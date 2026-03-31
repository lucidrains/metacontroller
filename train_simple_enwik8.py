from __future__ import annotations

import gzip
import random
from pathlib import Path

import tqdm
import fire
import numpy as np

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

from metacontroller.simple_metacontroller import TransformerWithMetacontroller, extract_grpo_data, z_score, policy_loss

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def cycle(loader):
    while True:
        for data in loader:
            yield data

def divisible_by(num, den):
    return (num % den) == 0

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return "".join(list(map(decode_token, tokens)))

def visualize_segments(
    tokens: Tensor,
    switch_betas: Tensor,
    delimiter = " ❚ ",
    threshold = 0.5
):
    tokens_list = tokens.tolist()
    switches = (switch_betas.flatten() >= threshold).tolist()

    segments = []

    for token, switch in zip(tokens_list, switches):
        if switch:
            segments.append(delimiter)

        segments.append(decode_token(token))

    return ''.join(segments)

def quantile_str_from_meta_output(meta_output, switch_entropy_quantiles):
    if not exists(meta_output.quantile_indices) or not exists(switch_entropy_quantiles):
        return ''

    q_idx = meta_output.quantile_indices[0].item()
    q_val = switch_entropy_quantiles[q_idx]
    return f' [quantile: {q_val}]'

def last_step(t):
    """Extract last step from grpo tensors, handling None and 1d quantile_indices."""
    if not exists(t):
        return None
    if t.ndim == 1:
        return t
    return t[:, -1:]

# exclamation rewarding

class ExclamationRewarder:
    def __init__(self, cap = 10):
        self.cap = cap

    def __call__(self, texts: list[str], batch_size = 32) -> list[float]:
        return [float(min(self.cap, text.count('!'))) for text in texts]

# sentiment rewarding

SENTIMENT_MODEL_NAME = "arnabdhar/tinybert-imdb"

class SentimentRewarder:
    def __init__(self, device: str | None = None):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_NAME)
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def __call__(self, texts: list[str], batch_size = 32) -> list[float]:
        results = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                padding = True,
                truncation = True,
                return_tensors = "pt"
            ).to(self.device)

            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim = -1)
            results.extend(probs[:, 1].tolist())

        return results

@torch.no_grad()
def sample(
    model,
    prompt: Tensor,
    seq_len: int,
    temperature = 1.,
):
    model.eval()

    prompt_seq_len = prompt.shape[-1]
    sample_num_times = max(0, seq_len - prompt_seq_len)

    state = prompt
    action = prompt[:, 1:]

    # initial cache derivation

    action_dist, meta_output, cache = model(
        state = state,
        actions = action,
        return_loss = False,
        return_cache = True,
        use_temporal_sequence_embed = False
    )

    next_state = model.action_readout.sample(action_dist[:, -1:], temperature = temperature)
    state = torch.cat((state, next_state), dim = -1)
    action = next_state

    all_switch_betas = []

    for i in range(sample_num_times):
        action_dist, meta_output, next_cache = model(
            state = state[:, -1:],
            actions = action,
            return_loss = False,
            cache = cache,
            return_cache = True,
            use_temporal_sequence_embed = False
        )

        all_switch_betas.append(meta_output.switch_beta[:, -1:])
        cache = next_cache

        if i < (sample_num_times - 1):
            next_state = model.action_readout.sample(action_dist[:, -1:], temperature = temperature)

            state = torch.cat((state, next_state), dim = -1)
            action = next_state

    return state[:, prompt_seq_len:], torch.cat(all_switch_betas, dim = -1)

# dataset

class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return self.data.size(0) // self.seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len - 1, (1,))
        full_seq = self.data[rand_start : rand_start + self.seq_len + 1].long()
        return full_seq

# train function

def train(
    num_batches = 20000,
    batch_size = 4,
    grad_accum_every = 4,
    learning_rate = 2e-4,
    bc_state_loss_weight = 1.,
    bc_action_loss_weight = 1.,
    ratio_loss_weight = 4.0,
    control_penalty_loss_weight = 0.1,
    latent_ar_loss_weight = 0.1,
    next_switch_pred_loss_weight = 0.1,
    predict_next_switch_embed = False,
    validate_every = 100,
    generate_every = 250,
    prime_length = 16,
    generate_length = 160,
    seq_len = 128,
    dim = 512,
    dim_code_bits = 8,
    depth = 3,
    kl_loss_weight = 1.,
    heads = 8,
    attn_dim_head = 64,
    switch_entropy_quantiles = (0.5, 0.75, 0.85, 0.9),
    eval_switch_entropy_quantile = 0.85,
    target_avg_token_length = 8.,
    temporal_sequence_embed_prob = 0.25,
    num_latent_pred_steps = 2,
    residual_stream_dropout = 0.,
    residual_stream_drop_prob = 0.,
    cpu = False,
    checkpoint_path = './results-simple-enwik8/train-enwik8.pt',
    enwik8_path = './data/enwik8.gz',
    grpo_phase = False,
    num_grpo_batches = 10000,
    grpo_group_size = 32,
    grpo_beta = 0.04,
    grpo_eps_clip = (0.2, 0.28), # DAPO epsilon
    grpo_learning_rate = 1e-4,
    grpo_reward_var_threshold = 0.0,
    grpo_reward_type = 'exclamation',
    use_wandb = False,
):
    # accelerator

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters = True)
    accelerator = Accelerator(cpu = cpu, kwargs_handlers = [ddp_kwargs])

    if exists(switch_entropy_quantiles):
        accelerator.print(f'training with entropy quantiles: {switch_entropy_quantiles}')
        accelerator.print(f'eval quantile target: {eval_switch_entropy_quantile}')

    if grpo_phase and use_wandb and accelerator.is_main_process:
        import wandb
        wandb.init(project = 'metacontroller-simple-enwik8', name = 'grpo')

    # ensure checkpoint directory exists

    checkpoint_dir = Path(checkpoint_path).parent
    checkpoint_dir.mkdir(parents = True, exist_ok = True)

    data_path = Path(enwik8_path)

    if not data_path.exists():
        accelerator.print(f'enwik8 data not found at {enwik8_path}')
        return

    with gzip.open(str(data_path)) as file:
        data = np.frombuffer(file.read(int(95e6)), dtype = np.uint8).copy()
        np_train, np_valid = np.split(data, [int(90e6)])
        data_train, data_val = torch.from_numpy(np_train), torch.from_numpy(np_valid)

    train_dataset = TextSamplerDataset(data_train, seq_len)
    val_dataset = TextSamplerDataset(data_val, seq_len)
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)

    # initialize merged model

    model = TransformerWithMetacontroller(
        dim = dim,
        state_embed_readout = dict(num_discrete = 256),
        action_embed_readout = dict(num_discrete = 256),
        dim_code_bits = dim_code_bits,
        lower_body = dict(depth = depth, heads = heads, attn_dim_head = attn_dim_head),
        upper_body = dict(depth = depth, heads = heads, attn_dim_head = attn_dim_head),
        emitter_decoder = dict(depth = 2, heads = heads, attn_dim_head = attn_dim_head),
        dim_queries_keys = 256,
        target_avg_token_length = target_avg_token_length,
        temporal_sequence_embed_prob = temporal_sequence_embed_prob,
        temporal_sequence_embedder = dict(
            depth = 1,
            heads = heads,
            attn_dim_head = attn_dim_head
        ),
        switch_entropy_quantiles = switch_entropy_quantiles,
        eval_switch_entropy_quantile = eval_switch_entropy_quantile,
        num_latent_pred_steps = num_latent_pred_steps,
        predict_next_switch_embed = predict_next_switch_embed,
        residual_stream_dropout = residual_stream_dropout,
        residual_stream_drop_prob = residual_stream_drop_prob,
        kl_loss_weight = kl_loss_weight
    )

    # optimize jointly

    optim = AdamW(model.parameters(), lr = learning_rate)

    grpo_parameters = model.internal_metacontroller_parameters()
    grpo_optim = AdamW(grpo_parameters, lr = grpo_learning_rate)

    model, optim, grpo_optim, train_loader, val_loader = accelerator.prepare(
        model, optim, grpo_optim, train_loader, val_loader
    )

    train_loader = cycle(train_loader)
    val_loader = cycle(val_loader)

    if grpo_phase:
        model.freeze_entropy_quantiles = True

        pretrained_path = Path(checkpoint_path)
        assert pretrained_path.exists(), f'pretrained checkpoint not found at {pretrained_path}'
        accelerator.print(f'loading pretrained checkpoint from {pretrained_path}')
        accelerator.unwrap_model(model).load(str(pretrained_path))

        if grpo_reward_type == 'sentiment':
            rewarder = SentimentRewarder(device = accelerator.device)
        else:
            rewarder = ExclamationRewarder()

        # validation sanity check from discovery checkpoint

        model.eval()
        with torch.no_grad():
            valid_data = next(val_loader)
            state = valid_data[:, :-1]
            actions = valid_data[:, 1:]

            outputs, meta_output = model(
                state = state,
                actions = actions,
                return_loss = True
            )

            segmented_str = visualize_segments(valid_data[0], meta_output.switch_beta[0])
            q_str = quantile_str_from_meta_output(meta_output, switch_entropy_quantiles)
            accelerator.print(f'\n\n[START VALIDATION] SEGMENTED{q_str}: {segmented_str}\n')

        best_reward_seen_so_far = 0.

        pbar = tqdm.tqdm(total = num_grpo_batches, mininterval = 10.0, desc = 'grpo')
        i = 0

        while i < num_grpo_batches:
            model.eval()

            # sample prompt

            data = next(train_loader)
            prompt = data[0:1, :prime_length]

            prompt = prompt.to(accelerator.device).repeat(grpo_group_size, 1)

            prompt_seq_len = prompt.shape[-1]
            sample_num_times = max(2, generate_length - prompt_seq_len)

            state = prompt
            action = prompt[:, 1:]

            with torch.no_grad():
                action_dist, meta_output, cache = model(
                    state = state,
                    actions = action,
                    return_loss = False,
                    return_cache = True,
                    use_temporal_sequence_embed = False
                )

                next_state = model.action_readout.sample(action_dist[:, -1:], temperature = 1.)
                state = torch.cat((state, next_state), dim = -1)

                first_grpo_data = extract_grpo_data(model, meta_output)
                first_grpo_data = type(first_grpo_data)(*map(last_step, first_grpo_data))
                grpo_data_list = [first_grpo_data]

                for _ in range(sample_num_times - 1):
                    action_dist, meta_output, next_cache = model(
                        state = state[:, -1:],
                        actions = state[:, -1:],
                        cache = cache,
                        return_loss = False,
                        return_cache = True,
                        use_temporal_sequence_embed = False
                    )

                    grpo_data_list.append(extract_grpo_data(model, meta_output))

                    next_state = model.action_readout.sample(action_dist[:, -1:], temperature = 1.)
                    state = torch.cat((state, next_state), dim = -1)

                    cache = next_cache

            # extract generated text and calculate reward

            generated = state[:, prompt_seq_len:]
            texts = [decode_tokens(t.tolist()) for t in generated]

            rewards = rewarder(texts, batch_size = grpo_group_size)
            rewards_tensor = torch.tensor(rewards, device = accelerator.device, dtype = torch.float32)

            reward_var = rewards_tensor.var()
            if reward_var <= grpo_reward_var_threshold:
                continue

            mean_reward = rewards_tensor.mean().item()
            max_reward = rewards_tensor.max().item()

            if max_reward > best_reward_seen_so_far:
                best_reward_seen_so_far = max_reward
                if accelerator.is_main_process:
                    best_idx = rewards_tensor.argmax().item()
                    accelerator.print(f'\n*** NEW BEST REWARD: {best_reward_seen_so_far:.3f} ***')
                    accelerator.print(f'PROMPT: {decode_tokens(prompt[best_idx].tolist())}')
                    accelerator.print(f'GENERATED: {texts[best_idx]}\n')

            # grpo tensors

            states, actions, log_probs, switch_betas, quantile_indices = zip(*grpo_data_list)
            group_states = torch.cat(states, dim = 1)
            group_actions = torch.cat(actions, dim = 1)
            group_log_probs = torch.cat(log_probs, dim = 1)
            group_switch_betas = torch.cat(switch_betas, dim = 1)

            group_quantile_indices = None
            if exists(quantile_indices[0]):
                group_quantile_indices = torch.cat(quantile_indices, dim = 1)

            # policy loss

            advantages = z_score(rewards_tensor)

            model.train()

            loss = policy_loss(
                model,
                state = group_states,
                old_log_probs = group_log_probs,
                actions = group_actions,
                advantages = advantages,
                switch_betas = group_switch_betas,
                mask = (group_switch_betas > 0.5),
                eps_clip = grpo_eps_clip,
                quantile_indices = group_quantile_indices
            )

            _, kl_loss = model.get_action_dist_for_internal_rl(
                group_states,
                switch_betas = group_switch_betas,
                quantile_indices = group_quantile_indices,
                return_kl_loss = True
            )

            switch_mask = (group_switch_betas > 0.5).float()
            kl_loss = (kl_loss * switch_mask).sum() / switch_mask.sum().clamp(min = 1e-5)

            total_loss = loss + kl_loss * grpo_beta

            accelerator.backward(total_loss)
            accelerator.clip_grad_norm_(model.parameters(), 0.5)
            grpo_optim.step()
            grpo_optim.zero_grad()

            loss_val = loss.item()

            with torch.no_grad():
                switch_density = (group_switch_betas > 0.5).float().mean().item()

            if accelerator.is_main_process and use_wandb:
                import wandb
                if exists(wandb.run):
                    log_dict = dict(
                        loss = loss_val,
                        kl_loss = kl_loss.item(),
                        mean_reward = mean_reward,
                        max_reward = max_reward,
                        best_reward_seen_so_far = best_reward_seen_so_far,
                        reward_var = reward_var.item(),
                        switch_density = switch_density
                    )

                    if divisible_by(i, generate_every):
                        table = wandb.Table(columns = ['Prompt', 'Generated', 'Reward'])
                        table.add_data(decode_tokens(prompt[0].tolist()), texts[0], rewards[0])
                        log_dict['samples'] = table

                    wandb.log(log_dict)

            if divisible_by(i, 10):
                pbar.set_postfix(loss = f'{loss_val:.3f}', kl = f'{kl_loss.item():.3f}', reward = f'{mean_reward:.3f}')
                tqdm.tqdm.write(f'{i}: loss: {loss_val:.3f} reward: {mean_reward:.3f} max: {max_reward:.3f}')

            # save checkpoint

            if divisible_by(i, 1000) and i > 0:
                accelerator.print(f'saving GRPO checkpoint to {checkpoint_path}')
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save(str(Path(checkpoint_path).with_stem(f'{Path(checkpoint_path).stem}-grpo')))

            i += 1
            pbar.update(1)

        accelerator.print(f'saving final GRPO checkpoint to {checkpoint_path}')
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save(str(Path(checkpoint_path).with_stem(f'{Path(checkpoint_path).stem}-grpo')))
        return

    method_desc = f'Entropy ({len(switch_entropy_quantiles)} thresholds)' if exists(switch_entropy_quantiles) else 'QK'
    pbar = tqdm.tqdm(range(num_batches), mininterval = 10.0, desc = f'training jointly | {method_desc}')

    for i in pbar:
        model.train()

        last_loss = 0.
        last_bc_action_loss = 0.
        last_bc_state_loss = 0.
        last_ratio_loss = 0.
        last_latent_ar_loss = 0.
        last_kl_loss = 0.
        last_next_switch_pred_loss = 0.
        last_control_penalty_loss = 0.
        last_switch_density = 0.

        for _ in range(grad_accum_every):
            data = next(train_loader)
            state = data[:, :-1]
            actions = data[:, 1:]

            losses, meta_output = model(
                state = state,
                actions = actions,
                return_loss = True
            )

            loss = (losses.bc_state_loss + 0.5) * bc_state_loss_weight + (losses.bc_action_loss + 0.5) * bc_action_loss_weight + losses.ratio_loss * ratio_loss_weight + losses.latent_loss * latent_ar_loss_weight + losses.kl_loss * kl_loss_weight + losses.next_switch_pred_loss * next_switch_pred_loss_weight + losses.control_penalty_loss * control_penalty_loss_weight

            last_loss = loss.item()
            last_bc_state_loss = losses.bc_state_loss.item()
            last_bc_action_loss = losses.bc_action_loss.item()
            last_ratio_loss = losses.ratio_loss.item()
            last_latent_ar_loss = losses.latent_ar_loss.item()
            last_sigreg_loss = losses.sigreg_loss.item() if isinstance(losses.sigreg_loss, torch.Tensor) else losses.sigreg_loss
            last_kl_loss = losses.kl_loss.item() if isinstance(losses.kl_loss, torch.Tensor) else losses.kl_loss
            last_next_switch_pred_loss = losses.next_switch_pred_loss.item()
            last_control_penalty_loss = losses.control_penalty_loss.item() if isinstance(losses.control_penalty_loss, torch.Tensor) else losses.control_penalty_loss
            last_switch_density = (meta_output.switch_beta > 0.5).float().mean().item()

            accelerator.backward(loss / grad_accum_every)

        accelerator.clip_grad_norm_(model.parameters(), 0.5)
        optim.step()
        optim.zero_grad()

        # logging

        if divisible_by(i, 10):
            log_str = f'{i}: loss: {last_loss:.3f} bc_action: {last_bc_action_loss:.3f} bc_state: {last_bc_state_loss:.3f} ratio: {last_ratio_loss:.3f} latent_ar: {last_latent_ar_loss:.3f} sigreg: {last_sigreg_loss:.3f} kl: {last_kl_loss:.3f} next_sw: {last_next_switch_pred_loss:.3f}'
            if control_penalty_loss_weight > 0.:
                log_str += f' ctrl_pen: {last_control_penalty_loss:.3f}'
            log_str += f' density: {last_switch_density:.3f}'
            tqdm.tqdm.write(log_str)
            pbar.set_postfix(bc_action = f'{last_bc_action_loss:.3f}', kl = f'{last_kl_loss:.3f}', density = f'{last_switch_density:.3f}')

        if divisible_by(i, validate_every):
            model.eval()
            with torch.no_grad():
                valid_data = next(val_loader)
                val_state = valid_data[:, :-1]
                val_actions = valid_data[:, 1:]

                losses, meta_output = model(
                    state = val_state,
                    actions = val_actions,
                    return_loss = True,
                    use_temporal_sequence_embed = False
                )

                loss_val = (losses.bc_state_loss + 0.5) * bc_state_loss_weight + (losses.bc_action_loss + 0.5) * bc_action_loss_weight + losses.ratio_loss * ratio_loss_weight + losses.latent_loss * latent_ar_loss_weight + losses.kl_loss * kl_loss_weight + losses.next_switch_pred_loss * next_switch_pred_loss_weight + losses.control_penalty_loss * control_penalty_loss_weight

                segmented_str = visualize_segments(val_state[0], meta_output.switch_beta[0])
                q_str = quantile_str_from_meta_output(meta_output, switch_entropy_quantiles)
                accelerator.print(f'\n\nSEGMENTED{q_str}: {segmented_str}\n')

                accelerator.print(f'{i}: validation loss: {loss_val.item():.3f} (bc: {losses.bc_action_loss.item():.3f})')

        if divisible_by(i, generate_every):
            model.eval()
            inp = random.choice(val_dataset)[:prime_length]
            inp = inp.to(accelerator.device)

            prime = decode_tokens(inp.tolist())
            accelerator.print(f'\n\nPROMPT: {prime}')

            prompt = inp[None, ...]
            sampled, sampled_switch_betas = sample(model, prompt, generate_length)

            sampled_segmented_str = visualize_segments(sampled[0], sampled_switch_betas[0])
            accelerator.print(f'GENERATED [quantile: {eval_switch_entropy_quantile}]: {sampled_segmented_str}\n')

        # save checkpoint

        if divisible_by(i, 1000) and i > 0:
            accelerator.print(f'saving checkpoint to {checkpoint_path}')
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save(checkpoint_path)

    # final save

    accelerator.print(f'saving final checkpoint to {checkpoint_path}')
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save(checkpoint_path)

if __name__ == '__main__':
    fire.Fire(train)
