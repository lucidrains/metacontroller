from __future__ import annotations
# /// script
# dependencies = [
#   "metacontroller-pytorch>=0.2.18",
#   "transformers",
#   "accelerate",
#   "fire",
#   "torch",
#   "torch-einops-utils>=0.0.30",
#   "einops",
#   "tqdm",
#   "numpy<2",
#   "jax",
#   "wandb",
#   "streaming-deep-rl==0.1.6",
#   "x-mlps-pytorch>=0.0.12"
# ]
# ///

# modify jax above to jax[cuda11|12|13] for gpu

import os
import gzip
import random
from pathlib import Path

import tqdm
import fire
import numpy as np

import torch
from torch.nn import Module
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from einops import rearrange

from metacontroller import MetaController, Transformer, binary_entropy
from metacontroller.metacontroller import extract_grpo_data, z_score
from metacontroller.metacontroller_with_binary_mapper import MetaControllerWithBinaryMapper

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from x_mlps_pytorch.normed_mlp import MLP
from streaming_deep_rl import StreamingACLambda

# helpers

def exists(v):
    return v is not None

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

def log_params(print_fn, title, params):
    bar = '=' * 50
    print_fn(f'\n{bar}')
    print_fn(title)
    print_fn(bar)
    for label, value in params:
        print_fn(f'{label:<20s} {value}')
    print_fn(f'{bar}\n')

def visualize_segments(
    tokens: Tensor,
    switch_betas: Tensor,
    delimiter = " \u275A ",
    threshold = 0.5
):
    tokens_list = tokens.tolist()
    switches = (switch_betas >= threshold).tolist()

    segments = []

    for i in range(len(switches)):
        if switches[i]:
            segments.append(delimiter)
        segments.append(decode_tokens([tokens_list[i]]))

    return ''.join(segments)

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

# sampling


@torch.no_grad()
def sample(
    model,
    prompt: Tensor,
    seq_len: int,
    temperature = 1.,
    force_behavior_cloning = False,
    meta_controller = None
):
    model.eval()

    prompt_seq_len = prompt.shape[-1]
    sample_num_times = max(0, seq_len - prompt_seq_len)

    state = prompt
    action = prompt[:, 1:]

    (action_dist, _), cache = model(
        state = state,
        actions = action,
        return_cache = True,
        return_state_action_cache = True,
        force_behavior_cloning = force_behavior_cloning,
        meta_controller = meta_controller
    )

    actual_model = getattr(model, 'model', model)
    next_action = next_state = actual_model.action_readout.sample(action_dist[:, -1:], temperature = temperature)

    state = torch.cat((state, next_state), dim = -1)

    for _ in range(sample_num_times):
        (action_dist, _), next_cache = model(
            state = state[:, -1:],
            actions = state[:, -1:],
            cache = cache,
            return_state_action_cache = True,
            force_behavior_cloning = force_behavior_cloning,
            meta_controller = meta_controller
        )

        next_state = actual_model.action_readout.sample(action_dist[:, -1:], temperature = temperature)

        state = torch.cat((state, next_state), dim = -1)

        cache = next_cache

    return state[:, prompt_seq_len:]

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

# action proposer wrapper for streaming RL

class ActionProposerMLP(Module):
    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp

    def forward(self, x, **kwargs):
        return self.mlp(x), None

# train function

def train(
    bc_phase = False,
    num_bc_batches = 10000,
    num_discovery_batches = 10000,
    discovery_warmup_steps = 100,
    batch_size = 4,
    grad_accum_every = 4,
    learning_rate = 1e-4,
    discovery_learning_rate = 1e-4,
    bc_state_loss_weight = 1.,
    bc_action_loss_weight = 1.,
    discovery_state_loss_weight = 1.,
    discovery_action_loss_weight = 1.,
    discovery_kl_loss_weight = 0.2,
    discovery_kl_loss_warmup_steps = 0,
    discovery_entropy_loss_weight = 0.75,
    discovery_entropy_loss_threshold = 0.4,
    discovery_negative_entropy_loss_weight = 0.75,
    ratio_loss_weight = 4.0,
    ratio_loss_final_weight = 0.5,
    validate_every = 100,
    generate_every = 250,
    prime_length = 16,
    generate_length = 160,
    seq_len = 128,
    dim = 512,
    dim_meta_controller = 128,
    dim_latent = 64,
    depth = 3,
    heads = 8,
    attn_dim_head = 48,
    hypernetwork_low_rank = 8,
    target_temporal_segment_len = 8,
    use_binary_mapper = True,
    dim_code_bits = 8,
    kl_loss_threshold = 0.1,
    switch_temperature = 0.1,
    discovery_phase = False,
    discovery_hard_switch = False,
    metacontroller_residual_stream_dropout_prob = 0.1,
    cpu = False,
    checkpoint_path = './results-enwik8/train-enwik8.pt',
    enwik8_path = './data/enwik8.gz',
    grpo_phase = False,
    num_grpo_batches = 1000,
    grpo_group_size = 32,
    grpo_learning_rate = 1e-4,
    grpo_reward_var_threshold = 0.0,
    grpo_hard_switch = True,
    grpo_reward_type = 'exclamation',
    evo_phase = False,
    num_evo_generations = 50,
    evo_population_size = 32,
    evo_noise_std = 0.02,
    streaming_rl_phase = False,
    num_streaming_rl_batches = 1000,
    streaming_rl_delay_steps = 16,
    streaming_rl_regen_reg_rate = 0.0,
    streaming_rl_regen_reg_every = 1,
    streaming_rl_val_min = -1.,
    streaming_rl_val_max = 50.,
    streaming_rl_num_bins = 64,
    streaming_rl_entropy_weight = 0.01,
    streaming_rl_update_only_on_switch = True,
):
    # accelerator

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(cpu = cpu, kwargs_handlers=[ddp_kwargs])

    assert sum([bc_phase, discovery_phase, grpo_phase, evo_phase, streaming_rl_phase]) == 1, 'exactly one training phase must be active (bc, discovery, grpo, evo, or streaming_rl)'

    if (grpo_phase or evo_phase or streaming_rl_phase) and accelerator.is_main_process:
        import wandb

        if streaming_rl_phase:
            wandb_name = "streaming_rl"
        elif grpo_phase:
            wandb_name = "grpo"
        else:
            wandb_name = "evo"

        wandb.init(project="metacontroller-enwik8", name=wandb_name)

    # ensure checkpoint directory exists

    checkpoint_dir = Path(checkpoint_path).parent
    checkpoint_dir.mkdir(parents = True, exist_ok = True)

    # hyperparameter summary

    params = [
        ('BC Phase', bc_phase),
        ('Discovery Phase', discovery_phase),
        ('BC Steps', num_bc_batches),
        ('Discovery Steps', num_discovery_batches),
        ('Batch Size', batch_size),
        ('Grad Accum Every', grad_accum_every),
        ('BC LR', learning_rate),
        ('Disc LR', discovery_learning_rate),
        ('Seq Len', seq_len),
        ('CPU', cpu),
        ('Model Dim', dim),
        ('MC Dim', dim_meta_controller),
        ('Latent Dim', dim_latent),
        ('Disc Hard Switch', discovery_hard_switch),
        ('Binary Mapper', f'{use_binary_mapper} (bits: {dim_code_bits} temp: {switch_temperature})'),
        ('Depth', depth),
        ('Heads', heads),
        ('Target Seg Len', target_temporal_segment_len),
        ('Hypernet Rank', hypernetwork_low_rank),
        ('Warmup Steps', discovery_warmup_steps),
        ('Checkpoint', checkpoint_path),
    ]

    if grpo_phase:
        params += [
            ('GRPO Batches', num_grpo_batches),
            ('GRPO Group Size', grpo_group_size),
            ('GRPO LR', grpo_learning_rate),
            ('GRPO Hard Switch', grpo_hard_switch),
            ('GRPO Reward', grpo_reward_type),
        ]
    elif streaming_rl_phase:
        params += [
            ('SRL Batches', num_streaming_rl_batches),
            ('SRL Delay', streaming_rl_delay_steps),
            ('SRL Val Min', streaming_rl_val_min),
            ('SRL Val Max', streaming_rl_val_max),
            ('SRL Bins', streaming_rl_num_bins),
            ('SRL Entropy', streaming_rl_entropy_weight),
            ('SRL Update Sw.', streaming_rl_update_only_on_switch),
        ]
    elif evo_phase:
        params += [
            ('EVO Generations', num_evo_generations),
            ('EVO Population', evo_population_size),
            ('EVO Noise Std', evo_noise_std),
        ]

    log_params(accelerator.print, 'Training enwik8', params)

    # dataset

    data_path = Path(enwik8_path)

    if not data_path.exists():
        accelerator.print(f"enwik8 data not found at {enwik8_path}")
        return

    with gzip.open(str(data_path)) as file:
        data = np.frombuffer(file.read(int(95e6)), dtype=np.uint8).copy()
        np_train, np_valid = np.split(data, [int(90e6)])
        data_train, data_val = torch.from_numpy(np_train), torch.from_numpy(np_valid)

    train_dataset = TextSamplerDataset(data_train, seq_len)
    val_dataset = TextSamplerDataset(data_val, seq_len)
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)

    # model

    if not use_binary_mapper:
        meta_controller_klass = MetaController
        meta_controller_kwargs = dict(
            dim_model = dim,
            dim_meta_controller = dim_meta_controller,
            dim_latent = dim_latent,
            hypernetwork_low_rank = hypernetwork_low_rank,
            kl_loss_weight = discovery_kl_loss_weight,
            kl_loss_warmup_steps = discovery_kl_loss_warmup_steps,
            target_temporal_segment_len = target_temporal_segment_len,
            ratio_loss_weight = ratio_loss_weight,
            ratio_loss_chunk_size = 4 * target_temporal_segment_len,
            ratio_loss_final_weight = ratio_loss_final_weight,
            ratio_loss_warmdown_steps = num_discovery_batches,
            hard_switch = discovery_hard_switch
        )
    else:
        meta_controller_klass = MetaControllerWithBinaryMapper
        meta_controller_kwargs = dict(
            dim_model = dim,
            dim_meta_controller = dim_meta_controller,
            dim_code_bits = dim_code_bits,
            kl_loss_threshold = kl_loss_threshold,
            switch_temperature = switch_temperature,
            hypernetwork_low_rank = hypernetwork_low_rank,
            kl_loss_weight = discovery_kl_loss_weight,
            kl_loss_warmup_steps = discovery_kl_loss_warmup_steps,
            target_temporal_segment_len = target_temporal_segment_len,
            ratio_loss_weight = ratio_loss_weight,
            ratio_loss_chunk_size = 4 * target_temporal_segment_len,
            ratio_loss_final_weight = ratio_loss_final_weight,
            ratio_loss_warmdown_steps = num_discovery_batches,
            hard_switch = discovery_hard_switch
        )

    meta_controller = meta_controller_klass(**meta_controller_kwargs)

    model_kwargs = dict(
        dim = dim,
        state_embed_readout = dict(num_discrete = 256),
        action_embed_readout = dict(num_discrete = 256),
        lower_body = dict(depth = depth, heads = heads, attn_dim_head = attn_dim_head),
        upper_body = dict(depth = depth, heads = heads, attn_dim_head = attn_dim_head),
        lower_transformer_post_norm = True,
        meta_controller = None
    )

    model = Transformer(**model_kwargs)

    # load checkpoint if it exists or if we are in discovery phase or grpo phase or streaming rl phase

    if grpo_phase or evo_phase or streaming_rl_phase:
        discovery_checkpoint_path = str(Path(checkpoint_path).with_stem(f"{Path(checkpoint_path).stem}-discovery"))

        load_path = discovery_checkpoint_path

        if exists(load_path):
            accelerator.print(f"loading meta controller checkpoint from {load_path}")
            meta_controller = meta_controller_klass.init_and_load(load_path, strict = False)
        else:
            accelerator.print(f"error: refinement phase requested but no base checkpoint found")
            return

        # for backbone, if loading from discovery, we might want to also load the backbone that was saved at discovery end
        # (though currently discovery only saves mc, but it's safer to check)

        if Path(checkpoint_path).exists():
            accelerator.print(f"loading backbone checkpoint from {checkpoint_path}")
            model = Transformer.init_and_load(checkpoint_path, strict = False)

    elif discovery_phase or Path(checkpoint_path).exists():
        if Path(checkpoint_path).exists():
            accelerator.print(f"loading checkpoint from {checkpoint_path}")
            model = Transformer.init_and_load(checkpoint_path, strict = False)
        elif discovery_phase:
            accelerator.print(f"warning: discovery phase requested but {checkpoint_path} not found. training from scratch.")

    model.meta_controller = meta_controller

    # optimizer

    bc_optim = AdamW(model.parameters(), lr = learning_rate)

    discovery_optim = AdamW(meta_controller.discovery_parameters(), lr = discovery_learning_rate)

    grpo_optim = AdamW(meta_controller.internal_rl_parameters(), lr = grpo_learning_rate)

    # streaming RL initialization

    if streaming_rl_phase:
        # simple MLP actor for streaming RL (replaces the transformer action proposer)

        actor_mlp = MLP(dim, dim, dim, dim, norm_elementwise_affine = False)

        meta_controller.action_proposer = ActionProposerMLP(actor_mlp)

        critic_mlp = MLP(dim, dim_latent, dim_latent, dim_latent, norm_elementwise_affine = False, activate_last = True)

        # Determine the correct dimension from the loaded meta controller
        if use_binary_mapper:
            actual_dim_latent = getattr(meta_controller, 'num_codes', 2 ** dim_code_bits)
        else:
            actual_dim_latent = getattr(meta_controller, 'dim_latent', dim_latent)

        streaming_agent = StreamingACLambda(
            actor = actor_mlp,
            critic = critic_mlp,
            dim_state = dim,
            dim_actor = dim,
            dim_critic = dim_latent,
            num_continuous_actions = actual_dim_latent,
            val_min = streaming_rl_val_min,
            val_max = streaming_rl_val_max,
            num_bins = streaming_rl_num_bins,
            entropy_weight = streaming_rl_entropy_weight,
            adaptive = True,
            actor_use_ema = True,
            delay_steps = streaming_rl_delay_steps,
            regen_reg_rate = streaming_rl_regen_reg_rate,
            regen_reg_every = streaming_rl_regen_reg_every
        )

        streaming_agent.to(accelerator.device)
    else:
        streaming_agent = None

    model, meta_controller, bc_optim, discovery_optim, grpo_optim, train_loader, val_loader = accelerator.prepare(
        model, meta_controller, bc_optim, discovery_optim, grpo_optim, train_loader, val_loader
    )

    if streaming_rl_phase:
        streaming_agent = accelerator.prepare(streaming_agent)

    train_loader = cycle(train_loader)
    val_loader = cycle(val_loader)

    # validation sanity check if starting from discovery phase

    if discovery_phase:
        model.eval()
        with torch.no_grad():
            valid_data = next(val_loader)
            state = valid_data[:, :-1]
            actions = valid_data[:, 1:]

            state_loss, action_loss = model(
                state = state,
                actions = actions,
                force_behavior_cloning = True,
                meta_controller = meta_controller
            )

            loss_val = state_loss + action_loss
            accelerator.print(f"\ninitial BC validation loss: {loss_val.item():.3f} (state: {state_loss.item():.3f} action: {action_loss.item():.3f})")

            # discovery sanity check with ablated control signal

            discovery_losses, _ = model(
                state = state,
                actions = actions,
                discovery_phase = True,
                control_signal_multiplier = 0.0,
                return_meta_controller_output = True,
                meta_controller = meta_controller
            )

            state_recon_loss, action_recon_loss, _, _ = discovery_losses
            accelerator.print(f"initial ablated discovery loss: {state_recon_loss.item() + action_recon_loss.item():.3f} (state: {state_recon_loss.item():.3f} action: {action_recon_loss.item():.3f})\n")

    # training

    if grpo_phase or streaming_rl_phase:
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

            outputs = model(
                state = state,
                actions = actions,
                discovery_phase = True,
                return_meta_controller_output = True,
                meta_controller = meta_controller
            )

            discovery_losses, meta_output = outputs

            segmented_str = visualize_segments(valid_data[0], meta_output.switch_beta[0], threshold = 0.5)
            accelerator.print(f"\n\n[START VALIDATION] SEGMENTED: {segmented_str}\n")

        pbar_desc = "streaming_rl" if streaming_rl_phase else "grpo"
        num_batches = num_streaming_rl_batches if streaming_rl_phase else num_grpo_batches
        pbar = tqdm.tqdm(total = num_batches, mininterval = 10.0, desc = pbar_desc)
        i = 0

        while i < num_batches:
            model.eval()
            meta_controller.train_internal_rl()

            # 1. Sample prompt
            data = next(train_loader)
            prompt = data[0:1, :prime_length] # (1, prime_length)

            group_size = 1 if streaming_rl_phase else grpo_group_size
            prompt = prompt.to(accelerator.device).repeat(group_size, 1)

            prompt_seq_len = prompt.shape[-1]
            sample_num_times = max(2, generate_length - prompt_seq_len)

            state = prompt
            action = prompt[:, 1:]

            with torch.no_grad():
                out, cache = model(
                    state = state,
                    actions = action,
                    return_cache = True,
                    force_behavior_cloning = False,
                    meta_controller = meta_controller,
                    hard_switch = grpo_hard_switch
                )
                action_dist = out[0] if isinstance(out, tuple) else out

                next_state = model.action_readout.sample(action_dist[:, -1:], temperature = 1.)
                state = torch.cat((state, next_state), dim = -1)

                def last(t):
                    return t[:, -1:]

                first_grpo_data = extract_grpo_data(meta_controller, cache)
                first_grpo_data = type(first_grpo_data)(*map(last, first_grpo_data))
                grpo_data_list = [first_grpo_data]

                for _ in range(sample_num_times - 1):
                    out, next_cache = model(
                        state = state[:, -1:],
                        actions = state[:, -1:],
                        cache = cache,
                        return_cache = True,
                        force_behavior_cloning = False,
                        meta_controller = meta_controller,
                        hard_switch = grpo_hard_switch
                    )

                    action_dist = out[0] if isinstance(out, tuple) else out
                    grpo_data_list.append(extract_grpo_data(meta_controller, next_cache))

                    next_state = model.action_readout.sample(action_dist[:, -1:], temperature = 1.)
                    state = torch.cat((state, next_state), dim = -1)

                    cache = next_cache

            # Extract generated text and calculate reward
            generated = state[:, prompt_seq_len:]
            texts = [decode_tokens(t.tolist()) for t in generated]

            rewards = rewarder(texts, batch_size = group_size)
            rewards_tensor = torch.tensor(rewards, device = accelerator.device, dtype = torch.float32)

            reward_var = rewards_tensor.var()
            if reward_var <= grpo_reward_var_threshold:
                continue

            mean_reward = rewards_tensor.mean().item()
            max_reward = rewards_tensor.max().item()

            # Grpo tensors
            states, actions, log_probs, switch_betas = zip(*grpo_data_list)
            group_states = torch.cat(states, dim = 1) # (G, T, D)
            group_actions = torch.cat(actions, dim = 1) # (G, T, D_latent)
            group_log_probs = torch.cat(log_probs, dim = 1) # (G, T, D_latent)
            group_switch_betas = torch.cat(switch_betas, dim = 1) # (G, T)

            if streaming_rl_phase:
                num_samples, num_steps, _ = group_states.shape

                for s in range(num_samples):
                    sample_reward = rewards_tensor[s]

                    for t in range(num_steps):
                        is_last = (t == num_steps - 1)
                        is_switch = group_switch_betas[s, t] > 0.5

                        # temporal compression: optionally only update on switches or last step
                        if streaming_rl_update_only_on_switch and not (is_switch or is_last):
                            continue

                        metrics = streaming_agent.update(
                            state = group_states[s, t],
                            action = group_actions[s, t],
                            next_state = group_states[s, t + 1] if not is_last else torch.zeros_like(group_states[s, t]),
                            reward = sample_reward if is_last else torch.zeros_like(sample_reward),
                            is_terminal = is_last,
                            drain = is_last
                        )

                loss_val = metrics.td_error if exists(metrics) else 0.

            else:
                # GRPO Phase
                advantages = z_score(rewards_tensor)

                # Policy loss
                loss = meta_controller.policy_loss(
                    state = group_states,
                    old_log_probs = group_log_probs,
                    actions = group_actions,
                    advantages = advantages,
                    mask = (group_switch_betas > 0.5)
                )

                accelerator.backward(loss)
                accelerator.clip_grad_norm_(meta_controller.internal_rl_parameters(), 0.5)
                grpo_optim.step()
                grpo_optim.zero_grad()

                loss_val = loss.item()

            # calculate hard switch density for logging

            with torch.no_grad():
                switch_density = (group_switch_betas > 0.5).float().mean().item()

            if accelerator.is_main_process:
                import wandb
                if exists(wandb.run):
                    log_dict = dict(
                        loss = loss_val,
                        mean_reward = mean_reward,
                        max_reward = max_reward,
                        reward_var = reward_var.item(),
                        switch_density = switch_density
                    )

                    if streaming_rl_phase and exists(metrics):
                        log_dict.update(metrics._asdict())

                    if divisible_by(i, generate_every):
                        table = wandb.Table(columns = ['Prompt', 'Generated', 'Reward'])
                        table.add_data(decode_tokens(prompt[0].tolist()), texts[0], rewards[0])
                        log_dict['samples'] = table

                    wandb.log(log_dict)

            if divisible_by(i, 10):
                pbar.set_postfix(loss = f"{loss_val:.3f}", reward = f"{mean_reward:.3f}")
                tqdm.tqdm.write(f"{i}: loss: {loss_val:.3f} reward: {mean_reward:.3f} max: {max_reward:.3f}")

            if divisible_by(i, generate_every):
                accelerator.print(f"\nPROMPT: {decode_tokens(prompt[0].tolist())}")
                accelerator.print(f"GENERATED ({rewards[0]:.3f}): {decode_tokens(generated[0].tolist())}\n")

            i += 1
            pbar.update(1)

        # Save checkpoint
        checkpoint_suffix = "streaming-rl" if streaming_rl_phase else "grpo"
        phase_checkpoint_path = str(Path(checkpoint_path).with_stem(f"{Path(checkpoint_path).stem}-{checkpoint_suffix}"))
        accelerator.print(f"saving {checkpoint_suffix.upper()} checkpoint (metacontroller only) to {phase_checkpoint_path}")
        accelerator.wait_for_everyone()
        unwrapped_meta_controller = accelerator.unwrap_model(meta_controller)
        unwrapped_meta_controller.save(phase_checkpoint_path)
        if not evo_phase:
            return

    if evo_phase:
        if grpo_reward_type == 'sentiment':
            rewarder = SentimentRewarder(device = accelerator.device)
        else:
            rewarder = ExclamationRewarder()

        @torch.no_grad()
        def sentiment_env(transformer):
            transformer.eval()

            # handle x_evolution Noisable wrapper
            mc = getattr(transformer, 'meta_controller', None)
            if not exists(mc):
                mc = transformer.model.meta_controller

            mc.train_internal_rl()

            num_eval_samples = 8
            data = next(train_loader)
            prompt = data[:num_eval_samples, :prime_length]
            prompt = prompt.to(accelerator.device)

            generated = sample(
                transformer,
                prompt = prompt,
                seq_len = generate_length,
                meta_controller = mc
            )

            texts = [decode_tokens(t.tolist()) for t in generated]
            rewards = rewarder(texts, batch_size = num_eval_samples)

            fitness = np.mean(rewards)

            if accelerator.is_main_process:
                import wandb
                if exists(wandb.run):
                    wandb.log(dict(evo_fitness = fitness))

            return fitness

        accelerator.print(f"starting ES optimization for {num_evo_generations} generations...")

        model.meta_controller = meta_controller

        model.evolve(
            num_generations = num_evo_generations,
            environment = sentiment_env,
            noise_population_size = evo_population_size,
            noise_scale = evo_noise_std
        )

        evo_checkpoint_path = str(Path(checkpoint_path).with_stem(f"{Path(checkpoint_path).stem}-evo"))
        accelerator.print(f"saving ES checkpoint (metacontroller only) to {evo_checkpoint_path}")
        accelerator.wait_for_everyone()
        unwrapped_meta_controller = accelerator.unwrap_model(meta_controller)
        unwrapped_meta_controller.save(evo_checkpoint_path)
        return

    if bc_phase:
        start_step = 0
        total_steps = num_bc_batches
    else:
        start_step = num_bc_batches
        total_steps = num_bc_batches + num_discovery_batches
    
    pbar = tqdm.tqdm(range(start_step, total_steps), mininterval = 10.0, desc = "training")

    for i in pbar:
        is_discovering = i >= num_bc_batches

        if is_discovering:
            model.train_discovery(eval_rest = True, meta_controller = meta_controller)
        else:
            model.train()

        optim = discovery_optim if is_discovering else bc_optim

        last_loss = 0.
        last_state_loss = 0.
        last_action_loss = 0.
        last_switch_density = 0.
        last_ratio_loss = 0.
        last_kl_loss = 0.

        for _ in range(grad_accum_every):
            data = next(train_loader)

            state = data[:, :-1]
            actions = data[:, 1:]

            multiplier = 1.0
            if is_discovering and discovery_warmup_steps > 0:
                discovery_step = i - num_bc_batches
                multiplier = min(1.0, discovery_step / discovery_warmup_steps)

            if is_discovering:
                meta_controller.maybe_increment_kl_loss_step()

            outputs = model(
                state = state,
                actions = actions,
                discovery_phase = is_discovering,
                force_behavior_cloning = not is_discovering,
                control_signal_multiplier = multiplier,
                return_meta_controller_output = is_discovering,
                meta_controller = meta_controller,
                metacontroller_residual_stream_dropout = metacontroller_residual_stream_dropout_prob
            )

            if is_discovering:
                discovery_losses, meta_output = outputs
                obs_loss, action_recon_loss, kl_loss, ratio_loss = discovery_losses

                entropy_loss = binary_entropy(meta_output.switch_beta).mean()
                entropy_loss = (entropy_loss - discovery_entropy_loss_threshold).relu()

                # dynamic entropy weight
                # if density is 0, apply negative entropy weight to push switch betas towards 0.5

                switch_density = (meta_output.switch_beta > 0.5).float().mean()

                entropy_weight = discovery_entropy_loss_weight
                if switch_density == 0:
                    entropy_weight = -discovery_negative_entropy_loss_weight

                loss = (action_recon_loss + 0.5) * discovery_action_loss_weight + \
                       (obs_loss + 0.5) * discovery_state_loss_weight + \
                       kl_loss + \
                       entropy_loss * entropy_weight + \
                       ratio_loss

                last_state_loss = obs_loss.item()
                last_action_loss = action_recon_loss.item()
                last_switch_density = (meta_output.switch_beta > 0.5).float().mean().item()
                last_ratio_loss = ratio_loss.item()
                last_kl_loss = kl_loss.item()
                last_kl_weight = meta_controller.current_kl_loss_weight
                last_entropy_loss = entropy_loss.item()
            else:
                state_loss, action_loss = outputs
                loss = (action_loss + 0.5) * bc_action_loss_weight + \
                       (state_loss + 0.5) * bc_state_loss_weight

                last_state_loss = state_loss.item()
                last_action_loss = action_loss.item()

            accelerator.backward(loss / grad_accum_every)
            last_loss = loss.item()

        accelerator.clip_grad_norm_(model.parameters(), 0.5)
        optim.step()
        optim.zero_grad()

        # logging

        if divisible_by(i, 10):
            phase = 'discovering' if is_discovering else 'cloning'
            action_loss_key = 'action_recon' if is_discovering else 'action'

            log_str = f"{i}: loss: {last_loss:.3f} ({phase}) state: {last_state_loss:.3f} {action_loss_key}: {last_action_loss:.3f}"

            if is_discovering:
                log_str += f" density: {last_switch_density:.3f} kl: {last_kl_loss:.3f} kl_weight: {last_kl_weight:.3f} entropy: {last_entropy_loss:.3f} ratio: {last_ratio_loss:.3f}"
                pbar.set_postfix(state=f"{last_state_loss:.3f}", action_recon=f"{last_action_loss:.3f}", density=f"{last_switch_density:.3f}", kl=f"{last_kl_loss:.3f}", kl_weight=f"{last_kl_weight:.3f}")
            else:
                pbar.set_postfix(state=f"{last_state_loss:.3f}", action=f"{last_action_loss:.3f}")

            # print every batch loss to validate speed and values
            tqdm.tqdm.write(log_str)

        if divisible_by(i, validate_every):
            model.eval()
            with torch.no_grad():
                valid_data = next(val_loader)

                state = valid_data[:, :-1]
                actions = valid_data[:, 1:]

                outputs = model(
                    state = state,
                    actions = actions,
                    discovery_phase = is_discovering,
                    force_behavior_cloning = not is_discovering,
                    return_meta_controller_output = is_discovering,
                    meta_controller = meta_controller
                )

                if is_discovering:
                    discovery_losses, meta_output = outputs
                    loss_val = discovery_losses.action_recon + discovery_losses.state_pred

                    segmented_str = visualize_segments(valid_data[0], meta_output.switch_beta[0], threshold = 0.5)
                    accelerator.print(f"\n\nSEGMENTED: {segmented_str}\n")
                else:
                    loss_val = outputs.action + outputs.state

                accelerator.print(f"{i}: validation loss: {loss_val.item():.3f}")

        if not is_discovering and divisible_by(i, generate_every):
            model.eval()
            inp = random.choice(val_dataset)[:prime_length]
            inp = inp.to(accelerator.device)

            prime = decode_tokens(inp)
            accelerator.print(f"\nPROMPT: {prime}")

            prompt = inp[None, ...]
            sampled = sample(model, prompt, generate_length, force_behavior_cloning = True, meta_controller = meta_controller)

            output = decode_tokens(sampled[0])
            accelerator.print(f"\nGENERATED: {output}\n")

        # save checkpoint at the end of BC phase

        if i == (num_bc_batches - 1):
            accelerator.print(f"saving BC checkpoint to {checkpoint_path}")
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save(checkpoint_path)

    # save discovery checkpoint if we finished discovery phase

    if total_steps > num_bc_batches:
        discovery_checkpoint_path = str(Path(checkpoint_path).with_stem(f"{Path(checkpoint_path).stem}-discovery"))
        accelerator.print(f"saving discovery checkpoint (metacontroller only) to {discovery_checkpoint_path}")
        accelerator.wait_for_everyone()
        unwrapped_meta_controller = accelerator.unwrap_model(meta_controller)
        unwrapped_meta_controller.save(discovery_checkpoint_path)

        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save(checkpoint_path)

if __name__ == '__main__':
    fire.Fire(train)
