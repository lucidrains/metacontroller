# /// script
# dependencies = [
#   "metacontroller-pytorch>=0.2.17",
#   "accelerate",
#   "fire",
#   "torch",
#   "einops",
#   "tqdm",
#   "numpy",
#   "jax"
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
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from accelerate import Accelerator
from einops import rearrange

from metacontroller import MetaController, Transformer, binary_entropy
from metacontroller.metacontroller_with_binary_mapper import MetaControllerWithBinaryMapper

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

def visualize_segments(
    tokens: Tensor,
    switch_betas: Tensor,
    delimiter = " \u275A ",
    threshold = 0.5
):
    tokens_list = tokens.tolist()
    switches = (switch_betas >= threshold).tolist()

    segments = []
    curr_segment = []

    for i in range(len(switches)):
        curr_segment.append(tokens_list[i])
        if switches[i]:
            segments.append(decode_tokens(curr_segment))
            curr_segment = []

    # Append any remaining tokens
    curr_segment.extend(tokens_list[len(switches):])

    if curr_segment:
        segments.append(decode_tokens(curr_segment))

    return delimiter.join(segments)

# sampling

@torch.no_grad()
def sample(
    model,
    prompt: Tensor,
    seq_len: int,
    temperature = 1.,
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
        force_behavior_cloning = True,
        meta_controller = meta_controller
    )

    next_action = next_state = model.action_readout.sample(action_dist[:, -1:], temperature = temperature)

    state = torch.cat((state, next_state), dim = -1)

    for _ in range(sample_num_times):

        (action_dist, _), next_cache = model(
            state = state[:, -1:],
            actions = state[:, -1:],
            cache = cache,
            return_state_action_cache = True,
            force_behavior_cloning = True,
            meta_controller = meta_controller
        )

        next_state = model.action_readout.sample(action_dist[:, -1:], temperature = temperature)

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

# train function

def train(
    num_bc_batches = 5000,
    num_discovery_batches = 5000,
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
    prime_length = 128,
    generate_length = 512,
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
    dim_code_bits = 4,
    switching_unit_type = 'gru',
    dim_queries_keys = 256,
    boundary_threshold = 0.5,
    switching_unit_decoder_heads = 8,
    switching_unit_decoder_attn_dim_head = 64,
    kl_loss_threshold = 0.1,
    switch_temperature = 0.1,
    discovery_phase = False,
    sequential_latent_action_selection = False,
    discovery_hard_switch = False,
    cpu = False,
    checkpoint_path = './results-enwik8/train-enwik8.pt',
    enwik8_path = './data/enwik8.gz',
):
    # accelerator

    accelerator = Accelerator(cpu = cpu)

    # ensure checkpoint directory exists

    checkpoint_dir = Path(checkpoint_path).parent
    checkpoint_dir.mkdir(parents = True, exist_ok = True)

    # hyperparameter summary

    accelerator.print(f"""
{'='*50}
Training enwik8
{'='*50}
Discovery Phase:    {discovery_phase}
BC Steps:           {num_bc_batches}
Discovery Steps:    {num_discovery_batches}
Batch Size:         {batch_size}
Grad Accum Every:   {grad_accum_every}
BC Learning Rate:   {learning_rate}
Disc Learning Rate: {discovery_learning_rate}
BC Loss Weights:    state: {bc_state_loss_weight} action: {bc_action_loss_weight}
Disc Loss Weights:  state: {discovery_state_loss_weight} action: {discovery_action_loss_weight} kl: {discovery_kl_loss_weight} kl_warmup: {discovery_kl_loss_warmup_steps} entropy: {discovery_entropy_loss_weight} (hinge: {discovery_entropy_loss_threshold}) neg_entropy: {discovery_negative_entropy_loss_weight} ratio: {ratio_loss_weight} -> {ratio_loss_final_weight}
Seq Len:            {seq_len}
CPU:                {cpu}
Model Dim:          {dim}
MC Dim:             {dim_meta_controller}
Latent Dim:         {dim_latent}
Sequential Selection: {sequential_latent_action_selection}
Disc Hard Switch:   {discovery_hard_switch}
Binary Mapper:      True (bits: {dim_code_bits} type: {switching_unit_type} qk_dim: {dim_queries_keys} thresh: {boundary_threshold} kl_thresh: {kl_loss_threshold} temp: {switch_temperature})
Depth:              {depth}
Heads:              {heads}
Target Seg Len:     {target_temporal_segment_len}
Hypernet Rank:      {hypernetwork_low_rank}
Warmup Steps:       {discovery_warmup_steps}
Checkpoint Path:    {checkpoint_path}
{'='*50}
""")

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
        meta_controller = MetaController(
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
            sequential_latent_action_selection = sequential_latent_action_selection,
            hard_switch = discovery_hard_switch
        )
    else:
        meta_controller = MetaControllerWithBinaryMapper(
            dim_model = dim,
            dim_meta_controller = dim_meta_controller,
            dim_code_bits = dim_code_bits,
            switching_unit_type = switching_unit_type,
            dim_queries_keys = dim_queries_keys,
            boundary_threshold = boundary_threshold,
            switching_unit_decoder_kwargs = dict(
                heads = switching_unit_decoder_heads,
                attn_dim_head = switching_unit_decoder_attn_dim_head,
                polar_pos_emb = True
            ),
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
            sequential_latent_action_selection = sequential_latent_action_selection,
            hard_switch = discovery_hard_switch
        )

    model = Transformer(
        dim = dim,
        state_embed_readout = dict(num_discrete = 256),
        action_embed_readout = dict(num_discrete = 256),
        lower_body = dict(depth = depth, heads = heads, attn_dim_head = attn_dim_head),
        upper_body = dict(depth = depth, heads = heads, attn_dim_head = attn_dim_head),
        meta_controller = None
    )

    # load checkpoint if it exists or if we are in discovery phase

    if discovery_phase or Path(checkpoint_path).exists():
        if Path(checkpoint_path).exists():
            accelerator.print(f"loading checkpoint from {checkpoint_path}")
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.load(checkpoint_path, strict = False)
        elif discovery_phase:
            accelerator.print(f"warning: discovery phase requested but {checkpoint_path} not found. training from scratch.")

    # optimizer

    bc_optim = AdamW(model.parameters(), lr = learning_rate)

    discovery_optim = AdamW(meta_controller.discovery_parameters(), lr = discovery_learning_rate)

    model, meta_controller, bc_optim, discovery_optim, train_loader, val_loader = accelerator.prepare(
        model, meta_controller, bc_optim, discovery_optim, train_loader, val_loader
    )

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

    start_step = num_bc_batches if discovery_phase else 0
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
                meta_controller = meta_controller
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
            sampled = sample(model, prompt, generate_length, meta_controller = meta_controller)
            
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

if __name__ == '__main__':
    fire.Fire(train)
