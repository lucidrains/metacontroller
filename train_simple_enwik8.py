import os
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

from metacontroller.simple_metacontroller import TransformerWithMetacontroller

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

        segments.append(decode_tokens([token]))

    return ''.join(segments)

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
        return_cache = True
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
            return_cache = True
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
    ratio_loss_weight = 2.0,
    latent_ar_loss_weight = 0.1,
    validate_every = 100,
    generate_every = 250,
    prime_length = 16,
    generate_length = 160,
    seq_len = 128,
    dim = 512,
    dim_latent = 64,
    depth = 3,
    heads = 8,
    attn_dim_head = 48,
    target_avg_token_length = 8.,
    residual_stream_dropout = 0.25,
    residual_stream_drop_prob = 0.05,
    pred_loss_to_switch_weight = 0.1,
    cpu = False,
    checkpoint_path = './results-simple-enwik8/train-enwik8.pt',
    enwik8_path = './data/enwik8.gz',
):
    # accelerator

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters = True)
    accelerator = Accelerator(cpu = cpu, kwargs_handlers = [ddp_kwargs])

    # ensure checkpoint directory exists

    checkpoint_dir = Path(checkpoint_path).parent
    checkpoint_dir.mkdir(parents = True, exist_ok = True)

    data_path = Path(enwik8_path)

    if not data_path.exists():
        accelerator.print(f"enwik8 data not found at {enwik8_path}")
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
        dim_latent = dim_latent,
        lower_body = dict(depth = depth, heads = heads, attn_dim_head = attn_dim_head),
        upper_body = dict(depth = depth, heads = heads, attn_dim_head = attn_dim_head),
        emitter_decoder = dict(depth = 1, heads = heads, attn_dim_head = attn_dim_head),
        dim_queries_keys = 256,
        target_avg_token_length = target_avg_token_length,
        residual_stream_dropout = residual_stream_dropout,
        residual_stream_drop_prob = residual_stream_drop_prob,
        pred_loss_to_switch_weight = pred_loss_to_switch_weight,
    )

    # optimize jointly

    optim = AdamW(model.parameters(), lr = learning_rate)

    model, optim, train_loader, val_loader = accelerator.prepare(
        model, optim, train_loader, val_loader
    )

    train_loader = cycle(train_loader)
    val_loader = cycle(val_loader)

    pbar = tqdm.tqdm(range(num_batches), mininterval = 10.0, desc = "training jointly")

    for i in pbar:
        model.train()

        last_loss = 0.
        last_bc_action_loss = 0.
        last_bc_state_loss = 0.
        last_ratio_loss = 0.
        last_latent_ar_loss = 0.
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

            bc_state_loss, bc_action_loss, ratio_loss, latent_ar_loss = losses

            loss = (bc_state_loss + 0.5) * bc_state_loss_weight + (bc_action_loss + 0.5) * bc_action_loss_weight + ratio_loss * ratio_loss_weight + latent_ar_loss * latent_ar_loss_weight

            last_loss = loss.item()
            last_bc_state_loss = bc_state_loss.item()
            last_bc_action_loss = bc_action_loss.item()
            last_ratio_loss = ratio_loss.item()
            last_latent_ar_loss = latent_ar_loss.item()
            last_switch_density = (meta_output.switch_beta > 0.5).float().mean().item()

            accelerator.backward(loss / grad_accum_every)

        accelerator.clip_grad_norm_(model.parameters(), 0.5)
        optim.step()
        optim.zero_grad()

        # logging

        if divisible_by(i, 10):
            log_str = f"{i}: loss: {last_loss:.3f} bc_action: {last_bc_action_loss:.3f} bc_state: {last_bc_state_loss:.3f} ratio: {last_ratio_loss:.3f} latent_ar: {last_latent_ar_loss:.3f} density: {last_switch_density:.3f}"
            tqdm.tqdm.write(log_str)
            pbar.set_postfix(bc_action = f"{last_bc_action_loss:.3f}", density = f"{last_switch_density:.3f}")

        if divisible_by(i, validate_every):
            model.eval()
            with torch.no_grad():
                valid_data = next(val_loader)
                val_state = valid_data[:, :-1]
                val_actions = valid_data[:, 1:]

                losses, meta_output = model(
                    state = val_state,
                    actions = val_actions,
                    return_loss = True
                )

                v_bc_state, v_bc_action, v_ratio, v_latent_ar = losses
                loss_val = (v_bc_state + 0.5) * bc_state_loss_weight + (v_bc_action + 0.5) * bc_action_loss_weight + v_ratio * ratio_loss_weight + v_latent_ar * latent_ar_loss_weight

                segmented_str = visualize_segments(val_state[0], meta_output.switch_beta[0], threshold = 0.5)
                accelerator.print(f"\n\nSEGMENTED: {segmented_str}\n")
                accelerator.print(f"{i}: validation loss: {loss_val.item():.3f} (bc: {v_bc_action.item():.3f})")

        if divisible_by(i, generate_every):
            model.eval()
            inp = random.choice(val_dataset)[:prime_length]
            inp = inp.to(accelerator.device)

            prime = decode_tokens(inp.tolist())
            accelerator.print(f"\n\nPROMPT: {prime}")

            prompt = inp[None, ...]
            sampled, sampled_switch_betas = sample(model, prompt, generate_length)

            sampled_segmented_str = visualize_segments(sampled[0], sampled_switch_betas[0], threshold = 0.5)
            accelerator.print(f"GENERATED: {sampled_segmented_str}\n")

        # save checkpoint

        if divisible_by(i, 1000) and i > 0:
            accelerator.print(f"saving checkpoint to {checkpoint_path}")
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save(checkpoint_path)

    # final save

    accelerator.print(f"saving final checkpoint to {checkpoint_path}")
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save(checkpoint_path)

if __name__ == '__main__':
    fire.Fire(train)
