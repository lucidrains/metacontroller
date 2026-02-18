# /// script
# dependencies = [
#   "accelerate",
#   "fire",
#   "memmap-replay-buffer>=0.0.23",
#   "metacontroller-pytorch>=0.1.0",
#   "torch",
#   "einops",
#   "tqdm",
#   "wandb",
#   "gymnasium",
#   "minigrid",
#   "sentence-transformers",
#   "matplotlib"
# ]
# ///

import fire
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from memmap_replay_buffer import ReplayBuffer
from einops import rearrange
from torch_einops_utils import maybe, lens_to_mask

import matplotlib.pyplot as plt
import wandb

from metacontroller import MetaController, Transformer, binary_entropy
from metacontroller.transformer_with_resnet import TransformerWithResnet

from babyai_env import get_mission_embedding

import minigrid
import gymnasium as gym

# helpers

from torch.nn import init
import torch.nn as nn

def initialize_weights_xavier(module):
    if isinstance(module, nn.Linear):
        # Apply Xavier uniform initialization to weights
        init.xavier_uniform_(module.weight)
        # Initialize biases to zero if they exist
        if module.bias is not None:
            init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        # Apply Xavier uniform initialization to convolutional layers
        init.xavier_uniform_(module.weight)
        if module.bias is not None:
            init.zeros_(module.bias)

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d


def visualize_switch_betas(
    switch_betas,      # (B, T-1)
    episode_lens,      # (B,) or None
    gradient_step,
    num_samples = 3
):
    """
    Visualize switch betas for randomly sampled sequences in the batch.
    Logs a single stacked figure to wandb.
    """
    B, T_minus_1 = switch_betas.shape
    
    # randomly sample sequences from the batch
    num_samples = min(num_samples, B)
    sample_indices = np.random.choice(B, size=num_samples, replace=False)
    
    # create figure with num_samples subplots
    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 3 * num_samples))
    fig.suptitle(f'Step {gradient_step} | Switch Betas Visualization', fontsize=10)
    
    # handle single subplot case
    if num_samples == 1:
        axes = [axes]
    
    for i, idx in enumerate(sample_indices):
        # get episode length for this sample (if available)
        if episode_lens is not None:
            ep_len = int(episode_lens[idx].item())
        else:
            ep_len = T_minus_1
        
        # extract data for this sample
        sample_switch_betas = switch_betas[idx, :ep_len-1].detach().cpu()  # (T-1,)
        
        ax = axes[i]
        
        # plot switch betas
        ax.plot(sample_switch_betas.numpy(), label='switch betas', linewidth=2)
        ax.set_xlabel('timesteps')
        ax.set_ylabel(f'switch betas (sample {idx})')
        ax.legend(loc='upper right')
    
    plt.tight_layout()
    
    # log to wandb
    wandb.log({
        f"switch_betas/step_{gradient_step}": wandb.Image(fig)
    }, step=gradient_step)
    
    plt.close(fig)

def train(
    input_dir = None,
    env_id = None,
    cloning_epochs = 10,
    discovery_epochs = 10,
    batch_size = 128,
    gradient_accumulation_steps = None, 
    lr = 1e-4,
    discovery_lr = 1e-4,
    weight_decay = 0.03,
    discovery_weight_decay = 0.03,
    dim = 512,
    depth = 6,
    heads = 8,
    dim_head = 64,
    switch_temperature = 1.,
    use_wandb = False,
    wandb_project = "metacontroller-babyai-bc",
    checkpoint_path = "transformer_bc.pt",
    meta_controller_checkpoint_path = "meta_controller_discovery.pt",
    load_transformer_weights_path = None,
    load_meta_controller_weights_path = None,
    save_steps = 1000,
    state_loss_weight = 1.,
    action_loss_weight = 1.,
    discovery_action_recon_loss_weight = 1.,
    discovery_obs_loss_weight = 1e-3,
    discovery_kl_loss_weight = 0.05,
    discovery_entropy_loss_weight = 0.1,
    discovery_negative_entropy_loss_weight = 0.75,
    discovery_ratio_loss_weight = 1.0,
    discovery_kl_loss_warmup_steps = 0,
    discovery_switch_warmup_steps = 1,
    discovery_switch_lr_scale = 0.1,
    max_grad_norm = 1.,
    use_resnet = False,
    condition_on_mission_embed = False,
    mission_embed_dim = 384,
    lr_schedule = "cosine",  # "cosine" or "constant"
    lr_min_ratio = 0.05,     # minimum LR as fraction of initial
    eval_steps = 1000
):

    def store_checkpoint(step:int = None, is_discovering: bool = False):
        if accelerator.is_main_process:

            if exists(step):
                # Add step to checkpoint filenames
                checkpoint_path_with_step = checkpoint_path.replace('.pt', f'_step_{step}.pt')
                meta_controller_checkpoint_path_with_step = meta_controller_checkpoint_path.replace('.pt', f'_step_{step}.pt')
            else:
                checkpoint_path_with_step = checkpoint_path
                meta_controller_checkpoint_path_with_step = meta_controller_checkpoint_path

            if not is_discovering or not exists(step):
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save(checkpoint_path_with_step)

            if is_discovering:
                unwrapped_meta_controller = accelerator.unwrap_model(meta_controller)
                unwrapped_meta_controller.save(meta_controller_checkpoint_path_with_step)

            accelerator.print(f"Model saved to {checkpoint_path_with_step}, MetaController to {meta_controller_checkpoint_path_with_step}")


    # check for yaml file
    

    # accelerator

    # DDP: some parameters are unused in BC-only steps (e.g. meta_controller) or conditional paths
    accelerator = Accelerator(
        log_with = "wandb" if use_wandb else None,
        kwargs_handlers = [DistributedDataParallelKwargs(find_unused_parameters = True)],
    )

    if use_wandb:
        accelerator.init_trackers(
            wandb_project,
            config = {
                "cloning_epochs": cloning_epochs,
                "discovery_epochs": discovery_epochs,
                "batch_size": batch_size,
                "lr": lr,
                "dim": dim,
                "depth": depth,
                "heads": heads,
                "dim_head": dim_head,
                "env_id": env_id,
                "state_loss_weight": state_loss_weight,
                "action_loss_weight": action_loss_weight,
                "discovery_ratio_loss_weight": discovery_ratio_loss_weight,
                "lr_schedule": lr_schedule,
                "lr_min_ratio": lr_min_ratio if lr_schedule == "cosine" else None,
                "eval_steps": eval_steps,
            }
        )

    # replay buffer and dataloader

    input_path = Path(input_dir)
    assert input_path.exists(), f"Input directory {input_dir} does not exist"

    replay_buffer = ReplayBuffer.from_folder(input_path)
    dataloader = replay_buffer.dataloader(batch_size = batch_size)

    # state shape and action dimension
    # state: (B, T, H, W, C) or (B, T, D)

    state_shape = replay_buffer.shapes['state']
    if use_resnet: state_dim = 256
    else: state_dim = int(torch.tensor(state_shape).prod().item())

    # deduce num_actions from the environment

    from babyai_env import create_env
    temp_env = create_env(env_id)
    num_actions = int(temp_env.action_space.n)
    temp_env.close()

    accelerator.print(f"Detected state_dim: {state_dim}, num_actions: {num_actions} from env: {env_id}")

    # meta controller

    meta_controller = MetaController(
        dim,
        switch_temperature = switch_temperature,
        ratio_loss_weight = discovery_ratio_loss_weight,
        kl_loss_weight = discovery_kl_loss_weight,
        kl_loss_warmup_steps = discovery_kl_loss_warmup_steps,
        apply_kl_loss_weight = False
    )

    # transformer
    
    transformer_class = TransformerWithResnet if use_resnet else Transformer

    transformer_kwargs = dict(
        dim = dim,
        state_embed_readout = dict(num_continuous = state_dim),
        action_embed_readout = dict(num_discrete = num_actions),
        lower_body = dict(depth = depth, heads = heads, attn_dim_head = dim_head),
        upper_body = dict(depth = depth, heads = heads, attn_dim_head = dim_head),
        meta_controller = meta_controller,
        dim_condition = mission_embed_dim if condition_on_mission_embed else None
    )
    if use_resnet: transformer_kwargs["use_layernorm"] = True # resnet won't suffer from grad. accum.

    model = transformer_class(**transformer_kwargs)

    # load pre-trained weights for fine-tuning if given / otherwise xavier init
    
    if exists(load_transformer_weights_path):
        _path = Path(load_transformer_weights_path)
        assert _path.exists(), f"load_transformer_weights_path {load_transformer_weights_path} does not exist"
        if hasattr(model, "load"):
            model.load(str(_path))
        else:
            state = torch.load(_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state, strict=False)
        accelerator.print(f"Loaded transformer weights from {load_transformer_weights_path}")
    else:
        for name, child in model.named_children():
            if name != "meta_controller":
                child.apply(initialize_weights_xavier)
        accelerator.print("Initialized transformer with Xavier")

    if exists(load_meta_controller_weights_path):
        _path = Path(load_meta_controller_weights_path)
        assert _path.exists(), f"load_meta_controller_weights_path {load_meta_controller_weights_path} does not exist"
        if hasattr(meta_controller, "load"):
            meta_controller.load(str(_path))
        else:
            state = torch.load(_path, map_location="cpu", weights_only=True)
            meta_controller.load_state_dict(state, strict=False)
        accelerator.print(f"Loaded meta_controller weights from {load_meta_controller_weights_path}")
    else:
        meta_controller.apply(initialize_weights_xavier)
        accelerator.print("Initialized meta_controller with Xavier")

    # optimizer

    optim_model = AdamW(model.parameters(), lr = lr, weight_decay = weight_decay)

    optim_meta_controller = AdamW([
        {"params": meta_controller.discovery_parameters_non_switching(), "lr": discovery_lr},
        {"params": meta_controller.discovery_parameters_switching_unit(), "lr": 0.0},
    ], weight_decay = discovery_weight_decay)

    # prepare

    model, optim_model, optim_meta_controller, dataloader = accelerator.prepare(model, optim_model, optim_meta_controller, dataloader)

    # optional LR schedule for BC phase
    scheduler_model = None
    if lr_schedule == "cosine" and cloning_epochs > 0:
        num_bc_steps = cloning_epochs * len(dataloader)
        scheduler_model = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim_model, T_max=num_bc_steps, eta_min=lr * lr_min_ratio
        )
        accelerator.print(f"Using cosine LR schedule for BC: {num_bc_steps} steps, eta_min={lr * lr_min_ratio:.2e}")

    # training
    
    old_discovery_obs_loss_weight = discovery_obs_loss_weight
    old_discovery_action_recon_loss_weight = discovery_action_recon_loss_weight

    gradient_step = 0
    for epoch in range(cloning_epochs + discovery_epochs):

        total_losses = defaultdict(float)

        progress_bar = tqdm(dataloader, desc = f"Epoch {epoch}", disable = not accelerator.is_local_main_process)

        is_discovering = (epoch >= cloning_epochs) # discovery phase is BC with metacontroller tuning

        if is_discovering:
            model.train_discovery()
        else:
            model.train()

        optim = optim_model if not is_discovering else optim_meta_controller

        for batch in progress_bar:
            
            # batch is a NamedTuple (e.g. MemoryMappedBatch)
            # state: (B, T, 7, 7, 3), action: (B, T)
            # normalize inputs to be first in [0, 1] by dividing by 255
            # then normalize w.r.t. mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
            states = batch['state'].float()
            states = torch.clamp(states / 255.0, min=0.0, max=1.0)
            states = (states - torch.tensor([0.485, 0.456, 0.406]).to(states.device)) / torch.tensor([0.229, 0.224, 0.225]).to(states.device)

            actions = batch['action'].long()
            episode_lens = batch.get('_lens')
            mission_embeddings = batch.get('mission_embedding')

            if condition_on_mission_embed and exists(mission_embeddings):
                mission_embeddings = mission_embeddings.to(accelerator.device)
            else:
                mission_embeddings = None

            # flatten state: (B, T, 7, 7, 3) -> (B, T, 147)
            if not use_resnet: states = rearrange(states, 'b t ... -> b t (...)')

            with accelerator.accumulate(model):
                losses, meta_controller_output = model(
                    state=states,
                    actions=actions,
                    episode_lens = episode_lens,
                    discovery_phase = is_discovering,
                    force_behavior_cloning = not is_discovering,
                    return_meta_controller_output = True,
                    condition = mission_embeddings
                )

                if is_discovering:
                    obs_loss, action_recon_loss, kl_loss, ratio_loss = losses

                    entropy_loss = binary_entropy(meta_controller_output.switch_beta).mean()

                    switch_mask = maybe(lens_to_mask)(episode_lens, meta_controller_output.switch_beta.shape[1])
                    if exists(switch_mask):
                        n_valid = switch_mask.sum()
                        entropy_loss = (binary_entropy(meta_controller_output.switch_beta) * switch_mask).sum() / n_valid.item()
                        last_hard_switch_density = ((meta_controller_output.switch_beta > 0.5).float() * switch_mask).sum().item() / n_valid.item()
                        last_soft_switch_density = (meta_controller_output.switch_beta * switch_mask).sum().item() / n_valid.item()
                    else:
                        entropy_loss = binary_entropy(meta_controller_output.switch_beta).mean()
                        last_hard_switch_density = (meta_controller_output.switch_beta > 0.5).float().mean().item()
                        last_soft_switch_density = meta_controller_output.switch_beta.mean().item()

                    # zero betas -> feasibility restoration -> losses: (entropy, kl)
                    if last_hard_switch_density < 0.1:
                        entropy_weight = -discovery_negative_entropy_loss_weight
                        discovery_obs_loss_weight = 0
                        discovery_action_recon_loss_weight = 0
                    else:
                        entropy_weight = discovery_entropy_loss_weight
                        discovery_obs_loss_weight = old_discovery_obs_loss_weight
                        discovery_action_recon_loss_weight = old_discovery_action_recon_loss_weight

                    loss = (
                        obs_loss * discovery_obs_loss_weight + 
                        action_recon_loss * discovery_action_recon_loss_weight +
                        # manually multiply kl loss by the warmup weight, since researcher set apply_kl_loss_weight = False in the MetaController init
                        kl_loss * meta_controller_output.kl_loss_weight +
                        entropy_loss * entropy_weight +
                        ratio_loss * discovery_ratio_loss_weight
                    )

                    # LR warmup + permanent scale for switching unit: ramp to discovery_lr * scale over warmup, then stay there
                    warmup_factor = min(1.0, (gradient_step + 1) / discovery_switch_warmup_steps)
                    switch_lr = discovery_lr * discovery_switch_lr_scale * warmup_factor
                    optim_meta_controller.param_groups[1]["lr"] = switch_lr

                    log = dict(
                        obs_loss = obs_loss.item(),
                        action_recon_loss = action_recon_loss.item(),
                        kl_loss = kl_loss.item(),
                        kl_loss_weight = meta_controller_output.kl_loss_weight,
                        entropy_loss = entropy_loss.item(),
                        ratio_loss = ratio_loss.item(),
                        hard_switch_density = last_hard_switch_density,
                        switch_lr_warmup = warmup_factor,
                    )
                else:
                    state_loss, action_loss = losses

                    loss = (
                        state_loss * state_loss_weight +
                        action_loss * action_loss_weight
                    )

                    log = dict(
                        state_loss = state_loss.item(),
                        action_loss = action_loss.item(),
                    )

                # gradient accumulation

                if gradient_accumulation_steps is not None: loss /= gradient_accumulation_steps

                # backprop

                accelerator.backward(loss)

                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = max_grad_norm)

                if gradient_accumulation_steps is None or gradient_step % gradient_accumulation_steps == 0:
                    optim.step()
                    optim.zero_grad()

                    if not is_discovering and scheduler_model is not None:
                        scheduler_model.step()

                    if is_discovering:
                        m = accelerator.unwrap_model(model)
                        m.meta_controller_maybe_increment_kl_loss_step()

            # log on backprop

            if gradient_accumulation_steps is None or gradient_step % gradient_accumulation_steps == 0:
            
                for key, value in log.items():
                    total_losses[key] += value

                if is_discovering: prefix = "discovery_phase"
                else: prefix = "behavior_cloning"

                m = accelerator.unwrap_model(model)
                log_dict = {
                    **log,
                    f"{prefix}_total_loss": loss.item(),
                    f"{prefix}_grad_norm": grad_norm.item()
                }

                if m.running_bc_state_loss is not None:
                    log_dict["running_bc_state_loss"] = m.running_bc_state_loss.squeeze().item()
                if m.running_bc_action_loss is not None:
                    log_dict["running_bc_action_loss"] = m.running_bc_action_loss.squeeze().item()
                if m.running_discovery_state_loss is not None:
                    log_dict["running_discovery_state_loss"] = m.running_discovery_state_loss.squeeze().item()
                if m.running_discovery_action_loss is not None:
                    log_dict["running_discovery_action_loss"] = m.running_discovery_action_loss.squeeze().item()

                if not is_discovering and scheduler_model is not None:
                    log_dict["behavior_cloning_lr"] = scheduler_model.get_last_lr()[0]

                accelerator.log(log_dict, step=gradient_step)

                progress_bar.set_postfix(**log)


            gradient_step += 1

            # checkpoint 

            if gradient_step % eval_steps == 10:
                # visualize switch betas during discovery phase
                if is_discovering and use_wandb and accelerator.is_main_process:
                    visualize_switch_betas(
                        switch_betas = meta_controller_output.switch_beta,
                        episode_lens = episode_lens,
                        gradient_step = gradient_step,
                        num_samples = 3
                    )

        avg_losses = {k: v / len(dataloader) for k, v in total_losses.items()}
        avg_losses_str = ", ".join([f"{k}={v:.4f}" for k, v in avg_losses.items()])
        accelerator.print(f"Epoch {epoch}: {avg_losses_str}")

    # save weights
    accelerator.wait_for_everyone()
    store_checkpoint(None, is_discovering)

    accelerator.end_training()

if __name__ == "__main__":
    fire.Fire(train)
