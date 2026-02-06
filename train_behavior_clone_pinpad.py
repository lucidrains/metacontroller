# /// script
# dependencies = [
#   "accelerate",
#   "fire",
#   "memmap-replay-buffer>=0.0.23",
#   "metacontroller-pytorch>=0.0.49",
#   "torch",
#   "einops",
#   "tqdm",
#   "wandb",
#   "gymnasium",
#   "minigrid",
#   "matplotlib"
# ]
# ///

import fire
from tqdm import tqdm
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW

from accelerate import Accelerator
from memmap_replay_buffer import ReplayBuffer
from einops import rearrange

import matplotlib.pyplot as plt
import wandb

from metacontroller import MetaController, Transformer

import gymnasium as gym
from gymnasium.envs.registration import register

# Import PinPad environment classes from gather_pinpad_trajs.py
from gather_pinpad_trajs import PinPad, OneHotFullyObsWrapper

# ============================================================================
# Helpers
# ============================================================================

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def create_pinpad_env(env_id, num_objects, obj_seq, room_size, num_rows, num_cols):
    """Register and create a PinPad environment."""
    if env_id in gym.envs.registry:
        del gym.envs.registry[env_id]
    
    register(
        id=env_id,
        entry_point=PinPad,
        kwargs={
            "num_objects": num_objects,
            "obj_seq": obj_seq,
            "room_size": room_size,
            "num_rows": num_rows,
            "num_cols": num_cols
        },
    )
    
    env = gym.make(env_id, obj_seq=obj_seq)
    env = OneHotFullyObsWrapper(env)
    return env


def visualize_switch_betas_vs_labels(
    switch_betas,      # (B, T-1, C)
    labels,            # (B, T)
    episode_lens,      # (B,) or None
    gradient_step,
    num_samples=3,
    num_random_dims=2
):
    """
    Visualize switch betas vs GT labels for randomly sampled sequences in the batch.
    Logs a single stacked figure to wandb.
    """
    B, T_minus_1, C = switch_betas.shape
    
    # randomly sample sequences from the batch
    num_samples = min(num_samples, B)
    sample_indices = np.random.choice(B, size=num_samples, replace=False)
    
    # create figure with 2 * num_samples subplots (labels + switch_betas for each sample)
    fig, axes = plt.subplots(2 * num_samples, 1, figsize=(12, 3 * num_samples), sharex=False)
    fig.suptitle(f'Step {gradient_step} | Note: -1 in labels = explore (no specific subgoal)', fontsize=10)
    
    for i, idx in enumerate(sample_indices):
        # get episode length for this sample (if available)
        if episode_lens is not None:
            ep_len = int(episode_lens[idx].item())
        else:
            ep_len = T_minus_1
        
        # extract data for this sample
        sample_switch_betas = switch_betas[idx, :ep_len-1, :].detach().cpu()  # (T-1, C)
        sample_labels = labels[idx, :ep_len].cpu()  # (T,)
        
        # compute mean across final dimension
        switch_betas_mean = sample_switch_betas.mean(dim=-1)  # (T-1,)
        
        # randomly sample dimensions
        random_dims = np.random.choice(C, size=min(num_random_dims, C), replace=False)
        
        # get axes for this sample pair
        ax1 = axes[2 * i]      # labels
        ax2 = axes[2 * i + 1]  # switch betas
        
        # top plot: labels (align with switch_betas by using labels[:-1])
        ax1.plot(sample_labels[:-1].numpy(), label=f'labels (sample {idx})', color='red')
        ax1.set_ylabel('labels')
        ax1.legend(loc='upper right')
        
        # bottom plot: switch betas
        ax2.plot(switch_betas_mean.numpy(), label='switch betas mean', linewidth=2)
        for dim in random_dims:
            ax2.plot(sample_switch_betas[:, dim].numpy(), label=f'switch betas dim={dim}', alpha=0.7)
        ax2.set_xlabel('timesteps')
        ax2.set_ylabel('switch betas')
        ax2.legend(loc='upper right')
    
    plt.tight_layout()
    
    # log to wandb
    wandb.log({
        f"switch_betas_vs_labels/step_{gradient_step}": wandb.Image(fig)
    }, step=gradient_step)
    
    plt.close(fig)


# ============================================================================
# Training
# ============================================================================

def train(
    input_dir = "pinpad_demonstrations",
    env_id = "PinPad-v0",
    # PinPad environment parameters (must match the demo collection settings)
    num_objects = 4,
    default_obj_seq = [0,1,2],  # placeholder for env registration
    room_size = 8,
    num_rows = 1,
    num_cols = 1,
    # Training parameters
    cloning_epochs = 10,
    discovery_epochs = 10,
    batch_size = 128,
    gradient_accumulation_steps = None, 
    lr = 1e-4,
    discovery_lr = 1e-4,
    weight_decay = 0.03,
    discovery_weight_decay = 0.03,
    dim = 512,
    depth = 2,
    heads = 8,
    dim_head = 64,
    use_wandb = False,
    wandb_project = "metacontroller-pinpad-bc",
    checkpoint_path = "transformer_pinpad_bc.pt",
    meta_controller_checkpoint_path = "meta_controller_pinpad_discovery.pt",
    save_steps = 1000,
    state_loss_weight = 1.,
    action_loss_weight = 1.,
    discovery_action_recon_loss_weight = 1.,
    discovery_kl_loss_weight = 1.,
    discovery_switch_loss_weight = 2e-2, # it is low
    target_switch_rate = 0.15,
    max_grad_norm = 1.,
):

    def store_checkpoint(step: int | None = None):
        if accelerator.is_main_process:
            if exists(step):
                checkpoint_path_with_step = checkpoint_path.replace('.pt', f'_step_{step}.pt')
                meta_controller_checkpoint_path_with_step = meta_controller_checkpoint_path.replace('.pt', f'_step_{step}.pt')
            else:
                checkpoint_path_with_step = checkpoint_path
                meta_controller_checkpoint_path_with_step = meta_controller_checkpoint_path

            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save(checkpoint_path_with_step)

            unwrapped_meta_controller = accelerator.unwrap_model(meta_controller)
            unwrapped_meta_controller.save(meta_controller_checkpoint_path_with_step)

            accelerator.print(f"Model saved to {checkpoint_path_with_step}, MetaController to {meta_controller_checkpoint_path_with_step}")

    # accelerator

    accelerator = Accelerator(log_with = "wandb" if use_wandb else None)

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
                "action_loss_weight": action_loss_weight
            }
        )

    # replay buffer and dataloader

    input_path = Path(input_dir)
    assert input_path.exists(), f"Input directory {input_dir} does not exist"

    replay_buffer = ReplayBuffer.from_folder(input_path)
    dataloader = replay_buffer.dataloader(batch_size = batch_size)

    # state shape and action dimension
    # state: (B, T, H, W, C) where C is num_channels (one-hot encoded)

    state_shape = replay_buffer.shapes['state']
    state_dim = int(torch.tensor(state_shape).prod().item())  # H * W * C

    # deduce num_actions from a temporary environment

    temp_env = create_pinpad_env(env_id, num_objects, default_obj_seq, room_size, num_rows, num_cols)
    num_actions = int(temp_env.action_space.n)
    temp_env.close()

    accelerator.print(f"Detected state_dim: {state_dim}, num_actions: {num_actions} from env: {env_id}")
    accelerator.print(f"State shape from replay buffer: {state_shape}")

    # meta controller

    meta_controller = MetaController(dim, target_switch_rate = target_switch_rate)

    # transformer (no ResNet for one-hot encoded states)

    model = Transformer(
        dim = dim,
        state_embed_readout = dict(num_continuous = state_dim),
        action_embed_readout = dict(num_discrete = num_actions),
        lower_body = dict(depth = depth, heads = heads, attn_dim_head = dim_head),
        upper_body = dict(depth = depth, heads = heads, attn_dim_head = dim_head),
        meta_controller = meta_controller,
    )

    # optimizer

    optim_model = AdamW(model.parameters(), lr = lr, weight_decay = weight_decay)
    optim_meta_controller = AdamW(meta_controller.discovery_parameters(), lr = discovery_lr, weight_decay = discovery_weight_decay)

    # prepare

    model, optim_model, optim_meta_controller, dataloader = accelerator.prepare(model, optim_model, optim_meta_controller, dataloader)

    # training
    gradient_step = 0
    for epoch in range(cloning_epochs + discovery_epochs):

        model.train()
        from collections import defaultdict
        total_losses = defaultdict(float)

        progress_bar = tqdm(dataloader, desc = f"Epoch {epoch}", disable = not accelerator.is_local_main_process)

        is_discovering = (epoch >= cloning_epochs)  # discovery phase is BC with metacontroller tuning

        optim = optim_model if not is_discovering else optim_meta_controller

        for batch in progress_bar:
            
            # batch fields: state (B, T, H, W, C), action (B, T), label (B, T)
            # state is one-hot encoded: (B, T, H, W, num_channels)

            states = batch['state'].float()
            actions = batch['action'].long()
            labels = batch['label'].long()
            episode_lens = batch.get('_lens')

            # flatten state: (B, T, H, W, C) -> (B, T, H*W*C)
            states = rearrange(states, 'b t ... -> b t (...)')

            with accelerator.accumulate(model):
                losses, meta_controller_output = model(
                    states,
                    actions,
                    episode_lens = episode_lens,
                    discovery_phase = is_discovering,
                    force_behavior_cloning = not is_discovering,
                    return_meta_controller_output = True,
                )

                if is_discovering:
                    action_recon_loss, kl_loss, switch_loss = losses

                    loss = (
                        action_recon_loss * discovery_action_recon_loss_weight +
                        kl_loss * discovery_kl_loss_weight +
                        switch_loss * discovery_switch_loss_weight
                    )

                    log = dict(
                        action_recon_loss = action_recon_loss.item(),
                        kl_loss = kl_loss.item(),
                        switch_loss = switch_loss.item(),
                        switch_density = meta_controller_output.switch_beta.mean().item()
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

                if gradient_accumulation_steps is not None: 
                    loss /= gradient_accumulation_steps

                # backprop

                accelerator.backward(loss)

                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = max_grad_norm)

                if gradient_accumulation_steps is None or gradient_step % gradient_accumulation_steps == 0:
                    optim.step()
                    optim.zero_grad()

            # log
            
            for key, value in log.items():
                total_losses[key] += value

            if is_discovering: 
                prefix = "discovery_phase"
            else: 
                prefix = "behavior_cloning"

            accelerator.log({
                **log,
                f"{prefix}_total_loss": loss.item(),
                f"{prefix}_grad_norm": grad_norm.item()
            })

            progress_bar.set_postfix(**log)
            gradient_step += 1

            # checkpoint 

            if gradient_step % save_steps == 0:
                accelerator.wait_for_everyone()
                store_checkpoint(gradient_step)

                # visualize switch betas vs GT labels during discovery phase
                if is_discovering and use_wandb and accelerator.is_main_process:
                    visualize_switch_betas_vs_labels(
                        switch_betas=meta_controller_output.switch_beta,
                        labels=batch['label'],
                        episode_lens=episode_lens,
                        gradient_step=gradient_step,
                        num_samples=3,
                        num_random_dims=2
                    )

        avg_losses = {k: v / len(dataloader) for k, v in total_losses.items()}
        avg_losses_str = ", ".join([f"{k}={v:.4f}" for k, v in avg_losses.items()])
        accelerator.print(f"Epoch {epoch}: {avg_losses_str}")

    # save weights
    accelerator.wait_for_everyone()
    store_checkpoint()

    accelerator.end_training()

if __name__ == "__main__":
    fire.Fire(train)
