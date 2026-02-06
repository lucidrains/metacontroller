# /// script
# dependencies = [
#   "accelerate",
#   "fire",
#   "memmap-replay-buffer>=0.0.23",
#   "metacontroller-pytorch>=0.0.57",
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

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from accelerate import Accelerator
from memmap_replay_buffer import ReplayBuffer
from einops import rearrange

import matplotlib.pyplot as plt
import wandb

from metacontroller import MetaController, Transformer
from metacontroller.transformer_with_resnet import TransformerWithResnet

from babyai_env import get_mission_embedding

import minigrid
import gymnasium as gym

# helpers

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
    input_dir = "babyai-minibosslevel-trajectories",
    env_id = "BabyAI-MiniBossLevel-v0",
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
    wandb_project = "metacontroller-babyai-bc",
    checkpoint_path = "transformer_bc.pt",
    meta_controller_checkpoint_path = "meta_controller_discovery.pt",
    save_steps = 1000,
    state_loss_weight = 1.,
    action_loss_weight = 1.,
    discovery_action_recon_loss_weight = 1.,
    discovery_kl_loss_weight = 1.,
    discovery_switch_loss_weight = 2e-2, # it is low
    target_switch_rate = 0.15,
    max_grad_norm = 1.,
    use_resnet = False,
    condition_on_mission_embed = False,
    mission_embed_dim = 384
):

    def store_checkpoint(step:int | None = None):
        if accelerator.is_main_process:

            if exists(step):
                # Add step to checkpoint filenames
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

    meta_controller = MetaController(dim, target_switch_rate = target_switch_rate)

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

    model = transformer_class(**transformer_kwargs)

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

        is_discovering = (epoch >= cloning_epochs) # discovery phase is BC with metacontroller tuning

        optim = optim_model if not is_discovering else optim_meta_controller

        for batch in progress_bar:
            
            # batch is a NamedTuple (e.g. MemoryMappedBatch)
            # state: (B, T, 7, 7, 3), action: (B, T)

            states = batch['state'].float()
            actions = batch['action'].long()
            episode_lens = batch.get('_lens')
            mission_embeddings = batch.get('mission_embedding')

            if condition_on_mission_embed and exists(mission_embeddings):
                mission_embeddings = mission_embeddings.to(accelerator.device)
            else:
                mission_embeddings = None

            # use resnet18 to embed visual observations

            if use_resnet: 
                states = model.visual_encode(states)
            else: # flatten state: (B, T, 7, 7, 3) -> (B, T, 147)
                states = rearrange(states, 'b t ... -> b t (...)')


            with accelerator.accumulate(model):
                losses, meta_controller_output = model(
                    states,
                    actions,
                    episode_lens = episode_lens,
                    discovery_phase = is_discovering,
                    force_behavior_cloning = not is_discovering,
                    return_meta_controller_output = True,
                    condition = mission_embeddings
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

                if gradient_accumulation_steps is not None: loss /= gradient_accumulation_steps

                # backprop

                accelerator.backward(loss)

                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = max_grad_norm)

                if gradient_accumulation_steps is None or gradient_step % gradient_accumulation_steps == 0:
                    optim.step()
                    optim.zero_grad()

            # log
            
            for key, value in log.items():
                total_losses[key] += value

            if is_discovering: prefix = "discovery_phase"
            else: prefix = "behavior_cloning"

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
    store_checkpoint()

    accelerator.end_training()

if __name__ == "__main__":
    fire.Fire(train)
