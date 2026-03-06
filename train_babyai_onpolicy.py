# /// script
# dependencies = [
#   "fire",
#   "gymnasium",
#   "gymnasium[other]",
#   "metacontroller-pytorch>=0.1.0",
#   "torch-einops-utils>=0.0.27",
#   "minigrid",
#   "tqdm",
#   "wandb",
#   "sentence-transformers"
# ]
# ///
# To train with multiple GPUs:
# 1. accelerate config
# 2. accelerate launch train_babyai_onpolicy.py [args]

from fire import Fire
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch import cat, tensor, stack, Tensor
from torch.optim import Adam

import matplotlib.pyplot as plt
import wandb

from einops import rearrange

def set_requires_grad(network: nn.Module, grad_val: bool):
    for param in network.parameters():
        param.requires_grad = grad_val

from torch_einops_utils import pad_sequence, pad_sequence_and_cat, lens_to_mask

from accelerate import Accelerator

from babyai_env import (
    create_env,
    get_missions_embeddings,
)
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
from metacontroller.metacontroller import Transformer, MetaController, z_score, extract_grpo_data
from metacontroller.transformer_with_resnet import TransformerWithResnet
from metacontroller.metacontroller_with_binary_mapper import MetaControllerWithBinaryMapper
from torch.nn.parallel import DistributedDataParallel
from functools import partial

MODALITY_RESNET_RGB = "resnet_rgb"
MODALITY_RAW_RGB = "raw_rgb"
MODALITY_SYMBOLIC = "symbolic"

# Patch x_transformers so checkpoint config can unpickle (Identity was removed/moved in some versions)
try:
    import x_transformers.x_transformers as _xt_module
    if not hasattr(_xt_module, "Identity"):
        _xt_module.Identity = torch.nn.Identity
except Exception:
    pass

# research entry point

def reward_shaping_fn(
    cumulative_rewards: Tensor, # float(num_episodes,)
    all_rewards: Tensor,        # float(num_episodes, max_timesteps)
    episode_lens: Tensor,       # int(num_episodes,)
    reject_threshold_cumulative_reward_variance: float = 0.
) -> Tensor | None:
    """
    researchers can modify this function to engineer rewards
    or return None to reject the entire batch
    """

    if exists(reject_threshold_cumulative_reward_variance):
        if cumulative_rewards.var() < reject_threshold_cumulative_reward_variance:
            return None

    return cumulative_rewards

def should_reject_group_based_on_switch_betas(
    switch_betas: Tensor,
    episode_lens: Tensor
):
    return switch_betas.sum().item() == 0.

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

MINIGRID_ACTION_NAMES = ['left', 'right', 'forward', 'pickup', 'drop', 'toggle', 'done']

def visualize_eval_trajectory(
    unwrapped_model,
    unwrapped_meta_controller,
    env_make_fn,
    modality: str,
    gradient_step: int,
    max_timesteps: int = 500,
    seed: int = 0,
    use_wandb: bool = False,
    accelerator = None,
    condition_on_mission_embed: bool = False,
    fps: int = 4,
):
    device = accelerator.device if exists(accelerator) else 'cpu'

    env = env_make_fn()
    state, _ = env.reset(seed=seed)

    mission = env.unwrapped.mission

    mission_embed = None
    if condition_on_mission_embed:
        mission_embed = get_missions_embeddings([mission]).to(device)

    cache = None
    past_action_id = None

    frames = []
    actions_list = []
    rewards_list = []
    switch_betas_list = []

    for step in range(max_timesteps):
        rgb_frame = env.render()  # full RGB view (H, W, 3) uint8

        image_tensor = torch.from_numpy(state['image']).float().unsqueeze(0).to(device)

        if modality == MODALITY_RESNET_RGB:
            image_tensor = torch.clamp(image_tensor / 255.0, min=0.0, max=1.0)
            image_tensor = (image_tensor - torch.tensor([0.485, 0.456, 0.406], device=device)) / torch.tensor([0.229, 0.224, 0.225], device=device)
            image_tensor = rearrange(image_tensor, 'b h w c -> b 1 h w c')
        elif modality == MODALITY_RAW_RGB:
            image_tensor = torch.clamp(image_tensor / 255.0, min=0.0, max=1.0)
            image_tensor = (image_tensor - torch.tensor([0.485, 0.456, 0.406], device=device)) / torch.tensor([0.229, 0.224, 0.225], device=device)
            image_tensor = rearrange(image_tensor, 'b h w c -> b 1 (h w c)')
        elif modality == MODALITY_SYMBOLIC:
            image_tensor = rearrange(image_tensor, 'b h w c -> b 1 (h w c)')

        if torch.is_tensor(past_action_id):
            past_action_id = past_action_id.long()

        with torch.no_grad():
            logits, cache = unwrapped_model(
                state=image_tensor,
                actions=past_action_id,
                meta_controller=unwrapped_meta_controller,
                return_cache=True,
                return_raw_action_dist=True,
                cache=cache,
                condition=mission_embed if condition_on_mission_embed else None,
            )

        action = unwrapped_model.action_readout.sample(logits)
        past_action_id = action

        grpo_data = extract_grpo_data(unwrapped_meta_controller, cache)
        switch_beta = grpo_data.switch_beta.item()

        action_id = action.item()
        next_state, reward, terminated, truncated, _ = env.step(action_id)

        action_name = MINIGRID_ACTION_NAMES[action_id] if action_id < len(MINIGRID_ACTION_NAMES) else str(action_id)

        frames.append(rgb_frame)
        actions_list.append(action_name)
        rewards_list.append(reward)
        switch_betas_list.append(switch_beta)

        if terminated or truncated:
            break

        state = next_state

    env.close()

    rendered = []
    dpi = 80
    for i, (frame, act, rew, sb) in enumerate(zip(frames, actions_list, rewards_list, switch_betas_list)):
        fig, (ax_img, ax_plot) = plt.subplots(1, 2, figsize=(7, 3.4), dpi=dpi,
                                               gridspec_kw={'width_ratios': [1, 2]})
        fig.suptitle(f'mission: "{mission}"', fontsize=8, y=0.98)

        ax_img.imshow(frame)
        ax_img.set_title(f"a={act}  r={rew:.2f}  β={sb:.2f}", fontsize=9)
        ax_img.set_xticks([])
        ax_img.set_yticks([])

        ax_plot.plot(range(i + 1), switch_betas_list[:i + 1], linewidth=2, color='#1f77b4')
        ax_plot.set_xlim(0, max_timesteps)
        ax_plot.set_ylim(-0.05, 1.05)
        ax_plot.set_xlabel('timesteps')
        ax_plot.set_ylabel('switch betas')

        fig.tight_layout(pad=0.4)
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
        rendered.append(img)
        plt.close(fig)

    video_np = np.stack(rendered)  # (T, H, W, 3)
    video_np = rearrange(video_np, 't h w c -> 1 t c h w')

    if use_wandb and exists(accelerator) and accelerator.is_main_process:
        tracker = accelerator.get_tracker("wandb")
        if exists(tracker):
            tracker.log(
                {"eval/trajectory_video": wandb.Video(video_np, fps=fps, format="mp4")},
                step=gradient_step,
            )


def visualize_switch_betas(
    switch_betas,      # (B, T-1)
    episode_lens,      # (B,) or None
    gradient_step,
    num_samples = 3,
    use_wandb = False,
    accelerator = None
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
    if use_wandb and exists(accelerator):
        tracker = accelerator.get_tracker("wandb")
        if exists(tracker):
            tracker.log({
                f"switch_betas/step_{gradient_step}": wandb.Image(fig)
            }, step=gradient_step)
    
    plt.close(fig)



# main

def main(
    seed: int | None = 456,
    npy_seedfile = None,
    npy_seedfile_seeds_limit = None,
    env_name = 'BabyAI-BossLevel-v0',
    num_episodes = int(10e6),
    max_timesteps = 500,
    render_every_eps = 1_000,
    kl_loss_weight = 0.01,
    video_folder = None,
    transformer_weights_path: str | None = None,
    meta_controller_weights_path: str | None = None,
    output_meta_controller_path = 'metacontroller_rl_trained.pt',
    modality = MODALITY_RAW_RGB,
    lr = 3e-5,
    save_steps = 100,
    eval_steps = 50,
    batch_size = 16,
    max_grad_norm = None,
    use_wandb = False,
    wandb_project = 'metacontroller-babyai-rl',
    reject_threshold_cumulative_reward_variance = None,
    condition_on_mission_embed = False,
    use_binary_mapper = False,
    env_shared_memory = True,
    env_context = 'fork'
):

    torch.manual_seed(seed)

    if not exists(max_grad_norm): max_grad_norm = float('inf')

    def store_checkpoint(step: int):
        if accelerator.is_main_process:
            meta_controller_checkpoint_path_with_step = output_meta_controller_path.replace('.pt', f'_step_{step}.pt')
            unwrapped_meta_controller = accelerator.unwrap_model(meta_controller)
            unwrapped_meta_controller.save(meta_controller_checkpoint_path_with_step)
            accelerator.print(f"MetaController to {meta_controller_checkpoint_path_with_step}")

    # seed selection priority: 1) numpy seedfile, 2) fixed seed arg, 3) random

    group_seeds = None
    if exists(npy_seedfile):
        group_seeds = np.load(npy_seedfile, allow_pickle=True).astype(int).tolist()
        if npy_seedfile_seeds_limit != None:
            group_seeds = group_seeds[:npy_seedfile_seeds_limit]
        
    seed_index = 0

    def get_next_seed():
        nonlocal seed_index
        if group_seeds is not None:
            s = group_seeds[seed_index % len(group_seeds)]
            seed_index += 1
            return s
        if exists(seed):
            return seed
        return torch.randint(0, 1000000, (1,)).item()

    # batch_size = number of simultaneous trajectories (the group) = number of parallel envs
    assert batch_size >= 2, "batch_size must be at least 2 for relative comparison (GRPO)"

    # accelerator

    accelerator = Accelerator(log_with = 'wandb' if use_wandb else None)

    if use_wandb:
        accelerator.init_trackers(wandb_project)

    # environment

    env_make_fn = partial(
        create_env,
        env_name,
        render_mode = 'rgb_array',
        video_folder = video_folder,
        render_every_eps = render_every_eps,
        use_symbolic = (modality == MODALITY_SYMBOLIC)
    )

    env = AsyncVectorEnv([env_make_fn] * batch_size, shared_memory = env_shared_memory, context = env_context)


    # load models

    model = None
    if exists(transformer_weights_path):
        weights_path = Path(transformer_weights_path)
        assert weights_path.exists(), f"transformer weights not found at {weights_path}"
        transformer_klass = TransformerWithResnet if modality == MODALITY_RESNET_RGB else Transformer
        model = transformer_klass.init_and_load(str(weights_path), strict = False)
        model.eval()

    meta_controller = None
    if exists(meta_controller_weights_path):
        weights_path = Path(meta_controller_weights_path)
        assert weights_path.exists(), f"meta controller weights not found at {weights_path}"
        meta_controller_klass = MetaControllerWithBinaryMapper if use_binary_mapper else MetaController
        meta_controller = meta_controller_klass.init_and_load(str(weights_path), strict = False)
        meta_controller.eval()

    meta_controller = default(meta_controller, getattr(model, 'meta_controller', None))
    assert exists(meta_controller), "MetaController must be present for reinforcement learning"

    # optimizer

    optim = Adam(meta_controller.internal_rl_parameters(), lr = lr)

    # prepare

    model, meta_controller, optim = accelerator.prepare(model, meta_controller, optim)

    unwrapped_model = accelerator.unwrap_model(model) if exists(model) else None
    unwrapped_meta_controller = accelerator.unwrap_model(meta_controller)

    # rollouts

    num_batch_updates = num_episodes // batch_size

    pbar = tqdm(range(num_batch_updates), desc = 'training')
    
    print("starting training")
    unwrapped_model.eval()
    unwrapped_meta_controller.train()

    set_requires_grad(unwrapped_model, False)
    set_requires_grad(unwrapped_meta_controller, True)

    group_rejections = 0

    for gradient_step in pbar:

        # one group of batch_size trajectories per gradient step (for GRPO relative comparison)
        group_seed = get_next_seed()
        env_seeds = [group_seed] * batch_size

        state, _ = env.reset(seed = env_seeds)

        if condition_on_mission_embed:
            missions = env.get_attr('mission')
            mission_embed = get_missions_embeddings(missions)
            mission_embed = mission_embed.to(accelerator.device)

        cache = None
        past_action_id = None

        iteration_states = []
        iteration_log_probs = []
        iteration_switch_betas = []
        iteration_latent_actions = []
        iteration_rewards = []

        dones = torch.zeros(batch_size, dtype = torch.bool)

        all_steps_dones = []

        # rollout: batch of batch_size parallel trajectories (one env per trajectory)
        for step in range(max_timesteps):
            # state['image']: (batch_size, H, W, C)
            image = state['image']
            image_tensor = torch.from_numpy(image).float().to(accelerator.device)

            # rgb -> norm and multichannel, ready for resnet
            if modality == MODALITY_RESNET_RGB:
                image_tensor = torch.clamp(image_tensor / 255.0, min=0.0, max=1.0)
                image_tensor = (image_tensor - torch.tensor([0.485, 0.456, 0.406]).to(image_tensor.device)) / torch.tensor([0.229, 0.224, 0.225]).to(image_tensor.device)
                image_tensor = rearrange(image_tensor, 'b h w c -> b 1 h w c')

            # raw rgb -> norm and flatten

            elif modality == MODALITY_RAW_RGB:
                image_tensor = torch.clamp(image_tensor / 255.0, min=0.0, max=1.0)
                image_tensor = (image_tensor - torch.tensor([0.485, 0.456, 0.406]).to(image_tensor.device)) / torch.tensor([0.229, 0.224, 0.225]).to(image_tensor.device)
                image_tensor = rearrange(image_tensor, 'b h w c -> b 1 (h w c)')
            
            # symbolic -> just flatten

            elif modality == MODALITY_SYMBOLIC:
                image_tensor = rearrange(image_tensor, 'b h w c -> b 1 (h w c)')

            # actions

            if torch.is_tensor(past_action_id):
                past_action_id = past_action_id.long()

            with torch.no_grad():
                logits, cache = unwrapped_model(
                    state=image_tensor,
                    actions=past_action_id,
                    meta_controller = unwrapped_meta_controller,
                    return_cache = True,
                    return_raw_action_dist = True,
                    cache = cache,
                    condition = mission_embed if condition_on_mission_embed else None
                )

            action = unwrapped_model.action_readout.sample(logits)
            past_action_id = action

            grpo_data = extract_grpo_data(unwrapped_meta_controller, cache)

            iteration_states.append(grpo_data.state)
            iteration_log_probs.append(grpo_data.log_prob)
            iteration_switch_betas.append(grpo_data.switch_beta)
            iteration_latent_actions.append(grpo_data.action)

            next_state, reward, terminated, truncated, _ = env.step(action.cpu().numpy())

            reward_tensor = torch.from_numpy(reward).float().to(accelerator.device)
            iteration_rewards.append(rearrange(reward_tensor, 'b -> b 1'))

            all_steps_dones.append(dones.clone())
            dones |= torch.from_numpy(terminated | truncated)

            if dones.all():
                break

            state = next_state

        episode_lens = (~torch.stack(all_steps_dones)).sum(dim = 0).to(accelerator.device)

        cur_states = cat(iteration_states, dim = 1)
        cur_log_probs = cat(iteration_log_probs, dim = 1)
        cur_switch_betas = cat(iteration_switch_betas, dim = 1)
        cur_latent_actions = cat(iteration_latent_actions, dim = 1)
        cur_rewards = cat(iteration_rewards, dim = 1)

        # pack group data (single rollout → one group of batch_size trajectories)

        group_states = cur_states
        group_log_probs = cur_log_probs
        group_switch_betas = cur_switch_betas
        group_latent_actions = cur_latent_actions
        
        group_step_rewards = cur_rewards

        # mask rewards after done

        mask = lens_to_mask(episode_lens, group_step_rewards.shape[-1])
        group_step_rewards = group_step_rewards * mask

        # compute advantages via z-score (GRPO style)

        cumulative_rewards = group_step_rewards.sum(dim = -1)

        # reward shaping hook

        shaped_rewards = reward_shaping_fn(
            cumulative_rewards,
            group_step_rewards,
            episode_lens,
            reject_threshold_cumulative_reward_variance = reject_threshold_cumulative_reward_variance
        )

        if not exists(shaped_rewards):
            accelerator.print(f'group rejected - variance of {cumulative_rewards.var().item():.4f} is lower than threshold of {reject_threshold_cumulative_reward_variance}')
            group_rejections += 1
            continue

        # gather across GPUs if multi-GPU

        all_shaped_rewards = accelerator.gather(shaped_rewards)

        # compute advantages via z-score (GRPO style)

        all_advantages = z_score(all_shaped_rewards).float()

        # slice back to local process

        process_index = accelerator.process_index
        num_local_trajectories = shaped_rewards.shape[0]

        group_advantages = all_advantages[process_index * num_local_trajectories: (process_index + 1) * num_local_trajectories]

        if torch.any(torch.isnan(group_advantages)):
            accelerator.print(f'group rejected - advantages contained NaNs')
            group_rejections += 1
            continue

        # whether to reject group based on switch betas (as it determines the mask for learning)

        if should_reject_group_based_on_switch_betas(group_switch_betas, episode_lens):
            accelerator.print(f'group rejected - switch betas for the entire group does not meet criteria for learning')
            group_rejections += 1
            continue

        # learn on this group directly (on-policy GRPO)

        # meta_controller.train_internal_rl(eval_rest = True)

        policy_loss, kl_loss = unwrapped_meta_controller.policy_loss(
            group_states,
            group_log_probs,
            group_latent_actions,
            group_advantages,
            group_switch_betas,
            episode_lens = episode_lens,
            kl_loss_weight=kl_loss_weight,
            eps_clip=0.2,
            return_kl_loss=True
        )

        loss = (
            policy_loss 
            + kl_loss
        )

        accelerator.backward(loss)

        grad_norm = accelerator.clip_grad_norm_(meta_controller.parameters(), max_grad_norm)

        optim.step()
        optim.zero_grad()

        # meta_controller.eval()

        pbar.set_postfix(
            loss = f'{loss.item():.4f}',
            grad_norm = f'{grad_norm.item():.4f}',
            reward = f'{cumulative_rewards.mean().item():.4f}',
            seeds = f'{group_seeds}'
        )

        accelerator.log({
            'loss': loss.item(),
            'policy_loss': policy_loss.item(),
            'kl_loss': kl_loss.item(),
            'grad_norm': grad_norm.item(),
            'reward': cumulative_rewards.mean().item(),
            'reward_std': cumulative_rewards.std().item(),
            'switch_density': group_switch_betas.mean().item(),
            'group_rejections': group_rejections,
            'env_seed': group_seed,
        }, step=gradient_step)

        accelerator.print(f'loss: {loss.item():.4f}, grad_norm: {grad_norm.item():.4f}, reward: {cumulative_rewards.mean().item():.4f}, seeds: {group_seeds}')

        if gradient_step % save_steps == 0:
            store_checkpoint(gradient_step)

        if gradient_step % eval_steps == 0 and gradient_step > 0:
            visualize_switch_betas(
                switch_betas = group_switch_betas,
                episode_lens = episode_lens,
                gradient_step = gradient_step,
                num_samples = 3,
                use_wandb = use_wandb,
                accelerator = accelerator
            )

            visualize_eval_trajectory(
                unwrapped_model = unwrapped_model,
                unwrapped_meta_controller = unwrapped_meta_controller,
                env_make_fn = env_make_fn,
                modality = modality,
                gradient_step = gradient_step,
                max_timesteps = max_timesteps,
                seed = group_seed,
                use_wandb = use_wandb,
                accelerator = accelerator,
                condition_on_mission_embed = condition_on_mission_embed,
            )

    env.close()

    # save

    if exists(output_meta_controller_path):
        unwrapped_meta_controller.save(output_meta_controller_path)
        accelerator.print(f'MetaController weights saved to {output_meta_controller_path}')

if __name__ == '__main__':
    Fire(main)
