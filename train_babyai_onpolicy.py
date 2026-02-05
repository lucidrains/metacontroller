# /// script
# dependencies = [
#   "fire",
#   "gymnasium",
#   "gymnasium[other]",
#   "metacontroller-pytorch>=0.0.49",
#   "minigrid",
#   "tqdm",
#   "wandb",
#   "sentence-transformers"
# ]
# ///

from fire import Fire
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
from torch import cat, tensor, stack, Tensor
from torch.optim import Adam

from einops import rearrange

from torch_einops_utils import pad_sequence

from accelerate import Accelerator

from babyai_env import (
    create_env,
    get_missions_embeddings,
    transform_to_symbolic
)
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
from metacontroller.metacontroller import Transformer, MetaController, z_score, extract_grpo_data
from metacontroller.transformer_with_resnet import TransformerWithResnet
from functools import partial

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

# main

def main(
    seed: int | None = None,
    npy_seedfile = None,
    env_name = 'BabyAI-BossLevel-v0',
    num_episodes = int(10e6),
    max_timesteps = 500,
    render_every_eps = 1_000,
    video_folder = './recordings',
    transformer_weights_path: str | None = None,
    meta_controller_weights_path: str | None = None,
    output_meta_controller_path = 'metacontroller_rl_trained.pt',
    use_resnet = False,
    lr = 1e-4,
    save_steps = 100,
    num_groups = 16,
    max_grad_norm = 1.0,
    use_wandb = False,
    wandb_project = 'metacontroller-babyai-rl',
    reject_threshold_cumulative_reward_variance = 0.1,
    condition_on_mission_embed = False
):

    def store_checkpoint(step: int):
        if accelerator.is_main_process:
            meta_controller_checkpoint_path_with_step = output_meta_controller_path.replace('.pt', f'_step_{step}.pt')
            unwrapped_meta_controller = accelerator.unwrap_model(meta_controller)
            unwrapped_meta_controller.save(meta_controller_checkpoint_path_with_step)
            accelerator.print(f"MetaController to {meta_controller_checkpoint_path_with_step}")

    # seed selection priority: 1) fixed seed arg, 2) numpy seedfile, 3) random

    group_seeds = None
    if not exists(seed) and exists(npy_seedfile):
        group_seeds = np.load(npy_seedfile, allow_pickle=True).astype(int).tolist()
        
    seed_index = 0

    def get_next_seed():
        nonlocal seed_index
        if exists(seed):
            return seed
        if group_seeds is not None:
            s = group_seeds[seed_index % len(group_seeds)]
            seed_index += 1
            return s
        return torch.randint(0, 1000000, (1,)).item()

    # accelerator

    accelerator = Accelerator(log_with = 'wandb' if use_wandb else None)

    if use_wandb:
        accelerator.init_trackers(wandb_project)

    # environment

    # environment

    env_make_fn = partial(
        create_env,
        env_name,
        render_mode = 'rgb_array',
        video_folder = video_folder,
        render_every_eps = render_every_eps,
        use_symbolic = False
    )

    env = AsyncVectorEnv([env_make_fn], shared_memory = False, context = 'spawn')

    # load models

    model = None
    if exists(transformer_weights_path):
        weights_path = Path(transformer_weights_path)
        assert weights_path.exists(), f"transformer weights not found at {weights_path}"
        
        transformer_klass = TransformerWithResnet if use_resnet else Transformer

        model = transformer_klass.init_and_load(str(weights_path), strict = False)
        
        model.eval()

    meta_controller = None
    if exists(meta_controller_weights_path):
        weights_path = Path(meta_controller_weights_path)
        assert weights_path.exists(), f"meta controller weights not found at {weights_path}"
        meta_controller = MetaController.init_and_load(str(weights_path), strict = False)
        meta_controller.eval()

    meta_controller = default(meta_controller, getattr(model, 'meta_controller', None))
    assert exists(meta_controller), "MetaController must be present for reinforcement learning"

    # optimizer

    optim = Adam(meta_controller.internal_rl_parameters(), lr = lr)

    # prepare

    model, meta_controller, optim = accelerator.prepare(model, meta_controller, optim)

    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_meta_controller = accelerator.unwrap_model(meta_controller)

    # rollouts

    num_batch_updates = num_episodes // num_groups

    pbar = tqdm(range(num_batch_updates), desc = 'training')

    for gradient_step in pbar:

        all_states = []
        all_log_probs = []
        all_switch_betas = []
        all_latent_actions = []
        all_cumulative_rewards = []
        all_step_rewards = []
        all_episode_lens = []

        # every group has a shared seed (for GRPO relative comparison)

        group_seed = get_next_seed()

        for i in range(num_groups):

            state, _ = env.reset(seed = group_seed)
            state['image'] = transform_to_symbolic(state['image'])

            if condition_on_mission_embed:
                missions = env.get_attr('mission')
                mission_embed = get_missions_embeddings(missions)
                mission_embed = mission_embed.to(accelerator.device)

            cache = None
            past_action_id = None

            states = []
            log_probs = []
            switch_betas = []
            latent_actions = []

            total_reward = 0.
            step_rewards = []
            episode_len = max_timesteps

            # one episode rollout

            for step in range(max_timesteps):

                image = state['image']
                image_tensor = torch.from_numpy(image).float().to(accelerator.device)

                if use_resnet:
                    image_tensor = rearrange(image_tensor, 'b h w c -> b 1 h w c')
                    image_tensor = unwrapped_model.visual_encode(image_tensor)
                else:
                    image_tensor = rearrange(image_tensor, 'b h w c -> b 1 (h w c)')

                if torch.is_tensor(past_action_id):
                    past_action_id = past_action_id.long()

                with torch.no_grad():
                    logits, cache = unwrapped_model(
                        image_tensor,
                        past_action_id,
                        meta_controller = unwrapped_meta_controller,
                        return_cache = True,
                        return_raw_action_dist = True,
                        cache = cache,
                        condition = mission_embed if condition_on_mission_embed else None
                    )

                action = unwrapped_model.action_readout.sample(logits)
                past_action_id = action

                # GRPO collection

                grpo_data = extract_grpo_data(unwrapped_meta_controller, cache)

                states.append(grpo_data.state)
                log_probs.append(grpo_data.log_prob)
                switch_betas.append(grpo_data.switch_beta)
                latent_actions.append(grpo_data.action)

                next_state, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
                next_state['image'] = transform_to_symbolic(next_state['image'])

                total_reward += reward[0]
                step_rewards.append(reward[0])
                done = terminated[0] or truncated[0]

                if done:
                    episode_len = step + 1
                    break

                state = next_state

            # store episode data (concatenate timesteps)
            # each has shape (1, timesteps, ...)

            all_states.append(cat(states, dim=1).squeeze(0))           # (timesteps, state_dim)
            all_log_probs.append(cat(log_probs, dim=1).squeeze(0))     # (timesteps,)
            all_switch_betas.append(cat(switch_betas, dim=1).squeeze(0))  # (timesteps,)
            all_latent_actions.append(cat(latent_actions, dim=1).squeeze(0))  # (timesteps,)

            all_cumulative_rewards.append(tensor(total_reward))
            all_step_rewards.append(tensor(step_rewards))
            all_episode_lens.append(episode_len)

        # compute advantages via z-score (GRPO style)

        cumulative_rewards = stack(all_cumulative_rewards)
        episode_lens = tensor(all_episode_lens)

        # pad step rewards for reward shaping hook

        padded_step_rewards = pad_sequence(all_step_rewards, dim = 0)

        # reward shaping hook

        shaped_rewards = reward_shaping_fn(
            cumulative_rewards,
            padded_step_rewards,
            episode_lens,
            reject_threshold_cumulative_reward_variance = reject_threshold_cumulative_reward_variance
        )

        if not exists(shaped_rewards):
            accelerator.print(f'group rejected - variance of {cumulative_rewards.var().item():.4f} is lower than threshold of {reject_threshold_cumulative_reward_variance}')
            continue

        group_advantages = z_score(shaped_rewards).float()

        # pad episodes to same length for batching

        padded_states, episode_lens = pad_sequence(all_states, dim = 0, return_lens = True)           # (num_groups, max_len, state_dim)
        padded_log_probs = pad_sequence(all_log_probs, dim = 0)     # (num_groups, max_len)
        padded_switch_betas = pad_sequence(all_switch_betas, dim = 0)  # (num_groups, max_len)
        padded_latent_actions = pad_sequence(all_latent_actions, dim = 0)  # (num_groups, max_len)

        # whether to reject group based on switch betas (as it determines the mask for learning)

        if should_reject_group_based_on_switch_betas(padded_switch_betas, episode_lens):
            accelerator.print(f'group rejected - switch betas for the entire group does not meet criteria for learning')
            continue

        # learn on this group directly (on-policy GRPO)

        meta_controller.train()

        loss = meta_controller.policy_loss(
            padded_states.to(accelerator.device),
            padded_log_probs.to(accelerator.device),
            padded_latent_actions.to(accelerator.device),
            group_advantages.to(accelerator.device),
            (padded_switch_betas == 1.).to(accelerator.device),
            episode_lens = episode_lens.to(accelerator.device)
        )

        accelerator.backward(loss)

        grad_norm = accelerator.clip_grad_norm_(meta_controller.parameters(), max_grad_norm)

        optim.step()
        optim.zero_grad()

        meta_controller.eval()

        pbar.set_postfix(
            loss = f'{loss.item():.4f}',
            grad_norm = f'{grad_norm.item():.4f}',
            reward = f'{cumulative_rewards.mean().item():.4f}',
            seed = f'{group_seed}'
        )

        accelerator.log({
            'loss': loss.item(),
            'grad_norm': grad_norm.item(),
            'reward': cumulative_rewards.mean().item(),
            'reward_std': cumulative_rewards.std().item(),
        })

        accelerator.print(f'loss: {loss.item():.4f}, grad_norm: {grad_norm.item():.4f}, reward: {cumulative_rewards.mean().item():.4f}, seed: {group_seed}')

        if gradient_step % save_steps == 0:
            store_checkpoint(gradient_step)

    env.close()

    # save

    if exists(output_meta_controller_path):
        unwrapped_meta_controller.save(output_meta_controller_path)
        accelerator.print(f'MetaController weights saved to {output_meta_controller_path}')

if __name__ == '__main__':
    Fire(main)
