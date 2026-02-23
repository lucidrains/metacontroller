# /// script
# dependencies = [
#   "fire",
#   "gymnasium",
#   "gymnasium[other]",
#   "memmap-replay-buffer>=0.0.12",
#   "metacontroller-pytorch>=0.1.0",
#   "minigrid",
#   "tqdm",
#   "wandb",
#   "sentence-transformers"
# ]
# ///
# To train with multiple GPUs:
# 1. accelerate config
# 2. accelerate launch train_babyai.py [args]

from fire import Fire
from pathlib import Path
from functools import partial
from shutil import rmtree
from tqdm import tqdm

import numpy as np
import torch
from torch import cat, tensor, stack, Tensor
from torch.optim import Adam

from einops import rearrange

from torch_einops_utils import maybe, pad_at_dim, lens_to_mask, masked_mean, align_dims_left, pad_sequence, pad_sequence_and_cat

from accelerate import Accelerator

import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv

from babyai_env import (
    create_env,
    get_missions_embeddings,
    transform_to_symbolic
)

from memmap_replay_buffer import ReplayBuffer

from metacontroller.metacontroller import Transformer, MetaController, policy_loss, z_score, extract_grpo_data
from metacontroller.transformer_with_resnet import TransformerWithResnet

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

def divisible_by(num, den):
    return (num % den) == 0


# main

def main(
    npy_seedfile = None,
    env_name = 'BabyAI-BossLevel-v0',
    gradient_accumulation_steps = None,
    num_episodes = int(10e6),
    max_timesteps = 500,
    buffer_size = 1_000,
    render_every_eps = 1_000,
    video_folder = './recordings',
    seed: int | None = None,
    transformer_weights_path: str | None = None,
    meta_controller_weights_path: str | None = None,
    output_meta_controller_path = 'metacontroller_rl_trained.pt',
    use_resnet = False,
    num_epochs = 3,
    lr = 1e-4,
    save_steps = 100,
    batch_size = 16,
    num_groups = 16,
    max_grad_norm = 1.0,
    use_wandb = False,
    wandb_project = 'metacontroller-babyai-rl',
    reject_threshold_cumulative_reward_variance = 0.,
    condition_on_mission_embed = False,
    num_envs = 1,
    env_shared_memory = False,
    env_context = 'spawn'
):

    def store_checkpoint(step:int):
        if accelerator.is_main_process:
            # Add step to checkpoint filenames
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

    # check num_envs

    assert divisible_by(num_envs, num_groups) or divisible_by(num_groups, num_envs), f"one of num_groups ({num_groups}) or num_envs ({num_envs}) must be divisible by the other"
    assert num_groups >= 2, "num_groups must be at least 2 for relative comparison (GRPO)"

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

    env = AsyncVectorEnv([env_make_fn] * num_envs, shared_memory = env_shared_memory, context = env_context)



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

    unwrapped_model = accelerator.unwrap_model(model) if exists(model) else None
    unwrapped_meta_controller = accelerator.unwrap_model(meta_controller)

    # replay buffer

    replay_buffer = ReplayBuffer(
        './replay-data',
        max_episodes = buffer_size,
        max_timesteps = max_timesteps + 1,
        fields = unwrapped_meta_controller.replay_buffer_field_dict,
        meta_fields = dict(advantages = 'float'),
        overwrite = True,
        circular = True
    )

    # rollouts

    num_batch_updates = num_episodes // num_groups

    num_rollout_iterations = max(1, num_groups // num_envs)
    num_groups_per_iteration = max(1, num_envs // num_groups)


    pbar = tqdm(range(num_batch_updates), desc = 'training')

    gradient_step = 0

    for _ in pbar:

        all_states = []
        all_log_probs = []
        all_switch_betas = []
        all_latent_actions = []
        all_step_rewards = []
        all_episode_lens = []

        # seeds for each group (for GRPO relative comparison)

        if num_envs <= num_groups:
            group_seeds = [get_next_seed()]
            env_seeds = group_seeds * num_envs
        else:
            group_seeds = [get_next_seed() for _ in range(num_groups_per_iteration)]
            env_seeds = [s for s in group_seeds for _ in range(num_groups)]

        for _ in range(num_rollout_iterations):

            state, _ = env.reset(seed = env_seeds)
            state['image'] = transform_to_symbolic(state['image'])

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

            dones = torch.zeros(num_envs, dtype = torch.bool)

            all_steps_dones = []

            # rollout parallel trajectories

            for step in range(max_timesteps):

                image = state['image']
                image_tensor = torch.from_numpy(image).float().to(accelerator.device)

                if use_resnet:
                    image_tensor = rearrange(image_tensor, 'b h w c -> b 1 h w c')
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

                iteration_states.append(grpo_data.state)
                iteration_log_probs.append(grpo_data.log_prob)
                iteration_switch_betas.append(grpo_data.switch_beta)
                iteration_latent_actions.append(grpo_data.action)

                next_state, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
                next_state['image'] = transform_to_symbolic(next_state['image'])

                reward_tensor = torch.from_numpy(reward).float().to(accelerator.device)
                iteration_rewards.append(rearrange(reward_tensor, 'b -> b 1'))

                all_steps_dones.append(dones.clone())
                dones |= torch.from_numpy(terminated | truncated)

                if dones.all():
                    break

                state = next_state

            # post-process for each environment

            episode_lens = (~torch.stack(all_steps_dones)).sum(dim = 0)

            cur_states = cat(iteration_states, dim = 1)
            cur_log_probs = cat(iteration_log_probs, dim = 1)
            cur_switch_betas = cat(iteration_switch_betas, dim = 1)
            cur_latent_actions = cat(iteration_latent_actions, dim = 1)
            cur_rewards = cat(iteration_rewards, dim = 1)

            # store environment data in batches per iteration

            all_states.append(cur_states)
            all_log_probs.append(cur_log_probs)
            all_switch_betas.append(cur_switch_betas)
            all_latent_actions.append(cur_latent_actions)
            all_step_rewards.append(cur_rewards)
            all_episode_lens.append(episode_lens)

        # align all sequences across rollout iterations
 
        group_states = pad_sequence_and_cat(all_states, dim = 1, dim_cat = 0)
        group_log_probs = pad_sequence_and_cat(all_log_probs, dim = 1, dim_cat = 0)
        group_switch_betas = pad_sequence_and_cat(all_switch_betas, dim = 1, dim_cat = 0)
        group_latent_actions = pad_sequence_and_cat(all_latent_actions, dim = 1, dim_cat = 0)
        
        group_step_rewards = pad_sequence_and_cat(all_step_rewards, dim = 0, dim_cat = 1)

        episode_lens = cat(all_episode_lens, dim = 0).to(accelerator.device)
        
        # mask rewards after done

        mask = lens_to_mask(episode_lens, group_step_rewards.shape[-1])
        group_step_rewards = group_step_rewards * mask

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
            continue

        # gather across GPUs if multi-GPU

        all_shaped_rewards = accelerator.gather(shaped_rewards)

        # compute advantages via z-score (across all gathered rewards)

        all_advantages = z_score(all_shaped_rewards).float()

        # slice back to local process

        process_index = accelerator.process_index
        num_local_trajectories = shaped_rewards.shape[0]

        group_advantages = all_advantages[process_index * num_local_trajectories: (process_index + 1) * num_local_trajectories]

        # whether to reject group based on switch betas (as it determines the mask for learning)

        if should_reject_group_based_on_switch_betas(group_switch_betas, episode_lens):
            accelerator.print(f'group rejected - switch betas for the entire group does not meet criteria for learning')
            continue

        num_trajectories_collected = max(num_groups, num_envs)

        for i in range(num_trajectories_collected):
            L = episode_lens[i]
            replay_buffer.store_episode(
                states = group_states[i, :L],
                log_probs = group_log_probs[i, :L],
                switch_betas = group_switch_betas[i, :L],
                latent_actions = group_latent_actions[i, :L],
                advantages = group_advantages[i]
            )

        # learn

        if len(replay_buffer) >= buffer_size:
            dl = replay_buffer.dataloader(batch_size = batch_size, shuffle = True)
            dl = accelerator.prepare(dl)

            meta_controller.train()

            for epoch in range(num_epochs):
                for batch in dl:
                    loss = meta_controller.policy_loss(
                        batch['states'],
                        batch['log_probs'],
                        batch['latent_actions'],
                        batch['advantages'],
                        batch['switch_betas'] == 1.,
                        episode_lens = batch['_lens']
                    )

                    # gradient acc

                    if gradient_accumulation_steps != None: loss /= gradient_accumulation_steps

                    accelerator.backward(loss)

                    if gradient_accumulation_steps == None or gradient_step % gradient_accumulation_steps == 0:

                        grad_norm = accelerator.clip_grad_norm_(meta_controller.parameters(), max_grad_norm)

                        optim.step()
                        optim.zero_grad()

                        pbar.set_postfix(
                            epoch = epoch,
                            loss = f'{loss.item():.4f}',
                            grad_norm = f'{grad_norm.item():.4f}',
                            reward = f'{cumulative_rewards.mean().item():.4f}'
                        )

                        accelerator.log({
                            'loss': loss.item(),
                            'grad_norm': grad_norm.item(),
                            'reward': cumulative_rewards.mean().item()
                        })

                    # checkpointing
                    
                    if gradient_step % save_steps == 0:
                        store_checkpoint(gradient_step)

            meta_controller.eval()

            accelerator.print(f'loss: {loss.item():.4f}, grad_norm: {grad_norm.item():.4f}, reward: {cumulative_rewards.mean().item():.4f}')

    env.close()

    # save

    if exists(output_meta_controller_path):
        unwrapped_meta_controller.save(output_meta_controller_path)
        accelerator.print(f'MetaController weights saved to {output_meta_controller_path}')

if __name__ == '__main__':
    Fire(main)
