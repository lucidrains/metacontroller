# /// script
# dependencies = [
#   "fire",
#   "gymnasium",
#   "gymnasium[other]",
#   "memmap-replay-buffer>=0.0.12",
#   "metacontroller-pytorch",
#   "minigrid",
#   "tqdm"
# ]
# ///

from fire import Fire
from pathlib import Path
from functools import partial
from shutil import rmtree
from tqdm import tqdm

import torch
from torch import cat, tensor, stack
from torch.optim import Adam

from einops import rearrange

from accelerate import Accelerator

from babyai_env import create_env
from memmap_replay_buffer import ReplayBuffer
from metacontroller.metacontroller import Transformer, MetaController, policy_loss, z_score
from metacontroller.transformer_with_resnet import TransformerWithResnet

# research entry point

def reward_shaping_fn(
    cumulative_rewards: torch.Tensor,
    all_rewards: torch.Tensor,
    episode_lens: torch.Tensor
) -> torch.Tensor | None:
    """
    researchers can modify this function to engineer rewards
    or return None to reject the entire batch
    
    cumulative_rewards: (num_episodes,)
    all_rewards: (num_episodes, max_timesteps)
    episode_lens: (num_episodes,)
    """
    return cumulative_rewards

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# main

def main(
    env_name: str = 'BabyAI-BossLevel-v0',
    num_episodes: int = int(10e6),
    max_timesteps: int = 500,
    buffer_size: int = 5_000,
    render_every_eps: int = 1_000,
    video_folder: str = './recordings',
    seed: int | None = None,
    transformer_weights_path: str | None = None,
    meta_controller_weights_path: str | None = None,
    output_meta_controller_path: str = 'metacontroller_rl_trained.pt',
    use_resnet: bool = False,
    lr: float = 1e-4,
    num_groups: int = 16,
    max_grad_norm: float = 1.0,
    use_wandb: bool = False,
    wandb_project: str = 'metacontroller-babyai-rl'
):
    # accelerator

    accelerator = Accelerator(log_with = 'wandb' if use_wandb else None)

    if use_wandb:
        accelerator.init_trackers(wandb_project)

    # environment

    env = create_env(
        env_name,
        render_mode = 'rgb_array',
        video_folder = video_folder,
        render_every_eps = render_every_eps
    )

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

    # replay buffer

    replay_buffer = ReplayBuffer(
        './replay-data',
        max_episodes = buffer_size,
        max_timesteps = max_timesteps + 1,
        fields = meta_controller.replay_buffer_field_dict,
        meta_fields = dict(advantages = 'float'),
        overwrite = True,
        circular = True
    )

    # rollouts

    num_batch_updates = num_episodes // num_groups

    pbar = tqdm(range(num_batch_updates), desc = 'training')

    for _ in pbar:

        all_episodes = []
        all_cumulative_rewards = []
        all_step_rewards = []
        all_episode_lens = []

        group_seed = default(seed, torch.randint(0, 1000000, (1,)).item())

        for _ in range(num_groups):

            state, *_ = env.reset(seed = group_seed)

            cache = None
            past_action_id = None

            states = []
            log_probs = []
            switch_betas = []
            latent_actions = []

            total_reward = 0.
            step_rewards = []
            episode_len = max_timesteps

            for step in range(max_timesteps):

                image = state['image']
                image_tensor = torch.from_numpy(image).float().to(accelerator.device)

                if use_resnet:
                    image_tensor = rearrange(image_tensor, 'h w c -> 1 1 h w c')
                    image_tensor = model.visual_encode(image_tensor)
                else:
                    image_tensor = rearrange(image_tensor, 'h w c -> 1 1 (h w c)')

                if torch.is_tensor(past_action_id):
                    past_action_id = past_action_id.long()

                with torch.no_grad():
                    logits, cache = unwrapped_model(
                        image_tensor,
                        past_action_id,
                        meta_controller = unwrapped_meta_controller,
                        return_cache = True,
                        return_raw_action_dist = True,
                        cache = cache
                    )

                action = unwrapped_model.action_readout.sample(logits)
                past_action_id = action
                action = action.squeeze()

                # GRPO collection

                meta_output = cache.prev_hiddens.meta_controller
                old_log_probs = unwrapped_meta_controller.log_prob(meta_output.action_dist, meta_output.actions)

                states.append(meta_output.input_residual_stream)
                log_probs.append(old_log_probs)
                switch_betas.append(meta_output.switch_beta)
                latent_actions.append(meta_output.actions)

                next_state, reward, terminated, truncated, *_ = env.step(action.cpu().numpy())

                total_reward += reward
                step_rewards.append(reward)
                done = terminated or truncated

                if done:
                    episode_len = step + 1
                    break

                state = next_state

            # store episode

            all_episodes.append((
                cat(states, dim = 1).squeeze(0),
                cat(log_probs, dim = 1).squeeze(0),
                cat(switch_betas, dim = 1).squeeze(0),
                cat(latent_actions, dim = 1).squeeze(0)
            ))

            all_cumulative_rewards.append(tensor(total_reward))
            all_step_rewards.append(tensor(step_rewards))
            all_episode_lens.append(episode_len)

        # compute advantages

        cumulative_rewards = stack(all_cumulative_rewards)
        episode_lens = tensor(all_episode_lens)

        # pad step rewards

        max_len = max(all_episode_lens)
        padded_step_rewards = torch.zeros(num_episodes, max_len)

        for i, (rewards, length) in enumerate(zip(all_step_rewards, all_episode_lens)):
            padded_step_rewards[i, :length] = rewards

        # reward shaping hook

        shaped_rewards = reward_shaping_fn(cumulative_rewards, padded_step_rewards, episode_lens)

        if not exists(shaped_rewards):
            continue

        group_advantages = z_score(shaped_rewards)

        group_states, group_log_probs, group_switch_betas, group_latent_actions = zip(*all_episodes)

        for states, log_probs, switch_betas, latent_actions, advantages in zip(group_states, group_log_probs, group_switch_betas, group_latent_actions, group_advantages):
            replay_buffer.store_episode(
                states = states,
                log_probs = log_probs,
                switch_betas = switch_betas,
                latent_actions = latent_actions,
                advantages = advantages
            )

        # learn

        if len(replay_buffer) >= buffer_size:
            dl = replay_buffer.dataloader(batch_size = num_groups)
            dl = accelerator.prepare(dl)

            meta_controller.train()

            batch = next(iter(dl))

            loss = meta_controller.policy_loss(
                batch['states'],
                batch['log_probs'],
                batch['latent_actions'],
                batch['advantages'],
                batch['switch_betas'] == 1.,
                episode_lens = batch['_lens']
            )

            accelerator.backward(loss)

            grad_norm = accelerator.clip_grad_norm_(meta_controller.parameters(), max_grad_norm)

            optim.step()
            optim.zero_grad()

            meta_controller.eval()

            pbar.set_postfix(
                loss = f'{loss.item():.4f}',
                grad_norm = f'{grad_norm.item():.4f}',
                reward = f'{cumulative_rewards.mean().item():.4f}'
            )

            accelerator.log({
                'loss': loss.item(),
                'grad_norm': grad_norm.item()
            })

            accelerator.print(f'loss: {loss.item():.4f}, grad_norm: {grad_norm.item():.4f}')

    env.close()

    # save

    if exists(output_meta_controller_path):
        unwrapped_meta_controller.save(output_meta_controller_path)
        accelerator.print(f'MetaController weights saved to {output_meta_controller_path}')

if __name__ == '__main__':
    Fire(main)
