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
from tqdm import tqdm
from shutil import rmtree

import torch

import gymnasium as gym
import minigrid

from memmap_replay_buffer import ReplayBuffer

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

# main

def main(
    env_name = 'BabyAI-BossLevel-v0',
    num_episodes = int(10e6),
    max_timesteps = 500,
    buffer_size = 5_000,
    render_every_eps = 1_000,
    video_folder = './recordings',
    seed = None
):

    # environment

    env = gym.make(env_name, render_mode = 'rgb_array')

    rmtree(video_folder, ignore_errors = True)

    env = gym.wrappers.RecordVideo(
        env = env,
        video_folder = video_folder,
        name_prefix = 'babyai',
        episode_trigger = lambda eps_num: divisible_by(eps_num, render_every_eps),
        disable_logger = True
    )

    # replay

    replay_buffer = ReplayBuffer(
        './replay-data',
        max_episodes = buffer_size,
        max_timesteps = max_timesteps + 1,
        fields = dict(
            action = 'int',
            state_image = ('float', (7, 7, 3)),
            state_direction = 'int'
        ),
        overwrite = True,
        circular = True
    )

    # rollouts

    for _ in tqdm(range(num_episodes)):

        state, *_ = env.reset(seed = seed)

        for _ in range(max_timesteps):

            action = torch.randint(0, 7, ())
            next_state, reward, terminated, truncated, *_ = env.step(action.numpy())

            done = terminated or truncated

            if done:
                break

            state = next_state

# running

if __name__ == '__main__':
    Fire(main)
