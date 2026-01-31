from pathlib import Path
from shutil import rmtree

import gymnasium as gym
import minigrid
from minigrid.wrappers import FullyObsWrapper, SymbolicObsWrapper

# functions

def divisible_by(num, den):
    return (num % den) == 0

# env creation

def create_env(
    env_id,
    render_mode = 'rgb_array',
    video_folder = None,
    render_every_eps = 1000
):
    # register minigrid environments if needed
    minigrid.register_minigrid_envs()

    # environment
    env = gym.make(env_id, render_mode = render_mode)
    env = FullyObsWrapper(env)
    env = SymbolicObsWrapper(env)

    if video_folder is not None:
        video_folder = Path(video_folder)
        rmtree(video_folder, ignore_errors = True)

        env = gym.wrappers.RecordVideo(
            env = env,
            video_folder = str(video_folder),
            name_prefix = 'babyai',
            episode_trigger = lambda eps_num: divisible_by(eps_num, render_every_eps),
            disable_logger = True
        )

    return env
