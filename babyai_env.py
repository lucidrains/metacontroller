import string
import torch
import numpy as np
from functools import lru_cache
from pathlib import Path
from shutil import rmtree

import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import Wrapper

import minigrid
from minigrid.wrappers import SymbolicObsWrapper, RGBImgPartialObsWrapper

from sentence_transformers import SentenceTransformer

# Mission space compatible with gymnasium shared memory (for AsyncVectorEnv)
BABYAI_MISSION_TEXT_MAX_LENGTH = 512
BABYAI_MISSION_CHARSET = " " + string.ascii_lowercase + string.ascii_uppercase + string.digits + "',.-"


class BabyAISharedMemoryWrapper(Wrapper):
    """
    Replaces the BabyAI mission observation space (BabyAIMissionSpace) with
    gymnasium.spaces.Text so that AsyncVectorEnv(..., shared_memory=True) works.
    Observation values are unchanged: mission remains a string.
    """

    def __init__(self, env, max_mission_length=BABYAI_MISSION_TEXT_MAX_LENGTH, charset=BABYAI_MISSION_CHARSET):
        super().__init__(env)
        if "mission" not in env.observation_space.spaces:
            return
        mission_space = env.observation_space.spaces["mission"]
        # Replace only if it's minigrid's MissionSpace (no create_shared_memory in gymnasium)
        if type(mission_space).__name__ in ("BabyAIMissionSpace", "MissionSpace"):
            self._mission_text_space = spaces.Text(max_length=max_mission_length, charset=charset)
            self.observation_space = spaces.Dict(
                {
                    **{k: v for k, v in env.observation_space.spaces.items() if k != "mission"},
                    "mission": self._mission_text_space,
                }
            )
        else:
            self._mission_text_space = None

    def _process_obs(self, obs):
        if self._mission_text_space is None:
            return obs
        # Mission stays as string; Text space accepts it for shared memory encode
        mission = obs.get("mission", "")
        if len(mission) > self._mission_text_space.max_length:
            mission = mission[: self._mission_text_space.max_length]
        obs = dict(obs)
        obs["mission"] = mission
        return obs

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self._process_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._process_obs(obs), reward, terminated, truncated, info

# sbert

_sbert = None

def get_sbert(local_files_only: bool = False, device: str = "cpu"):
    global _sbert
    if _sbert is None:
        _sbert = SentenceTransformer('all-MiniLM-L6-v2', local_files_only=local_files_only, device=device)
    return _sbert

@lru_cache(maxsize = 1000)
def get_mission_embedding(mission: str, sbert: object = None):
    if not sbert: sbert = get_sbert()
    embedding = sbert.encode(mission, convert_to_tensor = True)
    return embedding

_mission_cache = {}

def get_missions_embeddings(missions: list[str], sbert: object = None, batch_size: int = 64):
    if not sbert: sbert = get_sbert()

    # identify uncached missions

    unique_missions = list(set(missions))
    uncached_missions = [m for m in unique_missions if m not in _mission_cache]

    if len(uncached_missions) > 0:
        embeddings = sbert.encode(uncached_missions, convert_to_tensor=True, batch_size=batch_size, show_progress_bar=True)

        for mission, embedding in zip(uncached_missions, embeddings):
            _mission_cache[mission] = embedding

    # return stacked embeddings

    return torch.stack([_mission_cache[m] for m in missions])

# functions

def divisible_by(num, den):
    return (num % den) == 0

def transform_to_symbolic(images):
    # images: (..., W, H, 3) - assume last channel is (type, color, state)
    # as returned by FullyObsWrapper (which includes the agent)

    # We want (..., W, H, 3) where:
    # ch 0: x coord
    # ch 1: y coord
    # ch 2: object type

    *batch_dims, w, h, c = images.shape

    grid_x, grid_y = np.mgrid[:w, :h]

    # broadcast grid_x, grid_y to batch dims

    for _ in range(len(batch_dims)):
        grid_x = np.expand_dims(grid_x, axis = 0)
        grid_y = np.expand_dims(grid_y, axis = 0)

    grid_x = np.broadcast_to(grid_x, (*batch_dims, w, h))
    grid_y = np.broadcast_to(grid_y, (*batch_dims, w, h))

    images = images.astype(np.int64)

    # Output array
    symbolic = np.zeros(images.shape, dtype = np.int64)
    symbolic[..., 0] = grid_x
    symbolic[..., 1] = grid_y
    symbolic[..., 2] = np.where(images[..., 0] == 1, -1, images[..., 0]) # map empty (1) to -1

    return symbolic

# env creation

def create_env(
    env_id,
    render_mode = 'rgb_array',
    video_folder = None,
    render_every_eps = 1000,
    use_symbolic = True
):
    # register minigrid environments if needed
    # minigrid.register_minigrid_envs()

    # environment
    env = gym.make(env_id, render_mode = render_mode)

    if use_symbolic:
        env = SymbolicObsWrapper(env)
    else:
        env = RGBImgPartialObsWrapper(env)

    env = BabyAISharedMemoryWrapper(env)

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
