import torch
import numpy as np
from functools import lru_cache
from pathlib import Path
from shutil import rmtree

import gymnasium as gym
import minigrid
from minigrid.wrappers import FullyObsWrapper, SymbolicObsWrapper

from sentence_transformers import SentenceTransformer

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
    minigrid.register_minigrid_envs()

    # environment
    env = gym.make(env_id, render_mode = render_mode)
    env = FullyObsWrapper(env)

    if use_symbolic:
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
