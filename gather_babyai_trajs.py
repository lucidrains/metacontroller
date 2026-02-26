
# /// script
# dependencies = [
#   "gymnasium",
#   "minigrid",
#   "tqdm",
#   "fire",
#   "memmap-replay-buffer>=0.0.29",
#   "loguru",
#   "sentence-transformers"
# ]
# ///

# taken with modifications from https://github.com/ddidacus/bot-minigrid-babyai/blob/main/tests/get_trajectories.py

import fire
import random
import multiprocessing
from loguru import logger
from pathlib import Path

import warnings
warnings.filterwarnings("ignore", category = UserWarning)

import numpy as np

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

import torch
import minigrid
import gymnasium as gym
from babyai_env import get_missions_embeddings, get_sbert
from minigrid.utils.baby_ai_bot import BabyAIBot
from minigrid.wrappers import FullyObsWrapper, SymbolicObsWrapper, RGBImgPartialObsWrapper

from gymnasium import spaces
from gymnasium.core import ObservationWrapper
from minigrid.core.constants import OBJECT_TO_IDX

from memmap_replay_buffer import ReplayBuffer

# Difficulty thresholds based on mission length
EASY_MAX_LENGTH = 30      # easy: 0 to 30
MEDIUM_MAX_LENGTH = 75    # medium: 30 to 75, hard: > 75

# helpers

def exists(val):
    return val is not None

def sample(prob):
    return random.random() < prob

def get_mission_for_seed(env_id, seed):
    """
    Get the mission string for a given seed.
    Returns (mission_length, mission_string).
    """
    env = gym.make(env_id, render_mode="rgb_array")
    env.reset(seed=seed)
    mission = env.unwrapped.mission
    env.close()
    return len(mission), mission

def get_mission_length(env_id, seed):
    """
    Get the mission length for a given seed.
    Returns the length of the mission string.
    """
    length, _ = get_mission_for_seed(env_id, seed)
    return length

def categorize_seeds_by_difficulty(env_id, num_seeds_per_level, level_difficulty=None):
    """
    Scan seeds and categorize them by difficulty based on mission length.
    Also collects the mission string for each seed.

    Args:
        env_id: Environment ID
        num_seeds_per_level: Number of seeds needed per difficulty level.
            Either an int (same quota for all levels) or a list of ints
            in the same order as level_difficulty.
        level_difficulty: List of levels to collect seeds for.
            Supported: 'easy', 'medium', 'hard'
            If None, collects for ['easy', 'hard'].

    Returns:
        tuple of:
            - dict with keys for each requested level, each containing a list of seeds
            - dict mapping seed -> mission string
    """
    if level_difficulty is None:
        level_difficulty = ['easy', 'hard']

    if isinstance(num_seeds_per_level, int):
        level_to_quota = {level: num_seeds_per_level for level in level_difficulty}
    else:
        level_to_quota = {level: num_seeds_per_level[i] for i, level in enumerate(level_difficulty)}

    seeds = {level: [] for level in level_difficulty}
    seed_to_mission = {}

    total_needed = sum(level_to_quota[level] for level in level_difficulty)
    print(f"Scanning seeds to categorize by difficulty (quotas {level_to_quota} for {level_difficulty})...")

    with tqdm(total=total_needed, desc="Categorizing seeds") as pbar:
        seed = 1
        all_done = False
        while not all_done:
            all_done = all(len(seeds[level]) >= level_to_quota[level] for level in level_difficulty)

            try:
                mission_length, mission = get_mission_for_seed(env_id, seed)

                # easy: mission length <= 30
                if 'easy' in level_difficulty and mission_length <= EASY_MAX_LENGTH and len(seeds['easy']) < level_to_quota['easy']:
                    seeds['easy'].append(seed)
                    seed_to_mission[seed] = mission
                    pbar.update(1)
                # medium: mission length 30 < len <= 75
                elif 'medium' in level_difficulty and mission_length > EASY_MAX_LENGTH and mission_length <= MEDIUM_MAX_LENGTH and len(seeds['medium']) < level_to_quota['medium']:
                    seeds['medium'].append(seed)
                    seed_to_mission[seed] = mission
                    pbar.update(1)
                # hard: mission length > 75
                elif 'hard' in level_difficulty and mission_length > MEDIUM_MAX_LENGTH and len(seeds['hard']) < level_to_quota['hard']:
                    seeds['hard'].append(seed)
                    seed_to_mission[seed] = mission
                    pbar.update(1)
            except Exception as e:
                logger.warning(f"Error getting mission for seed {seed}: {e}")

            seed += 1

    return seeds, seed_to_mission

# agent

class BabyAIBotEpsilonGreedy:
    def __init__(self, env, random_action_prob = 0., num_actions = None):
        self.expert = BabyAIBot(env)
        self.random_action_prob = random_action_prob

        logger.info(f"num_actions: {num_actions}")

        if num_actions is not None:
            self.num_actions = num_actions
        else:
            self.num_actions = env.action_space.n
        self.last_action = None

    def __call__(self, state):
        if sample(self.random_action_prob):
            action = torch.randint(0, self.num_actions, ()).item()
        else:
            action = self.expert.replan(self.last_action)

        self.last_action = action
        return action

# functions

def collect_single_episode(env_id, seed, num_steps, random_action_prob, state_shape, num_actions=None, use_rgb_states = True):
    """
    Collect a single episode of demonstrations.
    Returns tuple of (episode_state, episode_action, success, episode_length, seed)
    Note: mission embeddings are pre-computed in main process and looked up by seed.
    """
    if env_id not in gym.envs.registry:
        minigrid.register_minigrid_envs()

    env = gym.make(env_id, render_mode="rgb_array", highlight=False)
    if use_rgb_states:
        env = RGBImgPartialObsWrapper(env)
    else:
        env = SymbolicObsWrapper(env)

    try:
        state_obs, _ = env.reset(seed=seed)
        episode_state = np.zeros((num_steps, *state_shape), dtype=np.uint8)
        episode_action = np.zeros(num_steps, dtype=np.uint8)

        expert = BabyAIBotEpsilonGreedy(env.unwrapped, random_action_prob=random_action_prob, num_actions=num_actions)

        for _step in range(num_steps):
            try:
                action = expert(state_obs)
            except Exception:
                env.close()
                return None, None, False, 0, seed

            episode_state[_step] = state_obs["image"]

            episode_action[_step] = action

            state_obs, reward, terminated, truncated, info = env.step(action)

            if terminated:
                env.close()
                return episode_state, episode_action, True, _step + 1, seed

        env.close()
        return episode_state, episode_action, False, num_steps, seed

    except Exception:
        env.close()
        return None, None, False, 0, seed

def collect_demonstrations(
    use_rgb_states = True,
    env_id = "BabyAI-MiniBossLevel-v0",
    num_seeds = 100,
    num_episodes_per_seed = 100,
    num_steps = 500,
    random_action_prob = 0.05,
    num_workers = None,
    difficulty = "easy",
    output_dir = "babyai-minibosslevel-trajectories",
    mission_embed_dim = 384,
    num_actions = None,         # if the actual number of actions is different from the environment's action space, provide it here (for BabyAI-MiniBossLevel-v0, it's 4: https://minigrid.farama.org/environments/babyai/MiniBossLevel/)
):
    """
    The BabyAI Bot should be able to solve all BabyAI environments,
    allowing us therefore to generate demonstrations.
    Parallelized version using ProcessPoolExecutor.

    difficulty: single level ('easy', 'medium', 'hard') or a list of levels.
        If a list, num_seeds is divided among the levels and seeds from all
        levels are merged into the same output (same numpy files and seeds file).
    """

    # Normalize difficulty to list and validate
    if isinstance(difficulty, str):
        difficulty = [difficulty]
    assert all(d in ['easy', 'medium', 'hard'] for d in difficulty), f"difficulty must be in ['easy','medium','hard'], got {difficulty}"

    # Divide num_seeds among difficulty levels (remainder goes to first levels)
    n_levels = len(difficulty)
    base = num_seeds // n_levels
    remainder = num_seeds % n_levels
    num_seeds_per_level = [base + (1 if i < remainder else 0) for i in range(n_levels)]

    # Register minigrid envs if not already registered
    if env_id not in gym.envs.registry:
        minigrid.register_minigrid_envs()

    # Determine state shape from environment
    temp_env = gym.make(env_id)
    if use_rgb_states:
        temp_env = RGBImgPartialObsWrapper(temp_env)
    else:
        temp_env = SymbolicObsWrapper(temp_env)
    state_shape = temp_env.observation_space['image'].shape
    temp_env.close()

    logger.info(f"Detected state shape: {state_shape} for env {env_id} and num_actions: {num_actions}")

    if not exists(num_workers):
        num_workers = multiprocessing.cpu_count()

    total_episodes = num_seeds * num_episodes_per_seed

    # Filter seeds: collect per level with quotas, then merge
    seeds, seed_to_mission = categorize_seeds_by_difficulty(
        env_id, num_seeds_per_level=num_seeds_per_level, level_difficulty=difficulty
    )
    # Merge seeds from all difficulty levels (order: first level, then second, ...)
    selected_seeds = [s for level in difficulty for s in seeds[level]]

    # pre-compute embeddings on master process (harder to handle with subprocesses)

    logger.info("Pre-computing mission embeddings with GPU...")
    sbert = get_sbert(local_files_only = False, device = "cuda" if torch.cuda.is_available() else "cpu")
    missions = [seed_to_mission[s] for s in selected_seeds]
    embeddings = get_missions_embeddings(missions, sbert).cpu().numpy()
    seed_to_embedding = {s: embeddings[i] for i, s in enumerate(selected_seeds)}

    successful = 0
    progressbar = tqdm(total=total_episodes)

    output_folder = Path(output_dir)

    fields = {
        'state': ('float', state_shape),
        'action': ('float', ()),
    }

    meta_fields = {
        'mission_embedding': ('float', (mission_embed_dim,)),
    }

    buffer = ReplayBuffer(
        folder = output_folder,
        max_episodes = total_episodes,
        max_timesteps = num_steps,
        fields = fields,
        meta_fields = meta_fields,
        overwrite = True,
    )

    max_pending = num_workers
    all_seeds = selected_seeds * num_episodes_per_seed

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        seed_iter = iter(all_seeds)
        futures = {}

        # submit

        for _ in range(min(max_pending, len(all_seeds))):
            seed = next(seed_iter, None)
            if exists(seed):
                future = executor.submit(collect_single_episode, env_id, seed, num_steps, random_action_prob, state_shape, num_actions, use_rgb_states)
                futures[future] = seed

        # collect

        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)

            for future in done:
                seed = futures.pop(future)
                episode_state, episode_action, success, episode_length, returned_seed = future.result()

                if success and exists(episode_state):
                    buffer.store_episode(
                        state = episode_state[:episode_length],
                        action = episode_action[:episode_length],
                        mission_embedding = seed_to_embedding[returned_seed],
                    )
                    successful += 1

                progressbar.update(1)
                progressbar.set_description(f"success rate = {successful}/{progressbar.n:.2f}")

                seed = next(seed_iter, None)
                if exists(seed):
                    new_future = executor.submit(collect_single_episode, env_id, seed, num_steps, random_action_prob, state_shape, num_actions, use_rgb_states)
                    futures[new_future] = seed

    buffer.flush()
    progressbar.close()

    seeds_array = np.array(selected_seeds)
    seeds_path = output_folder / "seeds.npy"
    np.save(seeds_path, seeds_array)

    logger.info(f"Saved {successful} trajectories to {output_dir}")
    logger.info(f"Saved {len(seeds_array)} unique seeds to {seeds_path}")

if __name__ == "__main__":
    fire.Fire(collect_demonstrations)