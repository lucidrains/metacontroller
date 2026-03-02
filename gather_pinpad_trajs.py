# create and customize my own env.
import torch
import numpy as np
import itertools
import matplotlib.pyplot as plt

import fire
import random
import multiprocessing
from loguru import logger
from pathlib import Path
import imageio

import warnings
warnings.filterwarnings("ignore", category = UserWarning)

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import ObservationWrapper
from gymnasium.envs.registration import register

# MiniGrid imports
from minigrid.envs.babyai.goto import GoToSeq
from minigrid.core.world_object import Ball, Box, Key, Goal
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX
from minigrid.envs.babyai.core.verifier import GoToInstr, ObjDesc, BeforeInstr, AndInstr
from minigrid.utils.baby_ai_bot import BabyAIBot
from minigrid.wrappers import RGBImgObsWrapper

from memmap_replay_buffer import ReplayBuffer

# Allowed types and colors (no grey to distinguish from walls)
ALLOWED_TYPES = ["ball", "box", "key"]
ALLOWED_COLORS = ["red", "green", "blue", "purple", "yellow"]

# Object constructors
OBJ_CONSTRUCTORS = {
    "ball": Ball,
    "box": Box,
    "key": Key,
}

# Allowed types and colors (no grey to distinguish from walls)
ALLOWED_TYPES = ["ball", "box", "key"]
ALLOWED_COLORS = ["red", "green", "blue", "purple", "yellow"]

# Object constructors
OBJ_CONSTRUCTORS = {
    "ball": Ball,
    "box": Box,
    "key": Key,
}

class PinPad(GoToSeq):
    def __init__(self, num_objects, obj_seq, room_size=8, num_rows=1, num_cols=1, seed=None, **kwargs):
        """
        Args:
            num_objects: Number of distinct objects to generate (total pool across episodes).
            obj_seq: Specifies the objects to visit and in what order for this episode.
                         Every index must be in [0, num_objects - 1].
            room_size: Size of each room
            num_rows: Number of room rows
            num_cols: Number of room columns
            seed: Random seed for object generation
        """
        assert num_objects <= len(ALLOWED_TYPES) * len(ALLOWED_COLORS), \
            f"Max distinct objects is {len(ALLOWED_TYPES) * len(ALLOWED_COLORS)}"

        # Validate obj_indices
        assert len(obj_seq) >= 1, "obj_seq must have at least 1 element"
        assert all(0 <= idx < num_objects for idx in obj_seq), \
            f"All indices in obj_seq must be in [0, {num_objects - 1}]"

        self.num_objects = num_objects
        self.obj_seq = obj_seq

        # Generate all possible (type, color) combinations and sample
        all_combinations = list(itertools.product(ALLOWED_TYPES, ALLOWED_COLORS))

        if seed is not None:
            rng = random.Random(seed)
            self.object_specs = rng.sample(all_combinations, num_objects)
        else:
            # sample the first num_objects
            self.object_specs = all_combinations[:num_objects]

        # Create mapping: (type, color) -> object_index (0 to num_objects-1)
        self.obj_to_idx = {spec: idx for idx, spec in enumerate(self.object_specs)}

        # One-hot channels:
        # 0: empty cell, 1: agent, 2: wall cell, 3+: distinct objects
        self.num_channels = 3 + num_objects

        super().__init__(
            room_size=room_size,
            num_rows=num_rows,
            num_cols=num_cols,
            num_dists=0,  # We handle objects ourselves
            **kwargs
        )

    def gen_mission(self):
        self.place_agent()

        # Place all objects in the room (not just the ones in the task sequence)
        for idx in range(self.num_objects):
            obj_type, color = self.object_specs[idx]
            obj = OBJ_CONSTRUCTORS[obj_type](color)
            self.place_in_room(0, 0, obj)

        # Build instruction based on obj_indices (which objects to visit and in what order)
        def make_goto(obj_idx):
            t, c = self.object_specs[obj_idx]
            return GoToInstr(ObjDesc(t, c))

        # number of objects to visit (i.e. subgoals)
        num_to_visit = len(self.obj_seq)

        if num_to_visit == 1:
            self.instrs = make_goto(self.obj_seq[0])

        elif num_to_visit == 2:
            # "go to A, then go to B"
            self.instrs = BeforeInstr(
                make_goto(self.obj_seq[0]),
                make_goto(self.obj_seq[1])
            )

        elif num_to_visit == 3:
            # "go to A and go to B, then go to C"

            # AndInstr is a combination of instructions that can be executed in any order
            # BeforeInstr is a combination of instructions that must be executed in the specified order
            # BeforeInstr can only be used for 2 instructions, and MiniGrid doesn't support chaining BeforeInstr s we resort to AndInstr, which should be ok for the purpose of generating expert demonstrations for BC - we can just generate the subgoal GT lables post-hoc.
            self.instrs = BeforeInstr(
                AndInstr(make_goto(self.obj_seq[0]), make_goto(self.obj_seq[1])),
                make_goto(self.obj_seq[2])
            )

        elif num_to_visit == 4:
            # "go to A and go to B, then go to C and go to D"

            # AndInstr is a combination of instructions that can be executed in any order
            # BeforeInstr is a combination of instructions that must be executed in the specified order
            # BeforeInstr can only be used for 2 instructions, and MiniGrid doesn't support chaining BeforeInstr s we resort to AndInstr, which should be ok for the purpose of generating expert demonstrations for BC - we can just generate the subgoal GT lables post-hoc.
            self.instrs = BeforeInstr(
                AndInstr(make_goto(self.obj_seq[0]), make_goto(self.obj_seq[1])),
                AndInstr(make_goto(self.obj_seq[2]), make_goto(self.obj_seq[3]))
            )

        else:
            raise ValueError(f"Max 4 objects supported for sequenced missions, got {num_to_visit}")

class OneHotFullyObsWrapper(ObservationWrapper):
    """
    One-hot encoding of the fully observable grid.

    Channels:
        0: empty space (None cells)
        1: agent
        2: wall
        3+: distinct objects (indexed by environment's obj_to_idx)
    """

    def __init__(self, env):
        super().__init__(env)

        # Get num_channels from the unwrapped environment
        unwrapped = self.env.unwrapped
        assert hasattr(unwrapped, 'num_channels'), \
            "Environment must have 'num_channels' attribute (use PinPidEnv)"

        self.num_channels = unwrapped.num_channels
        self.obj_to_idx = unwrapped.obj_to_idx

        width = unwrapped.width
        height = unwrapped.height

        new_image_space = spaces.Box(
            low=0,
            high=1,
            shape=(width, height, self.num_channels),
            dtype="uint8",
        )

        self.observation_space = spaces.Dict({
            **self.observation_space.spaces,
            "one_hot": new_image_space,
        })

    def observation(self, obs):
        env = self.unwrapped
        width, height = env.width, env.height

        # Get full grid encoding: [object_type, color, state]
        full_grid = env.grid.encode()

        # Initialize one-hot: (width, height, num_channels)
        one_hot = np.zeros((width, height, self.num_channels), dtype=np.uint8)

        for x in range(width):
            for y in range(height):
                obj_type_id = full_grid[x, y, 0]
                color_id = full_grid[x, y, 1]

                if obj_type_id == OBJECT_TO_IDX["unseen"] or obj_type_id == 0:
                    # Empty / unseen -> channel 0
                    one_hot[x, y, 0] = 1
                elif obj_type_id == OBJECT_TO_IDX["wall"]:
                    # Wall -> channel 2
                    one_hot[x, y, 2] = 1
                elif obj_type_id in [OBJECT_TO_IDX["ball"], OBJECT_TO_IDX["box"], OBJECT_TO_IDX["key"]]:
                    # Distinct object -> look up by (type, color)
                    idx_to_type = {v: k for k, v in OBJECT_TO_IDX.items()}
                    idx_to_color = {v: k for k, v in COLOR_TO_IDX.items()}

                    obj_type = idx_to_type[obj_type_id]
                    color = idx_to_color[color_id]

                    if (obj_type, color) in self.obj_to_idx:
                        obj_idx = self.obj_to_idx[(obj_type, color)]
                        one_hot[x, y, 3 + obj_idx] = 1
                    else:
                        # Unknown object, treat as empty
                        one_hot[x, y, 0] = 1
                else:
                    # Other (door, goal, lava, etc.) -> treat as empty for now
                    one_hot[x, y, 0] = 1

        # Agent position -> channel 1
        agent_x, agent_y = env.agent_pos
        one_hot[agent_x, agent_y, :] = 0  # Clear any other channel
        one_hot[agent_x, agent_y, 1] = 1  # Set agent channel

        return {**obs, "one_hot": one_hot}

# helpers
def exists(val):
    return val is not None

def sample(prob):
    return random.random() < prob

def get_mission_length(env_id, seed):
    """
    Get the mission length for a given seed.
    Returns the length of the mission string.
    """
    env = gym.make(env_id, render_mode="rgb_array")
    env.reset(seed=seed)
    length = len(env.unwrapped.mission)
    env.close()
    return length



class BabyAIBotEpsilonGreedy:
    def __init__(self, env, random_action_prob = 0., num_actions = None):
        self.expert = BabyAIBot(env)
        self.random_action_prob = random_action_prob

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

def collect_single_episode(env_id, obj_seq, seed, num_steps, random_action_prob, state_shape, one_hot=False, visualize=False, num_actions=None):
    """
    Collect a single episode of demonstrations.

    Args:
        env_id: registered environment ID
        obj_seq: sequence of object indices to visit
        seed: random seed for environment reset
        num_steps: max steps per episode
        random_action_prob: probability of taking random action
        state_shape: shape of state observations

    Returns:
        tuple of (episode_state, episode_action, success, episode_length)
    """

    env = gym.make(env_id, render_mode="rgb_array", obj_seq=obj_seq)

    if one_hot:
        env = OneHotFullyObsWrapper(env)
    else:
        env = RGBImgObsWrapper(env)

    try:
        state_obs, _ = env.reset(seed=seed)

        episode_state = np.zeros((num_steps, *state_shape), dtype=np.float32)
        episode_action = np.zeros(num_steps, dtype=np.float32)
        episode_label = -np.ones(num_steps, dtype=np.float32)     # these are the per-step subgoal labels (i.e. object indices)

        expert = BabyAIBotEpsilonGreedy(env.unwrapped, random_action_prob=random_action_prob, num_actions=num_actions)

        if visualize:
            frames = []
        else:
            frames = None

        for _step in range(num_steps):
            try:
                action = expert(state_obs)

                # extract the intended subgoal
                if hasattr(expert.expert, 'stack') and expert.expert.stack:
                    # each entry in self.expert.stack is a SubGoal instance
                    # current subgoal is the last entry in the stack
                    subgoal = expert.expert.stack[-1]

                    if subgoal.reason == "Explore":
                        obj_idx = -1   # -1 indicates explore
                    else:
                        assert subgoal.reason is None
                        assert isinstance(subgoal.datum, ObjDesc)
                        obj_idx = env.unwrapped.obj_to_idx[(subgoal.datum.type, subgoal.datum.color)]
            except Exception as e:
                print("Error: ", e)
                env.close()
                return None, None, None, None, False, 0

            episode_label[_step] = obj_idx
            if one_hot:
                episode_state[_step] = state_obs["one_hot"]
            else:
                episode_state[_step] = state_obs["image"]
            episode_action[_step] = action

            state_obs, reward, terminated, truncated, info = env.step(action)

            if visualize:
                frames.append(env.render())

            if terminated:
                env.close()
                return frames, episode_label, episode_state, episode_action, True, _step + 1

        env.close()
        return frames, episode_label, episode_state, episode_action, False, num_steps
    except Exception as e:
        print("Error: ", e)
        env.close()
        return None, None, None, None, False, 0

def collect_demonstrations(
    env_id = "PinPad-v0",
    num_objects = 4,            # number of distinct objects (across episodes)
    room_size = 8,              # size of each room (width and height)
    num_rows = 1,               # number of rows of rooms
    num_cols = 1,               # number of columns of rooms
    obj_seqs = [[0,1,2]],     # list of object sequences to visit
    seeds = None,               # list of seeds (if None, uses range(num_seeds))
    num_seeds = 100,            # number of seeds (ignored if seeds is provided)
    num_steps = 100,            # max steps per episode
    random_action_prob = 0.05,
    num_workers = None,
    output_dir = "pinpad_demonstrations",
    visualize = False,
    one_hot = False,
    num_actions = None,         # if the actual number of actions is different from the environment's action space, provide it here
):
    """
    Collect demonstrations using BabyAI Bot for the PinPad environment.

    For each (obj_seq, seed) pair, collects one episode.
    Total episodes = len(obj_seqs) * len(seeds)

    Parallelized using ProcessPoolExecutor.
    """

    # Register the environment
    if env_id in gym.envs.registry:
        del gym.envs.registry[env_id]

    register(
        id=env_id,
        entry_point=PinPad,
        kwargs={
            "num_objects": num_objects,
            "obj_seq": obj_seqs[0],  # placeholder, will be overridden at gym.make
            "room_size": room_size,
            "num_rows": num_rows,
            "num_cols": num_cols
        },
    )

    # Determine state shape from environment
    temp_env = gym.make(env_id, obj_seq=obj_seqs[0])
    if one_hot:
        temp_env = OneHotFullyObsWrapper(temp_env)
        state_shape = temp_env.observation_space['one_hot'].shape
    else:
        temp_env = RGBImgObsWrapper(temp_env)
        state_shape = temp_env.observation_space['image'].shape
    temp_env.close()

    logger.info(f"Detected state shape: {state_shape} for env {env_id}")
    logger.info(f"num actions: {num_actions}")

    # Display object specs
    # i.e. which object index corresponds to which (type, color)
    logger.info("Object specs:")
    for idx, spec in enumerate(temp_env.unwrapped.object_specs):
        logger.info(f"Object {idx}: {spec}")

    # Display each task
    logger.info(f"Tasks: {obj_seqs}")

    if not exists(num_workers):
        num_workers = multiprocessing.cpu_count()

    # Build list of seeds
    if seeds is None:
        seeds = list(range(num_seeds))

    # Create all (obj_seq, seed) pairs
    all_tasks = [(obj_seq, seed) for obj_seq in obj_seqs for seed in seeds]
    total_episodes = len(all_tasks)

    logger.info(f"Collecting {total_episodes} episodes ({len(obj_seqs)} Tasks x {len(seeds)} seeds)")

    successful = 0
    progressbar = tqdm(total=total_episodes)

    output_folder = Path(output_dir)

    fields = {
        'state': ('float', state_shape),
        'action': ('float', ()),
        'label': ('float', ()),          # subgoal labels (i.e. object indices)
    }

    buffer = ReplayBuffer(
        folder = output_folder,
        max_episodes = total_episodes,
        max_timesteps = num_steps,
        fields = fields,
        overwrite = True
    )

    # Parallel execution with bounded pending futures to avoid OOM
    max_pending = num_workers

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        task_iter = iter(all_tasks)
        futures = {}

        # Initial batch of submissions
        for _ in range(min(max_pending, total_episodes)):
            task = next(task_iter, None)
            if exists(task):
                obj_seq, seed = task
                future = executor.submit(
                    collect_single_episode, env_id, obj_seq, seed,
                    num_steps, random_action_prob, state_shape, one_hot, visualize, num_actions
                )
                futures[future] = task

        # Process completed tasks and submit new ones
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)

            for future in done:
                del futures[future]
                frames, episode_label, episode_state, episode_action, success, episode_length = future.result()

                if success and exists(episode_state):
                    buffer.store_episode(
                        state = episode_state[:episode_length],
                        action = episode_action[:episode_length],
                        label = episode_label[:episode_length],
                    )

                    # if visualize, save frames as .gif
                    # assert frames is not None
                    if visualize:
                        assert frames is not None
                        frames_path = output_folder / f"episode_{successful}.gif"
                        imageio.mimsave(frames_path, frames, fps=3)

                    successful += 1

                progressbar.update(1)
                progressbar.set_description(f"success rate = {successful}/{progressbar.n:.2f}")

                # Submit a new task to replace the completed one
                task = next(task_iter, None)
                if exists(task):
                    obj_seq, seed = task
                    new_future = executor.submit(
                        collect_single_episode, env_id, obj_seq, seed,
                        num_steps, random_action_prob, state_shape, one_hot, visualize, num_actions
                    )
                    futures[new_future] = task

    buffer.flush()
    progressbar.close()

    # Save metadata for reproducibility
    metadata = {
        'obj_seqs': obj_seqs,
        'seeds': seeds,
        'num_objects': num_objects,
        'room_size': room_size,
    }
    metadata_path = output_folder / "metadata.npy"
    np.save(metadata_path, metadata, allow_pickle=True)

    logger.info(f"Saved {successful} trajectories to {output_dir}")
    logger.info(f"Saved metadata to {metadata_path}")

if __name__ == "__main__":
    fire.Fire(collect_demonstrations)

    """
    # How to run this script in terminal:

    python gather_pinpad_trajs.py \
        --num_objects 4 \
        --obj_seqs "[[0,1,2],[0,2,1],[2,3,1],[0,2,3]]" \     # tasks i.e. list of object sequences to visit
        --room_size 8 \
        --num_rows 1 \
        --num_cols 1 \
        --num_seeds 10 \
        --num_steps 100 \
        --output_dir pinpad_demonstrations \
        --visualize True
    """
