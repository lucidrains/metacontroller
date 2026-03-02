# /// script
# dependencies = [
#   "fire",
#   "gymnasium",
#   "gymnasium[other]",
#   "metacontroller-pytorch>=0.1.0",
#   "minigrid",
#   "tqdm",
#   "x-evolution>=0.1.27",
#   "einops",
#   "sentence-transformers",
#   "accelerate"
# ]
# ///
# To train with multiple GPUs:
# 1. accelerate config
# 2. accelerate launch train_babyai_evo_strat.py [args]

from __future__ import annotations
import fire
from pathlib import Path
from shutil import rmtree
import numpy as np

import torch
from torch import nn, Tensor, tensor
from torch.nn import Module
from einops import rearrange

from babyai_env import create_env, get_mission_embedding
from metacontroller.metacontroller import Transformer, MetaController
from torch_einops_utils.device import module_device

from accelerate import Accelerator

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# default fitness function

def default_fitness_fn(
    rewards: list[float],
    states: list[any],
    actions: list[any],
    next_states: list[any],
    infos: list[any]
) -> float:
    """
    researchers can modify this function to engineer their own rewards and fitness scores
    processing the entire episode at once for every noise vector of the population separately
    """
    return sum(rewards)

# babyai environment for ES

class BabyAIEnvironment(Module):
    def __init__(
        self,
        env_id = 'BabyAI-BossLevel-v0',
        video_folder = './recordings_babyai_es',
        render_every_eps = 100,
        max_steps = 500,
        use_resnet = False,
        condition_on_mission_embed = False,
        fitness_fn = default_fitness_fn,
        accelerator: Accelerator | None = None
    ):
        super().__init__()
        self.accelerator = accelerator

        self.env_id = env_id
        self.video_folder = video_folder
        self.render_every_eps = render_every_eps
        self.max_steps = max_steps
        self.use_resnet = use_resnet
        self.condition_on_mission_embed = condition_on_mission_embed
        self.fitness_fn = fitness_fn

        # initial env creation for observation space etc. if needed
        self.env = create_env(
            self.env_id,
            render_mode = 'rgb_array',
            video_folder = self.video_folder,
            render_every_eps = self.render_every_eps
        )

    def forward(self, model):
        model.eval()

        device = module_device(model)

        seed = torch.randint(0, int(1e6), ()).item()
        state, _ = self.env.reset(seed = seed)

        if self.condition_on_mission_embed:
            mission = self.env.unwrapped.mission
            mission_embed = get_mission_embedding(mission)
            mission_embed = mission_embed.to(device)
            if mission_embed.ndim == 1:
                mission_embed = mission_embed.unsqueeze(0)

        step = 0
        cache = None
        past_action_id = None

        episode_rewards = []
        episode_states = []
        episode_actions = []
        episode_next_states = []
        episode_infos = []

        while step < self.max_steps:
            image = state['image']
            image_tensor = torch.from_numpy(image).float().to(device)

            if self.use_resnet:
                image_tensor = rearrange(image_tensor, 'h w c -> 1 1 h w c')
                image_tensor = model.visual_encode(image_tensor)
            else:
                image_tensor = rearrange(image_tensor, 'h w c -> 1 1 (h w c)')

            if torch.is_tensor(past_action_id):
                past_action_id = past_action_id.long()

            with torch.no_grad():
                logits, cache = model(
                    image_tensor,
                    past_action_id,
                    return_cache = True,
                    return_raw_action_dist = True,
                    cache = cache,
                    condition = mission_embed if self.condition_on_mission_embed else None
                )

            action = model.action_readout.sample(logits)
            past_action_id = action
            action_id = action.squeeze()

            next_state, reward, terminated, truncated, info = self.env.step(action_id.cpu().numpy().item())

            episode_rewards.append(reward)
            episode_states.append(state)
            episode_actions.append(action_id)
            episode_next_states.append(next_state)
            episode_infos.append(info)

            done = terminated or truncated
            if done:
                break

            state = next_state
            step += 1

        return self.fitness_fn(
            episode_rewards,
            episode_states,
            episode_actions,
            episode_next_states,
            episode_infos
        )

def main(
    env_id = 'BabyAI-BossLevel-v0',
    num_generations = 100,
    max_steps = 500,
    render_every_eps = 100,
    video_folder = './recordings_babyai_es',
    transformer_weights_path: str | None = None,
    meta_controller_weights_path: str | None = None,
    output_meta_controller_path = 'metacontroller_es_trained.pt',
    use_resnet = False,
    noise_population_size = 50,
    noise_scale = 1e-2,
    learning_rate = 1e-3,
    condition_on_mission_embed = False,
    fitness_fn = default_fitness_fn
):
    accelerator = Accelerator()

    if accelerator.is_main_process:
        rmtree(video_folder, ignore_errors = True)

    assert noise_population_size >= 2, "noise_population_size must be at least 2 for evolutionary strategies"

    # load model

    assert exists(transformer_weights_path), "Transformer weights must be provided"

    # lazy import to avoid unnecessary dependencies if not used
    from metacontroller.transformer_with_resnet import TransformerWithResnet as TransformerResnet
    transformer_klass = TransformerResnet if use_resnet else Transformer

    model = transformer_klass.init_and_load(transformer_weights_path, strict = False)
    model.eval()

    if exists(meta_controller_weights_path):
        meta_controller = MetaController.init_and_load(meta_controller_weights_path, strict = False)
        model.meta_controller = meta_controller

    assert exists(model.meta_controller), "MetaController must be present for evolution"

    # setup environment

    babyai_env = BabyAIEnvironment(
        env_id = env_id,
        video_folder = video_folder,
        render_every_eps = render_every_eps,
        max_steps = max_steps,
        use_resnet = use_resnet,
        condition_on_mission_embed = condition_on_mission_embed,
        fitness_fn = fitness_fn,
        accelerator = accelerator
    )

    # evolve

    model.train_internal_rl(eval_rest = True)

    model.evolve(
        num_generations = num_generations,
        environment = babyai_env,
        noise_population_size = noise_population_size,
        noise_scale = noise_scale,
        learning_rate = learning_rate,
        accelerator = accelerator,
        sync_on_init = False
    )

    # save

    if accelerator.is_main_process:
        model.meta_controller.save(output_meta_controller_path)
        print(f'MetaController weights saved to {output_meta_controller_path}')

if __name__ == '__main__':
    fire.Fire(main)
