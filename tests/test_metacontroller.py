import pytest
param = pytest.mark.parametrize

from pathlib import Path
from functools import partial

import torch
from torch import cat
from metacontroller.metacontroller import Transformer, MetaController, policy_loss, z_score
from metacontroller.metacontroller_with_binary_mapper import MetaControllerWithBinaryMapper

from einops import rearrange

# functions

def exists(v):
    return v is not None

# test

@param('use_binary_mapper_variant, switch_per_latent_dim', [
    (False, False),
    (False, True),
    (True, False)
])
@param('action_discrete', (False, True))
@param('variable_length', (False, True))
def test_metacontroller(
    use_binary_mapper_variant,
    switch_per_latent_dim,
    action_discrete,
    variable_length
):

    state = torch.randn(2, 128, 384)
    episode_lens = torch.tensor([64, 64]) if variable_length else None

    if action_discrete:
        actions = torch.randint(0, 4, (2, 128))
        action_embed_readout = dict(num_discrete = 4)
        assert_shape = (4,)
    else:
        actions = torch.randn(2, 128, 8)
        action_embed_readout = dict(num_continuous = 8)
        assert_shape = (8, 2)

    # behavioral cloning phase

    model = Transformer(
        dim = 512,
        action_embed_readout = action_embed_readout,
        state_embed_readout = dict(num_continuous = 384),
        lower_body = dict(depth = 2,),
        upper_body = dict(depth = 2,),
    )

    state_clone_loss, action_clone_loss = model(state, actions, episode_lens = episode_lens)
    (state_clone_loss + 0.5 * action_clone_loss).backward()

    # discovery and internal rl phase with meta controller

    if not use_binary_mapper_variant:
        meta_controller = MetaController(
            dim_model = 512,
            dim_meta_controller = 256,
            dim_latent = 128,
            switch_per_latent_dim = switch_per_latent_dim
        )
    else:
        meta_controller = MetaControllerWithBinaryMapper(
            dim_model = 512,
            dim_meta_controller = 256,
            switch_per_code = switch_per_latent_dim,
            dim_code_bits = 8, # 2 ** 8 = 256 codes
        )

    # discovery phase

    (action_recon_loss, kl_loss, switch_loss) = model(state, actions, meta_controller = meta_controller, discovery_phase = True, episode_lens = episode_lens)
    (action_recon_loss + kl_loss * 0.1 + switch_loss * 0.2).backward()

    # internal rl - done iteratively

    # simulate grpo

    all_episodes = []
    all_rewards = []

    for _ in range(3): # group of 3
        subset_state = state[:1]

        cache = None
        past_action_id = None

        states = []
        log_probs = []
        switch_betas = []
        latent_actions = []

        for one_state in subset_state.unbind(dim = 1):
            one_state = rearrange(one_state, 'b d -> b 1 d')

            logits, cache = model(one_state, past_action_id, meta_controller = meta_controller, return_cache = True)

            past_action_id = model.action_readout.sample(logits)

            # get log prob from meta controller latent actions

            meta_output = cache.prev_hiddens.meta_controller

            old_log_probs = meta_controller.log_prob(meta_output.action_dist, meta_output.actions)

            states.append(meta_output.input_residual_stream)
            log_probs.append(old_log_probs)
            switch_betas.append(meta_output.switch_beta)
            latent_actions.append(meta_output.actions)

        # accumulate across time for the episode data

        all_episodes.append((
            cat(states, dim = 1),
            cat(log_probs, dim = 1),
            cat(switch_betas, dim = 1),
            cat(latent_actions, dim = 1)
        ))

        all_rewards.append(torch.randn(1))

    # calculate advantages using z-score

    rewards = cat(all_rewards)
    advantages = z_score(rewards)

    assert advantages.shape == (3,)

    # simulate a policy loss update over the entire group

    group_states, group_log_probs, group_switch_betas, group_latent_actions = map(partial(cat, dim = 0), zip(*all_episodes))

    loss = policy_loss(
        meta_controller,
        group_states,
        group_log_probs,
        group_latent_actions,
        advantages,
        group_switch_betas == 1.,
        episode_lens = episode_lens[:1].repeat(3) if exists(episode_lens) else None
    )

    loss.backward()

    # evolutionary strategies over grpo

    model.meta_controller = meta_controller
    model.evolve(1, lambda _: 1., noise_population_size = 2)

    # saving and loading

    meta_controller.save('./meta_controller.pt')

    meta_controller_klass = meta_controller.__class__
    rehydrated_meta_controller = meta_controller_klass.init_and_load('./meta_controller.pt')

    model.save('./trained.pt')

    rehydrated_model = Transformer.init_and_load('./trained.pt', strict = False)

    Path('./meta_controller.pt').unlink()
    Path('./trained.pt').unlink()
