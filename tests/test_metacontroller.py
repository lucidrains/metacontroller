import pytest
param = pytest.mark.parametrize

from pathlib import Path

import torch
from metacontroller.metacontroller import Transformer, MetaController
from metacontroller.metacontroller_with_binary_mapper import MetaControllerWithBinaryMapper

from einops import rearrange

@param('use_binary_mapper_variant', (False, True))
@param('action_discrete', (False, True))
@param('switch_per_latent_dim', (False, True))
@param('variable_length', (False, True))
def test_metacontroller(
    use_binary_mapper_variant,
    action_discrete,
    switch_per_latent_dim,
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
            dim_code_bits = 8, # 2**8 = 256 codes
        )

    # discovery phase

    (action_recon_loss, kl_loss, switch_loss) = model(state, actions, meta_controller = meta_controller, discovery_phase = True, episode_lens = episode_lens)
    (action_recon_loss + kl_loss * 0.1 + switch_loss * 0.2).backward()

    # internal rl - done iteratively

    cache = None
    past_action_id = None

    for one_state in state.unbind(dim = 1):
        one_state = rearrange(one_state, 'b d -> b 1 d')

        logits, cache = model(one_state, past_action_id, meta_controller = meta_controller, return_cache = True)

        assert logits.shape == (2, 1, *assert_shape)
        past_action_id = model.action_readout.sample(logits)

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
