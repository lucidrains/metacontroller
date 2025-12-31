import pytest
param = pytest.mark.parametrize

import torch
from metacontroller.metacontroller import Transformer, MetaController

@param('action_discrete', (False, True))
@param('discovery_phase', (False, True))
@param('switch_per_latent_dim', (False, True))
def test_metacontroller(
    action_discrete,
    discovery_phase,
    switch_per_latent_dim
):

    state = torch.randn(1, 1024, 384)

    if action_discrete:
        actions = torch.randint(0, 4, (1, 1024))
        action_embed_readout = dict(num_discrete = 4)
        assert_shape = (4,)
    else:
        actions = torch.randn(1, 1024, 8)
        action_embed_readout = dict(num_continuous = 8)
        assert_shape = (8, 2)

    # behavioral cloning pahse

    model = Transformer(
        dim = 512,
        action_embed_readout = action_embed_readout,
        state_embed_readout = dict(num_continuous = 384),
        lower_body = dict(depth = 2,),
        upper_body = dict(depth = 2,),
    )

    state_clone_loss, action_clone_loss = model(state, actions)
    (state_clone_loss + 0.5 * action_clone_loss).backward()

    # discovery and internal rl phase with meta controller

    meta_controller = MetaController(
        dim_latent = 512,
        switch_per_latent_dim = switch_per_latent_dim
    )

    logits, cache = model(state, actions, meta_controller = meta_controller, discovery_phase = discovery_phase, return_cache = True)

    assert logits.shape == (1, 1024, *assert_shape)

    logits, cache = model(state, actions, meta_controller = meta_controller, discovery_phase = discovery_phase, return_cache = True, cache = cache)
    logits, cache = model(state, actions, meta_controller = meta_controller, discovery_phase = discovery_phase, return_cache = True, cache = cache)

    assert logits.shape == (1, 1, *assert_shape)

    model.meta_controller = meta_controller
    model.evolve(1, lambda _: 1., noise_population_size = 2)
