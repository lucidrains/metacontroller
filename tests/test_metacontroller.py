import pytest
param = pytest.mark.parametrize

from pathlib import Path

import torch
from metacontroller.metacontroller import Transformer, MetaController

from einops import rearrange

@param('action_discrete', (False, True))
@param('switch_per_latent_dim', (False, True))
def test_metacontroller(
    action_discrete,
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

    # behavioral cloning phase

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
        dim_model = 512,
        dim_meta_controller = 256,
        dim_latent = 128,
        switch_per_latent_dim = switch_per_latent_dim
    )

    # discovery phase

    (action_recon_loss, kl_loss, switch_loss) = model(state, actions, meta_controller = meta_controller, discovery_phase = True)
    (action_recon_loss + kl_loss * 0.1 + switch_loss * 0.2).backward()

    # internal rl - done iteratively

    cache = None
    past_action_id = None

    for one_state in state.unbind(dim = 1):
        one_state = rearrange(one_state, 'b d -> b 1 d')

        logits, cache = model(one_state, past_action_id, meta_controller = meta_controller, return_cache = True)

        assert logits.shape == (1, 1, *assert_shape)
        past_action_id = model.action_readout.sample(logits)

    # evolutionary strategies over grpo

    model.meta_controller = meta_controller
    model.evolve(1, lambda _: 1., noise_population_size = 2)

    # saving and loading

    meta_controller.save('./meta_controller.pt')

    rehydrated_meta_controller = MetaController.init_and_load('./meta_controller.pt')

    model.save('./trained.pt')

    rehydrated_model = Transformer.init_and_load('./trained.pt', strict = False)

    Path('./meta_controller.pt').unlink()
    Path('./trained.pt').unlink()
