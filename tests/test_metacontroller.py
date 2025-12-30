import pytest
param = pytest.mark.parametrize

import torch
from metacontroller.metacontroller import Transformer, MetaController

@param('discovery_phase', (False, True))
@param('switch_per_latent_dim', (False, True))
def test_metacontroller(
    discovery_phase,
    switch_per_latent_dim
):

    ids = torch.randint(0, 256, (1, 1024))

    model = Transformer(
        512,
        embed = dict(num_discrete = 256),
        lower_body = dict(depth = 2,),
        upper_body = dict(depth = 2,),
        readout = dict(num_discrete = 256)
    )

    meta_controller = MetaController(
        512,
        switch_per_latent_dim = switch_per_latent_dim
    )

    logits = model(ids, meta_controller = meta_controller, discovery_phase = discovery_phase)

    assert logits.shape == (1, 1024, 256)
