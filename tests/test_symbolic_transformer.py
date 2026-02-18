
import torch
from torch import nn
from metacontroller import TransformerForSymbolicGrid, MetaController

def test_symbolic_transformer():
    batch = 2
    time = 5
    width = 7
    height = 7
    channels = 3
    dim = 512

    meta_controller = MetaController(
        dim_model = dim
    )

    transformer = TransformerForSymbolicGrid(
        dim,                             # model dimension
        num_discrete = (11, 6, 3),       # Type, Color, State
        grid_size = 7,                   # Square grid dimension
        action_embed_readout = dict(
            num_discrete = 10            # Action discrete buckets
        ),
        lower_body = dict(
            depth = 2                    # Transformer lower body depth
        ),
        upper_body = dict(
            depth = 2                    # Transformer upper body depth
        ),
        meta_controller = meta_controller
    )

    state = torch.randint(0, 3, (batch, time, width, height, channels))
    actions = torch.randint(0, 10, (batch, time))

    # Discovery testing
    discovery_losses = transformer(
        state, 
        actions = actions, 
        discovery_phase = True
    )
    (discovery_losses.state_pred + discovery_losses.action_recon).backward()

    # BC testing
    bc_losses = transformer(
        state, 
        actions = actions, 
        force_behavior_cloning = True
    )
    (bc_losses.state + bc_losses.action).backward()
