from __future__ import annotations

import torch
from torch import nn, Tensor
from torch.nn import Module, ModuleList
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce

from metacontroller.metacontroller import Transformer
from discrete_continuous_embed_readout import EmbedAndReadout

from torch_einops_utils.save_load import save_load

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# convnext components

class ConvNextBlock(Module):
    def __init__(
        self,
        dim,
        expansion_factor = 4,
        kernel_size = 3
    ):
        super().__init__()
        self.dw_conv = nn.Conv2d(dim, dim, kernel_size = kernel_size, padding = kernel_size // 2, groups = dim)
        self.norm = nn.LayerNorm(dim)
        self.pw_mlp = nn.Sequential(
            nn.Linear(dim, expansion_factor * dim),
            nn.GELU(),
            nn.Linear(expansion_factor * dim, dim)
        )

    def forward(self, x):
        # x: (batch, dim, height, width)
        residual = x
        x = self.dw_conv(x)
        x = rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        x = self.pw_mlp(x)
        x = rearrange(x, 'b h w c -> b c h w')
        return residual + x

class SpatialProcessor(Module):
    def __init__(
        self,
        dim,
        depth = 4,
        kernel_size = 3
    ):
        super().__init__()
        self.layers = ModuleList([])
        for _ in range(depth):
            self.layers.append(ConvNextBlock(dim, kernel_size = kernel_size))

        self.norm = nn.LayerNorm(dim, bias = False)

    def forward(self, x):
        # x: (batch, time, width, height, dim)
        batch, time, width, height, dim = x.shape
        x = rearrange(x, 'b t w h d -> (b t) d w h')
        for layer in self.layers:
            x = layer(x)

        x = rearrange(x, '(b t) d w h -> b t w h d', b = batch, t = time)
        return self.norm(x)

# wrapped readout for spatial projection

class WrappedReadout(Module):
    def __init__(
        self,
        projection,
        readout
    ):
        super().__init__()
        self.projection = projection
        self.readout = readout

    def forward(self, x):
        x = self.projection(x)
        return self.readout(x)

    def calculate_loss(self, *args, **kwargs):
        return self.readout.calculate_loss(*args, **kwargs)

# transformer for symbolic grid

@save_load()
class TransformerForSymbolicGrid(Transformer):
    def __init__(
        self,
        dim,
        num_discrete = (11, 6, 3), # Type, Color, State
        grid_size = 7,
        spatial_processor_depth = 4,
        convnext_kernel_size = 3,
        **kwargs
    ):
        assert 'state_embed_readout' not in kwargs, 'state_embed_readout should not be passed to TransformerForSymbolicGrid, use num_discrete instead'

        kwargs['state_embed_readout'] = dict(
            num_discrete = num_discrete
        )

        super().__init__(dim = dim, **kwargs)

        self.symbolic_embed, self.symbolic_readout = self.state_embed, self.state_readout

        self.spatial_processor = SpatialProcessor(
            dim,
            depth = spatial_processor_depth,
            kernel_size = convnext_kernel_size
        )

        self.to_model_dim = Reduce('b t w h d -> b t d', 'mean')

        # override state_embed and state_readout to use symbolic pipeline

        self.state_embed = nn.Sequential(
            self.symbolic_embed,
            self.spatial_processor,
            self.to_model_dim
        )

        projection = nn.Sequential(
            nn.Linear(dim, (grid_size ** 2) * dim),
            Rearrange('b t (w h d) -> b t w h d', w = grid_size, h = grid_size, d = dim)
        )

        self.state_readout = WrappedReadout(projection, self.symbolic_readout)
