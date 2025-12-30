from __future__ import annotations
from functools import partial

import torch
from torch import nn
from torch.nn import Module, GRU, Linear, Identity
import torch.nn.functional as F

# einops

import einx
from einops import einsum, rearrange, repeat
from einops.layers.torch import Rearrange

# external modules

from x_transformers import Decoder
from x_mlps_pytorch import Feedforwards

from discrete_continuous_embed_readout import Embed, Readout

from assoc_scan import AssocScan

# constants

LinearNoBias = partial(Linear, bias = False)

# helper functions

def exists(v):
    return v is not None

def identity(t):
    return t

def default(v, d):
    return v if exists(v) else d

# meta controller

class MetaController(Module):
    def __init__(
        self,
        dim_latent,
        *,
        decoder_expansion_factor = 2.,
        decoder_depth = 1,
        hypernetwork_low_rank = 16,
        assoc_scan_kwargs: dict = dict()
    ):
        super().__init__()

        # switching unit

        self.switching_unit = nn.GRU(dim_latent, dim_latent, batch_first = True)
        self.to_switching_unit_beta = nn.Linear(dim_latent, 1, bias = False)

        self.switch_gating = AssocScan(**assoc_scan_kwargs)

        # decoder

        assert hypernetwork_low_rank < dim_latent

        dim_decoder_hidden = int(dim_latent * decoder_expansion_factor)

        self.decoder = Feedforwards(
            dim_in = dim_latent,
            dim = dim_decoder_hidden,
            depth = decoder_depth,
            dim_out = 2 * hypernetwork_low_rank * dim_latent
        )

        self.to_hyper_network_weights = Rearrange('... (two d r) -> two ... d r', two = 2, r = hypernetwork_low_rank)

    def forward(
        self,
        latent
    ):

        switching_unit_gru_out, switching_unit_gru_hidden = self.switching_unit(latent)

        switch_beta = self.to_switching_unit_beta(switching_unit_gru_out).sigmoid()

        batch, _, dim = latent.shape
        latent_for_gating = rearrange(latent, 'b n d -> (b d) n')
        switch_beta = repeat(switch_beta, 'b n 1 -> (b d) n', d = dim)

        gated_latent = self.switch_gating(latent_for_gating, switch_beta)

        gated_latent = rearrange(gated_latent, '(b d) n -> b n d', b = batch)

        # decoder

        decoder_out = self.decoder(gated_latent)

        w1, w2 = self.to_hyper_network_weights(decoder_out)
        hypernetwork_weight = einsum(w1, w2, '... i r, ... j r -> ... i j')

        # generating the residual stream controlling signal

        control_signal = einsum(gated_latent, hypernetwork_weight, '... d1, ... d1 d2 -> ... d1')

        return latent + control_signal

# main transformer, which is subsumed into the environment after behavioral cloning

class Transformer(Module):
    def __init__(
        self,
        dim,
        *,
        embed: Embed | dict,
        lower_body: Decoder | dict,
        upper_body: Decoder | dict,
        readout: Readout | dict
    ):
        super().__init__()

        if isinstance(embed, dict):
            embed = Embed(dim = dim, **embed)

        if isinstance(lower_body, dict):
            lower_body = Decoder(dim = dim, **lower_body)

        if isinstance(upper_body, dict):
            upper_body = Decoder(dim = dim, **upper_body)

        if isinstance(readout, dict):
            readout = Readout(dim = dim, **readout)

        self.embed = embed
        self.lower_body = lower_body
        self.upper_body = upper_body
        self.readout = readout

    def forward(
        self,
        ids,
        meta_controller: Module = Identity(),
        return_latents = False
    ):
        embed = self.embed(ids)

        latents = self.lower_body(embed)

        # meta controller acts on latents here

        modified_latents = meta_controller(latents)

        # modified latents sent back

        attended = self.upper_body(latents)

        dist_params = self.readout(attended)

        if not return_latents:
            return dist_params

        return dist_params, latents
