from __future__ import annotations
from functools import partial

import torch
from torch import nn, cat, stack, tensor
from torch.nn import Module, GRU, Linear, Identity
import torch.nn.functional as F

# einops

import einx
from einops import einsum, rearrange, repeat, reduce
from einops.layers.torch import Rearrange

# external modules

from x_transformers import Decoder
from x_mlps_pytorch import Feedforwards
from x_evolution import EvoStrategy

from discrete_continuous_embed_readout import Embed, Readout

from assoc_scan import AssocScan

# constants

LinearNoBias = partial(Linear, bias = False)

GRU = partial(GRU, batch_first = True)

# helper functions

def exists(v):
    return v is not None

def identity(t):
    return t

def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None

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

        # there are two phases, the first (discovery ssl phase) uses acausal with some ssm i don't really believe in - let's just use a bidirectional GRU as placeholders

        self.bidirectional_temporal_compressor = GRU(dim_latent, dim_latent, bidirectional = True) # revisit naming

        self.emitter = GRU(dim_latent * 2, dim_latent * 2)
        self.emitter_to_action_mean_log_var = LinearNoBias(dim_latent * 2, dim_latent * 2)

        # internal rl phase substitutes the acausal + emitter with a causal ssm

        self.action_proposer = GRU(dim_latent, dim_latent)
        self.action_proposer_mean_log_var = LinearNoBias(dim_latent, dim_latent * 2)

        # switching unit

        self.switching_unit = GRU(dim_latent, dim_latent)
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

        self.register_buffer('zero', tensor(0.), persistent = False)

    def discovery_parameters(self):
        return [
            *self.bidirectional_temporal_compressor.parameters(),
            *self.emitter.parameters(),
            *self.emitter_to_action_mean_log_var.parameters()
            *self.decoder.parameters(),
            *self.switch_gating
        ]

    def internal_rl_parameters(self):
        return [
            *self.action_proposer.parameters(),
            *self.action_proposer_mean_log_var.parameters(),
            *self.decoder.parameters(),
            *self.switch_gating
        ]

    def forward(
        self,
        residual_stream,
        discovery_phase = False
    ):

        if discovery_phase:
            temporal_compressed, _ = self.bidirectional_temporal_compressor(residual_stream)
            temporal_compressed = reduce(temporal_compressed, '... (two d) -> ... d', 'mean', two = 2)

            proposed_action_hidden, _ = self.emitter(cat((temporal_compressed, residual_stream), dim = -1))
            proposed_action = self.emitter_to_action_mean_log_var(proposed_action_hidden)

        else: # else internal rl phase
            proposed_action_hidden, _ = self.action_proposer(residual_stream)
            proposed_action = self.action_proposer_mean_log_var(proposed_action_hidden)

        # sample from the gaussian as the action from the meta controller

        mean, log_var = proposed_action.chunk(2, dim = -1)

        std = (0.5 * log_var).exp()
        sampled_action_intents = mean + torch.randn_like(mean) * std

        # need to encourage normal distribution

        vae_kl_loss = self.zero

        if discovery_phase:
            vae_kl_loss = (0.5 * (
                log_var.exp()
                + mean.square()
                - log_var
                - 1.
            )).sum(dim = -1).mean()

        # switching unit timer

        batch, _, dim = sampled_action_intents.shape

        switching_unit_gru_out, switching_unit_gru_hidden = self.switching_unit(residual_stream)

        switch_beta = self.to_switching_unit_beta(switching_unit_gru_out).sigmoid()

        action_intent_for_gating = rearrange(sampled_action_intents, 'b n d -> (b d) n')
        switch_beta = repeat(switch_beta, 'b n 1 -> (b d) n', d = dim)

        forget = 1. - switch_beta
        gated_action_intent = self.switch_gating(action_intent_for_gating * forget, switch_beta)

        gated_action_intent = rearrange(gated_action_intent, '(b d) n -> b n d', b = batch)

        # decoder

        decoder_out = self.decoder(gated_action_intent)

        w1, w2 = self.to_hyper_network_weights(decoder_out)
        hypernetwork_weight = einsum(w1, w2, '... i r, ... j r -> ... i j')

        # generating the residual stream controlling signal

        control_signal = einsum(gated_action_intent, hypernetwork_weight, '... d1, ... d1 d2 -> ... d1')

        modified_residual_stream = residual_stream + control_signal

        return modified_residual_stream, vae_kl_loss

# main transformer, which is subsumed into the environment after behavioral cloning

class Transformer(Module):
    def __init__(
        self,
        dim,
        *,
        embed: Embed | dict,
        lower_body: Decoder | dict,
        upper_body: Decoder | dict,
        readout: Readout | dict,
        meta_controller: MetaController | None = None
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

        # meta controller

        self.meta_controller = meta_controller

    def evolve(
        self,
        environment,
        **kwargs
    ):
        assert exists(self.meta_controller), '`meta_controller` must be defined on init for evolutionary strategies to be straightforwardly applied'

        evo_strat = EvoStrategy(
            self,
            environment = environment,
            params_to_optimize = self.meta_controller.internal_rl_parameters(),
            **kwargs
        )

        evo_strat()

    def forward(
        self,
        ids,
        meta_controller: Module | None = None,
        discovery_phase = False,
        return_latents = False
    ):
        meta_controller = default(meta_controller, self.meta_controller, Identity())

        embed = self.embed(ids)

        residual_stream = self.lower_body(embed)

        # meta controller acts on residual stream here

        modified_residual_stream, vae_aux_loss = meta_controller(residual_stream, discovery_phase = discovery_phase)

        # modified residual stream sent back

        attended = self.upper_body(modified_residual_stream)

        dist_params = self.readout(attended)

        if not return_latents:
            return dist_params

        return dist_params, latents
