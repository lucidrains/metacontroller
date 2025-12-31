from __future__ import annotations
from contextlib import nullcontext

from functools import partial
from collections import namedtuple
from loguru import logger

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

from discrete_continuous_embed_readout import Embed, Readout, EmbedAndReadout

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

# tensor helpers

def straight_through(src, tgt):
    return tgt + src - src.detach()

# meta controller

MetaControllerOutput = namedtuple('MetaControllerOutput', (
    'prev_hiddens',
    'action_dist',
    'actions',
    'kl_loss'
))

class MetaController(Module):
    def __init__(
        self,
        dim_latent,
        *,
        switch_per_latent_dim = True,
        decoder_expansion_factor = 2.,
        decoder_depth = 1,
        hypernetwork_low_rank = 16,
        assoc_scan_kwargs: dict = dict()
    ):
        super().__init__()

        # there are two phases, the first (discovery ssl phase) uses acausal with some ssm i don't really believe in - let's just use a bidirectional GRU as placeholders

        self.bidirectional_temporal_compressor = GRU(dim_latent, dim_latent, bidirectional = True) # revisit naming

        self.emitter = GRU(dim_latent * 2, dim_latent * 2)
        self.emitter_to_action_mean_log_var = Readout(dim_latent * 2, num_continuous = dim_latent)

        # internal rl phase substitutes the acausal + emitter with a causal ssm

        self.action_proposer = GRU(dim_latent, dim_latent)
        self.action_proposer_mean_log_var = Readout(dim_latent, num_continuous = dim_latent)

        # switching unit

        self.switch_per_latent_dim = switch_per_latent_dim

        self.switching_unit = GRU(dim_latent, dim_latent)
        self.to_switching_unit_beta = nn.Linear(dim_latent, dim_latent if switch_per_latent_dim else 1, bias = False)

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
            *self.emitter_to_action_mean_log_var.parameters(),
            *self.decoder.parameters(),
            *self.switch_gating.parameters()
        ]

    def internal_rl_parameters(self):
        return [
            *self.action_proposer.parameters(),
            *self.action_proposer_mean_log_var.parameters()
        ]

    def forward(
        self,
        residual_stream,
        cache: MetaControllerOutput | None = None,
        discovery_phase = False,
        hard_switch = False,
        temperature = 1.
    ):

        # destruct prev cache

        prev_action_proposer_hidden, prev_switching_unit_gru_hidden, prev_switch_gated_hiddens = cache.prev_hiddens if exists(cache) else ((None,) * 3)

        # getting proposed action for the two phases

        next_action_proposer_hidden = None

        if discovery_phase:
            logger.warning('meta controller cache being passed back in for discovery phase, which does not make sense given bidirectional encoder')

            temporal_compressed, _ = self.bidirectional_temporal_compressor(residual_stream)
            temporal_compressed = reduce(temporal_compressed, '... (two d) -> ... d', 'mean', two = 2)

            proposed_action_hidden, _ = self.emitter(cat((temporal_compressed, residual_stream), dim = -1))
            readout = self.emitter_to_action_mean_log_var

        else: # else internal rl phase

            proposed_action_hidden, next_action_proposer_hidden = self.action_proposer(residual_stream, prev_action_proposer_hidden)
            readout = self.action_proposer_mean_log_var

        # sample from the gaussian as the action from the meta controller

        action_dist = readout(proposed_action_hidden)

        sampled_action = readout.sample(action_dist, temperature = temperature)

        # switching unit timer

        batch, _, dim = sampled_action.shape

        switching_unit_gru_out, next_switching_unit_gru_hidden = self.switching_unit(residual_stream, prev_switching_unit_gru_hidden)

        switch_beta = self.to_switching_unit_beta(switching_unit_gru_out).sigmoid()

        # need to encourage normal distribution

        kl_loss = self.zero

        if discovery_phase:
            mean, log_var = action_dist.unbind(dim = -1)

            kl_loss = (0.5 * (
                log_var.exp()
                + mean.square()
                - log_var
                - 1.
            ))

            kl_loss = kl_loss * switch_beta
            kl_loss = kl_loss.sum(dim = -1).mean()

        # maybe hard switch, then use associative scan

        if hard_switch:
            hard_switch_beta = (switch_beta > 0.5).float()
            switch_beta = straight_through(switch_beta, hard_switch_beta)

        forget = 1. - switch_beta
        gated_action = self.switch_gating(switch_beta, sampled_action * forget, prev = prev_switch_gated_hiddens)

        next_switch_gated_action = gated_action[:, -1]

        # decoder

        decoder_out = self.decoder(gated_action)

        w1, w2 = self.to_hyper_network_weights(decoder_out)
        hypernetwork_weight = einsum(w1, w2, '... i r, ... j r -> ... i j')

        # generating the residual stream controlling signal

        control_signal = einsum(gated_action, hypernetwork_weight, '... d1, ... d1 d2 -> ... d1')

        modified_residual_stream = residual_stream + control_signal

        # returning

        next_hiddens = (
            next_action_proposer_hidden,
            next_switching_unit_gru_hidden,
            next_switch_gated_action
        )

        return modified_residual_stream, MetaControllerOutput(next_hiddens, action_dist, sampled_action, kl_loss)

# main transformer, which is subsumed into the environment after behavioral cloning

TransformerOutput = namedtuple('TransformerOutput', (
    'residual_stream_latent',
    'prev_hiddens'
))

class Transformer(Module):
    def __init__(
        self,
        dim,
        *,
        state_embed_readout: dict,
        action_embed_readout: dict,
        lower_body: Decoder | dict,
        upper_body: Decoder | dict,
        meta_controller: MetaController | None = None
    ):
        super().__init__()

        if isinstance(lower_body, dict):
            lower_body = Decoder(dim = dim, **lower_body)

        if isinstance(upper_body, dict):
            upper_body = Decoder(dim = dim, **upper_body)

        self.state_embed, self.state_readout = EmbedAndReadout(dim, **state_embed_readout)
        self.action_embed, self.action_readout = EmbedAndReadout(dim, **action_embed_readout)

        self.lower_body = lower_body
        self.upper_body = upper_body

        # meta controller

        self.meta_controller = meta_controller

        self.register_buffer('zero', tensor(0.), persistent = False)

    def evolve(
        self,
        num_generations,
        environment,
        **kwargs
    ):
        assert exists(self.meta_controller), '`meta_controller` must be passed in or defined on init for evolutionary strategies to be straightforwardly applied'

        evo_strat = EvoStrategy(
            self,
            num_generations = num_generations,
            environment = environment,
            params_to_optimize = self.meta_controller.internal_rl_parameters(),
            **kwargs
        )

        evo_strat()

    def forward(
        self,
        state,
        action_ids,
        meta_controller: Module | None = None,
        cache: TransformerOutput | None = None,
        discovery_phase = False,
        meta_controller_temperature = 1.,
        return_raw_action_dist = False,
        return_latents = False,
        return_cache = False,
    ):
        meta_controller = default(meta_controller, self.meta_controller)

        meta_controlling = exists(meta_controller)

        behavioral_cloning = not meta_controlling and not return_raw_action_dist

        # by default, if meta controller is passed in, transformer is no grad

        lower_transformer_context = nullcontext if not meta_controlling else torch.no_grad
        meta_controller_context = nullcontext if meta_controlling else torch.no_grad
        upper_transformer_context = nullcontext if (not meta_controlling or discovery_phase) else torch.no_grad

        # handle cache

        lower_transformer_hiddens, meta_hiddens, upper_transformer_hiddens = cache.prev_hiddens if exists(cache) else ((None,) * 3)

        # handle maybe behavioral cloning

        if behavioral_cloning:
            state, target_state = state[:, :-1], state[:, 1:]
            action_ids, target_action_ids = action_ids[:, :-1], action_ids[:, 1:]

        # transformer lower body

        with lower_transformer_context():

            state_embed = self.state_embed(state)
            action_embed = self.action_embed(action_ids)

            embed = state_embed + action_embed

            residual_stream, next_lower_hiddens = self.lower_body(embed, cache = lower_transformer_hiddens, return_hiddens = True)

        # meta controller acts on residual stream here

        with meta_controller_context():

            if exists(meta_controller):
                modified_residual_stream, next_meta_hiddens = meta_controller(residual_stream, cache = meta_hiddens, discovery_phase = discovery_phase, temperature = meta_controller_temperature)
            else:
                modified_residual_stream, next_meta_hiddens = residual_stream, None

        # modified residual stream sent back to transformer upper body

        with upper_transformer_context():

            attended, next_upper_hiddens = self.upper_body(modified_residual_stream, cache = upper_transformer_hiddens, return_hiddens = True)

            # head readout

            dist_params = self.action_readout(attended)

        # maybe return behavior cloning loss

        if behavioral_cloning:
            state_dist_params = self.state_readout(attended)
            state_clone_loss = self.state_readout.calculate_loss(state_dist_params, target_state)

            action_clone_loss = self.action_readout.calculate_loss(dist_params, target_action_ids)

            return state_clone_loss, action_clone_loss

        # returning

        return_one = not (return_latents or return_cache)

        if return_one:
            return dist_params

        return dist_params, TransformerOutput(residual_stream, (next_lower_hiddens, next_meta_hiddens, next_upper_hiddens))
