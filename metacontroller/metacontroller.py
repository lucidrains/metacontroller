from __future__ import annotations
from contextlib import nullcontext

from functools import partial
from collections import namedtuple
from loguru import logger

import torch
from torch import nn, cat, stack, tensor, Tensor
from torch.nn import Module, GRU, Linear, Identity
import torch.nn.functional as F

# einops

import einx
from einops import einsum, rearrange, repeat, reduce
from einops.layers.torch import Rearrange

# external modules

from x_transformers import Encoder, Decoder
from x_mlps_pytorch import Feedforwards
from x_evolution import EvoStrategy

from discrete_continuous_embed_readout import Embed, Readout, EmbedAndReadout

from assoc_scan import AssocScan

from torch_einops_utils import maybe, pad_at_dim, lens_to_mask
from torch_einops_utils.save_load import save_load

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
    'kl_loss',
    'switch_loss'
))

@save_load()
class MetaController(Module):
    def __init__(
        self,
        dim_model,
        *,
        dim_meta_controller = 256,
        dim_latent = 128,
        switch_per_latent_dim = True,
        decoder_expansion_factor = 2.,
        decoder_depth = 1,
        hypernetwork_low_rank = 16,
        assoc_scan_kwargs: dict = dict(),
        bidirectional_temporal_encoder_kwargs: dict = dict(
            attn_dim_head = 32,
            heads = 8
        )
    ):
        super().__init__()
        dim_meta = default(dim_meta_controller, dim_model)

        # the linear that brings from model dimension 

        self.model_to_meta = Linear(dim_model, dim_meta)

        # there are two phases, the first (discovery ssl phase) uses acausal with some ssm i don't really believe in - let's just use bidirectional attention as placeholder

        self.bidirectional_temporal_encoder = Encoder(dim = dim_meta, depth = 1, **bidirectional_temporal_encoder_kwargs)

        self.emitter = GRU(dim_meta * 2, dim_meta * 2)
        self.emitter_to_action_mean_log_var = Readout(dim_meta * 2, num_continuous = dim_latent)

        # internal rl phase substitutes the acausal + emitter with a causal ssm

        self.action_proposer = GRU(dim_meta, dim_meta)
        self.action_proposer_mean_log_var = Readout(dim_meta, num_continuous = dim_latent)

        # switching unit

        self.switch_per_latent_dim = switch_per_latent_dim


        self.dim_latent = dim_latent
        self.switching_unit = GRU(dim_meta + dim_latent, dim_meta)
        self.to_switching_unit_beta = nn.Linear(dim_meta, dim_latent if switch_per_latent_dim else 1, bias = False)

        self.switch_gating = AssocScan(**assoc_scan_kwargs)

        # decoder

        assert hypernetwork_low_rank < dim_latent

        dim_decoder_hidden = int(dim_latent * decoder_expansion_factor)

        self.decoder = Feedforwards(
            dim_in = dim_latent,
            dim = dim_decoder_hidden,
            depth = decoder_depth,
            dim_out = 2 * hypernetwork_low_rank * dim_model
        )

        self.to_hyper_network_weights = Rearrange('... (two d r) -> two ... d r', two = 2, r = hypernetwork_low_rank)

        self.register_buffer('zero', tensor(0.), persistent = False)

    def discovery_parameters(self):
        return [
            *self.model_to_meta.parameters(),
            *self.bidirectional_temporal_encoder.parameters(),
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
        temperature = 1.,
        episode_lens: Tensor | None = None
    ):
        device = residual_stream.device

        # destruct prev cache

        prev_action_proposer_hidden, prev_switching_unit_gru_hidden, prev_switch_gated_hiddens, prev_sampled_latent_action = cache.prev_hiddens if exists(cache) else ((None,) * 4)

        # getting proposed action for the two phases

        next_action_proposer_hidden = None

        meta_embed = self.model_to_meta(residual_stream)

        if discovery_phase:
            logger.warning('meta controller cache being passed back in for discovery phase, which does not make sense given bidirectional encoder')

            mask = maybe(lens_to_mask)(episode_lens, meta_embed.shape[1])

            encoded_temporal = self.bidirectional_temporal_encoder(meta_embed, mask = mask)

            proposed_action_hidden, _ = self.emitter(cat((encoded_temporal, meta_embed), dim = -1))
            readout = self.emitter_to_action_mean_log_var

        else: # else internal rl phase

            proposed_action_hidden, next_action_proposer_hidden = self.action_proposer(meta_embed, prev_action_proposer_hidden)
            readout = self.action_proposer_mean_log_var

        # sample from the gaussian as the action from the meta controller

        action_dist = readout(proposed_action_hidden)

        sampled_latent_action = readout.sample(action_dist, temperature = temperature)

        # switching unit timer

        batch, seq_len, dim = sampled_latent_action.shape

        # initialize prev sampled latent action to be zeros if not available (for first timestep and for discovery phase)

        if not exists(prev_sampled_latent_action):
            prev_sampled_latent_action = torch.zeros(batch, 1, self.dim_latent, device = device)

        if discovery_phase:
            z_prev = cat((prev_sampled_latent_action, sampled_latent_action[:, :-1]), dim = 1)

        else:
            # else during inference, use the previous sampled latent action

            assert seq_len == 1, f'inference RL phase must be done one token at a time'
            z_prev = prev_sampled_latent_action

        # switch input is previous latent action and the embedding

        switch_input = torch.cat((meta_embed, z_prev), dim=-1)

        switching_unit_gru_out, next_switching_unit_gru_hidden = self.switching_unit(
            switch_input, 
            prev_switching_unit_gru_hidden
        )

        switch_beta = self.to_switching_unit_beta(switching_unit_gru_out).sigmoid()

        # need to encourage normal distribution

        kl_loss = switch_loss = self.zero

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

            # encourage less switching

            switch_loss = switch_beta.mean()

        # maybe hard switch, then use associative scan

        if hard_switch:
            hard_switch_beta = (switch_beta > 0.5).float()
            switch_beta = straight_through(switch_beta, hard_switch_beta)

        forget = 1. - switch_beta
        gated_action = self.switch_gating(switch_beta, sampled_latent_action * forget, prev = prev_switch_gated_hiddens)

        next_switch_gated_action = gated_action[:, -1]

        # decoder

        decoder_out = self.decoder(gated_action)

        w1, w2 = self.to_hyper_network_weights(decoder_out)
        hypernetwork_weight = einsum(w1, w2, '... i r, ... j r -> ... i j')

        # generating the residual stream controlling signal

        control_signal = einsum(residual_stream, hypernetwork_weight, '... d1, ... d1 d2 -> ... d1')

        # returning

        next_hiddens = (
            next_action_proposer_hidden,
            next_switching_unit_gru_hidden,
            next_switch_gated_action,
            sampled_latent_action[:, -1:]
        )

        return control_signal, MetaControllerOutput(next_hiddens, action_dist, sampled_latent_action, kl_loss, switch_loss)

# main transformer, which is subsumed into the environment after behavioral cloning

TransformerOutput = namedtuple('TransformerOutput', (
    'residual_stream_latent',
    'prev_hiddens'
))

@save_load()
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
            lower_body = Decoder(dim = dim, pre_norm_has_final_norm = False, **lower_body)

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
        actions: Tensor | None = None,
        meta_controller: Module | None = None,
        cache: TransformerOutput | None = None,
        discovery_phase = False,
        meta_controller_temperature = 1.,
        return_raw_action_dist = False,
        return_latents = False,
        return_cache = False,
        episode_lens: Tensor | None = None
    ):
        device = state.device

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

        if behavioral_cloning or (meta_controlling and discovery_phase):
            assert exists(actions), f'`actions` cannot be empty when doing discovery or behavioral cloning'

            state, target_state = state[:, :-1], state[:, 1:]
            actions, target_actions = actions[:, :-1], actions[:, 1:]

            if exists(episode_lens):
                episode_lens = (episode_lens - 1).clamp(min = 0)

        # transformer lower body

        with lower_transformer_context():

            state_embed = self.state_embed(state)

            # handle no past action for first timestep

            if exists(actions):
                action_embed = self.action_embed(actions)
            else:
                action_embed = state_embed[:, 0:0] # empty action embed

            if action_embed.shape[-2] == (state_embed.shape[-2] - 1):
                action_embed = pad_at_dim(action_embed, (1, 0), dim = 1)

            embed = state_embed + action_embed

            residual_stream, next_lower_hiddens = self.lower_body(embed, cache = lower_transformer_hiddens, return_hiddens = True)

        # meta controller acts on residual stream here

        with meta_controller_context():

            if exists(meta_controller):
                control_signal, next_meta_hiddens = meta_controller(residual_stream, cache = meta_hiddens, discovery_phase = discovery_phase, temperature = meta_controller_temperature, episode_lens = episode_lens)
            else:
                control_signal, next_meta_hiddens = self.zero, None

            modified_residual_stream = residual_stream + control_signal

        # modified residual stream sent back to transformer upper body

        with upper_transformer_context():

            attended, next_upper_hiddens = self.upper_body(modified_residual_stream, cache = upper_transformer_hiddens, return_hiddens = True)

            # head readout

            dist_params = self.action_readout(attended)

        # maybe return behavior cloning loss

        if behavioral_cloning:
            loss_mask = None
            if exists(episode_lens):
                loss_mask = lens_to_mask(episode_lens, state.shape[1])

            state_dist_params = self.state_readout(attended)
            state_clone_loss = self.state_readout.calculate_loss(state_dist_params, target_state, mask = loss_mask)

            action_clone_loss = self.action_readout.calculate_loss(dist_params, target_actions, mask = loss_mask)

            return state_clone_loss, action_clone_loss

        elif meta_controlling and discovery_phase:

            action_recon_loss = self.action_readout.calculate_loss(dist_params, target_actions)

            return action_recon_loss, next_meta_hiddens.kl_loss, next_meta_hiddens.switch_loss

        # returning

        return_one = not (return_latents or return_cache)

        if return_one:
            return dist_params

        return dist_params, TransformerOutput(residual_stream, (next_lower_hiddens, next_meta_hiddens, next_upper_hiddens))
