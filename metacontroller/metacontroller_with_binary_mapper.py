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

from assoc_scan import AssocScan

from torch_einops_utils import maybe, pad_at_dim, lens_to_mask, masked_mean, align_dims_left, pad_right_ndim_to
from torch_einops_utils.save_load import save_load

from vector_quantize_pytorch import BinaryMapper

from metacontroller.metacontroller import MetaControllerOutput

# constants

LinearNoBias = partial(Linear, bias = False)

GRU = partial(GRU, batch_first = True)

# helper functions

def exists(v):
    return v is not None

def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None

def straight_through(src, tgt):
    return tgt + src - src.detach()

def log(t, eps = 1e-20):
    return t.clamp_min(eps).log()

# meta controller

@save_load()
class MetaControllerWithBinaryMapper(Module):
    def __init__(
        self,
        dim_model,
        *,
        dim_meta_controller = 256,
        dim_code_bits = 4,
        switch_per_code = False,
        decoder_expansion_factor = 2.,
        decoder_depth = 1,
        hypernetwork_low_rank = 16,
        assoc_scan_kwargs: dict = dict(),
        bidirectional_temporal_encoder_kwargs: dict = dict(
            attn_dim_head = 32, heads = 8
        ),
        kl_loss_threshold = 0.
    ):
        super().__init__()

        assert not switch_per_code, 'switch_per_code is not supported for binary mapper'

        dim_meta = default(dim_meta_controller, dim_model)

        self.model_to_meta = Linear(dim_model, dim_meta)

        self.bidirectional_temporal_encoder = Encoder(dim = dim_meta, depth = 1, **bidirectional_temporal_encoder_kwargs)

        self.emitter = GRU(dim_meta * 2, dim_meta * 2)
        self.emitter_to_binary_logits = Linear(dim_meta * 2, dim_code_bits)

        self.action_proposer = GRU(dim_meta, dim_meta)
        self.proposer_to_binary_logits = Linear(dim_meta, dim_code_bits)

        # binary mapper
        # proposed in https://arxiv.org/abs/2510.17558 as a more stable alternative to VAE by François Fleuret

        self.binary_mapper = BinaryMapper(
            bits = dim_code_bits,
            kl_loss_threshold = kl_loss_threshold
        )

        self.dim_code_bits = dim_code_bits
        self.num_codes = self.binary_mapper.num_codes

        # switching unit

        self.switch_per_code = switch_per_code

        self.switching_unit = GRU(dim_meta + self.num_codes, dim_meta)
        self.to_switching_unit_beta = nn.Linear(dim_meta, self.num_codes if switch_per_code else 1, bias = False)

        self.switch_gating = AssocScan(**assoc_scan_kwargs)

        # decoder

        assert hypernetwork_low_rank < self.num_codes

        dim_decoder_hidden = int(self.num_codes * decoder_expansion_factor)

        self.decoder = Feedforwards(
            dim_in = self.num_codes,
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
            *self.emitter_to_binary_logits.parameters(),
            *self.binary_mapper.parameters(),
            *self.decoder.parameters(),
            *self.switch_gating.parameters()
        ]

    def internal_rl_parameters(self):
        return [
            *self.action_proposer.parameters(),
            *self.proposer_to_binary_logits.parameters()
        ]

    def log_prob(
        self,
        action_dist,
        sampled_latent_action
    ):
        action_prob = action_dist.sigmoid()
        probs = stack((action_prob, 1. - action_prob), dim = -1)
        log_probs = log(probs)

        indices = sampled_latent_action.argmax(dim = -1)
        codes = self.binary_mapper.codes[indices].long()

        codes = rearrange(codes, '... -> ... 1')
        action_log_probs = log_probs.gather(-1, codes)

        return rearrange(action_log_probs, '... 1 -> ...')

    def forward(
        self,
        residual_stream,
        cache: MetaControllerOutput | None = None,
        discovery_phase = False,
        hard_switch = None,
        temperature = 1.,
        episode_lens: Tensor | None = None
    ):
        device = residual_stream.device

        # destruct prev cache

        prev_action_proposer_hidden, prev_switching_unit_gru_hidden, prev_switch_gated_hiddens, prev_sampled_code = cache.prev_hiddens if exists(cache) else ((None,) * 4)

        # getting proposed action for the two phases

        next_action_proposer_hidden = None

        meta_embed = self.model_to_meta(residual_stream)

        hard_switch = default(hard_switch, not discovery_phase) # think during internal RL phase, it needs to be a hard switch, then only the actions emitted during the switch is reinforced

        if discovery_phase:
            mask = maybe(lens_to_mask)(episode_lens, meta_embed.shape[1])

            encoded_temporal = self.bidirectional_temporal_encoder(meta_embed, mask = mask)

            proposed_action_hidden, _ = self.emitter(cat((encoded_temporal, meta_embed), dim = -1))
            to_logits = self.emitter_to_binary_logits

        else: # else internal rl phase

            proposed_action_hidden, next_action_proposer_hidden = self.action_proposer(meta_embed, prev_action_proposer_hidden)
            to_logits = self.proposer_to_binary_logits

        # sample from the binary mapper

        binary_logits = to_logits(proposed_action_hidden)

        one_hot, kl_loss = self.binary_mapper(
            binary_logits,
            temperature = temperature,
            reduce_aux_kl_loss = False
        )

        # bottled action is now the one-hot sparse codes (with straight-through)

        sampled_codes = one_hot

        # switching unit timer

        batch, seq_len, dim = sampled_codes.shape

        if not exists(prev_sampled_code):
            prev_sampled_code = torch.zeros(batch, 1, self.num_codes, device = device)

        if discovery_phase:
            z_prev = cat((prev_sampled_code, sampled_codes[:, :-1]), dim = 1)
        else:
            assert seq_len == 1, f'inference RL phase must be done one token at a time'
            z_prev = prev_sampled_code

        switch_input = torch.cat((meta_embed, z_prev), dim=-1)

        switching_unit_gru_out, next_switching_unit_gru_hidden = self.switching_unit(
            switch_input, 
            prev_switching_unit_gru_hidden
        )

        switch_beta = self.to_switching_unit_beta(switching_unit_gru_out).sigmoid()

        # losses

        switch_loss = self.zero

        if discovery_phase:
            # weight unreduced kl loss by switch gates

            kl_loss, switch_beta = align_dims_left((kl_loss, switch_beta))

            weighted_kl_loss = kl_loss * switch_beta
            kl_loss = weighted_kl_loss.sum(dim = -1).mean()

            # encourage less switching

            switch_loss = switch_beta.mean()
        else:
            kl_loss = self.zero

        # maybe hard switch, then use associative scan

        if hard_switch:
            hard_switch_beta = (switch_beta > 0.5).float()
            switch_beta = straight_through(switch_beta, hard_switch_beta)

        forget = 1. - switch_beta

        # gated codes (or soft distribution)

        gated_codes = self.switch_gating(switch_beta, sampled_codes * forget, prev = prev_switch_gated_hiddens)

        next_switch_gated_codes = gated_codes[:, -1]

        # decoder

        decoder_out = self.decoder(gated_codes)

        w1, w2 = self.to_hyper_network_weights(decoder_out)
        hypernetwork_weight = einsum(w1, w2, '... i r, ... j r -> ... i j')

        # generating the residual stream controlling signal

        control_signal = einsum(residual_stream, hypernetwork_weight, '... d1, ... d1 d2 -> ... d1')

        # returning

        next_hiddens = (
            next_action_proposer_hidden,
            next_switching_unit_gru_hidden,
            next_switch_gated_codes,
            sampled_codes[:, -1:]
        )

        return control_signal, MetaControllerOutput(next_hiddens, residual_stream, binary_logits, sampled_codes, switch_beta, kl_loss, switch_loss)
