from __future__ import annotations

from metacontroller.metacontroller import ActionProposerWrapper
from contextlib import nullcontext

from functools import partial
from collections import namedtuple
from loguru import logger

import torch
from torch import nn, cat, tensor, ones, zeros, stack, Tensor
from torch.nn import Module, Linear, GRU, Parameter
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity, sigmoid

# einops

import einx
from einops import einsum, rearrange, repeat, reduce
from einops.layers.torch import Rearrange

# external modules

from x_transformers import Encoder, Decoder, Attention
from x_mlps_pytorch import Feedforwards

from assoc_scan import AssocScan

from torch_einops_utils import maybe, pad_at_dim, lens_to_mask, align_dims_left, masked_mean
from torch_einops_utils.save_load import save_load

from vector_quantize_pytorch import BinaryMapper

from metacontroller.metacontroller import MetaControllerOutput, policy_loss, ratio_loss, BidirectionalSequenceEmbedder, CausalSequenceEmbedder
from metacontroller.sequential_action_selection import perform_sequential_action_selection

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

# meta controller classes

@save_load()
class GRUSwitchingUnit(Module):
    def __init__(
        self,
        dim_model,
        dim_meta,
        num_codes,
        switch_temperature = 0.1
    ):
        super().__init__()
        self.gru = GRU(dim_model + dim_meta + num_codes, dim_meta)
        self.to_beta = Linear(dim_meta, 1, bias = False)
        self.switch_temperature = switch_temperature

    def forward(
        self,
        residual_stream,
        meta_embed_prev,
        sampled_codes_prev,
        prev_hidden = None
    ):
        switch_input = cat((residual_stream, meta_embed_prev, sampled_codes_prev), dim = -1)

        gru_out, next_hidden = self.gru(switch_input, prev_hidden)

        beta_logit = self.to_beta(gru_out)
        beta = (beta_logit / self.switch_temperature).sigmoid()
        beta = rearrange(beta, '... 1 -> ...')

        return beta, next_hidden

@save_load()
class MetaControllerWithBinaryMapper(Module):
    def __init__(
        self,
        dim_model,
        *,
        dim_meta_controller = 256,
        dim_code_bits = 4,
        decoder_expansion_factor = 2.,
        decoder_depth = 1,
        hypernetwork_low_rank = 16,
        assoc_scan_kwargs: dict = dict(),
        internal_sequence_embedder: Module | dict = dict(
            attn_dim_head = 32,
            heads = 8,
            depth = 2,
            polar_pos_emb = True
        ),
        bidirectional = True,
        action_proposer: Module | dict = dict(
            depth = 2,
            attn_dim_head = 32,
            heads = 8,
            polar_pos_emb = True
        ),
        kl_loss_threshold = 0.,
        switch_temperature = 0.1,
        target_temporal_segment_len = 4, # set to target segment length driven by ratio loss
        ratio_loss_weight = 1.,
        pool_embedded_sequence = True,
        dim_sequence_summary_embed = 32,
        hard_switch = None,
        kl_loss_weight = 1.,
        kl_loss_warmup_steps = 0,
        apply_kl_loss_weight = True,
        ratio_loss_chunk_size = None,
        ratio_loss_final_weight = None,
        ratio_loss_warmdown_steps = 0,
    ):
        super().__init__()
        self.dim_model = dim_model
        self.hard_switch = hard_switch

        self.kl_loss_weight = kl_loss_weight
        self.kl_loss_warmup_steps = kl_loss_warmup_steps
        self.register_buffer('kl_loss_step_count', tensor(0.))

        self.apply_kl_loss_weight = apply_kl_loss_weight

        dim_meta = default(dim_meta_controller, dim_model)

        self.summary_gru = GRU(dim_model, dim_model)

        self.model_to_meta = Linear(dim_model, dim_meta)

        if isinstance(internal_sequence_embedder, dict):
            embedder_klass = BidirectionalSequenceEmbedder if bidirectional else CausalSequenceEmbedder

            if bidirectional:
                internal_sequence_embedder['pool'] = pool_embedded_sequence

            internal_sequence_embedder = embedder_klass(dim = dim_model, **internal_sequence_embedder)

        self.internal_sequence_embedder = internal_sequence_embedder

        self.to_sequence_summary_embed = Linear(dim_model, dim_sequence_summary_embed)

        self.emitter = GRU(dim_meta + dim_model + dim_sequence_summary_embed, dim_meta * 2)
        self.emitter_to_binary_logits = Linear(dim_meta * 2, dim_code_bits)

        # internal rl phase substitutes the acausal + emitter with a causal ssm
        # can be configurable, but defaults to x-transformers.Decoder

        if isinstance(action_proposer, dict):
            action_proposer = ActionProposerWrapper(
                Decoder(dim = dim_model, **action_proposer),
                cache_key = 'cache',
                return_cache_key = 'return_hiddens'
            )

        self.action_proposer = action_proposer
        self.proposer_to_binary_logits = Linear(dim_model, dim_code_bits)

        # binary mapper
        # proposed in https://arxiv.org/abs/2510.17558 as a more stable alternative to VAE by François Fleuret

        self.binary_mapper = BinaryMapper(
            bits = dim_code_bits,
            kl_loss_threshold = kl_loss_threshold
        )

        self.dim_code_bits = dim_code_bits
        self.num_codes = self.binary_mapper.num_codes

        self.switching_unit = GRUSwitchingUnit(
            dim_model = dim_model,
            dim_meta = dim_meta_controller,
            num_codes = self.num_codes,
            switch_temperature = switch_temperature
        )

        self.switch_temperature = switch_temperature

        self.switch_gating = AssocScan(**assoc_scan_kwargs)

        # turn off the ratio loss by setting the weight to 0

        assert ratio_loss_weight >= 0.

        self.has_ratio_loss = ratio_loss_weight > 0.

        self.ratio_loss_weight = ratio_loss_weight
        self.target_temporal_segment_len = target_temporal_segment_len
        self.ratio_loss_chunk_size = ratio_loss_chunk_size

        self.ratio_loss_final_weight = ratio_loss_final_weight
        self.ratio_loss_warmdown_steps = ratio_loss_warmdown_steps

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

    @staticmethod
    def create_regular_switch_beta(batch, seq_len, frequency, offset = 0, device = None):
        steps = torch.arange(seq_len, device = device) + offset
        switch_beta = ((steps + 1) % frequency == 0).float()
        return repeat(switch_beta, 'n -> b n', b = batch)

    @property
    def replay_buffer_field_dict(self):
        return dict(
            states = ('float', self.dim_model),
            log_probs = ('float', self.dim_code_bits),
            switch_betas = 'float',
            latent_actions = ('float', self.num_codes)
        )

    def maybe_increment_kl_loss_step(self):
        if self.kl_loss_warmup_steps > 0:
            self.kl_loss_step_count.add_(1)

    def reset_kl_loss_warmup(self):
        self.kl_loss_step_count.zero_()

    @property
    def current_kl_loss_weight(self):
        if self.kl_loss_warmup_steps == 0:
            return self.kl_loss_weight

        step = self.kl_loss_step_count.item()
        warmup_factor = min(1.0, step / self.kl_loss_warmup_steps)
        return self.kl_loss_weight * warmup_factor

    @property
    def current_ratio_loss_weight(self):
        if not exists(self.ratio_loss_final_weight) or self.ratio_loss_warmdown_steps == 0:
            return self.ratio_loss_weight

        step = self.kl_loss_step_count.item()
        warmdown_factor = min(1.0, step / self.ratio_loss_warmdown_steps)
        return self.ratio_loss_weight + (self.ratio_loss_final_weight - self.ratio_loss_weight) * warmdown_factor

    def discovery_parameters(self):
        params = [
            *self.summary_gru.parameters(),
            *self.model_to_meta.parameters(),
            *self.internal_sequence_embedder.parameters(),
            *self.to_sequence_summary_embed.parameters(),
            *self.emitter.parameters(),
            *self.emitter_to_binary_logits.parameters(),
            *self.binary_mapper.parameters(),
            *self.switching_unit.parameters(),
            *self.decoder.parameters(),
            *self.switch_gating.parameters()
        ]

        return params

    def internal_rl_parameters(self):
        return [
            *self.action_proposer.parameters(),
            *self.proposer_to_binary_logits.parameters()
        ]

    def train_internal_rl(self, eval_rest = False):
        if eval_rest:
            self.eval()

        self.action_proposer.train()
        self.proposer_to_binary_logits.train()

    def get_action_dist_for_internal_rl(
        self,
        residual_stream,
        return_kl_loss = False
    ):
        proposed_action_hidden, _ = self.action_proposer(residual_stream)

        logits = self.proposer_to_binary_logits(proposed_action_hidden)

        if not return_kl_loss:
            return logits

        return logits, self.calculate_kl_loss(logits)

    def calculate_kl_loss(
        self,
        binary_logits
    ):
        # hack for now, need to fix binary mapper upstream at vq-pytorch

        _, kl_loss = self.binary_mapper(binary_logits, reduce_aux_kl_loss = False)
        return kl_loss

    def log_prob(
        self,
        action_dist,
        sampled_latent_action
    ):
        log_probs = stack((
            F.logsigmoid(action_dist),
            F.logsigmoid(-action_dist)
        ), dim = -1)

        indices = sampled_latent_action.argmax(dim = -1)
        codes = self.binary_mapper.codes[indices].long()

        codes = rearrange(codes, '... -> ... 1')
        action_log_probs = log_probs.gather(-1, codes)
        action_log_probs = rearrange(action_log_probs, '... 1 -> ...')

        return action_log_probs

    def forward(
        self,
        residual_stream,
        cache: MetaControllerOutput | None = None,
        discovery_phase = False,
        hard_switch = None,
        ablate_switch_beta: Tensor | None = None,
        switch_beta_frequency: int | None = None,
        ablate_offset = 0,
        temperature = 1.,
        episode_lens: Tensor | None = None
    ):
        seq_len = residual_stream.shape[1]
        device = residual_stream.device

        # destruct prev cache

        prev_summarized, prev_action_proposer_hidden, prev_switching_unit_hidden, prev_switch_gated_hiddens = cache.prev_hiddens if exists(cache) else ((None,) * 4)

        # getting proposed action for the two phases

        next_action_proposer_hidden = None

        # summarizing the input residual stream, then projecting it to meta controller dimension

        summarized, next_summarized = self.summary_gru(residual_stream, prev_summarized)

        meta_embed = self.model_to_meta(summarized)

        # construct meta embed (projected summarized) with a previous, so it can be used for emitter and switching unit

        if not exists(prev_summarized):
            prev_summarized = torch.zeros_like(next_summarized)

        prev_summarized = rearrange(prev_summarized, 'n b d -> b n d')

        meta_embed_prev = cat((
            self.model_to_meta(prev_summarized),
            meta_embed[:, :-1]
        ), dim = 1)

        hard_switch = default(hard_switch, self.hard_switch, not discovery_phase) # think during internal RL phase, it needs to be a hard switch, then only the actions emitted during the switch is reinforced

        if discovery_phase:

            if exists(prev_action_proposer_hidden):
                logger.warning('meta controller cache being passed back in for discovery phase, which does not make sense given bidirectional encoder')

            mask = maybe(lens_to_mask)(episode_lens, meta_embed.shape[1])

            kwargs = dict()
            if exists(episode_lens):
                kwargs['episode_lens'] = episode_lens

            encoded_temporal = self.internal_sequence_embedder(residual_stream, mask = mask, **kwargs)

            summarized_sequence_embed = self.to_sequence_summary_embed(encoded_temporal)

            emitter_input = cat((
                residual_stream,
                meta_embed_prev,
                summarized_sequence_embed
            ), dim = -1)

            proposed_action_hidden, _ = self.emitter(emitter_input)
            to_logits = self.emitter_to_binary_logits

        else: # else internal rl phase

            proposed_action_hidden, next_action_proposer_hidden = self.action_proposer(
                residual_stream,
                cache = prev_action_proposer_hidden
            )

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

        batch, seq_len, _ = sampled_codes.shape

        if not exists(prev_switch_gated_hiddens):
            prev_switch_gated_hiddens = torch.zeros(batch, 1, self.num_codes, device = device)

        z_prev = prev_switch_gated_hiddens

        is_ablating_switch = exists(ablate_switch_beta) or exists(switch_beta_frequency)

        if seq_len > 1 and not is_ablating_switch:

            # resolve hard switch

            resolved_hard_switch = default(hard_switch, self.hard_switch, not discovery_phase)

            # call jax

            sequential_selection_output = perform_sequential_action_selection(
                self.switching_unit.gru,
                self.switching_unit.to_beta,
                residual_stream,
                meta_embed_prev,
                sampled_codes,
                prev_switching_unit_hidden,
                z_prev,
                self.switch_temperature,
                resolved_hard_switch
            )

            switch_beta = sequential_selection_output.switch_beta
            gated_codes = sequential_selection_output.gated_action
            next_switching_unit_hidden = sequential_selection_output.next_switching_unit_gru_hidden

            next_switch_gated_codes = gated_codes[:, -1:]

        else:
            if exists(switch_beta_frequency):
                ablate_switch_beta = self.create_regular_switch_beta(batch, seq_len, switch_beta_frequency, offset = ablate_offset, device = device)

            if is_ablating_switch:
                switch_beta = ablate_switch_beta

                if switch_beta.ndim == 1:
                    switch_beta = rearrange(switch_beta, 'b -> b 1')

                next_switching_unit_hidden = prev_switching_unit_hidden
            else:
                switch_beta, next_switching_unit_hidden = self.switching_unit(
                    residual_stream,
                    meta_embed_prev,
                    z_prev,
                    prev_switching_unit_hidden
                )

            # maybe hard switch, then use associative scan

            switch_beta_for_gate = switch_beta

            if hard_switch:
                hard_switch_beta = (switch_beta_for_gate > 0.5).float()
                switch_beta_for_gate = straight_through(switch_beta_for_gate, hard_switch_beta)

            forget_gate = 1. - switch_beta_for_gate

            # gated codes (or soft distribution)
            gated_sampled_codes = einx.multiply('b n d, b n', sampled_codes, switch_beta_for_gate)

            gated_codes = self.switch_gating(forget_gate, gated_sampled_codes, prev = prev_switch_gated_hiddens)

            next_switch_gated_codes = gated_codes[:, -1:]

        # losses

        if discovery_phase:
            kl_loss_weight = self.current_kl_loss_weight if self.apply_kl_loss_weight else 1.

            if kl_loss.ndim == 3:
                kl_loss = reduce(kl_loss, 'b n d -> b n', 'sum')

            kl_loss = masked_mean(kl_loss, mask)
            kl_loss = kl_loss * kl_loss_weight
        else:
            kl_loss = self.zero
            kl_loss_weight = self.zero

        decoder_out = self.decoder(gated_codes)

        w1, w2 = self.to_hyper_network_weights(decoder_out)
        hypernetwork_weight = einsum(w1, w2, '... i r, ... j r -> ... i j')

        # generating the residual stream controlling signal

        control_signal = einsum(residual_stream, hypernetwork_weight, '... d1, ... d1 d2 -> ... d1')

        # maybe ratio loss

        aux_ratio_loss = self.zero

        if self.has_ratio_loss:

            aux_ratio_loss = self.ratio_loss(
                self.target_temporal_segment_len,
                switch_beta,
                episode_lens = episode_lens,
                chunk_size = self.ratio_loss_chunk_size
            )

            aux_ratio_loss = aux_ratio_loss * self.current_ratio_loss_weight

        # returning

        next_hiddens = (
            next_summarized,
            next_action_proposer_hidden,
            next_switching_unit_hidden,
            next_switch_gated_codes
        )

        return control_signal, MetaControllerOutput(next_hiddens, residual_stream, binary_logits, sampled_codes, switch_beta, kl_loss, kl_loss_weight, aux_ratio_loss)

MetaControllerWithBinaryMapper.policy_loss = policy_loss
MetaControllerWithBinaryMapper.ratio_loss = ratio_loss
