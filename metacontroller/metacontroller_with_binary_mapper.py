from __future__ import annotations

from functools import partial
from loguru import logger

import torch
from torch import cat, tensor, Tensor
from torch.nn import Module, Linear, GRU
import torch.nn.functional as F

# einops

import einx
from einops import einsum, rearrange, repeat, reduce
from einops.layers.torch import Rearrange

# external modules

from x_transformers import Decoder
from x_mlps_pytorch import Feedforwards

from assoc_scan import AssocScan

from torch_einops_utils import maybe, lens_to_mask, and_masks, masked_mean, straight_through
from torch_einops_utils.shape import size
from torch_einops_utils.save_load import save_load

from vector_quantize_pytorch import BinaryMapper

from metacontroller.metacontroller import (
    exists,
    default,
    identity,
    ActionProposerWrapper,
    MetaControllerOutput,
    policy_loss,
    ratio_loss,
    reduce_discovery_samples,
    CausalSequenceEmbedder,
    sample_straight_through,
    categorical_entropy_loss
)
from metacontroller.sequential_action_selection import perform_sequential_action_selection

# constants

LinearNoBias = partial(Linear, bias = False)

GRU = partial(GRU, batch_first = True)

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
        hard_switch_straight_through = False,
        kl_loss_weight = 1.,
        kl_loss_warmup_steps = 0,
        apply_kl_loss_weight = True,
        gate_kl_with_switch_beta = False,
        hard_prob = 1.,
        binary_temperature = 1.,
        use_marginal_entropy_loss = False,
        sample_entropy_weight = 0.1,
        ratio_loss_chunk_size = None,
        ratio_loss_final_weight = None,
        ratio_loss_warmdown_steps = 0,
        discovery_num_samples = 1,
        discovery_softmin_temp = 0.1
    ):
        super().__init__()
        self.dim_model = dim_model
        self.hard_switch = hard_switch
        self.hard_switch_straight_through = hard_switch_straight_through

        self.kl_loss_weight = kl_loss_weight
        self.kl_loss_warmup_steps = kl_loss_warmup_steps
        self.register_buffer('kl_loss_step_count', tensor(0.))

        self.apply_kl_loss_weight = apply_kl_loss_weight
        self.gate_kl_with_switch_beta = gate_kl_with_switch_beta
        self.hard_prob = hard_prob
        self.binary_temperature = binary_temperature
        self.use_marginal_entropy_loss = use_marginal_entropy_loss
        self.sample_entropy_weight = sample_entropy_weight

        self.discovery_num_samples = discovery_num_samples
        self.discovery_softmin_temp = discovery_softmin_temp

        dim_meta = default(dim_meta_controller, dim_model)

        self.summary_gru = GRU(dim_model, dim_model)

        self.model_to_meta = Linear(dim_model, dim_meta)

        if isinstance(internal_sequence_embedder, dict):
            internal_sequence_embedder = CausalSequenceEmbedder(dim = dim_model, pool = pool_embedded_sequence, **internal_sequence_embedder)

        self.internal_sequence_embedder = internal_sequence_embedder

        self.to_sequence_summary_embed = Linear(dim_model, dim_sequence_summary_embed)

        self.emitter = Feedforwards(
            dim_in = dim_meta + dim_model + dim_sequence_summary_embed,
            dim = dim_meta * 2,
            dim_out = dim_meta * 2,
            depth = 2
        )
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

        assert hypernetwork_low_rank <= dim_model

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
        return [
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

    def discovery_parameters_switching_unit(self):
        return [
            *self.switching_unit.parameters()
        ]

    def discovery_parameters_non_switching(self):
        return [
            *self.summary_gru.parameters(),
            *self.model_to_meta.parameters(),
            *self.internal_sequence_embedder.parameters(),
            *self.to_sequence_summary_embed.parameters(),
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
        return self.binary_mapper.log_prob(action_dist, one_hot=sampled_latent_action, sum_bits=False)

    def forward(
        self,
        residual_stream,
        cache: MetaControllerOutput | None = None,
        discovery_phase = False,
        hard_switch: bool | None = None,
        hard_switch_straight_through: bool | None = None,
        ablate_switch_beta: Tensor | None = None,
        switch_beta_frequency: int | None = None,
        ablate_offset = 0,
        temperature = 1.,
        episode_lens: Tensor | None = None,
        num_samples: int | None = None
    ):
        seq_len = size(residual_stream, 'b [n] d')
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
                logger.warning('meta controller cache being passed back in for discovery phase, which does not make sense during discovery')

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

            proposed_action_hidden = self.emitter(emitter_input)
            to_logits = self.emitter_to_binary_logits

        else: # else internal rl phase

            proposed_action_hidden, next_action_proposer_hidden = self.action_proposer(
                residual_stream,
                cache = prev_action_proposer_hidden
            )

            to_logits = self.proposer_to_binary_logits

        # sample from the binary mapper

        binary_logits = to_logits(proposed_action_hidden)

        effective_temp = self.binary_temperature * temperature
        num_samples = default(num_samples, 1)
        has_multi_samples = num_samples > 1
        use_soft_code_probs = has_multi_samples or self.hard_prob < 1.

        if use_soft_code_probs or (discovery_phase and self.use_marginal_entropy_loss):
            codes_float = self.binary_mapper.codes.float()
            scaled_logits = binary_logits / effective_temp

            soft_G = (
                einsum(F.logsigmoid(scaled_logits), codes_float, '... bits, codes bits -> ... codes') +
                einsum(F.logsigmoid(-scaled_logits), 1. - codes_float, '... bits, codes bits -> ... codes')
            ).exp()

        if use_soft_code_probs:
            sample_probs = repeat(soft_G, 'b ... -> n b ...', n = num_samples) if has_multi_samples else soft_G
            one_hot = sample_straight_through(sample_probs, self.hard_prob, training = self.training)
            one_hot = rearrange(one_hot, 'n b ... -> (n b) ...') if has_multi_samples else one_hot

            kl_loss = self.zero if self.use_marginal_entropy_loss else self.binary_mapper(binary_logits, temperature = effective_temp, reduce_aux_kl_loss = False, calc_aux_loss = discovery_phase)[1]
        else:
            one_hot, kl_loss = self.binary_mapper(
                binary_logits,
                temperature = effective_temp,
                reduce_aux_kl_loss = False,
                calc_aux_loss = discovery_phase and not self.use_marginal_entropy_loss
            )

        # bottled action is now the one-hot sparse codes (with straight-through)

        sampled_codes = one_hot

        # switching unit timer

        batch, seq_len, _ = sampled_codes.shape
        input_residual_stream = residual_stream
        repeat_batch = partial(repeat, pattern = 'b ... -> (n b) ...', n = num_samples) if has_multi_samples else identity

        if has_multi_samples:
            repeat_hidden = partial(repeat, pattern = 'l b d -> l (n b) d', n = num_samples)

            residual_stream = repeat_batch(residual_stream)
            meta_embed_prev = repeat_batch(meta_embed_prev)
            ablate_switch_beta = maybe(repeat_batch)(ablate_switch_beta)
            prev_switching_unit_hidden = maybe(repeat_hidden)(prev_switching_unit_hidden)
            prev_switch_gated_hiddens = maybe(repeat_batch)(prev_switch_gated_hiddens)

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
                resolved_hard_switch,
                default(hard_switch_straight_through, self.hard_switch_straight_through),
                is_first_step = not exists(cache)
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

                # force beta to 1 on first step (no cache) to match parallel path

                if not exists(cache):
                    switch_beta = torch.ones_like(switch_beta)

            # maybe hard switch, then use associative scan

            switch_beta_for_gate = switch_beta

            if hard_switch:
                hard_switch_beta = (switch_beta_for_gate > 0.5).float()

                if default(hard_switch_straight_through, self.hard_switch_straight_through):
                    switch_beta_for_gate = straight_through(switch_beta_for_gate, hard_switch_beta)
                else:
                    switch_beta_for_gate = hard_switch_beta

            forget_gate = 1. - switch_beta_for_gate

            # gated codes (or soft distribution)
            gated_sampled_codes = einx.multiply('b n d, b n', sampled_codes, switch_beta_for_gate)

            gated_codes = self.switch_gating(forget_gate, gated_sampled_codes, prev = prev_switch_gated_hiddens)

            next_switch_gated_codes = gated_codes[:, -1:]

        # losses

        kl_loss_weight = self.zero

        if discovery_phase:
            kl_loss_weight = self.current_kl_loss_weight if self.apply_kl_loss_weight else 1.
            switch_beta_for_kl = reduce(switch_beta, '(n b) l -> b l', 'mean', n = num_samples) if has_multi_samples else switch_beta

            if self.use_marginal_entropy_loss:
                switch_mask = switch_beta_for_kl > 0.5
                kl_mask = and_masks((mask, switch_mask)) if self.gate_kl_with_switch_beta else mask
                kl_loss = categorical_entropy_loss(soft_G, mask = kl_mask, sample_entropy_weight = self.sample_entropy_weight)
            else:
                if kl_loss.ndim == 3:
                    kl_loss = reduce(kl_loss, 'b n d -> b n', 'sum')

                if self.gate_kl_with_switch_beta and kl_loss.ndim == 2:
                    kl_loss = kl_loss * switch_beta_for_kl

                kl_loss = masked_mean(kl_loss, mask)

            kl_loss = kl_loss * kl_loss_weight
        else:
            kl_loss = self.zero

        decoder_out = self.decoder(gated_codes)

        w1, w2 = self.to_hyper_network_weights(decoder_out)
        hypernetwork_weight = einsum(w1, w2, '... i r, ... j r -> ... i j')

        # generating the residual stream controlling signal

        control_signal = einsum(residual_stream, hypernetwork_weight, '... d1, ... d1 d2 -> ... d2')

        # maybe ratio loss

        aux_ratio_loss = self.zero

        if self.has_ratio_loss:
            aux_ratio_loss = self.ratio_loss(
                self.target_temporal_segment_len,
                switch_beta,
                episode_lens = maybe(repeat_batch)(episode_lens),
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

        return control_signal, MetaControllerOutput(next_hiddens, input_residual_stream, binary_logits, sampled_codes, switch_beta, kl_loss, kl_loss_weight, aux_ratio_loss, gated_codes)

MetaControllerWithBinaryMapper.policy_loss = policy_loss
MetaControllerWithBinaryMapper.ratio_loss = ratio_loss
MetaControllerWithBinaryMapper.reduce_discovery_samples = reduce_discovery_samples
