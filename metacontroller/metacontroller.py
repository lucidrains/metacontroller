from __future__ import annotations
from contextlib import nullcontext

from functools import partial
from collections import namedtuple
from loguru import logger

import torch
from torch import nn, cat, stack, tensor, is_tensor, Tensor
from torch.nn import Module, GRU, Linear, Identity

import torch.nn.functional as F

from metacontroller.sequential_action_selection import perform_sequential_action_selection, SequentialActionSelectionOutput

# einops

import einx
from einops import einsum, rearrange, repeat, reduce
from einops.layers.torch import Rearrange

# external modules

from x_transformers import Encoder, Decoder
from x_transformers.x_transformers import PolarEmbedding, Identity

# sequential selection moved to separate module

from x_mlps_pytorch import Feedforwards
from x_evolution import EvoStrategy

from discrete_continuous_embed_readout import Embed, Readout, EmbedAndReadout

from assoc_scan import AssocScan

from torch_einops_utils import maybe, pad_left_at_dim, pad_at_dim, lens_to_mask, masked_mean, align_dims_left
from torch_einops_utils.device import module_device, move_inputs_to_module_device
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

def log(t, eps = 1e-8):
    return (t + eps).log()

def binary_entropy(p):
    return -((p * log(p)) + (1. - p) * log(1. - p))

def straight_through(src, tgt):
    return tgt + src - src.detach()

# action proposer wrapper
# normalizes any action proposer to a standard interface for MetaController

@save_load()
class ActionProposerWrapper(Module):
    def __init__(
        self,
        module: Module,
        cache_key = 'cache',
        return_cache_key = 'return_hiddens'
    ):
        super().__init__()
        self.module = module
        self.cache_key = cache_key
        self.return_cache_key = return_cache_key

    def forward(
        self,
        x,
        cache = None
    ):
        kwargs = {self.return_cache_key: True}

        if exists(cache):
            kwargs[self.cache_key] = cache

        out = self.module(x, **kwargs)

        return out if isinstance(out, tuple) else (out, None)

# losses

BehavioralCloningLosses = namedtuple('BehavioralCloningLosses', (
    'state',
    'action'
))

DiscoveryLosses = namedtuple('DiscoveryLosses', (
    'state_pred',
    'action_recon',
    'kl',
    'ratio'
))

SequentialSelectionOutput = namedtuple('SequentialSelectionOutput', (
    'switch_beta',
    'gated_action',
    'next_switching_unit_gru_hidden'
))

# moved to sequential_selection.py

class LossNormalizer(Module):
    def __init__(
        self,
        num_losses = 1,
        beta = 0.999,
        eps = 1e-6
    ):
        super().__init__()
        self.register_buffer('exp_avg_sq', torch.ones(num_losses))
        self.beta = beta
        self.eps = eps

    @property
    def loss_scale(self):
        return self.exp_avg_sq.sqrt()

    def forward(
        self,
        losses: Tensor,
        update_ema = None
    ):
        exp_avg_sq, beta = self.exp_avg_sq, self.beta
        update_ema = default(update_ema, self.training)

        # get the rms value - as mentioned at the end of section 3 in the paper

        rms = exp_avg_sq.sqrt()

        if update_ema:
            decay = 1. - beta

            # update the ema

            exp_avg_sq.lerp_(losses.detach().square(), decay)

        # then normalize

        assert losses.numel() == rms.numel()

        normed_losses = losses / rms.clamp(min = self.eps)

        return normed_losses

# meta controller classes

@save_load()
class BidirectionalSequenceEmbedder(Module):
    def __init__(
        self,
        dim,
        pool = True,
        **kwargs
    ):
        super().__init__()
        self.pool = pool
        self.encoder = Encoder(dim = dim, **kwargs)

    def forward(
        self,
        x,
        mask = None,
        **kwargs
    ):
        encoded = self.encoder(x, mask = mask)

        if not self.pool:
            return encoded

        mean_pooled = masked_mean(encoded, mask, dim = 1)
        return repeat(mean_pooled, 'b d -> b n d', n = x.shape[1])

@save_load()
class CausalSequenceEmbedder(Module):
    def __init__(
        self,
        dim,
        **kwargs
    ):
        super().__init__()
        self.decoder = Decoder(dim = dim, **kwargs)

    def forward(
        self,
        x,
        mask = None,
        **kwargs
    ):
        return self.decoder(x, mask = mask)

# meta controller

MetaControllerOutput = namedtuple('MetaControllerOutput', (
    'prev_hiddens',
    'input_residual_stream',
    'action_dist',
    'actions',
    'switch_beta',
    'kl_loss',
    'kl_loss_weight',
    'ratio_loss'
))

GRPOOutput = namedtuple('GRPOOutput', (
    'state',
    'action',
    'log_prob',
    'switch_beta'
))

def z_score(t, dim = None, eps = 1e-8):
    kwargs = dict(dim = dim, keepdim = True) if exists(dim) else dict()
    return (t - t.mean(**kwargs)) / (t.std(**kwargs) + eps)

@move_inputs_to_module_device
def policy_loss(
    meta_controller,
    state,
    old_log_probs,
    actions,
    advantages,
    mask = None,
    episode_lens = None,
    eps_clip: float | tuple[float, float] = 0.2,
    switch_beta_frequency: int | None = None,
    return_kl_loss = False,
    kl_loss_weight = 0.
):
    if exists(switch_beta_frequency):
        batch, seq_len = state.shape[:2]
        mask = meta_controller.create_regular_switch_beta(batch, seq_len, switch_beta_frequency, device = state.device) > 0.5

    # if switch betas sent as mask without converting to bool, just take care of it

    if mask.dtype == torch.float:
        mask = mask > 0.5

    # get new log probs

    action_dist_out = meta_controller.get_action_dist_for_internal_rl(state, return_kl_loss = return_kl_loss)

    if return_kl_loss:
        action_dist, kl_loss = action_dist_out
    else:
        action_dist = action_dist_out

    new_log_probs = meta_controller.log_prob(action_dist, actions)

    # calculate ratio

    ratio = (new_log_probs - old_log_probs).exp()

    # align ratio and advantages

    ratio, advantages = align_dims_left((ratio, advantages))

    # ppo surrogate loss

    surr1 = ratio * advantages

    if isinstance(eps_clip, (float, int)):
        eps_clip = (eps_clip, eps_clip)

    eps_lower, eps_upper = eps_clip
    ratio_clip = ratio.clamp(1 - eps_lower, 1 + eps_upper)

    surr2 = ratio_clip * advantages

    losses = -torch.min(surr1, surr2)

    # masking

    if mask.ndim == 3:
        mask = rearrange(mask, 'b n 1 -> b n')

    if exists(episode_lens):
        episode_mask = lens_to_mask(episode_lens, mask.shape[1])
        mask = mask & episode_mask

    losses = reduce(losses, 'b n d -> b n', 'sum')

    grpo_loss = masked_mean(losses, mask)

    if not return_kl_loss:
        return grpo_loss

    if kl_loss.ndim == 3:
        kl_loss = reduce(kl_loss, 'b n d -> b n', 'sum')

    kl_loss = masked_mean(kl_loss, mask)

    return grpo_loss, kl_loss * kl_loss_weight

@move_inputs_to_module_device
def ratio_loss(
    meta_controller,
    target_segment_len: int,
    switch_beta,                # float[b n]
    hard_switch_beta = None,    # bool[b n]
    episode_lens = None,        # int[b]
    chunk_size = None
):
    # Hwang et al. https://arxiv.org/abs/2507.07955 eq (10)

    seq_len = switch_beta.shape[-1]
    mask = maybe(lens_to_mask)(episode_lens, seq_len)

    if not exists(hard_switch_beta):
        hard_switch_beta = switch_beta > 0.5

    if exists(chunk_size):
        num_chunks = seq_len // chunk_size

        if num_chunks > 1:
            chunk_limit = num_chunks * chunk_size
            chunk_fn = lambda t: rearrange(t[:, :chunk_limit], 'b (k n) -> (b k) n', n = chunk_size)

            switch_beta = chunk_fn(switch_beta)
            hard_switch_beta = maybe(chunk_fn)(hard_switch_beta)
            mask = maybe(chunk_fn)(mask)

    N = target_segment_len
    G = masked_mean(switch_beta, mask, dim = -1)
    F = masked_mean(hard_switch_beta.float(), mask, dim = -1)

    losses = (N / (N - 1.)) * ((N - 1.) * F * G + (1. - F) * (1. - G))

    return losses.mean()

def extract_grpo_data(meta_controller, transformer_output):
    meta_output = transformer_output.prev_hiddens.meta_controller

    state = meta_output.input_residual_stream
    action = meta_output.actions
    switch_beta = meta_output.switch_beta

    log_prob = meta_controller.log_prob(meta_output.action_dist, action)

    return GRPOOutput(state, action, log_prob, switch_beta)

@save_load()
class MetaController(Module):
    def __init__(
        self,
        dim_model,
        *,
        dim_meta_controller = 256,
        dim_latent = 128,
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
        compact_sequence_embedding = False,
        bidirectional = True,
        pool_embedded_sequence = True,
        dim_sequence_summary_embed = 32, # the summary embedding from the bidirectional network needs to be bottlenecked
        action_proposer: Module | dict = dict(
            depth = 2,
            attn_dim_head = 32,
            heads = 8,
            polar_pos_emb = True
        ),
        switch_temperature = 1.,
        target_temporal_segment_len = 4, # set to target segment length driven by ratio loss
        ratio_loss_weight = 1.,
        ratio_loss_chunk_size = None,
        hard_switch = None,
        kl_loss_weight = 1.,
        kl_loss_warmup_steps = 0,
        apply_kl_loss_weight = True,
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
        
        self.ratio_loss_final_weight = ratio_loss_final_weight
        self.ratio_loss_warmdown_steps = ratio_loss_warmdown_steps

        self.compact_sequence_embedding = compact_sequence_embedding
        
        dim_meta = default(dim_meta_controller, dim_model)

        # the linear that brings from model dimension 

        self.summary_gru = GRU(dim_model, dim_model) # eq (12)

        self.model_to_meta = Linear(dim_model, dim_meta)

        # there are two phases, the first (discovery ssl phase) uses acausal with some ssm i don't really believe in - let's just use bidirectional attention as placeholder

        # use transformer encoder and map the input to a self-attnded sequence

        if not self.compact_sequence_embedding and isinstance(internal_sequence_embedder, dict):
            embedder_klass = BidirectionalSequenceEmbedder if bidirectional else CausalSequenceEmbedder
            
            if bidirectional:
                internal_sequence_embedder['pool'] = pool_embedded_sequence

            internal_sequence_embedder = embedder_klass(dim = dim_model, **internal_sequence_embedder)
            self.internal_sequence_embedder = internal_sequence_embedder

        # compact sequence embedder -> use GRU last hidden state

        else:
            self.internal_sequence_embedder = GRU(dim_model, dim_model)

        self.to_sequence_summary_embed = Linear(dim_model, dim_sequence_summary_embed)

        self.emitter = GRU(dim_meta + dim_model + dim_sequence_summary_embed, dim_meta * 2)

        self.emitter_to_action_mean_log_var = Readout(dim_meta * 2, num_continuous = dim_latent)

        if isinstance(action_proposer, dict):
            # default to Decoder, wrapped for standard interface

            action_proposer = ActionProposerWrapper(
                Decoder(dim = dim_model, **action_proposer),
                cache_key = 'cache',
                return_cache_key = 'return_hiddens'
            )

        self.action_proposer = action_proposer
        self.action_proposer_mean_log_var = Readout(dim_model, num_continuous = dim_latent)

        # switching unit

        self.dim_latent = dim_latent
        self.switching_unit = GRU(dim_model + dim_meta + dim_latent, dim_meta)

        self.to_switching_unit_beta = nn.Linear(dim_meta, 1, bias = False)
        nn.init.xavier_uniform_(self.to_switching_unit_beta.weight)

        self.switch_temperature = switch_temperature

        self.switch_gating = AssocScan(**assoc_scan_kwargs)

        # turn off the ratio loss by setting the weight to 0

        assert ratio_loss_weight >= 0.

        self.has_ratio_loss = ratio_loss_weight > 0.

        self.ratio_loss_weight = ratio_loss_weight
        self.ratio_loss_chunk_size = ratio_loss_chunk_size
        self.target_temporal_segment_len = target_temporal_segment_len

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

    @staticmethod
    def create_regular_switch_beta(batch, seq_len, frequency, offset = 0, device = None):
        steps = torch.arange(seq_len, device = device) + offset
        switch_beta = ((steps + 1) % frequency == 0).float()
        return repeat(switch_beta, 'n -> b n', b = batch)

    @property
    def replay_buffer_field_dict(self):
        return dict(
            states = ('float', self.dim_model),
            log_probs = ('float', self.dim_latent),
            switch_betas = 'float',
            latent_actions = ('float', self.dim_latent)
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
            *self.emitter_to_action_mean_log_var.parameters(),
            *self.switching_unit.parameters(),
            *self.to_switching_unit_beta.parameters(),
            *self.decoder.parameters(),
            *self.switch_gating.parameters()
        ]

    def discovery_parameters_switching_unit(self):
        return [
            *self.switching_unit.parameters(),
            *self.to_switching_unit_beta.parameters(),
        ]

    def discovery_parameters_non_switching(self):
        return [
            *self.summary_gru.parameters(),
            *self.model_to_meta.parameters(),
            *self.internal_sequence_embedder.parameters(),
            *self.to_sequence_summary_embed.parameters(),
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

    def train_internal_rl(self, eval_rest = False):

        if isinstance(self.action_proposer, ActionProposerWrapper):
            self.action_proposer.module.train()
        else:
            self.action_proposer.train()

        self.action_proposer_mean_log_var.train()

    def get_action_dist_for_internal_rl(
        self,
        residual_stream,
        return_kl_loss = False
    ):

        proposed_action_hidden, _ = self.action_proposer(residual_stream)

        action_dist = self.action_proposer_mean_log_var(proposed_action_hidden)

        if not return_kl_loss:
            return action_dist

        return action_dist, self.calculate_kl_loss(action_dist)

    def calculate_kl_loss(
        self,
        action_dist
    ):
        mean, log_var = action_dist.unbind(dim = -1)

        kl_loss = (0.5 * (
            log_var.exp()
            + mean.square()
            - log_var
            - 1.
        ))

        return kl_loss

    def log_prob(
        self,
        action_dist,
        sampled_latent_action
    ):
        return self.action_proposer_mean_log_var.log_prob(action_dist, sampled_latent_action)

    def forward(
        self,
        residual_stream,
        cache: MetaControllerOutput | None = None,
        discovery_phase = False,
        hard_switch: bool | None = None,
        ablate_switch_beta: Tensor | None = None,
        switch_beta_frequency: int | None = None,
        ablate_offset = 0,
        temperature = 1.,
        episode_lens: Tensor | None = None
    ):
        seq_len = residual_stream.shape[1]
        device = residual_stream.device

        # destruct prev cache

        prev_summarized, prev_action_proposer_hidden, prev_switching_unit_gru_hidden, gated_prev_latent = cache.prev_hiddens if exists(cache) else ((None,) * 4)

        # getting proposed action for the two phases

        next_action_proposer_hidden = None

        # eq (12) - summarizing the input residual stream, then projecting it to meta controller dimension
        # h_t
        summarized, next_summarized = self.summary_gru(residual_stream, prev_summarized)

        meta_embed = self.model_to_meta(summarized)

        # construct meta embed (projected summarized) with a previous, so it can be used for emitter and switching unit
        # convert to batch first

        if not exists(prev_summarized):
            prev_summarized = torch.zeros_like(next_summarized)

        prev_summarized = rearrange(prev_summarized, 'n b d -> b n d')

        # their h_(t-1)

        # B, L, D
        meta_embed_prev = cat((
            self.model_to_meta(prev_summarized),
            meta_embed[:, :-1]
        ), dim = 1)

        # depending on discovery or internal RL phases

        if discovery_phase:

            if exists(prev_action_proposer_hidden):
                logger.warning('meta controller cache being passed back in for discovery phase, which does not make sense given bidirectional encoder')

            mask = maybe(lens_to_mask)(episode_lens, meta_embed.shape[1])

            kwargs = dict()
            if exists(episode_lens):
                kwargs['episode_lens'] = episode_lens

            # s(e_1:T) is a single compact embedding

            if self.compact_sequence_embedding:

                # pack_padded_sequence requires lengths on CPU (PyTorch contract)
                lengths_cpu = episode_lens.cpu().long()
                packed_residual_streams = nn.utils.rnn.pack_padded_sequence(
                    residual_stream, lengths_cpu, batch_first=True, enforce_sorted=False
                )

                # f_emb
                _, summarized_sequence = self.internal_sequence_embedder(packed_residual_streams) # B, L, D -> layer, B, D
                summarized_sequence = summarized_sequence[-1] # B, D

                # channel-mixing projection
                summarized_sequence_embed = self.to_sequence_summary_embed(summarized_sequence) # B, 32
                summarized_sequence_embed = summarized_sequence_embed.unsqueeze(1).repeat(1, residual_stream.shape[1], 1) # B, L, 32

            # otherwise treat it as an encoder output sequence (if it's the same)

            else:
                encoded_residual_stream = self.internal_sequence_embedder(residual_stream, mask = mask, **kwargs)
                summarized_sequence_embed = self.to_sequence_summary_embed(encoded_residual_stream)

            # eq 15

            # residual_stream
            emitter_input = cat((
                residual_stream,
                meta_embed_prev,
                summarized_sequence_embed
            ), dim = -1)

            proposed_action_hidden, _ = self.emitter(emitter_input)

            readout = self.emitter_to_action_mean_log_var

        else: # else internal rl phase

            proposed_action_hidden, next_action_proposer_hidden = self.action_proposer(
                residual_stream,
                cache = prev_action_proposer_hidden
            )

            readout = self.action_proposer_mean_log_var

        # sample from the gaussian as the action from the meta controller

        action_dist = readout(proposed_action_hidden)

        sampled_latent_action = readout.sample(action_dist, temperature = temperature)

        # switching unit timer

        batch, seq_len, dim = sampled_latent_action.shape

        # initialize prev sampled latent action / gated action to be zeros if not available (for first timestep and for discovery phase)

        if not exists(gated_prev_latent):
            gated_prev_latent = torch.zeros(batch, 1, self.dim_latent, device = device)

        z_prev = gated_prev_latent

        is_ablating_switch = exists(ablate_switch_beta) or exists(switch_beta_frequency)

        if seq_len > 1 and not is_ablating_switch:

            # resolve hard switch

            resolved_hard_switch = default(hard_switch, self.hard_switch, not discovery_phase)

            # call jax
            
            sequential_selection_output = perform_sequential_action_selection(
                self.switching_unit,
                self.to_switching_unit_beta,
                residual_stream,
                meta_embed_prev,
                sampled_latent_action,
                prev_switching_unit_gru_hidden,
                z_prev,
                self.switch_temperature,
                resolved_hard_switch
            )
            
            switch_beta = sequential_selection_output.switch_beta
            gated_action = sequential_selection_output.gated_action
            next_switching_unit_gru_hidden = sequential_selection_output.next_switching_unit_gru_hidden

            next_switch_gated_action = gated_action[:, -1:]

        else:
            if exists(switch_beta_frequency):
                ablate_switch_beta = self.create_regular_switch_beta(batch, seq_len, switch_beta_frequency, offset = ablate_offset, device = device)

            if is_ablating_switch:
                switch_beta = ablate_switch_beta

                if switch_beta.ndim == 1:
                    switch_beta = rearrange(switch_beta, 'b -> b 1')

                next_switching_unit_gru_hidden = prev_switching_unit_gru_hidden
            else:
                # switch input is previous latent action and the embedding
                # eq (16)

                switch_input = cat((
                    residual_stream,
                    meta_embed_prev,
                    z_prev
                ), dim = -1)

                switching_unit_gru_out, next_switching_unit_gru_hidden = self.switching_unit(
                    switch_input, 
                    prev_switching_unit_gru_hidden
                )

                switch_beta_logit = self.to_switching_unit_beta(switching_unit_gru_out)

                switch_beta = (switch_beta_logit / self.switch_temperature).sigmoid()
                switch_beta = rearrange(switch_beta, '... 1 -> ...')

            # maybe hard switch

            hard_switch = default(hard_switch, self.hard_switch, not discovery_phase)

            switch_beta_for_gate = switch_beta

            if hard_switch:
                hard_switch_beta = (switch_beta_for_gate > 0.5).float()
                switch_beta_for_gate = straight_through(switch_beta_for_gate, hard_switch_beta)

            # associative scan

            forget_gate = 1. - switch_beta_for_gate

            gated_sampled_latent_action = einx.multiply('b n d, b n', sampled_latent_action, switch_beta_for_gate)
            gated_action = self.switch_gating(forget_gate, gated_sampled_latent_action, prev = gated_prev_latent)

            next_switch_gated_action = gated_action[:, -1:]

        # need to encourage normal distribution

        kl_loss = self.zero

        if discovery_phase:
            kl_loss = self.calculate_kl_loss(action_dist)

            kl_loss = reduce(kl_loss, 'b n d -> b n', 'sum')
            kl_loss = masked_mean(kl_loss, mask)

            kl_loss_weight = self.current_kl_loss_weight if self.apply_kl_loss_weight else 1.
            kl_loss = kl_loss * kl_loss_weight

        else:
            kl_loss = self.zero
            kl_loss_weight = self.zero

        # decoder

        decoder_out = self.decoder(gated_action)

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
            next_switching_unit_gru_hidden,
            next_switch_gated_action
        )

        return control_signal, MetaControllerOutput(next_hiddens, residual_stream, action_dist, sampled_latent_action, switch_beta, kl_loss, kl_loss_weight if discovery_phase else self.zero, aux_ratio_loss)

MetaController.policy_loss = policy_loss
MetaController.ratio_loss = ratio_loss

# main transformer, which is subsumed into the environment after behavioral cloning

Hiddens = namedtuple('Hiddens', (
    'lower_body',
    'meta_controller',
    'upper_body'
))

TransformerOutput = namedtuple('TransformerOutput', (
    'residual_stream_latent',
    'prev_hiddens',
    'cache_steps'
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
        meta_controller: MetaController | None = None,
        dim_condition = None,
        state_loss_detach_target_state = True,
        embed_past_actions = True,
        normalize_state_action_losses = False,
        loss_normalizer_beta = 0.999
    ):
        super().__init__()

        self.normalize_state_action_losses = normalize_state_action_losses
        self.bc_state_loss_normalizer = None
        self.bc_action_loss_normalizer = None
        self.discovery_state_loss_normalizer = None
        self.discovery_action_loss_normalizer = None

        if normalize_state_action_losses:
            self.bc_state_loss_normalizer = LossNormalizer(num_losses = 1, beta = loss_normalizer_beta)
            self.bc_action_loss_normalizer = LossNormalizer(num_losses = 1, beta = loss_normalizer_beta)
            self.discovery_state_loss_normalizer = LossNormalizer(num_losses = 1, beta = loss_normalizer_beta)
            self.discovery_action_loss_normalizer = LossNormalizer(num_losses = 1, beta = loss_normalizer_beta)

        if exists(dim_condition):
            transformer_kwargs = dict(
                dim_condition = dim_condition,
                use_adaptive_rmsnorm = True,
                polar_pos_emb = True
            )
        else:
            transformer_kwargs = dict(
                use_rmsnorm = True,
                polar_pos_emb = True
            )

        if isinstance(lower_body, dict):
            lower_body = Decoder(dim = dim, pre_norm_has_final_norm = False, **transformer_kwargs, **lower_body)

            # x_transformers passes condition into final_norm when need_condition; nn.Identity() rejects kwargs → use wrapper
            # remove at later date, should be fixed in latest x-transformers
            lower_body.final_norm = Identity()

        if isinstance(upper_body, dict):
            upper_body = Decoder(dim = dim, **transformer_kwargs, **upper_body)

        self.state_embed, self.state_readout = EmbedAndReadout(dim, **state_embed_readout)
        action_embed, self.action_readout = EmbedAndReadout(dim, **action_embed_readout)

        self.action_embed = action_embed if embed_past_actions else None

        self.lower_body = lower_body
        self.upper_body = upper_body

        # polar positional embedding

        lower_attn_dim_head = lower_body.attn_dim_head
        lower_heads = lower_body.attn_heads

        upper_attn_dim_head = upper_body.attn_dim_head
        upper_heads = upper_body.attn_heads

        assert lower_attn_dim_head == upper_attn_dim_head
        assert lower_heads == upper_heads

        # meta controller

        self.meta_controller = meta_controller 

        # detaching the target state for the state loss - for the visual encoder based latent state ar prediction

        self.state_loss_detach_target_state = state_loss_detach_target_state

        self.register_buffer('zero', tensor(0.), persistent = False)

        # ensure devices match
        
        if exists(self.meta_controller): self._ensure_consistent_device(self.meta_controller)

    def meta_controller_maybe_increment_kl_loss_step(self):
        if not exists(self.meta_controller):
            return

        m = self.meta_controller
        if hasattr(m, 'module'):
            m = m.module

        m.maybe_increment_kl_loss_step()

    def meta_controller_reset_kl_loss_warmup(self):
        if not exists(self.meta_controller):
            return

        self.meta_controller.reset_kl_loss_warmup()

    @property
    def meta_controller_kl_loss_weight(self):
        if not exists(self.meta_controller):
            return None

        return self.meta_controller.kl_loss_weight

    @property
    def meta_controller_current_kl_loss_weight(self):
        if not exists(self.meta_controller):
            return 0.

        return self.meta_controller.current_kl_loss_weight

    @property
    def running_bc_state_loss(self):
        if not self.normalize_state_action_losses:
            return None

        return self.bc_state_loss_normalizer.loss_scale

    @property
    def running_bc_action_loss(self):
        if not self.normalize_state_action_losses:
            return None

        return self.bc_action_loss_normalizer.loss_scale

    @property
    def running_discovery_state_loss(self):
        if not self.normalize_state_action_losses:
            return None

        return self.discovery_state_loss_normalizer.loss_scale

    @property
    def running_discovery_action_loss(self):
        if not self.normalize_state_action_losses:
            return None

        return self.discovery_action_loss_normalizer.loss_scale

    def _ensure_consistent_device(self, network):
        self.model_device = module_device(self)
        if module_device(network) != self.model_device:
            network.to(self.model_device)

    def train_discovery(
        self,
        eval_rest = False,
        meta_controller = None
    ):
        meta_controller = default(meta_controller, self.meta_controller)
        assert exists(meta_controller)

        if eval_rest:
            self.eval()

        meta_controller.train()

    def train_internal_rl(
        self,
        eval_rest = False,
        meta_controller = None
    ):
        meta_controller = default(meta_controller, self.meta_controller)
        assert exists(meta_controller)

        if eval_rest:
            self.eval()

        meta_controller.train_internal_rl()

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
        force_behavior_cloning = False,
        meta_controller_temperature = 1.,
        return_raw_action_dist = False,
        return_latents = False,
        return_cache = False,
        return_state_action_cache = False,
        episode_lens: Tensor | None = None,
        return_meta_controller_output = False,
        return_residual_stream = False,
        return_action_logits = False,
        condition = None,
        return_embed = False,
        control_signal_multiplier = 1.,
        ablate_switch_beta: Tensor | None = None,
        switch_beta_frequency: int | None = None,
        update_loss_ema: bool | None = None,
        hard_switch: bool | None = None
    ):

        device = state.device

        # meta controller is either given or already given at init

        if exists(meta_controller): self._ensure_consistent_device(meta_controller)

        meta_controller = default(meta_controller, self.meta_controller)

        if force_behavior_cloning:
            assert not discovery_phase, 'discovery phase cannot be set to True if force behavioral cloning is set to True'
            meta_controller = None

        has_meta_controller = exists(meta_controller)

        assert not (discovery_phase and not has_meta_controller), 'meta controller must be made available during discovery phase'

        behavioral_cloning = force_behavior_cloning or (not has_meta_controller and not return_raw_action_dist)

        # by default, if meta controller is passed in, transformer is no grad

        lower_transformer_context = nullcontext if not has_meta_controller else torch.no_grad
        meta_controller_context = nullcontext if has_meta_controller else torch.no_grad
        upper_transformer_context = nullcontext if (not has_meta_controller or discovery_phase) else torch.no_grad

        # handle cache

        lower_transformer_hiddens, meta_hiddens, upper_transformer_hiddens = cache.prev_hiddens if exists(cache) else ((None,) * 3)

        cache_steps = cache.cache_steps if exists(cache) else 0

        seq_len = state.shape[1]
        is_single_token = seq_len == 1

        if (behavioral_cloning or discovery_phase) and is_single_token:
            assert exists(cache) or return_cache, 'behavioral cloning or discovery phase must be used with sequences of length > 1 if not doing inference (caching)'

        calc_auto_regressive_loss = (behavioral_cloning or discovery_phase) and not is_single_token

        if calc_auto_regressive_loss:
            # during behavior cloning and discovery phase, the network is predicting / reconstructing the next token

            assert exists(actions), f'`actions` cannot be empty when doing discovery or behavioral cloning'

            state, target_state = state, state[:, 1:]

            if self.state_loss_detach_target_state:
                target_state = target_state.detach().clone()

            # actions

            target_actions = actions

            if seq_len == actions.shape[1]:
                actions = actions[:, :-1]

            # masking

            state_loss_mask = None
            action_loss_mask = None

            if exists(episode_lens):
                loss_mask = lens_to_mask(episode_lens, state.shape[1])

                state_loss_mask = loss_mask[:, :-1]
                action_loss_mask = loss_mask

        # transformer lower body

        with lower_transformer_context():

            state_embed = self.state_embed(state)

            # handle no past action for first timestep

            action_embed = 0.

            if exists(actions) and exists(self.action_embed):
                action_embed = self.action_embed(actions)

            if is_tensor(action_embed) and action_embed.shape[1] == (seq_len - 1):
                action_embed = pad_left_at_dim(action_embed, 1, dim = 1)

            embed = state_embed + action_embed

            if return_embed:
                return embed

            residual_stream, next_lower_hiddens = self.lower_body(
                embed,
                condition = condition,
                cache = lower_transformer_hiddens,
                return_hiddens = True
            )

        # meta controller acts on residual stream here

        with meta_controller_context():

            if exists(meta_controller) and not behavioral_cloning:
                meta_cache = None if discovery_phase else meta_hiddens
                control_signal, next_meta_hiddens = meta_controller(
                    residual_stream,
                    cache = meta_cache,
                    discovery_phase = discovery_phase,
                    temperature = meta_controller_temperature,
                    episode_lens = episode_lens,
                    ablate_switch_beta = ablate_switch_beta,
                    switch_beta_frequency = switch_beta_frequency,
                    ablate_offset = cache_steps,
                    hard_switch = hard_switch
                )
            else:
                control_signal, next_meta_hiddens = self.zero, None

            if control_signal_multiplier != 1.0:
                control_signal = control_signal * control_signal_multiplier

            modified_residual_stream = residual_stream + control_signal

        # modified residual stream sent back to transformer upper body

        with upper_transformer_context():

            attended, next_upper_hiddens = self.upper_body(
                modified_residual_stream,
                condition = condition,
                cache = upper_transformer_hiddens,
                return_hiddens = True
            )

            # head readout

            dist_params = self.action_readout(attended)

        # if behavior or discovery, calculate state prediction

        if behavioral_cloning or discovery_phase:
            state_dist_params = self.state_readout(attended[:, :-1])

        # caching related

        next_cache_steps = cache_steps + seq_len
        return_cache_value = TransformerOutput(residual_stream, Hiddens(next_lower_hiddens, next_meta_hiddens, next_upper_hiddens), next_cache_steps)

        # maybe early return action and state dist with cache, for char lm
        # cleanup all the returns later

        if return_state_action_cache:
            return (dist_params, state_dist_params), return_cache_value

        # maybe return behavior cloning loss

        if calc_auto_regressive_loss and behavioral_cloning:

            # state

            state_clone_loss = self.state_readout.calculate_loss(state_dist_params, target_state, mask = state_loss_mask)

            # action

            action_clone_loss = self.action_readout.calculate_loss(dist_params, target_actions, mask = action_loss_mask)

            # loss normalization

            if self.normalize_state_action_losses:
                state_clone_loss = self.bc_state_loss_normalizer(state_clone_loss, update_ema = update_loss_ema)
                action_clone_loss = self.bc_action_loss_normalizer(action_clone_loss, update_ema = update_loss_ema)

            losses = BehavioralCloningLosses(state_clone_loss, action_clone_loss)

            if return_action_logits:
                if return_residual_stream:
                    if not return_meta_controller_output:
                        return losses, dist_params, residual_stream
                    return losses, dist_params, next_meta_hiddens, residual_stream
                if not return_meta_controller_output:
                    return losses, dist_params
                return losses, dist_params, next_meta_hiddens
            if return_residual_stream:
                if not return_meta_controller_output:
                    return losses, residual_stream

                return losses, next_meta_hiddens, residual_stream

            if not return_meta_controller_output:
                return losses

            return losses, next_meta_hiddens

        elif calc_auto_regressive_loss and discovery_phase:

            # state

            state_clone_loss = self.state_readout.calculate_loss(state_dist_params, target_state, mask = state_loss_mask)

            # action

            action_recon_loss = self.action_readout.calculate_loss(dist_params, target_actions, mask = action_loss_mask)

            # loss normalization

            if self.normalize_state_action_losses:
                state_clone_loss = self.discovery_state_loss_normalizer(state_clone_loss, update_ema = update_loss_ema)
                action_recon_loss = self.discovery_action_loss_normalizer(action_recon_loss, update_ema = update_loss_ema)

            losses = DiscoveryLosses(state_clone_loss, action_recon_loss, next_meta_hiddens.kl_loss, next_meta_hiddens.ratio_loss)

            if not return_meta_controller_output:
                return losses

            return losses, next_meta_hiddens

        # returning

        return_one = not (return_latents or return_cache)

        if return_one:
            return dist_params

        return dist_params, return_cache_value
