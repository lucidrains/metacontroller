from __future__ import annotations
from collections import namedtuple

import torch
from torch import nn, cat, tensor, is_tensor, Tensor, arange
from torch.nn import Module, Linear, Parameter, RMSNorm, LayerNorm
from torch.nn.functional import cosine_similarity, pad

import einx
from einx import multiply
from einops import rearrange, repeat, reduce

from x_transformers import Decoder

from discrete_continuous_embed_readout import EmbedAndReadout, Readout
from assoc_scan import AssocScan

from torch_einops_utils import pad_left_at_dim, masked_mean, align_dims_left, lens_to_mask
from torch_einops_utils.device import move_inputs_to_module_device
from torch_einops_utils.save_load import save_load

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

def cosine_distance(x, y):
    return (1. - cosine_similarity(x, y, dim = -1)) * 0.5

# losses

def calculate_ratio_loss(
    switch_probs: Tensor,
    boundary_mask: Tensor,
    target_avg_token_length: float,
    mask: Tensor | None = None
):
    N = target_avg_token_length

    F_soft = (switch_probs - 0.5).sigmoid()
    F_mask = straight_through(F_soft, boundary_mask.float())
    G = switch_probs

    ratio_loss_unreduced = N / (N - 1) * ((N - 1) * F_mask * G + (1. - F_mask) * (1. - G))
    return masked_mean(ratio_loss_unreduced, mask)

# named tuples

MetaControllerOutput = namedtuple('MetaControllerOutput', (
    'prev_hiddens',
    'input_residual_stream',
    'action_dist',
    'actions',
    'switch_beta',
    'ratio_loss'
))

GRPOOutput = namedtuple('GRPOOutput', (
    'state',
    'action',
    'log_prob',
    'switch_beta'
))

TransformerWithMetacontrollerCache = namedtuple('TransformerWithMetacontrollerCache', (
    'lower_body',
    'emitter_decoder',
    'upper_body',
    'prev_key',
    'prev_pred_residual_stream',
    'prev_gated_action',
    'cache_steps'
))

# grpo helpers

def z_score(t, dim = None, eps = 1e-8):
    kwargs = dict(dim = dim, keepdim = True) if exists(dim) else dict()
    return (t - t.mean(**kwargs)) / (t.std(**kwargs) + eps)

def extract_grpo_data(model, meta_output):
    state = meta_output.input_residual_stream
    action = meta_output.actions
    switch_beta = meta_output.switch_beta
    log_prob = model.log_prob(meta_output.action_dist, action)
    return GRPOOutput(state, action, log_prob, switch_beta)

@move_inputs_to_module_device
def policy_loss(
    model,
    state,
    old_log_probs,
    actions,
    advantages,
    mask = None,
    episode_lens = None,
    eps_clip: float | tuple[float, float] = 0.2,
):
    action_dist = model.get_action_dist_for_internal_rl(state)
    new_log_probs = model.log_prob(action_dist, actions)

    ratio = (new_log_probs - old_log_probs).exp()
    ratio, advantages = align_dims_left((ratio, advantages))

    surr1 = ratio * advantages

    if isinstance(eps_clip, (float, int)):
        eps_clip = (eps_clip, eps_clip)

    eps_lower, eps_upper = eps_clip
    surr2 = ratio.clamp(1 - eps_lower, 1 + eps_upper) * advantages

    losses = -torch.min(surr1, surr2)

    if exists(mask) and mask.ndim == 3:
        mask = rearrange(mask, 'b n 1 -> b n')

    if exists(episode_lens):
        seq_len = mask.shape[1] if exists(mask) else losses.shape[1]
        episode_mask = torch.arange(seq_len, device = losses.device) < episode_lens.unsqueeze(-1)
        mask = mask & episode_mask if exists(mask) else episode_mask

    losses = reduce(losses, 'b n d -> b n', 'sum')
    return masked_mean(losses, mask)

# main class

@save_load()
class TransformerWithMetacontroller(Module):
    def __init__(
        self,
        dim,
        *,
        state_embed_readout: dict,
        action_embed_readout: dict,
        embed_past_actions = True,
        dim_latent = 128,
        emitter_decoder: dict = dict(),
        lower_body: dict = dict(),
        upper_body: dict = dict(),
        dim_queries_keys = 256,
        target_avg_token_length = 8.,
        residual_stream_dropout = 0.,
        residual_stream_drop_prob = 0.,
        pred_loss_to_switch_weight = 0.,
        assoc_scan_kwargs: dict = dict()
    ):
        super().__init__()
        self.dim_model = dim
        self.dim_latent = dim_latent

        # embeddings

        self.state_embed, self.state_readout = EmbedAndReadout(dim, **state_embed_readout)
        action_embed, self.action_readout = EmbedAndReadout(dim, **action_embed_readout)

        self.action_embed = action_embed if embed_past_actions else None

        # lower body

        self.lower_body = Decoder(
            dim = dim,
            use_rmsnorm = True,
            polar_pos_emb = True,
            **lower_body
        )

        # emitter decoder

        self.emitter_decoder = Decoder(
            dim = dim,
            use_rmsnorm = True,
            polar_pos_emb = True,
            **emitter_decoder
        )

        # qk switching unit

        self.to_queries_keys = nn.Sequential(
            RMSNorm(dim),
            Linear(dim, dim_queries_keys * 2, bias = False)
        )

        self.to_pred_residual_stream = nn.Sequential(
            RMSNorm(dim),
            Linear(dim, dim, bias = False)
        )

        self.start_key_token = Parameter(torch.randn(dim_queries_keys) * 1e-2)
        self.target_avg_token_length = target_avg_token_length

        # latent readout

        self.latent_readout = Readout(dim, num_continuous = dim_latent)

        # associative scan

        self.assoc_scan = AssocScan(**assoc_scan_kwargs)

        # latent to control signal

        self.to_control_signal = Linear(dim_latent, dim, bias = False)

        self.residual_stream_dropout = nn.Dropout(residual_stream_dropout)
        self.residual_stream_drop_prob = residual_stream_drop_prob
        self.pred_loss_to_switch_weight = pred_loss_to_switch_weight
        assert self.pred_loss_to_switch_weight < 1., 'pred_loss_to_switch_weight must be strictly less than 1. so predictive loss is never wholly responsible for switching'

        # upper body

        self.upper_body = Decoder(
            dim = dim,
            use_rmsnorm = True,
            polar_pos_emb = True,
            **upper_body
        )

        self.modified_residual_stream_norm = LayerNorm(dim, bias = False)

        self.register_buffer('zero', tensor(0.), persistent = False)

    # grpo interface

    @property
    def replay_buffer_field_dict(self):
        return dict(
            states = ('float', self.dim_model),
            log_probs = 'float',
            switch_betas = 'float',
            latent_actions = ('float', self.dim_latent)
        )

    def get_action_dist_for_internal_rl(self, residual_stream):
        emitter_out = self.emitter_decoder(residual_stream)
        return self.latent_readout(emitter_out)

    def log_prob(self, dist_params, actions):
        return self.latent_readout.log_prob(dist_params, actions)

    # forward

    def forward(
        self,
        state,
        actions: Tensor | None = None,
        episode_lens: Tensor | None = None,
        return_loss = True,
        cache: TransformerWithMetacontrollerCache | None = None,
        return_cache = False
    ):
        batch, seq_len, device = *state.shape[:2], state.device

        mask = lens_to_mask(episode_lens, seq_len) if exists(episode_lens) else None
        target_state = target_actions = None

        if return_loss:
            target_state = state[:, 1:]

            assert exists(actions), '`actions` cannot be empty when return_loss is True'

            target_actions = actions

            if seq_len == actions.shape[1]:
                actions = actions[:, :-1]

        lower_body_cache, emitter_decoder_cache, upper_body_cache, prev_key, prev_pred_residual_stream, prev_gated_action, cache_steps = default(cache, (None,) * 6 + (0,))

        # embed and process through lower body

        state_embed = self.state_embed(state)
        action_embed_out = 0.

        if exists(actions) and exists(self.action_embed):
            action_embed_out = self.action_embed(actions)

        if is_tensor(action_embed_out) and action_embed_out.shape[1] == (seq_len - 1):
            action_embed_out = pad_left_at_dim(action_embed_out, 1, dim = 1)

        embed = state_embed + action_embed_out

        residual_stream, next_lower_body_cache = self.lower_body(embed, cache = lower_body_cache, return_hiddens = True)

        # derive switching probabilities from qk cosine similarity

        queries, keys = self.to_queries_keys(residual_stream).chunk(2, dim = -1)

        if not exists(prev_key):
            start_keys = repeat(self.start_key_token, 'd -> b 1 d', b = batch)
            keys_with_prev = cat((start_keys, keys), dim = 1)
        else:
            keys_with_prev = cat((prev_key, keys), dim = 1)

        next_prev_key = keys_with_prev[:, -1:]

        switch_probs = cosine_distance(queries, keys_with_prev[:, :-1])

        if cache_steps == 0:
            switch_probs = torch.where(
                arange(seq_len, device = device) == 0,
                1.,
                switch_probs
            )

        # latent ar loss + optional predictive switch blending

        latent_ar_loss = None
        pred_residual_stream = None

        if return_loss or self.pred_loss_to_switch_weight > 0.:
            pred_residual_stream = self.to_pred_residual_stream(residual_stream)

        if return_loss:
            per_token_latent_ar_loss = cosine_distance(pred_residual_stream[:, :-1], residual_stream[:, 1:].detach())

            # handle mask for loss (shift by 1 like targets)
            loss_mask = mask[:, 1:] if exists(mask) else None
            latent_ar_loss = masked_mean(per_token_latent_ar_loss, loss_mask)

        if self.pred_loss_to_switch_weight > 0.:
            if not exists(prev_pred_residual_stream):
                _per_token_loss = cosine_distance(pred_residual_stream[:, :-1], residual_stream[:, 1:].detach())
                predictive_difficulty = pad(_per_token_loss, (1, 0), value = 1.)
            else:
                pred_residual_stream_with_prev = cat((prev_pred_residual_stream, pred_residual_stream[:, :-1]), dim = 1)
                predictive_difficulty = cosine_distance(pred_residual_stream_with_prev, residual_stream.detach())

            switch_probs = switch_probs.lerp(predictive_difficulty, self.pred_loss_to_switch_weight)

        next_prev_pred_residual_stream = pred_residual_stream[:, -1:] if exists(pred_residual_stream) else None

        boundary_mask = switch_probs > 0.5
        switch_probs_hard = straight_through(switch_probs, boundary_mask.float())

        # obtain latents from emitter + readout

        emitter_out, next_emitter_decoder_cache = self.emitter_decoder(residual_stream, cache = emitter_decoder_cache, return_hiddens = True)
        action_dist = self.latent_readout(emitter_out)
        sampled_latent_action = self.latent_readout.sample(action_dist, differentiable = True)

        # gated action via associative scan

        gated_sampled_latent_action = multiply('b n d, b n', sampled_latent_action, switch_probs_hard)
        forget_gate = 1. - switch_probs_hard

        scanned_latent_action = self.assoc_scan(forget_gate, gated_sampled_latent_action, prev = prev_gated_action)
        next_prev_gated_action = scanned_latent_action[:, -1:]

        # confidence scaling for gradient routing

        confidence = torch.where(boundary_mask, switch_probs, 1. - switch_probs)
        confidence_scale = straight_through(confidence, 1.)

        # latent to control signal

        control_signal = self.to_control_signal(scanned_latent_action)
        control_signal = multiply('b n d, b n', control_signal, confidence_scale)

        dropped_residual_stream = self.residual_stream_dropout(residual_stream)

        if self.training and self.residual_stream_drop_prob > 0.:
            drop_mask = torch.rand(batch, device = device) > self.residual_stream_drop_prob
            dropped_residual_stream = einx.where('b, b n d, -> b n d', drop_mask, dropped_residual_stream, self.zero)

        modified_residual_stream = self.modified_residual_stream_norm(dropped_residual_stream + control_signal)

        # upper body

        attended, next_upper_body_cache = self.upper_body(modified_residual_stream, cache = upper_body_cache, return_hiddens = True)
        dist_params = self.action_readout(attended)

        state_dist_params = None
        if return_loss:
            state_dist_params = self.state_readout(attended[:, :-1])

        # cache

        next_cache = TransformerWithMetacontrollerCache(
            lower_body = next_lower_body_cache,
            emitter_decoder = next_emitter_decoder_cache,
            upper_body = next_upper_body_cache,
            prev_key = next_prev_key,
            prev_pred_residual_stream = next_prev_pred_residual_stream,
            prev_gated_action = next_prev_gated_action,
            cache_steps = cache_steps + seq_len
        )

        # losses

        bc_state_loss = bc_action_loss = ratio_loss = None

        if return_loss:
            loss_mask = mask[:, 1:] if exists(mask) else None

            bc_state_loss = self.state_readout.calculate_loss(state_dist_params, target_state, mask = loss_mask)
            bc_action_loss = self.action_readout.calculate_loss(dist_params, target_actions, mask = loss_mask)
            ratio_loss = calculate_ratio_loss(switch_probs, boundary_mask, self.target_avg_token_length, mask = mask)

        meta_output = MetaControllerOutput(
            prev_hiddens = None,
            input_residual_stream = residual_stream,
            action_dist = action_dist,
            actions = sampled_latent_action,
            switch_beta = switch_probs,
            ratio_loss = ratio_loss
        )

        if not return_loss:
            ret = (dist_params, meta_output)
        else:
            ret = (
                (bc_state_loss, bc_action_loss, ratio_loss, latent_ar_loss),
                meta_output
            )

        if return_cache:
            return (*ret, next_cache)

        return ret

TransformerWithMetacontroller.policy_loss = policy_loss
