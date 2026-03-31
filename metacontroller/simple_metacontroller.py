from __future__ import annotations
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn, cat, tensor, is_tensor, Tensor, arange
from torch.nn import Module, Linear, Parameter, RMSNorm, LayerNorm
from torch.nn.functional import cosine_similarity, pad, logsigmoid

import einx
from einx import multiply
from einops import rearrange, repeat, reduce

from x_transformers import Decoder, Encoder

from discrete_continuous_embed_readout import EmbedAndReadout, Readout
from assoc_scan import AssocScan
from vector_quantize_pytorch import BinaryMapper

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

class MetaControllerLosses(NamedTuple):
    bc_state_loss: Tensor | None
    bc_action_loss: Tensor | None
    ratio_loss: Tensor | None
    latent_loss: Tensor | float
    latent_ar_loss: Tensor | float
    sigreg_loss: Tensor | float
    kl_loss: Tensor | float
    next_switch_pred_loss: Tensor | float
    control_penalty_loss: Tensor | float

class MetaControllerOutput(NamedTuple):
    input_residual_stream: Tensor
    action_dist: Tensor
    actions: Tensor
    switch_beta: Tensor
    ratio_loss: Tensor | None
    kl_loss: Tensor | float
    kl_loss_weight: float | Tensor
    latent_pred_entropy: Tensor | None
    quantile_indices: Tensor | None = None
    sigreg_loss: Tensor | float = 0.
    control_penalty_loss: Tensor | float = 0.

class GRPOOutput(NamedTuple):
    state: Tensor
    action: Tensor
    log_prob: Tensor
    switch_beta: Tensor
    quantile_indices: Tensor | None = None

class TransformerWithMetacontrollerCache(NamedTuple):
    lower_body: Tensor | None
    emitter_decoder: Tensor | None
    upper_body: Tensor | None
    prev_key: Tensor | None
    prev_gated_action: Tensor | None
    quantile_indices: Tensor | None
    cache_steps: int

# grpo helpers

def z_score(t, dim = None, eps = 1e-8):
    kwargs = dict(dim = dim, keepdim = True) if exists(dim) else dict()
    return (t - t.mean(**kwargs)) / (t.std(**kwargs) + eps)

def extract_grpo_data(model, meta_output):
    state = meta_output.input_residual_stream
    action = meta_output.actions
    switch_beta = meta_output.switch_beta
    log_prob = model.log_prob(meta_output.action_dist, action)
    return GRPOOutput(state, action, log_prob, switch_beta, meta_output.quantile_indices)

@move_inputs_to_module_device
def policy_loss(
    model,
    state,
    old_log_probs,
    actions,
    advantages,
    switch_betas = None,
    mask = None,
    episode_lens = None,
    eps_clip: float | tuple[float, float] = 0.2,
    quantile_indices = None
):
    action_dist = model.get_action_dist_for_internal_rl(state, switch_betas = switch_betas, quantile_indices = quantile_indices)
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
        dim_code_bits = 4,
        emitter_decoder: dict = dict(),
        lower_body: dict = dict(),
        upper_body: dict = dict(),
        temporal_sequence_embedder: dict | None = None,
        dim_sequence_summary_embed: int = 32,
        temporal_sequence_embed_prob: float = 1.,
        switch_entropy_quantiles: tuple[float, ...] | None = None,
        eval_switch_entropy_quantile: float = 0.5,
        switch_entropy_quantile_lr: float = 0.1,
        dim_queries_keys = 256,
        target_avg_token_length = 8.,
        residual_stream_dropout = 0.,
        residual_stream_drop_prob = 0.,
        kl_loss_weight = 1.,
        kl_loss_threshold = 0.,
        freeze_entropy_quantiles = False,
        num_latent_pred_steps = 2,
        future_entropy_weights: tuple[float, ...] | None = None,
        predict_next_switch_embed = False,
        sigreg_lambda = 0.05,
        sigreg_loss_kwargs: dict | None = None,
        assoc_scan_kwargs: dict = dict(),
        control_penalty_floor = 0.5
    ):
        super().__init__()
        self.dim_model = dim
        self.dim_code_bits = dim_code_bits
        self.sigreg_lambda = sigreg_lambda
        self.sigreg_loss_kwargs = default(sigreg_loss_kwargs, dict(num_slices = 256))

        self.kl_loss_weight = kl_loss_weight
        self.control_penalty_floor = control_penalty_floor
        self.freeze_entropy_quantiles = freeze_entropy_quantiles

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

        # temporal sequence embedder

        self.temporal_embedder = None
        if exists(temporal_sequence_embedder):
            self.temporal_embedder = Encoder(
                dim = dim,
                use_rmsnorm = True,
                polar_pos_emb = True,
                **temporal_sequence_embedder
            )

            self.to_sequence_summary_embed = nn.Sequential(
                Linear(dim, dim_sequence_summary_embed),
                Linear(dim_sequence_summary_embed, dim)
            )

        self.temporal_sequence_embed_prob = temporal_sequence_embed_prob

        # entropy quantile switching unit

        self.switch_entropy_quantile_lr = switch_entropy_quantile_lr

        self.switch_entropy_quantiles = None
        self.num_quantiles = 0
        self.eval_switch_entropy_quantile_idx = 0

        if exists(switch_entropy_quantiles):
            if isinstance(switch_entropy_quantiles, (float, int)):
                switch_entropy_quantiles = (switch_entropy_quantiles,)

            self.switch_entropy_quantiles = tuple(switch_entropy_quantiles)
            self.num_quantiles = len(self.switch_entropy_quantiles)

            assert eval_switch_entropy_quantile in self.switch_entropy_quantiles, f'eval_switch_entropy_quantile ({eval_switch_entropy_quantile}) must be one of switch_entropy_quantiles {self.switch_entropy_quantiles}'
            self.eval_switch_entropy_quantile_idx = self.switch_entropy_quantiles.index(eval_switch_entropy_quantile)

            self.register_buffer('running_entropy_quantiles', tensor([10.] * self.num_quantiles))
            self.quantile_embed = nn.Embedding(self.num_quantiles, dim)
        else:
            self.to_queries_keys = nn.Sequential(
                RMSNorm(dim),
                Linear(dim, dim_queries_keys * 2, bias = False)
            )

            self.start_key_token = Parameter(torch.randn(dim_queries_keys) * 1e-2)

        self.predict_next_switch_embed = predict_next_switch_embed
        self.next_switch_embed_pred_head = None
        if predict_next_switch_embed:
            self.next_switch_embed_pred_head = nn.Sequential(
                RMSNorm(dim),
                Linear(dim, dim)
            )

        self.num_latent_pred_steps = num_latent_pred_steps

        if not exists(future_entropy_weights):
            future_entropy_weights = tuple(0.5 ** i for i in range(num_latent_pred_steps))

        assert len(future_entropy_weights) == num_latent_pred_steps, f'expected future_entropy_weights to have length {num_latent_pred_steps}'
        self.register_buffer('future_entropy_weights', tensor(future_entropy_weights, dtype = torch.float32), persistent = False)

        self.latent_readout = Readout(
            dim = dim,
            num_continuous = dim * num_latent_pred_steps,
            continuous_dist_kwargs = dict(log_var_clamp_range = (-5., 3.))
        )

        self.switch_embed = nn.Embedding(2, dim)

        self.target_avg_token_length = target_avg_token_length

        # binary mapper

        self.emitter_to_binary_logits = Linear(dim, dim_code_bits, bias = False)

        self.binary_mapper = BinaryMapper(
            bits = dim_code_bits,
            kl_loss_threshold = kl_loss_threshold
        )
        self.num_codes = self.binary_mapper.num_codes

        # associative scan

        self.assoc_scan = AssocScan(**assoc_scan_kwargs)

        # latent to control signal

        self.to_control_signal = Linear(self.num_codes, dim, bias = False)

        self.residual_stream_dropout = nn.Dropout(residual_stream_dropout)
        self.residual_stream_drop_prob = residual_stream_drop_prob

        # upper body

        self.upper_body = Decoder(
            dim = dim,
            use_rmsnorm = True,
            polar_pos_emb = True,
            **upper_body
        )

        self.modified_residual_stream_norm = LayerNorm(dim, bias = False)

        self.register_buffer('zero', tensor(0.), persistent = False)

    @staticmethod
    def sigreg_loss(
        x,
        mask = None,
        num_slices = 1024,
        domain = (-5, 5),
        num_knots = 17,
        reduction = 'mean'
    ):
        # Randall Balestriero - https://arxiv.org/abs/2511.08544

        dim, device = x.shape[-1], x.device

        if exists(mask):
            x = x[mask]

        if x.numel() == 0:
            return torch.tensor(0., device = device, requires_grad = True)

        # slice sampling

        rand_projs = torch.randn((num_slices, dim), device = device)
        rand_projs = F.normalize(rand_projs, dim = -1)

        # integration points

        t = torch.linspace(*domain, num_knots, device = device)

        # theoretical CF for N(0, 1) and Gauss. window

        exp_f = (-0.5 * t.square()).exp()

        # empirical CF

        x_t = einx.dot('... d, m d -> (...) m', x, rand_projs)

        x_t = multiply('n m, t -> n m t', x_t, t)
        ecf = (1j * x_t).exp().mean(dim = 0)

        # weighted L2 distance

        err = ecf.sub(exp_f).abs().square().mul(exp_f)

        loss = torch.trapz(err, t, dim = -1)

        if reduction == 'mean':
            return loss.mean()

        return loss

    # grpo interface

    @property
    def replay_buffer_field_dict(self):
        return {
            'state': self.dim_model,
            'action': self.dim_code_bits,
            'log_prob': (),
            'switch_beta': ()
        }

    def internal_metacontroller_parameters(self):
        return [
            *self.emitter_decoder.parameters(),
            *self.emitter_to_binary_logits.parameters()
        ]

    def _add_quantile_embed(self, x, quantile_indices):
        """Add quantile conditioning embedding to input tensor."""
        if not exists(self.switch_entropy_quantiles) or not exists(quantile_indices):
            return x

        q_embeds = self.quantile_embed(quantile_indices)

        if q_embeds.ndim == 2 and x.ndim == 3:
            q_embeds = rearrange(q_embeds, 'b d -> b 1 d')

        return x + q_embeds

    def get_action_dist_for_internal_rl(self, residual_stream, switch_betas = None, quantile_indices = None, return_kl_loss = False):
        emitter_input = residual_stream

        if exists(switch_betas):
            switch_indices = (switch_betas > 0.5).long()
            emitter_input = emitter_input + self.switch_embed(switch_indices)

        emitter_input = self._add_quantile_embed(emitter_input, quantile_indices)

        emitter_out = self.emitter_decoder(emitter_input)
        binary_logits = self.emitter_to_binary_logits(emitter_out)

        if not return_kl_loss:
            return binary_logits

        return binary_logits, self.calculate_kl_loss(binary_logits)

    def calculate_kl_loss(self, binary_logits):
        _, kl_loss = self.binary_mapper(binary_logits, reduce_aux_kl_loss = False)
        return kl_loss

    def log_prob(self, dist_params, actions):
        log_probs = torch.stack((
            logsigmoid(dist_params),
            logsigmoid(-dist_params)
        ), dim = -1)

        indices = actions.argmax(dim = -1)
        codes = self.binary_mapper.codes[indices].long()

        codes = rearrange(codes, '... -> ... 1')
        action_log_probs = log_probs.gather(-1, codes)
        action_log_probs = rearrange(action_log_probs, '... 1 -> ...')

        return action_log_probs

    # forward

    def forward(
        self,
        state,
        actions: Tensor | None = None,
        episode_lens: Tensor | None = None,
        return_loss = True,
        cache: TransformerWithMetacontrollerCache | None = None,
        return_cache = False,
        use_temporal_sequence_embed: bool | None = None,
        update_entropy_quantile: bool | None = None
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

        lower_body_cache, emitter_decoder_cache, upper_body_cache, prev_key, prev_gated_action, cached_quantile_indices, cache_steps = default(cache, (None,) * 6 + (0,))

        # embed and process through lower body

        state_embed = self.state_embed(state)
        action_embed_out = 0.

        if exists(actions) and exists(self.action_embed):
            action_embed_out = self.action_embed(actions)

        if is_tensor(action_embed_out) and action_embed_out.shape[1] == (seq_len - 1):
            action_embed_out = pad_left_at_dim(action_embed_out, 1, dim = 1)

        embed = state_embed + action_embed_out

        residual_stream, next_lower_body_cache = self.lower_body(embed, cache = lower_body_cache, return_hiddens = True)

        dist_params = self.latent_readout(residual_stream)

        latent_pred_entropy = self.latent_readout.entropy(dist_params)
        latent_pred_entropy = rearrange(latent_pred_entropy, 'b n (s d) -> b n s d', s = self.num_latent_pred_steps)
        latent_pred_entropy = latent_pred_entropy.sum(dim = -1)

        latent_pred_entropy = (latent_pred_entropy * self.future_entropy_weights).sum(dim = -1)

        sigreg_loss = self.zero

        if return_loss:
            targets = residual_stream[:, 1:].detach()
            num_targets = targets.shape[1]

            latent_ar_loss = self.zero

            if num_targets >= self.num_latent_pred_steps:
                unfolded_targets = targets.unfold(1, self.num_latent_pred_steps, 1)
                unfolded_targets = rearrange(unfolded_targets, 'b n d s -> b n (s d)')

                loss_inputs = residual_stream[:, :num_targets - self.num_latent_pred_steps + 1]

                loss_mask = None
                if exists(mask):
                    unfolded_mask = mask[:, 1:].unfold(1, self.num_latent_pred_steps, 1)
                    loss_mask = unfolded_mask.all(dim = -1)

                latent_ar_loss = self.latent_readout(
                    loss_inputs,
                    targets = unfolded_targets,
                    return_loss = True,
                    loss_mask = loss_mask
                )

                sigreg_loss = self.sigreg_loss(residual_stream, mask = mask, **self.sigreg_loss_kwargs)
                latent_loss = latent_ar_loss + sigreg_loss * self.sigreg_lambda

        # derive switching probabilities from qk cosine similarity or entropy quantile

        quantile_indices = None

        if exists(self.switch_entropy_quantiles):
            if exists(cached_quantile_indices):
                quantile_indices = cached_quantile_indices
            elif self.training and not self.freeze_entropy_quantiles:
                quantile_indices = torch.randint(0, self.num_quantiles, (batch,), device = device)
            else:
                quantile_indices = torch.full((batch,), self.eval_switch_entropy_quantile_idx, device = device, dtype = torch.long)

            entropy_thresholds = self.running_entropy_quantiles[quantile_indices]
            entropy_thresholds = rearrange(entropy_thresholds, 'b -> b 1')

            switch_probs = (latent_pred_entropy > entropy_thresholds).float()

            should_update = default(update_entropy_quantile, self.training and not self.freeze_entropy_quantiles)

            if should_update:
                with torch.no_grad():
                    if exists(mask):
                        flat_entropy = latent_pred_entropy[mask].detach().reshape(-1)
                    else:
                        flat_entropy = latent_pred_entropy.detach().reshape(-1)

                    if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
                        gathered_entropy = [torch.zeros_like(flat_entropy) for _ in range(torch.distributed.get_world_size())]
                        torch.distributed.all_gather(gathered_entropy, flat_entropy)
                        flat_entropy = torch.cat(gathered_entropy)

                    if flat_entropy.numel() > 0:
                        q_tensor = tensor(self.switch_entropy_quantiles, device = flat_entropy.device, dtype = flat_entropy.dtype)
                        batch_quantiles = torch.quantile(flat_entropy, q_tensor)
                        self.running_entropy_quantiles.lerp_(batch_quantiles, self.switch_entropy_quantile_lr)

            next_prev_key = None
        else:
            qk_input = residual_stream

            should_use_temporal_embed = default(use_temporal_sequence_embed, return_loss)

            if should_use_temporal_embed and exists(self.temporal_embedder):
                temporal_encoded = self.temporal_embedder(residual_stream, mask = mask)
                temporal_pooled = masked_mean(temporal_encoded, mask, dim = 1)
                temporal_bottlenecked = self.to_sequence_summary_embed(temporal_pooled)

                if self.training and self.temporal_sequence_embed_prob < 1.:
                    batch_prob_mask = torch.rand(batch, device = device) < self.temporal_sequence_embed_prob
                    temporal_bottlenecked = torch.where(
                        rearrange(batch_prob_mask, 'b -> b 1'),
                        temporal_bottlenecked,
                        self.zero
                    )

                temporal_repeated = repeat(temporal_bottlenecked, 'b d -> b n d', n = seq_len)
                qk_input = qk_input + temporal_repeated

            queries, keys = self.to_queries_keys(qk_input).chunk(2, dim = -1)

            if not exists(prev_key):
                start_keys = repeat(self.start_key_token, 'd -> b 1 d', b = batch)
                keys_with_prev = cat((start_keys, keys), dim = 1)
            else:
                keys_with_prev = cat((prev_key, keys), dim = 1)

            next_prev_key = keys_with_prev[:, -1:]

            switch_probs = cosine_distance(queries, keys_with_prev[:, :-1])

        # always switch on the first token

        if cache_steps == 0:
            switch_probs = pad_left_at_dim(switch_probs[:, 1:], 1, dim = 1)

        boundary_mask = switch_probs > 0.5
        switch_probs_hard = straight_through(switch_probs, boundary_mask.float())

        # obtain latents from emitter + readout

        switch_indices = boundary_mask.long()
        switch_embeds = self.switch_embed(switch_indices)
        emitter_input = residual_stream + switch_embeds

        emitter_input = self._add_quantile_embed(emitter_input, quantile_indices)

        emitter_out, next_emitter_decoder_cache = self.emitter_decoder(emitter_input, cache = emitter_decoder_cache, return_hiddens = True)
        binary_logits = self.emitter_to_binary_logits(emitter_out)

        sampled_latent_action, kl_loss = self.binary_mapper(binary_logits, reduce_aux_kl_loss = False)

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
            prev_gated_action = next_prev_gated_action,
            quantile_indices = quantile_indices,
            cache_steps = cache_steps + seq_len
        )

        # losses

        next_switch_pred_loss = self.zero
        bc_state_loss = bc_action_loss = ratio_loss = None

        if return_loss:
            loss_mask = mask[:, 1:] if exists(mask) else None

            bc_state_loss = self.state_readout.calculate_loss(state_dist_params, target_state, mask = loss_mask)
            bc_action_loss = self.action_readout.calculate_loss(dist_params, target_actions, mask = loss_mask)
            ratio_loss = calculate_ratio_loss(switch_probs, boundary_mask, self.target_avg_token_length, mask = mask)

            if self.predict_next_switch_embed:
                batch, seq_len, _ = emitter_out.shape

                flat_mask = rearrange(boundary_mask, 'b n -> (b n)')

                flat_emitter_out = rearrange(emitter_out, 'b n d -> (b n) d')
                flat_residual = rearrange(residual_stream, 'b n d -> (b n) d')

                switch_preds = self.next_switch_embed_pred_head(flat_emitter_out[flat_mask])
                switch_targets = flat_residual[flat_mask]

                batch_indices = torch.arange(batch, device = device)
                batch_indices = repeat(batch_indices, 'b -> (b n)', n = seq_len)

                switch_batch_indices = batch_indices[flat_mask]

                valid_next_mask = switch_batch_indices[:-1] == switch_batch_indices[1:]

                switch_preds = switch_preds[:-1][valid_next_mask]
                switch_targets = switch_targets[1:][valid_next_mask]

                next_switch_pred_loss = self.zero
                if switch_preds.numel() > 0:
                    next_switch_pred_loss = F.l1_loss(switch_preds, switch_targets.detach())

            if kl_loss.ndim == 3:
                kl_loss = reduce(kl_loss, 'b n d -> b n', 'sum')

            kl_loss = masked_mean(kl_loss, mask)

            control_penalty_loss = self.zero
            valid_mask = boundary_mask & loss_mask if exists(loss_mask) else boundary_mask

            if valid_mask.any():
                res_norm = dropped_residual_stream[valid_mask].norm(dim = -1)
                ctrl_norm = control_signal[valid_mask].norm(dim = -1)
                ratio = res_norm / ctrl_norm.clamp(min = 1e-5)
                control_penalty_loss = F.relu(ratio - self.control_penalty_floor).mean()

        else:
            kl_loss = self.zero
            control_penalty_loss = self.zero

        meta_output = MetaControllerOutput(
            input_residual_stream = residual_stream,
            action_dist = binary_logits,
            actions = sampled_latent_action,
            switch_beta = switch_probs_hard,
            ratio_loss = ratio_loss,
            kl_loss = kl_loss,
            kl_loss_weight = self.kl_loss_weight,
            latent_pred_entropy = latent_pred_entropy,
            quantile_indices = quantile_indices,
            sigreg_loss = sigreg_loss,
            control_penalty_loss = control_penalty_loss
        )

        if not return_loss:
            ret = (dist_params, meta_output)
        else:
            losses = MetaControllerLosses(
                bc_state_loss = bc_state_loss,
                bc_action_loss = bc_action_loss,
                ratio_loss = ratio_loss,
                latent_loss = latent_loss,
                latent_ar_loss = latent_ar_loss,
                sigreg_loss = sigreg_loss,
                kl_loss = kl_loss,
                next_switch_pred_loss = next_switch_pred_loss,
                control_penalty_loss = control_penalty_loss
            )

            ret = (
                losses,
                meta_output
            )

        if return_cache:
            return (*ret, next_cache)

        return ret

TransformerWithMetacontroller.policy_loss = policy_loss
