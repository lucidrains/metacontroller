from __future__ import annotations
from typing import Literal
from contextlib import nullcontext

from functools import partial
from collections import namedtuple
from loguru import logger

import torch
from torch import nn, cat, stack, tensor, is_tensor, Tensor
from torch.nn import Module, Linear, Identity, GRU, Parameter

import torch.nn.functional as F

from metacontroller.sequential_action_selection import perform_sequential_action_selection, SequentialActionSelectionOutput

# einops

import einx
from einops import einsum, rearrange, repeat, reduce
from einops.layers.torch import Rearrange

# external modules

from x_transformers import Decoder
from x_transformers.x_transformers import PolarEmbedding, Identity

# sequential selection moved to separate module

from x_mlps_pytorch import Feedforwards, create_mlp
from x_evolution import EvoStrategy

from discrete_continuous_embed_readout import Embed, Readout, EmbedAndReadout

from assoc_scan import AssocScan

from torch_einops_utils import maybe, pad_left_at_dim, pad_right_at_dim, pad_at_dim, lens_to_mask, and_masks, masked_mean, align_dims_left, exclusive_cumsum, pack_with_inverse, straight_through, z_score
from torch_einops_utils.shape import shape, size
from torch_einops_utils.device import module_device, move_inputs_to_module_device
from torch_einops_utils.save_load import save_load

# constants

LinearNoBias = partial(Linear, bias = False)

GRU = partial(GRU, batch_first = True)

# helper functions

def exists(val):
    return val is not None

def compact(iterable):
    return tuple(filter(exists, iterable))

def identity(t):
    return t

def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None

# tensor helpers

def log(t, eps = 1e-20):
    return t.clamp_min(eps).log()

def gumbel_noise(t):
    return -log(-log(torch.rand_like(t)))

def binary_entropy(p):
    return -((p * log(p)) + (1. - p) * log(1. - p))

def categorical_entropy(probs):
    return -(probs * log(probs)).sum(dim = -1)

def l2norm(t, dim = -1):
    return F.normalize(t, p = 2, dim = dim)

def categorical_entropy_loss(probs, mask = None, sample_entropy_weight = 0.1):
    active = probs[mask] if (exists(mask) and mask.any()) else rearrange(probs, '... k -> (...) k')
    mean_probs = reduce(active, 'n k -> k', 'mean')
    return -categorical_entropy(mean_probs) + sample_entropy_weight * categorical_entropy(active).mean()

def normalized_categorical_mi(probs):
    num_classes = probs.shape[-1]
    if num_classes <= 1 or probs.shape[-2] <= 1:
        return probs.sum() * 0.
    marginal = probs.mean(dim = -2)
    mi = categorical_entropy(marginal) - categorical_entropy(probs).mean(dim = -1)
    return mi.mean() / float(torch.tensor(num_classes, dtype = probs.dtype, device = probs.device).log())

def sample_straight_through(
    probs,
    hard_prob = 0.5,
    training = True,
    gumbel_sample = None
):
    has_multi_samples = probs.ndim >= 4
    gumbel_sample = default(gumbel_sample, has_multi_samples)

    if training and gumbel_sample:
        gumbel = gumbel_noise(probs)

        if has_multi_samples:
            gumbel[0] = 0.  # first candidate is greedy

        indices = (log(probs) + gumbel).argmax(dim = -1)
    else:
        indices = probs.argmax(dim = -1)

    hard_one_hot = F.one_hot(indices, size(probs, '... [k]')).float()

    if not training:
        return hard_one_hot

    st_probs = straight_through(probs, hard_one_hot)

    if hard_prob >= 1.:
        return st_probs

    mask_shape = probs[:1, ..., :1] if has_multi_samples else probs[..., :1]
    is_hard = torch.zeros_like(mask_shape).bernoulli_(hard_prob).bool()
    return torch.where(is_hard, st_probs, probs)

# next latent prediction - residual dynamics network

class ResidualDynamics(Module):
    def __init__(
        self,
        dim,
        cond_dim = None,
        hidden_dim = None,
        num_layers = 3
    ):
        super().__init__()
        hidden_dim = default(hidden_dim, dim)
        cond_dim = default(cond_dim, dim)

        self.net = nn.Sequential(
            nn.LayerNorm(dim + cond_dim),
            create_mlp(
                dim = hidden_dim,
                depth = num_layers,
                dim_in = dim + cond_dim,
                dim_out = dim
            )
        )

        last_linear = self.net[-1].layers[-1]
        nn.init.zeros_(last_linear.weight)
        nn.init.zeros_(last_linear.bias)

    def forward(
        self,
        next_token_embeds, # (b n d)
        curr_latent        # (b n d)
    ):
        dynamics_input = cat((curr_latent, next_token_embeds), dim = -1)
        return curr_latent + self.net(dynamics_input)

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
class CausalSequenceEmbedder(Module):
    def __init__(
        self,
        dim,
        pool = True,
        **kwargs
    ):
        super().__init__()
        self.pool = pool
        self.decoder = Decoder(dim = dim, **kwargs)

    def forward(
        self,
        x,
        mask = None,
        **kwargs
    ):
        encoded = self.decoder(x, mask = mask)

        if not self.pool:
            return encoded

        mean_pooled = masked_mean(encoded, mask, dim = 1)
        return repeat(mean_pooled, 'b d -> b n d', n = size(x, 'b [n] d'))

class LatentReadout(Module):
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def sample(self, dist: Tensor, temperature: float = 1., differentiable: bool = True) -> Tensor:
        raise NotImplementedError

    def calculate_kl_loss(self, dist: Tensor, mask: Tensor | None = None) -> Tensor:
        raise NotImplementedError

    def log_prob(self, dist: Tensor, sampled_latent_action: Tensor) -> Tensor:
        raise NotImplementedError

    def clone_for_proposer(self, dim_proposer: int) -> LatentReadout:
        raise NotImplementedError

@save_load()
class DiscreteLatentReadout(LatentReadout):
    dim_log_prob = 1

    def __init__(
        self,
        dim,
        dim_latent,
        num_discrete = 4,
        hard_prob = 0.5,
        temperature = 0.2,
        sample_entropy_weight = 0.1,
        shared_prototypes: Parameter | None = None
    ):
        super().__init__()
        self.dim_latent = dim_latent
        self.num_discrete = num_discrete
        self.hard_prob = hard_prob
        self.temperature = temperature
        self.sample_entropy_weight = sample_entropy_weight
        self.to_logits = Linear(dim, num_discrete)

        if exists(shared_prototypes):
            self._prototypes = None
            self._shared_prototypes = (shared_prototypes,)
        else:
            self._prototypes = Parameter(torch.empty(num_discrete, dim_latent))
            nn.init.orthogonal_(self._prototypes, gain = (dim_latent ** 0.5) * 0.5)
            self._shared_prototypes = None

    @property
    def prototypes(self) -> Parameter:
        return self._shared_prototypes[0] if exists(self._shared_prototypes) else self._prototypes

    def forward(self, x):
        return self.to_logits(x)

    def sample(self, logits, temperature = 1., differentiable = True):
        probs = (logits / (self.temperature * temperature)).softmax(dim = -1)
        chosen = sample_straight_through(probs, self.hard_prob, training = self.training and differentiable)
        return einsum(chosen, self.prototypes, '... k, k d -> ... d')

    def calculate_kl_loss(self, logits, mask = None):
        probs = (logits / self.temperature).softmax(dim = -1)
        return categorical_entropy_loss(probs, mask = mask, sample_entropy_weight = self.sample_entropy_weight)

    def log_prob(self, logits, sampled_latent_action):
        idx = torch.cdist(sampled_latent_action, self.prototypes).argmin(dim = -1)
        return F.log_softmax(logits / self.temperature, dim = -1).gather(-1, rearrange(idx, '... -> ... 1'))

    def clone_for_proposer(self, dim_proposer: int) -> DiscreteLatentReadout:
        return DiscreteLatentReadout(
            dim_proposer,
            dim_latent = self.dim_latent,
            num_discrete = self.num_discrete,
            hard_prob = self.hard_prob,
            temperature = self.temperature,
            sample_entropy_weight = self.sample_entropy_weight,
            shared_prototypes = self.prototypes
        )

# meta controller

MetaControllerOutput = namedtuple('MetaControllerOutput', (
    'prev_hiddens',
    'input_residual_stream',
    'action_dist',
    'actions',
    'switch_beta',
    'kl_loss',
    'kl_loss_weight',
    'ratio_loss',
    'gated_actions'
), defaults = (None,))

GRPOOutput = namedtuple('GRPOOutput', (
    'state',
    'action',
    'log_prob',
    'switch_beta'
))

# TPO - Jean Kaddour https://arxiv.org/abs/2604.06159

def tpo_target(log_scores, u, eta = 1.0):
    return (F.log_softmax(log_scores, dim = -1) + u / eta).softmax(dim = -1)

def tpo_loss(log_p, q):
    return -einsum(q, log_p, '... k, ... k -> ...').mean()

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
    kl_loss_weight = 0.,
    use_tpo = False,
    tpo_eta = 1.0,
    tpo_group_size = None
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

    # masking

    if mask.ndim == 3:
        mask = rearrange(mask, 'b n 1 -> b n')

    if exists(episode_lens):
        episode_mask = lens_to_mask(episode_lens, mask.shape[1])
        mask = mask & episode_mask

    if use_tpo:
        episode_lens_float = mask.sum(dim = 1).clamp(min = 1.).float()

        advantages = advantages.flatten()

        group_size = default(tpo_group_size, advantages.shape[0])
        reshape_fn = lambda t: rearrange(t, '(b k) -> b k', k = group_size)

        u = reshape_fn(advantages)

        # target distribution q is computed from old log probs and detached

        old_log_prob_per_step = old_log_probs.sum(dim = -1) if old_log_probs.ndim == 3 else old_log_probs
        old_log_scores = (old_log_prob_per_step * mask).sum(dim = 1) / episode_lens_float

        q = tpo_target(reshape_fn(old_log_scores), u, eta = tpo_eta).detach()

        # log_p is computed from the current policy

        new_log_prob_per_step = new_log_probs.sum(dim = -1) if new_log_probs.ndim == 3 else new_log_probs
        new_log_scores = (new_log_prob_per_step * mask).sum(dim = 1) / episode_lens_float

        log_p = F.log_softmax(reshape_fn(new_log_scores), dim = -1)

        loss_val = tpo_loss(log_p, q)

        # kl penalty not needed

        if not return_kl_loss:
            return loss_val

        return loss_val, loss_val.new_tensor(0.)

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

    losses = reduce(losses, 'b n d -> b n', 'sum')

    loss_val = masked_mean(losses, mask)

    if not return_kl_loss:
        return loss_val

    if kl_loss.ndim == 3:
        kl_loss = reduce(kl_loss, 'b n d -> b n', 'sum')

    kl_loss = masked_mean(kl_loss, mask)

    return loss_val, kl_loss * kl_loss_weight

@move_inputs_to_module_device
def ratio_loss(
    meta_controller,
    target_segment_len: int,
    switch_beta,                # float[b n]
    hard_switch_beta = None,    # bool[b n]
    episode_lens = None,        # int[b]
    chunk_size = None,
    return_unreduced = False
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

    if return_unreduced:
        return losses

    return losses.mean()

def extract_grpo_data(meta_controller, transformer_output):
    meta_output = transformer_output.prev_hiddens.meta_controller

    state = meta_output.input_residual_stream
    action = meta_output.actions
    switch_beta = meta_output.switch_beta

    log_prob = meta_controller.log_prob(meta_output.action_dist, action)

    return GRPOOutput(state, action, log_prob, switch_beta)

@move_inputs_to_module_device
def reduce_discovery_samples(
    meta_controller,
    meta_output: MetaControllerOutput,
    action_recon_losses: Tensor,
    state_pred_losses: Tensor | None = None,
    episode_lens: Tensor | None = None,
    softmin_temp: float | None = None,
    eps: float = 1e-6,
    return_meta_controller_output: bool = False
):
    # Gladstone et al. https://arxiv.org/abs/2607.27372 - best-of-N over discovery samples

    action_dist, sampled_actions, switch_beta = meta_output.action_dist, meta_output.actions, meta_output.switch_beta
    batch, seq_len = action_dist.shape[:2]
    num_samples = sampled_actions.shape[0] // batch

    if num_samples == 1:
        state_loss = state_pred_losses.mean() if exists(state_pred_losses) else meta_controller.zero
        losses = DiscoveryLosses(state_loss, action_recon_losses.mean(), meta_output.kl_loss, meta_output.ratio_loss)
        return (losses, meta_output) if return_meta_controller_output else losses

    softmin_temp = default(softmin_temp, meta_controller.discovery_softmin_temp)

    episode_lens = maybe(repeat)(episode_lens, 'b -> (n b)', n = num_samples)
    mask = maybe(lens_to_mask)(episode_lens, seq_len)

    def to_sample_batch(losses):
        if not exists(losses):
            return None

        if losses.ndim == 0:
            return repeat(losses, ' -> n b', n = num_samples, b = batch)

        if losses.shape == (num_samples, batch):
            return losses

        if losses.ndim > 2:
            losses = reduce(losses, 'b t ... -> b t', 'mean')

        if losses.ndim == 2:
            loss_mask = mask[:, :losses.shape[-1]] if exists(mask) else None
            losses = masked_mean(losses, loss_mask, dim = -1)

        return rearrange(losses, '(n b) -> n b', n = num_samples)

    action_recon = to_sample_batch(action_recon_losses)
    state_pred = to_sample_batch(state_pred_losses)

    ratio = None

    if meta_controller.has_ratio_loss:
        # per-trajectory ratio, unchunked for candidate weighting

        ratio = meta_controller.ratio_loss(
            meta_controller.target_temporal_segment_len,
            switch_beta,
            episode_lens = episode_lens,
            return_unreduced = True
        )
        ratio = rearrange(ratio, '(n b) -> n b', n = num_samples) * meta_controller.current_ratio_loss_weight

    total_loss = action_recon + default(state_pred, meta_controller.zero) + default(ratio, meta_controller.zero)

    weights = F.softmin(total_loss.detach() / max(softmin_temp, eps), dim = 0)
    weighted_mean = lambda losses: einsum(weights, losses, 'n b, n b -> b').mean()

    action_recon_loss = weighted_mean(action_recon)
    state_pred_loss = default(maybe(weighted_mean)(state_pred), meta_controller.zero)
    ratio_loss = default(maybe(weighted_mean)(ratio), meta_output.ratio_loss)

    losses = DiscoveryLosses(state_pred_loss, action_recon_loss, meta_output.kl_loss, ratio_loss)

    if not return_meta_controller_output:
        return losses

    best_idx = total_loss.detach().argmin(dim = 0) * batch + torch.arange(batch, device = sampled_actions.device)

    def select(t, dim = 0):
        if not exists(t) or t.shape[dim] != num_samples * batch:
            return t

        return t.index_select(dim, best_idx)

    prev_summarized, prev_proposer_hidden, switch_hidden, switch_gated = meta_output.prev_hiddens

    reduced_meta_output = meta_output._replace(
        prev_hiddens = (prev_summarized, prev_proposer_hidden, select(switch_hidden, dim = 1), select(switch_gated)),
        actions = select(sampled_actions),
        switch_beta = select(switch_beta),
        ratio_loss = ratio_loss,
        gated_actions = select(meta_output.gated_actions)
    )

    return losses, reduced_meta_output

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
        pool_embedded_sequence = True,
        dim_sequence_summary_embed = 32, # the summary embedding from the sequence embedder needs to be bottlenecked
        latent_readout: Literal['continuous', 'discrete'] | Module | tuple[Module, Module] = 'continuous',
        num_discrete_latents = 4,
        discrete_hard_prob = 0.5,
        discrete_temperature = 0.2,
        discrete_sample_entropy_weight = 0.1,
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
        hard_switch_straight_through = False,
        kl_loss_weight = 1.,
        kl_loss_warmup_steps = 0,
        apply_kl_loss_weight = True,
        gate_kl_with_switch_beta = False,
        ratio_loss_final_weight = None,
        ratio_loss_warmdown_steps = 0,
        latent_action_barrier_scale: float | None = None,
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

        self.ratio_loss_final_weight = ratio_loss_final_weight
        self.ratio_loss_warmdown_steps = ratio_loss_warmdown_steps

        self.discovery_num_samples = discovery_num_samples
        self.discovery_softmin_temp = discovery_softmin_temp

        self.compact_sequence_embedding = compact_sequence_embedding

        dim_meta = default(dim_meta_controller, dim_model)

        # the linear that brings from model dimension

        self.summary_gru = GRU(dim_model, dim_model) # eq (12)

        self.model_to_meta = Linear(dim_model, dim_meta)

        if self.compact_sequence_embedding:
            self.internal_sequence_embedder = GRU(dim_model, dim_model)
        elif isinstance(internal_sequence_embedder, dict):
            self.internal_sequence_embedder = CausalSequenceEmbedder(dim = dim_model, pool = pool_embedded_sequence, **internal_sequence_embedder)
        else:
            self.internal_sequence_embedder = internal_sequence_embedder

        self.to_sequence_summary_embed = Linear(dim_model, dim_sequence_summary_embed)

        self.emitter = Feedforwards(
            dim_in = dim_meta + dim_model + dim_sequence_summary_embed,
            dim = dim_meta * 2,
            dim_out = dim_meta * 2,
            depth = 2
        )

        if isinstance(action_proposer, dict):
            # default to Decoder, wrapped for standard interface

            action_proposer = ActionProposerWrapper(
                Decoder(dim = dim_model, **action_proposer),
                cache_key = 'cache',
                return_cache_key = 'return_hiddens'
            )

        self.action_proposer = action_proposer

        if latent_readout == 'continuous':
            self.emitter_to_action_mean_log_var = Readout(dim_meta * 2, num_continuous = dim_latent)
            self.action_proposer_mean_log_var = Readout(dim_model, num_continuous = dim_latent)
        elif latent_readout == 'discrete':
            self.emitter_to_action_mean_log_var = DiscreteLatentReadout(
                dim_meta * 2,
                dim_latent = dim_latent,
                num_discrete = num_discrete_latents,
                hard_prob = discrete_hard_prob,
                temperature = discrete_temperature,
                sample_entropy_weight = discrete_sample_entropy_weight
            )
            self.action_proposer_mean_log_var = self.emitter_to_action_mean_log_var.clone_for_proposer(dim_model)
        elif isinstance(latent_readout, tuple):
            self.emitter_to_action_mean_log_var, self.action_proposer_mean_log_var = latent_readout
        elif isinstance(latent_readout, Module):
            self.emitter_to_action_mean_log_var = latent_readout
            self.action_proposer_mean_log_var = latent_readout.clone_for_proposer(dim_model) if hasattr(latent_readout, 'clone_for_proposer') else Readout(dim_model, num_continuous = dim_latent)
        else:
            raise ValueError(f'invalid latent_readout: {latent_readout}')

        # switching unit

        self.dim_latent = dim_latent
        self.switching_unit = GRU(dim_model + dim_meta + dim_latent, dim_meta)

        self.to_switching_unit_beta = nn.Linear(dim_meta, 1, bias = False)
        nn.init.xavier_uniform_(self.to_switching_unit_beta.weight)

        self.switch_temperature = switch_temperature

        self.switch_gating = AssocScan(**assoc_scan_kwargs)

        # latent action barrier from https://zzk273.github.io/LATENT/

        self.latent_action_barrier_scale = latent_action_barrier_scale

        # turn off the ratio loss by setting the weight to 0

        assert ratio_loss_weight >= 0.

        self.has_ratio_loss = ratio_loss_weight > 0.

        self.ratio_loss_weight = ratio_loss_weight

        self.ratio_loss_chunk_size = ratio_loss_chunk_size
        self.target_temporal_segment_len = target_temporal_segment_len

        # decoder

        assert hypernetwork_low_rank <= dim_model

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
        dim_log_prob = getattr(self.action_proposer_mean_log_var, 'dim_log_prob', self.dim_latent)
        return dict(
            states = ('float', self.dim_model),
            log_probs = ('float', dim_log_prob),
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
        if eval_rest:
            self.eval()

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
        action_dist,
        mask = None
    ):
        if hasattr(self.emitter_to_action_mean_log_var, 'calculate_kl_loss'):
            return self.emitter_to_action_mean_log_var.calculate_kl_loss(action_dist, mask = mask)

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
                logger.warning('meta controller cache being passed back in for discovery phase, which does not make sense during discovery')

            mask = maybe(lens_to_mask)(episode_lens, meta_embed.shape[1])

            kwargs = dict()
            if exists(episode_lens):
                kwargs['episode_lens'] = episode_lens

            # s(e_1:T) is a single compact embedding

            if self.compact_sequence_embedding:
                seq_input = nn.utils.rnn.pack_padded_sequence(residual_stream, episode_lens.cpu().long(), batch_first = True, enforce_sorted = False) if exists(episode_lens) else residual_stream
                _, summarized_sequence = self.internal_sequence_embedder(seq_input)
                summarized_sequence = summarized_sequence[-1] # B, D

                # channel-mixing projection
                summarized_sequence_embed = self.to_sequence_summary_embed(summarized_sequence) # B, 32
                summarized_sequence_embed = repeat(summarized_sequence_embed, 'b d -> b n d', n = seq_len)

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

            proposed_action_hidden = self.emitter(emitter_input)

            readout = self.emitter_to_action_mean_log_var

        else: # else internal rl phase

            proposed_action_hidden, next_action_proposer_hidden = self.action_proposer(
                residual_stream,
                cache = prev_action_proposer_hidden
            )

            readout = self.action_proposer_mean_log_var

        # sample from the gaussian as the action from the meta controller

        action_dist = readout(proposed_action_hidden)

        num_samples = default(num_samples, 1)
        has_multi_samples = num_samples > 1

        action_dist_for_sample = repeat(action_dist, 'b ... -> n b ...', n = num_samples) if has_multi_samples else action_dist
        sampled_latent_action = readout.sample(action_dist_for_sample, temperature = temperature, differentiable = True)
        sampled_latent_action = rearrange(sampled_latent_action, 'n b ... -> (n b) ...') if has_multi_samples else sampled_latent_action

        bounded_latent_action = sampled_latent_action

        if exists(self.latent_action_barrier_scale):
            scale = self.latent_action_barrier_scale
            bounded_latent_action = scale * (sampled_latent_action / scale).tanh()

        # switching unit timer

        batch, seq_len, dim = sampled_latent_action.shape
        input_residual_stream = residual_stream
        repeat_batch = partial(repeat, pattern = 'b ... -> (n b) ...', n = num_samples) if has_multi_samples else identity

        if has_multi_samples:
            repeat_hidden = partial(repeat, pattern = 'l b d -> l (n b) d', n = num_samples)

            residual_stream = repeat_batch(residual_stream)
            meta_embed_prev = repeat_batch(meta_embed_prev)
            ablate_switch_beta = maybe(repeat_batch)(ablate_switch_beta)
            prev_switching_unit_gru_hidden = maybe(repeat_hidden)(prev_switching_unit_gru_hidden)
            gated_prev_latent = maybe(repeat_batch)(gated_prev_latent)

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
                bounded_latent_action,
                prev_switching_unit_gru_hidden,
                z_prev,
                self.switch_temperature,
                resolved_hard_switch,
                default(hard_switch_straight_through, self.hard_switch_straight_through),
                is_first_step = not exists(cache)
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

                # force beta to 1 on first step (no cache) to match parallel path

                if not exists(cache):
                    switch_beta = torch.ones_like(switch_beta)

            # maybe hard switch

            hard_switch = default(hard_switch, self.hard_switch, not discovery_phase)

            switch_beta_for_gate = switch_beta

            if hard_switch:
                hard_switch_beta = (switch_beta_for_gate > 0.5).float()

                if default(hard_switch_straight_through, self.hard_switch_straight_through):
                    switch_beta_for_gate = straight_through(switch_beta_for_gate, hard_switch_beta)
                else:
                    switch_beta_for_gate = hard_switch_beta

                switch_beta = switch_beta_for_gate # return the gating beta, consistent with the parallel path

            # associative scan

            forget_gate = 1. - switch_beta_for_gate

            gated_sampled_latent_action = einx.multiply('b n d, b n', bounded_latent_action, switch_beta_for_gate)
            gated_action = self.switch_gating(forget_gate, gated_sampled_latent_action, prev = gated_prev_latent)

            next_switch_gated_action = gated_action[:, -1:]

        # need to encourage normal distribution

        kl_loss = kl_loss_weight = self.zero

        if discovery_phase:
            switch_beta_for_kl = reduce(switch_beta, '(n b) l -> b l', 'mean', n = num_samples) if has_multi_samples else switch_beta
            sw_mask = switch_beta_for_kl > 0.5
            kl_mask = and_masks((mask, sw_mask)) if self.gate_kl_with_switch_beta else mask

            kl_loss = self.calculate_kl_loss(action_dist, mask = kl_mask)

            if kl_loss.ndim == 3:
                kl_loss = reduce(kl_loss, 'b n d -> b n', 'sum')

            if self.gate_kl_with_switch_beta and kl_loss.ndim == 2:
                kl_loss = kl_loss * switch_beta_for_kl

            if kl_loss.ndim >= 2:
                kl_loss = masked_mean(kl_loss, mask)

            kl_loss_weight = self.current_kl_loss_weight if self.apply_kl_loss_weight else 1.
            kl_loss = kl_loss * kl_loss_weight

        # decoder

        decoder_out = self.decoder(gated_action)

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
            next_switching_unit_gru_hidden,
            next_switch_gated_action
        )

        return control_signal, MetaControllerOutput(next_hiddens, input_residual_stream, action_dist, sampled_latent_action, switch_beta, kl_loss, kl_loss_weight, aux_ratio_loss, gated_action)

MetaController.policy_loss = policy_loss
MetaController.ratio_loss = ratio_loss
MetaController.reduce_discovery_samples = reduce_discovery_samples

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
        embed_past_actions = True,
        normalize_state_action_losses = False,
        loss_normalizer_beta = 0.999,
        lower_transformer_post_norm = False,
        discovery_phase_state_clone_loss = False,
        dynamics_hidden_dim = None,
        dynamics_num_layers = 3,
        next_latent_loss_weight = 1.0
    ):
        super().__init__()

        self.normalize_state_action_losses = normalize_state_action_losses
        self.discovery_phase_state_clone_loss = discovery_phase_state_clone_loss
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
            lower_body = Decoder(
                dim = dim,
                pre_norm_has_final_norm = lower_transformer_post_norm,
                **transformer_kwargs,
                **lower_body
            )

        if isinstance(upper_body, dict):
            upper_body = Decoder(dim = dim, **transformer_kwargs, **upper_body)

        self.state_embed, self.state_readout = EmbedAndReadout(dim, **state_embed_readout)
        action_embed, self.action_readout = EmbedAndReadout(dim, **action_embed_readout)

        self.action_embed = action_embed if embed_past_actions else None

        self.lower_body = lower_body
        self.upper_body = upper_body

        lower_attn_dim_head = lower_body.attn_dim_head
        lower_heads = lower_body.attn_heads

        upper_attn_dim_head = upper_body.attn_dim_head
        upper_heads = upper_body.attn_heads

        assert lower_attn_dim_head == upper_attn_dim_head
        assert lower_heads == upper_heads

        # meta controller

        self.meta_controller = meta_controller

        # next latent prediction related - https://arxiv.org/abs/2511.05963

        self.next_latent_loss_fn = nn.SmoothL1Loss(reduction = 'none')
        self.next_latent_loss_weight = next_latent_loss_weight

        self.has_next_latent_loss = next_latent_loss_weight > 0.
        self.has_action_readout = exists(self.action_readout)

        self.next_latent_prediction_action_embed, _ = EmbedAndReadout(dim, **action_embed_readout)

        self.dynamics_model = ResidualDynamics(
            dim = dim,
            cond_dim = dim,
            hidden_dim = dynamics_hidden_dim,
            num_layers = dynamics_num_layers
        )

        # aux default loss

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
        return_action_cache = False,
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
        hard_switch: bool | None = None,
        metacontroller_residual_stream_dropout: float = 0.,
        discovery_num_samples: int | None = None,
        discovery_softmin_temp: float | None = None
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

            # actions

            target_actions = actions

            if seq_len == actions.shape[1]:
                actions = actions[:, :-1]

            # masking

            action_loss_mask = None

            if exists(episode_lens):
                loss_mask = lens_to_mask(episode_lens, state.shape[1])
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

        resolved_num_samples = 1
        if discovery_phase and exists(meta_controller) and meta_controller.training:
            resolved_num_samples = default(discovery_num_samples, getattr(meta_controller, 'discovery_num_samples', 1))

        has_multi_samples = resolved_num_samples > 1
        repeat_batch = partial(repeat, pattern = 'b ... -> (n b) ...', n = resolved_num_samples) if has_multi_samples else identity

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
                    hard_switch = hard_switch,
                    num_samples = resolved_num_samples
                )
            else:
                control_signal, next_meta_hiddens = self.zero, None

            if control_signal_multiplier != 1.0:
                control_signal = control_signal * control_signal_multiplier

            if metacontroller_residual_stream_dropout > 0.:
                is_training = meta_controller.training if exists(meta_controller) else self.training
                residual_stream = F.dropout(residual_stream, p = metacontroller_residual_stream_dropout, training = is_training)

            residual_stream_for_upper = repeat_batch(residual_stream)
            condition_for_upper = maybe(repeat_batch)(condition)

            modified_residual_stream = residual_stream_for_upper + control_signal

        # modified residual stream sent back to transformer upper body

        with upper_transformer_context():

            attended, next_upper_hiddens = self.upper_body(
                modified_residual_stream,
                condition = condition_for_upper,
                cache = upper_transformer_hiddens,
                return_hiddens = True
            )

            # head readout

            dist_params = self.action_readout(attended)

        # latent for returning

        latent = attended

        if calc_auto_regressive_loss and (behavioral_cloning or discovery_phase):

            next_latent_loss = self.zero
            latent_loss_unreduced = None

            if self.has_next_latent_loss and (behavioral_cloning or self.discovery_phase_state_clone_loss):
                action_embeds_for_prediction = repeat_batch(self.next_latent_prediction_action_embed(actions))

                curr_latent, target_latent = attended[:, :-1], attended[:, 1:].detach()
                mask = maybe(repeat_batch)(action_loss_mask[:, :-1]) if exists(action_loss_mask) else None

                dynamics_out = self.dynamics_model(
                    next_token_embeds = action_embeds_for_prediction,
                    curr_latent = curr_latent
                )

                latent_loss_unreduced = self.next_latent_loss_fn(
                    dynamics_out,
                    target_latent
                )

                next_latent_loss = masked_mean(latent_loss_unreduced, mask)

        # caching related

        next_cache_steps = cache_steps + seq_len
        return_cache_value = TransformerOutput(residual_stream, Hiddens(next_lower_hiddens, next_meta_hiddens, next_upper_hiddens), next_cache_steps)

        # maybe early return action dist with cache

        if return_action_cache:
            return dist_params, return_cache_value

        # maybe return behavior cloning loss

        if calc_auto_regressive_loss and behavioral_cloning:

            # state

            state_clone_loss = self.next_latent_loss_weight * next_latent_loss

            # action

            action_clone_loss = self.action_readout.calculate_loss(dist_params, target_actions, mask = action_loss_mask)

            # loss normalization

            if self.normalize_state_action_losses:
                state_clone_loss = self.bc_state_loss_normalizer(state_clone_loss, update_ema = update_loss_ema)
                action_clone_loss = self.bc_action_loss_normalizer(action_clone_loss, update_ema = update_loss_ema)

            losses = BehavioralCloningLosses(state_clone_loss, action_clone_loss)

            ret = (
                losses,
                dist_params if return_action_logits else None,
                next_meta_hiddens if return_meta_controller_output else None,
                residual_stream if return_residual_stream else None
            )

            ret = compact(ret)
            return ret[0] if len(ret) == 1 else ret

        elif calc_auto_regressive_loss and discovery_phase:

            action_recon_loss = self.action_readout.calculate_loss(
                dist_params,
                repeat_batch(target_actions),
                mask = maybe(repeat_batch)(action_loss_mask),
                return_unreduced_loss = has_multi_samples
            )

            state_pred_loss = None
            if self.discovery_phase_state_clone_loss and exists(latent_loss_unreduced):
                state_pred_loss = self.next_latent_loss_weight * (latent_loss_unreduced if has_multi_samples else next_latent_loss)

            losses, next_meta_hiddens = meta_controller.reduce_discovery_samples(
                next_meta_hiddens,
                action_recon_loss,
                state_pred_losses = state_pred_loss,
                episode_lens = episode_lens,
                softmin_temp = discovery_softmin_temp,
                return_meta_controller_output = True
            )

            # loss normalization

            if self.normalize_state_action_losses:
                state_clone_loss, action_recon_loss, *rest_losses = losses

                if self.discovery_phase_state_clone_loss:
                    state_clone_loss = self.discovery_state_loss_normalizer(state_clone_loss, update_ema = update_loss_ema)

                action_recon_loss = self.discovery_action_loss_normalizer(action_recon_loss, update_ema = update_loss_ema)

                losses = DiscoveryLosses(state_clone_loss, action_recon_loss, *rest_losses)

            if not return_meta_controller_output:
                return losses

            return losses, next_meta_hiddens

        # returning

        return_one = not (return_latents or return_cache)

        if return_one:
            return dist_params

        return dist_params, return_cache_value
