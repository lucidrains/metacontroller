from __future__ import annotations
from contextlib import nullcontext
from collections import namedtuple
from loguru import logger

import torch
from torch import nn, cat, tensor, is_tensor, Tensor, arange
from torch.nn import Module, Linear, Parameter, RMSNorm, LayerNorm
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity, pad

import einx
from einx import multiply
from einops import einsum, rearrange, repeat, reduce
from einops.layers.torch import Rearrange

from x_transformers import Decoder
from x_mlps_pytorch import Feedforwards

from discrete_continuous_embed_readout import EmbedAndReadout, Readout
from assoc_scan import AssocScan

from torch_einops_utils import maybe, pad_left_at_dim, pad_at_dim, lens_to_mask, masked_mean, align_dims_left
from torch_einops_utils.device import module_device, move_inputs_to_module_device
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

def frac_gradient(t, frac = 1.):
    if frac == 1:
        return t
    t_grad = t * frac
    return straight_through(t_grad, t)

def binary_entropy(p):
    eps = 1e-8
    return -((p * (p + eps).log()) + (1. - p) * ((1. - p + eps).log()))

def calculate_ratio_loss(switch_probs: Tensor, target_avg_token_length: float):
    N = target_avg_token_length
    boundary_mask = switch_probs > 0.5
    
    F_soft = (switch_probs - 0.5).sigmoid()
    F_mask = straight_through(F_soft, boundary_mask.float()).mean(dim = -1)
    G = switch_probs.mean(dim = -1)
    
    ratio_loss = N / (N - 1) * ((N - 1) * F_mask * G + (1. - F_mask) * (1. - G))
    return ratio_loss.mean()

# constants

MetaControllerOutput = namedtuple('MetaControllerOutput', (
    'prev_hiddens',
    'input_residual_stream',
    'action_dist',
    'actions',
    'switch_beta',
    'ratio_loss'
))

TransformerWithMetacontrollerCache = namedtuple('TransformerWithMetacontrollerCache', (
    'lower_body',
    'emitter_decoder',
    'upper_body',
    'prev_key',
    'prev_gated_action',
    'cache_steps'
))

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
        switch_temperature = 0.1,
        target_avg_token_length = 8.,
        assoc_scan_kwargs: dict = dict()
    ):
        super().__init__()
        
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
        
        # qk module

        self.to_queries_keys = nn.Sequential(
            RMSNorm(dim),
            Linear(dim, dim_queries_keys * 2, bias = False)
        )

        self.start_key_token = Parameter(torch.randn(dim_queries_keys) * 1e-2)
        self.target_avg_token_length = target_avg_token_length
        
        # latent space (simple gaussian)

        self.latent_readout = Readout(
            dim, 
            num_continuous = dim_latent
        )
        
        # associative scan for sequence chunking

        self.assoc_scan = AssocScan(**assoc_scan_kwargs)
        
        # latent to control signal

        self.to_control_signal = Linear(dim_latent, dim, bias = False)
        
        # switch temperature

        self.switch_temperature = switch_temperature

        # upper body

        self.upper_body = Decoder(
            dim = dim,
            use_rmsnorm = True,
            polar_pos_emb = True,
            **upper_body
        )
        
        self.modified_residual_stream_norm = LayerNorm(dim, bias = False)

        self.register_buffer('zero', tensor(0.), persistent = False)

    def forward(
        self,
        state,
        actions: Tensor | None = None,
        return_loss = True,
        cache: TransformerWithMetacontrollerCache | None = None,
        return_cache = False
    ):
        batch, seq_len, device = *state.shape[:2], state.device
        
        target_state = target_actions = None

        if return_loss:
            target_state = state[:, 1:]
            
            assert exists(actions), '`actions` cannot be empty when turning on return_loss'
            
            target_actions = actions
            
            if seq_len == actions.shape[1]:
                actions = actions[:, :-1]
            
        lower_body_cache, emitter_decoder_cache, upper_body_cache, prev_key, prev_gated_action, cache_steps = default(cache, (None,) * 5 + (0,))

        # embed and process through lower body

        state_embed = self.state_embed(state)
        action_embed_out = 0.
        
        if exists(actions) and exists(self.action_embed):
            action_embed_out = self.action_embed(actions)
            
        if is_tensor(action_embed_out) and action_embed_out.shape[1] == (seq_len - 1):
            action_embed_out = pad_left_at_dim(action_embed_out, 1, dim = 1)
            
        embed = state_embed + action_embed_out
        
        residual_stream, next_lower_body_cache = self.lower_body(embed, cache = lower_body_cache, return_hiddens = True)
        
        # derive switching probabilities from qk module

        queries, keys = self.to_queries_keys(residual_stream).chunk(2, dim = -1)
        
        if not exists(prev_key):
            start_keys = repeat(self.start_key_token, 'd -> b 1 d', b = batch)
            keys_with_prev = cat((start_keys, keys), dim = 1)
        else:
            keys_with_prev = cat((prev_key, keys), dim = 1)
            
        next_prev_key = keys_with_prev[:, -1:]
        
        cosine_sim = cosine_similarity(queries, keys_with_prev[:, :-1], dim = -1)
        switch_probs = (-cosine_sim / self.switch_temperature).sigmoid()
        
        # obtain latents from emitter + readout

        emitter_out, next_emitter_decoder_cache = self.emitter_decoder(residual_stream, cache = emitter_decoder_cache, return_hiddens = True)
        action_dist = self.latent_readout(emitter_out)
        
        sampled_latent_action = self.latent_readout.sample(action_dist, differentiable = True)
        
        # gated action and associative scan for sequence chunking

        gated_sampled_latent_action = multiply('b n d, b n', sampled_latent_action, switch_probs)
        forget_gate = 1. - switch_probs
        
        scanned_latent_action = self.assoc_scan(forget_gate, gated_sampled_latent_action, prev = prev_gated_action)
        next_prev_gated_action = scanned_latent_action[:, -1:]
        
        # latent to control signal

        control_signal = self.to_control_signal(scanned_latent_action)
        modified_residual_stream = residual_stream + control_signal
        
        # apply layernorm without bias

        modified_residual_stream = self.modified_residual_stream_norm(modified_residual_stream)
        
        # process sequence upper body

        attended, next_upper_body_cache = self.upper_body(modified_residual_stream, cache = upper_body_cache, return_hiddens = True)
        dist_params = self.action_readout(attended)
        
        state_dist_params = None
        if return_loss:
            state_dist_params = self.state_readout(attended[:, :-1])
        
        next_cache_steps = cache_steps + seq_len
        next_cache = TransformerWithMetacontrollerCache(
            lower_body = next_lower_body_cache,
            emitter_decoder = next_emitter_decoder_cache,
            upper_body = next_upper_body_cache,
            prev_key = next_prev_key,
            prev_gated_action = next_prev_gated_action,
            cache_steps = next_cache_steps
        )
        
        # losses

        bc_state_loss = bc_action_loss = ratio_loss = None

        if return_loss:
            # a. bc loss
            bc_state_loss = self.state_readout.calculate_loss(state_dist_params, target_state)
            bc_action_loss = self.action_readout.calculate_loss(dist_params, target_actions)
            
            # b. ratio loss (from dynamicsequencechunker)
            ratio_loss = calculate_ratio_loss(switch_probs, self.target_avg_token_length)
        
        meta_output = MetaControllerOutput(
            prev_hiddens = None,
            input_residual_stream = residual_stream,
            action_dist = action_dist,
            actions = sampled_latent_action,
            switch_beta = switch_probs,
            ratio_loss = ratio_loss
        )
        
        ret = (dist_params, meta_output) if not return_loss else ((bc_state_loss, bc_action_loss, ratio_loss), meta_output)

        if return_cache:
            return (*ret, next_cache)

        return ret
