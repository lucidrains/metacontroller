from __future__ import annotations
from functools import partial
from collections import namedtuple

import torch
from torch import cat, Tensor
from torch.nn import GRU, Linear

from einops import rearrange
import einx

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# output container

SequentialActionSelectionOutput = namedtuple('SequentialActionSelectionOutput', [
    'switch_beta', 
    'gated_action', 
    'next_switching_unit_gru_hidden'
])

# JAX initialization - optional for CPU or if jax is not installed

try:
    import jax
    import jax.numpy as jnp
    from jax import lax, vjp
    from jax import config
    config.update("jax_enable_x64", True)
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

# JAX implementation - optimized with lax.scan

def jax_sequential_selection_scan(
    weight_ih,
    weight_hh,
    bias_ih,
    bias_hh,
    weight_beta,
    bias_beta,
    residual_stream,
    meta_embeddings,
    sampled_latent_actions,
    gru_hidden,
    current_latent_action,
    switch_temperature,
    hard_switch
):
    """
    Optimized JAX implementation of sequential latent action selection.
    Uses pre-computed input projections for static inputs (residual stream, meta embeddings)
    to minimize GEMMs inside the scan loop.
    """
    
    # dimensions
    
    dim_latent = current_latent_action.shape[-1]
    
    # input-to-hidden projections for static components (residual stream + meta embeddings)
    
    weight_static = weight_ih[:, :-dim_latent]
    weight_z = weight_ih[:, -dim_latent:]
    
    projected_static = jnp.concatenate([residual_stream, meta_embeddings], axis = -1) @ weight_static.T
    
    if exists(bias_ih):
        projected_static = projected_static + bias_ih
    
    # transpose for lax.scan indexing (seq_len, batch, dim)

    projected_static_T = rearrange(projected_static, 'b n d -> n b d')
    sampled_latent_actions_T = rearrange(sampled_latent_actions, 'b n d -> n b d')
    
    def scan_fn(carry, x):
        h_prev, z_prev = carry
        proj_static, sampled_latent = x
        
        # 1. GRU Cell Step
        
        # project previous interpolated action (dynamic inside loop)
        proj_z = z_prev @ weight_z.T
        proj_input = proj_static + proj_z

        gates_hh = h_prev @ weight_hh.T
        
        if exists(bias_hh):
            gates_hh = gates_hh + bias_hh
        
        i_r, i_z, i_n = jnp.split(proj_input, 3, axis = -1)
        h_r, h_z, h_n = jnp.split(gates_hh, 3, axis = -1)
        
        reset_gate = jax.nn.sigmoid(i_r + h_r)
        update_gate = jax.nn.sigmoid(i_z + h_z)
        new_gate = jnp.tanh(i_n + reset_gate * h_n)
        
        h_next = (1 - update_gate) * new_gate + update_gate * h_prev
        
        # 2. Beta (Switching Logit)

        logit = h_next @ weight_beta.T
        
        if exists(bias_beta):
            logit = logit + bias_beta
            
        beta = jax.nn.sigmoid(logit / switch_temperature)
        
        if hard_switch:
            beta_hard = (beta > 0.5).astype(beta.dtype)
            beta = beta + jax.lax.stop_gradient(beta_hard - beta)
            
        # 3. Gating (Momentum-like transition)
        
        beta = jnp.squeeze(beta, axis = -1)
        forget_gate = 1.0 - beta
        
        z_next = einx.multiply('... d, ... -> ... d', z_prev, forget_gate) + \
                 einx.multiply('... d, ... -> ... d', sampled_latent, beta)
        
        return (h_next, z_next), (beta, z_next)

    initial_carry = (gru_hidden, current_latent_action)
    inputs = (projected_static_T, sampled_latent_actions_T)

    # perform scan

    final_carry, (betas, gated_actions) = lax.scan(
        scan_fn,
        initial_carry,
        inputs
    )
    
    # return batch-first tensors and final hidden state

    return (
        rearrange(betas, 'n b -> b n'), 
        rearrange(gated_actions, 'n b d -> b n d'), 
        final_carry[0]
    )

# Torch-JAX AutoGrad Bridge

if HAS_JAX:
    from torch.utils.dlpack import from_dlpack as torch_from_dlpack
    from jax.dlpack import from_dlpack as jax_from_dlpack

    @partial(jax.jit, static_argnums = (11, 12))
    def jax_sequential_selection_scan_jitted(
        weight_ih,
        weight_hh,
        bias_ih,
        bias_hh,
        weight_beta,
        bias_beta,
        residual_stream,
        meta_embeddings,
        sampled_latent_actions,
        gru_hidden,
        current_latent_action,
        switch_temperature,
        hard_switch
    ):
        return jax_sequential_selection_scan(
            weight_ih, weight_hh, bias_ih, bias_hh, weight_beta, bias_beta, 
            residual_stream, meta_embeddings, sampled_latent_actions, gru_hidden, current_latent_action,
            switch_temperature, hard_switch
        )

    def jitted_vjp_logic(
        weight_ih,
        weight_hh,
        bias_ih,
        bias_hh,
        weight_beta,
        bias_beta,
        residual_stream,
        meta_embeddings,
        sampled_latent_actions,
        gru_hidden,
        current_latent_action,
        switch_temperature,
        hard_switch
    ):
        return vjp(
            lambda *args: jax_sequential_selection_scan_jitted(*args, switch_temperature, hard_switch),
            weight_ih, weight_hh, bias_ih, bias_hh, weight_beta, bias_beta, 
            residual_stream, meta_embeddings, sampled_latent_actions, gru_hidden, current_latent_action
        )

    # dlpack conversion helpers

    def t2j(t):
        if not exists(t):
            return None
        if not isinstance(t, torch.Tensor):
            return t
        return jax_from_dlpack(t.detach().contiguous())

    def j2t(j, device):
        if not exists(j):
            return None
        if isinstance(j, torch.Tensor):
            return j.to(device)
        return torch_from_dlpack(j).to(device)

    class JaxSequentialSelection(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
            weight_beta,
            bias_beta,
            residual_stream,
            meta_embeddings,
            sampled_latent_actions,
            gru_hidden,
            current_latent_action,
            switch_temperature,
            hard_switch
        ):
            device = residual_stream.device
            ctx.device = device
            
            jax_args = [t2j(t) for t in (weight_ih, weight_hh, bias_ih, bias_hh, weight_beta, bias_beta, residual_stream, meta_embeddings, sampled_latent_actions, gru_hidden, current_latent_action)]
            
            outputs, ctx.vjp_fn = jitted_vjp_logic(*jax_args, float(switch_temperature), bool(hard_switch))
            
            ctx.out_shapes = [output.shape for output in outputs]
            ctx.out_dtypes = [output.dtype for output in outputs]
            
            return j2t(outputs[0], device), j2t(outputs[1], device), j2t(outputs[2], device)

        @staticmethod
        def backward(ctx, grad_betas, grad_gated_actions, grad_h_final):
            device = ctx.device

            def to_jax_grad(grad, shape, dtype):
                if not exists(grad):
                    return jnp.zeros(shape, dtype = dtype)
                return t2j(grad)

            jax_grads = (
                to_jax_grad(grad_betas, ctx.out_shapes[0], ctx.out_dtypes[0]),
                to_jax_grad(grad_gated_actions, ctx.out_shapes[1], ctx.out_dtypes[1]),
                to_jax_grad(grad_h_final, ctx.out_shapes[2], ctx.out_dtypes[2])
            )
            
            jax_input_grads = ctx.vjp_fn(jax_grads)
            torch_grads = tuple(j2t(grad, device) for grad in jax_input_grads)
            
            return torch_grads + (None, None)

    def torch_jax_sequential_selection(
        weight_ih,
        weight_hh,
        bias_ih,
        bias_hh,
        weight_beta,
        bias_beta,
        residual_stream,
        meta_embeddings,
        sampled_latent_actions,
        gru_hidden,
        current_latent_action,
        switch_temperature,
        hard_switch
    ):
        return JaxSequentialSelection.apply(weight_ih, weight_hh, bias_ih, bias_hh, weight_beta, bias_beta, residual_stream, meta_embeddings, sampled_latent_actions, gru_hidden, current_latent_action, switch_temperature, hard_switch)

# reference PyTorch implementation

def pytorch_sequential_action_selection(
    gru: GRU, 
    to_beta: Linear, 
    residual_stream: Tensor, 
    meta_embeddings: Tensor, 
    sampled_latent_actions: Tensor, 
    hidden_init: Tensor | None, 
    latent_init: Tensor, 
    switch_temperature: float, 
    hard_switch: bool
):
    batch, seq_len, _ = residual_stream.shape
    betas, gated = [], []
    
    hidden, latent = hidden_init, latent_init
    
    for t in range(seq_len):
        
        # 1. GRU Step
        # input is current residual, current meta, and PREVIOUS latent

        input_t = cat([
            residual_stream[:, t:t+1],
            meta_embeddings[:, t:t+1],
            latent
        ], dim = -1)
        
        _, hidden = gru(input_t, hidden)
        
        # 2. Beta Calculation

        beta_logit = to_beta(hidden)
        beta = (beta_logit / switch_temperature).sigmoid()
        beta = rearrange(beta, '1 b 1 -> b 1')
        
        if hard_switch: 
            beta_hard = (beta > 0.5).float()
            beta = beta + (beta_hard - beta).detach()

        # 3. Latent Momentum Gating

        forget_gate = 1.0 - beta
        sampled_latent = sampled_latent_actions[:, t:t+1]
        
        latent = einx.multiply('b 1 d, b 1 -> b 1 d', latent, forget_gate) + \
                 einx.multiply('b 1 d, b 1 -> b 1 d', sampled_latent, beta)
        
        betas.append(beta)
        gated.append(latent)
        
    return SequentialActionSelectionOutput(
        cat(betas, dim = 1),
        cat(gated, dim = 1),
        hidden
    )

# Primary Discovery Dispatcher

def perform_sequential_action_selection(
    gru: GRU, 
    to_beta: Linear, 
    residual_stream: Tensor, 
    meta_embeddings: Tensor, 
    sampled_latent_actions: Tensor, 
    hidden_init: Tensor | None, 
    latent_init: Tensor, 
    switch_temperature: float, 
    hard_switch: bool
):
    """
    Dispatcher that prioritizes optimized JAX implementation on GPU.
    Falls back to PyTorch eager loop on CPU or if JAX is missing.
    """
    
    if HAS_JAX and residual_stream.is_cuda:
        try:
            device = residual_stream.device
            weight_ih, weight_hh = gru.weight_ih_l0, gru.weight_hh_l0
            bias_ih, bias_hh = gru.bias_ih_l0, gru.bias_hh_l0
            weight_beta, bias_beta = to_beta.weight, to_beta.bias
            
            # reshape inputs for JAX-friendly format

            gru_hidden = rearrange(hidden_init, '1 b h -> b h') if exists(hidden_init) else torch.zeros(residual_stream.shape[0], gru.hidden_size, device = device)
            current_latent_action = rearrange(latent_init, 'b 1 l -> b l')
            
            outputs = torch_jax_sequential_selection(
                weight_ih, weight_hh, bias_ih, bias_hh, weight_beta, bias_beta, 
                residual_stream, meta_embeddings, sampled_latent_actions, 
                gru_hidden, current_latent_action, 
                switch_temperature, hard_switch
            )
            
            switch_betas, gated_actions, final_hidden = outputs
            
            return SequentialActionSelectionOutput(
                switch_betas,
                gated_actions,
                final_hidden.unsqueeze(0)
            )
            
        except (RuntimeError, TypeError) as e:
            # fallback on specific JAX/DLPack integration failures
            import logging
            logging.warning(f"JAX fallback triggered: {e}")
            pass

    return pytorch_sequential_action_selection(gru, to_beta, residual_stream, meta_embeddings, sampled_latent_actions, hidden_init, latent_init, switch_temperature, hard_switch)
