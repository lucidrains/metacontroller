from __future__ import annotations
from functools import partial
from collections import namedtuple

import torch
from torch import cat, Tensor
from torch.nn import GRU, Linear

from einops import rearrange

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

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

# output container

SequentialSelectionOutput = namedtuple('SequentialSelectionOutput', [
    'switch_beta', 
    'gated_action', 
    'next_switching_unit_gru_hidden'
])

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
    gated_prev_latent,
    sampled_prev_latent,
    switch_temperature,
    hard_switch
):
    """
    Optimized JAX implementation of sequential latent action selection.
    Uses pre-computed input projections to minimize GEMMs inside the scan loop.
    """
    
    # input-to-hidden projections
    
    # beta_t depends on S_{t-1}
    # we need to concatenate initial sampled latent with all but last proposal
    
    prev_sampled_latent_actions = jnp.concatenate([
        rearrange(sampled_prev_latent, '... -> 1 ...'),
        sampled_latent_actions[:, :-1]
    ], axis = 1)

    gru_input_all = jnp.concatenate([
        residual_stream, 
        meta_embeddings, 
        prev_sampled_latent_actions
    ], axis = -1)
    
    projected_input_hidden_all = gru_input_all @ weight_ih.T + bias_ih
    
    # transpose for lax.scan indexing (seq_len, batch, dim)

    projected_input_hidden_all_T = rearrange(projected_input_hidden_all, 'b n d -> n b d')
    sampled_latent_actions_T = rearrange(sampled_latent_actions, 'b n d -> n b d')
    
    def scan_fn(carry, x):
        h_prev, z_prev = carry
        projected_input_hidden, sampled_latent = x
        
        # 1. GRU Cell Step

        gates_hh = h_prev @ weight_hh.T + bias_hh
        
        i_r, i_z, i_n = jnp.split(projected_input_hidden, 3, axis = -1)
        h_r, h_z, h_n = jnp.split(gates_hh, 3, axis = -1)
        
        reset_gate = jax.nn.sigmoid(i_r + h_r)
        update_gate = jax.nn.sigmoid(i_z + h_z)
        new_gate = jnp.tanh(i_n + reset_gate * h_n)
        
        h_next = (1 - update_gate) * new_gate + update_gate * h_prev
        
        # 2. Beta (Switching Logit)

        logit = (h_next @ weight_beta.T) + bias_beta
        beta = jax.nn.sigmoid(logit / switch_temperature)
        
        if hard_switch:
            beta = jnp.where(beta > 0.5, 1.0, 0.0)
            
        # 3. Gating (Momentum-like transition)

        z_next = z_prev * (1.0 - beta) + sampled_latent * beta
        
        return (h_next, z_next), (beta, z_next)

    initial_carry = (gru_hidden, gated_prev_latent)
    inputs = (projected_input_hidden_all_T, sampled_latent_actions_T)

    # perform scan

    final_carry, (betas, gated_actions) = lax.scan(
        scan_fn,
        initial_carry,
        inputs
    )
    
    # return batch-first tensors and final hidden state

    return (
        rearrange(betas, 'n b d -> b n d'), 
        rearrange(gated_actions, 'n b d -> b n d'), 
        final_carry[0]
    )

# Torch-JAX AutoGrad Bridge

if HAS_JAX:
    from torch.utils.dlpack import from_dlpack as torch_from_dlpack
    from jax.dlpack import from_dlpack as jax_from_dlpack

    # Define the backward-supporting JIT function at module level for persistent caching

    @partial(jax.jit, static_argnums = (12, 13))
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
        gated_prev_latent,
        sampled_prev_latent,
        switch_temperature,
        hard_switch
    ):
        return vjp(
            lambda *args: jax_sequential_selection_scan(*args, switch_temperature, hard_switch),
            weight_ih, weight_hh, bias_ih, bias_hh, weight_beta, bias_beta, 
            residual_stream, meta_embeddings, sampled_latent_actions, gru_hidden, gated_prev_latent, sampled_prev_latent
        )

    # dlpack conversion helpers

    def t2j(t):
        if not exists(t):
            return None
        if not isinstance(t, torch.Tensor):
            return t
        
        # detach/contiguous required for DLPack export in autograd functions

        return jax_from_dlpack(t.detach().contiguous())

    def j2t(j, device):
        if not exists(j):
            return None
        if isinstance(j, torch.Tensor):
            return j.to(device)
            
        return torch_from_dlpack(j).to(device)

    class JaxSequentialSelection(torch.autograd.Function):
        """
        Bridges Torch autograd to JAX optimized scan.
        Uses DLPack for zero-copy tensor sharing.
        """
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
            gated_prev_latent,
            sampled_prev_latent,
            switch_temperature,
            hard_switch
        ):
            device = residual_stream.device
            ctx.device = device
            
            # convert all torch tensors to jax

            jax_args = [t2j(t) for t in (weight_ih, weight_hh, bias_ih, bias_hh, weight_beta, bias_beta, residual_stream, meta_embeddings, sampled_latent_actions, gru_hidden, gated_prev_latent, sampled_prev_latent)]
            
            # use module-level JIT to compute forward and retrieve VJP function

            outputs, ctx.vjp_fn = jitted_vjp_logic(*jax_args, float(switch_temperature), bool(hard_switch))
            
            ctx.out_shapes = [output.shape for output in outputs]
            ctx.out_dtypes = [output.dtype for output in outputs]
            
            return j2t(outputs[0], device), j2t(outputs[1], device), j2t(outputs[2], device)

        @staticmethod
        def backward(ctx, grad_betas, grad_gated_actions, grad_h_final):
            device = ctx.device

            # fill None gradients with zeros to satisfy JAX VJP expectations

            def to_jax_grad(grad, shape, dtype):
                if not exists(grad):
                    return jnp.zeros(shape, dtype = dtype)
                return t2j(grad)

            jax_grads = (
                to_jax_grad(grad_betas, ctx.out_shapes[0], ctx.out_dtypes[0]),
                to_jax_grad(grad_gated_actions, ctx.out_shapes[1], ctx.out_dtypes[1]),
                to_jax_grad(grad_h_final, ctx.out_shapes[2], ctx.out_dtypes[2])
            )
            
            # backprop through JAX

            jax_input_grads = ctx.vjp_fn(jax_grads)
            
            # convert input array gradients back to Torch

            torch_grads = tuple(j2t(grad, device) for grad in jax_input_grads)
            
            # return grads for arrays, None for static scalars

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
        gated_prev_latent,
        sampled_prev_latent,
        switch_temperature,
        hard_switch
    ):
        return JaxSequentialSelection.apply(weight_ih, weight_hh, bias_ih, bias_hh, weight_beta, bias_beta, residual_stream, meta_embeddings, sampled_latent_actions, gru_hidden, gated_prev_latent, sampled_prev_latent, switch_temperature, hard_switch)

# reference PyTorch implementation

def pytorch_sequential_selection(
    gru: GRU, 
    to_beta: Linear, 
    residual_stream: Tensor, 
    meta_embeddings: Tensor, 
    sampled_latent_actions: Tensor, 
    hidden_init: Tensor | None, 
    gated_prev_latent: Tensor, 
    sampled_prev_latent: Tensor, 
    switch_temperature: float, 
    hard_switch: bool
):
    batch, seq_len, _ = residual_stream.shape
    betas, gated = [], []
    
    hidden, latent = hidden_init, gated_prev_latent
    
    for t in range(seq_len):
        
        # 1. GRU Step
        # beta_t depends on S_{t-1}

        prev_sampled_latent = sampled_latent_actions[:, t-1:t] if t > 0 else sampled_prev_latent

        input_t = cat([
            residual_stream[:, t:t+1],
            meta_embeddings[:, t:t+1],
            prev_sampled_latent
        ], dim = -1)
        
        _, hidden = gru(input_t, hidden)
        
        # 2. Beta Calculation

        beta_input = rearrange(hidden, '1 b d -> b 1 d')
        logit = to_beta(beta_input)
        beta = (logit / switch_temperature).sigmoid()
        
        if hard_switch: 
            beta = (beta > 0.5).float()
            
        # 3. Latent Momentum Gating

        latent = latent * (1.0 - beta) + sampled_latent_actions[:, t:t+1] * beta
        
        betas.append(beta)
        gated.append(latent)
        
    return SequentialSelectionOutput(
        cat(betas, dim = 1).squeeze(-1),
        cat(gated, dim = 1),
        hidden
    )

# Primary Discovery Dispatcher

def perform_sequential_selection(
    gru: GRU, 
    to_beta: Linear, 
    residual_stream: Tensor, 
    meta_embeddings: Tensor, 
    sampled_latent_actions: Tensor, 
    hidden_init: Tensor | None, 
    gated_prev_latent: Tensor, 
    sampled_prev_latent: Tensor, 
    switch_temperature: float, 
    hard_switch: bool
):
    """
    Dispatcher that prioritizes optimized JAX implementation on GPU.
    Falls back to PyTorch eager loop on CPU or if JAX is missing.
    """
    
    if HAS_JAX:
        try:
            device = residual_stream.device
            weight_ih, weight_hh = gru.weight_ih_l0, gru.weight_hh_l0
            bias_ih, bias_hh = gru.bias_ih_l0, gru.bias_hh_l0
            weight_beta, bias_beta = to_beta.weight, to_beta.bias
            
            # reshape inputs for JAX-friendly format

            gru_hidden = rearrange(hidden_init, '1 b h -> b h') if exists(hidden_init) else torch.zeros(residual_stream.shape[0], gru.hidden_size, device = device)
            gated_prev_latent_jax = rearrange(gated_prev_latent, 'b 1 l -> b l')
            sampled_prev_latent_jax = rearrange(sampled_prev_latent, 'b 1 l -> b l')
            
            outputs = torch_jax_sequential_selection(
                weight_ih, weight_hh, bias_ih, bias_hh, weight_beta, bias_beta, 
                residual_stream, meta_embeddings, sampled_latent_actions, 
                gru_hidden, gated_prev_latent_jax, sampled_prev_latent_jax,
                switch_temperature, hard_switch
            )
            
            switch_betas, gated_actions, final_hidden = outputs
            
            return SequentialSelectionOutput(
                switch_betas.squeeze(-1),
                gated_actions,
                final_hidden.unsqueeze(0)
            )
            
        except Exception:
            # fallback on any JAX-specific failure
            pass

    return pytorch_sequential_selection(gru, to_beta, residual_stream, meta_embeddings, sampled_latent_actions, hidden_init, gated_prev_latent, sampled_prev_latent, switch_temperature, hard_switch)
