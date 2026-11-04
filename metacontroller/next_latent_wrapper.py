from __future__ import annotations
from collections import namedtuple
from copy import deepcopy
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn, tensor, is_tensor, Tensor
from torch.nn import Module

from einops import repeat, reduce
from torch_einops_utils import maybe, pad_left_at_dim, lens_to_mask, masked_mean
from torch_einops_utils.save_load import save_load

from metacontroller.metacontroller import (
    exists,
    default,
    compact,
    identity,
    ResidualDynamics,
    Transformer
)

# constants

Losses = namedtuple('Losses', (
    'action_recon',
    'next_latent',
    'kl',
    'meta_kl',
    'ratio'
))

# next latent prediction wrapper for metacontroller discovery (Teoh et al. https://arxiv.org/abs/2511.05963)

@save_load()
class NextLatentWrapper(Module):
    def __init__(
        self,
        model: Transformer,
        meta_controller: Module | None = None,
        *,
        dim: int | None = None,
        dim_action: int | None = None,
        dynamics_hidden_dim: int | None = None,
        dynamics_num_layers: int = 3,
        next_latent_loss_weight: float = 1.,
        kl_loss_weight: float = 1.,
        use_gated_action: bool = True
    ):
        super().__init__()
        meta_controller = default(meta_controller, model.meta_controller)
        assert exists(meta_controller), '`meta_controller` must be provided either on `Transformer` or passed to `NextLatentWrapper`'

        self.model = model
        self.meta_controller = meta_controller

        dim = default(dim, getattr(meta_controller, 'dim_model', None), model.upper_body.dim)
        dim_action = default(dim_action, getattr(meta_controller, 'dim_latent', None), getattr(meta_controller, 'num_codes', None))

        self.dynamics_model = ResidualDynamics(
            dim = dim,
            cond_dim = dim_action,
            hidden_dim = dynamics_hidden_dim,
            num_layers = dynamics_num_layers
        )

        self.loss_fn = nn.SmoothL1Loss(reduction = 'none')

        self.next_latent_loss_weight = next_latent_loss_weight
        self.kl_loss_weight = kl_loss_weight

        self.has_next_latent_loss = next_latent_loss_weight > 0.
        self.has_kl_loss = kl_loss_weight > 0.

        self.use_gated_action = use_gated_action

        self.register_buffer('zero', tensor(0.), persistent = False)

    def discovery_parameters(self):
        return [
            *self.meta_controller.discovery_parameters(),
            *self.dynamics_model.parameters()
        ]

    def discovery_parameters_non_switching(self):
        return [
            *self.meta_controller.discovery_parameters_non_switching(),
            *self.dynamics_model.parameters()
        ]

    def forward(
        self,
        state: Tensor | None = None,
        actions: Tensor | None = None,
        *,
        residual_stream: Tensor | None = None,
        episode_lens: Tensor | None = None,
        condition: Tensor | None = None,
        ablate_switch_beta: Tensor | None = None,
        switch_beta_frequency: int | None = None,
        hard_switch: bool | None = None,
        meta_controller_temperature: float = 1.,
        control_signal_multiplier: float = 1.,
        metacontroller_residual_stream_dropout: float = 0.,
        discovery_num_samples: int | None = None,
        discovery_softmin_temp: float | None = None,
        return_loss_breakdown: bool = False,
        return_meta_controller_output: bool = False
    ):
        assert exists(state) ^ exists(residual_stream), 'either `state` or `residual_stream` must be passed in'
        assert exists(actions), '`actions` must be provided'

        model, meta_controller = self.model, self.meta_controller

        seq_len = residual_stream.shape[1] if exists(residual_stream) else state.shape[1]
        assert seq_len > 1, f'sequence length ({seq_len}) must be greater than 1 for next latent prediction'

        action_loss_mask = maybe(lens_to_mask)(episode_lens, seq_len)

        # 1. lower body (if residual_stream not already given)

        if not exists(residual_stream):
            past_actions = actions[:, :-1] if actions.shape[1] == seq_len else actions

            with torch.no_grad():
                state_embed = model.state_embed(state)
                action_embed = 0.

                if exists(past_actions) and exists(model.action_embed):
                    action_embed = model.action_embed(past_actions)

                if is_tensor(action_embed) and action_embed.shape[1] == (seq_len - 1):
                    action_embed = pad_left_at_dim(action_embed, 1, dim = 1)

                residual_stream = model.lower_body(
                    state_embed + action_embed,
                    condition = condition
                )

        # 2. meta controller discovery pass

        num_samples = default(discovery_num_samples, getattr(meta_controller, 'discovery_num_samples', 1)) if meta_controller.training else 1
        has_multi_samples = num_samples > 1
        repeat_batch = partial(repeat, pattern = 'b ... -> (n b) ...', n = num_samples) if has_multi_samples else identity

        control_signal, meta_output = meta_controller(
            residual_stream,
            discovery_phase = True,
            temperature = meta_controller_temperature,
            episode_lens = episode_lens,
            ablate_switch_beta = ablate_switch_beta,
            switch_beta_frequency = switch_beta_frequency,
            hard_switch = hard_switch,
            num_samples = num_samples
        )

        if control_signal_multiplier != 1.:
            control_signal = control_signal * control_signal_multiplier

        residual_stream = F.dropout(residual_stream, p = metacontroller_residual_stream_dropout, training = meta_controller.training)

        modified_residual_stream = repeat_batch(residual_stream) + control_signal

        # 3. upper body (with final rmsnorm) and action readout

        attended = model.upper_body(
            modified_residual_stream,
            condition = maybe(repeat_batch)(condition)
        )

        dist_params = model.action_readout(attended)

        action_recon_loss = model.action_readout.calculate_loss(
            dist_params,
            repeat_batch(actions),
            mask = maybe(repeat_batch)(action_loss_mask),
            return_unreduced_loss = has_multi_samples
        )

        # 4. 1-step next latent prediction (conditioned on metacontroller action) and rollout kl loss

        latent_actions = default(meta_output.gated_actions if self.use_gated_action else None, meta_output.actions)

        curr_latent, target_latent = attended[:, :-1], attended[:, 1:].detach()
        transition_mask = maybe(repeat_batch)(action_loss_mask[:, 1:]) if exists(action_loss_mask) else None

        dynamics_out = self.dynamics_model(
            next_token_embeds = latent_actions[:, :-1],
            curr_latent = curr_latent
        )

        next_latent_loss = self.zero
        latent_loss_unreduced = self.zero

        if self.has_next_latent_loss:
            latent_loss_unreduced = self.loss_fn(dynamics_out, target_latent)
            latent_loss_unreduced = reduce(latent_loss_unreduced, 'b n d -> b n', 'mean')
            next_latent_loss = masked_mean(latent_loss_unreduced, transition_mask)

        kl_loss = self.zero
        kl_loss_unreduced = self.zero

        if self.has_kl_loss:
            frozen_action_readout = deepcopy(model.action_readout)
            frozen_action_readout.requires_grad_(False)

            with torch.no_grad():
                target_dist = frozen_action_readout(target_latent)

            pred_dist = frozen_action_readout(dynamics_out)

            kl_loss_unreduced = frozen_action_readout.kl_div(target_dist, pred_dist)

            if kl_loss_unreduced.ndim == 3:
                kl_loss_unreduced = reduce(kl_loss_unreduced, 'b n d -> b n', 'sum')

            kl_loss = masked_mean(kl_loss_unreduced, transition_mask)

        # 5. reduce discovery samples

        nlp_loss_unreduced = (
            self.next_latent_loss_weight * latent_loss_unreduced +
            self.kl_loss_weight * kl_loss_unreduced
        )

        discovery_losses, reduced_meta_output = meta_controller.reduce_discovery_samples(
            meta_output,
            action_recon_loss,
            state_pred_losses = nlp_loss_unreduced if has_multi_samples else None,
            episode_lens = episode_lens,
            softmin_temp = discovery_softmin_temp,
            return_meta_controller_output = True
        )

        total_loss = (
            discovery_losses.action_recon +
            discovery_losses.kl +
            discovery_losses.ratio +
            self.next_latent_loss_weight * next_latent_loss +
            self.kl_loss_weight * kl_loss
        )

        losses = Losses(
            discovery_losses.action_recon,
            next_latent_loss,
            kl_loss,
            discovery_losses.kl,
            discovery_losses.ratio
        )

        ret = (
            total_loss,
            losses if return_loss_breakdown else None,
            reduced_meta_output if return_meta_controller_output else None
        )

        ret = compact(ret)
        return ret[0] if len(ret) == 1 else ret
