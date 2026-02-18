import pytest
param = pytest.mark.parametrize

from shutil import rmtree
from pathlib import Path
from functools import partial

import torch
from torch import nn, cat
from metacontroller.metacontroller import Transformer, MetaController, ActionProposerWrapper, policy_loss, z_score, extract_grpo_data
from metacontroller.metacontroller_with_binary_mapper import MetaControllerWithBinaryMapper

from torch_einops_utils.save_load import save_load

from memmap_replay_buffer import ReplayBuffer

from einops import rearrange, repeat

# functions

def exists(v):
    return v is not None

# test

@param('accept_condition', (False, True))
@param('action_discrete', (False, True))
@param('embed_past_actions', (False, True))
@param('variable_length', (False, True))
@param('normalize_state_action_losses', (False, True))
@param('variant', (False, True))
def test_metacontroller(
    variant,
    action_discrete,
    embed_past_actions,
    variable_length,
    accept_condition,
    normalize_state_action_losses
):
    use_binary_mapper_variant = variant
    switching_unit_type = 'gru'

    dim_model = 32
    dim_meta = 16
    seq_len = 16

    state = torch.randn(2, seq_len, 384)
    episode_lens = torch.tensor([64, 64]) if variable_length else None

    if action_discrete:
        actions = torch.randint(0, 4, (2, seq_len))
        action_embed_readout = dict(num_discrete = 4)
        assert_shape = (4,)
    else:
        actions = torch.randn(2, seq_len, 8)
        action_embed_readout = dict(num_continuous = 8)
        assert_shape = (8, 2)

    # maybe conditioning

    condition = None
    condition_kwargs = dict()

    if accept_condition:
        condition_kwargs = dict(
            dim_condition = 384,
        )

        condition = torch.randn(2, 384)

    # behavioral cloning phase

    model = Transformer(
        dim = dim_model,
        action_embed_readout = action_embed_readout,
        state_embed_readout = dict(num_continuous = 384),
        lower_body = dict(depth = 1, attn_dim_head = 16, heads = 2),
        upper_body = dict(depth = 1, attn_dim_head = 16, heads = 2),
        embed_past_actions = embed_past_actions,
        normalize_state_action_losses = normalize_state_action_losses,
        **condition_kwargs
    )

    assert exists(model.running_bc_state_loss) == normalize_state_action_losses
    assert exists(model.running_bc_action_loss) == normalize_state_action_losses
    assert exists(model.running_discovery_state_loss) == normalize_state_action_losses
    assert exists(model.running_discovery_action_loss) == normalize_state_action_losses

    state_clone_loss, action_clone_loss = model(state, actions, condition = condition, episode_lens = episode_lens)
    (state_clone_loss + 0.5 * action_clone_loss).backward()

    # discovery and internal rl phase with meta controller

    action_proposer_kwargs = dict()
    action_proposer_kwargs['action_proposer'] = dict(depth = 1, attn_dim_head = 16, heads = 2)

    if not use_binary_mapper_variant:
        meta_controller = MetaController(
            dim_model = dim_model,
            dim_meta_controller = dim_meta,
            dim_latent = 64,
            internal_sequence_embedder = dict(attn_dim_head = 16, heads = 2, depth = 1),
            **action_proposer_kwargs
        )
    else:
        meta_controller = MetaControllerWithBinaryMapper(
            dim_model = dim_model,
            dim_meta_controller = dim_meta,
            dim_code_bits = 8,
            internal_sequence_embedder = dict(attn_dim_head = 16, heads = 2, depth = 1),
            **action_proposer_kwargs
        )

    # discovery phase

    (_, action_recon_loss, kl_loss, ratio_loss) = model(state, actions, condition = condition, meta_controller = meta_controller, discovery_phase = True, episode_lens = episode_lens)
    (action_recon_loss + kl_loss * 0.1 + ratio_loss * 0.1).backward()

    # internal rl - done iteratively

    # replay buffer

    test_folder = './test-buffer-for-grpo'

    replay_buffer = ReplayBuffer(
        test_folder,
        max_episodes = 3,
        max_timesteps = 256,
        circular = True,
        fields = meta_controller.replay_buffer_field_dict,
        meta_fields = dict(
            advantages = 'float'
        )
    )

    # simulate grpo

    all_episodes = []
    all_rewards = []

    one_state = state[:1]
    one_condition = condition[:1] if exists(condition) else None

    for _ in range(3): # group of 3

        cache = None
        past_action_id = None

        grpo_data_list = []

        for timestep_state in one_state.unbind(dim = 1):
            timestep_state = rearrange(timestep_state, 'b d -> b 1 d')

            logits, cache = model(timestep_state, past_action_id, condition = one_condition, meta_controller = meta_controller, cache = cache, return_cache = True)

            past_action_id = model.action_readout.sample(logits)

            # extract grpo data and store

            grpo_data = extract_grpo_data(meta_controller, cache)
            grpo_data_list.append(grpo_data)

        # accumulate across time for the episode data

        states, actions, log_probs, switch_betas = zip(*grpo_data_list)

        all_episodes.append((
            cat(states, dim = 1),
            cat(log_probs, dim = 1),
            cat(switch_betas, dim = 1),
            cat(actions, dim = 1)
        ))

        all_rewards.append(torch.randn(1))

    # calculate advantages using z-score

    rewards = cat(all_rewards)
    group_advantages = z_score(rewards)

    assert group_advantages.shape == (3,)

    # simulate a policy loss update over the entire group

    group_states, group_log_probs, group_switch_betas, group_latent_actions = map(partial(cat, dim = 0), zip(*all_episodes))
    
    # parallel verification

    parallel_action_dist = meta_controller.get_action_dist_for_internal_rl(group_states)
    parallel_log_probs = meta_controller.log_prob(parallel_action_dist, group_latent_actions)

    assert torch.allclose(parallel_log_probs, group_log_probs, atol = 1e-5), 'parallel log probs do not match stored log probs'

    for states, log_probs, switch_betas, latent_actions, advantages in zip(group_states, group_log_probs, group_switch_betas, group_latent_actions, group_advantages):
        replay_buffer.store_episode(
            states = states,
            log_probs = log_probs,
            switch_betas = switch_betas,
            latent_actions = latent_actions,
            advantages = advantages
        )

    dl = replay_buffer.dataloader(batch_size = 3)

    batch = next(iter(dl))

    loss = meta_controller.policy_loss(
        batch['states'],
        batch['log_probs'],
        batch['latent_actions'],
        batch['advantages'],
        batch['switch_betas'] == 1.,
        episode_lens = batch['_lens']
    )

    loss.backward()

    # evolutionary strategies over grpo

    model.meta_controller = meta_controller
    model.evolve(1, lambda _: 1., noise_population_size = 2)

    # saving and loading

    meta_controller.save('./meta_controller.pt')

    meta_controller_klass = meta_controller.__class__
    rehydrated_meta_controller = meta_controller_klass.init_and_load('./meta_controller.pt')

    model.save('./trained.pt')

    rehydrated_model = Transformer.init_and_load('./trained.pt', strict = False)

    Path('./meta_controller.pt').unlink()
    Path('./trained.pt').unlink()

    rmtree(test_folder, ignore_errors = True)

def test_kl_loss_warmup_e2e():
    dim_model = 64
    kl_loss_weight = 0.2
    kl_loss_warmup_steps = 10
    
    mc = MetaController(
        dim_model = dim_model,
        kl_loss_weight = kl_loss_weight,
        kl_loss_warmup_steps = kl_loss_warmup_steps
    )
    
    transformer = Transformer(
        dim = dim_model,
        state_embed_readout = dict(num_continuous = dim_model),
        action_embed_readout = dict(num_continuous = dim_model),
        lower_body = dict(depth = 1),
        upper_body = dict(depth = 1),
        meta_controller = mc
    )
    
    # Step 0
    state = torch.randn(1, 2, dim_model)
    actions = torch.randn(1, 2, dim_model)
    
    _, output = transformer(state, actions, discovery_phase = True, return_meta_controller_output = True)
    assert output.kl_loss_weight == 0.0
    assert output.kl_loss == 0.0
    
    # Step 5
    for _ in range(5):
        transformer.meta_controller_maybe_increment_kl_loss_step()
        
    assert transformer.meta_controller_current_kl_loss_weight == 0.1
    
    _, output5 = transformer(state, actions, discovery_phase = True, return_meta_controller_output = True)
    assert output5.kl_loss_weight == 0.1
    
    # Step 10
    for _ in range(5):
        transformer.meta_controller_maybe_increment_kl_loss_step()

    assert transformer.meta_controller_current_kl_loss_weight == 0.2

    _, output10 = transformer(state, actions, discovery_phase = True, return_meta_controller_output = True)
    assert output10.kl_loss_weight == 0.2

    # verify scaling (kl_loss should be doubled if weight is doubled)
    if output5.kl_loss > 0:
        assert torch.isclose(output10.kl_loss / output5.kl_loss, torch.tensor(2.0), atol = 1e-4)

    # test reset
    transformer.meta_controller_reset_kl_loss_warmup()
    assert transformer.meta_controller_current_kl_loss_weight == 0.0
    _, output_reset = transformer(state, actions, discovery_phase = True, return_meta_controller_output = True)
    assert output_reset.kl_loss_weight == 0.0
    assert output_reset.kl_loss == 0.0

    # test transformer accessor
    assert transformer.meta_controller_kl_loss_weight == 0.2

    # test apply_kl_loss_weight = False
    mc_no_weight = MetaController(
        dim_model = dim_model,
        kl_loss_weight = 0.2,
        kl_loss_warmup_steps = 10,
        apply_kl_loss_weight = False
    )
    
    transformer_no_weight = Transformer(
        dim = dim_model,
        state_embed_readout = dict(num_continuous = dim_model),
        action_embed_readout = dict(num_continuous = dim_model),
        lower_body = dict(depth = 1),
        upper_body = dict(depth = 1),
        meta_controller = mc_no_weight
    )

    _, output_no_weight = transformer_no_weight(state, actions, discovery_phase = True, return_meta_controller_output = True)
    # weight shouldn't be applied despite step 0 (which would be weight 0)
    assert output_no_weight.kl_loss_weight == 1.0
    assert output_no_weight.kl_loss > 0.0

def test_transformer_embed_parity():
    dim_model = 512
    dim_meta = 256
    dim_latent = 128
    seq_len = 10
    batch = 1

    model = Transformer(
        dim = dim_model,
        action_embed_readout = dict(num_continuous = 8),
        state_embed_readout = dict(num_continuous = 384),
        lower_body = dict(depth = 1),
        upper_body = dict(depth = 1),
        meta_controller = MetaController(
            dim_model = dim_model,
            dim_meta_controller = dim_meta,
            dim_latent = dim_latent
        )
    )

    state = torch.randn(batch, seq_len, 384)
    actions = torch.randn(batch, seq_len, 8)

    bc_embeds = model(state, actions, force_behavior_cloning = True, return_embed = True)

    discovery_embeds = model(state, actions, discovery_phase = True, return_embed = True)

    sequential_embeds = []

    states = state.unbind(dim = 1)
    past_actions = [None, *actions[:, :-1].unbind(dim = 1)]

    for t_state, t_past_action in zip(states, past_actions):
        t_state = rearrange(t_state, 'b d -> b 1 d')

        if exists(t_past_action):
            t_past_action = rearrange(t_past_action, 'b d -> b 1 d')

        embed_t = model(
            t_state,
            actions = t_past_action,
            return_embed = True
        )

        sequential_embeds.append(embed_t)

    sequential_embeds = cat(sequential_embeds, dim = 1)

    assert torch.allclose(bc_embeds, discovery_embeds, atol = 1e-6)
    assert torch.allclose(bc_embeds, sequential_embeds, atol = 1e-6)

def test_transformer_bc_parity():
    dim_model = 512
    dim_meta = 256
    dim_latent = 128
    seq_len = 10
    batch = 1

    model = Transformer(
        dim = dim_model,
        action_embed_readout = dict(num_continuous = 8),
        state_embed_readout = dict(num_continuous = 384),
        lower_body = dict(depth = 1),
        upper_body = dict(depth = 1),
        meta_controller = MetaController(
            dim_model = dim_model,
            dim_meta_controller = dim_meta,
            dim_latent = dim_latent
        )
    )

    state = torch.randn(batch, seq_len, 384)
    actions = torch.randn(batch, seq_len, 8)

    # parallel forward
    _, parallel_logits = model(state, actions, force_behavior_cloning = True, return_action_logits = True)

    # sequential forward
    sequential_logits = []
    cache = None
    
    states = state.unbind(dim = 1)
    past_actions = [None, *actions[:, :-1].unbind(dim = 1)]

    for t_state, t_past_action in zip(states, past_actions):
        t_state = rearrange(t_state, 'b d -> b 1 d')
        if exists(t_past_action):
            t_past_action = rearrange(t_past_action, 'b d -> b 1 d')

        logits, cache = model(
            t_state,
            actions = t_past_action,
            force_behavior_cloning = True,
            return_cache = True,
            cache = cache
        )
        sequential_logits.append(logits)

    sequential_logits = torch.cat(sequential_logits, dim = 1)

    assert torch.allclose(parallel_logits, sequential_logits, atol = 1e-5)

def test_discovery_vs_bc_ablation_parity():
    dim_model = 512
    dim_meta = 256
    dim_latent = 128
    seq_len = 32
    batch = 2

    model = Transformer(
        dim = dim_model,
        action_embed_readout = dict(num_continuous = 8),
        state_embed_readout = dict(num_continuous = 384),
        lower_body = dict(depth = 1),
        upper_body = dict(depth = 1),
        meta_controller = MetaController(
            dim_model = dim_model,
            dim_meta_controller = dim_meta,
            dim_latent = dim_latent
        )
    )

    model.eval()

    state = torch.randn(batch, seq_len + 1, 384)
    actions = torch.randn(batch, seq_len + 1, 8)

    # BC phase losses
    with torch.no_grad():
        bc_losses = model(
            state,
            actions = actions,
            force_behavior_cloning = True
        )

    # Discovery phase losses with ablated control signal
    with torch.no_grad():
        discovery_losses, _ = model(
            state,
            actions = actions,
            discovery_phase = True,
            control_signal_multiplier = 0.,
            return_meta_controller_output = True
        )

    assert torch.allclose(bc_losses.state, discovery_losses.state_pred, atol = 1e-6)
    assert torch.allclose(bc_losses.action, discovery_losses.action_recon, atol = 1e-6)

def test_switch_ablation():
    dim = 64
    batch = 2
    seq_len = 8
    frequency = 4

    meta_controller = MetaController(
        dim_model = dim,
        dim_meta_controller = 32,
        dim_latent = 32
    )

    transformer = Transformer(
        dim = dim,
        state_embed_readout = dict(num_continuous = dim),
        action_embed_readout = dict(num_continuous = dim),
        lower_body = dict(depth = 1),
        upper_body = dict(depth = 1),
        meta_controller = meta_controller
    )

    # test helper
    ablate_switch_beta = MetaController.create_regular_switch_beta(batch, seq_len, frequency)
    expected = torch.zeros(batch, seq_len)
    expected[:, 3] = 1.
    expected[:, 7] = 1.
    assert torch.allclose(ablate_switch_beta, expected)

    # test transformer discovery ablation
    state = torch.randn(batch, seq_len, dim)
    actions = torch.randn(batch, seq_len, dim)
    
    losses, meta_output = transformer(
        state,
        actions = actions,
        discovery_phase = True,
        ablate_switch_beta = ablate_switch_beta,
        return_meta_controller_output = True
    )
    assert torch.allclose(meta_output.switch_beta, ablate_switch_beta)

    # test transformer frequency ablation
    losses, meta_output = transformer(
        state,
        actions = actions,
        discovery_phase = True,
        switch_beta_frequency = frequency,
        return_meta_controller_output = True
    )
    assert torch.allclose(meta_output.switch_beta, ablate_switch_beta)

    # test with cache
    out1, cache1 = transformer(torch.randn(batch, 1, dim), switch_beta_frequency = frequency, return_cache = True)
    assert torch.all(cache1.prev_hiddens.meta_controller.switch_beta == 0.)

    for _ in range(2):
        _, cache1 = transformer(torch.randn(batch, 1, dim), switch_beta_frequency = frequency, cache = cache1, return_cache = True)

    _, cache1 = transformer(torch.randn(batch, 1, dim), switch_beta_frequency = frequency, cache = cache1, return_cache = True)
    assert torch.all(cache1.prev_hiddens.meta_controller.switch_beta == 1.)



def test_sequential_selection_parallel_vs_iterative():
    """
    Validates that parallel discovery (full sequence) matches
    iterative cached discovery when sequential_latent_action_selection is on.
    """

    dim = 64
    seq_len = 8
    batch = 1
    dim_latent = 32
    dim_meta = 64

    mc = MetaController(
        dim_model = dim,
        dim_meta_controller = dim_meta,
        dim_latent = dim_latent
    )

    # force context-free components for bit-perfect parity

    class IdentityEmbedder(nn.Module):
        def forward(self, x, mask = None):
            return x

    class LinearEmitter(nn.Module):
        def __init__(self, dim_in, dim_out):
            super().__init__()
            self.linear = nn.Linear(dim_in, dim_out)
        def forward(self, x, h = None):
            return self.linear(x), None

    mc.internal_sequence_embedder = IdentityEmbedder()

    dim_emitter_in = dim_meta + dim + dim_latent
    mc.emitter = LinearEmitter(dim_emitter_in, dim_meta * 2)

    mc.eval()

    residual_stream = torch.randn(batch, seq_len, dim)

    # 1. Parallel discovery (full sequence, uses JAX scan or PyTorch loop)

    torch.manual_seed(42)
    with torch.no_grad():
        _, mc_out_parallel = mc(
            residual_stream,
            discovery_phase = True
        )

    # 2. Iterative cached discovery (step-by-step)

    iter_switch_betas = []
    cache = None
    torch.manual_seed(42)

    for t in range(seq_len):
        x_t = residual_stream[:, t:t+1]

        with torch.no_grad():
            _, out_t = mc(
                x_t,
                cache = cache,
                discovery_phase = True
            )

        cache = out_t
        iter_switch_betas.append(out_t.switch_beta)

    iter_switch_beta = torch.cat(iter_switch_betas, dim = 1)

    # compare switch betas

    assert torch.allclose(mc_out_parallel.switch_beta, iter_switch_beta, atol = 1e-5), \
        f'switch beta mismatch: max diff = {(mc_out_parallel.switch_beta - iter_switch_beta).abs().max().item()}'

def test_jax_pytorch_parity():
    from metacontroller.sequential_action_selection import (
        pytorch_sequential_action_selection,
        torch_jax_sequential_selection,
        HAS_JAX
    )
    
    if not HAS_JAX:
        pytest.skip("JAX not installed")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dim_model, dim_meta, dim_latent, seq_len, batch = 64, 32, 16, 8, 2

    gru = nn.GRU(dim_model + dim_meta + dim_latent, dim_meta, batch_first=True).to(device)
    to_beta = nn.Linear(dim_meta, 1, bias = False).to(device)

    rs = torch.randn(batch, seq_len, dim_model, device = device)
    me = torch.randn(batch, seq_len, dim_meta, device = device)
    sla = torch.randn(batch, seq_len, dim_latent, device = device)
    h0 = torch.randn(1, batch, dim_meta, device = device)
    z0 = torch.randn(batch, 1, dim_latent, device = device)
    
    with torch.no_grad():
        pt_out = pytorch_sequential_action_selection(gru, to_beta, rs, me, sla, h0, z0, 1.0, False)
        
        jax_beta, jax_action, jax_hidden = torch_jax_sequential_selection(
            gru.weight_ih_l0, gru.weight_hh_l0, gru.bias_ih_l0, gru.bias_hh_l0,
            to_beta.weight, to_beta.bias, rs, me, sla, h0.squeeze(0), z0.squeeze(1), 1.0, False
        )

    assert torch.allclose(pt_out.switch_beta, jax_beta, atol = 1e-6)
    assert torch.allclose(pt_out.gated_action, jax_action, atol = 1e-6)
    assert torch.allclose(pt_out.next_switching_unit_gru_hidden, jax_hidden.unsqueeze(0), atol = 1e-6)

def test_compact_sequence_embedder():
    from metacontroller.compact_sequence_embedder import CompactSequenceEmbedder
    
    dim = 64
    seq_len = 10
    batch = 2
    
    embedder = CompactSequenceEmbedder(dim = dim)
    
    x = torch.randn(batch, seq_len, dim)
    episode_lens = torch.tensor([5, 10])
    
    out = embedder(x, episode_lens = episode_lens)
    
    assert out.shape == (batch, seq_len, dim)
    
    # manual check: last hidden of first sequence (len 5) should be repeated for out[0]
    _, h = embedder.gru(x[0:1, :5])
    expected0 = repeat(h[-1], '1 d -> n d', n = seq_len)
    
    assert torch.allclose(out[0], expected0, atol = 1e-6)
