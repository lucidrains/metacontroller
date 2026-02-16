import pytest
param = pytest.mark.parametrize

from shutil import rmtree
from pathlib import Path
from functools import partial

import torch
from torch import cat
from metacontroller.metacontroller import Transformer, MetaController, ActionProposerWrapper, policy_loss, z_score, extract_grpo_data
from metacontroller.metacontroller_with_binary_mapper import MetaControllerWithBinaryMapper
from minGRU_pytorch import minGRU

from memmap_replay_buffer import ReplayBuffer

from einops import rearrange

# functions

def exists(v):
    return v is not None

# test

@param('accept_condition', (False, True))
@param('action_discrete', (False, True))
@param('embed_past_actions', (False, True))
@param('variable_length', (False, True))
@param('use_mingru', (False, True))
@param('normalize_state_action_losses', (False, True))
@param('variant', [
    (False, 'qk'),
    (False, 'gru'),
    (True, 'qk'),
    (True, 'gru')
])
def test_metacontroller(
    variant,
    action_discrete,
    embed_past_actions,
    variable_length,
    use_mingru,
    accept_condition,
    normalize_state_action_losses
):
    use_binary_mapper_variant, switching_unit_type = variant
    dim_model = 512
    dim_meta = 256

    state = torch.randn(2, 128, 384)
    episode_lens = torch.tensor([64, 64]) if variable_length else None

    if action_discrete:
        actions = torch.randint(0, 4, (2, 128))
        action_embed_readout = dict(num_discrete = 4)
        assert_shape = (4,)
    else:
        actions = torch.randn(2, 128, 8)
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
        lower_body = dict(depth = 2,),
        upper_body = dict(depth = 2,),
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

    if use_mingru:
        action_proposer_kwargs = dict(
            action_proposer = ActionProposerWrapper(
                minGRU(dim = dim_model),
                cache_key = 'prev_hidden',
                return_cache_key = 'return_next_prev_hidden'
            )
        )

    if not use_binary_mapper_variant:
        meta_controller = MetaController(
            dim_model = dim_model,
            dim_meta_controller = dim_meta,
            dim_latent = 128,
            **action_proposer_kwargs
        )
    else:
        meta_controller = MetaControllerWithBinaryMapper(
            dim_model = dim_model,
            dim_meta_controller = dim_meta,
            dim_code_bits = 8,
            switching_unit_type = switching_unit_type,
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
