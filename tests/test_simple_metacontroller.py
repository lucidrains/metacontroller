import tempfile
import shutil
from functools import partial

import torch
from torch import cat

from memmap_replay_buffer import ReplayBuffer
from metacontroller.simple_metacontroller import TransformerWithMetacontroller, extract_grpo_data, policy_loss, z_score

# helpers

def exists(v):
    return v is not None

MODEL_KWARGS = dict(
    dim = 128,
    state_embed_readout = dict(num_discrete = 256),
    action_embed_readout = dict(num_discrete = 256),
    dim_latent = 32,
    lower_body = dict(depth = 1, heads = 4, attn_dim_head = 16),
    upper_body = dict(depth = 1, heads = 4, attn_dim_head = 16),
    emitter_decoder = dict(depth = 1, heads = 4, attn_dim_head = 16),
    dim_queries_keys = 64,
    target_avg_token_length = 4.,
    pred_loss_to_switch_weight = 0.5,
)

def test_caching_parity():
    model = TransformerWithMetacontroller(**MODEL_KWARGS)
    model.eval()

    batch_size = 2
    seq_len = 16

    with torch.no_grad():
        x = torch.randint(0, 256, (batch_size, seq_len))

        # deterministic sampling for parity

        model.latent_readout.sample = lambda dist, **kwargs: dist.unbind(dim = -1)[0]

        state = x[:, :-1]
        actions = x[:, 1:]

        # parallel

        dist_params_parallel, meta_output_parallel = model(
            state = state,
            actions = actions,
            return_loss = False
        )

        # sequential

        dist_params_seq = []
        switch_betas_seq = []
        action_dist_seq = []

        cache = None

        for i in range(seq_len - 1):
            dist_params, meta_output, cache = model(
                state = state[:, i:i+1],
                actions = actions[:, i:i+1],
                cache = cache,
                return_loss = False,
                return_cache = True
            )

            dist_params_seq.append(dist_params)
            switch_betas_seq.append(meta_output.switch_beta)
            action_dist_seq.append(meta_output.action_dist)

        dist_params_seq = cat(dist_params_seq, dim = 1)
        switch_betas_seq = cat(switch_betas_seq, dim = 1)
        action_dist_seq = cat(action_dist_seq, dim = 1)

        assert torch.allclose(dist_params_parallel, dist_params_seq, atol = 1e-5)
        assert torch.allclose(meta_output_parallel.switch_beta, switch_betas_seq, atol = 1e-5)
        assert torch.allclose(meta_output_parallel.action_dist, action_dist_seq, atol = 1e-5)

def test_grpo_parity():
    seq_len = 16
    num_rollouts = 3

    model = TransformerWithMetacontroller(**MODEL_KWARGS)

    test_folder = tempfile.mkdtemp()

    try:
        replay_buffer = ReplayBuffer(
            test_folder,
            max_episodes = num_rollouts,
            max_timesteps = seq_len * 2,
            circular = True,
            fields = model.replay_buffer_field_dict,
            meta_fields = dict(advantages = 'float')
        )

        one_state = torch.randint(0, 256, (1, seq_len))
        actions = torch.randint(0, 256, (1, seq_len))

        all_episodes = []
        all_rewards = []

        # simulate rollouts

        with torch.no_grad():
            for _ in range(num_rollouts):
                cache = None
                grpo_data_list = []

                for i in range(seq_len - 1):
                    _, meta_output, cache = model(
                        state = one_state[:, i:i+1],
                        actions = actions[:, i:i+1],
                        cache = cache,
                        return_loss = False,
                        return_cache = True
                    )

                    grpo_data_list.append(extract_grpo_data(model, meta_output))

                states, latent_actions, log_probs, switch_betas = zip(*grpo_data_list)

                all_episodes.append((
                    cat(states, dim = 1),
                    cat(log_probs, dim = 1),
                    cat(switch_betas, dim = 1),
                    cat(latent_actions, dim = 1)
                ))

                all_rewards.append(torch.randn(1))

        # verify parallel log probs match sequential

        rewards = cat(all_rewards)
        group_advantages = z_score(rewards)

        group_states, group_log_probs, group_switch_betas, group_latent_actions = map(partial(cat, dim = 0), zip(*all_episodes))

        parallel_action_dist = model.get_action_dist_for_internal_rl(group_states)
        parallel_log_probs = model.log_prob(parallel_action_dist, group_latent_actions)

        assert torch.allclose(parallel_log_probs, group_log_probs, atol = 1e-4)

        # verify policy loss backward

        loss = policy_loss(
            model,
            group_states,
            group_log_probs,
            group_latent_actions,
            group_advantages
        )

        loss.backward()

    finally:
        shutil.rmtree(test_folder)
