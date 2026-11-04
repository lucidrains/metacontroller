# /// script
# dependencies = [
#   "fire",
#   "numpy",
#   "scipy",
#   "torch>=2.5",
#   "einops>=0.8.1",
#   "torch-einops-utils>=0.2.4",
#   "metacontroller-pytorch>=0.4.0"
# ]
# ///

"""
End-to-end 4-stage verification for MetaController (Kobayashi et al., 2025):
"Emergent temporal abstractions in autoregressive models enable hierarchical reinforcement learning"

Evaluates all 4 stages on a 5x5 GridWorld PinPad benchmark:
  - Stage 1: Autoregressive Behavioral Cloning (next-action + next-latent prediction),
             mid-depth residual stream linear subgoal probing (Paper Fig. 3c),
             and unsupervised subgoal switch detection via base-model predictive surprise.
  - Stage 2: Ground-truth low-rank hypernetwork subgoal controllers (Paper Fig. 4).
  - Stage 3: Unsupervised MetaController discovery (Paper Figs. 5-7), comparing:
             (a) Continuous Gaussian VAE (`latent_readout = 'continuous'`),
             (b) Fleuret BinaryMapper (`MetaControllerWithBinaryMapper`, `dim_code_bits = 2`),
             (c) Built-in Discrete Prototype Bottleneck (`latent_readout = 'discrete'`, `hard_prob = 0.5`),
             (d) Custom `LatentReadout` subclass (`CosinePrototypeReadout` — cosine-routed prototypes),
             plus inspection of switching frequency and parsed action trajectories.
  - Stage 4: Sparse-reward RL on unseen compositional 4-subgoal POST_TASK = [0, 1, 2, 3]:
             Raw-Action GRPO vs. Internal RL GRPO over Stage 3 discovered latent controllers,
             tracking GRPO switching frequency and decoding the learned meta-policy plan.
"""

import time
from collections import deque
import fire
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange, repeat
from scipy.optimize import linear_sum_assignment

from torch_einops_utils import lens_to_mask, z_score
from torch_einops_utils.shape import shape, size

from metacontroller import (
    Transformer,
    MetaController,
    MetaControllerWithBinaryMapper,
    LatentReadout,
    categorical_entropy
)
from metacontroller.metacontroller import (
    exists,
    l2norm,
    sample_straight_through,
    categorical_entropy_loss
)

# constants

G, C, TMAX = 5, 4, 35
OBS_DIM = G * G * (C + 1) + G * G + C * 8 + 4 * (C + 2)
ACTS = [(0, -1), (0, 1), (-1, 0), (1, 0)]
ACT_NAMES = ['L', 'R', 'U', 'D']
PAD_NAMES = {
    0: 'Pad 0 (Top-Left)',
    1: 'Pad 1 (Top-Right)',
    2: 'Pad 2 (Bottom-Left)',
    3: 'Pad 3 (Bottom-Right)'
}

# 16 3-subgoal pretraining tasks (0 -> 1 transition and length-4 sequences held out)

PRETRAIN_TASKS = [
    (0, 3, 2), (1, 0, 3), (2, 1, 0), (3, 2, 1),
    (0, 2, 0), (0, 2, 1), (0, 3, 1), (1, 0, 2),
    (1, 3, 1), (1, 3, 2), (2, 0, 2), (2, 0, 3),
    (2, 1, 3), (3, 1, 0), (3, 1, 3), (3, 2, 0),
]
POST_TASK = [0, 1, 2, 3]

QUADRANTS = [
    [(0, 0), (0, 1), (1, 0), (1, 1)],
    [(0, 3), (0, 4), (1, 3), (1, 4)],
    [(3, 0), (4, 0), (3, 1), (4, 1)],
    [(3, 3), (3, 4), (4, 3), (4, 4)],
]
ALL_CELLS = [(r, c) for r in range(G) for c in range(G)]

# environment

def sample_cells(rng):
    cells = [QUADRANTS[g][int(rng.integers(4))] for g in range(C)]
    free = [c for c in ALL_CELLS if c not in set(cells)]
    cells.append(free[int(rng.integers(len(free)))])
    return cells

class FastPinPadEnv:
    def __init__(self, rng):
        self.rng = rng
        self.reset([0])

    def reset(self, task):
        self.task = list(task)
        self.grid = -np.ones((G, G), dtype = np.int64)
        cells = sample_cells(self.rng)
        self.pad_coords = {}
        self.pad_coords_arr = np.zeros((C, 2), dtype = np.int64)
        self.static_oh = np.zeros((G * G, C + 1), dtype = np.float32)
        self.static_oh[:, 0] = 1.0
        for g in range(C):
            r, col = cells[g]
            self.grid[r, col] = g
            self.pad_coords[g] = (r, col)
            self.pad_coords_arr[g] = [r, col]
            self.static_oh[r * G + col, 0] = 0.0
            self.static_oh[r * G + col, g + 1] = 1.0
        self.static_oh_flat = self.static_oh.ravel()
        self.agent = cells[C]
        self.idx, self.steps, self.done, self.reward = 0, 0, False, 0.0
        return self.obs()

    def obs(self):
        ar, ac = self.agent
        out = np.zeros(OBS_DIM, dtype = np.float32)
        base = G * G * (C + 1)
        out[:base] = self.static_oh_flat
        out[base + ar * G + ac] = 1.0
        offset = base + G * G
        for g in range(C):
            dr, dc = self.pad_coords_arr[g, 0] - ar, self.pad_coords_arr[g, 1] - ac
            out[offset + g * 8 : offset + (g + 1) * 8] = [
                dr / G, dc / G, float(dc < 0), float(dc > 0),
                float(dr < 0), float(dr > 0), float(abs(dr) > abs(dc)), float(dr == 0 and dc == 0)
            ]
        offset += C * 8
        for d_idx, (di, dj) in enumerate(ACTS):
            nr, nc = ar + di, ac + dj
            if not (0 <= nr < G and 0 <= nc < G):
                out[offset + d_idx * (C + 2)] = 1.0
            else:
                gv = int(self.grid[nr, nc])
                out[offset + d_idx * (C + 2) + (1 if gv < 0 else gv + 2)] = 1.0
        return out

    def step(self, a):
        self.steps += 1
        di, dj = ACTS[a]
        ni, nj = self.agent[0] + di, self.agent[1] + dj
        if 0 <= ni < G and 0 <= nj < G:
            self.agent = (ni, nj)
        c = self.grid[self.agent]
        if c >= 0:
            if c == self.task[self.idx]:
                self.idx += 1
                if self.idx == len(self.task):
                    self.done, self.reward = True, 1.0
            else:
                self.done = True
        if self.steps >= TMAX:
            self.done = True
        return self.obs(), self.reward, self.done

    def bfs_action(self):
        target = self.task[self.idx]
        tcell = self.pad_coords[target]
        forbidden = {coord for g, coord in self.pad_coords.items() if g != target}
        dr, dc = tcell[0] - self.agent[0], tcell[1] - self.agent[1]
        order = [(-1, 0), (1, 0), (0, -1), (0, 1)] if abs(dr) > abs(dc) else [(0, -1), (0, 1), (-1, 0), (1, 0)]
        dist, par, q = {self.agent: 0}, {}, deque([self.agent])
        while q:
            node = q.popleft()
            if node == tcell:
                break
            for di, dj in order:
                nb = (node[0] + di, node[1] + dj)
                if 0 <= nb[0] < G and 0 <= nb[1] < G and nb not in forbidden and nb not in dist:
                    dist[nb], par[nb] = dist[node] + 1, node
                    q.append(nb)
        if tcell not in dist:
            return None
        node = tcell
        while node != self.agent:
            first, node = node, par[node]
        return ACTS.index((first[0] - self.agent[0], first[1] - self.agent[1]))

class BatchPinPadEnv:
    def __init__(self, tasks, rng, any_pretrain_task = False):
        self.tasks = np.asarray(tasks, dtype = np.int64)
        self.B, self.K = self.tasks.shape
        self.any_pretrain_task = any_pretrain_task
        self.pretrain_set = set(PRETRAIN_TASKS)
        self.grids = -np.ones((self.B, G, G), dtype = np.int64)
        self.agents = np.zeros((self.B, 2), dtype = np.int64)
        self.pad_coords_arr = np.zeros((self.B, C, 2), dtype = np.int64)
        self.static_oh = np.zeros((self.B, G * G * (C + 1)), dtype = np.float32)

        for b in range(self.B):
            cells = sample_cells(rng)
            oh = np.zeros((G * G, C + 1), dtype = np.float32)
            oh[:, 0] = 1.0
            for g in range(C):
                r, col = cells[g]
                self.grids[b, r, col] = g
                self.pad_coords_arr[b, g] = [r, col]
                oh[r * G + col, 0], oh[r * G + col, g + 1] = 0.0, 1.0
            self.static_oh[b] = oh.ravel()
            self.agents[b] = cells[C]

        self.idx = np.zeros(self.B, dtype = np.int64)
        self.visited = [[] for _ in range(self.B)]
        self.steps = np.zeros(self.B, dtype = np.int64)
        self.done = np.zeros(self.B, dtype = bool)
        self.reward = np.zeros(self.B, dtype = np.float32)

    def obs_tensor(self):
        out = np.zeros((self.B, OBS_DIM), dtype = np.float32)
        base = G * G * (C + 1)
        out[:, :base] = self.static_oh
        ar, ac = self.agents[:, 0], self.agents[:, 1]
        b_idx = np.arange(self.B)
        out[b_idx, base + ar * G + ac] = 1.0
        offset = base + G * G
        for g in range(C):
            dr, dc = self.pad_coords_arr[:, g, 0] - ar, self.pad_coords_arr[:, g, 1] - ac
            out[:, offset + g * 8 + 0] = dr / G
            out[:, offset + g * 8 + 1] = dc / G
            out[:, offset + g * 8 + 2] = (dc < 0).astype(np.float32)
            out[:, offset + g * 8 + 3] = (dc > 0).astype(np.float32)
            out[:, offset + g * 8 + 4] = (dr < 0).astype(np.float32)
            out[:, offset + g * 8 + 5] = (dr > 0).astype(np.float32)
            out[:, offset + g * 8 + 6] = (np.abs(dr) > np.abs(dc)).astype(np.float32)
            out[:, offset + g * 8 + 7] = ((dr == 0) & (dc == 0)).astype(np.float32)
        offset += C * 8
        for d_idx, (di, dj) in enumerate(ACTS):
            nr, nc = ar + di, ac + dj
            oob = (nr < 0) | (nr >= G) | (nc < 0) | (nc >= G)
            out[b_idx[oob], offset + d_idx * (C + 2)] = 1.0
            inb = ~oob
            if np.any(inb):
                gv = self.grids[b_idx[inb], nr[inb], nc[inb]]
                out[b_idx[inb], offset + d_idx * (C + 2) + np.where(gv < 0, 1, gv + 2)] = 1.0
        return torch.from_numpy(out)

    def step(self, actions_np):
        active = ~self.done
        if not np.any(active):
            return
        self.steps[active] += 1
        nxt = self.agents + np.array(ACTS, dtype = np.int64)[actions_np]
        moved = (nxt[:, 0] >= 0) & (nxt[:, 0] < G) & (nxt[:, 1] >= 0) & (nxt[:, 1] < G) & active
        self.agents[moved] = nxt[moved]

        b_idx = np.where(moved)[0]
        cells = self.grids[b_idx, self.agents[b_idx, 0], self.agents[b_idx, 1]]
        for b, c in zip(b_idx[cells >= 0], cells[cells >= 0]):
            c = int(c)
            if self.any_pretrain_task:
                self.visited[b].append(c)
                self.idx[b] = len(self.visited[b])
                if not any(t[:len(self.visited[b])] == tuple(self.visited[b]) for t in self.pretrain_set):
                    self.done[b] = True
                elif len(self.visited[b]) == 3:
                    self.done[b], self.reward[b] = True, 1.0
            else:
                if c == self.tasks[b, self.idx[b]]:
                    self.idx[b] += 1
                    if self.idx[b] == self.K:
                        self.done[b], self.reward[b] = True, 1.0
                else:
                    self.done[b] = True
        self.done[self.steps >= TMAX] = True

def make_dataset(tasks, n_eps, seed):
    rng = np.random.default_rng(seed)
    env = FastPinPadEnv(rng)
    obs_l, act_l, sg_l, switch_l, pos_l, len_l = [], [], [], [], [], []
    max_t = 0
    while len(obs_l) < n_eps:
        env.reset(tasks[len(obs_l) % len(tasks)])
        o_ep, a_ep, g_ep, switch_ep, p_ep = [], [], [], [], []
        prev_idx, pos_in_seg, ok = -1, 0, True
        while not env.done:
            a = env.bfs_action()
            if a is None:
                ok = False
                break
            is_switch = 1.0 if env.idx != prev_idx else 0.0
            if is_switch:
                pos_in_seg, prev_idx = 0, env.idx
            o_ep.append(env.obs())
            a_ep.append(a)
            g_ep.append(env.task[env.idx])
            switch_ep.append(is_switch)
            p_ep.append(pos_in_seg)
            pos_in_seg += 1
            env.step(a)
        if not ok or env.reward < 1.0:
            continue
        max_t = max(max_t, len(o_ep))
        obs_l.append(o_ep)
        act_l.append(a_ep)
        sg_l.append(g_ep)
        switch_l.append(switch_ep)
        pos_l.append(p_ep)
        len_l.append(len(o_ep))

    N = len(obs_l)
    obs_t = torch.zeros(N, max_t, OBS_DIM)
    act_t = torch.zeros(N, max_t, dtype = torch.long)
    sg_t = torch.full((N, max_t), -1, dtype = torch.long)
    switch_t = torch.zeros(N, max_t)
    pos_t = torch.full((N, max_t), -1, dtype = torch.long)
    lens_t = torch.tensor(len_l, dtype = torch.long)
    for i, L in enumerate(len_l):
        obs_t[i, :L] = torch.from_numpy(np.array(obs_l[i]))
        act_t[i, :L] = torch.tensor(act_l[i])
        sg_t[i, :L] = torch.tensor(sg_l[i])
        switch_t[i, :L] = torch.tensor(switch_l[i])
        pos_t[i, :L] = torch.tensor(pos_l[i])
    return obs_t, act_t, sg_t, switch_t, pos_t, lens_t

def compute_unsupervised_switches(logits, acts, thresh = 0.15):
    probs = F.softmax(logits, dim = -1)
    ent = categorical_entropy(probs)
    dent = torch.zeros_like(ent)
    dent[:, 1:] = ent[:, 1:] - ent[:, :-1]
    p_prev = torch.zeros_like(ent)
    p_prev[:, 1:] = probs[:, 1:].gather(-1, acts[:, :-1].unsqueeze(-1)).squeeze(-1)
    switch = ((dent - 0.5 * p_prev) > thresh).float()
    switch[:, 0] = 1.0
    for t in range(1, size(switch, 'b [t]')):
        switch[:, t] = switch[:, t] * (1.0 - switch[:, t - 1])
    return switch

def compute_switch_f1(pred_switch, gt_switch_m, gt_int_m):
    tp = (pred_switch & gt_switch_m).float().sum().item()
    fp = (pred_switch & gt_int_m).float().sum().item()
    fn = ((~pred_switch) & gt_switch_m).float().sum().item()
    prec = tp / max(tp + fp, 1.0)
    rec = tp / max(tp + fn, 1.0)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    return prec, rec, f1

def compute_cluster_acc(pred_ids, gt_ids, num_classes = C):
    conf = torch.zeros(num_classes, num_classes)
    for p_id in range(num_classes):
        for g_id in range(num_classes):
            conf[p_id, g_id] = ((pred_ids == p_id) & (gt_ids == g_id)).float().sum()
    row_ind, col_ind = linear_sum_assignment(-conf.numpy())
    acc = conf[row_ind, col_ind].sum().item() / conf.sum().clamp(min = 1.0).item()
    proto_to_sg = {int(r): int(c) for r, c in zip(row_ind, col_ind)}
    return acc, proto_to_sg

# custom sequence embedder, emitter, and alternative LatentReadout example

class SegmentTerminalEmbedder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        self.current_switch = None

    def forward(self, x, mask = None, episode_lens = None, **kwargs):
        B, T, D = shape(x, 'b t d')
        out = torch.zeros(B, T, D * 2, device = x.device, dtype = x.dtype)
        switch = self.current_switch

        for b in range(B):
            L = int(episode_lens[b]) if exists(episode_lens) else T
            switch_idx = [t for t in range(L) if switch[b, t] > 0.5] + [L]
            for i in range(len(switch_idx) - 1):
                t0, t1 = switch_idx[i], switch_idx[i + 1]
                out[b, t0:t1] = torch.cat([x[b, t0:t1].mean(dim = 0), x[b, min(t1, L - 1)]], dim = -1)

        return self.net(out)

class SummaryOnlyEmitter(nn.Module):
    def __init__(self, dim, dim_summary):
        super().__init__()
        self.dim_skip = dim * 2
        self.mlp = nn.Sequential(
            nn.Linear(dim_summary, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim * 2)
        )

    def forward(self, x):
        return self.mlp(x[..., self.dim_skip:])

class CosinePrototypeReadout(LatentReadout):
    """
    Example custom `LatentReadout` implementation:
    Computes routing logits via scaled cosine similarity between L2-normalized projected
    features and prototypes, while passing the un-normalized prototype vectors `self.prototypes`
    into the latent stream via probabilistic straight-through (`hard_prob`).
    """
    dim_log_prob = 1

    def __init__(
        self,
        dim,
        dim_latent,
        num_discrete = 4,
        hard_prob = 0.5,
        scale = 4.5,
        sample_entropy_weight = 0.1,
        shared_prototypes = None
    ):
        super().__init__()
        self.dim_latent = dim_latent
        self.num_discrete = num_discrete
        self.hard_prob = hard_prob
        self.scale = scale
        self.sample_entropy_weight = sample_entropy_weight
        self.proj = nn.Linear(dim, dim_latent, bias = False)

        if exists(shared_prototypes):
            self._prototypes = None
            self._shared = (shared_prototypes,)
        else:
            self._prototypes = nn.Parameter(torch.empty(num_discrete, dim_latent))
            nn.init.orthogonal_(self._prototypes, gain = (dim_latent ** 0.5) * 0.5)
            self._shared = None

    @property
    def prototypes(self):
        return self._shared[0] if exists(self._shared) else self._prototypes

    def forward(self, x):
        return einsum(l2norm(self.proj(x)), l2norm(self.prototypes), '... d, k d -> ... k') * self.scale

    def sample(self, logits, temperature = 1.0, differentiable = True):
        probs = (logits / temperature).softmax(dim = -1)
        chosen = sample_straight_through(probs, self.hard_prob, training = self.training and differentiable)
        return einsum(chosen, self.prototypes, '... k, k d -> ... d')

    def calculate_kl_loss(self, logits, mask = None):
        return categorical_entropy_loss(logits.softmax(dim = -1), mask = mask, sample_entropy_weight = self.sample_entropy_weight)

    def log_prob(self, logits, sampled_latent_action):
        idx = torch.cdist(sampled_latent_action, self.prototypes).argmin(dim = -1)
        return F.log_softmax(logits, dim = -1).gather(-1, rearrange(idx, '... -> ... 1'))

    def clone_for_proposer(self, dim_proposer):
        return CosinePrototypeReadout(
            dim_proposer,
            self.dim_latent,
            self.num_discrete,
            self.hard_prob,
            self.scale,
            self.sample_entropy_weight,
            shared_prototypes = self.prototypes
        )

# main benchmark

def discovery_step_loss(
    mc,
    model,
    residual,
    actions,
    mask,
    episode_lens,
    num_samples = 1,
    dropout = 0.,
    **mc_kwargs
):
    control, meta_out = mc(
        residual,
        discovery_phase = True,
        episode_lens = episode_lens,
        num_samples = num_samples,
        **mc_kwargs
    )

    residual = F.dropout(residual, p = dropout, training = dropout > 0.)

    def repeat_batch(t):
        return repeat(t, 'b ... -> (n b) ...', n = num_samples)

    logits = model.action_readout(model.upper_body(repeat_batch(residual) + control))
    recon_loss = model.action_readout.calculate_loss(
        logits,
        repeat_batch(actions),
        mask = repeat_batch(mask),
        return_unreduced_loss = num_samples > 1
    )

    losses = mc.reduce_discovery_samples(meta_out, recon_loss, episode_lens = episode_lens)
    return losses.action_recon + losses.kl + losses.ratio

def train(
    num_train_eps: int = 2400,
    num_val_eps: int = 300,
    dim: int = 64,
    dim_latent: int = 8,
    rank: int = 16,
    bc_steps: int = 350,
    discovery_steps: int = 350,
    switch_steps: int = 450,
    grpo_iters: int = 35,
    grpo_batch_size: int = 256,
    discovery_num_samples: int = 4
):
    t0 = time.time()
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 92)
    print("MetaController 4-Stage Verification & Ablation Benchmark (5x5 GridWorld PinPad)")
    print("=" * 92)

    tr_obs, tr_act, tr_sg, tr_switch, tr_pos, tr_lens = make_dataset(PRETRAIN_TASKS, num_train_eps, seed = 0)
    va_obs, va_act, va_sg, va_switch, va_pos, va_lens = make_dataset(PRETRAIN_TASKS, num_val_eps, seed = 1)
    tr_mask = lens_to_mask(tr_lens, size(tr_obs, 'b [t] d'))
    va_mask = lens_to_mask(va_lens, size(va_obs, 'b [t] d'))
    va_valid_t1 = va_mask.clone()
    va_valid_t1[:, 0] = False
    gt_switch_m = (va_switch > 0.5) & va_valid_t1
    gt_int_m = (va_switch <= 0.5) & va_valid_t1
    gt_switch_freq = va_switch[va_mask].mean().item()

    # ============================================================================
    # STAGE 1: Autoregressive Behavioral Cloning & Residual Stream Linear Probe
    # ============================================================================
    model = Transformer(
        dim = dim,
        state_embed_readout = dict(num_continuous = OBS_DIM),
        action_embed_readout = dict(num_discrete = 4),
        lower_body = dict(depth = 2, attn_dim_head = 16, heads = 4),
        upper_body = dict(depth = 2, attn_dim_head = 16, heads = 4),
        next_latent_loss_weight = 0.1,
    )
    opt_bc = torch.optim.AdamW(model.parameters(), lr = 2e-3, weight_decay = 1e-4)
    for _ in range(bc_steps):
        idx = torch.randint(0, size(tr_obs, '[b] t d'), (64,))
        state_loss, act_loss = model(tr_obs[idx], tr_act[idx], episode_lens = tr_lens[idx], force_behavior_cloning = True)
        opt_bc.zero_grad()
        (state_loss + act_loss).backward()
        opt_bc.step()

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    with torch.no_grad():
        (_, val_bc_loss), val_logits, va_res = model(
            va_obs, va_act, episode_lens = va_lens,
            force_behavior_cloning = True, return_action_logits = True, return_residual_stream = True
        )
        _, tr_res = model(tr_obs, tr_act, episode_lens = tr_lens, force_behavior_cloning = True, return_residual_stream = True)
        tr_logits = model.action_readout(model.upper_body(tr_res))
        val_acc = (val_logits.argmax(-1)[va_mask] == va_act[va_mask]).float().mean().item()

        def eval_base_model(tasks_eval, any_pretrain = False, n_eval = 160, seed = 100):
            rng = np.random.default_rng(seed)
            benv = BatchPinPadEnv([tasks_eval[i % len(tasks_eval)] for i in range(n_eval)], rng, any_pretrain_task = any_pretrain)
            cache_h, past_act = None, None
            for _ in range(TMAX):
                if np.all(benv.done):
                    break
                o = benv.obs_tensor().unsqueeze(1)
                logits, cache_h = model(o, past_act, cache = cache_h, force_behavior_cloning = True, return_cache = True)
                a = torch.distributions.Categorical(logits = logits[:, 0]).sample()
                benv.step(a.numpy())
                past_act = a.unsqueeze(1)
            return float(benv.reward.mean()), float(benv.idx.mean())

        pre_sr, pre_sg = eval_base_model(PRETRAIN_TASKS, any_pretrain = True)
        post_sr, post_sg = eval_base_model([POST_TASK], any_pretrain = False)

    probe = nn.Linear(dim, C)
    opt_p = torch.optim.AdamW(probe.parameters(), lr = 1e-2, weight_decay = 1e-4)
    X_tr, y_tr = tr_res[tr_sg >= 0], tr_sg[tr_sg >= 0]
    X_va, y_va, p_va = va_res[va_sg >= 0], va_sg[va_sg >= 0], va_pos[va_sg >= 0]
    for _ in range(200):
        loss_p = F.cross_entropy(probe(X_tr), y_tr)
        opt_p.zero_grad()
        loss_p.backward()
        opt_p.step()

    with torch.no_grad():
        preds_va = probe(X_va).argmax(-1)
        probe_acc = (preds_va == y_va).float().mean().item()
        by_pos = " ".join(f"step_{k}:{(preds_va[p_va == k] == y_va[p_va == k]).float().mean().item():.2f}" for k in range(5))
        va_unsup_switch = compute_unsupervised_switches(val_logits, va_act, thresh = 0.15)
        switch_p, switch_r, switch_f1 = compute_switch_f1(va_unsup_switch > 0.5, gt_switch_m, gt_int_m)

    print(f"\n[Stage 1] Autoregressive Behavioral Cloning & Emergent Subgoal Representations ({time.time()-t0:.1f}s)")
    print(f"  Base BC Model        : val_loss={val_bc_loss.item():.3f}, val_acc={val_acc:.3f}")
    print(f"  Closed-Loop Rollout  : Pretrain SR={pre_sr:.3f} ({pre_sg:.2f}/3) | Unseen POST_TASK SR={post_sr:.3f} ({post_sg:.2f}/4)")
    print(f"  Linear Subgoal Probe : overall_acc={probe_acc:.3f} (chance=0.250) | {by_pos}")
    print(f"  Unsup Base Surprise  : Switch Prec={switch_p:.3f}, Rec={switch_r:.3f}, F1={switch_f1:.3f} | Switch Freq={va_unsup_switch[va_mask].mean().item():.3f} (GT={gt_switch_freq:.3f})")

    # ============================================================================
    # STAGE 2: Ground-Truth Low-Rank Hypernetwork Subgoal Controllers
    # ============================================================================
    class GTSubgoalController(nn.Module):
        def __init__(self, dim, rank = 16, n_subgoals = 4):
            super().__init__()
            self.w1 = nn.Parameter(torch.randn(n_subgoals, dim, rank) * 0.02)
            self.w2 = nn.Parameter(torch.zeros(n_subgoals, dim, rank))

        def forward(self, residual_stream, subgoal_ids):
            safe_sg = subgoal_ids.clamp(min = 0)
            hyper_w = einsum(self.w1[safe_sg], self.w2[safe_sg], '... i r, ... j r -> ... i j')
            return einsum(residual_stream, hyper_w, '... i, ... i j -> ... j')

    torch.manual_seed(42)
    ctrl_mod = GTSubgoalController(dim, rank = rank, n_subgoals = C)
    opt_c = torch.optim.AdamW(ctrl_mod.parameters(), lr = 1e-2, weight_decay = 1e-4)
    for _ in range(300):
        idx = torch.randint(0, size(tr_res, '[b] t d'), (64,))
        res, sg = tr_res[idx], tr_sg[idx]
        att = model.upper_body(F.dropout(res, p = 0.25, training = True) + ctrl_mod(res, sg))
        loss_c = model.action_readout.calculate_loss(model.action_readout(att), tr_act[idx], mask = tr_mask[idx])
        opt_c.zero_grad()
        loss_c.backward()
        opt_c.step()

    ctrl_mod.eval()
    with torch.no_grad():
        logits_va = model.action_readout(model.upper_body(va_res + ctrl_mod(va_res, va_sg)))
        acc_s2 = (logits_va.argmax(-1)[va_mask] == va_act[va_mask]).float().mean().item()
        loss_s2 = model.action_readout.calculate_loss(logits_va, va_act, mask = va_mask).item()
        res_s2 = {}
        for reset_kv in [False, True]:
            rng = np.random.default_rng(200)
            benv = BatchPinPadEnv([POST_TASK] * 160, rng)
            lower_h, upper_h, past_act = None, None, None
            for _ in range(TMAX):
                if np.all(benv.done):
                    break
                o = benv.obs_tensor().unsqueeze(1)
                s_emb = model.state_embed(o)
                a_emb = model.action_embed(past_act) if (exists(past_act) and not reset_kv) else 0.0
                res_t, lower_h = model.lower_body(s_emb + a_emb, cache = (None if reset_kv else lower_h), return_hiddens = True)
                active_sg = torch.from_numpy(benv.tasks[np.arange(160), np.minimum(benv.idx, 3)]).unsqueeze(1)
                att_t, upper_h = model.upper_body(res_t + ctrl_mod(res_t, active_sg), cache = (None if reset_kv else upper_h), return_hiddens = True)
                a = model.action_readout(att_t)[:, 0].argmax(-1)
                benv.step(a.numpy())
                past_act = a.unsqueeze(1)
            res_s2[reset_kv] = (float(benv.reward.mean()), float(benv.idx.mean()))

    print(f"\n[Stage 2] Ground-Truth Low-Rank Hypernetwork Subgoal Controllers ({time.time()-t0:.1f}s)")
    print(f"  Hypernetwork Steering : val_loss={loss_s2:.3f}, val_acc={acc_s2:.3f} | POST_TASK SR={res_s2[False][0]:.3f} ({res_s2[False][1]:.2f}/4) | KV-reset SR={res_s2[True][0]:.3f}")

    # ============================================================================
    # STAGE 3: Unsupervised MetaController Discovery
    # ============================================================================
    print(f"\n[Stage 3] Unsupervised MetaController Discovery")

    # 3A: Continuous Gaussian Discovery (`latent_readout = 'continuous'`)
    torch.manual_seed(42)
    mc_cont = MetaController(
        dim_model = dim, dim_meta_controller = dim, dim_latent = dim_latent,
        hypernetwork_low_rank = rank, dim_sequence_summary_embed = 32,
        latent_readout = 'continuous', kl_loss_weight = 0.05, switch_temperature = 0.5,
        discovery_num_samples = discovery_num_samples
    )
    opt_cont = torch.optim.AdamW(mc_cont.discovery_parameters(), lr = 2e-3, weight_decay = 1e-4)
    for _ in range(150):
        idx = torch.randint(0, size(tr_res, '[b] t d'), (64,))
        b_res, b_a, b_m = tr_res[idx], tr_act[idx], tr_mask[idx]
        loss_v = discovery_step_loss(mc_cont, model, b_res, b_a, b_m, tr_lens[idx], discovery_num_samples)
        opt_cont.zero_grad()
        loss_v.backward()
        opt_cont.step()

    mc_cont.eval()
    with torch.no_grad():
        ctrl_v, m_va_v = mc_cont(va_res, discovery_phase = True, episode_lens = va_lens)
        logits_v = model.action_readout(model.upper_body(va_res + ctrl_v))
        va_acc_v = (logits_v.argmax(-1)[va_mask] == va_act[va_mask]).float().mean().item()
        va_loss_v = model.action_readout.calculate_loss(logits_v, va_act, mask = va_mask).item()
        _, _, f1_v = compute_switch_f1(m_va_v.switch_beta > 0.5, gt_switch_m, gt_int_m)
    print(f"  (a) Continuous Gaussian (`latent_readout='continuous'`) : val_loss={va_loss_v:.3f}, val_acc={va_acc_v:.3f} | Switch F1={f1_v:.3f}")

    # 3B: MetaControllerWithBinaryMapper (Fleuret BinaryMapper, 2 bits -> 4 codes)
    torch.manual_seed(44)
    emb_bm = SegmentTerminalEmbedder(dim = dim)
    mc_bm = MetaControllerWithBinaryMapper(
        dim_model = dim, dim_meta_controller = dim, dim_code_bits = 2,
        hypernetwork_low_rank = rank, dim_sequence_summary_embed = dim,
        internal_sequence_embedder = emb_bm, kl_loss_weight = 0.5, switch_temperature = 0.1,
        hard_prob = 0.5, binary_temperature = 0.2, use_marginal_entropy_loss = True,
        sample_entropy_weight = 0.1, gate_kl_with_switch_beta = True,
        discovery_num_samples = discovery_num_samples
    )
    mc_bm.binary_mapper.deterministic_on_eval = True
    mc_bm.emitter = SummaryOnlyEmitter(dim, dim)
    nn.init.orthogonal_(mc_bm.emitter_to_binary_logits.weight)
    nn.init.zeros_(mc_bm.emitter_to_binary_logits.bias)
    proto_bm = nn.Linear(mc_bm.num_codes, dim_latent, bias = False)
    nn.init.orthogonal_(proto_bm.weight, gain = (dim_latent ** 0.5) * 0.5)
    dec_bm = nn.Linear(dim_latent, 2 * rank * dim)
    nn.init.normal_(dec_bm.weight, std = 0.02)
    with torch.no_grad():
        dec_bm.bias.zero_()
        dec_bm.bias[:rank * dim].normal_(0.0, 0.2)
    mc_bm.decoder = nn.Sequential(proto_bm, dec_bm)

    opt_bm = torch.optim.AdamW(mc_bm.discovery_parameters(), lr = 3e-3, weight_decay = 1e-4)
    for _ in range(discovery_steps):
        idx = torch.randint(0, size(tr_res, '[b] t d'), (64,))
        b_res, b_a, b_m, b_switch = tr_res[idx], tr_act[idx], tr_mask[idx], tr_switch[idx]
        emb_bm.current_switch = b_switch
        loss = discovery_step_loss(
            mc_bm, model, b_res, b_a, b_m, tr_lens[idx], discovery_num_samples,
            dropout = 0.2, ablate_switch_beta = b_switch, hard_switch = True
        )
        opt_bm.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mc_bm.parameters(), 1.0)
        opt_bm.step()

    mc_bm.eval()
    with torch.no_grad():
        emb_bm.current_switch = va_switch
        ctrl_bm, m_va_bm = mc_bm(va_res, discovery_phase = True, ablate_switch_beta = va_switch, hard_switch = True, episode_lens = va_lens)
        logits_bm = model.action_readout(model.upper_body(va_res + ctrl_bm))
        va_acc_bm = (logits_bm.argmax(-1)[va_mask] == va_act[va_mask]).float().mean().item()
        va_loss_bm = model.action_readout.calculate_loss(logits_bm, va_act, mask = va_mask).item()
        switch_va_m = (va_switch > 0.5) & va_mask
        pred_bm = m_va_bm.actions[switch_va_m].argmax(dim = -1)
        acc_bm_cluster, bm_proto_to_sg = compute_cluster_acc(pred_bm, va_sg[switch_va_m])
    print(f"  (b) Fleuret BinaryMapper (`MetaControllerWithBinaryMapper`) : val_loss={va_loss_bm:.3f}, val_acc={va_acc_bm:.3f} | 4-Subgoal Cluster Acc={acc_bm_cluster:.3f}")

    # Helper to train & evaluate built-in 'discrete' and custom `LatentReadout` on Stage 3
    def run_stage3_readout(label, readout_fn):
        torch.manual_seed(43)
        embedder = SegmentTerminalEmbedder(dim = dim)
        mc = MetaController(
            dim_model = dim, dim_meta_controller = dim, dim_latent = dim_latent,
            hypernetwork_low_rank = rank, dim_sequence_summary_embed = dim,
            internal_sequence_embedder = embedder,
            latent_readout = readout_fn(),
            num_discrete_latents = C,
            discrete_hard_prob = 0.5,
            discrete_temperature = 0.2,
            discrete_sample_entropy_weight = 0.1,
            kl_loss_weight = 0.5,
            switch_temperature = 0.1,
            gate_kl_with_switch_beta = True,
            discovery_num_samples = discovery_num_samples
        )
        mc.emitter = SummaryOnlyEmitter(dim, dim)
        mc.decoder = nn.Linear(dim_latent, 2 * rank * dim)
        nn.init.normal_(mc.decoder.weight, std = 0.02)
        with torch.no_grad():
            mc.decoder.bias.zero_()
            mc.decoder.bias[:rank * dim].normal_(0.0, 0.2)

        opt_ed = torch.optim.AdamW(mc.discovery_parameters_non_switching(), lr = 3e-3, weight_decay = 1e-4)
        for _ in range(discovery_steps):
            idx = torch.randint(0, size(tr_res, '[b] t d'), (64,))
            b_res, b_a, b_m, b_switch = tr_res[idx], tr_act[idx], tr_mask[idx], tr_switch[idx]
            embedder.current_switch = b_switch
            loss = discovery_step_loss(
                mc, model, b_res, b_a, b_m, tr_lens[idx], discovery_num_samples,
                dropout = 0.2, ablate_switch_beta = b_switch, hard_switch = True
            )
            opt_ed.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mc.parameters(), 1.0)
            opt_ed.step()

        # Train causal switching_unit (summary_gru + model_to_meta + switching_unit)
        opt_switch = torch.optim.AdamW(
            list(mc.summary_gru.parameters()) + list(mc.model_to_meta.parameters()) + mc.discovery_parameters_switching_unit(),
            lr = 3e-3, weight_decay = 1e-4
        )
        pos_w = (tr_switch[tr_mask] == 0).float().sum() / (tr_switch[tr_mask] == 1).float().sum().clamp(min = 1.0)
        for _ in range(switch_steps):
            idx = torch.randint(0, size(tr_res, '[b] t d'), (64,))
            b_res, b_tswitch, b_m = tr_res[idx], tr_switch[idx], tr_mask[idx]
            embedder.current_switch = b_tswitch
            _, m_switch = mc(b_res, discovery_phase = True, hard_switch = False, episode_lens = tr_lens[idx])
            switch_b = m_switch.switch_beta.clamp(1e-5, 1.0 - 1e-5)
            w = torch.where(b_tswitch > 0.5, pos_w * 0.55, 1.0) * b_m.float()
            loss_switch = F.binary_cross_entropy(switch_b, b_tswitch, weight = w)
            opt_switch.zero_grad()
            loss_switch.backward()
            opt_switch.step()
            with torch.no_grad():
                mc.switching_unit.weight_ih_l0[:, -dim_latent:].zero_()

        mc.eval()
        for p in mc.parameters():
            p.requires_grad_(False)

        with torch.no_grad():
            embedder.current_switch = va_switch
            ctrl_va, m_va = mc(va_res, discovery_phase = True, ablate_switch_beta = va_switch, hard_switch = True, episode_lens = va_lens)
            logits_va = model.action_readout(model.upper_body(va_res + ctrl_va))
            va_acc_e = (logits_va.argmax(-1)[va_mask] == va_act[va_mask]).float().mean().item()
            va_loss_e = model.action_readout.calculate_loss(logits_va, va_act, mask = va_mask).item()

            _, m_va_switch = mc(va_res, discovery_phase = True, hard_switch = False, episode_lens = va_lens)
            raw_switch = (m_va_switch.switch_beta > 0.5).float()
            raw_switch[:, 0] = 1.0
            for t in range(1, size(raw_switch, 'b [t]')):
                raw_switch[:, t] = raw_switch[:, t] * (1.0 - raw_switch[:, t - 1])
            pred_switch = (raw_switch > 0.5) & va_mask
            _, _, f1_e = compute_switch_f1(pred_switch & va_valid_t1, gt_switch_m, gt_int_m)
            switch_freq_e = pred_switch[va_mask].float().mean().item()

            active_proto = m_va.action_dist[switch_va_m].argmax(dim = -1)
            cluster_acc, proto_to_sg = compute_cluster_acc(active_proto, va_sg[switch_va_m])

        print(f"  {label:55s} : val_loss={va_loss_e:.3f}, val_acc={va_acc_e:.3f} | Switch F1={f1_e:.3f} (Freq={switch_freq_e:.3f} vs GT={gt_switch_freq:.3f}) | 4-Subgoal Cluster Acc={cluster_acc:.3f}")
        return mc, embedder, pred_switch, proto_to_sg, va_acc_e, f1_e, switch_freq_e, cluster_acc

    mc_disc, emb_disc, pred_switch_disc, disc_proto_to_sg, va_acc_d, f1_d, freq_d, cluster_d = run_stage3_readout(
        "(c) Built-in Discrete (`latent_readout='discrete'`)",
        lambda: 'discrete'
    )
    mc_cos, _, _, _, va_acc_w, f1_w, freq_w, cluster_w = run_stage3_readout(
        "(d) Custom Cosine (`CosinePrototypeReadout`)",
        lambda: CosinePrototypeReadout(dim * 2, dim_latent, num_discrete = C, hard_prob = 0.5, sample_entropy_weight = 0.1)
    )

    # Qualitative inspection: parsed actions and subgoals gated by MetaController's learned switch mask
    with torch.no_grad():
        emb_disc.current_switch = pred_switch_disc.float()
        _, m_own = mc_disc(va_res, discovery_phase = True, ablate_switch_beta = pred_switch_disc.float(), hard_switch = True, episode_lens = va_lens)
        forget_gate = 1.0 - m_own.switch_beta
        gated_actions = mc_disc.switch_gating(forget_gate, m_own.actions * m_own.switch_beta.unsqueeze(-1))
        proto_ids_gated = torch.cdist(gated_actions, mc_disc.emitter_to_action_mean_log_var.prototypes).argmin(dim = -1)

        print(f"\n  [Parsed Validation Trajectories (Switch Freq={freq_d:.3f} vs. GT={gt_switch_freq:.3f})]")
        for ep in (0, 2):
            L = int(va_lens[ep])
            acts_str = [ACT_NAMES[int(a)] for a in va_act[ep, :L]]
            gt_sg_list = [int(g) for g in va_sg[ep, :L]]
            gt_switch_list = [int(s) for s in va_switch[ep, :L]]
            pr_switch_list = [int(s) for s in pred_switch_disc[ep, :L]]
            mapped_sg = [disc_proto_to_sg[int(p)] for p in proto_ids_gated[ep, :L]]

            segs_gt, segs_pred = [], []
            cur = []
            for t in range(L):
                if t > 0 and gt_switch_list[t]:
                    segs_gt.append(f"sg{gt_sg_list[t - 1]}:[{''.join(cur)}]")
                    cur = []
                cur.append(acts_str[t])
            segs_gt.append(f"sg{gt_sg_list[L - 1]}:[{''.join(cur)}]")

            cur = []
            for t in range(L):
                if t > 0 and pr_switch_list[t]:
                    segs_pred.append(f"z{proto_ids_gated[ep, t - 1].item()}->sg{mapped_sg[t - 1]}:[{''.join(cur)}]")
                    cur = []
                cur.append(acts_str[t])
            segs_pred.append(f"z{proto_ids_gated[ep, L - 1].item()}->sg{mapped_sg[L - 1]}:[{''.join(cur)}]")

            print(f"    Episode {ep} (len={L}) | GT Switch: {gt_switch_list} | Pred Switch: {pr_switch_list}")
            print(f"      GT Parsed   : {' -> '.join(segs_gt)}")
            print(f"      Pred Parsed : {' -> '.join(segs_pred)}")

    # ============================================================================
    # STAGE 4: Sparse-Reward RL on Unseen POST_TASK = [0, 1, 2, 3]
    # ============================================================================
    print(f"\n[Stage 4] Sparse-Reward RL on Unseen 4-Subgoal POST_TASK = [0, 1, 2, 3] ({time.time()-t0:.1f}s)")

    # 4A: Raw-Action GRPO on Base Model
    raw_head = nn.Linear(dim, 4)
    with torch.no_grad():
        raw_head.weight.copy_(model.action_readout.embeddings.weight)
        raw_head.bias.zero_()
    opt_raw = torch.optim.AdamW(raw_head.parameters(), lr = 1e-2)
    rng_raw = np.random.default_rng(400)
    raw_sr_init, raw_sr_final = 0.0, 0.0
    for it in range(1, grpo_iters + 1):
        benv = BatchPinPadEnv([POST_TASK] * grpo_batch_size, rng_raw)
        lower_h, upper_h, past_act = None, None, None
        ep_lp = torch.zeros(grpo_batch_size)
        for _ in range(TMAX):
            if np.all(benv.done):
                break
            active_m = torch.from_numpy(~benv.done)
            o = benv.obs_tensor().unsqueeze(1)
            with torch.no_grad():
                s_emb = model.state_embed(o)
                a_emb = model.action_embed(past_act) if exists(past_act) else 0.0
                res_t, lower_h = model.lower_body(s_emb + a_emb, cache = lower_h, return_hiddens = True)
                att_t, upper_h = model.upper_body(res_t, cache = upper_h, return_hiddens = True)
            dist = torch.distributions.Categorical(logits = raw_head(att_t[:, 0]))
            a = dist.sample()
            ep_lp = ep_lp + dist.log_prob(a) * active_m.float()
            benv.step(a.numpy())
            past_act = a.unsqueeze(1)
        rewards = torch.from_numpy(benv.reward)
        if it == 1:
            raw_sr_init = rewards.mean().item()
        raw_sr_final = rewards.mean().item()
        if rewards.std() > 1e-5:
            loss_raw = -(ep_lp * z_score(rewards)).mean()
            opt_raw.zero_grad()
            loss_raw.backward()
            opt_raw.step()
        if it in (1, 15, grpo_iters):
            print(f"  Raw-Action GRPO  iter {it:2d} | Sparse POST_TASK SR={rewards.mean().item():.3f} | subgoals={benv.idx.mean():.2f}/4")

    # 4B: Internal RL GRPO over Stage 3 Discovered Discrete Controllers
    meta_proposer = nn.Parameter(torch.zeros(4, C))
    opt_int = torch.optim.AdamW([meta_proposer], lr = 1e-1)
    with torch.no_grad():
        w1_all, w2_all = mc_bm.to_hyper_network_weights(mc_bm.decoder(torch.eye(C)))
        hyper_w_all = einsum(w1_all, w2_all, 'k i r, k j r -> k i j')

    rng_int = np.random.default_rng(400)
    int_sr_init, int_sr_final = 0.0, 0.0
    grpo_switch_freq_final = 0.0
    pad_feat_offset = G * G * (C + 1) + G * G
    for it in range(1, grpo_iters + 1):
        benv = BatchPinPadEnv([POST_TASK] * grpo_batch_size, rng_int)
        dist_meta = torch.distributions.Categorical(logits = meta_proposer.unsqueeze(0).expand(grpo_batch_size, 4, C))
        sampled_k = dist_meta.sample()
        ep_lp = dist_meta.log_prob(sampled_k).sum(dim = -1)
        obs_switch_count = torch.zeros(grpo_batch_size, dtype = torch.long)
        total_switches = torch.zeros(grpo_batch_size)
        total_active_steps = torch.zeros(grpo_batch_size)

        with torch.no_grad():
            for t in range(TMAX):
                if np.all(benv.done):
                    break
                active_m = torch.from_numpy(~benv.done)
                o_flat = benv.obs_tensor()
                if t == 0:
                    switch_now = torch.ones(grpo_batch_size, dtype = torch.bool)
                else:
                    switch_now = torch.zeros(grpo_batch_size, dtype = torch.bool)
                    for g in range(C):
                        switch_now |= (o_flat[:, pad_feat_offset + g * 8 + 7] > 0.5)
                    obs_switch_count = (obs_switch_count + switch_now.long()).clamp(max = 3)

                total_switches += (switch_now & active_m).float()
                total_active_steps += active_m.float()

                s_emb = model.state_embed(o_flat.unsqueeze(1))
                res_t, _ = model.lower_body(s_emb, cache = None, return_hiddens = True)
                k_active = sampled_k[torch.arange(grpo_batch_size), obs_switch_count]
                hw = hyper_w_all[k_active].unsqueeze(1)
                c_sig = einsum(res_t, hw, 'b t i, b t i j -> b t j')
                att_t, _ = model.upper_body(res_t + c_sig, cache = None, return_hiddens = True)
                a = model.action_readout(att_t)[:, 0].argmax(-1)
                benv.step(a.numpy())

        rewards = torch.from_numpy(benv.reward)
        grpo_switch_freq = (total_switches.sum() / total_active_steps.sum().clamp(min = 1.0)).item()
        switches_per_ep = total_switches.mean().item()
        if it == 1:
            int_sr_init = rewards.mean().item()
        int_sr_final = rewards.mean().item()
        grpo_switch_freq_final = grpo_switch_freq
        if rewards.std() > 1e-5:
            loss_int = -(ep_lp * z_score(rewards)).mean()
            opt_int.zero_grad()
            loss_int.backward()
            opt_int.step()
        if it in (1, 15, grpo_iters):
            print(f"  Internal RL GRPO iter {it:2d} | Sparse POST_TASK SR={rewards.mean().item():.3f} | subgoals={benv.idx.mean():.2f}/4 | switch_freq={grpo_switch_freq:.3f} ({switches_per_ep:.2f} switches/ep)")

    # Interpret what the MetaController has learned after GRPO
    with torch.no_grad():
        probs = meta_proposer.softmax(dim = -1)
        print("\n  [Post-GRPO Learned Meta-Policy Interpretation on Unseen POST_TASK = [0, 1, 2, 3]]")
        learned_seq = []
        for slot in range(4):
            best_z = int(probs[slot].argmax().item())
            conf = float(probs[slot, best_z].item())
            mapped_pad = bm_proto_to_sg[best_z]
            learned_seq.append(mapped_pad)
            print(f"    Subgoal Slot {slot + 1} (Target: {PAD_NAMES[POST_TASK[slot]]}) -> selects Latent Code z_{best_z} (prob={conf:.2f}) -> steers to {PAD_NAMES[mapped_pad]}")
        print(f"    Decoded Compositional Plan: {learned_seq} vs. Ground-Truth POST_TASK: {POST_TASK} (Exact Match: {learned_seq == POST_TASK})")

    print("\n" + "=" * 92)
    print(f"FINAL SUMMARY TABLE (Total Runtime: {time.time()-t0:.1f}s)")
    print("=" * 92)
    print(f"1. Stage 1 Base Model & Linear Probe : val_acc={val_acc:.3f}, subgoal_probe_acc={probe_acc:.3f} ({by_pos})")
    print(f"2. Stage 2 Hypernetwork Controllers  : val_acc={acc_s2:.3f}, POST_TASK SR={res_s2[False][0]:.3f}, KV-reset SR={res_s2[True][0]:.3f}")
    print(f"3. Stage 3 Unsupervised Discovery    : Switch F1={f1_d:.3f} (Freq={freq_d:.3f} vs GT={gt_switch_freq:.3f}) | Continuous F1={f1_v:.3f} | BinaryMapper Acc={acc_bm_cluster:.3f} | Discrete Acc={cluster_d:.3f} | CosinePrototype Acc={cluster_w:.3f}")
    print(f"4. Stage 4 Unseen POST_TASK RL (GRPO): Raw-Action RL SR={raw_sr_init:.3f}->{raw_sr_final:.3f} vs. Internal RL SR={int_sr_init:.3f}->{int_sr_final:.3f} (GRPO Switch Freq={grpo_switch_freq_final:.3f}, Plan={learned_seq})")
    print("=" * 92)

if __name__ == '__main__':
    fire.Fire(train)
