# /// script
# dependencies = [
#   "accelerate",
#   "fire",
#   "memmap-replay-buffer>=0.0.23",
#   "metacontroller-pytorch>=0.0.49",
#   "torch",
#   "einops",
#   "tqdm",
#   "wandb",
#   "gymnasium",
#   "minigrid",
#   "matplotlib"
# ]
# ///

import logging
import fire
from tqdm import tqdm
from pathlib import Path

logger = logging.getLogger(__name__)

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.nn import init
import torch.nn as nn

from accelerate import Accelerator
from memmap_replay_buffer import ReplayBuffer
from einops import rearrange

import matplotlib.pyplot as plt
import wandb

from metacontroller import MetaController, Transformer
from metacontroller.transformer_with_resnet import TransformerWithResnet

import gymnasium as gym
from gymnasium.envs.registration import register

# Import PinPad environment classes from gather_pinpad_trajs.py
from minigrid.wrappers import RGBImgObsWrapper
from gather_pinpad_trajs import PinPad, OneHotFullyObsWrapper

# ============================================================================
# Helpers
# ============================================================================

def initialize_weights_xavier(module):
    if isinstance(module, nn.Linear):
        init.xavier_uniform_(module.weight)
        if module.bias is not None:
            init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        init.xavier_uniform_(module.weight)
        if module.bias is not None:
            init.zeros_(module.bias)

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def create_pinpad_env(env_id, num_objects, obj_seq, room_size, num_rows, num_cols):
    """Register and create a PinPad environment."""
    if env_id in gym.envs.registry:
        del gym.envs.registry[env_id]

    register(
        id=env_id,
        entry_point=PinPad,
        kwargs={
            "num_objects": num_objects,
            "obj_seq": obj_seq,
            "room_size": room_size,
            "num_rows": num_rows,
            "num_cols": num_cols
        },
    )

    env = gym.make(env_id, obj_seq=obj_seq)
    #env = OneHotFullyObsWrapper(env)
    env = RGBImgObsWrapper(env)
    return env


class SubgoalProbe(nn.Module):
    """Linear probe: residual stream (..., dim) -> subgoal logits (..., num_objects)."""

    def __init__(self, dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(dim, num_classes)

    def forward(self, residual_stream: torch.Tensor) -> torch.Tensor:
        return self.linear(residual_stream)


def visualize_switch_betas_vs_labels(
    switch_betas,      # (B, T-1)
    labels,            # (B, T)
    episode_lens,      # (B,) or None
    gradient_step,
    num_samples=3,
):
    """
    Visualize switch betas vs GT labels for randomly sampled sequences in the batch.
    Logs a single stacked figure to wandb.
    """
    B, T_minus_1 = switch_betas.shape

    # randomly sample sequences from the batch
    num_samples = min(num_samples, B)
    sample_indices = np.random.choice(B, size=num_samples, replace=False)

    # create figure with 2 * num_samples subplots (labels + switch_betas for each sample)
    fig, axes = plt.subplots(2 * num_samples, 1, figsize=(12, 3 * num_samples), sharex=False)
    fig.suptitle(f'Step {gradient_step} | Note: -1 in labels = explore (no specific subgoal)', fontsize=10)

    for i, idx in enumerate(sample_indices):
        # get episode length for this sample (if available)
        if episode_lens is not None:
            ep_len = int(episode_lens[idx].item())
        else:
            ep_len = T_minus_1

        # extract data for this sample
        sample_switch_betas = switch_betas[idx, :ep_len-1].detach().cpu()  # (T-1,)
        sample_labels = labels[idx, :ep_len].cpu()  # (T,)

        # get axes for this sample pair
        ax1 = axes[2 * i]      # labels
        ax2 = axes[2 * i + 1]  # switch betas

        # top plot: labels (align with switch_betas by using labels[:-1])
        ax1.plot(sample_labels[:-1].numpy(), label=f'labels (sample {idx})', color='red')
        ax1.set_ylabel('labels')
        ax1.legend(loc='upper right')

        # bottom plot: switch betas
        ax2.plot(sample_switch_betas.numpy(), label='switch betas', linewidth=2)
        ax2.set_xlabel('timesteps')
        ax2.set_ylabel('switch betas')
        ax2.legend(loc='upper right')

    plt.tight_layout()

    # log to wandb
    wandb.log({
        f"switch_betas_vs_labels/step_{gradient_step}": wandb.Image(fig)
    }, step=gradient_step)

    plt.close(fig)


# ============================================================================
# Training
# ============================================================================

def train(
    input_dir = "pinpad_demonstrations",
    env_id = "PinPad-v0",
    # PinPad environment parameters (must match the demo collection settings)
    num_objects = 4,
    default_obj_seq = [0,1,2],  # placeholder for env registration
    room_size = 8,
    num_rows = 1,
    num_cols = 1,
    num_actions = None,
    # Training parameters
    cloning_epochs = 10,
    probe_epochs = 5,
    batch_size = 128,
    gradient_accumulation_steps = None,
    lr = 1e-4,
    weight_decay = 0.03,
    dim = 512,
    depth = 2,
    heads = 8,
    dim_head = 64,
    switch_temperature = 1.,
    use_wandb = False,
    wandb_project = "metacontroller-pinpad-bc",
    wandb_run_name = None,
    checkpoint_path = "transformer_pinpad_bc.pt",
    load_transformer_weights_path = None,
    save_steps = 1000,
    state_loss_weight = 1.,
    action_loss_weight = 1.,
    max_grad_norm = 1.,
    use_resnet = False,
    resnet_type = 'resnet18',
    condition_on_mission_embed = False,
    mission_embed_dim = 384,
    # Linear probe (subgoal prediction from residual stream, after BC)
    probe_lr = 1e-3,
    probe_checkpoint_path = "subgoal_probe_pinpad.pt",
    load_probe_path = None,
    use_layernorm = False,
):

    def store_checkpoint(step: int | None = None):
        if accelerator.is_main_process:
            if exists(step):
                checkpoint_path_with_step = checkpoint_path.replace('.pt', f'_step_{step}.pt')
            else:
                checkpoint_path_with_step = checkpoint_path
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save(checkpoint_path_with_step)
            accelerator.print(f"Model saved to {checkpoint_path_with_step}")

    # accelerator

    accelerator = Accelerator(log_with = "wandb" if use_wandb else None)

    if use_wandb:
        init_kwargs = {}
        if exists(wandb_run_name):
            init_kwargs["wandb"] = {"name": wandb_run_name}
        accelerator.init_trackers(
            wandb_project,
            config = {
                "cloning_epochs": cloning_epochs,
                "probe_epochs": probe_epochs,
                "probe_lr": probe_lr,
                "batch_size": batch_size,
                "lr": lr,
                "dim": dim,
                "depth": depth,
                "heads": heads,
                "dim_head": dim_head,
                "env_id": env_id,
                "state_loss_weight": state_loss_weight,
                "action_loss_weight": action_loss_weight,
            },
            init_kwargs = init_kwargs
        )

    # replay buffer and dataloader

    input_path = Path(input_dir)
    assert input_path.exists(), f"Input directory {input_dir} does not exist"

    replay_buffer = ReplayBuffer.from_folder(input_path)
    dataloader = replay_buffer.dataloader(batch_size = batch_size)

    # state shape and action dimension
    # state: (B, T, H, W, C) where C is num_channels (one-hot encoded)

    state_shape = replay_buffer.shapes['state']
    if use_resnet: state_dim = 256
    else:
        state_dim = int(torch.tensor(state_shape).prod().item())

    temp_env = create_pinpad_env(env_id, num_objects, default_obj_seq, room_size, num_rows, num_cols)
    if num_actions is None:
        num_actions = int(temp_env.action_space.n)
    temp_env.close()

    accelerator.print(f"Detected state_dim: {state_dim}, num_actions: {num_actions} from env: {env_id}")
    accelerator.print(f"State shape from replay buffer: {state_shape}")

    meta_controller = MetaController(dim, switch_temperature=switch_temperature)

    transformer_class = TransformerWithResnet if use_resnet else Transformer

    transformer_kwargs = dict(
        dim=dim,
        state_embed_readout=dict(num_continuous=state_dim),
        action_embed_readout=dict(num_discrete=num_actions),
        lower_body=dict(depth=depth, heads=heads, attn_dim_head=dim_head),
        upper_body=dict(depth=depth, heads=heads, attn_dim_head=dim_head),
        meta_controller=meta_controller,
        use_layernorm=use_layernorm,
        resnet_type=resnet_type if use_resnet else None,
        encoder_kwargs=dict(depth=depth, heads=heads, attn_dim_head=dim_head)
    )

    model = transformer_class(**transformer_kwargs)

    if exists(load_transformer_weights_path):
        _path = Path(load_transformer_weights_path)
        assert _path.exists(), f"load_transformer_weights_path {load_transformer_weights_path} does not exist"
        if hasattr(model, "load"):
            model.load(str(_path))
        else:
            state = torch.load(_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state, strict=False)
        accelerator.print(f"Loaded transformer weights from {load_transformer_weights_path}")
    else:
        for name, child in model.named_children():
            if name != "meta_controller":
                child.apply(initialize_weights_xavier)
        accelerator.print("Initialized transformer with Xavier")

    meta_controller.apply(initialize_weights_xavier)
    accelerator.print("Initialized meta_controller with Xavier")

    optim_model = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    probe = SubgoalProbe(dim=dim, num_classes=num_objects)
    if exists(load_probe_path):
        _path = Path(load_probe_path)
        assert _path.exists(), f"load_probe_path {load_probe_path} does not exist"
        state = torch.load(_path, map_location="cpu", weights_only=True)
        probe.load_state_dict(state, strict=True)
        accelerator.print(f"Loaded probe weights from {load_probe_path}")
    optim_probe = AdamW(probe.parameters(), lr=probe_lr)

    # prepare

    model, optim_model, probe, optim_probe, dataloader = accelerator.prepare(
        model, optim_model, probe, optim_probe, dataloader
    )

    # training
    gradient_step = 0
    for epoch in range(cloning_epochs + probe_epochs):
        from collections import defaultdict
        total_losses = defaultdict(float)

        is_probe = (epoch >= cloning_epochs)
        if is_probe:
            if epoch == cloning_epochs:
                logger.info("Linear probing began.")
            model.eval()
            probe.train()
        else:
            if epoch == 0:
                logger.info("Behavior cloning started.")
            model.train()

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}", disable=not accelerator.is_local_main_process)

        for batch in progress_bar:

            # normalize inputs to be first in [0, 1] by dividing by 255
            # then normalize w.r.t. mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
            states = batch['state'].float()
            states = torch.clamp(states / 255.0, min=0.0, max=1.0)
            states = (states - torch.tensor([0.485, 0.456, 0.406]).to(states.device)) / torch.tensor([0.229, 0.224, 0.225]).to(states.device)

            actions = batch['action'].long()

            labels = batch.get('label')
            if labels is not None:
                labels = labels.long()
            episode_lens = batch.get('_lens')

            if condition_on_mission_embed and exists(mission_embeddings):
                mission_embeddings = mission_embeddings.to(accelerator.device)
            else:
                mission_embeddings = None

            if not use_resnet: # flatten state: (B, T, 7, 7, 3) -> (B, T, 147)
                states = rearrange(states, 'b t ... -> b t (...)')

            if is_probe:
                # Linear probe: get pre-control residual stream via BC forward (teacher-forced), then train probe
                if labels is None:
                    progress_bar.set_postfix(probe_skip="no_labels")
                    continue

                with torch.no_grad():
                    _, residual_stream = model(
                        state=states,
                        actions=actions,
                        episode_lens=episode_lens,
                        force_behavior_cloning=True,
                        return_residual_stream=True,
                        condition=mission_embeddings,
                    )
                # residual_stream is (B, T-1, dim) from BC's state[:, :-1]
                labels_probe = labels[:, :-1]  # (B, T-1) to align with residual_stream
                logits = probe(residual_stream)  # (B, T-1, num_objects)
                seq_len = logits.shape[1]
                #seq_len = episode_lens.min().item()

                # Valid positions: within episode and label != -1 (explore)
                if episode_lens is not None:
                    ep_len_eff = (episode_lens - 1).clamp(min=0)
                    within_episode = torch.arange(seq_len-1, device=logits.device)[None, :] < ep_len_eff[:, None]
                else:
                    within_episode = torch.ones(logits.shape[0], seq_len, dtype=torch.bool, device=logits.device)
                valid = (labels_probe >= 0) & within_episode

                if not valid.any():
                    progress_bar.set_postfix(probe_skip="no_valid")
                    continue

                target = labels_probe.clone()
                target[~valid] = -100
                probe_loss = F.cross_entropy(
                    logits.view(-1, num_objects),
                    target.view(-1),
                    ignore_index=-100,
                )
                if gradient_accumulation_steps is not None:
                    probe_loss = probe_loss / gradient_accumulation_steps

                pred = logits.argmax(dim=-1)
                probe_acc = (pred[valid] == labels_probe[valid]).float().mean().item()

                accelerator.backward(probe_loss)
                grad_norm = torch.nn.utils.clip_grad_norm_(probe.parameters(), max_norm=max_grad_norm)
                if gradient_accumulation_steps is None or gradient_step % gradient_accumulation_steps == 0:
                    optim_probe.step()
                    optim_probe.zero_grad()

                log = dict(probe_loss=probe_loss.item(), probe_acc=probe_acc, probe_grad_norm=grad_norm.item())
                prefix = "probe_phase"
                loss_for_log = probe_loss
                grad_norm_for_log = grad_norm.item()

            else:
                # Behavior cloning
                with accelerator.accumulate(model):
                    losses, action_logits = model(
                        state=states,
                        actions=actions,
                        episode_lens=episode_lens,
                        discovery_phase=False,
                        force_behavior_cloning=True,
                        return_meta_controller_output=False,
                        return_action_logits=True,
                        condition=mission_embeddings,
                    )
                    state_loss, action_loss = losses

                    loss = (
                        state_loss * state_loss_weight +
                        action_loss * action_loss_weight
                    )
                    if gradient_accumulation_steps is not None:
                        loss = loss / gradient_accumulation_steps

                    # Action prediction accuracy (aligns with model: state[:, :-1] -> target_actions = actions[:, :-1])
                    target_actions = actions[:, :-1]
                    pred_actions = action_logits.argmax(dim=-1)
                    seq_len = action_logits.shape[1]
                    if episode_lens is not None:
                        ep_len_eff = (episode_lens - 1).clamp(min=0)
                        within_episode = torch.arange(seq_len, device=action_logits.device)[None, :] < ep_len_eff[:, None]
                    else:
                        within_episode = torch.ones(action_logits.shape[0], seq_len, dtype=torch.bool, device=action_logits.device)

                    # modify the action accuracy s.t. only the predictions within each episode_length are considered
                    action_acc = (pred_actions[within_episode] == target_actions[within_episode]).float().mean().item()
                    #action_acc = (pred_actions[:, :episode_lens.min().item() - 1] == target_actions[:, :episode_lens.min().item() - 1]).float().mean().item()

                    accelerator.backward(loss)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                    if gradient_accumulation_steps is None or gradient_step % gradient_accumulation_steps == 0:
                        optim_model.step()
                        optim_model.zero_grad()

                log = dict(state_loss=state_loss.item(), action_loss=action_loss.item(), action_acc=action_acc)
                prefix = "behavior_cloning"
                loss_for_log = loss
                grad_norm_for_log = grad_norm.item()

            for key, value in log.items():
                total_losses[key] += value

            accelerator.log({
                **log,
                f"{prefix}_total_loss": loss_for_log.item(),
                f"{prefix}_grad_norm": grad_norm_for_log,
            })
            progress_bar.set_postfix(**log)
            gradient_step += 1

            if gradient_step % save_steps == 0:
                accelerator.wait_for_everyone()
                store_checkpoint(gradient_step)
                if is_probe and accelerator.is_main_process:
                    torch.save(accelerator.unwrap_model(probe).state_dict(), probe_checkpoint_path)
                    accelerator.print(f"Probe saved to {probe_checkpoint_path}")

        avg_losses = {k: v / len(dataloader) for k, v in total_losses.items()}
        avg_losses_str = ", ".join([f"{k}={v:.4f}" for k, v in avg_losses.items()])
        accelerator.print(f"Epoch {epoch}: {avg_losses_str}")
        if not is_probe and epoch == cloning_epochs - 1:
            logger.info("Behavior cloning ended.")

    # save weights
    accelerator.wait_for_everyone()
    store_checkpoint(None)
    if accelerator.is_main_process:
        unwrapped_probe = accelerator.unwrap_model(probe)
        torch.save(unwrapped_probe.state_dict(), probe_checkpoint_path)
        accelerator.print(f"Probe saved to {probe_checkpoint_path}")

    accelerator.end_training()

if __name__ == "__main__":
    fire.Fire(train)
