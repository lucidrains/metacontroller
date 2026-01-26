# /// script
# dependencies = [
#   "accelerate",
#   "fire",
#   "memmap-replay-buffer>=0.0.23",
#   "metacontroller-pytorch",
#   "torch",
#   "einops",
#   "tqdm",
#   "wandb",
#   "gymnasium",
#   "minigrid"
# ]
# ///

import fire
from tqdm import tqdm
from pathlib import Path

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from accelerate import Accelerator
from memmap_replay_buffer import ReplayBuffer
from einops import rearrange

from metacontroller.metacontroller import Transformer
from metacontroller.metacontroller_with_resnet import TransformerWithResnetEncoder

import minigrid
import gymnasium as gym

def train(
    input_dir: str = "babyai-minibosslevel-trajectories",
    env_id: str = "BabyAI-MiniBossLevel-v0",
    cloning_epochs: int = 10,
    discovery_epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-4,
    dim: int = 512,
    depth: int = 2,
    heads: int = 8,
    dim_head: int = 64,
    use_wandb: bool = False,
    wandb_project: str = "metacontroller-babyai-bc",
    checkpoint_path: str = "transformer_bc.pt",
    state_loss_weight: float = 1.,
    action_loss_weight: float = 1.,
    use_resnet: bool = False
):
    # accelerator

    accelerator = Accelerator(log_with = "wandb" if use_wandb else None)

    if use_wandb:
        accelerator.init_trackers(
            wandb_project,
            config = {
                "cloning_epochs": cloning_epochs,
                "discovery_epochs": discovery_epochs,
                "batch_size": batch_size,
                "lr": lr,
                "dim": dim,
                "depth": depth,
                "heads": heads,
                "dim_head": dim_head,
                "env_id": env_id,
                "state_loss_weight": state_loss_weight,
                "action_loss_weight": action_loss_weight
            }
        )

    # replay buffer and dataloader

    input_path = Path(input_dir)
    assert input_path.exists(), f"Input directory {input_dir} does not exist"

    replay_buffer = ReplayBuffer.from_folder(input_path)
    dataloader = replay_buffer.dataloader(batch_size = batch_size)

    # state shape and action dimension
    # state: (B, T, H, W, C) or (B, T, D)
    state_shape = replay_buffer.shapes['state']
    if use_resnet: state_dim = 256
    else: state_dim = int(torch.tensor(state_shape).prod().item())

    # deduce num_actions from the environment
    minigrid.register_minigrid_envs()
    temp_env = gym.make(env_id)
    num_actions = int(temp_env.action_space.n)
    temp_env.close()

    accelerator.print(f"Detected state_dim: {state_dim}, num_actions: {num_actions} from env: {env_id}")

    # transformer
    
    transformer_class = TransformerWithResnetEncoder if use_resnet else Transformer
    model = transformer_class(
        dim = dim,
        state_embed_readout = dict(num_continuous = state_dim),
        action_embed_readout = dict(num_discrete = num_actions),
        lower_body = dict(depth = depth, heads = heads, attn_dim_head = dim_head),
        upper_body = dict(depth = depth, heads = heads, attn_dim_head = dim_head)
    )

    # optimizer

    optim = Adam(model.parameters(), lr = lr)

    # prepare

    model, optim, dataloader = accelerator.prepare(model, optim, dataloader)

    # training
    for epoch in range(cloning_epochs + discovery_epochs):
        model.train()
        total_state_loss = 0.
        total_action_loss = 0.

        progress_bar = tqdm(dataloader, desc = f"Epoch {epoch}", disable = not accelerator.is_local_main_process)
        is_discovering = (epoch >= cloning_epochs) # discovery phase is BC with metacontroller tuning

        for batch in progress_bar:
            # batch is a NamedTuple (e.g. MemoryMappedBatch)
            # state: (B, T, 7, 7, 3), action: (B, T)

            states = batch['state'].float()
            actions = batch['action'].long()
            episode_lens = batch.get('_lens')

            # use resnet18 to embed visual observations
            if use_resnet: 
                states = model.visual_encode(states)
            else: # flatten state: (B, T, 7, 7, 3) -> (B, T, 147)
                states = rearrange(states, 'b t ... -> b t (...)')

            with accelerator.accumulate(model):
                state_loss, action_loss = model(states, actions, episode_lens = episode_lens, discovery_phase=is_discovering)
                loss = state_loss * state_loss_weight + action_loss * action_loss_weight

                accelerator.backward(loss)

                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optim.step()
                optim.zero_grad()

            # log

            total_state_loss += state_loss.item()
            total_action_loss += action_loss.item()

            accelerator.log({
                "state_loss": state_loss.item(),
                "action_loss": action_loss.item(),
                "total_loss": loss.item(),
                "grad_norm": grad_norm.item()
            })

            progress_bar.set_postfix(
                state_loss = state_loss.item(),
                action_loss = action_loss.item()
            )

        avg_state_loss = total_state_loss / len(dataloader)
        avg_action_loss = total_action_loss / len(dataloader)

        accelerator.print(f"Epoch {epoch}: state_loss={avg_state_loss:.4f}, action_loss={avg_action_loss:.4f}")

    # save weights

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save(checkpoint_path)
        accelerator.print(f"Model saved to {checkpoint_path}")

    accelerator.end_training()

if __name__ == "__main__":
    fire.Fire(train)
