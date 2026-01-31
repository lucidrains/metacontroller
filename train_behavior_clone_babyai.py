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

from metacontroller.metacontroller import Transformer, MetaController
from metacontroller.transformer_with_resnet import TransformerWithResnet

import minigrid
import gymnasium as gym

def train(
    input_dir = "babyai-minibosslevel-trajectories",
    env_id = "BabyAI-MiniBossLevel-v0",
    cloning_epochs = 10,
    discovery_epochs = 10,
    batch_size = 32,
    lr = 1e-4,
    discovery_lr = 1e-4,
    dim = 512,
    depth = 2,
    heads = 8,
    dim_head = 64,
    use_wandb = False,
    wandb_project = "metacontroller-babyai-bc",
    checkpoint_path = "transformer_bc.pt",
    meta_controller_checkpoint_path = "meta_controller_discovery.pt",
    state_loss_weight = 1.,
    action_loss_weight = 1.,
    discovery_action_recon_loss_weight = 1.,
    discovery_kl_loss_weight = 1.,
    discovery_switch_loss_weight = 1.,
    max_grad_norm = 1.,
    use_resnet = False
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
    from babyai_env import create_env
    temp_env = create_env(env_id)
    num_actions = int(temp_env.action_space.n)
    temp_env.close()

    accelerator.print(f"Detected state_dim: {state_dim}, num_actions: {num_actions} from env: {env_id}")

    # transformer
    
    transformer_class = TransformerWithResnet if use_resnet else Transformer

    model = transformer_class(
        dim = dim,
        state_embed_readout = dict(num_continuous = state_dim),
        action_embed_readout = dict(num_discrete = num_actions),
        lower_body = dict(depth = depth, heads = heads, attn_dim_head = dim_head),
        upper_body = dict(depth = depth, heads = heads, attn_dim_head = dim_head)
    )

    # meta controller

    meta_controller = MetaController(dim)

    # optimizer

    optim_model = Adam(model.parameters(), lr = lr)

    optim_meta_controller = Adam(meta_controller.discovery_parameters(), lr = discovery_lr)

    # prepare

    model, optim_model, optim_meta_controller, dataloader = accelerator.prepare(model, optim_model, optim_meta_controller, dataloader)

    # training

    for epoch in range(cloning_epochs + discovery_epochs):

        model.train()
        from collections import defaultdict
        total_losses = defaultdict(float)

        progress_bar = tqdm(dataloader, desc = f"Epoch {epoch}", disable = not accelerator.is_local_main_process)

        is_discovering = (epoch >= cloning_epochs) # discovery phase is BC with metacontroller tuning

        optim = optim_model if not is_discovering else optim_meta_controller

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
                losses = model(
                    states,
                    actions,
                    episode_lens = episode_lens,
                    discovery_phase = is_discovering,
                    meta_controller = meta_controller if is_discovering else None
                )

                if is_discovering:
                    action_recon_loss, kl_loss, switch_loss = losses

                    loss = (
                        action_recon_loss * discovery_action_recon_loss_weight +
                        kl_loss * discovery_kl_loss_weight +
                        switch_loss * discovery_switch_loss_weight
                    )

                    log = dict(
                        action_recon_loss = action_recon_loss.item(),
                        kl_loss = kl_loss.item(),
                        switch_loss = switch_loss.item()
                    )
                else:
                    state_loss, action_loss = losses

                    loss = (
                        state_loss * state_loss_weight +
                        action_loss * action_loss_weight
                    )

                    log = dict(
                        state_loss = state_loss.item(),
                        action_loss = action_loss.item(),
                    )

                # backprop

                accelerator.backward(loss)

                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = max_grad_norm)

                optim.step()
                optim.zero_grad()

            # log
            
            for key, value in log.items():
                total_losses[key] += value

            accelerator.log({
                **log,
                "total_loss": loss.item(),
                "grad_norm": grad_norm.item()
            })

            progress_bar.set_postfix(**log)

        avg_losses = {k: v / len(dataloader) for k, v in total_losses.items()}
        avg_losses_str = ", ".join([f"{k}={v:.4f}" for k, v in avg_losses.items()])
        accelerator.print(f"Epoch {epoch}: {avg_losses_str}")

    # save weights

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:

        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save(checkpoint_path)

        unwrapped_meta_controller = accelerator.unwrap_model(meta_controller)
        unwrapped_meta_controller.save(meta_controller_checkpoint_path)

        accelerator.print(f"Model saved to {checkpoint_path}, MetaController to {meta_controller_checkpoint_path}")

    accelerator.end_training()

if __name__ == "__main__":
    fire.Fire(train)
