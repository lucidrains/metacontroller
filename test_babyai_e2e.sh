#!/bin/bash
set -e

# 1. Gather trajectories
echo "Gathering trajectories..."
uv run gather_babyai_trajs.py \
    --num_seeds 1000 \
    --num_episodes_per_seed 100 \
    --num_steps 500 \
    --output_dir end_to_end_trajectories \
    --env_id BabyAI-MiniBossLevel-v0 \
    --num_actions 7

# 2. Behavioral cloning
echo "Training behavioral cloning model..."
ACCELERATE_USE_CPU=true ACCELERATE_MIXED_PRECISION=no uv run train_behavior_clone_babyai.py \
    --input_dir end_to_end_trajectories \
    --env_id BabyAI-MiniBossLevel-v0 \
    --checkpoint_path end_to_end_model.pt \
    --lr 5e-5 \
    --discovery_lr 5e-5 \
    --batch_size 256 \
    --cloning_epochs 10 --discovery_epochs 10 \
    --switch_temperature 0.1 \
    --state_loss_weight 0.001 \
    --discovery_action_recon_loss_weight 1.0 \
    --discovery_switch_warmup_steps 1 \
    --discovery_switch_lr_scale 1.0 \
    --discovery_kl_loss_weight 0.01 \
    --discovery_obs_loss_weight 0.005 \
    --discovery_ratio_loss_weight 1.0 \
    --save_steps 500 \
    --dim 256 \
    --heads 8 \
    --dim_head 64 \
    --depth 6 \
    --max_grad_norm 1.0 \
    --condition_on_mission_embed \
    --use_resnet --use_wandb

# 3. Inference rollouts
echo "Running inference rollouts..."
uv run train_babyai.py \
    --transformer_weights_path end_to_end_model.pt \
    --meta_controller_weights_path meta_controller_discovery.pt \
    --env_name BabyAI-MiniBossLevel-v0 \
    --num_episodes 1000 \
    --buffer_size 1000 \
    --max_timesteps 100 \
    --num_groups 16 \
    --lr 1e-4 \
    --use_resnet
