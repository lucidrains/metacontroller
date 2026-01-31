#!/bin/bash
set -e

# 1. Gather trajectories
echo "Gathering trajectories..."
uv run gather_babyai_trajs.py \
    --num_seeds 100 \
    --num_episodes_per_seed 10 \
    --num_steps 500 \
    --output_dir end_to_end_trajectories \
    --env_id BabyAI-MiniBossLevel-v0

# 2. Behavioral cloning
echo "Training behavioral cloning model..."
ACCELERATE_USE_CPU=true ACCELERATE_MIXED_PRECISION=no uv run train_behavior_clone_babyai.py \
    --cloning_epochs 10 \
    --discovery_epochs 10 \
    --batch_size 256 \
    --input_dir end_to_end_trajectories \
    --env_id BabyAI-MiniBossLevel-v0 \
    --checkpoint_path end_to_end_model.pt \
    --use_resnet

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
