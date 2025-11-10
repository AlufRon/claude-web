#!/bin/bash

# TTT Training Script with Memory Optimization
# Uses PyTorch memory management optimizations

echo "🚀 Running TTT Training with Memory Optimizations"

# Set PyTorch memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Activate environment
conda activate moshi_ttt_fixed

# Run training with minimal TTT config
echo "Starting training with 4 TTT layers..."
torchrun --standalone --nnodes=1 --nproc_per_node=1 train.py configs/ttt_minimal_48gb.yaml

echo "Training completed or stopped."