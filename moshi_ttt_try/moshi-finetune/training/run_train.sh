#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Moshi TTT Training Script
# Usage: ./run_train.sh [config_file] [gpu_id]
# 
# Examples:
#   ./run_train.sh                           # Use default config on GPU 0
#   ./run_train.sh example/moshi_7B.yaml     # Use specific config on GPU 0  
#   ./run_train.sh example/moshi_7B.yaml 1   # Use specific config on GPU 1
# -----------------------------------------------------------------------------

set -e  # Exit on any error

# Configuration
DEFAULT_CONFIG="example/moshi_7B.yaml"
CONFIG_FILE=${1:-$DEFAULT_CONFIG}
GPU_ID=${2:-0}

# Restrict to specified GPU
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Error: Config file '$CONFIG_FILE' not found!"
    echo "Available configs:"
    ls -la example/*.yaml 2>/dev/null || echo "  No configs found in example/"
    exit 1
fi

# Check if conda environment is activated
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "moshi_ttt_fixed" ]; then
    echo "⚠️  Warning: Expected conda environment 'moshi_ttt_fixed' but found '$CONDA_DEFAULT_ENV'"
    echo "Please run: conda activate moshi_ttt_fixed"
    exit 1
fi

# Print training info
echo "🚀 Starting Moshi TTT Training"
echo "================================"
echo "📄 Config file: $CONFIG_FILE"
echo "🎯 GPU ID: $GPU_ID"
echo "🐍 Conda env: $CONDA_DEFAULT_ENV"
echo "📂 Working dir: $(pwd)"
echo ""

# Find torchrun (try multiple locations)
TORCHRUN=""
for cmd in torchrun python -m torch.distributed.run; do
    if command -v $cmd >/dev/null 2>&1; then
        TORCHRUN=$cmd
        break
    fi
done

if [ -z "$TORCHRUN" ]; then
    echo "❌ Error: Could not find torchrun command!"
    echo "Please ensure PyTorch is properly installed"
    exit 1
fi

echo "🔧 Using torchrun: $TORCHRUN"
echo ""

# Execute the training
echo "🏃 Starting training..."
$TORCHRUN \
  --nproc-per-node=1 \
  --master-port=29999 \
  finetune/train.py \
  --config $CONFIG_FILE

echo ""
echo "✅ Training completed!"