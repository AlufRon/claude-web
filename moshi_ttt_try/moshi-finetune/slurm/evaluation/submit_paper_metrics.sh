#!/bin/bash
#SBATCH --job-name=paper_metrics
#SBATCH --output=/home/alufr/ttt_tests/moshi-finetune/logs/evaluation/paper_metrics_%j.log
#SBATCH --error=/home/alufr/ttt_tests/moshi-finetune/logs/evaluation/paper_metrics_%j.err
#SBATCH --time=10:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --partition=main
#SBATCH --constraint=rtx_6000
#SBATCH --exclude=cs-6000-01,cs-6000-02,cs-6000-03,cs-6000-04,ise-6000-08,ise-6000-07

# Script to run paper metrics evaluation on a TTT checkpoint or baseline Moshi
#
# Usage:
#   For TTT checkpoint:
#     sbatch submit_paper_metrics.sh [checkpoint_dir] [max_samples] [config_file] [librilight_max_length]
#   For baseline:
#     sbatch submit_paper_metrics.sh --baseline [max_samples] [config_file] [librilight_max_length]
#   Skip LibriLight:
#     sbatch submit_paper_metrics.sh [checkpoint_dir] [max_samples] [config_file] skip
#
# LibriLight max length options:
#   - 5000   (default, ~5-10 min, quick eval)
#   - 24000  (full eval, ~20-30 min)
#   - skip   (skip LibriLight entirely)

set -e  # Exit on error

# Check if first argument is --baseline
if [ "$1" == "--baseline" ]; then
    BASELINE_MODE=true
    CHECKPOINT_DIR=""
    MAX_SAMPLES=${2:-50}
    CONFIG_FILE=${3:-"/home/alufr/ttt_tests/moshi-finetune/example/moshi_7B_multilayer_with_ttt.yaml"}
    LIBRILIGHT_LENGTH=${4:-5000}
else
    BASELINE_MODE=false
    CHECKPOINT_DIR=${1:-"/sise/eliyanac-group/ron_al/dailytalk_finetune_from_librilight6/checkpoints/checkpoint_003000/consolidated/"}
    MAX_SAMPLES=${2:-50}
    CONFIG_FILE=${3:-"/home/alufr/ttt_tests/moshi-finetune/example/moshi_7B_multilayer_with_ttt.yaml"}
    LIBRILIGHT_LENGTH=${4:-5000}
fi

echo "=================================================="
echo "Paper Metrics Evaluation"
echo "=================================================="
if [ "$BASELINE_MODE" == "true" ]; then
    echo "Mode: BASELINE (no TTT)"
else
    echo "Checkpoint: $CHECKPOINT_DIR"
fi
echo "Max samples per task: $MAX_SAMPLES"
echo "Config file: $CONFIG_FILE"
if [ "$LIBRILIGHT_LENGTH" == "skip" ]; then
    echo "LibriLight: SKIPPED"
else
    echo "LibriLight max length: $LIBRILIGHT_LENGTH tokens"
fi
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=================================================="
echo ""

# Activate conda environment
source ~/.bashrc
conda activate moshi_ttt_fixed

# Verify environment
echo "🐍 Active conda environment: $CONDA_DEFAULT_ENV"
echo "🐍 Python: $(which python)"
echo "🐍 Python version: $(python --version)"
echo ""

# GPU check
echo "🔧 GPU Check:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# Run paper metrics evaluation
echo "🚀 Starting paper metrics evaluation..."
echo ""

cd /home/alufr/ttt_tests/moshi-finetune

# Add current directory to PYTHONPATH for module imports
export PYTHONPATH="/home/alufr/ttt_tests/moshi-finetune:$PYTHONPATH"

# Build command with LibriLight options
if [ "$LIBRILIGHT_LENGTH" == "skip" ]; then
    LIBRILIGHT_ARGS="--skip-librilight"
else
    LIBRILIGHT_ARGS="--librilight-max-length $LIBRILIGHT_LENGTH"
fi

if [ "$BASELINE_MODE" == "true" ]; then
    python evaluation/scripts/run_paper_metrics_on_checkpoint.py \
        --baseline \
        --max-samples $MAX_SAMPLES \
        --config "$CONFIG_FILE" \
        --device cuda \
        --hf-repo kyutai/moshiko-pytorch-bf16 \
        $LIBRILIGHT_ARGS
else
    python evaluation/scripts/run_paper_metrics_on_checkpoint.py \
        --checkpoint "$CHECKPOINT_DIR" \
        --max-samples $MAX_SAMPLES \
        --config "$CONFIG_FILE" \
        --device cuda \
        --hf-repo kyutai/moshiko-pytorch-bf16 \
        $LIBRILIGHT_ARGS
fi

EXIT_CODE=$?

echo ""
echo "=================================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Paper metrics evaluation completed successfully!"
else
    echo "❌ Paper metrics evaluation failed with exit code: $EXIT_CODE"
fi
echo "=================================================="

exit $EXIT_CODE
