#!/bin/bash
#SBATCH --partition=main
#SBATCH --job-name=paper_metrics
#SBATCH --output=/home/alufr/ttt_tests/moshi-finetune/logs/evaluation/paper_metrics_%j.log
#SBATCH --error=/home/alufr/ttt_tests/moshi-finetune/logs/evaluation/paper_metrics_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --constraint=rtx_6000
#SBATCH --mem=50G
#SBATCH --time=08:00:00
#SBATCH --exclude=cs-6000-01,cs-6000-02,cs-6000-03,ise-6000-08

# SLURM job script for paper metrics evaluation
# This script is called by submit_paper_metrics_wrapper.sh
# Environment variables are set via --export flag

set -e  # Exit on error

echo "=================================================="
echo "Paper Metrics Evaluation"
echo "=================================================="
if [ "$BASELINE_MODE" == "true" ]; then
    echo "Mode: BASELINE (no TTT)"
else
    echo "Mode: TTT Checkpoint"
    echo "Checkpoint: $CHECKPOINT_DIR"
fi
echo "Max samples per task: $MAX_SAMPLES"
echo "Config file: $CONFIG_FILE"
echo "HF repo: $HF_REPO"
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

# GPU diagnostics
echo "🔧 GPU Diagnostics:"
echo "🔧 Hostname: $(hostname)"
echo "🔧 SLURM_JOB_ID: $SLURM_JOB_ID"
echo "🔧 CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
echo "🔧 nvidia-smi output:"
nvidia-smi
echo ""

# Run paper metrics evaluation
echo "🚀 Starting paper metrics evaluation..."
echo ""

cd /home/alufr/ttt_tests/moshi-finetune

# Add current directory to PYTHONPATH for module imports
export PYTHONPATH="/home/alufr/ttt_tests/moshi-finetune:$PYTHONPATH"

if [ "$BASELINE_MODE" == "true" ]; then
    echo "Running baseline evaluation (no TTT)..."
    python evaluation/scripts/run_paper_metrics_on_checkpoint.py \
        --baseline \
        --max-samples $MAX_SAMPLES \
        --config "$CONFIG_FILE" \
        --device cuda \
        --hf-repo "$HF_REPO" \
        --output ./baseline_paper_metrics_results.json
else
    echo "Running TTT checkpoint evaluation..."
    python evaluation/scripts/run_paper_metrics_on_checkpoint.py \
        --checkpoint "$CHECKPOINT_DIR" \
        --max-samples $MAX_SAMPLES \
        --config "$CONFIG_FILE" \
        --device cuda \
        --hf-repo "$HF_REPO"
fi

EXIT_CODE=$?

echo ""
echo "=================================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Paper metrics evaluation completed successfully!"
    if [ "$BASELINE_MODE" == "true" ]; then
        echo "📊 Results saved to: ./baseline_paper_metrics_results.json"
    else
        echo "📊 Results saved to: $CHECKPOINT_DIR/paper_metrics_results.json"
    fi
else
    echo "❌ Paper metrics evaluation failed with exit code: $EXIT_CODE"
fi
echo "=================================================="

exit $EXIT_CODE
