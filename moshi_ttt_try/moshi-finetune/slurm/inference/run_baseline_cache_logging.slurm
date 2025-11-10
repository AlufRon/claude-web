#!/bin/bash
#SBATCH --job-name=moshi_cache_log
#SBATCH --output=logs/inference/cache_logging_%j.log
#SBATCH --error=logs/inference/cache_logging_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

echo "=================================="
echo "Baseline Moshi with Cache Logging"
echo "=================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Started: $(date)"
echo "=================================="
echo ""

# Environment
source ~/.bashrc
conda activate moshi_ttt_fixed

echo "🐍 Active conda environment: $CONDA_DEFAULT_ENV"
echo "🐍 Python path: $(which python)"
echo "🐍 Python version: $(python --version)"
echo ""

# GPU diagnostics
echo "🔧 GPU Diagnostics:"
echo "🔧 Hostname: $(hostname)"
echo "🔧 SLURM_JOB_ID: $SLURM_JOB_ID"
echo "🔧 CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "🔧 nvidia-smi output:"
nvidia-smi
echo ""

# Change to working directory
cd /home/alufr/ttt_tests/moshi-finetune

# Input audio file (the one showing the jump)
INPUT_AUDIO="/sise/eliyanac-group/ron_al/examples/combined_2673329117028914160_24000hz.wav"
OUTPUT_AUDIO="./output_baseline_cache_logged_${SLURM_JOB_ID}.wav"

echo "📁 Input audio: $INPUT_AUDIO"
echo "📁 Output audio: $OUTPUT_AUDIO"
echo ""

# Run inference with cache logging
echo "🚀 Starting inference..."
python inference/run_baseline_with_cache_logging.py \
    --input "$INPUT_AUDIO" \
    --output "$OUTPUT_AUDIO" \
    --device cuda \
    --log-interval 100

EXIT_CODE=$?

echo ""
echo "=================================="
echo "Job completed with exit code: $EXIT_CODE"
echo "Finished: $(date)"
echo "=================================="

exit $EXIT_CODE
