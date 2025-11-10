#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Submit Baseline Moshi Inference (No TTT)
# Usage: ./submit_baseline_inference.sh [input_audio] [output_audio] [hf_repo] [repetition_penalty] [repetition_window]
#
# Examples:
#   ./submit_baseline_inference.sh input.wav output.wav
#   ./submit_baseline_inference.sh input.wav output.wav kyutai/moshiko-pytorch-bf16 1.15 64
#   ./submit_baseline_inference.sh   # Use defaults
#
# Note: Repetition penalty uses default 1.0 (disabled) for baseline compatibility.
#       To enable, pass values like: ./submit_baseline_inference.sh input.wav output.wav kyutai/moshiko-pytorch-bf16 1.15 64
# -----------------------------------------------------------------------------

set -e

# Default values
DEFAULT_INPUT="/sise/eliyanac-group/ron_al/examples/combined_199561851133275930_24000hz.wav"
DEFAULT_OUTPUT="./output_baseline.wav"
DEFAULT_HF_REPO="kyutai/moshiko-pytorch-bf16"
DEFAULT_REPETITION_PENALTY="1.3"
DEFAULT_REPETITION_WINDOW="64"

# Get parameters from command line or use defaults
INPUT_AUDIO=${1:-$DEFAULT_INPUT}
OUTPUT_AUDIO=${2:-$DEFAULT_OUTPUT}
HF_REPO=${3:-$DEFAULT_HF_REPO}
REPETITION_PENALTY=${4:-$DEFAULT_REPETITION_PENALTY}
REPETITION_WINDOW=${5:-$DEFAULT_REPETITION_WINDOW}

# Check if input audio exists
if [ ! -f "$INPUT_AUDIO" ]; then
    echo "❌ Error: Input audio file '$INPUT_AUDIO' not found!"
    exit 1
fi

echo "🚀 Submitting Baseline Moshi Inference (No TTT)"
echo "=============================================="
echo "🎤 Input audio: $INPUT_AUDIO"
echo "🔊 Output audio: $OUTPUT_AUDIO"
echo "🤗 HF repo: $HF_REPO"
echo "🔄 Repetition penalty: $REPETITION_PENALTY"
echo "📏 Repetition window: $REPETITION_WINDOW"
echo "📋 SLURM script: run_baseline_inference.slurm"
echo ""

# Export variables to environment
export INPUT_AUDIO
export OUTPUT_AUDIO
export HF_REPO
export REPETITION_PENALTY
export REPETITION_WINDOW

# Submit the job with all environment variables
JOB_ID=$(sbatch \
    --export=ALL \
    run_baseline_inference.slurm | awk '{print $4}')

echo "✅ Job submitted successfully!"
echo "🎯 Job ID: $JOB_ID"
echo ""
echo "📊 Monitor your job with:"
echo "  squeue -u $USER"
echo "  squeue -j $JOB_ID"
echo ""
echo "📄 Check logs with:"
echo "  tail -f /home/alufr/ttt_tests/moshi-finetune/logs/inference/moshi_baseline.$JOB_ID.log"
echo "  tail -f /home/alufr/ttt_tests/moshi-finetune/logs/inference/moshi_baseline.$JOB_ID.err"
echo ""
echo "❌ Cancel job with:"
echo "  scancel $JOB_ID"
echo ""
echo "🔊 Output will be saved to: $OUTPUT_AUDIO"
