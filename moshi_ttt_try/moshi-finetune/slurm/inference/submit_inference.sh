#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# SLURM Inference Job Submission Script
# Usage: ./submit_inference.sh [checkpoint_dir] [input_audio] [output_audio] [hf_repo] [repetition_penalty] [repetition_window]
#
# Examples:
#   ./submit_inference.sh /path/to/checkpoint input.wav output.wav
#   ./submit_inference.sh /path/to/checkpoint input.wav output.wav kyutai/moshiko-pytorch-bf16 1.15 64
#   ./submit_inference.sh   # Use defaults
# -----------------------------------------------------------------------------

set -e  # Exit on any error

# Default values
DEFAULT_CHECKPOINT=/sise/eliyanac-group/ron_al/dailytalk_finetune_from_librilight6/checkpoints/checkpoint_002000/consolidated
DEFAULT_INPUT="/sise/eliyanac-group/ron_al/examples/combined_2198538934964547889_24000hz.wav"
DEFAULT_OUTPUT="./output_training_sample.wav"
DEFAULT_HF_REPO="kyutai/moshiko-pytorch-bf16"
DEFAULT_REPETITION_PENALTY="1.0"
DEFAULT_REPETITION_WINDOW="64"

# Get parameters from command line or use defaults
CHECKPOINT_DIR=${1:-$DEFAULT_CHECKPOINT}
INPUT_AUDIO=${2:-$DEFAULT_INPUT}
OUTPUT_AUDIO=${3:-$DEFAULT_OUTPUT}
HF_REPO=${4:-$DEFAULT_HF_REPO}
REPETITION_PENALTY=${5:-$DEFAULT_REPETITION_PENALTY}
REPETITION_WINDOW=${6:-$DEFAULT_REPETITION_WINDOW}

# Check if checkpoint exists
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "❌ Error: Checkpoint directory '$CHECKPOINT_DIR' not found!"
    echo ""
    echo "Available checkpoints in /sise/eliyanac-group/ron_al/:"
    find /sise/eliyanac-group/ron_al -type d -name "checkpoint_*" 2>/dev/null | head -10 || echo "  (none found)"
    exit 1
fi

# Check if input audio exists
if [ ! -f "$INPUT_AUDIO" ]; then
    echo "❌ Error: Input audio file '$INPUT_AUDIO' not found!"
    exit 1
fi

echo "🚀 Submitting Moshi TTT Inference Job"
echo "====================================="
echo "📁 Checkpoint: $CHECKPOINT_DIR"
echo "🎤 Input audio: $INPUT_AUDIO"
echo "🔊 Output audio: $OUTPUT_AUDIO"
echo "🤗 HF repo: $HF_REPO"
echo "🔄 Repetition penalty: $REPETITION_PENALTY"
echo "📏 Repetition window: $REPETITION_WINDOW"
echo "📋 SLURM script: run_inference_ttt.slurm"
echo ""

# Submit the job with environment variables
JOB_ID=$(sbatch \
    --export=CHECKPOINT_DIR="$CHECKPOINT_DIR",INPUT_AUDIO="$INPUT_AUDIO",OUTPUT_AUDIO="$OUTPUT_AUDIO",HF_REPO="$HF_REPO",REPETITION_PENALTY="$REPETITION_PENALTY",REPETITION_WINDOW="$REPETITION_WINDOW" \
    run_inference_ttt.slurm | awk '{print $4}')

echo "✅ Job submitted successfully!"
echo "🎯 Job ID: $JOB_ID"
echo ""
echo "📊 Monitor your job with:"
echo "  squeue -u $USER"
echo "  squeue -j $JOB_ID"
echo ""
echo "📄 Check logs with:"
echo "  tail -f /home/alufr/ttt_tests/moshi-finetune/logs/inference/moshi_inference.$JOB_ID.log"
echo "  tail -f /home/alufr/ttt_tests/moshi-finetune/logs/inference/moshi_inference.$JOB_ID.err"
echo ""
echo "❌ Cancel job with:"
echo "  scancel $JOB_ID"
echo ""
echo "🔊 Output will be saved to: $OUTPUT_AUDIO"
