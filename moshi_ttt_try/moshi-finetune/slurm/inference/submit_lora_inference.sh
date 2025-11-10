#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Submit LoRA Moshi Inference
# Usage: ./submit_lora_inference.sh [checkpoint_dir] [input_audio] [output_audio] [hf_repo]
#
# Examples:
#   ./submit_lora_inference.sh /path/to/checkpoint input.wav output.wav
#   ./submit_lora_inference.sh /path/to/checkpoint input.wav output.wav kyutai/moshiko-pytorch-bf16
#   ./submit_lora_inference.sh   # Use defaults
# -----------------------------------------------------------------------------

set -e

# Default values
DEFAULT_CHECKPOINT="/sise/eliyanac-group/ron_al/dailytalk_finetune_from_librilight2255lora122/checkpoints/checkpoint_025000/consolidated/"
DEFAULT_INPUT="/sise/eliyanac-group/ron_al/examples/combined_2673329117028914160_24000hz.wav"
DEFAULT_OUTPUT="./output_lora.wav"
DEFAULT_HF_REPO="kyutai/moshiko-pytorch-bf16"

# Get parameters from command line or use defaults
CHECKPOINT_DIR=${1:-$DEFAULT_CHECKPOINT}
INPUT_AUDIO=${2:-$DEFAULT_INPUT}
OUTPUT_AUDIO=${3:-$DEFAULT_OUTPUT}
HF_REPO=${4:-$DEFAULT_HF_REPO}

# Check if checkpoint directory exists
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "❌ Error: Checkpoint directory '$CHECKPOINT_DIR' not found!"
    exit 1
fi

# Check if lora.safetensors exists
if [ ! -f "$CHECKPOINT_DIR/lora.safetensors" ]; then
    echo "❌ Error: lora.safetensors not found in '$CHECKPOINT_DIR'!"
    exit 1
fi

# Check if input audio exists
if [ ! -f "$INPUT_AUDIO" ]; then
    echo "❌ Error: Input audio file '$INPUT_AUDIO' not found!"
    exit 1
fi

echo "🚀 Submitting LoRA Moshi Inference"
echo "=============================================="
echo "📁 Checkpoint: $CHECKPOINT_DIR"
echo "🎤 Input audio: $INPUT_AUDIO"
echo "🔊 Output audio: $OUTPUT_AUDIO"
echo "🤗 HF repo: $HF_REPO"
echo "📋 SLURM script: run_lora_inference.slurm"
echo ""

# Export variables to environment
export CHECKPOINT_DIR
export INPUT_AUDIO
export OUTPUT_AUDIO
export HF_REPO

# Submit the job with all environment variables
JOB_ID=$(sbatch \
    --export=ALL \
    run_lora_inference.slurm | awk '{print $4}')

echo "✅ Job submitted successfully!"
echo "🎯 Job ID: $JOB_ID"
echo ""
echo "📊 Monitor your job with:"
echo "  squeue -u $USER"
echo "  squeue -j $JOB_ID"
echo ""
echo "📄 Check logs with:"
echo "  tail -f /home/alufr/ttt_tests/moshi-finetune/logs/inference/moshi_lora.$JOB_ID.log"
echo "  tail -f /home/alufr/ttt_tests/moshi-finetune/logs/inference/moshi_lora.$JOB_ID.err"
echo ""
echo "❌ Cancel job with:"
echo "  scancel $JOB_ID"
echo ""
echo "🔊 Output will be saved to: $OUTPUT_AUDIO"
