#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# SLURM Inference Job with BALANCED Temperature (More Talkative but Coherent)
# Usage: ./submit_inference_balanced.sh [checkpoint_dir] [input_audio] [output_audio]
# -----------------------------------------------------------------------------

set -e  # Exit on any error

# Default values - WITH MODERATE TEMPERATURES FOR BALANCE
DEFAULT_CHECKPOINT="/sise/eliyanac-group/ron_al/dailytalk_finetune_from_librilight5/checkpoints/checkpoint_001600/consolidated"
DEFAULT_INPUT="/sise/eliyanac-group/ron_al/seamless_interaction/daily_format_output/47.wav"
DEFAULT_OUTPUT="./output_balanced.wav"
DEFAULT_HF_REPO="kyutai/moshiko-pytorch-bf16"

# Temperature settings (MODERATE increase for balance)
# Baseline: 0.8 audio, 0.9 text
# Talkative (1.2, 1.1) = too much gibberish
# Balanced: slight increase only
DEFAULT_TEMP="0.9"          # Audio temperature (was 0.8, trying 0.9 - only 12% increase)
DEFAULT_TEMP_TEXT="0.95"    # Text temperature (was 0.9, trying 0.95 - only 5% increase)

# Get parameters from command line or use defaults
CHECKPOINT_DIR=${1:-$DEFAULT_CHECKPOINT}
INPUT_AUDIO=${2:-$DEFAULT_INPUT}
OUTPUT_AUDIO=${3:-$DEFAULT_OUTPUT}
HF_REPO=${4:-$DEFAULT_HF_REPO}
TEMP=${5:-$DEFAULT_TEMP}
TEMP_TEXT=${6:-$DEFAULT_TEMP_TEXT}

# Check if checkpoint exists
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "❌ Error: Checkpoint directory '$CHECKPOINT_DIR' not found!"
    exit 1
fi

# Check if input audio exists
if [ ! -f "$INPUT_AUDIO" ]; then
    echo "❌ Error: Input audio file '$INPUT_AUDIO' not found!"
    exit 1
fi

echo "🚀 Submitting Moshi TTT Inference Job (BALANCED MODE)"
echo "===================================================="
echo "📁 Checkpoint: $CHECKPOINT_DIR"
echo "🎤 Input audio: $INPUT_AUDIO"
echo "🔊 Output audio: $OUTPUT_AUDIO"
echo "🤗 HF repo: $HF_REPO"
echo "🌡️  Audio temp: $TEMP (balanced: slightly more talkative)"
echo "🌡️  Text temp: $TEMP_TEXT (balanced: slightly more varied)"
echo "📋 SLURM script: run_inference_ttt_talkative.slurm"
echo ""

# Submit the job with environment variables
JOB_ID=$(sbatch \
    --export=CHECKPOINT_DIR="$CHECKPOINT_DIR",INPUT_AUDIO="$INPUT_AUDIO",OUTPUT_AUDIO="$OUTPUT_AUDIO",HF_REPO="$HF_REPO",TEMP="$TEMP",TEMP_TEXT="$TEMP_TEXT" \
    run_inference_ttt_talkative.slurm | awk '{print $4}')

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
echo ""
echo "ℹ️  Temperature Guide:"
echo "  • Too low (< 0.8): Model too quiet/conservative"
echo "  • Balanced (0.9-0.95): Good coherence, slightly more talkative"
echo "  • Too high (> 1.1): Gibberish/incoherent"
