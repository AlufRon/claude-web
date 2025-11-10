#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Submit TTT Gating Alpha Test to SLURM
# Usage: ./submit_gating_test.sh [gating_alpha] [checkpoint_dir] [input_audio]
#
# Examples:
#   ./submit_gating_test.sh 1.0    # Test pure TTT (100%)
#   ./submit_gating_test.sh 0.0    # Test pure base Moshi (0% TTT)
#   ./submit_gating_test.sh 0.5    # Test 50/50 blend
#   ./submit_gating_test.sh 0.3    # Test original training target
# -----------------------------------------------------------------------------

set -e  # Exit on any error

# Get gating alpha from argument or use default
GATING_ALPHA=${1:-1.0}

# Default values
DEFAULT_CHECKPOINT=/sise/eliyanac-group/ron_al/librilight_ttt_pretrain_fixed_weights2/checkpoints/checkpoint_033000/consolidated/
DEFAULT_INPUT="/sise/eliyanac-group/ron_al/examples/combined_-6632801910088531808_24000hz.wav"
DEFAULT_HF_REPO="kyutai/moshiko-pytorch-bf16"

CHECKPOINT_DIR=${2:-$DEFAULT_CHECKPOINT}
INPUT_AUDIO=${3:-$DEFAULT_INPUT}
OUTPUT_AUDIO="./output_gating_${GATING_ALPHA}.wav"
HF_REPO=${4:-$DEFAULT_HF_REPO}

# Validate gating alpha is a number between 0 and 1
if ! [[ "$GATING_ALPHA" =~ ^[0-9]*\.?[0-9]+$ ]] || (( $(echo "$GATING_ALPHA < 0" | bc -l) )) || (( $(echo "$GATING_ALPHA > 1" | bc -l) )); then
    echo "❌ Error: Gating alpha must be a number between 0.0 and 1.0"
    echo "   You provided: $GATING_ALPHA"
    exit 1
fi

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

echo "🔬 Submitting TTT Gating Alpha Test"
echo "===================================="
echo "🎛️  Gating Alpha: $GATING_ALPHA"
echo "📁 Checkpoint: $CHECKPOINT_DIR"
echo "🎤 Input audio: $INPUT_AUDIO"
echo "🔊 Output audio: $OUTPUT_AUDIO"
echo "🤗 HF repo: $HF_REPO"
echo "📋 SLURM script: test_gating_alpha.slurm"
echo ""
echo "What this test means:"
echo "  • 0.0 = Pure base Moshi (0% TTT) - baseline"
echo "  • 0.3 = Original training target (30% TTT)"
echo "  • 0.5 = Balanced blend (50% TTT)"
echo "  • 1.0 = Pure TTT (100%) - see what TTT learned"
echo ""

# Submit the job with environment variables
JOB_ID=$(sbatch \
    --export=GATING_ALPHA="$GATING_ALPHA",CHECKPOINT_DIR="$CHECKPOINT_DIR",INPUT_AUDIO="$INPUT_AUDIO",OUTPUT_AUDIO="$OUTPUT_AUDIO",HF_REPO="$HF_REPO" \
    test_gating_alpha.slurm | awk '{print $4}')

echo "✅ Job submitted successfully!"
echo "🎯 Job ID: $JOB_ID"
echo ""
echo "📊 Monitor your job with:"
echo "  squeue -u $USER"
echo "  squeue -j $JOB_ID"
echo ""
echo "📄 Check logs with:"
echo "  tail -f /home/alufr/ttt_tests/moshi-finetune/logs/training/moshi_gating_$JOB_ID.log"
echo "  tail -f /home/alufr/ttt_tests/moshi-finetune/logs/training/moshi_gating_$JOB_ID.err"
echo ""
echo "❌ Cancel job with:"
echo "  scancel $JOB_ID"
echo ""
echo "🔊 Output will be saved to: $OUTPUT_AUDIO"
echo ""
echo "💡 Recommended test sequence:"
echo "  1. ./submit_gating_test.sh 0.0   # Baseline (pure Moshi)"
echo "  2. ./submit_gating_test.sh 1.0   # Pure TTT (what did TTT learn?)"
echo "  3. ./submit_gating_test.sh 0.3   # Original training target"
echo "  4. ./submit_gating_test.sh 0.5   # Balanced blend"
