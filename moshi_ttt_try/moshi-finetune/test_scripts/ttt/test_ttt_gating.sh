#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Test TTT by forcing gating alpha to specific value
# Usage: ./test_ttt_gating.sh [gating_alpha]
#
# Examples:
#   ./test_ttt_gating.sh 0.5   # Use 50% TTT, 50% base Moshi
#   ./test_ttt_gating.sh 1.0   # Use 100% TTT only
#   ./test_ttt_gating.sh 0.0   # Use 0% TTT (base Moshi only)
# -----------------------------------------------------------------------------

set -e

# Get gating alpha from argument or use default
GATING_ALPHA=${1:-0.5}
CHECKPOINT=${2:-/sise/eliyanac-group/ron_al/librilight_ttt_pretrain_fixed_weights2/checkpoints/checkpoint_033000/consolidated/}
INPUT=${3:-/sise/eliyanac-group/ron_al/examples/combined_-6632801910088531808_24000hz.wav}
OUTPUT=${4:-./output_gating_${GATING_ALPHA}.wav}

echo "🔬 Testing TTT with Forced Gating Alpha"
echo "========================================"
echo "🎛️  Gating Alpha: $GATING_ALPHA"
echo "📁 Checkpoint: $CHECKPOINT"
echo "🎤 Input: $INPUT"
echo "🔊 Output: $OUTPUT"
echo ""
echo "What this means:"
echo "  • 0.0 = Use only base Moshi (0% TTT)"
echo "  • 0.5 = Blend 50% TTT + 50% base"
echo "  • 1.0 = Use only TTT output (100% TTT)"
echo ""

python run_inference_with_ttt.py \
    --checkpoint "$CHECKPOINT" \
    --hf-repo kyutai/moshiko-pytorch-bf16 \
    --force-gating-alpha $GATING_ALPHA \
    "$INPUT" \
    "$OUTPUT"

echo ""
echo "✅ Done! Output saved to: $OUTPUT"
echo ""
echo "Try different alpha values to compare:"
echo "  ./test_ttt_gating.sh 0.0   # Base Moshi only"
echo "  ./test_ttt_gating.sh 0.3   # 30% TTT (original training target)"
echo "  ./test_ttt_gating.sh 0.5   # 50/50 blend"
echo "  ./test_ttt_gating.sh 1.0   # TTT only (see what TTT learned)"
