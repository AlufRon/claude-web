#!/bin/bash
# Submit Figure 5 diagnostic job
# Usage: ./submit_figure5_diagnostic.sh [checkpoint_dir] [input_audio]

CHECKPOINT_DIR="${1:-/sise/eliyanac-group/ron_al/dailytalk_finetune_from_librilight5/checkpoints/checkpoint_006800/consolidated}"
INPUT_AUDIO="${2:-/sise/eliyanac-group/ron_al/examples/combined_-6632801910088531808_24000hz.wav}"
OUTPUT_DIR="${3:-./figure5_diagnostics}"

echo "🔬 Submitting Figure 5 Diagnostic Job"
echo "======================================"
echo "📁 Checkpoint: $CHECKPOINT_DIR"
echo "🎤 Input: $INPUT_AUDIO"
echo "📂 Output: $OUTPUT_DIR"
echo ""

sbatch \
    --export=ALL,CHECKPOINT_DIR="$CHECKPOINT_DIR",INPUT_AUDIO="$INPUT_AUDIO",OUTPUT_DIR="$OUTPUT_DIR" \
    run_figure5_diagnostic.slurm

echo ""
echo "✅ Job submitted! Monitor with: tail -f /home/alufr/ttt_tests/moshi-finetune/logs/evaluation/ttt_figure5_*.log"
echo "📊 Results will be in: $OUTPUT_DIR"
