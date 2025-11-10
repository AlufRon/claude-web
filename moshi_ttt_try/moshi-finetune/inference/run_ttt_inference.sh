#!/bin/bash
# Simple wrapper script for TTT inference

CHECKPOINT_DIR=/sise/eliyanac-group/ron_al/dailytalk_finetune_from_librilight5/checkpoints/checkpoint_000500/consolidated
INPUT_AUDIO="${1:-/sise/eliyanac-group/ron_al/seamless_interaction/daily_format_output/47.wav}"
OUTPUT_AUDIO="${2:-./output_training_sample.wav}"

echo "Running TTT Inference..."
echo "Checkpoint: $CHECKPOINT_DIR"
echo "Input: $INPUT_AUDIO"
echo "Output: $OUTPUT_AUDIO"
echo ""

python run_inference_with_ttt.py \
    --checkpoint "$CHECKPOINT_DIR" \
    --hf-repo kyutai/moshiko-pytorch-bf16 \
    "$INPUT_AUDIO" \
    "$OUTPUT_AUDIO"
