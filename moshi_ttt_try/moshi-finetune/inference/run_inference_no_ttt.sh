#!/bin/bash
# Simple wrapper script for standard Moshi inference (no TTT)

CHECKPOINT_DIR=/sise/eliyanac-group/ron_al/librilight_frozen_pretrain_fixed_weights/checkpoints/checkpoint_015000/consolidated/
INPUT_AUDIO="${1:-/sise/eliyanac-group/ron_al/seamless_interaction/daily_format_output/0.wav}"
OUTPUT_AUDIO="${2:-./output_no_ttt.wav}"

echo "Running Standard Moshi Inference (no TTT)..."
echo "Checkpoint: $CHECKPOINT_DIR"
echo "Input: $INPUT_AUDIO"
echo "Output: $OUTPUT_AUDIO"
echo ""

# Use the standard Moshi inference script with your checkpoint
python -m moshi.run_inference \
    --hf-repo kyutai/moshiko-pytorch-bf16 \
    --moshi-weight "$CHECKPOINT_DIR/model.safetensors" \
    --config "$CHECKPOINT_DIR/config.json" \
    "$INPUT_AUDIO" \
    "$OUTPUT_AUDIO"
