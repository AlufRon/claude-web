#!/bin/bash
# Inference script for LoRA checkpoints (no TTT)

CHECKPOINT_DIR=/sise/eliyanac-group/ron_al/librilight_lora_pretrain_fixed_weights2/checkpoints/checkpoint_003000/consolidated/
INPUT_AUDIO="${1:-/sise/eliyanac-group/ron_al/librilight/extracted_medium/medium/100/emerald_city_librivox_64kb_mp3/emeraldcity_30_baum_64kb.flac}"
OUTPUT_AUDIO="${2:-./output_lora_sample.wav}"

echo "Running LoRA Inference..."
echo "Checkpoint: $CHECKPOINT_DIR"
echo "Input: $INPUT_AUDIO"
echo "Output: $OUTPUT_AUDIO"
echo ""

# Use standard Moshi inference - it will:
# 1. Load base model from HuggingFace
# 2. Apply LoRA adapters from checkpoint
# 3. Use config from checkpoint

cd /home/alufr/ttt_tests/moshi/moshi

python -m moshi.run_inference \
    --hf-repo kyutai/moshiko-pytorch-bf16 \
    --config "$CHECKPOINT_DIR/config.json" \
    --moshi-weight "$CHECKPOINT_DIR/lora.safetensors" \
    "$INPUT_AUDIO" \
    "$OUTPUT_AUDIO"
