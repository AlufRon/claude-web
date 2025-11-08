#!/bin/bash
# Quick Start Script for TTT Dataset Preprocessing
# This script demonstrates the complete preprocessing workflow on a small test set

set -e  # Exit on error

echo "=========================================="
echo "TTT Dataset Preprocessing - Quick Start"
echo "=========================================="

# Configuration
DATASET_NAME="SALT-Research/DeepDialogue-xtts"
RAW_DATA_DIR="./deepdialogue-xtts-sample"
PROCESSED_DIR="./processed_data_test"
NUM_TEST_DIALOGUES=10

echo ""
echo "[1/6] Checking dependencies..."
python -c "import torch, torchaudio, transformers; print('✅ All dependencies installed')" || {
    echo "❌ Missing dependencies. Installing..."
    pip install torch torchaudio transformers datasets tqdm
}

echo ""
echo "[2/6] Downloading sample dataset ($NUM_TEST_DIALOGUES dialogues)..."
if [ ! -d "$RAW_DATA_DIR" ]; then
    echo "For this quick start, we'll simulate the DeepDialogue-xtts structure."
    echo "To process the full dataset, download it from HuggingFace:"
    echo "  huggingface-cli download $DATASET_NAME --repo-type dataset --local-dir ./deepdialogue-xtts"
    echo ""
    echo "Creating test data structure..."
    python ../examples/create_test_data.py --output_dir "$RAW_DATA_DIR" --num_dialogues "$NUM_TEST_DIALOGUES"
else
    echo "✅ Sample data already exists at $RAW_DATA_DIR"
fi

echo ""
echo "[3/6] Preprocessing dataset..."
python ../deepdialogue_preprocessor.py \
    --dataset_dir "$RAW_DATA_DIR" \
    --audio_dir "$RAW_DATA_DIR/segments" \
    --output_dir "$PROCESSED_DIR" \
    --tokenizer meta-llama/Llama-3.1-8B-Instruct \
    --max_dialogues "$NUM_TEST_DIALOGUES" \
    --format pt

echo ""
echo "[4/6] Validating processed data..."
python ../validate_ttt_data.py \
    --data_dir "$PROCESSED_DIR" \
    --output "$PROCESSED_DIR/validation_report.json"

if [ $? -eq 0 ]; then
    echo "✅ Validation passed!"
else
    echo "❌ Validation failed. Check $PROCESSED_DIR/validation_report.json"
    exit 1
fi

echo ""
echo "[5/6] Testing dataset loader..."
python ../ttt_dataset.py \
    --data_dir "$PROCESSED_DIR" \
    --curriculum_stage 8k \
    --num_samples 3

echo ""
echo "[6/6] Checking preprocessing statistics..."
python -c "
import json
with open('$PROCESSED_DIR/preprocessing_stats.json') as f:
    stats = json.load(f)
print('')
print('='*60)
print('PREPROCESSING STATISTICS')
print('='*60)
print(f'Dialogues processed: {stats[\"total_dialogues\"]}')
print(f'Total turns: {stats[\"total_turns\"]}')
print(f'Total tokens: {stats[\"total_tokens\"]:,}')
print(f'Audio duration: {stats[\"total_audio_duration_sec\"]/3600:.2f} hours')
print(f'Avg tokens/turn: {stats[\"total_tokens\"]/stats[\"total_turns\"]:.1f}')
print('='*60)
"

echo ""
echo "=========================================="
echo "✅ QUICK START COMPLETE!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Review processed data in: $PROCESSED_DIR"
echo "  2. Check validation report: $PROCESSED_DIR/validation_report.json"
echo "  3. To process full dataset:"
echo "     - Download: huggingface-cli download $DATASET_NAME --repo-type dataset --local-dir ./deepdialogue-xtts"
echo "     - Process: python deepdialogue_preprocessor.py --dataset_dir ./deepdialogue-xtts --audio_dir ./deepdialogue-xtts/segments --output_dir ./processed_data_full"
echo "  4. Implement TTT modules (see docs/TRAINING_STRATEGY_ANALYSIS.md)"
echo "  5. Begin training with curriculum learning (8k → 16k → 32k → 64k)"
echo ""
