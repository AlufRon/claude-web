#!/bin/bash
#SBATCH --job-name=ttt_figure5
#SBATCH --output=/home/alufr/ttt_tests/moshi-finetune/logs/evaluation/ttt_figure5_%j.log
#SBATCH --error=/home/alufr/ttt_tests/moshi-finetune/logs/evaluation/ttt_figure5_%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

# TTT Figure 5 Diagnostic Job
# This runs inference and generates Figure 5 plots to diagnose TTT learning

set -e

echo "🔬 TTT Figure 5 Diagnostic Analysis"
echo "====================================="

# Activate conda environment
source ~/.bashrc
conda activate moshi_ttt_fixed

# Print environment info
echo "🐍 Active conda environment: $(conda info --envs | grep '*' | awk '{print $1}')"
echo "🐍 Python path: $(which python)"
echo "🐍 Python version: $(python --version)"

# GPU diagnostics
echo "🔧 GPU Diagnostics:"
echo "🔧 Hostname: $(hostname)"
echo "🔧 SLURM_JOB_ID: $SLURM_JOB_ID"
echo "🔧 CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "🔧 nvidia-smi output:"
nvidia-smi

# Configuration
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/sise/eliyanac-group/ron_al/librilight_ttt_pretrain_fixed_weights2/checkpoints/checkpoint_033000/consolidated/}"
INPUT_AUDIO="${INPUT_AUDIO:-/sise/eliyanac-group/ron_al/examples/combined_-6632801910088531808_24000hz.wav}"
OUTPUT_DIR="${OUTPUT_DIR:-./figure5_diagnostics}"
HF_REPO="${HF_REPO:-kyutai/moshiko-pytorch-bf16}"
MAX_T="${MAX_T:-2048}"
TTT_LAYERS="${TTT_LAYERS:-29 30 31}"

echo ""
echo "📋 Configuration:"
echo "📁 Checkpoint: $CHECKPOINT_DIR"
echo "🎤 Input audio: $INPUT_AUDIO"
echo "📂 Output dir: $OUTPUT_DIR"
echo "🤗 HF repo: $HF_REPO"
echo "📊 Max T: $MAX_T"
echo "🎯 TTT layers: $TTT_LAYERS"
echo ""

# Change to moshi-finetune directory
cd /home/alufr/ttt_tests/moshi-finetune

# Add current directory to PYTHONPATH for module imports
export PYTHONPATH="/home/alufr/ttt_tests/moshi-finetune:$PYTHONPATH"

# Run Figure 5 diagnostic
echo "🚀 Starting Figure 5 diagnostic inference..."
python inference/run_inference_with_figure5.py \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --input-audio "$INPUT_AUDIO" \
    --output-dir "$OUTPUT_DIR" \
    --hf-repo "$HF_REPO" \
    --max-T "$MAX_T" \
    --ttt-layers $TTT_LAYERS

echo ""
echo "✅ Figure 5 diagnostic complete!"
echo "📊 Check results in: $OUTPUT_DIR"
echo "   - ttt_diagnostic_report.txt (text analysis)"
echo "   - figure5_layer*.png (plots)"
echo "   - output_with_figure5.wav (generated audio)"
