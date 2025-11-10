#!/usr/bin/env bash
#SBATCH --partition=main
#SBATCH --job-name=moshi_baseline
#SBATCH --output=/home/alufr/ttt_tests/moshi-finetune/logs/inference/moshi_baseline.%j.log
#SBATCH --error=/home/alufr/ttt_tests/moshi-finetune/logs/inference/moshi_baseline.%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --constraint=rtx_6000
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# Pin to the first (and only) visible GPU
export CUDA_VISIBLE_DEVICES=0

# Fix CUDA memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Load & activate conda environment
module load anaconda
eval "$(conda shell.bash hook)"
conda activate moshi_ttt_fixed

# Force proper environment activation
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# Print environment info
echo "🐍 Active conda environment: $CONDA_DEFAULT_ENV"
echo "🐍 Python path: $(which python)"
echo "🐍 Python version: $(python --version)"

# GPU diagnostics
echo "🔧 GPU Diagnostics:"
echo "🔧 Hostname: $(hostname)"
echo "🔧 SLURM_JOB_ID: $SLURM_JOB_ID"
echo "🔧 CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "🔧 nvidia-smi output:"
nvidia-smi

echo "🚀 Starting Baseline Moshi Inference (No TTT)"
echo "============================================="
echo "🎤 Input audio: $INPUT_AUDIO"
echo "🔊 Output audio: $OUTPUT_AUDIO"
echo "🤗 HF repo: $HF_REPO"
echo ""
echo "Note: Baseline uses standard moshi.run_inference"
echo "      Repetition penalty is built into LMGen with default 1.0 (disabled)"
echo "      To enable repetition penalty for baseline, modify moshi.run_inference"
echo ""

# Add Moshi to Python path - uses our modified Moshi with repetition penalty support
export PYTHONPATH="/home/alufr/ttt_tests/moshi:$PYTHONPATH"

# Change to moshi-finetune directory so output files are saved there
cd /home/alufr/ttt_tests/moshi-finetune

# Run baseline Moshi inference using the standard Moshi script
# Note: This uses our modified LMGen which defaults to repetition_penalty=1.0 (disabled)
# The baseline maintains backward compatibility while having the capability built-in
python -m moshi.run_inference \
    --hf-repo "$HF_REPO" \
    --device cuda \
    --batch-size 1 \
    "$INPUT_AUDIO" \
    "$OUTPUT_AUDIO"

EXIT_CODE=$?

echo ""
echo "============================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Baseline inference completed successfully!"
    echo "🔊 Output saved to: $OUTPUT_AUDIO"
else
    echo "❌ Baseline inference failed with exit code: $EXIT_CODE"
fi
echo "============================================="

exit $EXIT_CODE
