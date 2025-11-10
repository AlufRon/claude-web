#!/usr/bin/env bash
#SBATCH --partition=main                  # use main partition
#SBATCH --job-name=moshi_lora        # job name
#SBATCH --output=/home/alufr/ttt_tests/moshi-finetune/logs/inference/moshi_lora.%j.log   # STDOUT → logs/inference/moshi_lora.<JOBID>.log
#SBATCH --error=/home/alufr/ttt_tests/moshi-finetune/logs/inference/moshi_lora.%j.err    # STDERR → logs/inference/moshi_lora.<JOBID>.err
#SBATCH --ntasks=1                        # one task/process
#SBATCH --cpus-per-task=4                 # CPU cores
#SBATCH --gpus=1                          # 1 GPU
#SBATCH --constraint=rtx_6000             # prefer RTX 6000
#SBATCH --mem=50G                         # host RAM
#SBATCH --time=2:00:00                    # walltime hh:mm:ss (2 hours)
#SBATCH --exclude=cs-6000-01,cs-6000-02,cs-6000-03,cs-6000-04,ise-6000-08,ise-6000-07   # exclude problematic nodes

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

# Verify environment
echo "🐍 Active conda environment: $CONDA_DEFAULT_ENV"
echo "🐍 Python path: $(which python)"
echo "🐍 Python version: $(python --version)"

# GPU Diagnostics
echo "🔧 GPU Diagnostics:"
echo "🔧 Hostname: $(hostname)"
echo "🔧 SLURM_JOB_ID: $SLURM_JOB_ID"
echo "🔧 CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Test nvidia-smi
if command -v nvidia-smi &> /dev/null; then
    echo "🔧 nvidia-smi output:"
    nvidia-smi || echo "❌ nvidia-smi failed"
else
    echo "❌ nvidia-smi not found"
fi

# Get parameters from environment variables (set by submit script)
CHECKPOINT_DIR=${CHECKPOINT_DIR}
INPUT_AUDIO=${INPUT_AUDIO}
OUTPUT_AUDIO=${OUTPUT_AUDIO}
HF_REPO=${HF_REPO:-kyutai/moshiko-pytorch-bf16}

# Change to moshi-finetune directory
cd /home/alufr/ttt_tests/moshi-finetune

# Add Moshi to PYTHONPATH
export PYTHONPATH="/home/alufr/ttt_tests/moshi:$PYTHONPATH"

echo "🚀 Starting Moshi LoRA Inference"
echo "=============================="
echo "📁 Checkpoint: $CHECKPOINT_DIR"
echo "🎤 Input audio: $INPUT_AUDIO"
echo "🔊 Output audio: $OUTPUT_AUDIO"
echo "🤗 HF repo: $HF_REPO"
echo ""

# Verify checkpoint exists
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "❌ Error: Checkpoint directory '$CHECKPOINT_DIR' not found!"
    exit 1
fi

# Verify input audio exists
if [ ! -f "$INPUT_AUDIO" ]; then
    echo "❌ Error: Input audio file '$INPUT_AUDIO' not found!"
    exit 1
fi

# Run LoRA inference using standard moshi.run_inference
echo "🏃 Starting LoRA inference..."
python -m moshi.run_inference \
    --config "$CHECKPOINT_DIR/config.json" \
    --lora-weight "$CHECKPOINT_DIR/lora.safetensors" \
    --hf-repo "$HF_REPO" \
    --device cuda \
    --batch-size 1 \
    "$INPUT_AUDIO" \
    "$OUTPUT_AUDIO"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ Inference completed successfully!"
    echo "🔊 Output saved to: $OUTPUT_AUDIO"
else
    echo ""
    echo "❌ Inference failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi
