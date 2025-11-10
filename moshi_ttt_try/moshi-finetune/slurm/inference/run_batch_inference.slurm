#!/usr/bin/env bash
#SBATCH --partition=main                  # use main partition
#SBATCH --job-name=moshi_batch_inference  # job name
#SBATCH --output=/home/alufr/ttt_tests/moshi-finetune/logs/batch_inference/moshi_batch_inference.%j.log   # STDOUT → logs/batch_inference/moshi_batch_inference.<JOBID>.log
#SBATCH --error=/home/alufr/ttt_tests/moshi-finetune/logs/batch_inference/moshi_batch_inference.%j.err    # STDERR → logs/batch_inference/moshi_batch_inference.<JOBID>.err
#SBATCH --ntasks=1                        # one task/process
#SBATCH --cpus-per-task=4                 # CPU cores
#SBATCH --gpus=1                          # 1 GPU
#SBATCH --constraint=rtx_6000             # prefer RTX 6000
#SBATCH --mem=50G                         # host RAM
#SBATCH --time=24:00:00                    # walltime hh:mm:ss (24 hours for batch processing)
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
INPUT_FILES=${INPUT_FILES}
OUTPUT_DIR=${OUTPUT_DIR}
HF_REPO=${HF_REPO:-kyutai/moshiko-pytorch-bf16}
MAX_LENGTH=${MAX_LENGTH}
COMPUTE_PERPLEXITY=${COMPUTE_PERPLEXITY:-false}
GENERATE_AUDIO=${GENERATE_AUDIO:-true}
ENABLE_TTT_DIAGNOSTICS=${ENABLE_TTT_DIAGNOSTICS:-false}
DIAGNOSTIC_LOG_FREQUENCY=${DIAGNOSTIC_LOG_FREQUENCY:-100}

# Change to moshi-finetune directory
cd /home/alufr/ttt_tests/moshi-finetune

# Add current directory to PYTHONPATH for module imports
export PYTHONPATH="/home/alufr/ttt_tests/moshi-finetune:$PYTHONPATH"

echo "🚀 Starting Moshi TTT Batch Inference"
echo "===================================="
echo "📁 Checkpoint: $CHECKPOINT_DIR"
echo "🎤 Input files: $(echo $INPUT_FILES | wc -w) files"
echo "📂 Output directory: $OUTPUT_DIR"
echo "🤗 HF repo: $HF_REPO"
echo "📏 Max length: ${MAX_LENGTH:-"unlimited"}"
echo "📊 Compute perplexity: $COMPUTE_PERPLEXITY"
echo "🎵 Generate audio: $GENERATE_AUDIO"
echo "🔍 TTT Diagnostics: $ENABLE_TTT_DIAGNOSTICS (frequency: $DIAGNOSTIC_LOG_FREQUENCY)"
echo ""

# Verify checkpoint exists
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "❌ Error: Checkpoint directory '$CHECKPOINT_DIR' not found!"
    exit 1
fi

# Verify at least one input file exists
FIRST_FILE=$(echo $INPUT_FILES | awk '{print $1}')
if [ ! -f "$FIRST_FILE" ]; then
    echo "❌ Error: First input audio file '$FIRST_FILE' not found!"
    echo "   Full input list: $INPUT_FILES"
    exit 1
fi

# Create output directory and logs subdirectory
mkdir -p "$OUTPUT_DIR"
mkdir -p /home/alufr/ttt_tests/moshi-finetune/logs/batch_inference

# Build command arguments
CMD_ARGS=(
    "--checkpoint" "$CHECKPOINT_DIR"
    "--hf-repo" "$HF_REPO"
    "--output-dir" "$OUTPUT_DIR"
)

# Add input files as space-separated arguments
for file in $INPUT_FILES; do
    CMD_ARGS+=("--input")
    CMD_ARGS+=("$file")
done

# Add max length if specified
if [ -n "$MAX_LENGTH" ]; then
    CMD_ARGS+=("--max-length" "$MAX_LENGTH")
fi

# Add perplexity computation if requested
if [ "$COMPUTE_PERPLEXITY" = "true" ]; then
    CMD_ARGS+=("--compute-perplexity")
fi

# Add audio generation if requested
if [ "$GENERATE_AUDIO" = "true" ]; then
    CMD_ARGS+=("--generate-audio")
fi

# Add TTT diagnostics if requested
if [ "$ENABLE_TTT_DIAGNOSTICS" = "true" ]; then
    CMD_ARGS+=("--enable-ttt-diagnostics")
    CMD_ARGS+=("--diagnostic-log-frequency" "$DIAGNOSTIC_LOG_FREQUENCY")
fi

# Run batch inference
echo "🏃 Starting TTT batch inference..."
echo "🏃 Command: python inference/run_batch_inference.py ${CMD_ARGS[*]}"
echo ""

python inference/run_batch_inference.py "${CMD_ARGS[@]}"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ Batch inference completed successfully!"
    echo "📊 Results saved to: $OUTPUT_DIR"
    echo ""
    echo "📄 Files created:"
    if [ -d "$OUTPUT_DIR" ]; then
        ls -la "$OUTPUT_DIR" | grep -E "\.(json|pt)$" || echo "   No result files found"
    else
        echo "   Output directory not created"
    fi
else
    echo ""
    echo "❌ Batch inference failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi