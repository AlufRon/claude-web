#!/usr/bin/env bash
#SBATCH --partition=main                  # use main partition like working interactive
#SBATCH --job-name=moshi_ttt              # job name
#SBATCH --output=/home/alufr/ttt_tests/moshi-finetune/logs/training/moshi_ttt.%j.log         # STDOUT → logs/training/moshi_ttt.<JOBID>.log
#SBATCH --error=/home/alufr/ttt_tests/moshi-finetune/logs/training/moshi_ttt.%j.err          # STDERR → logs/training/moshi_ttt.<JOBID>.err
#SBATCH --ntasks=1                        # one task/process
#SBATCH --cpus-per-task=3                 # keep your original CPU count
#SBATCH --gpus=1                          # 1 GPU (like interactive)
#SBATCH --constraint=rtx_6000             # prefer RTX 6000, but allow fallback
#SBATCH --mem=50G                         # host RAM
#SBATCH --time=48:00:00                   # walltime hh:mm:ss (24 hours)
#SBATCH --exclude=cs-6000-04,cs-6000-02,cs-6000-01,cs-6000-03,ise-6000-04,ise-6000-07   # exclude nodes with CUDA/module/memory issues
#SBATCH --requeue                         # allow requeue on node failure

# pin to the first (and only) visible GPU
export CUDA_VISIBLE_DEVICES=0

# Fix CUDA memory fragmentation (prevents OOM errors during torch.compile)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Direct conda activation without module system (fixes SLURM env retrieval issues)
# Source conda initialization from system conda
source /etc/profile.d/conda.sh

# Activate environment directly
conda activate moshi_ttt_fixed

# Verify environment activation (critical)
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "moshi_ttt_fixed" ]; then
    echo "❌ ERROR: Conda environment activation failed!"
    echo "Expected: moshi_ttt_fixed"
    echo "Got: $CONDA_DEFAULT_ENV"
    exit 1
fi

# Force proper environment activation
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# Verify environment activation
echo "🐍 Active conda environment: $CONDA_DEFAULT_ENV"
echo "🐍 CONDA_PREFIX: $CONDA_PREFIX"
echo "🐍 Python path: $(which python)"
echo "🐍 Python version: $(python --version)"

# Test critical imports
echo "🐍 Testing critical imports..."
python -c "
try:
    import torch
    print(f'✅ PyTorch {torch.__version__} available')
except ImportError as e:
    print(f'❌ PyTorch import failed: {e}')
    exit(1)

try:
    import torch.distributed
    print('✅ torch.distributed available')
except ImportError as e:
    print(f'❌ torch.distributed import failed: {e}')

print(f'Python executable: {__import__(\"sys\").executable}')
" || {
    echo "❌ Critical imports failed - environment not properly activated"
    echo "🚨 Conda environment issue - cannot proceed"
    exit 1
}

# GPU Diagnostics
echo "🔧 GPU Diagnostics:"
echo "🔧 Hostname: $(hostname)"
echo "🔧 SLURM_JOB_ID: $SLURM_JOB_ID"
echo "🔧 SLURM_PROCID: $SLURM_PROCID"
echo "🔧 SLURM_LOCALID: $SLURM_LOCALID"
echo "🔧 SLURM_GPUS_ON_NODE: $SLURM_GPUS_ON_NODE"
echo "🔧 CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Test nvidia-smi
if command -v nvidia-smi &> /dev/null; then
    echo "🔧 nvidia-smi output:"
    nvidia-smi || echo "❌ nvidia-smi failed"
else
    echo "❌ nvidia-smi not found"
fi

# Test CUDA availability with Python (with timeout to prevent hanging)
echo "🔧 Testing CUDA availability..."
timeout 30 python -c "
import torch
import sys
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA device count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')

    # Test basic CUDA operation with simple tensor
    try:
        print('Testing basic CUDA operation...')
        x = torch.randn(10, 10, device='cuda')
        y = x * 2.0  # Simple operation to verify CUDA works
        print('✅ Basic CUDA operations work')
    except Exception as e:
        print(f'❌ CUDA operations failed: {e}')
        sys.exit(1)
else:
    print('❌ CUDA not available to PyTorch')
    sys.exit(1)
"

CUDA_TEST_EXIT=$?
if [ $CUDA_TEST_EXIT -eq 124 ]; then
    echo "❌ CUDA test timed out (30s) - node $(hostname) has CUDA issues"
    echo "🚨 Requesting job requeue to try different node"
    exit 1
elif [ $CUDA_TEST_EXIT -ne 0 ]; then
    echo "❌ Python CUDA test failed with exit code $CUDA_TEST_EXIT"
    echo "🚨 GPU not available on node $(hostname) - requesting job requeue"
    exit 1
fi

# Final GPU validation before training
echo "🔧 Final GPU validation..."
python -c "
import torch
assert torch.cuda.is_available(), 'CUDA must be available'
assert torch.cuda.device_count() > 0, 'At least one GPU required'
print('✅ GPU validation passed - proceeding with training')
" || {
    echo "❌ Final GPU validation failed"
    echo "🚨 Cannot proceed with training - requesting job requeue"
    exit 1
}

# allow overrides via sbatch --export
YAML=${YAML:-example/moshi_7B.yaml}
# Use dynamic port based on job ID to avoid conflicts
MASTER_PORT=${MASTER_PORT:-$((29000 + ($SLURM_JOB_ID % 1000)))}

# Change to our moshi-finetune directory where train.py is located
cd /home/alufr/ttt_tests/moshi-finetune

# Add current directory to PYTHONPATH for module imports
export PYTHONPATH="/home/alufr/ttt_tests/moshi-finetune:$PYTHONPATH"

echo "🚀 Starting Moshi TTT Training"
echo "=============================="
echo "📄 Config file: $YAML"
echo "📂 Working directory: $(pwd)"
echo "🔧 Master port: $MASTER_PORT"
echo ""

# Verify config file exists
if [ ! -f "$YAML" ]; then
    echo "❌ Error: Config file '$YAML' not found!"
    echo "Available configs:"
    ls -la example/*.yaml 2>/dev/null || echo "  No configs found in example/"
    exit 1
fi

# Run training
TORCHRUN=${TORCHRUN:-torchrun}
echo "🏃 Starting training with torchrun..."
$TORCHRUN \
  --nproc-per-node=1 \
  --master-port=$MASTER_PORT \
  train.py \
  --config $YAML

echo ""
echo "✅ Training completed!"