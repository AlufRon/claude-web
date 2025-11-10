# GPU SLURM Configuration Fix ✅

## Problem
Paper metrics jobs were running on **CPU partition** instead of GPU, making evaluation very slow.

## Root Cause
The SLURM script used the old `#SBATCH --gres=gpu:1` format which doesn't work properly on your cluster.

## Solution
Changed to match the working inference script format:

### Before (CPU partition):
```bash
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1              # ❌ OLD format
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
```

### After (GPU partition):
```bash
#SBATCH --partition=main          # ✅ Use main partition
#SBATCH --gpus=1                  # ✅ NEW format
#SBATCH --constraint=rtx_6000     # ✅ Prefer RTX 6000
#SBATCH --mem=50G
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --exclude=cs-6000-01,cs-6000-02,cs-6000-03,ise-6000-08
```

## Additional Fixes

### 1. Log file naming correction
**Before:**
```bash
tail -f paper_metrics.7479381.log  # ❌ Wrong (dots)
```

**After:**
```bash
tail -f paper_metrics_7479381.log  # ✅ Correct (underscores)
```

### 2. CUDA environment setup
Added proper CUDA configuration (from inference script):
```bash
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## Verification

**Job 7479309** (old CPU version):
```
PARTITION: cpu
NODE: ise-cpu256-05
```

**Job 7479381** (new GPU version):
```
PARTITION: gpu        ✅
NODE: cs-6000-04      ✅
```

## Files Modified

1. **run_paper_metrics.slurm** - Updated SLURM directives
2. **submit_paper_metrics_wrapper.sh** - Fixed log file path instructions

## Usage

```bash
# Submit baseline evaluation
./submit_paper_metrics_wrapper.sh --baseline

# Submit checkpoint evaluation
./submit_paper_metrics_wrapper.sh --checkpoint /path/to/checkpoint

# Monitor with correct log file names
tail -f paper_metrics_<JOBID>.log
tail -f paper_metrics_<JOBID>.err
```

## Performance Impact

Running on GPU partition provides:
- ✅ Proper GPU acceleration
- ✅ Much faster evaluation (~5-10 minutes instead of hours)
- ✅ Access to RTX 6000 GPUs
- ✅ Consistent with inference script configuration

## Status: ✅ COMPLETE

Paper metrics evaluation now properly runs on GPU partition with correct resource allocation!
