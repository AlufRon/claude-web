#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# SLURM Job Submission Script
# Usage: ./submit_job.sh [config_file]
# 
# Examples:
#   ./submit_job.sh                        # Use default config
#   ./submit_job.sh example/moshi_7B.yaml  # Use specific config
# -----------------------------------------------------------------------------

set -e  # Exit on any error

# Configuration
DEFAULT_CONFIG="../../example/moshi_7B_memory_optimized.yaml"
CONFIG_FILE=${1:-$DEFAULT_CONFIG}

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Error: Config file '$CONFIG_FILE' not found!"
    echo "Available configs:"
    ls -la ../../example/*.yaml 2>/dev/null || echo "  No configs found in ../../example/"
    exit 1
fi

# Convert to absolute path (sbatch needs this because the SLURM script changes directory)
CONFIG_FILE=$(readlink -f "$CONFIG_FILE")

echo "🚀 Submitting Moshi TTT Training Job"
echo "===================================="
echo "📄 Config file: $CONFIG_FILE"
echo "📋 SLURM script: train_moshi_ttt.slurm"
echo ""

# Submit the job
JOB_ID=$(sbatch --export=YAML=$CONFIG_FILE train_moshi_ttt.slurm | awk '{print $4}')

echo "✅ Job submitted successfully!"
echo "🎯 Job ID: $JOB_ID"
echo ""
echo "📊 Monitor your job with:"
echo "  squeue -u $USER"
echo "  squeue -j $JOB_ID"
echo ""
echo "📄 Check logs with:"
echo "  tail -f /home/alufr/ttt_tests/moshi-finetune/logs/training/moshi_ttt.$JOB_ID.log"
echo "  tail -f /home/alufr/ttt_tests/moshi-finetune/logs/training/moshi_ttt.$JOB_ID.err"
echo ""
echo "❌ Cancel job with:"
echo "  scancel $JOB_ID"