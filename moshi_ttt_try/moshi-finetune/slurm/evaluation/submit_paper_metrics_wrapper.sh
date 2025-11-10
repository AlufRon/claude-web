#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# SLURM Paper Metrics Submission Script
# Usage: ./submit_paper_metrics_wrapper.sh [options]
#
# Examples:
#   ./submit_paper_metrics_wrapper.sh --baseline 50
#   ./submit_paper_metrics_wrapper.sh /path/to/checkpoint 100 config.yaml 5000
#   ./submit_paper_metrics_wrapper.sh /path/to/checkpoint 100 config.yaml skip
#   ./submit_paper_metrics_wrapper.sh   # Use defaults (TTT checkpoint)
# -----------------------------------------------------------------------------

set -e  # Exit on any error

# Default values
DEFAULT_CHECKPOINT="/sise/eliyanac-group/ron_al/dailytalk_finetune_from_librilight10/checkpoints/checkpoint_000100/consolidated/"
DEFAULT_MAX_SAMPLES=2000
DEFAULT_CONFIG="/home/alufr/ttt_tests/moshi-finetune/example/moshi_7B_multilayer_with_ttt.yaml"
DEFAULT_LIBRILIGHT_LENGTH=45000

# Parse arguments
if [ "$1" == "--baseline" ]; then
    MODE="baseline"
    CHECKPOINT_DIR=""
    MAX_SAMPLES=${2:-$DEFAULT_MAX_SAMPLES}
    CONFIG_FILE=${3:-$DEFAULT_CONFIG}
    LIBRILIGHT_LENGTH=${4:-$DEFAULT_LIBRILIGHT_LENGTH}
elif [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --baseline [max_samples] [config] [librilight_length]    Evaluate baseline Moshi (no TTT)"
    echo "  checkpoint_dir [max_samples] [config] [librilight_length] Evaluate TTT checkpoint"
    echo ""
    echo "LibriLight length options:"
    echo "  5000     Quick evaluation (~10-15 min total)"
    echo "  24000    Full evaluation (~30-40 min total)"
    echo "  skip     Skip LibriLight entirely (~5-10 min total)"
    echo ""
    echo "Examples:"
    echo "  $0 --baseline 50                                    # Baseline with LibriLight (5k)"
    echo "  $0 /path/to/checkpoint 100                          # TTT checkpoint with LibriLight (5k)"
    echo "  $0 /path/to/checkpoint 100 config.yaml 24000        # Full LibriLight (24k)"
    echo "  $0 /path/to/checkpoint 100 config.yaml skip         # Skip LibriLight"
    echo "  $0                                                  # Default: TTT checkpoint, LibriLight (5k)"
    exit 0
else
    MODE="checkpoint"
    CHECKPOINT_DIR=${1:-$DEFAULT_CHECKPOINT}
    MAX_SAMPLES=${2:-$DEFAULT_MAX_SAMPLES}
    CONFIG_FILE=${3:-$DEFAULT_CONFIG}
    LIBRILIGHT_LENGTH=${4:-$DEFAULT_LIBRILIGHT_LENGTH}
fi

# Validate checkpoint directory for non-baseline mode
if [ "$MODE" == "checkpoint" ] && [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "❌ Error: Checkpoint directory '$CHECKPOINT_DIR' not found!"
    echo ""
    echo "Available checkpoints:"
    find /sise/eliyanac-group/ron_al -type d -name "checkpoint_*" 2>/dev/null | head -10 || echo "  (none found)"
    exit 1
fi

# Validate config file
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Error: Config file '$CONFIG_FILE' not found!"
    exit 1
fi

echo "🚀 Submitting Paper Metrics Evaluation Job"
echo "==========================================="
if [ "$MODE" == "baseline" ]; then
    echo "📊 Mode: BASELINE (no TTT)"
else
    echo "📁 Checkpoint: $CHECKPOINT_DIR"
fi
echo "📈 Max samples: $MAX_SAMPLES per task"
echo "📄 Config: $CONFIG_FILE"
if [ "$LIBRILIGHT_LENGTH" == "skip" ]; then
    echo "🔬 LibriLight: SKIPPED"
else
    echo "🔬 LibriLight: $LIBRILIGHT_LENGTH tokens"
fi
echo ""

# Submit the job with appropriate arguments using submit_paper_metrics.sh
if [ "$MODE" == "baseline" ]; then
    JOB_ID=$(sbatch submit_paper_metrics.sh --baseline "$MAX_SAMPLES" "$CONFIG_FILE" "$LIBRILIGHT_LENGTH" | awk '{print $4}')
else
    JOB_ID=$(sbatch submit_paper_metrics.sh "$CHECKPOINT_DIR" "$MAX_SAMPLES" "$CONFIG_FILE" "$LIBRILIGHT_LENGTH" | awk '{print $4}')
fi

echo "✅ Job submitted successfully!"
echo "🎯 Job ID: $JOB_ID"
echo ""
echo "📊 Monitor your job with:"
echo "  squeue -u $USER"
echo "  squeue -j $JOB_ID"
echo ""
echo "📄 Check logs with:"
echo "  tail -f /home/alufr/ttt_tests/moshi-finetune/logs/evaluation/paper_metrics_$JOB_ID.log"
echo "  tail -f /home/alufr/ttt_tests/moshi-finetune/logs/evaluation/paper_metrics_$JOB_ID.err"
echo ""
echo "❌ Cancel job with:"
echo "  scancel $JOB_ID"
echo ""
if [ "$MODE" == "baseline" ]; then
    echo "📊 Results will be saved to: ./baseline_paper_metrics_results.json"
else
    echo "📊 Results will be saved to: $CHECKPOINT_DIR/paper_metrics_results.json"
fi
