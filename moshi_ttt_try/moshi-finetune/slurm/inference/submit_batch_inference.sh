#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# SLURM Batch Inference Job Submission Script
# Usage: ./submit_batch_inference.sh [checkpoint_dir] [input_pattern] [output_dir] [hf_repo] [max_length] [compute_perplexity]
#
# Examples:
#   # Generate audio (default mode)
#   ./submit_batch_inference.sh /path/to/checkpoint "audio/*.wav" ./results
#   
#   # Generate audio with custom settings
#   ./submit_batch_inference.sh /path/to/checkpoint "audio1.wav audio2.wav" ./results kyutai/moshiko-pytorch-bf16 1000 false true
#   
#   # Compute perplexity (research mode)
#   ./submit_batch_inference.sh /path/to/checkpoint "audio1.wav" ./results kyutai/moshiko-pytorch-bf16 1000 true false
#   
#   ./submit_batch_inference.sh   # Use defaults (generate audio)
# -----------------------------------------------------------------------------

set -e  # Exit on any error

# Default values
DEFAULT_CHECKPOINT=/sise/eliyanac-group/ron_al/dailytalk_finetune_from_librilight6/checkpoints/checkpoint_002000/consolidated
DEFAULT_INPUT="/sise/eliyanac-group/ron_al/examples/combined_1595476768245534437_24000hz.wav"
DEFAULT_OUTPUT_DIR="/home/alufr/ttt_tests/moshi-finetune/batch_results"
DEFAULT_HF_REPO="kyutai/moshiko-pytorch-bf16"
DEFAULT_MAX_LENGTH=""  # No limit by default
DEFAULT_COMPUTE_PERPLEXITY="false"  # Changed default to false
DEFAULT_GENERATE_AUDIO="true"      # Default to generating audio
DEFAULT_ENABLE_TTT_DIAGNOSTICS="true"  # Enabled by default for analysis
DEFAULT_DIAGNOSTIC_LOG_FREQUENCY="100"  # Log every 100 steps

# Get parameters from command line or use defaults
CHECKPOINT_DIR=${1:-$DEFAULT_CHECKPOINT}
INPUT_PATTERN=${2:-$DEFAULT_INPUT}
OUTPUT_DIR=${3:-$DEFAULT_OUTPUT_DIR}
HF_REPO=${4:-$DEFAULT_HF_REPO}
MAX_LENGTH=${5:-$DEFAULT_MAX_LENGTH}
COMPUTE_PERPLEXITY=${6:-$DEFAULT_COMPUTE_PERPLEXITY}
GENERATE_AUDIO=${7:-$DEFAULT_GENERATE_AUDIO}
ENABLE_TTT_DIAGNOSTICS=${8:-$DEFAULT_ENABLE_TTT_DIAGNOSTICS}
DIAGNOSTIC_LOG_FREQUENCY=${9:-$DEFAULT_DIAGNOSTIC_LOG_FREQUENCY}

# Check if checkpoint exists
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "❌ Error: Checkpoint directory '$CHECKPOINT_DIR' not found!"
    echo ""
    echo "Available checkpoints in /sise/eliyanac-group/ron_al/:"
    find /sise/eliyanac-group/ron_al -type d -name "checkpoint_*" 2>/dev/null | head -10 || echo "  (none found)"
    exit 1
fi

# Expand input pattern and check if files exist
INPUT_FILES=""
if [[ "$INPUT_PATTERN" == *"*"* ]]; then
    # Handle glob patterns
    INPUT_FILES=$(ls $INPUT_PATTERN 2>/dev/null | tr '\n' ' ' | sed 's/ $//')
    if [ -z "$INPUT_FILES" ]; then
        echo "❌ Error: No files found matching pattern '$INPUT_PATTERN'!"
        exit 1
    fi
else
    # Handle space-separated file list
    INPUT_FILES="$INPUT_PATTERN"
    for file in $INPUT_FILES; do
        if [ ! -f "$file" ]; then
            echo "❌ Error: Input audio file '$file' not found!"
            exit 1
        fi
    done
fi

# Count files
FILE_COUNT=$(echo $INPUT_FILES | wc -w)

echo "🚀 Submitting Moshi TTT Batch Inference Job"
echo "==========================================="
echo "📁 Checkpoint: $CHECKPOINT_DIR"
echo "🎤 Input files: $FILE_COUNT files"
echo "   $(echo $INPUT_FILES | cut -c1-100)$([ ${#INPUT_FILES} -gt 100 ] && echo "...")"
echo "📂 Output directory: $OUTPUT_DIR"
echo "🤗 HF repo: $HF_REPO"
echo "📏 Max length: ${MAX_LENGTH:-"unlimited"}"
echo "📊 Compute perplexity: $COMPUTE_PERPLEXITY"
echo "🎵 Generate audio: $GENERATE_AUDIO"
echo "🔍 TTT Diagnostics: $ENABLE_TTT_DIAGNOSTICS (frequency: $DIAGNOSTIC_LOG_FREQUENCY)"
echo "📋 SLURM script: run_batch_inference.slurm"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Submit the job with environment variables
JOB_ID=$(sbatch \
    --export=CHECKPOINT_DIR="$CHECKPOINT_DIR",INPUT_FILES="$INPUT_FILES",OUTPUT_DIR="$OUTPUT_DIR",HF_REPO="$HF_REPO",MAX_LENGTH="$MAX_LENGTH",COMPUTE_PERPLEXITY="$COMPUTE_PERPLEXITY",GENERATE_AUDIO="$GENERATE_AUDIO",ENABLE_TTT_DIAGNOSTICS="$ENABLE_TTT_DIAGNOSTICS",DIAGNOSTIC_LOG_FREQUENCY="$DIAGNOSTIC_LOG_FREQUENCY" \
    run_batch_inference.slurm | awk '{print $4}')

echo "✅ Job submitted successfully!"
echo "🎯 Job ID: $JOB_ID"
echo ""
echo "📊 Monitor your job with:"
echo "  squeue -u $USER"
echo "  squeue -j $JOB_ID"
echo ""
echo "📄 Check logs with:"
echo "  tail -f /home/alufr/ttt_tests/moshi-finetune/logs/batch_inference/moshi_batch_inference.$JOB_ID.log"
echo "  tail -f /home/alufr/ttt_tests/moshi-finetune/logs/batch_inference/moshi_batch_inference.$JOB_ID.err"
echo ""
echo "❌ Cancel job with:"
echo "  scancel $JOB_ID"
echo ""
echo "📊 Results will be saved to: $OUTPUT_DIR"
echo "   - results.json (metrics and metadata)"
echo "   - logits.pt (model outputs)"