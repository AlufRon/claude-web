#!/bin/bash
# Paper Metrics Dashboard - Update Script
#
# This script updates the dashboard with new paper metrics results.
# Run this after evaluating new checkpoints to refresh the dashboard.
#
# Usage:
#   ./update_dashboard.sh
#   ./update_dashboard.sh --checkpoint-dirs /custom/path

set -e  # Exit on error

echo "🔄 Updating Paper Metrics Dashboard..."
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Default arguments
CHECKPOINT_DIRS="${1:-/sise/eliyanac-group/ron_al}"
LOG_DIR="${2:-/home/alufr/ttt_tests/moshi-finetune/logs/evaluation}"
OUTPUT="dashboard_data.json"

echo "📂 Configuration:"
echo "   Checkpoint dirs: $CHECKPOINT_DIRS"
echo "   Log directory: $LOG_DIR"
echo "   Output file: $OUTPUT"
echo ""

# Check if Python script exists
if [ ! -f "aggregate_paper_metrics.py" ]; then
    echo "❌ Error: aggregate_paper_metrics.py not found!"
    echo "   Make sure you're in the dashboard directory."
    exit 1
fi

# Step 1: Run aggregation script
echo "📊 Step 1/2: Aggregating paper metrics data..."
python aggregate_paper_metrics.py \
    --checkpoint-dirs "$CHECKPOINT_DIRS" \
    --log-dir "$LOG_DIR" \
    --output "$OUTPUT"

EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "❌ Dashboard aggregation failed with exit code $EXIT_CODE"
    echo "   Check the error messages above for details."
    exit $EXIT_CODE
fi

echo "   ✅ Dashboard data aggregated!"
echo ""

# Step 2: Regenerate standalone HTML
echo "📦 Step 2/2: Regenerating standalone HTML..."
./create_standalone.sh

EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "❌ HTML generation failed with exit code $EXIT_CODE"
    echo "   Data was updated at: $(pwd)/$OUTPUT"
    exit $EXIT_CODE
fi

echo ""
echo "✅ Dashboard fully updated!"
echo ""
echo "📊 Generated files:"
echo "   • dashboard_data.json (data)"
echo "   • dashboard_standalone.html (standalone - download this!)"
echo "   • dashboard.html (requires local server)"
echo ""
echo "💡 Usage:"
echo "   • Download dashboard_standalone.html to view on any computer"
echo "   • Or run: firefox dashboard_standalone.html &"
echo ""
