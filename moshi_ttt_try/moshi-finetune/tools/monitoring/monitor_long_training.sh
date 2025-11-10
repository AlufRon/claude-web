#!/bin/bash
# Monitor long LibriLight TTT training
# Usage: ./monitor_long_training.sh [job_id]

JOB_ID=${1:-7331353}
LOG_FILE="moshi_ttt.${JOB_ID}.log"

echo "=========================================="
echo "🔍 Monitoring TTT Training Job: ${JOB_ID}"
echo "=========================================="
echo ""

# Check if job is running
echo "📊 Job Status:"
squeue -j ${JOB_ID} 2>/dev/null || echo "❌ Job not found or completed"
echo ""

# Check if log file exists
if [ ! -f "${LOG_FILE}" ]; then
    echo "❌ Log file not found: ${LOG_FILE}"
    exit 1
fi

# Show training progress
echo "📈 Training Progress (Last 10 steps):"
grep "step:" ${LOG_FILE} | tail -10
echo ""

# Show checkpoint saves
echo "💾 Checkpoint Saves:"
grep "Done dumping checkpoint" ${LOG_FILE} | tail -5
echo ""

# Show memory usage
echo "🧠 Recent Memory Usage:"
grep "Memory Step.*after_backward" ${LOG_FILE} | tail -3
echo ""

# Calculate progress
TOTAL_STEPS=50000
CURRENT_STEP=$(grep "step:" ${LOG_FILE} | tail -1 | grep -oP 'step: \K\d+' || echo "0")
if [ "$CURRENT_STEP" -gt 0 ]; then
    PROGRESS=$(echo "scale=2; ($CURRENT_STEP / $TOTAL_STEPS) * 100" | bc)
    echo "✅ Progress: ${CURRENT_STEP}/${TOTAL_STEPS} steps (${PROGRESS}%)"
    
    # Estimate time remaining
    ELAPSED=$(grep "step:" ${LOG_FILE} | tail -1 | grep -oP '\d+:\d+:\d+' || echo "00:00:00")
    echo "⏱️  Elapsed time: ${ELAPSED}"
    
    # Show latest loss
    LATEST_LOSS=$(grep "step:" ${LOG_FILE} | tail -1 | grep -oP 'loss: \K[\d.]+' || echo "N/A")
    echo "📉 Latest loss: ${LATEST_LOSS}"
    
    # Show ETA
    LATEST_ETA=$(grep "step:" ${LOG_FILE} | tail -1 | grep -oP 'ETA: >(.*)' | tail -1)
    if [ ! -z "$LATEST_ETA" ]; then
        echo "🎯 ETA: ${LATEST_ETA}"
    fi
fi
echo ""

# Show any errors
ERROR_COUNT=$(grep -i "error\|exception\|failed" ${LOG_FILE} | wc -l)
if [ "$ERROR_COUNT" -gt 0 ]; then
    echo "⚠️  Found ${ERROR_COUNT} error/warning messages"
    echo "Recent errors:"
    grep -i "error\|exception\|failed" ${LOG_FILE} | tail -3
else
    echo "✅ No errors detected"
fi
echo ""

echo "=========================================="
echo "📝 Quick Commands:"
echo "  View real-time logs:  tail -f ${LOG_FILE}"
echo "  View all progress:    grep 'step:' ${LOG_FILE}"
echo "  View checkpoints:     ls -lh /sise/eliyanac-group/ron_al/librilight_ttt_long_pretrain/checkpoints/"
echo "  Cancel job:           scancel ${JOB_ID}"
echo "=========================================="
