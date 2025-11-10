#!/bin/bash
# Evaluation script with Inner Loop Loss Tracking enabled
# This script loads the config, modifies it for quick evaluation, and enables Figure 4 plotting

set -e

CHECKPOINT_DIR="/sise/eliyanac-group/ron_al/seamless_moshinbn010/checkpoints/checkpoint_000100/consolidated"
CONFIG_PATH="/sise/eliyanac-group/ron_al/seamless_moshinbn010/args.yaml"
TEMP_CONFIG="/tmp/eval_config_inner_loop.yaml"

echo "============================================"
echo "TTT-Moshi Evaluation with Inner Loop Logging"
echo "============================================"
echo ""
echo "Checkpoint: $CHECKPOINT_DIR"
echo "Config: $CONFIG_PATH"
echo ""

# Copy config and modify it
echo "📝 Preparing evaluation config..."
cp "$CONFIG_PATH" "$TEMP_CONFIG"

# Use Python to modify the YAML file
python3 << 'EOF'
import yaml
import sys

config_path = "/tmp/eval_config_inner_loop.yaml"

# Load config
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Reduce evaluation samples for faster testing
if 'paper_metrics' not in config:
    config['paper_metrics'] = {}

config['paper_metrics']['sblimp_max_pairs'] = 5
config['paper_metrics']['swuggy_max_pairs'] = 5
config['paper_metrics']['sstory_max_pairs'] = 5
config['paper_metrics']['tstory_max_pairs'] = 5

# Enable inner loop loss tracking (Figure 4)
if 'ttt' not in config:
    config['ttt'] = {}

config['ttt']['log_inner_loop_losses'] = True
config['ttt']['inner_loop_log_interval'] = 1
config['ttt']['save_inner_loop_plots'] = True
config['ttt']['inner_loop_plot_dir'] = './evaluation_plots/inner_loop'

# Save modified config
with open(config_path, 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

print("✅ Config updated:")
print(f"  - sBLIMP/sWUGGY/sStoryCloze/tStoryCloze: 5 pairs each")
print(f"  - Inner loop logging: ENABLED")
print(f"  - Plot directory: ./evaluation_plots/inner_loop")
EOF

echo ""
echo "🚀 Starting evaluation..."
echo ""

# Run evaluation
python eval_from_checkpoint.py \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --config "$TEMP_CONFIG"

echo ""
echo "============================================"
echo "✅ Evaluation Complete!"
echo "============================================"
echo ""
echo "📊 Check for inner loop plots in:"
echo "   ./evaluation_plots/inner_loop/"
echo ""
echo "Files you should see:"
echo "  - inner_loop_losses_seq_0.png (Figure 4 visualization)"
echo "  - inner_loop_losses_seq_0.json (raw data)"
echo "  - inner_loop_summary.png (if multiple sequences)"
echo ""
