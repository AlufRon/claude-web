#!/usr/bin/env python3
"""
Quick evaluation script with Inner Loop Loss Tracking
Modifies config to use 5 samples for quick testing and enables Figure 4 plotting
"""

import yaml
import subprocess
import sys
from pathlib import Path

# Configuration
CHECKPOINT_DIR = "/sise/eliyanac-group/ron_al/seamless_moshinbn010/checkpoints/checkpoint_000100/consolidated"
CONFIG_PATH = "/sise/eliyanac-group/ron_al/seamless_moshinbn010/args.yaml"
TEMP_CONFIG = "/tmp/eval_config_inner_loop.yaml"

def main():
    print("=" * 80)
    print("TTT-Moshi Evaluation with Inner Loop Loss Tracking")
    print("=" * 80)
    print()
    print(f"Checkpoint: {CHECKPOINT_DIR}")
    print(f"Config: {CONFIG_PATH}")
    print()
    
    # Load and modify config
    print("📝 Preparing evaluation config...")
    with open(CONFIG_PATH, 'r') as f:
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
    with open(TEMP_CONFIG, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print("✅ Config updated:")
    print("  - sBLIMP/sWUGGY/sStoryCloze/tStoryCloze: 5 pairs each")
    print("  - Inner loop logging: ENABLED")
    print("  - Plot directory: ./evaluation_plots/inner_loop")
    print()
    
    # Run evaluation
    print("🚀 Starting evaluation...")
    print()
    
    cmd = [
        sys.executable,
        "eval_from_checkpoint.py",
        "--checkpoint_dir", CHECKPOINT_DIR,
        "--config", TEMP_CONFIG
    ]
    
    result = subprocess.run(cmd, cwd="/home/alufr/ttt_tests/moshi-finetune")
    
    if result.returncode == 0:
        print()
        print("=" * 80)
        print("✅ Evaluation Complete!")
        print("=" * 80)
        print()
        print("📊 Check for inner loop plots in:")
        print("   ./evaluation_plots/inner_loop/")
        print()
        print("Files you should see:")
        print("  - inner_loop_losses_seq_0.png (Figure 4 visualization)")
        print("  - inner_loop_losses_seq_0.json (raw data)")
        print("  - inner_loop_summary.png (if multiple sequences)")
        print()
    else:
        print()
        print("❌ Evaluation failed with return code:", result.returncode)
        sys.exit(result.returncode)

if __name__ == "__main__":
    main()
