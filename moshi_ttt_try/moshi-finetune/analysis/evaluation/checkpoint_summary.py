#!/usr/bin/env python3
"""
TTT-Moshi Checkpoint System Summary
=================================

This script provides an overview of the checkpoint system status and capabilities.
Run this to verify everything is working and get quick help.

Usage:
    python checkpoint_summary.py [--check-system] [--show-examples]
"""

import argparse
import logging
import sys
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("checkpoint_summary")

def print_header():
    """Print the header information."""
    print("=" * 60)
    print("🔧 TTT-MOSHI CHECKPOINT SYSTEM SUMMARY")
    print("=" * 60)
    print()

def check_system_status():
    """Check if the checkpoint system components are available."""
    logger.info("🔍 Checking checkpoint system components...")
    
    # Check core files
    core_files = [
        "finetune/checkpointing.py",
        "train.py", 
        "finetune/eval.py",
        "test_checkpoint_verification.py",
        "eval_from_checkpoint.py",
        "CHECKPOINT_USER_GUIDE.md"
    ]
    
    all_present = True
    for file_path in core_files:
        if Path(file_path).exists():
            logger.info(f"✅ {file_path}")
        else:
            logger.error(f"❌ {file_path} - MISSING")
            all_present = False
    
    print()
    
    # Check imports
    logger.info("🔍 Checking Python dependencies...")
    try:
        import torch
        import safetensors
        from finetune.checkpointing import Checkpointer
        from finetune.args import TrainArgs
        logger.info("✅ All required Python modules available")
        modules_ok = True
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        modules_ok = False
    
    print()
    
    # Overall status
    if all_present and modules_ok:
        logger.info("🎉 CHECKPOINT SYSTEM STATUS: ✅ READY")
        return True
    else:
        logger.error("⚠️ CHECKPOINT SYSTEM STATUS: ❌ ISSUES DETECTED")
        return False

def show_quick_examples():
    """Show quick usage examples."""
    logger.info("🚀 QUICK USAGE EXAMPLES")
    logger.info("-" * 25)
    print()
    
    examples = [
        ("Train with checkpoints", 
         "python train.py --config configs/your_config.yaml"),
        
        ("Resume training", 
         "torchrun --nproc_per_node=1 train.py --config configs/your_config.yaml --checkpoint.resume true"),
        
        ("Verify checkpoint system",
         "python test_checkpoint_verification.py"),
        
        ("Evaluate checkpoint",
         "python eval_from_checkpoint.py --checkpoint_dir runs/exp/checkpoints/checkpoint_005000 --config configs/your_config.yaml"),
        
        ("Check checkpoint contents",
         "ls runs/your_experiment/checkpoints/"),
    ]
    
    for i, (description, command) in enumerate(examples, 1):
        logger.info(f"{i}. {description}:")
        logger.info(f"   {command}")
        print()

def show_config_reference():
    """Show key configuration options."""
    logger.info("⚙️ KEY CONFIGURATION OPTIONS")
    logger.info("-" * 30)
    print()
    
    config_yaml = """# In your YAML config file:
do_ckpt: true              # Enable checkpointing
ckpt_freq: 1000           # Save every N steps (0 = only at end)
num_ckpt_keep: 3          # Keep last N checkpoints
save_adapters: true       # Save only adapters (smaller files)

ttt:
  enable: true            # TTT parameters auto-saved in checkpoints
  layers: "1,5,10"       # Which layers have TTT
  base_lr: 1.0
  mini_batch_size: 16"""
    
    print(config_yaml)
    print()

def show_file_overview():
    """Show overview of checkpoint-related files."""
    logger.info("📁 CHECKPOINT SYSTEM FILES")
    logger.info("-" * 27)
    print()
    
    files = [
        ("train.py", "Main training script with checkpoint integration"),
        ("finetune/checkpointing.py", "Core checkpoint save/load implementation"),
        ("test_checkpoint_verification.py", "Verify TTT parameters are saved/loaded correctly"),
        ("eval_from_checkpoint.py", "Evaluate saved checkpoints"),
        ("CHECKPOINT_USER_GUIDE.md", "Complete user documentation"),
        ("finetune/eval.py", "Standard evaluation functions"),
        ("finetune/paper_metrics.py", "Paper metrics evaluation"),
    ]
    
    for filename, description in files:
        status = "✅" if Path(filename).exists() else "❌"
        logger.info(f"{status} {filename:<35} - {description}")
    
    print()

def show_troubleshooting():
    """Show common troubleshooting tips."""
    logger.info("🔧 TROUBLESHOOTING TIPS")
    logger.info("-" * 20)
    print()
    
    tips = [
        ("Distributed training error", "Use: torchrun --nproc_per_node=1 train.py"),
        ("TTT parameters missing", "Check ttt.enable: true in config"),
        ("Checkpoint not found", "Use full path: runs/exp/checkpoints/checkpoint_NNNNNN"),
        ("Out of memory", "Reduce batch_size or use save_adapters: true"),
        ("Parameter mismatch", "Ensure same TTT layers and LoRA config"),
    ]
    
    for problem, solution in tips:
        logger.info(f"• {problem}:")
        logger.info(f"  → {solution}")
        print()

def main():
    parser = argparse.ArgumentParser(description="TTT-Moshi checkpoint system summary")
    parser.add_argument("--check-system", action="store_true", 
                       help="Check if all system components are available")
    parser.add_argument("--show-examples", action="store_true",
                       help="Show usage examples")
    parser.add_argument("--full", action="store_true",
                       help="Show complete overview (default)")
    
    args = parser.parse_args()
    
    # Default to full overview if no specific option chosen
    if not any([args.check_system, args.show_examples]):
        args.full = True
    
    print_header()
    
    success = True
    
    if args.check_system or args.full:
        success = check_system_status()
        print()
    
    if args.full:
        show_file_overview()
        show_config_reference()
        show_quick_examples()
        show_troubleshooting()
    
    if args.show_examples:
        show_quick_examples()
    
    # Footer
    logger.info("📚 DOCUMENTATION")
    logger.info("-" * 15)
    logger.info("📖 Complete guide: CHECKPOINT_USER_GUIDE.md")
    logger.info("🧪 Test system:    python test_checkpoint_verification.py")
    logger.info("📊 Evaluate:       python eval_from_checkpoint.py --help")
    print()
    
    if success:
        logger.info("✅ Ready to train and checkpoint TTT-Moshi models!")
        return 0
    else:
        logger.error("❌ Please fix the issues above before proceeding.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)