#!/usr/bin/env python3
"""
Script to evaluate a trained checkpoint with inner loop loss tracking enabled.

This loads a checkpoint and runs LibriLight evaluation with Figure 4 plotting.

Usage:
    python evaluate_checkpoint_with_inner_loop.py --checkpoint <path_to_checkpoint>
"""

import argparse
import torch
from pathlib import Path
import yaml
import logging

from finetune.args import TrainArgs
from finetune.train import load_model
from finetune.paper_metrics import PaperMetricsEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoint with inner loop loss tracking")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--config", type=str, help="Optional config YAML (defaults to checkpoint's config)")
    parser.add_argument("--output-dir", type=str, default="./evaluation_plots/inner_loop",
                       help="Directory to save inner loop plots")
    parser.add_argument("--max-sequences", type=int, default=1,
                       help="Number of LibriLight sequences to evaluate")
    args = parser.parse_args()
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load config
    if args.config:
        config_path = Path(args.config)
        logger.info(f"Loading config from: {config_path}")
    else:
        # Try to find config next to checkpoint
        config_path = checkpoint_path.parent.parent / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(
                f"Config not found: {config_path}. Please specify with --config"
            )
        logger.info(f"Using config from checkpoint directory: {config_path}")
    
    # Load and modify config to enable inner loop tracking
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    
    # Enable inner loop loss tracking
    if 'ttt' not in config_dict:
        config_dict['ttt'] = {}
    
    config_dict['ttt']['log_inner_loop_losses'] = True
    config_dict['ttt']['save_inner_loop_plots'] = True
    config_dict['ttt']['inner_loop_plot_dir'] = args.output_dir
    config_dict['ttt']['inner_loop_log_interval'] = 1  # Log every position
    
    # Ensure LibriLight evaluation is enabled
    if 'paper_metrics' not in config_dict:
        config_dict['paper_metrics'] = {}
    
    config_dict['paper_metrics']['paper_metrics_eval'] = True
    
    # Limit sequences for faster evaluation
    if 'librilight_num_sequences' in config_dict['paper_metrics']:
        config_dict['paper_metrics']['librilight_num_sequences'] = min(
            config_dict['paper_metrics'].get('librilight_num_sequences', 1),
            args.max_sequences
        )
    
    logger.info(f"\n{'='*80}")
    logger.info("Configuration for inner loop tracking:")
    logger.info(f"  log_inner_loop_losses: {config_dict['ttt']['log_inner_loop_losses']}")
    logger.info(f"  save_inner_loop_plots: {config_dict['ttt']['save_inner_loop_plots']}")
    logger.info(f"  inner_loop_plot_dir: {config_dict['ttt']['inner_loop_plot_dir']}")
    logger.info(f"  max_sequences: {args.max_sequences}")
    logger.info(f"{'='*80}\n")
    
    # Parse config into TrainArgs
    from simple_parsing import ArgumentParser
    parser = ArgumentParser()
    parser.add_arguments(TrainArgs, dest="args")
    
    # Convert config_dict to command line args format
    # This is a simplified version - you may need to adjust based on your config structure
    temp_config_path = Path("/tmp/eval_config.yaml")
    with open(temp_config_path, 'w') as f:
        yaml.dump(config_dict, f)
    
    train_args = parser.parse_args([f"--config_path={temp_config_path}"])
    args_obj = train_args.args
    
    # Load model
    logger.info("Loading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # NOTE: You'll need to adapt load_model() call based on your checkpoint format
    # This is a placeholder - adjust according to your actual checkpoint loading code
    try:
        # Try loading with standard checkpoint format
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Extract model state dict
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        logger.info(f"Checkpoint loaded successfully")
        logger.info(f"  Keys in checkpoint: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'N/A'}")
        
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        raise
    
    # Initialize model (you'll need to implement this based on your model architecture)
    logger.info("Initializing model...")
    
    # This is where you'd normally load your model architecture
    # For now, just show what would happen:
    logger.warning(
        "\n⚠️  MODEL LOADING NOT IMPLEMENTED IN THIS TEMPLATE\n"
        "You need to add code to:\n"
        "  1. Initialize your model architecture\n"
        "  2. Load the checkpoint state dict\n"
        "  3. Move model to correct device\n"
        "\nExample:\n"
        "  from moshi.models import get_moshi_lm\n"
        "  model = get_moshi_lm(args_obj.model_config)\n"
        "  model.load_state_dict(state_dict)\n"
        "  model = model.to(device)\n"
        "  model.eval()\n"
    )
    
    # Initialize paper metrics evaluator
    logger.info("Initializing evaluator...")
    
    # You'll need to initialize mimi encoder and tokenizer here
    logger.warning(
        "\n⚠️  MIMI/TOKENIZER INITIALIZATION NOT IMPLEMENTED\n"
        "You need to add code to:\n"
        "  1. Load Mimi encoder\n"
        "  2. Load interleaved tokenizer\n"
        "\nExample:\n"
        "  from moshi.models import get_mimi\n"
        "  mimi = get_mimi(...)\n"
        "  tokenizer = get_tokenizer(...)\n"
    )
    
    # evaluator = PaperMetricsEvaluator(
    #     mimi_encoder=mimi,
    #     interleaved_tokenizer=tokenizer,
    #     device=device,
    #     config=config_dict.get('paper_metrics', {})
    # )
    
    logger.info("\n" + "="*80)
    logger.info("NEXT STEPS:")
    logger.info("="*80)
    logger.info("""
This template shows you how to enable inner loop tracking for checkpoint evaluation.

To make this work, you need to:

1. Complete the model loading code (lines marked with ⚠️)
2. Complete the MIMI/tokenizer initialization
3. Run LibriLight evaluation with the evaluator
4. The plots will be automatically saved to: {args.output_dir}

The inner loop loss tracking is ALREADY IMPLEMENTED in the TTT-MLP code.
When you call the model with log_losses=True, it will automatically:
  - Compute reconstruction losses at each mini-batch
  - Return them alongside outputs
  - Generate Figure 4 plots when save_inner_loop_plots=True

See INNER_LOOP_IMPLEMENTATION.md for full documentation.
""")
    logger.info("="*80 + "\n")
    
    # What you would run after completing the above:
    # logger.info("Running LibriLight evaluation with inner loop tracking...")
    # results = evaluator.evaluate_librilight_only(model)
    # logger.info(f"Results: {results}")
    # logger.info(f"Plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
