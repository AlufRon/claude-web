#!/usr/bin/env python3
"""
Checkpoint-Based Evaluation Script for TTT-Moshi
===============================================

This script loads a saved checkpoint and runs evaluation on it.
Supports both standard evaluation and paper metrics evaluation.

Usage:
    python eval_from_checkpoint.py --checkpoint_dir /path/to/checkpoint --config /path/to/config.yaml

Features:
- Load TTT-Moshi checkpoints
- Run standard evaluation (perplexity)
- Run paper metrics evaluation (optional)
- Verify TTT parameter loading
- Support for both full model and LoRA checkpoints
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from contextlib import ExitStack

# Disable torch dynamo to avoid compilation issues during evaluation
import torch._dynamo
torch._dynamo.config.suppress_errors = True

import torch
import torch.distributed as dist
import safetensors.torch

from finetune.args import TrainArgs
from finetune.checkpointing import Checkpointer
from finetune.wrapped_model import get_fsdp_model
from finetune.utils import TrainState
from finetune.distributed import set_device, BACKEND, get_rank, get_world_size
from finetune.data.data_loader import build_data_loader
from finetune.data.interleaver import InterleavedTokenizer, Interleaver
from finetune.eval import evaluate
from moshi.models import loaders

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("eval_checkpoint")

def setup_distributed():
    """Setup distributed training for single GPU evaluation."""
    if "LOCAL_RANK" not in os.environ:
        # Setup for single GPU evaluation
        import socket
        
        # Find a free port to avoid conflicts
        def find_free_port():
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', 0))
                s.listen(1)
                port = s.getsockname()[1]
            return port
        
        free_port = find_free_port()
        
        os.environ["LOCAL_RANK"] = "0"
        os.environ["RANK"] = "0" 
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(free_port)
        
        logger.info(f"Using free port: {free_port}")
    
    set_device()
    dist.init_process_group(backend=BACKEND)

def load_checkpoint_state_dict(checkpoint_dir: Path, is_lora_only: bool = False):
    """Load state dict from checkpoint directory."""
    # Smart path detection: handle both checkpoint_dir and checkpoint_dir/consolidated
    if checkpoint_dir.name == "consolidated":
        # User provided the consolidated directory directly
        consolidated_dir = checkpoint_dir
    else:
        # User provided the checkpoint directory, need to add /consolidated
        consolidated_dir = checkpoint_dir / "consolidated"
    
    if not consolidated_dir.exists():
        raise FileNotFoundError(f"Consolidated checkpoint not found: {consolidated_dir}")
    
    # Determine checkpoint file
    checkpoint_file = Checkpointer.consolidated_path(consolidated_dir, save_only_lora=is_lora_only)
    if not checkpoint_file.exists():
        # Try the other format
        checkpoint_file = Checkpointer.consolidated_path(consolidated_dir, save_only_lora=not is_lora_only)
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"No checkpoint file found in {consolidated_dir}")
    
    logger.info(f"Loading checkpoint from: {checkpoint_file}")
    state_dict = safetensors.torch.load_file(checkpoint_file)
    
    return state_dict, checkpoint_file

def load_training_config(checkpoint_dir: Path):
    """Load training configuration from checkpoint if available."""
    consolidated_dir = checkpoint_dir / "consolidated"
    training_config_path = consolidated_dir / "training_config.json"
    
    if training_config_path.exists():
        logger.info(f"📋 Loading training config from checkpoint: {training_config_path}")
        import json
        with open(training_config_path, 'r') as f:
            training_config = json.load(f)
        return training_config
    else:
        logger.warning("⚠️ No training_config.json found in checkpoint. Using provided config.")
        return None

def verify_ttt_parameters(model, state_dict):
    """Verify that TTT parameters are loaded correctly."""
    logger.info("🔍 Verifying TTT parameter loading...")
    
    # Count TTT parameters in model
    model_ttt_params = {}
    for name, param in model.named_parameters():
        if any(ttt_key in name.lower() for ttt_key in ['ttt', 'w1', 'w2', 'gating_alpha']):
            model_ttt_params[name] = param
    
    # Count TTT parameters in checkpoint
    checkpoint_ttt_params = {}
    for name, tensor in state_dict.items():
        if any(ttt_key in name.lower() for ttt_key in ['ttt', 'w1', 'w2', 'gating_alpha']):
            checkpoint_ttt_params[name] = tensor
    
    logger.info(f"Model TTT parameters: {len(model_ttt_params)}")
    logger.info(f"Checkpoint TTT parameters: {len(checkpoint_ttt_params)}")
    
    # Verify key TTT parameters match
    ttt_verified = True
    for name in list(model_ttt_params.keys())[:5]:  # Check first 5 for brevity
        if name in checkpoint_ttt_params:
            model_val = model_ttt_params[name].cpu().to(torch.float16)
            checkpoint_val = checkpoint_ttt_params[name].to(torch.float16)
            
            if torch.allclose(model_val, checkpoint_val, atol=1e-3, rtol=1e-3):
                logger.info(f"✅ {name} - loaded correctly")
            else:
                logger.error(f"❌ {name} - mismatch detected")
                ttt_verified = False
        else:
            logger.error(f"❌ {name} - missing from checkpoint")
            ttt_verified = False
    
    return ttt_verified

def run_standard_evaluation(model, args, interleaved_tokenizer):
    """Run standard evaluation (perplexity)."""
    logger.info("📊 Running standard evaluation...")
    
    # Build evaluation data loader
    eval_data_loader = build_data_loader(
        instruct_tokenizer=interleaved_tokenizer,
        args=args.data,
        batch_size=args.batch_size,
        seed=None,
        rank=get_rank(),
        world_size=get_world_size(),
        is_eval=True,
    )
    
    # Create a dummy state for evaluation
    state = TrainState(max_steps=1)
    state.step = 0
    
    # Run evaluation
    evaluate(model, eval_data_loader, state, args)
    
    return {
        'eval_loss': state.this_eval_loss,
        'eval_perplexity': state.this_eval_perplexity,
        'audio_loss': state.this_audio_loss,
        'text_loss': state.this_text_loss
    }

def run_paper_metrics_evaluation(model, args, mimi, interleaved_tokenizer):
    """Run paper metrics evaluation if configured."""
    if not hasattr(args, 'paper_metrics') or not args.paper_metrics.get('paper_metrics_eval', False):
        logger.info("📋 Paper metrics evaluation not configured")
        return None
    
    logger.info("📊 Running paper metrics evaluation...")
    
    try:
        from finetune.paper_metrics import PaperMetricsEvaluator
        
        # Convert TrainArgs to dict to pass full config including TTT section
        config_dict = args.to_dict() if hasattr(args, 'to_dict') else vars(args)
        
        evaluator = PaperMetricsEvaluator(
            mimi,
            interleaved_tokenizer,
            config=config_dict
        )
        
        # Run full evaluation INCLUDING LibriLight long context
        results = evaluator.evaluate_all(model)
        
        logger.info("✅ Paper metrics evaluation completed")
        return results
        
    except ImportError as e:
        logger.warning(f"Paper metrics not available: {e}")
        return None
    except Exception as e:
        logger.error(f"Paper metrics evaluation failed: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Evaluate TTT-Moshi from checkpoint")
    parser.add_argument("--checkpoint_dir", type=str, required=True, 
                       help="Path to checkpoint directory. Supports both formats: "
                            "checkpoint_001000 or checkpoint_001000/consolidated")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to training config YAML file")
    parser.add_argument("--eval_type", choices=["standard", "paper_metrics", "both"], default="both",
                       help="Type of evaluation to run")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Override batch size for evaluation")
    parser.add_argument("--no_ttt_verify", action="store_true",
                       help="Skip TTT parameter verification")
    
    args = parser.parse_args()
    
    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        logger.error(f"Checkpoint directory not found: {checkpoint_dir}")
        return 1
    
    # Setup distributed training
    setup_distributed()
    
    # Load configuration
    logger.info(f"📋 Loading configuration from: {args.config}")
    train_args = TrainArgs.load(args.config, drop_extra_fields=False)
    
    # Try to load training config from checkpoint for proper restoration
    checkpoint_training_config = load_training_config(checkpoint_dir)
    if checkpoint_training_config:
        # Override critical TTT settings from checkpoint
        if 'ttt' in checkpoint_training_config:
            logger.info("🔧 Using TTT configuration from checkpoint...")
            checkpoint_ttt = checkpoint_training_config['ttt']
            logger.info(f"   TTT mini_batch_size: {checkpoint_ttt.get('mini_batch_size', 'unknown')}")
            logger.info(f"   TTT layers: {checkpoint_ttt.get('layers', 'unknown')}")
            logger.info(f"   TTT base_lr: {checkpoint_ttt.get('base_lr', 'unknown')}")
            
            # Update training args with checkpoint TTT config
            train_args.ttt.mini_batch_size = checkpoint_ttt.get('mini_batch_size', train_args.ttt.mini_batch_size)
            train_args.ttt.layers = checkpoint_ttt.get('layers', train_args.ttt.layers)
            train_args.ttt.base_lr = checkpoint_ttt.get('base_lr', train_args.ttt.base_lr)
            train_args.ttt.enable = checkpoint_ttt.get('enable', train_args.ttt.enable)
            
        # Use original batch size from training if not overridden
        if args.batch_size == 1 and 'batch_size' in checkpoint_training_config:
            original_batch_size = checkpoint_training_config['batch_size']
            logger.info(f"🔧 Using original training batch_size: {original_batch_size}")
            train_args.batch_size = original_batch_size
    
    # Override batch size if explicitly specified
    if args.batch_size != 1:
        train_args.batch_size = args.batch_size
        logger.info(f"🔧 Overriding batch_size to: {args.batch_size}")
    
    with ExitStack() as exit_stack:
        try:
            # Load Moshi components
            logger.info("📦 Loading Moshi model and components...")
            checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
                hf_repo=train_args.moshi_paths.hf_repo_id,
                moshi_weights=train_args.moshi_paths.moshi_path,
                mimi_weights=train_args.moshi_paths.mimi_path,
                tokenizer=train_args.moshi_paths.tokenizer_path,
                config_path=train_args.moshi_paths.config_path,
            )
            
            # Load and setup model
            model = get_fsdp_model(train_args, checkpoint_info)
            
            # Load Mimi for audio processing
            mimi = checkpoint_info.get_mimi(device="cuda")
            mimi.eval()
            for p in mimi.parameters():
                p.requires_grad = False
            
            # Setup tokenizers
            spm = checkpoint_info.get_text_tokenizer()
            interleaver = Interleaver(
                spm,
                mimi.frame_rate,
                model.text_padding_token_id,
                model.end_of_text_padding_id,
                model.zero_token_id,
                keep_main_only=True,
            )
            interleaved_tokenizer = InterleavedTokenizer(
                mimi, interleaver, duration_sec=train_args.duration_sec
            )
            
            # Load checkpoint
            logger.info(f"📂 Loading checkpoint from: {checkpoint_dir}")
            state_dict, checkpoint_file = load_checkpoint_state_dict(checkpoint_dir)
            
            # Load state dict into model
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                logger.warning(f"Missing keys in checkpoint: {len(missing_keys)}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys in checkpoint: {len(unexpected_keys)}")
            
            logger.info("✅ Checkpoint loaded successfully")
            
            # Fix mixed precision issues by ensuring all parameters have consistent dtypes
            logger.info("🔧 Fixing mixed precision compatibility...")
            target_dtype = torch.bfloat16  # Use bfloat16 as target
            
            for name, param in model.named_parameters():
                if param.dtype != target_dtype:
                    param.data = param.data.to(target_dtype)
            
            logger.info(f"✅ All parameters converted to {target_dtype}")
            
            # Verify TTT parameters if requested
            if not args.no_ttt_verify:
                ttt_verified = verify_ttt_parameters(model, state_dict)
                if not ttt_verified:
                    logger.warning("⚠️ TTT parameter verification failed, but continuing...")
            
            # Set model to evaluation mode
            model.eval()
            
            # Run evaluations
            results = {}
            
            if args.eval_type in ["standard", "both"]:
                logger.info("=" * 50)
                logger.info("RUNNING STANDARD EVALUATION")
                logger.info("=" * 50)
                
                standard_results = run_standard_evaluation(model, train_args, interleaved_tokenizer)
                results['standard'] = standard_results
                
                logger.info("📊 Standard Evaluation Results:")
                for key, value in standard_results.items():
                    logger.info(f"   {key}: {value:.4f}")
            
            if args.eval_type in ["paper_metrics", "both"]:
                logger.info("=" * 50)
                logger.info("RUNNING PAPER METRICS EVALUATION")
                logger.info("=" * 50)
                
                paper_results = run_paper_metrics_evaluation(model, train_args, mimi, interleaved_tokenizer)
                if paper_results:
                    results['paper_metrics'] = paper_results
                    
                    logger.info("📊 Paper Metrics Results:")
                    for benchmark, result in paper_results.items():
                        if not benchmark.startswith('_'):  # Skip internal keys
                            logger.info(f"   {benchmark}: {result}")
            
            # Print final summary
            logger.info("=" * 50)
            logger.info("EVALUATION SUMMARY")
            logger.info("=" * 50)
            logger.info(f"Checkpoint: {checkpoint_file}")
            logger.info(f"Configuration: {args.config}")
            logger.info(f"Evaluation types: {args.eval_type}")
            
            if 'standard' in results:
                std = results['standard']
                logger.info(f"📊 Standard Metrics:")
                logger.info(f"   Perplexity: {std['eval_perplexity']:.4f}")
                logger.info(f"   Loss: {std['eval_loss']:.4f}")
            
            if 'paper_metrics' in results:
                logger.info(f"📊 Paper Metrics: {len(results['paper_metrics'])} benchmarks completed")
            
            logger.info("✅ Evaluation completed successfully!")
            return 0
            
        except KeyboardInterrupt:
            logger.info("Evaluation interrupted by user")
            return 1
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)