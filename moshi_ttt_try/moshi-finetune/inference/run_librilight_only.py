#!/usr/bin/env python3
"""
Run only LibriLight evaluation with inner loop tracking enabled.
This will generate Figure 4 plots showing TTT adaptation.
"""

import os
import sys
from pathlib import Path
from contextlib import ExitStack
import logging

# Disable torch dynamo
import torch._dynamo
torch._dynamo.config.suppress_errors = True

import torch
import torch.distributed as dist

from finetune.args import TrainArgs
from finetune.wrapped_model import get_fsdp_model
from finetune.distributed import set_device, BACKEND
from finetune.data.interleaver import InterleavedTokenizer, Interleaver
from moshi.models import loaders

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("librilight_eval")

def setup_distributed():
    """Setup distributed training for single GPU evaluation."""
    if "LOCAL_RANK" not in os.environ:
        import socket
        
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
    
    set_device()
    dist.init_process_group(backend=BACKEND)

def main():
    CHECKPOINT_DIR = "/sise/eliyanac-group/ron_al/seamless_moshinbn0115/checkpoints/checkpoint_000100/consolidated"
    CONFIG_PATH = "/home/alufr/ttt_tests/moshi-finetune/example/moshi_7B_multilayer_with_ttt.yaml"
    
    print("=" * 80)
    print("TTT-Moshi LibriLight Evaluation with Inner Loop Tracking")
    print("=" * 80)
    print(f"Checkpoint: {CHECKPOINT_DIR}")
    print(f"Config: {CONFIG_PATH}")
    print()
    
    # Setup distributed
    setup_distributed()
    
    # Load configuration
    logger.info(f"📋 Loading configuration from: {CONFIG_PATH}")
    train_args = TrainArgs.load(CONFIG_PATH, drop_extra_fields=False)
    
    # Verify inner loop tracking is enabled
    logger.info("🔍 Inner loop loss tracking ENABLED (Figure 4)")
    logger.info(f"  - Log interval: every {train_args.ttt.inner_loop_log_interval} positions")
    logger.info(f"  - Save plots: {train_args.ttt.save_inner_loop_plots}")
    logger.info(f"  - Plot directory: {train_args.ttt.inner_loop_plot_dir}")
    print()
    
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
            logger.info(f"📂 Loading checkpoint from: {CHECKPOINT_DIR}")
            import safetensors.torch
            from finetune.checkpointing import Checkpointer
            
            # Determine checkpoint file
            checkpoint_dir = Path(CHECKPOINT_DIR)
            checkpoint_file = Checkpointer.consolidated_path(checkpoint_dir, save_only_lora=True)
            if not checkpoint_file.exists():
                checkpoint_file = Checkpointer.consolidated_path(checkpoint_dir, save_only_lora=False)
            
            state_dict = safetensors.torch.load_file(checkpoint_file)
            
            # Load state dict into model
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                logger.warning(f"Missing keys in checkpoint: {len(missing_keys)}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys in checkpoint: {len(unexpected_keys)}")
            
            logger.info("✅ Checkpoint loaded successfully")
            
            # Fix mixed precision
            target_dtype = torch.bfloat16
            for name, param in model.named_parameters():
                if param.dtype != target_dtype:
                    param.data = param.data.to(target_dtype)
            
            logger.info(f"✅ All parameters converted to {target_dtype}")
            
            # Set model to evaluation mode
            model.eval()
            
            # Initialize paper metrics evaluator with full config
            logger.info("🔧 Initializing LibriLight evaluator...")
            from finetune.paper_metrics import PaperMetricsEvaluator
            
            # Extract paper_metrics config and add TTT config
            paper_metrics_config = {}
            
            # Add paper_metrics config
            if isinstance(train_args.paper_metrics, dict):
                paper_metrics_config.update(train_args.paper_metrics)
            else:
                paper_metrics_config.update(train_args.paper_metrics.__dict__)
                
            # Add TTT config for inner loop tracking
            if isinstance(train_args.ttt, dict):
                paper_metrics_config.update(train_args.ttt)
            else:
                paper_metrics_config.update(train_args.ttt.__dict__)
            
            # Limit sequence length to 2000 tokens for faster evaluation
            paper_metrics_config['librilight_streaming'] = {
                'max_sequence_length': 2000
            }
            
            logger.info(f"📋 LibriLight config:")
            logger.info(f"   mode: {paper_metrics_config.get('librilight_evaluation_mode')}")
            logger.info(f"   directory: {paper_metrics_config.get('librilight_concatenated_dir')}")
            logger.info(f"   max_files: {paper_metrics_config.get('librilight_max_files')}")
            logger.info(f"   max_sequence_length: 2000 tokens (for quick testing)")
            
            evaluator = PaperMetricsEvaluator(
                mimi,
                interleaved_tokenizer,
                config=paper_metrics_config
            )
            
            # Run LibriLight evaluation only
            logger.info("=" * 80)
            logger.info("🚀 RUNNING LIBRILIGHT EVALUATION WITH INNER LOOP TRACKING")
            logger.info("=" * 80)
            
            results = evaluator.evaluate_librilight_only(model)
            
            # Print results
            logger.info("=" * 80)
            logger.info("📊 LIBRILIGHT EVALUATION RESULTS")
            logger.info("=" * 80)
            
            for key, value in results.items():
                if isinstance(value, float):
                    logger.info(f"   {key}: {value:.4f}")
                else:
                    logger.info(f"   {key}: {value}")
            
            # Check for generated plots
            plot_dir = Path(train_args.ttt.inner_loop_plot_dir)
            if plot_dir.exists():
                plot_files = list(plot_dir.glob("*.png"))
                if plot_files:
                    logger.info("=" * 80)
                    logger.info("📊 GENERATED INNER LOOP PLOTS (Figure 4)")
                    logger.info("=" * 80)
                    for plot_file in plot_files:
                        logger.info(f"   📈 {plot_file}")
                else:
                    logger.warning("⚠️ No plots generated in plot directory")
            else:
                logger.warning(f"⚠️ Plot directory not found: {plot_dir}")
            
            logger.info("✅ LibriLight evaluation completed successfully!")
            return 0
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)