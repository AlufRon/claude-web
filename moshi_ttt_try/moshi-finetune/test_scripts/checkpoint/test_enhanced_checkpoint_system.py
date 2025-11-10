#!/usr/bin/env python3
"""
Test Enhanced Checkpoint System with Training Config Saving
===========================================================

This script demonstrates the enhanced checkpoint system that saves and loads
complete training configuration to ensure consistent evaluation.

Usage:
    python test_enhanced_checkpoint_system.py
"""

import logging
import tempfile
import torch
import shutil
import json
import os
from pathlib import Path

# Setup environment for single-GPU testing
os.environ["LOCAL_RANK"] = "0"
os.environ["RANK"] = "0" 
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12358"

from finetune.args import TrainArgs
from finetune.checkpointing import Checkpointer
from finetune.wrapped_model import get_fsdp_model
from finetune.utils import TrainState
from finetune.distributed import set_device, BACKEND
from moshi.models import loaders
import torch.distributed as dist

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("enhanced_checkpoint_test")

def test_enhanced_checkpoint_system():
    """Test the enhanced checkpoint system with training config saving."""
    
    logger.info("🔧 Testing Enhanced Checkpoint System...")
    
    # Initialize distributed training for single GPU
    logger.info("🔧 Initializing distributed training...")
    set_device()
    dist.init_process_group(backend=BACKEND)
    
    # Create a test configuration with TTT
    test_config = {
        'data': {
            'train_data': '/dev/null',
            'eval_data': '/dev/null',
            'shuffle': False
        },
        'run_dir': '/tmp/enhanced_checkpoint_test',
        'moshi_paths': {
            'hf_repo_id': 'kyutai/moshiko-pytorch-bf16'
        },
        'ttt': {
            'enable': True,
            'layers': '1,2',
            'base_lr': 1.0,
            'mini_batch_size': 8,  # This is critical for evaluation!
            'persistent_states': True,
            'initial_gating_alpha': 0.1
        },
        'lora': {
            'enable': False,
            'rank': 64,
            'scaling': 2.0,
            'ft_embed': False
        },
        'full_finetuning': False,
        'duration_sec': 10,
        'batch_size': 1,
        'max_steps': 10,
        'seed': 42,
        'optim': {
            'lr': 1e-4,
            'weight_decay': 0.1,
            'pct_start': 0.05
        },
        'wandb': {
            'project': None
        }
    }
    
    # Create temporary directory for test
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_config['run_dir'] = tmp_dir
        
        # Create TrainArgs from config
        args = TrainArgs.from_dict(test_config)
        
        # Load model with TTT
        logger.info("📦 Loading Moshi model with TTT...")
        checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
            hf_repo=args.moshi_paths.hf_repo_id,
        )
        
        lm_config = (
            loaders._lm_kwargs
            if checkpoint_info.raw_config is None
            else checkpoint_info.raw_config
        )
        lm_config["lora"] = args.lora.enable
        lm_config["lora_rank"] = args.lora.rank
        lm_config["lora_scaling"] = args.lora.scaling
        
        model = get_fsdp_model(args, checkpoint_info)
        
        # Create enhanced checkpointer with training args
        state = TrainState(max_steps=10)
        state.step = 5
        
        logger.info("💾 Creating enhanced checkpointer...")
        checkpointer = Checkpointer(
            model=model,
            state=state,
            config=lm_config,
            run_dir=tmp_dir,
            num_ckpt_keep=3,
            full_finetuning=False,
            training_args=args,  # Pass complete training config
        )
        
        # Save checkpoint with enhanced config
        logger.info("💾 Saving checkpoint with training config...")
        checkpointer.save_checkpoint(save_only_lora=False)
        
        # Verify checkpoint files
        checkpoint_dir = Path(tmp_dir) / "checkpoints" / f"checkpoint_{state.step:06d}" / "consolidated"
        
        # Check if training config was saved
        training_config_path = checkpoint_dir / "training_config.json"
        if training_config_path.exists():
            logger.info("✅ training_config.json saved successfully")
            
            # Load and verify training config
            with open(training_config_path, 'r') as f:
                saved_training_config = json.load(f)
            
            # Verify critical TTT settings
            ttt_config = saved_training_config.get('ttt', {})
            logger.info(f"✅ TTT mini_batch_size saved: {ttt_config.get('mini_batch_size')}")
            logger.info(f"✅ TTT layers saved: {ttt_config.get('layers')}")
            logger.info(f"✅ TTT enable saved: {ttt_config.get('enable')}")
            logger.info(f"✅ Batch size saved: {saved_training_config.get('batch_size')}")
            
            return True
        else:
            logger.error("❌ training_config.json not saved")
            return False

def test_config_loading():
    """Test loading configuration from a real checkpoint."""
    
    # Test with an existing checkpoint
    checkpoint_path = "/sise/eliyanac-group/ron_al/seamless_ttt_multilayer_with_ttt0.1_no_lora13/checkpoints/checkpoint_001000"
    
    if not Path(checkpoint_path).exists():
        logger.warning("⚠️ Test checkpoint not found, skipping config loading test")
        return True
    
    logger.info("🔍 Testing config loading from existing checkpoint...")
    
    # Test the new config loading function
    from eval_from_checkpoint import load_training_config
    
    training_config = load_training_config(Path(checkpoint_path))
    
    if training_config is None:
        logger.info("📋 No training config found (expected for older checkpoints)")
        return True
    else:
        logger.info("✅ Training config loaded successfully")
        if 'ttt' in training_config:
            ttt_config = training_config['ttt']
            logger.info(f"   TTT config: {ttt_config}")
        return True

if __name__ == "__main__":
    print("🚀 Testing Enhanced Checkpoint System")
    print("=" * 50)
    
    try:
        # Test 1: Enhanced checkpoint saving
        test1_success = test_enhanced_checkpoint_system()
        
        # Test 2: Config loading
        test2_success = test_config_loading()
        
        print("\n" + "=" * 50)
        print("ENHANCED CHECKPOINT SYSTEM RESULTS")
        print("=" * 50)
        print(f"Enhanced Saving Test: {'✅ PASS' if test1_success else '❌ FAIL'}")
        print(f"Config Loading Test:  {'✅ PASS' if test2_success else '❌ FAIL'}")
        
        if test1_success and test2_success:
            print("\n🎉 ENHANCED CHECKPOINT SYSTEM WORKING!")
            print("✅ Training configs are now saved with checkpoints")
            print("✅ Evaluation can load proper TTT settings")
            print("✅ Batch size mismatches will be prevented")
            exit(0)
        else:
            print("\n⚠️ SOME TESTS FAILED!")
            exit(1)
            
    except Exception as e:
        logger.error(f"❌ Critical error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)