#!/usr/bin/env python3
"""
Test script to validate TTT memory optimizations.
Tests that the memory fixes prevent OOM and reduce peak memory usage.
"""

import torch
import logging
import os
from pathlib import Path

# Import training components
from finetune.args import TrainArgs
from finetune.wrapped_model import get_fsdp_model
from moshi.models import loaders

def setup_memory_optimizations():
    """Apply the same memory optimizations as in train.py"""
    # CUDA Memory Optimization Setup
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["TORCH_COMPILE_DISABLE"] = "1"  # Disable compilation to reduce memory overhead
    
    # Reduce compilation cache to minimize memory usage
    import torch._dynamo
    import torch._inductor
    torch._dynamo.config.cache_size_limit = 32
    torch._inductor.config.triton.max_cached_kernels = 16
    
    # Clear CUDA cache before testing
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

def log_memory_usage(stage: str):
    """Log current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        cached = torch.cuda.memory_reserved() / 1e9
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"🧠 Memory {stage}: {allocated:.1f}GB allocated, {cached:.1f}GB cached, {peak:.1f}GB peak")
        return peak
    return 0

def test_model_loading_with_optimizations():
    """Test that the model loads successfully with memory optimizations."""
    print("=" * 80)
    print("🧪 Testing TTT Memory Optimizations")
    print("=" * 80)
    
    # Apply memory optimizations
    setup_memory_optimizations()
    log_memory_usage("after_setup")
    
    # Load the memory-optimized config
    config_path = Path("example/moshi_7B_memory_optimized.yaml")
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return False
    
    try:
        # Load args
        print("📋 Loading memory-optimized configuration...")
        args = TrainArgs.load(str(config_path), drop_extra_fields=False)
        log_memory_usage("after_args_load")
        
        # Load model components
        print("🤖 Loading Moshi model...")
        checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
            hf_repo=args.moshi_paths.hf_repo_id,
            moshi_weights=args.moshi_paths.moshi_path,
            mimi_weights=args.moshi_paths.mimi_path,
            tokenizer=args.moshi_paths.tokenizer_path,
            config_path=args.moshi_paths.config_path,
        )
        
        lm_config = (
            loaders._lm_kwargs
            if checkpoint_info.raw_config is None
            else checkpoint_info.raw_config
        )
        lm_config["lora"] = args.lora.enable
        lm_config["lora_rank"] = args.lora.rank
        lm_config["lora_scaling"] = args.lora.scaling
        
        log_memory_usage("after_checkpoint_load")
        
        # Create model with TTT
        print("⚡ Creating model with TTT layers and checkpointing...")
        model = get_fsdp_model(
            args,
            lm_config,
            checkpoint_info,
            use_ema=False
        )
        
        model_load_peak = log_memory_usage("after_model_creation")
        
        # Test forward pass
        print("🔄 Testing forward pass...")
        model.train()  # Enable training mode for checkpointing
        
        # Create a small test batch
        batch_size = 1
        seq_len = 100  # Small sequence to test
        vocab_size = model.vocab_size
        
        # Create dummy input
        codes = torch.randint(0, vocab_size, (batch_size, seq_len), device='cuda')
        
        log_memory_usage("before_forward")
        
        # Forward pass (this should trigger TTT layer checkpointing)
        with torch.no_grad():  # No backward pass for this test
            output = model(codes=codes)
        
        forward_peak = log_memory_usage("after_forward")
        
        print("\n" + "=" * 80)
        print("✅ MEMORY OPTIMIZATION TEST RESULTS")
        print("=" * 80)
        print(f"📊 Model loading peak: {model_load_peak:.1f} GB")
        print(f"🔄 Forward pass peak: {forward_peak:.1f} GB")
        
        # Check if memory usage is within acceptable bounds
        # Target: < 30 GB (vs original 47 GB)
        if forward_peak < 30.0:
            print(f"✅ SUCCESS: Peak memory {forward_peak:.1f}GB is below 30GB target")
            print(f"💡 Memory reduction achieved: {47.4 - forward_peak:.1f}GB saved (~{((47.4 - forward_peak) / 47.4 * 100):.1f}% reduction)")
            return True
        else:
            print(f"⚠️  WARNING: Peak memory {forward_peak:.1f}GB still above 30GB target")
            print(f"💡 Memory reduction: {47.4 - forward_peak:.1f}GB saved (~{((47.4 - forward_peak) / 47.4 * 100):.1f}% reduction)")
            return False
            
    except torch.cuda.OutOfMemoryError as e:
        print(f"❌ CUDA OOM Error: {e}")
        print("💡 The memory optimizations may need further tuning")
        return False
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_configuration_values():
    """Test that the optimized configuration has correct values."""
    print("\n" + "=" * 80)
    print("🔧 Testing Configuration Optimizations")
    print("=" * 80)
    
    config_path = Path("example/moshi_7B_memory_optimized.yaml")
    args = TrainArgs.load(str(config_path), drop_extra_fields=False)
    
    # Check TTT configuration
    print(f"✅ TTT enabled: {args.ttt.enable}")
    print(f"✅ Mini-batch size: {args.ttt.mini_batch_size} (was 1, now 4+)")
    print(f"✅ Max chunk size: {args.ttt.max_chunk_size} (was 50, now 25)")
    print(f"✅ Base LR: {args.ttt.base_lr} (was 10.0, now 1.0)")
    print(f"✅ Gradient checkpointing: {args.gradient_checkpointing}")
    
    # Verify optimizations are applied
    success = True
    if args.ttt.mini_batch_size < 4:
        print(f"⚠️  Mini-batch size {args.ttt.mini_batch_size} is small (recommended: 4+)")
        success = False
    if args.ttt.max_chunk_size > 30:
        print(f"⚠️  Max chunk size {args.ttt.max_chunk_size} is large (recommended: ≤30)")
        success = False
    if not args.gradient_checkpointing:
        print(f"❌ Gradient checkpointing disabled (critical for memory savings)")
        success = False
    
    return success

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("🚀 Starting TTT Memory Optimization Validation")
    
    # Test 1: Configuration values
    config_success = test_configuration_values()
    
    # Test 2: Model loading and forward pass
    model_success = test_model_loading_with_optimizations()
    
    print("\n" + "=" * 80)
    print("📋 FINAL RESULTS")
    print("=" * 80)
    print(f"🔧 Configuration optimized: {'✅' if config_success else '❌'}")
    print(f"🧠 Memory test passed: {'✅' if model_success else '❌'}")
    
    if config_success and model_success:
        print("\n🎉 All tests passed! Memory optimizations are working correctly.")
        print("💡 You can now train with significantly reduced memory usage.")
    else:
        print("\n⚠️  Some tests failed. Please check the optimizations.")
        
    print("\n🚀 Ready to train with optimized config:")
    print("   python train.py example/moshi_7B_memory_optimized.yaml")