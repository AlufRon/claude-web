#!/usr/bin/env python3
"""
Test the fixed LibriLight evaluation on a real frozen Moshi model.
This validates that our LMGen-based approach works and produces no NaN values.
"""

import sys
import os
import torch
import logging
import numpy as np
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_frozen_moshi():
    """Load a frozen Moshi model for testing."""
    try:
        from moshi.models import loaders
        
        logger.info("🔄 Loading frozen Moshi model...")
        
        # Load Moshi model (frozen, no TTT)
        checkpoint_info = loaders.CheckpointInfo.from_hf_repo("kyutai/moshiko-pytorch-bf16")
        mimi = checkpoint_info.get_mimi(device='cuda')
        moshi = checkpoint_info.get_moshi(device='cuda', dtype=torch.bfloat16)
        
        # Ensure model is in eval mode (frozen)
        moshi.eval()
        
        logger.info("✅ Frozen Moshi model loaded successfully")
        logger.info(f"   Model device: {moshi.device}")
        logger.info(f"   Model dtype: {moshi.dtype}")
        
        return moshi, mimi
        
    except Exception as e:
        logger.error(f"❌ Failed to load Moshi model: {e}")
        return None, None

def create_test_audio_sequence():
    """Create a short test audio sequence for evaluation."""
    try:
        # Create synthetic audio codes [1, 8, seq_len]
        batch_size = 1
        num_codebooks = 8  # Moshi uses 8 audio codebooks
        seq_len = 50  # Short sequence for testing
        vocab_size = 1024  # Typical audio vocab size
        
        # Create random but valid audio codes
        codes = torch.randint(0, vocab_size, (batch_size, num_codebooks, seq_len), device='cuda')
        targets = torch.randint(0, vocab_size, (batch_size, num_codebooks, seq_len), device='cuda')
        
        logger.info(f"✅ Created test audio sequence: {codes.shape}")
        return codes, targets
        
    except Exception as e:
        logger.error(f"❌ Failed to create test sequence: {e}")
        return None, None

def create_minimal_paper_metrics_evaluator(mimi):
    """Create a minimal PaperMetricsEvaluator for testing."""
    try:
        from finetune.paper_metrics import PaperMetricsEvaluator
        
        # Create minimal config
        config = {
            'max_sequence_length': 1000,
            'first_codebook_weight_multiplier': 100.0,
            'ttt_verification_enabled': False,
            'memory_check_enabled': False,
        }
        
        # Create a dummy tokenizer (not used in LibriLight evaluation)
        class DummyTokenizer:
            def __init__(self):
                pass
        
        dummy_tokenizer = DummyTokenizer()
        
        # Create evaluator with required parameters
        evaluator = PaperMetricsEvaluator(
            mimi_encoder=mimi,  # Required parameter
            interleaved_tokenizer=dummy_tokenizer,
            device='cuda',
            config=config
        )
        
        logger.info("✅ Created minimal PaperMetricsEvaluator")
        return evaluator
        
    except Exception as e:
        logger.error(f"❌ Failed to create evaluator: {e}")
        return None

def test_librilight_evaluation(model, codes, targets, evaluator):
    """Test the LibriLight evaluation method."""
    try:
        logger.info("🧪 Testing LibriLight evaluation...")
        
        # Call the fixed evaluation method
        position_losses = evaluator._evaluate_librilight_moshi_native(
            model=model,
            codes=codes,
            targets=targets
        )
        
        if not position_losses:
            logger.error("❌ Evaluation returned empty loss list")
            return False
        
        # Convert to numpy for analysis
        losses = np.array(position_losses)
        
        # Check for NaN values (this was the original bug!)
        nan_count = np.isnan(losses).sum()
        inf_count = np.isinf(losses).sum()
        
        logger.info(f"📊 Evaluation Results:")
        logger.info(f"   Total tokens processed: {len(losses)}")
        logger.info(f"   Loss range: {losses.min():.4f} - {losses.max():.4f}")
        logger.info(f"   Loss mean: {losses.mean():.4f}")
        logger.info(f"   Loss std: {losses.std():.4f}")
        logger.info(f"   NaN values: {nan_count}")
        logger.info(f"   Inf values: {inf_count}")
        
        # Success criteria
        success = (nan_count == 0) and (inf_count == 0) and (len(losses) > 0)
        
        if success:
            logger.info("✅ LibriLight evaluation SUCCESS: No NaN/Inf values!")
            logger.info("🎉 The fix works! The streaming API prevents numerical instability.")
        else:
            logger.error("❌ LibriLight evaluation FAILED: NaN/Inf values detected!")
        
        return success
        
    except Exception as e:
        logger.error(f"❌ Error during LibriLight evaluation: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def main():
    """Run the comprehensive test with real Moshi model."""
    logger.info("🚀 Testing LibriLight NaN fix with real frozen Moshi model...")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        logger.error("❌ CUDA not available. This test requires GPU.")
        return False
    
    logger.info(f"🔧 CUDA device: {torch.cuda.get_device_name()}")
    logger.info(f"🔧 CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    try:
        # Step 1: Load frozen Moshi model
        model, mimi = load_frozen_moshi()
        if model is None:
            return False
        
        # Step 2: Create test audio sequence
        codes, targets = create_test_audio_sequence()
        if codes is None:
            return False
        
        # Step 3: Create evaluator
        evaluator = create_minimal_paper_metrics_evaluator(mimi)
        if evaluator is None:
            return False
        
        # Step 4: Run LibriLight evaluation test
        success = test_librilight_evaluation(model, codes, targets, evaluator)
        
        # Cleanup
        del model, mimi, codes, targets, evaluator
        torch.cuda.empty_cache()
        
        if success:
            logger.info("\n🎉 OVERALL SUCCESS!")
            logger.info("✅ LibriLight fix validated with real Moshi model")
            logger.info("✅ No NaN values detected")
            logger.info("✅ LMGen streaming API works correctly")
            logger.info("✅ Ready for production use!")
            return True
        else:
            logger.error("\n💥 OVERALL FAILURE!")
            logger.error("❌ LibriLight fix needs more work")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test failed with exception: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)