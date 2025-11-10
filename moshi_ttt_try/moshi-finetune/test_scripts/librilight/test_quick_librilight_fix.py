#!/usr/bin/env python3
"""
Quick test of the LibriLight logits fix with minimal sequence length.
Just verify we get different loss values at different positions.
"""

import sys
import os
import torch
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_quick_librilight_fix():
    """Quick test with very short sequence to verify loss variability."""
    try:
        # Import required modules
        from moshi.models import loaders
        from finetune.paper_metrics import PaperMetricsEvaluator
        
        logger.info("🚀 Quick LibriLight Logits Fix Test")
        logger.info("🎯 Testing with SHORT sequence to verify different losses")
        
        # Load Moshi model
        logger.info("🔄 Loading Moshi model...")
        checkpoint_info = loaders.CheckpointInfo.from_hf_repo("kyutai/moshiko-pytorch-bf16")
        mimi = checkpoint_info.get_mimi(device='cuda')
        moshi = checkpoint_info.get_moshi(device='cuda', dtype=torch.bfloat16)
        
        # Put model in evaluation mode
        moshi.eval()
        
        # Create evaluator with VERY short config for quick test
        config = {
            'librilight_audio_dir': '/sise/eliyanac-group/ron_al/librilight/extracted_medium/medium/',
            'librilight_book_name': 'emerald_city_librivox_64kb_mp3',
            'librilight_evaluation_mode': 'single_book',
            'librilight_max_chapters': 1,  
            'librilight_num_sequences': 1,
            'librilight_speaker_id': '100',
            'max_sequence_length': 100,  # VERY short for quick test!
        }
        
        # Create minimal tokenizer interface
        class MinimalTokenizer:
            def __init__(self):
                self.device = torch.device('cuda')
        
        evaluator = PaperMetricsEvaluator(
            mimi_encoder=mimi,
            interleaved_tokenizer=MinimalTokenizer(),
            device='cuda',
            config=config
        )
        
        logger.info("✅ Running QUICK LibriLight test...")
        
        # Run the evaluation
        results = evaluator.evaluate_librilight_only(moshi)
        
        logger.info("📊 Quick LibriLight Results:")
        for key, value in results.items():
            if isinstance(value, (int, float)):
                logger.info(f"   {key}: {value:.6f}")
        
        # Extract position losses and check for variability
        position_losses = []
        position_keys = []
        for key, value in results.items():
            if 'loss' in key and '_' in key and isinstance(value, (int, float)):
                position_losses.append(value)
                position_keys.append(key)
        
        if len(position_losses) >= 2:
            # Check if we have different values
            unique_values = set(f"{v:.6f}" for v in position_losses)
            
            logger.info(f"🔍 Found {len(position_losses)} position losses:")
            for key, loss in zip(position_keys, position_losses):
                logger.info(f"   {key}: {loss:.6f}")
            
            if len(unique_values) > 1:
                logger.info("🎉 SUCCESS: Different loss values found!")
                logger.info(f"✅ Unique values: {sorted(unique_values)}")
                logger.info("✅ Fixed the constant 2.3026 bug!")
                return True
            else:
                logger.warning("⚠️ Still getting constant loss values:")
                logger.warning(f"❌ All values: {unique_values}")
                return False
        else:
            logger.warning(f"⚠️ Only found {len(position_losses)} position losses - need more for comparison")
            return False
        
    except Exception as e:
        logger.error(f"❌ Error in quick test: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = test_quick_librilight_fix()
    if success:
        print("\n🎉 LIBRILIGHT LOGITS FIX VERIFIED!")
    else:
        print("\n💥 LIBRILIGHT LOGITS FIX FAILED!")
    sys.exit(0 if success else 1)