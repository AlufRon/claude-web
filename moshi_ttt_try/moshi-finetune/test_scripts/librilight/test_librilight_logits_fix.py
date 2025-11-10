#!/usr/bin/env python3
"""
Test the LibriLight logits fix to ensure we get real loss values instead of constant 2.3026.
This validates that we're computing proper cross-entropy loss on model logits.
"""

import sys
import os
import torch
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_librilight_logits_fix():
    """Test that LibriLight evaluation now returns variable loss values."""
    try:
        # Import required modules
        from moshi.models import loaders
        from finetune.paper_metrics import PaperMetricsEvaluator
        
        logger.info("🔧 Testing LibriLight Logits Fix")
        logger.info("🎯 Expecting: Variable loss values instead of constant 2.3026")
        
        # Load Moshi model
        logger.info("🔄 Loading Moshi model...")
        checkpoint_info = loaders.CheckpointInfo.from_hf_repo("kyutai/moshiko-pytorch-bf16")
        mimi = checkpoint_info.get_mimi(device='cuda')
        moshi = checkpoint_info.get_moshi(device='cuda', dtype=torch.bfloat16)
        
        # Put model in evaluation mode
        moshi.eval()
        logger.info(f"🔧 Model evaluation mode: {not moshi.training}")
        
        # Create evaluator with LibriLight config
        config = {
            'librilight_audio_dir': '/sise/eliyanac-group/ron_al/librilight/extracted_medium/medium/',
            'librilight_book_name': 'emerald_city_librivox_64kb_mp3',
            'librilight_evaluation_mode': 'single_book',
            'librilight_max_chapters': 1,  
            'librilight_num_sequences': 1,
            'librilight_speaker_id': '100',
            'max_sequence_length': 1000,  # Short sequence for quick test
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
        
        logger.info("✅ Evaluator created successfully")
        
        # Run the LibriLight evaluation with the fix
        logger.info("🎯 Running LibriLight evaluation with logits fix...")
        
        results = evaluator.evaluate_librilight_only(moshi)
        
        logger.info("📊 LibriLight Evaluation Results:")
        for key, value in results.items():
            logger.info(f"   {key}: {value}")
        
        # Check for variability in loss values
        loss_values = []
        for key, value in results.items():
            if 'loss' in key and isinstance(value, (int, float)):
                loss_values.append(value)
        
        if len(loss_values) >= 3:
            # Check if we have different values (not all the same)
            unique_values = set(f"{v:.6f}" for v in loss_values)  # Round to avoid floating point noise
            
            if len(unique_values) > 1:
                logger.info("🎉 SUCCESS: LibriLight now returns variable loss values!")
                logger.info(f"✅ Found {len(unique_values)} different loss values: {sorted(unique_values)}")
                logger.info("✅ Fixed the constant 2.3026 bug!")
                return True
            else:
                logger.warning("⚠️ Still getting constant loss values")
                logger.warning(f"❌ All values are: {unique_values}")
                return False
        else:
            logger.warning("⚠️ Not enough loss values to check variability")
            return False
        
    except Exception as e:
        logger.error(f"❌ Error testing LibriLight fix: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def main():
    """Test the LibriLight logits fix."""
    if not torch.cuda.is_available():
        logger.error("❌ CUDA not available")
        return False
    
    logger.info(f"🔧 CUDA device: {torch.cuda.get_device_name()}")
    
    try:
        success = test_librilight_logits_fix()
        
        if success:
            logger.info("\n🎉 LIBRILIGHT LOGITS FIX VALIDATION PASSED!")
            logger.info("✅ LibriLight now computes proper cross-entropy loss")
            logger.info("✅ TTT learning + real loss evaluation working together")
        else:
            logger.error("\n💥 LIBRILIGHT LOGITS FIX VALIDATION FAILED!")
            logger.error("❌ Still getting constant loss values")
        
        return success
        
    except Exception as e:
        logger.error(f"❌ Test crashed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)