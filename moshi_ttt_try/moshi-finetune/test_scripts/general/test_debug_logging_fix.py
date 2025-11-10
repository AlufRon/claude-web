#!/usr/bin/env python3
"""
Test script to verify that DEBUG logging is working and to test LibriLight evaluation
with comprehensive logging to identify the exact failure point.
"""

import logging
import sys
import torch
from pathlib import Path

# Add current directory to Python path
sys.path.append(str(Path(__file__).parent))

# Import the logging setup
from finetune.monitoring.utils import set_logger

# Set up DEBUG logging
set_logger(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_debug_logging():
    """Test that DEBUG logging is working"""
    logger.debug("🐛 DEBUG message - this should be visible now!")
    logger.info("ℹ️ INFO message - this should also be visible")
    logger.warning("⚠️ WARNING message - this should definitely be visible")
    logger.error("🚨 ERROR message - this should be very visible")
    
    print("✅ If you see the DEBUG message above, logging level change worked!")

def test_quick_librilight_evaluation():
    """Test LibriLight evaluation with a very short sequence to trigger the debug messages"""
    try:
        # Import necessary modules
        from moshi.models import loaders  
        from finetune.paper_metrics import create_paper_metrics_evaluator
        
        logger.info("🧪 Starting quick LibriLight evaluation test with DEBUG logging...")
        
        # Load minimal Moshi models
        logger.debug("🔍 Loading MIMI encoder...")
        mimi = loaders.get_mimi()
        mimi = mimi.cuda().eval()
        
        logger.debug("🔍 Loading interleaved tokenizer...")  
        tokenizer = loaders.get_interleaved_tokenizer()
        
        # Create evaluator with debug-friendly config
        config = {
            'librilight_audio_dir': '/sise/eliyanac-group/ron_al/librilight/extracted_medium/medium/',
            'librilight_speaker_id': '100',
            'librilight_book_name': 'emerald_city_librivox_64kb_mp3', 
            'librilight_evaluation_mode': 'single_book',
            'librilight_max_chapters': 1,  # Just 1 chapter for quick test
            'librilight_num_sequences': 1,
        }
        
        logger.debug("🔍 Creating paper metrics evaluator...")
        evaluator = create_paper_metrics_evaluator(mimi, tokenizer, device="cuda", config=config)
        
        # Set evaluator to use fixed streaming with extensive logging
        evaluator.use_fixed_streaming = True
        evaluator.max_sequence_length = 100  # Very short sequence for quick test
        evaluator.verify_loss_computation = True
        evaluator.ttt_verification_enabled = True
        evaluator.memory_check_enabled = True
        
        logger.info("🧪 Loading minimal Moshi model...")
        
        # Load minimal Moshi model (no TTT for this debug test)
        model_args = {
            'hf_repo_id': 'kyutai/moshiko-pytorch-bf16',
        }
        
        model = loaders.get_moshi_lm(**model_args)
        model = model.cuda().eval()
        
        logger.info("🧪 Running LibriLight evaluation with extensive DEBUG logging...")
        logger.info("🧪 This should now show all the hidden debug messages that were causing silent failure!")
        
        # Run the evaluation - this should now show detailed debug logs
        results = evaluator.evaluate_librilight_only(model)
        
        logger.info(f"🧪 Results: {results}")
        
        if results.get('librilight_samples', 0) > 0:
            logger.info("✅ LibriLight evaluation succeeded!")
        else:
            logger.error("🚨 LibriLight evaluation still failing - check debug logs above for the exact failure point")
            
    except Exception as e:
        logger.error(f"🚨 Test failed with exception: {e}")
        import traceback
        logger.error(f"🚨 Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    logger.info("🧪 Testing DEBUG logging fix...")
    
    # Test 1: Verify debug logging works
    test_debug_logging()
    
    # Test 2: Quick LibriLight evaluation with debug logging
    if torch.cuda.is_available():
        test_quick_librilight_evaluation()
    else:
        logger.warning("⚠️ CUDA not available, skipping LibriLight test")
        
    logger.info("🧪 Debug logging test completed!")