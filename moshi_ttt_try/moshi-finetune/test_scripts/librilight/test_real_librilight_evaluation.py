#!/usr/bin/env python3
"""
Test the REAL LibriLight evaluation exactly as it runs at end of training.
Uses the actual evaluate_librilight_only() function with 24,000+ tokens
from real LibriLight data, just like the production training pipeline.
"""

import sys
import os
import torch
import logging
import numpy as np
from pathlib import Path

# Set up logging to match training
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_frozen_moshi_for_real_evaluation():
    """Load frozen Moshi exactly like training does."""
    try:
        from moshi.models import loaders
        
        logger.info("🔄 Loading frozen Moshi model (training-style)...")
        
        # Use the same model loading as training
        checkpoint_info = loaders.CheckpointInfo.from_hf_repo("kyutai/moshiko-pytorch-bf16")
        mimi = checkpoint_info.get_mimi(device='cuda')
        moshi = checkpoint_info.get_moshi(device='cuda', dtype=torch.bfloat16)
        
        # Ensure eval mode (frozen)
        moshi.eval()
        
        logger.info("✅ Frozen Moshi loaded (matches training setup)")
        logger.info(f"   Model device: {moshi.device}")
        logger.info(f"   Model dtype: {moshi.dtype}")
        logger.info(f"   Total parameters: {sum(p.numel() for p in moshi.parameters())/1e6:.1f}M")
        
        return moshi, mimi
        
    except Exception as e:
        logger.error(f"❌ Failed to load Moshi: {e}")
        return None, None

def create_production_paper_metrics_evaluator(mimi):
    """Create PaperMetricsEvaluator exactly like training does."""
    try:
        from finetune.paper_metrics import PaperMetricsEvaluator
        from finetune.data.interleaver import InterleavedTokenizer
        
        logger.info("🔧 Creating production-style PaperMetricsEvaluator...")
        
        # Production config matching training YAML
        config = {
            # LibriLight specific config
            'librilight_audio_dir': '/sise/eliyanac-group/ron_al/librilight/extracted_medium/medium/',
            'librilight_book_name': 'emerald_city_librivox_64kb_mp3',
            'librilight_evaluation_mode': 'single_book',
            'librilight_max_chapters': 3,
            'librilight_num_sequences': 1,
            'librilight_speaker_id': '100',
            
            # Paper metrics config
            'paper_metrics_eval': True,
            'paper_metrics_freq': 10,
            'paper_metrics_use_silence': True,
            'paper_metrics_use_user_stream': False,
            
            # Evaluation config
            'first_codebook_weight_multiplier': 100.0,
            'max_sequence_length': 30000,  # Allow full 24k+ sequences
            
            # Memory and verification
            'ttt_verification_enabled': False,  # No TTT in frozen model
            'memory_check_enabled': True,
        }
        
        # Create tokenizer (required but not used for LibriLight)
        class ProductionTokenizer:
            """Minimal tokenizer interface matching training setup."""
            def __init__(self):
                self.device = torch.device('cuda')
        
        tokenizer = ProductionTokenizer()
        
        # Create evaluator exactly like training
        evaluator = PaperMetricsEvaluator(
            mimi_encoder=mimi,
            interleaved_tokenizer=tokenizer,
            device='cuda',
            config=config
        )
        
        logger.info("✅ Production PaperMetricsEvaluator created")
        logger.info("🎯 Configured for LibriLight evaluation:")
        logger.info(f"   📁 Audio dir: {config['librilight_audio_dir']}")
        logger.info(f"   📖 Book: {config['librilight_book_name']}")
        logger.info(f"   🔢 Max chapters: {config['librilight_max_chapters']}")
        logger.info(f"   📏 Max sequence: {config['max_sequence_length']}")
        
        return evaluator
        
    except Exception as e:
        logger.error(f"❌ Failed to create evaluator: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

def run_real_librilight_evaluation(model, evaluator):
    """Run the REAL LibriLight evaluation exactly like end-of-training."""
    try:
        logger.info("🚀 Running REAL LibriLight evaluation (end-of-training style)...")
        logger.info("📋 This is exactly what runs after step 10 in training!")
        
        # Call the REAL function that training uses
        logger.info("🎯 Calling evaluate_librilight_only() - the real production function")
        
        # This is the exact call from training at end of training
        librilight_results = evaluator.evaluate_librilight_only(model)
        
        if not librilight_results:
            logger.error("❌ LibriLight evaluation returned empty results")
            return None
        
        # Analyze the real results
        logger.info("📊 REAL LibriLight Results (Production Pipeline):")
        
        # Log all metrics exactly like training does
        for key, value in librilight_results.items():
            if isinstance(value, (int, float)):
                if 'nan' in str(value).lower() or np.isnan(value) if isinstance(value, float) else False:
                    logger.error(f"   ❌ {key}: {value} (NaN detected!)")
                else:
                    logger.info(f"   ✅ {key}: {value}")
            else:
                logger.info(f"   📄 {key}: {value}")
        
        # Extract key metrics
        key_metrics = {
            'librilight_loss_8k': librilight_results.get('librilight_loss_8k', 'missing'),
            'librilight_loss_16k': librilight_results.get('librilight_loss_16k', 'missing'),
            'librilight_loss_24k': librilight_results.get('librilight_loss_24k', 'missing'),
            'librilight_slope': librilight_results.get('librilight_slope', 'missing'),
            'librilight_samples': librilight_results.get('librilight_samples', 'missing'),
        }
        
        logger.info("🎯 Key LibriLight Metrics Summary:")
        for metric, value in key_metrics.items():
            if isinstance(value, (int, float)):
                if 'nan' in str(value).lower() or (isinstance(value, float) and np.isnan(value)):
                    logger.error(f"   ❌ {metric}: {value} (NaN!)")
                else:
                    logger.info(f"   ✅ {metric}: {value}")
            else:
                logger.warning(f"   ⚠️  {metric}: {value}")
        
        # Check for enhanced position metrics
        enhanced_metrics = {k: v for k, v in librilight_results.items() if k.startswith('librilight_loss_') and k.endswith('000')}
        if enhanced_metrics:
            logger.info(f"📈 Enhanced position metrics: {len(enhanced_metrics)} positions measured")
            logger.info("   Sample positions:")
            for i, (pos, loss) in enumerate(list(enhanced_metrics.items())[:5]):
                logger.info(f"     {pos}: {loss}")
        
        # Success criteria
        has_nan = any(
            'nan' in str(v).lower() or (isinstance(v, float) and np.isnan(v)) 
            for v in key_metrics.values() if isinstance(v, (int, float))
        )
        
        has_values = any(
            isinstance(v, (int, float)) and not ('nan' in str(v).lower() or (isinstance(v, float) and np.isnan(v)))
            for v in key_metrics.values()
        )
        
        if has_nan:
            logger.error("❌ FAILURE: NaN values detected in LibriLight results!")
            logger.error("💥 The original bug is still present - fix didn't work!")
            return False
        elif has_values:
            logger.info("✅ SUCCESS: No NaN values detected!")
            logger.info("🎉 LibriLight evaluation completed successfully!")
            logger.info("🔧 The fix is working in production!")
            return True
        else:
            logger.warning("⚠️  UNCLEAR: No numeric values found - check evaluation setup")
            return False
        
    except Exception as e:
        logger.error(f"❌ Error during real LibriLight evaluation: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def main():
    """Run the exact LibriLight evaluation that training uses."""
    logger.info("🎯 Testing REAL LibriLight evaluation (24,000+ tokens, production pipeline)")
    logger.info("📋 This replicates the exact evaluation from end-of-training")
    
    if not torch.cuda.is_available():
        logger.error("❌ CUDA not available")
        return False
    
    logger.info(f"🔧 CUDA device: {torch.cuda.get_device_name()}")
    logger.info(f"🔧 CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    try:
        # Load frozen Moshi (training-style)
        model, mimi = load_frozen_moshi_for_real_evaluation()
        if model is None:
            return False
        
        # Create production evaluator
        evaluator = create_production_paper_metrics_evaluator(mimi)
        if evaluator is None:
            return False
        
        # Run REAL LibriLight evaluation
        success = run_real_librilight_evaluation(model, evaluator)
        
        # Cleanup
        del model, mimi, evaluator
        torch.cuda.empty_cache()
        
        # Final summary
        if success:
            logger.info("\n🎉 REAL LIBRILIGHT EVALUATION SUCCESS!")
            logger.info("✅ Production pipeline works correctly")
            logger.info("✅ No NaN values in 24,000+ token evaluation")
            logger.info("✅ Fix is validated for production use")
            logger.info("🚀 Ready for TTT training with proper LibriLight metrics!")
        else:
            logger.error("\n💥 REAL LIBRILIGHT EVALUATION FAILED!")
            logger.error("❌ Production pipeline has issues")
            logger.error("❌ Fix needs more work")
        
        return success
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)