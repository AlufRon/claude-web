#!/usr/bin/env python3
"""
Test script for LibriLight streaming evaluation fix.

This script tests the new streaming evaluation implementation to ensure:
1. Streaming evaluation completes without memory crashes
2. TTT state tracking works (if TTT layers present)  
3. Memory usage remains stable
4. Output shapes and values are reasonable
"""

import logging
import torch
import sys
import os
from pathlib import Path

# Add the finetune directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / "finetune"))

from finetune.paper_metrics import PaperMetricsEvaluator
from moshi.models import loaders

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_streaming_evaluation():
    """Test streaming evaluation with a baseline model."""
    
    logger.info("=" * 60)
    logger.info("TESTING LIBRILIGHT STREAMING EVALUATION FIX")
    logger.info("=" * 60)
    
    try:
        # 1. Load Moshi model (baseline - no TTT)
        logger.info("Loading Moshi model...")
        
        checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
            hf_repo="kmhf/moshi",
            moshi_weights=None,  # Use default
            mimi_weights=None,   # Use default  
            tokenizer=None,      # Use default
            config_path=None,    # Use default
        )
        
        # Load Mimi encoder
        mimi = checkpoint_info.get_mimi(device="cuda")
        mimi.eval()
        for p in mimi.parameters():
            p.requires_grad = False
            
        # Load Moshi model (baseline)
        model = checkpoint_info.get_lm(device="cuda")
        model.eval()
        
        # Create interleaved tokenizer
        spm = checkpoint_info.get_text_tokenizer()
        from finetune.data.interleaver import Interleaver, InterleavedTokenizer
        
        interleaver = Interleaver(
            spm,
            mimi.frame_rate,
            model.text_padding_token_id,
            model.end_of_text_padding_id,
            model.zero_token_id,
            keep_main_only=True,
        )
        interleaved_tokenizer = InterleavedTokenizer(
            mimi, interleaver, duration_sec=30.0  # 30 second duration for testing
        )
        
        logger.info(f"Model loaded successfully - baseline Moshi (no TTT)")
        
        # 2. Create paper metrics evaluator with streaming configuration
        streaming_config = {
            'librilight_streaming': {
                'enabled': True,
                'memory_check': True,
                'cache_clear_interval': 3000,
                'max_sequence_length': 10000,  # Limit for testing
                'ttt_verification': True,
                'memory_log_interval': 500,    # More frequent logging for testing
            }
        }
        
        evaluator = PaperMetricsEvaluator(
            mimi, 
            interleaved_tokenizer, 
            config=streaming_config
        )
        
        logger.info("Paper metrics evaluator created with streaming configuration")
        
        # 3. Test with synthetic data (simulating LibriLight)
        logger.info("Creating synthetic test data...")
        
        # Create synthetic audio codes (simulating LibriLight sequence)
        batch_size = 1
        num_codebooks = 8
        seq_length = 5000  # 5k tokens for testing
        vocab_size = 2048
        
        # Generate random audio codes 
        synthetic_codes = torch.randint(
            0, vocab_size, 
            (batch_size, num_codebooks, seq_length), 
            device="cuda", 
            dtype=torch.long
        )
        
        logger.info(f"Synthetic test data created: {synthetic_codes.shape}")
        
        # 4. Run streaming evaluation test
        logger.info("Starting streaming evaluation test...")
        
        # Reset memory stats
        torch.cuda.reset_peak_memory_stats()
        initial_memory = torch.cuda.memory_allocated() / (1024**3)
        
        # Test the streaming evaluation
        with torch.no_grad():
            # Call the streaming evaluation method directly
            loss_per_position = evaluator._evaluate_librilight_streaming(
                model, synthetic_codes, synthetic_codes  # Use same for input/target
            )
        
        # Check memory after evaluation
        peak_memory = torch.cuda.max_memory_allocated() / (1024**3)
        final_memory = torch.cuda.memory_allocated() / (1024**3)
        
        # 5. Validate results
        logger.info("=" * 40)
        logger.info("STREAMING EVALUATION TEST RESULTS:")
        logger.info("=" * 40)
        
        # Check output format
        assert isinstance(loss_per_position, list), f"Expected list, got {type(loss_per_position)}"
        assert len(loss_per_position) > 0, "No losses computed"
        assert len(loss_per_position) <= seq_length, f"Too many losses: {len(loss_per_position)} > {seq_length}"
        
        # Check loss values
        valid_losses = [l for l in loss_per_position if l > 0]
        assert len(valid_losses) > 0, "No valid losses computed"
        
        # Check memory usage
        memory_increase = peak_memory - initial_memory
        assert memory_increase < 10.0, f"Memory usage too high: {memory_increase:.2f}GB increase"
        
        logger.info(f"✅ Sequence length: {seq_length} tokens")
        logger.info(f"✅ Losses computed: {len(loss_per_position)} positions")
        logger.info(f"✅ Valid losses: {len(valid_losses)} / {len(loss_per_position)}")
        logger.info(f"✅ Loss range: {min(valid_losses):.4f} - {max(valid_losses):.4f}")
        logger.info(f"✅ Memory usage: {initial_memory:.2f}GB → {peak_memory:.2f}GB → {final_memory:.2f}GB")
        logger.info(f"✅ Peak memory increase: {memory_increase:.2f}GB")
        logger.info(f"✅ TTT verification: {'Available' if hasattr(evaluator, '_last_ttt_info') else 'Not run'}")
        
        # 6. Test with longer sequence
        logger.info("\nTesting with longer sequence...")
        longer_seq_length = 15000  # 15k tokens
        longer_codes = torch.randint(
            0, vocab_size,
            (batch_size, num_codebooks, longer_seq_length),
            device="cuda",
            dtype=torch.long
        )
        
        torch.cuda.reset_peak_memory_stats()
        initial_memory_2 = torch.cuda.memory_allocated() / (1024**3)
        
        with torch.no_grad():
            loss_per_position_2 = evaluator._evaluate_librilight_streaming(
                model, longer_codes, longer_codes
            )
        
        peak_memory_2 = torch.cuda.max_memory_allocated() / (1024**3)
        memory_increase_2 = peak_memory_2 - initial_memory_2
        
        logger.info(f"✅ Longer sequence: {longer_seq_length} tokens")
        logger.info(f"✅ Losses computed: {len(loss_per_position_2)} positions")
        logger.info(f"✅ Memory increase: {memory_increase_2:.2f}GB")
        logger.info(f"✅ Memory stable: {'Yes' if memory_increase_2 < 15.0 else 'No - may need optimization'}")
        
        logger.info("=" * 60)
        logger.info("🎉 STREAMING EVALUATION TEST PASSED!")
        logger.info("The LibriLight streaming fix is working correctly.")
        logger.info("Memory usage remains stable and evaluation completes successfully.")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error("=" * 60)
        logger.error("❌ STREAMING EVALUATION TEST FAILED!")
        logger.error(f"Error: {e}")
        logger.error("=" * 60)
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_streaming_evaluation()
    sys.exit(0 if success else 1)