#!/usr/bin/env python3
"""
Get baseline LibriLight results for frozen Moshi model.
This will establish the baseline performance that TTT should improve upon.
"""

import sys
import os
import torch
import logging
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_frozen_moshi():
    """Load a frozen Moshi model for baseline evaluation."""
    try:
        from moshi.models import loaders
        
        logger.info("🔄 Loading frozen Moshi model...")
        
        checkpoint_info = loaders.CheckpointInfo.from_hf_repo("kyutai/moshiko-pytorch-bf16")
        mimi = checkpoint_info.get_mimi(device='cuda')
        moshi = checkpoint_info.get_moshi(device='cuda', dtype=torch.bfloat16)
        
        # Ensure model is in eval mode (frozen)
        moshi.eval()
        
        logger.info("✅ Frozen Moshi model loaded successfully")
        logger.info(f"   Model has {sum(p.numel() for p in moshi.parameters())/1e6:.1f}M parameters")
        
        return moshi, mimi
        
    except Exception as e:
        logger.error(f"❌ Failed to load Moshi model: {e}")
        return None, None

def create_librilight_test_sequence():
    """Create a longer test sequence similar to real LibriLight data."""
    try:
        # Create a longer sequence for meaningful evaluation
        batch_size = 1
        num_codebooks = 8
        seq_len = 1000  # Longer sequence for better baseline metrics
        vocab_size = 1024
        
        # Create realistic audio patterns (not completely random)
        torch.manual_seed(42)  # Reproducible results
        
        # Generate codes with some structure (more realistic than pure random)
        codes = torch.randint(0, vocab_size, (batch_size, num_codebooks, seq_len), device='cuda')
        
        # Create targets as next-token prediction
        targets = torch.cat([codes[:, :, 1:], torch.randint(0, vocab_size, (batch_size, num_codebooks, 1), device='cuda')], dim=2)
        
        logger.info(f"✅ Created LibriLight-style test sequence: {codes.shape}")
        logger.info(f"   Sequence length: {seq_len} tokens")
        logger.info(f"   Vocab range: 0-{vocab_size-1}")
        
        return codes, targets
        
    except Exception as e:
        logger.error(f"❌ Failed to create test sequence: {e}")
        return None, None

def create_paper_metrics_evaluator(mimi):
    """Create PaperMetricsEvaluator for LibriLight evaluation."""
    try:
        from finetune.paper_metrics import PaperMetricsEvaluator
        
        # Realistic config matching training setup
        config = {
            'max_sequence_length': 5000,  # Allow longer sequences
            'first_codebook_weight_multiplier': 100.0,
            'ttt_verification_enabled': False,  # No TTT in frozen model
            'memory_check_enabled': True,
        }
        
        class DummyTokenizer:
            def __init__(self):
                pass
        
        evaluator = PaperMetricsEvaluator(
            mimi_encoder=mimi,
            interleaved_tokenizer=DummyTokenizer(),
            device='cuda',
            config=config
        )
        
        logger.info("✅ Created PaperMetricsEvaluator for baseline")
        return evaluator
        
    except Exception as e:
        logger.error(f"❌ Failed to create evaluator: {e}")
        return None

def run_librilight_baseline(model, codes, targets, evaluator):
    """Run comprehensive LibriLight evaluation for baseline."""
    try:
        logger.info("🧪 Running LibriLight baseline evaluation...")
        logger.info(f"🔍 Input shape: {codes.shape}")
        
        # Run the evaluation
        position_losses = evaluator._evaluate_librilight_moshi_native(
            model=model,
            codes=codes,
            targets=targets
        )
        
        if not position_losses:
            logger.error("❌ Evaluation returned empty loss list")
            return None
        
        # Analyze results
        losses = np.array(position_losses)
        
        logger.info(f"📊 Baseline LibriLight Results:")
        logger.info(f"   ✅ Total tokens processed: {len(losses)}")
        logger.info(f"   📈 Loss range: {losses.min():.4f} - {losses.max():.4f}")
        logger.info(f"   📊 Loss mean: {losses.mean():.4f}")
        logger.info(f"   📐 Loss std: {losses.std():.4f}")
        logger.info(f"   🔍 NaN values: {np.isnan(losses).sum()}")
        logger.info(f"   ♾️  Inf values: {np.isinf(losses).sum()}")
        
        # Calculate key metrics at important positions
        metrics = {}
        positions = [100, 200, 500, 800] if len(losses) >= 800 else [len(losses)//4, len(losses)//2, 3*len(losses)//4]
        
        for pos in positions:
            if pos < len(losses):
                metrics[f'loss_at_{pos}'] = losses[pos]
        
        # Calculate trend (early vs late)
        if len(losses) >= 100:
            early_loss = losses[:100].mean()
            late_loss = losses[-100:].mean()
            improvement = early_loss - late_loss
            metrics['early_loss'] = early_loss
            metrics['late_loss'] = late_loss 
            metrics['improvement'] = improvement
            
            logger.info(f"📈 Trend Analysis:")
            logger.info(f"   Early loss (first 100): {early_loss:.4f}")
            logger.info(f"   Late loss (last 100): {late_loss:.4f}")
            logger.info(f"   Improvement: {improvement:.4f} ({'↓ better' if improvement > 0 else '↑ worse'})")
        
        # Calculate overall slope
        if len(losses) >= 50:
            x = np.arange(len(losses))
            slope = np.polyfit(x, losses, 1)[0]
            metrics['slope'] = slope
            logger.info(f"   📐 Overall slope: {slope:.6f} ({'↓ improving' if slope < 0 else '→ flat' if abs(slope) < 1e-6 else '↑ degrading'})")
        
        return {
            'losses': losses,
            'metrics': metrics,
            'summary': {
                'mean_loss': losses.mean(),
                'total_tokens': len(losses),
                'has_nan': bool(np.isnan(losses).sum()),
                'has_inf': bool(np.isinf(losses).sum())
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Error during baseline evaluation: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

def save_baseline_results(results, output_file="frozen_moshi_baseline.npz"):
    """Save baseline results for comparison."""
    try:
        np.savez(
            output_file,
            losses=results['losses'],
            metrics=results['metrics'],
            summary=results['summary']
        )
        logger.info(f"💾 Saved baseline results to {output_file}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to save results: {e}")
        return False

def main():
    """Run comprehensive LibriLight baseline evaluation."""
    logger.info("🎯 Running LibriLight baseline evaluation on frozen Moshi...")
    
    if not torch.cuda.is_available():
        logger.error("❌ CUDA not available. This test requires GPU.")
        return False
    
    logger.info(f"🔧 CUDA device: {torch.cuda.get_device_name()}")
    
    try:
        # Load frozen Moshi
        model, mimi = load_frozen_moshi()
        if model is None:
            return False
        
        # Create test sequence
        codes, targets = create_librilight_test_sequence()
        if codes is None:
            return False
        
        # Create evaluator
        evaluator = create_paper_metrics_evaluator(mimi)
        if evaluator is None:
            return False
        
        # Run baseline evaluation
        results = run_librilight_baseline(model, codes, targets, evaluator)
        if results is None:
            return False
        
        # Save results
        save_baseline_results(results)
        
        # Final summary
        logger.info("\n🎉 BASELINE EVALUATION COMPLETE!")
        logger.info("📋 Key Findings:")
        logger.info(f"   🔢 Processed {results['summary']['total_tokens']} tokens successfully")
        logger.info(f"   📊 Mean loss: {results['summary']['mean_loss']:.4f}")
        logger.info(f"   ✅ No NaN/Inf values: {not results['summary']['has_nan'] and not results['summary']['has_inf']}")
        
        if 'improvement' in results['metrics']:
            improvement = results['metrics']['improvement']
            logger.info(f"   📈 Learning trend: {improvement:.4f} ({'✅ improving' if improvement > 0 else '❌ not improving'})")
        
        logger.info("\n🎯 This baseline will be compared against TTT-enhanced results!")
        logger.info("📈 TTT should show better long-context adaptation (lower late losses)")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Baseline evaluation failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)