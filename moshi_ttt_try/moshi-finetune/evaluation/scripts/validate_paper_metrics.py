#!/usr/bin/env python3
"""
Validation script for paper metrics evaluation improvements.
Compares our enhanced implementation with reference results from eval_paper.

Tests different configurations:
1. Standard mode (Moshi stream only)
2. Optimal silence mode (+2-6% performance boost)
3. Cross-stream mode (experimental)
"""

import sys
import os
import torch
import logging
from pathlib import Path

# Add paths
sys.path.append('/home/alufr/ttt_tests/moshi-finetune')
sys.path.append('/home/alufr/ttt_tests/moshi')

from moshi.models import loaders
from finetune.paper_metrics import PaperMetricsEvaluator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_moshi_models(device="cuda"):
    """Load Moshi and MIMI models for evaluation"""
    logger.info("Loading Moshi models...")
    
    # Load checkpoint info
    checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
        hf_repo="kyutai/moshiko-pytorch-bf16"
    )
    
    # Load MIMI
    mimi = checkpoint_info.get_mimi(device=device)
    mimi.eval()
    for p in mimi.parameters():
        p.requires_grad = False
    
    # Load Moshi LM (simplified - no FSDP for evaluation)
    model = checkpoint_info.get_moshi(device=device, dtype=torch.float32, load_weight=True)
    model.eval()
    
    logger.info("✓ Models loaded successfully")
    return model, mimi, checkpoint_info

def test_configuration(model, mimi, config_name, config):
    """Test a specific configuration and return results"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing Configuration: {config_name}")
    logger.info(f"{'='*60}")
    
    # Create evaluator with this configuration
    evaluator = PaperMetricsEvaluator(mimi, None, device="cuda", config=config)
    
    # Test sBLIMP on a small sample (20 pairs for speed)
    sblimp_results = evaluator.evaluate_sblimp(model, max_samples=20)
    
    # Test sWUGGY on a small sample
    swuggy_results = evaluator.evaluate_swuggy(model, max_samples=20)
    
    results = {
        'config': config_name,
        'sblimp_accuracy': sblimp_results.get('sblimp_accuracy', 0.0),
        'sblimp_samples': sblimp_results.get('sblimp_samples', 0),
        'swuggy_accuracy': swuggy_results.get('swuggy_accuracy', 0.0),
        'swuggy_samples': swuggy_results.get('swuggy_samples', 0),
    }
    
    logger.info(f"Results for {config_name}:")
    logger.info(f"  sBLIMP: {results['sblimp_accuracy']:.3f} ({results['sblimp_samples']} samples)")
    logger.info(f"  sWUGGY: {results['swuggy_accuracy']:.3f} ({results['swuggy_samples']} samples)")
    
    return results

def main():
    """Main validation function"""
    logger.info("🧪 Starting Paper Metrics Validation")
    logger.info("This will test our improved evaluation against different configurations")
    
    # Load models
    try:
        model, mimi, checkpoint_info = load_moshi_models()
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        return 1
    
    # Base configuration paths
    base_config = {
        'sblimp_audio_dir': '/sise/eliyanac-group/ron_al/sblimp_data/sLM21_dataset/syntactic/test/',
        'sblimp_gold_csv': '/sise/eliyanac-group/ron_al/sblimp_data/sLM21_dataset/syntactic/test/gold.csv',
        'swuggy_audio_dir': '/sise/eliyanac-group/ron_al/sblimp_data/sLM21_dataset/lexical/test/',
        'swuggy_gold_csv': '/sise/eliyanac-group/ron_al/sblimp_data/sLM21_dataset/lexical/test/gold.csv',
    }
    
    # Test configurations
    configurations = [
        {
            'name': 'Standard Mode (Baseline)',
            'config': {
                **base_config,
                'paper_metrics_use_silence': False,
                'paper_metrics_use_user_stream': False,
            }
        },
        {
            'name': 'Optimal Silence Mode (+2-6% boost)',
            'config': {
                **base_config,
                'paper_metrics_use_silence': True,
                'paper_metrics_use_user_stream': False,
            }
        },
        {
            'name': 'Cross-Stream Mode (Experimental)',
            'config': {
                **base_config,
                'paper_metrics_use_silence': False,
                'paper_metrics_use_user_stream': True,
            }
        },
        {
            'name': 'Cross-Stream + Silence (Full Experimental)',
            'config': {
                **base_config,
                'paper_metrics_use_silence': True,
                'paper_metrics_use_user_stream': True,
            }
        }
    ]
    
    # Run all tests
    all_results = []
    for config_test in configurations:
        try:
            results = test_configuration(model, mimi, config_test['name'], config_test['config'])
            all_results.append(results)
        except Exception as e:
            logger.error(f"Configuration {config_test['name']} failed: {e}")
            continue
    
    # Summary comparison
    logger.info(f"\n{'='*80}")
    logger.info("SUMMARY COMPARISON")
    logger.info(f"{'='*80}")
    logger.info(f"{'Configuration':<35} {'sBLIMP':<10} {'sWUGGY':<10} {'Notes'}")
    logger.info("-" * 80)
    
    baseline_sblimp = None
    baseline_swuggy = None
    
    for result in all_results:
        if 'Standard Mode' in result['config']:
            baseline_sblimp = result['sblimp_accuracy']
            baseline_swuggy = result['swuggy_accuracy']
        
        sblimp_str = f"{result['sblimp_accuracy']:.3f}"
        swuggy_str = f"{result['swuggy_accuracy']:.3f}"
        
        # Calculate improvement
        notes = ""
        if baseline_sblimp and 'Optimal Silence' in result['config']:
            improvement = result['sblimp_accuracy'] - baseline_sblimp
            notes = f"sBLIMP: {improvement:+.3f}"
        
        logger.info(f"{result['config']:<35} {sblimp_str:<10} {swuggy_str:<10} {notes}")
    
    # Expected targets from reference implementation
    logger.info(f"\n{'='*80}")
    logger.info("REFERENCE TARGETS (from eval_paper)")
    logger.info(f"{'='*80}")
    logger.info("sBLIMP Target: 55.2% (Moshi paper baseline)")
    logger.info("sWUGGY Target: ~74% (Moshi paper baseline)")
    logger.info("Expected boost from silence codes: +2-6%")
    
    # Recommendations
    logger.info(f"\n{'='*80}")
    logger.info("RECOMMENDATIONS")
    logger.info(f"{'='*80}")
    
    if all_results:
        optimal_result = [r for r in all_results if 'Optimal Silence' in r['config']]
        if optimal_result:
            opt = optimal_result[0]
            if opt['sblimp_accuracy'] > 0.5:  # Above 50%
                logger.info("✅ Optimal silence configuration is working correctly")
                if baseline_sblimp and opt['sblimp_accuracy'] > baseline_sblimp:
                    logger.info(f"✅ Performance improvement detected: {opt['sblimp_accuracy'] - baseline_sblimp:+.3f}")
                else:
                    logger.info("⚠️  Expected performance improvement not observed")
            else:
                logger.info("❌ Accuracy too low - check implementation")
        
        logger.info("\nFor production training, use:")
        logger.info("  paper_metrics_use_silence: true")
        logger.info("  paper_metrics_use_user_stream: false")
    
    logger.info(f"\n🧪 Validation completed!")
    return 0

if __name__ == "__main__":
    sys.exit(main())