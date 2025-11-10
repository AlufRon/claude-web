#!/usr/bin/env python3
"""
Simpler debug script - just enable TORCH_SHOW_CPP_STACKTRACES and run.
"""

import torch
import sys
import os
import logging
from pathlib import Path

# Enable detailed PyTorch error tracing
os.environ['TORCH_SHOW_CPP_STACKTRACES'] = '1'

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add necessary paths
sys.path.insert(0, str(Path(__file__).parent.parent / "moshi"))
sys.path.insert(0, str(Path(__file__).parent))

def main():
    import yaml
    from run_paper_metrics_on_checkpoint import load_ttt_model
    from finetune.paper_metrics import PaperMetricsEvaluator
    
    logger.info("="*80)
    logger.info("SIMPLE DTYPE MISMATCH DEBUGGER")
    logger.info("="*80)
    
    # Load config
    config_path = Path("example/moshi_7B_multilayer_with_ttt.yaml")
    logger.info(f"\n📄 Loading config from {config_path}")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Checkpoint path
    checkpoint_path = Path("/sise/eliyanac-group/ron_al/dailytalk_finetune_from_librilight6/checkpoints/checkpoint_002500/consolidated")
    
    logger.info(f"\n🏗️  Loading model from {checkpoint_path}")
    model, checkpoint_info = load_ttt_model(str(checkpoint_path), device="cuda")
    
    # Load MIMI encoder properly
    logger.info("\n🎤 Loading MIMI encoder...")
    mimi = checkpoint_info.get_mimi(device="cuda")
    logger.info(f"✅ MIMI loaded: {type(mimi).__name__}")
    
    # Create evaluator
    evaluator = PaperMetricsEvaluator(
        mimi_encoder=mimi,
        interleaved_tokenizer=None,
        device="cuda",
        config=config.get('paper_metrics', {})
    )
    
    # Try to load and process ONE audio file
    test_audio = "/sise/eliyanac-group/ron_al/sblimp_data/sLM21_dataset/syntactic/test/aAAAZvtMsGyf.wav"
    
    if Path(test_audio).exists():
        logger.info(f"\n🎵 Testing with audio file: {test_audio}")
        
        # Encode audio
        logger.info("\n1️⃣  Encoding audio...")
        codes = evaluator._encode_audio(test_audio)
        logger.info(f"✅ Encoded codes: shape={codes.shape}, dtype={codes.dtype}")
        
        # Try to compute likelihood - this should fail with detailed traceback
        logger.info("\n2️⃣  Computing likelihood (this should fail with detailed error)...")
        
        # Don't catch the exception - let it propagate with full traceback
        likelihood = evaluator._compute_likelihood(model, codes)
        logger.info(f"✅ Likelihood computed: {likelihood}")
    else:
        logger.error(f"Test audio file not found: {test_audio}")
        return 1
    
    logger.info("\n" + "="*80)
    logger.info("DEBUG COMPLETE")
    logger.info("="*80)
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        logger.error(f"\n{'='*80}")
        logger.error(f"EXCEPTION CAUGHT: {e}")
        logger.error(f"{'='*80}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
