#!/usr/bin/env python3
"""
Quick test to verify our depformer fix works in isolation.
"""

import sys
import torch
from pathlib import Path

# Add current directory to Python path
sys.path.append(str(Path(__file__).parent))

# Import the logging setup
from finetune.monitoring.utils import set_logger
import logging

# Set up DEBUG logging
set_logger(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_depformer_fix():
    """Test our depformer fix with minimal Moshi setup"""
    try:
        # Load minimal Moshi models
        from moshi.models import loaders
        
        logger.info("🧪 Loading Moshi model...")
        model = loaders.get_moshi_lm(hf_repo_id='kyutai/moshiko-pytorch-bf16')
        model = model.cuda().eval()
        
        logger.info(f"🧪 Model loaded. dep_q={model.dep_q}")
        logger.info(f"🧪 Model has depformer: {model.depformer is not None}")
        
        # Test our depformer function
        from finetune.paper_metrics import PaperMetricsEvaluator
        evaluator = PaperMetricsEvaluator(None, None, device="cuda", config={})
        
        # Create test inputs similar to what the evaluation uses
        B = 1
        text_token = torch.zeros(B, dtype=torch.long, device=model.device)
        transformer_out = torch.randn(B, 1, model.dim, dtype=model.dtype, device=model.device)
        
        logger.info(f"🧪 Test inputs: text_token.shape={text_token.shape}, transformer_out.shape={transformer_out.shape}")
        logger.info(f"🧪 Test dtypes: text_token.dtype={text_token.dtype}, transformer_out.dtype={transformer_out.dtype}")
        
        # Test our fixed depformer function
        logger.info("🧪 Testing depformer function...")
        audio_logits = evaluator._get_audio_logits_from_depformer(model, text_token, transformer_out)
        
        if audio_logits is not None:
            logger.info(f"✅ SUCCESS! audio_logits.shape={audio_logits.shape}")
            logger.info(f"✅ Expected shape: [B={B}, dep_q={model.dep_q}, vocab_size]")
            return True
        else:
            logger.error("❌ FAILED: audio_logits is None")
            return False
            
    except Exception as e:
        logger.error(f"❌ FAILED with exception: {e}")
        import traceback
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    logger.info("🧪 Testing depformer fix...")
    
    if torch.cuda.is_available():
        success = test_depformer_fix()
        if success:
            logger.info("🎉 Depformer fix works!")
        else:
            logger.error("💥 Depformer fix failed!")
    else:
        logger.warning("⚠️ CUDA not available, skipping test")