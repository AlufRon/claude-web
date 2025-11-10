#!/usr/bin/env python3
"""
Ultra-minimal test to find dtype mismatch - no hooks, just run and fail.
"""

import torch
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent / "moshi"))
sys.path.insert(0, str(Path(__file__).parent))

import yaml
from run_paper_metrics_on_checkpoint import load_ttt_model
from finetune.paper_metrics import PaperMetricsEvaluator

# Load config
with open("example/moshi_7B_multilayer_with_ttt.yaml") as f:
    config = yaml.safe_load(f)

# Load model
checkpoint_path = "/sise/eliyanac-group/ron_al/dailytalk_finetune_from_librilight6/checkpoints/checkpoint_002500/consolidated"
logger.info(f"Loading model from {checkpoint_path}...")
model, checkpoint_info = load_ttt_model(checkpoint_path, device="cuda")

# Load MIMI
logger.info("Loading MIMI...")
mimi = checkpoint_info.get_mimi(device="cuda")

# Create evaluator
evaluator = PaperMetricsEvaluator(
    mimi_encoder=mimi,
    interleaved_tokenizer=None,
    device="cuda",
    config=config.get('paper_metrics', {})
)

# Test audio
test_audio = "/sise/eliyanac-group/ron_al/sblimp_data/sLM21_dataset/syntactic/test/aAAAZvtMsGyf.wav"
logger.info(f"Encoding audio: {test_audio}")
codes = evaluator._encode_audio(test_audio)
logger.info(f"Encoded: shape={codes.shape}, dtype={codes.dtype}")

# This will fail with dtype error - let it crash with full traceback
logger.info("Computing likelihood (will fail)...")
logger.info("Directly calling model forward to bypass error handling...")

# Manually do what _compute_likelihood does, but without error handling
with torch.no_grad():
    codes = codes.to("cuda")
    B, K, T = codes.shape
    
    # Create input tensor
    input_codes = torch.full(
        (B, model.num_codebooks, T),
        model.zero_token_id,
        device=codes.device,
        dtype=codes.dtype
    )
    
    # Place audio codes
    audio_offset = model.audio_offset
    input_codes[:, audio_offset:audio_offset + K, :] = codes
    
    logger.info(f"Input codes: shape={input_codes.shape}, dtype={input_codes.dtype}")
    
    # Forward pass - THIS is where it should fail
    logger.info("Calling model forward...")
    output = model(codes=input_codes)
    
    logger.info(f"Output: {output}")

sys.exit(0)
