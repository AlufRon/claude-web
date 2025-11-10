#!/usr/bin/env python3
"""
Quick test script to verify TTT persistence logging is working.

This script processes just 50 tokens to quickly check if:
1. Weight hashes are logged
2. scan() outputs are logged
3. Inner updates are logged
4. We can see if weights persist or reset

Usage:
    python test_persistence_logging.py

Expected output (if weights reset every token):
    [TTT-PERSIST-CHECK] Layer 29 Token 0: W1_hash=123456, W2_hash=789012
    [TTT-INNER-UPDATE] Layer 29 Position 0: W1_change=0.00420000, W2_change=0.00350000
    [TTT-SCAN-OUTPUT] Layer 29 Token 0: W1_change=0.00420000, W2_change=0.00350000
    [TTT-PERSIST-CHECK] Layer 29 Token 1: W1_hash=123456, W2_hash=789012  ← SAME (reset!)
    [TTT-PERSIST-CHECK] Layer 29 Token 2: W1_hash=123456, W2_hash=789012  ← SAME (reset!)

Expected output (if weights persist):
    [TTT-PERSIST-CHECK] Layer 29 Token 0: W1_hash=123456, W2_hash=789012
    [TTT-SCAN-OUTPUT] Layer 29 Token 0: W1_change=0.00420000, W2_change=0.00350000
    [TTT-PERSIST-CHECK] Layer 29 Token 1: W1_hash=456789, W2_hash=012345  ← DIFFERENT!
    [TTT-PERSIST-CHECK] Layer 29 Token 2: W1_hash=789012, W2_hash=345678  ← DIFFERENT!
"""

import os
import sys
import torch
import logging

# Add moshi-finetune to path
sys.path.insert(0, '/home/alufr/ttt_tests/moshi-finetune')

from finetune.librilight_loader import LibriLightLoader
from finetune.librilight_simple import evaluate_librilight_simple
from moshi.models import loaders, LMGen

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("=" * 80)
    logger.info("TTT PERSISTENCE LOGGING TEST")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Loading model...")

    # Load model
    model_path = "/home/alufr/ttt_tests/moshi-finetune/checkpoints/pretrained"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = loaders.load_lm_model_local(
        model_path,
        device=device
    )

    # Create LMGen
    lm_gen = LMGen(model, temp=0.8, temp_text=0.7, top_p=0.95, top_k=0, check=False)

    # Load MIMI encoder
    mimi = loaders.get_mimi(model_path, device=device)

    logger.info(f"Model loaded on {device}")
    logger.info("")

    # Load LibriLight data (just 50 tokens for quick test)
    loader = LibriLightLoader(
        manifest_path="/sise/eliyanac-group/ron_al/librilight/extracted_medium/medium/0/0_0.flac.json",
        data_root="/sise/eliyanac-group/ron_al/librilight/extracted_medium/medium",
        device=device,
        mimi_encoder=mimi
    )

    logger.info("Loading 50 tokens for quick test...")
    audio_codes, audio_targets = loader.get_batch(max_tokens=50)
    logger.info(f"Loaded {audio_codes.shape[-1]} tokens")
    logger.info("")

    logger.info("=" * 80)
    logger.info("RUNNING EVALUATION (watch for persistence logs)")
    logger.info("=" * 80)
    logger.info("")

    # Run evaluation with persistence logging
    position_losses = evaluate_librilight_simple(
        model=model,
        lm_gen=lm_gen,
        audio_codes=audio_codes,
        audio_targets=audio_targets,
        mimi=mimi,
        max_length=50,
        first_codebook_weight_multiplier=100.0
    )

    logger.info("")
    logger.info("=" * 80)
    logger.info("TEST COMPLETE")
    logger.info("=" * 80)
    logger.info("")
    logger.info("✅ Check the logs above for:")
    logger.info("   1. [TTT-PERSIST-CHECK] - Weight hashes at each token")
    logger.info("   2. [TTT-INNER-UPDATE] - Weight changes within forward pass")
    logger.info("   3. [TTT-SCAN-OUTPUT] - scan() return values")
    logger.info("")
    logger.info("🔍 DIAGNOSIS:")
    logger.info("   - If W1_hash/W2_hash NEVER change → weights reset every token (BUG!)")
    logger.info("   - If W1_hash/W2_hash DO change → weights persist (WORKING!)")
    logger.info("")
    logger.info(f"Average loss: {sum(position_losses)/len(position_losses):.4f}")

if __name__ == "__main__":
    main()
