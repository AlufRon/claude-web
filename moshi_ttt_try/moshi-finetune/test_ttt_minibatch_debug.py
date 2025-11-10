#!/usr/bin/env python3
"""
Quick test to understand TTT mini-batch behavior during inference.
"""

import sys
import torch
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "moshi"))

from inference.run_inference_with_ttt import load_ttt_model, load_checkpoint_config

def main():
    logger.info("=" * 80)
    logger.info("TTT Mini-Batch Debug Test")
    logger.info("=" * 80)

    # Use the checkpoint from the logs
    checkpoint_dir = Path("/sise/eliyanac-group/ron_al/dailytalk_finetune_from_librilight20lorattt/checkpoints/checkpoint_002000/consolidated")
    hf_repo = "kyutai/moshiko-pytorch-bf16"
    device = "cuda"

    logger.info(f"Loading model from {checkpoint_dir}")

    # Load model
    model = load_ttt_model(
        checkpoint_dir=checkpoint_dir,
        hf_repo=hf_repo,
        device=device
    )

    if model is None:
        logger.error("Failed to load model")
        return 1

    logger.info("Model loaded successfully!")
    logger.info("=" * 80)
    logger.info("Running test forward pass to trigger logging...")
    logger.info("=" * 80)

    # Test with a typical inference sequence length
    model.eval()
    with torch.no_grad():
        # Moshi uses 17 codebooks total (dep_q structure)
        num_codebooks = model.num_codebooks
        batch_size = 1
        seq_len = 64  # This is what we saw in the logs

        # Create dummy input
        dummy_codes = torch.randint(0, 2048, (batch_size, num_codebooks, seq_len), device=device)

        logger.info(f"Input shape: {dummy_codes.shape} (batch={batch_size}, codebooks={num_codebooks}, seq_len={seq_len})")
        logger.info("Running forward pass...")

        output = model(dummy_codes)

        logger.info(f"Output logits shape: {output.logits.shape}")
        logger.info(f"Output text_logits shape: {output.text_logits.shape}")

    logger.info("=" * 80)
    logger.info("Test complete! Check the [TTT-*] logs above")
    logger.info("=" * 80)

    return 0

if __name__ == "__main__":
    sys.exit(main())
