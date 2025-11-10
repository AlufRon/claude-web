#!/usr/bin/env python3
"""
Test script for batch inference implementation.

This script validates that non-streaming batch inference works correctly
before using it for TTT evaluation.

Usage:
    python tests/test_batch_inference.py --checkpoint /path/to/checkpoint
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import numpy as np

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.batch_inference import load_batch_inference

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_basic_forward(batch_inf, device='cuda'):
    """Test 1: Basic forward pass works."""
    logger.info("\n" + "="*80)
    logger.info("TEST 1: Basic Forward Pass")
    logger.info("="*80)

    try:
        # Create dummy input
        B, T = 2, 256  # 2 sequences, 256 tokens (~20s audio)
        dep_q = batch_inf.dep_q

        # Create random codes
        codes = torch.randint(0, batch_inf.card, (B, 17, T), device=device)

        logger.info(f"Input shape: {codes.shape}")

        # Forward pass
        output = batch_inf.forward(codes)

        logger.info(f"Output logits shape: {output['logits'].shape}")
        logger.info(f"Output text_logits shape: {output['text_logits'].shape}")

        # Check output shapes
        assert output['logits'].shape == (B, dep_q, T, batch_inf.card), \
            f"Wrong logits shape: {output['logits'].shape}"
        assert output['text_logits'].shape[:-1] == (B, 1, T), \
            f"Wrong text_logits shape: {output['text_logits'].shape}"
        assert output['mask'].shape == (B, dep_q, T), \
            f"Wrong mask shape: {output['mask'].shape}"
        assert output['text_mask'].shape == (B, 1, T), \
            f"Wrong text_mask shape: {output['text_mask'].shape}"

        logger.info("✅ TEST 1 PASSED: Forward pass successful")
        return True

    except Exception as e:
        logger.error(f"❌ TEST 1 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_audio_encoding_decoding(batch_inf, device='cuda'):
    """Test 2: Audio encoding and decoding works."""
    logger.info("\n" + "="*80)
    logger.info("TEST 2: Audio Encoding/Decoding")
    logger.info("="*80)

    try:
        # Create dummy audio (2 seconds at 24kHz)
        B = 2
        samples = 2 * 24000
        audio = torch.randn(B, 1, samples, device=device)

        logger.info(f"Input audio shape: {audio.shape}")

        # Encode
        codes = batch_inf.encode_audio(audio)
        logger.info(f"Encoded codes shape: {codes.shape}")

        # Check shape
        assert codes.shape[0] == B, "Wrong batch size"
        assert codes.shape[1] == batch_inf.dep_q, "Wrong number of codebooks"

        # Decode
        audio_reconstructed = batch_inf.decode_audio(codes)
        logger.info(f"Decoded audio shape: {audio_reconstructed.shape}")

        # Check reconstruction shape (might be slightly different due to padding)
        assert audio_reconstructed.shape[0] == B, "Wrong batch size"
        assert audio_reconstructed.shape[1] == 1, "Wrong number of channels"

        logger.info("✅ TEST 2 PASSED: Encoding/decoding successful")
        return True

    except Exception as e:
        logger.error(f"❌ TEST 2 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_forward_audio(batch_inf, device='cuda'):
    """Test 3: End-to-end forward_audio works."""
    logger.info("\n" + "="*80)
    logger.info("TEST 3: End-to-End forward_audio")
    logger.info("="*80)

    try:
        # Create dummy audio
        B = 2
        samples = 2 * 24000  # 2 seconds
        audio = torch.randn(B, 1, samples, device=device)

        logger.info(f"Input audio shape: {audio.shape}")

        # Forward audio
        output = batch_inf.forward_audio(audio, include_text=True)

        logger.info(f"Output logits shape: {output['logits'].shape}")
        logger.info(f"Output text_logits shape: {output['text_logits'].shape}")

        # Check shapes
        B_out, dep_q, T, card = output['logits'].shape
        assert B_out == B, "Wrong batch size"
        assert dep_q == batch_inf.dep_q, "Wrong number of codebooks"

        logger.info("✅ TEST 3 PASSED: End-to-end forward_audio successful")
        return True

    except Exception as e:
        logger.error(f"❌ TEST 3 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_loss_computation(batch_inf, device='cuda'):
    """Test 4: Loss computation works."""
    logger.info("\n" + "="*80)
    logger.info("TEST 4: Loss Computation")
    logger.info("="*80)

    try:
        # Create dummy data
        B, dep_q, T = 2, batch_inf.dep_q, 128
        card = batch_inf.card

        logits = torch.randn(B, dep_q, T, card, device=device)
        targets = torch.randint(0, card, (B, dep_q, T), device=device)
        mask = torch.ones(B, dep_q, T, device=device)

        logger.info(f"Logits shape: {logits.shape}")
        logger.info(f"Targets shape: {targets.shape}")

        # Compute loss
        loss = batch_inf.compute_loss(logits, targets, mask=mask, reduction='mean')
        logger.info(f"Loss (mean): {loss.item():.4f}")

        # Check loss is valid
        assert not torch.isnan(loss), "Loss is NaN"
        assert not torch.isinf(loss), "Loss is Inf"
        assert loss.item() > 0, "Loss is not positive"

        # Test with codebook weights
        codebook_weights = torch.tensor([3.0] + [1.0] * (dep_q - 1), device=device)
        loss_weighted = batch_inf.compute_loss(
            logits, targets, mask=mask,
            reduction='mean',
            codebook_weights=codebook_weights
        )
        logger.info(f"Loss (weighted): {loss_weighted.item():.4f}")

        assert not torch.isnan(loss_weighted), "Weighted loss is NaN"

        logger.info("✅ TEST 4 PASSED: Loss computation successful")
        return True

    except Exception as e:
        logger.error(f"❌ TEST 4 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_perplexity_computation(batch_inf, device='cuda'):
    """Test 5: Perplexity computation works."""
    logger.info("\n" + "="*80)
    logger.info("TEST 5: Perplexity Computation")
    logger.info("="*80)

    try:
        # Create dummy audio
        B = 2
        samples = 2 * 24000  # 2 seconds
        audio = torch.randn(B, 1, samples, device=device)

        logger.info(f"Input audio shape: {audio.shape}")

        # Compute perplexity
        results = batch_inf.evaluate_perplexity(audio)

        logger.info(f"Perplexity: {results['perplexity']:.4f}")
        logger.info(f"Loss: {results['loss']:.4f}")
        logger.info(f"Per-codebook perplexity: {[f'{p:.4f}' for p in results['per_codebook_perplexity']]}")

        # Check results are valid
        assert not np.isnan(results['perplexity']), "Perplexity is NaN"
        assert not np.isinf(results['perplexity']), "Perplexity is Inf"
        assert results['perplexity'] > 0, "Perplexity is not positive"
        assert len(results['per_codebook_perplexity']) == batch_inf.dep_q, "Wrong number of per-codebook perplexities"

        logger.info("✅ TEST 5 PASSED: Perplexity computation successful")
        return True

    except Exception as e:
        logger.error(f"❌ TEST 5 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_ttt_configuration(batch_inf, device='cuda'):
    """Test 6: TTT layers are properly configured."""
    logger.info("\n" + "="*80)
    logger.info("TEST 6: TTT Configuration")
    logger.info("="*80)

    try:
        # Check if model has TTT layers
        ttt_layers_found = 0

        if hasattr(batch_inf.model, 'transformer') and hasattr(batch_inf.model.transformer, 'layers'):
            for i, layer in enumerate(batch_inf.model.transformer.layers):
                if hasattr(layer, 'seq_modeling_block'):
                    seq_block = layer.seq_modeling_block
                    if hasattr(seq_block, 'ttt_layer'):
                        ttt_layers_found += 1
                        ttt_layer = seq_block.ttt_layer

                        # Check mini_batch_size
                        if hasattr(ttt_layer, 'mini_batch_size'):
                            logger.info(f"Layer {i}: TTT mini_batch_size = {ttt_layer.mini_batch_size}")
                        else:
                            logger.warning(f"Layer {i}: TTT layer has no mini_batch_size attribute")

        if ttt_layers_found > 0:
            logger.info(f"✅ Found {ttt_layers_found} TTT layers")
            logger.info("✅ TEST 6 PASSED: TTT layers configured")
        else:
            logger.info("ℹ️  No TTT layers found (baseline Moshi)")
            logger.info("✅ TEST 6 PASSED (no TTT layers to configure)")

        return True

    except Exception as e:
        logger.error(f"❌ TEST 6 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_non_streaming_state(batch_inf, device='cuda'):
    """Test 7: Model is not in streaming mode."""
    logger.info("\n" + "="*80)
    logger.info("TEST 7: Non-Streaming State Verification")
    logger.info("="*80)

    try:
        # Check transformer streaming state
        if hasattr(batch_inf.model, 'transformer'):
            if hasattr(batch_inf.model.transformer, '_streaming_state'):
                state = batch_inf.model.transformer._streaming_state
                assert state is None, f"Transformer streaming state is not None: {state}"
                logger.info("✅ Transformer is NOT in streaming mode")

        # Check depformer streaming state
        if hasattr(batch_inf.model, 'depformer') and batch_inf.model.depformer is not None:
            if hasattr(batch_inf.model.depformer, '_streaming_state'):
                state = batch_inf.model.depformer._streaming_state
                assert state is None, f"Depformer streaming state is not None: {state}"
                logger.info("✅ Depformer is NOT in streaming mode")

        logger.info("✅ TEST 7 PASSED: Model is in non-streaming mode")
        return True

    except Exception as e:
        logger.error(f"❌ TEST 7 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_deterministic_output(batch_inf, device='cuda'):
    """Test 8: Output is deterministic."""
    logger.info("\n" + "="*80)
    logger.info("TEST 8: Deterministic Output")
    logger.info("="*80)

    try:
        # Create dummy input
        B, T = 1, 128
        codes = torch.randint(0, batch_inf.card, (B, 17, T), device=device)

        # Run twice
        output1 = batch_inf.forward(codes)
        output2 = batch_inf.forward(codes)

        # Compare
        diff = (output1['logits'] - output2['logits']).abs().max().item()
        logger.info(f"Max difference between two runs: {diff:.10f}")

        # Should be exactly the same (or very close due to floating point)
        assert diff < 1e-5, f"Output is not deterministic: diff={diff}"

        logger.info("✅ TEST 8 PASSED: Output is deterministic")
        return True

    except Exception as e:
        logger.error(f"❌ TEST 8 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def parse_args():
    parser = argparse.ArgumentParser(description="Test batch inference implementation")
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to consolidated checkpoint directory'
    )
    parser.add_argument(
        '--hf-repo',
        type=str,
        default='kyutai/moshiko-pytorch-bf16',
        help='HuggingFace repository ID'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to run tests on'
    )
    parser.add_argument(
        '--ttt-mini-batch-size',
        type=int,
        default=16,
        help='TTT mini_batch_size to test'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("="*80)
    logger.info("BATCH INFERENCE TESTS")
    logger.info("="*80)

    # Load batch inference
    logger.info(f"\nLoading model from {args.checkpoint}...")
    batch_inf = load_batch_inference(
        checkpoint_dir=args.checkpoint,
        hf_repo=args.hf_repo,
        device=args.device,
        ttt_mini_batch_size=args.ttt_mini_batch_size,
    )

    # Run tests
    tests = [
        ("Basic Forward Pass", test_basic_forward),
        ("Audio Encoding/Decoding", test_audio_encoding_decoding),
        ("End-to-End forward_audio", test_forward_audio),
        ("Loss Computation", test_loss_computation),
        ("Perplexity Computation", test_perplexity_computation),
        ("TTT Configuration", test_ttt_configuration),
        ("Non-Streaming State", test_non_streaming_state),
        ("Deterministic Output", test_deterministic_output),
    ]

    passed = 0
    failed = 0

    for test_name, test_fn in tests:
        try:
            if test_fn(batch_inf, args.device):
                passed += 1
            else:
                failed += 1
        except Exception as e:
            failed += 1
            logger.error(f"❌ {test_name} FAILED with exception: {str(e)}")
            import traceback
            traceback.print_exc()

    # Summary
    logger.info("\n" + "="*80)
    logger.info("TEST SUMMARY")
    logger.info("="*80)
    logger.info(f"Passed: {passed}/{len(tests)}")
    logger.info(f"Failed: {failed}/{len(tests)}")

    if failed == 0:
        logger.info("✅ ALL TESTS PASSED!")
        return 0
    else:
        logger.error(f"❌ {failed} TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
