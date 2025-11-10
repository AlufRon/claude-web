#!/usr/bin/env python3
"""
Run Baseline Moshi Inference with Cache Logging

This script runs vanilla Moshi (no TTT) with detailed cache behavior logging
to diagnose the token 6100 activation jump issue.

Usage:
    python run_baseline_with_cache_logging.py \\
        --input /path/to/audio.wav \\
        --output output.wav
"""

import argparse
import sys
from pathlib import Path

# Add paths
moshi_path = Path('/home/alufr/ttt_tests/moshi/moshi')
sys.path.insert(0, str(moshi_path))

# CRITICAL: Apply logging patch BEFORE importing Moshi modules
print("🔧 Applying cache logging patch...")
from cache_logging_patch import instrument_cache_logging
instrument_cache_logging()

# Now import Moshi
print("📥 Importing Moshi modules...")
import torch
import sphn
from moshi.models import loaders
from moshi.client_utils import log
from moshi.run_inference import InferenceState


def main():
    parser = argparse.ArgumentParser(
        description="Run baseline Moshi inference with cache logging"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input audio file path"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output_baseline_logged.wav",
        help="Output audio file path (default: output_baseline_logged.wav)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (default: cuda)"
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        default=loaders.DEFAULT_REPO,
        help=f"HuggingFace repo (default: {loaders.DEFAULT_REPO})"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size (default: 1)"
    )
    parser.add_argument(
        "--cfg-coef",
        type=float,
        default=1.0,
        help="CFG coefficient (default: 1.0)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model dtype (default: bfloat16)"
    )

    args = parser.parse_args()

    # Parse dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    print(f"\n{'='*80}")
    print("🎤 BASELINE MOSHI INFERENCE WITH CACHE LOGGING")
    print(f"{'='*80}")
    print(f"  Input audio: {args.input}")
    print(f"  Output audio: {args.output}")
    print(f"  Device: {args.device}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Dtype: {args.dtype}")
    print(f"{'='*80}\n")

    # Check input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Error: Input file not found: {args.input}")
        return 1

    # Load checkpoint
    log("info", f"retrieving checkpoint from {args.hf_repo}")
    checkpoint_info = loaders.CheckpointInfo.from_hf_repo(args.hf_repo)

    # Load Mimi
    log("info", "loading mimi encoder/decoder")
    mimi = checkpoint_info.get_mimi(device=args.device)
    log("info", "✅ mimi loaded")

    # Load LM
    log("info", "loading moshi language model")
    lm = checkpoint_info.get_moshi(device=args.device, dtype=dtype)
    log("info", "✅ moshi loaded")

    # Fix RoPE max_period for long-context generation (6000+ tokens)
    print("🔧 Applying RoPE fix for long-context generation...")
    if hasattr(lm, 'transformer') and hasattr(lm.transformer, 'rope') and lm.transformer.rope is not None:
        lm.transformer.rope.max_period = 100_000
        lm.transformer.max_period = 100_000
        print(f"✅ Set main transformer RoPE max_period: 10,000 → 100,000")
    if hasattr(lm, 'depformer') and hasattr(lm.depformer, 'transformer'):
        if hasattr(lm.depformer.transformer, 'rope') and lm.depformer.transformer.rope is not None:
            lm.depformer.transformer.rope.max_period = 100_000
            lm.depformer.transformer.max_period = 100_000
            print(f"✅ Set depformer RoPE max_period: 10,000 → 100,000")

    # Increase KV cache context size to avoid wraparound at 3000 tokens
    print("🔧 Increasing KV cache context size...")
    new_context = 8000
    if hasattr(lm, 'transformer') and hasattr(lm.transformer, 'layers'):
        old_context = None
        for layer in lm.transformer.layers:
            if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'context'):
                if old_context is None:
                    old_context = layer.self_attn.context
                layer.self_attn.context = new_context
        if old_context is not None:
            print(f"✅ Set transformer context: {old_context} → {new_context}")

    # Load tokenizer
    text_tokenizer = checkpoint_info.get_text_tokenizer()
    log("info", "✅ text tokenizer loaded")

    print(f"\n{'='*80}")
    print("📖 MODEL CONFIGURATION")
    print(f"{'='*80}")
    print(f"  Model: {checkpoint_info.model_type}")
    print(f"  LM dim: {lm.dim}")
    try:
        print(f"  LM layers: {len(lm.transformer.layers)}")
        # Try to find context
        first_layer = lm.transformer.layers[0]
        if hasattr(first_layer, 'seq_modeling_block'):
            if hasattr(first_layer.seq_modeling_block, 'context'):
                print(f"  Context (cache capacity): {first_layer.seq_modeling_block.context}")
            else:
                print(f"  Context (cache capacity): Will be logged when cache is created")
        else:
            print(f"  Context (cache capacity): Will be logged when cache is created")
    except Exception as e:
        print(f"  LM layers: Error getting info - {e}")
    print(f"{'='*80}\n")

    # Load input audio
    log("info", f"loading audio from {args.input}")
    in_pcms, _ = sphn.read(args.input, sample_rate=mimi.sample_rate)
    in_pcms = torch.from_numpy(in_pcms).to(device=args.device)
    in_pcms = in_pcms[None, 0:1].expand(args.batch_size, -1, -1)

    duration_sec = in_pcms.shape[-1] / mimi.sample_rate
    expected_tokens = int(duration_sec * 12.5 * 8)  # 12.5 Hz * 8 codebooks

    print(f"\n{'='*80}")
    print("🎵 AUDIO INFO")
    print(f"{'='*80}")
    print(f"  Duration: {duration_sec:.1f} seconds")
    print(f"  Sample rate: {mimi.sample_rate} Hz")
    print(f"  Input shape: {in_pcms.shape}")
    print(f"  Expected tokens: ~{expected_tokens}")
    print(f"  Expected wraparounds (capacity=3000): {expected_tokens // 3000}")
    print(f"{'='*80}\n")

    # Create inference state
    log("info", "initializing inference state")
    state = InferenceState(
        checkpoint_info,
        mimi,
        text_tokenizer,
        lm,
        args.batch_size,
        args.cfg_coef,
        args.device,
        **checkpoint_info.lm_gen_config,
    )

    print("\n🚀 Starting inference with cache logging...\n")

    # Run inference
    try:
        with torch.no_grad():
            out_items = state.run(in_pcms)
    except KeyboardInterrupt:
        print("\n⚠️  Inference interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Save output
    if args.output and out_items:
        outfile = Path(args.output)
        for index, (_, out_pcm) in enumerate(out_items):
            if len(out_items) > 1:
                outfile_ = outfile.with_name(f"{outfile.stem}-{index}{outfile.suffix}")
            else:
                outfile_ = outfile

            duration = out_pcm.shape[1] / mimi.sample_rate
            log("info", f"writing {outfile_} with duration {duration:.1f} sec.")
            sphn.write_wav(
                str(outfile_), out_pcm[0].numpy(), sample_rate=mimi.sample_rate
            )
            log("info", "✅ output saved")

    print(f"\n{'='*80}")
    print("✅ INFERENCE COMPLETE")
    print(f"{'='*80}")
    if args.output:
        print(f"  Output saved to: {args.output}")
    print(f"{'='*80}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
