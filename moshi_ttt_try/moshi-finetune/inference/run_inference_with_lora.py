#!/usr/bin/env python3
"""
Simple wrapper to run Moshi inference with LoRA-finetuned checkpoints.
This loads the base model from HuggingFace and applies LoRA adapters.
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import safetensors.torch
import sphn

sys.path.insert(0, str(Path(__file__).parent))

from moshi.models.loaders import CheckpointInfo
from moshi.models import LMGen
from moshi.client_utils import log

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Run Moshi inference with LoRA-finetuned checkpoint"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to consolidated checkpoint directory (contains lora.safetensors)"
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        default="kyutai/moshiko-pytorch-bf16",
        help="HuggingFace repository for base Moshi model"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on (default: cuda)"
    )
    parser.add_argument(
        "infile",
        type=str,
        help="Input audio file (WAV/FLAC format)"
    )
    parser.add_argument(
        "outfile",
        type=str,
        help="Output audio file (WAV format)"
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("Moshi Inference with LoRA")
    logger.info("=" * 80)

    checkpoint_dir = Path(args.checkpoint)
    
    # Load base model from HuggingFace
    log("info", "loading moshi components from HuggingFace")
    checkpoint_info = CheckpointInfo.from_hf_repo(args.hf_repo)
    
    # Load Mimi (audio codec)
    log("info", "loading mimi")
    mimi = checkpoint_info.get_mimi(device=args.device)
    log("info", "mimi loaded")
    
    # Load text tokenizer
    text_tokenizer = checkpoint_info.get_text_tokenizer()
    
    # Load base Moshi model
    log("info", "loading base moshi model")
    model = checkpoint_info.get_moshi(
        device=args.device,
        dtype=torch.bfloat16,
        load_weight=True
    )
    log("info", "base moshi model loaded")
    
    # Load LoRA adapters
    lora_path = checkpoint_dir / "lora.safetensors"
    if not lora_path.exists():
        raise FileNotFoundError(f"lora.safetensors not found in {checkpoint_dir}")
    
    log("info", f"loading LoRA adapters from {lora_path.name}")
    lora_state_dict = safetensors.torch.load_file(str(lora_path))
    
    # Load LoRA weights (strict=False because base model already has weights)
    missing, unexpected = model.load_state_dict(lora_state_dict, strict=False)
    
    logger.info(f"✅ Loaded {len(lora_state_dict)} LoRA parameters")
    if missing:
        logger.info(f"   (Skipped {len(missing)} base model parameters - expected)")

    # Fix RoPE max_period for long-context generation (6000+ tokens)
    logger.info("🔧 Applying RoPE fix for long-context generation...")
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'rope') and model.transformer.rope is not None:
        model.transformer.rope.max_period = 100_000
        model.transformer.max_period = 100_000
        logger.info(f"✅ Set main transformer RoPE max_period: 10,000 → 100,000")
    if hasattr(model, 'depformer') and hasattr(model.depformer, 'transformer'):
        if hasattr(model.depformer.transformer, 'rope') and model.depformer.transformer.rope is not None:
            model.depformer.transformer.rope.max_period = 100_000
            model.depformer.transformer.max_period = 100_000
            logger.info(f"✅ Set depformer RoPE max_period: 10,000 → 100,000")

    # Increase KV cache context size to avoid wraparound at 3000 tokens
    logger.info("🔧 Increasing KV cache context size...")
    new_context = 8000
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'layers'):
        old_context = None
        for layer in model.transformer.layers:
            if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'context'):
                if old_context is None:
                    old_context = layer.self_attn.context
                layer.self_attn.context = new_context
        if old_context is not None:
            logger.info(f"✅ Set transformer context: {old_context} → {new_context}")

    # Load input audio
    log("info", f"loading input file {args.infile}")
    in_pcms, _ = sphn.read(args.infile, sample_rate=mimi.sample_rate)
    in_pcms = torch.from_numpy(in_pcms).to(device=args.device)
    in_pcms = in_pcms[None, 0:1]  # Single sample inference
    
    # Run inference
    log("info", "running inference")
    with torch.no_grad():
        # Simple forward pass (no streaming for now)
        codes = mimi.encode(in_pcms)
        
        # Create LM generator
        lm_gen = LMGen(model, **checkpoint_info.lm_gen_config)
        lm_gen.streaming_forever(1)
        mimi.streaming_forever(1)
        
        # Process in chunks
        out_pcms = []
        frame_size = int(mimi.sample_rate / mimi.frame_rate)
        chunks = [chunk for chunk in in_pcms.split(frame_size, dim=2) 
                  if chunk.shape[-1] == frame_size]
        
        for chunk in chunks:
            codes = mimi.encode(chunk)
            tokens = lm_gen.step(codes)
            if tokens is not None and model.dep_q > 0:
                out_pcm = mimi.decode(tokens[:, 1:])
                out_pcms.append(out_pcm)
        
        # Concatenate output
        if out_pcms:
            final_out = torch.cat(out_pcms, dim=2)
        else:
            logger.warning("No output generated")
            final_out = torch.zeros_like(in_pcms)
    
    # Save output
    log("info", f"writing {args.outfile}")
    sphn.write_wav(
        args.outfile,
        final_out[0, 0].cpu().numpy(),
        sample_rate=mimi.sample_rate
    )
    log("info", "done!")


if __name__ == "__main__":
    main()
