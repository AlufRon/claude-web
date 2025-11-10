#!/usr/bin/env python3
"""
Minimal wrapper to run Moshi inference with TTT-finetuned checkpoints.

This script loads a TTT checkpoint and runs streaming inference using the
existing run_inference.py infrastructure.

Usage:
    python run_inference_with_ttt.py \\
        --checkpoint /path/to/checkpoint_000100/consolidated \\
        --hf-repo kyutai/moshiko-pytorch-bf16
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import safetensors.torch
import sphn  # Moshi's audio library
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from moshi.models.loaders import CheckpointInfo
from moshi.models import LMGen
from moshi.utils.compile import no_cuda_graph  # For TTT compatibility
from moshi.client_utils import Printer, RawPrinter, log  # For text output formatting
from moshi.modules.lora import replace_all_linear_with_lora  # For LoRA support
from finetune.ttt_integration import apply_ttt_to_model, verify_ttt_integration
from finetune.args import TTTArgs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def seed_all(seed: int):
    """
    Set random seed for reproducible inference.
    Matches the behavior of moshi/run_inference.py for deterministic outputs.

    Args:
        seed: Random seed value (Moshi uses 4242 by default)
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def load_checkpoint_config(checkpoint_dir: Path) -> Dict[str, Any]:
    """
    Load configuration files from checkpoint directory.

    Args:
        checkpoint_dir: Path to consolidated checkpoint directory

    Returns:
        Dictionary with 'training_config' and 'config' keys
    """
    checkpoint_dir = Path(checkpoint_dir)

    # Load training config (contains TTT settings)
    training_config_path = checkpoint_dir / "training_config.json"
    if not training_config_path.exists():
        raise FileNotFoundError(f"training_config.json not found in {checkpoint_dir}")

    with open(training_config_path, 'r') as f:
        training_config = json.load(f)

    # Load model config (contains LM architecture)
    config_path = checkpoint_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {checkpoint_dir}")

    with open(config_path, 'r') as f:
        model_config = json.load(f)

    logger.info(f"✅ Loaded configs from {checkpoint_dir}")

    return {
        'training_config': training_config,
        'config': model_config
    }


def create_ttt_args_from_config(ttt_config: Dict[str, Any]) -> TTTArgs:
    """
    Create TTTArgs object from training_config.json TTT section.

    Args:
        ttt_config: TTT configuration dictionary from training_config.json

    Returns:
        TTTArgs object
    """
    # Create TTTArgs with exact settings from checkpoint
    args = TTTArgs(
        enable=ttt_config.get('enable', False),
        layers=ttt_config.get('layers', 'none'),
        base_lr=ttt_config.get('base_lr', 0.001),
        mini_batch_size=ttt_config.get('mini_batch_size', 32),
        persistent_states=ttt_config.get('persistent_states', True),
        initial_gating_alpha=ttt_config.get('initial_gating_alpha', 0.1),
        log_inner_loop_losses=ttt_config.get('log_inner_loop_losses', False),
        inner_loop_log_interval=ttt_config.get('inner_loop_log_interval', 1),
        save_inner_loop_plots=ttt_config.get('save_inner_loop_plots', False),
        inner_loop_plot_dir=ttt_config.get('inner_loop_plot_dir', './evaluation_plots/inner_loop'),
    )

    # Add multi-layer TTT-MLP configuration if present
    if 'ttt_mlp_layers' in ttt_config:
        args.ttt_mlp_layers = ttt_config['ttt_mlp_layers']

    if 'ttt_mlp_expansion_factor' in ttt_config:
        args.ttt_mlp_expansion_factor = ttt_config['ttt_mlp_expansion_factor']

    if 'ttt_mlp_hidden_dims' in ttt_config:
        args.ttt_mlp_hidden_dims = ttt_config['ttt_mlp_hidden_dims']

    # Add attention context configuration if present
    if 'ttt_layer_context' in ttt_config:
        args.ttt_layer_context = ttt_config['ttt_layer_context']

    if 'non_ttt_layer_context' in ttt_config:
        args.non_ttt_layer_context = ttt_config['non_ttt_layer_context']

    logger.info(f"✅ Created TTTArgs from checkpoint config:")
    logger.info(f"   Layers: {args.layers}")
    logger.info(f"   Base LR: {args.base_lr}")
    logger.info(f"   Mini batch size: {args.mini_batch_size}")
    logger.info(f"   Persistent states: {args.persistent_states}")
    logger.info(f"   Initial gating alpha: {args.initial_gating_alpha}")

    if hasattr(args, 'ttt_mlp_layers'):
        logger.info(f"   TTT-MLP layers: {args.ttt_mlp_layers}")
    if hasattr(args, 'ttt_mlp_expansion_factor'):
        logger.info(f"   TTT-MLP expansion factor: {args.ttt_mlp_expansion_factor}")
    if hasattr(args, 'ttt_layer_context'):
        logger.info(f"   TTT layer context: {args.ttt_layer_context}")
    if hasattr(args, 'non_ttt_layer_context'):
        logger.info(f"   Non-TTT layer context: {args.non_ttt_layer_context}")

    return args


def load_ttt_model(checkpoint_dir: Path, hf_repo: str, device: str = "cuda") -> torch.nn.Module:
    """
    Load Moshi model with TTT checkpoint.

    Args:
        checkpoint_dir: Path to consolidated checkpoint directory
        hf_repo: HuggingFace repository ID for base Moshi model
        device: Device to load model on

    Returns:
        Model with TTT layers and loaded weights
    """
    checkpoint_dir = Path(checkpoint_dir)

    # Step 1: Load checkpoint configs
    logger.info("📂 Loading checkpoint configuration...")
    configs = load_checkpoint_config(checkpoint_dir)
    training_config = configs['training_config']
    model_config = configs['config']

    # Step 2: Detect LoRA and TTT configuration
    ttt_config = training_config.get('ttt', {})
    lora_config = training_config.get('lora', {})

    ttt_enabled = ttt_config.get('enable', False)
    lora_enabled = lora_config.get('enable', False)

    logger.info(f"📊 Checkpoint configuration:")
    logger.info(f"   TTT enabled: {ttt_enabled}")
    logger.info(f"   LoRA enabled: {lora_enabled}")

    # Create TTTArgs if TTT is enabled
    ttt_args = None
    if ttt_enabled:
        ttt_args = create_ttt_args_from_config(ttt_config)

    # Step 3: Load base Moshi model (without weights first)
    logger.info("🔨 Loading base Moshi model...")
    checkpoint_info = CheckpointInfo.from_hf_repo(hf_repo)

    # Load model structure
    model = checkpoint_info.get_moshi(
        device=device,
        dtype=torch.bfloat16,
        load_weight=True  # Load base Moshi weights
    )

    logger.info(f"✅ Loaded base Moshi model from {hf_repo}")

    # Step 4: Apply LoRA integration (if enabled in checkpoint)
    if lora_enabled:
        logger.info("🔧 Applying LoRA integration...")
        lora_rank = lora_config.get('rank', 128)
        lora_scaling = lora_config.get('scaling', 2.0)

        logger.info(f"   LoRA rank: {lora_rank}")
        logger.info(f"   LoRA scaling: {lora_scaling}")

        # Replace all linear layers with LoRA
        replace_all_linear_with_lora(model, lora_rank, lora_scaling, device=device)
        logger.info("✅ LoRA integration applied")
    else:
        logger.info("⏭️  LoRA not enabled in checkpoint")

    # Step 5: Apply TTT integration (if enabled in checkpoint)
    if ttt_enabled:
        logger.info("🧠 Applying TTT integration...")
        apply_ttt_to_model(model, ttt_args, model_config)

        # Verify TTT integration
        if not verify_ttt_integration(model):
            raise RuntimeError("TTT integration verification failed!")

        logger.info("✅ TTT integration applied")
    else:
        logger.info("⏭️  TTT not enabled in checkpoint")

    # Ensure base model is in bfloat16, but we'll keep TTT/LoRA weights for precision
    logger.info("🔄 Converting base model to bfloat16...")
    model = model.to(torch.bfloat16)

    # Step 6: Load finetuned parameters (TTT/LoRA/both) from checkpoint
    logger.info("📥 Loading finetuned parameters from checkpoint...")
    weights_path = checkpoint_dir / "lora.safetensors"

    if not weights_path.exists():
        raise FileNotFoundError(f"lora.safetensors not found in {checkpoint_dir}")

    state_dict = safetensors.torch.load_file(str(weights_path))
    logger.info(f"✅ Loaded {len(state_dict)} parameters from {weights_path.name}")

    # CRITICAL: Keep TTT weights in float32 for accumulating small updates during inference
    # During training, the optimizer maintains float32 master weights, but during inference
    # we directly update the weights, so we need float32 precision to avoid losing tiny updates
    # LoRA weights can stay in their checkpoint dtype (typically float32 or bfloat16)
    if ttt_enabled:
        logger.info("🔄 Keeping TTT weights in float32 for precision (ttt_norm can be bfloat16)...")
        for key in state_dict.keys():
            # Keep weights/biases in float32, but ttt_norm can be bfloat16 (less critical)
            if 'ttt' in key.lower():
                if 'ttt_norm' in key:
                    state_dict[key] = state_dict[key].to(torch.bfloat16)
                else:
                    state_dict[key] = state_dict[key].to(torch.float32)

    # Load finetuned parameters (strict=False for partial loading)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    # Handle loading results
    if missing:
        logger.warning(f"⚠️  Missing keys (expected for partial loading): {len(missing)} keys")

    if unexpected:
        # Only treat as error if neither LoRA nor TTT are enabled
        if not lora_enabled and not ttt_enabled:
            logger.error(f"❌ Unexpected keys in checkpoint: {unexpected[:5]}")
            raise RuntimeError("Unexpected keys in checkpoint!")
        else:
            # With LoRA/TTT, unexpected keys from strict=False are normal
            logger.debug(f"   Unexpected keys handled (strict=False): {len(unexpected)}")

    logger.info("✅ Checkpoint loaded successfully!")

    # Step 7: Handle gating_alpha override (only if TTT is enabled)
    if ttt_enabled:
        # Override gating_alpha if explicitly requested in config
        # Respects the override_gating_alpha_on_resume flag (no hardcoded policies)
        override_gating = ttt_config.get('override_gating_alpha_on_resume', False)
        config_alpha = ttt_config.get('initial_gating_alpha', None)

        if override_gating and config_alpha is not None:
            logger.info(f"🔧 Overriding loaded gating_alpha with config value: {config_alpha}")
            logger.info(f"   (override_gating_alpha_on_resume = {override_gating})")
            for name, module in model.named_modules():
                if hasattr(module, 'gating_alpha') and isinstance(module.gating_alpha, torch.nn.Parameter):
                    with torch.no_grad():
                        module.gating_alpha.fill_(config_alpha)
                    logger.info(f"   Set {name}.gating_alpha = {config_alpha}")
        else:
            # Show the actual gating_alpha values loaded from checkpoint
            logger.info(f"ℹ️  Using gating_alpha from checkpoint (override_gating_alpha_on_resume = {override_gating})")
            if config_alpha is not None:
                logger.info(f"   Config specifies initial_gating_alpha = {config_alpha}, but checkpoint values will be used")

            # Log the loaded gating_alpha values
            logger.info(f"🔧 Loaded gating_alpha values from checkpoint:")
            for name, module in model.named_modules():
                if hasattr(module, 'gating_alpha') and isinstance(module.gating_alpha, torch.nn.Parameter):
                    # gating_alpha can be scalar or per-dimension tensor
                    if module.gating_alpha.numel() == 1:
                        alpha_value = module.gating_alpha.item()
                        logger.info(f"   {name}.gating_alpha = {alpha_value:.6f}")
                    else:
                        alpha_mean = module.gating_alpha.mean().item()
                        alpha_min = module.gating_alpha.min().item()
                        alpha_max = module.gating_alpha.max().item()
                        logger.info(f"   {name}.gating_alpha: mean={alpha_mean:.6f}, min={alpha_min:.6f}, max={alpha_max:.6f}")

    # Step 8: Enable persistence logging (optional but useful)
    if ttt_enabled:
        enable_persistence_logging(model)

    # Step 9: Fix RoPE max_period for long-context generation (6000+ tokens)
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

    # Step 10: Set context size - from training config (not hardcoded!)
    logger.info("🔧 Setting KV cache context size from training config...")

    # Get from ttt_config, default to Moshi's default (3000) if not present
    # This ensures train/inference match for attention context
    ttt_context = ttt_config.get('ttt_layer_context', 3000)
    non_ttt_context = ttt_config.get('non_ttt_layer_context', 3000)

    logger.info(f"   TTT layer context: {ttt_context} tokens")
    logger.info(f"   Non-TTT layer context: {non_ttt_context} tokens")

    modified_ttt = 0
    modified_non_ttt = 0

    if hasattr(model, 'transformer') and hasattr(model.transformer, 'layers'):
        old_context = None
        for layer in model.transformer.layers:
            # Handle regular StreamingTransformerLayer (non-TTT)
            if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'context'):
                if old_context is None:
                    old_context = layer.self_attn.context
                layer.self_attn.context = non_ttt_context
                modified_non_ttt += 1
            # Handle HybridStreamingTransformerLayer (TTT layers)
            elif hasattr(layer, 'original_layer') and hasattr(layer.original_layer, 'self_attn'):
                if hasattr(layer.original_layer.self_attn, 'context'):
                    if old_context is None:
                        old_context = layer.original_layer.self_attn.context
                    layer.original_layer.self_attn.context = ttt_context  # Context from training config
                    modified_ttt += 1
        if old_context is not None:
            logger.info(f"✅ Set context: {old_context} → TTT layers: {ttt_context} ({modified_ttt} layers), Non-TTT: {non_ttt_context} ({modified_non_ttt} layers)")

    # Also increase depformer context if needed
    if hasattr(model, 'depformer') and hasattr(model.depformer, 'transformer'):
        if hasattr(model.depformer.transformer, 'layers'):
            old_context = None
            depformer_layers = 0
            for layer in model.depformer.transformer.layers:
                if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'context'):
                    if old_context is None:
                        old_context = layer.self_attn.context
                    layer.self_attn.context = non_ttt_context
                    depformer_layers += 1
            if old_context is not None:
                logger.info(f"✅ Set depformer context: {old_context} → {non_ttt_context} ({depformer_layers} layers)")

    return model


def print_ttt_statistics_summary(model: torch.nn.Module) -> None:
    """
    Print summary statistics for TTT output and gating across all layers.

    Args:
        model: Model with TTT layers
    """
    import numpy as np

    logger.info("=" * 80)
    logger.info("TTT Statistics Summary")
    logger.info("=" * 80)

    for layer_idx, layer in enumerate(model.transformer.layers):
        if not hasattr(layer, 'seq_modeling_block'):
            continue
        if not hasattr(layer.seq_modeling_block, 'ttt_layer'):
            continue

        # Check for accumulated stats
        ttt_instance = layer.seq_modeling_block.ttt_layer.ttt
        hybrid_block = layer.seq_modeling_block

        has_output_stats = hasattr(ttt_instance, '_output_stats')
        has_gating_stats = hasattr(hybrid_block, '_gating_stats')

        if not has_output_stats and not has_gating_stats:
            continue

        logger.info(f"\n📊 Layer {layer_idx} Statistics:")

        # TTT Output statistics
        if has_output_stats:
            stats = ttt_instance._output_stats
            if stats['norms']:
                norms = np.array(stats['norms'])
                maxs = np.array(stats['maxs'])
                logger.info(f"  TTT Output (across {len(norms)} tokens):")
                logger.info(f"    Norm:  mean={norms.mean():.2f}, std={norms.std():.2f}, "
                           f"min={norms.min():.2f}, max={norms.max():.2f}")
                logger.info(f"    Max:   mean={maxs.mean():.2f}, std={maxs.std():.2f}, "
                           f"min={maxs.min():.2f}, max={maxs.max():.2f}")

        # Gating statistics
        if has_gating_stats:
            stats = hybrid_block._gating_stats
            if stats['pre_gate_norms']:
                pre_norms = np.array(stats['pre_gate_norms'])
                post_norms = np.array(stats['post_gate_norms'])
                residual_norms = np.array(stats['residual_norms'])
                alphas = np.array(stats['alphas'])
                gate_ratios = np.array(stats['gate_ratios'])

                logger.info(f"  Gating (across {len(pre_norms)} tokens):")
                logger.info(f"    Pre-gate norm:   mean={pre_norms.mean():.2f}, std={pre_norms.std():.2f}")
                logger.info(f"    Post-gate norm:  mean={post_norms.mean():.2f}, std={post_norms.std():.2f}")
                logger.info(f"    Residual norm:   mean={residual_norms.mean():.2f}, std={residual_norms.std():.2f}")
                logger.info(f"    Alpha:           mean={alphas.mean():.6f}, std={alphas.std():.6f}")
                logger.info(f"    Gate ratio:      mean={gate_ratios.mean():.4f}, std={gate_ratios.std():.4f}")
                logger.info(f"    TTT contribution: {100 * post_norms.mean() / residual_norms.mean():.1f}% of residual")

    logger.info("=" * 80)


def enable_persistence_logging(model: torch.nn.Module) -> None:
    """
    Enable TTT persistence logging for debugging/verification.

    Args:
        model: Model with TTT layers
    """
    enabled_count = 0

    for layer in model.transformer.layers:
        # Check for HybridStreamingTransformerLayer
        if hasattr(layer, 'seq_modeling_block'):
            # Check for HybridSeqModelingBlock with TTT
            if hasattr(layer.seq_modeling_block, 'ttt_layer'):
                # Check for TTTWrapper with actual TTT instance
                if hasattr(layer.seq_modeling_block.ttt_layer, 'ttt'):
                    # Set flag on the actual TTTMLP instance
                    layer.seq_modeling_block.ttt_layer.ttt._persistence_check_enabled = True
                    enabled_count += 1

    if enabled_count > 0:
        logger.info(f"✅ Enabled persistence logging for {enabled_count} TTT layers")
    else:
        logger.warning("⚠️  No TTT layers found for persistence logging")


def plot_waveform(audio_data: np.ndarray, sample_rate: int, output_path: str):
    """
    Create and save a waveform plot showing when the model is speaking.

    Args:
        audio_data: Audio waveform as numpy array (1D)
        sample_rate: Sample rate in Hz
        output_path: Path where the plot should be saved
    """
    try:
        # Calculate time axis
        duration = len(audio_data) / sample_rate
        time = np.linspace(0, duration, len(audio_data))

        # Create figure with good size for visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8))

        # Plot 1: Full waveform
        ax1.plot(time, audio_data, linewidth=0.5, color='#2E86AB')
        ax1.set_xlabel('Time (seconds)', fontsize=12)
        ax1.set_ylabel('Amplitude', fontsize=12)
        ax1.set_title('Generated Audio Waveform - Full View', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, duration)

        # Add zero line
        ax1.axhline(y=0, color='red', linestyle='--', linewidth=0.5, alpha=0.5)

        # Plot 2: Amplitude envelope (shows speech activity)
        # Calculate envelope using absolute value with smoothing
        window_size = int(sample_rate * 0.02)  # 20ms window
        envelope = np.abs(audio_data)

        # Smooth the envelope
        if len(envelope) > window_size:
            envelope = np.convolve(envelope, np.ones(window_size)/window_size, mode='same')

        ax2.fill_between(time, envelope, alpha=0.6, color='#A23B72', label='Speech Activity')
        ax2.plot(time, envelope, linewidth=0.8, color='#A23B72')
        ax2.set_xlabel('Time (seconds)', fontsize=12)
        ax2.set_ylabel('Amplitude Envelope', fontsize=12)
        ax2.set_title('Speech Activity (Amplitude Envelope) - Shows When Model Speaks', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, duration)
        ax2.legend(loc='upper right')

        # Add threshold line to show significant speech
        threshold = np.percentile(envelope, 75)  # 75th percentile
        ax2.axhline(y=threshold, color='green', linestyle='--', linewidth=1,
                   alpha=0.7, label=f'Activity Threshold ({threshold:.3f})')
        ax2.legend(loc='upper right')

        # Tight layout
        plt.tight_layout()

        # Save plot
        plot_path = output_path.replace('.wav', '_waveform.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        log("info", f"waveform plot saved to {plot_path}")

        # Also print some statistics
        speech_ratio = np.sum(envelope > threshold) / len(envelope) * 100
        log("info", f"speech activity: {speech_ratio:.1f}% of audio above threshold")

    except Exception as e:
        logger.error(f"Failed to create waveform plot: {e}")


def run_audio_inference(
    model: torch.nn.Module,
    mimi,
    text_tokenizer,
    checkpoint_info,
    audio_path: str,
    output_path: str,
    device: str,
    batch_size: int = 1,
    repetition_penalty: float = 1.0,
    repetition_window: int = 64,
):
    """
    Run inference on audio file, following Moshi's run_inference.py workflow.

    Args:
        repetition_penalty: Penalty for tokens that appear in recent history (1.0 = disabled)
        repetition_window: Number of recent tokens to track for repetition penalty
    """
    logger.info("=" * 80)
    logger.info("Running Audio Inference with TTT")
    logger.info("=" * 80)

    # Setup printer (same as original Moshi)
    if sys.stdout.isatty():
        printer = Printer()
    else:
        printer = RawPrinter()

    # Load audio
    log("info", f"loading audio from {audio_path}")
    in_pcms, _ = sphn.read(audio_path, sample_rate=mimi.sample_rate)
    in_pcms = torch.from_numpy(in_pcms).to(device=device)
    in_pcms = in_pcms[None, 0:1].expand(batch_size, -1, -1)

    duration = in_pcms.shape[2] / mimi.sample_rate
    log("info", f"loaded {duration:.1f}s of audio at {mimi.sample_rate} Hz")

    # IMPORTANT: Disable CUDA graphs for TTT compatibility
    # TTT layers move tensors to device during forward pass, which is not allowed during CUDA graph capture
    log("info", "disabling CUDA graphs for TTT compatibility")

    with no_cuda_graph():
        # Create LMGen for streaming inference
        log("info", "initializing streaming inference")
        # Override lm_gen_config with repetition penalty parameters
        lm_gen_config = checkpoint_info.lm_gen_config.copy()
        lm_gen_config['repetition_penalty'] = repetition_penalty
        lm_gen_config['repetition_window'] = repetition_window
        # Remove unsupported keys before passing to LMGen
        lm_gen_config.pop('repetition_penalty', None)
        lm_gen_config.pop('repetition_window', None)
        lm_gen = LMGen(model, **lm_gen_config)

        # Log repetition penalty settings
        if repetition_penalty != 1.0:
            log("info", f"repetition penalty enabled: {repetition_penalty} (window: {repetition_window})")
        else:
            log("info", "repetition penalty disabled")

        # Setup streaming
        frame_size = int(mimi.sample_rate / mimi.frame_rate)
        mimi.streaming_forever(batch_size)
        lm_gen.streaming_forever(batch_size)

        # Process audio frame by frame
        from collections import deque
        chunks = deque([
            chunk
            for chunk in in_pcms.split(frame_size, dim=2)
            if chunk.shape[-1] == frame_size
        ])

        out_pcms_list = []
        out_text_tokens = []
        first_frame = True
        ntokens = 0

        import time
        start_time = time.time()

        # Print header like original Moshi
        printer.log(
            "info",
            f"starting inference, "
            f"sampling: {lm_gen.use_sampling}, "
            f"audio temp: {lm_gen.temp}, "
            f"text temp: {lm_gen.temp_text}",
        )
        printer.print_header()

        while chunks:
            chunk = chunks.popleft()
            codes = mimi.encode(chunk)

            if first_frame:
                # First frame needs special handling
                tokens = lm_gen.step(codes)
                if max(lm_gen.lm_model.delays) > 0:
                    assert tokens is None
                first_frame = False

            tokens = lm_gen.step(codes)
            if tokens is None:
                continue

            ntokens += 1

            # Decode audio output
            if lm_gen.lm_model.dep_q > 0:
                out_pcm = mimi.decode(tokens[:, 1:]).cpu()
                text_token = tokens[:, 0].cpu()

                out_pcms_list.append(out_pcm[0])
                out_text_tokens.append(text_token[0])

                # Print text output using Moshi's printer (same as original)
                if text_token[0].item() not in [0, 3]:
                    text = text_tokenizer.id_to_piece(text_token[0].item())
                    text = text.replace("▁", " ")
                    printer.print_token(text)

        dt = time.time() - start_time
        printer.log(
            "info",
            f"processed {ntokens} steps in {dt:.0f}s, {1000 * dt / ntokens:.2f}ms/step",
        )

    # Save output (same format as original Moshi)
    if out_pcms_list and output_path:
        out_pcm = torch.cat(out_pcms_list, dim=1)
        out_duration = out_pcm.shape[1] / mimi.sample_rate

        log("info", f"writing {output_path} with duration {out_duration:.1f} sec.")
        sphn.write_wav(output_path, out_pcm[0].numpy(), sample_rate=mimi.sample_rate)

        # Create waveform plot
        log("info", "creating waveform plot...")
        plot_waveform(out_pcm[0].numpy(), mimi.sample_rate, output_path)

    # Print TTT statistics summary
    print_ttt_statistics_summary(model)

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run Moshi inference with TTT-finetuned checkpoint"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to consolidated checkpoint directory (e.g., checkpoint_000100/consolidated)"
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        default="kyutai/moshiko-pytorch-bf16",
        help="HuggingFace repository for base Moshi model (default: kyutai/moshiko-pytorch-bf16)"
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
        nargs="?",
        default=None,
        help="Input audio file (WAV format)"
    )
    parser.add_argument(
        "outfile",
        type=str,
        nargs="?",
        default=None,
        help="Output audio file (WAV format)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for inference (default: 1)"
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="Repetition penalty for token sampling (default: 1.0 = disabled, recommended: 1.15)"
    )
    parser.add_argument(
        "--repetition-window",
        type=int,
        default=64,
        help="Window size for tracking recent tokens for repetition penalty (default: 64)"
    )

    args = parser.parse_args()

    # Set random seed for deterministic inference (matches moshi/run_inference.py)
    seed_all(4242)
    logger.info("🎲 Random seed set to 4242 for deterministic inference")

    # Load TTT model
    logger.info("=" * 80)
    logger.info("Moshi Inference with TTT")
    logger.info("=" * 80)

    # Load checkpoint info and components
    checkpoint_dir = Path(args.checkpoint)
    configs = load_checkpoint_config(checkpoint_dir)
    training_config = configs['training_config']

    # Load checkpoint info for mimi and tokenizer
    log("info", "loading moshi components")
    checkpoint_info = CheckpointInfo.from_hf_repo(args.hf_repo)

    # Load Mimi (audio codec)
    log("info", "loading mimi")
    mimi = checkpoint_info.get_mimi(device=args.device)
    log("info", "mimi loaded")

    # Load text tokenizer
    text_tokenizer = checkpoint_info.get_text_tokenizer()

    # Load TTT model
    log("info", "loading moshi with TTT")
    model = load_ttt_model(
        checkpoint_dir=checkpoint_dir,
        hf_repo=args.hf_repo,
        device=args.device
    )

    if model is None:
        log("error", "failed to load TTT model")
        return 1

    log("info", "moshi with TTT loaded")

    # Run audio inference if input provided (just like original Moshi)
    if args.infile:
        try:
            model.eval()
            with torch.no_grad():
                success = run_audio_inference(
                    model=model,
                    mimi=mimi,
                    text_tokenizer=text_tokenizer,
                    checkpoint_info=checkpoint_info,
                    audio_path=args.infile,
                    output_path=args.outfile if args.outfile else "",
                    device=args.device,
                    batch_size=args.batch_size,
                    repetition_penalty=args.repetition_penalty,
                    repetition_window=args.repetition_window,
                )

            if success:
                return 0
            else:
                return 1

        except Exception as e:
            log("error", f"audio inference failed: {e}")
            import traceback
            log("error", f"traceback:\n{traceback.format_exc()}")
            return 1

    else:
        # Test mode: just verify model loads
        log("info", "no input file provided, running test")
        try:
            model.eval()
            with torch.no_grad():
                num_codebooks = model.num_codebooks
                batch_size = 1
                seq_len = 10
                dummy_codes = torch.randint(0, 2048, (batch_size, num_codebooks, seq_len), device=args.device)
                output = model(dummy_codes)
                log("info", f"test passed! model ready for inference")
                log("info", f"output shapes: logits={output.logits.shape}, text_logits={output.text_logits.shape}")

        except Exception as e:
            log("error", f"test failed: {e}")
            return 1

        log("info", "model loaded and tested successfully")
        log("info", "usage: python run_inference_with_ttt.py --checkpoint <checkpoint_dir> <input.wav> <output.wav>")

    return 0


if __name__ == "__main__":
    sys.exit(main())
