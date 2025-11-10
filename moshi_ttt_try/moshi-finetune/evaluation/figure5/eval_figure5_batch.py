"""
Batch Evaluation Script for Figure 5 - TTT Loss Trajectories

This script processes a long LibriLight sequence in ONE forward pass (batch mode),
allowing scan() to update W₀ → W₁ → W₂ ... → W₂₀₄₇ within that single forward pass.

This is the correct way to replicate the paper's Figure 4, which shows cumulative TTT learning.

Usage:
    python eval_figure5_batch.py --checkpoint_path <path> --librilight_file <path>
"""

import torch
import logging
import argparse
from pathlib import Path
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model_and_checkpoint(checkpoint_path: str):
    """Load model from checkpoint."""
    logger.info(f"Loading checkpoint from: {checkpoint_path}")

    # Import Moshi model
    from moshi.models import loaders

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Extract config and state_dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        config = checkpoint.get('config', None)
    else:
        state_dict = checkpoint
        config = None

    # TODO: Properly reconstruct model from config
    # For now, we need to figure out how to load the model architecture

    logger.info(f"Checkpoint loaded: {len(state_dict)} keys")
    return None, state_dict, config

def load_librilight_audio(audio_file: str, mimi_encoder, max_tokens: int = 2048):
    """Load and encode LibriLight audio file."""
    logger.info(f"Loading audio from: {audio_file}")

    import torchaudio

    # Load audio
    waveform, sample_rate = torchaudio.load(audio_file)

    # Resample if needed
    if sample_rate != mimi_encoder.sample_rate:
        logger.info(f"Resampling from {sample_rate} to {mimi_encoder.sample_rate}")
        resampler = torchaudio.transforms.Resample(sample_rate, mimi_encoder.sample_rate)
        waveform = resampler(waveform)

    # Ensure correct shape [B, C, T]
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0).unsqueeze(0)
    elif waveform.dim() == 2:
        waveform = waveform.unsqueeze(0)

    # Encode with MIMI
    logger.info(f"Encoding audio with MIMI...")
    with torch.no_grad():
        codes = mimi_encoder.encode(waveform)  # [B, K, T]

    logger.info(f"Encoded {codes.shape[-1]} tokens")

    # Limit to max_tokens if specified
    if max_tokens and codes.shape[-1] > max_tokens:
        codes = codes[:, :, :max_tokens]
        logger.info(f"Limited to {max_tokens} tokens")

    return codes

def evaluate_figure5_batch(model, codes, device='cuda'):
    """
    Evaluate Figure 5 by processing entire sequence in batch mode.

    This allows scan() to process all mini-batches in one forward pass,
    enabling cumulative TTT learning W₀ → W₁ → W₂ ... → Wₜ.
    """
    logger.info(f"\n{'='*80}")
    logger.info("Starting Figure 5 Batch Evaluation")
    logger.info(f"{'='*80}")

    model.eval()
    model = model.to(device)
    codes = codes.to(device)

    B, K, T = codes.shape
    logger.info(f"Sequence shape: B={B}, K={K} codebooks, T={T} tokens")

    # Enable Figure 5 logging
    from moshi_ttt.models.ssm.ops.ttt_mlp import fig5_set_logging, fig5_clear, fig5_get

    fig5_clear()
    fig5_set_logging(True, max_T=T)
    logger.info(f"✅ Figure 5 logging enabled for {T} positions")

    # Reset stream position for all TTT layers
    reset_count = 0
    for layer in model.transformer.layers:
        if hasattr(layer, 'seq_modeling_block'):
            block = layer.seq_modeling_block
            if hasattr(block, 'ttt_layer'):
                ttt = block.ttt_layer
                if hasattr(ttt, 'ttt') and hasattr(ttt.ttt, 'stream_position'):
                    ttt.ttt.stream_position = 0
                    reset_count += 1

    logger.info(f"Reset stream_position for {reset_count} TTT layers")

    # Prepare input: need 17 codebooks for Moshi
    # [0] = text (can be zeros for audio-only)
    # [1-8] = Moshi audio (LibriLight)
    # [9-16] = User audio (zeros for audio-only evaluation)

    inp = torch.zeros(B, 17, T, device=device, dtype=codes.dtype)
    inp[:, 1:9, :] = codes  # Place LibriLight audio in Moshi stream

    logger.info(f"Prepared input tensor: {inp.shape}")

    # Forward pass through model (batch mode - entire sequence at once)
    logger.info(f"\n🚀 Starting batch forward pass ({T} tokens)...")

    with torch.no_grad():
        try:
            # Create dummy sequence metadata (simplified)
            from moshi_ttt.utils import SequenceMetadata
            seq_metadata = SequenceMetadata(
                is_multiscene=False,
                init_offset=None,
                base_offset=None,
                num_chunks=1,
                text_length=0,
            )

            # Process through model
            # Note: This is simplified - actual Moshi forward may require more setup
            output = model(inp, seq_metadata)

            logger.info(f"✅ Forward pass complete! Output shape: {output.shape}")

        except Exception as e:
            logger.error(f"❌ Forward pass failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    # Collect Figure 5 data
    logger.info(f"\n📊 Collecting Figure 5 data...")
    fig5_data = fig5_get()

    if not fig5_data:
        logger.warning("No Figure 5 data collected!")
        return None

    logger.info(f"Collected data for {len(fig5_data)} layers")
    for layer_id, data in fig5_data.items():
        num_positions = len([c for c in data['cnt'] if c > 0])
        logger.info(f"  Layer {layer_id}: {num_positions} positions")

    return fig5_data

def plot_figure5(fig5_data, output_dir='./evaluation_plots/figure5_batch'):
    """Generate Figure 5 plots from collected data."""
    logger.info(f"\n📈 Generating Figure 5 plots...")

    from finetune.figure5_plotting import plot_figure5_trajectories

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Plot Figure 5
    plot_path = plot_figure5_trajectories(
        fig5_data,
        output_dir=output_dir,
        smooth_window=10,
        layers_to_plot=[29, 30, 31]
    )

    logger.info(f"✅ Figure 5 plot saved to: {plot_path}")
    return plot_path

def main():
    parser = argparse.ArgumentParser(description='Evaluate Figure 5 in batch mode')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--librilight_file', type=str, required=True, help='Path to LibriLight audio file (.flac or .wav)')
    parser.add_argument('--max_tokens', type=int, default=2048, help='Maximum tokens to process')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--output_dir', type=str, default='./evaluation_plots/figure5_batch', help='Output directory for plots')

    args = parser.parse_args()

    logger.info("="*80)
    logger.info("Figure 5 Batch Evaluation Script")
    logger.info("="*80)
    logger.info(f"Checkpoint: {args.checkpoint_path}")
    logger.info(f"Audio file: {args.librilight_file}")
    logger.info(f"Max tokens: {args.max_tokens}")
    logger.info(f"Device: {args.device}")
    logger.info("="*80)

    # Check if files exist
    if not Path(args.checkpoint_path).exists():
        logger.error(f"Checkpoint not found: {args.checkpoint_path}")
        sys.exit(1)

    if not Path(args.librilight_file).exists():
        logger.error(f"Audio file not found: {args.librilight_file}")
        sys.exit(1)

    # Load model and checkpoint
    model, state_dict, config = load_model_and_checkpoint(args.checkpoint_path)

    if model is None:
        logger.error("Failed to load model - need to implement model reconstruction")
        logger.info("\n⚠️  TODO: Implement proper model loading from checkpoint")
        logger.info("This requires:")
        logger.info("1. Reconstruct model architecture from config")
        logger.info("2. Load state_dict into model")
        logger.info("3. Load MIMI encoder")
        sys.exit(1)

    # Load LibriLight audio
    # mimi = load_mimi_encoder()  # TODO: Load MIMI
    # codes = load_librilight_audio(args.librilight_file, mimi, args.max_tokens)

    # Evaluate Figure 5
    # fig5_data = evaluate_figure5_batch(model, codes, args.device)

    # Plot results
    # if fig5_data:
    #     plot_figure5(fig5_data, args.output_dir)
    #     logger.info("\n✅ Figure 5 evaluation complete!")
    # else:
    #     logger.error("❌ Failed to collect Figure 5 data")

if __name__ == "__main__":
    main()
