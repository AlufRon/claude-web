#!/usr/bin/env python3
"""
Run batch (non-streaming) inference for Moshi with TTT.

This script demonstrates how to use batch inference mode for evaluation,
which is essential for proper TTT testing. Unlike streaming inference,
batch mode processes entire sequences at once.

Usage:
    python run_batch_inference.py \\
        --checkpoint /path/to/checkpoint_000100/consolidated \\
        --hf-repo kyutai/moshiko-pytorch-bf16 \\
        --input audio1.wav audio2.wav \\
        --output-dir ./batch_results
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
import torchaudio
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.batch_inference import load_batch_inference, BatchInference

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_waveform(audio_data: np.ndarray, sample_rate: int, output_path: str, title: str = "Audio Waveform"):
    """
    Plot audio waveform (amplitude vs time) and save to file.
    
    Args:
        audio_data: Audio samples as numpy array
        sample_rate: Sample rate in Hz
        output_path: Path to save the plot
        title: Plot title
    """
    # Create time axis
    duration = len(audio_data) / sample_rate
    time_axis = np.linspace(0, duration, len(audio_data))
    
    # Create figure with high DPI for quality
    plt.figure(figsize=(12, 6), dpi=300)
    
    # Plot waveform
    plt.plot(time_axis, audio_data, linewidth=0.5, color='blue', alpha=0.8)
    
    # Customize plot
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Amplitude', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add statistics as text
    rms = np.sqrt(np.mean(audio_data**2))
    peak = np.max(np.abs(audio_data))
    stats_text = f'Duration: {duration:.2f}s\nRMS: {rms:.4f}\nPeak: {peak:.4f}\nSample Rate: {sample_rate}Hz'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             fontsize=10)
    
    # Set limits with some padding
    plt.xlim(0, duration)
    y_max = max(abs(audio_data.min()), abs(audio_data.max())) * 1.1
    plt.ylim(-y_max, y_max)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ Saved waveform plot: {output_path}")


def plot_comparison_waveform(input_audio: np.ndarray, generated_audio: np.ndarray, 
                           sample_rate: int, output_path: str, title: str = "Audio Comparison"):
    """
    Plot input vs generated audio waveforms for comparison.
    
    Args:
        input_audio: Input audio samples as numpy array
        generated_audio: Generated audio samples as numpy array  
        sample_rate: Sample rate in Hz
        output_path: Path to save the plot
        title: Plot title
    """
    # Create time axes
    input_duration = len(input_audio) / sample_rate
    generated_duration = len(generated_audio) / sample_rate
    input_time = np.linspace(0, input_duration, len(input_audio))
    generated_time = np.linspace(0, generated_duration, len(generated_audio))
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), dpi=300)
    
    # Plot input audio
    ax1.plot(input_time, input_audio, linewidth=0.5, color='blue', alpha=0.8)
    ax1.set_title('Input Audio', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time (seconds)', fontsize=12)
    ax1.set_ylabel('Amplitude', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Add input stats
    input_rms = np.sqrt(np.mean(input_audio**2))
    input_peak = np.max(np.abs(input_audio))
    input_stats = f'Duration: {input_duration:.2f}s | RMS: {input_rms:.4f} | Peak: {input_peak:.4f}'
    ax1.text(0.02, 0.95, input_stats, transform=ax1.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
             fontsize=10)
    
    # Plot generated audio  
    ax2.plot(generated_time, generated_audio, linewidth=0.5, color='red', alpha=0.8)
    ax2.set_title('Generated Audio (Continuation)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Time (seconds)', fontsize=12)
    ax2.set_ylabel('Amplitude', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Add generated stats
    gen_rms = np.sqrt(np.mean(generated_audio**2))
    gen_peak = np.max(np.abs(generated_audio))
    gen_stats = f'Duration: {generated_duration:.2f}s | RMS: {gen_rms:.4f} | Peak: {gen_peak:.4f}'
    ax2.text(0.02, 0.95, gen_stats, transform=ax2.transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8),
             fontsize=10)
    
    # Set consistent y-limits
    all_audio = np.concatenate([input_audio, generated_audio])
    y_max = max(abs(all_audio.min()), abs(all_audio.max())) * 1.1
    ax1.set_ylim(-y_max, y_max)
    ax2.set_ylim(-y_max, y_max)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ Saved comparison plot: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run batch inference for Moshi with TTT")

    # Model arguments
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
        help='HuggingFace repository ID for base Moshi model'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to run inference on'
    )

    # Input/Output arguments
    parser.add_argument(
        '--input',
        type=str,
        nargs='+',
        required=True,
        help='Input audio files (one or more)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./batch_results',
        help='Directory to save results'
    )

    # Inference arguments
    parser.add_argument(
        '--ttt-mini-batch-size',
        type=int,
        default=None,
        help='Override TTT mini_batch_size (default: use checkpoint config)'
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=None,
        help='Maximum sequence length in tokens (None = process full audio)'
    )
    parser.add_argument(
        '--compute-perplexity',
        action='store_true',
        help='Compute and display perplexity metrics'
    )
    parser.add_argument(
        '--generate-audio',
        action='store_true',
        help='Generate audio output (like streaming inference but in batch mode)'
    )

    # Codebook weighting
    parser.add_argument(
        '--codebook-weights',
        type=str,
        default=None,
        help='Comma-separated weights for each codebook (e.g., "3,1,1,1,1,1,1,1")'
    )

    return parser.parse_args()


def load_audio_files(
    audio_paths: List[str],
    target_sr: int = 24000,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Load and resample audio files to target sample rate.

    Args:
        audio_paths: List of paths to audio files
        target_sr: Target sample rate (Moshi uses 24kHz)
        device: Device to load audio on

    Returns:
        audio_batch: [B, 1, samples] audio waveforms
    """
    audio_list = []

    for path in audio_paths:
        logger.info(f"Loading {path}...")

        # Load audio
        waveform, sr = torchaudio.load(path)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if needed
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr,
                new_freq=target_sr
            )
            waveform = resampler(waveform)

        audio_list.append(waveform)

    # Find max length for padding
    max_len = max(a.shape[-1] for a in audio_list)

    # Pad all to same length
    audio_padded = []
    for waveform in audio_list:
        if waveform.shape[-1] < max_len:
            pad_len = max_len - waveform.shape[-1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))
        audio_padded.append(waveform)

    # Stack to batch
    audio_batch = torch.stack(audio_padded, dim=0)  # [B, 1, samples]

    logger.info(f"✅ Loaded {len(audio_paths)} audio files")
    logger.info(f"   Batch shape: {audio_batch.shape}")
    logger.info(f"   Sample rate: {target_sr} Hz")
    logger.info(f"   Duration: {audio_batch.shape[-1] / target_sr:.2f}s")

    return audio_batch.to(device)


def truncate_to_max_length(
    audio: torch.Tensor,
    max_length: int,
    frame_size: int = 1920,  # Mimi frame size (80ms at 24kHz)
) -> torch.Tensor:
    """
    Truncate audio to maximum length in tokens.

    Args:
        audio: [B, 1, samples] audio
        max_length: Maximum length in tokens
        frame_size: Samples per token (frame)

    Returns:
        Truncated audio [B, 1, samples_truncated]
    """
    max_samples = max_length * frame_size
    if audio.shape[-1] > max_samples:
        logger.info(f"Truncating audio: {audio.shape[-1]} → {max_samples} samples ({max_length} tokens)")
        audio = audio[:, :, :max_samples]
    return audio


def parse_codebook_weights(weights_str: str, dep_q: int = 8) -> torch.Tensor:
    """
    Parse codebook weights from string.

    Args:
        weights_str: Comma-separated weights (e.g., "3,1,1,1,1,1,1,1")
        dep_q: Number of codebooks

    Returns:
        weights: [dep_q] tensor
    """
    weights = [float(w) for w in weights_str.split(',')]
    if len(weights) != dep_q:
        raise ValueError(f"Expected {dep_q} weights, got {len(weights)}")
    return torch.tensor(weights)


def run_batch_inference(args):
    """Main function to run batch inference."""

    logger.info("="*80)
    logger.info("Batch Inference for Moshi with TTT")
    logger.info("="*80)

    # Create unique output directory with timestamp and checkpoint info
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Extract checkpoint name for folder
    checkpoint_path = Path(args.checkpoint)
    checkpoint_name = checkpoint_path.parent.name  # e.g., "checkpoint_008700"
    
    # Extract input file info for folder name
    input_file_name = Path(args.input[0]).stem if args.input else "unknown"
    num_files = len(args.input) if args.input else 0
    
    if num_files > 1:
        folder_suffix = f"{num_files}files"
    else:
        # Use first 20 chars of filename to keep folder name reasonable
        folder_suffix = input_file_name[:20]
    
    # Create unique directory: base_dir/checkpoint_name_timestamp_input/
    base_output_dir = Path(args.output_dir)
    unique_output_dir = base_output_dir / f"{checkpoint_name}_{timestamp}_{folder_suffix}"
    unique_output_dir.mkdir(exist_ok=True, parents=True)
    
    output_dir = unique_output_dir
    logger.info(f"📁 Output directory: {output_dir}")
    logger.info(f"🕐 Timestamp: {timestamp}")
    logger.info(f"📊 Checkpoint: {checkpoint_name}")
    logger.info(f"🎤 Input files: {num_files} files")

    # Step 1: Load model and create BatchInference wrapper
    logger.info("\n" + "="*80)
    logger.info("Step 1: Loading Model")
    logger.info("="*80)

    batch_inf = load_batch_inference(
        checkpoint_dir=args.checkpoint,
        hf_repo=args.hf_repo,
        device=args.device,
        ttt_mini_batch_size=args.ttt_mini_batch_size,
    )

    # Step 2: Load audio files
    logger.info("\n" + "="*80)
    logger.info("Step 2: Loading Audio Files")
    logger.info("="*80)

    audio = load_audio_files(args.input, device=args.device)

    # Truncate if max_length specified
    if args.max_length is not None:
        audio = truncate_to_max_length(audio, args.max_length)
        
    # Plot input audio waveforms
    logger.info("Creating input audio waveform plots...")
    for b in range(audio.shape[0]):
        input_filename = Path(args.input[b]).stem
        input_audio_np = audio[b, 0].cpu().numpy()  # Remove batch and channel dims
        input_plot_path = output_dir / f'{input_filename}_input_waveform.png'
        plot_waveform(
            audio_data=input_audio_np,
            sample_rate=24000,
            output_path=str(input_plot_path),
            title=f"Input Audio Waveform - {input_filename}"
        )

    # Step 3: Run batch inference
    logger.info("\n" + "="*80)
    logger.info("Step 3: Running Batch Inference")
    logger.info("="*80)

    with torch.no_grad():
        # Encode audio to codes
        logger.info("Encoding audio...")
        codes = batch_inf.encode_audio(audio)
        B, K, T = codes.shape
        logger.info(f"✅ Encoded: [{B}, {K}, {T}]")

        # Forward pass
        logger.info("Running forward pass...")
        output = batch_inf.forward_audio(audio, include_text=True)

        logits = output['logits']        # [B, dep_q, T, card]
        text_logits = output['text_logits']  # [B, 1, T, text_card]
        mask = output['mask']            # [B, dep_q, T]
        text_mask = output['text_mask']  # [B, 1, T]

        logger.info(f"✅ Forward pass complete:")
        logger.info(f"   Logits shape: {logits.shape}")
        logger.info(f"   Text logits shape: {text_logits.shape}")

    # Step 4: Generate or Compute Metrics
    logger.info("\n" + "="*80)
    if args.generate_audio:
        logger.info("Step 4: Generating Audio Output")
    else:
        logger.info("Step 4: Computing Metrics")
    logger.info("="*80)

    results = {}

    if args.generate_audio:
        # Generate Moshi conversation response (same length as input)
        logger.info("Generating Moshi conversation response...")
        
        with torch.no_grad():
            # Create LMGen for conversation (like streaming inference does)
            from moshi.models.lm import LMGen
            from moshi.utils.compile import no_cuda_graph
            
            logger.info("Disabling CUDA graphs for TTT compatibility...")
            
            # CRITICAL: Disable CUDA graphs for TTT compatibility
            # TTT layers move tensors to device during forward pass, which is not allowed during CUDA graph capture
            with no_cuda_graph():
                logger.info("Creating LMGen for conversation...")
                lm_gen = LMGen(
                    batch_inf.model,
                    temp=0.8,        # Audio temperature (like streaming)
                    temp_text=0.7,   # Text temperature
                    use_sampling=True,  # Enable sampling for conversation
                    top_k=250,       # Top-k sampling
                    cfg_coef=1.0,    # No CFG for conversation
                )
                
                logger.info("Starting batch conversation inference...")
                logger.info(f"Processing user audio: {T} tokens → Generating Moshi response: {T} tokens")
                
                # Response tokens (same length as input)
                response_tokens_list = []
                
                with lm_gen.streaming(batch_size=B):
                    # Process input audio token-by-token (like streaming conversation)
                    for t in range(T):
                        # Get current user audio token
                        user_audio_token = codes[:, :, t:t+1]  # [B, 8, 1]
                        
                        # Feed to LMGen: User speaks → Moshi responds
                        tokens = lm_gen.step(user_audio_token)
                        
                        if tokens is not None:
                            response_tokens_list.append(tokens)
                            
                        if (t + 1) % 100 == 0:
                            logger.info(f"Processed {t+1}/{T} conversation tokens...")
                
                logger.info(f"Generated {len(response_tokens_list)} response tokens (same length as input)")
                
                if response_tokens_list:
                    # Concatenate response tokens
                    response_tokens = torch.cat(response_tokens_list, dim=-1)  # [B, K+1, T_response]
                    logger.info(f"Response tokens shape: {response_tokens.shape}")
                    
                    # Extract and process Moshi text output (same as streaming)
                    moshi_text_tokens = response_tokens[:, 0, :].cpu()  # [B, T_response] - text is first token
                    
                    # Convert text tokens to readable text (same as streaming inference)
                    logger.info("Processing Moshi text output...")
                    moshi_text_output = []
                    valid_text_tokens = 0
                    
                    for b in range(B):
                        batch_text = []
                        for t in range(moshi_text_tokens.shape[1]):
                            text_token_id = moshi_text_tokens[b, t].item()
                            
                            # Skip padding tokens (same logic as streaming: 0, 3)
                            if text_token_id not in [0, 3]:
                                try:
                                    # Load text tokenizer to decode (same way as streaming inference)
                                    if 'text_tokenizer' not in locals():
                                        from inference.run_inference_with_ttt import CheckpointInfo
                                        checkpoint_info = CheckpointInfo.from_hf_repo(args.hf_repo)
                                        text_tokenizer = checkpoint_info.get_text_tokenizer()
                                    
                                    text = text_tokenizer.id_to_piece(text_token_id)
                                    text = text.replace("▁", " ")  # Convert sentencepiece underscores to spaces
                                    batch_text.append(text)
                                    valid_text_tokens += 1
                                except Exception as e:
                                    logger.warning(f"Could not decode text token {text_token_id}: {e}")
                        
                        batch_text_str = "".join(batch_text)
                        moshi_text_output.append(batch_text_str)
                        
                        if batch_text_str.strip():
                            logger.info(f"📝 Moshi text output (batch {b}): '{batch_text_str.strip()}'")
                        else:
                            logger.info(f"📝 Moshi text output (batch {b}): [No text/silence]")
                    
                    logger.info(f"✅ Processed {valid_text_tokens} valid text tokens")
                    
                    # Extract Moshi audio tokens (skip text token)
                    moshi_audio_codes = response_tokens[:, 1:, :]  # [B, K_audio, T_response]
                    
                    # Decode to Moshi response audio
                    moshi_response_audio = batch_inf.decode_audio(moshi_audio_codes)
                    logger.info(f"Moshi response audio shape: {moshi_response_audio.shape}")
                    
                    # Save Moshi response audio
                    for b in range(B):
                        input_filename = Path(args.input[b]).stem
                        output_file = output_dir / f'{input_filename}_moshi_response.wav'
                        moshi_audio_np = moshi_response_audio[b, 0].cpu().numpy()  # Remove batch and channel dims
                        
                        # Save with torchaudio
                        import torchaudio
                        moshi_audio_tensor = torch.from_numpy(moshi_audio_np).unsqueeze(0)  # Add channel dim back
                        torchaudio.save(str(output_file), moshi_audio_tensor, 24000)
                        
                        logger.info(f"✅ Saved Moshi response audio: {output_file}")
                        
                        # Plot Moshi response audio waveform
                        plot_path = output_dir / f'{input_filename}_moshi_response_waveform.png'
                        plot_waveform(
                            audio_data=moshi_audio_np,
                            sample_rate=24000,
                            output_path=str(plot_path),
                            title=f"Moshi Response Audio Waveform - {input_filename}"
                        )
                        
                        # Plot comparison of user input vs Moshi response
                        user_audio_np = audio[b, 0].cpu().numpy()  # Get original user input for this batch
                        comparison_path = output_dir / f'{input_filename}_conversation_waveform.png'
                        plot_comparison_waveform(
                            input_audio=user_audio_np,
                            generated_audio=moshi_audio_np,
                            sample_rate=24000,
                            output_path=str(comparison_path),
                            title=f"User vs Moshi Conversation - {input_filename}"
                        )
                    
                    # Save Moshi text output to files
                    logger.info("Saving Moshi text output...")
                    text_files = []
                    for b in range(B):
                        input_filename = Path(args.input[b]).stem
                        text_file = output_dir / f'{input_filename}_moshi_text.txt'
                        
                        with open(text_file, 'w', encoding='utf-8') as f:
                            f.write("=== MOSHI TEXT OUTPUT ===\n")
                            f.write(f"Input file: {args.input[b]}\n")
                            f.write(f"Duration: {T * 1920 / 24000:.2f} seconds\n")
                            f.write(f"Response tokens: {len(response_tokens_list)}\n")
                            f.write(f"Valid text tokens: {valid_text_tokens}\n")
                            f.write("=" * 30 + "\n\n")
                            
                            if moshi_text_output[b].strip():
                                f.write(moshi_text_output[b])
                            else:
                                f.write("[Moshi generated no text - audio-only response]")
                            
                            f.write(f"\n\n{'='*30}\n")
                            f.write("Note: This is what Moshi 'said' during the conversation.\n")
                            f.write("Empty text is normal - Moshi often responds with audio-only.\n")
                        
                        text_files.append(f'{input_filename}_moshi_text.txt')
                        logger.info(f"✅ Saved Moshi text: {text_file}")
                    
                    results['generated_files'] = [f'{Path(args.input[b]).stem}_moshi_response.wav' for b in range(B)]
                    results['text_files'] = text_files
                    results['generation_method'] = 'batch_conversation_inference'
                    results['response_tokens'] = len(response_tokens_list)
                    results['input_tokens'] = T
                    results['valid_text_tokens'] = valid_text_tokens
                    results['temperature'] = 0.8
                    results['note'] = 'Moshi conversation response - same length as user input'
                else:
                    logger.warning("No response tokens were generated")
                    results['generated_files'] = []
                    results['generation_method'] = 'conversation_failed'

    if args.compute_perplexity:
        # Parse codebook weights if provided
        codebook_weights = None
        if args.codebook_weights is not None:
            codebook_weights = parse_codebook_weights(
                args.codebook_weights,
                batch_inf.dep_q
            ).to(args.device)
            logger.info(f"Using codebook weights: {codebook_weights.tolist()}")

        # Compute perplexity
        logger.info("Computing perplexity...")
        perplexity_results = batch_inf.evaluate_perplexity(
            audio,
            target_codes=codes,
            codebook_weights=codebook_weights,
        )

        results['perplexity'] = perplexity_results['perplexity']
        results['loss'] = perplexity_results['loss']
        results['per_codebook_perplexity'] = perplexity_results['per_codebook_perplexity']

        logger.info(f"✅ Metrics computed:")
        logger.info(f"   Perplexity: {results['perplexity']:.4f}")
        logger.info(f"   Loss: {results['loss']:.4f}")
        logger.info(f"   Per-codebook perplexity:")
        for q, perp in enumerate(results['per_codebook_perplexity']):
            logger.info(f"      Codebook {q}: {perp:.4f}")

    # Step 5: Save results
    logger.info("\n" + "="*80)
    logger.info("Step 5: Saving Results")
    logger.info("="*80)

    # Save metrics to JSON
    results_file = output_dir / 'results.json'
    results_data = {
        'checkpoint': str(args.checkpoint),
        'hf_repo': args.hf_repo,
        'input_files': args.input,
        'num_sequences': B,
        'sequence_length': T,
        'audio_codebooks': K,
        'metrics': results,
    }

    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)

    logger.info(f"✅ Saved results to {results_file}")

    # Save logits (optional - can be large)
    logits_file = output_dir / 'logits.pt'
    torch.save({
        'logits': logits.cpu(),
        'text_logits': text_logits.cpu(),
        'mask': mask.cpu(),
        'text_mask': text_mask.cpu(),
        'codes': codes.cpu(),
    }, logits_file)

    logger.info(f"✅ Saved logits to {logits_file}")

    # Summary
    logger.info("\n" + "="*80)
    logger.info("Summary")
    logger.info("="*80)
    logger.info(f"✅ Batch inference complete!")
    logger.info(f"   Processed {B} sequences of length {T} tokens")
    logger.info(f"   Results saved to {output_dir}")

    if args.compute_perplexity:
        logger.info(f"   Perplexity: {results['perplexity']:.4f}")

    return results


def main():
    args = parse_args()
    results = run_batch_inference(args)
    return results


if __name__ == "__main__":
    main()
