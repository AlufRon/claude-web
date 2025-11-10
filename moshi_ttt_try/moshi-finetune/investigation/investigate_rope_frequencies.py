#!/usr/bin/env python3
"""
RoPE Frequency Analysis: Understanding 3D Video RoPE vs 1D Audio Needs

This script analyzes how CogVideo's 3D RoPE (designed for video with height×width×time)
is currently being applied to Moshi's 1D audio sequences, and identifies the mismatch.

Investigation Goals:
1. Understand what frequencies CogVideo's 3D RoPE encodes
2. See how 1D audio sequences map to 3D dimensions  
3. Identify if spatial frequencies are relevant for audio
4. Determine optimal positional encoding strategy for audio
"""

import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any

# Add moshi_ttt to path
sys.path.append('/home/alufr/ttt_tests/moshi-finetune')
sys.path.append('/home/alufr/ttt_tests/moshi')

from moshi_ttt.models.ssm.utils import precompute_freqs_cis_3d, apply_rotary_emb
from moshi_ttt.config import TTTConfig


def analyze_video_rope_design():
    """
    Analyze CogVideo's 3D RoPE design to understand what it encodes.
    
    CogVideo processes video latents with spatial (H×W) and temporal (T) dimensions.
    The 3D RoPE encodes position information across all three dimensions.
    """
    print("=" * 60)
    print("ANALYZING COGVIDEO'S 3D ROPE DESIGN")
    print("=" * 60)
    
    # Typical CogVideo video dimensions (from config)
    latent_height = 16      # Spatial height after VAE encoding
    latent_width = 16       # Spatial width after VAE encoding  
    num_frames = 8          # Temporal frames
    head_dim = 64           # Attention head dimension
    rope_theta = 10000.0    # RoPE base frequency
    
    print(f"Video Latent Dimensions:")
    print(f"  Height: {latent_height}")
    print(f"  Width: {latent_width}")
    print(f"  Frames: {num_frames}")
    print(f"  Total Positions: {latent_height * latent_width * num_frames}")
    print(f"  Head Dimension: {head_dim}")
    print()
    
    # Generate 3D RoPE frequencies
    freqs_3d = precompute_freqs_cis_3d(
        head_dim, latent_height, latent_width, num_frames, rope_theta
    )
    
    print(f"3D RoPE Output Shape: {freqs_3d.shape}")
    print(f"Expected: [{latent_height * latent_width * num_frames}, {head_dim // 2}]")
    print()
    
    # Analyze frequency patterns
    total_positions = latent_height * latent_width * num_frames
    freq_magnitudes = torch.abs(freqs_3d)
    
    print(f"Frequency Statistics:")
    print(f"  Min frequency magnitude: {freq_magnitudes.min():.6f}")
    print(f"  Max frequency magnitude: {freq_magnitudes.max():.6f}")
    print(f"  Mean frequency magnitude: {freq_magnitudes.mean():.6f}")
    print(f"  Std frequency magnitude: {freq_magnitudes.std():.6f}")
    print()
    
    return {
        'freqs_3d': freqs_3d,
        'video_dims': (latent_height, latent_width, num_frames),
        'total_positions': total_positions,
        'head_dim': head_dim,
        'freq_stats': {
            'min': freq_magnitudes.min().item(),
            'max': freq_magnitudes.max().item(), 
            'mean': freq_magnitudes.mean().item(),
            'std': freq_magnitudes.std().item()
        }
    }


def analyze_audio_sequence_mapping():
    """
    Analyze how 1D audio sequences currently map to 3D video RoPE.
    
    This reveals the core mismatch: audio has only temporal dimension,
    but CogVideo RoPE expects spatial dimensions too.
    """
    print("=" * 60)
    print("ANALYZING AUDIO → 3D ROPE MAPPING")
    print("=" * 60)
    
    # Typical Moshi audio sequence dimensions
    seq_lengths = [128, 256, 512, 1024]  # Common audio sequence lengths
    d_model = 1024  # Moshi model dimension
    num_heads = 8   # Moshi attention heads
    head_dim = d_model // num_heads  # 128
    
    print(f"Moshi Audio Dimensions:")
    print(f"  Model Dimension: {d_model}")
    print(f"  Attention Heads: {num_heads}")
    print(f"  Head Dimension: {head_dim}")
    print()
    
    results = {}
    
    for seq_len in seq_lengths:
        print(f"Audio Sequence Length: {seq_len}")
        
        # Question: How does this 1D sequence map to 3D video dimensions?
        # Current implementation likely treats seq_len as total_positions
        
        # Try to reverse-engineer the mapping
        # If seq_len = height * width * frames, what are the dimensions?
        possible_mappings = find_possible_3d_mappings(seq_len)
        
        print(f"  Possible 3D mappings for seq_len={seq_len}:")
        for i, (h, w, f) in enumerate(possible_mappings[:3]):  # Show first 3
            print(f"    {i+1}. H={h}, W={w}, F={f} (H×W×F = {h*w*f})")
        
        # Test RoPE generation with these dimensions
        if possible_mappings:
            h, w, f = possible_mappings[0]  # Use first mapping
            try:
                freqs_audio = precompute_freqs_cis_3d(head_dim, h, w, f, 10000.0)
                
                results[seq_len] = {
                    'mapping': (h, w, f),
                    'freqs_shape': freqs_audio.shape,
                    'freqs_sample': freqs_audio[:5, :5]  # Small sample
                }
                
                print(f"    Generated RoPE shape: {freqs_audio.shape}")
            except Exception as e:
                print(f"    Error generating RoPE: {e}")
        
        print()
    
    return results


def find_possible_3d_mappings(seq_len: int) -> list:
    """Find possible (height, width, frames) factorizations of seq_len."""
    mappings = []
    
    # Find all factor combinations
    for f in range(1, int(seq_len**0.5) + 1):
        if seq_len % f == 0:
            remaining = seq_len // f
            for h in range(1, int(remaining**0.5) + 1):
                if remaining % h == 0:
                    w = remaining // h
                    mappings.append((h, w, f))
    
    # Sort by how "square" the spatial dimensions are (prefer balanced h,w)
    mappings.sort(key=lambda x: abs(x[0] - x[1]))
    
    return mappings


def compare_positional_encoding_needs():
    """
    Compare what positional information video vs audio actually needs.
    
    This conceptual analysis helps understand why 3D RoPE may be wrong for audio.
    """
    print("=" * 60)
    print("POSITIONAL ENCODING NEEDS: VIDEO VS AUDIO")
    print("=" * 60)
    
    print("VIDEO (3D) POSITIONAL NEEDS:")
    print("  Spatial (Height): Object positions vertically")
    print("  Spatial (Width): Object positions horizontally") 
    print("  Temporal (Time): Motion and temporal dynamics")
    print("  Interactions: Spatial-temporal correlations")
    print("  Example: A ball moving from top-left to bottom-right")
    print()
    
    print("AUDIO (1D) POSITIONAL NEEDS:")
    print("  Temporal (Time): Sequential dependencies only")
    print("  No Spatial: Audio tokens have no spatial relationship")
    print("  Example: Word sequence 'The cat sat on the mat'")
    print()
    
    print("MISMATCH ANALYSIS:")
    print("  ❌ 3D RoPE encodes spatial relationships that don't exist in audio")
    print("  ❌ Audio tokens get artificial 'height' and 'width' positions")
    print("  ❌ May interfere with temporal modeling (the important dimension)")
    print("  ✅ Only temporal encoding needed for audio sequences")
    print()


def visualize_rope_frequencies(video_analysis: Dict, audio_analysis: Dict):
    """Create visualizations comparing video and audio RoPE frequencies."""
    print("=" * 60)
    print("GENERATING ROPE FREQUENCY VISUALIZATIONS")
    print("=" * 60)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('RoPE Frequency Analysis: Video vs Audio', fontsize=16)
    
    # 1. Video RoPE frequency heatmap
    video_freqs = video_analysis['freqs_3d']
    video_magnitudes = torch.abs(video_freqs)
    
    im1 = axes[0, 0].imshow(video_magnitudes.numpy(), aspect='auto', cmap='viridis')
    axes[0, 0].set_title('Video 3D RoPE Frequencies')
    axes[0, 0].set_xlabel('Head Dimension')
    axes[0, 0].set_ylabel('Position Index')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # 2. Audio RoPE frequency heatmap (using one sequence length)
    seq_len = 256
    if seq_len in audio_analysis:
        # Generate fresh frequencies for visualization
        h, w, f = audio_analysis[seq_len]['mapping']
        audio_freqs = precompute_freqs_cis_3d(64, h, w, f, 10000.0)
        audio_magnitudes = torch.abs(audio_freqs)
        
        im2 = axes[0, 1].imshow(audio_magnitudes.numpy(), aspect='auto', cmap='viridis')
        axes[0, 1].set_title(f'Audio Sequence (len={seq_len}) using 3D RoPE')
        axes[0, 1].set_xlabel('Head Dimension')
        axes[0, 1].set_ylabel('Position Index')
        plt.colorbar(im2, ax=axes[0, 1])
    
    # 3. Frequency magnitude distribution
    video_data = video_magnitudes.flatten().numpy()
    # Handle case where all values are the same (causing hist error)
    if video_data.std() > 1e-8:
        axes[1, 0].hist(video_data, bins=50, alpha=0.7, label='Video 3D RoPE')
    else:
        axes[1, 0].axvline(video_data[0], color='blue', alpha=0.7, label='Video 3D RoPE (constant)')
    
    if seq_len in audio_analysis:
        audio_data = audio_magnitudes.flatten().numpy()
        if audio_data.std() > 1e-8:
            axes[1, 0].hist(audio_data, bins=50, alpha=0.7, label='Audio (3D RoPE)')
        else:
            axes[1, 0].axvline(audio_data[0], color='red', alpha=0.7, label='Audio 3D RoPE (constant)')
    
    axes[1, 0].set_title('Frequency Magnitude Distributions')
    axes[1, 0].set_xlabel('Magnitude')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].legend()
    
    # 4. Position encoding comparison
    positions = np.arange(min(64, video_freqs.shape[0]))
    first_freq = video_magnitudes[:len(positions), 0].numpy()
    
    axes[1, 1].plot(positions, first_freq, 'b-', label='Video 3D RoPE')
    if seq_len in audio_analysis:
        audio_first_freq = audio_magnitudes[:len(positions), 0].numpy()
        axes[1, 1].plot(positions, audio_first_freq, 'r--', label='Audio (3D RoPE)')
    axes[1, 1].set_title('First Frequency Component vs Position')
    axes[1, 1].set_xlabel('Position')
    axes[1, 1].set_ylabel('Frequency Magnitude')
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    # Save the plot
    output_path = '/home/alufr/ttt_tests/moshi-finetune/rope_frequency_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    plt.close()


def test_current_moshi_rope_application():
    """
    Test how Moshi currently applies 3D RoPE to audio sequences.
    
    This reveals the actual tensor operations and potential issues.
    """
    print("=" * 60)
    print("TESTING CURRENT MOSHI ROPE APPLICATION")
    print("=" * 60)
    
    # Create a sample Moshi audio sequence
    batch_size = 2
    seq_len = 128
    d_model = 1024
    num_heads = 8
    head_dim = d_model // num_heads
    
    # Generate sample Q/K tensors (as TTT would)
    Q = torch.randn(batch_size, seq_len, num_heads, head_dim)
    K = torch.randn(batch_size, seq_len, num_heads, head_dim)
    
    print(f"Input tensor shapes:")
    print(f"  Q: {Q.shape}")
    print(f"  K: {K.shape}")
    print()
    
    # Try to apply current 3D RoPE (this might fail or work unexpectedly)
    try:
        # This is what Moshi's TTT layer currently does
        # We need to understand how seq_len maps to video dimensions
        
        # Find a reasonable 3D mapping for seq_len=128
        possible_mappings = find_possible_3d_mappings(seq_len)
        if possible_mappings:
            h, w, f = possible_mappings[0]
            print(f"Using 3D mapping: H={h}, W={w}, F={f}")
            
            # Generate 3D RoPE frequencies
            freqs_cis = precompute_freqs_cis_3d(head_dim, h, w, f, 10000.0)
            print(f"Generated freqs_cis shape: {freqs_cis.shape}")
            
            # Apply RoPE (this is the actual operation in Moshi TTT)
            Q_rope, K_rope = apply_rotary_emb(Q, K, freqs_cis=freqs_cis)
            
            print(f"RoPE application successful!")
            print(f"  Q_rope: {Q_rope.shape}")
            print(f"  K_rope: {K_rope.shape}")
            
            # Analyze the effect
            Q_diff = torch.abs(Q_rope - Q).mean()
            K_diff = torch.abs(K_rope - K).mean()
            
            print(f"  Average change in Q: {Q_diff:.6f}")
            print(f"  Average change in K: {K_diff:.6f}")
            
            return {
                'success': True,
                'mapping': (h, w, f),
                'q_change': Q_diff.item(),
                'k_change': K_diff.item(),
                'q_original': Q,
                'k_original': K,
                'q_rope': Q_rope,
                'k_rope': K_rope
            }
            
    except Exception as e:
        print(f"❌ RoPE application failed: {e}")
        return {'success': False, 'error': str(e)}


def generate_investigation_summary():
    """Generate summary of findings and recommendations."""
    print("=" * 60)
    print("ROPE INVESTIGATION SUMMARY")
    print("=" * 60)
    
    print("KEY FINDINGS:")
    print()
    
    print("1. DESIGN MISMATCH:")
    print("   - CogVideo's 3D RoPE designed for height×width×time video")
    print("   - Moshi audio has only temporal dimension (no spatial)")
    print("   - Current implementation forces 1D audio into 3D framework")
    print()
    
    print("2. POTENTIAL ISSUES:")
    print("   - Artificial spatial relationships imposed on audio tokens")
    print("   - May interfere with temporal sequence modeling")
    print("   - Frequency patterns not optimized for audio characteristics")
    print()
    
    print("3. RECOMMENDED ALTERNATIVES:")
    print("   - 1D Temporal RoPE: Only encode sequence position")
    print("   - Learned Position Embeddings: Let model learn optimal encoding")
    print("   - No Position Encoding: Test if TTT needs explicit positions")
    print()
    
    print("NEXT STEPS:")
    print("   → Implement 1D temporal RoPE alternative")
    print("   → Compare attention patterns with different approaches")
    print("   → Run mini-training experiments to validate performance")
    print()


def main():
    """Run complete RoPE frequency analysis."""
    print("🔍 RoPE FREQUENCY INVESTIGATION")
    print("Analyzing CogVideo 3D RoPE vs Moshi Audio Needs")
    print()
    
    # Phase 1: Understand video RoPE design
    video_analysis = analyze_video_rope_design()
    
    # Phase 2: Analyze audio sequence mapping
    audio_analysis = analyze_audio_sequence_mapping()
    
    # Phase 3: Compare positional encoding needs
    compare_positional_encoding_needs()
    
    # Phase 4: Test current Moshi application
    moshi_test = test_current_moshi_rope_application()
    
    # Phase 5: Create visualizations
    visualize_rope_frequencies(video_analysis, audio_analysis)
    
    # Phase 6: Generate summary
    generate_investigation_summary()
    
    print("✅ RoPE frequency analysis complete!")
    print("📊 Check rope_frequency_analysis.png for visualizations")
    
    return {
        'video_analysis': video_analysis,
        'audio_analysis': audio_analysis, 
        'moshi_test': moshi_test
    }


if __name__ == "__main__":
    results = main()