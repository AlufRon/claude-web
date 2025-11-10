#!/usr/bin/env python3
"""
Attention Pattern Analysis: Comparing RoPE Approaches for Audio TTT

This script compares how different positional encoding approaches affect attention patterns
in TTT processing. We analyze the Q@K^T attention matrices to understand how RoPE choice
impacts the model's ability to attend to different positions in audio sequences.

Comparison Matrix:
1. No RoPE - Vanilla attention without positional encoding
2. 3D RoPE - Current CogVideo approach (spatial + temporal)
3. 1D RoPE - Proposed temporal-only approach
4. Learned PE - Learnable positional embeddings

Key Questions:
- Do spatial dimensions in 3D RoPE create artifacts in attention?
- Does 1D temporal RoPE produce cleaner patterns for audio?
- How do different approaches affect long-range dependencies?
"""

import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List, Optional

# Add paths
sys.path.append('/home/alufr/ttt_tests/moshi-finetune')
sys.path.append('/home/alufr/ttt_tests/moshi')

from moshi_ttt.models.ssm.utils import precompute_freqs_cis_3d, apply_rotary_emb
from moshi_ttt.config import TTTConfig


class AttentionPatternAnalyzer:
    """Analyzes attention patterns with different positional encoding approaches."""
    
    def __init__(self, seq_len: int = 128, d_model: int = 512, num_heads: int = 8):
        self.seq_len = seq_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        print(f"Initializing Attention Pattern Analyzer:")
        print(f"  Sequence Length: {seq_len}")
        print(f"  Model Dimension: {d_model}")
        print(f"  Attention Heads: {num_heads}")
        print(f"  Head Dimension: {self.head_dim}")
        print()
    
    def create_test_audio_sequence(self, batch_size: int = 2) -> torch.Tensor:
        """Create a realistic test audio sequence with structure."""
        # Create audio with some realistic patterns
        # - Low frequency base pattern
        # - Higher frequency details
        # - Some periodic structure
        
        t = torch.linspace(0, 4*np.pi, self.seq_len)
        
        # Base pattern (simulates speech rhythm)
        base_freq = torch.sin(t.unsqueeze(-1) * torch.linspace(0.1, 2.0, self.d_model))
        
        # Add harmonic structure
        harmonics = torch.sin(3 * t.unsqueeze(-1) * torch.linspace(0.1, 2.0, self.d_model)) * 0.3
        
        # Add noise for realism
        noise = torch.randn(self.seq_len, self.d_model) * 0.1
        
        audio_pattern = base_freq + harmonics + noise
        
        # Expand to batch
        audio_sequence = audio_pattern.unsqueeze(0).repeat(batch_size, 1, 1)
        
        return audio_sequence
    
    def project_to_qk(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project input to Q and K tensors."""
        batch_size = x.shape[0]
        
        # Simple linear projections (like in TTT)
        W_q = torch.randn(self.d_model, self.d_model) * 0.02
        W_k = torch.randn(self.d_model, self.d_model) * 0.02
        
        Q = x @ W_q  # [B, seq_len, d_model]
        K = x @ W_k  # [B, seq_len, d_model]
        
        # Reshape to multi-head format
        Q = Q.view(batch_size, self.seq_len, self.num_heads, self.head_dim)
        K = K.view(batch_size, self.seq_len, self.num_heads, self.head_dim)
        
        # L2 normalize (as done in TTT)
        Q = F.normalize(Q, p=2, dim=-1)
        K = F.normalize(K, p=2, dim=-1)
        
        return Q, K
    
    def apply_no_rope(self, Q: torch.Tensor, K: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """No positional encoding - baseline attention."""
        return Q, K
    
    def apply_3d_rope(self, Q: torch.Tensor, K: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply 3D RoPE as currently used in Moshi (from CogVideo)."""
        # Find reasonable 3D mapping for sequence length
        mappings = self._find_3d_mappings(self.seq_len)
        if not mappings:
            print(f"Warning: No 3D mapping found for seq_len={self.seq_len}")
            return Q, K
            
        h, w, f = mappings[0]  # Use first mapping
        
        try:
            freqs_cis = precompute_freqs_cis_3d(self.head_dim, h, w, f, 10000.0)
            Q_rope, K_rope = apply_rotary_emb(Q, K, freqs_cis=freqs_cis)
            return Q_rope, K_rope
        except Exception as e:
            print(f"Warning: 3D RoPE failed: {e}")
            return Q, K
    
    def apply_1d_rope(self, Q: torch.Tensor, K: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply 1D temporal RoPE (our proposed alternative)."""
        # Simple 1D RoPE implementation
        freqs = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        positions = torch.arange(self.seq_len, dtype=torch.float)
        
        # Create rotation matrices for each position
        angles = positions.unsqueeze(-1) * freqs.unsqueeze(0)  # [seq_len, head_dim//2]
        cos_angles = torch.cos(angles)
        sin_angles = torch.sin(angles)
        
        # Apply rotation to Q and K
        Q_rope = self._apply_1d_rotation(Q, cos_angles, sin_angles)
        K_rope = self._apply_1d_rotation(K, cos_angles, sin_angles)
        
        return Q_rope, K_rope
    
    def apply_learned_pe(self, Q: torch.Tensor, K: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply learned positional embeddings."""
        # Create learned position embeddings
        pos_emb = torch.randn(self.seq_len, self.head_dim) * 0.1
        
        # Add to Q and K
        Q_pe = Q + pos_emb.unsqueeze(0).unsqueeze(2)
        K_pe = K + pos_emb.unsqueeze(0).unsqueeze(2)
        
        return Q_pe, K_pe
    
    def _apply_1d_rotation(self, x: torch.Tensor, cos_angles: torch.Tensor, sin_angles: torch.Tensor) -> torch.Tensor:
        """Apply 1D rotation to tensor."""
        batch_size, seq_len, num_heads, head_dim = x.shape
        
        # Split into even/odd dimensions for rotation
        x_even = x[..., 0::2]  # [B, seq_len, num_heads, head_dim//2]
        x_odd = x[..., 1::2]   # [B, seq_len, num_heads, head_dim//2]
        
        # Apply rotation
        cos = cos_angles.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, head_dim//2]
        sin = sin_angles.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, head_dim//2]
        
        x_even_rot = x_even * cos - x_odd * sin
        x_odd_rot = x_even * sin + x_odd * cos
        
        # Interleave back
        x_rot = torch.empty_like(x)
        x_rot[..., 0::2] = x_even_rot
        x_rot[..., 1::2] = x_odd_rot
        
        return x_rot
    
    def _find_3d_mappings(self, seq_len: int) -> List[Tuple[int, int, int]]:
        """Find possible (h, w, f) factorizations."""
        mappings = []
        for f in range(1, int(seq_len**0.5) + 1):
            if seq_len % f == 0:
                remaining = seq_len // f
                for h in range(1, int(remaining**0.5) + 1):
                    if remaining % h == 0:
                        w = remaining // h
                        mappings.append((h, w, f))
        
        # Sort by spatial balance
        mappings.sort(key=lambda x: abs(x[0] - x[1]))
        return mappings
    
    def compute_attention_matrix(self, Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """Compute attention matrix Q@K^T."""
        # Q: [B, seq_len, num_heads, head_dim]
        # K: [B, seq_len, num_heads, head_dim]
        
        # Transpose for attention computation
        Q_t = Q.transpose(1, 2)  # [B, num_heads, seq_len, head_dim]
        K_t = K.transpose(1, 2)  # [B, num_heads, seq_len, head_dim]
        
        # Compute attention scores
        attn_scores = torch.matmul(Q_t, K_t.transpose(-2, -1))  # [B, num_heads, seq_len, seq_len]
        
        # Scale by sqrt(head_dim)
        attn_scores = attn_scores / (self.head_dim ** 0.5)
        
        return attn_scores
    
    def analyze_attention_patterns(self) -> Dict[str, torch.Tensor]:
        """Compare attention patterns across different RoPE approaches."""
        print("=" * 60)
        print("ANALYZING ATTENTION PATTERNS")
        print("=" * 60)
        
        # Create test sequence
        audio_seq = self.create_test_audio_sequence()
        Q, K = self.project_to_qk(audio_seq)
        
        print(f"Input shapes: Q={Q.shape}, K={K.shape}")
        print()
        
        results = {}
        
        # Test each approach
        approaches = {
            'no_rope': self.apply_no_rope,
            '3d_rope': self.apply_3d_rope,
            '1d_rope': self.apply_1d_rope,
            'learned_pe': self.apply_learned_pe
        }
        
        for name, apply_fn in approaches.items():
            print(f"Testing {name}...")
            
            try:
                Q_mod, K_mod = apply_fn(Q, K)
                attn_matrix = self.compute_attention_matrix(Q_mod, K_mod)
                
                results[name] = {
                    'Q': Q_mod,
                    'K': K_mod,
                    'attention_matrix': attn_matrix,
                    'attention_weights': F.softmax(attn_matrix, dim=-1)
                }
                
                # Compute statistics
                attn_weights = results[name]['attention_weights']
                entropy = self._compute_attention_entropy(attn_weights)
                locality = self._compute_attention_locality(attn_weights)
                
                print(f"  Attention entropy: {entropy:.4f}")
                print(f"  Attention locality: {locality:.4f}")
                
            except Exception as e:
                print(f"  Error: {e}")
                results[name] = {'error': str(e)}
        
        print()
        return results
    
    def _compute_attention_entropy(self, attn_weights: torch.Tensor) -> float:
        """Compute average attention entropy (measure of focus)."""
        # attn_weights: [B, num_heads, seq_len, seq_len]
        log_attn = torch.log(attn_weights + 1e-8)
        entropy = -(attn_weights * log_attn).sum(dim=-1)  # [B, num_heads, seq_len]
        return entropy.mean().item()
    
    def _compute_attention_locality(self, attn_weights: torch.Tensor) -> float:
        """Compute attention locality (how much attention focuses on nearby positions)."""
        batch_size, num_heads, seq_len, _ = attn_weights.shape
        
        # Create distance matrix
        positions = torch.arange(seq_len, dtype=torch.float)
        distances = torch.abs(positions.unsqueeze(1) - positions.unsqueeze(0))
        
        # Weight attention by distance
        weighted_distances = attn_weights * distances.unsqueeze(0).unsqueeze(0)
        avg_distance = weighted_distances.sum(dim=-1).mean()  # Average attended distance
        
        # Normalize by sequence length
        return (avg_distance / seq_len).item()
    
    def visualize_attention_patterns(self, results: Dict) -> None:
        """Create visualizations comparing attention patterns."""
        print("=" * 60)
        print("GENERATING ATTENTION PATTERN VISUALIZATIONS")
        print("=" * 60)
        
        # Filter successful results
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        
        if not valid_results:
            print("No valid results to visualize!")
            return
        
        num_approaches = len(valid_results)
        fig, axes = plt.subplots(2, num_approaches, figsize=(4*num_approaches, 8))
        if num_approaches == 1:
            axes = axes.reshape(2, 1)
        
        fig.suptitle('Attention Pattern Comparison: RoPE Approaches for Audio', fontsize=14)
        
        for i, (name, data) in enumerate(valid_results.items()):
            attn_weights = data['attention_weights']
            
            # Take first batch, first head for visualization
            attn_map = attn_weights[0, 0].detach().numpy()
            
            # 1. Full attention matrix
            im1 = axes[0, i].imshow(attn_map, cmap='Blues', aspect='auto')
            axes[0, i].set_title(f'{name.replace("_", " ").title()}\nAttention Matrix')
            axes[0, i].set_xlabel('Key Position')
            axes[0, i].set_ylabel('Query Position')
            plt.colorbar(im1, ax=axes[0, i])
            
            # 2. Attention pattern analysis
            # Show how attention spreads from a few query positions
            query_positions = [self.seq_len//4, self.seq_len//2, 3*self.seq_len//4]
            
            for j, q_pos in enumerate(query_positions):
                attn_pattern = attn_map[q_pos]
                axes[1, i].plot(attn_pattern, alpha=0.7, label=f'Query {q_pos}')
            
            axes[1, i].set_title(f'{name.replace("_", " ").title()}\nAttention Distribution')
            axes[1, i].set_xlabel('Key Position')
            axes[1, i].set_ylabel('Attention Weight')
            axes[1, i].legend()
            axes[1, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save visualization
        output_path = '/home/alufr/ttt_tests/moshi-finetune/attention_pattern_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {output_path}")
        plt.close()
    
    def compare_attention_statistics(self, results: Dict) -> None:
        """Compare statistical properties of attention patterns."""
        print("=" * 60)
        print("ATTENTION PATTERN STATISTICS COMPARISON")
        print("=" * 60)
        
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        
        statistics = {}
        
        for name, data in valid_results.items():
            attn_weights = data['attention_weights']
            
            # Compute various statistics
            stats = {
                'entropy': self._compute_attention_entropy(attn_weights),
                'locality': self._compute_attention_locality(attn_weights),
                'max_attention': attn_weights.max().item(),
                'min_attention': attn_weights.min().item(),
                'std_attention': attn_weights.std().item(),
                'sparsity': (attn_weights < 0.01).float().mean().item()
            }
            
            statistics[name] = stats
        
        # Print comparison table
        print(f"{'Approach':<12} {'Entropy':<8} {'Locality':<9} {'Max':<6} {'Min':<8} {'Std':<8} {'Sparsity':<8}")
        print("-" * 65)
        
        for name, stats in statistics.items():
            print(f"{name:<12} {stats['entropy']:<8.4f} {stats['locality']:<9.4f} "
                  f"{stats['max_attention']:<6.4f} {stats['min_attention']:<8.6f} "
                  f"{stats['std_attention']:<8.4f} {stats['sparsity']:<8.4f}")
        
        print()
        
        # Analyze differences
        print("KEY OBSERVATIONS:")
        
        if 'no_rope' in statistics and '3d_rope' in statistics:
            entropy_diff = statistics['3d_rope']['entropy'] - statistics['no_rope']['entropy']
            locality_diff = statistics['3d_rope']['locality'] - statistics['no_rope']['locality']
            
            print(f"  3D RoPE vs No RoPE:")
            print(f"    Entropy difference: {entropy_diff:+.4f} ({'more focused' if entropy_diff < 0 else 'more spread'})")
            print(f"    Locality difference: {locality_diff:+.4f} ({'more local' if locality_diff < 0 else 'more global'})")
        
        if '1d_rope' in statistics and '3d_rope' in statistics:
            entropy_diff = statistics['1d_rope']['entropy'] - statistics['3d_rope']['entropy']
            locality_diff = statistics['1d_rope']['locality'] - statistics['3d_rope']['locality']
            
            print(f"  1D RoPE vs 3D RoPE:")
            print(f"    Entropy difference: {entropy_diff:+.4f} ({'more focused' if entropy_diff < 0 else 'more spread'})")
            print(f"    Locality difference: {locality_diff:+.4f} ({'more local' if locality_diff < 0 else 'more global'})")
        
        print()
        
        return statistics


def main():
    """Run complete attention pattern analysis."""
    print("🔍 ATTENTION PATTERN ANALYSIS")
    print("Comparing RoPE Approaches for Audio TTT Processing")
    print()
    
    # Test with different sequence lengths
    seq_lengths = [64, 128, 256]
    
    all_results = {}
    
    for seq_len in seq_lengths:
        print(f"Testing sequence length: {seq_len}")
        print("=" * 40)
        
        analyzer = AttentionPatternAnalyzer(seq_len=seq_len, d_model=512, num_heads=8)
        
        # Analyze attention patterns
        results = analyzer.analyze_attention_patterns()
        
        # Generate statistics
        statistics = analyzer.compare_attention_statistics(results)
        
        # Create visualizations (only for first sequence length to avoid clutter)
        if seq_len == seq_lengths[0]:
            analyzer.visualize_attention_patterns(results)
        
        all_results[seq_len] = {
            'results': results,
            'statistics': statistics
        }
        
        print()
    
    # Generate summary
    print("=" * 60)
    print("ATTENTION PATTERN ANALYSIS SUMMARY")
    print("=" * 60)
    
    print("FINDINGS ACROSS SEQUENCE LENGTHS:")
    
    for seq_len in seq_lengths:
        stats = all_results[seq_len]['statistics']
        print(f"\nSequence Length {seq_len}:")
        
        if 'no_rope' in stats and '3d_rope' in stats:
            no_rope_entropy = stats['no_rope']['entropy']
            rope_3d_entropy = stats['3d_rope']['entropy']
            print(f"  Entropy: No RoPE={no_rope_entropy:.3f}, 3D RoPE={rope_3d_entropy:.3f}")
            
        if '1d_rope' in stats:
            rope_1d_entropy = stats['1d_rope']['entropy']
            print(f"  1D RoPE entropy: {rope_1d_entropy:.3f}")
    
    print("\nRECOMMENDATIONS:")
    print("  → Check attention_pattern_analysis.png for visual comparison")
    print("  → Lower entropy = more focused attention")
    print("  → Lower locality = better long-range modeling")
    print("  → Next: Run TTT impact analysis to see effect on weight updates")
    print()
    
    print("✅ Attention pattern analysis complete!")
    
    return all_results


if __name__ == "__main__":
    results = main()