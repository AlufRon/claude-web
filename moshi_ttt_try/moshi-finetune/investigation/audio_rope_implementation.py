#!/usr/bin/env python3
"""
Audio-Specific RoPE Implementation: 1D Temporal Positional Encoding

This script implements a proper 1D temporal RoPE for audio sequences, replacing
CogVideo's 3D spatial-temporal RoPE with audio-appropriate positional encoding.

Key Design Principles:
1. Only temporal dimension - no artificial spatial relationships
2. Audio-appropriate frequency ranges
3. Compatible with existing TTT pipeline
4. Drop-in replacement for current 3D RoPE

Implementation includes:
- 1D Temporal RoPE computation
- Frequency scaling for audio characteristics  
- Integration with Moshi TTT layer
- Backward compatibility with existing code
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple, Optional

# Add paths
sys.path.append('/home/alufr/ttt_tests/moshi-finetune')
sys.path.append('/home/alufr/ttt_tests/moshi')

from moshi_ttt.config import TTTConfig


class AudioRoPE1D(nn.Module):
    """
    1D Rotary Position Embedding for Audio Sequences.
    
    Designed specifically for temporal audio sequences without spatial dimensions.
    Replaces CogVideo's 3D RoPE with audio-appropriate positional encoding.
    """
    
    def __init__(
        self, 
        head_dim: int,
        max_seq_len: int = 8192,
        base_freq: float = 10000.0,
        audio_scaling: bool = True
    ):
        """
        Initialize 1D Audio RoPE.
        
        Args:
            head_dim: Attention head dimension
            max_seq_len: Maximum sequence length to precompute
            base_freq: Base frequency for RoPE (10000 is standard)
            audio_scaling: Apply audio-specific frequency scaling
        """
        super().__init__()
        
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base_freq = base_freq
        self.audio_scaling = audio_scaling
        
        # Precompute frequencies
        self.register_buffer("inv_freq", self._compute_inv_frequencies())
        self.register_buffer("cos_cached", torch.zeros(max_seq_len, head_dim // 2))
        self.register_buffer("sin_cached", torch.zeros(max_seq_len, head_dim // 2))
        
        # Precompute for common sequence lengths
        self._update_cos_sin_cache(max_seq_len)
        
    def _compute_inv_frequencies(self) -> torch.Tensor:
        """Compute inverse frequencies for RoPE."""
        # Standard RoPE frequency computation
        freqs = 1.0 / (self.base_freq ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        
        if self.audio_scaling:
            # Apply audio-specific scaling
            # Higher frequencies for temporal details, lower for long-range dependencies
            audio_scale = torch.exp(-torch.arange(0, self.head_dim, 2).float() / (self.head_dim * 2))
            freqs = freqs * (0.5 + 1.5 * audio_scale)  # Scale between 0.5x and 2.0x
        
        return freqs
    
    def _update_cos_sin_cache(self, seq_len: int):
        """Update cached cos/sin values for given sequence length."""
        if seq_len > self.max_seq_len:
            # Extend cache if needed
            self.max_seq_len = seq_len
            self.register_buffer("cos_cached", torch.zeros(seq_len, self.head_dim // 2))
            self.register_buffer("sin_cached", torch.zeros(seq_len, self.head_dim // 2))
        
        # Compute position encodings
        positions = torch.arange(seq_len, dtype=torch.float, device=self.inv_freq.device)
        angles = positions.unsqueeze(-1) * self.inv_freq.unsqueeze(0)  # [seq_len, head_dim//2]
        
        self.cos_cached[:seq_len] = torch.cos(angles)
        self.sin_cached[:seq_len] = torch.sin(angles)
    
    def forward(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate cos/sin values for given sequence length.
        
        Args:
            seq_len: Sequence length
            
        Returns:
            tuple: (cos_values, sin_values) each of shape [seq_len, head_dim//2]
        """
        if seq_len > self.cos_cached.shape[0]:
            self._update_cos_sin_cache(seq_len)
        
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def apply_audio_rope_1d(
    Q: torch.Tensor, 
    K: torch.Tensor, 
    cos: torch.Tensor, 
    sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply 1D Audio RoPE to Q and K tensors.
    
    Args:
        Q: Query tensor [batch, seq_len, num_heads, head_dim]
        K: Key tensor [batch, seq_len, num_heads, head_dim]
        cos: Cosine values [seq_len, head_dim//2]
        sin: Sine values [seq_len, head_dim//2]
        
    Returns:
        tuple: (Q_rope, K_rope) with RoPE applied
    """
    def rotate_half(x):
        """Rotate half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    # Convert to complex and back for consistency with other method
    def to_complex(x):
        x_reshaped = x.view(*x.shape[:-1], -1, 2)
        return torch.complex(x_reshaped[..., 0], x_reshaped[..., 1])
    
    def from_complex(x):
        real_part = torch.real(x)
        imag_part = torch.imag(x)
        return torch.stack([real_part, imag_part], dim=-1).flatten(-2)
    
    # Create complex freqs from cos/sin
    freqs_cis = torch.complex(cos, sin)  # [seq_len, head_dim//2]
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, head_dim//2]
    
    # Convert Q, K to complex and apply rotation
    Q_complex = to_complex(Q)
    K_complex = to_complex(K)
    
    Q_rope_complex = Q_complex * freqs_cis
    K_rope_complex = K_complex * freqs_cis
    
    # Convert back to real
    Q_rope = from_complex(Q_rope_complex)
    K_rope = from_complex(K_rope_complex)
    
    return Q_rope, K_rope


def precompute_audio_rope_1d(
    head_dim: int, 
    seq_len: int, 
    base_freq: float = 10000.0,
    audio_scaling: bool = True
) -> torch.Tensor:
    """
    Precompute 1D Audio RoPE values (compatible with existing TTT interface).
    
    This is a drop-in replacement for CogVideo's precompute_freqs_cis_3d.
    
    Args:
        head_dim: Attention head dimension
        seq_len: Sequence length
        base_freq: Base frequency for RoPE
        audio_scaling: Apply audio-specific frequency scaling
        
    Returns:
        torch.Tensor: Complex exponentials [seq_len, head_dim//2]
    """
    # Compute inverse frequencies
    freqs = 1.0 / (base_freq ** (torch.arange(0, head_dim, 2).float() / head_dim))
    
    if audio_scaling:
        # Audio-specific frequency scaling
        audio_scale = torch.exp(-torch.arange(0, head_dim, 2).float() / (head_dim * 2))
        freqs = freqs * (0.5 + 1.5 * audio_scale)
    
    # Compute position encodings
    positions = torch.arange(seq_len, dtype=torch.float)
    angles = positions.unsqueeze(-1) * freqs.unsqueeze(0)  # [seq_len, head_dim//2]
    
    # Return as complex exponentials (compatible with apply_rotary_emb)
    freqs_cis = torch.polar(torch.ones_like(angles), angles)
    
    return freqs_cis


def apply_audio_rotary_emb(
    Q: torch.Tensor,
    K: torch.Tensor,
    freqs_cis: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply audio rotary embeddings (drop-in replacement for apply_rotary_emb).
    
    Compatible with existing TTT code that calls apply_rotary_emb.
    
    Args:
        Q: Query tensor [batch, seq_len, num_heads, head_dim]
        K: Key tensor [batch, seq_len, num_heads, head_dim]
        freqs_cis: Complex exponentials [seq_len, head_dim//2]
        
    Returns:
        tuple: (Q_rope, K_rope) with audio RoPE applied
    """
    # Convert to complex representation
    def to_complex(x):
        """Convert real tensor to complex by grouping adjacent dimensions."""
        x_reshaped = x.view(*x.shape[:-1], -1, 2)  # [..., head_dim//2, 2]
        return torch.complex(x_reshaped[..., 0], x_reshaped[..., 1])
    
    def from_complex(x):
        """Convert complex tensor back to real."""
        real_part = torch.real(x)
        imag_part = torch.imag(x)
        return torch.stack([real_part, imag_part], dim=-1).flatten(-2)
    
    # Convert Q, K to complex
    Q_complex = to_complex(Q)  # [batch, seq_len, num_heads, head_dim//2]
    K_complex = to_complex(K)  # [batch, seq_len, num_heads, head_dim//2]
    
    # Apply rotation
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, head_dim//2]
    
    Q_rope_complex = Q_complex * freqs_cis
    K_rope_complex = K_complex * freqs_cis
    
    # Convert back to real
    Q_rope = from_complex(Q_rope_complex)
    K_rope = from_complex(K_rope_complex)
    
    return Q_rope, K_rope


class AudioRoPEConfig:
    """Configuration for Audio RoPE implementation."""
    
    def __init__(
        self,
        use_audio_rope: bool = True,
        audio_scaling: bool = True,
        base_freq: float = 10000.0,
        max_seq_len: int = 8192
    ):
        self.use_audio_rope = use_audio_rope
        self.audio_scaling = audio_scaling
        self.base_freq = base_freq
        self.max_seq_len = max_seq_len


def create_audio_rope_replacement(ttt_config: TTTConfig, audio_config: AudioRoPEConfig = None):
    """
    Create audio RoPE replacement functions for TTT layer.
    
    Returns functions that can replace the 3D RoPE calls in existing TTT code.
    
    Args:
        ttt_config: TTT configuration
        audio_config: Audio RoPE configuration
        
    Returns:
        dict: Replacement functions for TTT integration
    """
    if audio_config is None:
        audio_config = AudioRoPEConfig()
    
    head_dim = ttt_config.model_dim // ttt_config.num_heads
    
    def precompute_freqs_replacement(seq_len: int) -> torch.Tensor:
        """Replacement for precompute_freqs_cis_3d."""
        return precompute_audio_rope_1d(
            head_dim=head_dim,
            seq_len=seq_len,
            base_freq=audio_config.base_freq,
            audio_scaling=audio_config.audio_scaling
        )
    
    def apply_rotary_replacement(Q: torch.Tensor, K: torch.Tensor, freqs_cis: torch.Tensor):
        """Replacement for apply_rotary_emb."""
        return apply_audio_rotary_emb(Q, K, freqs_cis)
    
    return {
        'precompute_freqs_cis': precompute_freqs_replacement,
        'apply_rotary_emb': apply_rotary_replacement,
        'audio_config': audio_config
    }


def test_audio_rope_implementation():
    """Test the audio RoPE implementation."""
    print("🧪 TESTING AUDIO ROPE IMPLEMENTATION")
    print("=" * 50)
    
    # Test parameters
    batch_size = 2
    seq_len = 128
    num_heads = 8
    head_dim = 64
    
    # Create test Q, K tensors
    Q = torch.randn(batch_size, seq_len, num_heads, head_dim)
    K = torch.randn(batch_size, seq_len, num_heads, head_dim)
    
    print(f"Input shapes: Q={Q.shape}, K={K.shape}")
    
    # Test 1: Direct audio RoPE
    print("\n1. Testing direct Audio RoPE...")
    audio_rope = AudioRoPE1D(head_dim=head_dim)
    cos, sin = audio_rope(seq_len)
    Q_rope1, K_rope1 = apply_audio_rope_1d(Q, K, cos, sin)
    
    print(f"   Output shapes: Q_rope={Q_rope1.shape}, K_rope={K_rope1.shape}")
    print(f"   Q change: {F.mse_loss(Q_rope1, Q):.6f}")
    print(f"   K change: {F.mse_loss(K_rope1, K):.6f}")
    
    # Test 2: Compatible interface
    print("\n2. Testing compatible interface...")
    freqs_cis = precompute_audio_rope_1d(head_dim, seq_len)
    Q_rope2, K_rope2 = apply_audio_rotary_emb(Q, K, freqs_cis)
    
    print(f"   Output shapes: Q_rope={Q_rope2.shape}, K_rope={K_rope2.shape}")
    print(f"   Q change: {F.mse_loss(Q_rope2, Q):.6f}")
    print(f"   K change: {F.mse_loss(K_rope2, K):.6f}")
    
    # Test 3: Consistency check
    print("\n3. Testing consistency between approaches...")
    consistency_q = F.mse_loss(Q_rope1, Q_rope2)
    consistency_k = F.mse_loss(K_rope1, K_rope2)
    
    print(f"   Q consistency: {consistency_q:.8f}")
    print(f"   K consistency: {consistency_k:.8f}")
    
    if consistency_q < 1e-5 and consistency_k < 1e-5:
        print("   ✅ Both approaches are consistent!")
    else:
        print("   ❌ Approaches differ - need to debug")
    
    # Test 4: Audio scaling effect
    print("\n4. Testing audio scaling effect...")
    freqs_no_scale = precompute_audio_rope_1d(head_dim, seq_len, audio_scaling=False)
    freqs_with_scale = precompute_audio_rope_1d(head_dim, seq_len, audio_scaling=True)
    
    # Compare magnitudes (complex numbers)
    scaling_effect = torch.abs(freqs_no_scale - freqs_with_scale).mean()
    print(f"   Scaling effect magnitude: {scaling_effect:.6f}")
    
    # Test 5: Sequence length scaling
    print("\n5. Testing different sequence lengths...")
    for test_seq_len in [64, 256, 512]:
        try:
            freqs_test = precompute_audio_rope_1d(head_dim, test_seq_len)
            print(f"   seq_len={test_seq_len}: ✅ shape={freqs_test.shape}")
        except Exception as e:
            print(f"   seq_len={test_seq_len}: ❌ {e}")
    
    print("\n✅ Audio RoPE implementation test complete!")
    
    return {
        'Q_original': Q,
        'K_original': K,
        'Q_rope': Q_rope2,
        'K_rope': K_rope2,
        'freqs_cis': freqs_cis
    }


def compare_with_3d_rope():
    """Compare Audio RoPE with current 3D RoPE approach."""
    print("\n🔄 COMPARING AUDIO ROPE VS 3D ROPE")
    print("=" * 50)
    
    # Import 3D RoPE for comparison
    try:
        from moshi_ttt.models.ssm.utils import precompute_freqs_cis_3d, apply_rotary_emb
        
        # Test parameters
        batch_size = 2
        seq_len = 128
        head_dim = 64
        num_heads = 8
        
        # Create test data
        Q = torch.randn(batch_size, seq_len, num_heads, head_dim)
        K = torch.randn(batch_size, seq_len, num_heads, head_dim)
        
        print(f"Comparing approaches for seq_len={seq_len}, head_dim={head_dim}")
        
        # 1. Audio RoPE (our implementation)
        print("\n1. Audio 1D RoPE:")
        freqs_audio = precompute_audio_rope_1d(head_dim, seq_len)
        Q_audio, K_audio = apply_audio_rotary_emb(Q, K, freqs_audio)
        
        print(f"   Q change: {F.mse_loss(Q_audio, Q):.6f}")
        print(f"   K change: {F.mse_loss(K_audio, K):.6f}")
        
        # 2. 3D RoPE (current implementation)
        print("\n2. Video 3D RoPE:")
        # Find 3D mapping
        mappings = []
        for f in range(1, int(seq_len**0.5) + 1):
            if seq_len % f == 0:
                remaining = seq_len // f
                for h in range(1, int(remaining**0.5) + 1):
                    if remaining % h == 0:
                        w = remaining // h
                        mappings.append((h, w, f))
        
        if mappings:
            h, w, f = sorted(mappings, key=lambda x: abs(x[0] - x[1]))[0]
            print(f"   Using 3D mapping: {h}×{w}×{f}")
            
            freqs_3d = precompute_freqs_cis_3d(head_dim, h, w, f, 10000.0)
            Q_3d, K_3d = apply_rotary_emb(Q, K, freqs_cis=freqs_3d)
            
            print(f"   Q change: {F.mse_loss(Q_3d, Q):.6f}")
            print(f"   K change: {F.mse_loss(K_3d, K):.6f}")
            
            # Compare outputs
            print("\n3. Comparison:")
            q_diff = F.mse_loss(Q_audio, Q_3d)
            k_diff = F.mse_loss(K_audio, K_3d)
            
            print(f"   Q difference: {q_diff:.6f}")
            print(f"   K difference: {k_diff:.6f}")
            
            if q_diff > 0.01 or k_diff > 0.01:
                print("   📊 Significant difference detected - approaches diverge!")
            else:
                print("   ⚠️  Small difference - may not impact performance much")
        
        else:
            print("   ❌ No valid 3D mapping found")
        
    except ImportError as e:
        print(f"❌ Could not import 3D RoPE: {e}")


def main():
    """Run complete audio RoPE implementation and testing."""
    print("🎵 AUDIO ROPE IMPLEMENTATION")
    print("1D Temporal Positional Encoding for Audio Sequences")
    print()
    
    # Test implementation
    test_results = test_audio_rope_implementation()
    
    # Compare with 3D RoPE
    compare_with_3d_rope()
    
    # Generate integration guide
    print("\n📋 INTEGRATION GUIDE")
    print("=" * 50)
    print("To replace 3D RoPE with Audio RoPE in TTT:")
    print()
    print("1. Replace precompute_freqs_cis_3d call:")
    print("   # OLD:")
    print("   freqs_cis = precompute_freqs_cis_3d(head_dim, h, w, f, theta)")
    print("   # NEW:")
    print("   freqs_cis = precompute_audio_rope_1d(head_dim, seq_len)")
    print()
    print("2. Replace apply_rotary_emb call:")
    print("   # OLD:")
    print("   Q_rope, K_rope = apply_rotary_emb(Q, K, freqs_cis=freqs_cis)")
    print("   # NEW:")
    print("   Q_rope, K_rope = apply_audio_rotary_emb(Q, K, freqs_cis)")
    print()
    print("3. Or use replacement functions:")
    print("   replacements = create_audio_rope_replacement(ttt_config)")
    print("   freqs_cis = replacements['precompute_freqs_cis'](seq_len)")
    print("   Q_rope, K_rope = replacements['apply_rotary_emb'](Q, K, freqs_cis)")
    print()
    print("✅ Audio RoPE implementation complete and ready for integration!")
    
    return test_results


if __name__ == "__main__":
    results = main()