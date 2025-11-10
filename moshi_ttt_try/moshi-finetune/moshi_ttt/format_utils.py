"""
Step 3.2.2: Format conversion utilities
Convert between Moshi format [B, seq_len, d_model] and TTT format [B, H, NC, C, HD]
"""

import torch
from typing import Tuple
import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))
from config import TTTConfig


class SequenceMetadata:
    """Sequence metadata for Moshi TTT processing (adapted from Video-DiT)"""
    def __init__(self, seq_length: int, mini_batch_size: int):
        self.seq_length = seq_length
        self.mini_batch_size = mini_batch_size
        # For Moshi (1D audio), we don't have video-specific fields
        self.is_multiscene = False  # No multi-scene for audio
        self.seq_text_length = 0  # Audio only, no text component


def create_sequence_metadata(x: torch.Tensor, ttt_config: TTTConfig) -> SequenceMetadata:
    """
    Create sequence metadata for TTT processing.
    """
    B, seq_len, d_model = x.shape
    return SequenceMetadata(
        seq_length=seq_len,
        mini_batch_size=ttt_config.mini_batch_size
    )


def moshi_to_ttt_format(
    x: torch.Tensor, 
    ttt_config: TTTConfig
) -> Tuple[torch.Tensor, dict]:
    """
    Convert Moshi format [B, seq_len, d_model] to TTT format [B, H, NC, C, HD] with right-padding + mask.
    
    Args:
        x: Input tensor in Moshi format [B, seq_len, d_model]
        ttt_config: TTT configuration containing dimensions
        
    Returns:
        tuple: (converted_tensor, metadata_dict)
        - converted_tensor: [B, H, NC, C, HD] format for TTT processing
        - metadata_dict: Information needed for reverse conversion including pad_mask
    """
    B, seq_len, d_model = x.shape
    H, HD = ttt_config.num_heads, ttt_config.head_dim
    C = ttt_config.mini_batch_size
    
    # Validate dimensions
    assert d_model == ttt_config.model_dim, f"Model dim mismatch: {d_model} != {ttt_config.model_dim}"
    assert H * HD == d_model, f"Head dimensions don't match: {H} * {HD} != {d_model}"
    assert C > 0, "mini_batch_size must be > 0"
    assert seq_len >= 0, "seq_len must be >= 0"

    # Compute padded length using ceil division
    NC = (seq_len + C - 1) // C  # ceil division
    padded_len = NC * C
    pad_tokens = padded_len - seq_len

    # Right-pad along time dimension if needed
    if pad_tokens > 0:
        pad = x.new_zeros(B, pad_tokens, d_model)
        x_padded = torch.cat([x, pad], dim=1)
    else:
        x_padded = x

    # [B, padded_len, d_model] -> [B, NC, C, d_model]
    # Use reshape to be safe on non-contiguous inputs
    x_chunked = x_padded.reshape(B, NC, C, d_model)

    # [B, NC, C, d_model] -> [B, NC, C, H, HD] -> [B, H, NC, C, HD]
    x_heads = x_chunked.reshape(B, NC, C, H, HD)
    x_ttt = x_heads.permute(0, 3, 1, 2, 4).contiguous()

    # Build pad mask in TTT layout for convenient broadcasting in losses/updates
    # True = valid, False = padding
    if pad_tokens > 0:
        valid = torch.arange(padded_len, device=x.device).unsqueeze(0) < seq_len
        valid = valid.view(1, NC, C, 1).expand(B, NC, C, 1)  # [B, NC, C, 1]
        pad_mask = valid.permute(0, 3, 1, 2).unsqueeze(-1)   # [B, 1, NC, C, 1]
        pad_mask = pad_mask.expand(B, H, NC, C, 1).contiguous()
    else:
        pad_mask = x_ttt.new_ones(B, H, NC, C, 1, dtype=torch.bool)

    metadata = {
        "original_shape": (B, seq_len, d_model),
        "padded_shape": (B, padded_len, d_model),
        "NC": NC, "C": C, "H": H, "HD": HD,
        "pad_tokens": pad_tokens,
        "pad_mask": pad_mask,  # boolean mask [B, H, NC, C, 1]
    }
    return x_ttt, metadata


def ttt_to_moshi_format(
    x_ttt: torch.Tensor, 
    metadata: dict
) -> torch.Tensor:
    """
    Convert TTT format [B, H, NC, C, HD] back to Moshi format [B, seq_len, d_model], slicing off right-padding.
    
    Args:
        x_ttt: Tensor in TTT format [B, H, NC, C, HD]
        metadata: Metadata from the forward conversion
        
    Returns:
        torch.Tensor: [B, seq_len, d_model] format for Moshi (original shape preserved)
    """
    B, H, NC, C, HD = x_ttt.shape
    B0, seq_len, d_model = metadata["original_shape"]
    padded_len = metadata["padded_shape"][1]
    assert B == B0, f"Batch size mismatch: {B} != {B0}"
    assert H == metadata["H"], f"Head count mismatch: {H} != {metadata['H']}"
    assert NC == metadata["NC"], f"Chunk count mismatch: {NC} != {metadata['NC']}"
    assert C == metadata["C"], f"Chunk size mismatch: {C} != {metadata['C']}"
    assert HD == metadata["HD"], f"Head dim mismatch: {HD} != {metadata['HD']}"

    # [B, H, NC, C, HD] -> [B, NC, C, H, HD] -> [B, NC, C, d_model] -> [B, padded_len, d_model]
    x_heads = x_ttt.permute(0, 2, 3, 1, 4).contiguous()
    x_chunked = x_heads.reshape(B, NC, C, H * HD)
    x_flat = x_chunked.reshape(B, padded_len, d_model)

    # Slice back to original seq_len to keep residual adds shape-safe
    return x_flat[:, :seq_len, :]


def test_format_conversions(ttt_config: TTTConfig, batch_size: int = 2, seq_len: int = 63) -> bool:
    """
    Test that format conversions are reversible and preserve data with pad+mask approach.
    
    Args:
        ttt_config: TTT configuration
        batch_size: Batch size for test
        seq_len: Sequence length for test
        
    Returns:
        bool: True if all tests pass
    """
    d_model = ttt_config.model_dim
    
    # Create test input
    x_original = torch.randn(batch_size, seq_len, d_model)
    
    # Forward conversion
    x_ttt, metadata = moshi_to_ttt_format(x_original, ttt_config)
    
    # Check TTT format shape (using ceil division for padding)
    expected_H = ttt_config.num_heads
    expected_HD = ttt_config.head_dim
    expected_C = ttt_config.mini_batch_size
    expected_NC = (seq_len + expected_C - 1) // expected_C  # Ceil division
    
    expected_ttt_shape = (batch_size, expected_H, expected_NC, expected_C, expected_HD)
    assert x_ttt.shape == expected_ttt_shape, f"TTT shape mismatch: {x_ttt.shape} != {expected_ttt_shape}"
    
    # Reverse conversion should return the ORIGINAL length, not the padded length
    x_recovered = ttt_to_moshi_format(x_ttt, metadata)
    assert x_recovered.shape == (batch_size, seq_len, d_model), f"Recovered shape mismatch: {x_recovered.shape} != {(batch_size, seq_len, d_model)}"
    
    # Data preservation on valid positions
    max_diff = (x_recovered - x_original).abs().max().item()
    assert max_diff < 1e-6, f"Data not preserved: max diff = {max_diff}"
    
    # Verify pad_mask correctness
    pad_mask = metadata["pad_mask"]
    
    assert pad_mask.shape == (batch_size, expected_H, expected_NC, expected_C, 1), f"Pad mask shape mismatch: {pad_mask.shape}"
    assert pad_mask.dtype == torch.bool, f"Pad mask should be boolean, got {pad_mask.dtype}"
    
    # Check that valid positions are True and padding positions are False
    expected_valid_tokens = seq_len
    actual_valid_tokens = pad_mask.sum().item()
    expected_total_valid = batch_size * expected_H * expected_valid_tokens
    assert actual_valid_tokens == expected_total_valid, f"Pad mask valid count mismatch: {actual_valid_tokens} != {expected_total_valid}"
    
    return True
