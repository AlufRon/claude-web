# Simplified utilities for Moshi TTT integration
from dataclasses import dataclass
from typing import Optional
import torch

@dataclass
class SequenceMetadata:
    """Simplified sequence metadata for Moshi (1D sequences)"""
    init_offset: Optional[int] = None
    base_offset: Optional[int] = None
    text_length: Optional[int] = None
    num_chunks: int = 1
    seq_text_length: int = 0  # For audio: no text component
    is_multiscene: bool = False  # For audio: no video scenes
    position_ids: Optional[torch.Tensor] = None  # [B, seq_len] for RoPE (optional, defaults to None)

def full_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Get full tensor (placeholder for tensor parallelism)"""
    return tensor

def shard_tensor(tensor: torch.Tensor, device_mesh, dim: int) -> torch.Tensor:
    """Shard tensor (placeholder for tensor parallelism)"""
    return tensor

def place_into(target: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    """Place target into source's tensor structure (placeholder)

    In distributed setting, this would place 'target' (local tensor) into
    the structure of 'source' (potentially distributed tensor).
    For non-distributed case, just return target.

    CRITICAL: Must return 'target' (first arg), not 'source' (second arg)!
    """
    return target  # Fixed: was returning source (wrong!)

def to_local(tensor: torch.Tensor) -> torch.Tensor:
    """Convert to local tensor (placeholder)"""
    return tensor
