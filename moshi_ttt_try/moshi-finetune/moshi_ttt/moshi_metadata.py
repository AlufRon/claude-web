"""
Step 3.2.3: Create sequence metadata for Moshi
Adapt Video-DiT's SequenceMetadata for 1D audio sequences
"""

import torch
from typing import Optional, Dict, Any
import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))
from config import TTTConfig


class MoshiSequenceMetadata:
    """
    Sequence metadata for Moshi audio sequences, adapted from Video-DiT's SequenceMetadata.
    
    Handles 1D audio sequences instead of 2D video sequences.
    Manages chunking and streaming requirements for Moshi.
    """
    
    def __init__(
        self,
        batch_size: int,
        seq_len: int,
        d_model: int,
        ttt_config: TTTConfig,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        position_ids: Optional[torch.Tensor] = None
    ):
        """
        Initialize sequence metadata for Moshi audio processing.
        
        Args:
            batch_size: Batch size
            seq_len: Sequence length (number of audio tokens)
            d_model: Model dimension
            ttt_config: TTT configuration
            device: Torch device
            dtype: Torch dtype
            position_ids: Position IDs for RoPE [B, seq_len] (optional)
        """
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.d_model = d_model
        self.ttt_config = ttt_config
        self.device = device or torch.device('cpu')
        self.dtype = dtype or torch.float32
        self.position_ids = position_ids  # Store position_ids for RoPE
        
        # Calculate chunking parameters
        self.mini_batch_size = ttt_config.mini_batch_size
        self.NC = (seq_len + self.mini_batch_size - 1) // self.mini_batch_size  # Ceiling division
        self.C = self.mini_batch_size
        
        # TTT dimensions
        self.H = ttt_config.num_heads
        self.HD = ttt_config.head_dim
        
        # Truncation information (replacing padding logic)
        self.truncated_len = self.NC * self.C
        self.truncated_tokens = max(0, seq_len - self.truncated_len)
        self.was_truncated = self.truncated_tokens > 0
        
        # Shape information
        self.moshi_shape = (batch_size, seq_len, d_model)
        self.ttt_shape = (batch_size, self.H, self.NC, self.C, self.HD)
        self.truncated_moshi_shape = (batch_size, self.truncated_len, d_model)
        
    def get_format_info(self) -> Dict[str, Any]:
        """Get format conversion information."""
        return {
            'original_shape': self.moshi_shape,
            'truncated_shape': self.truncated_moshi_shape,
            'ttt_shape': self.ttt_shape,
            'NC': self.NC,
            'C': self.C,
            'H': self.H,
            'HD': self.HD,
            'was_truncated': self.was_truncated,
            'truncated_tokens': self.truncated_tokens,
            'seq_len': self.seq_len,
            'd_model': self.d_model
        }
        
    def validate_moshi_tensor(self, x: torch.Tensor) -> bool:
        """Validate that tensor matches expected Moshi format."""
        expected_shape = self.moshi_shape
        if x.shape != expected_shape:
            raise ValueError(f"Moshi tensor shape mismatch: {x.shape} != {expected_shape}")
        return True
        
    def validate_ttt_tensor(self, x: torch.Tensor) -> bool:
        """Validate that tensor matches expected TTT format."""
        expected_shape = self.ttt_shape
        if x.shape != expected_shape:
            raise ValueError(f"TTT tensor shape mismatch: {x.shape} != {expected_shape}")
        return True
        
    def create_truncation_mask(self) -> Optional[torch.Tensor]:
        """
        Create mask for truncated sequences.
        
        Returns:
            torch.Tensor or None: Mask tensor [B, truncated_len] where all values are True
                                 (since truncated tokens are simply discarded, no masking needed)
        """
        if not self.was_truncated:
            return None
            
        # For truncated sequences, all remaining tokens are real data
        # No False values needed since truncated tokens are discarded
        mask = torch.ones(self.batch_size, self.truncated_len, device=self.device, dtype=torch.bool)
        return mask
        
    def create_ttt_truncation_mask(self) -> Optional[torch.Tensor]:
        """
        Create truncation mask in TTT format [B, H, NC, C].
        
        Returns:
            torch.Tensor or None: Mask tensor where True indicates real data
        """
        moshi_mask = self.create_truncation_mask()
        if moshi_mask is None:
            return None
            
        # Reshape mask to match TTT chunking: [B, truncated_len] -> [B, NC, C]
        mask_chunked = moshi_mask.view(self.batch_size, self.NC, self.C)
        
        # Expand for heads: [B, NC, C] -> [B, H, NC, C]
        mask_ttt = mask_chunked.unsqueeze(1).expand(self.batch_size, self.H, self.NC, self.C)
        
        return mask_ttt
        
    def get_streaming_info(self) -> Dict[str, Any]:
        """Get information for streaming compatibility."""
        return {
            'chunk_size': self.C,
            'num_chunks': self.NC,
            'total_padded_length': self.padded_len,
            'effective_length': self.seq_len,
            'supports_streaming': True,  # Moshi supports streaming
            'chunk_boundaries': list(range(0, self.padded_len + 1, self.C))
        }
        
    def __repr__(self) -> str:
        return (
            f"MoshiSequenceMetadata("
            f"batch_size={self.batch_size}, "
            f"seq_len={self.seq_len}, "
            f"d_model={self.d_model}, "
            f"NC={self.NC}, "
            f"C={self.C}, "
            f"truncated_tokens={self.truncated_tokens})"
        )
