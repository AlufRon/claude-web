# Moshi-compatible configuration for TTT layers
from dataclasses import dataclass
from typing import Optional

@dataclass
class TTTConfig:
    """Configuration for TTT layers in Moshi"""
    
    # Model dimensions
    model_dim: int = 1024
    num_heads: int = 8 
    
    # TTT-specific configs
    ttt_base_lr: float = 0.1          # Video-DiT exact value
    mini_batch_size: int = 16
    scan_checkpoint_group_size: int = 16  # Video-DiT proven config (fixes checkpoint error)
    ssm_layer: str = "ttt_mlp"        # or "ttt_linear"
    
    # SSM Gating config (Video-DiT configs.py line 44)
    gating_alpha_init: float = 0.1   # Video-DiT exact value - use 0.01 for multi-layer
    
    # TTT Output Normalization (NEW)
    normalize_ttt_output: bool = False  # Enable learnable output scaling
    target_output_norm: float = 25.0    # Target L2 norm for TTT output

    # RoPE Configuration (following ttt-lm-pytorch pattern)
    use_rope: bool = False  # Enable Rotary Position Embeddings (default: False for backward compatibility)
    rope_theta: float = 10000.0  # Base for RoPE frequency computation
    latent_height: int = 32
    latent_width: int = 32
    compressed_num_frames: int = 16
    
    # Continuous RoPE Configuration (for chunked training)
    rope_continuous: bool = False  # Enable continuous positions across chunks from same file
    rope_reset_on_new_file: bool = True  # Auto-reset positions when new file starts
    
    @property
    def head_dim(self) -> int:
        return self.model_dim // self.num_heads
