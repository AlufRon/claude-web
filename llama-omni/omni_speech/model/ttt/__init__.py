"""
Test-Time Training (TTT) Implementation for Llama-Omni

This module provides TTT layers that can replace standard attention mechanisms
to enable unlimited context length through test-time adaptation.

Key Components:
- utils.py: Core utility functions (LayerNorm, L2 loss, RoPE)
- ops.py: TTT operations (ttt_linear, ttt_mlp)
- ttt_layer.py: TTT layer implementations
- cache.py: State management for conversation-level persistence
- logger.py: Logging and CSV tracking utilities

Based on:
- TTT-Video (CVPR 2025): https://github.com/test-time-training/ttt-video-dit
- TTT-LM-Kernels: https://github.com/test-time-training/ttt-lm-kernels

Usage:
    # Enable TTT in config
    config = OmniSpeechConfig(
        use_ttt=True,
        ttt_layer_type="ttt_linear",
        ttt_mini_batch_size=64,
        ttt_layer_indices=[24, 25, 26, 27, 28, 29, 30, 31],  # Top 8 layers
    )

    # Model will automatically use TTT in specified layers
    model = OmniSpeechLlamaForCausalLM(config)

    # State is managed via TTTCache
    from omni_speech.model.ttt import create_ttt_cache_from_config
    cache = create_ttt_cache_from_config(config, max_batch_size=8)
"""

# Core utilities
from .utils import (
    ln_fwd,
    ln_fused_l2_bwd,
    apply_rotary_pos_emb,
    precompute_freqs_cis,
    scan,
    log_ttt_state,
)

# TTT operations
from .ops import (
    ttt_linear,
    compute_ttt_linear_mini_batch,
)

# TTT layers
from .ttt_layer import (
    TTTLinearLayer,
    TTTMLPLayer,
)

# State management
from .cache import (
    TTTCache,
    create_ttt_cache_from_config,
)

# Logging
from .logger import (
    TTTCSVLogger,
    TTTStatsTracker,
    setup_ttt_logging,
    log_ttt_update,
)

__all__ = [
    # Utils
    'ln_fwd',
    'ln_fused_l2_bwd',
    'apply_rotary_pos_emb',
    'precompute_freqs_cis',
    'scan',
    'log_ttt_state',
    # Ops
    'ttt_linear',
    'compute_ttt_linear_mini_batch',
    # Layers
    'TTTLinearLayer',
    'TTTMLPLayer',
    # Cache
    'TTTCache',
    'create_ttt_cache_from_config',
    # Logging
    'TTTCSVLogger',
    'TTTStatsTracker',
    'setup_ttt_logging',
    'log_ttt_update',
]

__version__ = '0.1.0'
