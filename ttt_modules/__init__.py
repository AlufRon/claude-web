"""
TTT Modules for Llama-Omni Integration

Implements Test-Time Training (TTT) layers for unlimited context speech generation.

Key components:
- TTTMLP: Drop-in replacement for Llama attention layers
- ttt_mlp: Core TTT operation with analytical gradients
- Integration utilities for Llama-Omni model
"""

from ttt_modules.ttt_layer import TTTMLP
from ttt_modules.ops.ttt_mlp import ttt_mlp, compute_mini_batch

__all__ = [
    'TTTMLP',
    'ttt_mlp',
    'compute_mini_batch',
]

__version__ = '0.1.0'
