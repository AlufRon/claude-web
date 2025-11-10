#!/usr/bin/env python3
"""Test script to check if live logging is working during a simple TTT forward pass"""

import sys
sys.path.append('.')
import torch

# Import TTT components
from moshi_ttt.models.ssm.ops.ttt_mlp import (
    set_inner_loop_logging, 
    get_collected_inner_loop_losses, 
    _collected_inner_loop_losses,
    _global_log_inner_loop_losses
)
from moshi_ttt.ttt_layer import TTTMLP, TTTConfig

print("=== TEST: Live TTT Logging ===")

# Enable logging
print("1. Enabling inner loop logging...")
set_inner_loop_logging(True)
print(f"   Global flag: {_global_log_inner_loop_losses}")

# Create TTT config and layer
print("2. Creating TTT layer...")
config = TTTConfig(
    dim=512,
    heads=8,
    lr=0.001,
    mini_batch_size=8,
    use_glu=False,
    use_tanh_gating=False,
    use_gating=True,
    initial_alpha=0.01
)

ttt_layer = TTTMLP(config, layer_id=25, use_kernel=False)

# Create test inputs
print("3. Creating test inputs...")
batch_size = 1
seq_len = 16
X = torch.randn(batch_size, config.heads, seq_len // config.mini_batch_size, config.mini_batch_size, config.dim // config.heads)

print(f"   Input shape: {X.shape}")

# Run forward pass
print("4. Running TTT forward pass...")
try:
    output = ttt_layer(X, X, X)
    print(f"   Output shape: {output.shape}")
    print("   ✅ Forward pass completed")
except Exception as e:
    print(f"   ❌ Forward pass failed: {e}")
    sys.exit(1)

# Check collected losses
print("5. Checking collected losses...")
print(f"   Global losses dict: {_collected_inner_loop_losses}")
print(f"   Dict length: {len(_collected_inner_loop_losses)}")

if _collected_inner_loop_losses:
    for layer_id, losses in _collected_inner_loop_losses.items():
        print(f"   Layer {layer_id}: {len(losses)} losses")
        if losses:
            print(f"     Sample losses: {losses[:3]}")

# Test get function
print("6. Testing get_collected_inner_loop_losses()...")
collected = get_collected_inner_loop_losses()
print(f"   Returned: {type(collected)}, length: {len(collected) if isinstance(collected, dict) else 'N/A'}")

if isinstance(collected, dict) and collected:
    for layer_id, losses in collected.items():
        print(f"   Layer {layer_id}: {len(losses)} losses")

print("=== TEST COMPLETE ===")