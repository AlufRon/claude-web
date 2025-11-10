#!/usr/bin/env python3
"""Debug script to check what losses are collected"""

import sys
sys.path.append('.')

# Import the loss collection function
from moshi_ttt.models.ssm.ops.ttt_mlp import get_collected_inner_loop_losses, _collected_inner_loop_losses

print("=== DEBUG: Loss Collection State ===")
print(f"Global losses dict: {_collected_inner_loop_losses}")
print(f"Type: {type(_collected_inner_loop_losses)}")
print(f"Length: {len(_collected_inner_loop_losses)}")

if _collected_inner_loop_losses:
    print("Keys:")
    for key in _collected_inner_loop_losses.keys():
        print(f"  {key} ({type(key)}): {len(_collected_inner_loop_losses[key])} losses")

# Try getting losses
losses = get_collected_inner_loop_losses()
print(f"\nget_collected_inner_loop_losses() returned:")
print(f"  Type: {type(losses)}")
print(f"  Content: {losses}")

if isinstance(losses, dict):
    print("  Keys:")
    for key in losses.keys():
        print(f"    {key} ({type(key)}): {len(losses[key]) if isinstance(losses[key], list) else 'not a list'}")