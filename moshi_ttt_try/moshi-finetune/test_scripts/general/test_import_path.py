#!/usr/bin/env python3
"""Test different import paths for the global flag"""

import sys
sys.path.append('.')

print("=== IMPORT PATH TEST ===")

# Test 1: Direct import (what I was doing)
print("1. Testing direct import...")
from moshi_ttt.models.ssm.ops.ttt_mlp import set_inner_loop_logging, _global_log_inner_loop_losses
print(f"   Initial flag: {_global_log_inner_loop_losses}")
set_inner_loop_logging(True)
print(f"   After enable: {_global_log_inner_loop_losses}")

# Test 2: Module import (what might be happening)
print("\n2. Testing module import...")
import moshi_ttt.models.ssm.ops.ttt_mlp as ttt_mlp
print(f"   Module flag: {ttt_mlp._global_log_inner_loop_losses}")

# Test 3: Check if they're the same object
print("\n3. Object identity check...")
print(f"   Same object? {_global_log_inner_loop_losses is ttt_mlp._global_log_inner_loop_losses}")

# Test 4: Check module locations
print("\n4. Module locations...")
print(f"   Module file: {ttt_mlp.__file__}")

print("\n=== TEST COMPLETE ===")