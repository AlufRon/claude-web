#!/usr/bin/env python3
"""Simple test to check if set_inner_loop_logging works"""

import sys
sys.path.append('.')

# Import TTT components
from moshi_ttt.models.ssm.ops.ttt_mlp import (
    set_inner_loop_logging, 
    _global_log_inner_loop_losses
)

print("=== SIMPLE LOGGING TEST ===")

print(f"1. Initial global flag: {_global_log_inner_loop_losses}")

print("2. Calling set_inner_loop_logging(True)...")
set_inner_loop_logging(True)

print(f"3. After enabling - global flag: {_global_log_inner_loop_losses}")

print("4. Calling set_inner_loop_logging(False)...")
set_inner_loop_logging(False)

print(f"5. After disabling - global flag: {_global_log_inner_loop_losses}")

print("=== TEST COMPLETE ===")