#!/usr/bin/env python3
"""Test import issue properly"""

import sys
sys.path.append('.')

print("=== IMPORT ISSUE DEBUG ===")

# Import the module first
import moshi_ttt.models.ssm.ops.ttt_mlp as ttt_module

print(f"1. Module imported from: {ttt_module.__file__}")
print(f"2. Initial flag value: {ttt_module._global_log_inner_loop_losses}")

# Now import the function and variable
from moshi_ttt.models.ssm.ops.ttt_mlp import set_inner_loop_logging, _global_log_inner_loop_losses

print(f"3. Direct import flag value: {_global_log_inner_loop_losses}")
print(f"4. Same object? {_global_log_inner_loop_losses is ttt_module._global_log_inner_loop_losses}")

# Try enabling logging
print(f"5. Calling set_inner_loop_logging(True)...")
set_inner_loop_logging(True)

print(f"6. After enable - module flag: {ttt_module._global_log_inner_loop_losses}")
print(f"7. After enable - direct flag: {_global_log_inner_loop_losses}")

print("=== TEST COMPLETE ===")