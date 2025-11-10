#!/usr/bin/env python3
"""Check dtypes of all parameters in checkpoint"""

import torch
from safetensors.torch import load_file
from pathlib import Path
from collections import defaultdict

checkpoint_path = Path("/sise/eliyanac-group/ron_al/dailytalk_finetune_from_librilight6/checkpoints/checkpoint_002500/consolidated/lora.safetensors")

print(f"Loading checkpoint: {checkpoint_path}")
state_dict = load_file(str(checkpoint_path))

print(f"\nTotal parameters: {len(state_dict)}")

# Group by dtype
dtype_groups = defaultdict(list)
for key, tensor in state_dict.items():
    dtype_groups[tensor.dtype].append(key)

print("\nParameters by dtype:")
for dtype, keys in sorted(dtype_groups.items(), key=lambda x: str(x[0])):
    print(f"\n{dtype}: {len(keys)} parameters")
    for key in keys[:5]:  # Show first 5
        print(f"  - {key}")
    if len(keys) > 5:
        print(f"  ... and {len(keys) - 5} more")
