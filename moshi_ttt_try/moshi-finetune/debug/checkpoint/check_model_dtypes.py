#!/usr/bin/env python3
"""Check if model has mixed dtypes after loading"""

import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "moshi"))
sys.path.insert(0, str(Path(__file__).parent))

from run_paper_metrics_on_checkpoint import load_ttt_model

checkpoint_path = "/sise/eliyanac-group/ron_al/dailytalk_finetune_from_librilight6/checkpoints/checkpoint_002800/consolidated"

print(f"Loading model from {checkpoint_path}...")
model, checkpoint_info = load_ttt_model(checkpoint_path, device="cuda")

print("\nChecking all parameter dtypes...")

dtypes = {}
for name, param in model.named_parameters():
    dtype = param.dtype
    if dtype not in dtypes:
        dtypes[dtype] = []
    dtypes[dtype].append(name)

print(f"\nFound {len(dtypes)} different dtypes:\n")
for dtype, names in sorted(dtypes.items(), key=lambda x: len(x[1]), reverse=True):
    print(f"{dtype}: {len(names)} parameters")
    for name in names[:10]:
        print(f"  - {name}")
    if len(names) > 10:
        print(f"  ... and {len(names) - 10} more")
    print()

if len(dtypes) > 1:
    print("⚠️  WARNING: Model has MIXED dtypes!")
else:
    print("✅ All parameters have the same dtype")
