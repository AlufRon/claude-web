#!/usr/bin/env python3
"""
Cache Logging Patch - CUDA Graph Safe Version

Only logs cache creation (capacity, wraparound points).
No logging during inference to avoid breaking CUDA graphs.

The cache creation info alone tells us:
- Capacity = 3000 tokens = 30 seconds
- First wraparound at token 3000
- Second wraparound at token 6000
- User's activation jump at token 6100 = 100 tokens after second wraparound!

Usage:
    from cache_logging_patch import instrument_cache_logging
    instrument_cache_logging()
"""

import sys
from pathlib import Path

def instrument_cache_logging(min_capacity=1000):
    """
    Instrument RingKVCache to log creation info only.

    Args:
        min_capacity: Only log caches >= this size (default 1000, filters depformer)
    """
    # Import Moshi modules
    moshi_path = Path('/home/alufr/ttt_tests/moshi/moshi')
    if str(moshi_path) not in sys.path:
        sys.path.insert(0, str(moshi_path))

    from moshi.modules.transformer import RingKVCache

    # Store original __init__
    original_init = RingKVCache.__init__

    def init_with_logging(self, batch_size, num_heads, dim_per_head, capacity,
                          respect_exec_mask=True, device=None, dtype=None):
        """Log cache creation - safe, happens before CUDA graphs"""

        # Call original
        if device is None:
            import torch
            device = torch.device("cuda")
        if dtype is None:
            import torch
            dtype = torch.bfloat16

        original_init(self, batch_size, num_heads, dim_per_head, capacity,
                     respect_exec_mask, device, dtype)

        # Only log large caches (main transformer, not depformer)
        if capacity >= min_capacity:
            cache_memory_mb = (2 * batch_size * num_heads * capacity * dim_per_head * 2) / (1024**2)
            tokens_per_second = 12.5 * 8
            time_capacity = capacity / tokens_per_second

            print(f"\n{'='*80}")
            print(f"📦 RingKVCache [Capacity: {capacity} = {time_capacity:.1f} sec]")
            print(f"{'='*80}")
            print(f"  Heads: {num_heads}, Dim/head: {dim_per_head}")
            print(f"  Memory: {cache_memory_mb:.1f} MB")
            print(f"")
            print(f"  🔄 Wraparound schedule:")
            print(f"     Token {capacity:>5}: First wraparound  (loses tokens 0-{capacity-1})")
            print(f"     Token {2*capacity:>5}: Second wraparound (loses tokens 0-{2*capacity-1})")
            print(f"     Token {3*capacity:>5}: Third wraparound  (loses tokens 0-{3*capacity-1})")
            print(f"")
            print(f"  ⚠️  Context window: Only last {capacity} tokens accessible at any time")
            print(f"{'='*80}\n")

    # Apply patch (only to __init__, NOT to complete())
    RingKVCache.__init__ = init_with_logging

    print("✅ RingKVCache instrumented (cache creation logging only)")
    print(f"   Min capacity: {min_capacity} (filters small depformer caches)")
    print(f"   Safe for CUDA graphs - no inference-time logging")
    print()


if __name__ == "__main__":
    print("This module should be imported, not run directly")
    print("\nUsage:")
    print("  from cache_logging_patch import instrument_cache_logging")
    print("  instrument_cache_logging()")
