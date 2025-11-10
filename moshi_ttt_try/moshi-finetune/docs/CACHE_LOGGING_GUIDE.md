# Cache Logging Guide

**Purpose**: Diagnose the token 6100 activation jump by tracking KV cache wraparound behavior.

**Created**: 2025-10-27
**Related**: See `RESIDUAL_JUMP_INVESTIGATION.md` for background

---

## Quick Start

### Option 1: SLURM (Recommended)

```bash
cd /home/alufr/ttt_tests/moshi-finetune
sbatch slurm/inference/run_baseline_cache_logging.slurm
```

### Option 2: Direct Execution

```bash
cd /home/alufr/ttt_tests/moshi-finetune
conda activate moshi_ttt_fixed

python inference/run_baseline_with_cache_logging.py \
    --input /sise/eliyanac-group/ron_al/examples/combined_2673329117028914160_24000hz.wav \
    --output output_baseline_logged.wav \
    --log-interval 100
```

---

## What This Does

### 1. **Instruments RingKVCache**
The `cache_logging_patch.py` monkey-patches Moshi's KV cache to log:
- Cache creation with capacity (expected: 3000 tokens = 30 seconds)
- Every 100 tokens: offset, positions, wraparound status
- Critical tokens: 3000, 3001, 6000, 6001, 6100, etc.
- Wraparound events when cache overwrites old entries

### 2. **Runs Baseline Moshi**
- Pure vanilla Moshi (NO TTT, NO LoRA)
- Same problematic audio showing jump at token 6100
- Logs all cache activity to stdout

### 3. **Tracks Correlation**
Key questions answered:
- Does first wraparound occur at token 3001? (capacity = 3000)
- Does second wraparound occur at token 6001? (2× capacity)
- Does activation jump happen ~100 tokens after second wraparound?

---

## Expected Output

### Cache Creation
```
================================================================================
📦 RingKVCache Created
================================================================================
  Cache ID: 140234567890
  Capacity: 3000 positions (30.0 seconds)
  Batch size: 1
  Num heads: 32
  Dim per head: 128
  Respect exec mask: True
  Cache memory: 48.0 MB
  Wraparound will occur at token 3000
================================================================================
```

### Normal Operation (Token 100)
```
================================================================================
📊 Cache Update at Token 100 (regular interval)
================================================================================
  Cache capacity: 3000 (30.0 sec)
  Current end_offset: 100
  New end_offset: 101
  Tokens added this step: 1
  Cache write indices: [100..100]
  Total tokens processed: 100
  Number of wraparounds: 0
  Valid token positions: [0..99] (all tokens accessible)
================================================================================
```

### First Wraparound (Token 3000-3001)
```
================================================================================
🔄 FIRST WRAPAROUND!
📊 Cache Update at Token 3000 (FIRST WRAPAROUND)
================================================================================
  Cache capacity: 3000 (30.0 sec)
  Current end_offset: 3000
  New end_offset: 3001
  Tokens added this step: 1
  Cache write indices: [0..0]
  Total tokens processed: 3000
  Number of wraparounds: 1
  ⚠️  WRAPPING: Cache is OVERWRITING old entries
  Valid token positions in cache: [0..2999]
  Lost context: tokens [0..-1] no longer accessible
  Overwriting: positions [0..0]
  Losing: tokens [0..0]
================================================================================
```

### Second Wraparound (Token 6000-6001)
```
================================================================================
🔄 SECOND WRAPAROUND!
📊 Cache Update at Token 6000 (SECOND WRAPAROUND)
================================================================================
  Cache capacity: 3000 (30.0 sec)
  Current end_offset: 6000
  New end_offset: 6001
  Tokens added this step: 1
  Cache write indices: [0..0]
  Total tokens processed: 6000
  Number of wraparounds: 2
  ⚠️  WRAPPING: Cache is OVERWRITING old entries
  Valid token positions in cache: [3000..5999]
  Lost context: tokens [0..2999] no longer accessible
  Overwriting: positions [0..0]
  Losing: tokens [3000..3000]
================================================================================
```

### Token 6100 (Where Jump Occurs)
```
================================================================================
📊 Cache Update at Token 6100 (regular interval)
================================================================================
  Cache capacity: 3000 (30.0 sec)
  Current end_offset: 6100
  New end_offset: 6101
  Tokens added this step: 1
  Cache write indices: [100..100]
  Total tokens processed: 6100
  Number of wraparounds: 2
  ⚠️  WRAPPING: Cache is OVERWRITING old entries
  Valid token positions in cache: [3100..6099]
  Lost context: tokens [0..3099] no longer accessible
  Overwriting: positions [100..100]
  Losing: tokens [3100..3100]
================================================================================
```

---

## What To Look For

### ✅ Confirms Cache Hypothesis
If you see:
1. **First wraparound at token 3001** (capacity = 3000)
2. **Second wraparound at token 6001** (2× capacity)
3. **Activation jump at token 6100** (documented in your original logs)

**Conclusion**: Jump is 100 tokens after second wraparound → cache-related

### ❌ Refutes Cache Hypothesis
If you see:
1. Wraparounds occur but NO correlation with jump timing
2. Jump happens at different token in baseline vs TTT inference
3. Jump doesn't occur in baseline at all

**Conclusion**: Not cache-related, investigate other causes

---

## Analysis After Run

### 1. Check Log Files
```bash
# For SLURM
cat logs/inference/cache_logging_*.err | grep -E "WRAPAROUND|Token 6[01]"

# For direct execution
grep -E "WRAPAROUND|Token 6[01]" <output>
```

### 2. Compare with TTT Logs
Original TTT jump location:
```
Token 6000: resid=16.4
Token 6100: resid=27.5  ← JUMP
```

Does baseline show similar pattern?

### 3. Count Wraparounds
```bash
grep "WRAPAROUND" logs/inference/cache_logging_*.err | wc -l
```

Expected: 2-3 wraparounds for ~60 seconds of audio

---

## Files Created

| File | Purpose |
|------|---------|
| `inference/cache_logging_patch.py` | Monkey-patch for RingKVCache logging |
| `inference/run_baseline_with_cache_logging.py` | Wrapper inference script |
| `slurm/inference/run_baseline_cache_logging.slurm` | SLURM job script |
| `docs/CACHE_LOGGING_GUIDE.md` | This guide |

---

## Technical Details

### Cache Behavior

**Capacity**: 3000 positions = 30 seconds of audio
- Formula: `3000 tokens / 8 codebooks / 12.5 Hz = 30 seconds`

**Ring Buffer**: Overwrites oldest entries when full
- Token 0-2999: Fill cache
- Token 3000+: Start overwriting from position 0
- Token 6000+: Overwriting for second time

**Position Tracking**:
- `end_offset`: Total tokens processed
- Cache indices: `end_offset % capacity`
- RoPE positions: True sequence positions (no modulo)

### Why This Matters

If cache hypothesis is correct:
1. Model loses access to tokens 0-2999 at wraparound #2
2. This sudden context loss causes behavioral change
3. Manifests as activation magnitude jump
4. Happens ~100 tokens later as effect propagates through layers

### Alternative Explanations

If cache is NOT the cause:
1. **Training length**: Model trained on 15s (1500 tokens), breaks at 4× (6000 tokens)
2. **RoPE limits**: Rotary embeddings degrade at long positions
3. **Cumulative errors**: Autoregressive drift over long sequences
4. **Coincidence**: 60 seconds is just when generation quality degrades

---

## Next Steps

After running this experiment:

1. **If cache-related**:
   - Increase cache capacity in training
   - Implement better sliding window
   - Test with modified capacity

2. **If not cache-related**:
   - Focus on training sequence length
   - Investigate RoPE position encoding limits
   - Test with different context windows

3. **Document findings** in `RESIDUAL_JUMP_INVESTIGATION.md`

---

## Troubleshooting

### Import Error
```
ModuleNotFoundError: No module named 'moshi'
```
**Fix**: Ensure `/home/alufr/ttt_tests/moshi/moshi` is in path (script handles this)

### CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**Fix**: Use smaller batch size or shorter audio clip

### No Logging Output
**Check**:
1. Patch is applied before Moshi import
2. `instrument_cache_logging()` is called
3. stdout is not buffered (use `python -u` if needed)

---

## References

- Original issue: [RESIDUAL_JUMP_INVESTIGATION.md](RESIDUAL_JUMP_INVESTIGATION.md)
- Moshi code: `/home/alufr/ttt_tests/moshi/moshi/moshi/modules/transformer.py`
- Cache capacity: Line 94 in `moshi/models/loaders.py` → `"context": 3000`
