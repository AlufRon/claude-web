# TTT Persistence Verification - Quick Start

## What This Does

Adds **clean, precise logging** to check if TTT weights persist during streaming evaluation or reset every token.

## Quick Test (30 seconds)

```bash
cd /home/alufr/ttt_tests/moshi-finetune
conda activate moshi_ttt_fixed
python test_persistence_logging.py
```

## What To Look For

### ❌ Bug: Weights Reset (Current Hypothesis)
```
[TTT-PERSIST-CHECK] Layer 29 Token 0: W1_hash=123456, W2_hash=789012
[TTT-PERSIST-CHECK] Layer 29 Token 1: W1_hash=123456, W2_hash=789012  ← SAME!
[TTT-PERSIST-CHECK] Layer 29 Token 2: W1_hash=123456, W2_hash=789012  ← SAME!
```
**Diagnosis**: Hashes NEVER change → weights reset every token → **BUG CONFIRMED**

### ✅ Working: Weights Persist
```
[TTT-PERSIST-CHECK] Layer 29 Token 0: W1_hash=123456, W2_hash=789012
[TTT-PERSIST-CHECK] Layer 29 Token 1: W1_hash=456789, W2_hash=012345  ← DIFFERENT!
[TTT-PERSIST-CHECK] Layer 29 Token 2: W1_hash=789012, W2_hash=345678  ← DIFFERENT!
```
**Diagnosis**: Hashes DO change → weights persist → **WORKING AS EXPECTED**

## Additional Logs

You'll also see:

1. **`[TTT-INNER-UPDATE]`**: Confirms TTT updates weights within a single forward pass
   ```
   [TTT-INNER-UPDATE] Layer 29 Position 0: W1_change=0.00420000, W2_change=0.00350000
   ```

2. **`[TTT-SCAN-OUTPUT]`**: Shows scan() returns updated weights (but may not persist)
   ```
   [TTT-SCAN-OUTPUT] Layer 29 Token 0: W1_change=0.00420000 (scan returned, NOT PERSISTING)
   ```

## What Was Changed

- **ttt_layer.py**: Added hash logging before/after TTT
- **ttt_mlp.py**: Added inner update tracking
- **librilight_simple.py**: Enable logging for TTT layers
- **test_persistence_logging.py**: Quick test script

All changes are **read-only logging** - no behavior modifications.

## Full Evaluation

To test with full LibriLight evaluation:
```bash
sbatch finetune/scripts/slurm/run_paper_metrics.sh
```

Check logs for `[TTT-PERSIST-CHECK]` patterns.

## See Full Details

Read `TTT_PERSISTENCE_VERIFICATION_PLAN.md` for:
- Detailed problem analysis
- Root cause hypothesis
- Fix options (if bug confirmed)
- Expected impact on LibriLight metrics
