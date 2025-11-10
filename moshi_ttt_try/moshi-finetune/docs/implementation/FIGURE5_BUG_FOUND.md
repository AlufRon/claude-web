# 🔥 CRITICAL BUG FOUND IN FIGURE 5 LOGGING

## The Bug

**File**: `moshi_ttt/models/ssm/ops/ttt_mlp.py`
**Lines**: 184-196

```python
# For Figure 5: add frozen W₀ and position tracking if enabled
if _inner_fig5_enabled and layer_id is not None:
    # Detach frozen initial weights (W₀) for Figure 5 logging
    num_mini_batches = inputs["XK"].shape[0]
    # Weights already have appropriate batch dimensions, just expand them
    inputs["W1_0"] = W1_init.detach().expand(num_mini_batches, -1, -1, -1, -1)  # BUG!
    inputs["b1_0"] = b1_init.detach().expand(num_mini_batches, -1, -1, -1, -1)  # BUG!
    inputs["W2_0"] = W2_init.detach().expand(num_mini_batches, -1, -1, -1, -1)  # BUG!
    inputs["b2_0"] = b2_init.detach().expand(num_mini_batches, -1, -1, -1, -1)  # BUG!
```

## The Problem

**`W1_init` is NOT the frozen initial weights W₀!**

`W1_init` comes from line 157:
```python
W1_init = params_dict["W1_states"].to(torch.float32)
```

And `params_dict["W1_states"]` gets **UPDATED** by `scan()` as it processes mini-batches!

So what we're calling "l0" (loss with frozen W₀) is actually:
- **Position 0**: Uses W₀ (correct)
- **Position 1**: Uses W₁ (wrong! should use W₀)
- **Position 2**: Uses W₂ (wrong! should use W₀)
- **Position t**: Uses Wₜ (wrong! should use W₀)

**This explains why the "frozen" W₀ loss changes over the sequence!**

From `figure5_stats_librilight.json`:
```json
"layer_29": {
  "initial_W0_loss": 1.972,  // Actually l(W₀; x₀) ✓
  "final_W0_loss": 2.005,    // Actually l(W₂₀₄₇; x₂₀₄₇) ✗ Should be l(W₀; x₂₀₄₇)
}
```

## The Fix

We need to **freeze W₀ at the very beginning** before entering the scan loop:

```python
def ttt_mlp(...):
    init_params_dict = {
        "W1_states": W1_init,
        "b1_states": b1_init,
        "W2_states": W2_init,
        "b2_states": b2_init,
        ...
    }

    # ✅ FREEZE INITIAL WEIGHTS BEFORE SCAN
    if _inner_fig5_enabled and layer_id is not None:
        # Clone and freeze W₀ (these will NEVER change)
        W1_frozen = W1_init.detach().clone()
        b1_frozen = b1_init.detach().clone()
        W2_frozen = W2_init.detach().clone()
        b2_frozen = b2_init.detach().clone()

    inputs = {...}
    inputs = tree_map(lambda x: x.permute(2, 0, 1, 3, 4), inputs)

    if _inner_fig5_enabled and layer_id is not None:
        num_mini_batches = inputs["XK"].shape[0]
        # ✅ Use the FROZEN weights for all mini-batches
        inputs["W1_0"] = W1_frozen.expand(num_mini_batches, -1, -1, -1, -1)
        inputs["b1_0"] = b1_frozen.expand(num_mini_batches, -1, -1, -1, -1)
        inputs["W2_0"] = W2_frozen.expand(num_mini_batches, -1, -1, -1, -1)
        inputs["b2_0"] = b2_frozen.expand(num_mini_batches, -1, -1, -1, -1)
        ...
```

## What This Means for the 99% "Improvement"

The 99% improvement we saw is **WRONG** because:

```
Current calculation:
improvement = (l0_final - lafter_final) / l0_final
            = (l(W₂₀₄₇; x₂₀₄₇) - l(W₂₀₄₇_updated; x₂₀₄₇)) / l(W₂₀₄₇; x₂₀₄₇)
            = (2.0 - 0.01) / 2.0
            = 99.5%

This is comparing:
- W₂₀₄₇ (weights BEFORE processing x₂₀₄₇)
- W₂₀₄₇_updated (weights AFTER processing x₂₀₄₇)

Which is just measuring the immediate gradient step benefit!
```

**What we SHOULD be measuring:**

```
Correct calculation:
improvement = (l(W₀; x₂₀₄₇) - l(W₂₀₄₇_updated; x₂₀₄₇)) / l(W₀; x₂₀₄₇)

This compares:
- W₀ (FROZEN initial weights)
- W₂₀₄₇_updated (weights after processing 2047 tokens)

This tells us: "How much better is the adapted model vs the un-adapted model on token 2047?"
```

## The Real Question

**If W₀ loss changes from 1.972 → 2.005, what's actually changing?**

Since W₀ should be frozen, the only thing that can change is the **input data** (X1, reconstruction_target).

But the input data is SUPPOSED to change - each token is different!

So the "frozen" W₀ loss varying across positions is **EXPECTED** - it just means different tokens have different inherent difficulty.

**The bug is that we're not using frozen W₀ at all - we're using Wₜ!**

## Impact Analysis

### What Figure 5 is Currently Measuring

1. **"l0" (blue line)**: Actually l(Wₜ₋₁; xₜ) - loss with weights from previous token
2. **"lprev" (orange line)**: l(Wₜ₋₁; xₜ) - loss before gradient step
3. **"lafter" (green line)**: l(Wₜ; xₜ) - loss after gradient step

**Wait... "l0" and "lprev" are THE SAME!**

That's why at position 0 they're identical:
```
l0 (frozen weights): 1.979173
lprev (current weights): 1.979173
```

At position 0, both use W₀ because no updates have happened yet.

But then they diverge because:
- l0 (supposedly W₀) actually tracks Wₜ₋₁
- lprev explicitly uses Wₜ₋₁

So they should be identical at ALL positions!

**Let me check the actual plot to see if this is true...**

If l0 and lprev are nearly identical throughout, that confirms the bug.

### What Figure 5 SHOULD Be Measuring

1. **l0 (blue)**: l(W₀; xₜ) - how well do frozen initial weights predict each token?
   - Should vary based on token difficulty
   - Should NOT trend downward (W₀ never changes!)

2. **lprev (orange)**: l(Wₜ₋₁; xₜ) - how well do accumulated updates predict current token?
   - Should trend downward as model adapts
   - Measures cumulative learning benefit

3. **lafter (green)**: l(Wₜ; xₜ) - after one more gradient step
   - Should be below lprev (immediate gradient benefit)
   - Shows one-step improvement

**Gap between blue and orange = cumulative TTT learning**
**Gap between orange and green = single gradient step benefit**

## Next Steps

1. ✅ **Confirm the bug** by checking if l0 ≈ lprev in the actual plot
2. ⚠️ **Fix the bug** by freezing W₀ before the scan loop
3. 🔬 **Re-run evaluation** to get correct Figure 5
4. 📊 **Analyze true improvement** with correct frozen W₀

## Expected Results After Fix

With the fix, we should see:

- **l0 (blue)**: Relatively flat or slightly varying (different token difficulties)
- **lprev (orange)**: Trending downward (cumulative learning)
- **lafter (green)**: Below lprev, also trending down

**The gap between blue and orange will show the TRUE TTT benefit!**

Currently, the 99% "improvement" is mostly just showing that one gradient step helps a lot, not that cumulative TTT adaptation is working.

---

## Summary

**Bug**: W₀ not actually frozen - using Wₜ₋₁ instead
**Impact**: Figure 5 shows wrong metrics, 99% "improvement" is misleading
**Fix**: Clone and freeze W₀ before scan loop
**Priority**: HIGH - this invalidates our current Figure 5 results
