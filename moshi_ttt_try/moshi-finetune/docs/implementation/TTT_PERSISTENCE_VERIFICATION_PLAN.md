# TTT Weight Persistence Verification

## Problem Statement

From LOG_ANALYSIS_7237224.md, we identified a critical mystery:
- **Figure 5** shows 99% improvement in TTT reconstruction loss over 2047 tokens
- **LibriLight** shows no improvement (loss increases slightly)
- **Hypothesis**: TTT weights may be resetting every token instead of persisting

## Root Cause Analysis

After code review, we identified the likely issue in `ttt_layer.py:676-726`:

```python
def ttt(self, inputs, layer_id=None):
    # CREATES FRESH STATES FROM self.W1/W2 EVERY FORWARD PASS
    W1_states = torch.tile(self.W1.unsqueeze(0), dims=(B, 1, 1, 1))
    b1_states = torch.tile(self.b1.unsqueeze(0), dims=(B, 1, 1, 1))
    # ...

    # Calls scan() which returns updated weights
    XQW_batch = ttt_mlp(...)

    # BUT: Updated weights from scan() are NEVER saved back to self.W1/W2
    # Next forward pass starts fresh again!
```

**The Problem**:
- `scan()` computes updated weights W₁, W₂ within a single forward pass
- These updated weights are **discarded** after the forward pass
- Next token starts with the same initial `self.W1`, `self.W2` again
- **No persistence between tokens during streaming evaluation!**

## Solution: Add Diagnostic Logging

We added **4 strategic log points** to verify this hypothesis without cluttering logs:

### 1. Weight Hash Logging (BEFORE TTT) - `ttt_layer.py:681-685`
```python
# Log hash of self.W1/W2 BEFORE creating states
if layer_id is not None and hasattr(self, '_persistence_check_enabled'):
    W1_hash = torch.sum(self.W1.data * 1000000).item() % 1000000
    W2_hash = torch.sum(self.W2.data * 1000000).item() % 1000000
    logger.info(f"[TTT-PERSIST-CHECK] Layer {layer_id} Token {self.stream_position}: W1_hash={W1_hash:.0f}, W2_hash={W2_hash:.0f}")
```

**Purpose**: Shows if `self.W1`/`self.W2` parameters ever change between forward passes.

### 2. scan() Output Logging (AFTER TTT) - `ttt_layer.py:731-735`
```python
# Log weight changes returned by scan()
if layer_id is not None and hasattr(self, '_persistence_check_enabled'):
    W1_change = (final_params["W1_states"] - W1_states).abs().max().item()
    W2_change = (final_params["W2_states"] - W2_states).abs().max().item()
    logger.info(f"[TTT-SCAN-OUTPUT] Layer {layer_id} Token {self.stream_position}: W1_change={W1_change:.8f}, W2_change={W2_change:.8f} (scan returned updated weights, BUT NOT PERSISTING TO self.W1/W2)")
```

**Purpose**: Confirms `scan()` IS computing updates (just not persisting them).

### 3. Inner Update Logging - `ttt_mlp.py:286-295`
```python
# Log weight changes during compute_mini_batch (once per layer)
if layer_id is not None and layer_id not in _persistence_check_logged_layers:
    _persistence_check_logged_layers.add(layer_id)
    w1_change = (W1_last - W1_init).abs().max().item()
    w2_change = (W2_last - W2_init).abs().max().item()
    print(f"[TTT-INNER-UPDATE] Layer {layer_id} Position {pos_t}: W1_change={w1_change:.8f}, W2_change={w2_change:.8f} (TTT updating WITHIN forward pass)")
```

**Purpose**: Shows TTT IS working within a single forward pass.

### 4. Enable Persistence Logging - `librilight_simple.py:46-59`
```python
def enable_ttt_persistence_logging(model):
    """Enable persistence check logging for all TTT layers."""
    enabled_count = 0
    for layer in model.transformer.layers:
        if hasattr(layer, 'ttt_layer'):
            if hasattr(layer.ttt_layer, 'ttt'):
                layer.ttt_layer.ttt._persistence_check_enabled = True
                enabled_count += 1
    logger.info(f"✅ Enabled persistence logging for {enabled_count} TTT layers")
    return enabled_count
```

**Purpose**: Activates logging during evaluation.

## Expected Output

### Scenario A: Weights Reset Every Token (BUG - Current State)
```
[TTT-PERSIST-CHECK] Layer 29 Token 0: W1_hash=123456, W2_hash=789012
[TTT-INNER-UPDATE] Layer 29 Position 0: W1_change=0.00420000, W2_change=0.00350000
[TTT-SCAN-OUTPUT] Layer 29 Token 0: W1_change=0.00420000, W2_change=0.00350000
[TTT-PERSIST-CHECK] Layer 29 Token 1: W1_hash=123456, W2_hash=789012  ← SAME (reset!)
[TTT-PERSIST-CHECK] Layer 29 Token 2: W1_hash=123456, W2_hash=789012  ← SAME (reset!)
[TTT-PERSIST-CHECK] Layer 29 Token 3: W1_hash=123456, W2_hash=789012  ← SAME (reset!)
```

**Interpretation**:
- ❌ W1_hash/W2_hash NEVER change
- ❌ Every token starts with same initial weights
- ❌ scan() computes updates but they're discarded
- ❌ Explains why LibriLight loss doesn't improve!

### Scenario B: Weights Persist (WORKING - Expected)
```
[TTT-PERSIST-CHECK] Layer 29 Token 0: W1_hash=123456, W2_hash=789012
[TTT-INNER-UPDATE] Layer 29 Position 0: W1_change=0.00420000, W2_change=0.00350000
[TTT-SCAN-OUTPUT] Layer 29 Token 0: W1_change=0.00420000, W2_change=0.00350000
[TTT-PERSIST-CHECK] Layer 29 Token 1: W1_hash=456789, W2_hash=012345  ← DIFFERENT!
[TTT-PERSIST-CHECK] Layer 29 Token 2: W1_hash=789012, W2_hash=345678  ← DIFFERENT!
[TTT-PERSIST-CHECK] Layer 29 Token 3: W1_hash=234567, W2_hash=890123  ← DIFFERENT!
```

**Interpretation**:
- ✅ W1_hash/W2_hash DO change between tokens
- ✅ Each token starts with previous token's updated weights
- ✅ scan() updates are persisted
- ✅ Should see LibriLight loss improve over sequence

## Testing

### Quick Test (50 tokens - 30 seconds)
```bash
cd /home/alufr/ttt_tests/moshi-finetune
conda activate moshi_ttt_fixed
python test_persistence_logging.py
```

This script:
1. Loads the TTT-Moshi model
2. Processes just 50 tokens from LibriLight
3. Shows persistence logging in action
4. Takes ~30 seconds

### Full Evaluation (3000 tokens - 10 minutes)
Use the existing paper metrics script:
```bash
sbatch finetune/scripts/slurm/run_paper_metrics.sh
```

Look for `[TTT-PERSIST-CHECK]` logs in the output.

## Files Modified

### 1. `moshi_ttt/models/ssm/ttt_layer.py`
- Added weight hash logging before TTT (lines 681-685)
- Changed to use `ttt_mlp_with_states()` instead of `ttt_mlp()` (line 712)
- Added scan() output logging (lines 731-735)

### 2. `moshi_ttt/models/ssm/ops/ttt_mlp.py`
- Added global variable `_persistence_check_logged_layers` (line 48)
- Added inner update logging in `compute_mini_batch()` (lines 286-295)

### 3. `finetune/librilight_simple.py`
- Added `enable_ttt_persistence_logging()` function (lines 46-59)
- Call to enable logging during evaluation (lines 101-103)

### 4. `test_persistence_logging.py` (NEW)
- Quick test script to verify logging works
- Processes 50 tokens in ~30 seconds
- Clear pass/fail interpretation

## Next Steps

### Step 1: Verify the Bug
Run `test_persistence_logging.py` and check if hashes change:
- **Hashes SAME** → Bug confirmed, weights reset every token
- **Hashes DIFFERENT** → Weights persist, bug is elsewhere

### Step 2: If Bug Confirmed - Fix Options

#### Option A: Store and restore states in TTTMLP
```python
def ttt(self, inputs, layer_id=None):
    # Use stored states if available
    if hasattr(self, '_W1_states_persistent'):
        W1_states = self._W1_states_persistent
        b1_states = self._b1_states_persistent
        W2_states = self._W2_states_persistent
        b2_states = self._b2_states_persistent
    else:
        W1_states = torch.tile(self.W1.unsqueeze(0), dims=(B, 1, 1, 1))
        # ... initialize others

    # Run TTT
    XQW_batch, final_params = ttt_mlp_with_states(...)

    # CRITICAL: Store updated states for next forward
    self._W1_states_persistent = final_params["W1_states"].detach()
    self._b1_states_persistent = final_params["b1_states"].detach()
    self._W2_states_persistent = final_params["W2_states"].detach()
    self._b2_states_persistent = final_params["b2_states"].detach()

    return XQW_batch
```

#### Option B: Copy states back to parameters (Video-DiT style)
```python
# After scan() returns final_params
if self.training or persistent_states_enabled:
    # Copy updated states back to parameters
    self.W1.data.copy_(final_params["W1_states"][0])  # Remove batch dim
    self.b1.data.copy_(final_params["b1_states"][0])
    self.W2.data.copy_(final_params["W2_states"][0])
    self.b2.data.copy_(final_params["b2_states"][0])
```

### Step 3: Re-evaluate
After fix, run full LibriLight evaluation:
- Should see loss DECREASE over sequence
- Figure 5 and LibriLight should align
- Confirms TTT adaptation is working end-to-end

## Impact

If weights ARE resetting (Scenario A):
- **Explains LOG_ANALYSIS mystery**: TTT works within each forward, but resets between tokens
- **Explains flat LibriLight loss**: Model never benefits from adaptation
- **Simple fix**: Store and restore states between forward passes
- **Expected improvement**: Loss should decrease over sequence like Figure 5 shows

## Validation Metrics

After fix, we should see:
1. ✅ Weight hashes change between tokens
2. ✅ LibriLight loss decreases over sequence (negative slope)
3. ✅ Figure 5 and LibriLight metrics align
4. ✅ Later positions have lower loss than early positions

---

## Summary

We've added **minimal, clean logging** to definitively answer:
> **Do TTT weights persist between streaming tokens, or reset every time?**

Run `test_persistence_logging.py` to find out in 30 seconds!
