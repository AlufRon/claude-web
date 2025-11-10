# TTT Explosion Bug - Fix v2

## 🚨 **Problem: First Fix Didn't Work**

The initial fix (clamping std) didn't work because:

1. **`@torch.compile` cached the old version**: The function was JIT-compiled, so changes weren't taking effect
2. **The clamping happened too late**: Still dividing by `(std + eps)` instead of just `std`
3. **eps was too small**: `1e-8` is too close to zero

## 🔧 **Updated Fix (v2)**

### **Changes Made**

#### 1. **Removed `@torch.compile` decorator** (line 274)
```python
# @torch.compile  # TEMPORARILY DISABLED for debugging
def ln_reconstruction_target(self, XV, XK):
```

**Why**: Forces function to recompile, ensuring changes take effect immediately.

#### 2. **Increased epsilon** (line 301)
```python
eps = 1e-6  # INCREASED from 1e-8 to 1e-6 for stability
```

**Why**: Larger safety margin for numerical stability.

#### 3. **Use `torch.maximum` instead of clamp** (line 319)
```python
std = torch.maximum(std, torch.tensor(eps, device=std.device, dtype=std.dtype))
```

**Why**: More direct - ensures `std >= eps` before division.

#### 4. **Divide by `std` directly** (line 322)
```python
XV = (XV - mean) / std  # No more + eps
```

**Why**: Since we already clamped `std >= eps`, no need to add eps again.

#### 5. **Added extensive debug logging**
- Log `(XV - XK)` range before normalization
- Log `std` statistics (min, max, mean, num below thresholds)
- This will show us exactly what's happening

### **Expected Behavior**

With these changes:
1. If `std < 1e-6`: clamped to `1e-6`
2. Division: `XV / 1e-6` gives max amplification of 1,000,000x (finite)
3. Should prevent NaN/Inf values
4. Training should proceed past backward pass

### **Debug Output to Expect**

```
🔍 [Before normalization - XV-XK]
  (XV - XK) range: [-X.XXXXXX, X.XXXXXX]
  (XV - XK) global std: X.XXXXXX

🔍 [std values before clamping]
  std range: [X.XXXXXXXXXXXX, X.XXXXXXXXXXXX]
  std mean: X.XXXXXXXXXXXX
  num std < 1e-6: XXX / XXXX
  num std < 1e-8: XXX / XXXX
```

This will tell us:
- How many std values are problematically small
- Whether clamping is actually helping
- What the actual distribution of std values looks like

## 📋 **Next Steps**

1. **Run training again** to see the new debug output
2. **Check if explosion is prevented** - output should be < 1000 instead of millions
3. **If still exploding**: Need to investigate other causes (maybe issue is elsewhere)
4. **Once working**: Re-enable `@torch.compile` for performance

## 🔍 **Alternative Hypotheses (if this doesn't work)**

If v2 still doesn't fix it:
1. **Problem might be in ttt_mlp.py**: The explosion might happen during TTT gradient updates, not in ln_reconstruction_target
2. **Problem might be in format conversion**: moshi_to_ttt_format or ttt_to_moshi_format
3. **Problem might be in XV projection**: The wv (value projection) might be producing extreme values

---

**Status**: v2 fix applied, awaiting test results
**Date**: 2025-10-10
**Files Modified**: `moshi_ttt/models/ssm/ttt_layer.py`
