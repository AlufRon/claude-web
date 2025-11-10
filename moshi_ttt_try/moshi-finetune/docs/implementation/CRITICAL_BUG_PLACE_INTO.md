# Critical Bug: place_into() Returning Wrong Argument

## 🔥 **The Bug**

In `moshi_ttt/utils.py`, the `place_into()` function was returning the WRONG argument:

```python
def place_into(target: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    """Place source into target (placeholder)"""
    return source  # BUG: Returns second argument instead of first!
```

## 🔍 **How This Caused the Explosion**

### **The Code Flow**

In `ln_reconstruction_target()` (line 283 of ttt_layer.py):

```python
XV = XV - XK  # XV now contains the difference
std = place_into(to_local(XV).std(dim=-1, keepdim=True), XV)
#                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  ^^
#                computed std (should be returned)        XV (was being returned!)
```

### **What Should Happen**

1. Compute `XV - XK`
2. Compute `std` of the result over head dimension
3. `place_into(computed_std, XV)` should return `computed_std`
4. Normalize using `(XV - mean) / (std + eps)`

### **What Actually Happened**

1. Compute `XV - XK`
2. Compute `std` of the result
3. `place_into(computed_std, XV)` **returns XV** (the difference values!)
4. Normalize using `(XV - mean) / (XV + eps)` ← **Dividing by XV itself!**

This caused:
- `std` variable to contain `(XV - XK)` values (range ±30)
- Division by `(std + eps)` where `std` could be negative or very small
- Massive amplification when `std` is close to zero
- "Negative std" values in debug output (actually XV values)

## 📊 **Evidence from Log 7204433**

Lines 181-190 show the smoking gun:

```
🔍 [STD DISTRIBUTION in ln_reconstruction_target]
  std min: -28.7500000000    ← IMPOSSIBLE: std cannot be negative!
  std max: 27.0000000000     ← This is XV range, not std range!
  std mean: -0.0067443848
  eps value: 0.0000100000
  std < eps (1e-5): 397301/786432 (50.5%)
```

The "std" values have the SAME RANGE as `XV - XK`:
- XV input: -20.25 to 17.75
- XK input: -22.62 to 19.00
- Expected `XV - XK` range: ~-43 to +40
- Actual "std" range: -28.75 to 27.00 ✓ **Matches XV - XK values!**

True std values would be:
- Always positive
- Typically in range 0.1 to 10
- Never negative

## ✅ **The Fix**

Changed line 33 in `moshi_ttt/utils.py`:

```python
def place_into(target: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    """Place target into source's tensor structure (placeholder)"""
    return target  # FIXED: was 'return source'
```

## 🎯 **Why Video-DiT Worked**

Video-DiT has the SAME call pattern:
```python
std = place_into(to_local(XV).std(dim=-1, keepdim=True), XV)
```

But their `place_into()` **correctly returns the first argument**:

```python
def place_into(local_tensor: torch.Tensor, dt: DTensor | torch.Tensor):
    if not isinstance(dt, DTensor):
        return local_tensor  # Returns first argument (correct!)
    return DTensor.from_local(local_tensor, ...)
```

## 🔬 **Verification Strategy**

### **Before Fix (Log 7204433)**
```
std min: -28.75 (impossible!)
std max: 27.00 (matches XV range)
XV output: min=-61696, max=139264 (6000x explosion)
```

### **After Fix (Expected)**
```
std min: ~0.01 (positive, reasonable)
std max: ~10.0 (positive, reasonable)
XV output: min≈-25, max≈25 (normalized + XK, ~±20 range)
```

## 📋 **Impact Analysis**

### **What This Bug Affected**

1. **All normalization**: Every call to `ln_reconstruction_target()` was broken
2. **All TTT layers**: Layers 29, 30, 31 all affected
3. **Training impossible**: Forward pass produced garbage values
4. **Backward pass failed**: NaN/Inf propagated from extreme values

### **Why Previous "Fixes" Didn't Work**

1. **eps = 1e-5**: Helped slightly, but `std` was still XV values, not actual std
2. **Parameter initialization**: Parameters were fine, problem was in computation
3. **Clamping std**: Tried to clamp what we thought was std, but it was XV!

## 🚀 **Next Steps**

- [x] Fix `place_into()` to return first argument
- [ ] Run training to verify fix works
- [ ] Expect normalized output (±20 range, not thousands)
- [ ] Verify training proceeds past backward pass
- [ ] Remove debug logging and re-enable `@torch.compile`

---

**Root Cause**: Placeholder function had incorrect implementation
**Symptom**: 6000x value explosion
**Diagnosis Time**: Multiple failed attempts with wrong hypotheses
**Actual Cause**: Single-line bug in utility function
**Lesson**: Always verify placeholder/stub implementations match the original!
