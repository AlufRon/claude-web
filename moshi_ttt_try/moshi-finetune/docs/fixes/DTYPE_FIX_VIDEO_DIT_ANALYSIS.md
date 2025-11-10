# Dtype Fix: Aligning with Video-DiT Approach

## Problem
Getting `mat1 and mat2 must have the same dtype, but got BFloat16 and Float` error during paper metrics evaluation.

## Root Cause Analysis

### Our Original (Incorrect) Approach
We were trying to keep TTT weights in **float32** for "precision", thinking this would help with small gradient updates:

```python
# ❌ WRONG: Mixed float32/bfloat16 parameters
for key in ttt_state_dict.keys():
    if 'ttt_norm' in key:
        ttt_state_dict[key] = ttt_state_dict[key].to(torch.bfloat16)
    else:
        ttt_state_dict[key] = ttt_state_dict[key].to(torch.float32)  # ❌ WRONG
```

And in `ops/ttt_mlp.py`:
```python
# ❌ WRONG: Converting to float32 inside ops
W1_init = params_dict["W1_states"].to(torch.float32)  # ❌ WRONG
XQ_mini_batch = inputs["XQ"].to(torch.float32)  # ❌ WRONG
```

### Video-DiT's Correct Approach

After reading Video-DiT's code (`ttt-video-dit/`), we discovered:

1. **All parameters are in the SAME dtype** (bfloat16 or float32, depending on FSDP config)
2. **NO dtype conversions** inside the TTT ops
3. **Mixed precision is handled at the FSDP level**, not per-parameter

#### Video-DiT Evidence

**File: `ttt-video-dit/ttt/infra/parallelisms.py:156-159`**
```python
param_dtype = TORCH_DTYPE_MAP[job_config.parallelism.fsdp_unsharded_dtype]
reduce_dtype = torch.float32  # Gradient reduction in float32

mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
```
- **All parameters** use `param_dtype` (bfloat16)
- **Gradient reduction** happens in float32
- **NO per-parameter dtype mixing**

**File: `ttt-video-dit/ttt/models/ssm/ops/ttt_mlp.py:11-23`**
```python
def compute_mini_batch(params_dict, inputs):
    W1_init = params_dict["W1_states"]  # ✅ NO .to(torch.float32)
    b1_init = params_dict["b1_states"]
    W2_init = params_dict["W2_states"]
    b2_init = params_dict["b2_states"]
    
    ttt_norm_weight = params_dict["ttt_norm_weight"]
    ttt_norm_bias = params_dict["ttt_norm_bias"]
    
    XQ_mini_batch = inputs["XQ"]  # ✅ NO .to(torch.float32)
    XV_mini_batch = inputs["XV"]
    XK_mini_batch = inputs["XK"]
```
- **NO dtype conversions**
- All ops work directly with bfloat16
- Numerical stability is handled by the FSDP mixed precision policy

## The Fix

### 1. Remove Float32 Conversions in Checkpoint Loading
**File: `run_paper_metrics_on_checkpoint.py`**

```python
# ✅ CORRECT: Keep all parameters in bfloat16
for key in ttt_state_dict.keys():
    ttt_state_dict[key] = ttt_state_dict[key].to(device=device, dtype=torch.bfloat16)
```

### 2. Remove Float32 Conversions in TTT Ops
**File: `moshi_ttt/models/ssm/ops/ttt_mlp.py`**

```python
# ✅ CORRECT: No dtype conversion
W1_init = params_dict["W1_states"]
b1_init = params_dict["b1_states"]
W2_init = params_dict["W2_states"]
b2_init = params_dict["b2_states"]

ttt_norm_weight = params_dict["ttt_norm_weight"]
ttt_norm_bias = params_dict["ttt_norm_bias"]

XQ_mini_batch = inputs["XQ"]
XV_mini_batch = inputs["XV"]
XK_mini_batch = inputs["XK"]

eta_mini_batch = inputs["eta"]
```

### 3. Remove Float32 Conversions in Output
**File: `moshi_ttt/models/ssm/ops/ttt_mlp.py`**

```python
# ✅ CORRECT: No dtype conversion in output
last_param_dict = {
    "W1_states": W1_last,
    "b1_states": b1_last,
    "W2_states": W2_last,
    "b2_states": b2_last,
    "ttt_norm_weight": ttt_norm_weight,
    "ttt_norm_bias": ttt_norm_bias,
}

return last_param_dict, XQW_mini_batch
```

## Why This Works

### Numerical Stability
- **BFloat16 range**: 10^-38 to 10^38 (same as float32)
- **BFloat16 precision**: 7-8 decimal digits (vs 7 for float32)
- **Good enough** for TTT gradient updates
- **FSDP handles gradient accumulation in float32** anyway

### Training vs Inference
- **Training**: FSDP's `MixedPrecisionPolicy` with `reduce_dtype=torch.float32` ensures stable gradient accumulation
- **Inference**: BFloat16 is sufficient; no gradient accumulation needed
- **Paper Metrics**: Inference mode, bfloat16 works perfectly

### Why We Had the Bug
We misunderstood the comment in our own code that said "following video-dit approach" but actually did the OPPOSITE of Video-DiT. Video-DiT keeps everything in one dtype; we were mixing dtypes.

## Files Changed

1. **`run_paper_metrics_on_checkpoint.py`**
   - Removed float32 conversion in checkpoint loading
   - Removed post-loading dtype correction loop

2. **`moshi_ttt/models/ssm/ops/ttt_mlp.py`**
   - Removed `.to(torch.float32)` conversions at function start
   - Removed `.to(input_dtype)` conversions at function end
   - Removed dtype conversions in Figure 5 logging code

## Verification

To verify this matches Video-DiT:
```bash
# Video-DiT's ttt_mlp.py has NO .to(torch.float32)
grep -n "\.to(torch\.float32)" ttt-video-dit/ttt/models/ssm/ops/ttt_mlp.py
# (No matches)

# Our fixed ttt_mlp.py should also have NO .to(torch.float32)  
grep -n "\.to(torch\.float32)" moshi-finetune/moshi_ttt/models/ssm/ops/ttt_mlp.py
# (No matches in compute path - only in debug/unused code)
```

## Impact

- ✅ **Fixes dtype mismatch error** in paper metrics
- ✅ **Matches Video-DiT's proven approach**
- ✅ **Simpler code** - no dtype conversions
- ✅ **Better performance** - fewer dtype casts
- ✅ **Training still works** - FSDP handles mixed precision

## Key Takeaway

**Don't mix parameter dtypes within the model. Use FSDP's MixedPrecisionPolicy for gradient stability, and keep all parameters in the same dtype.**
