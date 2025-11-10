# Figure 5 Diagnostic Fix - Summary

## Problem
Figure 5 diagnostics were not collecting data during inference, showing:
```
⚠️  No Figure 5 data collected
```

## Root Cause
The multi-layer TTT-MLP implementation (`compute_multi_layer_mini_batch`) was missing the Figure 5 logging code that existed in the single-layer version. While `stream_position` tracking was already in place and being passed correctly, the actual loss computation and logging calls were never added to the multi-layer code path.

## Solution
Added Figure 5 logging to `compute_multi_layer_mini_batch()` in `/home/alufr/ttt_tests/moshi-finetune/moshi_ttt/models/ssm/ops/ttt_mlp.py`:

### Changes Made:

1. **Computed l0 and lprev (BEFORE gradient update):**
   - `l0`: Loss with frozen initial weights (W₀)
   - `lprev`: Loss with current weights before update (Wₜ₋₁)
   - Uses multi-layer forward pass through all N layers

2. **Computed lafter (AFTER gradient update):**
   - `lafter`: Loss with updated weights (Wₜ)
   - Computed after gradient descent step completes

3. **Added logging call:**
   - `_fig5_log(layer_id, pos_t, l0.item(), lprev.item(), lafter.item())`
   - Logs all three losses for each token position

### How Position Tracking Works:
- `stream_position` counter in `TTTMLPMultiLayer` class tracks absolute token position
- Increments by `num_mini_batch` after each forward pass
- Passed as `stream_pos_base` parameter to `ttt_mlp_multi_layer()`
- Converted to per-token indices: `torch.arange(stream_pos_base, stream_pos_base + num_mini_batches)`
- Each mini-batch iteration extracts its `pos_t` from the inputs dict

### Expected Behavior After Fix:
```
📊 Figure 5 diagnostic enabled (max_T=2048, layers=[29, 30, 31])
[Figure 5] Logging enabled: max_T=2048
...
📊 Generating Figure 5 diagnostic report...
✅ Generated diagnostic report in figure5_diagnostics/
```

The diagnostic will show:
- **Blue curve** (l0): Loss with frozen initial weights
- **Orange curve** (lprev): Loss before each update  
- **Green curve** (lafter): Loss after each update
- **Expected healthy**: Blue > Orange > Green (decreasing)
- **Expected for failed TTT**: High losses, minimal improvement, wrong ordering

## Files Modified:
1. `/home/alufr/ttt_tests/moshi-finetune/moshi_ttt/models/ssm/ops/ttt_mlp.py`
   - Added Figure 5 logging to `compute_multi_layer_mini_batch()`
   - ~40 lines of new code for l0, lprev, lafter computation

## Testing:
Run inference again:
```bash
cd /home/alufr/ttt_tests/moshi-finetune
./submit_inference.sh
```

Check for Figure 5 output:
```bash
ls -la figure5_diagnostics/
cat figure5_diagnostics/ttt_diagnostic_report.txt
```

The diagnostic will reveal why TTT training failed (likely: huge losses >10, minimal improvement <0.001, high violation rate >50%).
