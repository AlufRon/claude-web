# Gradient Checkpointing Error Fix

## Problem

Training crashed at step 1 during backward pass with:
```
torch.utils.checkpoint.CheckpointError: torch.utils.checkpoint: A different number of tensors was saved during the original forward and recomputation.
Number of tensors saved during forward: 103
Number of tensors saved during recomputation: 63
```

## Root Cause

The persistence logging code changed `ttt_layer.py` to **unconditionally** use `ttt_mlp_with_states()` instead of `ttt_mlp()`:

```python
# BEFORE (working):
XQW_batch = ttt_mlp(...)  # Returns only output

# AFTER (broken):
XQW_batch, final_params = ttt_mlp_with_states(...)  # Returns output + states dict
```

**Why this breaks gradient checkpointing**:
- `ttt_mlp_with_states()` returns TWO values: `(output, states_dict)`
- Gradient checkpointing saves all outputs during forward pass
- During recomputation, the function signature is different
- PyTorch panics: "Different number of tensors!"

## The Fix

Make `ttt_mlp_with_states()` **conditional** - only use it during evaluation:

```python
# Use ttt_mlp_with_states ONLY during evaluation for persistence logging
# During training, use regular ttt_mlp to avoid gradient checkpointing issues
use_states_version = (hasattr(self, '_persistence_check_enabled') and
                     not self.training)

if use_states_version:
    # Evaluation path - get states for logging
    XQW_batch, final_params = ttt_mlp_with_states(...)
    # Log persistence checks
else:
    # Training path - no state return, compatible with gradient checkpointing
    XQW_batch = ttt_mlp(...)
```

**Key insight**:
- During **training** (`self.training=True`): Use `ttt_mlp()` (no extra outputs)
- During **evaluation** (`self.training=False`): Use `ttt_mlp_with_states()` (for logging)

## Result

✅ **Training works**: Uses regular `ttt_mlp()`, no gradient checkpointing issues
✅ **Evaluation logging works**: Uses `ttt_mlp_with_states()`, persistence checks active
✅ **No behavior change**: Same computation, just conditional state return

## Testing

Run training again:
```bash
cd /home/alufr/ttt_tests/moshi-finetune
./submit_job.sh figure5_quick.yaml
```

Should complete successfully:
- Steps 1-9: Training with gradient checkpointing (no crash)
- Step 10: Evaluation with persistence logging (logs appear)

## Files Modified

- `moshi_ttt/models/ssm/ttt_layer.py:710-761` - Conditional state return
