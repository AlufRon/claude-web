# Persistence Logging Detection Fix

## Problem

After adding persistence verification logging to TTT layers, the log showed:
```
✅ Enabled persistence logging for 0 TTT layers
```

This meant the `enable_ttt_persistence_logging()` function wasn't finding any TTT layers, so no persistence logs would appear.

## Root Cause

**Wrong layer detection logic** in `finetune/librilight_simple.py:46-59`:

```python
# BEFORE (WRONG):
for layer in model.transformer.layers:
    if hasattr(layer, 'ttt_layer'):  # ❌ This attribute doesn't exist!
        if hasattr(layer.ttt_layer, 'ttt'):
            layer.ttt_layer.ttt._persistence_check_enabled = True
```

**The actual layer hierarchy** after TTT integration is:

```
model.transformer.layers[i]  # When i is a TTT-enabled layer
└── HybridStreamingTransformerLayer
    └── seq_modeling_block (HybridSeqModelingBlock)
        └── ttt_layer (TTTWrapper)
            └── ttt (TTTMLP instance)  ← Need to set flag HERE
```

**Why the old code failed**:
- Checked `layer.ttt_layer` directly
- But `HybridStreamingTransformerLayer` doesn't have `ttt_layer` attribute
- It has `seq_modeling_block` which contains `ttt_layer`
- So the check always failed, never finding TTT layers

## The Fix

**Corrected layer detection** in `finetune/librilight_simple.py:46-69`:

```python
# AFTER (CORRECT):
for layer in model.transformer.layers:
    # Check for HybridStreamingTransformerLayer
    if hasattr(layer, 'seq_modeling_block'):
        # Check for HybridSeqModelingBlock with TTT
        if hasattr(layer.seq_modeling_block, 'ttt_layer'):
            # Check for TTTWrapper with actual TTT instance
            if hasattr(layer.seq_modeling_block.ttt_layer, 'ttt'):
                # Set flag on the actual TTTMLP instance
                layer.seq_modeling_block.ttt_layer.ttt._persistence_check_enabled = True
                enabled_count += 1
```

**Why this works**:
1. Check `layer.seq_modeling_block` - exists on `HybridStreamingTransformerLayer` ✅
2. Check `seq_modeling_block.ttt_layer` - exists on `HybridSeqModelingBlock` ✅
3. Check `ttt_layer.ttt` - exists on `TTTWrapper` ✅
4. Set `ttt._persistence_check_enabled = True` on the actual `TTTMLP` instance ✅

## How TTT Integration Works

When `apply_ttt_to_model()` is called in `finetune/ttt_integration.py:104-206`:

```python
for layer_idx in layer_indices:
    original_layer = transformer_layers[layer_idx]  # StreamingTransformerLayer

    # Wrap it in HybridStreamingTransformerLayer
    hybrid_layer = HybridStreamingTransformerLayer(
        original_layer,  # Stored in seq_modeling_block.original_layer
        ttt_config,
        persistent_states,
        layer_idx
    )

    # Replace the original layer
    transformer_layers[layer_idx] = hybrid_layer
```

Inside `HybridStreamingTransformerLayer.__init__()`:

```python
# Create hybrid seq modeling block (attention + TTT)
self.seq_modeling_block = HybridSeqModelingBlock(
    original_layer,
    ttt_config,
    persistent_states,
    layer_id
)
```

Inside `HybridSeqModelingBlock.__init__()`:

```python
# Create Video-DiT TTT layer using TTTWrapper
from .models.ssm.ttt_layer import TTTWrapper
self.ttt_layer = TTTWrapper(ttt_config)
```

Inside `TTTWrapper.__init__()`:

```python
# Create the actual TTT instance
self.ttt = TTTMLP(ttt_config)
```

## Expected Result

After this fix, when you run training with TTT enabled (layers 29, 30, 31), you should see:

```
✅ Enabled persistence logging for 3 TTT layers
```

Then during evaluation, you'll see the persistence verification logs:

```
[TTT-PERSIST-CHECK] Layer 29 Token 0: W1_hash=123456, W2_hash=789012
[TTT-PERSIST-CHECK] Layer 29 Token 1: W1_hash=123456, W2_hash=789012  ← Same? Reset!
[TTT-PERSIST-CHECK] Layer 29 Token 1: W1_hash=234567, W2_hash=890123  ← Different? Persisting! ✅
```

## Files Modified

- `finetune/librilight_simple.py:46-69` - Fixed layer detection logic

## Next Steps

Run training again to see:
1. "✅ Enabled persistence logging for 3 TTT layers" (not 0)
2. Actual persistence logs showing W1/W2 hash values
3. Determine if weights persist or reset by comparing hash values across tokens
