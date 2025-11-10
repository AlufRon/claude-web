# CRITICAL BUG: TTT Layer Attention Context Never Set

**Date**: 2025-11-01
**Severity**: CRITICAL
**Status**: ✅ FIXED
**Impact**: TTT layers were using default attention context instead of intended restricted context

---

## Bug Summary

**Issue**: Attribute name mismatch prevented TTT layer attention contexts from being set during both training and inference.

**Root Cause**: Code looked for `layer.wrapped_layer.self_attn` but the actual attribute is `layer.original_layer.self_attn`.

**Result**: All TTT layers used default Moshi context (3000 tokens) regardless of configuration.

---

## Detection

From inference log:
```
INFO:__main__:✅ Set context: 3000 → TTT layers: 50 (0 layers), Non-TTT: 100 (21 layers)
```

**Red flag**: `0 layers` detected as TTT despite having TTT-enabled checkpoint!

---

## Root Cause Analysis

### Incorrect Code (BEFORE FIX)

**File**: `inference/run_inference_with_ttt.py:355`
```python
elif hasattr(layer, 'wrapped_layer') and hasattr(layer.wrapped_layer, 'self_attn'):
    #                    ^^^^^^^^^^^^                ^^^^^^^^^^^^
    #                    WRONG ATTRIBUTE NAME!
```

**File**: `finetune/wrapped_model.py:388`
```python
elif hasattr(layer, 'wrapped_layer') and hasattr(layer.wrapped_layer, 'self_attn'):
    #                    ^^^^^^^^^^^^                ^^^^^^^^^^^^
    #                    SAME BUG IN TRAINING!
```

### Actual Class Structure

**File**: `moshi_ttt/hybrid_layer.py:364-374`
```python
class HybridStreamingTransformerLayer(StreamingModule[_LayerState]):
    def __init__(self, original_layer: StreamingTransformerLayer, ...):
        super().__init__()

        self.original_layer = original_layer  # ← CORRECT NAME
        #    ^^^^^^^^^^^^^^
        self.layer_id = layer_id

        # Create hybrid seq modeling block
        self.seq_modeling_block = HybridSeqModelingBlock(original_layer, ...)
```

**Hierarchy**:
```
HybridStreamingTransformerLayer
  ├─ self.original_layer (StreamingTransformerLayer)  ← Correct!
  │    └─ self.self_attn
  └─ self.seq_modeling_block (HybridSeqModelingBlock)
       └─ self.original_layer (same as above)
```

### Why It Wasn't Caught Earlier

1. **Silent Failure**: The `elif` branch was never entered (attribute doesn't exist)
2. **Fallback Behavior**: Layers kept their default context (3000 tokens)
3. **No Error Message**: Python's `hasattr()` returns `False` silently
4. **Logs Were Misleading**: Showed "0 layers" but no error raised

---

## Impact Assessment

### During Training

**Configuration** (example):
```yaml
ttt:
  ttt_layer_context: 50        # Intended: 4s local attention
  non_ttt_layer_context: 100   # Intended: 8s for non-TTT
```

**Actual Behavior**:
```
✅ Set attention context: TTT layers=50 (0 layers), Non-TTT=100 (21 layers)
```

- ❌ TTT layers: Used default 3000 tokens (NOT 50!)
- ✅ Non-TTT layers: Correctly set to 100 tokens
- ❌ No differentiation between TTT and non-TTT layers

**Training Impact**:

| Intended Behavior | Actual Behavior | Consequence |
|-------------------|-----------------|-------------|
| TTT layer sees 50 tokens (4s) | TTT layer sees 3000 tokens (240s) | TTT didn't learn to be aggressive memory |
| Attention = local, TTT = global | Attention = very long, TTT = redundant | Division of labor not established |
| Strong gradient signal for TTT | Weak gradient signal | TTT weights may not have learned compression |
| Force TTT to compress history | Attention handles most context | TTT undertrained |

### During Inference

**Loaded Config**:
```python
ttt_context = ttt_config.get('ttt_layer_context', 3000)     # 50 from config
non_ttt_context = ttt_config.get('non_ttt_layer_context', 3000)  # 100 from config
```

**Actual Application**:
```
✅ Set context: 3000 → TTT layers: 50 (0 layers), Non-TTT: 100 (21 layers)
```

- ❌ TTT layers: Kept default 3000 (NOT changed to 50!)
- ✅ Non-TTT layers: Correctly changed to 100
- ❌ Train/inference mismatch (if training had been correct)

---

## The Fix

### Corrected Code (AFTER FIX)

**File**: `inference/run_inference_with_ttt.py:355`
```python
elif hasattr(layer, 'original_layer') and hasattr(layer.original_layer, 'self_attn'):
    #                    ^^^^^^^^^^^^^^                ^^^^^^^^^^^^^^
    #                    CORRECT ATTRIBUTE NAME!
    if hasattr(layer.original_layer.self_attn, 'context'):
        layer.original_layer.self_attn.context = ttt_context
        modified_ttt += 1
```

**File**: `finetune/wrapped_model.py:388`
```python
elif hasattr(layer, 'original_layer') and hasattr(layer.original_layer, 'self_attn'):
    #                    ^^^^^^^^^^^^^^                ^^^^^^^^^^^^^^
    #                    CORRECT ATTRIBUTE NAME!
    if hasattr(layer.original_layer.self_attn, 'context'):
        if ttt_context is not None:
            layer.original_layer.self_attn.context = ttt_context
            modified_ttt += 1
```

### Expected Behavior After Fix

**Training Log** (after fix):
```
✅ Set attention context: TTT layers=50 (3 layers), Non-TTT=100 (18 layers)
```

**Inference Log** (after fix):
```
✅ Set context: 3000 → TTT layers: 50 (3 layers), Non-TTT: 100 (18 layers)
```

Now TTT layers will actually get their intended context!

---

## Verification

### Test 1: Check Detection During Training

**Before Fix**:
```bash
grep "Set attention context" logs/training.log
# Output: TTT layers=50 (0 layers), Non-TTT=100 (21 layers)
```

**After Fix**:
```bash
grep "Set attention context" logs/training.log
# Output: TTT layers=50 (3 layers), Non-TTT=100 (18 layers)
```

### Test 2: Check Detection During Inference

**Before Fix**:
```bash
grep "Set context:" logs/inference.log
# Output: Set context: 3000 → TTT layers: 50 (0 layers), Non-TTT: 100 (21 layers)
```

**After Fix**:
```bash
grep "Set context:" logs/inference.log
# Output: Set context: 3000 → TTT layers: 50 (3 layers), Non-TTT: 100 (18 layers)
```

### Test 3: Verify Actual Context Values

Add debug logging:
```python
for layer_idx, layer in enumerate(model.transformer.layers):
    if hasattr(layer, 'original_layer') and hasattr(layer.original_layer, 'self_attn'):
        context = layer.original_layer.self_attn.context
        print(f"Layer {layer_idx} (TTT): context={context}")
    elif hasattr(layer, 'self_attn'):
        context = layer.self_attn.context
        print(f"Layer {layer_idx} (regular): context={context}")
```

**Expected Output** (with ttt_layers="15,16,31", ttt_context=50, non_ttt_context=100):
```
Layer 0 (regular): context=100
Layer 1 (regular): context=100
...
Layer 15 (TTT): context=50      ← TTT layer!
Layer 16 (TTT): context=50      ← TTT layer!
...
Layer 31 (TTT): context=50      ← TTT layer!
```

---

## Implications for Existing Checkpoints

### Checkpoints Trained Before Fix

**Status**: All existing checkpoints were trained with **WRONG context settings**

**What Actually Happened**:
- Config said: `ttt_layer_context: 50`
- Reality: TTT layers used context=3000 (60x larger!)
- TTT never learned aggressive compression
- TTT never learned to be the primary memory layer

**Options**:

#### Option 1: Retrain with Correct Context (RECOMMENDED)

```yaml
# Use Video-DiT-inspired aggressive settings
ttt:
  ttt_layer_context: 50          # 4s local attention for TTT
  non_ttt_layer_context: 100     # 8s for non-TTT layers
```

**Benefits**:
- TTT will learn proper division of labor
- Matches Video-DiT's proven approach
- Strong gradient signal for memory learning

#### Option 2: Continue with Large Context

```yaml
# Accept that previous training used large context
ttt:
  ttt_layer_context: 3000        # What was actually used
  non_ttt_layer_context: 3000    # What was actually used
```

**Benefits**:
- Matches what model actually learned
- No need to retrain
- But defeats the purpose of TTT (no memory layer)

#### Option 3: Fine-tune from Existing Checkpoint

```yaml
# Start from existing checkpoint, use correct context
ttt:
  ttt_layer_context: 50
  non_ttt_layer_context: 100
```

**Caveat**: Model was trained with wrong context, may need adjustment period

---

## Recommendations

### For New Training

1. ✅ **Use the fixed code** (already applied)
2. ✅ **Set aggressive TTT context** (50-100 tokens)
3. ✅ **Verify logs show non-zero TTT layers**
4. ✅ **Monitor that context is actually being applied**

### Example Config (Video-DiT-Inspired)

```yaml
duration_sec: 80  # Long sequences to test memory

ttt:
  enable: true
  layers: "middle"  # Middle 50% of layers

  # CRITICAL: These will now actually be applied!
  ttt_layer_context: 50          # 4s at 12.5 Hz (5% coverage)
  non_ttt_layer_context: 100     # 8s for non-TTT (10% coverage)

  base_lr: 0.1
  mini_batch_size: 16
  persistent_states: true
```

### For Inference

1. ✅ **Use the fixed code** (already applied)
2. ✅ **Load context from checkpoint config** (already implemented)
3. ✅ **Verify non-zero TTT layers in logs**

### Monitoring

Add to training/inference scripts:
```python
# After setting context
logger.info("=" * 80)
logger.info("CONTEXT VERIFICATION:")
ttt_count = 0
for idx, layer in enumerate(model.transformer.layers):
    if hasattr(layer, 'original_layer'):
        ctx = layer.original_layer.self_attn.context
        logger.info(f"  Layer {idx} [TTT]: context={ctx}")
        ttt_count += 1
    else:
        ctx = layer.self_attn.context
        logger.info(f"  Layer {idx} [regular]: context={ctx}")

if ttt_count == 0:
    logger.error("❌ WARNING: No TTT layers detected!")
else:
    logger.info(f"✅ Verified {ttt_count} TTT layers with correct context")
logger.info("=" * 80)
```

---

## Timeline

- **Bug Introduced**: Unknown (likely from initial implementation)
- **Bug Active**: All training runs until 2025-11-01
- **Bug Discovered**: 2025-11-01 (inference log showing "0 layers")
- **Bug Fixed**: 2025-11-01 (this commit)
- **Verification**: Pending (next training/inference run)

---

## Affected Files

✅ **Fixed**:
- `inference/run_inference_with_ttt.py:355` - Changed `wrapped_layer` → `original_layer`
- `finetune/wrapped_model.py:388` - Changed `wrapped_layer` → `original_layer`

⚠️ **To Check**:
- `evaluation/scripts/run_paper_metrics_on_checkpoint.py` - May have same bug
- Any other files that iterate over layers and check for TTT

---

## Lessons Learned

1. **Explicit is better than implicit**: Attribute names should be obvious
2. **Fail loudly**: Silent failures (hasattr returning False) are dangerous
3. **Verify logs**: "0 layers" should have raised immediate red flags
4. **Test integration thoroughly**: End-to-end tests would have caught this
5. **Log actual values**: Don't just log intent, log reality

---

## Action Items

- [x] Fix inference script
- [x] Fix training script
- [ ] Check evaluation scripts for same bug
- [ ] Add verification logging to detect "0 layers" and error out
- [ ] Add unit test to verify layer detection works
- [ ] Document expected log output in README
- [ ] Retrain models with correct context settings
- [ ] Compare performance before/after fix

---

## Conclusion

**This was a critical bug** that prevented the core TTT mechanism (limited attention + global memory via persistent states) from working as intended.

All training runs before this fix:
- ❌ Did NOT restrict TTT layer attention to intended values
- ❌ Did NOT establish proper division of labor
- ❌ May not have learned TTT as effective memory layer

**After this fix**:
- ✅ TTT layers get their intended restricted context
- ✅ Division of labor is established (attention=local, TTT=global)
- ✅ Training signals force TTT to learn compression
- ✅ Matches Video-DiT's proven architecture

**Recommendation**: Retrain with fixed code and aggressive context restrictions (50-100 tokens for TTT layers) to properly evaluate TTT's memory capabilities.
