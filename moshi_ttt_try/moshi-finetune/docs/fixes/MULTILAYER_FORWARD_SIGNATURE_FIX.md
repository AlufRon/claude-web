# Multi-Layer TTT-MLP Forward Signature Fix

## 🔴 CRITICAL BUG DISCOVERED

**Job 7113935** crashed immediately during first forward pass with:
```
TypeError: TTTBase.forward() missing 1 required positional argument: 'seq_metadata'
```

## 🔍 ROOT CAUSE ANALYSIS

### The Problem

The previous fix (Job 7113897 approach) created `TTTMLPMultiLayer` directly in `hybrid_layer.py`:

```python
# BROKEN CODE - Job 7113935
from .models.ssm.ttt_layer import TTTMLPMultiLayer
self.ttt_layer = TTTMLPMultiLayer(ttt_config, use_kernel=False)
```

### Why It Failed

1. **Missing Forward Method**: `TTTMLPMultiLayer` inherits from `TTTBase` but does **NOT** override `forward()`
2. **Only Has `ttt()` Method**: The class only implements `ttt(inputs)` for the inner TTT computation
3. **Wrong Call Signature**: When `hybrid_layer.py` calls `self.ttt_layer(x_padded, seq_metadata)`:
   - It expects: `forward(hidden_states, seq_metadata)` 
   - Gets called on: `TTTBase.forward()` (which has different signature expecting RoPE freqs)
   - Result: **TypeError**

### Architecture Comparison

#### Video-DiT Architecture (models/ssm/ttt_layer.py):
```
TTTWrapper (has forward(x, seq_metadata))
    ↓
  self.ttt = TTTMLPMultiLayer (only has ttt(inputs) method)
```

#### Moshi 2-Layer Architecture (moshi_ttt/ttt_layer.py):
```
TTTMLP (has forward(hidden_states, seq_metadata))
    ↓
  Built-in ttt(inputs) method
```

### The Logs Told Us The Truth

**Job 7113935 logs showed:**
```
2025-10-09 10:45:29 - moshi_ttt.hybrid_layer - INFO - [Hybrid] Layer 27: Creating 3-layer TTT-MLP
2025-10-09 10:45:31 - moshi_ttt.models.ssm.ttt_layer - INFO - [TTT-MLP-MULTI] Total TTT parameters: 12,619,776
```

✅ **Creation succeeded** - all 5 layers initialized with multi-layer architecture

```
File "moshi_ttt/hybrid_layer.py", line 329, in _apply_ttt_processing_impl
    ttt_output = self.ttt_layer(x_padded, seq_metadata)
...
TypeError: TTTBase.forward() missing 1 required positional argument: 'seq_metadata'
```

❌ **Forward failed** - calling wrong forward signature

## ✅ THE FIX

### Use TTTWrapper Instead

`TTTWrapper` is the **correct** entry point that:
1. Has the proper `forward(x, seq_metadata)` signature
2. Internally selects `TTTMLPMultiLayer` (3+ layers) or `TTTMLP` (2 layers)
3. Handles RoPE computation automatically
4. Matches the Video-DiT architecture pattern

### Code Changes

**File: `moshi_ttt/hybrid_layer.py`** (lines 67-73)

**BEFORE (Job 7113935 - BROKEN):**
```python
# Create Video-DiT TTT layer - choose implementation based on config
num_layers = getattr(ttt_config, 'ttt_mlp_layers', 2)
if num_layers >= 3:
    # Use multi-layer implementation for 3+ layers
    logger.info(f"[Hybrid] Layer {layer_id}: Creating {num_layers}-layer TTT-MLP")
    from .models.ssm.ttt_layer import TTTMLPMultiLayer
    self.ttt_layer = TTTMLPMultiLayer(ttt_config, use_kernel=False)
else:
    # Use standard 2-layer implementation
    logger.info(f"[Hybrid] Layer {layer_id}: Creating standard 2-layer TTT-MLP")
    from .ttt_layer import TTTMLP
    self.ttt_layer = TTTMLP(ttt_config, layer_id=layer_id, use_kernel=False)
```

**AFTER (FIXED):**
```python
# Create Video-DiT TTT layer using TTTWrapper
# TTTWrapper will internally choose TTTMLPMultiLayer (3+ layers) or TTTMLP (2 layers)
num_layers = getattr(ttt_config, 'ttt_mlp_layers', 2)
logger.info(f"[Hybrid] Layer {layer_id}: Creating TTT-MLP with {num_layers} layers via TTTWrapper")
from .models.ssm.ttt_layer import TTTWrapper
self.ttt_layer = TTTWrapper(ttt_config)
```

### State Save/Restore Update

Updated to handle the wrapper by accessing the inner TTT instance:

**BEFORE:**
```python
if hasattr(self.ttt_layer, 'W1'):
    saved_state['W1'] = self.ttt_layer.W1.clone().detach()
```

**AFTER:**
```python
# Access the actual TTT instance (either directly or through wrapper)
ttt_instance = getattr(self.ttt_layer, 'ttt', self.ttt_layer)

if hasattr(ttt_instance, 'W1'):
    saved_state['W1'] = ttt_instance.W1.clone().detach()
```

## 🎯 EXPECTED BEHAVIOR (Next Run)

### Initialization Logs
```
[Hybrid] Layer 27: Creating TTT-MLP with 3 layers via TTTWrapper
[TTT] ✅ USING MULTI-LAYER TTT-MLP with 3 layers
[TTT-MLP-MULTI] Initializing Multi-Layer TTT-MLP
[TTT-MLP-MULTI] Number of layers: 3
[TTT-MLP-MULTI] Layer dimensions: [128, 512, 512, 128]
[TTT-MLP-MULTI] Total TTT parameters: 12,619,776
```

### Forward Pass Logs
```
[TTT-MLP-MULTI] 🚀 First forward pass - Multi-layer TTT-MLP active!
[TTT-MLP-MULTI] Processing batch: B=1, L=..., num_mini_batch=...
[TTT-MLP-MULTI] Using 3 layers with dimensions [128, 512, 512, 128]
```

### Parameter Count
- **Per layer**: 12,619,776 parameters
- **5 layers total**: 63,098,880 parameters
- **Previous runs**: 21,696,480 parameters (2-layer)
- **Increase**: 2.91× more parameters ✅

## 📊 VERIFICATION CHECKLIST

After next training run, verify:

1. ✅ **No TypeError** - Forward pass succeeds
2. ✅ **Correct parameter count** - Should see ~63M TTT params (not 21.7M)
3. ✅ **Multi-layer logs appear** - `[TTT-MLP-MULTI]` messages during training
4. ✅ **Architecture confirmed** - Logs show 3-layer with [128, 512, 512, 128] dimensions
5. ⚠️ **Memory usage increases** - Expected ~30-50% more GPU memory
6. ⚠️ **Slower training** - More parameters = more computation time

## 🔧 TECHNICAL DETAILS

### TTTWrapper Decision Logic

From `moshi_ttt/models/ssm/ttt_layer.py` (lines 28-40):
```python
if config.ssm_layer == "ttt_linear":
    logger.info(f"[TTT] Initializing TTTLinear layer")
    self.ttt = TTTLinear(config, use_kernel=False)
elif config.ssm_layer == "ttt_mlp":
    # Use multi-layer implementation if 3+ layers, else use standard 2-layer
    if hasattr(config, 'ttt_mlp_layers') and getattr(config, 'ttt_mlp_layers', 2) >= 3:
        num_layers = getattr(config, 'ttt_mlp_layers', 2)
        logger.info(f"[TTT] ✅ USING MULTI-LAYER TTT-MLP with {num_layers} layers")
        self.ttt = TTTMLPMultiLayer(config, use_kernel=False)
    else:
        logger.info(f"[TTT] Initializing standard 2-layer TTT-MLP")
        self.ttt = TTTMLP(config, use_kernel=False)
```

### Config Flow (Still Correct)
```
YAML (ttt_mlp_layers: 3)
  ↓
TTTArgs (ttt_mlp_layers=3)
  ↓
TTTConfig (ttt_mlp_layers=3, ssm_layer="ttt_mlp")
  ↓
TTTWrapper (checks config.ttt_mlp_layers >= 3)
  ↓
TTTMLPMultiLayer (3-layer implementation)
```

## 🚨 WHAT WENT WRONG IN JOB 7113935

### Timeline of Events

1. **10:45:29** - Config correctly detected 3 layers
2. **10:45:29-10:45:32** - All 5 TTT layers created with multi-layer architecture (12.6M params each)
3. **10:45:35** - TTT parameter tracker found 110 parameters ✅
4. **10:45:38** - Started first forward pass
5. **10:45:39** - **CRASH** - TypeError on forward call

### Why The Fix Worked Locally But Failed In Training

The test script (`test_multilayer_logging.py`) probably:
- Tested initialization only (which worked)
- Didn't call `forward()` on the layer
- Or used a different code path

The actual training revealed the real issue: **missing forward method**.

## 💡 KEY INSIGHT

**Always use the architecture's intended entry point!**

- Video-DiT uses `TTTWrapper` → we must use `TTTWrapper`
- Don't bypass the wrapper to instantiate inner classes directly
- The wrapper provides the public API contract (`forward` signature)
- Inner classes (`TTTMLPMultiLayer`, `TTTMLP`) are implementation details

## 🎓 LESSONS LEARNED

1. **Layer instantiation patterns matter** - Can't mix and match wrapper vs direct instantiation
2. **Test forward passes, not just initialization** - Creation != Execution
3. **Check base class methods** - `TTTBase.forward()` had wrong signature for our use case
4. **Follow the reference architecture** - Video-DiT uses `TTTWrapper` for a reason
5. **Logs can be misleading** - Successful init doesn't mean successful forward

## 📝 FILES MODIFIED

1. **moshi_ttt/hybrid_layer.py**
   - Line 67-73: Use `TTTWrapper` instead of `TTTMLPMultiLayer` directly
   - Line 100: Access `ttt_instance` via `getattr(self.ttt_layer, 'ttt', self.ttt_layer)` in save
   - Line 143: Access `ttt_instance` via `getattr(self.ttt_layer, 'ttt', self.ttt_layer)` in restore

## ✅ READY FOR NEXT TRAINING RUN

The code is now properly structured to:
- Use Video-DiT's architecture pattern correctly
- Handle 3-layer TTT-MLP through the proper wrapper
- Maintain state save/restore compatibility
- Provide correct forward signature for all code paths
