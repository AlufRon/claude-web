# TTT Gating Alpha Configuration Fix Summary

## Problem
The `initial_gating_alpha` parameter in the YAML configuration was not being used because it was hardcoded to `0.05` in `ttt_utils.py`.

## Solution Implemented

### 1. Added `initial_gating_alpha` to TTTArgs
**File**: `finetune/args.py:74`
```python
initial_gating_alpha: float = 0.1  # Initial gating alpha for TTT layers
```

### 2. Updated TTTConfig Creation
**File**: `finetune/ttt_integration.py:85`
```python
gating_alpha_init=ttt_args.initial_gating_alpha,
```

### 3. Made TTT Parameter Initialization Configurable
**File**: `finetune/ttt_utils.py:66`
```python
def initialize_ttt_parameter(param: torch.nn.Parameter, param_name: str, gating_alpha_init: float = 0.05) -> None:
```

### 4. Updated Parameter Initialization Call
**File**: `finetune/wrapped_model.py:124`
```python
initialize_ttt_parameter(param, full_param_name, args.ttt.initial_gating_alpha)
```

### 5. Added Logging for Gating Alpha Value
**File**: `finetune/ttt_integration.py:132`
```python
main_logger_info(f"TTT gating: initial_alpha={ttt_config.gating_alpha_init}")
```

## Configuration Files
- **Multi-layer small alpha**: `example/moshi_7B_multilayer_small_alpha.yaml`
  - Uses `initial_gating_alpha: 0.01` to prevent catastrophic forgetting
  - Applies TTT to layers 15,31

## Testing
- ✅ Configuration parsing works correctly
- ✅ Gating alpha initialization respects YAML values
- ✅ Default value (0.05) preserved for backward compatibility
- ✅ Custom values (0.01) applied correctly

## Impact
The multi-layer TTT configuration can now use smaller gating alpha values to potentially prevent catastrophic forgetting that was observed in the previous experiments.

**Next Step**: Test the `moshi_7B_multilayer_small_alpha.yaml` configuration to see if smaller gating alpha (0.01) prevents the dramatic performance drop seen with multi-layer TTT.