# TTT Parameter Optimization Investigation

## Summary

**CRITICAL FINDING**: TTT parameters are NOT being optimized during Moshi training because the training setup doesn't account for TTT-specific parameter group requirements.

## Investigation Results

### ✅ TTT Implementation is Working
- **Test Result**: `test_ttt_parameters_update.py` PASSED
- **Evidence**: TTT parameters receive gradients and update correctly in isolation
- **18 TTT parameters** all show gradient flow and parameter updates
- **Forward/backward pass** works without errors

### ❌ Training Configuration Issue
- **Problem**: Moshi training uses standard optimizer setup
- **Missing**: TTT-specific parameter grouping with higher learning rates
- **Current**: `AdamW(model.parameters(), lr=args.optim.lr)` treats all parameters equally

## TTT-Video-DiT Parameter Group Analysis

### Parameter Classification
TTT-Video-DiT uses sophisticated parameter grouping:

```python
TTT_PARAMETER_PATTERNS = ["ttt", "ssm"]
NO_WEIGHT_DECAY_PATTERNS = ["bias", "norm", "b1", "b2"]

def is_ttt_parameter(param_name: str) -> bool:
    return any(pattern in param_name.lower() for pattern in TTT_PARAMETER_PATTERNS)
```

### Parameter Groups
1. **TTT parameters with weight decay**
2. **TTT parameters without weight decay** (biases, norms)  
3. **Regular parameters with weight decay**
4. **Regular parameters without weight decay**

### Learning Rate Configuration
- **Regular parameters**: `lr=1e-5` (base learning rate)
- **TTT parameters**: `lr_ssm=1e-4` (10x higher learning rate)
- **Ratio**: TTT parameters get **10x higher learning rate**

## Current Moshi Training Setup

### Optimizer Creation (train.py:254-261)
```python
optimizer = AdamW(
    model.parameters(),          # ← All parameters get same treatment
    lr=args.optim.lr,           # ← Single learning rate for all
    betas=(0.9, 0.95),
    weight_decay=args.optim.weight_decay,
)
```

### TTT Configuration (example/moshi_7B.yaml)
```yaml
ttt:
  base_lr: 0.1                  # ← Not used in optimizer setup
  enable: true
  layers: '31'
  mini_batch_size: 2
```

## Root Cause Analysis

### Why TTT Parameters Don't Change

1. **Insufficient Learning Rate**: TTT parameters get `lr=0.002` (from Moshi config) instead of needed `lr=0.1` (TTT config)
2. **No Parameter Grouping**: All parameters treated equally instead of TTT-specific groups
3. **Configuration Disconnect**: `ttt.base_lr: 0.1` is ignored by optimizer setup

### Evidence from Training Logs
```
ttt_alpha: 0.050049 - ttt_raw: 0.050049 - ttt_Δ: 0.00e+00
```
- **Constant values**: Parameters never update
- **Delta always zero**: No parameter changes detected

## Solution Requirements

### 1. Implement TTT Parameter Groups
Create optimizer with separate parameter groups:
- TTT parameters: `lr = ttt_config.base_lr` (0.1)
- Regular parameters: `lr = args.optim.lr` (0.002)

### 2. Parameter Detection Logic
```python
def is_ttt_parameter(name: str) -> bool:
    ttt_patterns = ['ttt', 'ssm', 'wq', 'wk', 'wv', 'wo', 'W1', 'W2', 'b1', 'b2', 'learnable_ttt']
    return any(pattern in name.lower() for pattern in ttt_patterns)
```

### 3. Modified Optimizer Setup
```python
# Separate TTT and regular parameters
ttt_params = []
regular_params = []

for name, param in model.named_parameters():
    if is_ttt_parameter(name):
        ttt_params.append(param)
    else:
        regular_params.append(param)

# Create parameter groups
param_groups = [
    {'params': regular_params, 'lr': args.optim.lr},
    {'params': ttt_params, 'lr': args.ttt.base_lr}  # Higher LR for TTT
]

optimizer = AdamW(param_groups, ...)
```

## Implementation Priority

### High Priority Fixes
1. **Parameter Group Optimizer**: Implement TTT-aware parameter grouping
2. **Learning Rate Scaling**: Use `ttt.base_lr` for TTT parameters
3. **Training Integration**: Modify `train.py` optimizer setup

### Verification Steps
1. **Parameter Detection**: Ensure TTT parameters are correctly identified
2. **Learning Rate Assignment**: Verify TTT parameters get higher learning rates
3. **Training Test**: Confirm TTT parameters update during training

## Expected Outcome

After implementing TTT parameter groups:
- `ttt_alpha` should change from `0.050049` to varying values
- `ttt_Δ` should show non-zero changes
- TTT parameters should receive gradients and update at higher learning rates

## Files to Modify

1. **`train.py`**: Optimizer setup (lines 254-261)
2. **`finetune/wrapped_model.py`**: TTT parameter detection
3. **`moshi_ttt/config.py`**: Parameter group configuration

## Testing Strategy

1. **Unit Test**: Verify parameter group creation
2. **Integration Test**: Test optimizer with TTT parameter groups  
3. **Training Test**: Run short training to confirm parameter updates
4. **Production Test**: Full training run with parameter monitoring

---

**CONCLUSION**: The TTT implementation is correct, but the training setup needs TTT-specific parameter grouping with higher learning rates to enable TTT parameter optimization.