# TTT Save/Restore Implementation Plan

## Problem Statement
TTT parameter reset after evaluation causes backward graph errors:
- **Issue**: `RuntimeError: Trying to backward through the graph a second time`
- **Root Cause**: Calling `init_weights()` during training destroys computation graph
- **Current Approach**: Reset TTT parameters to random values after evaluation
- **Why Reset Needed**: Prevent data leakage from evaluation back to training

## Solution: Save/Restore Pattern

Replace random reset with save/restore pattern to avoid computation graph destruction.

### Core Concept
```python
# Before evaluation
saved_state = save_ttt_states()

# During evaluation: TTT learns from eval data ✅ 
run_evaluation()

# After evaluation: restore pre-eval state ✅
restore_ttt_states(saved_state)

# Continue training with clean TTT (no eval contamination) ✅
```

## Implementation Steps

### Step 1: Add Save/Restore Methods to HybridSeqModelingBlock

**File**: `moshi_ttt/hybrid_layer.py`

```python
def save_ttt_states(self):
    """Save current TTT parameter values"""
    if not self.persistent_states:
        return None
    
    return {
        'W1': self.ttt_layer.W1.clone().detach(),
        'b1': self.ttt_layer.b1.clone().detach(), 
        'W2': self.ttt_layer.W2.clone().detach(),
        'b2': self.ttt_layer.b2.clone().detach()
    }

def restore_ttt_states(self, saved_state):
    """Restore TTT parameters from saved state"""
    if not self.persistent_states or saved_state is None:
        return
    
    with torch.no_grad():
        self.ttt_layer.W1.copy_(saved_state['W1'])
        self.ttt_layer.b1.copy_(saved_state['b1'])
        self.ttt_layer.W2.copy_(saved_state['W2'])
        self.ttt_layer.b2.copy_(saved_state['b2'])
```

### Step 2: Propagate Methods Through Layer Hierarchy

**File**: `moshi_ttt/hybrid_layer.py`

```python
# In HybridStreamingTransformerLayer
def save_ttt_states(self):
    return self.seq_modeling_block.save_ttt_states()

def restore_ttt_states(self, saved_state):
    self.seq_modeling_block.restore_ttt_states(saved_state)
```

### Step 3: Add Model-Level Methods

**File**: `finetune/model_integration.py` (or where model wrapper is)

```python
def save_ttt_states(model):
    """Save TTT states from all hybrid layers"""
    saved_states = {}
    for name, module in model.named_modules():
        if hasattr(module, 'save_ttt_states'):
            state = module.save_ttt_states()
            if state is not None:
                saved_states[name] = state
    return saved_states

def restore_ttt_states(model, saved_states):
    """Restore TTT states to all hybrid layers"""
    for name, module in model.named_modules():
        if hasattr(module, 'restore_ttt_states') and name in saved_states:
            module.restore_ttt_states(saved_states[name])
```

### Step 4: Update Evaluation Code

**File**: `finetune/paper_metrics.py`

**Replace this:**
```python
# CURRENT (BROKEN)
if hasattr(model, 'reset_ttt_states'):
    model.reset_ttt_states()  # Breaks computation graph
```

**With this:**
```python
# NEW (SAFE)
saved_ttt_states = save_ttt_states(model)

try:
    # Run evaluation - TTT adapts to eval data
    results = run_librilight_evaluation(model, ...)
    
finally:
    # Always restore pre-eval state 
    restore_ttt_states(model, saved_ttt_states)
    logger.info("✅ TTT states restored - no eval contamination")
```

### Step 5: Remove Broken Reset Method

**File**: `moshi_ttt/hybrid_layer.py`

```python
def reset_ttt_states(self):
    """DEPRECATED: Use save/restore pattern instead"""
    logger.warning("⚠️  reset_ttt_states() is deprecated")
    logger.warning("   Use save_ttt_states() and restore_ttt_states() instead")
    logger.warning("   This prevents computation graph errors")
```

## Advantages of Save/Restore

✅ **No computation graph destruction**
- `copy_()` doesn't break backward pass
- Safe to call during training

✅ **Precise state control**  
- Restore exact pre-evaluation state
- No random reinitialization

✅ **Cleaner evaluation isolation**
- Evaluation gets its own sandbox
- Training state preserved perfectly

✅ **Exception safety**
- Use try/finally to ensure restoration
- Even if evaluation crashes

## Testing Plan

### Test 1: Basic Save/Restore
```python
# Save state
state = model.save_ttt_states()

# Modify parameters (simulate evaluation)
# ... 

# Restore and verify
model.restore_ttt_states(state)
assert torch.allclose(W1_original, model.W1)
```

### Test 2: No Computation Graph Errors
```python
# Forward pass
output = model(input)
loss = output.sum()

# Save/restore cycle  
state = model.save_ttt_states()
model.restore_ttt_states(state)

# Should not error
loss.backward()  # ✅ Should work
```

### Test 3: Evaluation Isolation
```python
# Training state
train_state = model.save_ttt_states()

# Evaluation modifies TTT
eval_output = model(eval_input)

# Restore training state
model.restore_ttt_states(train_state)

# Verify no eval contamination
assert torch.allclose(train_W1, model.W1)
```

## Risk Mitigation

- **Memory**: Saving states uses extra memory, but TTT params are small
- **Performance**: `clone()` and `copy_()` are fast operations  
- **Correctness**: Extensive testing to ensure exact restoration

## Implementation Order

1. **Step 1-2**: Add methods to hybrid layers
2. **Step 5**: Remove broken reset method  
3. **Step 3**: Add model-level methods
4. **Step 4**: Update evaluation code
5. **Testing**: Verify no computation graph errors
6. **Integration**: Test full training pipeline