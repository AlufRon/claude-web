# LibriLight Streaming Evaluation Fix - Detailed Implementation Plan

## 🚨 Problem Statement

The current LibriLight evaluation uses **training methodology** (batch processing entire sequences) instead of **Moshi's streaming inference methodology**, causing:

1. **Memory Crashes**: 24k×24k attention matrices (~37GB) exceed GPU capacity
2. **Invalid Comparison**: Both baseline and TTT models crash, making comparison meaningless  
3. **Wrong Architecture Usage**: Not leveraging Moshi's natural 3k context window + KV cache system
4. **No TTT Advantage**: TTT can't demonstrate its strength (accumulating patterns beyond sliding window)

## 🎯 Solution Overview

**Replace batch processing with streaming evaluation** that:
- Uses `model.streaming()` context manager and `model.step()` for token-by-token processing
- Respects Moshi's 3000-token context window limitation
- Allows TTT to accumulate knowledge while staying within memory constraints
- Provides fair comparison: Baseline (3k sliding window) vs TTT (3k sliding window + global patterns)

## 📋 Detailed Implementation Plan

### **Phase 1: Create Streaming Evaluation Infrastructure**

#### **Step 1.1: Add Streaming LibriLight Method**
- **File**: `finetune/paper_metrics.py`
- **Method**: `_evaluate_librilight_streaming(model, codes, targets)`
- **Purpose**: Core streaming evaluation logic

**Implementation Details:**
```python
def _evaluate_librilight_streaming(self, model, codes, targets):
    """
    Stream-based evaluation using Moshi's natural inference mode.
    
    Key differences from batch processing:
    - Uses model.streaming() context manager (enables KV cache)
    - Processes one token at a time with model.step()
    - Respects Moshi's 3k context window limitation
    - Allows TTT to accumulate patterns across entire sequence
    """
    model.eval()
    
    # Pre-evaluation memory cleanup
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    # Enter Moshi streaming mode
    with model.streaming(batch_size=1):
        position_losses = []
        
        seq_length = codes.shape[-1]  # Total sequence length
        logger.info(f"Starting streaming evaluation: {seq_length} tokens")
        
        for t in range(seq_length):
            # Process single token (Moshi's natural mode)
            current_codes = codes[:, :, t:t+1]  # [1, 8, 1]
            
            # Create 17-codebook input (Moshi format)
            inp = torch.zeros(1, 17, 1, device=codes.device, dtype=codes.dtype)
            inp[:, 1:9] = current_codes  # Place audio in codebooks 1-8
            
            with torch.no_grad():
                # Streaming step: Updates KV cache + TTT weights
                out = model.step(inp)
                logits = out.logits[:, :8, -1].float()  # [1, 8, vocab_size]
            
            # Compute loss for this position
            target = targets[:, :, t]  # [1, 8]
            loss = self._compute_position_loss(logits, target)
            position_losses.append(loss)
            
            # Memory management: Clear cache every 3k tokens
            if t % 3000 == 0 and t > 0:
                torch.cuda.empty_cache()
                logger.debug(f"Processed {t} tokens, cleared cache")
                
        logger.info(f"Streaming evaluation complete: {len(position_losses)} losses computed")
        return position_losses
```

#### **Step 1.2: Add Position Loss Computation Helper**
- **Method**: `_compute_position_loss(logits, target)`
- **Purpose**: Handle per-position loss calculation with proper masking

**Implementation Details:**
```python
def _compute_position_loss(self, logits, target):
    """
    Compute loss for single position with proper masking.
    Handles invalid tokens and ensures finite results.
    """
    # Create mask for valid positions
    valid_mask = (target >= 0) & (target < logits.size(-1))
    
    if valid_mask.sum() > 0:
        # Compute cross-entropy loss
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target.view(-1),
            reduction='none'
        )
        
        # Apply mask and get mean
        masked_loss = loss * valid_mask.view(-1).float()
        valid_count = valid_mask.sum().float()
        
        if valid_count > 0 and torch.isfinite(masked_loss).all():
            avg_loss = masked_loss.sum() / valid_count
            if torch.isfinite(avg_loss):
                return avg_loss.item()
    
    return 0.0  # Fallback for invalid positions
```

#### **Step 1.3: Add TTT State Tracking**
- **Method**: `_track_ttt_state_changes(model)`
- **Purpose**: Verify TTT is actually learning during streaming

**Implementation Details:**
```python
def _track_ttt_state_changes(self, model):
    """
    Capture TTT weights before/after evaluation to verify learning.
    Returns dict with state change information.
    """
    ttt_info = {'has_ttt': False, 'weights_changed': False}
    
    try:
        # Check if model has TTT layers
        ttt_layers = []
        for name, module in model.named_modules():
            if 'ttt' in name.lower() or hasattr(module, 'ttt_layer'):
                ttt_layers.append((name, module))
        
        if ttt_layers:
            ttt_info['has_ttt'] = True
            ttt_info['num_ttt_layers'] = len(ttt_layers)
            
            # Capture initial weights
            initial_weights = {}
            for name, module in ttt_layers:
                if hasattr(module, 'ttt_layer'):
                    initial_weights[name] = module.ttt_layer.state_dict()
            
            ttt_info['initial_weights'] = initial_weights
            
    except Exception as e:
        logger.warning(f"Could not track TTT state: {e}")
    
    return ttt_info

def _verify_ttt_learning(self, model, initial_ttt_info):
    """Verify TTT weights changed during evaluation"""
    if not initial_ttt_info['has_ttt']:
        return initial_ttt_info
    
    try:
        changes_detected = False
        for name, module in model.named_modules():
            if name in initial_ttt_info['initial_weights']:
                current_weights = module.ttt_layer.state_dict()
                initial_weights = initial_ttt_info['initial_weights'][name]
                
                for param_name, param_tensor in current_weights.items():
                    if param_name in initial_weights:
                        if not torch.equal(param_tensor, initial_weights[param_name]):
                            changes_detected = True
                            break
                
                if changes_detected:
                    break
        
        initial_ttt_info['weights_changed'] = changes_detected
        logger.info(f"TTT verification: {initial_ttt_info['num_ttt_layers']} layers, weights_changed={changes_detected}")
        
    except Exception as e:
        logger.warning(f"Could not verify TTT learning: {e}")
    
    return initial_ttt_info
```

### **Phase 2: Integrate Streaming with Existing Evaluation**

#### **Step 2.1: Modify Main Evaluation Method**
- **Method**: Modify `evaluate_long_context_loss()`
- **Purpose**: Replace batch processing with streaming calls

**Key Changes:**
1. Replace the mega-sequence batch forward pass (lines 799-851)
2. Add streaming evaluation call
3. Maintain existing metrics format
4. Add TTT tracking

**Implementation Approach:**
```python
# Replace this section in evaluate_long_context_loss():
# OLD (lines 799-851): Batch processing
# with torch.no_grad():
#     out = model(inp)  # CRASHES with 24k tokens

# NEW: Streaming processing
initial_ttt_info = self._track_ttt_state_changes(model)

try:
    # Stream-based evaluation (no memory crashes)
    loss_per_position = self._evaluate_librilight_streaming(
        model, codes, target_codes
    )
    
    # Verify TTT learning occurred
    final_ttt_info = self._verify_ttt_learning(model, initial_ttt_info)
    
    # Store TTT info for logging
    self._last_ttt_info = final_ttt_info
    
    all_position_losses.append(loss_per_position)
    
except Exception as e:
    logger.error(f"Streaming evaluation failed: {e}")
    # Handle gracefully...
```

#### **Step 2.2: Enhanced Memory Management**
- **Purpose**: Ensure reliable evaluation across different hardware configurations

**Implementation:**
```python
def _ensure_memory_availability(self, required_gb=4.0):
    """Ensure sufficient GPU memory for streaming evaluation"""
    try:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Check available memory
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        current_allocated = torch.cuda.memory_allocated() / (1024**3)
        available = total_memory - current_allocated
        
        logger.info(f"Memory status: {available:.1f}GB available of {total_memory:.1f}GB total")
        
        if available < required_gb:
            logger.warning(f"Low memory: {available:.1f}GB < {required_gb:.1f}GB required")
            # Additional cleanup
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            
        return available >= required_gb
        
    except Exception as e:
        logger.warning(f"Memory check failed: {e}")
        return True  # Assume OK
```

### **Phase 3: Configuration and Validation**

#### **Step 3.1: Add Streaming Configuration Options**
- **Purpose**: Make streaming behavior configurable

**Configuration Options:**
```yaml
paper_metrics:
  librilight_streaming:
    enabled: true                    # Use streaming evaluation
    memory_check: true              # Check memory before evaluation
    cache_clear_interval: 3000      # Clear cache every N tokens
    max_sequence_length: 50000      # Limit sequence length if needed
    ttt_verification: true          # Verify TTT weight changes
```

#### **Step 3.2: Enhanced Logging**
- **Purpose**: Provide detailed feedback about streaming evaluation

**Log Information:**
- Memory usage before/during/after evaluation
- TTT state changes (if applicable)
- Processing speed (tokens/second)
- Context window utilization
- Comparison with batch processing (when applicable)

### **Phase 4: Testing and Validation**

#### **Step 4.1: Unit Testing**
- **Test File**: `test_streaming_evaluation.py`
- **Coverage**: Individual method testing

**Test Cases:**
1. `test_streaming_evaluation_basic()`: Simple streaming evaluation
2. `test_memory_management()`: Memory cleanup and monitoring
3. `test_ttt_state_tracking()`: TTT weight change detection
4. `test_position_loss_computation()`: Per-position loss accuracy
5. `test_error_handling()`: Graceful failure handling

#### **Step 4.2: Integration Testing**
- **Test**: Compare streaming vs batch on short sequences
- **Purpose**: Validate numerical accuracy
- **Expectation**: Similar results for sequences under 3k tokens

#### **Step 4.3: Performance Testing**
- **Test**: Memory usage and processing speed
- **Metrics**: GPU memory peak, tokens/second, accuracy maintenance
- **Validation**: Streaming should use <5GB consistently

#### **Step 4.4: TTT vs Baseline Comparison**
- **Test**: Run both models with streaming evaluation
- **Expected Results**:
  - Both complete evaluation without crashes ✅
  - TTT shows more negative slope than baseline ✅
  - TTT weights demonstrably change during evaluation ✅
  - Memory usage remains stable for both models ✅

## 🔧 Technical Implementation Details

### **Memory Management Strategy**
1. **Pre-evaluation**: Clear cache, check available memory
2. **During evaluation**: Clear cache every 3k tokens (context window size)
3. **Post-evaluation**: Final cleanup and memory reporting

### **Error Handling**
1. **Memory errors**: Graceful fallback to shorter sequences
2. **Model errors**: Detailed logging and safe defaults
3. **Invalid tokens**: Proper masking and finite value checking

### **Backward Compatibility**
1. **Existing metrics**: All current metrics maintained
2. **Configuration**: Streaming is opt-in, batch processing remains default
3. **API**: No breaking changes to public interfaces

### **Performance Optimization**
1. **Streaming overhead**: Minimize per-token overhead
2. **Memory efficiency**: Only store necessary intermediate results
3. **GPU utilization**: Balance memory usage with processing speed

## 📊 Expected Outcomes

### **Memory Usage Comparison**
| Method | Memory Usage | Status |
|--------|--------------|---------|
| Current (Batch) | 37GB+ for 24k tokens | 💥 CRASH |
| New (Streaming) | ~3GB constant | ✅ SUCCESS |

### **TTT vs Baseline Results**
| Model | Context Access | Expected Slope |
|-------|----------------|----------------|
| Baseline | 3k sliding window only | Near zero (plateau) |
| TTT | 3k window + global patterns | Negative (continued improvement) |

### **Processing Capabilities**
- **Sequence Length**: No practical limit (memory constant)
- **Evaluation Time**: Linear with sequence length
- **Memory Stability**: Consistent throughout evaluation
- **Hardware Requirements**: Standard GPU (8GB+ recommended)

## 📝 Files to be Modified

1. **`finetune/paper_metrics.py`** (Primary changes)
   - Add streaming evaluation methods
   - Modify main evaluation logic
   - Add TTT state tracking
   - Enhanced memory management

2. **Configuration files** (Optional)
   - Add streaming evaluation parameters
   - Memory management settings

3. **Documentation** (This file)
   - Implementation plan and rationale
   - Usage examples and validation results

## ✅ Success Criteria

### **Functional Requirements**
- ✅ Both baseline and TTT models complete LibriLight evaluation
- ✅ Memory usage remains within GPU limits (<5GB)
- ✅ TTT weights demonstrably update during streaming evaluation
- ✅ Clear slope difference between TTT and baseline models
- ✅ Numerical accuracy maintained vs batch processing (short sequences)

### **Non-Functional Requirements**
- ✅ Backward compatibility with existing evaluation pipeline
- ✅ Configurable behavior (streaming vs batch)
- ✅ Comprehensive error handling and logging
- ✅ Performance acceptable (processing speed reasonable)
- ✅ Memory management reliable across hardware configurations

### **Validation Requirements**
- ✅ Unit tests pass for all new methods
- ✅ Integration tests show consistent results
- ✅ Performance tests demonstrate memory efficiency
- ✅ TTT vs baseline comparison shows expected differences

## 🚀 Implementation Timeline

1. **Phase 1** (Infrastructure): Core streaming methods and helpers
2. **Phase 2** (Integration): Modify existing evaluation pipeline
3. **Phase 3** (Configuration): Add configuration options and enhanced logging
4. **Phase 4** (Testing): Comprehensive testing and validation

**Total Estimated Time**: Implementation ready for testing within this session.

## 📋 Post-Implementation Validation Checklist

- [ ] Streaming evaluation completes without memory errors
- [ ] TTT state tracking works correctly
- [ ] Baseline vs TTT comparison shows clear differences
- [ ] Memory usage remains stable throughout evaluation
- [ ] All existing metrics continue to work
- [ ] Error handling prevents crashes
- [ ] Performance is acceptable for practical use
- [ ] Documentation updated with new methodology

---

**Note**: This plan addresses the fundamental architectural mismatch between TTT evaluation methodology and Moshi's streaming design. The fix will enable proper comparison of TTT's long-term pattern accumulation vs baseline's sliding window limitation.