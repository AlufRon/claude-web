# TTT Memory Optimization Strategy

## Problem Summary
TTT training uses **44.2 GB peak memory** vs **19.7 GB** for baseline - a **124% increase**.
However, only **20.9 GB** is actively allocated during most of training.
**The 23.3 GB overhead is temporary memory during the backward pass.**

## Root Cause Analysis

### Why the Peak Occurs
1. **Gradient Computation**: TTT backward pass stores intermediate activations
2. **State Persistence**: TTT maintains state across sequence positions  
3. **Large Matrix Operations**: Attention mechanisms with (batch × seq_len × hidden_dim)
4. **No Gradient Checkpointing**: All activations stored for backward pass

### Key Observation
- Forward pass: ~21 GB
- Backward pass: ~44 GB (peak!)
- **23 GB of temporary gradient tensors**

## Optimization Strategies (Priority Order)

### 🔥 **Priority 1: Gradient Checkpointing** (Expected savings: 15-18 GB)

**Implementation:**
```python
# In finetune/ttt_integration.py

from torch.utils.checkpoint import checkpoint

class TTTLayerWithCheckpointing(nn.Module):
    def forward(self, x, *args, **kwargs):
        # Use gradient checkpointing for TTT forward pass
        if self.training:
            return checkpoint(
                self._forward_impl,
                x,
                *args,
                use_reentrant=False,  # More memory efficient
                **kwargs
            )
        else:
            return self._forward_impl(x, *args, **kwargs)
    
    def _forward_impl(self, x, *args, **kwargs):
        # Actual TTT computation here
        pass
```

**Benefits:**
- Trades computation for memory
- Recomputes activations during backward instead of storing them
- Should reduce peak from 44 GB to ~28-30 GB

### 🔥 **Priority 2: Selective Activation Caching** (Expected savings: 3-5 GB)

**Implementation:**
```python
# Clear TTT state more aggressively

class TTTLayer(nn.Module):
    def forward(self, x):
        # Process in chunks
        results = []
        for chunk in x.chunk(self.num_chunks):
            result = self.ttt_forward(chunk)
            results.append(result.detach())  # Detach intermediate results
            
            # Clear cache after each chunk
            if hasattr(self, 'ttt_state'):
                self.ttt_state = self.ttt_state.detach()
        
        return torch.cat(results)
```

### 🔥 **Priority 3: Memory-Efficient Attention** (Expected savings: 4-6 GB)

**Use Flash Attention or similar:**
```python
# In TTT layer

from flash_attn import flash_attn_func

def efficient_ttt_attention(self, q, k, v):
    # Flash attention is O(1) memory instead of O(N²)
    return flash_attn_func(
        q, k, v,
        dropout_p=0.0,
        causal=True
    )
```

### ⚡ **Priority 4: Mixed Precision for TTT** (Expected savings: 2-3 GB)

**Use bfloat16 or float16 for TTT computations:**
```python
class TTTLayer(nn.Module):
    def forward(self, x):
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            # TTT computation in lower precision
            output = self.ttt_computation(x)
        return output.to(x.dtype)  # Convert back
```

### 🛠️ **Priority 5: Optimize Chunk Processing** (Expected savings: 1-2 GB)

**Process smaller chunks with dynamic sizing:**
```python
def dynamic_chunk_size(available_memory_gb, current_seq_len):
    """Adjust chunk size based on available memory."""
    if available_memory_gb < 25:
        return min(25, current_seq_len // 4)  # Smaller chunks
    elif available_memory_gb < 35:
        return min(50, current_seq_len // 2)
    else:
        return min(100, current_seq_len)  # Larger chunks if memory available
```

### 🛠️ **Priority 6: Explicit Memory Management** (Expected savings: 1-2 GB)

**Clear CUDA cache strategically:**
```python
def training_step_with_memory_management(self, batch, step):
    output = self.model(batch)
    loss = compute_loss(output)
    
    loss.backward()
    
    # Clear cache after backward pass (where peak occurs)
    if step % 10 == 0:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)  # More memory efficient than zero_grad()
```

## Implementation Roadmap

### Phase 1: Quick Wins (1-2 hours)
1. ✅ Add gradient checkpointing to TTT layers
2. ✅ Use `zero_grad(set_to_none=True)` 
3. ✅ Clear CUDA cache after backward pass

**Expected reduction: 44 GB → 32-35 GB**

### Phase 2: Medium Effort (4-6 hours)
1. ✅ Implement selective activation caching
2. ✅ Add memory-efficient attention (Flash Attention)
3. ✅ Dynamic chunk sizing based on memory

**Expected reduction: 32-35 GB → 25-28 GB**

### Phase 3: Advanced (1-2 days)
1. ✅ Full mixed precision for TTT
2. ✅ Optimize TTT state management
3. ✅ Implement reversible layers for TTT

**Expected reduction: 25-28 GB → 22-24 GB**

## Monitoring & Validation

### Memory Profiling Code
```python
import torch.cuda

def profile_memory_by_component():
    """Profile memory usage at each stage."""
    torch.cuda.reset_peak_memory_stats()
    
    # Forward pass
    output = model(batch)
    forward_peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"Forward peak: {forward_peak:.2f} GB")
    
    torch.cuda.reset_peak_memory_stats()
    
    # Backward pass
    loss.backward()
    backward_peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"Backward peak: {backward_peak:.2f} GB")
```

### Success Metrics
- ✅ Peak memory < 30 GB (vs current 44 GB)
- ✅ Memory efficiency > 70% (vs current 47%)
- ✅ No OOM errors
- ✅ Training speed not significantly impacted (<20% slowdown)

## Expected Final Result

| Metric | Before | After Phase 1 | After Phase 2 | Target |
|--------|--------|---------------|---------------|---------|
| Peak Memory | 44.2 GB | 32-35 GB | 25-28 GB | <30 GB |
| Allocated | 20.9 GB | 20.9 GB | 20.9 GB | ~21 GB |
| Overhead | 23.3 GB | 11-14 GB | 4-7 GB | <9 GB |
| Efficiency | 47.3% | ~63% | ~75% | >70% |

## Code Changes Needed

### Files to Modify:
1. `finetune/ttt_integration.py` - Add gradient checkpointing
2. `finetune/ttt_layer.py` - Implement memory-efficient attention
3. `train.py` - Add memory management in training loop
4. `finetune/wrapped_model.py` - Enable mixed precision for TTT

Each modification should be done incrementally and tested!
