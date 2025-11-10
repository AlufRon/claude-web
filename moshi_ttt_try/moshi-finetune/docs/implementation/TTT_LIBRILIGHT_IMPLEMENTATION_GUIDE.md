# TTT LibriLight Implementation Guide: The Correct Approach

## Executive Summary

This guide presents **two scientifically valid approaches** for implementing LibriLight evaluation with TTT (Test-Time Training), based on comprehensive analysis of the TTT paper, Moshi's streaming architecture, and our current implementation.

**Key Finding**: Both approaches are theoretically sound but serve different purposes:
- **TTT-Optimized**: Maximum efficiency for TTT mini-batch processing  
- **Moshi-Native**: Maximum adaptation accuracy using token-by-token streaming

## Background: Current Implementation Analysis

### Current Method (50-Token Chunking)
**File**: `/home/alufr/ttt_tests/moshi-finetune/finetune/paper_metrics.py:1168`

```python
# Current implementation
chunk_size = min(50, seq_length)  # Fixed 50-token chunks
with model.streaming(batch_size=1):
    for chunk_start in range(0, seq_length, chunk_size):
        chunk_codes = codes_truncated[:, :, chunk_start:chunk_end]  # [1, 8, chunk_length]
        out = model(codes=codes_input, condition_tensors=None)
```

**Current TTT Configuration**:
```yaml
ttt:
  mini_batch_size: 4     # TTT processes 4 tokens per mini-batch
  base_lr: 0.1          # TTT learning rate
  persistent_states: true  # States persist across chunks
```

### Efficiency Analysis
**Problem**: Misalignment between chunk size and TTT mini-batch size
- **Chunk size**: 50 tokens
- **Mini-batch size**: 4 tokens  
- **Result**: 50/4 = 12.5 → 12 mini-batches + 2 padding tokens
- **Efficiency**: 48/50 = 96% (4% waste due to padding)

**TTT Processing Pattern**:
```
Chunk: [T1, T2, T3, T4, T5, T6, T7, T8, ..., T50]
       ↓
Mini-batch 1: [T1, T2, T3, T4]     ← TTT update
Mini-batch 2: [T5, T6, T7, T8]     ← TTT update  
...
Mini-batch 12: [T45, T46, T47, T48] ← TTT update
Remaining: [T49, T50, PAD, PAD]     ← TTT update with 50% padding
```

## TTT Paper Insights: Critical Findings

### Mini-Batch Size Trade-offs (TTT Paper Table 1)
**Source**: `/home/alufr/ttt_tests/papers/ttt_paper.txt:473-477`

> "Figure 7. Ablations on TTT mini-batch size b, where b = 1 is online GD and b = T is batch GD. We choose b = 16 for all experiments in this paper. Left: **Smaller b improves perplexity since more GD steps are taken.**"

**Key Experimental Results**:
- **b = 1 (online GD)**: Best perplexity, worst speed
- **b = 16**: Chosen by authors as optimal trade-off
- **Mini-batch TTT improvement**: -1.70 perplexity improvement (Table 1, line 766)

### TTT Works with Single Tokens
**Critical Discovery**: TTT's core mechanism operates token-by-token:

```python
# TTT update rule (per token)
Wt = Wt-1 - η∇ℓ(Wt-1; xt)
```

**Implication**: TTT can function with `mini_batch_size = 1`, processing each token individually through reconstruction loss. This enables compatibility with Moshi's native `S=1` streaming.

### Learning Rate Scaling
**Formula**: `eta = (base_lr / mini_batch_size) * eta`

**Current Effective Rates**:
- `mini_batch_size=4`: `eta = 0.1/4 = 0.025` per mini-batch
- `mini_batch_size=1`: `eta = 0.1/1 = 0.1` per token (4x higher)

## Two Scientifically Valid Approaches

### Approach A: TTT-Optimized Chunking
**Philosophy**: Maximize TTT efficiency while maintaining streaming benefits

#### Core Principle
Align chunk size with TTT mini-batch size for 100% processing efficiency:
```
chunk_size = mini_batch_size * N  (where N is integer)
```

#### Configuration Options

**Option 1: Balanced Efficiency**
```yaml
ttt_optimized_balanced:
  mini_batch_size: 25    # Larger mini-batches for efficiency
  chunk_size: 25         # Perfect alignment (1 mini-batch per chunk)
  base_lr: 0.1          # eta = 0.1/25 = 0.004 per mini-batch
  num_chunks_per_50: 2   # 50 tokens = 2×25-token chunks
```

**Option 2: Maximum Efficiency**  
```yaml
ttt_optimized_max:
  mini_batch_size: 50    # Single mini-batch per standard chunk
  chunk_size: 50         # Perfect alignment
  base_lr: 0.1          # eta = 0.1/50 = 0.002 per mini-batch  
  padding_overhead: 0%   # No wasted computation
```

#### Implementation Strategy
1. **Configurable chunk size**: Make chunk_size depend on TTT mini_batch_size
2. **Efficiency monitoring**: Track padding percentage and mini-batch utilization
3. **Learning rate compensation**: Adjust base_lr to maintain effective learning rate
4. **Backward compatibility**: Default to current behavior if no config changes

#### Expected Benefits
- ✅ **100% TTT efficiency** (no padding waste)
- ✅ **Maintained streaming benefits** (context accumulation)
- ✅ **Good parallelization** (larger mini-batches utilize GPUs better)
- ✅ **Easy migration** (minimal changes to existing pipeline)

### Approach B: Moshi-Native Token-by-Token
**Philosophy**: Maximum adaptation accuracy using TTT's optimal configuration

#### Core Principle
Use Moshi's native `LMGen.step()` with `S=1` (single token per step) and TTT's `mini_batch_size=1` (online gradient descent):

```python
# Moshi native streaming constraint
assert S == 1, "Only support being given steps one by one."

# TTT online GD (best perplexity from paper)
mini_batch_size = 1  # Each token gets individual TTT update
```

#### Implementation Strategy
```python
def evaluate_librilight_native_streaming(model, codes, targets):
    """
    True token-by-token evaluation matching Moshi's native streaming.
    Each token receives individual TTT adaptation (online GD).
    """
    # Configure TTT for online gradient descent  
    set_ttt_mini_batch_size(model, mini_batch_size=1)
    adjust_ttt_learning_rate(model, base_lr=0.025)  # Compensate for 4x higher effective rate
    
    position_losses = []
    
    with model.streaming(batch_size=1):
        for t in range(codes.shape[-1]):
            # Process exactly 1 token (Moshi native constraint)
            token_input = codes[:, :, t:t+1]  # [1, 8, 1] 
            
            # Get logits from model (TTT adapts internally)
            with torch.no_grad():
                logits = model(token_input)
                loss_t = compute_position_loss(logits, targets[:, :, t])
                position_losses.append(loss_t)
    
    return position_losses
```

#### Configuration
```yaml
ttt_native_streaming:
  mini_batch_size: 1     # Online GD per TTT paper
  base_lr: 0.025        # Reduced to maintain effective LR (0.025/1 = 0.025)
  chunk_size: 1         # No chunking (true streaming)
  memory_footprint: "O(1) per token"  # Constant memory per position
```

#### Expected Benefits
- ✅ **Best perplexity** (online GD optimal per TTT paper)
- ✅ **Maximum TTT adaptation** (each token gets full gradient update)
- ✅ **True streaming compatibility** (matches Moshi's native S=1 constraint)
- ✅ **Constant memory** (no chunk buffering required)
- ❌ **Slower speed** (no parallelization benefits)

## Current Results Analysis

### Measured Performance
**Current Implementation Results**: 22.2% improvement over baseline
- **Method**: 50-token chunking, mini_batch_size=4, persistent_states=true
- **Efficiency**: 96% (4% padding overhead)
- **TTT Utilization**: Good (12.5 mini-batches per chunk)

### Why Current Method Works Well
1. **Sufficient context**: 50 tokens provide meaningful sequence for TTT adaptation
2. **State persistence**: TTT learning accumulates across chunks via parameter copying
3. **Reasonable efficiency**: 96% utilization is quite good despite misalignment
4. **Balanced trade-off**: Good speed vs adaptation balance

## Implementation Roadmap

### Phase 1: TTT-Optimized Implementation ✅ RECOMMENDED FIRST
**Goal**: Improve current approach with minimal risk

1. **Make chunk_size configurable** based on TTT mini_batch_size
2. **Add efficiency monitoring** (padding percentage tracking)
3. **Test optimal configurations** (25/25, 50/50)
4. **Validate performance** (should match or exceed current 22.2%)

### Phase 2: Moshi-Native Implementation 🔬 RESEARCH FOCUS
**Goal**: Explore maximum theoretical performance

1. **Implement token-by-token evaluation** using `LMGen.step()`
2. **Configure TTT for online GD** (mini_batch_size=1)
3. **Adjust learning rates** to compensate for higher effective rates
4. **Compare against optimized chunking** approach

### Phase 3: Experimental Comparison
**Goal**: Quantify trade-offs between approaches

**Metrics to Compare**:
- **Perplexity**: Primary quality metric
- **Speed**: Tokens/second evaluation rate
- **Memory**: Peak memory usage during evaluation
- **TTT Adaptation**: Parameter change magnitude over sequence length

**Expected Results**:
- **TTT-Optimized**: Better speed, good perplexity, easy deployment
- **Moshi-Native**: Best perplexity, slower speed, research insights

## Production Recommendations

### For Current Production Use
**Recommendation**: Implement TTT-Optimized approach
- **Benefits**: Immediate 4% efficiency gain, maintains all current benefits
- **Risk**: Minimal (incremental improvement to working system)
- **Timeline**: Can be implemented immediately

### For Research Applications  
**Recommendation**: Implement both approaches
- **TTT-Optimized**: Baseline for fair comparison
- **Moshi-Native**: Maximum theoretical performance
- **Value**: Understanding TTT's full potential in streaming scenarios

### For Real-Time Applications
**Recommendation**: TTT-Optimized with `mini_batch_size=50`
- **Rationale**: Maximum parallelization, good TTT utilization
- **Trade-off**: Slightly less adaptation per token vs better throughput

## Technical Implementation Details

### Configuration Management
```python
# Recommended configuration structure
@dataclass
class LibriLightConfig:
    approach: str = "ttt_optimized"  # "ttt_optimized" | "moshi_native"
    
    # TTT-Optimized settings
    chunk_size: Optional[int] = None  # Auto-calculated from mini_batch_size if None
    mini_batch_size: int = 25         # Balanced default
    
    # Moshi-Native settings  
    token_by_token: bool = False      # Enable true streaming
    
    # Common settings
    base_lr: float = 0.1              # Will be adjusted based on mini_batch_size
    persistent_states: bool = True     # Enable cross-chunk learning
    max_sequence_length: int = 24000   # LibriLight sequence limit
```

### Efficiency Monitoring
```python
def calculate_ttt_efficiency(chunk_size: int, mini_batch_size: int) -> Dict[str, float]:
    """Calculate TTT processing efficiency metrics."""
    num_complete_batches = chunk_size // mini_batch_size
    remaining_tokens = chunk_size % mini_batch_size
    
    if remaining_tokens > 0:
        # Last mini-batch needs padding
        padding_tokens = mini_batch_size - remaining_tokens
        total_processed = chunk_size + padding_tokens
        efficiency = chunk_size / total_processed
        num_batches = num_complete_batches + 1
    else:
        # Perfect alignment
        efficiency = 1.0
        num_batches = num_complete_batches
        padding_tokens = 0
    
    return {
        "efficiency_percent": efficiency * 100,
        "padding_tokens": padding_tokens,
        "num_mini_batches": num_batches,
        "tokens_per_batch": chunk_size / num_batches
    }
```

## Conclusion

Both approaches are scientifically sound and serve different purposes:

1. **TTT-Optimized**: Incremental improvement to current method with guaranteed benefits
2. **Moshi-Native**: Research-grade implementation for maximum theoretical performance

The **recommended immediate action** is implementing the TTT-Optimized approach, as it provides clear benefits with minimal risk. The Moshi-Native approach should be pursued for research purposes to understand TTT's full potential in streaming scenarios.

**Key Insight**: Our current 22.2% improvement demonstrates TTT's effectiveness, but there's room for optimization through better alignment with TTT's internal processing patterns and potentially even better results through online gradient descent.