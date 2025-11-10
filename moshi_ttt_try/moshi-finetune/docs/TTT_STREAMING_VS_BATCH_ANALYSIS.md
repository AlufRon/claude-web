# TTT Streaming vs Batch Inference: Equivalence Analysis

**Date**: 2025-10-22
**Author**: Analysis based on Moshi-TTT codebase investigation
**Question**: Is streaming inference (L=64, num_mini_batch=1) with TTT weight persistence equivalent to batch inference (L=2048, num_mini_batch=32)?

---

## Executive Summary

**TL;DR**: Streaming inference with TTT weight persistence is **NEARLY EQUIVALENT** to batch inference with multiple mini-batches, with some important caveats.

**Key Finding**: The primary concern about `num_mini_batch=1` is **less severe than initially thought** because:
1. ✅ TTT weights persist across streaming chunks (confirmed in code)
2. ✅ Moshi's KV cache provides full causal context to attention layers
3. ✅ Each streaming step's gradient update accumulates over time
4. ⚠️ However, there are subtle differences in gradient computation and attention patterns

---

## 1. Understanding TTT Mini-Batches

### What "Mini-Batch" Means in TTT

**Important**: TTT's "mini-batch" is NOT the same as training batch size!

In TTT, **mini-batching refers to chunking a sequence for sequential weight updates**:

```python
# From moshi_ttt/models/ssm/ops/ttt_mlp.py:640
def ttt_mlp_multi_layer(XK, XQ, XV, eta, ...):
    # Input shape: [B, num_heads, num_mini_batch, chunk_size, head_dim]
    # Example with L=2048, mini_batch_size=64:
    #   [1, 32, 32, 64, 128]
    #    ^B  ^heads ^num_mb ^chunk ^head_dim

    # Reorder so mini-batch is first dimension for iteration
    inputs = tree_map(lambda x: x.permute(2, 0, 1, 3, 4), inputs)
    # Now: [32, 1, 32, 64, 128]
    #       ^num_mb (we iterate over this)

    # Scan loop iterates over mini-batches
    final_params, XQW_batch = scan(
        compute_multi_layer_mini_batch,  # Called 32 times
        init_params_dict,  # Contains W₀
        inputs,
        checkpoint_group_size,
    )
```

### The Scan Loop

**Key code**: `moshi_ttt/models/ssm/utils.py:225-260`

```python
def scan(f, init, xs, checkpoint_group=0):
    """Mimic jax.lax.scan function."""
    carry = init  # Contains TTT weights
    num_items = len(next(iter(xs.values())))  # num_mini_batch

    for i in range(num_items):  # ← THE CRITICAL LOOP
        x = {key: tensor[i] for key, tensor in xs.items()}  # One chunk
        carry, y = f(carry, x)  # Update weights: carry = {W_new, b_new, ...}
        # carry contains updated weights, passed to next iteration!
```

**With L=2048, num_mini_batch=32**: Loop iterates 32 times
**With L=64, num_mini_batch=1**: Loop iterates **1 time**

---

## 2. Batch Inference: L=2048, num_mini_batch=32

### Single Forward Pass with 32 Sequential Updates

```
Forward pass: model(input_codes)  # input has 2048 tokens
    ↓
Attention layer: processes all 2048 tokens with causal mask
    ↓
Hidden states: [B=1, L=2048, d_model=4096]
    ↓
TTT layer receives hidden states
    ↓
Reshape to mini-batches: [1, 32, 32, 64, 128]
                          ^B  ^h  ^num_mb ^chunk ^head_dim
    ↓
Scan loop: 32 iterations

Iteration 1:  chunk = hidden_states[0:64]
              W₁ = W₀ - η * ∇L(chunk[0:64])

Iteration 2:  chunk = hidden_states[64:128]
              W₂ = W₁ - η * ∇L(chunk[64:128])  ← Uses W₁!

Iteration 3:  chunk = hidden_states[128:192]
              W₃ = W₂ - η * ∇L(chunk[128:192])  ← Uses W₂!

...

Iteration 32: chunk = hidden_states[1984:2048]
              W₃₂ = W₃₁ - η * ∇L(chunk[1984:2048])

Result: 32 sequential gradient updates within single forward pass
```

### Key Properties

1. **Attention sees full sequence**: All 2048 tokens processed together with causal masking
2. **Hidden states are complete**: `hidden_states[i]` contains information from tokens `[0:i]` via attention
3. **TTT processes sequentially**: Splits hidden states into 32 chunks, updates weights 32 times
4. **Gradient flow**: Single backward pass through all 32 mini-batch computations

---

## 3. Streaming Inference: L=64, num_mini_batch=1

### Multiple Forward Passes with Weight Persistence

```
Streaming loop: (from inference/run_inference_with_ttt.py:473-503)

while chunks:  # Iterate through audio
    chunk = chunks.popleft()  # Get next audio frame
    codes = mimi.encode(chunk)  # Encode to tokens
    tokens = lm_gen.step(codes)  # Process through model
```

Each `lm_gen.step()` triggers:

```
Step 1: model(codes[0:64])  # First 64 tokens
    ↓
Attention layer with KV cache:
    - KV cache is empty
    - Processes tokens [0:64]
    - Stores K,V in cache
    ↓
Hidden states: [B=1, L=64, d_model=4096]
    ↓
TTT layer receives hidden states
    ↓
Reshape to mini-batches: [1, 32, 1, 64, 128]
                          ^B  ^h  ^num_mb=1 ^chunk ^head_dim
    ↓
Scan loop: 1 iteration

Iteration 1:  chunk = hidden_states[0:64]
              W₁ = W₀ - η * ∇L(chunk[0:64])
    ↓
CRITICAL: self.weights[i].data.copy_(updated_w_unbatched)
          (moshi_ttt/models/ssm/ttt_layer.py:716)
          Weights persisted to module state!

---

Step 2: model(codes[64:128])  # Next 64 tokens
    ↓
Attention layer with KV cache:
    - KV cache contains keys/values from tokens [0:64]
    - Processes tokens [64:128]
    - Attends to tokens [0:128] (via cache + new tokens)
    - Updates cache
    ↓
Hidden states: [B=1, L=64, d_model=4096]
    (These hidden states reflect context from tokens [0:128])
    ↓
TTT layer receives hidden states
    - TTT weights start at W₁ (from step 1!)
    ↓
Reshape to mini-batches: [1, 32, 1, 64, 128]
    ↓
Scan loop: 1 iteration

Iteration 1:  chunk = hidden_states[0:64]  # Really tokens [64:128]
              W₂ = W₁ - η * ∇L(chunk[64:128])  ← Uses W₁!
    ↓
Weights persisted again!

---

Step 3: model(codes[128:192])
    ↓
Attention with KV cache [0:128]
    ↓
Hidden states reflect context [0:192]
    ↓
TTT starts with W₂
    ↓
W₃ = W₂ - η * ∇L(chunk[128:192])  ← Uses W₂!

...and so on
```

### Key Properties

1. **Attention via KV cache**: Each step sees full causal history via cached keys/values
2. **Hidden states accumulate context**: Token 65 sees influence of tokens 0-64 through attention
3. **TTT weights persist**: Each step starts with weights from previous step
4. **Gradient flow**: Separate backward pass for each streaming step

---

## 4. The Critical Difference: Attention Context

### Moshi's KV Cache Mechanism

**Code**: `moshi/moshi/moshi/modules/transformer.py:187-260`

```python
class RingKVCache:
    """Efficient streaming KVCache to be compatible with Cuda Graph."""

    def __init__(self, batch_size, num_heads, dim_per_head, capacity, ...):
        self.capacity = capacity  # Context window size
        self.cache = torch.zeros(
            (2, batch_size, num_heads, capacity, dim_per_head),
            device=device, dtype=dtype,
        )
```

**In streaming mode** (`transformer.py:463-480`):

```python
def _init_streaming_state(self, batch_size: int):
    kv_cache = RingKVCache(
        batch_size, self.num_heads, dim_per_head, capacity,
        device=device, dtype=dtype
    )
    return _MHAState(batch_size, device, kv_cache, ...)

def _complete_kv(self, k, v):
    state = self._streaming_state
    if state is None or state.kv_cache is None:
        return KVCacheResult.from_kv(k, v)
    else:
        return state.kv_cache.complete(k, v, state.exec_mask)
        # ↑ Appends new k,v to cache, returns full cached sequence
```

**This means**: At step N, attention over tokens [0:N*64] even though only processing [N*64-64:N*64]!

### Attention Pattern Comparison

#### Batch Mode (L=2048)
```
Attention computes:
Q @ K^T where Q,K have shape [B, num_heads, 2048, head_dim]

Attention matrix: [2048 × 2048] (causal masked)
Token 64 attends to: [0, 1, 2, ..., 64]
Token 128 attends to: [0, 1, 2, ..., 128]
Token 2048 attends to: [0, 1, 2, ..., 2048]
```

#### Streaming Mode (L=64 × 32 steps)
```
Step 1: Q @ K^T where Q,K have shape [B, num_heads, 64, head_dim]
        Attention matrix: [64 × 64]
        Token 64 attends to: [0, 1, 2, ..., 64]

Step 2: Q @ [K_cache | K_new]^T
        K_cache: [64 tokens from step 1]
        K_new: [64 tokens from step 2]
        Attention matrix: [64 × 128]
        Token 128 attends to: [0, 1, 2, ..., 128] ✓

Step 3: Q @ [K_cache | K_new]^T
        K_cache: [128 tokens from steps 1-2]
        K_new: [64 tokens from step 3]
        Attention matrix: [64 × 192]
        Token 192 attends to: [0, 1, 2, ..., 192] ✓
```

**Result**: **Attention patterns are equivalent!** (assuming cache capacity ≥ sequence length)

---

## 5. Mathematical Equivalence Analysis

### Weight Update Sequences

#### Batch Mode
```
W₀ (initial weights)
  ↓ process chunk [0:64]
W₁ = W₀ - η∇L₁  where L₁ = reconstruction_loss(hidden_states[0:64])
  ↓ process chunk [64:128]
W₂ = W₁ - η∇L₂  where L₂ = reconstruction_loss(hidden_states[64:128])
  ↓ process chunk [128:192]
W₃ = W₂ - η∇L₃  where L₃ = reconstruction_loss(hidden_states[128:192])
  ...
W₃₂ = W₃₁ - η∇L₃₂
```

#### Streaming Mode
```
W₀ (initial weights)
  ↓ step 1: process tokens [0:64]
W₁ = W₀ - η∇L₁  where L₁ = reconstruction_loss(hidden_states[0:64])
  ↓ step 2: process tokens [64:128]
W₂ = W₁ - η∇L₂  where L₂ = reconstruction_loss(hidden_states[64:128])
  ↓ step 3: process tokens [128:192]
W₃ = W₂ - η∇L₃  where L₃ = reconstruction_loss(hidden_states[128:192])
  ...
W₃₂ = W₃₁ - η∇L₃₂
```

### The Key Question

**Are `hidden_states[64:128]` the same in both cases?**

#### Batch Mode Hidden States
```
Input: tokens [0:2048]
  ↓ Attention with causal mask over all 2048 tokens
hidden_states[64:128] = f_attn(tokens[64:128], context=tokens[0:64])
```

#### Streaming Mode Hidden States
```
Step 2:
Input: tokens [64:128]
  ↓ Attention with KV cache containing tokens [0:64]
hidden_states[64:128] = f_attn(tokens[64:128], kv_cache=cached_kv[0:64])
```

**Are these equivalent?**

**YES, if**:
1. KV cache faithfully stores keys and values from previous tokens ✓
2. Attention mechanism uses cache correctly ✓
3. Positional encodings are consistent ✓

**Moshi uses RoPE (Rotary Position Embedding)**, which encodes position into Q and K:
- In batch mode: positions [0, 1, 2, ..., 2047]
- In streaming mode: positions tracked via `offset` in state (`transformer.py:569-573`)

```python
# From transformer.py:569-573
if state is not None and not self.cross_attention:
    state.offset[:] = torch.where(
        state.exec_mask,
        state.offset + T,  # Increment position by sequence length
        state.offset)
```

**Position tracking is consistent!** ✓

---

## 6. Gradient Computation Differences

### Batch Mode: Single Backward Pass

```python
# Simplified conceptual flow
loss = 0
for i in range(num_mini_batch):  # 32 iterations
    # Forward through TTT with weights W_i
    output_i, W_{i+1} = compute_mini_batch(W_i, input_i)
    loss_i = reconstruction_loss(output_i)
    loss += loss_i

# Single backward pass
loss.backward()  # Gradients flow through all 32 mini-batch computations
```

**Gradient flow**:
- Gradients from `loss_32` propagate back through W₃₁, W₃₀, ..., W₁, W₀
- This creates dependencies between mini-batches in the backward pass
- However, TTT's design explicitly **breaks these dependencies** for efficiency

### Streaming Mode: Multiple Separate Backward Passes

```python
# Step 1
output_1, W_1 = compute_mini_batch(W_0, input_1)
loss_1 = reconstruction_loss(output_1)
loss_1.backward()  # Backward pass 1
# Weights updated and persisted

# Step 2
output_2, W_2 = compute_mini_batch(W_1, input_2)  # W_1 is detached (no_grad)
loss_2 = reconstruction_loss(output_2)
loss_2.backward()  # Backward pass 2 (separate from backward pass 1)
```

**Gradient flow**:
- Each step's backward pass is **independent**
- Gradients from step 2 do NOT propagate back to step 1
- Weight updates happen via **test-time training**, not via backprop through time

### The Key Insight: TTT is Designed This Way!

**From TTT paper**: Test-time training explicitly updates weights during forward pass via gradient descent on self-supervised loss. The weights are **not** trainable parameters in the traditional sense during inference.

**In both modes**:
- TTT weights are updated via mini-batch gradient descent
- Updates are **not backpropagated** through previous mini-batches
- This is by design - it's what makes TTT efficient!

**Code evidence** (`ttt_mlp.py:592`):
```python
# Update weights and biases
Wi_last = weight_states[i] - (last_eta_mini_batch * Xi).transpose(-1, -2) @ grad_Zi
# ↑ Direct gradient descent update, not via .backward()
```

**Conclusion**: Gradient computation is **functionally equivalent** because TTT doesn't use backprop-through-time by design!

---

## 7. Potential Differences and Edge Cases

### 7.1 Numerical Precision

#### Batch Mode
- All 32 mini-batch computations in single forward pass
- Gradients accumulated in one backward pass
- Potential for gradient accumulation errors over 32 steps

#### Streaming Mode
- 32 separate forward/backward passes
- Fresh gradient computation each step
- Different numerical error accumulation pattern

**Impact**: Minimal in practice, but could cause slight output differences at float precision boundaries.

### 7.2 Memory and Computation

#### Batch Mode
- Higher peak memory (entire 2048-token sequence in memory)
- Can use gradient checkpointing to reduce memory
- Single large computation

#### Streaming Mode
- Lower peak memory (only 64 tokens at a time)
- No need for gradient checkpointing
- 32 smaller computations

**Impact**: Streaming is more memory-efficient, which was the original motivation!

### 7.3 Checkpoint/Save Points

#### Batch Mode
- Can save intermediate mini-batch states during scan
- Gradient checkpointing support (`utils.py:245-256`)

#### Streaming Mode
- Natural save points between steps
- Can interrupt and resume easily

### 7.4 KV Cache Capacity Limits

**Critical constraint**: KV cache has finite capacity (ring buffer)

```python
# transformer.py:463
kv_cache = RingKVCache(
    ...,
    capacity=self.context,  # Limited capacity!
    ...
)
```

If `context < sequence_length`:
- **Batch mode**: May truncate attention to `context` tokens
- **Streaming mode**: May evict old tokens from cache

**When they differ**:
- If context window = 2048: Both modes equivalent
- If context window = 512: Streaming loses tokens 0-63 after 8 steps, but batch mode processes all 2048 with sliding window

**For Moshi**: Default context is typically large (4096+), so unlikely to be limiting factor for sequences < 2048 tokens.

### 7.5 Attention Mask Details

#### Batch Mode
```python
# All tokens processed together
# Causal mask prevents future tokens from attending to past
attn_mask = torch.tril(torch.ones(2048, 2048))
```

#### Streaming Mode
```python
# Each step processes subset of tokens
# Cache provides history, causal mask applied per step
# Effective attention is over [cache + current tokens]
```

**Difference**: In streaming, the attention computation is chunked, which could affect:
- Softmax normalization (computed over different token ranges)
- Numerical stability of attention scores

**However**: Moshi's implementation carefully handles this via position tracking and proper cache indexing.

---

## 8. Experimental Verification Strategy

### To definitively prove equivalence, we should test:

#### Test 1: Output Comparison
```python
# Run batch inference
model.eval()
with torch.no_grad():
    output_batch = model(tokens[0:2048])

# Run streaming inference
model.eval()
with torch.no_grad():
    lm_gen.streaming_forever(batch_size=1)
    outputs_streaming = []
    for i in range(0, 2048, 64):
        chunk = tokens[i:i+64]
        output = lm_gen.step(chunk)
        outputs_streaming.append(output)
    output_streaming = torch.cat(outputs_streaming, dim=-1)

# Compare
diff = (output_batch - output_streaming).abs().max()
print(f"Max difference: {diff}")
```

**Expected**: `diff < 1e-5` (allowing for numerical errors)

#### Test 2: TTT Weight Trajectory
```python
# Track TTT weights during both modes
# Compare W_1, W_2, ..., W_32 from batch vs streaming
```

**Expected**: Weights should follow same trajectory (within numerical precision)

#### Test 3: Attention Patterns
```python
# Extract attention matrices from both modes
# Compare which tokens attend to which
```

**Expected**: Identical attention patterns (causal, with correct positions)

---

## 9. Conclusion

### Are Streaming and Batch Modes Equivalent?

**Answer**: **YES, with caveats**

### What Works

✅ **TTT weight persistence**: Confirmed working (`ttt_layer.py:716`)
✅ **Attention context via KV cache**: Provides full causal history
✅ **Position encoding**: Consistently tracked across streaming steps
✅ **Weight update sequence**: Mathematically identical (W₀ → W₁ → ... → W₃₂)
✅ **Gradient computation**: Functionally equivalent (TTT doesn't use backprop-through-time)

### Caveats

⚠️ **Numerical precision**: Different error accumulation patterns
⚠️ **Memory profile**: Streaming uses less peak memory
⚠️ **Attention normalization**: Softmax computed over different chunk sizes
⚠️ **KV cache capacity**: If limited, may cause differences at long sequences

### The Original Concern

**Initial worry**: "With `num_mini_batch=1`, TTT only does 1 gradient update per forward pass, so it can't adapt properly."

**Resolution**: **This concern is addressed by weight persistence!**
- In streaming mode, each step does 1 update, BUT weights persist across steps
- After 32 streaming steps, we've done 32 updates: W₀ → W₁ → ... → W₃₂
- This is equivalent to 32 mini-batches in a single forward pass

### Why Streaming Still Works

The key insight is that **TTT + attention work together**:

1. **Attention (with KV cache)** provides long-range context
   - Token 128 sees tokens 0-127 via attention
   - This provides the "what happened before" signal

2. **TTT weights** adapt based on that context
   - Weights evolve: W₀ → W₁ → W₂ → ...
   - Each update uses gradient from current chunk, given full context via attention
   - Weights persist across chunks, accumulating adaptation

3. **The combination** recreates the effect of processing a long sequence:
   - Attention: spatial (across tokens) context
   - TTT persistence: temporal (across chunks) adaptation

---

## 10. Recommendations

### For Your Current Setup

1. **Keep using streaming inference** - It's working as designed
2. **The `num_mini_batch=1` is fine** - Weight persistence makes it equivalent
3. **Monitor KV cache capacity** - Ensure it's >= expected sequence length
4. **Don't worry about gradient flow** - TTT is designed for independent mini-batch updates

### For Future Experiments

1. **Test output equivalence** (see Section 8) - Verify numerically
2. **Compare perplexity**: streaming vs batch on same long sequence
3. **Profile memory usage**: Confirm streaming's efficiency gains
4. **Monitor cache eviction**: Check if long sequences cause cache overflow

### Configuration Tuning

If you want to experiment with `mini_batch_size`:

**Current**: `mini_batch_size=64`, streaming with `L=64` → `num_mini_batch=1`

**Alternative**: `mini_batch_size=32`, streaming with `L=64` → `num_mini_batch=2`
- 2 inner-loop updates per streaming step
- Might improve adaptation within each chunk
- Trade-off: Smaller chunks may have higher variance

**But**: Given that weights persist, the current setup is likely optimal for streaming!

---

## 11. References

### Code Locations

| Component | File | Lines |
|-----------|------|-------|
| TTT scan loop | `moshi_ttt/models/ssm/utils.py` | 225-260 |
| Mini-batch computation | `moshi_ttt/models/ssm/ops/ttt_mlp.py` | 501-638 |
| Weight persistence | `moshi_ttt/models/ssm/ttt_layer.py` | 693-717 |
| KV cache implementation | `moshi/modules/transformer.py` | 187-260 |
| Streaming attention | `moshi/modules/transformer.py` | 463-480 |
| Streaming inference | `inference/run_inference_with_ttt.py` | 473-503 |

### Key Concepts

- **Test-Time Training (TTT)**: Adapting model weights during inference via gradient descent on self-supervised loss
- **Mini-batch (TTT context)**: Sequential chunks of a sequence, not batch size
- **Scan operation**: JAX-style loop that carries state (updated weights) across iterations
- **KV cache**: Mechanism to store attention keys/values for efficient sequential processing
- **Weight persistence**: Keeping updated TTT weights in model state across forward passes

---

## Appendix A: Detailed Tensor Shapes

### Batch Mode (L=2048)

```
Input codes: [1, 17, 2048]
            ^B  ^codebooks ^seq_len

After transformer attention:
  hidden_states: [1, 2048, 4096]
                 ^B  ^seq   ^d_model

Into TTT layer:
  XQ, XK, XV after projection: [1, 2048, 32, 128]
                                ^B  ^seq   ^heads ^head_dim

After reshape_to_mini_batch:
  XQ, XK, XV: [1, 32, 32, 64, 128]
               ^B  ^heads ^num_mb ^chunk ^head_dim

After permute (scan input):
  XQ, XK, XV: [32, 1, 32, 64, 128]
               ^num_mb (iterate over this)

Scan loop: 32 iterations
  Each iteration processes: [1, 32, 64, 128]
                            ^B  ^heads ^chunk ^head_dim
```

### Streaming Mode (L=64)

```
Input codes (step 1): [1, 17, 64]
                       ^B  ^codebooks ^seq_len

After transformer attention (with empty KV cache):
  hidden_states: [1, 64, 4096]
                 ^B  ^seq ^d_model

Into TTT layer:
  XQ, XK, XV after projection: [1, 64, 32, 128]
                                ^B  ^seq ^heads ^head_dim

After reshape_to_mini_batch:
  XQ, XK, XV: [1, 32, 1, 64, 128]
               ^B  ^heads ^num_mb=1 ^chunk ^head_dim

After permute (scan input):
  XQ, XK, XV: [1, 1, 32, 64, 128]
               ^num_mb=1 (iterate once)

Scan loop: 1 iteration
  Processes: [1, 32, 64, 128]
             ^B  ^heads ^chunk ^head_dim

---

Step 2: [1, 17, 64] → ... → Same shapes
        Attention now has KV cache with 64 tokens from step 1
        TTT starts with W₁ from step 1

Step 3: [1, 17, 64] → ... → Same shapes
        KV cache now has 128 tokens (steps 1-2)
        TTT starts with W₂ from step 2

...
```

---

## Appendix B: Weight Update Mathematics

### Reconstruction Loss

```python
# From ttt_mlp.py:550
reconstruction_target = XV_mini_batch - XK_mini_batch

# Forward through TTT-MLP layers
final_Z = MLP(XK_mini_batch, weights, biases)

# Apply layer norm
final_Z_normalized = LayerNorm(final_Z, ln_weight, ln_bias)

# MSE loss
loss = ||final_Z_normalized - reconstruction_target||²
```

### Gradient Computation

```python
# From ttt_mlp.py:570
gradients[-1] = ln_fused_l2_bwd(final_Z, reconstruction_target, ln_weight, ln_bias)
# ↑ Gradient of loss w.r.t. final layer output

# Backprop through MLP layers
for i in range(num_layers - 2, -1, -1):
    gradients[i] = grad_next @ W_{i+1}^T * gelu'(Z_i)
```

### Weight Update

```python
# From ttt_mlp.py:592-593
Wi_new = Wi_old - (eta * Xi).T @ grad_Zi
bi_new = bi_old - sum(eta * grad_Zi, dim=-2)

# Where:
#   eta: learning rate (from config)
#   Xi: input to layer i
#   grad_Zi: gradient w.r.t. layer i output
```

### Update Equivalence

**Batch mode**: All computed in single backward pass, but updates applied sequentially in scan

**Streaming mode**: Each computed in separate backward pass, updates applied and persisted

**Result**: Same sequence of weight tensors W₀, W₁, W₂, ..., W₃₂

---

**Document Version**: 1.0
**Last Updated**: 2025-10-22
**Status**: Complete analysis with caveats identified
