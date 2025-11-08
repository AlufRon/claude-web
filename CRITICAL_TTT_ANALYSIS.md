# CRITICAL TTT Implementation Analysis

**Date**: 2025-11-08
**Status**: ⚠️ CURRENT IMPLEMENTATION INCOMPLETE - MAJOR ISSUES FOUND

## Executive Summary

After deep analysis of the original TTT implementations (ttt-video-dit, ttt-lm-kernels) compared to our current implementation, I've identified **7 CRITICAL missing components**. The current implementation only supports training/prefill mode and is **completely missing the inference/decode path** required for long-form speech generation.

**Bottom line**: The current implementation will NOT work for actual generation. It needs major rework before testing.

---

## Part 1: What We Got RIGHT ✅

### 1.1 Float32 for Inner States
```python
# Our implementation (CORRECT):
self.W1 = nn.Parameter(torch.normal(..., dtype=torch.float32))
self.b1 = nn.Parameter(torch.zeros(..., dtype=torch.float32))
```
✅ All W1, b1 parameters are float32 as required.

### 1.2 Mini-Batch Reshaping
```python
# Our implementation (CORRECT):
def reshape_for_mini_batches(self, hidden_states):
    B, L, D = hidden_states.shape
    K = self.mini_batch_size
    if L % K != 0:
        raise ValueError(...)
    num_mb = L // K
    reshaped = hidden_states.reshape(B, num_mb, K, D)
    return reshaped, num_mb
```
✅ Correctly reshapes sequences into mini-batches.

### 1.3 RoPE Within Mini-Batches
```python
# Our implementation (CORRECT):
cos, sin = precompute_freqs_cis(self.head_dim, self.mini_batch_size, ...)
```
✅ RoPE is computed for positions 0 to K-1 within each mini-batch, not globally.

### 1.4 Analytical Gradients
```python
# Our implementation (CORRECT):
grad_l_wrt_Z1 = ln_fused_l2_bwd(Z1, reconstruction_target, ...)
W1_last = W1_init - (last_eta * XK).transpose(-1, -2) @ grad_l_wrt_Z1
```
✅ Uses analytical gradients from ln_fused_l2_bwd, not autograd.

### 1.5 Sequential Mini-Batch Processing
```python
# Our implementation (CORRECT):
final_params, XQW_batch = scan(compute_ttt_linear_mini_batch, init_params_dict, inputs, ...)
```
✅ Processes mini-batches sequentially with state carrying forward.

---

## Part 2: What We Got WRONG ❌

### 2.1 ❌ CRITICAL: Missing Decode Path

**Issue**: Our implementation only has the PREFILL path (processing full mini-batches). The DECODE path (processing ONE token at a time during generation) is **completely missing**.

**Why this matters**: During text generation, the model generates one token at a time. TTT must:
1. Accumulate gradients across K tokens in a mini-batch
2. Apply the accumulated gradients only at the K-th token
3. Reset gradient buffers for the next mini-batch

**What the original does**:
```python
# From ttt-lm-kernels/ttt/modeling_ttt.py

# For tokens 1 to K-1 (NOT last in mini-batch):
def ttt_linear_decode_token(states, inputs, ...):
    W1 = states['W1_init']
    W1_grad = states['W1_grad']  # Accumulated gradients

    # Compute gradient for this token
    W1_grad.add_(XK.transpose(-1, -2) @ ilr_mul_dl_dZ1)  # ACCUMULATE

    # Compute output using TEMPORARY updated weights (don't modify W1!)
    W1_bar = W1 - (token_idx * W1_grad)  # Temporary
    Z1_bar = XQ @ W1_bar + b1_bar

    return Z1_bar  # W1_grad still contains accumulated gradients

# For token K (LAST in mini-batch):
def ttt_linear_decode_last_token_in_mini_batch(states, inputs, ...):
    W1 = states['W1_init']
    W1_grad = states['W1_grad']

    # Final accumulation
    W1_grad.add_(XK.transpose(-1, -2) @ ilr_mul_dl_dZ1)

    # ACTUALLY update W1 (not temporary)
    W1.sub_(token_idx * W1_grad)
    b1.sub_(token_idx * b1_grad)

    # Reset gradients for next mini-batch
    W1_grad.zero_()
    b1_grad.zero_()

    Z1_bar = XQ @ W1 + b1
    return Z1_bar
```

**What we have**:
```python
# NOTHING! We only have compute_ttt_linear_mini_batch which processes K tokens at once.
```

**Impact**: ⚠️ **BLOCKER** - Cannot use model for generation at all.

---

### 2.2 ❌ CRITICAL: Missing Gradient Buffers

**Issue**: The cache needs to store gradient accumulators (W1_grad, b1_grad), not just the states (W1, b1).

**What the original does**:
```python
# From ttt-lm-kernels/ttt/generation.py:TTTCache

class TTTCache:
    def __init__(self, ...):
        self.params_dict = {
            "W1_init": [...],      # Actual state [B*nh, f, f]
            "W1_grad": [...],      # Gradient accumulator [B*nh, f, f]
            "b1_init": [...],      # Actual state [B*nh, 1, f]
            "b1_grad": [...],      # Gradient accumulator [B*nh, 1, f]
            "conv_cache": [...],   # Conv1d state
        }
```

**What we have**:
```python
# llama-omni/omni_speech/model/ttt/cache.py:TTTCache

@dataclass
class TTTCache:
    params_dict: Dict = field(default_factory=lambda: defaultdict(dict))

    # We only store W1_states and b1_states, NO gradient buffers!
```

**Impact**: ⚠️ **BLOCKER** - Cannot accumulate gradients during decode.

---

### 2.3 ❌ CRITICAL: Missing Token Index

**Issue**: During decode, TTT needs to know the position within the current mini-batch (0 to K-1) to:
1. Determine which RoPE embeddings to use
2. Scale the gradient update appropriately
3. Detect when to apply accumulated gradients (at position K-1)

**What the original does**:
```python
# From ttt-lm-kernels/ttt/modeling_ttt.py:256-259

# Pre-computed reciprocal of position: [1, 1/2, 1/3, ..., 1/K]
token_idx = 1. / torch.arange(1, self.mini_batch_size + 1).reshape(1, 1, -1, 1)
self.register_buffer('token_idx', token_idx, persistent=False)

# During decode:
inner_mini_batch_step_offset = cache_params.seqlen_offset % self.mini_batch_size
token_idx = self.token_idx[:, :, inner_mini_batch_step_offset, :] + \
            self.learnable_token_idx_bias[:, :, inner_mini_batch_step_offset, :]
```

**What we have**:
```python
# NOTHING! We don't track position within mini-batch.
```

**Impact**: ⚠️ **BLOCKER** - Cannot know when to apply gradients or which RoPE to use.

---

### 2.4 ❌ CRITICAL: Missing Causal Conv1d

**Issue**: TTT-LM-Kernels applies causal 1D convolution to Q and K projections. This is important for capturing local patterns.

**What the original does**:
```python
# From ttt-lm-kernels/ttt/modeling_ttt.py:271-286, 325-375

self.conv_q = nn.Conv1d(
    self.hidden_size, self.hidden_size,
    bias=True,
    kernel_size=self.conv_kernel,  # Typically 4
    groups=self.hidden_size,  # Depthwise
    padding=self.conv_kernel - 1,
)
self.conv_k = nn.Conv1d(...)

def conv_qk_fused(self, XQK, cache_params, is_prefill):
    if is_prefill:
        XQ = causal_conv1d_fn(XQK, conv_q_weights, ...)
        XK = causal_conv1d_fn(XQK, conv_k_weights, ...)
        # Cache last few tokens for decode
        cache_params["conv_cache"][layer_idx].copy_(F.pad(XQK, (kernel-N, 0)))
    else:
        # Decode: update cache, convolve single token
        XQ = causal_conv1d_update(XQK, cache_params["conv_cache"][layer_idx], ...)
        XK = causal_conv1d_update(...)
```

**What we have**:
```python
# NOTHING! We go straight from hidden_states to Q/K/V projections.
```

**Impact**: ⚠️ **MAJOR** - Missing a component that improves quality. Not a blocker, but quality will be worse.

---

### 2.5 ❌ CRITICAL: Missing Gate Mechanism

**Issue**: The output should be gated with a learned gate projection.

**What the original does**:
```python
# From ttt-lm-kernels/ttt/modeling_ttt.py:306-323, 441-442

def _get_QKV_ttt_lr(self, hidden_states):
    XQKV_ttt_lr = self.qkv_learnable_ttt_lr_proj(hidden_states)  # Project to 3*F + nh
    XQKV, ttt_lr = torch.split(XQKV_ttt_lr, [3*D, num_heads], dim=-1)

    XQK, XGate, XV = torch.split(XQKV, self.hidden_size, dim=-1)  # Split into Q+K, Gate, V
    return XQK, XV, XGate, ttt_lr

# Output:
output_hidden_states = gelu(XGate) * self.post_norm(ttt_process_output)
```

**What we have**:
```python
# ttt_layer.py:400-403
output = self.o_proj(output)
output = self.post_norm(output)
return output  # No gate!
```

**Impact**: ⚠️ **MAJOR** - Missing gating mechanism that controls information flow. Could affect quality significantly.

---

### 2.6 ❌ MEDIUM: Wrong Forward Signature

**Issue**: Our forward() signature matches standard Llama attention, but it needs `is_prefill` and `is_last_in_mini_batch` flags to route to the correct code path.

**What the original does**:
```python
# From ttt-lm-kernels/ttt/modeling_ttt.py:466-503

def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    cache_params: Optional[TTTCache] = None,
    is_prefill: Optional[bool] = None,  # NEW!
    is_last_in_mini_batch: Optional[bool] = None,  # NEW!
):
    if is_prefill:
        # Process full mini-batches
        ttt_process_output = self.prefill_sequence(...)
    else:
        if is_last_in_mini_batch:
            # Apply accumulated gradients
            ttt_process_output = self.decode_last_token_in_mini_batch(...)
        else:
            # Accumulate gradients
            ttt_process_output = self.decode_token(...)
```

**What we have**:
```python
# ttt_layer.py:254-263
def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,  # Standard Llama name
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs
):
    # Only has prefill path!
```

**Impact**: ⚠️ **BLOCKER** - Cannot route to decode path even if we implement it.

---

### 2.7 ❌ HIGH: State Persistence Bug

**Issue**: We don't extract and return the final W1, b1 from the TTT computation.

**What should happen**:
```python
# After ttt_linear() finishes:
final_params, XQW_batch = scan(compute_ttt_linear_mini_batch, ...)

# Extract final state
W1_final = final_params["W1_states"]
b1_final = final_params["b1_states"]

# Return for caching
new_ttt_state = (W1_final, b1_final) if use_cache else None
return output, None, new_ttt_state
```

**What we have**:
```python
# ttt_layer.py:405-408
# TODO: Extract final W1, b1 from ttt_linear
new_ttt_state = None if not use_cache else (W1_init, b1_init)  # WRONG! Returns INITIAL state
```

**Impact**: ⚠️ **BLOCKER** - State doesn't persist across batches, defeating the purpose of TTT.

---

## Part 3: Detailed Comparison Table

| Feature | Original TTT-LM-Kernels | Our Implementation | Status |
|---------|------------------------|-------------------|---------|
| **Prefill Mode** | ✅ Processes K tokens at once | ✅ Implemented | ✅ DONE |
| **Decode Mode** | ✅ Processes 1 token at a time | ❌ Not implemented | ❌ MISSING |
| **Gradient Accumulation** | ✅ W1_grad, b1_grad buffers | ❌ No gradient buffers | ❌ MISSING |
| **Token Index** | ✅ Tracks position in mini-batch | ❌ No tracking | ❌ MISSING |
| **Causal Conv1d** | ✅ On Q and K | ❌ Not implemented | ❌ MISSING |
| **Gate Mechanism** | ✅ gelu(gate) * norm(output) | ❌ Just norm(output) | ❌ MISSING |
| **Float32 States** | ✅ W1, b1 are float32 | ✅ W1, b1 are float32 | ✅ DONE |
| **RoPE in Mini-Batch** | ✅ position % K | ✅ Correct | ✅ DONE |
| **L2 Normalize Q/K** | ✅ F.normalize(Q/K, p=2) | ✅ Implemented | ✅ DONE |
| **Learnable TTT LR** | ✅ Per-head projection | ✅ Per-head projection | ✅ DONE |
| **State Persistence** | ✅ Returns final W1, b1 | ❌ Returns initial | ❌ BUG |
| **Forward Signature** | ✅ Has is_prefill, is_last_in_mb | ❌ Standard Llama | ❌ WRONG |
| **Cache Structure** | ✅ W1/b1 + grads + conv | ❌ Only W1/b1 | ❌ INCOMPLETE |

---

## Part 4: Architecture Differences

### 4.1 TTT-Video-Dit (Training-Focused)
- **Purpose**: Non-causal video generation
- **Mode**: Prefill only (no autoregressive decode)
- **Features**:
  - Processes full sequences in mini-batches
  - No conv1d (not needed for non-causal)
  - Simpler: just W1, b1 updates
  - Used for training and full-sequence inference

### 4.2 TTT-LM-Kernels (Inference-Focused)
- **Purpose**: Causal language modeling with autoregressive generation
- **Mode**: Both prefill AND decode
- **Features**:
  - Prefill: Processes full sequences
  - Decode: One token at a time with gradient accumulation
  - Causal conv1d for local patterns
  - Gate mechanism for output
  - Triton kernels for fast decode
  - **THIS IS WHAT WE NEED FOR LLAMA-OMNI**

### 4.3 Our Implementation
- **Purpose**: Attempted Llama-Omni integration
- **Mode**: Only prefill (incomplete)
- **Issues**: Missing all decode-specific features

---

## Part 5: Training vs Inference Requirements

### 5.1 Training (What We Have)
```python
# Training forward pass:
for batch in dataloader:
    outputs = model(input_ids=batch["input_ids"], labels=batch["labels"])
    loss = outputs.loss
    loss.backward()
    optimizer.step()

# TTT behavior during training:
# - Processes full sequences in mini-batches (prefill mode)
# - TTT inner loop updates W1, b1 during forward pass
# - Outer loop (standard backprop) updates Q/K/V projections, learnable_lr, etc.
# - No need for decode mode during training
```

✅ Our current implementation WORKS for training!

### 5.2 Inference (What We're Missing)
```python
# Inference with generation:
cache = TTTCache(...)
for step in range(max_new_tokens):
    # Generate ONE token
    outputs = model.generate(input_ids=current_ids, past_key_values=cache, max_new_tokens=1)
    new_token = outputs[:, -1]
    current_ids = torch.cat([current_ids, new_token], dim=1)

    # TTT behavior during decode:
    # - Process ONE token
    # - Accumulate gradients in W1_grad, b1_grad
    # - Every K tokens, apply accumulated gradients to W1, b1
    # - Use conv cache for causal conv1d
    # - Track position within mini-batch
```

❌ Our current implementation DOES NOT WORK for generation!

---

## Part 6: What Needs to Be Fixed

### Priority 1: BLOCKERS (Cannot work without these)

#### Fix 1: Add Gradient Buffers to Cache
```python
# llama-omni/omni_speech/model/ttt/cache.py

@dataclass
class TTTCache:
    # Existing
    params_dict: Dict = field(default_factory=lambda: defaultdict(dict))

    def reset(self, ...):
        for layer_idx in range(num_layers):
            # States
            self.params_dict["W1_init"][layer_idx] = torch.zeros(B*nh, f, f, dtype=torch.float32)
            self.params_dict["b1_init"][layer_idx] = torch.zeros(B*nh, 1, f, dtype=torch.float32)

            # ADD: Gradient accumulators
            self.params_dict["W1_grad"][layer_idx] = torch.zeros(B*nh, f, f, dtype=torch.float32)
            self.params_dict["b1_grad"][layer_idx] = torch.zeros(B*nh, 1, f, dtype=torch.float32)

            # ADD: Conv cache (if using conv)
            self.params_dict["conv_cache"][layer_idx] = torch.zeros(B, kernel_size-1, F)
```

#### Fix 2: Implement Decode Functions
```python
# llama-omni/omni_speech/model/ttt/ops.py

def ttt_linear_decode_token(states, inputs, ttt_norm_weight, ttt_norm_bias):
    """Process one token, accumulate gradients, DON'T update W1/b1"""
    W1 = states['W1_init']
    b1 = states['b1_init']
    W1_grad = states['W1_grad']
    b1_grad = states['b1_grad']

    XV, XK, XQ, token_idx, ttt_lr = inputs[...]

    # 1. Forward
    Z1 = XK @ W1 + b1

    # 2. Loss gradient
    dl_dZ1 = ln_fused_l2_bwd(Z1, XV - XK, ttt_norm_weight, ttt_norm_bias)

    # 3. Accumulate gradients (DON'T apply yet)
    W1_grad.add_(XK.transpose(-1, -2) @ (ttt_lr * dl_dZ1))
    b1_grad.add_(ttt_lr * dl_dZ1)

    # 4. Compute output with TEMPORARY updated weights
    W1_bar = W1 - (token_idx * W1_grad)
    b1_bar = b1 - (token_idx * b1_grad)
    Z1_bar = XQ @ W1_bar + b1_bar

    return Z1_bar

def ttt_linear_decode_last_token_in_mini_batch(states, inputs, ttt_norm_weight, ttt_norm_bias):
    """Process last token in mini-batch, apply accumulated gradients"""
    W1 = states['W1_init']
    b1 = states['b1_init']
    W1_grad = states['W1_grad']
    b1_grad = states['b1_grad']

    XV, XK, XQ, token_idx, ttt_lr = inputs[...]

    # 1-2. Forward + loss gradient (same as above)
    Z1 = XK @ W1 + b1
    dl_dZ1 = ln_fused_l2_bwd(Z1, XV - XK, ttt_norm_weight, ttt_norm_bias)

    # 3. Final accumulation
    W1_grad.add_(XK.transpose(-1, -2) @ (ttt_lr * dl_dZ1))
    b1_grad.add_(ttt_lr * dl_dZ1)

    # 4. ACTUALLY update W1, b1 (not temporary)
    W1.sub_(token_idx * W1_grad)
    b1.sub_(token_idx * b1_grad)

    # 5. Reset gradients for next mini-batch
    W1_grad.zero_()
    b1_grad.zero_()

    # 6. Compute output
    Z1_bar = XQ @ W1 + b1

    return Z1_bar
```

#### Fix 3: Add Token Index Tracking
```python
# llama-omni/omni_speech/model/ttt/ttt_layer.py

class TTTLinearLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()

        # ADD: Token index buffer
        token_idx = 1. / torch.arange(1, self.mini_batch_size + 1).reshape(1, 1, -1, 1)
        self.register_buffer('token_idx', token_idx, persistent=False)
        self.learnable_token_idx_bias = nn.Parameter(
            torch.zeros((1, 1, self.mini_batch_size, 1))
        )

    def forward(self, hidden_states, cache_params, is_prefill, is_last_in_mini_batch, **kwargs):
        if is_prefill:
            # Use full token_idx: [1, 1, K, 1]
            token_idx = self.token_idx + self.learnable_token_idx_bias
            # ... prefill logic
        else:
            # Use token_idx for current position
            inner_mini_batch_step_offset = cache_params.seqlen_offset % self.mini_batch_size
            token_idx = self.token_idx[:, :, inner_mini_batch_step_offset, :] + \
                        self.learnable_token_idx_bias[:, :, inner_mini_batch_step_offset, :]

            # Route to decode function
            if is_last_in_mini_batch:
                output = ttt_linear_decode_last_token_in_mini_batch(...)
            else:
                output = ttt_linear_decode_token(...)
```

#### Fix 4: Update Forward Signature
```python
# llama-omni/omni_speech/model/ttt/ttt_layer.py

def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,  # This becomes TTTCache
    output_attentions: bool = False,
    use_cache: bool = False,
    is_prefill: Optional[bool] = None,  # ADD
    is_last_in_mini_batch: Optional[bool] = None,  # ADD
    **kwargs
):
    # Route based on mode
    if is_prefill:
        output = self._forward_prefill(...)
    else:
        output = self._forward_decode(hidden_states, past_key_value, is_last_in_mini_batch, ...)

    return output, None, updated_cache
```

#### Fix 5: Fix State Persistence
```python
# llama-omni/omni_speech/model/ttt/ops.py:ttt_linear

def ttt_linear(...):
    final_params, XQW_batch = scan(compute_ttt_linear_mini_batch, ...)

    # Extract final state
    W1_final = final_params["W1_states"]
    b1_final = final_params["b1_states"]

    # Return state (not just output)
    return XQW_batch, W1_final, b1_final  # ADD final states to return
```

```python
# llama-omni/omni_speech/model/ttt/ttt_layer.py

def forward(self, ...):
    # ...
    output, W1_final, b1_final = ttt_linear(...)  # Unpack all returns

    # Reshape output
    output = output.reshape(B, L, D)
    output = self.o_proj(output)
    output = self.post_norm(output)

    # Store final state in cache
    if use_cache:
        cache_params.update(self.layer_idx, W1_final, b1_final)

    return output, None, cache_params
```

### Priority 2: MAJOR (Quality will be poor without these)

#### Fix 6: Add Causal Conv1d
```python
# llama-omni/omni_speech/model/ttt/ttt_layer.py

class TTTLinearLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()

        # ADD: Causal convolutions
        self.conv_kernel = 4  # Standard
        self.conv_q = nn.Conv1d(
            self.hidden_size, self.hidden_size,
            bias=True,
            kernel_size=self.conv_kernel,
            groups=self.hidden_size,  # Depthwise
            padding=self.conv_kernel - 1,
        )
        self.conv_k = nn.Conv1d(...)

    def conv_qk_fused(self, XQK, cache_params, is_prefill):
        from causal_conv1d import causal_conv1d_fn, causal_conv1d_update

        if is_prefill:
            XQK = XQK.transpose(-1, -2)  # [B, N, D] -> [B, D, N]
            XQ = causal_conv1d_fn(XQK, self.conv_q.weight, self.conv_q.bias)
            XK = causal_conv1d_fn(XQK, self.conv_k.weight, self.conv_k.bias)
            XQ = XQ.transpose(-1, -2)  # [B, D, N] -> [B, N, D]
            XK = XK.transpose(-1, -2)

            # Cache last few tokens
            if cache_params is not None:
                cache_params["conv_cache"][self.layer_idx].copy_(
                    F.pad(XQK, (self.conv_kernel - XQK.size(-1), 0))
                )
        else:
            # Decode: single token
            XQK = XQK[:, 0, :]  # [B, 1, D] -> [B, D]
            XQ = causal_conv1d_update(
                XQK, cache_params["conv_cache"][self.layer_idx],
                self.conv_q.weight, self.conv_q.bias
            )
            XK = causal_conv1d_update(...)
            XQ = XQ.unsqueeze(1)  # [B, D] -> [B, 1, D]
            XK = XK.unsqueeze(1)

        return XQ, XK
```

Note: Requires `pip install causal-conv1d`

#### Fix 7: Add Gate Mechanism
```python
# llama-omni/omni_speech/model/ttt/ttt_layer.py

class TTTLinearLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()

        # MODIFY: Joint projection for Q, K, V, Gate, LR
        self.qkv_gate_lr_proj = nn.Linear(
            self.hidden_size,
            3 * self.hidden_size + self.num_heads,  # Q + K + V + Gate + LR
            bias=False
        )

    def forward(self, hidden_states, ...):
        # Project all at once
        qkv_gate_lr = self.qkv_gate_lr_proj(hidden_states)

        # Split
        qkv_gate, ttt_lr = torch.split(qkv_gate_lr, [3*D, num_heads], dim=-1)
        XQ, XK, XGate, XV = torch.split(qkv_gate, [D, D, D, D], dim=-1)

        # ... TTT processing ...

        # Output with gate
        output = F.gelu(XGate) * self.post_norm(ttt_output)
        output = self.o_proj(output)

        return output, None, cache
```

### Priority 3: OPTIONAL (Nice to have)

#### Fix 8: Add Triton Kernels
The original has Triton kernels for fast decode. We can start without them and add later if needed.

#### Fix 9: Add Auto-Padding
Currently we raise an error if sequence length is not divisible by mini_batch_size. We should add automatic padding.

---

## Part 7: Implementation Strategy

### Phase 1: Fix Blockers (MUST DO BEFORE TESTING)
**Goal**: Make generation work at all

1. **Add gradient buffers to cache** (2 hours)
   - Modify TTTCache class
   - Add W1_grad, b1_grad initialization
   - Add reset logic

2. **Implement decode functions** (4 hours)
   - `ttt_linear_decode_token`
   - `ttt_linear_decode_last_token_in_mini_batch`
   - Test with dummy inputs

3. **Add token index tracking** (2 hours)
   - Add token_idx buffer
   - Add learnable_token_idx_bias parameter
   - Update forward to compute current index

4. **Update forward signature** (2 hours)
   - Add is_prefill, is_last_in_mini_batch parameters
   - Add routing logic
   - Ensure cache is passed through correctly

5. **Fix state persistence** (2 hours)
   - Modify ttt_linear to return final states
   - Update forward to extract and cache states
   - Test that state persists across batches

**Timeline**: ~12 hours of focused work

### Phase 2: Add Major Features (RECOMMENDED)
**Goal**: Match original quality

6. **Add causal conv1d** (4 hours)
   - Install causal-conv1d package
   - Add conv layers
   - Implement conv_qk_fused
   - Add conv cache management

7. **Add gate mechanism** (2 hours)
   - Modify projections to include gate
   - Update forward to apply gating
   - Verify no regression

**Timeline**: +6 hours

### Phase 3: Polish (OPTIONAL)
**Goal**: Production ready

8. **Add Triton kernels** (8 hours if needed)
9. **Add auto-padding** (2 hours)
10. **Comprehensive testing** (4 hours)

**Timeline**: +14 hours

---

## Part 8: Testing Plan

### Test 1: Unit Tests
```python
# Test decode token
def test_decode_token():
    # Setup
    cache = TTTCache(...)
    layer = TTTLinearLayer(config)

    # Process K tokens one by one
    for i in range(K):
        is_last = (i == K - 1)
        output = layer(
            hidden_states=input[i:i+1],  # Single token
            past_key_value=cache,
            is_prefill=False,
            is_last_in_mini_batch=is_last
        )

    # Verify state was updated at last token
    assert cache.params_dict["W1_grad"][0].abs().max() < 1e-6  # Should be zeroed
    assert not torch.allclose(cache.params_dict["W1_init"][0], initial_W1)  # Should be updated
```

### Test 2: Integration Test
```python
# Test full generation
def test_generation():
    model = OmniSpeechLlamaForCausalLM(config)
    cache = TTTCache.from_config(config, max_batch_size=1)

    prompt = torch.randint(0, vocab_size, (1, 64))

    # Generate 256 tokens (4 mini-batches of 64)
    output = model.generate(
        input_ids=prompt,
        max_new_tokens=256,
        past_key_values=cache,
        use_cache=True
    )

    assert output.shape[1] == 64 + 256
    # Verify no gibberish (check perplexity)
```

### Test 3: Long Context Test
```python
# Test 1-hour conversation (45k tokens @ 12.5Hz)
def test_long_context():
    model = OmniSpeechLlamaForCausalLM(config)
    cache = TTTCache.from_config(config)

    # Generate in chunks
    for chunk_idx in range(100):
        output = model.generate(
            input_ids=prev_output,
            max_new_tokens=450,  # ~3.6 seconds
            past_key_values=cache,
            use_cache=True
        )

        # Check no gibberish
        perplexity = compute_perplexity(output)
        assert perplexity < threshold, f"Gibberish detected at chunk {chunk_idx}"
```

---

## Part 9: Estimated Time to Fix

| Phase | Tasks | Hours | Risk |
|-------|-------|-------|------|
| Phase 1 (Blockers) | Cache + decode + token_idx + signature + persistence | 12 | Medium |
| Phase 2 (Quality) | Conv1d + gate | 6 | Low |
| Phase 3 (Polish) | Triton + padding + tests | 14 | Low |
| **TOTAL** | | **32 hours** | |

**Realistic estimate with debugging**: 40-50 hours over 5-7 days

---

## Part 10: Recommendation

### What to Do Next

1. **DO NOT TEST CURRENT IMPLEMENTATION** - It will not work for generation. Don't waste time.

2. **Fix Phase 1 first** - The 5 blocker issues must be fixed before any testing.

3. **Test incrementally** - After each fix, run unit tests to verify it works.

4. **Add Phase 2 features** - Conv1d and gate are important for quality. Don't skip.

5. **Phase 3 is optional** - Triton kernels can wait until we confirm the approach works.

### Alternative: Use TTT-LM-Kernels Directly

**Pros**:
- Already complete
- Battle-tested
- Has Triton kernels

**Cons**:
- Designed for text LM, not multimodal
- Different architecture (no Llama compatibility layer)
- Would need significant adaptation for Llama-Omni

**Recommendation**: Fix our implementation. It's closer to working than starting over.

---

## Part 11: Key Insights from Analysis

1. **TTT is NOT just a drop-in attention replacement** - It needs significant infrastructure:
   - Gradient accumulation buffers
   - Token position tracking
   - Prefill/decode mode switching
   - Special cache structure

2. **Training ≠ Inference** - Our implementation works for training but not inference. These are fundamentally different code paths.

3. **The original has TWO implementations**:
   - ttt-video-dit: For training (what we copied from)
   - ttt-lm-kernels: For inference (what we need)

4. **Moshi's gibberish was likely due to**:
   - KV cache wraparound at 3000 tokens
   - Not TTT itself
   - Llama-Omni with proper TTT should work better

5. **The user's critical analysis was spot-on** - State persistence is broken, and that's just the tip of the iceberg.

---

## Appendix: Code References

### A. Original Implementations
- **ttt-video-dit**: `/home/user/claude-web/ttt-video-dit/ttt/models/ssm/`
  - `ops/ttt_linear.py`: Prefill implementation
  - `ttt_layer.py`: TTT layer wrapper

- **ttt-lm-kernels**: `/home/user/claude-web/ttt-lm-kernels/ttt/`
  - `modeling_ttt.py`: Full prefill + decode implementation
  - `generation.py`: TTTCache class
  - `triton_kernel/`: Fast kernels

### B. Our Implementation
- `/home/user/claude-web/llama-omni/omni_speech/model/ttt/`
  - `ttt_layer.py`: Layer (only prefill)
  - `ops.py`: TTT algorithm (only prefill)
  - `cache.py`: Cache (incomplete)
  - `utils.py`: Helper functions
  - `logger.py`: Logging
  - `integration.py`: Model integration

### C. Llama-Omni
- `/home/user/claude-web/llama-omni/omni_speech/model/language_model/`
  - `omni_speech_llama.py`: Main model class

---

## Summary

**Current Status**: 40% complete (training works, inference doesn't)

**Critical Issues**: 7 blockers, 2 major quality issues

**Time to Fix**: 40-50 hours

**Next Step**: Fix Phase 1 blockers before any testing

**Expected Outcome**: After fixes, should work for unlimited context speech generation
