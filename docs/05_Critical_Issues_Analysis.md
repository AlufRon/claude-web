# Critical Issues Analysis: Why TTT Integration Has Failed So Far

Based on implementation experience and deep code investigation.

---

## Executive Summary

**The Problem**: Adding TTT to Moshi for long context has been attempted but **consistently produces gibberish after 5-7 minutes**, despite:
- Loss decreasing during training ✓
- Extended RoPE ✗
- Larger attention context ✗
- TTT layers added ✗
- Local attention + global TTT attempted ✗

**Root Causes Identified**:
1. **State reset bug in training**: TTT state resets per batch instead of per conversation
2. **Streaming mini-batch mismatch**: Video-DiT processes 64-token batches; Moshi generates 1 frame at a time
3. **FP32 precision not enforced**: Inner states need FP32 but may be cast to BF16
4. **Local+global architecture misunderstood**: Non-causal diffusion pattern doesn't directly apply to causal autoregressive

---

## Issue 1: FP32 Precision for Inner States ✓ CONFIRMED

### What the Code Shows

**Video-DiT explicitly uses FP32** (`ttt-video-dit/ttt/models/ssm/linear_triton.py:82-88`):
```python
# EXPLICIT FP32 for inner states
W1_last = torch.empty(B, NH, F, F, device=device, dtype=torch.float32)
b1_last = torch.empty(B, NH, 1, F, device=device, dtype=torch.float32)

# Cast to FP32 before processing
W1_init.to(torch.float32).contiguous()
b1_init.to(torch.float32).contiguous()

# Convert outputs back to mixed precision
return XQW_batch.to(mp_dtype)  # BF16/FP16
```

### What You Need

```python
class TTTLinear(nn.Module):
    def __init__(self, d_model, num_heads, dtype=torch.bfloat16):
        # Model uses BF16
        self.q_proj = nn.Linear(d_model, d_model, dtype=dtype)

        # BUT inner states MUST be FP32
        self.W1 = nn.Parameter(
            torch.normal(0, 0.02, size=(num_heads, head_dim, head_dim),
                        dtype=torch.float32)  # ← FP32!
        )
        self.b1 = nn.Parameter(
            torch.zeros(num_heads, 1, head_dim,
                       dtype=torch.float32)  # ← FP32!
        )
```

**Validation**:
```python
# In forward pass, verify dtypes
assert self.W1.dtype == torch.float32, f"W1 must be FP32, got {self.W1.dtype}"
assert activations.dtype == torch.bfloat16, f"Activations should be BF16"
```

---

## Issue 2: No-Grad Update Mechanism ⚠️ MISUNDERSTOOD

### Clarification from Code

**TTT does NOT bypass torch.no_grad()**. Instead, it uses **analytical gradients within the computation graph**.

**Evidence** (`ttt-video-dit/ttt/models/ssm/ops/utils.py:21-48`):
```python
def ln_fused_l2_bwd(x, l2_target, gamma, beta, eps=1e-8):
    """Analytical backward through LayerNorm + L2 loss."""
    # Forward
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    std = torch.sqrt(var + eps)
    x_hat = (x - mu) / std
    y = gamma * x_hat + beta

    # MANUAL gradient (but still differentiable!)
    grad_output = y - l2_target

    # Analytical backward through LayerNorm
    grad_x_hat = grad_output * gamma
    grad_x = (1.0 / D) * (D * grad_x_hat - ...) / std

    return grad_x  # Still part of computation graph!
```

**The update** (`ttt-video-dit/ttt/models/ssm/ops/ttt_linear.py:28-41`):
```python
# This IS part of autograd graph
W1_updated = W1_init - (eta * X.T) @ grad_l_wrt_Z1

# The entire TTT forward pass is differentiable
# Outer loop backprop flows through this!
```

**Why this matters**:
- TTT updates work in `model.train()` mode (gradients flow)
- TTT updates work in `model.eval()` + `torch.no_grad()` mode (uses analytical grad)
- The "inner loop" gradient is computed analytically, not via .backward()

---

## Issue 3: State Reset Logic 🔴 CRITICAL BUG

### Current Behavior (WRONG for Speech)

**Training** (`ttt-video-dit/ttt/models/ssm/ttt_layer.py:360-398`):
```python
def forward(self, inputs):
    # PROBLEM: State RESETS every forward() call!
    W1_states = torch.tile(self.W1.unsqueeze(0), dims=(B, 1, 1, 1))
    b1_states = torch.tile(self.b1.unsqueeze(0), dims=(B, 1, 1, 1))

    # Process sequence with fresh state
    output = ttt_linear(XK, XQ, XV, eta, W1_states, b1_states, ...)

    # Final state is DISCARDED
    return output
```

**Problem**:
- Each batch gets fresh W1, b1
- Model never learns to use persistent state
- Long conversations are chopped into independent chunks

### What You Need for Speech Training

**Conversation-level state persistence**:

```python
class ConversationAwareTTTLayer(nn.Module):
    def __init__(self):
        self.W1_param = nn.Parameter(...)  # Learned init
        self.b1_param = nn.Parameter(...)
        self.conversation_state = None  # Persistent across batches

    def reset_conversation(self):
        """Call when starting NEW conversation."""
        self.conversation_state = None

    def forward(self, x, is_new_conversation=False):
        if is_new_conversation or self.conversation_state is None:
            # Fresh state for new conversation
            W1 = self.W1_param.clone()
            b1 = self.b1_param.clone()
        else:
            # Continue from previous batch's state
            W1, b1 = self.conversation_state

        # Process current batch
        output, W1_updated, b1_updated = ttt_process(x, W1, b1)

        # CRITICAL: Save state for next batch in SAME conversation
        self.conversation_state = (W1_updated.detach(), b1_updated.detach())

        return output
```

**Dataset modification**:
```python
# In your dataloader
for conversation_id, batches in dataset:
    # Signal new conversation
    model.ttt_layer.reset_conversation()

    for batch in batches:
        # All batches in same conversation share state
        output = model(batch, is_new_conversation=False)
```

**This is why loss went down but quality didn't improve**:
- Model learned to predict next token (local patterns)
- But never learned to use long-range TTT memory (state kept resetting!)

---

## Issue 4: Local Attention + Global TTT for Autoregressive 🔴 CRITICAL

### Video-DiT Pattern (Non-Causal)

**What they do** (`ttt-video-dit/ttt/models/cogvideo/dit.py`):
```python
# Process entire video at once (non-causal diffusion)
def forward(self, video_tokens):  # [B, T=1000, D]
    # 1. Local attention: overlapping 12-frame windows
    for i in range(num_windows):
        start = i * 12
        end = start + 13  # 1 frame overlap!
        window = tokens[:, start:end]
        attn_out[start:end] += attention(window, is_causal=False)
    attn_out /= overlap_counts  # Average overlaps

    # 2. Global TTT: bidirectional over entire sequence
    ttt_out_fwd = ttt(tokens)  # x1 → x2 → ... → xT
    ttt_out_bwd = reverse(ttt(reverse(tokens)))  # xT → ... → x1

    # 3. Combine
    return gate_α * attn_out + gate_β * ttt_out_fwd + gate_γ * ttt_out_bwd
```

**Key properties**:
- Non-causal attention (sees future)
- Overlapping windows (computed multiple times)
- Bidirectional TTT (forward + backward passes)
- Processes entire sequence at once

### Autoregressive Adaptation (What You Need)

```python
class AutoregressiveTTTLayer(nn.Module):
    def __init__(self, window_size=256):
        self.attn_window = window_size
        self.ttt_state = None  # Persistent state

    def forward(self, x, position):
        # 1. Causal sliding window attention
        window_start = max(0, position - self.attn_window)
        window = self.kv_cache[window_start:position+1]  # Only past!

        local_out = self.attention(
            q=x,
            k=window,
            v=window,
            is_causal=True  # Can't see future
        )

        # 2. Forward-only global TTT
        if self.ttt_state is None:
            self.ttt_state = self.init_ttt_state()

        # Update state with current token
        global_out, self.ttt_state = self.ttt_step(
            x,
            prev_state=self.ttt_state
        )

        # 3. Gated combination (no backward pass!)
        return local_out + self.gate_alpha * global_out
```

**Key differences**:
- ✅ Causal attention (no future access)
- ✅ Non-overlapping windows (single pass per token)
- ✅ Forward-only TTT (no backward pass)
- ✅ Token-by-token streaming (not batch processing)

### Why Your Implementation Didn't Work

**Likely issues**:
1. Tried to use overlapping windows in causal model (breaks causality)
2. Tried bidirectional TTT in autoregressive model (sees future)
3. State still resetting despite "global" TTT

---

## Issue 5: The Gibberish Problem 🔴 ROOT CAUSE ANALYSIS

### What You Experienced

```
Timeline:
0-5 min:  ✓ Coherent speech
5-7 min:  ⚠️ Quality degrades
7+ min:   ❌ Complete gibberish

Attempted fixes:
❌ Extended RoPE → no improvement
❌ Larger KV cache → no improvement
❌ Added TTT layers → no improvement
❌ Fine-tuned on long data → loss decreased, quality didn't improve
```

### Hypothesis 1: TTT State Resetting (MOST LIKELY)

**Evidence**:
- Video-DiT code resets state every forward()
- Your training probably does the same
- Loss decreases (local prediction works) but long-range coherence fails

**Test**:
```python
# Add logging to TTT layer
def forward(self, x):
    if self.conversation_state is None:
        print(f"⚠️  WARNING: State reset at step {self.step_counter}")

    # ... rest of forward
```

**If you see resets mid-conversation → This is the bug!**

### Hypothesis 2: FP32 Precision Issue

**Evidence**:
- TTT inner states need FP32 for numerical stability
- Moshi uses BF16 by default
- After ~300 steps (5min @ 12.5Hz = 3750 frames), accumulated FP16 errors explode

**Test**:
```python
# Check dtypes during generation
@torch.no_grad()
def generate_debug(model, steps=5000):
    for step in range(steps):
        output = model.step(...)

        if step % 500 == 0:
            # Check TTT state dtypes
            for layer in model.transformer.layers:
                if hasattr(layer, 'W1'):
                    print(f"Step {step}: W1 dtype = {layer.W1.dtype}")
                    print(f"W1 range: [{layer.W1.min():.4f}, {layer.W1.max():.4f}]")

                    # Check for numerical explosion
                    if torch.isnan(layer.W1).any():
                        print(f"❌ NaN detected in W1 at step {step}!")
                    if layer.W1.abs().max() > 1e4:
                        print(f"⚠️  W1 values exploding at step {step}!")
```

### Hypothesis 3: Position Embedding Breakdown

**Evidence**:
- RoPE is trained on sequences up to training context (e.g., 3000 tokens)
- At 5-7 minutes: 3750-5250 frames @ 12.5 Hz
- Beyond training length, RoPE extrapolation fails

**But**: You said extending RoPE didn't help, so this is less likely.

### Hypothesis 4: Streaming Mini-Batch Mismatch

**Evidence**:
- TTT needs mini-batches (e.g., 64 tokens) for stable updates
- Moshi streams 1 frame at a time
- Without proper buffering, TTT gradients are too noisy

**Current TTT-video code expects**:
```python
# Reshape into mini-batches
X = X.view(B, num_mini_batch, 64, D)  # Expects 64-token chunks
```

**Moshi streaming does**:
```python
# One frame at a time
for frame in stream:
    output = model.step(frame)  # Single token!
```

**Mismatch**: TTT can't form mini-batches in streaming!

---

## Issue 6: Streaming TTT Design 🔴 NOT IN EXISTING CODE

### The Challenge

**TTT implementations assume**:
- Offline processing (entire sequence available)
- Mini-batches of 16-64 tokens
- Batch gradient descent updates

**Moshi requires**:
- Frame-by-frame generation (1 frame = 80ms)
- Immediate output (can't buffer 64 frames = 5 seconds!)
- Low latency (200ms total)

### Solution: Hybrid Buffered TTT

**Approach**:
```python
class StreamingTTT(nn.Module):
    def __init__(self, mini_batch_size=16):
        self.buffer = []  # Token buffer
        self.mini_batch_size = mini_batch_size
        self.W_state = None  # Current TTT state
        self.W_grad_accum = None  # Gradient accumulator

    def streaming_forward(self, x_t, force_update=False):
        """Process single token in streaming mode."""

        # Add to buffer
        self.buffer.append(x_t)

        # Check if buffer full or forced update
        if len(self.buffer) >= self.mini_batch_size or force_update:
            # Process buffered tokens as mini-batch
            X_batch = torch.stack(self.buffer)  # [mini_batch_size, D]

            # TTT update with mini-batch
            output_batch, self.W_state = self.ttt_mini_batch(
                X_batch, self.W_state
            )

            # Clear buffer
            self.buffer = []

            return output_batch[-1]  # Return last token output
        else:
            # Buffer not full: use current W without updating
            output_t = self.ttt_no_update(x_t, self.W_state)
            return output_t
```

**Trade-off**:
- Latency: Added buffering (16 frames @ 12.5Hz = 1.28 seconds)
- vs Stability: Proper mini-batch updates

**Alternative: Per-Token Updates**:
```python
def streaming_forward_no_buffer(self, x_t):
    """Update TTT state after every token (less stable)."""

    # Compute gradient with single token
    grad = self.compute_gradient(x_t, self.W_state)

    # Accumulate gradient
    if self.W_grad_accum is None:
        self.W_grad_accum = grad
    else:
        self.W_grad_accum += grad

    # Apply accumulated gradient every N tokens
    self.token_counter += 1
    if self.token_counter % self.update_frequency == 0:
        self.W_state -= self.lr * self.W_grad_accum
        self.W_grad_accum.zero_()

    # Forward with current W
    output = self.W_state @ x_t
    return output
```

---

## Issue 7: Minimal Code Changes ✓ STRATEGY

### What to Copy from Existing Code

**From ttt-lm-kernels** (most production-ready):
```
ttt-lm-kernels/ttt/
├── generation.py                    ← TTTCache class
├── modeling_ttt.py                  ← State management
└── triton_kernel/
    ├── ttt_linear_decode.py         ← Streaming updates
    └── ttt_linear_prefill.py        ← Batch updates
```

**From ttt-video-dit** (PyTorch, easier to modify):
```
ttt-video-dit/ttt/models/ssm/
├── ttt_layer.py                     ← TTTWrapper class
├── ops/
│   ├── ttt_linear.py                ← Core TTT logic
│   └── utils.py                     ← ln_fused_l2_bwd
└── linear_triton.py                 ← FP32 handling
```

### Integration Pattern

```python
# 1. Copy entire file and modify imports
cp ttt-video-dit/ttt/models/ssm/ttt_layer.py → moshi/moshi/modules/ttt_layer.py

# 2. Minimal changes to ttt_layer.py:
# - Change: from ttt.models.configs → from .configs
# - Add: Conversation state management (50 lines)
# - Add: Streaming buffer (30 lines)
# - Keep: Everything else unchanged

# 3. Modify transformer.py (20 lines):
from .ttt_layer import TTTWrapper

if use_ttt:
    self.ttt = TTTWrapper(config)
else:
    self.ttt = None
```

**Total new code**: ~100 lines (state management + streaming buffer)
**Total copied code**: ~400 lines (ttt_layer.py)
**Total modified code**: ~20 lines (transformer.py)

---

## Recommended Action Plan

### Phase 0: Debug Current Implementation (1 week)

Before adding new code, understand WHY it failed:

**Test 1: State Reset Detection**
```python
# Add to your current TTT layer
class DebugTTT(nn.Module):
    def __init__(self):
        self.reset_counter = 0
        self.step_counter = 0

    def forward(self, x, is_new_conversation=False):
        if is_new_conversation:
            self.reset_counter += 1
            print(f"🔄 State reset #{self.reset_counter} at step {self.step_counter}")

        self.step_counter += 1
        # ... rest of forward
```

**Test 2: Dtype Verification**
```python
# Check if FP32 is maintained
assert self.W1.dtype == torch.float32, "W1 must be FP32!"
```

**Test 3: Generation Quality Over Time**
```python
# Generate 10 minutes, save every 30 seconds
outputs = []
for i in range(10):  # 10 minutes
    output_30s = model.generate(duration=30)
    outputs.append(output_30s)

    # Evaluate quality
    wer = compute_wer(output_30s)
    mcd = compute_mcd(output_30s)
    print(f"Minute {i+1}: WER={wer:.2f}, MCD={mcd:.2f}")

    # Check for degradation
    if wer > 0.5:  # Gibberish threshold
        print(f"❌ Gibberish detected at minute {i+1}!")
        break
```

### Phase 1: Fix State Management (1 week)

**Goal**: Make TTT state persist across batches in same conversation.

**Changes**:
1. Modify dataloader to group batches by conversation_id
2. Add `reset_conversation()` method to TTT layer
3. Verify state persists within conversation, resets between

**Test**:
```python
# Train on 10-minute conversations
model.train()
for conversation in dataset:
    model.reset_conversation()

    for batch in conversation.batches:
        loss = model(batch)
        loss.backward()
        optimizer.step()

    # State should have evolved over 10 minutes!
    print(f"Final state norm: {model.W1.norm():.4f}")
```

### Phase 2: Implement Streaming TTT (2 weeks)

**Goal**: Make TTT work with Moshi's frame-by-frame generation.

**Approach**: Use buffered updates (mini-batch=16)

**Test**:
```python
# Stream 10 minutes frame-by-frame
model.eval()
with model.streaming(batch_size=1):
    for frame_idx in range(7500):  # 10 min @ 12.5Hz
        output = model.step(input_frame, frame_idx)

        # Every 1 minute, check quality
        if frame_idx % 750 == 0:
            quality = evaluate(output)
            print(f"Minute {frame_idx//750}: Quality={quality:.3f}")
```

### Phase 3: Validate on Long Audio (1 week)

**Goal**: Confirm gibberish problem is solved.

**Test cases**:
1. 5-minute generation (should be perfect)
2. 10-minute generation (critical test)
3. 30-minute generation (stress test)
4. 1-hour generation (ultimate goal)

---

## Checklist for Next Attempt

Before integrating TTT again, ensure:

### Code Requirements
- [ ] W1, b1, W2, b2 explicitly `dtype=torch.float32`
- [ ] State persists across batches in training
- [ ] State persists for entire conversation in inference
- [ ] Streaming buffer implemented (mini-batch=16 or 64)
- [ ] State reset only on `is_new_conversation=True`

### Testing Requirements
- [ ] State reset logging added
- [ ] Dtype verification in forward pass
- [ ] Quality metrics logged every N steps
- [ ] Long generation test (10+ minutes)
- [ ] Training loss AND generation quality both tracked

### Architecture Requirements
- [ ] Causal sliding window attention (not overlapping)
- [ ] Forward-only TTT (not bidirectional)
- [ ] Proper gating initialization (α≈0.1)
- [ ] Mini-batch processing for TTT updates

---

## Key Takeaways

1. **FP32 is mandatory**: Your instinct was correct - code confirms this
2. **State management is the bug**: Video-DiT resets state per sequence; you need per-conversation
3. **Streaming requires buffering**: Can't do mini-batch TTT with 1 token at a time
4. **Local+global pattern is different for causal**: No overlapping windows, no bidirectional
5. **Debug first, integrate second**: Understand WHY previous attempts failed before trying again

The gibberish problem is likely **state resetting mid-conversation** + **FP16 precision errors**. Fix these first before adding more complexity.
