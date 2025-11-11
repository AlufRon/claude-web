# Critical Issues in Moshi-TTT Implementation

**Date**: 2025-11-10
**Status**: Code-Verified Analysis
**Branch**: `claude/deep-code-review-ttt-011CUzpCD2kNLGH5UnuHN7hF`

---

## Executive Summary

Through deep code analysis and end-to-end training flow verification, **5 critical issues** have been identified and verified in the Moshi-TTT implementation. These range from architectural limitations to implementation bugs that prevent TTT from functioning correctly.

| # | Issue | Severity | Type | Status |
|---|-------|----------|------|--------|
| **1** | Ring KV Cache Information Loss | 🔴 **Critical** | Architectural | ✅ Verified |
| **2** | Reconstruction Target Normalization Bug | 🟠 **Major** | Implementation | ✅ Verified |
| **3** | Batch Size > 1 Incompatibility | 🟡 **Medium** | Implementation | ✅ Verified |
| **4** | Gradient Flow Corruption | 🔴 **Critical** | Implementation | ✅ Verified |
| **5** | Cross-File State Contamination | 🔴 **Critical** | Implementation | ✅ Verified |

---

## Issue #1: Ring KV Cache Information Loss 🔴

### The Problem

**Moshi's attention layer uses a ring buffer with fixed capacity that permanently discards old tokens, preventing TTT from ever seeing the complete sequence history.**

### Code Evidence

**File**: `moshi/moshi/moshi/modules/transformer.py`

```python
# Line 187-233
class RingKVCache:
    def __init__(self, batch_size, num_heads, dim_per_head, capacity: int, ...):
        self.capacity = capacity  # Fixed capacity (default: 3000 tokens)
        self.cache = torch.zeros(
            (2, batch_size, num_heads, capacity, dim_per_head), ...
        )

    def complete(self, k, v, exec_mask):
        # Line 232-233: MODULO WRAPAROUND
        indexes = torch.arange(T, ...) + self.end_offset.view(-1, 1)
        indexes = indexes % self.capacity  # ← OLD TOKENS OVERWRITTEN!

        # Line 240-241: Scatter overwrites cache at wrapped indexes
        self.cache[0].scatter_(2, this_indexes, k)  # Keys overwritten
        self.cache[1].scatter_(2, this_indexes, v)  # Values overwritten
```

**File**: `moshi/moshi/moshi/modules/transformer.py:460-466`

```python
if self.context is None:
    if self.weights_per_step:
        capacity = self.weights_per_step
    else:
        raise RuntimeError("Cannot create streaming KVCache without context")
else:
    capacity = self.context  # ← Sets ring buffer size to context (default 3000)

kv_cache = RingKVCache(batch_size, self.num_heads, dim_per_head, capacity, ...)
```

### Data Flow Analysis

**At position 3750 (5 minutes @ 12.5 Hz):**

```
Timeline:
Tokens 0-3000:   Stored in ring buffer [positions 0-2999]
Tokens 3001-3750: New tokens arrive
                  → Positions 0-749 OVERWRITTEN (modulo wraparound)
                  → Ring buffer now contains [750-3750]
                  → Tokens 0-749 PERMANENTLY LOST

Attention Query at 3750:
  ├─ KV Cache: Only tokens [750-3750] available
  ├─ Attention output: Weighted combination of values[750-3750]
  └─ ❌ NO INFORMATION about tokens 0-749

TTT Input:
  ├─ Receives: Attention output (missing 0-749 info)
  ├─ Q/K/V Projections: All derived from incomplete attention output
  └─ ❌ CANNOT reconstruct lost information
```

### Why This Is Different from Video-DiT

**Video-DiT Architecture** (from `ttt-video-dit/ttt/models/cogvideo/dit.py:163-276`):

```python
def _attn_forward(self, vid_emb, text_emb, seq_metadata):
    for i in range(num_attn_steps):  # Process each 3-second segment
        start_idx = i * self.attn_length * tokens_per_frame
        end_idx = (self.prefix + (i + 1) * self.attn_length) * tokens_per_frame

        # Extract COMPLETE segment from input
        cur_emb = torch.cat([text_emb[:, ...], vid_emb[:, start_idx:end_idx]], dim=1)

        # Attention sees ALL tokens in segment
        attn_output = attention(cur_emb)
        output_vid_emb[:, start_idx:end_idx] += attn_output

    return output_vid_emb  # All segments complete

def _ssm_forward(self, emb, seq_metadata):
    # TTT receives FULL concatenated sequence
    emb = forward_ssm(emb, seq_metadata)  # ← Gets ALL segment outputs!
```

**Key Difference**:
- **Video-DiT**: Attention processes **complete 3-second segments**, TTT sees concatenation of ALL segment outputs
- **Moshi-TTT**: Attention uses **ring buffer**, old tokens are **discarded**, TTT never sees complete information

### Impact

**Critical architectural limitation**. Even with perfect TTT implementation:
- Sequences > 3000 tokens: TTT cannot maintain coherent long-term memory
- Information loss is **permanent and unrecoverable**
- 5-minute audio (3750 tokens @ 12.5 Hz): 20% of history lost

### Why TTT Memory Doesn't Save This

**User's question**: "TTT should see the full context. Why wouldn't it?"

**Answer**: TTT can only remember what it was shown. The data flow is:

```
Raw Input → Attention (ring buffer) → TTT
              ↓
         [Tokens 0-749 lost here]
              ↓
         TTT never receives this data
```

Even though TTT maintains weights W across time:
1. At position 3000: W₃₀₀₀ contains information about tokens 0-3000 ✓
2. Positions 3001-3750: W is updated based on attention outputs
3. But attention outputs at 3750 **lack information about 0-749**
4. Reconstruction target at 3750: `XV - XK` (both missing 0-749 info)
5. Gradient computation: Based on incomplete reconstruction target
6. Weight update: `W₃₇₅₀ = W₃₇₄₉ - η × gradient`
   - Gradient has no signal to preserve 0-749 information
   - After 750 updates, old information progressively **washes out**

**TTT's W contains residual information but no gradient signal to maintain it.**

---

## Issue #2: Reconstruction Target Normalization Bug 🟠

### The Problem

**Moshi normalizes XV directly instead of normalizing the difference (XV - XK), resulting in incorrect reconstruction targets.**

### Code Evidence

**Video-DiT (CORRECT)** - `ttt-video-dit/ttt/models/ssm/ttt_layer.py:220-235`:

```python
def ln_reconstruction_target(self, XV, XK):
    # Step 1: Compute difference FIRST
    XV = XV - XK

    # Step 2: Normalize the DIFFERENCE
    mean = XV.mean(dim=-1, keepdim=True)
    std = XV.std(dim=-1, keepdim=True)
    XV = (XV - mean) / (std + 1e-8)

    # Step 3: Apply learned scale/shift
    XV = self.ttt_norm_weight * XV + self.ttt_norm_bias

    # Step 4: Add XK back
    return XV + XK  # Returns: norm(XV - XK) + XK
```

Then in `compute_mini_batch` (line 32):
```python
reconstruction_target = XV_mini_batch - XK_mini_batch
# = [norm(XV - XK) + XK] - XK
# = norm(XV - XK)  ✓ CORRECT
```

**Moshi-TTT (INCORRECT)** - `moshi_ttt_try/moshi-finetune/moshi_ttt/ttt_layer.py:463-478`:

```python
def ln_reconstruction_target(self, XV, XK):
    """Layer norm reconstruction target following Video-DiT pattern"""
    B, L, num_heads, head_dim = XV.shape

    # Apply layer norm per head
    XV_normed = torch.zeros_like(XV)
    for h in range(num_heads):
        weight = self.ttt_norm_weight[h]
        bias = self.ttt_norm_bias[h]

        # Normalize XV DIRECTLY (not the difference!)
        XV_h = XV[:, :, h, :]
        XV_normed[:, :, h, :] = F.layer_norm(XV_h, (head_dim,), weight, bias, eps=1e-6)

    return XV_normed  # Returns: norm(XV)
```

Then in `compute_mini_batch` (line 141):
```python
reconstruction_target = XV_mini_batch - XK_mini_batch
# = norm(XV) - XK  ✗ WRONG!
```

### Mathematical Impact

**Correct** (Video-DiT):
```
Target = norm(XV - XK)
```
- Normalizes the residual connection
- Stable statistics (mean=0, std=1 for the difference)
- TTT learns to reconstruct the normalized difference

**Incorrect** (Moshi-TTT):
```
Target = norm(XV) - XK
```
- XK is subtracted AFTER normalization
- Unstable statistics (mean ≠ 0, std ≠ 1)
- TTT learns to reconstruct an incorrectly scaled target

### Impact

**Major implementation bug**:
- TTT reconstruction loss computed on wrong target
- Gradient updates train TTT to minimize the wrong objective
- May explain poor long-context performance
- Easy to fix but affects all trained checkpoints

---

## Issue #3: Batch Size > 1 Incompatibility 🟡

### The Problem

**Persistent state updates only save the first batch element, breaking multi-batch training.**

### Code Evidence

**File**: `moshi_ttt_try/moshi-finetune/moshi_ttt/ttt_layer.py:684-707`

```python
# Update model parameters with persistent states (JAX-style behavior)
# Extract final states and remove batch dimension for parameter update
with torch.no_grad():
    # final_states have shape [B, H, ...] - take first batch element [0] to get [H, ...]
    final_W1 = final_states["W1_states"][0]  # [B, H, HD, 4*HD] -> [H, HD, 4*HD]
    #                                    ↑
    #                                    Takes ONLY batch[0]!

    final_b1 = final_states["b1_states"][0]  # [B, H, 1, 4*HD] -> [H, 1, 4*HD]
    final_W2 = final_states["W2_states"][0]  # [B, H, 4*HD, HD] -> [H, 4*HD, HD]
    final_b2 = final_states["b2_states"][0]  # [B, H, 1, HD] -> [H, 1, HD]

    # Copy updated states to model parameters
    self.W1.data.copy_(final_W1)  # Overwrites with batch[0] state only
    self.b1.data.copy_(final_b1)
    self.W2.data.copy_(final_W2)
    self.b2.data.copy_(final_b2)
```

### Scenario Analysis

**Batch Size = 1 (Streaming Inference)**: ✓ Works correctly
- Only one batch element exists
- Taking `[0]` is correct

**Batch Size > 1 (Training)**: ✗ Fails
```
Initial: W₀ (shared across batch)
Forward pass with batch_size=4:
  ├─ Batch[0]: W₀ → W₁⁽⁰⁾ → W₂⁽⁰⁾ → ... → Wₙ⁽⁰⁾
  ├─ Batch[1]: W₀ → W₁⁽¹⁾ → W₂⁽¹⁾ → ... → Wₙ⁽¹⁾  ← DISCARDED!
  ├─ Batch[2]: W₀ → W₁⁽²⁾ → W₂⁽²⁾ → ... → Wₙ⁽²⁾  ← DISCARDED!
  └─ Batch[3]: W₀ → W₁⁽³⁾ → W₂⁽³⁾ → ... → Wₙ⁽³⁾  ← DISCARDED!

After forward: self.W1 = Wₙ⁽⁰⁾ (only batch[0]'s final state)
Next forward: All batches start from Wₙ⁽⁰⁾ (biased by batch[0]'s data)
```

### Impact

**Medium severity** (depends on usage):
- **If persistent_states only for inference (B=1)**: No impact
- **If persistent_states used during training (B>1)**: Critical bug
  - TTT states biased toward batch[0]
  - Batches 1...B-1 have incorrect initial states
  - Training dynamics corrupted

**Recommendation**: Document that `persistent_states=True` only supports `batch_size=1`, or fix to handle B>1.

---

## Issue #4: Gradient Flow Corruption 🟠

### The Problem

**Persistent state updates during forward pass (with `no_grad()`) interfere with optimizer updates, preventing proper training of initial TTT weights.**

### Code Evidence

**File**: `moshi_ttt_try/moshi-finetune/moshi_ttt/ttt_layer.py:626-707`

```python
def ttt(self, inputs):
    B = inputs["XV"].shape[0]

    # Initialize from model parameters (these should be trained by optimizer)
    W1_states = torch.tile(self.W1.unsqueeze(0), dims=(B, 1, 1, 1))
    #                      ^^^^^^^^
    #                      Gradients can flow back to self.W1

    b1_states = torch.tile(self.b1.unsqueeze(0), dims=(B, 1, 1, 1))
    W2_states = torch.tile(self.W2.unsqueeze(0), dims=(B, 1, 1, 1))
    b2_states = torch.tile(self.b2.unsqueeze(0), dims=(B, 1, 1, 1))

    # TTT inner loop computation (differentiable)
    XQW_batch, final_states = ttt_mlp_with_states(
        ..., W1_states, b1_states, W2_states, b2_states, ...
    )

    # DURING FORWARD PASS: Overwrite parameters with no_grad()
    with torch.no_grad():
        final_W1 = final_states["W1_states"][0]
        self.W1.data.copy_(final_W1)  # ← Overwrites self.W1.data!
        self.b1.data.copy_(final_states["b1_states"][0])
        self.W2.data.copy_(final_states["W2_states"][0])
        self.b2.data.copy_(final_states["b2_states"][0])

    return XQW_batch
```

### Timeline of Operations

**Training iteration `t`:**

```
1. Forward Pass (this method):
   ├─ self.W1 = W_t (from previous iteration or initialization)
   ├─ Create W1_states = tile(self.W1)
   ├─ TTT inner loop: W_t → W_{t+1} → ... → W_{t+n}
   ├─ XQW_batch = f(W_t, W_{t+1}, ..., W_{t+n})  [differentiable]
   └─ with torch.no_grad():
        self.W1.data.copy_(W_{t+n})  # ← Overwrites W_t with W_{t+n}!

2. Backward Pass:
   ├─ Compute gradients: ∂Loss/∂XQW_batch
   ├─ Backprop through TTT inner loop
   ├─ Gradients flow back to W1_states
   ├─ Since W1_states was created from self.W1 (via tile)
   └─ self.W1.grad accumulates (gradient with respect to W_t)

3. Optimizer Step:
   ├─ Read: self.W1 = W_{t+n}  (was overwritten in forward!)
   ├─ Read: self.W1.grad = ∂Loss/∂W_t  (gradient for OLD value)
   ├─ Update: self.W1 = W_{t+n} - lr × ∂Loss/∂W_t
   └─ ❌ WRONG! Applying gradient for W_t to value W_{t+n}
```

### The Core Problem

**Two concurrent updates to the same parameter**:

1. **TTT inner loop update** (during forward, no_grad):
   - `W_t → W_{t+n}` (test-time training adaptation)
   - Overwrites `self.W1.data`

2. **Optimizer update** (after backward):
   - Tries to update "initial" `W_t` based on gradients
   - But `W_t` was already replaced with `W_{t+n}`!
   - Applies `W_t`'s gradient to `W_{t+n}` (incorrect base value)

### The Architectural Incompatibility

**The fundamental issue: `self.W1` serves two conflicting roles simultaneously**

For sequential chunk training (where chunks should continue from previous state), you need:

1. **Trainable base weights** (W_base): Learned initialization, updated by optimizer
2. **Persistent state** (W_state): Current TTT state, carries across chunks

**Current implementation conflates these**, using `self.W1` for BOTH:
- As `nn.Parameter` → Trainable by optimizer (role #1)
- Overwritten each forward → Persistent state (role #2)

**This creates the gradient mismatch.**

### Comparison with Video-DiT

**Video-DiT** (`ttt-video-dit/ttt/models/ssm/ttt_layer.py:434-473`):
```python
def ttt(self, inputs):
    # Tile trainable parameter
    W1_states = torch.tile(self.W1.unsqueeze(0), dims=(B, 1, 1, 1))

    # Run TTT
    XQW_batch = ttt_mlp(..., W1_states, ...)

    return XQW_batch
    # ← NO PERSISTENT STATES! Each forward starts from learned self.W1
```

**Data flow**:
- Batch N: `self.W1` (learned) → TTT → output
- Backward: Gradients to `self.W1` ✓
- Optimizer: Updates `self.W1` ✓
- Batch N+1: Starts from **updated `self.W1`** (not from batch N's final state)

**Result**: Each batch is independent. Optimizer can train W1 correctly.

**Moshi-TTT with persistent_states=True**:
```python
def ttt(self, inputs):
    W1_states = torch.tile(self.W1.unsqueeze(0), dims=(B, 1, 1, 1))
    XQW_batch, final_states = ttt_mlp_with_states(..., W1_states, ...)

    # Overwrite parameter
    with torch.no_grad():
        self.W1.data.copy_(final_states["W1_states"][0])

    return XQW_batch
```

**Data flow**:
- Chunk N: `self.W1=A` → TTT → `final=Z` → **copy Z to self.W1.data**
- Backward: Gradients for `A`, but `self.W1` now contains `Z`! ✗
- Optimizer: Updates `Z` using gradients for `A` (mismatch!) ✗
- Chunk N+1: Starts from `Z` (persistent state works)

**Result**: Persistent state works, but gradient flow is broken.

### The Correct Architecture for Sequential Training

To support both trainable base weights AND persistent states, you need **separation**:

```python
class TTTMLP:
    def __init__(self, ...):
        # Trainable base weights (learned by optimizer)
        self.W1_base = nn.Parameter(torch.normal(0, 0.02, ...))
        self.b1_base = nn.Parameter(torch.zeros(...))
        self.W2_base = nn.Parameter(torch.normal(0, 0.02, ...))
        self.b2_base = nn.Parameter(torch.zeros(...))

        # Persistent state (updated by TTT, NOT by optimizer)
        self.register_buffer('W1_state', None)  # Initialized on first use
        self.register_buffer('b1_state', None)
        self.register_buffer('W2_state', None)
        self.register_buffer('b2_state', None)

    def reset_ttt_state(self):
        """Call at start of new file/conversation"""
        self.W1_state = self.W1_base.detach().clone()
        self.b1_state = self.b1_base.detach().clone()
        self.W2_state = self.W2_base.detach().clone()
        self.b2_state = self.b2_base.detach().clone()

    def ttt(self, inputs):
        # Initialize state from base weights if needed
        if self.W1_state is None:
            self.reset_ttt_state()

        # Use persistent state as initialization
        W1_init = self.W1_state.unsqueeze(0).expand(B, -1, -1, -1)
        b1_init = self.b1_state.unsqueeze(0).expand(B, -1, -1, -1)
        # ... same for W2, b2

        # Run TTT (gradients flow to W1_base through initial state dependency)
        XQW_batch, final_states = ttt_mlp_with_states(..., W1_init, ...)

        # Update persistent state for next chunk
        with torch.no_grad():
            self.W1_state.copy_(final_states["W1_states"][0])
            self.b1_state.copy_(final_states["b1_states"][0])
            # ... same for W2, b2

        return XQW_batch
```

**Gradient flow** (CORRECT):
1. `W1_init` comes from `W1_state` (buffer)
2. BUT `W1_state` was initialized from `W1_base` (parameter) via `reset_ttt_state()`
3. Gradients flow to `W1_base` through the dependency chain
4. Optimizer updates `W1_base` ✓
5. Persistent `W1_state` maintains across chunks ✓
6. Call `reset_ttt_state()` at start of new file to reinitialize from learned base

**This separates concerns**:
- `W1_base`: What the outer loop learns (good initialization)
- `W1_state`: What TTT adapts during a sequence (test-time training)

### Impact

**Major training bug in current implementation**:
- The "learned initial weights" `W₀` that should be trained by the outer loop cannot be properly optimized
- Each iteration tries to optimize `W_t`, but the parameter was already modified to `W_{t+n}`
- Optimizer updates are applied to the wrong base values
- Gradient descent on initial weights is corrupted

**Current design is fundamentally incompatible** with having trainable W1/W2 parameters when `persistent_states=True`

### Video-DiT Comparison

**Video-DiT** (`ttt-video-dit/ttt/models/ssm/ttt_layer.py:404-407`):

```python
class TTTMLP(TTTBase):
    def __init__(self, config: ModelConfig, use_kernel: bool = True):
        super().__init__(config)
        # These are nn.Parameters (trainable by optimizer)
        self.W1 = nn.Parameter(torch.normal(0, 0.02, size=(...)))
        self.b1 = nn.Parameter(torch.zeros(...))
        self.W2 = nn.Parameter(...)
        self.b2 = nn.Parameter(...)
```

**Video-DiT does NOT overwrite these parameters during forward pass**. Each forward pass:
1. Uses `self.W1` as initial weights
2. TTT inner loop creates temporary updated weights
3. Parameters remain unchanged (optimizer can train them properly)

**Video-DiT does NOT have persistent states across forward passes** - this is the key difference for streaming.

---

## Summary and Recommendations

### Issue Priority

1. **Issue #1 (Ring Buffer)**: 🔴 **Requires architectural redesign**
   - Cannot be fixed without changing attention mechanism
   - Needs chunked processing like Video-DiT or sliding window with full context preservation

2. **Issue #2 (Normalization)**: 🟠 **Easy fix, retraining required**
   - One-line code change in `ln_reconstruction_target()`
   - Must retrain all checkpoints

3. **Issue #4 (Gradient Flow)**: 🔴 **Architectural redesign required**
   - **Current design is fundamentally broken**: `self.W1` serves two conflicting roles (trainable parameter + persistent state)
   - **Quick fix**: Disable persistent_states during training (`persistent_states=False`)
   - **Proper fix**: Separate W_base (trainable parameters) from W_state (buffers)
   - See "The Correct Architecture for Sequential Training" section above for implementation

4. **Issue #3 (Batch Size)**: 🟡 **Document or fix**
   - Add assertion: `assert B == 1 when persistent_states=True`
   - Or implement proper multi-batch state management

### Next Steps

1. **Immediate**:
   - Fix Issue #2 (normalization bug) - one line change
   - **CRITICAL**: Set `persistent_states=False` in training config OR implement proper W_base/W_state separation
2. **Short-term**:
   - Implement proper persistent state architecture (separate parameters from buffers)
   - Add file/conversation boundary detection to reset states appropriately
3. **Long-term**:
   - Redesign attention mechanism to solve Issue #1 (ring buffer)

### Files Verified

- ✅ `moshi/moshi/moshi/modules/transformer.py` (Ring KV Cache)
- ✅ `moshi_ttt_try/moshi-finetune/moshi_ttt/ttt_layer.py` (All issues)
- ✅ `moshi_ttt_try/moshi-finetune/moshi_ttt/models/ssm/ops/ttt_mlp.py` (Reconstruction target)
- ✅ `ttt-video-dit/ttt/models/ssm/ttt_layer.py` (Comparison reference)
- ✅ `ttt-video-dit/ttt/models/cogvideo/dit.py` (Comparison reference)

---

## Issue #5: Cross-File State Contamination 🔴

### The Problem

**TTT weights persist across file boundaries without reset, causing state from one audio file to contaminate the adaptation for subsequent files with different speakers and acoustic conditions.**

### Discovery

This issue was discovered during comprehensive end-to-end training flow analysis (see `docs/FINAL_VERIFICATION.md`). Despite the data pipeline correctly tracking file boundaries, the training loop completely ignores this information.

### Code Evidence

#### 1. Data Pipeline Tracks File Boundaries ✅

**File**: `moshi_ttt_try/moshi-finetune/finetune/data/interleaver.py:305-315`

```python
# Track file_id and chunk_index for continuous RoPE
file_id = path  # Use path as unique file identifier
chunk_index = int(start_sec / self.duration_sec)  # Calculate chunk position

# Update chunk tracker (for sequential tracking if needed)
if file_id not in self._file_chunk_map:
    self._file_chunk_map[file_id] = 0
else:
    self._file_chunk_map[file_id] += 1

return Sample(codes, data.get("text_conditions", None),
              file_id=file_id, chunk_index=chunk_index)
```

**File**: `moshi_ttt_try/moshi-finetune/finetune/data/interleaver.py:17-22`

```python
@dataclass
class Sample:
    codes: torch.Tensor
    condition_attributes: ConditionAttributes | None = None
    file_id: str | None = None  # ← Track source file
    chunk_index: int | None = None  # ← Track chunk position within file
```

**File**: `moshi_ttt_try/moshi-finetune/finetune/data/interleaver.py:26-30`

```python
@dataclass
class Batch:
    codes: torch.Tensor
    condition_attributes: list[ConditionAttributes] | None = None
    file_id: str | None = None  # ← Available in batch
    chunk_index: int | None = None
```

#### 2. Sequential Processing Confirmed ✅

**File**: `moshi_ttt_try/moshi-finetune/finetune/data/dataset.py:250-261`

```python
def get_dataset_iterator(...):
    while True:
        for jsonl_file in source.jsonl_files:
            dataset = sphn.dataset_jsonl(...)

            if shuffle_at_epoch:  # FALSE for TTT training!
                dataset = dataset.shuffle(...)
            else:
                # SEQUENTIAL processing - chunks from same file are consecutive
                dataset = dataset.seq(skip=rank, step_by=world_size)

            for sample in dataset:
                wav = sample["data"][..., : sample["unpadded_len"]]
                result = instruct_tokenizer(wav, sample["start_time_sec"], sample["path"])
                if result is not None:
                    yield result
```

**Processing order**:
```
File1_chunk0 → File1_chunk1 → File1_chunk2 → ... → File1_chunkN →
File2_chunk0 → File2_chunk1 → ...  ← NEW FILE, but no state reset!
File3_chunk0 → ...
```

#### 3. Training Loop Ignores File Boundaries ❌

**File**: `moshi_ttt_try/moshi-finetune/training/train_ttt_production.py:234-298`

```python
while step < args.max_steps:
    step += 1
    step_start = time.time()

    # Get batch
    batch = next(data_loader)  # ← Contains file_id and chunk_index
    codes = batch.codes.to(device)

    # ❌ NO file_id check!
    # ❌ NO state reset logic!
    # batch.file_id is completely ignored!

    # Forward pass
    optimizer.zero_grad()

    # ... condition tensors setup ...

    # Model forward pass
    for mb_idx in range(args.num_microbatches):
        output = model(codes=codes, condition_tensors=condition_tensors)
        # ↑ TTT weights carry over from previous batch
        # ↑ Even if previous batch was from a different file!

        # Compute losses
        text_loss = compute_loss_with_mask(...)
        audio_loss = compute_loss_with_mask(...)
        mb_loss = text_loss + audio_loss
        mb_loss.backward()

    # Optimizer step
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
    optimizer.step()
    scheduler.step()
```

**Verification**: Grep search for `file_id` usage in training code:
```bash
grep -r "batch\.file_id\|file_id" moshi_ttt_try/moshi-finetune/training/train_ttt_production.py
# Result: No matches found!
```

#### 4. No State Reset Mechanism Exists ❌

**Verification**: Grep search for state reset logic:
```bash
grep -r "reset.*ttt\|reset.*state\|streaming_forever" moshi_ttt_try/moshi-finetune/training/
# Result: No matches found!
```

### Contamination Flow

**Iteration Trace** (with 2 files, each with 3 chunks):

```
Training Step 1: File 1, Chunk 0 (0-10s)
├─ TTT weights: W_init → W1_0
└─ State: Adapted to File 1, Chunk 0

Training Step 2: File 1, Chunk 1 (10-20s)
├─ TTT weights: W1_0 → W1_1 (continues from previous)
└─ State: Adapted to File 1, Chunks 0-1 ✅ CORRECT (same file)

Training Step 3: File 1, Chunk 2 (20-30s)
├─ TTT weights: W1_1 → W1_2 (continues from previous)
└─ State: Adapted to File 1, Chunks 0-2 ✅ CORRECT (same file)

Training Step 4: File 2, Chunk 0 (0-10s) ← NEW FILE!
├─ TTT weights: W1_2 → W2_0 (continues from File 1!)
└─ State: File 2 starts with File 1's final state ❌ WRONG!
    ↳ Should reset to W_base and start fresh
    ↳ Instead carries File 1's adaptations into File 2

Training Step 5: File 2, Chunk 1 (10-20s)
├─ TTT weights: W2_0 → W2_1 (contaminated by File 1)
└─ State: File 2 adaptation corrupted by File 1 ❌ WRONG!

Training Step 6: File 2, Chunk 2 (20-30s)
├─ TTT weights: W2_1 → W2_2 (contaminated by File 1)
└─ State: File 2 fully processed, but initial adaptation was wrong ❌ WRONG!
```

### Impact on Two-Loop Learning Paradigm

The TTT two-loop paradigm requires:

| Loop | Purpose | Current Behavior | Expected Behavior |
|------|---------|------------------|-------------------|
| **Outer Loop** (Optimizer) | Learn W_base that generalizes across files | Receives contaminated gradients | Should receive per-file adaptation signals |
| **Inner Loop** (TTT) | Adapt W_base to specific file | Carries state across files | Should reset at file boundaries |

**Current (BROKEN)**:
```
File 1: W_base → W_adapt_file1 (good)
File 2: W_adapt_file1 → W_adapt_file2 (contaminated!)
File 3: W_adapt_file2 → W_adapt_file3 (contaminated!)

Optimizer sees: W_adapt_file3 - W_base (WRONG!)
Should see: (W_adapt_file1 - W_base) + (W_adapt_file2 - W_base) + (W_adapt_file3 - W_base)
```

**Expected (CORRECT)**:
```
File 1: W_base → W_adapt_file1 (reset) → W_base
File 2: W_base → W_adapt_file2 (reset) → W_base
File 3: W_base → W_adapt_file3 (reset) → W_base

Optimizer receives: Per-file adaptation gradients
Learns: W_base that works well as starting point for all files
```

### Real-World Impact

**Scenario**: Training on diverse dataset

| File | Content | Expected TTT Behavior | Actual Behavior | Impact |
|------|---------|----------------------|-----------------|--------|
| File 1 | Male speaker, noisy | Adapt to male voice + noise reduction | ✅ Correct | ✅ Good |
| File 2 | Female speaker, quiet | Reset and adapt to female voice | ❌ Continues from File 1 | 🔴 Contaminated |
| File 3 | Child speaker, music | Reset and adapt to child + music | ❌ Continues from File 2 | 🔴 Contaminated |
| File 4 | Accented speech | Reset and adapt to accent | ❌ Continues from File 3 | 🔴 Contaminated |

**Consequences**:
1. **Incorrect adaptation**: File 2 starts with weights optimized for File 1's male speaker
2. **Slow convergence**: TTT must "unlearn" File 1 before adapting to File 2
3. **Degraded performance**: Especially on first chunks of new files
4. **Training instability**: Optimizer receives wrong gradient signals
5. **Violated assumptions**: Two-loop paradigm requires independent per-example adaptation

### Comparison with Video-DiT

**Video-DiT** (from `ttt-video-dit/ttt/models/ssm/ttt_layer.py:404-473`):

```python
def ttt(self, inputs):
    # NO persistent_states!
    # Each forward pass is independent

    B, S, H, P, C = inputs.shape

    # Initialize from base parameters (always fresh)
    W1_states = torch.tile(self.W1.unsqueeze(0), dims=(B, 1, 1, 1, 1))

    # Run TTT inner loop
    XQW_batch = ttt_mlp(inputs, ..., W1_states, ...)

    # NO state copying - states discarded after forward
    return XQW_batch  # Return output only
```

**Key difference**: Video-DiT processes complete videos in batches, with no cross-video state persistence. Each video gets independent TTT adaptation.

**Moshi-TTT**: Sequential chunks from long files NEED persistence within files, but SHOULD reset between files.

### Why This Wasn't Caught Earlier

1. **Data pipeline is correct**: File tracking works perfectly
2. **Information is available**: `batch.file_id` and `batch.chunk_index` are present
3. **Silent failure**: No error or warning - state just carries over silently
4. **Training still runs**: Loss decreases (but suboptimally)
5. **Requires end-to-end analysis**: Only visible when tracing complete data→training flow

### Related Issues

- **Issue #4** (Gradient Flow Corruption): Both issues relate to improper state management
- Fixing both together is recommended (see `docs/MOSHI_TTT_FIXES.md`)

---

## Training Configuration Issue 🔴

### Additional Finding: Persistent States Enabled By Default

**During training code review, discovered that Issue #4 is ACTIVE by default in training due to configuration.**

**File**: `finetune/args.py:74`
```python
@dataclass
class TTTArgs(Serializable):
    persistent_states: bool = True  # ← DEFAULT IS TRUE!
```

**Impact**: All training runs using default configuration have corrupted gradient flow for TTT weights (W1, W2, b1, b2).

**Production config** (`configs/production_ttt_dailytalk.yaml`) does NOT override this, so it inherits `persistent_states=True`.

**Fix**: Change default to `False` for training, or add training mode check in `ttt_layer.py:647`:
```python
if hasattr(self, 'persistent_states') and self.persistent_states and not self.training:
    # Only use persistent states during eval/inference
```

**See**: `docs/TRAINING_CODE_ANALYSIS.md` for complete training pipeline review.

---

**Analysis Date**: 2025-11-10
**Verification Status**: All issues code-verified with line-by-line analysis
**Training Pipeline**: Reviewed - configuration bug found (persistent_states=True by default)
**Confidence Level**: High (based on direct code reading and comparison with reference implementation)
