# Moshi-TTT Implementation Fixes

**Date**: 2025-11-10
**Status**: Proposed Fixes for Critical Issues
**Branch**: `claude/deep-code-review-ttt-011CUzpCD2kNLGH5UnuHN7hF`

---

## Overview

This document provides concrete, actionable fixes for all 4 critical issues identified in the Moshi-TTT implementation.

**See**: `docs/MOSHI_TTT_CRITICAL_ISSUES.md` for detailed analysis of each issue.

---

## Issue #1: Ring KV Cache Information Loss 🔴

### Problem Summary

Moshi's attention uses a ring buffer (capacity=3000 tokens) that permanently discards old tokens via modulo wraparound. TTT never sees the complete sequence history.

### Fix Option A: Increase Ring Buffer Capacity (Quick Fix)

**File**: `moshi/moshi/moshi/modules/transformer.py:208`

```python
class RingKVCache:
    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        dim_per_head: int,
        capacity: int,  # ← Increase this value
        ...
    ):
        self.capacity = capacity
```

**Change capacity default from 3000 to larger value**:

**File**: `moshi/moshi/moshi/modules/transformer.py:460-466`

```python
if self.context is None:
    if self.weights_per_step:
        capacity = self.weights_per_step
    else:
        raise RuntimeError("Cannot create streaming KVCache without context")
else:
    # OLD: capacity = self.context  # default 3000
    # NEW: capacity = self.context * 2  # 6000 tokens (~8 minutes @ 12.5 Hz)
    capacity = self.context * 2
```

**Pros**:
- Simple one-line change
- No architectural changes needed
- Doubles the effective context window

**Cons**:
- Increases memory usage (2x KV cache memory)
- Still has a limit (just delayed)
- Doesn't fundamentally solve the problem

**Memory Impact**: ~2GB additional memory per layer (for 2x increase)

---

### Fix Option B: Chunked Attention Processing (Proper Fix)

Implement Video-DiT style chunked attention where each chunk sees complete tokens.

**File**: Create new `moshi_ttt/chunked_attention.py`

```python
import torch
import torch.nn.functional as F
from typing import Optional

class ChunkedAttentionProcessor:
    """
    Process attention in chunks, each chunk seeing complete tokens.
    Similar to Video-DiT's approach.
    """

    def __init__(
        self,
        chunk_size: int = 750,  # 60 seconds @ 12.5 Hz
        overlap: int = 150,     # 12 seconds overlap for continuity
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def forward(
        self,
        query: torch.Tensor,       # [B, L, num_heads, head_dim]
        key: torch.Tensor,         # [B, L, num_heads, head_dim]
        value: torch.Tensor,       # [B, L, num_heads, head_dim]
        prefix_length: int = 0,    # Text tokens to prepend to each chunk
        prefix_key: Optional[torch.Tensor] = None,
        prefix_value: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Process attention in overlapping chunks.

        Args:
            query: Query tensor for full sequence
            key: Key tensor for full sequence
            value: Value tensor for full sequence
            prefix_length: Number of text tokens to include in each chunk
            prefix_key: Key for prefix tokens
            prefix_value: Value for prefix tokens

        Returns:
            Attention output for full sequence
        """
        B, L, num_heads, head_dim = query.shape

        # Calculate number of chunks
        effective_chunk = self.chunk_size - self.overlap
        num_chunks = (L + effective_chunk - 1) // effective_chunk

        output = torch.zeros_like(query)

        for i in range(num_chunks):
            # Calculate chunk boundaries
            start_idx = max(0, i * effective_chunk - self.overlap)
            end_idx = min(L, start_idx + self.chunk_size)

            # Extract chunk
            chunk_q = query[:, start_idx:end_idx]
            chunk_k = key[:, start_idx:end_idx]
            chunk_v = value[:, start_idx:end_idx]

            # Prepend prefix (text) if provided
            if prefix_key is not None and prefix_value is not None:
                chunk_k = torch.cat([prefix_key, chunk_k], dim=1)
                chunk_v = torch.cat([prefix_value, chunk_v], dim=1)

            # Apply attention to COMPLETE chunk
            chunk_output = F.scaled_dot_product_attention(
                chunk_q.transpose(1, 2),  # [B, num_heads, chunk_len, head_dim]
                chunk_k.transpose(1, 2),
                chunk_v.transpose(1, 2),
                is_causal=True,
            )
            chunk_output = chunk_output.transpose(1, 2)  # Back to [B, chunk_len, num_heads, head_dim]

            # Write to output (handle overlap by averaging)
            actual_start = i * effective_chunk
            actual_end = min(L, actual_start + (end_idx - start_idx))
            output_slice = output[:, actual_start:actual_end]

            if output_slice.abs().sum() > 0:
                # Overlap region - average with existing
                output[:, actual_start:actual_end] = (output_slice + chunk_output) / 2
            else:
                # First time writing to this region
                output[:, actual_start:actual_end] = chunk_output

        return output
```

**Integration**: Modify `StreamingTransformerLayer._sa_block()` to use chunked processor during training:

```python
def _sa_block(self, x: torch.Tensor) -> torch.Tensor:
    if self.training:
        # Use chunked attention for training (full context)
        q = self.self_attn.in_proj_q(x)
        k = self.self_attn.in_proj_k(x)
        v = self.self_attn.in_proj_v(x)

        chunked_processor = ChunkedAttentionProcessor(
            chunk_size=750,  # 60 seconds @ 12.5 Hz
            overlap=150,     # 12 seconds overlap
        )
        attn_output = chunked_processor.forward(q, k, v)
        x = self.self_attn.out_proj(attn_output)
    else:
        # Use ring buffer for streaming inference
        x = self.self_attn(x)

    return x
```

**Pros**:
- TTT sees complete sequence information
- No artificial memory limit
- Matches Video-DiT architecture

**Cons**:
- More complex implementation
- Requires training/eval mode distinction
- Slower training (more computation per chunk)

**Complexity**: High (2-3 days implementation + testing)

---

## Issue #2: Reconstruction Target Normalization Bug 🟠

### Problem Summary

The code normalizes `XV` directly instead of normalizing the difference `(XV - XK)`, resulting in incorrect reconstruction targets.

### Fix: Normalize the Difference

**File**: `moshi_ttt_try/moshi-finetune/moshi_ttt/ttt_layer.py:463-478`

**Current (INCORRECT)**:
```python
def ln_reconstruction_target(self, XV, XK):
    """Layer norm reconstruction target following Video-DiT pattern"""
    B, L, num_heads, head_dim = XV.shape

    # Apply layer norm per head
    XV_normed = torch.zeros_like(XV)
    for h in range(num_heads):
        weight = self.ttt_norm_weight[h]
        bias = self.ttt_norm_bias[h]

        # BUG: Normalizing XV directly
        XV_h = XV[:, :, h, :]
        XV_normed[:, :, h, :] = F.layer_norm(XV_h, (head_dim,), weight, bias, eps=1e-6)

    return XV_normed  # Returns norm(XV)
```

**Fixed (CORRECT)**:
```python
def ln_reconstruction_target(self, XV, XK):
    """Layer norm reconstruction target following Video-DiT pattern"""
    B, L, num_heads, head_dim = XV.shape

    # Step 1: Compute difference FIRST
    diff = XV - XK

    # Step 2: Normalize the DIFFERENCE per head
    diff_normed = torch.zeros_like(diff)
    for h in range(num_heads):
        weight = self.ttt_norm_weight[h]  # [head_dim]
        bias = self.ttt_norm_bias[h]      # [head_dim]

        # Normalize the difference
        diff_h = diff[:, :, h, :]  # [B, L, head_dim]
        diff_normed[:, :, h, :] = F.layer_norm(diff_h, (head_dim,), weight, bias, eps=1e-6)

    # Step 3: Add XK back (matches Video-DiT line 235)
    return diff_normed + XK  # Returns norm(XV - XK) + XK
```

**Or more efficiently** (vectorized):
```python
def ln_reconstruction_target(self, XV, XK):
    """Layer norm reconstruction target following Video-DiT pattern (vectorized)"""
    B, L, num_heads, head_dim = XV.shape

    # Compute difference
    diff = XV - XK  # [B, L, num_heads, head_dim]

    # Compute per-head statistics
    eps = 1e-6
    mean = diff.mean(dim=-1, keepdim=True)  # [B, L, num_heads, 1]
    std = diff.std(dim=-1, keepdim=True)     # [B, L, num_heads, 1]

    # Normalize
    diff_normed = (diff - mean) / (std + eps)

    # Apply per-head scale and shift
    # ttt_norm_weight: [num_heads, head_dim]
    # ttt_norm_bias: [num_heads, head_dim]
    weight = self.ttt_norm_weight.unsqueeze(0).unsqueeze(0)  # [1, 1, num_heads, head_dim]
    bias = self.ttt_norm_bias.unsqueeze(0).unsqueeze(0)      # [1, 1, num_heads, head_dim]

    diff_normed = diff_normed * weight + bias

    # Add XK back
    return diff_normed + XK
```

**Pros**:
- One function change
- Matches Video-DiT exactly
- Fixes TTT optimization objective

**Cons**:
- Requires retraining all checkpoints
- Previous checkpoints will be incompatible

**Complexity**: Low (15 minutes to implement, retrain required)

**Testing**:
```python
# Verify fix matches Video-DiT behavior
def test_ln_reconstruction_target():
    ttt_layer = TTTMLP(config)
    XV = torch.randn(2, 100, 8, 128)
    XK = torch.randn(2, 100, 8, 128)

    result = ttt_layer.ln_reconstruction_target(XV, XK)

    # Then in compute_mini_batch:
    reconstruction_target = result - XK
    # reconstruction_target should equal norm(XV - XK)

    # Verify statistics
    diff = XV - XK
    assert torch.allclose(
        reconstruction_target.mean(dim=-1),
        torch.zeros_like(reconstruction_target.mean(dim=-1)),
        atol=1e-5
    )
```

---

## Issue #3: Batch Size > 1 Incompatibility 🟡

### Problem Summary

Persistent state updates only save `batch[0]`, breaking multi-batch training.

### Fix Option A: Add Assertion (Quick Fix)

**File**: `moshi_ttt_try/moshi-finetune/moshi_ttt/ttt_layer.py:686`

```python
# Update model parameters with persistent states (JAX-style behavior)
# Extract final states and remove batch dimension for parameter update
with torch.no_grad():
    # Validate batch size
    if final_states["W1_states"].shape[0] > 1:
        raise RuntimeError(
            f"persistent_states=True only supports batch_size=1. "
            f"Got batch_size={final_states['W1_states'].shape[0]}. "
            f"For multi-batch training, set persistent_states=False in config."
        )

    # final_states have shape [B, H, ...] - take first batch element [0] to get [H, ...]
    final_W1 = final_states["W1_states"][0]  # [B, H, HD, 4*HD] -> [H, HD, 4*HD]
    ...
```

**Pros**:
- Fails fast with clear error message
- Prevents silent bugs
- Simple to implement

**Cons**:
- Doesn't fix the underlying issue
- Limits training flexibility

---

### Fix Option B: Per-Batch State Management (Proper Fix)

If you need multi-batch training with persistent states (unlikely), implement per-batch state tracking:

```python
class TTTMLP:
    def __init__(self, ...):
        ...
        # Per-batch persistent states
        self.register_buffer('W1_states_per_batch', None)  # [max_batch, H, HD, 4*HD]
        self.register_buffer('batch_state_initialized', None)  # [max_batch] bool mask
        self.max_batch_size = 32  # Configure based on training

    def ttt(self, inputs):
        B = inputs["XV"].shape[0]

        # Initialize per-batch states if needed
        if self.W1_states_per_batch is None:
            self.W1_states_per_batch = self.W1.unsqueeze(0).expand(
                self.max_batch_size, -1, -1, -1
            ).clone()
            self.batch_state_initialized = torch.zeros(
                self.max_batch_size, dtype=torch.bool, device=self.W1.device
            )

        # Use per-batch states
        W1_states = self.W1_states_per_batch[:B].clone()

        # Run TTT
        XQW_batch, final_states = ttt_mlp_with_states(..., W1_states, ...)

        # Update per-batch states
        with torch.no_grad():
            self.W1_states_per_batch[:B] = final_states["W1_states"]

        return XQW_batch
```

**Complexity**: Medium (but likely not needed)

---

## Issue #4: Gradient Flow Corruption 🔴

### Problem Summary

`self.W1` serves two conflicting roles: trainable parameter (updated by optimizer) and persistent state (overwritten each forward). This creates gradient mismatch.

### Fix Option A: Disable During Training (Quick Fix)

**File**: `moshi_ttt_try/moshi-finetune/finetune/args.py:74`

```python
@dataclass
class TTTArgs(Serializable):
    """Configuration for Test-Time Training (TTT) layers in Moshi"""
    enable: bool = False
    layers: str = "middle"
    base_lr: float = 1.0
    mini_batch_size: int = 16
    # OLD: persistent_states: bool = True
    # NEW: persistent_states: bool = False  # Disable for training
    persistent_states: bool = False
    initial_gating_alpha: float = 0.1
```

**Or add training mode check** in `moshi_ttt/ttt_layer.py:647`:

```python
# Check for persistent states support
# OLD: if hasattr(self, 'persistent_states') and self.persistent_states:
# NEW: Only use persistent states during evaluation/inference
if hasattr(self, 'persistent_states') and self.persistent_states and not self.training:
    # Use state-returning version for persistent TTT states (JAX-style)
    ...
else:
    # Use standard version (no state persistence)
    ...
```

**Pros**:
- Immediate fix
- Restores correct gradient flow
- No architectural changes

**Cons**:
- Loses persistent state capability during training
- Each chunk starts from scratch (no continuity across file chunks)

**Complexity**: Trivial (5 minutes)

---

### Fix Option B: Separate Parameters from Buffers (Proper Fix)

Implement the correct architecture that separates trainable base weights from persistent state.

**File**: `moshi_ttt_try/moshi-finetune/moshi_ttt/ttt_layer.py:591-625`

**Current (BROKEN)**:
```python
class TTTMLP(TTTBase):
    def __init__(self, config: TTTConfig, layer_id: int = None, use_kernel: bool = False):
        super().__init__(config, layer_id)

        # W1/W2 serve BOTH roles (conflicting!)
        self.W1 = nn.Parameter(torch.normal(0, 0.02, size=(...)))
        self.b1 = nn.Parameter(torch.zeros(...))
        self.W2 = nn.Parameter(torch.normal(0, 0.02, size=(...)))
        self.b2 = nn.Parameter(torch.zeros(...))
```

**Fixed (CORRECT)**:
```python
class TTTMLP(TTTBase):
    def __init__(self, config: TTTConfig, layer_id: int = None, use_kernel: bool = False):
        super().__init__(config, layer_id)

        # SEPARATE concerns:

        # 1. Trainable base weights (learned by optimizer)
        self.W1_base = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, self.head_dim, 4 * self.head_dim)))
        self.b1_base = nn.Parameter(torch.zeros(self.num_heads, 1, 4 * self.head_dim))
        self.W2_base = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, 4 * self.head_dim, self.head_dim)))
        self.b2_base = nn.Parameter(torch.zeros(self.num_heads, 1, self.head_dim))

        # 2. Persistent state (buffers, NOT trainable)
        self.register_buffer('W1_state', None)
        self.register_buffer('b1_state', None)
        self.register_buffer('W2_state', None)
        self.register_buffer('b2_state', None)

        # 3. Track whether state is initialized
        self.register_buffer('state_initialized', torch.tensor(False))

        self.use_kernel = use_kernel
        self.stream_position = 0

    def reset_ttt_state(self):
        """
        Reset TTT state to learned base weights.
        Call at start of new file or conversation.
        """
        with torch.no_grad():
            self.W1_state = self.W1_base.detach().clone()
            self.b1_state = self.b1_base.detach().clone()
            self.W2_state = self.W2_base.detach().clone()
            self.b2_state = self.b2_base.detach().clone()
            self.state_initialized.fill_(True)

    def init_weights(self):
        super().init_weights()
        nn.init.normal_(self.W1_base, mean=0.0, std=0.02)
        nn.init.zeros_(self.b1_base)
        nn.init.normal_(self.W2_base, mean=0.0, std=0.02)
        nn.init.zeros_(self.b2_base)

    def ttt(self, inputs):
        """TTT processing with proper persistent state handling"""
        B = inputs["XV"].shape[0]
        num_mini_batch = inputs["XV"].shape[2]

        # Initialize or use persistent state
        if hasattr(self, 'persistent_states') and self.persistent_states:
            # Initialize state from base if needed
            if self.W1_state is None or not self.state_initialized:
                self.reset_ttt_state()

            # Use persistent state as initialization
            W1_states = self.W1_state.unsqueeze(0).expand(B, -1, -1, -1)
            b1_states = self.b1_state.unsqueeze(0).expand(B, -1, -1, -1)
            W2_states = self.W2_state.unsqueeze(0).expand(B, -1, -1, -1)
            b2_states = self.b2_state.unsqueeze(0).expand(B, -1, -1, -1)
        else:
            # Use base weights directly (no persistence)
            W1_states = self.W1_base.unsqueeze(0).expand(B, -1, -1, -1)
            b1_states = self.b1_base.unsqueeze(0).expand(B, -1, -1, -1)
            W2_states = self.W2_base.unsqueeze(0).expand(B, -1, -1, -1)
            b2_states = self.b2_base.unsqueeze(0).expand(B, -1, -1, -1)

        # Checkpointing setup
        if self.training:
            checkpoint_group_size = min(max(self.scan_checkpoint_group_size, 1), num_mini_batch)
        else:
            checkpoint_group_size = 0

        # Run TTT with state return
        XQW_batch, final_states = ttt_mlp_with_states(
            inputs["XK"],
            inputs["XQ"],
            inputs["XV"],
            inputs["eta"],
            self.ttt_norm_weight,
            self.ttt_norm_bias,
            W1_states,
            b1_states,
            W2_states,
            b2_states,
            checkpoint_group_size,
        )

        # Update persistent state if enabled
        if hasattr(self, 'persistent_states') and self.persistent_states:
            with torch.no_grad():
                self.W1_state.copy_(final_states["W1_states"][0])
                self.b1_state.copy_(final_states["b1_states"][0])
                self.W2_state.copy_(final_states["W2_states"][0])
                self.b2_state.copy_(final_states["b2_states"][0])

        # Reshape and return
        original_length = inputs["original_length"]
        XQW_batch = XQW_batch.reshape(B, num_mini_batch * inputs["mini_batch_size"], self.width)
        XQW_batch = XQW_batch[:, :original_length, :]

        self.stream_position += num_mini_batch

        return XQW_batch
```

**Gradient Flow** (Now CORRECT):
1. Forward: `W1_state` → TTT → output (differentiable)
2. `W1_state` was initialized from `W1_base` via `reset_ttt_state()`
3. Gradients flow: output → TTT → `W1_state` dependency → **`W1_base`** ✓
4. Optimizer updates `W1_base` ✓
5. `W1_state` persists across chunks ✓

**Data Loader Integration** - detect file boundaries:

**File**: Create `moshi_ttt_try/moshi-finetune/finetune/data/boundary_detection.py`

```python
class FileAwareDataLoader:
    """Wrapper that tracks file boundaries for TTT state resets"""

    def __init__(self, base_loader, model):
        self.base_loader = base_loader
        self.model = model
        self.last_file_id = None

    def __iter__(self):
        for batch in self.base_loader:
            # Check if this is a new file
            current_file_id = batch.get('file_id', None)

            if current_file_id is not None and current_file_id != self.last_file_id:
                # New file detected - reset TTT states
                self._reset_ttt_states()
                self.last_file_id = current_file_id

            yield batch

    def _reset_ttt_states(self):
        """Reset TTT states in all hybrid layers"""
        for module in self.model.modules():
            if isinstance(module, TTTMLP):
                module.reset_ttt_state()
```

**Training Script Integration**:

```python
# In train_ttt_production.py
from finetune.data.boundary_detection import FileAwareDataLoader

# After creating data_loader
data_loader = FileAwareDataLoader(data_loader, model)
```

**Pros**:
- Correct gradient flow to base weights
- Persistent state works across chunks
- Base weights learned by optimizer
- State resets at file boundaries

**Cons**:
- More complex implementation
- Requires data loader modifications for boundary detection
- Needs careful state management

**Complexity**: High (2-3 days implementation + testing)

**Migration Path**:
1. Implement new `W1_base`/`W1_state` architecture
2. For existing checkpoints, copy `self.W1` → `self.W1_base` and `self.W1_state`
3. Add checkpoint converter script

---

## Issue #5: Cross-File State Contamination 🔴

### Problem Statement

TTT weights persist across file boundaries without reset, causing state from one audio file to contaminate adaptation for subsequent files. This violates the two-loop learning paradigm where the outer loop (optimizer) should learn W_base that generalizes, while the inner loop (TTT) should adapt W_base independently to each file.

**Discovered**: During end-to-end training flow analysis (see `docs/FINAL_VERIFICATION.md`)

**Evidence**:
- Data pipeline tracks `file_id` and `chunk_index` ✅
- Training loop completely ignores this information ❌
- No state reset mechanism exists ❌

### Fix Option A: Quick Fix (Disable persistent_states)

**Same as Issue #4 Quick Fix** - Disabling `persistent_states` prevents cross-file contamination but loses within-file continuity.

**Status**: Not recommended as primary fix (loses too much functionality)

### Fix Option B: Proper Fix (File Boundary Detection)

**Note**: This fix is INTEGRATED with Issue #4 Proper Fix. Both issues require proper state management and should be fixed together.

#### Component 1: State Reset Method (in TTT Layer)

**File**: `moshi_ttt_try/moshi-finetune/moshi_ttt/ttt_layer.py`

Add method to reset TTT states to base weights:

```python
def reset_ttt_state(self):
    """Reset TTT states to base weights (for file boundaries)"""
    if not hasattr(self, 'W1_base'):
        # Old architecture - no-op
        return

    # Reset states to base weights
    with torch.no_grad():
        if hasattr(self, 'W1_state'):
            self.W1_state.copy_(self.W1_base)
        if hasattr(self, 'b1_state'):
            self.b1_state.copy_(self.b1_base)
        if hasattr(self, 'W2_state'):
            self.W2_state.copy_(self.W2_base)
        if hasattr(self, 'b2_state'):
            self.b2_state.copy_(self.b2_base)
```

#### Component 2: File Boundary Detection (in Training Loop)

**File**: `moshi_ttt_try/moshi-finetune/training/train_ttt_production.py`

Add file boundary tracking to training loop:

```python
# Before training loop (line 230)
previous_file_id = None

# Inside training loop (after line 239)
while step < args.max_steps:
    step += 1
    step_start = time.time()

    # Get batch
    batch = next(data_loader)
    codes = batch.codes.to(device)

    # ✅ NEW: Check for file boundary
    current_file_id = batch.file_id
    if current_file_id is not None:
        if previous_file_id is not None and current_file_id != previous_file_id:
            # File boundary detected - reset TTT states
            logger.info(f"File boundary detected: {previous_file_id} → {current_file_id}")
            reset_ttt_states(model)

        previous_file_id = current_file_id

    # Forward pass
    optimizer.zero_grad()
    # ... rest of training loop ...

def reset_ttt_states(model):
    """Reset TTT states to base weights in all TTT layers"""
    count = 0
    for module in model.modules():
        if hasattr(module, 'reset_ttt_state'):
            module.reset_ttt_state()
            count += 1
    logger.info(f"Reset {count} TTT layers to base weights")
```

#### Component 3: Logging and Monitoring

Add metrics to track file boundaries and state resets:

```python
# In training loop logging (around line 306)
if step % args.log_freq == 0:
    # Existing logging...

    # NEW: Track file boundaries
    if hasattr(model, '_ttt_resets_count'):
        print(f"   TTT state resets: {model._ttt_resets_count}")

# In reset_ttt_states function
def reset_ttt_states(model):
    """Reset TTT states to base weights in all TTT layers"""
    count = 0
    for module in model.modules():
        if hasattr(module, 'reset_ttt_state'):
            module.reset_ttt_state()
            count += 1

    # Track resets
    if not hasattr(model, '_ttt_resets_count'):
        model._ttt_resets_count = 0
    model._ttt_resets_count += 1

    logger.info(f"Reset {count} TTT layers (total resets: {model._ttt_resets_count})")
```

### Complete Implementation (Issues #4 + #5 Together)

**Recommended approach**: Fix both Issue #4 and Issue #5 together, as they both require proper state management.

**Architecture changes**:
1. Separate `W_base` (trainable parameters) from `W_state` (persistent buffers)
2. Add `reset_ttt_state()` method to reset states to base
3. Add file boundary detection in training loop
4. Reset states when file boundary detected

**Data flow after fix**:
```
File 1, Chunk 0: W_base → W_state (adapt) → output → loss → ∂loss/∂W_base
File 1, Chunk 1: W_state (continue) → output → loss → ∂loss/∂W_base
File 1, Chunk 2: W_state (continue) → output → loss → ∂loss/∂W_base

[File boundary detected] → reset: W_state ← W_base

File 2, Chunk 0: W_base → W_state (fresh adapt) → output → loss → ∂loss/∂W_base
File 2, Chunk 1: W_state (continue) → output → loss → ∂loss/∂W_base
```

**Gradients received by optimizer**:
- Proper per-file adaptation signals
- No cross-file contamination
- Clean two-loop learning paradigm

### Testing

**Unit test**:
```python
def test_file_boundary_reset():
    """Verify TTT states reset at file boundaries"""
    model = load_model_with_ttt(persistent_states=True)
    batch1 = create_batch(file_id="/data/file1.wav", chunk_index=0)
    batch2 = create_batch(file_id="/data/file1.wav", chunk_index=1)
    batch3 = create_batch(file_id="/data/file2.wav", chunk_index=0)

    # Process File 1
    out1 = model(batch1.codes)
    W1_after_file1_chunk0 = model.transformer.layers[0].hybrid.W1_state.clone()

    out2 = model(batch2.codes)
    W1_after_file1_chunk1 = model.transformer.layers[0].hybrid.W1_state.clone()

    # State should have changed (adaptation)
    assert not torch.allclose(W1_after_file1_chunk0, W1_after_file1_chunk1)

    # File boundary - reset
    reset_ttt_states(model)

    # Process File 2
    out3 = model(batch3.codes)
    W1_after_file2_chunk0 = model.transformer.layers[0].hybrid.W1_state.clone()

    # State should be reset to base (not continue from File 1)
    W1_base = model.transformer.layers[0].hybrid.W1_base
    assert torch.allclose(W1_after_file2_chunk0, W1_base, atol=1e-3)
```

**Integration test**:
```python
def test_training_with_file_boundaries():
    """Verify training handles file boundaries correctly"""
    model = load_model_with_ttt(persistent_states=True)
    optimizer = AdamW(model.parameters(), lr=1e-4)

    # Create dataset with 2 files
    dataset = [
        create_batch(file_id="/data/file1.wav", chunk_index=0),
        create_batch(file_id="/data/file1.wav", chunk_index=1),
        create_batch(file_id="/data/file2.wav", chunk_index=0),  # File boundary
        create_batch(file_id="/data/file2.wav", chunk_index=1),
    ]

    previous_file_id = None
    reset_count = 0

    for batch in dataset:
        # File boundary detection
        if previous_file_id and batch.file_id != previous_file_id:
            reset_ttt_states(model)
            reset_count += 1
        previous_file_id = batch.file_id

        # Training step
        optimizer.zero_grad()
        output = model(batch.codes)
        loss = compute_loss(output, batch.codes)
        loss.backward()
        optimizer.step()

    # Should have detected 1 file boundary
    assert reset_count == 1
```

### Complexity and Impact

**Complexity**: Medium (requires Issue #4 proper fix first)
- Depends on W_base/W_state separation
- File boundary detection logic is straightforward
- Testing requires multi-file dataset

**Impact**: 🔴 Critical
- Restores proper two-loop learning
- Prevents cross-file contamination
- Enables correct per-file adaptation
- No performance overhead (reset is rare)

**Implementation time**: 1-2 days (including testing)

**Dependencies**:
- Issue #4 proper fix must be implemented first
- Requires W_base/W_state architecture

---

## Implementation Priority

### Immediate (Week 1)

1. **Issue #2 (Normalization Bug)** - MUST FIX
   - Complexity: Low
   - Impact: High (incorrect training objective)
   - Fix: Change `ln_reconstruction_target()` to normalize difference
   - Requires: Retraining all checkpoints

2. **Issue #4 Quick Fix (Disable persistent_states)** - MUST FIX
   - Complexity: Trivial
   - Impact: Critical (restore gradient flow)
   - Fix: Set `persistent_states=False` in config or add training mode check
   - Trade-off: Loses cross-chunk continuity (acceptable for initial fix)

### Short-term (Week 2-3)

3. **Issues #4 + #5 Proper Fix (Separate W_base/W_state + File Boundary Detection)** - RECOMMENDED
   - Complexity: High
   - Impact: Critical (enables proper persistent state training + prevents cross-file contamination)
   - Fix: Implement parameter/buffer separation architecture with file boundary detection
   - Enables: Sequential chunk training with proper gradients and per-file adaptation
   - Note: Both issues should be fixed together as they share the same state management architecture

4. **Issue #3 (Batch Size Assertion)** - NICE TO HAVE
   - Complexity: Low
   - Impact: Medium (prevents silent bugs)
   - Fix: Add validation in persistent state code path

### Long-term (Month 2-3)

5. **Issue #1 (Ring Buffer)** - ARCHITECTURAL IMPROVEMENT
   - Complexity: High
   - Impact: High (full context for TTT)
   - Fix Option A: Increase ring buffer capacity (quick)
   - Fix Option B: Implement chunked attention (proper)

---

## Testing Checklist

After implementing fixes:

### Unit Tests

- [ ] `test_ln_reconstruction_target()` - Verify normalization fix
- [ ] `test_gradient_flow_with_persistent_states()` - Verify W_base receives gradients
- [ ] `test_state_persistence_across_chunks()` - Verify W_state maintains across chunks
- [ ] `test_state_reset_at_boundaries()` - Verify reset_ttt_state() works
- [ ] `test_batch_size_validation()` - Verify B>1 assertion fires

### Integration Tests

- [ ] Train for 100 steps, verify loss decreases
- [ ] Verify W_base parameters change after training
- [ ] Verify W_state persists across chunks
- [ ] Test checkpoint save/load with new architecture
- [ ] Test eval mode (streaming) with persistent states

### Regression Tests

- [ ] Compare reconstruction targets before/after Issue #2 fix
- [ ] Verify gradient norms are reasonable after Issue #4 fix
- [ ] Profile memory usage after Issue #1 fixes

---

## Migration Guide

### For Existing Checkpoints

When implementing Issue #4 proper fix (W_base/W_state separation):

```python
def migrate_checkpoint(old_checkpoint_path, new_checkpoint_path):
    """Migrate old checkpoint to new W_base/W_state architecture"""
    checkpoint = torch.load(old_checkpoint_path)
    state_dict = checkpoint['model_state_dict']

    # Find all TTT layers
    for key in list(state_dict.keys()):
        if '.W1' in key and '.W1_base' not in key:
            # Old format: "layer.W1"
            # New format: "layer.W1_base" + "layer.W1_state"
            base_key = key.replace('.W1', '.W1_base')
            state_key = key.replace('.W1', '.W1_state')

            # Copy to both
            state_dict[base_key] = state_dict[key].clone()
            state_dict[state_key] = state_dict[key].clone()

            # Remove old key
            del state_dict[key]

        # Repeat for b1, W2, b2
        # ...

    checkpoint['model_state_dict'] = state_dict
    torch.save(checkpoint, new_checkpoint_path)
```

---

## Summary

| Issue | Priority | Complexity | Fix Time | Retrain? |
|-------|----------|------------|----------|----------|
| #2 Normalization | 🔴 Critical | Low | 15 min | Yes |
| #4 Quick Fix | 🔴 Critical | Trivial | 5 min | No |
| #4 + #5 Proper Fix | 🔴 Critical | High | 3-4 days | No (migration) |
| #3 Batch Size | 🟡 Medium | Low | 30 min | No |
| #1 Option A (Buffer) | 🟡 Medium | Low | 30 min | No |
| #1 Option B (Chunked) | 🟠 High | High | 2-3 days | Yes |

**Recommended Immediate Action**:
1. Fix Issue #2 (normalization) - start retraining
2. Fix Issues #4 + #5 quick (disable persistent_states during training) - temporarily disables cross-file contamination
3. Plan Issues #4 + #5 proper fix (W_base/W_state architecture + file boundary detection) - complete solution

---

**Document Version**: 1.0
**Last Updated**: 2025-11-10
**Status**: Ready for Implementation
