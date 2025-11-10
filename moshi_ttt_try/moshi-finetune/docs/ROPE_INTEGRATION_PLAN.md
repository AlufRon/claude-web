# RoPE Integration Plan - Surgical Implementation for Moshi-TTT

**Reference Implementation**: `/home/alufr/ttt_tests/ttt-lm-pytorch/ttt.py`
**Target Implementation**: `/home/alufr/ttt_tests/moshi-finetune/moshi_ttt/ttt_layer.py`
**Status**: Ready for implementation
**Date**: 2025-10-26

---

## Table of Contents
1. [Overview](#overview)
2. [Why RoPE is Critical](#why-rope-is-critical)
3. [Key Innovation: Position Modulo Trick](#key-innovation-position-modulo-trick)
4. [Files to Modify](#files-to-modify)
5. [New Code Components](#new-code-components)
6. [Integration Steps](#integration-steps)
7. [Testing Plan](#testing-plan)
8. [Rollback Plan](#rollback-plan)

---

## Overview

**Goal**: Add Rotary Position Embeddings (RoPE) to Moshi-TTT exactly as implemented in `ttt-lm-pytorch`, enabling proper position encoding for long-context audio processing.

**Architecture Pattern**:
- Current: `Input → QKV Projection → L2 Norm → TTT Processing → Output`
- After: `Input → QKV Projection → L2 Norm → **RoPE on Q/K** → TTT Processing → Output`

**Critical Insight**: TTT-LM uses `position_ids % mini_batch_size` to keep RoPE in a bounded range, preventing extrapolation issues seen with LoRA-only models.

---

## Why RoPE is Critical

### Problem: Long-Context Degradation
- **LoRA-only models**: Trained on 10-sec clips (positions 0-125) but see unbounded positions (5000+) during inference
- **Result**: Model degrades on long audio because RoPE extrapolates to never-seen positions
- **Reference**: User observation at `/home/alufr/ttt_tests/moshi-finetune/logs/inference/moshi_lora.7807993.log`

### Solution: Position Modulo Trick
```python
# From ttt-lm-pytorch line 869
cos, sin = self.rotary_emb(XV, position_ids % self.mini_batch_size)
```

**Why this works**:
1. TTT processes sequences in mini-batches (e.g., 64 tokens)
2. RoPE only needs to distinguish positions **within** each mini-batch
3. Position modulo ensures RoPE always sees positions 0-63, never extrapolates
4. TTT's persistent states handle longer-range dependencies naturally

---

## Key Innovation: Position Modulo Trick

### Mathematical Foundation

**Standard RoPE** (as used in Llama, Moshi attention):
```python
# Position grows unboundedly: 0, 1, 2, ..., 5000, 10000, ...
position_ids = torch.arange(0, seq_len)
cos, sin = rotary_emb(x, position_ids)
```
- **Problem**: If trained on positions 0-125, fails on 5000+

**TTT-LM RoPE** (position modulo):
```python
# Position wraps around: 0, 1, ..., 63, 0, 1, ..., 63, ...
position_ids_bounded = position_ids % mini_batch_size
cos, sin = rotary_emb(x, position_ids_bounded)
```
- **Benefit**: Always sees positions 0-63, no extrapolation
- **Why sufficient**: TTT states carry long-range information, RoPE only needs local context

### From ttt-lm-pytorch (line 869-874)
```python
# CRITICAL: Use modulo to keep positions in [0, mini_batch_size)
cos, sin = self.rotary_emb(XV, position_ids % self.mini_batch_size)

# Apply RoPE with permutation for JAX compatibility
XQ, XK = permute_qk(XQ, XK)
XQ, XK = apply_rotary_pos_emb(XQ, XK, cos, sin)
XQ, XK = undo_permute_qk(XQ, XK)
```

---

## Files to Modify

### 1. `/home/alufr/ttt_tests/moshi-finetune/moshi_ttt/ttt_layer.py`
**Changes**: Add RoPE classes and apply in `TTTBase.process_input()`

### 2. `/home/alufr/ttt_tests/moshi-finetune/moshi_ttt/config.py`
**Changes**: Add `rope_theta` parameter (already exists but unused)

### 3. `/home/alufr/ttt_tests/moshi-finetune/moshi_ttt/format_utils.py`
**Changes**: Add position_ids generation in `SequenceMetadata`

### 4. `/home/alufr/ttt_tests/moshi-finetune/moshi_ttt/hybrid_layer.py`
**Changes**: Pass position information through to TTT layer

---

## New Code Components

### Component 1: RotaryEmbedding Class

**Reference**: `ttt-lm-pytorch/ttt.py` lines 118-201

**Location**: Add to `/home/alufr/ttt_tests/moshi-finetune/moshi_ttt/ttt_layer.py` after imports

```python
class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) - EXACT copy from ttt-lm-pytorch.

    Key difference from standard RoPE: Only needs max_position_embeddings=mini_batch_size
    because TTT-LM uses position_ids % mini_batch_size.
    """
    def __init__(self, dim, max_position_embeddings=16, base=10000, device=None, scaling_factor=1.0):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Compute inverse frequencies: inv_freq[i] = 1 / (base^(2i/dim))
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        """Precompute cos/sin cache for all positions"""
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)
        t = t / self.scaling_factor

        # Compute frequencies: freqs[i] = position * inv_freq[i]
        freqs = torch.outer(t, self.inv_freq)

        # Different from paper, but matches ttt-lm-pytorch exactly
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, position_ids):
        """
        Args:
            x: Input tensor [B, H, NC, C, HD]
            position_ids: Position indices [B, seq_len] - will be used with modulo

        Returns:
            cos, sin: Cosine and sine values for RoPE [seq_len, head_dim]
        """
        # x: [bs, num_attention_heads, seq_len, head_size]
        if position_ids is None:
            # Default to sequential positions
            seq_len = x.shape[2] * x.shape[3]  # NC * C
            position_ids = torch.arange(seq_len, device=x.device, dtype=torch.long)
        else:
            # Flatten position_ids if needed
            if position_ids.dim() > 1:
                position_ids = position_ids.flatten()

        # if seq_len > self.max_seq_len_cached:
        #     self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        # Index into cached cos/sin using position_ids
        cos = self.cos_cached[position_ids].to(dtype=x.dtype)  # [seq_len, head_dim]
        sin = self.sin_cached[position_ids].to(dtype=x.dtype)  # [seq_len, head_dim]

        return cos, sin
```

**Key Implementation Notes**:
- `max_position_embeddings=mini_batch_size`: Only 16-64 positions needed!
- `inv_freq` buffer: Precomputed inverse frequencies
- `_set_cos_sin_cache()`: Precompute cos/sin for all positions
- `forward()`: Simply indexes into cache using `position_ids`

---

### Component 2: Helper Functions

**Reference**: `ttt-lm-pytorch/ttt.py` lines 204-270

**Location**: Add to `/home/alufr/ttt_tests/moshi-finetune/moshi_ttt/ttt_layer.py` after `RotaryEmbedding` class

```python
def rotate_half(x):
    """
    Rotates half the hidden dims of the input.

    For RoPE, we split features into pairs and rotate them.
    Example: [a, b, c, d] -> [-c, -d, a, b]
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """
    Apply Rotary Position Embedding to query and key tensors.

    Args:
        q: Query tensor [B, H, NC, C, HD]
        k: Key tensor [B, H, NC, C, HD]
        cos: Cosine values [seq_len, head_dim]
        sin: Sine values [seq_len, head_dim]
        position_ids: Not used (for API compatibility)
        unsqueeze_dim: Dimension to unsqueeze cos/sin for broadcasting

    Returns:
        q_embed, k_embed: Rotated query and key tensors

    Formula:
        RoPE(x) = x * cos + rotate_half(x) * sin
    """
    # Unsqueeze cos/sin for broadcasting: [seq_len, head_dim] -> [1, 1, seq_len, head_dim]
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # Apply rotation
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


def permute_qk(XQ, XK):
    """
    Permute Q and K dimensions for JAX compatibility.

    This matches ttt-lm-pytorch exactly and ensures compatibility
    with JAX-based reference implementations.

    Args:
        XQ: [B, H, NC, C, HD]
        XK: [B, H, NC, C, HD]

    Returns:
        XQ, XK with permuted dimensions
    """
    # From ttt-lm-pytorch line 234-235
    # Permute from [B, H, NC, C, HD] to format suitable for RoPE
    XQ = XQ.permute(0, 2, 3, 1, 4)  # [B, NC, C, H, HD]
    XK = XK.permute(0, 2, 3, 1, 4)  # [B, NC, C, H, HD]
    return XQ, XK


def undo_permute_qk(XQ, XK):
    """
    Undo the permutation applied by permute_qk().

    Args:
        XQ: [B, NC, C, H, HD]
        XK: [B, NC, C, H, HD]

    Returns:
        XQ, XK: [B, H, NC, C, HD]
    """
    # From ttt-lm-pytorch line 238-239
    XQ = XQ.permute(0, 3, 1, 2, 4)  # [B, H, NC, C, HD]
    XK = XK.permute(0, 3, 1, 2, 4)  # [B, H, NC, C, HD]
    return XQ, XK
```

**Key Implementation Notes**:
- `rotate_half()`: Core RoPE rotation operation
- `apply_rotary_pos_emb()`: Applies rotation to Q and K (NOT V!)
- `permute_qk()` / `undo_permute_qk()`: JAX compatibility (may not be strictly necessary for PyTorch, but matches reference exactly)

---

### Component 3: RoPE Initialization in TTTBase

**Reference**: `ttt-lm-pytorch/ttt.py` lines 666-672

**Location**: Modify `TTTBase.__init__()` in `/home/alufr/ttt_tests/moshi-finetune/moshi_ttt/ttt_layer.py`

**Current code** (line 220-234):
```python
def __init__(self, config: TTTConfig, layer_id: int = None):
    super().__init__()
    self.config = config
    self.layer_id = layer_id
    self.width = config.model_dim
    self.num_heads = config.num_heads
    self.head_dim = config.model_dim // config.num_heads
    self.mini_batch_size = config.mini_batch_size
    self.ttt_base_lr = config.ttt_base_lr
    self.scan_checkpoint_group_size = max(config.mini_batch_size // 4, 1)

    self._init_qkvo_proj()
    self._init_ttt_lr_gate()
    self._init_ttt_ln()
    self.post_norm = nn.LayerNorm(self.width, eps=1e-6)
```

**Add after line 229** (after `self.scan_checkpoint_group_size`):
```python
    # Initialize RoPE (CRITICAL: max_position_embeddings=mini_batch_size!)
    self.rope_theta = config.rope_theta
    self.rotary_emb = RotaryEmbedding(
        self.head_dim,
        max_position_embeddings=self.mini_batch_size,  # KEY: Only need mini_batch_size positions!
        base=self.rope_theta,
    )
```

**Key Implementation Notes**:
- `max_position_embeddings=self.mini_batch_size`: This is CRITICAL! Not sequence length!
- `base=self.rope_theta`: Typically 10000.0 (standard RoPE)
- RoPE is initialized per-layer, not shared across layers

---

### Component 4: RoPE Application in Forward Pass

**Reference**: `ttt-lm-pytorch/ttt.py` lines 869-874

**Location**: Modify `TTTBase.process_input()` in `/home/alufr/ttt_tests/moshi-finetune/moshi_ttt/ttt_layer.py`

**Current code** (lines 327-381) - relevant section:
```python
def process_input(self, hidden_states: torch.Tensor, seq_metadata: SequenceMetadata):
    """Process input following Video-DiT pattern"""
    B, L = hidden_states.shape[:2]
    mini_batch_size = self.mini_batch_size

    # Get Q/K/V projections
    XQ, XK, XV = self.get_qkv_projections(hidden_states)
    XQ = XQ.view(B, L, -1, self.head_dim)
    XK = XK.view(B, L, -1, self.head_dim)
    XV = XV.view(B, L, -1, self.head_dim)

    # L2 Norm (Video-DiT pattern)
    XQ = F.normalize(XQ, p=2, dim=-1)
    XK = F.normalize(XK, p=2, dim=-1)

    # Apply layer norm reconstruction target
    XV = self.ln_reconstruction_target(XV, XK)

    # ... rest of function (padding, reshaping, etc.)
```

**Modify after L2 normalization** (after line 340, before line 343):

```python
    # L2 Norm (Video-DiT pattern)
    XQ = F.normalize(XQ, p=2, dim=-1)
    XK = F.normalize(XK, p=2, dim=-1)

    # === ADD RoPE APPLICATION HERE ===
    # Generate position_ids from seq_metadata (or default to sequential)
    if hasattr(seq_metadata, 'position_ids') and seq_metadata.position_ids is not None:
        position_ids = seq_metadata.position_ids  # [B, L]
    else:
        # Default: sequential positions
        position_ids = torch.arange(L, device=hidden_states.device, dtype=torch.long).unsqueeze(0).expand(B, -1)

    # CRITICAL: Apply position modulo to keep positions bounded
    position_ids_bounded = position_ids % self.mini_batch_size

    # Reshape for RoPE: [B, L, H, HD] -> [B, H, L, HD]
    XQ_rope = XQ.permute(0, 2, 1, 3)  # [B, num_heads, L, head_dim]
    XK_rope = XK.permute(0, 2, 1, 3)  # [B, num_heads, L, head_dim]

    # Compute RoPE cos/sin (using bounded positions!)
    cos, sin = self.rotary_emb(XQ_rope, position_ids_bounded)

    # cos, sin are [L, head_dim], need to reshape for broadcasting
    # XQ_rope, XK_rope are [B, H, L, HD]
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, L, HD]
    sin = sin.unsqueeze(0).unsqueeze(0)  # [1, 1, L, HD]

    # Apply RoPE rotation
    XQ_rope = (XQ_rope * cos) + (rotate_half(XQ_rope) * sin)
    XK_rope = (XK_rope * cos) + (rotate_half(XK_rope) * sin)

    # Reshape back: [B, H, L, HD] -> [B, L, H, HD]
    XQ = XQ_rope.permute(0, 2, 1, 3)
    XK = XK_rope.permute(0, 2, 1, 3)
    # === END RoPE APPLICATION ===

    # Apply layer norm reconstruction target
    XV = self.ln_reconstruction_target(XV, XK)
```

**Alternative Implementation** (matching ttt-lm-pytorch more closely with permute_qk):

```python
    # L2 Norm (Video-DiT pattern)
    XQ = F.normalize(XQ, p=2, dim=-1)
    XK = F.normalize(XK, p=2, dim=-1)

    # === ADD RoPE APPLICATION HERE (ttt-lm-pytorch style) ===
    # After padding and reshaping to mini-batch format...
    # (Move this block to after line 363 where we have [B, H, NC, C, HD] format)
```

**BETTER LOCATION**: After reshaping to mini-batch format (after line 363):

**Current code** (lines 346-363):
```python
    # Convert to mini-batch format for TTT processing
    # Pad sequence to be divisible by mini_batch_size
    NC = (L + mini_batch_size - 1) // mini_batch_size
    padded_L = NC * mini_batch_size

    if padded_L > L:
        pad_len = padded_L - L
        XQ_pad = torch.zeros(B, pad_len, XQ.shape[2], self.head_dim, device=XQ.device, dtype=XQ.dtype)
        XK_pad = torch.zeros(B, pad_len, XK.shape[2], self.head_dim, device=XK.device, dtype=XK.dtype)
        XV_pad = torch.zeros(B, pad_len, XV.shape[2], self.head_dim, device=XV.device, dtype=XV.dtype)

        XQ = torch.cat([XQ, XQ_pad], dim=1)
        XK = torch.cat([XK, XK_pad], dim=1)
        XV = torch.cat([XV, XV_pad], dim=1)

    # Reshape to mini-batch format: [B, padded_L, H, HD] -> [B, H, NC, C, HD]
    XQ = XQ.view(B, NC, mini_batch_size, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
    XK = XK.view(B, NC, mini_batch_size, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
    XV = XV.view(B, NC, mini_batch_size, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
```

**Add immediately after line 363** (after reshaping to mini-batch format):

```python
    # === APPLY RoPE (ttt-lm-pytorch exact pattern) ===
    # Generate position_ids (default to sequential)
    if hasattr(seq_metadata, 'position_ids') and seq_metadata.position_ids is not None:
        position_ids = seq_metadata.position_ids[:, :padded_L]  # Use padded length
    else:
        position_ids = torch.arange(padded_L, device=XQ.device, dtype=torch.long).unsqueeze(0).expand(B, -1)

    # CRITICAL: Apply modulo to keep positions in [0, mini_batch_size)
    position_ids_bounded = position_ids % self.mini_batch_size

    # Compute cos/sin using bounded positions
    cos, sin = self.rotary_emb(XV, position_ids_bounded)

    # Reshape cos/sin for mini-batch format
    # cos, sin are [padded_L, head_dim]
    # Need [B, H, NC, C, HD] for broadcasting
    cos = cos.view(NC, mini_batch_size, self.head_dim).unsqueeze(0).unsqueeze(1)  # [1, 1, NC, C, HD]
    sin = sin.view(NC, mini_batch_size, self.head_dim).unsqueeze(0).unsqueeze(1)  # [1, 1, NC, C, HD]

    # Apply RoPE to XQ and XK (NOT XV!)
    XQ = (XQ * cos) + (rotate_half(XQ) * sin)
    XK = (XK * cos) + (rotate_half(XK) * sin)
    # === END RoPE APPLICATION ===
```

**Key Implementation Notes**:
- **Apply AFTER mini-batch reshaping**: Easier to match shapes
- **Use modulo**: `position_ids % self.mini_batch_size` is CRITICAL
- **Apply to Q and K only**: NOT to V!
- **Reshape cos/sin carefully**: Must broadcast correctly with `[B, H, NC, C, HD]`

---

### Component 5: Config Update

**Location**: `/home/alufr/ttt_tests/moshi-finetune/moshi_ttt/config.py`

**Current code** (line 27):
```python
    # For Video-DiT compatibility (not used in Moshi but needed for imports)
    rope_theta: float = 10000.0
```

**Update comment**:
```python
    # RoPE configuration
    rope_theta: float = 10000.0  # Base for RoPE frequency computation (standard value)
```

**No changes needed** - `rope_theta` already exists with correct default!

---

### Component 6: Position ID Generation

**Location**: `/home/alufr/ttt_tests/moshi-finetune/moshi_ttt/format_utils.py`

**Current SequenceMetadata class**:
```python
@dataclass
class SequenceMetadata:
    """Metadata for sequence processing (adapted from Video-DiT)"""
    batch_size: int
    sequence_length: int
    num_chunks: int
    chunk_size: int
```

**Add field**:
```python
@dataclass
class SequenceMetadata:
    """Metadata for sequence processing (adapted from Video-DiT)"""
    batch_size: int
    sequence_length: int
    num_chunks: int
    chunk_size: int
    position_ids: Optional[torch.Tensor] = None  # [B, seq_len] - position indices
```

---

### Component 7: Hybrid Layer Update

**Location**: `/home/alufr/ttt_tests/moshi-finetune/moshi_ttt/hybrid_layer.py`

**Need to ensure `position_ids` are passed through in `SequenceMetadata`**

**Current code** (where `seq_metadata` is created):
```python
seq_metadata = SequenceMetadata(
    batch_size=batch_size,
    sequence_length=seq_len,
    num_chunks=num_chunks,
    chunk_size=chunk_size,
)
```

**Update to**:
```python
# Generate position_ids for RoPE
# For streaming: maintain cumulative position across chunks
if not hasattr(self, '_cumulative_position'):
    self._cumulative_position = 0

position_ids = torch.arange(
    self._cumulative_position,
    self._cumulative_position + seq_len,
    device=hidden_states.device,
    dtype=torch.long
).unsqueeze(0).expand(batch_size, -1)

seq_metadata = SequenceMetadata(
    batch_size=batch_size,
    sequence_length=seq_len,
    num_chunks=num_chunks,
    chunk_size=chunk_size,
    position_ids=position_ids,
)

# Update cumulative position for next chunk (streaming)
if not self.training:
    self._cumulative_position += seq_len
```

**Key Implementation Notes**:
- **Cumulative positions**: For streaming inference, track position across chunks
- **Training vs inference**: In training, positions reset per sample; in inference, accumulate
- **Batch handling**: `position_ids` are `[B, seq_len]` to support different positions per batch item if needed

---

## Integration Steps

### Step 1: Add RoPE Components to ttt_layer.py

**File**: `/home/alufr/ttt_tests/moshi-finetune/moshi_ttt/ttt_layer.py`

1. **Add imports** (if needed):
   ```python
   from typing import Optional
   ```

2. **Add `RotaryEmbedding` class** after imports (before `ln_fwd` function)

3. **Add helper functions** after `RotaryEmbedding`:
   - `rotate_half()`
   - `apply_rotary_pos_emb()`
   - `permute_qk()`
   - `undo_permute_qk()`

4. **Update `TTTBase.__init__()`** to initialize RoPE (after line 229)

5. **Update `TTTBase.process_input()`** to apply RoPE (after line 363)

**Estimated changes**: ~200 lines added

---

### Step 2: Update Config

**File**: `/home/alufr/ttt_tests/moshi-finetune/moshi_ttt/config.py`

1. **Update comment** for `rope_theta` (line 27)

**Estimated changes**: 1 line modified

---

### Step 3: Update Format Utils

**File**: `/home/alufr/ttt_tests/moshi-finetune/moshi_ttt/format_utils.py`

1. **Add `position_ids` field** to `SequenceMetadata` dataclass

**Estimated changes**: 2 lines added

---

### Step 4: Update Hybrid Layer

**File**: `/home/alufr/ttt_tests/moshi-finetune/moshi_ttt/hybrid_layer.py`

1. **Find where `seq_metadata` is created**
2. **Add position_ids generation** before metadata creation
3. **Add cumulative position tracking** for streaming

**Estimated changes**: ~15 lines added

---

### Step 5: Testing

See [Testing Plan](#testing-plan) section below.

---

## Testing Plan

### Test 1: Unit Test - RoPE Components

**File**: Create `/home/alufr/ttt_tests/moshi-finetune/tests/test_rope_components.py`

```python
import torch
from moshi_ttt.ttt_layer import RotaryEmbedding, rotate_half, apply_rotary_pos_emb

def test_rotary_embedding_initialization():
    """Test RoPE initialization"""
    rope = RotaryEmbedding(dim=128, max_position_embeddings=64, base=10000)
    assert rope.dim == 128
    assert rope.max_position_embeddings == 64
    assert rope.inv_freq.shape == (64,)  # dim // 2

def test_rotary_embedding_forward():
    """Test RoPE forward pass"""
    rope = RotaryEmbedding(dim=128, max_position_embeddings=64, base=10000)
    x = torch.randn(2, 8, 10, 64, 128)  # [B, H, NC, C, HD]
    position_ids = torch.arange(0, 640).view(2, 320)  # [B, seq_len]

    cos, sin = rope(x, position_ids)
    assert cos.shape == (640, 128)  # [seq_len, head_dim]
    assert sin.shape == (640, 128)

    # Check bounded positions work
    position_ids_bounded = position_ids % 64
    cos_bounded, sin_bounded = rope(x, position_ids_bounded)
    assert cos_bounded.shape == (640, 128)

def test_rotate_half():
    """Test rotation operation"""
    x = torch.tensor([[1., 2., 3., 4.]])
    rotated = rotate_half(x)
    expected = torch.tensor([[-3., -4., 1., 2.]])
    assert torch.allclose(rotated, expected)

def test_apply_rotary_pos_emb():
    """Test RoPE application"""
    B, H, L, HD = 2, 8, 100, 128
    q = torch.randn(B, H, L, HD)
    k = torch.randn(B, H, L, HD)
    cos = torch.randn(L, HD)
    sin = torch.randn(L, HD)

    q_embed, k_embed = apply_rotary_pos_emb(q, k, cos, sin)
    assert q_embed.shape == q.shape
    assert k_embed.shape == k.shape

def test_position_modulo_invariance():
    """
    CRITICAL TEST: Verify that positions (0, 1, 2, ..., 63)
    produce same cos/sin as positions (64, 65, 66, ..., 127) when using modulo.
    """
    rope = RotaryEmbedding(dim=128, max_position_embeddings=64)
    x = torch.randn(1, 8, 1, 64, 128)

    # Positions 0-63
    pos_1 = torch.arange(0, 64).unsqueeze(0)
    cos_1, sin_1 = rope(x, pos_1)

    # Positions 64-127 with modulo
    pos_2 = torch.arange(64, 128).unsqueeze(0)
    pos_2_bounded = pos_2 % 64
    cos_2, sin_2 = rope(x, pos_2_bounded)

    # Should be identical because of modulo
    assert torch.allclose(cos_1, cos_2, atol=1e-6)
    assert torch.allclose(sin_1, sin_2, atol=1e-6)

if __name__ == "__main__":
    test_rotary_embedding_initialization()
    test_rotary_embedding_forward()
    test_rotate_half()
    test_apply_rotary_pos_emb()
    test_position_modulo_invariance()
    print("✅ All RoPE component tests passed!")
```

**Run**:
```bash
cd /home/alufr/ttt_tests/moshi-finetune
conda activate moshi_ttt_fixed
python tests/test_rope_components.py
```

---

### Test 2: Integration Test - TTT Layer with RoPE

**File**: Create `/home/alufr/ttt_tests/moshi-finetune/tests/test_ttt_with_rope.py`

```python
import torch
from moshi_ttt.ttt_layer import TTTMLP
from moshi_ttt.config import TTTConfig
from moshi_ttt.format_utils import SequenceMetadata

def test_ttt_forward_with_rope():
    """Test TTT forward pass with RoPE enabled"""
    config = TTTConfig(
        model_dim=1024,
        num_heads=8,
        mini_batch_size=64,
        rope_theta=10000.0,
    )

    ttt = TTTMLP(config, layer_id=0)
    ttt.eval()

    # Create input
    B, L = 2, 128
    hidden_states = torch.randn(B, L, config.model_dim)

    # Create position_ids
    position_ids = torch.arange(L).unsqueeze(0).expand(B, -1)

    seq_metadata = SequenceMetadata(
        batch_size=B,
        sequence_length=L,
        num_chunks=2,
        chunk_size=64,
        position_ids=position_ids,
    )

    # Forward pass
    with torch.no_grad():
        output = ttt(hidden_states, seq_metadata)

    assert output.shape == (B, L, config.model_dim)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()
    print("✅ TTT forward with RoPE: output shape correct, no NaN/Inf")

def test_long_sequence_with_modulo():
    """
    CRITICAL TEST: Verify position modulo prevents degradation on long sequences
    """
    config = TTTConfig(
        model_dim=512,
        num_heads=8,
        mini_batch_size=64,
        rope_theta=10000.0,
    )

    ttt = TTTMLP(config, layer_id=0)
    ttt.eval()

    # Test with very long sequence (positions up to 5000)
    B = 1
    L = 5000
    hidden_states = torch.randn(B, L, config.model_dim)

    position_ids = torch.arange(L).unsqueeze(0)

    seq_metadata = SequenceMetadata(
        batch_size=B,
        sequence_length=L,
        num_chunks=L // 64,
        chunk_size=64,
        position_ids=position_ids,
    )

    with torch.no_grad():
        output = ttt(hidden_states, seq_metadata)

    # Check that output is reasonable (no NaN, not degenerate)
    assert output.shape == (B, L, config.model_dim)
    assert not torch.isnan(output).any()
    assert output.std() > 0.01  # Not collapsed to zero
    print(f"✅ Long sequence test: L={L}, positions 0-{L-1}, output healthy")

def test_streaming_cumulative_positions():
    """Test that cumulative positions work for streaming inference"""
    config = TTTConfig(
        model_dim=512,
        num_heads=8,
        mini_batch_size=64,
    )

    ttt = TTTMLP(config, layer_id=0)
    ttt.eval()

    B = 1
    chunk_size = 128

    # Simulate 3 streaming chunks
    for i in range(3):
        hidden_states = torch.randn(B, chunk_size, config.model_dim)

        # Cumulative position_ids
        start_pos = i * chunk_size
        position_ids = torch.arange(start_pos, start_pos + chunk_size).unsqueeze(0)

        seq_metadata = SequenceMetadata(
            batch_size=B,
            sequence_length=chunk_size,
            num_chunks=2,
            chunk_size=64,
            position_ids=position_ids,
        )

        with torch.no_grad():
            output = ttt(hidden_states, seq_metadata)

        assert output.shape == (B, chunk_size, config.model_dim)
        print(f"✅ Streaming chunk {i}: positions {start_pos}-{start_pos+chunk_size-1}")

if __name__ == "__main__":
    test_ttt_forward_with_rope()
    test_long_sequence_with_modulo()
    test_streaming_cumulative_positions()
    print("\n✅ All TTT+RoPE integration tests passed!")
```

**Run**:
```bash
cd /home/alufr/ttt_tests/moshi-finetune
conda activate moshi_ttt_fixed
python tests/test_ttt_with_rope.py
```

---

### Test 3: End-to-End Test - Inference with RoPE

**File**: Modify existing inference script or create test

**Test Procedure**:
1. **Load LoRA checkpoint** (previously trained)
2. **Run inference on short audio** (10 sec) - should work same as before
3. **Run inference on long audio** (60 sec) - should NOT degrade now!
4. **Compare outputs**: With RoPE modulo, long audio should maintain quality

**Script**:
```bash
cd /home/alufr/ttt_tests/moshi-finetune

# Run inference on long audio (this failed before)
python inference/run_inference_with_lora.py \
    --checkpoint /sise/eliyanac-group/ron_al/dailytalk_finetune_from_librilight2255lora122/checkpoints/checkpoint_025000/consolidated/ \
    --input /path/to/long_audio_60sec.wav \
    --output output_lora_with_rope.wav \
    --device cuda

# Check that output is reasonable (listen to it)
```

**Expected Result**: Audio should NOT degrade over time (no gibberish at the end)

---

### Test 4: Comparison Test - With/Without RoPE

**Create a toggle to disable RoPE** for testing:

**In `TTTBase.process_input()`**:
```python
# Add config flag
use_rope = getattr(self.config, 'use_rope', True)

if use_rope:
    # Apply RoPE (new code)
    position_ids_bounded = position_ids % self.mini_batch_size
    cos, sin = self.rotary_emb(XV, position_ids_bounded)
    # ... apply rotation
else:
    # Skip RoPE (old behavior)
    pass
```

**Run both**:
```bash
# Without RoPE (should degrade on long context)
python test_inference.py --checkpoint ... --input long.wav --output without_rope.wav --use-rope False

# With RoPE (should maintain quality)
python test_inference.py --checkpoint ... --input long.wav --output with_rope.wav --use-rope True
```

**Compare**: Listen to both outputs, measure perplexity over time

---

## Rollback Plan

**If RoPE integration causes issues:**

### Immediate Rollback
1. **Git rollback** (if using version control):
   ```bash
   git checkout <previous-commit>
   ```

2. **Manual rollback**:
   - Remove `RotaryEmbedding` class from `ttt_layer.py`
   - Remove helper functions (`rotate_half`, `apply_rotary_pos_emb`, etc.)
   - Remove RoPE initialization in `TTTBase.__init__()`
   - Remove RoPE application in `TTTBase.process_input()`
   - Remove `position_ids` field from `SequenceMetadata`

### Fallback: Config Flag
**Best practice**: Add `use_rope` config flag from the start:

**In `TTTConfig`**:
```python
@dataclass
class TTTConfig:
    # ... existing fields
    use_rope: bool = True  # Enable/disable RoPE
```

**In `TTTBase.process_input()`**:
```python
if self.config.use_rope:
    # Apply RoPE
    pass
else:
    # Skip RoPE (backward compatible)
    pass
```

**Disable RoPE in config**:
```yaml
ttt:
  use_rope: false
```

---

## Key Differences from Video-DiT

**Video-DiT does NOT use RoPE** - this is a TTT-LM specific feature.

**Why?**
- Video-DiT processes 2D spatial+temporal patches with absolute positions
- TTT-LM processes 1D token sequences that can be very long
- TTT-LM needs position encoding to distinguish tokens within mini-batches

**Our implementation**:
- Follows TTT-LM pattern exactly
- Uses position modulo trick (`position_ids % mini_batch_size`)
- Applies after L2 norm, before TTT processing
- Only applies to Q and K, not V

---

## Expected Benefits

### 1. Long-Context Performance
- **Before**: LoRA degrades on audio >30 seconds (positions >125 extrapolation)
- **After**: Maintains quality on 60+ second audio (positions always in [0, 63])

### 2. Training Stability
- RoPE provides positional inductive bias
- Helps TTT distinguish tokens within mini-batches
- May improve convergence

### 3. Consistent with TTT-LM
- Matches reference implementation exactly
- Enables direct comparison with TTT-LM results
- Follows proven architecture pattern

---

## Critical Implementation Notes

### 🚨 CRITICAL: Position Modulo
```python
# CORRECT (ttt-lm-pytorch)
position_ids_bounded = position_ids % self.mini_batch_size
cos, sin = self.rotary_emb(x, position_ids_bounded)

# WRONG (will cause extrapolation issues!)
cos, sin = self.rotary_emb(x, position_ids)  # ❌ Unbounded positions
```

### 🚨 CRITICAL: max_position_embeddings
```python
# CORRECT
self.rotary_emb = RotaryEmbedding(
    self.head_dim,
    max_position_embeddings=self.mini_batch_size,  # ✅ Only 64 positions
)

# WRONG
self.rotary_emb = RotaryEmbedding(
    self.head_dim,
    max_position_embeddings=8192,  # ❌ Wastes memory, doesn't match usage
)
```

### 🚨 CRITICAL: Apply to Q and K Only
```python
# CORRECT
XQ = (XQ * cos) + (rotate_half(XQ) * sin)
XK = (XK * cos) + (rotate_half(XK) * sin)
# XV unchanged

# WRONG
XV = (XV * cos) + (rotate_half(XV) * sin)  # ❌ Don't apply to V!
```

### 🚨 CRITICAL: Apply After L2 Norm
```python
# CORRECT order
XQ = F.normalize(XQ, p=2, dim=-1)  # L2 norm first
XK = F.normalize(XK, p=2, dim=-1)
# Then apply RoPE
XQ = (XQ * cos) + (rotate_half(XQ) * sin)
XK = (XK * cos) + (rotate_half(XK) * sin)

# WRONG order
XQ = (XQ * cos) + (rotate_half(XQ) * sin)  # RoPE first
XQ = F.normalize(XQ, p=2, dim=-1)  # ❌ L2 norm would destroy RoPE!
```

---

## Summary Checklist

- [ ] Add `RotaryEmbedding` class to `ttt_layer.py`
- [ ] Add helper functions: `rotate_half`, `apply_rotary_pos_emb`, `permute_qk`, `undo_permute_qk`
- [ ] Initialize RoPE in `TTTBase.__init__()` with `max_position_embeddings=mini_batch_size`
- [ ] Apply RoPE in `TTTBase.process_input()` after L2 norm
- [ ] Use position modulo: `position_ids % self.mini_batch_size`
- [ ] Add `position_ids` field to `SequenceMetadata`
- [ ] Update `hybrid_layer.py` to generate and pass `position_ids`
- [ ] Implement cumulative position tracking for streaming
- [ ] Add `use_rope` config flag for easy enable/disable
- [ ] Write unit tests for RoPE components
- [ ] Write integration tests for TTT+RoPE
- [ ] Test on long-context audio (60+ seconds)
- [ ] Compare with/without RoPE to verify improvement
- [ ] Document all changes in code comments

---

## References

1. **ttt-lm-pytorch**: `/home/alufr/ttt_tests/ttt-lm-pytorch/ttt.py`
   - Lines 118-201: `RotaryEmbedding` class
   - Lines 204-270: Helper functions
   - Lines 666-672: RoPE initialization
   - Lines 869-874: RoPE application with modulo trick

2. **Current implementation**: `/home/alufr/ttt_tests/moshi-finetune/moshi_ttt/ttt_layer.py`
   - Video-DiT based, no RoPE currently

3. **User observation**: `/home/alufr/ttt_tests/moshi-finetune/logs/inference/moshi_lora.7807993.log`
   - LoRA degrades on long audio without RoPE

---

**END OF SURGICAL INTEGRATION PLAN**

This plan provides complete, surgical implementation details for adding RoPE to Moshi-TTT exactly as implemented in ttt-lm-pytorch, with special emphasis on the critical position modulo trick that prevents long-context extrapolation issues.
