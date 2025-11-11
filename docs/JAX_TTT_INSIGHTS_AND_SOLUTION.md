# JAX TTT-LM Analysis: The Missing Architectural Insight

**Date**: 2025-11-10
**Source**: `ttt-lm-jax` official implementation

---

## 🎯 **THE CRITICAL DISCOVERY**

The JAX TTT-LM implementation reveals **the fundamental architectural mistake** in the current Moshi-TTT implementation:

**Current Moshi-TTT**: `Input → Attention (KV Cache) → TTT → Output` ❌

**JAX TTT-LM**: `Input → TTT (NO KV Cache) → Output` ✅

**TTT is meant to REPLACE attention, not augment it!**

---

## Key Architectural Differences

### 1. **TTT Replaces Attention Entirely**

**JAX Implementation** (`model.py:698-714`):

```python
def __call__(self, hidden_states, ...):
    # Pre-normalize
    hidden_states_pre_normed = self.seq_norm(hidden_states)

    # EITHER attention OR TTT (not both!)
    if self.config.seq_modeling_block == "self_attention":
        seq_modeling_outputs = self.seq_modeling_block(
            hidden_states_pre_normed, attention_mask, position_ids, ...
        )
    else:  # TTT layers
        seq_modeling_outputs = self.seq_modeling_block(
            hidden_states_pre_normed, input_ids, position_ids, ...
        )

    # Add residual
    hidden_states = hidden_states + seq_modeling_output
```

**Key Insight**:
- TTT layers receive `hidden_states_pre_normed` (the **raw input**)
- NOT the output of attention
- TTT **replaces** the attention mechanism in those layers

**Moshi-TTT Implementation** (`hybrid_layer.py:257-265`):

```python
def _forward_impl(self, x, cross_attention_src):
    # Step 1: Attention processing
    attn_output = self._attn_forward(x, cross_attention_src)  # ← Uses KV cache

    # Step 2: TTT processing
    ttt_output = self._ttt_forward(attn_output)  # ← Receives corrupted input!

    return ttt_output
```

**Problem**: TTT receives attention output, which is already corrupted by the Ring KV Cache.

---

### 2. **No KV Cache for TTT Layers**

**JAX**: TTT layers don't use KV caching at all. The `__call__` signature:

```python
# ttt_layer.py:336-353
def __call__(
    self,
    hidden_states,
    input_ids=None,
    position_ids=None,
    deterministic: bool = True,
    output_ttt_stats: bool = False,
    ttt_lr_mult=1.0,
):
    # No attention_mask, no init_cache, no KV cache!
    XQ, XK, XV, eta, precompute_stats = self.get_ttt_inputs(hidden_states, position_ids)
    Z, ttt_stats = self.ttt(XQ, XK, XV, eta, input_ids)
    # ...
```

**vs. Attention layers** (`model.py:509-613`):

```python
def __call__(
    self,
    hidden_states,
    attention_mask,
    position_ids,
    deterministic: bool = True,
    init_cache: bool = False,  # ← KV cache support
    ...
):
    # Uses _concatenate_to_cache for KV caching
    if self.has_variable("cache", "cached_key") or init_cache:
        xk, xv, attention_mask = self._concatenate_to_cache(xk, xv, xq, attention_mask)
```

**Insight**: TTT's weight matrices (W₁, b₁, W₂, b₂) **ARE the memory**. No separate KV cache needed!

---

### 3. **Convolutional Layers for Q/K Computation**

**JAX TTTLinear** (`ttt_layer.py:444-498`):

```python
def setup_qkvo(self):
    self.wq = nn.Dense(...)

    # Causal convolutions for Q and K
    self.conv_q = conv_module(
        self.config.hidden_size,
        (self.config.conv_width,),  # Typically 4
        padding="CAUSAL",
        feature_group_count=self.config.hidden_size,
        ...
    )
    self.conv_k = conv_module(
        self.config.hidden_size,
        (self.config.conv_width,),
        padding="CAUSAL",
        ...
    )

    self.wv = nn.Dense(...)
    self.wo = nn.Dense(...)

def get_qkv_projections(self, batch):
    xqk, XV = self.wq(batch), self.wv(batch)
    XQ = self.conv_q(xqk)  # ← Causal conv for local context
    XK = self.conv_k(xqk)  # ← Causal conv for local context
    return XQ, XK, XV
```

**Why Convolutions?**
- Causal convolution (width=4) gives each token access to its 4 previous neighbors
- Provides **local context** without needing attention
- Much cheaper than full attention
- Works well with TTT's long-range compression

**Moshi-TTT**: Uses standard linear projections, relies on attention for local context (which breaks with Ring KV Cache).

---

### 4. **Position Modulo for RoPE**

**JAX** (`ttt_layer.py:260`):

```python
freqs_cis = jnp.take(self.freqs_cis, position_ids % self.mini_batch_size, axis=0)
XQ, XK = apply_rotary_emb(XQ, XK, freqs_cis=freqs_cis, dtype=self.dtype)
```

**Moshi-TTT** (`ttt_layer.py:508-510`):

```python
# CRITICAL: Apply position modulo to keep positions in [0, mini_batch_size)
position_ids_bounded = position_ids % self.mini_batch_size
```

**Status**: ✅ Already implemented correctly in Moshi-TTT

---

### 5. **Model Configuration: Mixed Attention/TTT**

**JAX Config** (`model.py:57-77, 94-114`):

```python
"125m": {  # Pure attention baseline
    "seq_modeling_block": "self_attention",
    ...
}

"125m-TTT": {  # Pure TTT model
    "seq_modeling_block": "ttt_linear",
    "mini_batch_size": 16,
    ...
}
```

**Typical Usage**:
- Some layers use `"self_attention"`
- Other layers use `"ttt_linear"` or `"ttt_mlp"`
- **Each layer is EITHER attention OR TTT**, not both!

---

## The Root Cause (Clarified)

The current Moshi-TTT implementation tried to **add TTT after attention** (Video-DiT style):

```
┌─────────────┐
│   Attention │ ← Uses Ring KV Cache (loses old tokens)
│  (corrupted)│
└──────┬──────┘
       │ (corrupted features)
       ▼
┌─────────────┐
│     TTT     │ ← Tries to compress what it never saw
└─────────────┘
```

But JAX TTT-LM uses **TTT as a replacement**:

```
┌─────────────┐
│     TTT     │ ← Receives raw input
│  (W₁,b₁,W₂,b₂)│ ← Weights are the memory
│  + Conv Q/K │ ← Local context from convolution
└─────────────┘
```

---

## 💡 **THE SOLUTION**

### Option A: Pure Replacement (Recommended)

**Replace attention entirely in some layers**:

```python
# Layer 0-15: Keep Moshi's attention
# Layer 16-27: Replace with TTT

class MoshiTTTLayer(nn.Module):
    def __init__(self, layer_id):
        if layer_id < 16:
            # Standard Moshi attention layer
            self.seq_block = MoshiAttentionBlock(...)
        else:
            # Pure TTT layer (NO attention!)
            self.seq_block = TTTBlock(...)  # Receives raw input

    def forward(self, x):
        # Pre-norm
        x_norm = self.layer_norm(x)

        # EITHER attention OR TTT
        seq_output = self.seq_block(x_norm)

        # Residual
        return x + seq_output
```

**Advantages**:
- ✅ TTT receives raw input (no information loss)
- ✅ No KV cache conflicts
- ✅ TTT weights naturally accumulate history
- ✅ Matches JAX architecture exactly

**What needs to change**:
1. Remove the hybrid attention+TTT architecture
2. Make TTT layers standalone (no attention inside)
3. Add causal convolutions for local context
4. Let TTT weights be the memory mechanism

---

### Option B: Hybrid with Separate Streams (Advanced)

**Keep both but make them independent**:

```python
class HybridLayer(nn.Module):
    def forward(self, x):
        # Path 1: Attention for local context (small window)
        attn_out = self.attention(x, kv_cache_size=100)  # Small cache

        # Path 2: TTT for long-range memory (independent!)
        ttt_out = self.ttt(x)  # Gets raw input, not attn output!

        # Combine
        combined = self.gate_attn(attn_out) + self.gate_ttt(ttt_out)
        return x + combined
```

**Advantages**:
- ✅ TTT gets raw input (no corruption)
- ✅ Attention handles local patterns
- ✅ TTT handles long-range dependencies

**Disadvantages**:
- More complex
- Higher compute cost
- Needs careful tuning

---

## Detailed Implementation Plan

### Step 1: Create Pure TTT Layers

```python
# moshi_ttt/pure_ttt_layer.py

class PureTTTLayer(nn.Module):
    """Pure TTT layer that REPLACES attention (JAX-style)"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Q/K/V projections with causal convolution
        self.wq = nn.Linear(config.model_dim, config.model_dim)
        self.conv_q = nn.Conv1d(
            config.model_dim, config.model_dim,
            kernel_size=4, padding='causal',
            groups=config.model_dim  # Depthwise
        )
        self.conv_k = nn.Conv1d(
            config.model_dim, config.model_dim,
            kernel_size=4, padding='causal',
            groups=config.model_dim
        )
        self.wv = nn.Linear(config.model_dim, config.model_dim)

        # TTT mechanism
        self.ttt = TTTMLP(config)  # Your existing TTT implementation

        # Output projection
        self.wo = nn.Linear(config.model_dim, config.model_dim)
        self.post_norm = nn.LayerNorm(config.model_dim)

    def forward(self, x):
        """
        Args:
            x: [B, L, D] - RAW input (not attention output!)
        """
        B, L, D = x.shape

        # Project to Q/K/V
        xqk = self.wq(x)  # [B, L, D]

        # Apply causal convolutions for local context
        # (gives each token access to previous 3 tokens)
        xqk_t = xqk.transpose(1, 2)  # [B, D, L]
        XQ = self.conv_q(xqk_t).transpose(1, 2)  # [B, L, D]
        XK = self.conv_k(xqk_t).transpose(1, 2)  # [B, L, D]
        XV = self.wv(x)  # [B, L, D]

        # TTT processing (your existing implementation)
        # TTT weights accumulate history across entire sequence
        ttt_output = self.ttt(XQ, XK, XV)

        # Output projection
        output = self.wo(ttt_output)
        output = self.post_norm(output)

        return output
```

### Step 2: Replace Moshi Layers

```python
# finetune/ttt_integration.py (modified)

def apply_ttt_to_model(model, ttt_args):
    """Apply pure TTT layers to replace attention in specified layers"""

    layer_indices = parse_layer_specification(ttt_args.layers, total_layers)

    for layer_idx in layer_indices:
        original_layer = model.transformer.layers[layer_idx]

        # Create PURE TTT layer (no attention inside!)
        pure_ttt_layer = PureTTTStreamingTransformerLayer(
            original_layer,  # For FFN and norms
            ttt_config,
            layer_idx
        )

        # Replace
        model.transformer.layers[layer_idx] = pure_ttt_layer
```

### Step 3: Streaming-Compatible Architecture

```python
class PureTTTStreamingTransformerLayer(StreamingModule):
    def __init__(self, original_layer, ttt_config, layer_id):
        super().__init__()
        self.layer_id = layer_id

        # Use original's FFN and norms
        self.ffn = original_layer.ffn
        self.norm1 = original_layer.norm1
        self.norm2 = original_layer.norm2

        # REPLACE attention with pure TTT
        self.seq_block = PureTTTLayer(ttt_config)

    def forward(self, x):
        # Self-attention block → Pure TTT block
        x = x + self.seq_block(self.norm1(x))

        # FFN block (unchanged)
        x = x + self.ffn(self.norm2(x))

        return x
```

---

## Why This Solves the Problem

### Current Problem:
```
Position 3750:
├─ Attention KV Cache: [750-3750] (tokens 0-749 LOST)
├─ Attention output: Missing tokens 0-749
├─ TTT receives: Corrupted features
└─ Result: Can't remember what it never saw ❌
```

### With Pure TTT Replacement:
```
Position 3750:
├─ TTT receives: RAW input x₃₇₅₀ ✅
├─ TTT weights: W₃₇₅₀ (accumulated from all 0-3749) ✅
├─ TTT computes: x₃₇₅₀ @ W₃₇₅₀ ✅
├─ Convolution: Provides local context (last 4 tokens) ✅
└─ Result: Full history available! ✅
```

**Key Points**:
1. TTT gets **raw input** at every position (no information loss)
2. TTT **weights** (W₁, b₁, W₂, b₂) contain **compressed history**
3. Weight updates: `W₃₇₅₀ = W₃₇₄₉ - η∇L` accumulate information
4. No Ring KV Cache involved at all!

---

## Validation Against JAX Code

### ✅ Confirmed Patterns:

1. **Block-level replacement** (`model.py:623-639`):
   - JAX uses `if seq_modeling_block == "self_attention"` vs `"ttt_linear"`
   - Each block is ONE type, not both

2. **Direct input to TTT** (`model.py:696-697, 709-711`):
   - `hidden_states_pre_normed = self.seq_norm(hidden_states)`
   - `seq_modeling_outputs = self.seq_modeling_block(hidden_states_pre_normed, ...)`
   - TTT receives normalized hidden states, NOT attention output

3. **Causal convolutions** (`ttt_layer.py:459-476, 630-647`):
   - Both TTTLinear and TTTMLP use `nn.Conv` with `padding="CAUSAL"`
   - Provides local context without attention

4. **No KV cache for TTT** (`ttt_layer.py:336-353`):
   - TTT `__call__` has no `init_cache` or `attention_mask` parameters
   - No cache management code

---

## Next Steps

### Immediate (Prototype):
1. ✅ Implement `PureTTTLayer` with causal convolutions
2. ✅ Test on single layer replacement
3. ✅ Verify TTT receives raw input (not attention output)

### Short-term (Validation):
4. ⏳ Train on short sequences (30s) with pure TTT layers
5. ⏳ Test on longer sequences (5-10 min)
6. ⏳ Compare perplexity: baseline vs pure TTT

### Long-term (Production):
7. ⏳ Mixed architecture: Attention (layers 0-15) + TTT (layers 16-27)
8. ⏳ Full training run with progressive sequence lengths
9. ⏳ Benchmark on 30+ minute conversations

---

## Conclusion

**The fundamental mistake**: Adding TTT after attention, where it receives corrupted input from the Ring KV Cache.

**The JAX solution**: Replace attention entirely with TTT in select layers, so TTT receives raw input and its weights become the memory mechanism.

**Why it works**:
- No information loss (TTT sees raw input)
- No KV cache conflicts (TTT doesn't use cache)
- Natural long-term memory (TTT weights accumulate)
- Local context from convolutions (replaces attention's role)

**Implementation complexity**: Medium
- Need to create pure TTT layers
- Add causal convolutions
- Integrate into Moshi's streaming architecture
- But architecture is simpler (no hybrid attention+TTT)

**Expected outcome**: TTT can now actually extend context because it processes complete information at every step, with its weight matrices naturally accumulating history over arbitrarily long sequences.

---

**Analysis Date**: 2025-11-10
**Source Code**: `ttt-lm-jax` (official JAX implementation)
**Key Files Analyzed**:
- `ttt-lm-jax/ttt/models/model.py`
- `ttt-lm-jax/ttt/models/ttt_layer.py`
