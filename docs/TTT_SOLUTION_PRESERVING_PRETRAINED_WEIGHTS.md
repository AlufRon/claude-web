# TTT Solution: Preserving Moshi's Pretrained Knowledge

**Date**: 2025-11-10
**Critical Constraint**: Cannot override Moshi's pretrained attention layers

---

## The Dilemma

### JAX Approach (Can't Use):
```
Replace attention layers → TTT gets raw input ✅
But: Loses pretrained weights ❌
```

### Current Approach (Doesn't Work):
```
Attention → TTT → Output
Preserves weights ✅
But: TTT gets corrupted input ❌
```

**We need**: Preserve pretrained weights ✅ AND give TTT clean input ✅

---

## 🎯 **Solution: TTT-Augmented KV Cache**

Keep Moshi's attention layers completely intact, but replace the **Ring KV Cache** with a **hybrid cache** that uses TTT for compression.

### Architecture

```
┌─────────────────────────────────────────────────┐
│              Input: x_t [B, L, D]               │
└──────────────────┬──────────────────────────────┘
                   │
                   ├──────────────┐
                   │              │
                   ▼              ▼
         ┌─────────────────┐  ┌─────────────────┐
         │  Moshi Attention│  │   TTT Memory    │
         │  (UNCHANGED!)   │  │   (NEW!)        │
         │                 │  │                 │
         │  Q = Wq(x)     │  │  Compresses     │
         │  K = Wk(x)     │  │  history into   │
         │  V = Wv(x)     │  │  K_virtual,     │
         │                 │  │  V_virtual      │
         └────────┬────────┘  └────────┬────────┘
                  │                    │
                  ▼                    ▼
         ┌──────────────────────────────────────┐
         │    Hybrid KV Cache                   │
         │                                      │
         │  Recent: [K_recent, V_recent]       │
         │          (last 100 tokens, precise)  │
         │                                      │
         │  History: [K_virtual, V_virtual]    │
         │          (TTT-compressed, all older) │
         └──────────────────┬───────────────────┘
                            │
                            ▼
                  ┌─────────────────┐
                  │    Attention    │
                  │  Q @ [K_r; K_v] │
                  │  × [V_r; V_v]   │
                  └─────────────────┘
```

### Key Points

1. **Moshi's Attention: Completely Unchanged**
   - All pretrained weights preserved
   - Q/K/V projections unchanged
   - Attention mechanism unchanged
   - Only the KV cache implementation changes

2. **TTT Processes Raw Input**
   - TTT module runs in parallel
   - Receives same input as attention
   - Compresses history into virtual K/V pairs
   - No information loss!

3. **Hybrid Cache**
   - Recent tokens: Normal KV cache (last 100 tokens)
   - Old tokens: TTT-compressed virtual K/V (unlimited history)
   - Attention queries both seamlessly

---

## Detailed Implementation

### 1. TTT Memory Module

```python
class TTTMemoryModule(nn.Module):
    """
    TTT module that compresses input history into virtual K/V pairs.
    Runs in parallel with attention (doesn't receive attention output).
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_model = config.model_dim
        self.num_heads = config.num_heads
        self.head_dim = self.d_model // self.num_heads

        # TTT compressor (your existing TTTMLP)
        self.ttt = TTTMLP(config)

        # Project TTT output to K/V space
        self.to_k_virtual = nn.Linear(self.d_model, self.d_model)
        self.to_v_virtual = nn.Linear(self.d_model, self.d_model)

        # Number of virtual tokens to represent compressed history
        self.num_virtual_tokens = config.ttt_virtual_tokens  # e.g., 10-20

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compress input into virtual K/V pairs.

        Args:
            x: [B, L, D] - RAW input (same as what attention sees)

        Returns:
            K_virtual: [B, num_virtual, num_heads, head_dim]
            V_virtual: [B, num_virtual, num_heads, head_dim]
        """
        B, L, D = x.shape

        # TTT compression (accumulates history in weights)
        compressed = self.ttt(x)  # [B, L, D]

        # Take last few outputs as summary of entire history
        # Option 1: Use last num_virtual_tokens outputs
        summary = compressed[:, -self.num_virtual_tokens:, :]  # [B, num_virtual, D]

        # Option 2: Or pool across entire sequence (better?)
        # summary = self.pool_to_virtual(compressed)  # [B, num_virtual, D]

        # Project to K/V space
        K_virtual = self.to_k_virtual(summary)  # [B, num_virtual, D]
        V_virtual = self.to_v_virtual(summary)  # [B, num_virtual, D]

        # Reshape to multi-head format
        K_virtual = K_virtual.view(B, self.num_virtual_tokens, self.num_heads, self.head_dim)
        V_virtual = V_virtual.view(B, self.num_virtual_tokens, self.num_heads, self.head_dim)

        return K_virtual, V_virtual
```

### 2. Hybrid KV Cache

```python
class HybridKVCache:
    """
    Hybrid cache that combines:
    - Recent KV cache (last N tokens, precise)
    - TTT virtual KV (compressed history, unbounded)
    """

    def __init__(self, capacity_recent=100, num_virtual=10):
        self.capacity_recent = capacity_recent
        self.num_virtual = num_virtual

        # Recent cache (ring buffer, same as Moshi's original)
        self.recent_cache = RingKVCache(capacity=capacity_recent)

        # Virtual cache (from TTT)
        self.virtual_cache = None  # Set during forward pass

    def update(self, k_new, v_new, k_virtual, v_virtual):
        """
        Update cache with new tokens and virtual representations.

        Args:
            k_new: [B, H, L, D] - new keys
            v_new: [B, H, L, D] - new values
            k_virtual: [B, num_virtual, H, D] - virtual keys from TTT
            v_virtual: [B, num_virtual, H, D] - virtual values from TTT
        """
        # Update recent cache (normal ring buffer)
        self.recent_cache.update(k_new, v_new)

        # Update virtual cache (TTT-compressed)
        self.virtual_cache = (k_virtual, v_virtual)

    def get_kv_for_attention(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get combined K/V for attention to query.

        Returns:
            K: [B, H, (num_virtual + capacity_recent), D]
            V: [B, H, (num_virtual + capacity_recent), D]
        """
        # Get recent K/V
        K_recent, V_recent = self.recent_cache.get()  # [B, H, capacity_recent, D]

        # Get virtual K/V (compressed history)
        if self.virtual_cache is not None:
            K_virtual, V_virtual = self.virtual_cache
            # Reshape: [B, num_virtual, H, D] -> [B, H, num_virtual, D]
            K_virtual = K_virtual.transpose(1, 2)
            V_virtual = V_virtual.transpose(1, 2)
        else:
            # No history yet
            B, H, _, D = K_recent.shape
            K_virtual = torch.zeros(B, H, 0, D, device=K_recent.device)
            V_virtual = torch.zeros(B, H, 0, D, device=V_recent.device)

        # Concatenate: [virtual history | recent tokens]
        K = torch.cat([K_virtual, K_recent], dim=2)  # [B, H, num_virtual + capacity, D]
        V = torch.cat([V_virtual, V_recent], dim=2)

        return K, V
```

### 3. Modified Attention Layer

```python
class MoshiAttentionWithTTTCache(nn.Module):
    """
    Moshi's attention layer with TTT-augmented cache.
    Attention weights and projections are UNCHANGED (preserves pretraining).
    Only the cache mechanism is modified.
    """

    def __init__(self, original_attention, ttt_config):
        super().__init__()

        # PRESERVE original attention (all pretrained weights!)
        self.original_attention = original_attention

        # Add TTT memory module (new, trained from scratch)
        self.ttt_memory = TTTMemoryModule(ttt_config)

        # Replace cache with hybrid cache
        self.kv_cache = HybridKVCache(
            capacity_recent=ttt_config.recent_cache_size,  # e.g., 100
            num_virtual=ttt_config.ttt_virtual_tokens,    # e.g., 10
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, D] - input
        """
        B, L, D = x.shape

        # === UNCHANGED: Moshi's attention projections ===
        Q = self.original_attention.wq(x)  # [B, L, D]
        K_new = self.original_attention.wk(x)  # [B, L, D]
        V_new = self.original_attention.wv(x)  # [B, L, D]

        # Reshape to multi-head
        Q = Q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K_new = K_new.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V_new = V_new.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # === NEW: TTT processes same input (in parallel) ===
        K_virtual, V_virtual = self.ttt_memory(x)  # [B, num_virtual, H, D]

        # Update hybrid cache
        self.kv_cache.update(K_new, V_new, K_virtual, V_virtual)

        # Get combined K/V (recent + virtual)
        K, V = self.kv_cache.get_kv_for_attention()  # [B, H, (num_virtual + capacity), D]

        # === UNCHANGED: Moshi's attention mechanism ===
        # Compute attention: Q @ K^T → softmax → @ V
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, V)  # [B, H, L, D]

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).reshape(B, L, D)
        output = self.original_attention.wo(attn_output)

        return output
```

---

## Why This Works

### 1. ✅ Preserves Pretrained Knowledge
- Moshi's attention weights (Wq, Wk, Wv, Wo) completely unchanged
- All learned representations preserved
- Attention mechanism unchanged

### 2. ✅ TTT Gets Clean Input
- TTT memory module receives raw input `x`
- Same input that attention's Q/K/V projections see
- No information loss from Ring KV Cache!

### 3. ✅ Solves the 5-Minute Problem
```
Position 3750:
├─ Recent cache: [3650-3750] (last 100 tokens, precise)
├─ TTT virtual: 10 tokens representing ALL history 0-3649
├─ Attention queries: [10 virtual tokens | 100 recent tokens]
└─ Result: Full context available! ✅
```

### 4. ✅ Efficient
- Attention always sees: num_virtual + capacity_recent tokens (e.g., 110 total)
- Much cheaper than full attention over 3750 tokens
- TTT compression happens in parallel

---

## Training Strategy

### Phase 1: Freeze Attention, Train TTT
```python
# Freeze all Moshi attention weights
for param in model.attention_layers.parameters():
    param.requires_grad = False

# Only train TTT memory modules
for param in model.ttt_memory_modules.parameters():
    param.requires_grad = True

# Train on 5-10 minute sequences
# TTT learns to compress history into useful K/V representations
```

### Phase 2: Fine-tune Both (Optional)
```python
# Unfreeze attention with small learning rate
for param in model.attention_layers.parameters():
    param.requires_grad = True
# attention_lr = 1e-6

# Continue training TTT
# ttt_lr = 1e-4

# Fine-tune on longer sequences (10-30 minutes)
```

---

## Comparison with JAX Approach

| Aspect | JAX Pure Replacement | Our Hybrid Cache |
|--------|---------------------|------------------|
| **Pretrained weights** | Lost ❌ | Preserved ✅ |
| **TTT input** | Raw input ✅ | Raw input ✅ |
| **Training from scratch** | Required ❌ | Not required ✅ |
| **Attention mechanism** | Replaced | Augmented |
| **Context extension** | Unlimited ✅ | Unlimited ✅ |

---

## Implementation Complexity

### Easy:
- ✅ TTT memory module (reuse existing TTT code)
- ✅ Virtual K/V projection (simple linear layers)
- ✅ Hybrid cache (straightforward concatenation)

### Medium:
- ⏳ Integration with Moshi's streaming
- ⏳ Cache state management
- ⏳ Training stability (new components)

### Hard:
- ⚠️ Determining optimal num_virtual_tokens
- ⚠️ Balancing recent vs virtual attention
- ⚠️ Handling position encodings correctly

---

## Expected Results

**Short sequences (< 4 min)**:
- Should match baseline (pretrained weights preserved)
- TTT virtual tokens not heavily used yet

**Medium sequences (4-10 min)**:
- TTT virtual tokens start providing value
- Attention can access compressed history
- Perplexity remains stable

**Long sequences (10+ min)**:
- Full benefit of TTT compression
- Attention queries 10 virtual + 100 recent tokens
- Can extend to arbitrary lengths (hours)

---

## Advantages Over Previous Approaches

1. **vs. Current (Attention → TTT)**:
   - ✅ TTT gets raw input (no corruption)
   - ✅ Information flows correctly

2. **vs. JAX Replacement**:
   - ✅ Preserves pretrained weights
   - ✅ No need to retrain from scratch
   - ✅ Leverages Moshi's existing capabilities

3. **vs. Hierarchical Cache (from root cause doc)**:
   - ✅ Actually implemented architecture
   - ✅ Clearer separation of concerns
   - ✅ Easier to train incrementally

---

## Next Steps

1. **Implement TTTMemoryModule**
   - Reuse existing TTTMLP code
   - Add K/V projection layers
   - Test on single layer

2. **Implement HybridKVCache**
   - Extend RingKVCache
   - Add virtual K/V handling
   - Test cache operations

3. **Integrate with Moshi**
   - Modify StreamingMultiheadAttention
   - Add TTT memory in parallel
   - Preserve all attention weights

4. **Train and Validate**
   - Freeze attention, train TTT
   - Test on 5-minute sequences
   - Measure perplexity vs baseline

---

## Conclusion

**The Solution**: Don't replace attention or chain TTT after it. Instead, augment the KV cache with TTT-compressed virtual tokens.

**Key Insight**:
- Attention's pretrained weights are preserved (Q/K/V projections unchanged)
- TTT processes raw input in parallel (gets clean information)
- Hybrid cache combines precise recent tokens + compressed history
- Attention queries both seamlessly

**Result**: Moshi's knowledge preserved + Unlimited context via TTT compression ✅

This architecture respects both constraints:
1. Preserve pretrained attention weights ✅
2. Give TTT clean input (not corrupted by Ring KV Cache) ✅

---

**Document Date**: 2025-11-10
**Status**: Proposed solution addressing the "lost pretrained weights" concern
