# Moshi Autoregressive Architecture - Quick Reference

## Key Files Location
```
/home/user/claude-web/moshi/
├── moshi/moshi/moshi/
│   ├── models/lm.py                         # LMModel (49-850) + LMGen (556-851)
│   ├── modules/transformer.py               # StreamingTransformer, MHA, RingKVCache
│   ├── modules/streaming.py                 # Streaming base classes
│   └── configs/moshi_7b_202409.json         # Model config
└── moshi_mlx/moshi_mlx/modules/kv_cache.py  # MLX KV cache variants
```

## 1. KV CACHE (Line 187-280 in transformer.py)

**Type**: Ring buffer (circular queue)
**Purpose**: Efficient streaming with bounded memory
**Key Code**:
```python
self.cache = torch.zeros((2, batch_size, num_heads, capacity, dim_per_head))
# cache[0] = keys, cache[1] = values
# Rotates: new_index = (end_offset + t) % capacity
```

**Benefits**:
- O(T) memory instead of O(T²)
- CUDA graph compatible
- Supports exec_mask for async batching

## 2. CAUSAL ATTENTION (Line 520-574)

**Rule**: `attn_bias = (pos_k >= 0) & (delta >= 0)`
**Meaning**: `query_pos >= key_pos` (strictly left-to-right)
**Context Window**: Default 3000 tokens = ~4 minutes @ 12.5Hz
**Optional**: Can set to None for infinite context

```python
# Current position + relative positions
pos_q = offset + torch.arange(T)  
pos_k = from_cache
delta = pos_q - pos_k
# Only attend if delta >= 0 (past or current token)
```

## 3. TOKEN GENERATION LOOP (Line 668-783)

**Method**: `LMGen._step()` - generates ONE timestep

**Steps**:
1. Update circular cache with previous tokens (with per-codebook delays)
2. Fetch current input from cache (aligned to delays)
3. Forward through main transformer (32 layers, 4096-dim)
4. Sample text token from logits
5. Forward through depformer (6 layers, 1024-dim) for 8 audio codebooks
6. Update offset and cache
7. Return tokens (with delay compensation)

**Not like diffusion**: No parallel denoising, pure sequential left-to-right

## 4. CONTEXT WINDOW

**Config**: `/home/user/claude-web/moshi/configs/moshi_7b_202409.json`
```json
{
  "context": 3000,              // Main: 3000 tokens = 240 sec = 4 min
  "depformer_context": 8        // Depth: Very local (8 tokens)
  "max_period": 10000           // RoPE max period
}
```

**At Runtime**: `set_attention_context(model, context=N)` changes context dynamically

**Cache Size**: `max_delay + 2` per codebook (small ring buffer)

## 5. ATTENTION TYPE

**Standard**: Scaled dot-product attention (F.scaled_dot_product_attention)
**NOT**: Mamba, linear attention, or SSMs
**Optimization**: Ring KV cache + context windowing + position offsets

**Optional Features**:
- **RoPE** (Rotary Position Embeddings): Relative positions, infinite extrapolation
- **Cross-attention**: For conditioning (text, speaker, etc.)
- **Weights per step**: Different linear layers per codebook (for depformer)

## 6. LAYER ARCHITECTURE

### Per Layer = Attention + FFN + Residuals (Pre-norm style)

```
Input x
├─ norm1
├─ self_attn (causal + KV cache)
├─ residual + layer_scale
├─ [optional: cross_attn]
├─ norm2
├─ FFN (gating: silu, gelu, etc.)
├─ residual + layer_scale
→ Output
```

### Two Transformers

1. **Main (32 layers)**
   - dim=4096, num_heads=32
   - num_layers=32
   - causal=True, context=3000
   - Processes temporal sequence
   - Outputs: text logits + hidden states

2. **Depformer (6 layers)**
   - dim=1024, num_heads=16
   - num_layers=6
   - causal=True, context=8
   - Refines audio codebooks
   - weights_per_step=True (per-codebook weighting)
   - Independent streaming state

## MOSHI vs COGVIDEOX (DIFFUSION)

| Aspect | Moshi | CogVideoX |
|--------|-------|-----------|
| Generation | Sequential, left-to-right | Parallel denoising (iterative) |
| Per-step | Previous tokens → next | Pure noise → content |
| Dependencies | Strict DAG (causal) | Fully connected (bidirectional) |
| Context | Limited window (3000) | Global image view |
| Memory | KV cache O(T) | Full activations O(T×D) |
| Speed | Single pass/token | 50+ denoising passes |
| Latency | ~80ms/token | ~15+ seconds/video |
| Real-time | Yes | No |

## STREAMING STATE (_LMGenState)

```python
cache              # [B, K, CT] ring buffer for tokens
initial            # [B, K, 1] special start tokens
offsets            # [B] position in unlimited sequence
offset_cpu         # CPU-side position tracker
graphed_main       # CUDA-graphed main transformer
graphed_depth      # CUDA-graphed depformer
condition_sum      # Conditioning features (optional)
condition_cross    # Cross-attention features (optional)
```

## IMPLEMENTATION SUMMARY

**Autoregressive** = Token-by-token, strictly causal, left-to-right
**Streaming** = Online processing, state management, bounded memory
**Efficient** = Ring KV cache, context windowing, CUDA graphs
**Real-time** = <100ms latency, full-duplex capable

**Difference from diffusion**: 
- Not iterative refinement (single forward pass)
- Not bidirectional (pure left-to-right)
- Not parallel (sequential generation)
- Instead: KV cache, causal masking, position tracking

---

## WHERE TO MODIFY FOR TTT INTEGRATION

To add TTT layers (for unlimited context):

1. **Replace top N layers** of main transformer with TTT modules
2. **Maintain streaming interface** (StreamingModule compatibility)
3. **Keep causal semantics** (TTT updates only from past tokens)
4. **Preserve KV cache** (TTT uses hidden state as context instead)

Key integration point: 
- `/home/user/claude-web/moshi/moshi/moshi/modules/transformer.py:844-862`
- Replace StreamingTransformerLayer with TTTLayer in layer stack
