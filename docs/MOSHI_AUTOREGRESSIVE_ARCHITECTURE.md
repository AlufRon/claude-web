# Moshi Autoregressive Architecture Analysis

## Executive Summary

Moshi is a **7B-parameter streaming autoregressive speech language model** designed for full-duplex real-time dialogue. Unlike diffusion models like CogVideoX that denoise from noise in parallel timesteps, Moshi generates tokens sequentially with strict left-to-right (causal) processing, similar to traditional LLMs.

---

## 1. KV CACHE IMPLEMENTATION

### Location & Architecture

**PyTorch Implementation** (Main): `/home/user/claude-web/moshi/moshi/moshi/modules/transformer.py:187-280`

**MLX Implementation** (Inference): `/home/user/claude-web/moshi/moshi_mlx/moshi_mlx/modules/kv_cache.py:1-197`

### Ring KV Cache (Streaming-Optimized)

```python
class RingKVCache:
    """Efficient streaming KVCache compatible with CUDA Graph execution."""
    
    def __init__(self, batch_size, num_heads, dim_per_head, capacity):
        # Ring buffer for circular queue behavior
        self.cache = torch.zeros(
            (2, batch_size, num_heads, capacity, dim_per_head),
            dtype=torch.bfloat16
        )
        # Two buffers: [0] for keys, [1] for values
        self.end_offset = torch.zeros(batch_size)
        self.capacity = capacity
```

**Key Features**:
- **Circular Buffer**: Reuses memory instead of growing unbounded
- **Exec Mask Support**: Allows asynchronous batch execution
- **Ring Rotation**: When full, overwrites oldest entries
- **Position Mapping**: Tracks which indices correspond to which timesteps

### Cache Update Process (Line 227-279)

```python
def complete(self, k, v, exec_mask):
    # 1. Store new K,V at current write position
    indexes = (torch.arange(T) + self.end_offset) % self.capacity
    self.cache[0].scatter_(2, indexes, k)  # Keys
    self.cache[1].scatter_(2, indexes, v)  # Values
    
    # 2. Calculate positions for attention
    positions = torch.where(
        delta <= 0,
        last_offset + delta,
        last_offset + delta - self.capacity
    )
    
    # 3. Mark invalid entries as -1
    positions = torch.where(invalid, -1, positions)
    
    # 4. Update offset
    self.end_offset += T
    
    return KVCacheResult(keys, values, positions)
```

### MLX Rotating KV Cache (MLX Framework)

```python
class RotatingKVCache:
    """MLX variant for Apple Silicon optimization."""
    
    def update_and_fetch(self, keys, values) -> tuple[mx.array, mx.array]:
        # Prefill mode: S > 1 (multiple tokens at once)
        if S > 1:
            # Trim old entries beyond max_size
            trim_size = self.keys.shape[2] - self.max_size + 1
            self.keys = self._trim(trim_size, self.keys, keys)
            self.values = self._trim(trim_size, self.values, values)
        
        # Decode mode: S == 1 (single token generation)
        else:
            # Rotate position and update single entry
            if self._idx == self.max_size:
                self._idx = self.keep  # Wrap around
            self.keys[..., self._idx, :] = keys
            self.values[..., self._idx, :] = values
```

---

## 2. CAUSAL ATTENTION & MASKING

### Location: `/home/user/claude-web/moshi/moshi/moshi/modules/transformer.py:520-574`

### Causal Mask Construction

```python
def forward(self, query, key, value):
    # Get current offset in sequence
    offset = state.offset  # Position in full sequence
    
    # Create position indices
    pos_q = offset.view(-1, 1, 1) + torch.arange(T, device=q.device)
    pos_k = pos_k[:, None]  # Key positions from cache
    
    # Causal constraint: can only attend to past/current
    delta = pos_q - pos_k  # Position difference
    attn_bias = (pos_k >= 0) & (delta >= 0)  # Strictly left-to-right
    
    # Context window: optional receptive field limit
    if self.context is not None:
        attn_bias = attn_bias & (delta < self.context)
    
    # Apply attention with bias
    x = F.scaled_dot_product_attention(q, k, v, attn_bias, dropout_p=0.0)
```

**Strictly Causal**: `delta >= 0` means:
- Position query >= position key
- Current token sees only past + itself
- NO bidirectional context
- Pure sequential left-to-right generation

### Context Window Management

**From Config** (`/home/user/claude-web/moshi/configs/moshi_7b_202409.json`):
```json
{
  "context": 3000,           // Main transformer
  "depformer_context": 8     // Secondary depformer
}
```

**Interpretation**:
- **3000 tokens** @ 12.5 Hz = **240 seconds = ~4 minutes** max context
- If context is `None`, infinite context (no performance optimization)
- Can be changed at inference time via `set_attention_context()`

---

## 3. TOKEN GENERATION LOOP

### Main Generation Class: `LMGen` 
**Location**: `/home/user/claude-web/moshi/moshi/moshi/models/lm.py:556-851`

### Sequential Generation Loop (Key Method: `_step`)

```python
class LMGen(StreamingModule[_LMGenState]):
    """Streaming autoregressive generation."""
    
    @torch.no_grad()
    def _step(self, input_tokens: torch.Tensor) -> torch.Tensor | None:
        """Generate ONE timestep of output."""
        
        # Step 1: Manage circular cache of recently generated tokens
        CT = state.cache.shape[2]  # Cache time dimension
        
        # Input positions (with delays per codebook)
        write_positions = (state.offsets[:, None] + delays[:, None]) % CT
        scatter_with_mask_(state.cache[:, ...], -1, write_positions, input_tokens, ...)
        
        # Step 2: Retrieve current input from cache (positions aligned with delays)
        is_init = state.offsets[:, None] <= delays[:, None]
        positions = (state.offsets % CT)[:, None].expand_as(is_init)
        input_ = state.cache.gather(dim=2, index=positions)
        input_ = torch.where(is_init, state.initial, input_)
        
        # Step 3: Forward through main transformer (SINGLE timestep)
        transformer_out, text_logits = state.graphed_main(
            input_, 
            state.condition_sum, 
            state.condition_cross
        )
        
        # Step 4: Sample text token from logits
        text_token = sample_token(
            text_logits.float(),
            use_sampling=True,
            temp=0.7,
            top_k=25
        )
        
        # Step 5: Run depformer (inner generation loop for audio codebooks)
        if state.graphed_depth is not None:
            audio_tokens = state.graphed_depth(text_token, transformer_out)
        
        # Step 6: Update offset and cache for next timestep
        state.offsets = state.offsets + 1
        state.offset_cpu += 1
        scatter_with_mask_(state.cache[:, :1], -1, positions, text_token, ...)
        
        # Step 7: Return generated tokens (with delay consideration)
        return out, transformer_out
```

### Depformer Inner Loop (Recursive Generation)

```python
def depformer_step(self, text_token, transformer_out):
    """Generate audio codebooks autoregressively given text token."""
    
    with lm_model.depformer.streaming(batch_size):
        prev_token = text_token
        
        # Generate ONE token per audio codebook
        for cb_index in range(lm_model.dep_q):  # e.g., 8 codebooks
            # Input: previous codebook token + transformer features
            input_ = prev_token[:, None, None]
            
            # Forward through depformer for current codebook
            logits = lm_model.forward_depformer(cb_index, input_, transformer_out)
            
            # Sample next token
            next_token = sample_token(logits.float(), ...)
            
            # Use as input for next codebook
            prev_token = next_token[:, 0, 0]  # Shape [B]
```

### Key Differences from Diffusion Models (Like CogVideoX)

| Aspect | Moshi (Autoregressive) | CogVideoX (Diffusion) |
|--------|----------------------|----------------------|
| **Generation** | Sequential, left-to-right | Parallel denoising steps |
| **Per-step Input** | Previous tokens + noise | Pure noise (iterative) |
| **Context** | All previous tokens (causal) | Global image context |
| **Dependencies** | Strict left-to-right | Bidirectional refinement |
| **Speed** | Single pass per token | Multiple refinement passes |
| **Memory** | KV cache O(T) | Full activations O(T×D) |

---

## 4. CONTEXT WINDOW MANAGEMENT

### Configuration Storage
**File**: `/home/user/claude-web/moshi/configs/moshi_7b_202409.json`

```json
{
  "context": 3000,              // Hard limit for attention window
  "max_period": 10000,          // For RoPE position embeddings
  "depformer_context": 8        // Per-codebook local context
}
```

### Runtime Context Setting
**File**: `/home/user/claude-web/moshi/moshi/moshi/modules/transformer.py:158-172`

```python
def set_attention_context(model: nn.Module, context: int | None = None):
    """Change context dynamically (before generating each sequence)."""
    for module in model.modules():
        if isinstance(module, StreamingMultiheadAttention):
            module.context = context  # Override default
```

### Context in Initialization
**File**: `/home/user/claude-web/moshi/moshi/moshi/models/lm.py:146-158`

```python
self.transformer = StreamingTransformer(
    d_model=dim,
    num_heads=num_heads,
    context=context,           # Set here: 3000 tokens
    causal=True,
    ...
)
```

### Cache Size Calculation
**File**: `/home/user/claude-web/moshi/moshi/moshi/models/lm.py:605-613`

```python
def _init_streaming_state(self, batch_size):
    lm_model = self.lm_model
    initial = lm_model._get_initial_token()
    
    # Ring cache with max_delay + 2 buffer
    cache = torch.full(
        (batch_size, self.lm_model.num_codebooks, self.max_delay + 2),
        lm_model.ungenerated_token_id,
        device=lm_model.device
    )
```

---

## 5. ATTENTION MECHANISM

### Type: Standard Scaled Dot-Product Attention with Modifications

**Location**: `/home/user/claude-web/moshi/moshi/moshi/modules/transformer.py:328-574`

### Standard Multi-Head Attention

```python
class StreamingMultiheadAttention(StreamingModule[_MHAState]):
    """Standard transformer attention with streaming and causal support."""
    
    def __init__(self, embed_dim, num_heads, causal=False, context=None, rope=None):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.causal = causal              # Enable causal masking
        self.context = context            # Optional context window
        self.rope = rope                  # Rotary position embeddings
        
        # Separate linear projections per timestep if weights_per_step
        self.in_projs = nn.ModuleList([...])   # Q,K,V projections
        self.out_projs = nn.ModuleList([...])  # Output projection
```

### Key Modifications for Streaming

1. **Weights Per Step** (When depformer uses multiple codebooks):
   ```python
   self.in_projs = nn.ModuleList([
       nn.Linear(embed_dim, 3*embed_dim) for _ in range(weights_per_step)
   ])
   ```
   - Different linear layers for each timestep
   - Enables per-codebook weighting schedule

2. **Rotary Position Embeddings (RoPE)**:
   ```python
   if self.rope:
       q, k = self.rope(q, k, offset, time_before_heads=False)
   ```
   - Relative position encoding
   - Enables unlimited extrapolation beyond training context

3. **Cross-Attention Support** (Optional):
   ```python
   if self.cross_attention:
       # Can attend to external conditioning (e.g., speaker embedding)
       k, v = self._get_cross_attention(key, value)
   ```

### Attention Computation (Line 520-562)

```python
def forward(self, query, key, value):
    B, T = query.shape[:2]
    
    # Project Q,K,V (with weights_per_step consideration)
    q, k, v = rearrange(
        apply_weights_per_step(self.in_projs, ..., query, offset_cpu),
        "b t (p h d) -> p b h t d", p=3, h=self.num_heads
    )
    
    # Apply rotary embeddings if present
    if self.rope:
        q, k = self.rope(q, k, offset, time_before_heads=False)
    
    # Complete KV cache (add to ring buffer)
    k, v, pos_k = self._complete_kv(k, v)
    pos_k = pos_k[:, None]
    
    # Build causal mask
    if self.causal:
        pos_q = offset.view(-1, 1, 1) + torch.arange(T, device=q.device).view(-1, 1)
        delta = pos_q - pos_k
        attn_bias = (pos_k >= 0) & (delta >= 0)  # STRICT CAUSALITY
        if self.context is not None:
            attn_bias = attn_bias & (delta < self.context)
        attn_bias = attn_bias[:, None]
    else:
        attn_bias = None
    
    # Scaled dot-product attention
    x = F.scaled_dot_product_attention(q, k, v, attn_bias, dropout_p=0.0)
    
    # Project output
    x = rearrange(x, "b h t d -> b t (h d)")
    x = apply_weights_per_step(self.out_projs, ..., x, offset_cpu)
    
    return x
```

### NOT Streaming Attention

Moshi uses **standard scaled dot-product attention**, not specialized streaming variants like:
- **Linear Attention** (Mamba, S4)
- **State Space Models**
- **Hierarchical attention**

Instead, it manages streaming via:
- KV cache with bounded memory (ring buffer)
- Context window limitation
- Position offset tracking

---

## 6. LAYER ARCHITECTURE

### Complete Layer Stack

**File**: `/home/user/claude-web/moshi/moshi/moshi/modules/transformer.py:586-778`

### Transformer Layer Structure

```python
class StreamingTransformerLayer(StreamingModule[_LayerState]):
    """Each layer = Attention + CrossAttn (optional) + FFN + Residuals"""
    
    def __init__(self, d_model, num_heads, dim_feedforward, causal=False, 
                 context=None, rope=None, norm="layer_norm", layer_scale=None,
                 gating="silu", cross_attention=False):
        
        # 1. SELF ATTENTION + NORM
        self.norm1 = LayerNorm(d_model)          # Pre-norm
        self.self_attn = StreamingMultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            causal=causal,
            context=context,
            rope=rope
        )
        self.layer_scale_1 = LayerScale(d_model) or Identity()
        
        # 2. OPTIONAL CROSS ATTENTION
        if cross_attention:
            self.norm_cross = LayerNorm(d_model)
            self.cross_attention = StreamingMultiheadAttention(
                embed_dim=d_model,
                cross_attention=True
            )
            self.layer_scale_cross = LayerScale(d_model) or Identity()
        
        # 3. FEED-FORWARD + NORM
        self.norm2 = LayerNorm(d_model)          # Pre-norm
        if gating == "none":
            self.linear1 = Linear(d_model, dim_feedforward)
            self.linear2 = Linear(dim_feedforward, d_model)
        else:
            self.gating = make_gating(gating, d_model, dim_feedforward)
        self.layer_scale_2 = LayerScale(d_model) or Identity()
```

### Forward Pass (Per-Layer Computation)

```python
def forward(self, x, cross_attention_src=None):
    """Residual connection + Pre-norm architecture."""
    
    # 1. Self-Attention Block
    x_orig = x
    x = self.norm1(x)
    update = self.self_attn(x, x, x)
    x = x_orig + self.layer_scale_1(update)     # Residual + LayerScale
    
    # 2. Cross-Attention Block (if present)
    if self.cross_attention is not None:
        x_orig = x
        x = self.norm_cross(x)
        update = self.cross_attention(x, cross_attention_src, cross_attention_src)
        x = x_orig + self.layer_scale_cross(update)
    
    # 3. Feed-Forward Block
    x_orig = x
    x = self.norm2(x)
    if self.gating is None:
        update = self.linear2(F.gelu(self.linear1(x)))
    else:
        update = self.gating(x)  # GLU, SiGLU, etc.
    x = x_orig + self.layer_scale_2(update)     # Residual
    
    return x
```

### Layer Stack (Transformer Container)

```python
class StreamingTransformer(StreamingModule[_TransformerState]):
    """Stack of StreamingTransformerLayers."""
    
    def __init__(self, d_model, num_heads, num_layers, dim_feedforward,
                 causal=False, context=None, positional_embedding="rope",
                 checkpointing=False, **kwargs):
        
        # 1. Positional Embeddings
        if positional_embedding in {"rope", "sin_rope"}:
            self.rope = RotaryEmbedding(max_period=max_period)
        
        # 2. Layer Stack
        self.layers = nn.ModuleList([
            StreamingTransformerLayer(
                d_model=d_model,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                causal=causal,
                context=context,
                rope=self.rope,
                **kwargs
            )
            for _ in range(num_layers)
        ])
        
        self.checkpointing = checkpointing
```

### Forward Pass (Full Transformer)

```python
def forward(self, x):
    B, T, C = x.shape
    
    # Get current position offset
    offsets = state.offsets if state else torch.zeros(1)
    
    # Add positional embeddings (sinusoidal + RoPE)
    if self.positional_embedding in {"sin", "sin_rope"}:
        positions = torch.arange(T) + offsets.view(-1, 1, 1)
        pos_emb = create_sin_embedding(positions, C, self.max_period)
        x = x + self.positional_scale * pos_emb
    
    # Pass through all layers
    for layer in self.layers:
        if self.checkpointing:
            x = torch_checkpoint(layer, x, use_reentrant=False)
        else:
            x = layer(x)
    
    # Update offset in streaming state
    if state:
        state.offsets = state.offsets + T
    
    return x
```

---

## 7. DUAL TRANSFORMER ARCHITECTURE

### Main Temporal Transformer

```python
self.transformer = StreamingTransformer(
    d_model=4096,              # 7B model
    num_heads=32,
    num_layers=32,             # 32 layers
    dim_feedforward=16896,     # 4× expansion
    causal=True,
    context=3000,              # 4-minute context
    rope=True,
    gating="silu",
    norm="rms_norm_f32"
)
```

**Role**: Process temporal sequence of audio codes
**Input**: Concatenated text + audio embeddings
**Output**: Hidden states for logit prediction

### Secondary Depformer (Depth Transformer)

```python
self.depformer = StreamingTransformer(
    d_model=1024,              # 100M total
    num_heads=16,
    num_layers=6,              # Only 6 layers
    dim_feedforward=4224,
    causal=True,
    context=8,                 # Very local context
    weights_per_step=True,     # Per-codebook weighting
    weights_per_step_schedule=[...],
    norm="rms_norm_f32"
)
self.depformer.set_streaming_detached(True)  # Independent streaming state
```

**Role**: Refine audio codebooks given text + transformer output
**Purpose**: Capture interdependencies between 8 audio codebooks
**Why separate**: 
- Depformer has different streaming semantics (per-codebook, not per-timestep)
- Much smaller (100M vs 7B)
- Operates on features from main transformer

### Token Prediction & Output

```python
# Text prediction from main transformer
self.text_linear = Linear(4096, 32000)  # → Text vocabulary
text_logits = self.text_linear(transformer_out)

# Audio prediction from depformer
self.linears = nn.ModuleList([
    Linear(1024, 2048)  # depformer_dim → card (vocabulary size)
    for _ in range(8)   # One per audio codebook
])
```

---

## MOSHI vs DIFFUSION MODELS: KEY DIFFERENCES

### CogVideoX Diffusion vs Moshi Autoregressive

```
┌─────────────────────────────────────────────────────────────────┐
│                     MOSHI (AUTOREGRESSIVE)                      │
├─────────────────────────────────────────────────────────────────┤
│ GENERATION PROCESS:                                             │
│ 1. Initialize with special tokens                              │
│ 2. FOR t=1 to T:                                               │
│    - Attend to all previous tokens (causal)                    │
│    - Predict token t given context [0...t-1]                  │
│    - Sample token t                                            │
│    - Add to context for next iteration                         │
│                                                                 │
│ CHARACTERISTICS:                                                │
│ • Sequential: Must generate left-to-right                      │
│ • Streaming: Can generate online token-by-token                │
│ • KV Cache: O(T) memory, efficient                            │
│ • Context: Up to 3000 tokens (4 minutes)                       │
│ • Speed: Single forward pass per token                         │
│ • Latency: ~80ms per token                                     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                   COGVIDEOX (DIFFUSION)                         │
├─────────────────────────────────────────────────────────────────┤
│ GENERATION PROCESS:                                             │
│ 1. Initialize with pure noise [B, T, H, W]                    │
│ 2. FOR step=1 to 50 (or more):                                │
│    - Denoise entire sequence with diffusion model              │
│    - Gradually reduce noise across all frames                  │
│    - Refine details iteratively                                │
│                                                                 │
│ CHARACTERISTICS:                                                │
│ • Parallel: Can generate all frames simultaneously             │
│ • Non-streaming: Requires full output upfront                 │
│ • Memory: O(T×D) for full activations                         │
│ • Context: Global image context (bidirectional)               │
│ • Speed: 50+ denoising steps (slow)                           │
│ • Latency: ~15+ seconds for video                             │
└─────────────────────────────────────────────────────────────────┘
```

### Fundamental Architecture Differences

| Dimension | Moshi (LLM-like) | CogVideoX (Diffusion) |
|-----------|------------------|----------------------|
| **Update Rule** | `p(x_t\|x_{<t})` | `p(x\|noise_{1:T})` |
| **Dependency Graph** | DAG (left-to-right) | Fully connected (all-to-all) |
| **Optimization** | Token likelihood | Noise prediction error |
| **Memory Pattern** | KV cache growth | Full batch activation |
| **Inference Path** | Single forward path | Multiple refinement paths |
| **Streaming Friendly** | Yes (inherent) | No (needs full sequence) |
| **Real-time Capable** | Yes | No (latency prohibitive) |

---

## IMPLEMENTATION FILES REFERENCE

### Core Architecture Files

| File | Lines | Purpose |
|------|-------|---------|
| `/home/user/claude-web/moshi/moshi/moshi/models/lm.py` | 1-850 | Main LMModel + LMGen generation |
| `/home/user/claude-web/moshi/moshi/moshi/modules/transformer.py` | 1-950+ | StreamingTransformer, MHA, KV cache |
| `/home/user/claude-web/moshi/moshi/moshi/modules/streaming.py` | 1-218 | Streaming base classes & state management |
| `/home/user/claude-web/moshi/moshi_mlx/moshi_mlx/modules/kv_cache.py` | 1-196 | MLX KV cache implementations |
| `/home/user/claude-web/moshi/configs/moshi_7b_202409.json` | 1-31 | Model hyperparameters & context |

### Generation Flow Files

| File | Lines | Purpose |
|------|-------|---------|
| `/home/user/claude-web/moshi/moshi/moshi/models/lm.py` | 556-851 | LMGen class (streaming generation) |
| `/home/user/claude-web/moshi/moshi/moshi/models/lm.py` | 668-783 | _step() method (per-timestep generation) |
| `/home/user/claude-web/moshi/moshi/moshi/models/lm.py` | 809-850 | depformer_step() (audio codebook refinement) |
| `/home/user/claude-web/moshi/moshi_mlx/moshi_mlx/models/generate.py` | 1-149 | MLX LmGen wrapper |

---

## STREAMING STATE MANAGEMENT

### State Dataclass (Line 523-553)

```python
@dataclass
class _LMGenState(State):
    cache: torch.Tensor                    # [B, K, CT] ring buffer
    initial: torch.Tensor                  # [B, K, 1] initial tokens
    graphed_main: CUDAGraphed             # Compiled main transformer
    graphed_depth: CUDAGraphed | None     # Compiled depformer
    offsets: torch.Tensor                  # [B] current position
    offset_cpu: int                        # CPU-side offset tracker
    condition_sum: torch.Tensor | None    # Conditioning features
    condition_cross: torch.Tensor | None  # Cross-attention conditioning
    cfg_is_masked_until: torch.Tensor | None  # Classifier-free guidance
    exit_stack: ExitStack                 # Resource cleanup
```

### Streaming Initialization (Line 605-666)

```python
def _init_streaming_state(self, batch_size):
    # Cache stores recently generated tokens (circular buffer)
    cache = torch.full(
        (batch_size, self.lm_model.num_codebooks, self.max_delay + 2),
        lm_model.ungenerated_token_id,
        device=lm_model.device
    )
    
    # Offsets track position in unlimited sequence
    offsets = torch.zeros(batch_size, device=lm_model.device, dtype=torch.long)
    
    # CUDA graphs for compilation
    graphed_main = CUDAGraphed(lm_model.forward_text, disable=False)
    graphed_depth = CUDAGraphed(self.depformer_step, disable=False)
    
    # Context manager for streaming
    with self.lm_model.streaming(batch_size):
        # Initializes streaming state in all submodules
        ...
```

---

## KEY INSIGHTS FOR DIFFUSION COMPARISON

### Why Autoregressive for Speech

1. **Streaming Requirement**: Real-time dialogue needs online processing
2. **Dependencies**: Speech quality depends heavily on context (prosody, consistency)
3. **Clarity**: Each token determines next audio characteristics
4. **Latency**: Autoregressive allows ultra-low latency (~80ms)

### Why Not Streaming Attention

Moshi doesn't use:
- **Linearized Attention** (Mamba): Would lose expressiveness
- **State Space Models**: Less expressive than transformer attention
- **Hierarchical Attention**: Added complexity without clear benefit

Instead:
- **KV Cache**: Standard approach, well-optimized
- **Context Windowing**: Limits memory while maintaining quality
- **Ring Buffer**: Efficient streaming with CUDA graphs

---

## CONCLUSION

Moshi represents a **streaming-optimized autoregressive language model** fundamentally different from diffusion models like CogVideoX:

- **Sequential generation**: One token at a time, left-to-right
- **Causal masking**: No future information leaks
- **KV caching**: Memory-efficient context management
- **Streaming native**: Online processing from day one
- **Real-time capable**: <100ms latency per token

The integration of TTT layers would leverage this architecture by replacing top transformer layers with test-time trainable modules, extending context to hours while maintaining the streaming semantics.

