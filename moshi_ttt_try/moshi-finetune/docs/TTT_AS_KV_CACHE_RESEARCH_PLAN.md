# Research Plan: TTT as KV Cache Replacement for Moshi

## Executive Summary

**Goal**: Replace or augment Moshi's fixed-size KV cache (context=3000) with a TTT-based unbounded memory system, enabling true long-context generation without cache wraparound.

**Key Insight**: KV cache stores compressed representations of past tokens for attention. TTT can learn a better compression that's unbounded and adaptive.

**Target**: Minimal architectural changes, maintain streaming capability, preserve generation quality.

---

## 🎯 Problem Statement

### Current Limitations

1. **KV Cache Wraparound** (Token 3000):
   - Fixed capacity: 3000 tokens
   - Oldest tokens discarded → loss of context
   - Causes repetition and quality degradation

2. **Memory Scaling**:
   - KV cache: O(n) memory where n = context length
   - Current: 3000 tokens × 32 layers × 4096 dim = ~400MB
   - Scaling to 30k tokens → 4GB just for cache!

3. **Quadratic Attention**:
   - Attention over n tokens: O(n²) compute
   - Limits practical context length

### Why TTT?

- **Linear complexity**: O(n) for sequence of length n
- **Learned compression**: Adapts what to remember
- **Unbounded**: No fixed capacity limit
- **Already integrated**: We have TTT in layer 30

---

## 🏗️ Proposed Architectures (4 Options)

### **Option 1: Hierarchical Cache (RECOMMENDED)**

**Architecture**:
```
┌─────────────────────────────────────────┐
│  Current Token: x_t                     │
└─────────────────┬───────────────────────┘
                  │
                  ▼
         ┌────────────────┐
         │   Attention    │
         └────────┬───────┘
                  │
      ┌───────────┴───────────┐
      │                       │
      ▼                       ▼
┌─────────────┐      ┌──────────────┐
│  KV Cache   │      │  TTT State   │
│  (Recent    │      │  (Compressed │
│   N=100)    │      │   History)   │
└─────────────┘      └──────────────┘
  Recent tokens       All old tokens
  (Precise attn)      (Learned repr)
```

**Key Components**:

1. **Recent Window**: Standard KV cache for last N tokens (N=100-500)
   - Precise attention for local context
   - Fast, no changes needed

2. **TTT Compression**: Processes tokens as they exit the window
   - Input: Old K, V pairs
   - Output: Compressed state vector(s)
   - Maintains summary of entire history

3. **Hybrid Attention**:
   ```python
   # Attention over recent window (standard)
   attn_recent = softmax(Q @ K_recent.T) @ V_recent

   # Query TTT compressed memory
   attn_old = query_ttt_memory(Q, ttt_state)

   # Combine (learnable gating)
   output = alpha * attn_recent + (1 - alpha) * attn_old
   ```

**Advantages**:
- ✅ Minimal changes to attention mechanism
- ✅ Unbounded context via TTT
- ✅ Precise local attention preserved
- ✅ Streaming-compatible
- ✅ Gradual degradation (not sudden at N tokens)

**Challenges**:
- Need to design `query_ttt_memory()` interface
- Training: How to update TTT during generation?
- Gating mechanism between recent/old

---

### **Option 2: TTT-Compressed KV Pairs**

**Architecture**:
```
Old KV pairs → TTT Compression → Compressed K', V'
Recent KV pairs → K_recent, V_recent

Attention input: K = [K', K_recent], V = [V', V_recent]
```

**How it works**:
1. When KV cache full, compress oldest M tokens with TTT
2. TTT outputs: compressed K', V' (fixed size, e.g., 10 tokens worth)
3. Attention operates over: [compressed old + recent] KV pairs

**Advantages**:
- ✅ No changes to attention mechanism!
- ✅ Compression happens "offline"
- ✅ Easy to implement

**Challenges**:
- TTT must output K, V format (not arbitrary state)
- Fixed compression ratio needed
- May lose fine-grained information

---

### **Option 3: TTT as Memory Bank (Cross-Attention Style)**

**Architecture**:
```
x_t → Q_t
TTT state → project to K_ttt, V_ttt
Recent cache → K_recent, V_recent

Attention: Q_t attends to [K_ttt, K_recent], [V_ttt, V_recent]
```

**How it works**:
1. Maintain TTT state for all history
2. Project TTT hidden state to K, V dimensions
3. Attention operates over: [TTT-derived KV + recent KV]

**Advantages**:
- ✅ TTT state is flexible representation
- ✅ Projection is learnable
- ✅ Standard attention mechanism

**Challenges**:
- How many K, V vectors from TTT state? (1? 10? 100?)
- Position information lost
- Training complexity

---

### **Option 4: Pure TTT Replacement (Most Radical)**

**Architecture**:
```
Remove attention KV cache entirely.
Replace with: TTT-only or TTT + Local sliding window attention
```

**Not recommended because**:
- Requires complete retraining
- Attention is good at local patterns
- Too invasive for Moshi

---

## 🎯 Recommended Approach: Hierarchical Cache (Option 1)

### Detailed Design

#### 1. **Architecture Components**

```python
class TTTCachedAttention(nn.Module):
    def __init__(self, dim, num_heads,
                 recent_window=100,  # Recent KV cache size
                 ttt_compression_ratio=10):  # 10 old tokens → 1 TTT token
        self.recent_window = recent_window

        # Standard attention components
        self.qkv_proj = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)

        # TTT memory compressor
        self.ttt_compressor = TTTLayer(
            dim=dim,
            num_heads=num_heads,
            # Output: compressed K, V representations
        )

        # Gating between recent and old
        self.memory_gating = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, cache_state):
        """
        Args:
            x: [B, 1, dim] - current token
            cache_state: {
                'recent_kv': Recent KV cache,
                'ttt_state': TTT compressed memory
            }
        """
        Q, K, V = self.qkv_proj(x).chunk(3, dim=-1)

        # 1. Attention over recent window (standard)
        attn_recent = self.attend(Q,
                                   cache_state['recent_kv']['K'],
                                   cache_state['recent_kv']['V'])

        # 2. Query TTT memory for old context
        if cache_state['ttt_state'] is not None:
            attn_old = self.query_ttt_memory(Q, cache_state['ttt_state'])
        else:
            attn_old = 0

        # 3. Gated combination
        alpha = torch.sigmoid(self.memory_gating)
        output = alpha * attn_recent + (1 - alpha) * attn_old

        return self.out_proj(output), updated_cache
```

#### 2. **TTT Memory Interface**

The key challenge: **How does attention query TTT state?**

**Solution A: TTT outputs pseudo-KV pairs**
```python
def query_ttt_memory(self, Q, ttt_state):
    """
    TTT state → Pseudo K, V pairs that attention can query
    """
    # Project TTT state to K, V dimensions
    # TTT state: [B, ttt_dim]
    # Output: K_ttt [B, n_virtual_tokens, head_dim]
    #         V_ttt [B, n_virtual_tokens, head_dim]

    K_ttt = self.ttt_to_k_proj(ttt_state)  # [B, 10, head_dim]
    V_ttt = self.ttt_to_v_proj(ttt_state)  # [B, 10, head_dim]

    # Standard attention
    scores = Q @ K_ttt.transpose(-2, -1) / sqrt(d_k)
    attn_weights = softmax(scores)
    output = attn_weights @ V_ttt

    return output
```

**Solution B: Cross-attention with TTT embedding**
```python
def query_ttt_memory(self, Q, ttt_state):
    """
    Cross-attention: Q from current token, KV from TTT
    """
    # TTT state is already a learned embedding of history
    # Treat it as a single "memory token"

    K_memory = self.state_to_k(ttt_state)  # [B, 1, head_dim]
    V_memory = self.state_to_v(ttt_state)  # [B, 1, head_dim]

    # Attention to single memory token
    scores = (Q @ K_memory.transpose(-2, -1)) / sqrt(d_k)
    attn_weight = softmax(scores)
    output = attn_weight @ V_memory

    return output
```

**Recommendation**: Solution A with n_virtual_tokens=10-50
- Balances expressiveness vs efficiency
- TTT learns to output multiple "summary tokens"

#### 3. **Cache Management**

```python
class HierarchicalCache:
    def __init__(self, recent_window=100):
        self.recent_window = recent_window
        self.recent_kv = deque(maxlen=recent_window)
        self.ttt_state = None
        self.ttt_updater = TTTCompressor()

    def add_token(self, k, v):
        # Add to recent cache
        self.recent_kv.append((k, v))

        # If cache full, compress oldest to TTT
        if len(self.recent_kv) == self.recent_window:
            # Get oldest token that's about to be evicted
            k_old, v_old = self.recent_kv[0]

            # Update TTT state with this token
            self.ttt_state = self.ttt_updater.update(
                self.ttt_state, k_old, v_old
            )
```

---

## 🔬 Implementation Roadmap

### Phase 1: Proof of Concept (2-3 days)

**Goal**: Verify basic concept works

1. **Implement simplified version**:
   - Recent window: 100 tokens
   - TTT memory: Single vector (not multiple tokens)
   - Test on single layer (layer 30)

2. **Components to build**:
   ```
   - `HierarchicalCache` class
   - `query_ttt_memory()` function (simple version)
   - Modified attention forward pass
   ```

3. **Test**:
   - Load existing checkpoint
   - Run inference with new cache
   - Compare quality vs standard cache

**Success criteria**: No crashes, reasonable output quality

---

### Phase 2: Architecture Refinement (1 week)

**Goal**: Optimize design decisions

1. **Experiments**:
   - Recent window size: 50, 100, 200, 500
   - TTT virtual tokens: 1, 10, 50, 100
   - Gating strategies: Fixed, learned, layer-dependent

2. **Metrics**:
   - Perplexity on long sequences (1k, 5k, 10k tokens)
   - Generation quality (human eval)
   - Memory usage
   - Inference speed

3. **Key questions to answer**:
   - What's the optimal recent window size?
   - How many virtual tokens does TTT need?
   - Where does quality degrade? (if at all)

---

### Phase 3: Training Integration (2 weeks)

**Goal**: Train with new cache mechanism

1. **Training challenges**:
   - TTT updates during training (gradient flow)
   - Backprop through cache operations
   - Curriculum: Start with short sequences, gradually increase

2. **Training strategy**:
   ```
   Stage 1: Freeze Moshi, train TTT compressor only (1k steps)
   Stage 2: Unfreeze recent few layers (5k steps)
   Stage 3: Full finetuning (10k steps)
   ```

3. **Datasets**:
   - Start: Existing DailyTalk (25s clips)
   - Progress: Concatenate clips → 1min sequences
   - Final: 5min+ long conversations

---

### Phase 4: Multi-Layer Deployment (1 week)

**Goal**: Scale to all layers

1. **Rollout strategy**:
   - Start: Apply to layers 25-31 (top layers)
   - Validate: Check quality doesn't degrade
   - Expand: Apply to more layers gradually

2. **Memory optimization**:
   - Share TTT compressor across layers?
   - Layer-specific recent window sizes?

---

### Phase 5: Production & Evaluation (Ongoing)

**Goal**: Real-world testing

1. **Benchmarks**:
   - Long-context metrics (paper metrics at 8k, 16k, 32k tokens)
   - Human evaluation
   - Conversation coherence over 10+ minutes

2. **Comparison baselines**:
   - Standard Moshi (context=3000)
   - Moshi with extended cache (context=8000)
   - Current TTT-hybrid (layer 30 only)
   - New TTT-cache hybrid

---

## 🔑 Critical Design Decisions

### Decision 1: How many virtual tokens from TTT?

**Options**:
- **1 token**: Most compressed, least information
- **10 tokens**: Balanced
- **100 tokens**: More expressive, higher cost

**Recommendation**: Start with 10, experiment with 1-50

**Rationale**:
- 10 tokens = 10 attention queries to memory
- Enough to capture multiple "concepts" from history
- Low overhead

---

### Decision 2: What does TTT compress?

**Option A: Compress K, V pairs directly**
```python
ttt_input = [k_old, v_old]  # Raw attention outputs
ttt_output = compressed_representation
```

**Option B: Compress hidden states**
```python
ttt_input = x_old  # Token embeddings before QKV projection
ttt_output = compressed_representation
```

**Recommendation**: Option B (compress hidden states)

**Rationale**:
- More flexible - TTT learns optimal compression
- x contains full information (K, V are projections)
- Easier to train

---

### Decision 3: When to update TTT?

**Option A: Every token** (Streaming)
```python
for token in sequence:
    process_token()
    if cache_full:
        update_ttt(oldest_token)
```

**Option B: Batch update** (Efficient)
```python
for token in sequence:
    process_token()
    if len(cache) % 100 == 0:
        update_ttt(oldest_100_tokens)
```

**Recommendation**: Option A for inference, Option B for training

**Rationale**:
- Inference: Need streaming for real-time
- Training: Batch updates more efficient

---

### Decision 4: TTT architecture for compression

**Option A: Small TTT-MLP** (2-3 layers)
- Fast, lightweight
- Dedicated to compression task

**Option B: Reuse existing TTT** (10 layers)
- More capacity
- But heavier

**Recommendation**: Option A

**Rationale**:
- Compression is different task than layer processing
- Want lightweight, fast updates
- Can be layer-specific

---

## 🧪 Key Experiments & Metrics

### Experiment 1: Proof of Concept

**Setup**:
- Single layer (30)
- Recent window: 100
- TTT virtual tokens: 10
- No training, just inference

**Metrics**:
- Does it run without crashing?
- Quality check: Generate 5k tokens
- Compare vs standard cache

**Expected outcome**: Slightly worse quality (untrained) but no catastrophic failure

---

### Experiment 2: Scaling Laws

**Question**: How does quality scale with recent window size?

**Setup**: Test recent_window = [50, 100, 200, 500, 1000]

**Metrics**:
- Perplexity at 5k tokens
- Memory usage
- Inference time per token

**Expected outcome**: Diminishing returns after 200-500

---

### Experiment 3: Training Effectiveness

**Question**: Can TTT learn better compression than fixed cache?

**Setup**:
- Train for 5k steps on long sequences (5min audio)
- Compare: TTT-cache vs extended fixed cache (context=8000)

**Metrics**:
- Paper metrics (sBLIMP, sstory, etc.) at 8k tokens
- Long-sequence perplexity (1k, 5k, 10k tokens)
- Qualitative: Conversation coherence

**Expected outcome**: TTT-cache should match or exceed fixed cache quality with lower memory

---

### Experiment 4: Unbounded Scaling

**Question**: Can it handle truly long contexts?

**Setup**: Generate 30k, 60k, 100k tokens (4-8 hours of audio!)

**Metrics**:
- Memory usage (should stay constant)
- Quality degradation curve
- Coherence over ultra-long contexts

**Expected outcome**: Graceful degradation, no hard limit

---

## 🚧 Implementation Challenges & Solutions

### Challenge 1: Gradient Flow Through TTT State

**Problem**: TTT state updated online - how to backprop through it?

**Solution Options**:

A. **Truncated BPTT** (Recommended)
- Backprop through recent N steps only
- TTT state treated as constant beyond N
- Standard in RNN training

B. **Synthetic Gradients**
- Predict gradient for TTT update
- Decouple from full backprop

C. **RL-style updates**
- Treat TTT compression as policy
- Optimize via reward (downstream task performance)

**Recommendation**: Start with A (truncated BPTT, N=100-500 steps)

---

### Challenge 2: Position Information

**Problem**: TTT state has no explicit position info - how does attention know "when"?

**Solution Options**:

A. **Relative position in virtual tokens**
- Each TTT virtual token gets relative position encoding
- token_0 = "oldest memory", token_9 = "recent memory"

B. **Time decay weighting**
- Weight virtual tokens by recency
- Attention naturally focuses on recent

C. **No explicit position**
- Let model learn implicitly

**Recommendation**: B + C - add time decay as optional, let model learn

---

### Challenge 3: Streaming Inference Compatibility

**Problem**: Moshi supports streaming (real-time generation) - must maintain this

**Solution**:
- TTT updates must be O(1) per token (not recompute all history)
- Already satisfied by TTT design (linear scan)
- Recent cache updates are O(1)

**Implementation**:
```python
# Streaming loop
while generating:
    token = generate_next()

    # O(1) updates
    cache.add_to_recent(token)  # O(1)
    if cache.recent_full():
        cache.compress_oldest_to_ttt()  # O(1) - single TTT update
```

**Verification**: Profile to ensure <5ms per token for cache operations

---

### Challenge 4: Memory Management

**Problem**: TTT state + recent cache - how much memory?

**Current (context=3000)**:
- 32 layers × 3000 tokens × 4096 dim × 2 (K+V) × 2 bytes (fp16)
- ≈ 1.6 GB per sample

**New (recent=100, TTT state)**:
- Recent cache: 32 × 100 × 4096 × 2 × 2 = 52 MB
- TTT state: 32 × 4096 dim × 4 bytes (fp32) = 0.5 MB
- **Total: ~53 MB** (30× reduction!)

**Trade-off**: Computation vs Memory
- More TTT updates (compute)
- Much less cache memory

---

## 📊 Success Criteria

### Minimum Viable Success (Phase 1)

- ✅ Runs without crashing
- ✅ Generates coherent audio for 5k+ tokens
- ✅ Memory usage <100MB per sample (vs 1.6GB baseline)

### Target Success (Phase 3)

- ✅ Quality matches standard cache up to 8k tokens
- ✅ Quality degrades gracefully beyond 8k tokens (not suddenly)
- ✅ Paper metrics within 2% of baseline at 8k tokens
- ✅ Can generate 30k+ tokens (4+ hours) without repetition

### Stretch Goals (Phase 5)

- ✅ Outperforms extended cache (context=8000) on long contexts
- ✅ Memory constant regardless of sequence length
- ✅ Enables 100k+ token generation (8+ hours of audio)
- ✅ Streaming latency <10ms per token

---

## 🔧 Code Structure

### New Files to Create

```
moshi-finetune/
├── moshi_ttt/
│   ├── hierarchical_cache.py          # NEW: Main cache implementation
│   ├── ttt_compressor.py              # NEW: TTT memory compressor
│   ├── cached_attention.py            # NEW: Modified attention with TTT cache
│   └── cache_utils.py                 # NEW: Helper functions
├── finetune/
│   ├── train_with_ttt_cache.py        # NEW: Training script
│   └── ttt_cache_args.py              # NEW: Config for TTT cache
└── tests/
    ├── test_hierarchical_cache.py     # NEW: Unit tests
    └── test_ttt_cache_integration.py  # NEW: Integration tests
```

### Modifications Needed

```
moshi_ttt/hybrid_layer.py:
  - Add option to use TTT cache instead of standard cache
  - Minimal changes, factory pattern

finetune/wrapped_model.py:
  - Add TTT cache configuration
  - Pass to model during creation

inference/run_inference_with_ttt.py:
  - Add flag --use_ttt_cache
  - Load TTT cache state if available
```

---

## 📅 Timeline Estimate

| Phase | Duration | Key Deliverables |
|-------|----------|-----------------|
| **Phase 1: PoC** | 3 days | Working prototype, basic tests |
| **Phase 2: Refinement** | 1 week | Optimized design, ablations |
| **Phase 3: Training** | 2 weeks | Trained model, quality metrics |
| **Phase 4: Multi-layer** | 1 week | Scaled to all layers |
| **Phase 5: Production** | 2 weeks | Benchmarks, evaluation |
| **Total** | **~6 weeks** | Production-ready system |

**Critical path items**:
1. Designing `query_ttt_memory()` interface (2-3 days thinking + prototyping)
2. Training recipe that actually improves over baseline (1-2 weeks experimentation)
3. Verifying streaming compatibility (2-3 days testing)

---

## 🎓 Research Questions to Answer

### Open Questions

1. **Expressiveness**: Can 10 virtual tokens from TTT really capture 10,000 tokens of history?
   - Need to measure: information retention, semantic similarity

2. **Training dynamics**: How quickly does TTT learn good compression?
   - Track: compression loss, downstream task performance over training

3. **Failure modes**: When/how does it break down?
   - Test: Adversarial sequences, edge cases, very long contexts

4. **Generalization**: Does TTT-trained on DailyTalk generalize to other domains?
   - Test: LibriSpeech, different speakers, different audio conditions

### Potential Publications

If successful, this could lead to:

1. **Workshop paper**: "Learned Memory Compression for Long-Context Speech Generation"
2. **Full paper**: "Hierarchical Cache with Test-Time Training for Unbounded Context in Autoregressive Models"
3. **Blog post**: Technical deep-dive on implementation

---

## 🔗 Related Work & Inspiration

### Papers to Read

1. **TTT Original Papers**:
   - "Learning to (Learn at Test Time): RNNs with Expressive Hidden States"
   - "Video-DiT: Video Generation with Test-Time Training"

2. **Long Context Methods**:
   - "Landmark Attention" (compress to landmarks)
   - "Memorizing Transformers" (external memory)
   - "∞-former" (continuous attention)

3. **Cache/Memory Compression**:
   - "Compressive Transformers"
   - "Perceiver" (learned compression of inputs)

### Similar Ideas in Literature

- **Compressive Transformers**: Compress old memories, but fixed compression (not learned)
- **Memorizing Transformers**: External memory bank, but uses k-NN (not learned retrieval)
- **Landmark Attention**: Select key tokens, but fixed selection strategy

**Our novelty**: Using TTT for **learned, adaptive** compression with **gradient-based** updates

---

## 💡 Key Insights & Intuitions

### Why This Should Work

1. **TTT is designed for this**: Online learning, compression, long sequences
2. **Hierarchical memory is proven**: Human memory works this way (working memory + long-term)
3. **Attention + compression is powerful**: Best of both worlds

### Why This Might Be Hard

1. **Interface challenge**: Attention and TTT are different paradigms
2. **Training complexity**: Gradients through online updates
3. **Cold start**: TTT needs data to learn compression

### How to Know If It's Working

**Early signs of success** (Phase 1):
- TTT virtual tokens have meaningful attention weights (not uniform)
- Attention sometimes focuses on TTT memory (not always recent)
- Quality doesn't catastrophically degrade

**Strong validation** (Phase 3):
- Perplexity curve is smooth (not sudden jump at N tokens)
- Can generate coherent 10k+ token sequences
- Paper metrics match or exceed baseline

**Publication-worthy** (Phase 5):
- Clearly outperforms baselines on long sequences
- Demonstrates unbounded scaling (100k+ tokens)
- Ablations show each component is necessary

---

## 🚀 Getting Started: Next Actions

### Immediate Next Steps (This Week)

1. **Read this document carefully** ✅
2. **Decide**: Is this worth pursuing? (vs just using context=8000)
3. **If yes**: Start Phase 1 PoC implementation
4. **Create**: `hierarchical_cache.py` skeleton

### First Code to Write

```python
# moshi_ttt/hierarchical_cache.py

class HierarchicalCache:
    """
    Hybrid cache: Recent KV pairs + TTT compressed memory.

    Design goals:
    - Drop-in replacement for standard KV cache
    - Streaming compatible (O(1) updates)
    - Minimal overhead
    """

    def __init__(self, recent_window=100, ttt_config=None):
        pass  # TODO

    def add_token(self, k, v):
        """Add new K, V pair to cache."""
        pass  # TODO

    def get_kv_for_attention(self):
        """Return K, V for attention computation."""
        pass  # TODO

    def compress_to_ttt(self):
        """Compress oldest tokens using TTT."""
        pass  # TODO
```

### Decision Point: After Phase 1 PoC

**Go/No-Go decision criteria**:
- ✅ **GO**: Quality within 10% of baseline, clear path to improvement
- ❌ **NO-GO**: Catastrophic quality loss, fundamental design flaws, too complex

If NO-GO, fall back to: TTT on more layers (proven approach)

---

## 📚 Appendix: Mathematical Formulation

### Standard Attention with KV Cache

```
Q_t = W_q @ x_t
K_t = W_k @ x_t
V_t = W_v @ x_t

Cache: K = [K_1, K_2, ..., K_{t-1}], V = [V_1, V_2, ..., V_{t-1}]

Attention:
  scores = Q_t @ K.T / sqrt(d_k)
  weights = softmax(scores)
  output = weights @ V

Memory: O(t × d)  where t = context length
```

### Proposed TTT-Cached Attention

```
Recent cache: K_recent = [K_{t-N}, ..., K_{t-1}], V_recent = similar

TTT state: s_t = TTT_update(s_{t-1}, x_{t-N-1})  # Compress oldest token

Virtual tokens from TTT:
  K_virtual = TT_to_K(s_t)  # [B, M, d]  where M << t
  V_virtual = TTT_to_V(s_t)  # [B, M, d]

Combined cache:
  K_combined = [K_virtual, K_recent]
  V_combined = [V_virtual, V_recent]

Attention:
  scores_recent = Q_t @ K_recent.T / sqrt(d_k)
  scores_virtual = Q_t @ K_virtual.T / sqrt(d_k)

  weights_recent = softmax(scores_recent)
  weights_virtual = softmax(scores_virtual)

  output_recent = weights_recent @ V_recent
  output_virtual = weights_virtual @ V_virtual

  output = alpha * output_recent + (1 - alpha) * output_virtual

Memory: O(N × d + M × d + |s_t|)
  where N = recent window (100), M = virtual tokens (10), |s_t| = TTT state size
  Much smaller than O(t × d) for large t!
```

---

## 🎯 Summary: What Makes This Exciting

1. **Solves real problem**: Cache wraparound at 3000 tokens is limiting
2. **Novel approach**: TTT for cache compression hasn't been done
3. **Practical impact**: Enables truly long conversations (hours)
4. **Minimal changes**: Can integrate with existing Moshi
5. **Research value**: Publishable if successful
6. **Fallback exists**: If fails, we have proven alternatives

**Risk**: Moderate - requires training and careful design
**Reward**: High - unbounded context with linear complexity

**Recommendation**:
- ✅ Worth pursuing as research project
- Start with Phase 1 PoC (3 days) to validate concept
- Re-evaluate after PoC before committing to full training

---

## 📞 Questions? Start Here:

1. Re-read "Proposed Architectures" section
2. Look at "Implementation Roadmap"
3. Check "Critical Design Decisions"
4. Review "Key Experiments"

**Then decide**: Phase 1 PoC or stick with context=8000 for now?
