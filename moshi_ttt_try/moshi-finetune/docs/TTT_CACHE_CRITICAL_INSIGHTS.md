# TTT as Cache: Critical Insights & Architecture

## 🎯 The Core Idea (One Paragraph)

Instead of dropping old tokens when KV cache fills (causing context loss at 3000 tokens), **compress them using TTT into a learned representation**. Recent tokens stay in standard KV cache for precise attention (local context). Old tokens are compressed by TTT into "virtual tokens" that attention can still query (global context). This gives us **unbounded memory** with **minimal architectural changes**.

---

## 🏗️ Recommended Architecture: Hierarchical Cache

```
┌─────────────────────────────────────────────────────────────┐
│                     Current Token x_t                        │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │  Attention   │
                    └──────┬───────┘
                           │
          ┌────────────────┴────────────────┐
          │                                 │
          ▼                                 ▼
  ┌───────────────┐                ┌─────────────────┐
  │  Recent Cache │                │   TTT Memory    │
  │   (Last 100   │                │   (Compressed   │
  │    tokens)    │                │     History)    │
  │               │                │                 │
  │  K_recent     │                │  TTT State      │
  │  V_recent     │                │      ↓          │
  │               │                │  Project to:    │
  │  Precise      │                │  K_virtual [10] │
  │  Local Attn   │                │  V_virtual [10] │
  └───────────────┘                └─────────────────┘
         ↓                                  ↓
         └──────────┬───────────────────────┘
                    ▼
              Gated Combine:
         α·attn_recent + (1-α)·attn_old
```

### Why This Works

1. **Recent tokens** (0-100 back): Full KV cache → Precise attention for local context
2. **Old tokens** (100+ back): TTT compressed → Approximate attention for distant context
3. **No hard cutoff**: Gradual transition, not sudden context loss
4. **Streaming compatible**: O(1) updates per token

---

## 🔑 5 Critical Design Decisions

### 1️⃣ How Many Virtual Tokens from TTT?

**Question**: TTT state → How many K, V pairs for attention?

**Options**:
- 1 token: Most compressed (entire history → 1 attention query)
- 10 tokens: **RECOMMENDED** - Balanced
- 100 tokens: More expressive but expensive

**Key Insight**:
- With 10 virtual tokens, attention can ask 10 "questions" to compressed memory
- Each virtual token represents a learned "summary aspect" of history
- Example: token_0 = overall topic, token_1 = speaker style, token_2 = recent emotions, etc.

**Critical to notice**:
- This is a **learned compression** - TTT figures out what to remember
- Unlike fixed compression (e.g., average), TTT adapts to what matters downstream
- During training, gradients tell TTT: "compress in a way that helps attention"

---

### 2️⃣ TTT Compressor Architecture

**Question**: What should TTT look like?

**Recommendation**: Small dedicated TTT-MLP (2-3 layers)

```python
class TTTCompressor(nn.Module):
    def __init__(self, dim=4096, num_virtual=10):
        self.ttt_mlp = TTT_MLP(
            input_dim=dim,
            hidden_dim=512,
            num_layers=2,
            mini_batch_size=1  # Online updates
        )
        self.state_to_kv = nn.Linear(dim, 2 * num_virtual * head_dim)

    def compress(self, old_token_embedding, current_ttt_state):
        # Update TTT state with old token
        new_state = self.ttt_mlp.update(current_ttt_state, old_token_embedding)

        # Project state to virtual K, V pairs
        kv = self.state_to_kv(new_state)  # [B, 2*10*128]
        k_virtual, v_virtual = kv.chunk(2, dim=-1)
        k_virtual = k_virtual.reshape(B, 10, 128)  # 10 virtual tokens
        v_virtual = v_virtual.reshape(B, 10, 128)

        return new_state, k_virtual, v_virtual
```

**Why small?**:
- Fast online updates (<1ms)
- Dedicated to compression task
- Can be layer-specific if needed

**Critical to notice**:
- TTT updates are **gradient-based** during training
- At inference, TTT still adapts (test-time training!) to current sequence
- This is different from static compression methods

---

### 3️⃣ Recent Window Size

**Question**: How many recent tokens in KV cache?

**Recommendation**: 100-200 tokens

**Reasoning**:
- 100 tokens ≈ 8 seconds of audio (at 12.5 Hz)
- Enough for immediate context (current sentence/utterance)
- Small enough to be efficient

**Critical to notice**:
- There's a **trade-off space** here:
  - Larger window → More precise, but more memory
  - Smaller window → More compression needed from TTT
- Optimal depends on task: conversation needs less local precision than transcription

**Ablation to run**: Test window = [50, 100, 200, 500]

---

### 4️⃣ Attention Combination Strategy

**Question**: How to combine attention over recent vs TTT memory?

**Option A: Separate then gate** (RECOMMENDED)
```python
attn_recent = softmax(Q @ K_recent.T) @ V_recent
attn_old = softmax(Q @ K_virtual.T) @ V_virtual
output = α * attn_recent + (1-α) * attn_old
```

**Option B: Concatenate then attend**
```python
K_combined = concat([K_virtual, K_recent])
V_combined = concat([V_virtual, V_recent])
output = softmax(Q @ K_combined.T) @ V_combined
```

**Recommendation**: Start with A, experiment with B

**Why A?**:
- α is **learnable** - model learns how much to trust each
- Separate softmax = separate attention distributions
- More controllable

**Critical to notice**:
- With Option A, attention weights over recent and old are **independent**
- Model can attend strongly to both, or prefer one over the other
- With Option B, attention weights **compete** (sum to 1 across both)

---

### 5️⃣ Training Strategy

**Question**: How to train this architecture?

**Recommendation**: 3-stage curriculum

```
Stage 1: Freeze Moshi, train TTT compressor only
  - Loss: Reconstruction of old K,V from TTT virtual tokens
  - Goal: TTT learns basic compression
  - Duration: 1k steps

Stage 2: Unfreeze attention, freeze lower layers
  - Loss: Standard language modeling
  - Goal: Attention learns to query TTT memory
  - Duration: 5k steps

Stage 3: Full finetuning
  - Loss: Standard language modeling
  - Goal: End-to-end optimization
  - Duration: 10k steps
```

**Critical to notice**:
- **Can't train end-to-end from scratch** - too many moving parts
- Need to initialize TTT compression first
- Curriculum goes from short sequences → long sequences

**Sequence length curriculum**:
```
Steps 0-1k:    512 tokens  (TTT sees ~400 old tokens)
Steps 1k-3k:   1024 tokens (TTT sees ~900 old tokens)
Steps 3k-5k:   2048 tokens (TTT sees ~1900 old tokens)
Steps 5k+:     4096+ tokens (TTT sees full history)
```

---

## ⚠️ Critical Things to Notice During Implementation

### 1. Gradient Flow

**Issue**: TTT state is updated online (one token at a time). How do gradients flow back?

**Solution**: Truncated BPTT (Backprop Through Time)
```python
# Only backprop through last N TTT updates
# Treat earlier TTT states as constants

# Example: N=100
for i, token in enumerate(sequence):
    with torch.no_grad() if i < len(sequence)-100 else contextlib.nullcontext():
        ttt_state = ttt.update(ttt_state, token)
```

**Why this matters**:
- Full BPTT through entire sequence = O(n²) memory (defeats the purpose!)
- Truncated BPTT = O(N) memory where N is fixed (e.g., 100)
- Still gets useful gradients for recent history

---

### 2. Position Information

**Issue**: TTT virtual tokens have no explicit position. How does attention know they're "old"?

**Solutions**:

A. **Relative position encoding per virtual token**
```python
# Add position encoding to each virtual token
k_virtual[0] += position_encoding(-1000)  # Very old
k_virtual[1] += position_encoding(-900)
...
k_virtual[9] += position_encoding(-100)   # Less old
```

B. **Time decay weighting** (simpler)
```python
# Weight virtual token attention by recency
decay = torch.exp(-torch.arange(10) * 0.1)  # [1.0, 0.9, ..., 0.4]
attn_old = attn_old * decay
```

C. **No explicit position** (let model learn)

**Recommendation**: Try B first, then experiment with A

**Critical observation**:
- Virtual tokens should be **ordered by time** (token_0 = oldest)
- But exact position doesn't matter - just relative "oldness"
- Model might learn to ignore very old virtual tokens naturally

---

### 3. Cold Start Problem

**Issue**: At start of sequence, TTT state is uninitialized. What to do?

**Solutions**:

A. **Zero initialization**
```python
ttt_state = torch.zeros(B, dim)
```

B. **Learned initialization**
```python
ttt_state = self.learned_init.expand(B, -1)
```

C. **No TTT initially**
```python
if num_tokens < recent_window:
    use_only_recent_cache()
else:
    use_hierarchical_cache()
```

**Recommendation**: C is safest

**Critical observation**:
- First ~100 tokens: TTT does nothing (no old context yet)
- This is fine! Recent cache handles it
- TTT only activates once we exceed recent window

---

### 4. Memory Efficiency Validation

**Check these numbers**:

**Standard Moshi (context=3000)**:
```
32 layers × 3000 tokens × 4096 dim × 2 (K+V) × 2 bytes (fp16)
= 1,572,864,000 bytes
≈ 1.5 GB per sample
```

**TTT-Cached Moshi (recent=100, virtual=10)**:
```
Recent cache: 32 × 100 × 4096 × 2 × 2 = 52,428,800 bytes ≈ 50 MB
TTT state: 32 × 4096 × 4 bytes (fp32) = 524,288 bytes ≈ 0.5 MB
Virtual K,V: 32 × 10 × 128 × 2 × 2 bytes = 163,840 bytes ≈ 0.16 MB

Total: ~51 MB (30× reduction!)
```

**Critical to verify**:
- Profile actual memory usage
- Make sure no memory leaks
- Check GPU memory growth over time

---

### 5. Streaming Compatibility

**Requirement**: Must work in streaming mode (real-time generation)

**Check**:
```python
# Each operation must be O(1) per token

✅ add_to_recent_cache(token)           # O(1) - deque append
✅ ttt_compressor.update(state, token)  # O(1) - single TTT step
✅ project_ttt_to_virtual_kv(state)     # O(1) - linear projection
✅ attention_over_combined(Q, K, V)     # O(N+M) where N,M are small

❌ recompute_all_ttt_state()            # O(t) - FORBIDDEN!
```

**Critical test**:
- Measure latency per token
- Should be <5ms for cache operations
- Profile with long sequences (10k+ tokens)

---

## 🧪 Key Validation Experiments

### Experiment 1: Does TTT Actually Compress?

**Goal**: Verify TTT learns meaningful compression

**Test**:
```python
# After training, check:
1. Can you reconstruct information from virtual tokens?
   - Predict token IDs from TTT virtual tokens
   - Should be better than random, worse than exact

2. Are virtual tokens diverse?
   - Measure cosine similarity between k_virtual[0], k_virtual[1], ...
   - Should NOT all be the same

3. Do attention weights make sense?
   - Plot attention over [virtual_0, ..., virtual_9, recent_0, ..., recent_99]
   - Should see smooth transition, not ignoring all virtual tokens
```

---

### Experiment 2: Does Compression Help?

**Goal**: Verify compression is better than alternatives

**Baselines to compare**:
```
A. Standard cache (context=3000)     - Hard limit
B. Extended cache (context=8000)     - More memory
C. Average pooling (old tokens → 1)  - Fixed compression
D. TTT cache (our method)            - Learned compression
```

**Metrics**:
- Perplexity at 5k, 10k, 20k tokens
- Memory usage
- Quality of long conversations

**Expected**:
- D should beat A (no hard limit)
- D should match B with less memory
- D should beat C (learned > fixed)

---

### Experiment 3: Scaling Behavior

**Goal**: Verify no sudden quality drops

**Test sequence lengths**: [1k, 2k, 5k, 10k, 20k, 50k, 100k tokens]

**Plot**:
```
Y-axis: Perplexity
X-axis: Sequence length

Expected curves:
- Standard cache: Flat until 3k, then JUMP, then flat (higher)
- Extended cache: Flat until 8k, then JUMP
- TTT cache: Smooth gradual increase (no jumps!)
```

**Critical observation**:
- Sudden jumps = hard failures (bad!)
- Gradual degradation = graceful (good!)
- TTT should show graceful degradation

---

## 🚨 Potential Failure Modes

### Failure Mode 1: TTT Ignores Old Context

**Symptom**: α ≈ 1.0 (only uses recent cache)

**Diagnosis**:
```python
# Check gating weight
print(f"α = {model.memory_gating.item()}")  # Should be 0.3-0.7

# Check attention to virtual tokens
attn_weights_to_virtual = attention[:, :10].mean()  # Should be >0.1
```

**Fixes**:
- Regularize α toward 0.5
- Pretrain TTT compressor better
- Increase number of virtual tokens

---

### Failure Mode 2: Virtual Tokens All Identical

**Symptom**: k_virtual[0] ≈ k_virtual[1] ≈ ... ≈ k_virtual[9]

**Diagnosis**:
```python
# Compute pairwise similarity
K = k_virtual  # [B, 10, 128]
sim = torch.cosine_similarity(K[:, :, None], K[:, None, :], dim=-1)
print(sim.mean())  # Should be <0.8
```

**Fixes**:
- Add diversity loss: `loss += -0.1 * sim.mean()`
- Use different initialization for each virtual token
- Increase hidden dimension in TTT

---

### Failure Mode 3: NaN/Inf in TTT State

**Symptom**: Model crashes during long sequences

**Diagnosis**:
```python
# Add assertions
assert torch.isfinite(ttt_state).all(), "TTT state has NaN/Inf!"
assert torch.isfinite(k_virtual).all(), "Virtual K has NaN/Inf!"
```

**Fixes**:
- Gradient clipping on TTT updates
- Layer normalization on TTT state
- Lower TTT learning rate

---

## 🎯 Success Metrics Summary

| Metric | Target | Why It Matters |
|--------|--------|----------------|
| **Memory** | <100 MB vs 1.5 GB baseline | Proves efficiency |
| **Perplexity @ 5k tokens** | Within 5% of extended cache | Proves quality |
| **Perplexity @ 20k tokens** | Better than standard cache | Proves unbounded benefit |
| **No quality jumps** | Smooth degradation curve | Proves graceful failure |
| **Attention to virtual** | >10% attention weight | Proves TTT is used |
| **Virtual diversity** | Cosine sim <0.8 | Proves meaningful compression |
| **Streaming latency** | <5ms per token | Proves practical |

---

## 📋 Implementation Checklist

### Phase 1: Proof of Concept (3 days)

- [ ] Implement `HierarchicalCache` class
- [ ] Implement `TTTCompressor` module
- [ ] Implement `query_ttt_memory()` function
- [ ] Modify attention to use hierarchical cache
- [ ] Test: Run inference without crashing
- [ ] Test: Generate 5k tokens and check quality
- [ ] Measure: Memory usage vs baseline
- [ ] **Decision point**: Go/no-go for Phase 2

### Phase 2: Experiments (1 week)

- [ ] Ablation: Recent window size (50, 100, 200, 500)
- [ ] Ablation: Virtual tokens (1, 5, 10, 20, 50)
- [ ] Ablation: TTT compressor size (2, 3, 5 layers)
- [ ] Ablation: Gating strategy (fixed, learned, per-layer)
- [ ] Metrics: Perplexity curves on long sequences
- [ ] Metrics: Memory and speed profiling
- [ ] **Decision point**: Best configuration found

### Phase 3: Training (2 weeks)

- [ ] Setup: Training data pipeline for long sequences
- [ ] Stage 1: Train TTT compressor (1k steps)
- [ ] Stage 2: Finetune attention (5k steps)
- [ ] Stage 3: Full finetuning (10k steps)
- [ ] Validation: Paper metrics at 8k, 16k tokens
- [ ] Validation: Qualitative evaluation (listen to outputs)
- [ ] **Decision point**: Beats baseline?

### Phase 4+: Production (ongoing)

- [ ] Scale to all layers
- [ ] Optimize for speed
- [ ] Comprehensive benchmarks
- [ ] Write paper/blog post

---

## 🎓 Final Insights

### What Makes This Hard

1. **Interface design**: Bridging attention and TTT paradigms
2. **Training dynamics**: Many hyperparameters to tune
3. **Validation**: Hard to know if working until fully trained

### What Makes This Exciting

1. **Novel approach**: Learned cache compression hasn't been done
2. **Practical impact**: Enables truly long conversations
3. **Research value**: Publishable if successful
4. **Elegant design**: Minimal changes to existing architecture

### The Core Bet

**Hypothesis**: TTT can learn a better compression of old context than simple strategies (averaging, subsampling) because it's trained end-to-end with gradients from the downstream task (language modeling).

**If true**: We get unbounded context with minimal quality loss

**If false**: We fall back to extended cache (context=8000) - still an improvement

### Decision Matrix

| Outcome | Action |
|---------|--------|
| Phase 1 PoC works well | → Continue to Phase 2 |
| Phase 1 PoC has issues | → Debug or pivot to TTT on more layers |
| Phase 2 experiments promising | → Continue to Phase 3 training |
| Phase 2 no clear winner | → Reconsider or try hybrid approaches |
| Phase 3 training beats baseline | → Scale up, publish! |
| Phase 3 training doesn't improve | → Analyze failure, iterate or stop |

---

## 🚀 Ready to Start?

**First step**: Implement the skeleton in Phase 1 checklist

**Key file to create**: `moshi_ttt/hierarchical_cache.py`

**Start simple**: Recent window=100, virtual tokens=10, no training yet

**Validation**: Can it generate 5k tokens without crashing? How's the quality?

**Then decide**: Continue or pivot?

Good luck! This is ambitious but feasible. 🎉
