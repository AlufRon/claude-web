# Persistent States: What You Lose vs What You Gain

**Date**: 2025-11-10
**Status**: Critical Design Decision
**Your Concern**: "Setting persistence false will make TTT not remember older forward passes..."

**You're absolutely right.** Let me explain the tradeoff honestly.

---

## The Core Question

**Should TTT weights carry information from earlier chunks to later chunks during training?**

This is THE fundamental architectural question for Moshi-TTT.

---

## Option A: persistent_states=False (What You LOSE)

### What Happens

```python
# Chunk 0 processing (File 1, 0-10s)
W_init = self.W1  # Base parameter
output_0 = forward(chunk_0)
  # Inside forward: W_init → [TTT adapts] → W_adapted_0
  # TTT adapts to chunk 0 (e.g., male speaker characteristics)
loss_0 = compute_loss(output_0)
loss_0.backward()
optimizer.step()  # Updates self.W1 (the base parameter)
# W_adapted_0 is DISCARDED ← No state persistence!

# Chunk 1 processing (File 1, 10-20s, SAME speaker)
W_init = self.W1  # Fresh start! Doesn't remember chunk 0's adaptation
output_1 = forward(chunk_1)
  # Inside forward: W_init → [TTT adapts] → W_adapted_1
  # TTT adapts to chunk 1 FROM SCRATCH, no memory of chunk 0
loss_1 = compute_loss(output_1)
loss_1.backward()
optimizer.step()
```

### What You LOSE

1. **No cross-chunk adaptation**:
   - Chunk 1 doesn't benefit from Chunk 0's TTT adaptation
   - Each chunk starts fresh from base weights
   - TTT re-learns speaker characteristics every 10 seconds

2. **No gradual refinement**:
   - Can't gradually adapt to speaker over multiple chunks
   - Can't build up speaker-specific weights
   - Each chunk is independent

3. **Potential performance loss**:
   - IF cross-chunk adaptation helps (unproven!)
   - Later chunks might have higher loss
   - Miss opportunity for within-file learning

### What You KEEP

1. **KV Cache still provides cross-chunk memory**:
   ```python
   # Chunk 1 attention still sees Chunk 0 tokens!
   attn_out = attention(
       query_chunk1,
       keys_from_chunks_0_and_1,  # ← Still has Chunk 0 info!
       values_from_chunks_0_and_1
   )
   ```
   - Explicit token-level memory (last 3000 tokens)
   - Not lost! Only TTT weight-level memory is lost

2. **Outer loop learning**:
   - Optimizer still learns from all chunks
   - `self.W1` (base parameter) improves over training
   - Learns good initialization that works across diverse chunks

### Example: What Actually Happens

```
File 1, 60 seconds, 6 chunks:

Chunk 0 (male speaker):
  W_base → TTT adapts to male → W_male
  Uses W_male for this chunk
  Discards W_male after ✗

Chunk 1 (same male speaker):
  W_base → TTT adapts to male AGAIN → W_male'
  Has to re-learn male characteristics ✗
  But: Attention sees chunk 0 tokens (KV cache) ✓
  But: W_base improves from gradient (outer loop) ✓

Chunk 2 (same male speaker):
  W_base → TTT adapts to male AGAIN → W_male''
  Still re-learning ✗
  But: Attention sees chunks 0-1 tokens ✓
  But: W_base keeps improving ✓
```

**The question**: Is re-adapting each chunk wasteful? Or is fresh adaptation actually better?

---

## Option B: persistent_states=True (What You GAIN)

### What Happens (WITH PROPER FIXES!)

```python
# Chunk 0 processing (File 1, 0-10s)
W_init = self.W1_base  # Base parameter (trainable)
W_state = self.W1_state  # State buffer (persistent)
W_state.copy_(W_init)  # Initialize state from base

output_0 = forward(chunk_0)
  # Inside forward: W_state → [TTT adapts] → W_state'
  # TTT adapts to chunk 0
  # self.W1_state.copy_(W_state')  ← State persists!
loss_0 = compute_loss(output_0)
loss_0.backward()  # Gradient flows to W1_base (not W1_state)
optimizer.step()  # Updates W1_base only

# Chunk 1 processing (File 1, 10-20s, SAME speaker)
W_state = self.W1_state  # Contains W_state' from chunk 0! ✓
output_1 = forward(chunk_1)
  # Inside forward: W_state' → [TTT continues] → W_state''
  # TTT continues adapting, builds on chunk 0's adaptation ✓
loss_1 = compute_loss(output_1)
loss_1.backward()  # Gradient still flows to W1_base
optimizer.step()

# File boundary (File 2 starts)
reset_ttt_states(model)  # W_state.copy_(W1_base) ← Reset!
# Now File 2 starts fresh
```

### What You GAIN

1. **Cross-chunk adaptation**:
   - Chunk 1 continues from Chunk 0's adapted weights
   - No re-learning of speaker characteristics
   - Gradual refinement across chunks

2. **Potentially lower loss on later chunks**:
   - IF adaptation helps
   - IF speaker characteristics persist
   - IF TTT can encode useful patterns in weights

3. **More like original TTT vision**:
   - Test-time training that adapts continuously
   - State carries forward
   - Memory in weights, not just KV cache

### What You MUST IMPLEMENT

**CRITICAL**: persistent_states=True requires proper fixes!

#### Fix A: Separate W_base and W_state

**Current (BROKEN)**:
```python
class TTTMLP:
    def __init__(self):
        self.W1 = nn.Parameter(...)  # Single parameter

    def ttt(self, inputs):
        W = self.W1  # Read parameter
        W_adapted = ttt_inner_loop(W, inputs)

        self.W1.data.copy_(W_adapted)  # ← BUG! Overwrite parameter
        # Gradient: ∂loss/∂W_initial
        # But self.W1 now contains W_adapted!
        # Optimizer: W_adapted = W_adapted - lr × ∂loss/∂W_initial
        # MISMATCH!
```

**Fixed (CORRECT)**:
```python
class TTTMLP:
    def __init__(self):
        # Base weights (trainable by optimizer)
        self.W1_base = nn.Parameter(...)

        # State buffer (NOT trainable, just persistent)
        self.register_buffer('W1_state', self.W1_base.clone())

    def reset_ttt_state(self):
        """Call at file boundaries"""
        with torch.no_grad():
            self.W1_state.copy_(self.W1_base)

    def ttt(self, inputs):
        # Initialize from state buffer
        W = self.W1_state.clone()

        # TTT inner loop (differentiable!)
        W_adapted = ttt_inner_loop(W, inputs)

        # Update state buffer (no gradient)
        with torch.no_grad():
            self.W1_state.copy_(W_adapted)

        # Return output (gradient flows through W_adapted → W → W1_base) ✓
        return output
```

**Key differences**:
- `W1_base`: nn.Parameter, optimizer updates it
- `W1_state`: nn.Buffer, optimizer ignores it
- Gradient: `∂loss/∂W1_base` (correct!)
- State: Persists in `W1_state` (correct!)

#### Fix B: File Boundary Detection

```python
# In training loop
previous_file_id = None

for batch in data_loader:
    if batch.file_id != previous_file_id:
        # File boundary detected!
        reset_ttt_states(model)  # Reset state to base
        previous_file_id = batch.file_id

    output = model(batch.codes)
    loss = compute_loss(output)
    loss.backward()
    optimizer.step()
```

### Example: What Actually Happens (WITH FIXES)

```
File 1, 60 seconds, 6 chunks:

Chunk 0 (male speaker):
  W_state ← W_base
  W_base → TTT adapts → W_male
  W_state ← W_male  ✓ (persists)
  Gradient: ∂loss/∂W_base (correct)

Chunk 1 (same male speaker):
  W_state = W_male  ✓ (continues from chunk 0!)
  W_male → TTT continues → W_male'
  W_state ← W_male'  ✓
  Gradient: ∂loss/∂W_base (correct)

  Result: Doesn't re-learn male characteristics! ✓

Chunk 2 (same male speaker):
  W_state = W_male'  ✓ (continues!)
  W_male' → TTT refines → W_male''
  W_state ← W_male''  ✓

[File 1 ends]

[File 2 starts]
File boundary detected! → reset_ttt_states()
  W_state ← W_base  ✓ (reset for new file)

Chunk 0 (female speaker):
  W_state = W_base  ✓ (fresh start, not contaminated!)
  W_base → TTT adapts → W_female
  W_state ← W_female  ✓
```

---

## The Honest Comparison

| Aspect | persistent_states=False | persistent_states=True (WITH FIXES) |
|--------|-------------------------|-------------------------------------|
| **Cross-chunk adaptation** | ❌ No | ✅ Yes |
| **Re-learn each chunk** | ⚠️ Yes (might be wasteful) | ✅ No (continues adapting) |
| **KV cache memory** | ✅ Still works | ✅ Still works |
| **Outer loop learning** | ✅ Still works | ✅ Still works |
| **Implementation complexity** | ✅ Simple | ⚠️ Complex (W_base/W_state separation) |
| **Bugs** | ✅ None | ⚠️ Must fix Issues #4 and #5 properly |
| **Training time** | ✅ Faster (simpler) | ⚠️ Slower (state management overhead) |
| **Video-DiT proven** | ✅ Yes | ⚠️ No (Moshi-specific) |
| **File boundary handling** | ✅ Not needed | ⚠️ Must detect and reset |
| **Gradient flow** | ✅ Clean | ✅ Clean (if fixed properly) |
| **Benefit proven** | ✅ Video-DiT works great | ❌ **UNKNOWN** - needs experiments! |

---

## The CRITICAL Unknown

**Does cross-chunk TTT adaptation actually help?**

### Arguments FOR (persistent_states=True might help):

1. **Speaker consistency**: Same speaker across chunks
   - Might benefit from speaker-specific weights
   - Avoid re-learning speaker characteristics
   - Faster adaptation to speaker

2. **Gradual refinement**: Cumulative learning
   - Chunk 2 benefits from Chunk 0+1 adaptations
   - Build up patterns over time
   - More information encoded in weights

3. **Beyond KV cache**: Weights encode patterns differently
   - KV cache: Explicit tokens (discrete memory)
   - TTT weights: Learned patterns (distributed memory)
   - Complementary information?

### Arguments AGAINST (persistent_states=False might be better):

1. **KV cache redundancy**:
   - Already provides cross-chunk memory
   - Explicit token storage
   - TTT weight memory might be redundant

2. **Fresh adaptation**:
   - Starting fresh might be BETTER
   - Avoids accumulating errors
   - Clean slate for each chunk

3. **Outer loop learns**:
   - W_base improves from all chunks
   - Already learns cross-chunk patterns
   - No need for state persistence

4. **Video-DiT proof**:
   - Works excellently without persistence
   - Proven approach
   - Why add complexity?

### The DATA We Don't Have

**WE DON'T KNOW which is better because experiments haven't been run!**

Specifically, we need:
```python
# Experiment: Compare per-chunk loss

Model A: persistent_states=False
Model B: persistent_states=True (with proper fixes)

For each file with N chunks:
    losses_A = []
    losses_B = []

    for chunk_i in file.chunks:
        loss_A = compute_loss(model_A(chunk_i))
        loss_B = compute_loss(model_B(chunk_i))

        losses_A.append(loss_A)
        losses_B.append(loss_B)

    # Question: Does loss_B decrease across chunks?
    # Expected if persistence helps:
    #   losses_B[0] > losses_B[1] > losses_B[2] > ...
    # Expected if no benefit:
    #   losses_B[i] ≈ losses_B[j] for all i,j
```

**This experiment has NOT been run!**

---

## My Honest Recommendation

### Phase 1: Quick Fix (This Week)

**Set `persistent_states=False`**

**Why**:
- ✅ Fixes all bugs immediately (Issues #3, #4, #5)
- ✅ Simple, proven approach (Video-DiT)
- ✅ Get a working baseline FAST
- ✅ No complex fixes needed
- ✅ Works with existing checkpoints

**Downside**:
- ⚠️ Loses cross-chunk TTT adaptation (unproven if valuable)
- ⚠️ Might miss performance gains (if they exist)

**Time**: 2 minutes to change, 1 week to train

### Phase 2: Proper Comparison (Weeks 2-4)

**Implement persistent_states=True WITH PROPER FIXES**

**Changes needed**:
1. Separate W_base (nn.Parameter) and W_state (nn.Buffer)
2. File boundary detection and reset
3. Gradient flow verification
4. Testing

**Time**: 1-2 weeks to implement and test

**Then RUN THE EXPERIMENT**:
- Train Model A (persistent_states=False) - already done in Phase 1
- Train Model B (persistent_states=True with fixes)
- Compare per-chunk losses
- Statistical significance test

**Possible outcomes**:

#### Outcome 1: Model B significantly better (p < 0.05, effect > 10%)
→ **Use persistent_states=True**
→ Complexity justified by performance gain
→ Validates original design

#### Outcome 2: Model A and B similar (p > 0.05)
→ **Use persistent_states=False**
→ Simpler is better when performance equal
→ Video-DiT approach wins

#### Outcome 3: Model B worse than Model A
→ **Use persistent_states=False**
→ Persistence actively hurts
→ Fresh adaptation is better

---

## The Two Paths Forward

### Path A: Simple & Fast (My recommendation for immediate action)

```
Week 1:
✅ Set persistent_states=False
✅ Fix normalization bug (Issue #2)
✅ Train baseline model
✅ Get working system

Week 2:
✅ Measure performance
✅ Establish baseline metrics

Week 3-4:
✅ Implement persistent_states=True properly (W_base/W_state)
✅ Train comparison model
✅ Run experiments
✅ Make evidence-based decision
```

**Pros**: Working system in 1 week, evidence-based decision by week 4

### Path B: Complex First (Higher risk)

```
Week 1-2:
⚠️ Implement W_base/W_state separation
⚠️ Implement file boundary detection
⚠️ Debug gradient flow
⚠️ Test thoroughly

Week 3:
✅ Train with persistent_states=True
✅ Hope it works

Week 4+:
❓ If bugs appear, debug more
❓ If performance bad, wasted 3 weeks
❓ No baseline to compare against
```

**Cons**: High risk, no baseline, might waste 3+ weeks

---

## What You're Really Asking

**"Will TTT be effective without cross-chunk memory?"**

**Honest answer**:

✅ **YES, it can still be effective** because:
1. KV cache provides cross-chunk token-level memory
2. Outer loop (optimizer) learns from all chunks
3. Video-DiT proves TTT works excellently without persistence
4. TTT still adapts within each chunk (mini-batch level)

⚠️ **BUT, it MIGHT be less effective** because:
1. Loses weight-level cross-chunk memory
2. Re-adapts to speaker every chunk (possibly wasteful)
3. Can't build up gradual refinement
4. Original TTT vision includes persistent adaptation

❓ **WE DON'T KNOW which is actually better** because:
1. Experiments haven't been run
2. Learning signal for persistence is unclear (see LEARNING_INCENTIVES_ANALYSIS.md)
3. KV cache might make weight persistence redundant
4. Fresh adaptation might actually be better

---

## My Final Recommendation

### Immediate Action (Phase 1)

**Set `persistent_states=False` and fix normalization bug**

This gives you:
- ✅ Working system in days, not weeks
- ✅ Clean baseline
- ✅ No bugs
- ✅ Proven approach

### Then Validate (Phase 2)

**Implement persistent_states=True properly and compare**

This gives you:
- ✅ Evidence-based decision
- ✅ Quantitative comparison
- ✅ Know what you're giving up (or gaining)
- ✅ No regrets

### Decision Tree

```
IF experiments show persistent_states=True significantly better:
    → Use it (complexity justified)
    → You were right to push back on my recommendation!

ELIF experiments show similar performance:
    → Use persistent_states=False (simpler)
    → Video-DiT approach validated

ELIF experiments show persistent_states=True worse:
    → Use persistent_states=False (proven better)
    → Fresh adaptation is superior
```

---

## You're Right to Question This

Your instinct is correct: **Setting persistent_states=False does lose cross-chunk TTT memory**.

The question is: **Is that memory valuable?**

Video-DiT says: No (works great without it)
Original TTT design says: Yes (intended for continuous adaptation)
Data says: ??? (experiments not run)

**Let's find out!** Start with simple (Phase 1), then test complex (Phase 2), then decide based on data.

---

**Bottom line**:
- You lose cross-chunk TTT weight memory
- You keep cross-chunk token memory (KV cache)
- You keep outer loop learning (optimizer)
- **Whether this matters is an empirical question**

Let's get Phase 1 working (1 week), then test Phase 2 properly (weeks 2-4), then make evidence-based decision.

---

**Document Version**: 1.0
**Date**: 2025-11-10
**Status**: Honest assessment of tradeoffs
