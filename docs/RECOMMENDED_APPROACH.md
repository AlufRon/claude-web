# Recommended Approach: Adding TTT to Moshi

**Date**: 2025-11-10
**Status**: Architectural Recommendation
**Branch**: `claude/deep-code-review-ttt-011CUzpCD2kNLGH5UnuHN7hF`

---

## Executive Summary

**Bottom Line**: Start with **Video-DiT's proven approach** (no persistent states), then add complexity only if measurements demonstrate clear benefit.

**Rationale**:
1. Video-DiT proves TTT works excellently WITHOUT persistent states
2. Current persistent states have 5 critical bugs and weak learning signal
3. Moshi's KV cache already provides cross-chunk memory
4. Scientific method: Start simple, add complexity only when data demands it

---

## Core Insight: Separate Memory from Adaptation

**The key realization**: TTT should focus on **LOCAL ADAPTATION**, not memory.

| Component | Purpose | Scope |
|-----------|---------|-------|
| **KV Cache** | Memory (store previous tokens) | Last 3000 tokens (~12.5s) |
| **Attention** | Information retrieval (query relevant history) | Global |
| **TTT** | Fast adaptation (learn local patterns) | Current mini-batch |

**Current problem**: TTT is trying to do memory (via persistent states), but KV cache already does this better!

**Better approach**: Let each component do what it's best at.

---

## Recommended Phased Approach

### Phase 1: Simplify & Stabilize (Week 1-2) 🎯 **PRIORITY**

**Goal**: Get a working, bug-free TTT-Moshi baseline

#### Changes:

**1.1 Fix Critical Bugs**

```python
# File: moshi_ttt_try/moshi-finetune/moshi_ttt/ttt_layer.py

# Fix Issue #2: Normalization
def ln_reconstruction_target(self, reconstruction_target):
    if self.reconstruction_target_norm:
        return F.layer_norm(
            reconstruction_target,  # ← Now receives (XV - XK), not XV!
            normalized_shape=(reconstruction_target.size(-1),),
            weight=None, bias=None, eps=1e-5,
        )
    return reconstruction_target

# In compute_mini_batch:
reconstruction_target = XV_mini_batch - XK_mini_batch
reconstruction_target = self.ln_reconstruction_target(reconstruction_target)  # ✓ Correct!
```

**1.2 Disable Persistent States (Video-DiT Style)**

```python
# File: moshi_ttt_try/moshi-finetune/finetune/args.py

@dataclass
class TTTArgs(Serializable):
    enable: bool = True
    layers: list[int] = field(default_factory=lambda: [0, 1, 2])

    # ✓ CHANGE: Default to False for training
    persistent_states: bool = False  # ← Changed from True

    mini_batch_size: int = 8
    eta: float = 1.0
    reconstruction_target_norm: bool = True
```

**1.3 Add Batch Size Validation**

```python
# File: moshi_ttt_try/moshi-finetune/moshi_ttt/ttt_layer.py

def ttt(self, inputs):
    B, H, NC, C, D = inputs.shape

    # Fix Issue #3: Validate batch size
    if self.persistent_states and B > 1:
        raise ValueError(
            f"persistent_states=True requires batch_size=1, got {B}. "
            f"Either set batch_size=1 or disable persistent_states."
        )

    # Rest of TTT implementation...
```

#### Expected Results:

- ✅ All 5 issues avoided (no persistent states = no Issues #3, #4, #5; fixes #2)
- ✅ Issue #1 (ring buffer) remains but can be addressed separately
- ✅ Clean, simple architecture matching proven Video-DiT approach
- ✅ Stable training with proper gradient flow

#### Testing:

```python
# Test 1: Training converges
python training/train_ttt_production.py --config configs/production_ttt_dailytalk.yaml

# Verify:
# - Loss decreases steadily
# - No NaN gradients
# - TTT parameters (W1, W2) receive gradients
# - Training completes without errors

# Test 2: Inference works
python inference/run_inference_with_ttt.py --infile test_audio.wav

# Verify:
# - Audio quality is good
# - No artifacts or errors
# - Real-time capable
```

#### Estimated Time: 2-3 days
- Bug fixes: 4 hours
- Testing: 1 day
- Documentation: 4 hours

---

### Phase 2: Measure Baseline Performance (Week 2) 🔬

**Goal**: Establish quantitative baseline for Video-DiT style TTT

#### Metrics to Measure:

**2.1 Training Metrics**

```python
# Track during training
metrics = {
    'train_loss': [],           # Overall loss
    'text_loss': [],            # Text prediction loss
    'audio_loss': [],           # Audio prediction loss
    'ttt_reconstruction_loss': [],  # TTT inner loop loss (if accessible)
    'learning_rate': [],
    'gradient_norms': {
        'W1': [],               # TTT parameter gradients
        'W2': [],
        'transformer': [],      # Base model gradients
    }
}
```

**2.2 Evaluation Metrics**

```python
# Test on held-out data
eval_metrics = {
    'perplexity': ...,          # Text prediction quality
    'mel_distance': ...,        # Audio reconstruction quality
    'speaker_similarity': ...,  # Preserve speaker characteristics
    'intelligibility': ...,     # ASR word error rate
}
```

**2.3 Per-Chunk Analysis** 🔍 **CRITICAL**

```python
# This reveals if TTT adapts within files
def analyze_per_chunk_loss(model, test_files):
    results = []

    for file in test_files:
        chunks = split_into_chunks(file, duration=10.0)
        file_losses = []

        for i, chunk in enumerate(chunks):
            loss = compute_loss(model(chunk))
            file_losses.append(loss)

        results.append({
            'file': file,
            'losses': file_losses,
            'trend': compute_trend(file_losses)  # Decreasing? Flat?
        })

    return results

# Expected for Video-DiT style:
# - Each chunk has similar loss (no adaptation across chunks)
# - Loss doesn't decrease within file
# - This is EXPECTED and OK!
```

#### Success Criteria:

- Training loss decreases steadily
- Evaluation metrics comparable to baseline Moshi
- No instabilities or divergence
- TTT parameters receive non-zero gradients

#### Estimated Time: 1 week
- Full training run: 3-5 days
- Evaluation: 1-2 days

---

### Phase 3: Test If Persistence Helps (Week 3-4) 🧪 **EXPERIMENTAL**

**Goal**: Determine if persistent states actually improve performance

#### Experiment Design:

**3.1 Three Conditions**

```python
# Condition A: Video-DiT style (baseline from Phase 2)
config_A = {
    'persistent_states': False,
    # Each chunk independent
}

# Condition B: Persistent within files, reset between files
config_B = {
    'persistent_states': True,
    'reset_at_file_boundaries': True,  # New feature!
    # Adapts across chunks within same file
}

# Condition C: Always persistent (current broken implementation - for comparison)
config_C = {
    'persistent_states': True,
    'reset_at_file_boundaries': False,
    # WARNING: Has Issues #4 and #5, will perform poorly
}
```

**3.2 Measurements**

```python
# Primary metric: Per-chunk loss pattern
for condition in [A, B, C]:
    per_chunk_results = analyze_per_chunk_loss(model_condition, test_files)

    # Questions:
    # 1. Does loss decrease across chunks within a file?
    #    - Condition A: Should be flat (no cross-chunk adaptation)
    #    - Condition B: Might decrease (if persistence helps)
    #    - Condition C: Unknown (bugs corrupt signal)

    # 2. Does first chunk of new file have higher loss?
    #    - Condition A: Same as other chunks (independent)
    #    - Condition B: Similar to first chunk (reset)
    #    - Condition C: Lower than it should be (contaminated)

    plot_per_chunk_losses(per_chunk_results, condition)
```

**3.3 Statistical Analysis**

```python
# Test significance
from scipy.stats import ttest_rel

# Compare Condition A vs B on chunk-level losses
chunks_later_in_file = get_chunks(position > 0)  # Not first chunk

losses_A = compute_losses(model_A, chunks_later_in_file)
losses_B = compute_losses(model_B, chunks_later_in_file)

t_stat, p_value = ttest_rel(losses_A, losses_B)

if p_value < 0.05 and mean(losses_B) < mean(losses_A):
    print("✓ Persistent states significantly improve later chunks")
else:
    print("✗ No significant benefit from persistent states")
```

#### Decision Tree:

```
IF Condition B significantly outperforms Condition A (p < 0.05):
    → Proceed to Phase 4 (implement proper persistent states)
    → Benefit is proven, worth the complexity

ELIF Condition B similar to Condition A (p > 0.05):
    → STOP at Phase 2 (use Video-DiT style)
    → No benefit from persistence, keep it simple

ELIF Condition B worse than Condition A:
    → Persistent states actively hurt (possible reasons below)
    → Definitely STOP at Phase 2
```

**Possible reasons persistence might hurt**:
1. Train/test mismatch (training chunks shuffled at file level)
2. Overfitting to specific files
3. Interference between different speakers/files
4. Gradient flow issues (even with fixes)

#### Estimated Time: 1-2 weeks
- Implement Condition B properly: 3-4 days
- Run experiments: 1 week (parallel training)
- Analysis: 2-3 days

---

### Phase 4: Proper Persistent States (Only if Phase 3 shows benefit)

**Goal**: Implement persistent states correctly with all fixes

#### Required Changes:

**4.1 Separate W_base and W_state**

```python
# File: moshi_ttt_try/moshi-finetune/moshi_ttt/ttt_layer.py

class TTTMLP(nn.Module):
    def __init__(self, ...):
        super().__init__()

        # Base weights (trainable parameters - updated by optimizer)
        self.W1_base = nn.Parameter(torch.normal(0, 0.02, size=(H, D, 4*D)))
        self.b1_base = nn.Parameter(torch.zeros(H, 1, 4*D))
        self.W2_base = nn.Parameter(torch.normal(0, 0.02, size=(H, 4*D, D)))
        self.b2_base = nn.Parameter(torch.zeros(H, 1, D))

        # State buffers (persistent across chunks - NOT updated by optimizer)
        self.register_buffer('W1_state', self.W1_base.data.clone())
        self.register_buffer('b1_state', self.b1_base.data.clone())
        self.register_buffer('W2_state', self.W2_base.data.clone())
        self.register_buffer('b2_state', self.b2_base.data.clone())

    def reset_ttt_state(self):
        """Reset states to base weights (call at file boundaries)"""
        with torch.no_grad():
            self.W1_state.copy_(self.W1_base)
            self.b1_state.copy_(self.b1_base)
            self.W2_state.copy_(self.W2_base)
            self.b2_state.copy_(self.b2_base)

    def ttt(self, inputs):
        B, H, NC, C, D = inputs.shape

        if self.training and hasattr(self, 'W1_state'):
            # TRAINING: Use states, update them, return gradient-connected output
            # Initialize from current states
            W1_states = self.W1_state.unsqueeze(0).expand(B, -1, -1, -1)
            b1_states = self.b1_state.unsqueeze(0).expand(B, -1, -1, -1)
            W2_states = self.W2_state.unsqueeze(0).expand(B, -1, -1, -1)
            b2_states = self.b2_state.unsqueeze(0).expand(B, -1, -1, -1)

            # CRITICAL: TTT inner loop must be part of computation graph!
            # Gradients flow: loss → XQW_batch → final_states → base_params
            XQW_batch, final_states = ttt_mlp_with_states(
                inputs, W1_states, b1_states, W2_states, b2_states, ...
            )

            # Update state buffers (no grad needed here)
            with torch.no_grad():
                self.W1_state.copy_(final_states["W1_states"][0])
                self.b1_state.copy_(final_states["b1_states"][0])
                self.W2_state.copy_(final_states["W2_states"][0])
                self.b2_state.copy_(final_states["b2_states"][0])

            # Gradient flows through final_states back to W1_base!
            # ∂loss/∂W1_base via: loss → XQW_batch → final_states → W1_states → W1_base
            return XQW_batch
        else:
            # INFERENCE or NO PERSISTENT STATES: Standard Video-DiT style
            W1_states = self.W1_base.unsqueeze(0).expand(B, -1, -1, -1)
            # ... same as above
```

**Key insight**:
- `W1_base` is `nn.Parameter` → optimizer updates it
- `W1_state` is `nn.Buffer` → optimizer ignores it
- Gradient flows: `loss → XQW → final_states → W1_states → W1_base`
- No parameter is both updated by TTT and optimizer!

**4.2 File Boundary Detection**

```python
# File: moshi_ttt_try/moshi-finetune/training/train_ttt_production.py

def reset_ttt_states(model):
    """Reset TTT states in all layers"""
    count = 0
    for module in model.modules():
        if hasattr(module, 'reset_ttt_state'):
            module.reset_ttt_state()
            count += 1
    logger.info(f"Reset {count} TTT layers to base weights")
    return count

# In training loop
previous_file_id = None

while step < args.max_steps:
    batch = next(data_loader)
    codes = batch.codes.to(device)

    # Check for file boundary
    if batch.file_id is not None:
        if previous_file_id is not None and batch.file_id != previous_file_id:
            # File boundary detected!
            logger.info(f"File boundary: {previous_file_id} → {batch.file_id}")
            reset_count = reset_ttt_states(model)

            # Optional: Track in metrics
            metrics_logger.log({'ttt_resets': reset_count}, step=step)

        previous_file_id = batch.file_id

    # Forward pass
    optimizer.zero_grad()
    output = model(codes=codes, condition_tensors=condition_tensors)

    # Loss and backward
    loss = compute_loss(output, codes, ...)
    loss.backward()  # Gradients flow to W1_base, W2_base (NOT W1_state!)

    optimizer.step()  # Updates W1_base, W2_base only
```

**4.3 Gradient Flow Verification**

```python
# Add to training loop for debugging
if step % args.log_freq == 0:
    # Check that base parameters receive gradients
    for name, param in model.named_parameters():
        if 'W1_base' in name or 'W2_base' in name:
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                logger.info(f"{name} grad norm: {grad_norm:.6f}")
            else:
                logger.warning(f"{name} has no gradient!")

    # Check that state buffers do NOT receive gradients
    for name, buffer in model.named_buffers():
        if 'W1_state' in name or 'W2_state' in name:
            if hasattr(buffer, 'grad') and buffer.grad is not None:
                logger.error(f"BUG: {name} (buffer) has gradient!")
```

#### Testing:

```python
# Test 1: Gradient flow to base parameters
def test_gradient_flow():
    model = create_model_with_ttt(persistent_states=True)
    optimizer = AdamW(model.parameters(), lr=1e-4)

    # Create two chunks from same "file"
    chunk0 = create_dummy_batch(file_id="file1", chunk_index=0)
    chunk1 = create_dummy_batch(file_id="file1", chunk_index=1)

    # Process chunk 0
    optimizer.zero_grad()
    output0 = model(chunk0.codes)
    loss0 = compute_loss(output0, chunk0.codes)
    loss0.backward()

    # Verify base parameters have gradients
    assert model.ttt_layer.W1_base.grad is not None, "W1_base should have gradient"
    assert model.ttt_layer.W1_base.grad.abs().sum() > 0, "W1_base gradient should be non-zero"

    optimizer.step()

    # Process chunk 1 (should use adapted state from chunk 0)
    optimizer.zero_grad()
    output1 = model(chunk1.codes)
    loss1 = compute_loss(output1, chunk1.codes)
    loss1.backward()

    # Verify gradients again
    assert model.ttt_layer.W1_base.grad is not None

    print("✓ Gradient flow test passed")

# Test 2: File boundary reset
def test_file_boundary_reset():
    model = create_model_with_ttt(persistent_states=True)

    # Process file 1
    chunk_file1 = create_dummy_batch(file_id="file1", chunk_index=0)
    output1 = model(chunk_file1.codes)
    W1_after_file1 = model.ttt_layer.W1_state.clone()

    # Reset at file boundary
    reset_ttt_states(model)
    W1_after_reset = model.ttt_layer.W1_state.clone()

    # Should be reset to base
    assert torch.allclose(W1_after_reset, model.ttt_layer.W1_base, atol=1e-6), \
        "State should reset to base weights"

    # Process file 2
    chunk_file2 = create_dummy_batch(file_id="file2", chunk_index=0)
    output2 = model(chunk_file2.codes)

    print("✓ File boundary reset test passed")
```

#### Estimated Time: 1-2 weeks
- Implementation: 3-4 days
- Testing: 2-3 days
- Training run: 1 week
- Comparison with Phase 2: 2 days

---

## Alternative: Hybrid Approach (Training vs Inference)

**Observation**: Training and inference have different goals

| Mode | Goal | Context | Best Approach |
|------|------|---------|---------------|
| **Training** | Learn generalizable W_base | Batches from many files | Video-DiT style? |
| **Inference** | Adapt to current speaker | Single streaming session | Persistent states? |

### Hybrid Design:

```python
class TTTMLP(nn.Module):
    def ttt(self, inputs):
        if self.training:
            # TRAINING: Video-DiT style (no persistence)
            W1_states = self.W1_base.unsqueeze(0).expand(B, -1, -1, -1)
            XQW_batch = ttt_mlp(inputs, W1_states, ...)
            return XQW_batch
        else:
            # INFERENCE: Persistent states (streaming adaptation)
            W1_states = self.W1_state.unsqueeze(0).expand(B, -1, -1, -1)
            XQW_batch, final_states = ttt_mlp_with_states(inputs, W1_states, ...)

            with torch.no_grad():
                self.W1_state.copy_(final_states["W1_states"][0])

            return XQW_batch
```

**Pros**:
- ✅ Training is simple and stable (Video-DiT style)
- ✅ Inference adapts to speaker (persistent states)
- ✅ Each mode optimized for its goal

**Cons**:
- ⚠️ Train/test mismatch (training doesn't see persistent behavior)
- ⚠️ May not learn W_base that benefits from persistence
- ⚠️ Inference behavior not trained for

**Verdict**: ⚠️ Risky due to train/test mismatch. Only try if Phase 3 shows inference benefits from persistence but training doesn't.

---

## Addressing Issue #1: Ring Buffer (Separate Concern)

**Problem**: Ring buffer loses tokens after 3000 steps (~12.5 seconds)

**Impact**: Limits long-term context, but separate from TTT state persistence issue

### Option 1A: Increase Ring Buffer Capacity (Quick Fix)

```python
# File: moshi/moshi/moshi/modules/transformer.py

# Change from:
context = 3000  # 12.5 seconds at 12Hz

# To:
context = 12000  # 50 seconds at 12Hz
```

**Pros**: Trivial change, extends context
**Cons**: 4x memory usage, may not be enough for very long audio

### Option 1B: Chunked Attention (Proper Fix)

Adopt Video-DiT's approach for Moshi:

```python
# Process audio in complete segments with overlap

def chunked_forward(audio_sequence, chunk_size=3000, overlap=500):
    chunks = []

    for i in range(0, len(audio_sequence), chunk_size - overlap):
        chunk = audio_sequence[i:i+chunk_size]

        # Full attention within chunk (no ring buffer!)
        attn_out = full_attention(chunk)

        # TTT processes chunk
        ttt_out = ttt_layer(attn_out)

        chunks.append(ttt_out[overlap//2:-overlap//2])  # Remove overlap

    return torch.cat(chunks, dim=1)
```

**Pros**: No information loss, complete attention
**Cons**: Requires architectural changes, slower (quadratic attention)

**Recommendation**: Try Option 1A first (simple), move to 1B only if needed.

---

## Final Recommendation: Start Simple, Validate, Then Complexify

### **Phase 1 (Week 1-2): Fix bugs, use Video-DiT style**
- ✅ Simple and proven
- ✅ Stable training
- ✅ No persistent states issues
- **Goal**: Working baseline

### **Phase 2 (Week 2): Measure baseline performance**
- 🔬 Quantitative metrics
- 🔬 Per-chunk loss analysis
- 🔬 Comparison with original Moshi
- **Goal**: Know what we're improving from

### **Phase 3 (Week 3-4): Test if persistence helps**
- 🧪 A/B test: persistent vs non-persistent
- 🧪 Statistical analysis
- 🧪 Decision: keep simple or add complexity
- **Goal**: Evidence-based decision

### **Phase 4 (Week 5-6): Only if Phase 3 proves benefit**
- 🏗️ Proper persistent states implementation
- 🏗️ W_base/W_state separation
- 🏗️ File boundary detection
- **Goal**: Production-ready persistent states

---

## Expected Outcomes

### Scenario A: Phase 3 shows NO benefit from persistence

**Outcome**: Ship Phase 2 (Video-DiT style)

**Result**:
- ✅ Simple, maintainable architecture
- ✅ Proven approach (Video-DiT)
- ✅ No persistence bugs
- ✅ Faster training (no state management overhead)

**This is a good outcome!** Simpler is better if performance is equal.

### Scenario B: Phase 3 shows SMALL benefit (5-10%)

**Decision point**: Is complexity worth small gain?

**Factors**:
- Engineering cost (maintenance, debugging)
- Training stability
- Inference latency
- Code complexity

**Recommendation**: Unless benefit is substantial, prefer simplicity.

### Scenario C: Phase 3 shows LARGE benefit (>15%)

**Outcome**: Proceed to Phase 4

**Result**:
- ✅ Justified complexity (clear performance gain)
- ✅ Proper implementation (W_base/W_state)
- ✅ Full testing and validation
- ✅ Production-ready

**This validates the original design intent!**

---

## Risk Mitigation

### Risk 1: TTT doesn't help at all

**Symptom**: Even Phase 1 (Video-DiT style) doesn't improve over baseline Moshi

**Possible causes**:
- TTT not suited for audio (proven in video though)
- Hyperparameters wrong (eta, mini_batch_size)
- Implementation bugs
- KV cache + attention is already sufficient

**Mitigation**:
- Verify TTT works on toy problem first
- Sweep hyperparameters
- Compare with TTT paper's audio results (if any)

### Risk 2: Phase 3 shows persistence hurts

**Symptom**: Condition B (persistent) worse than Condition A (non-persistent)

**Possible causes**:
- Overfitting to specific files
- Train/test mismatch
- Implementation still has bugs
- Persistent states genuinely harmful

**Mitigation**:
- Debug thoroughly
- Check for remaining bugs
- Analyze failure modes
- Accept result and stick with Video-DiT style

### Risk 3: Phase 4 implementation too complex

**Symptom**: Bugs, instabilities, hard to maintain

**Possible causes**:
- W_base/W_state separation is tricky
- Gradient flow issues
- State synchronization problems

**Mitigation**:
- Extensive unit tests
- Gradient checking
- Code review
- Consider if benefit justifies complexity

---

## Success Metrics

### Phase 1 Success:
- ✅ Training converges
- ✅ No gradient issues
- ✅ TTT parameters update
- ✅ Inference works

### Phase 2 Success:
- ✅ Metrics comparable to baseline Moshi
- ✅ TTT reconstruction loss decreases
- ✅ No performance regression
- ✅ Per-chunk analysis shows expected pattern

### Phase 3 Success:
- ✅ Clear measurement of persistence benefit (or lack thereof)
- ✅ Statistical significance
- ✅ Reproducible results
- ✅ Evidence-based decision

### Phase 4 Success (if reached):
- ✅ Performance gain matches Phase 3 prediction
- ✅ No training instabilities
- ✅ Proper gradient flow verified
- ✅ File boundaries handled correctly
- ✅ Production-ready code

---

## Timeline Summary

| Phase | Duration | Outcome |
|-------|----------|---------|
| Phase 1 | 1-2 weeks | Working baseline |
| Phase 2 | 1 week | Quantitative metrics |
| Phase 3 | 1-2 weeks | Evidence-based decision |
| Phase 4 | 1-2 weeks | (Only if Phase 3 warrants) |
| **Total** | **4-7 weeks** | **Production TTT-Moshi** |

**Fast track** (if Phase 3 shows no benefit): 3-4 weeks total

**Full track** (if Phase 3 shows benefit): 6-7 weeks total

---

## Conclusion

**The right approach is EMPIRICAL, not theoretical.**

1. **Start with what works** (Video-DiT's approach)
2. **Measure carefully** (quantitative metrics, statistical tests)
3. **Add complexity only if data demands it** (evidence-based decisions)
4. **Prefer simplicity** when performance is equal

**Current state**: 5 critical bugs, unvalidated assumptions, weak learning signal

**Recommended state**: Clean implementation, measured performance, justified design

**Philosophy**: "Make it work, make it right, make it fast" - in that order.

---

**Document Version**: 1.0
**Author**: Claude (Sonnet 4.5)
**Date**: 2025-11-10
**Status**: Ready for Implementation
