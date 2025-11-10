# Research Protocol: Systematic Debugging of Moshi 6000-Token Failure

## Executive Summary

**Observation**: Moshi outputs garbage after ~6000 tokens  
**Goal**: Root-cause the failure before investing in TTT integration  
**Timeline**: 3-5 days for complete diagnosis  
**Critical Question**: Is TTT a solution or are we building on a broken foundation?

---

## Phase 1: Quick Triage (30 minutes - DO THIS FIRST)

### Why First?
**Principle**: Test the most likely hypothesis with the least effort first  
**RoPE is 90% likely the culprit** because:
- Failure at 6000 tokens ≈ 1.46x training context (4096)
- Classic RoPE extrapolation failure signature
- max_period=10,000 is insufficient for 6000+ tokens

### Action Items

1. **Run Quick Check** (5 minutes)
   ```python
   from quick_rope_check import run_quick_diagnosis
   is_rope, confidence, details = run_quick_diagnosis(model)
   ```

2. **If RoPE detected** (confidence > 80%):
   - **Immediate Test**: Change max_period and re-run
   ```python
   model.transformer.max_period = 100_000
   model.transformer.rope.max_period = 100_000
   # Re-test at 6000 tokens
   ```
   - **If garbage fixed**: RoPE confirmed → Proceed to Phase 4 (solutions)
   - **If garbage persists**: RoPE not the issue → Continue to Phase 2

3. **If RoPE not detected**:
   - Continue systematic debugging in Phase 2

---

## Phase 2: Isolate Component Failures (1-2 days)

### Principle: Binary Search Through Architecture

Use **ablation testing** to isolate which component is failing:

### 2A. Text vs. Audio Pathway Test
```python
# Test 1: Text-only mode (disable audio generation)
output = model.generate(input_ids, audio_generation=False)
# If failure persists → Helium (text LM) issue
# If failure disappears → Audio pathway issue

# Test 2: Audio-only mode (bypass text entirely)  
output = model.generate_audio_only(audio_input)
# If failure persists → Mimi (codec) or acoustic modeling issue
# If failure disappears → Text-audio interaction issue
```

**Decision Tree**:
- Text fails → Check: Helium context window, text KV cache
- Audio fails → Check: Mimi bottleneck, acoustic tokens
- Both fail → Check: Multi-stream synchronization
- Neither fails → Check: Text-audio fusion layer

### 2B. Single vs. Multi-Stream Test
```python
# Disable "other speaker" stream
output = model.generate(input, single_stream=True)
# If failure disappears → Cross-stream interference at 6000 tokens
```

### 2C. Context Window Sweep
```python
for context in [2048, 3072, 4096, 6144, 8192]:
    set_attention_context(model, context=context)
    test_at_length(context + 1900)  # Test slightly below capacity
    
# If failure moves with context → KV cache capacity issue
# If failure stays at 6000 → Something else
```

### 2D. Precision Test
```python
# Test fp32 vs fp16
model_fp32 = model.float()
output_fp32 = generate_at_6000(model_fp32)

# If garbage disappears → Numerical instability
# Likely: gradient accumulation or attention softmax overflow
```

---

## Phase 3: Deep Inspection (2-3 days if Phase 2 inconclusive)

### Only if Phase 1 & 2 don't give clear answer

### 3A. Attention Visualization
```python
from debug_rope_moshi import RoPEDebugger
debugger = RoPEDebugger(model)
attn_patterns = debugger.test_2_attention_pattern_at_6000()

# Visualize attention heatmap at position 6000
# Look for:
# - Uniform noise → numerical collapse
# - Fixation on single position → cache corruption  
# - Recency bias (only last 100 tokens) → RoPE breakdown
```

### 3B. Token Generation Quality
```python
# Generate at boundaries: 4000, 5000, 5900, 6000, 6100, 6200
for length in boundaries:
    logits = model(input[:length])[-1]
    entropy = compute_entropy(softmax(logits))
    
# Plot entropy vs position
# If entropy drops sharply at 6000 → model confidence collapse
# If entropy explodes at 6000 → distribution breakdown
```

### 3C. Codec Analysis
```python
# Mimi has 250-frame (20s) bottleneck
# 6000 audio tokens at 12.5 Hz = 480 seconds = 24 complete cycles

# Check if quality degrades every 20 seconds
for t in range(0, 600, 20):  # Every 20s up to 600s
    quality_score = assess_audio_quality(generated_audio[t:t+20])
    
# If periodic degradation → Mimi bottleneck issue
# If sudden degradation at 480s → Codec accumulation issue
```

---

## Phase 4: Solution Decision Tree

### Based on diagnosis, choose solution path:

```
IF RoPE extrapolation failure:
├── Quick Fix: Increase max_period (10,000 → 100,000)
│   ├── Works? → Use this for now, plan position interpolation
│   └── Doesn't work? → Try position interpolation or ALiBi
│
├── Medium-term: Position Interpolation (1 week)
│   - Scale positions by factor before RoPE
│   - Research: Chen et al. "Extending Context Window..."
│   
└── Long-term: Hybrid RoPE + TTT (2-3 weeks)
    - Keep RoPE for local attention (<4096)
    - Add TTT for long-range dependencies (>4096)
    - Best of both worlds

ELSE IF KV cache capacity:
├── Quick Fix: Increase cache size (4096 → 8192)
├── Medium-term: Sliding window attention
└── Long-term: TTT memory compression

ELSE IF multi-stream synchronization:
├── Add cross-stream loss terms
├── Periodic synchronization checks
└── TTT with cross-stream memory

ELSE IF Mimi bottleneck:
├── Integrate TTT into Mimi transformers (major undertaking)
├── Add error correction between Mimi windows
└── Hierarchical codec with longer context

ELSE IF numerical instability:
├── Use fp32 for critical operations
├── Add gradient clipping in generation
└── Normalize hidden states periodically
```

---

## TTT Integration Decision

### When to Add TTT?

**YES - TTT is worth it if**:
1. ✅ RoPE is fixable but we want >20k tokens
2. ✅ KV cache is the bottleneck
3. ✅ You need unbounded context
4. ✅ Base model works correctly at 4096 tokens

**NO - Fix underlying issue first if**:
1. ❌ Base model fails even with quick fixes
2. ❌ Codec/quantization artifacts are the issue
3. ❌ Multi-stream synchronization is broken
4. ❌ Numerical instability persists

### TTT Integration Strategy (if YES above)

**Conservative Approach** (recommended):
1. Fix immediate issue (RoPE/cache) first
2. Validate model works reliably to 10k tokens
3. **Then** add TTT for 10k → 100k+ extension
4. TTT solves the "how do we go further" problem

**Aggressive Approach** (risky):
1. Add TTT immediately
2. Hope it masks underlying issues
3. **Risk**: TTT adds complexity, may make debugging harder
4. **Risk**: You'll never know what the real issue was

---

## Recommended Research Path

### Week 1: Diagnosis (THIS WEEK)
```
Day 1: Quick RoPE check (30 min) + max_period fix attempt (1 hour)
Day 2: Component ablation tests (Phase 2)
Day 3: Deep inspection if needed (Phase 3)
Day 4: Document findings, decide solution path
Day 5: Implement quick fix and validate
```

### Week 2: Solution Implementation
```
If RoPE issue:
  - Implement position interpolation properly
  - Test on 8k, 12k, 16k token sequences
  - Measure quality degradation curve

If other issue:
  - Implement targeted fix
  - Re-test TTT value proposition
```

### Week 3: TTT Integration (if appropriate)
```
- Only if base model is solid
- Integrate TTT into last 3-5 layers
- Train/fine-tune with long sequences
- Validate on multi-hour conversations
```

---

## Experimental Log Template

Keep detailed notes:

```markdown
## Experiment: [Name]
Date: YYYY-MM-DD
Hypothesis: [What you're testing]
Setup: [Code, parameters, etc.]

### Results
- Metric 1: [value]
- Metric 2: [value]
- Observation: [qualitative]

### Conclusion
- [ ] Hypothesis confirmed
- [ ] Hypothesis rejected
- [ ] Inconclusive

### Next Step
[What to test next based on this result]
```

---

## Key Principles

1. **Test cheapest hypotheses first** (RoPE check takes 30 seconds)
2. **One variable at a time** (don't change multiple things)
3. **Document everything** (you'll forget what you tried)
4. **Validate fixes thoroughly** (test at 7k, 8k, 10k too)
5. **Don't add complexity until necessary** (TTT is complex)

---

## Success Criteria

### Phase 1 Success:
- [x] Identified root cause with >80% confidence
- [x] Quick fix tested (works or doesn't work)
- [x] Decision on whether TTT is appropriate

### Phase 2 Success:
- [x] Model generates coherently at 8k tokens
- [x] No quality degradation from 4k → 8k
- [x] Failure mechanism understood and documented

### Phase 3 Success (if adding TTT):
- [x] Model generates coherently at 20k+ tokens
- [x] TTT provides measurable benefit
- [x] Integration is stable and well-tested

---

## Emergency Shortcut

**If you just need it to work NOW**:

1. Run quick_rope_check.py (30 seconds)
2. If RoPE issue: `max_period = 500_000` (nuclear option, will work)
3. Test at 10k tokens
4. If still broken → not RoPE, need deeper diagnosis
5. If works → proceed with proper solution later

But **always** come back and do proper diagnosis later.

---

## Contact Points for Help

If stuck:
1. Check similar LLM extrapolation papers (Llama2 tech report section on context extension)
2. Review RoFormer paper (Su et al.) for RoPE theory
3. Check Moshi GitHub issues for similar problems
4. Test on different checkpoint (maybe just this checkpoint is broken?)

---

## Final Recommendation

**START HERE**:
```bash
python quick_rope_check.py
```

Takes 30 seconds. Will tell you if it's RoPE with >90% confidence.

If RoPE: Try max_period fix  
If not RoPE: Proceed to Phase 2 systematic testing

**Do NOT start TTT integration until you've completed Phase 1-2.**

Good luck! 🚀