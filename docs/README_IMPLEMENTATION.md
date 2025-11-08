# TTT + Llama-Omni Implementation Guide

**Complete implementation plan for adding Test-Time Training to Llama-Omni for unlimited context speech generation**

---

## 📚 Documentation Structure

### Essential Reading Order

1. **START HERE: `13_Critical_Updates_To_Plan.md`** ⭐
   - Complete, production-ready implementation guide
   - Incorporates all feedback and analysis
   - Read this alongside base plan (Doc 10)
   - **Status**: 95%+ complete, ready to implement

2. **Base Plan: `10_TTT_Integration_Plan.md`**
   - Original comprehensive plan (1,297 lines)
   - Architecture, file copying, training infrastructure
   - **Status**: 68% complete (good foundation)

3. **Gap Analysis: `11_Critical_Points_Analysis.md`**
   - Comparison of docs requirements vs plan
   - What was covered, what was missing
   - **Result**: Identified 5 critical gaps

4. **Agent Review: `12_Agent_Review_Analysis.md`**
   - Analysis from experienced TTT implementer
   - 12 critical points identified
   - Specific code examples for missing pieces

### Background Research (Optional)

5. `09_Final_Verdict.md` - Why Llama-Omni over Moshi
6. `05_Critical_Issues_Analysis.md` - Why previous attempts failed
7. `08_Llama_Omni_Analysis.md` - Model architecture analysis

---

## 🎯 What You Need to Know

### The Problem
- **Goal**: Long-form speech generation (hours, not minutes)
- **Challenge**: Standard models limited to 4-minute context
- **Solution**: Add TTT (Test-Time Training) for unlimited context

### Why This Approach
- **Llama-Omni**: Speech-to-text-to-speech based on Llama 3.1
- **TTT**: Replaces attention in top 8 layers (24-31)
- **Advantage**: Fixed memory (2GB) vs. growing KV cache (24GB @ 48k tokens)

### Critical Success Factors

From agent review and analysis:

1. **FP32 Precision** (MANDATORY)
   - W1, b1, W2, b2 MUST be `torch.float32`
   - BF16 causes gibberish after ~3,750 updates (5-7 minutes)
   - Add assertions everywhere

2. **State Return & Persistence**
   - Must return `(output, None, cache)` for HuggingFace compatibility
   - Cache = `(W1, b1, W2, b2)` tuple
   - Never reset mid-conversation

3. **RoPE Position Reset**
   - Positions reset per mini-batch: [0-63, 0-63, 0-63, ...]
   - NOT global positions [0, 1, 2, ..., 8191]
   - This is non-obvious but critical

4. **Runtime Auto-Padding**
   - Auto-pad in forward() to 64-multiple
   - Trim padding from output
   - Not just in data collator

5. **Curriculum Training**
   - Stage 1: 8k context (2 days)
   - Stage 2: 16k context (3 days)
   - Stage 3: 32k context (4 days)
   - Stage 4: 64k context (5 days)
   - DON'T skip stages!

---

## 🚀 Implementation Checklist

### Phase 1: Core Implementation (Week 1)

**Priority 1: TTT Layer with State Return**
- [ ] Copy from `ttt-video-dit/ttt/models/ssm/ttt_layer.py`
- [ ] Remove video-specific code (interleave, 3D RoPE)
- [ ] Add state return: `return (output, None, cache)`
- [ ] Implement auto-padding + trimming
- [ ] Add FP32 assertions
- [ ] Test: `test_state_return_format()`

**Priority 2: TTT Operations**
- [ ] Copy `ops/ttt_mlp.py` (100% - just change imports)
- [ ] Copy `ops/ttt_linear.py` (100% - just change imports)
- [ ] Copy `ops/utils.py` (ln_fwd, ln_fused_l2_bwd, gelu_bwd)
- [ ] Test: `test_state_persistence_across_batches()`

**Priority 3: Integration**
- [ ] Create `omni_speech_llama_ttt.py`
- [ ] Replace `self_attn` in layers 24-31
- [ ] Load pretrained Llama-Omni weights
- [ ] Test: `test_fp32_maintained_during_forward()`

### Phase 2: Infrastructure (Week 2)

**Monitoring**
- [ ] Implement `TTTMonitor` class
- [ ] CSV logging (step, layer, W1 stats, loss)
- [ ] Visualization plots
- [ ] Test with sample training run

**Data Pipeline**
- [ ] Collect long-form data (100+ hours)
- [ ] Identify conversation boundaries
- [ ] Create curriculum datasets (8k, 16k, 32k, 64k)
- [ ] Implement padding to 64-multiple

**Tests**
- [ ] `test_auto_padding()`
- [ ] `test_progressive_lengths()`
- [ ] `test_multi_turn_generation()`
- [ ] All 6 critical tests pass

### Phase 3: Training (Week 3-7)

**Stage 1: 8k Context** (2 days)
- [ ] Start training with monitoring
- [ ] Check W1/b1 stats every 100 steps
- [ ] No NaN, max < 10
- [ ] TTT loss decreasing
- [ ] Save checkpoint

**Stage 2: 16k Context** (3 days)
- [ ] Load Stage 1 checkpoint
- [ ] Continue training
- [ ] Monitor quality

**Stage 3: 32k Context** (4 days)
- [ ] Load Stage 2 checkpoint
- [ ] Continue training
- [ ] Monitor quality

**Stage 4: 64k Context** (5 days)
- [ ] Load Stage 3 checkpoint
- [ ] Continue training
- [ ] Final evaluation

### Phase 4: Validation (Week 8)

**Quality Tests**
- [ ] Generate 1-hour coherent speech
- [ ] Generate 2-hour coherent speech
- [ ] No gibberish at any tested length
- [ ] Perplexity @ 1k, 5k, 10k, 30k, 60k

**Human Evaluation**
- [ ] 3+ raters
- [ ] Blind comparison vs baseline
- [ ] TTT preferred > 50%

**Performance**
- [ ] Memory < 3GB per sample
- [ ] Batch size 4 works
- [ ] Inference speed acceptable

---

## 🔥 Critical Warnings

### DON'T:

❌ **Skip curriculum stages** - Jumping to 64k will fail
❌ **Use FP16/BF16 for W1, b1** - Causes gibberish after 5-7 min
❌ **Forget to return cache** - Breaks HuggingFace compatibility
❌ **Use global RoPE positions** - Should reset per mini-batch
❌ **Train without monitoring** - You'll waste weeks debugging

### DO:

✅ **Follow curriculum**: 8k → 16k → 32k → 64k
✅ **Assert FP32 everywhere**: In `__init__` and `forward()`
✅ **Monitor W1/b1 stats**: Mean, std, max every 100 steps
✅ **Test progressively**: 1k, 2k, 4k, 8k before full training
✅ **Use troubleshooting guide**: When (not if) issues arise

---

## 📊 What Success Looks Like

### Week 1: Core Complete
```python
# This should work:
model = OmniSpeechLlamaForCausalLMTTT(config)
input_ids = torch.randint(0, 1000, (1, 128))
out = model(input_ids, use_cache=True)

assert out.past_key_values is not None
assert out.past_key_values[24][0].dtype == torch.float32
print("✅ Core implementation working")
```

### Week 2: Infrastructure Ready
```python
# Monitoring working:
monitor = TTTMonitor(log_dir="logs")
monitor.log_layer_state(24, W1, b1, W2, b2)
monitor.plot_evolution()
print("✅ Infrastructure ready")
```

### Week 3-7: Training Progress
```
Stage 1 (8k):  [===================] 100% | Loss: 2.45 | W1_max: 3.2 ✅
Stage 2 (16k): [===================] 100% | Loss: 2.31 | W1_max: 3.5 ✅
Stage 3 (32k): [===================] 100% | Loss: 2.28 | W1_max: 3.8 ✅
Stage 4 (64k): [===================] 100% | Loss: 2.25 | W1_max: 4.1 ✅
```

### Week 8: Validation Passed
```
Perplexity Results:
├─ 1k tokens:  12.5
├─ 5k tokens:  13.1  (+4.8%)
├─ 10k tokens: 13.8  (+10.4%)
├─ 30k tokens: 14.2  (+13.6%)
└─ 60k tokens: 14.5  (+16.0%)  ✅ < 20% degradation

Generation Quality:
├─ 1 hour:  Coherent ✅
├─ 2 hours: Coherent ✅
└─ No gibberish detected ✅

Human Evaluation:
└─ TTT preferred: 68% ✅ > 50%

🎉 READY FOR PRODUCTION
```

---

## 🐛 When Things Go Wrong

Use `docs/13_Critical_Updates_To_Plan.md` Section 5: Troubleshooting Guide

### Quick Diagnostic:

**Gibberish after 5-7 min?**
→ Check FP32 (most likely cause)

**Training unstable?**
→ Reduce `ttt_base_lr` to 0.1

**No quality improvement?**
→ Verify state actually updating

**Out of memory?**
→ Reduce batch size, enable gradient checkpointing

**See full troubleshooting guide for complete symptom → fix mapping**

---

## 📖 How to Use This Documentation

### For Implementers:

1. Read `13_Critical_Updates_To_Plan.md` - complete implementation guide
2. Reference `10_TTT_Integration_Plan.md` - base architecture details
3. Use troubleshooting guide when stuck
4. Follow pre-launch checklist before deployment

### For Reviewers:

1. Read `11_Critical_Points_Analysis.md` - coverage analysis
2. Read `12_Agent_Review_Analysis.md` - expert feedback
3. Verify all critical points addressed

### For Project Managers:

1. Timeline: 8 weeks (1 week prep + 5 weeks training + 2 weeks validation)
2. Resources: 4-8 GPUs, 100+ hours speech data
3. Risks: See troubleshooting guide for common issues
4. Success metrics: See "What Success Looks Like"

---

## 🎯 Final Checklist

Before starting implementation:

- [ ] Read `13_Critical_Updates_To_Plan.md` completely
- [ ] Understand why FP32 is critical (numerical stability)
- [ ] Understand why curriculum is critical (gradual learning)
- [ ] Understand state return format (HuggingFace compatibility)
- [ ] Understand RoPE position reset (per mini-batch)
- [ ] Have 100+ hours of long-form speech data
- [ ] Have 4-8 GPUs available for 8 weeks
- [ ] Have monitoring infrastructure ready
- [ ] Have rollback plan if training fails

Before deployment:

- [ ] All tests pass (6 critical tests)
- [ ] Generates coherent speech for 2+ hours
- [ ] No gibberish detected
- [ ] Perplexity degradation < 20% @ 60k
- [ ] Human eval: TTT preferred > 50%
- [ ] Memory usage < 3GB per sample
- [ ] Complete pre-launch checklist (Doc 13, Section 6)

---

## 📞 Getting Help

If you encounter issues:

1. Check troubleshooting guide (Doc 13, Section 5)
2. Review agent feedback (Doc 12)
3. Verify critical points addressed (Doc 11)
4. Check base plan for architecture details (Doc 10)

Common issues and solutions are documented in the troubleshooting guide.

---

## 🙏 Acknowledgments

This implementation plan incorporates:

- Deep code analysis of ttt-video-dit, ttt-lm-jax, llama-omni
- Lessons from docs 00-09 (research and analysis)
- Feedback from experienced TTT implementer
- Critical points analysis and gap identification

**Total Documentation**: ~5,000 lines across 13 documents
**Completeness**: 95%+ (production-ready)
**Timeline**: 8 weeks from start to deployment

---

**Ready to implement? Start with Doc 13! 🚀**
