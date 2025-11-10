# LibriLight Evaluation - Migration Guide

## Quick Start

The LibriLight evaluation has been simplified. Here's what you need to know:

### What Changed
- **Before**: 3 broken implementations, 600+ lines, complex configuration
- **After**: 1 clean implementation, 150 lines, simple and correct

### How to Use

```python
from finetune.paper_metrics import PaperMetricsEvaluator

# Create evaluator
evaluator = PaperMetricsEvaluator(
    mimi_encoder=mimi,
    interleaved_tokenizer=tokenizer,
    device='cuda',
    config=config
)

# Run LibriLight evaluation
results = evaluator.evaluate_librilight_long_context(model)

# Results format:
# {
#     'librilight_loss_8k': 2.84,     # Loss at 8k tokens
#     'librilight_loss_16k': 2.71,    # Loss at 16k tokens  
#     'librilight_loss_24k': 2.58,    # Loss at 24k tokens
#     'librilight_slope': -0.00001,   # Improvement rate (negative = getting better)
#     'librilight_samples': 24990     # Number of tokens evaluated
# }
```

## Configuration

### Required Settings
```yaml
librilight_audio_dir: "/path/to/librilight/data"
librilight_max_chapters: 3  # Number of chapters to evaluate
```

### Optional Settings
```yaml
librilight_streaming:
  max_sequence_length: 50000  # Memory safety limit (tokens)
```

### Removed Settings (No Longer Needed)
These config options have been removed:
- ❌ `use_fixed_streaming`
- ❌ `verify_loss_computation`
- ❌ `ttt_verification_enabled`
- ❌ `memory_check_enabled`
- ❌ `ttt_optimize_chunk_size`
- ❌ `ttt_prefer_efficiency`

## Key Fixes

### 1. Correct Text Token Usage
**Issue**: LibriLight is audio-only, but the old code used silence token (0) for text.

**Fix**: Now uses model's predicted text token.

```python
# OLD (wrong):
text_token = torch.zeros(...)  # Silence = "no speech"

# NEW (correct):
text_token = torch.argmax(text_logits[:, 0, 0, :], dim=-1)  # Model's prediction
```

**Why it matters**: The text token affects audio generation through the depformer. Using the correct token gives more accurate evaluation.

### 2. Audio Loss Only
**Issue**: Some code computed text loss when LibriLight has no text labels.

**Fix**: Only compute loss for audio codebooks (1-8).

```python
# Compute audio loss only
for cb in range(8):  # 8 audio codebooks
    loss += F.cross_entropy(audio_logits[cb], audio_target[cb])

return loss / 8  # Average over audio codebooks
```

### 3. Simplified Implementation
**Issue**: 3 different broken implementations, complex state management.

**Fix**: 1 clean implementation in `librilight_simple.py`.

## Understanding the Results

### Loss Values
- **Expected range**: 2-4 for audio tokens
- **Lower is better**: Model predicts tokens more accurately
- **Smooth progression**: Should decrease gradually with TTT

### Slope Metric
- **Negative slope**: Model is improving (TTT working)
- **Positive slope**: Model is degrading (problem!)
- **Zero slope**: No learning (TTT may be disabled)

Example:
```python
{
    'librilight_loss_8k': 3.12,   # Early in sequence
    'librilight_loss_16k': 2.89,  # Middle
    'librilight_loss_24k': 2.67,  # Late (better!)
    'librilight_slope': -0.00002  # Improving!
}
```

## Debugging

### If You See High Loss Values (>8)
This usually means:
1. Model hasn't been trained properly
2. Audio encoding failed
3. Wrong data being fed

**Check**:
```python
# Verify audio encoding
codes = evaluator._encode_audio(audio_path)
print(f"Audio codes shape: {codes.shape}")  # Should be [1, 8, T]
print(f"Audio codes range: {codes.min()}-{codes.max()}")  # Should be 0-1023

# Verify model forward pass
output = model(codes)
print(f"Logits shape: {output.logits.shape}")  # Should have vocab dimension
```

### If You See Erratic Loss Patterns
This usually means:
1. TTT is disabled or broken
2. Streaming state not working
3. Data corruption

**Check**:
```python
# Verify TTT layers exist
for name, module in model.named_modules():
    if hasattr(module, 'ttt_layer'):
        print(f"Found TTT layer: {name}")

# If no TTT layers, loss won't improve over time
```

### If Evaluation Crashes (OOM)
**Reduce sequence length**:
```yaml
librilight_streaming:
  max_sequence_length: 25000  # Reduce from 50000
```

**Or reduce number of chapters**:
```yaml
librilight_max_chapters: 1  # Reduce from 3
```

## Limitations

### Current Implementation
The simplified implementation still has the fundamental API limitation:

**Problem**: `LMGen.step()` returns sampled tokens, not logits. We can't compute loss from exact TTT-updated state.

**Workaround**: We do a separate forward pass to get logits. This is better than before (correct text token, audio loss only) but not perfect.

**Impact**: Loss values are approximate. They reflect TTT performance but may not be exact.

### Proper Solution
Add `step_with_logits()` to LMGen API in `moshi/models/lm.py`:

```python
def step_with_logits(self, input_tokens):
    """Like step() but returns logits for loss computation."""
    # ... normal step logic ...
    transformer_out, text_logits = state.graphed_main(...)
    
    # Get audio logits
    audio_logits = []
    for cb_idx in range(self.lm_model.dep_q):
        logits = self.lm_model.forward_depformer(cb_idx, text_input, transformer_out)
        audio_logits.append(logits)
    
    return {
        'tokens': output_tokens,
        'audio_logits': torch.stack(audio_logits, dim=1)
    }
```

Then evaluation would be trivial:
```python
for t in range(seq_length - 1):
    result = lm_gen.step_with_logits(codes[:, :, t:t+1])
    loss = F.cross_entropy(result['audio_logits'], targets[:, :, t+1])
```

## FAQs

### Q: Why is LibriLight audio-only?
A: LibriLight is from LibriSpeech audiobooks. It's designed for long-context audio evaluation without transcriptions.

### Q: Why do we need a text token if it's audio-only?
A: Moshi's architecture requires: `Audio → Transformer → Text Token → Depformer → Audio Prediction`

Even for audio-only, we need a text token (intermediate representation) to generate audio.

### Q: What text token should we use?
A: **Model's prediction** (what it thinks is being said). NOT silence token (0).

### Q: Is the current implementation correct?
A: **Better** than before, but not perfect due to API limitations. Good enough for evaluation.

### Q: Should I use this for published results?
A: The simplified implementation is more correct than the old one. But note the API limitation in your paper/report.

### Q: How do I know if TTT is working?
A: Check the slope metric. Negative slope = loss decreasing = TTT learning.

## Code Organization

### Main Files
1. **`finetune/paper_metrics.py`**
   - Overall metrics evaluator class
   - Wrapper for LibriLight evaluation
   - Integration with other metrics

2. **`finetune/librilight_simple.py`**
   - Core LibriLight evaluation logic
   - Clean, documented implementation
   - ~150 lines total

3. **`finetune/librilight_loader.py`**
   - Data loading utilities
   - Unchanged by cleanup

### Method Flow
```
evaluate_librilight_long_context()
  ↓
_evaluate_librilight_simple()  [wrapper in paper_metrics.py]
  ↓
evaluate_librilight_simple()  [main logic in librilight_simple.py]
  ↓
_compute_streaming_audio_loss()  [audio loss computation]
  ↓
aggregate_librilight_results()  [results formatting]
```

## Migration Checklist

If you're updating existing code:

- [ ] Remove old config options (`use_fixed_streaming`, etc.)
- [ ] Update LibriLight config to minimal settings
- [ ] Test evaluation runs without errors
- [ ] Verify loss values are in expected range (2-4)
- [ ] Check that slope is negative (if TTT enabled)
- [ ] Update any scripts that parse LibriLight results
- [ ] Document the API limitation if using for research

## Summary

**What you need to know**:
1. ✅ LibriLight evaluation is now simpler and more correct
2. ✅ Uses model's predicted text token (not silence)
3. ✅ Computes audio loss only (correct for audio-only data)
4. ⚠️  Still has API limitation (separate forward pass for loss)
5. 📝 Proper fix requires adding `step_with_logits()` to LMGen

**For most use cases, the current implementation is good enough!**
