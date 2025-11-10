# TTT Diagnostic Logging Guide

This guide explains how to use the comprehensive TTT diagnostic logging system to understand how TTT layers affect model behavior during inference.

## Overview

The diagnostic logger tracks detailed metrics every N steps to show:
- **Magnitude Analysis**: How strong are attention vs TTT outputs?
- **Gating Behavior**: How much does TTT contribute to the final output?
- **Cosine Similarity**: Are attention and TTT outputs aligned or orthogonal?
- **Distribution Statistics**: Mean, std, sparsity of outputs
- **Delta/Change Tracking**: How much outputs change over time
- **TTT Internal Weights**: Weight norms and updates

## Quick Start

### Enable Diagnostics During Batch Inference

```bash
# Basic usage (logs every 100 steps, default)
./submit_batch_inference.sh /path/to/checkpoint "input.wav" ./results "" "" false true true

# Custom frequency (logs every 50 steps)
./submit_batch_inference.sh /path/to/checkpoint "input.wav" ./results "" "" false true true 50
```

**Parameter order**:
1. Checkpoint directory
2. Input files
3. Output directory
4. HF repo (empty = default)
5. Max length (empty = unlimited)
6. Compute perplexity (false)
7. Generate audio (true)
8. **Enable TTT diagnostics (true/false)**
9. **Diagnostic log frequency (number)**

### Enable Diagnostics Programmatically

```python
from inference.enable_ttt_diagnostics import enable_ttt_diagnostics

# After loading model
enable_ttt_diagnostics(
    model=model,
    log_frequency=100,      # Log every 100 steps
    track_history=False     # Don't store history (saves memory)
)
```

## What Gets Logged

Every N steps (default: 100), you'll see detailed diagnostics like this:

```
================================================================================
📊 TTT Diagnostics - Layer 1 - Step 100
================================================================================

🔍 MAGNITUDE ANALYSIS:
   Attention:  1234.5678  (per-token:   3.0123)
   TTT:         234.5678  (per-token:   0.5678)
   Combined:   1456.7890  (per-token:   3.5432)
   TTT/Attn Ratio: 0.1900x

🎛️  GATING ANALYSIS:
   Alpha (mean):  0.048096
   Alpha (std):   0.025123
   Alpha (range): [-0.189453, 0.230469]
   Effective TTT weight: 0.951904

📐 COSINE SIMILARITY:
   Attn ↔ TTT:      0.1234
   Attn ↔ Combined: 0.9876
   TTT ↔ Combined:  0.2345

📈 DISTRIBUTION STATS:
   Attention:  μ=  0.0012, σ=  0.4567, sparsity= 12.3%
   TTT:        μ= -0.0023, σ=  0.2345, sparsity= 23.4%
   Combined:   μ=  0.0009, σ=  0.4890, sparsity= 10.5%

Δ CHANGE FROM PREVIOUS STEP:
   Attention Δ:     45.678901
   TTT Δ:           12.345678  (relative: 0.052635)

⚙️  TTT INTERNAL WEIGHTS:
   Weight norm: 32.250000
================================================================================
```

## Understanding the Metrics

### Magnitude Analysis

- **Attention magnitude**: Norm of attention output
- **TTT magnitude**: Norm of TTT output (before gating)
- **Combined magnitude**: Norm of final output (after gating)
- **TTT/Attn Ratio**: Shows relative strength of TTT vs attention

**What to look for**:
- Ratio > 1.0: TTT output is stronger than attention
- Ratio < 0.1: TTT has very little effect
- Typical range: 0.1 - 0.5 for well-tuned models

### Gating Analysis

- **Alpha mean**: Average gating value (lower = more TTT influence)
- **Effective TTT weight**: 1 - alpha (shows actual TTT contribution)

**What to look for**:
- Alpha ≈ 0.05: TTT contributes ~95% (strong TTT)
- Alpha ≈ 0.5: 50/50 blend
- Alpha ≈ 0.95: Mostly attention (weak TTT)

### Cosine Similarity

Measures directional alignment (-1 to 1):
- **+1**: Perfectly aligned (same direction)
- **0**: Orthogonal (independent directions)
- **-1**: Opposite directions

**What to look for**:
- `Attn ↔ TTT` ≈ 0: TTT provides independent information (good!)
- `Attn ↔ Combined` ≈ 1: Output mostly follows attention
- `TTT ↔ Combined` high: TTT dominates output

### Distribution Statistics

- **Mean (μ)**: Average activation value
- **Std (σ)**: Spread of activation values
- **Sparsity**: Percentage of near-zero values

**What to look for**:
- High sparsity (>50%): Very selective activations
- Low sparsity (<10%): Dense activations
- TTT sparsity > attention: TTT is more selective

### Delta/Change Tracking

Shows how much outputs change from previous step:
- **Absolute delta**: Total change magnitude
- **Relative change**: Change normalized by current magnitude

**What to look for**:
- Large deltas: Model is adapting/learning
- Small deltas: Model has converged
- TTT delta > attention: TTT is more dynamic

## Logging Frequency Guidelines

| Use Case | Frequency | Why |
|----------|-----------|-----|
| **Quick check** | 500-1000 | Minimal overhead, spot check |
| **Normal analysis** | 100 (default) | Good balance of detail vs log size |
| **Detailed debugging** | 10-50 | High detail, large logs |
| **Every step** | 1 | Maximum detail, very verbose |

**Note**: Lower frequency = more logs = larger files. For 20k token sequences:
- Frequency 100 = ~200 diagnostic outputs
- Frequency 10 = ~2000 diagnostic outputs

## Disabling Hash-Based Logging

The old hash-based "Weights CHANGED" logging has been **disabled by default** since it's redundant with the diagnostic logger. If you see excessive `✓ L1: Weights CHANGED` messages, they come from the TTT layer's internal logging, which now only logs every 100 steps.

## Performance Impact

Diagnostic logging adds minimal overhead:
- **Memory**: ~1-2MB per layer (no history tracking)
- **Compute**: <1% slowdown (just norm/similarity calculations)
- **Storage**: ~500KB per 1000 steps in logs

For inference on long sequences (>10k tokens), diagnostics are safe to enable.

## Example Use Cases

### 1. Verify TTT is Learning During Inference

Enable diagnostics and check:
- TTT weight norm is changing
- TTT delta is non-zero
- Gating alpha is evolving

### 2. Understand TTT Contribution

Check magnitude ratio and gating:
- Ratio 0.2, alpha 0.05 → Strong TTT contribution
- Ratio 0.01, alpha 0.95 → TTT barely used

### 3. Debug Unexpected Behavior

Enable high-frequency logging (10-20):
- Look for sudden magnitude spikes
- Check if cosine similarity changes dramatically
- Verify TTT weights are stable

### 4. Compare Checkpoints

Run with same input on different checkpoints:
- Compare gating alpha values
- Compare magnitude ratios
- See how training affects TTT contribution

## Troubleshooting

### "No TTT layers found"

The model doesn't have TTT layers. Check:
- Checkpoint is from TTT training run
- `training_config.json` shows `ttt.enable = true`

### Logs are too verbose

Increase log frequency:
```bash
# Log every 500 steps instead of 100
./submit_batch_inference.sh ... true 500
```

### Want to track history for plotting

Enable in Python:
```python
enable_ttt_diagnostics(model, log_frequency=100, track_history=True)

# Later, get history
history = model.transformer.layers[1].get_diagnostic_history()
```

## Related Files

- **Diagnostic Logger**: `moshi_ttt/diagnostic_logger.py`
- **Integration**: `moshi_ttt/hybrid_layer.py`
- **Enable Utility**: `inference/enable_ttt_diagnostics.py`
- **Batch Inference**: `inference/run_batch_inference.py`

## Architecture Reference

The diagnostic logger captures outputs at these points:

```
Input
  ↓
[Attention] ← Captured as attn_output
  ↓
[Format Conversion]
  ↓
[TTT Processing] ← Captured as ttt_output (before gating)
  ↓
[Gating: α*attn + (1-α)*ttt] ← Gating alpha captured
  ↓
[Format Conversion]
  ↓
Output ← Captured as combined_output
```

All three outputs are compared to understand TTT's effect.
