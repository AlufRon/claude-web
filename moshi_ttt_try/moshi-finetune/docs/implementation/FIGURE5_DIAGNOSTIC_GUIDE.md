# TTT Figure 5 Diagnostic Guide

## 🎯 Purpose

This diagnostic tool analyzes **why TTT inner-loop learning failed** by implementing the paper's Figure 5 methodology. It tracks three loss values at each token position during inference:

1. **l0 (Blue)**: Loss with frozen initial weights W₀ (no adaptation)
2. **lprev (Orange)**: Loss before gradient update (accumulated learning from previous tokens)
3. **lafter (Green)**: Loss after gradient update (immediate improvement from current token)

## ✅ Expected Healthy TTT Behavior

```
Blue > Orange > Green
 l0  > lprev  > lafter
```

- **Blue > Orange**: Shows cumulative benefit from adapting to context
- **Orange > Green**: Shows per-step learning is working
- **Decreasing trend**: All curves should go down as context grows

## ❌ Failed TTT Training Symptoms

### Symptom 1: **Lines Overlap**
```
Blue ≈ Orange ≈ Green
```
**Diagnosis**: TTT is not learning at all
- Inner-loop gradients are zero
- Learning rate too small
- Gradient flow broken

### Symptom 2: **Wrong Ordering**
```
Green > Orange  (update makes things worse!)
```
**Diagnosis**: Degenerative learning
- Learning rate too high (overshooting)
- Numerical instability
- Wrong gradient signs

### Symptom 3: **Huge Loss Values**
```
l0 > 10.0  (should be 0.01-1.0)
```
**Diagnosis**: Numerical explosion
- TTT weights diverging
- Learning rate way too high
- Overflow/NaN issues

### Symptom 4: **Increasing Trend**
```
lprev increases over time
```
**Diagnosis**: No context adaptation
- TTT not using context information
- Feature extraction broken
- Architecture mismatch

## 🚀 Usage

### Quick Start

```bash
cd /home/alufr/ttt_tests/moshi-finetune

# Use default checkpoint (LibriLight pretrained)
./submit_figure5_diagnostic.sh

# Or specify custom checkpoint
./submit_figure5_diagnostic.sh \
    /path/to/checkpoint/consolidated/ \
    /path/to/input/audio.wav \
    ./my_diagnostics_output/
```

### What It Does

1. **Loads TTT checkpoint** with Figure 5 logging enabled
2. **Runs streaming inference** on input audio
3. **Tracks 3 losses** at every token position (up to max_T=2048)
4. **Generates diagnostic report** with:
   - Loss statistics (mean, std, min, max)
   - Ordering violation rate
   - Per-step improvement metrics
   - Trend analysis
   - Automated diagnosis
5. **Creates Figure 5 plots** showing:
   - Three loss curves (blue/orange/green)
   - Per-step improvement
   - Cumulative benefit

### Output Files

```
figure5_diagnostics/
├── ttt_diagnostic_report.txt       # Text analysis with diagnosis
├── figure5_layer29.png             # Plot for layer 29
├── figure5_layer30.png             # Plot for layer 30
├── figure5_layer31.png             # Plot for layer 31
└── output_with_figure5.wav         # Generated audio
```

## 📊 Reading the Diagnostic Report

### Example: Healthy TTT

```
LAYER 29 ANALYSIS
================================================================

📊 Data Coverage:
   Valid positions: 1328 / 2048
   Position range: 0 to 1327

📈 Loss Statistics:
   l0 (W₀):     mean=0.125000, std=0.045000, min=0.080000, max=0.200000
   lprev (Wₜ₋₁): mean=0.085000, std=0.030000, min=0.055000, max=0.140000
   lafter (Wₜ):  mean=0.070000, std=0.025000, min=0.045000, max=0.115000

🎯 Learning Quality:
   Ordering violations: 12/1328 (0.9%)
   Avg per-step improvement: 0.015000

📉 Trend Analysis:
   l0 trend: -0.025000 (improving ✅)
   lprev trend: -0.030000 (adapting ✅)
```

**Interpretation**: 
- ✅ Losses in normal range (0.07-0.2)
- ✅ Clear ordering: l0 > lprev > lafter
- ✅ Positive per-step improvement (0.015)
- ✅ Decreasing trends (adapting to context)

### Example: Failed TTT (Your Current Situation)

```
LAYER 29 ANALYSIS
================================================================

📊 Data Coverage:
   Valid positions: 1328 / 2048
   Position range: 0 to 1327

📈 Loss Statistics:
   l0 (W₀):     mean=2.450000, std=1.200000, min=0.500000, max=15.000000
   lprev (Wₜ₋₁): mean=2.445000, std=1.198000, min=0.498000, max=14.980000
   lafter (Wₜ):  mean=2.440000, std=1.195000, min=0.495000, max=14.950000

🎯 Learning Quality:
   Ordering violations: 658/1328 (49.5%)
   Avg per-step improvement: 0.000050

🔍 Diagnosis:
   ⚠️  HUGE LOSSES: Average l0=2.45 >> 1.0
       → TTT weights may be diverging or numerically unstable
       → Check: learning rate too high?

   ❌ HIGH VIOLATION RATE: 49.5% positions violate l0 >= lprev >= lafter
       → TTT learning is NOT working as expected

   ❌ MINIMAL IMPROVEMENT: 0.000050 per step
       → TTT gradient updates have negligible effect
       → Check: learning rate too small? Gradient flow broken?

📉 Trend Analysis:
   l0 trend: +0.350000 (degrading ❌)
   lprev trend: +0.345000 (degrading ❌)
   
   ❌ lprev INCREASING over time
       → TTT is NOT adapting to context
```

**Interpretation**:
- ❌ Losses way too high (2.4 vs expected 0.1-0.5)
- ❌ Lines almost overlap (no learning)
- ❌ Tiny improvement (0.00005 vs expected 0.01-0.05)
- ❌ Increasing trend (getting worse, not better)

**Root Causes**:
1. TTT learning rate (0.01) might be causing instability
2. Inner-loop gradients not flowing properly
3. TTT weights learned random noise, not useful features

## 🔧 Next Steps Based on Diagnosis

### If Losses Are Huge (>10.0)
→ **Lower TTT learning rate**
```yaml
# In training config
ttt_base_lr: 0.001  # Try 0.001 instead of 0.01
```

### If Lines Overlap (minimal improvement)
→ **Check gradient flow**
```python
# Add gradient monitoring during training
for name, param in ttt_params:
    if param.grad is not None:
        print(f"{name}: grad_norm={param.grad.norm():.6f}")
```

### If Wrong Ordering (degenerative)
→ **Add gradient clipping**
```python
# In inner loop
torch.nn.utils.clip_grad_norm_(ttt_params, max_norm=1.0)
```

### If Increasing Trend (no adaptation)
→ **Check TTT architecture**
- Are features being extracted properly?
- Is the MLP too deep/shallow?
- Is normalization breaking signal?

## 📝 Checkpoints to Test

### LibriLight Pretrained (checkpoint_033000)
```bash
./submit_figure5_diagnostic.sh \
    /sise/eliyanac-group/ron_al/librilight_ttt_pretrain_fixed_weights2/checkpoints/checkpoint_033000/consolidated/
```

### DailyTalk Finetuned (checkpoint_005500)
```bash
./submit_figure5_diagnostic.sh \
    /sise/eliyanac-group/ron_al/dailytalk_finetune_from_librilight5/checkpoints/checkpoint_005500/consolidated/
```

### Earlier Checkpoint (before collapse)
```bash
./submit_figure5_diagnostic.sh \
    /sise/eliyanac-group/ron_al/librilight_ttt_pretrain_fixed_weights2/checkpoints/checkpoint_010000/consolidated/
```

## 🎓 Understanding Figure 5

From the TTT paper (Sun et al.):

> "Figure 5 shows three loss curves: the loss with initial weights W₀ (blue), 
> the loss before each gradient step (orange), and the loss after each gradient 
> step (green). The gap between blue and green shows the cumulative benefit of 
> test-time training, while the gap between orange and green shows the 
> per-step learning effectiveness."

**Key Insight**: If your Figure 5 plots don't show clear separation with Blue > Orange > Green, your TTT training fundamentally failed.

## ⚙️ Advanced Options

### Track More Positions
```bash
export MAX_T=4096  # Track up to 4096 tokens
./submit_figure5_diagnostic.sh
```

### Different Input Audio
```bash
./submit_figure5_diagnostic.sh \
    $CHECKPOINT_DIR \
    /path/to/longer/audio.wav  # Test on different data
```

### Monitor Job Progress
```bash
# Watch logs in real-time
tail -f ttt_figure5_*.log

# Check for errors
tail -f ttt_figure5_*.err
```

## 🐛 Troubleshooting

### "No Figure 5 data collected"
- Check that TTT layers are actually being used
- Verify `fig5_set_logging()` is called before inference
- Ensure sequence length > mini_batch_size (16)

### "AttributeError: fig5_set_logging"
- Update `moshi_ttt/models/ssm/ops/ttt_mlp.py` with Figure 5 code
- Check FIGURE5_IMPLEMENTATION_COMPLETE.md for details

### Plots show flat lines
- TTT might not be active during inference
- Check gating alpha values (if <<0.001, TTT is bypassed)
- Verify checkpoint actually contains TTT weights

---

## 📚 References

- TTT Paper: "Learning to Learn in Context" (Sun et al.)
- Figure 5 Implementation: `/home/alufr/ttt_tests/FIGURE5_IMPLEMENTATION_COMPLETE.md`
- TTT Integration: `/home/alufr/ttt_tests/moshi-finetune/finetune/ttt_integration.py`
