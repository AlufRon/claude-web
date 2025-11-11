# Exact Fixes Needed for Moshi-TTT

**Date**: 2025-11-10
**Priority**: CRITICAL - Implement immediately
**Estimated Time**: 2-3 hours

---

## Overview

This document provides **exact code changes** needed to fix the 3 most critical issues. These are simple, surgical fixes that can be applied immediately.

| Issue | File | Lines | Time | Impact |
|-------|------|-------|------|--------|
| **#2** Normalization Bug | `moshi_ttt/ttt_layer.py` | 463-478 | 10 min | 🔴 Critical |
| **#4** Gradient Flow (Quick Fix) | `finetune/args.py` | 74 | 2 min | 🔴 Critical |
| **#3** Batch Size Validation | `moshi_ttt/ttt_layer.py` | ~640 | 5 min | 🟡 Medium |
| **Config** Update Production Config | `configs/production_ttt_dailytalk.yaml` | N/A | 3 min | 🔴 Critical |

**Total Time**: ~20 minutes

---

## Fix #1: Normalization Bug (Issue #2) 🔴

### Problem

**File**: `moshi_ttt_try/moshi-finetune/moshi_ttt/ttt_layer.py`
**Lines**: 463-478

The `ln_reconstruction_target` function normalizes `XV` directly, but it should normalize `(XV - XK)`.

**Current (WRONG)**:
```python
def ln_reconstruction_target(self, XV, XK):
    """Layer norm reconstruction target following Video-DiT pattern"""
    B, L, num_heads, head_dim = XV.shape

    # Apply layer norm per head
    XV_normed = torch.zeros_like(XV)
    for h in range(num_heads):
        # Get weight and bias for this head
        weight = self.ttt_norm_weight[h]  # [head_dim]
        bias = self.ttt_norm_bias[h]      # [head_dim]

        # Apply layer norm: [B, L, head_dim]
        XV_h = XV[:, :, h, :]  # ← Normalizing XV, not (XV - XK)!
        XV_normed[:, :, h, :] = F.layer_norm(XV_h, (head_dim,), weight, bias, eps=1e-6)

    return XV_normed  # ← Returns normalized XV, should be normalized (XV - XK)
```

### Fix

**Replace lines 463-478** with:

```python
def ln_reconstruction_target(self, XV, XK):
    """Layer norm reconstruction target following Video-DiT pattern

    CRITICAL: Must normalize the DIFFERENCE (XV - XK), not XV alone.
    This is the TTT reconstruction target.
    """
    B, L, num_heads, head_dim = XV.shape

    # Compute difference FIRST (reconstruction target)
    diff = XV - XK  # ← CRITICAL: Compute difference before normalization

    # Apply layer norm per head to the DIFFERENCE
    diff_normed = torch.zeros_like(diff)
    for h in range(num_heads):
        # Get weight and bias for this head
        weight = self.ttt_norm_weight[h]  # [head_dim]
        bias = self.ttt_norm_bias[h]      # [head_dim]

        # Apply layer norm: [B, L, head_dim]
        diff_h = diff[:, :, h, :]  # ← Normalizing (XV - XK), CORRECT!
        diff_normed[:, :, h, :] = F.layer_norm(diff_h, (head_dim,), weight, bias, eps=1e-6)

    return diff_normed  # ← Returns normalized (XV - XK)
```

### Verification

**Test that the fix works**:

```python
# Add this test after the fix
def test_normalization_fix():
    ttt_layer = TTTMLP(...)

    # Create dummy inputs
    B, L, H, D = 2, 100, 8, 64
    XV = torch.randn(B, L, H, D)
    XK = torch.randn(B, L, H, D)

    # Call ln_reconstruction_target
    result = ttt_layer.ln_reconstruction_target(XV, XK)

    # Verify result is normalized (XV - XK), not XV
    # Check that result != normalized(XV)
    XV_only_norm = torch.zeros_like(XV)
    for h in range(H):
        weight = ttt_layer.ttt_norm_weight[h]
        bias = ttt_layer.ttt_norm_bias[h]
        XV_h = XV[:, :, h, :]
        XV_only_norm[:, :, h, :] = F.layer_norm(XV_h, (D,), weight, bias, eps=1e-6)

    # These should NOT be equal (if they are, fix didn't work)
    assert not torch.allclose(result, XV_only_norm, atol=1e-4), \
        "BUG: Still normalizing XV instead of (XV - XK)"

    # Verify result is close to normalized (XV - XK)
    diff = XV - XK
    diff_norm_expected = torch.zeros_like(diff)
    for h in range(H):
        weight = ttt_layer.ttt_norm_weight[h]
        bias = ttt_layer.ttt_norm_bias[h]
        diff_h = diff[:, :, h, :]
        diff_norm_expected[:, :, h, :] = F.layer_norm(diff_h, (D,), weight, bias, eps=1e-6)

    assert torch.allclose(result, diff_norm_expected, atol=1e-4), \
        "BUG: Result doesn't match normalized (XV - XK)"

    print("✓ Normalization fix verified!")

# Run test
test_normalization_fix()
```

### Impact

- **Training**: TTT will now optimize the correct reconstruction target
- **Performance**: Expect improved reconstruction accuracy
- **Required**: Must retrain all checkpoints with this fix

---

## Fix #2: Disable Persistent States for Training (Issue #4 Quick Fix) 🔴

### Problem

**File**: `moshi_ttt_try/moshi-finetune/finetune/args.py`
**Line**: 74

Persistent states are enabled by default, causing gradient flow corruption (Issue #4) and cross-file contamination (Issue #5) during training.

**Current (WRONG)**:
```python
@dataclass
class TTTArgs(Serializable):
    """Configuration for Test-Time Training (TTT) layers in Moshi"""
    enable: bool = False
    layers: str = "middle"
    base_lr: float = 1.0
    mini_batch_size: int = 16
    persistent_states: bool = True  # ← DEFAULT IS TRUE - causes Issues #4 and #5!
```

### Fix

**Change line 74** from:
```python
persistent_states: bool = True
```

**To**:
```python
persistent_states: bool = False  # Disabled for training (Video-DiT style). Prevents gradient flow bugs (Issue #4) and cross-file contamination (Issue #5).
```

### Complete Fixed Section

```python
@dataclass
class TTTArgs(Serializable):
    """Configuration for Test-Time Training (TTT) layers in Moshi"""
    enable: bool = False
    layers: str = "middle"  # "all", "middle", "none", or comma-separated indices like "1,3,5"
    base_lr: float = 1.0
    mini_batch_size: int = 16
    persistent_states: bool = False  # Disabled for training (Video-DiT style). Prevents gradient flow bugs (Issue #4) and cross-file contamination (Issue #5).
    initial_gating_alpha: float = 0.1  # Initial gating alpha for TTT layers
    override_gating_alpha_on_resume: bool = False  # Reset gating alpha when resuming from checkpoint
```

### Impact

- **Fixes Issue #4**: No gradient flow corruption (W parameters updated by optimizer only)
- **Fixes Issue #5**: No cross-file contamination (each chunk independent)
- **Fixes Issue #3**: Batch size > 1 becomes compatible (no state persistence)
- **Simplifies architecture**: Video-DiT proven approach
- **No retraining needed**: Can use existing checkpoints

### Note on Inference

For inference (streaming audio), persistent states CAN be enabled safely because:
- No optimizer running (no gradient conflict)
- `torch.no_grad()` (no gradient computation)
- Single file per session (no cross-file contamination)

To enable for inference only:
```python
# In inference script
model.eval()
with torch.no_grad():
    # Temporarily enable persistent states for streaming
    for module in model.modules():
        if hasattr(module, 'config') and hasattr(module.config, 'persistent_states'):
            module.config.persistent_states = True

    # Run inference...
```

---

## Fix #3: Batch Size Validation (Issue #3) 🟡

### Problem

**File**: `moshi_ttt_try/moshi-finetune/moshi_ttt/ttt_layer.py`
**Lines**: ~640-707 (in the `ttt` method)

When `persistent_states=True`, the code copies state from batch index `[0]` only. If batch size > 1, other batch items are ignored silently.

### Find the Location

Search for this code pattern:
```python
with torch.no_grad():
    self.W1.data.copy_(final_states["W1_states"][0])  # ← [0] = first batch item only!
```

This should be around lines 686-707 in the `ttt` method.

### Fix

**Add validation at the start of the `if self.persistent_states:` block**:

```python
def ttt(self, inputs):
    B, H, NC, C, D = inputs.shape

    # ... existing code ...

    if self.persistent_states:
        # ✓ ADD THIS VALIDATION
        if B > 1:
            raise ValueError(
                f"persistent_states=True requires batch_size=1, got batch_size={B}. "
                f"With persistent states, only the first batch item's state is saved. "
                f"Options: (1) Set batch_size=1 in training config, or "
                f"(2) Disable persistent_states (recommended for training)."
            )

        # Initialize from current parameters
        W1_states = torch.tile(self.W1.unsqueeze(0), dims=(B, 1, 1, 1))
        # ... rest of persistent states code ...
```

### Complete Fixed Code (Around Lines 640-650)

```python
def ttt(self, inputs):
    """
    Test-time training forward pass.

    Args:
        inputs: dict containing XQ, XK, XV, eta

    Returns:
        XQW_batch: Output after TTT adaptation [B, H, NC, C, D]
    """
    B, H, NC, C, D = inputs["XQ"].shape

    # Determine checkpoint group size
    if self.training:
        checkpoint_group_size = min(max(self.config.scan_checkpoint_group_size, 1), NC)
    else:
        checkpoint_group_size = 0  # No checkpointing during eval

    # ========== PERSISTENT STATES PATH ==========
    if self.persistent_states:
        # ✓ VALIDATION: Ensure batch_size=1 when using persistent states
        if B > 1:
            raise ValueError(
                f"persistent_states=True requires batch_size=1, got batch_size={B}. "
                f"With persistent states, only the first batch item's state is saved. "
                f"Options: (1) Set batch_size=1 in training config, or "
                f"(2) Disable persistent_states (recommended for training)."
            )

        # Initialize from current parameters (carrying state across chunks)
        W1_states = torch.tile(self.W1.unsqueeze(0), dims=(B, 1, 1, 1))
        b1_states = torch.tile(self.b1.unsqueeze(0), dims=(B, 1, 1, 1))
        W2_states = torch.tile(self.W2.unsqueeze(0), dims=(B, 1, 1, 1))
        b2_states = torch.tile(self.b2.unsqueeze(0), dims=(B, 1, 1, 1))

        # ... rest of code ...
```

### Impact

- **Safety**: Prevents silent bugs when batch_size > 1
- **Clear error messages**: Developer knows exactly what to fix
- **Minimal change**: Just adds validation, doesn't change logic

---

## Fix #4: Update Production Configuration 🔴

### Problem

**File**: `moshi_ttt_try/moshi-finetune/configs/production_ttt_dailytalk.yaml`

The production config likely doesn't explicitly set `persistent_states`, so it inherits the buggy default of `True`.

### Fix

**Add explicit `persistent_states: false` to the TTT section**:

```yaml
# TTT Configuration
ttt:
  enable: true
  layers: "middle"  # or "all" or specific indices
  base_lr: 1.0
  mini_batch_size: 16
  persistent_states: false  # ← ADD THIS LINE (Video-DiT style, prevents Issues #4 and #5)
  initial_gating_alpha: 0.1

  # Optional: Add comment explaining why
  # persistent_states disabled for training to prevent:
  #   - Issue #4: Gradient flow corruption
  #   - Issue #5: Cross-file state contamination
  # Using Video-DiT proven approach (independent chunks)
```

### Full Example Config

```yaml
# Production TTT-Moshi Training Configuration
# Updated: 2025-11-10 (Fixes Issues #2, #4, #5)

# Model paths
moshi_paths:
  hf_repo_id: "kyutai/moshiko-pytorch-bf16"

# Training parameters
max_steps: 10000
batch_size: 1  # Required when persistent_states=true (currently false, but keeping for safety)
learning_rate: 1e-4
gradient_checkpointing: true

# Data
data:
  train_data: "/path/to/dailytalk/train"
  eval_data: "/path/to/dailytalk/val"
  shuffle: false  # Sequential processing for TTT
  duration_sec: 10.0  # 10-second chunks

# TTT Configuration
ttt:
  enable: true
  layers: "middle"  # Apply TTT to middle layers
  base_lr: 1.0
  mini_batch_size: 16  # TTT processes 16 tokens per mini-batch
  persistent_states: false  # Video-DiT style (prevents Issues #4 and #5)
  initial_gating_alpha: 0.1

  # Multi-learning-rate configuration
  weight_lr_multiplier: 10.0  # TTT weights learn 10x faster
  alpha_lr_multiplier: 100.0  # Gating alpha learns 100x faster

  # Diagnostics (optional)
  log_inner_loop_losses: false  # Set true for debugging
  save_inner_loop_plots: false

# Optimizer
optim:
  lr: 1e-4
  weight_decay: 0.01
  pct_start: 0.1  # OneCycleLR warmup percentage

# Logging
log_freq: 10
eval_freq: 100
ckpt_freq: 1000
run_dir: "./runs/production_ttt_dailytalk"

# Weights & Biases (optional)
wandb:
  enabled: false
  project: "moshi-ttt"
  name: "production_dailytalk"
```

---

## Testing Checklist

After applying all fixes:

### 1. Unit Tests

```bash
# Test normalization fix
pytest tests/test_ttt_layer.py::test_normalization_fix -v

# Test batch size validation
pytest tests/test_ttt_layer.py::test_batch_size_validation -v
```

### 2. Quick Training Test

```bash
# Run 10 steps to verify training works
python training/train_ttt_production.py \
    --config configs/production_ttt_dailytalk.yaml \
    --max_steps 10 \
    --log_freq 1

# Verify:
# ✓ No errors
# ✓ Loss decreases or stays reasonable
# ✓ Logs show "persistent_states: false"
# ✓ All TTT parameters receive gradients
```

### 3. Inference Test

```bash
# Test inference still works
python inference/run_inference_with_ttt.py \
    --infile test_audio.wav \
    --outfile test_output.wav

# Verify:
# ✓ No errors
# ✓ Audio quality is good
# ✓ No artifacts
```

### 4. Gradient Flow Verification

Add to training script (temporary):

```python
# After loss.backward(), before optimizer.step()
if step % log_freq == 0:
    print("\n=== Gradient Check ===")
    for name, param in model.named_parameters():
        if 'W1' in name or 'W2' in name or 'ttt' in name:
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                print(f"{name}: grad_norm={grad_norm:.6f}")
            else:
                print(f"{name}: NO GRADIENT!")
    print("===================\n")
```

Expected output:
```
=== Gradient Check ===
transformer.layers.5.hybrid.W1: grad_norm=0.023451
transformer.layers.5.hybrid.b1: grad_norm=0.001234
transformer.layers.5.hybrid.W2: grad_norm=0.034567
transformer.layers.5.hybrid.b2: grad_norm=0.002345
===================
```

All TTT parameters should have non-zero gradients!

---

## Migration Guide

### For Existing Checkpoints

**Good news**: These fixes are **compatible with existing checkpoints**!

- Fix #1 (normalization): Changes training behavior, but doesn't affect model architecture
- Fix #2 (persistent_states): Config change only, doesn't affect saved weights
- Fix #3 (validation): Just adds a check, doesn't change anything

**To use existing checkpoint with new fixes**:

```python
# Load checkpoint normally
checkpoint = torch.load("checkpoint_step_5000.pt")
model.load_state_dict(checkpoint['model_state_dict'])

# Start training with fixed code
# - Normalization will be correct going forward
# - persistent_states=false will be used
# - No issues!
```

### For New Training Runs

**Recommended**:
1. Apply all fixes
2. Start fresh training run
3. Compare with old checkpoints to measure improvement

**Expected improvements**:
- Better reconstruction quality (Fix #1)
- Stable training (Fix #2)
- Correct gradient flow (Fix #2)

---

## Summary of Changes

| File | Change | Lines | Type |
|------|--------|-------|------|
| `moshi_ttt/ttt_layer.py` | Fix normalization | 463-478 | Code modification |
| `finetune/args.py` | Disable persistent_states | 74 | Config default |
| `moshi_ttt/ttt_layer.py` | Add batch size validation | ~640 | Add validation |
| `configs/*.yaml` | Update config | N/A | Config update |

**Total changes**:
- 2 code modifications
- 2 configuration changes
- ~30 lines of code changed
- ~20 minutes of work

---

## Expected Results After Fixes

### Training

✅ **Stable training**:
- Loss decreases smoothly
- No NaN gradients
- No divergence

✅ **Correct gradient flow**:
- All TTT parameters receive gradients
- W1, b1, W2, b2 all update properly
- No gradient/parameter mismatch

✅ **Better reconstruction**:
- TTT optimizes correct objective
- Reconstruction loss decreases properly
- Improved next-token prediction

### Inference

✅ **Still works correctly**:
- No changes needed for inference
- Can optionally enable persistent_states for streaming
- Same audio quality

---

## Next Steps

### Immediate (after applying fixes)

1. **Apply fixes** (20 minutes)
2. **Run tests** (30 minutes)
3. **Start training run** (1 week)

### Short-term (Week 2)

4. **Measure baseline performance** (see RECOMMENDED_APPROACH.md Phase 2)
5. **Compare with unfixed version** (if you have old checkpoints)
6. **Document improvements**

### Medium-term (Weeks 3-4)

7. **Run ablation study** (see RECOMMENDED_APPROACH.md Phase 3)
   - Test if persistent_states helps (with proper W_base/W_state separation)
   - Make evidence-based decision on architecture

---

## Questions?

**Q**: Can I apply these fixes to a running training job?

**A**: No, restart training. But you can resume from last checkpoint with new config.

**Q**: Will this break my existing checkpoints?

**A**: No! Fixes are compatible. Old checkpoints work with new code.

**Q**: Do I need to retrain from scratch?

**A**: Recommended for Fix #1 (normalization), but not required. You can continue from checkpoint.

**Q**: How long until I see improvements?

**A**: Immediately! Training should be more stable from step 1.

**Q**: What about Issue #1 (ring buffer)?

**A**: Separate issue. Can fix independently later (see RECOMMENDED_APPROACH.md).

---

**Document Version**: 1.0
**Date**: 2025-11-10
**Status**: Ready to implement immediately
**Estimated time**: 20 minutes
**Difficulty**: Easy (surgical changes)
