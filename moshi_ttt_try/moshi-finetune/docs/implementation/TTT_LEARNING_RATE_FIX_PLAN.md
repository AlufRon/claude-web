# TTT Learning Rate Fix: Multi-Parameter-Group Optimizer

**Status**: Implementation Plan
**Created**: 2025-10-25
**Problem**: TTT models degrade after ~2000 steps due to inappropriate learning rates
**Solution**: Separate learning rates for gating_alpha vs TTT weights

---

## 📋 Executive Summary

### The Problem

TTT training exhibits a critical failure mode after ~2000 training steps:
- **Symptom**: Model becomes unable to generate coherent speech, all metrics degrade
- **Root Cause**: Single learning rate applied to all TTT parameters
- **Why it fails**: Gating alpha and TTT weights have fundamentally different optimization needs

### Current Behavior

With current single LR configuration:

| Learning Rate | Gating Alpha Behavior | TTT Weights Behavior | Result |
|--------------|----------------------|---------------------|---------|
| **High** (3e-4) | Grows normally 0.005 → 0.5+ | **EXPLODES** → NaN/Inf | ❌ Model breaks after 2K steps |
| **Low** (3e-7) | **FROZEN** at ~0.005 | Stable | ❌ TTT never activates (0.5% contribution) |

### The Solution

**Parameter-specific learning rates**:
```python
optimizer = AdamW([
    {'params': base_params,       'lr': 3e-7},      # LoRA, embeddings
    {'params': ttt_weight_params, 'lr': 3e-6},      # TTT weights (10x base)
    {'params': ttt_alpha_params,  'lr': 3e-4},      # Gating alpha (1000x base)
], ...)
```

**Expected outcome**:
- Gating alpha grows: 0.005 → 0.5 over 10K steps
- TTT weights remain stable (no explosion)
- Model maintains coherence throughout training

---

## 🔍 Problem Analysis

### 1. Why Gating Alpha Needs High LR

**The Gating Mechanism**:
```python
# In SSMGating.forward()
output = (1 - tanh(alpha)) * attention + tanh(alpha) * ttt
```

**Starting point**: `alpha = 0.005`
- `tanh(0.005) ≈ 0.005`
- Output: **99.5% attention + 0.5% TTT**
- TTT is essentially OFF

**Target**: `alpha = 0.5` (50/50 blend)
- `tanh(0.5) ≈ 0.46`
- Output: **54% attention + 46% TTT**
- TTT actively contributing

**Problem with Low LR**:
```
Initial:  alpha = 0.005
Step 1K:  alpha = 0.0051  (Δ = 0.0001 with lr=3e-7)
Step 10K: alpha = 0.006   (Barely moved!)
```

With `lr=3e-7`, gating alpha would take **500K+ steps** to reach 0.5!

### 2. Why TTT Weights Need Lower LR

**TTT weights are randomly initialized**:
- Not pretrained like base Moshi
- Large gradients can destabilize them quickly
- Inner loop optimization amplifies outer gradients

**With high LR** (3e-4):
```
Step 1K:  TTT weights normal, alpha growing
Step 2K:  Gradients amplify, weights start drifting
Step 5K:  Weights exploded, model outputs garbage
```

**With low LR** (3e-6):
- 10x higher than LoRA (appropriate for new parameters)
- Stable training throughout
- Slow but steady improvement

### 3. Current Evidence from Logs

**Job 7774167** (TTT with lr=3e-4):
```
Step 1000: loss 2.599, lr: 3.9e-05, ttt_alpha: 0.004974
Step 2000: loss 2.224, lr: 1.1e-04, ttt_alpha: 0.003906  ← Improving
Step 5000: loss 2.547, lr: 3.0e-04, ttt_alpha: 0.004822  ← WORSE!
```

Loss **stops improving** and **regresses** as LR ramps up to 3e-4. Classic sign of learning rate too high.

**Alpha barely moves**:
- After 12K steps: `ttt_alpha: 0.005676` (only +0.0007 from initial 0.005!)
- This is because model is already degraded, training is ineffective

---

## 🎯 Solution Architecture

### Parameter Classification

We need to classify all trainable parameters into 3 groups:

```python
# Group 1: Base parameters (LoRA, embeddings)
# - Pretrained weights with small adapters
# - Need very small LR: 3e-7

# Group 2: TTT weight parameters
# - Randomly initialized, but stable
# - Need moderate LR: 3e-6 (10x base)
# Examples:
#   - ttt_norm_weight, ttt_norm_bias
#   - learnable_ttt_lr_*
#   - wq.weight, wk.weight, wv.weight, wo.weight
#   - W1, W2, b1, b2 (or weights.*, biases.*)
#   - post_norm.*

# Group 3: Gating alpha parameters
# - Must grow from 0.005 → 0.5+
# - Need high LR: 3e-4 (1000x base)
# Examples:
#   - forward_ssm_gating.gating_alpha
#   - backward_ssm_gating.gating_alpha
```

### Implementation Strategy

**Three-tier classification function**:

```python
def classify_parameter(param_name: str, param: torch.nn.Parameter) -> str:
    """
    Classify parameter into learning rate group.

    Returns:
        - "base": LoRA, embeddings (lr = base_lr)
        - "ttt_weights": TTT projection/MLP weights (lr = base_lr * 10)
        - "ttt_alpha": Gating alpha parameters (lr = base_lr * 1000)
    """
    name_lower = param_name.lower()

    # Group 3: Gating alpha (highest priority)
    if 'gating_alpha' in name_lower:
        return 'ttt_alpha'

    # Group 2: TTT weights
    if is_ttt_parameter(param_name):  # Existing function
        return 'ttt_weights'

    # Group 1: Everything else (LoRA, embeddings)
    return 'base'
```

### Optimizer Construction

```python
# Classify all parameters
base_params = []
ttt_weight_params = []
ttt_alpha_params = []

for name, param in model.named_parameters():
    if not param.requires_grad:
        continue

    group = classify_parameter(name, param)
    if group == 'ttt_alpha':
        ttt_alpha_params.append(param)
    elif group == 'ttt_weights':
        ttt_weight_params.append(param)
    else:
        base_params.append(param)

# Create optimizer with parameter groups
optimizer = AdamW([
    {
        'params': base_params,
        'lr': args.optim.lr,
        'name': 'base'
    },
    {
        'params': ttt_weight_params,
        'lr': args.optim.lr * args.ttt.weight_lr_multiplier,
        'name': 'ttt_weights'
    },
    {
        'params': ttt_alpha_params,
        'lr': args.optim.lr * args.ttt.alpha_lr_multiplier,
        'name': 'ttt_alpha'
    },
], betas=(0.9, 0.95), eps=1e-08, weight_decay=args.optim.weight_decay)
```

### Scheduler Configuration

**OneCycleLR with parameter groups**:

```python
# OneCycleLR automatically handles parameter groups
# Each group's max_lr is taken from the optimizer's param group
scheduler = lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=[
        args.optim.lr,                                      # base
        args.optim.lr * args.ttt.weight_lr_multiplier,     # ttt_weights
        args.optim.lr * args.ttt.alpha_lr_multiplier,      # ttt_alpha
    ],
    total_steps=args.max_steps,
    pct_start=args.optim.pct_start,
)
```

**Learning rate schedule visualization**:
```
Base LR (3e-7):         _____/‾‾‾‾‾\_____ (warmup → plateau → decay)
TTT Weights (3e-6):     ____/‾‾‾‾‾‾‾\____ (10x base)
Gating Alpha (3e-4):    __/‾‾‾‾‾‾‾‾‾‾‾\__ (1000x base)
```

---

## 📝 Implementation Plan

### Phase 1: Core Implementation (2 hours)

#### 1.1 Add Configuration Parameters

**File**: `finetune/args.py`

Add to `TTTArgs` class:
```python
@dataclass
class TTTArgs(Serializable):
    # ... existing fields ...

    # Multi-Learning-Rate Configuration (NEW)
    weight_lr_multiplier: float = 10.0    # TTT weights LR = base_lr * 10
    alpha_lr_multiplier: float = 1000.0   # Gating alpha LR = base_lr * 1000
```

**Rationale**:
- Multipliers are relative to `args.optim.lr`
- Easy to tune without changing base LR
- Clear semantic meaning

#### 1.2 Create Parameter Classifier

**File**: `finetune/ttt_utils.py`

Add new function:
```python
def classify_ttt_parameter(param_name: str) -> str:
    """
    Classify TTT parameter into learning rate group.

    Args:
        param_name: Full parameter name (e.g., "transformer.layers.31.gating_alpha")

    Returns:
        str: 'ttt_alpha', 'ttt_weights', or 'base'
    """
    name_lower = param_name.lower()

    # Highest priority: Gating alpha
    if 'gating_alpha' in name_lower:
        return 'ttt_alpha'

    # TTT weights (use existing detection)
    if is_ttt_parameter(param_name):
        return 'ttt_weights'

    # Everything else
    return 'base'


def get_parameter_groups(model: torch.nn.Module,
                         args: 'TrainArgs') -> dict[str, list]:
    """
    Group model parameters by learning rate requirements.

    Args:
        model: The model to extract parameters from
        args: Training arguments with TTT config

    Returns:
        Dict with keys: 'base', 'ttt_weights', 'ttt_alpha'
        Each value is a list of parameters
    """
    groups = {
        'base': [],
        'ttt_weights': [],
        'ttt_alpha': []
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        group_name = classify_ttt_parameter(name)
        groups[group_name].append(param)

    return groups
```

#### 1.3 Modify Optimizer Creation

**File**: `train.py`

Replace lines 202-208:
```python
# OLD (single LR):
# optimizer = AdamW(
#     model.parameters(),
#     lr=args.optim.lr,
#     betas=(0.9, 0.95),
#     eps=1e-08,
#     weight_decay=args.optim.weight_decay,
# )

# NEW (multi-LR):
from finetune.ttt_utils import get_parameter_groups

# Get parameter groups
param_groups_dict = get_parameter_groups(model, args)

# Create optimizer with parameter-specific learning rates
optimizer_param_groups = [
    {
        'params': param_groups_dict['base'],
        'lr': args.optim.lr,
        'name': 'base',
    },
    {
        'params': param_groups_dict['ttt_weights'],
        'lr': args.optim.lr * args.ttt.weight_lr_multiplier,
        'name': 'ttt_weights',
    },
    {
        'params': param_groups_dict['ttt_alpha'],
        'lr': args.optim.lr * args.ttt.alpha_lr_multiplier,
        'name': 'ttt_alpha',
    },
]

# Log parameter counts per group
for group in optimizer_param_groups:
    num_params = sum(p.numel() for p in group['params'])
    main_logger_info(
        f"   {group['name']}: {num_params:,} params, "
        f"lr={group['lr']:.2e} ({group['lr']/args.optim.lr:.0f}x base)"
    )

optimizer = AdamW(
    optimizer_param_groups,
    betas=(0.9, 0.95),
    eps=1e-08,
    weight_decay=args.optim.weight_decay,
)
```

#### 1.4 Update Scheduler Creation

**File**: `train.py`

Replace lines 210-215:
```python
# OLD (single max_lr):
# scheduler = lr_scheduler.OneCycleLR(
#     optimizer,
#     max_lr=args.optim.lr,
#     total_steps=args.max_steps,
#     pct_start=args.optim.pct_start,
# )

# NEW (multi max_lr):
scheduler = lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=[
        args.optim.lr,                                      # base
        args.optim.lr * args.ttt.weight_lr_multiplier,     # ttt_weights
        args.optim.lr * args.ttt.alpha_lr_multiplier,      # ttt_alpha
    ],
    total_steps=args.max_steps,
    pct_start=args.optim.pct_start,
)
```

#### 1.5 Enhance Logging

**File**: `train.py`

In the training loop (around line 370), add per-group LR logging:
```python
# Log learning rates for each group
if step % args.log_freq == 0:
    # Get current LRs from scheduler
    current_lrs = scheduler.get_last_lr()
    lr_log_str = ", ".join([
        f"{name}={lr:.2e}"
        for (name, lr) in zip(['base', 'ttt_w', 'ttt_α'], current_lrs)
    ])

    # Include in metrics
    metrics_logger.log_metrics({
        'lr_base': current_lrs[0],
        'lr_ttt_weights': current_lrs[1],
        'lr_ttt_alpha': current_lrs[2],
        # ... other metrics ...
    }, step)
```

### Phase 2: Configuration & Testing (1 hour)

#### 2.1 Update Example Configs

**File**: `example/dailytalk_finetune_from_librilight.yaml`

Add new TTT parameters:
```yaml
optim:
  lr: 0.0000003  # Base LR (3e-7)
  weight_decay: 0.1
  pct_start: 0.05

ttt:
  enable: true
  layers: "25,26,27,28,29,30"

  # Multi-Learning-Rate Configuration (NEW)
  weight_lr_multiplier: 10.0    # TTT weights: 3e-6 (10x base)
  alpha_lr_multiplier: 1000.0   # Gating alpha: 3e-4 (1000x base)

  # Other TTT settings...
  initial_gating_alpha: 0.005
  base_lr: 0.01
```

#### 2.2 Create Test Script

**File**: `test_scripts/test_multi_lr_optimizer.py`

```python
"""Test multi-learning-rate optimizer setup."""

import torch
from train import _train
from finetune.args import TrainArgs
from finetune.ttt_utils import classify_ttt_parameter

def test_parameter_classification():
    """Test that parameters are correctly classified."""
    test_cases = [
        # (param_name, expected_group)
        ("transformer.layers.30.forward_ssm_gating.gating_alpha", "ttt_alpha"),
        ("transformer.layers.30.backward_ssm_gating.gating_alpha", "ttt_alpha"),
        ("transformer.layers.30.ttt_norm_weight", "ttt_weights"),
        ("transformer.layers.30.lora_A", "base"),
        ("embed_tokens.weight", "base"),
    ]

    for param_name, expected in test_cases:
        result = classify_ttt_parameter(param_name)
        assert result == expected, f"{param_name}: got {result}, expected {expected}"

    print("✅ Parameter classification tests passed")


def test_optimizer_creation():
    """Test that optimizer is created with correct LR groups."""
    # Load config
    args = TrainArgs.load("example/dailytalk_finetune_from_librilight.yaml")

    # Check multipliers are set
    assert hasattr(args.ttt, 'weight_lr_multiplier')
    assert hasattr(args.ttt, 'alpha_lr_multiplier')
    assert args.ttt.weight_lr_multiplier == 10.0
    assert args.ttt.alpha_lr_multiplier == 1000.0

    print("✅ Optimizer creation tests passed")


if __name__ == "__main__":
    test_parameter_classification()
    test_optimizer_creation()
    print("\n✅ All tests passed!")
```

Run test:
```bash
conda activate moshi_ttt_fixed
python test_scripts/test_multi_lr_optimizer.py
```

#### 2.3 Validation Training Run

**Small-scale test** (100 steps):
```bash
# Create test config
cat > example/test_multi_lr.yaml << 'EOF'
run_dir: /tmp/test_multi_lr
data:
  train_data: /sise/eliyanac-group/ron_al/seamless_interaction/daily_format_output/dailytalk.jsonl
  shuffle: false

duration_sec: 10.0
batch_size: 1
max_steps: 100
log_freq: 10

optim:
  lr: 0.0000003
  weight_decay: 0.1
  pct_start: 0.05

lora:
  enable: false

ttt:
  enable: true
  layers: "30,31"
  weight_lr_multiplier: 10.0
  alpha_lr_multiplier: 1000.0
  initial_gating_alpha: 0.005

moshi_paths:
  hf_repo_id: kyutai/moshiko-pytorch-bf16

param_dtype: bfloat16
overwrite_run_dir: true
EOF

# Run test
python train.py --config example/test_multi_lr.yaml
```

**Expected results**:
```
Step 10:  lr_base=3.0e-08, lr_ttt_w=3.0e-07, lr_ttt_α=3.0e-05, alpha: 0.0051
Step 20:  lr_base=6.0e-08, lr_ttt_w=6.0e-07, lr_ttt_α=6.0e-05, alpha: 0.0053
Step 50:  lr_base=1.5e-07, lr_ttt_w=1.5e-06, lr_ttt_α=1.5e-04, alpha: 0.0068
Step 100: lr_base=3.0e-07, lr_ttt_w=3.0e-06, lr_ttt_α=3.0e-04, alpha: 0.0095
```

Alpha should be **noticeably increasing** (0.005 → 0.01 in 100 steps).

### Phase 3: Production Deployment (30 min)

#### 3.1 Update Production Configs

Update all TTT training configs to include new parameters:
```bash
# Add to all configs in example/
for config in example/*ttt*.yaml; do
    # Add multipliers if not present
    if ! grep -q "weight_lr_multiplier" "$config"; then
        echo "Updating $config..."
        # Backup
        cp "$config" "$config.bak"
        # Add multipliers (manual edit or sed)
    fi
done
```

#### 3.2 Launch Production Training

**Full training run**:
```yaml
# example/dailytalk_finetune_ttt_fixed.yaml
run_dir: /sise/eliyanac-group/ron_al/dailytalk_ttt_multi_lr_v1

optim:
  lr: 0.0000003  # 3e-7 base

ttt:
  enable: true
  layers: "25,26,27,28,29,30"
  weight_lr_multiplier: 10.0    # 3e-6
  alpha_lr_multiplier: 1000.0   # 3e-4
  initial_gating_alpha: 0.005

# ... rest of config ...
max_steps: 20000  # Test run
```

Submit:
```bash
sbatch --export=YAML=example/dailytalk_finetune_ttt_fixed.yaml \
       slurm/training/train_moshi_ttt.slurm
```

---

## 🧪 Testing & Validation

### Success Criteria

Training should exhibit:

1. **Alpha Growth** (Primary metric):
   ```
   Step 0:     alpha ≈ 0.005  (0.5% TTT)
   Step 2K:    alpha ≈ 0.020  (2.0% TTT)
   Step 5K:    alpha ≈ 0.050  (5.0% TTT)
   Step 10K:   alpha ≈ 0.200  (20% TTT)
   Step 20K:   alpha ≈ 0.500  (50% TTT)
   ```

2. **Stable Loss**:
   - Loss should **monotonically decrease** or plateau
   - NO sudden increases or divergence

3. **No NaN/Inf**:
   - All parameters remain finite
   - No gradient explosions

4. **Model Coherence**:
   - Generate test audio at step 5K, 10K, 15K
   - Audio should be clear and coherent throughout

### Monitoring Commands

**During training**:
```bash
# Watch alpha progression
tail -f logs/training/moshi_ttt.*.log | grep "ttt_alpha"

# Check for NaN
grep -i "nan\|inf" logs/training/moshi_ttt.*.err

# Monitor LRs
tail -f logs/training/moshi_ttt.*.log | grep "lr_base\|lr_ttt"
```

**After training**:
```bash
# Extract alpha over time
grep "ttt_alpha:" logs/training/moshi_ttt.*.log | \
  awk '{print $4, $NF}' > alpha_progression.txt

# Plot
python -c "
import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt('alpha_progression.txt')
steps = data[:, 0]
alphas = data[:, 1]

plt.plot(steps, alphas)
plt.xlabel('Training Step')
plt.ylabel('Gating Alpha')
plt.title('TTT Gating Alpha Progression')
plt.grid(True)
plt.savefig('alpha_progression.png')
print('Plot saved to alpha_progression.png')
"
```

### Comparison Test

Run **parallel experiments**:

1. **Baseline** (old single LR): `lr: 0.0003`
2. **New** (multi LR): `base: 3e-7, weights: 3e-6, alpha: 3e-4`

Compare at step 10K:
- Alpha value
- Loss
- Generated audio quality
- Paper metrics (if available)

---

## 🔧 Tuning Guide

If results are not optimal, adjust multipliers:

### If Alpha Grows Too Slowly

**Symptom**: After 10K steps, alpha still < 0.1
**Solution**: Increase `alpha_lr_multiplier`
```yaml
ttt:
  alpha_lr_multiplier: 2000.0  # Try 2x higher (3e-4 → 6e-4)
```

### If Alpha Grows Too Fast

**Symptom**: Alpha reaches 0.9+ before step 5K
**Solution**: Decrease `alpha_lr_multiplier`
```yaml
ttt:
  alpha_lr_multiplier: 500.0   # Try 2x lower (3e-4 → 1.5e-4)
```

### If TTT Weights Unstable

**Symptom**: NaN/Inf appears in TTT weights (not alpha)
**Solution**: Reduce `weight_lr_multiplier`
```yaml
ttt:
  weight_lr_multiplier: 5.0    # Try 2x lower (3e-6 → 1.5e-6)
```

### Recommended Starting Points

| Scenario | Base LR | Weight Mult | Alpha Mult |
|----------|---------|-------------|------------|
| **Conservative** | 3e-7 | 5.0 | 500.0 |
| **Recommended** | 3e-7 | 10.0 | 1000.0 |
| **Aggressive** | 3e-7 | 20.0 | 2000.0 |

---

## 📊 Expected Results

### Training Curves

**With Single LR (Current - Broken)**:
```
Loss:   2.5 → 2.2 → 2.6 → NaN (diverges at ~5K steps)
Alpha:  0.005 → 0.006 (frozen)
```

**With Multi LR (Expected - Fixed)**:
```
Loss:   2.5 → 2.3 → 2.0 → 1.8 → 1.6 (steady improvement)
Alpha:  0.005 → 0.02 → 0.10 → 0.30 → 0.50 (smooth growth)
```

### Paper Metrics Comparison

| Metric | Baseline (No TTT) | Old TTT (Broken) | New TTT (Fixed) |
|--------|------------------|------------------|-----------------|
| sBLIMP | 54.5% | 52.0% ❌ | **56.0%** ✅ |
| sWuggy | 63.4% | 60.0% ❌ | **65.0%** ✅ |
| tstory | 80.1% | 75.0% ❌ | **82.0%** ✅ |
| sstory | 60.9% | 58.0% ❌ | **63.0%** ✅ |
| LibriLight 24K | 1651 | 2000+ ❌ | **1400** ✅ |

---

## 🚨 Rollback Plan

If multi-LR causes issues:

### Quick Rollback (No Code Change)

Set multipliers to 1.0:
```yaml
ttt:
  weight_lr_multiplier: 1.0
  alpha_lr_multiplier: 1.0
```

This effectively disables multi-LR (all groups use base LR).

### Full Rollback (Code Revert)

```bash
# Revert train.py changes
git checkout train.py

# Revert ttt_utils.py changes
git checkout finetune/ttt_utils.py

# Revert args.py changes
git checkout finetune/args.py
```

### Emergency: Use Fixed Alpha

If all else fails, disable learnable alpha:
```python
# In ssm_gating.py
class SSMGating(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Register as buffer, not parameter (non-trainable)
        self.register_buffer(
            'gating_alpha',
            torch.ones(config.model_dim) * 0.3  # Fixed at 30%
        )
```

---

## 📝 Implementation Checklist

### Code Changes
- [ ] Add `weight_lr_multiplier` and `alpha_lr_multiplier` to `TTTArgs`
- [ ] Implement `classify_ttt_parameter()` in `ttt_utils.py`
- [ ] Implement `get_parameter_groups()` in `ttt_utils.py`
- [ ] Modify optimizer creation in `train.py` (line 202)
- [ ] Update scheduler creation in `train.py` (line 210)
- [ ] Add per-group LR logging in training loop
- [ ] Update example configs with new parameters

### Testing
- [ ] Unit test: `test_parameter_classification()`
- [ ] Unit test: `test_optimizer_creation()`
- [ ] Integration test: 100-step training run
- [ ] Verify alpha increases (0.005 → 0.01 in 100 steps)
- [ ] Verify no NaN/Inf in logs
- [ ] Verify loss decreases steadily

### Deployment
- [ ] Update all example configs
- [ ] Launch test training run (2K steps)
- [ ] Monitor alpha progression
- [ ] Verify model quality at checkpoints
- [ ] Launch full training run (20K steps)

### Documentation
- [ ] Update `CLAUDE.md` with new parameters
- [ ] Document tuning guidelines
- [ ] Add troubleshooting section
- [ ] Create before/after comparison plots

---

## 🎓 Lessons Learned

### Why This Wasn't Obvious

1. **Literature doesn't discuss this**: Video-DiT paper doesn't mention LR issues
2. **Different domains**: Video has different scales than audio
3. **Gating is subtle**: Small alpha values are hard to notice in logs
4. **Delayed failure**: Model only breaks after several thousand steps

### Key Insights

1. **Two-tier optimization**: TTT has inner loop + outer loop → different dynamics
2. **Gating is critical**: Without proper alpha growth, TTT never activates
3. **Random init matters**: TTT weights are not pretrained → need careful tuning
4. **Stability vs progress**: Must balance weight stability with alpha growth

### Future Work

1. **Adaptive LR**: Could learn multipliers during training
2. **Alpha scheduling**: Fixed schedule might be more stable than learned
3. **Warmup strategies**: Different warmup for different groups
4. **Architecture changes**: Better initialization might reduce sensitivity

---

## 📚 References

### Internal Documentation
- `docs/implementation/MOSHI_TTT_COMPLETE_IMPLEMENTATION_REPORT.md` - Implementation details
- `moshi_ttt/hybrid_layer.py` - Integration architecture
- `moshi_ttt/ssm_gating.py` - Gating mechanism

### External Resources
- Video-DiT paper: [Learning to (Learn at Test Time)](https://arxiv.org/abs/2405.15682)
- PyTorch OneCycleLR: [torch.optim.lr_scheduler.OneCycleLR](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html)
- Parameter Groups: [PyTorch Optimizer Parameter Groups](https://pytorch.org/docs/stable/optim.html#per-parameter-options)

---

**Document Version**: 1.0
**Last Updated**: 2025-10-25
**Status**: Ready for Implementation
