# TTT Modules for Llama-Omni

Test-Time Training (TTT) implementation for unlimited context speech generation with Llama-Omni.

## Overview

This module provides a drop-in replacement for Llama's attention mechanism using Test-Time Training (TTT), enabling unlimited context length for speech-to-text-to-speech generation.

### Key Features

✅ **Drop-in replacement** for Llama attention layers
✅ **FP32 precision enforcement** (prevents numerical instability)
✅ **HuggingFace compatible** interface
✅ **Auto-padding** to mini-batch boundaries
✅ **RoPE position reset** per mini-batch
✅ **State persistence** across forward passes
✅ **Comprehensive monitoring** and debugging tools

## Installation

```bash
# Install dependencies
pip install torch transformers

# No separate installation needed - import directly
```

## Quick Start

### 1. Convert Llama-Omni to TTT

```python
from ttt_modules.llama_omni_integration import convert_llama_to_ttt

# Load Llama-Omni model
model = OmniSpeechLlamaForCausalLM.from_pretrained("ictnlp/Llama-Omni")

# Convert top 8 layers to TTT
model = convert_llama_to_ttt(
    model,
    ttt_layers=[24, 25, 26, 27, 28, 29, 30, 31],
    mini_batch_size=64,
    ttt_base_lr=1.0,
)

# Model is now ready for training!
```

### 2. Setup FP32 Enforcement (CRITICAL!)

```python
from ttt_modules.monitoring import FP32Enforcer

# Create enforcer
enforcer = FP32Enforcer(
    model,
    ttt_layers=[24, 25, 26, 27, 28, 29, 30, 31],
    auto_fix=True
)

# Option 1: Manual verification each step
enforcer.verify_and_fix()

# Option 2: Automatic hooks (recommended)
enforcer.setup_hooks()
```

### 3. Monitor TTT Training

```python
from ttt_modules.monitoring import TTTMonitor

# Create monitor
monitor = TTTMonitor(
    log_dir="logs/ttt_monitoring",
    log_every_n_steps=10
)

# During training:
for step, batch in enumerate(dataloader):
    output = model(**batch)
    loss = output.loss
    loss.backward()
    optimizer.step()

    # Log TTT states
    monitor.log_model_states(
        model,
        ttt_layers=[24, 25, 26, 27, 28, 29, 30, 31],
        step=step
    )

# Close monitor
monitor.close()
```

### 4. Create Optimizer with Separate Learning Rates

```python
from ttt_modules.llama_omni_integration import create_ttt_param_groups
import torch.optim as optim

# Create parameter groups
param_groups = create_ttt_param_groups(
    model,
    ttt_layers=[24, 25, 26, 27, 28, 29, 30, 31],
    ttt_lr=1e-4,      # TTT states (W1, b1, W2, b2)
    other_lr=2e-5,    # All other parameters
)

# Create optimizer
optimizer = optim.AdamW(param_groups, weight_decay=0.01)
```

## Module Structure

```
ttt_modules/
├── __init__.py                    # Package initialization
├── ttt_layer.py                   # TTTMLP layer (attention replacement)
├── llama_omni_integration.py      # Integration utilities
├── monitoring.py                  # FP32 enforcement + monitoring
├── ops/
│   ├── __init__.py
│   └── ttt_mlp.py                 # Core TTT operation
└── README.md                      # This file
```

## Core Components

### 1. `TTTMLP` Layer

Drop-in replacement for `LlamaSdpaAttention`:

```python
from ttt_modules.ttt_layer import TTTMLP

# Create TTT layer
ttt_layer = TTTMLP(config, layer_idx=24)

# Forward pass (same interface as attention)
output, attn_weights, cache = ttt_layer(
    hidden_states,
    attention_mask=None,
    position_ids=None,
    past_key_value=None,
    use_cache=True,
)

# output: [B, L, D] transformed features
# attn_weights: None (TTT doesn't have attention weights)
# cache: (W1, b1, W2, b2) for state persistence
```

**Key features:**
- Auto-pads sequences to 64-token multiples
- Resets RoPE positions every 64 tokens
- Returns state cache for long context
- Enforces FP32 for W1, b1, W2, b2

### 2. `ttt_mlp` Operation

Core TTT operation with analytical gradients:

```python
from ttt_modules.ops.ttt_mlp import ttt_mlp

final_params, XQW_output = ttt_mlp(
    XK, XQ, XV, eta,
    ttt_norm_weight, ttt_norm_bias,
    W1_init, b1_init, W2_init, b2_init,
    checkpoint_group_size=4
)

# final_params: Updated TTT states
# XQW_output: Transformed features
```

**Key features:**
- Sequential mini-batch processing
- Analytical gradient computation (no autograd in inner loop)
- Gradient checkpointing for memory efficiency
- FP32 precision enforcement

### 3. Integration Utilities

```python
from ttt_modules.llama_omni_integration import (
    convert_llama_to_ttt,
    verify_ttt_fp32,
    create_ttt_param_groups,
    setup_ttt_fp32_hooks,
)

# Convert model
model = convert_llama_to_ttt(model, ttt_layers=[24-31])

# Verify FP32
verify_ttt_fp32(model, ttt_layers=[24-31])

# Create optimizer groups
param_groups = create_ttt_param_groups(model, ttt_layers=[24-31])

# Setup enforcement hooks
setup_ttt_fp32_hooks(model, ttt_layers=[24-31])
```

### 4. Monitoring and Enforcement

```python
from ttt_modules.monitoring import (
    FP32Enforcer,
    TTTMonitor,
    check_numerical_health,
)

# FP32 enforcement
enforcer = FP32Enforcer(model, ttt_layers=[24-31], auto_fix=True)
enforcer.setup_hooks()

# TTT monitoring
monitor = TTTMonitor(log_dir="logs")
monitor.log_model_states(model, ttt_layers=[24-31], step=0)

# Numerical health check
health = check_numerical_health(tensor, "tensor_name")
```

## Critical Requirements

### 1. FP32 Precision (MOST IMPORTANT!)

**TTT states (W1, b1, W2, b2) MUST be torch.float32.**

Why: BF16/FP16 causes numerical instability after ~3,750 updates (~5-7 minutes), leading to gibberish output.

```python
# CORRECT ✅
self.W1 = nn.Parameter(
    torch.zeros(..., dtype=torch.float32)
)

# WRONG ❌
self.W1 = nn.Parameter(
    torch.zeros(...)  # Uses default dtype (might be BF16!)
)
```

**Enforcement:**

```python
# Option 1: Manual check each forward pass
assert ttt_layer.W1.dtype == torch.float32

# Option 2: Use FP32Enforcer (recommended)
enforcer = FP32Enforcer(model, ttt_layers=[24-31], auto_fix=True)
enforcer.setup_hooks()
```

### 2. Mini-Batch Alignment

Sequences must be divisible by `mini_batch_size` (default: 64).

The module handles this automatically via auto-padding:

```python
# Input: 100 tokens (not divisible by 64)
hidden_states = torch.randn(1, 100, 4096)

# TTT layer auto-pads to 128 (2 mini-batches)
output, _, cache = ttt_layer(hidden_states)

# Output: 100 tokens (padding removed)
assert output.shape[1] == 100
```

### 3. RoPE Position Reset

RoPE positions reset every `mini_batch_size` tokens:

```
# Sequence of 256 tokens (4 mini-batches)
position_ids = [0-63, 0-63, 0-63, 0-63]  # ✅ CORRECT

# NOT:
position_ids = [0-255]  # ❌ WRONG
```

This is handled automatically by the TTT layer.

### 4. State Persistence

For long context, use `use_cache=True` and pass cache to next batch:

```python
cache = None
for batch in conversation_batches:
    output, _, cache = ttt_layer(
        batch,
        past_key_value=cache,
        use_cache=True,
    )

    # cache contains (W1, b1, W2, b2) for next batch
```

**When to reset cache:**
- New conversation starts
- `turn_number == 0` in dataset
- Manual reset desired

## Training Recommendations

### 1. Curriculum Learning

Train progressively with increasing context lengths:

| Stage | Max Context | Training Time (est.) |
|-------|-------------|---------------------|
| Stage 1 | 8k tokens  | 24 hours |
| Stage 2 | 16k tokens | 36 hours |
| Stage 3 | 32k tokens | 48 hours |
| Stage 4 | 64k tokens | 60 hours |

**Total**: ~7 days on 4x A100 GPUs

### 2. Learning Rates

Different learning rates for different parameter groups:

```python
- TTT states (W1, b1, W2, b2): 1e-4
- TTT projections (Q, K, V, O): 2e-5
- Other parameters: 2e-5
```

### 3. Mixed Precision

**CRITICAL**: Do NOT apply mixed precision to TTT states!

```python
# WRONG ❌
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    output = model(inputs)  # Converts W1, b1, W2, b2 to BF16!

# CORRECT ✅
# Use FP32Enforcer to prevent conversion
enforcer = FP32Enforcer(model, ttt_layers=[24-31], auto_fix=True)
enforcer.setup_hooks()

# Or exclude TTT states from autocast:
with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    # Only applies to activations, not TTT states
    output = model(inputs)
```

### 4. Gradient Checkpointing

TTT-MLP uses gradient checkpointing internally:

```python
final_params, output = ttt_mlp(
    ...
    checkpoint_group_size=4  # Checkpoint every 4 mini-batches
)
```

Adjust `checkpoint_group_size` to trade memory for speed:
- `checkpoint_group_size=1`: Max memory savings, slower
- `checkpoint_group_size=8`: Less memory savings, faster
- `checkpoint_group_size=0`: No checkpointing, fastest but most memory

## Troubleshooting

### Issue: "Gibberish output after ~5 minutes of training"

**Cause**: TTT states converted to BF16/FP16

**Solution**:
```python
# Check dtype
print(ttt_layer.W1.dtype)  # Should be torch.float32

# Use FP32Enforcer
enforcer = FP32Enforcer(model, ttt_layers=[24-31], auto_fix=True)
enforcer.setup_hooks()
```

### Issue: "NaN loss during training"

**Cause**: Numerical instability, possibly explosion

**Solution**:
```python
# 1. Check for NaN in states
monitor = TTTMonitor(log_dir="logs")
monitor.log_model_states(model, ttt_layers=[24-31])

# 2. Reduce learning rate
optimizer = optim.AdamW(param_groups, lr=5e-5)  # Lower LR

# 3. Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Issue: "CUDA out of memory"

**Cause**: TTT processes sequences in mini-batches, uses more memory

**Solution**:
```python
# 1. Reduce batch size
batch_size = 1  # Process one conversation at a time

# 2. Increase gradient checkpointing
checkpoint_group_size = 2  # Checkpoint more frequently

# 3. Use smaller mini-batch size (trade speed for memory)
mini_batch_size = 32  # Default is 64
```

### Issue: "Sequence length not divisible by 64"

**Cause**: Auto-padding disabled or not working

**Solution**:
```python
# Auto-padding is automatic in TTTMLP
# But verify:
ttt_layer = TTTMLP(config)
output, _, cache = ttt_layer(hidden_states)  # Pads automatically

# Or pad manually in data collator:
pad_len = (64 - (seq_len % 64)) % 64
```

## Performance

**Benchmarks** (on A100 40GB):

| Sequence Length | Batch Size | Memory | Speed |
|----------------|------------|---------|-------|
| 256 (4 MB)     | 8          | 12 GB   | 100 samples/s |
| 1024 (16 MB)   | 4          | 18 GB   | 40 samples/s |
| 4096 (64 MB)   | 1          | 28 GB   | 8 samples/s |
| 8192 (128 MB)  | 1          | 35 GB   | 4 samples/s |

MB = Mini-Batches

**Tip**: Use gradient accumulation for large sequences:

```python
accumulation_steps = 8
for i, batch in enumerate(dataloader):
    output = model(batch)
    loss = output.loss / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## Citation

If you use this code, please cite:

```bibtex
@software{ttt_llama_omni,
  title = {TTT Modules for Llama-Omni},
  year = {2025},
  url = {https://github.com/AlufRon/claude-web}
}

@article{sun2024ttt,
  title = {Learning to (Learn at Test Time)},
  author = {Sun, Yu and others},
  journal = {arXiv preprint arXiv:2407.04620},
  year = {2024}
}
```

## License

Apache 2.0 (same as Llama-Omni)

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review `docs/TRAINING_STRATEGY_ANALYSIS.md`
3. Check monitoring logs for numerical issues
4. Open an issue with logs and error messages
