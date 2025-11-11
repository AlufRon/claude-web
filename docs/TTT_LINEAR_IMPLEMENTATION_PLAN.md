# TTTLinear Implementation Plan: Step-by-Step Guide

**Date**: 2025-11-11
**Purpose**: Detailed implementation guide for TTT-as-LoRA architecture
**Approach**: Follow exact LoRA integration pattern in moshi-finetune

---

## Overview

This document provides a complete, step-by-step implementation plan for replacing LoRA with TTTLinear in the moshi-finetune codebase. The implementation follows the exact same integration points and patterns as the current LoRA implementation.

---

## Current LoRA Integration Points

Based on analysis of moshi-finetune code, LoRA is integrated at these points:

### 1. **Module Definition** (`moshi/moshi/moshi/modules/lora.py`)
- `LoRALinear` class (lines 44-123)
- `replace_all_linear_with_lora()` function (lines 5-22)
- `replace_lora_with_linear()` function for merging (lines 25-41)

### 2. **Configuration** (`moshi-finetune/finetune/args.py`)
- `LoraArgs` dataclass (lines 10-20)
- Fields: `enable`, `rank`, `scaling`, `ft_embed`

### 3. **Model Initialization** (`moshi-finetune/finetune/wrapped_model.py`)
- Pass LoRA kwargs to model creation (lines 125-130)
- Initialize LoRA parameters (lines 147-150)
- Freeze non-LoRA parameters (lines 174-181)
- Special FSDP policy for mixed trainable/frozen (lines 27-58)

### 4. **Training** (`moshi-finetune/train.py`)
- Validation of LoRA config (lines 93-96)
- Pass LoRA config to model (lines 141-143)
- Save adapters option (line 351)

### 5. **Checkpointing** (`moshi-finetune/finetune/checkpointing.py`)
- `save_only_lora` option to save just adapters

---

## TTTLinear Implementation Steps

### Phase 1: Create TTTLinear Module

**File**: `moshi/moshi/moshi/modules/ttt_linear.py` (new file)

**What to implement**:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class TTTLinear(nn.Module):
    """
    TTT-based linear adapter layer.

    Replaces LoRA's fixed low-rank adaptation with adaptive TTT-based transformation.
    Uses test-time training to adapt weights during forward pass.

    Architecture:
        output = frozen_W(x) + scaling * TTT_adapted(x)

    Where TTT_adapted uses:
        1. Project input to TTT space (theta_K, theta_Q, theta_V)
        2. Adapt inner weights W1/b1 via mini-batch gradient descent
        3. Project back to output space (theta_out)

    Args:
        in_features: Input dimension
        out_features: Output dimension
        ttt_inner_dim: TTT hidden dimension (analogous to LoRA rank)
        scaling: Scaling factor for TTT contribution
        mini_batch_size: Size of mini-batches for TTT updates
        device: Device for initialization
        dtype: Data type for parameters
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        ttt_inner_dim: int,
        scaling: float,
        mini_batch_size: int = 8,
        bias: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.ttt_inner_dim = ttt_inner_dim
        self.scaling = scaling
        self.mini_batch_size = mini_batch_size
        assert not bias
        self.bias = bias

        # Frozen pretrained weights (same as LoRA)
        self.frozen_W = nn.Linear(
            in_features,
            out_features,
            bias=False,
            device=device,
            dtype=dtype
        )

        # TTT projection layers (replace LoRA A/B)
        self.theta_K = nn.Linear(
            in_features,
            ttt_inner_dim,
            bias=False,
            device=device,
            dtype=dtype
        )
        self.theta_Q = nn.Linear(
            in_features,
            ttt_inner_dim,
            bias=False,
            device=device,
            dtype=dtype
        )
        self.theta_V = nn.Linear(
            in_features,
            ttt_inner_dim,
            bias=False,
            device=device,
            dtype=dtype
        )
        self.theta_out = nn.Linear(
            ttt_inner_dim,
            out_features,
            bias=False,
            device=device,
            dtype=dtype
        )

        # TTT meta-parameters (what outer loop optimizes)
        self.W1_base = nn.Parameter(
            torch.randn(ttt_inner_dim, ttt_inner_dim, device=device, dtype=dtype) * 0.02
        )
        self.b1_base = nn.Parameter(
            torch.zeros(ttt_inner_dim, device=device, dtype=dtype)
        )

        # Learnable TTT learning rate
        self.lr_gate = nn.Parameter(
            torch.tensor(-2.0, device=device, dtype=torch.float32)
        )

        # Layer norm for TTT
        self.ttt_norm = nn.LayerNorm(ttt_inner_dim, device=device, dtype=dtype)

        # Register load hook (like LoRA)
        self._register_load_state_dict_pre_hook(TTTLinear._load_hook, with_module=True)

    @staticmethod
    def _load_hook(module, state_dict, prefix, *_):
        """Handle loading pretrained weights into frozen_W."""
        key_name = prefix + "weight"
        if key_name in state_dict:
            w_ref = state_dict.pop(key_name)
            state_dict[prefix + 'frozen_W.weight'] = w_ref

    def forward(self, x: torch.Tensor):
        """
        Forward pass with TTT adaptation.

        Args:
            x: [B, T, in_features]

        Returns:
            output: [B, T, out_features]
        """
        # Frozen path (same as LoRA)
        frozen_out = self.frozen_W(x)  # [B, T, out_features]

        # TTT adaptation path
        ttt_out = self._ttt_forward(x)  # [B, T, out_features]

        # Combined output
        return frozen_out + self.scaling * ttt_out

    def _ttt_forward(self, x: torch.Tensor):
        """
        TTT forward pass with mini-batch adaptation.

        Args:
            x: [B, T, in_features]

        Returns:
            ttt_output: [B, T, out_features]
        """
        B, T, D_in = x.shape

        # Project to TTT space
        K = self.theta_K(x)  # [B, T, ttt_inner_dim]
        Q = self.theta_Q(x)  # [B, T, ttt_inner_dim]
        V = self.theta_V(x)  # [B, T, ttt_inner_dim]

        # Normalize K and Q (helps stability)
        K = F.normalize(K, dim=-1)
        Q = F.normalize(Q, dim=-1)

        # Initialize TTT weights from base parameters
        W = self.W1_base.clone()  # [ttt_inner_dim, ttt_inner_dim]
        b = self.b1_base.clone()  # [ttt_inner_dim]

        # Convert lr_gate to positive learning rate
        lr = torch.sigmoid(self.lr_gate)

        outputs = []

        # Process in mini-batches (TTT inner loop)
        for i in range(0, T, self.mini_batch_size):
            end = min(i + self.mini_batch_size, T)

            K_mb = K[:, i:end, :]  # [B, mb_size, ttt_inner_dim]
            Q_mb = Q[:, i:end, :]  # [B, mb_size, ttt_inner_dim]
            V_mb = V[:, i:end, :]  # [B, mb_size, ttt_inner_dim]

            # Reconstruction target (TTT objective)
            # Target is V - K (difference between value and key)
            target = V_mb - K_mb  # [B, mb_size, ttt_inner_dim]

            # Current prediction using W and b
            pred = K_mb @ W.T + b  # [B, mb_size, ttt_inner_dim]

            # TTT update: compute gradients and update W, b
            error = pred - target  # [B, mb_size, ttt_inner_dim]

            # Gradients for W and b
            grad_W = (K_mb.transpose(-2, -1) @ error).mean(0)  # [ttt_inner_dim, ttt_inner_dim]
            grad_b = error.mean(dim=(0, 1))  # [ttt_inner_dim]

            # Gradient descent step (TTT inner loop)
            W = W - lr * grad_W
            b = b - lr * grad_b

            # Compute adapted output for this mini-batch
            adapted = Q_mb @ W.T + b  # [B, mb_size, ttt_inner_dim]
            adapted = self.ttt_norm(adapted)
            outputs.append(adapted)

        # Concatenate all mini-batch outputs
        ttt_hidden = torch.cat(outputs, dim=1)  # [B, T, ttt_inner_dim]

        # Project back to output space
        ttt_out = self.theta_out(ttt_hidden)  # [B, T, out_features]

        return ttt_out

    def __repr__(self) -> str:
        return "{}Linear(in_features={}, out_features={}, ttt_inner_dim={})".format(
            "TTT", self.in_features, self.out_features, self.ttt_inner_dim
        )


def replace_all_linear_with_ttt(
    module,
    ttt_inner_dim: int,
    scaling: float,
    mini_batch_size: int = 8,
    device=None,
    dtype=None
):
    """
    Recursively replace all Linear layers with TTTLinear layers.

    Args:
        module: PyTorch module to convert
        ttt_inner_dim: TTT hidden dimension (analogous to LoRA rank)
        scaling: Scaling factor for TTT contribution
        mini_batch_size: Size of mini-batches for TTT updates
        device: Device for initialization
        dtype: Data type for parameters
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            if device is None:
                this_device = child.weight.device
            else:
                this_device = device
            if dtype is None:
                this_dtype = child.weight.dtype
            else:
                this_dtype = dtype

            ttt = TTTLinear(
                child.in_features,
                child.out_features,
                ttt_inner_dim,
                scaling,
                mini_batch_size=mini_batch_size,
                device=this_device,
                dtype=this_dtype
            )
            # Transfer frozen weights
            ttt.frozen_W = child
            setattr(module, name, ttt)
        else:
            replace_all_linear_with_ttt(
                child,
                ttt_inner_dim,
                scaling,
                mini_batch_size=mini_batch_size,
                device=device,
                dtype=dtype
            )


def replace_ttt_with_linear(module):
    """
    Recursively replace all TTTLinear layers with Linear layers.

    This merges the frozen weights with the TTT adaptation for inference.
    Note: TTT is adaptive, so this freezes it to a static transformation.
    """
    for name, child in module.named_children():
        if isinstance(child, TTTLinear):
            # For TTT, we can only return the frozen weights
            # (TTT adaptation is dynamic, can't be merged statically)
            new_linear = nn.Linear(
                child.frozen_W.in_features,
                child.frozen_W.out_features,
                bias=False,
                device=torch.device('meta'),
                dtype=child.frozen_W.weight.dtype
            )
            new_linear.weight = nn.Parameter(
                child.frozen_W.weight.data,
                requires_grad=child.frozen_W.weight.requires_grad
            )
            setattr(module, name, new_linear)
        else:
            replace_ttt_with_linear(child)
```

**Estimated time**: 2-3 hours

---

### Phase 2: Add Configuration

**File**: `moshi-finetune/finetune/args.py`

**Changes**:

```python
@dataclass
class TTTArgs(Serializable):
    """Configuration for TTT-based adapters."""
    enable: bool = False
    inner_dim: int = 16  # TTT hidden dimension (like LoRA rank)
    scaling: float = 2.0
    mini_batch_size: int = 8
    ft_embed: bool = False

    def __post_init__(self) -> None:
        if self.enable:
            assert self.inner_dim > 0, "TTT inner_dim must be positive"
            assert self.scaling > 0.0, "TTT scaling must be positive"
            assert self.mini_batch_size > 0, "mini_batch_size must be positive"


@dataclass
class TrainArgs(Serializable):
    # ... existing fields ...

    # LoRA (keep for backward compatibility)
    lora: LoraArgs | None = field(default_factory=LoraArgs)

    # TTT (new)
    ttt: TTTArgs | None = field(default_factory=TTTArgs)

    # ... rest of fields ...

    def __post_init__(self) -> None:
        # ... existing validation ...

        # Validate TTT/LoRA mutual exclusivity
        if self.ttt.enable and self.lora.enable:
            raise ValueError("Cannot enable both TTT and LoRA simultaneously")
```

**Estimated time**: 30 minutes

---

### Phase 3: Update Model Initialization

**File**: `moshi-finetune/finetune/wrapped_model.py`

**Changes**:

```python
def initialize_ttt_parameters(model: torch.nn.Module, param_dtype: torch.dtype):
    """
    Initialize TTTLinear layers.

    - theta_K, theta_Q, theta_V, theta_out: Kaiming uniform
    - W1_base: Small random values (0.02 std)
    - b1_base: Zeros
    - lr_gate: -2.0 (sigmoid gives ~0.12 learning rate)
    """
    for m_name, module in model.named_modules():
        if all(p.is_meta for p in module.parameters()):
            for p_name, param in module.named_parameters():
                module._parameters[p_name] = torch.nn.Parameter(
                    torch.empty_like(param, device="cpu", dtype=param_dtype)
                )
                param = module._parameters[p_name]

                # Initialize based on parameter name
                if "theta" in m_name:
                    # Projection layers: Kaiming uniform
                    torch.nn.init.kaiming_uniform_(param, a=math.sqrt(5))
                elif m_name.endswith("W1_base"):
                    # TTT weights: small random
                    torch.nn.init.normal_(param, mean=0.0, std=0.02)
                elif m_name.endswith("b1_base"):
                    # TTT bias: zeros
                    torch.nn.init.zeros_(param)
                elif m_name.endswith("lr_gate"):
                    # Learning rate gate: -2.0
                    torch.nn.init.constant_(param, -2.0)
                elif "ttt_norm" in m_name:
                    # LayerNorm: standard initialization
                    if "weight" in p_name:
                        torch.nn.init.ones_(param)
                    elif "bias" in p_name:
                        torch.nn.init.zeros_(param)
                else:
                    raise ValueError(f"Unexpected TTT parameter: {m_name}.{p_name}")


def get_fsdp_model(
    args: TrainArgs, checkpointer_info: CheckpointInfo
) -> FullyShardedDataParallel | LMModel:
    # ... existing code ...

    with torch.device("meta"):
        # Determine which adapter to use
        lm_kwargs_overrides = {
            "gradient_checkpointing": args.gradient_checkpointing,
        }

        if args.ttt.enable:
            lm_kwargs_overrides["ttt"] = True
            lm_kwargs_overrides["ttt_inner_dim"] = args.ttt.inner_dim
            lm_kwargs_overrides["ttt_scaling"] = args.ttt.scaling
            lm_kwargs_overrides["ttt_mini_batch_size"] = args.ttt.mini_batch_size
        elif args.lora.enable:
            lm_kwargs_overrides["lora"] = True
            lm_kwargs_overrides["lora_rank"] = args.lora.rank
            lm_kwargs_overrides["lora_scaling"] = args.lora.scaling

        model = checkpointer_info.get_moshi(
            device="meta",
            dtype=param_dtype,
            lm_kwargs_overrides=lm_kwargs_overrides,
            load_weight=False,
        )

    if get_rank() == 0:
        # ... load state dict ...

        if args.ttt.enable and not args.full_finetuning:
            logger.info("Initializing TTT layers ...")
            initialize_ttt_parameters(model, param_dtype)
        elif args.lora.enable and not args.full_finetuning:
            logger.info("Initializing lora layers ...")
            initialize_lora_parameters(model, param_dtype)

        # ... rest of initialization ...

    # ... rest of function ...

    # Freeze parameters
    if args.ttt.enable and not args.full_finetuning:
        for name, param in model.named_parameters():
            if "ttt" in name or "theta" in name:
                param.requires_grad = True
            elif args.ttt.ft_embed and "emb" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    elif args.lora.enable and not args.full_finetuning:
        for name, param in model.named_parameters():
            if "lora" in name:
                param.requires_grad = True
            elif args.lora.ft_embed and "emb" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    else:
        for param in model.parameters():
            param.requires_grad = True

    # ... rest of function ...

    auto_wrap_policy = get_fsdp_policy(args.ttt.enable or args.lora.enable)

    # ... rest remains same ...
```

**Estimated time**: 1-2 hours

---

### Phase 4: Update Moshi LMModel

**File**: `moshi/moshi/moshi/models/lm.py`

**Changes**: Need to modify `LMModel.__init__()` to accept TTT kwargs and apply TTT to layers.

Look for where LoRA is currently applied (should be in model initialization), and add equivalent TTT path:

```python
class LMModel(nn.Module):
    def __init__(
        self,
        # ... existing args ...
        lora: bool = False,
        lora_rank: int = 64,
        lora_scaling: float = 2.0,
        # New TTT args
        ttt: bool = False,
        ttt_inner_dim: int = 16,
        ttt_scaling: float = 2.0,
        ttt_mini_batch_size: int = 8,
        **kwargs
    ):
        super().__init__()

        # ... existing initialization ...

        # Apply adapters
        if ttt:
            from moshi.modules.ttt_linear import replace_all_linear_with_ttt
            replace_all_linear_with_ttt(
                self,
                ttt_inner_dim=ttt_inner_dim,
                scaling=ttt_scaling,
                mini_batch_size=ttt_mini_batch_size
            )
        elif lora:
            from moshi.modules.lora import replace_all_linear_with_lora
            replace_all_linear_with_lora(
                self,
                rank=lora_rank,
                scaling=lora_scaling
            )
```

**Note**: Need to check exact location in lm.py where LoRA is applied. Use grep to find it.

**Estimated time**: 1 hour

---

### Phase 5: Update Training Script

**File**: `moshi-finetune/train.py`

**Changes**:

```python
def _train(args: TrainArgs, exit_stack: ExitStack):
    # ... existing code ...

    # Validation (around line 93-96)
    if args.full_finetuning:
        assert not args.lora.enable and not args.ttt.enable, \
            "LoRA/TTT should not be enabled for full finetuning."
    else:
        assert args.lora.enable or args.ttt.enable, \
            "LoRA or TTT should be enabled for partial finetuning"
        assert not (args.lora.enable and args.ttt.enable), \
            "Cannot enable both LoRA and TTT"

    # ... rest of function ...

    # Checkpointing (around line 350)
    if args.do_ckpt and (
        (args.ckpt_freq > 0 and state.step % args.ckpt_freq == 0) or is_last_step
    ):
        save_adapters_only = not args.full_finetuning and args.save_adapters
        checkpointer.save_checkpoint(
            save_only_lora=save_adapters_only and args.lora.enable,
            save_only_ttt=save_adapters_only and args.ttt.enable,
            dtype=param_dtype,
        )
```

**Estimated time**: 30 minutes

---

### Phase 6: Update Checkpointing

**File**: `moshi-finetune/finetune/checkpointing.py`

**Changes**: Add `save_only_ttt` parameter similar to `save_only_lora`:

```python
class Checkpointer:
    def save_checkpoint(
        self,
        save_only_lora: bool = False,
        save_only_ttt: bool = False,
        dtype: torch.dtype = torch.bfloat16
    ):
        # ... existing code ...

        if save_only_ttt:
            # Save only TTT parameters
            state_dict_to_save = {
                k: v for k, v in state_dict.items()
                if "ttt" in k or "theta" in k or "lr_gate" in k
            }
        elif save_only_lora:
            # Save only LoRA parameters
            state_dict_to_save = {
                k: v for k, v in state_dict.items()
                if "lora" in k
            }
        else:
            state_dict_to_save = state_dict

        # ... rest of saving logic ...
```

**Estimated time**: 30 minutes

---

### Phase 7: Create Example Config

**File**: `moshi-finetune/example/moshi_7B_ttt.yaml` (new file)

**Content**:

```yaml
# TTT-based fine-tuning configuration
# Similar to LoRA config but uses TTT adapters

run_dir: ./runs/moshi_ttt_experiment
seed: 42

# Model paths
moshi_paths:
  hf_repo_id: "kyutai/moshiko-pytorch-bf16"

# TTT configuration
ttt:
  enable: true
  inner_dim: 16          # TTT hidden dimension (like LoRA rank)
  scaling: 2.0           # Scaling factor for TTT contribution
  mini_batch_size: 8     # Mini-batch size for TTT updates
  ft_embed: false        # Whether to fine-tune embeddings

# Training
duration_sec: 10
batch_size: 1
num_microbatches: 4
max_steps: 1000
max_norm: 1.0
log_freq: 10
ckpt_freq: 100
do_ckpt: true
save_adapters: true      # Save only TTT adapters (not full model)
num_ckpt_keep: 3

# Optimizer
optim:
  lr: 1e-4
  weight_decay: 0.1
  pct_start: 0.05

# Efficiency
gradient_checkpointing: true
param_dtype: "bfloat16"

# Data
data:
  # ... your data config ...

# Weights
first_codebook_weight_multiplier: 1.0
text_padding_weight: 0.5

# Wandb (optional)
wandb:
  project: null  # Set to your wandb project name
```

**Estimated time**: 15 minutes

---

## Implementation Order

### Week 1: Core Implementation

**Day 1-2**: Phase 1 (TTTLinear module)
- Create `moshi/moshi/moshi/modules/ttt_linear.py`
- Implement `TTTLinear` class
- Implement `replace_all_linear_with_ttt()`
- Write unit tests for TTTLinear

**Day 3**: Phase 2 (Configuration)
- Add `TTTArgs` to args.py
- Update `TrainArgs` validation

**Day 4**: Phase 4 (Model integration)
- Find where LoRA is applied in lm.py
- Add TTT path to model initialization

**Day 5**: Phases 3, 5, 6 (Integration)
- Update wrapped_model.py
- Update train.py
- Update checkpointing.py

### Week 2: Testing and Validation

**Day 6**: Phase 7 + Testing
- Create example config
- Write integration tests
- Test parameter initialization

**Day 7**: End-to-end validation
- Run small training test
- Verify gradients flow correctly
- Check parameter counts match expectations

---

## Testing Checklist

### Unit Tests

- [ ] TTTLinear forward pass produces correct shapes
- [ ] TTTLinear gradient flow works (backward pass)
- [ ] replace_all_linear_with_ttt replaces all Linear layers
- [ ] Parameter initialization is correct
- [ ] TTT mini-batch loop works with various sequence lengths

### Integration Tests

- [ ] Model initializes with TTT enabled
- [ ] Only TTT parameters have requires_grad=True
- [ ] Parameter count matches expectations
- [ ] Training loop runs without errors
- [ ] Checkpointing saves only TTT parameters
- [ ] Can load TTT checkpoint and resume training

### Validation Tests

- [ ] Compare parameter count: TTT vs LoRA
- [ ] Train for 100 steps, verify loss decreases
- [ ] Check memory usage (should be similar to LoRA)
- [ ] Verify TTT adapts during forward pass (W1 changes)

---

## Parameter Count Verification

**For fair comparison with LoRA**:

```python
# LoRA parameters (rank=64, in=512, out=512):
lora_params = 2 * in_features * rank
            = 2 * 512 * 64
            = 65,536

# TTT parameters (inner_dim=16, in=512, out=512):
ttt_params = 3 * in_features * inner_dim     # theta_K, theta_Q, theta_V
           + inner_dim * out_features        # theta_out
           + inner_dim * inner_dim           # W1_base
           + inner_dim                       # b1_base
           + inner_dim * 2                   # ttt_norm
           + 1                               # lr_gate
         = 3 * 512 * 16 + 16 * 512 + 16 * 16 + 16 + 32 + 1
         = 24,576 + 8,192 + 256 + 16 + 32 + 1
         = 33,073

# To match LoRA parameter count, solve:
# 4 * in_features * inner_dim + inner_dim^2 + 3*inner_dim + 1 = lora_params
# For in=512, out=512:
# 4 * 512 * inner_dim + inner_dim^2 + 3*inner_dim + 1 = 65,536
# inner_dim ≈ 32 gives ~65K parameters
```

**Recommendation**:
- Use `ttt_inner_dim=32` to match LoRA rank=64 parameter count
- Or use `ttt_inner_dim=16` for ~half the parameters (faster, less memory)

---

## Potential Issues and Solutions

### Issue 1: TTT is slower than LoRA

**Cause**: Mini-batch loop and gradient computations
**Solution**:
- Use smaller mini_batch_size (e.g., 4 instead of 8)
- Optimize inner loop with fused operations
- Use torch.compile() on _ttt_forward method

### Issue 2: Training instability

**Cause**: TTT learning rate too high/low
**Solution**:
- Initialize lr_gate to different value (-3.0 for lr≈0.05, -1.0 for lr≈0.27)
- Add gradient clipping inside TTT loop
- Add normalization on K and Q (already included)

### Issue 3: OOM (Out of Memory)

**Cause**: TTT computational graph larger than LoRA
**Solution**:
- Use torch.no_grad() around parts of TTT update (careful!)
- Reduce mini_batch_size
- Use gradient checkpointing more aggressively

### Issue 4: Doesn't improve over LoRA

**Cause**: TTT adaptation may not help for all tasks
**Solution**:
- This is valid experimental result!
- Try different inner_dim values
- Try different mini_batch_size values
- Analyze what TTT learns (visualize W1 changes)

---

## Success Criteria

### Minimum Viable Product (MVP)

- [ ] TTTLinear module implemented and tested
- [ ] Can train with TTT enabled
- [ ] Checkpointing works
- [ ] No crashes or NaNs

### Full Success

- [ ] TTT trains stably for 1000+ steps
- [ ] Loss decreases (model is learning)
- [ ] Can compare TTT vs LoRA fairly
- [ ] Memory usage reasonable (< 2x LoRA)
- [ ] Speed reasonable (< 3x slower than LoRA)

### Stretch Goals

- [ ] TTT achieves lower loss than LoRA
- [ ] TTT shows evidence of test-time adaptation
- [ ] Can visualize what TTT learns across chunks
- [ ] Inference with TTT adaptation works

---

## Comparison Experiment Design

Once implementation is complete, run this experiment:

```bash
# Experiment 1: LoRA baseline
python train.py example/moshi_7B_lora.yaml

# Experiment 2: TTT with matched parameters
python train.py example/moshi_7B_ttt.yaml

# Compare:
# - Final loss
# - Training speed (steps/sec)
# - Memory usage
# - Audio quality (if applicable)
```

**Fair comparison**:
- Same training data
- Same number of steps
- Similar parameter count (LoRA rank=64 vs TTT inner_dim=32)
- Same batch size, learning rate, etc.

---

## File Structure Summary

```
moshi/
  moshi/
    moshi/
      modules/
        lora.py              # Existing LoRA implementation
        ttt_linear.py        # NEW: TTT implementation

moshi-finetune/
  finetune/
    args.py                  # MODIFY: Add TTTArgs
    wrapped_model.py         # MODIFY: Add initialize_ttt_parameters, update get_fsdp_model
    checkpointing.py         # MODIFY: Add save_only_ttt
  train.py                   # MODIFY: Add TTT validation, update checkpointing
  example/
    moshi_7B_lora.yaml       # Existing LoRA config
    moshi_7B_ttt.yaml        # NEW: TTT config
  tests/
    test_ttt_linear.py       # NEW: Unit tests
```

---

## Next Steps After Implementation

1. **Verify correctness**: Run all tests, check gradients
2. **Small-scale experiment**: Train for 100 steps, verify loss decreases
3. **Parameter count check**: Verify TTT params match expectations
4. **Compare with LoRA**: Run fair comparison experiment
5. **Analyze results**: Does TTT help? By how much?
6. **Iterate**: Based on results, tune hyperparameters or improve implementation

---

## Summary

This implementation plan provides:

✅ **Complete code** for TTTLinear module
✅ **Step-by-step** integration guide following LoRA pattern
✅ **Configuration** for TTT hyperparameters
✅ **Testing checklist** for validation
✅ **Comparison experiment** design
✅ **Troubleshooting** guide for common issues

**Estimated total time**: 7-10 days for complete implementation and initial testing

**Key advantage**: Follows exact same pattern as LoRA, so integration is straightforward and low-risk.

---

**Document Version**: 1.0
**Date**: 2025-11-11
**Status**: Ready for implementation
