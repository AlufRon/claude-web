# TTT-Moshi Checkpoint System User Guide

**Complete guide to training, saving, loading, and evaluating TTT-Moshi checkpoints.**

---

## 🎯 Quick Start

### Train with Checkpoints
```bash
# Basic training with checkpoints every 1000 steps
python train.py --config configs/your_config.yaml
```

### Resume Training
```bash
# Resume from latest checkpoint automatically  
torchrun --nproc_per_node=1 train.py --config configs/your_config.yaml --checkpoint.resume true

# Resume from specific step
torchrun --nproc_per_node=1 train.py --config configs/your_config.yaml --checkpoint.resume true --checkpoint.resume_step 5000
```

### Evaluate Checkpoint
```bash
# Evaluate a saved checkpoint
python eval_from_checkpoint.py --checkpoint_dir runs/experiment/checkpoints/checkpoint_005000 --config configs/your_config.yaml
```

---

## 📋 Configuration Options

### In Your YAML Config
```yaml
# Checkpoint settings
do_ckpt: true              # Enable/disable checkpointing
ckpt_freq: 1000           # Save checkpoint every N steps (0 = only save at end)
num_ckpt_keep: 3          # Keep only last N checkpoints (null = keep all)
save_adapters: true       # Save only LoRA/TTT adapters (false = save full model)

# TTT settings (automatically saved in checkpoints)
ttt:
  enable: true
  layers: "1,5,10"        # Which layers have TTT
  base_lr: 1.0
  mini_batch_size: 16
  initial_gating_alpha: 0.1
```

---

## 🔄 Training Scenarios

### Scenario 1: Train from Scratch
```bash
# Start fresh training with periodic checkpoints
python train.py --config configs/my_experiment.yaml
```

**What happens:**
- Checkpoints saved every `ckpt_freq` steps to `run_dir/checkpoints/`
- All TTT parameters automatically included
- Old checkpoints deleted (keeps last `num_ckpt_keep`)

**Checkpoint structure:**
```
runs/my_experiment/
├── checkpoints/
│   ├── checkpoint_001000/
│   │   └── consolidated/
│   │       ├── consolidated.safetensors  # Full model
│   │       └── config.json
│   ├── checkpoint_002000/
│   └── checkpoint_003000/
└── args.yaml
```

### Scenario 2: Resume Interrupted Training
```bash
# Training was stopped at step 2500, resume from latest checkpoint
torchrun --nproc_per_node=1 train.py \
    --config configs/my_experiment.yaml \
    --checkpoint.resume true \
    --checkpoint.resume_step -1    # -1 = auto-find latest
```

**What happens:**
- Automatically finds `checkpoint_002000` (latest)
- Restores: model weights, optimizer state, scheduler, data position
- Training continues from step 2000 (not 2500!)
- TTT states preserved exactly

### Scenario 3: Fine-tune from Previous Checkpoint
```bash
# Start new training using weights from previous experiment
python train.py \
    --config configs/stage2_experiment.yaml \
    --checkpoint.init_state_dir /path/to/previous/checkpoint_005000
```

**What happens:**
- Loads model weights from previous checkpoint
- Optimizer starts fresh (new learning rate schedule)  
- Training starts from step 0
- TTT parameters transferred

### Scenario 4: Resume with Different Config
```bash
# Resume but with modified settings (e.g., different learning rate)
torchrun --nproc_per_node=1 train.py \
    --config configs/modified_experiment.yaml \
    --checkpoint.resume true \
    --checkpoint.resume_step 2000
```

**Important:** Model architecture must match (same TTT layers, same LoRA settings)

---

## 🔍 Evaluation Workflows

### Standard Evaluation
```bash
# Basic perplexity evaluation
python eval_from_checkpoint.py \
    --checkpoint_dir runs/experiment/checkpoints/checkpoint_005000 \
    --config configs/experiment.yaml \
    --eval_type standard
```

### Paper Metrics Evaluation  
```bash
# Run all paper benchmarks (SbLiMP, SStory, etc.)
python eval_from_checkpoint.py \
    --checkpoint_dir runs/experiment/checkpoints/checkpoint_005000 \
    --config configs/experiment.yaml \
    --eval_type paper_metrics
```

### Custom Evaluation
```bash
# Both standard and paper metrics
python eval_from_checkpoint.py \
    --checkpoint_dir runs/experiment/checkpoints/checkpoint_005000 \
    --config configs/experiment.yaml \
    --eval_type both \
    --batch_size 2    # Override batch size
```

---

## 🔧 TTT Parameter Handling

### What Gets Saved
All TTT parameters are automatically saved in checkpoints:

```python
# Core TTT parameters
- W1, W2, b1, b2                    # MLP weights and biases  
- learnable_ttt_lr_weight, _bias    # Adaptive learning rates
- ttt_norm_weight, ttt_norm_bias    # Normalization layers
- gating_alpha                      # Attention/TTT gating
- wq, wk, wv, wo weights/biases     # Attention projections
- post_norm weights/biases          # Post-attention normalization
```

### Verification
```bash
# Verify TTT parameters are saved correctly
python test_checkpoint_verification.py
```

Expected output:
```
✅ Found 40 TTT parameters out of 7830720576 total parameters
✅ Loaded 40 TTT parameters from checkpoint  
✅ Parameter values match (fp16 precision)
✅ CHECKPOINT SYSTEM VERIFICATION PASSED!
```

---

## 🗂️ File Structure Reference

### Complete Experiment Directory
```
runs/my_ttt_experiment/
├── args.yaml                      # Training configuration
├── checkpoints/
│   ├── checkpoint_001000/
│   │   └── consolidated/
│   │       ├── consolidated.safetensors    # Full model (7.8B params)
│   │       ├── lora.safetensors           # LoRA-only (if save_adapters=true)
│   │       └── config.json                # Model config
│   ├── checkpoint_002000/
│   └── checkpoint_003000/
├── wandb/                         # WandB logs (if enabled)
└── logs/                          # Training logs
```

### Checkpoint File Sizes
```bash
# Full model checkpoint (~15GB)
consolidated.safetensors: 15.6GB   # Full 7B Moshi + TTT parameters

# LoRA+TTT only checkpoint (~280MB)  
lora.safetensors: 286MB             # Only trainable adapters
```

---

## 📊 Monitoring and Debugging

### Check Available Checkpoints
```bash
# List all checkpoints in experiment
ls runs/my_experiment/checkpoints/
# Output: checkpoint_001000  checkpoint_002000  checkpoint_003000
```

### Inspect Checkpoint Contents
```python
import safetensors.torch

# Load and inspect checkpoint
state_dict = safetensors.torch.load_file("checkpoint_005000/consolidated/consolidated.safetensors")

# Count TTT parameters
ttt_params = [k for k in state_dict.keys() if any(x in k.lower() for x in ['ttt', 'w1', 'w2', 'gating'])]
print(f"TTT parameters: {len(ttt_params)}")

# Check parameter shapes
for name in ttt_params[:5]:
    print(f"{name}: {state_dict[name].shape}")
```

### Verify TTT Integration
```bash
# Run built-in verification
python test_checkpoint_verification.py

# Custom verification
python -c "
from finetune.wrapped_model import get_fsdp_model
from finetune.args import TrainArgs  
from moshi.models import loaders

args = TrainArgs.load('configs/my_config.yaml')
checkpoint_info = loaders.CheckpointInfo.from_hf_repo(args.moshi_paths.hf_repo_id)
model = get_fsdp_model(args, checkpoint_info)

ttt_count = sum(1 for n, p in model.named_parameters() if 'ttt' in n.lower())
print(f'TTT parameters: {ttt_count}')
"
```

---

## ⚡ Performance Tips

### Checkpoint Frequency
```yaml
# For development (frequent saves)
ckpt_freq: 100

# For production (less frequent, save disk space)  
ckpt_freq: 5000

# For final runs (only save at end)
ckpt_freq: 0
```

### Storage Management
```yaml
# Keep only last 3 checkpoints (saves disk space)
num_ckpt_keep: 3

# Keep all checkpoints (for analysis)
num_ckpt_keep: null

# Save only trainable adapters (much smaller files)
save_adapters: true
```

### Evaluation Optimization
```bash
# Fast evaluation (standard metrics only)
--eval_type standard

# Comprehensive evaluation (includes paper metrics)
--eval_type both

# Skip TTT verification (faster startup)
--no_ttt_verify
```

---

## 🚨 Troubleshooting

### Common Issues

#### 1. "Default process group has not been initialized"
**Problem:** Running without distributed setup
```bash
# Wrong:
python train.py --config ...

# Correct:
torchrun --nproc_per_node=1 train.py --config ...
```

#### 2. "Checkpoint directory not found"
**Problem:** Wrong checkpoint path
```bash
# Check available checkpoints
ls runs/experiment/checkpoints/

# Use full path to checkpoint directory
python eval_from_checkpoint.py --checkpoint_dir runs/experiment/checkpoints/checkpoint_005000
```

#### 3. "TTT parameters missing from checkpoint"
**Problem:** TTT not enabled during training
```yaml
# Make sure TTT is enabled in config
ttt:
  enable: true  # Must be true!
  layers: "1,5,10"
```

#### 4. "Parameter mismatch after loading"
**Problem:** Different model architecture
- Check TTT layer configuration matches
- Verify LoRA settings are identical
- Make sure model size is consistent

#### 5. "Out of memory during evaluation"
```bash
# Reduce batch size
--batch_size 1

# Or evaluate with smaller config
```

### Debug Commands
```bash
# Verify checkpoint system
python test_checkpoint_verification.py

# Check TTT parameter count
python -c "from your_model import *; print_ttt_params()"

# Test evaluation script
python eval_from_checkpoint.py --help
```

---

## 📝 Best Practices

### 1. Checkpoint Strategy
- **Development:** `ckpt_freq: 100`, `num_ckpt_keep: 5`
- **Production:** `ckpt_freq: 1000`, `num_ckpt_keep: 3`
- **Final runs:** `ckpt_freq: 5000`, `num_ckpt_keep: null`

### 2. Storage Management
- Use `save_adapters: true` for smaller checkpoints
- Set reasonable `num_ckpt_keep` to avoid filling disk
- Store checkpoints on fast storage for large models

### 3. Evaluation Workflow
- Always verify checkpoints with `test_checkpoint_verification.py`
- Use `eval_from_checkpoint.py` for standardized evaluation
- Run paper metrics evaluation for final model assessment

### 4. Resumption Safety
- Always backup configs before resuming with modifications
- Test resumption on small scale before large experiments
- Verify TTT parameters are preserved after resumption

---

## 🔗 Related Files

| File | Purpose |
|------|---------|
| `train.py` | Main training script with checkpoint integration |
| `finetune/checkpointing.py` | Core checkpoint save/load implementation |
| `eval_from_checkpoint.py` | Evaluation script for saved checkpoints |
| `test_checkpoint_verification.py` | Verification script for checkpoint system |
| `finetune/eval.py` | Standard evaluation functions |
| `finetune/paper_metrics.py` | Paper metrics evaluation |

---

**Need help?** Run `python test_checkpoint_verification.py` to verify your setup or check the troubleshooting section above.