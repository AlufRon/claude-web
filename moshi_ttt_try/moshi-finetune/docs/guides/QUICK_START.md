# ⚡ LibriLight Training - Quick Start Guide

## 🎯 One-Command Training Pipeline

### Step 1: Activate Environment
```bash
conda activate moshi_ttt_fixed
cd /home/alufr/ttt_tests/moshi-finetune
```

### Step 2: Pre-train on LibriLight (60K hours audio)
```bash
torchrun --nproc_per_node=4 train.py --config example/librilight_pretrain.yaml
```
**Duration:** ~24-48 hours | **Steps:** 10,000 | **Checkpoints:** Every 1000 steps

### Step 3: Fine-tune on DailyTalk (conversations)
```bash
# First: Edit example/dailytalk_from_librilight.yaml
# Set resume_from to your LibriLight checkpoint path
# Example: resume_from: /sise/eliyanac-group/ron_al/librilight_ttt_pretrain/checkpoints/checkpoint_010000/consolidated

torchrun --nproc_per_node=4 train.py --config example/dailytalk_from_librilight.yaml
```
**Duration:** ~4-6 hours | **Steps:** 2,000 | **Checkpoints:** Every 500 steps

### Step 4: Run Inference
```bash
cd /home/alufr/ttt_tests/moshi
python run_inference.py
```

---

## 🛠️ Resume Interrupted Training

### LibriLight Interrupted:
```bash
# Edit example/librilight_continue.yaml to set checkpoint path
torchrun --nproc_per_node=4 train.py --config example/librilight_continue.yaml
```

### DailyTalk Interrupted:
```bash
# Create config with resume_from pointing to interrupted checkpoint
# Set reset_optimizer: false, reset_step: false
torchrun --nproc_per_node=4 train.py --config your_resume_config.yaml
```

---

## 📊 Monitor Training

### Check Loss:
```bash
# Watch training logs
tail -f /sise/eliyanac-group/ron_al/librilight_ttt_pretrain/train.log
```

### Check Checkpoints:
```bash
# List saved checkpoints
ls -lh /sise/eliyanac-group/ron_al/librilight_ttt_pretrain/checkpoints/
```

### Check GPU Usage:
```bash
watch -n 1 nvidia-smi
```

---

## ✅ Verification Before Training

```bash
# 1. Check LibriLight index is complete
wc -l /sise/eliyanac-group/ron_al/librilight/extracted_medium/medium/librilight.jsonl
# Expected: 18,739 lines

# 2. Run tests
python tests/test_librilight_loading.py
# Expected: All tests pass ✅

# 3. Check environment
conda env list | grep moshi_ttt_fixed
# Expected: * next to moshi_ttt_fixed

# 4. Check GPU availability
nvidia-smi
# Expected: 4 GPUs visible
```

---

## 🔥 Pro Tips

1. **Use `tmux` for long training:**
   ```bash
   tmux new -s moshi_training
   # Run training
   # Detach: Ctrl+B, then D
   # Reattach: tmux attach -t moshi_training
   ```

2. **Monitor with WandB:**
   - Set `wandb.project` in config
   - View real-time metrics in browser
   - Track across multiple runs

3. **Save disk space:**
   - Set `num_ckpt_keep: 3` in config
   - Only keeps last 3 checkpoints
   - Deletes older ones automatically

4. **Faster debugging:**
   - Set `max_steps: 10` for quick test runs
   - Use `batch_size: 1` to reduce memory
   - Enable `gradient_checkpointing: false` for speed (if memory allows)

---

## 🚨 Common Issues

| Issue | Solution |
|-------|----------|
| OOM Error | Reduce `batch_size` or `duration_sec` |
| Slow training | Check GPU utilization, ensure 4 GPUs used |
| Index not found | Run `python scripts/create_librilight_index.py` |
| Checkpoint not found | Verify path in config is correct |

---

## 📁 Important Paths

```bash
# LibriLight data
/sise/eliyanac-group/ron_al/librilight/extracted_medium/medium/

# LibriLight index
/sise/eliyanac-group/ron_al/librilight/extracted_medium/medium/librilight.jsonl

# DailyTalk data
/sise/eliyanac-group/ron_al/daily-talk-contiguous/daily_train.jsonl

# Training outputs
/sise/eliyanac-group/ron_al/librilight_ttt_pretrain/          # LibriLight
/sise/eliyanac-group/ron_al/dailytalk_ttt_finetuned/          # DailyTalk

# Configs
/home/alufr/ttt_tests/moshi-finetune/example/*.yaml
```

---

## 🎯 Expected Results

### After LibriLight Pre-training:
- Loss: ~8-10 → ~3-5
- 10 checkpoints saved
- Model understands audio structure

### After DailyTalk Fine-tuning:
- Loss: ~5-6 → ~2-3
- Model understands conversations
- Ready for coherent speech inference

### After Inference:
- Clear, coherent speech output
- Natural-sounding audio
- Reasonable response time

---

**Need More Details?** → See `LIBRILIGHT_IMPLEMENTATION_COMPLETE.md`

**Ready to train!** 🚀
```bash
torchrun --nproc_per_node=4 train.py --config example/librilight_pretrain.yaml
```
