# Testing TTT Gating Alpha with SLURM

## Quick Start

```bash
cd /home/alufr/ttt_tests/moshi-finetune

# Test 1: Pure TTT (see what TTT actually learned)
./submit_gating_test.sh 1.0

# Test 2: Pure base Moshi (baseline)
./submit_gating_test.sh 0.0

# Test 3: 50/50 blend
./submit_gating_test.sh 0.5

# Test 4: Original training target
./submit_gating_test.sh 0.3
```

## What Each Test Reveals

### Test 1: `./submit_gating_test.sh 1.0` (Pure TTT - 100%)
**Forces model to use ONLY TTT output, ignoring base Moshi**

**If output is QUIET/SHORT:**
- ✅ Proves: TTT weights themselves learned to be quiet
- Root cause: Training data (DailyTalk) taught it conservative behavior
- Next step: Need to retrain with different data/objectives

**If output is TALKATIVE/GOOD:**
- ✅ Proves: TTT learned good features, but gating is ignoring them
- Root cause: Gating mechanism collapsed during training
- Next step: Fix gating to prevent collapse

### Test 2: `./submit_gating_test.sh 0.0` (Pure Base - 0%)
**Forces model to use ONLY base Moshi, ignoring TTT**

**Should be:**
- Talkative and conversational (normal Moshi behavior)
- Use this as your baseline for comparison

### Test 3: `./submit_gating_test.sh 0.5` (Balanced - 50%)
**Equal blend of TTT and base Moshi**

- Middle ground
- See how they combine

### Test 4: `./submit_gating_test.sh 0.3` (Original Target - 30%)
**What the training originally intended**

- This was the `initial_gating_alpha` in training config
- See if it works better than learned values (~0.02-9%)

## Files Created

1. **test_gating_alpha.slurm** - SLURM batch script for gating tests
2. **submit_gating_test.sh** - Submission wrapper script

## Advanced Usage

### Custom checkpoint and input:
```bash
./submit_gating_test.sh 1.0 \
    /path/to/checkpoint \
    /path/to/input.wav
```

### Monitor the job:
```bash
# Get job ID from submission output, then:
squeue -j JOB_ID
tail -f moshi_gating_JOB_ID.log
tail -f moshi_gating_JOB_ID.err
```

## Expected Output in Logs

You should see:
```
🔧 FORCING gating_alpha override to: 1.0
   This will replace learned gating values from checkpoint
   ✅ Set transformer.layers.29.seq_modeling_block.forward_ssm_gating = 1.0
   ✅ Set transformer.layers.29.seq_modeling_block.backward_ssm_gating = 1.0
   ...
🔧 Overrode 6 gating parameters

📊 Gating Alpha Statistics (per TTT layer):
📍 transformer.layers.29.seq_modeling_block.forward_ssm_gating:
   Effective TTT contribution: 100.00% (avg), 100.00% (max)
```

## Interpreting Results

### Scenario A: Alpha=1.0 is quiet, Alpha=0.0 is talkative
**Conclusion**: TTT weights learned quiet behavior from training data

**Solutions:**
1. Retrain with LibriLight only (no DailyTalk)
2. Filter training data to remove silence
3. Adjust loss to penalize silence
4. Use data augmentation to add more verbal samples

### Scenario B: Alpha=1.0 is talkative, Alpha=0.0 is talkative
**Conclusion**: TTT learned good features, but gating ignores them

**Solutions:**
1. Use earlier checkpoint (before gating collapsed)
2. Add gating regularization in training
3. Remove gating entirely (use fixed alpha)
4. Investigate why gating collapsed

### Scenario C: All alphas are similarly quiet
**Conclusion**: Problem might be elsewhere (base model, input, etc.)

**Solutions:**
1. Test with baseline Moshi (no TTT at all)
2. Try different input audio
3. Check temperature settings

## Output Files

Each test creates:
- `output_gating_0.0.wav` - Pure base Moshi
- `output_gating_0.3.wav` - 30% TTT
- `output_gating_0.5.wav` - 50% TTT
- `output_gating_1.0.wav` - Pure TTT

Compare these to understand what each component contributes!
