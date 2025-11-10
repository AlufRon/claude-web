# TTT Gating Override - Testing What TTT Actually Learned

## The Problem You Discovered

You're absolutely right - I was confusing two things:

1. **Gating alpha** (how much TTT vs base Moshi is blended)
2. **What TTT learned** (what the TTT weights themselves predict)

The real question: **Did TTT weights learn to predict silence/quiet outputs?**

## Current Situation

Your checkpoint shows:
- **Initial gating alpha**: 0.3 (30% TTT, 70% base Moshi) 
- **Learned gating alpha** (after training):
  - Layer 29: ~9% TTT contribution
  - Layer 30: ~0.02% TTT contribution  
  - Layer 31: ~1% TTT contribution

The gating learned to "turn off" TTT and rely mostly on base Moshi.

## The Experiment

Now you can test what TTT actually learned by **forcing different gating alpha values**:

### Test 1: Pure TTT (1.0)
```bash
./test_ttt_gating.sh 1.0
```
- **Forces 100% TTT, 0% base Moshi**
- **Answer**: If still quiet → TTT weights learned silence
- **Answer**: If talkative → Gating was the problem

### Test 2: Pure Base Moshi (0.0)
```bash
./test_ttt_gating.sh 0.0
```
- **Forces 0% TTT, 100% base Moshi**
- Should be talkative (normal Moshi)
- **Baseline for comparison**

### Test 3: Balanced (0.5)
```bash
./test_ttt_gating.sh 0.5
```
- **50/50 blend**
- Middle ground

### Test 4: Original Target (0.3)
```bash
./test_ttt_gating.sh 0.3
```
- **What training originally intended**
- 30% TTT, 70% base

## How It Works

The new `--force-gating-alpha` parameter:
1. Loads the checkpoint normally
2. **Overwrites** the learned gating values
3. Forces all TTT layers to use the specified alpha
4. Lets you see what TTT would output if it wasn't being ignored

## What Each Result Means

### If alpha=1.0 (pure TTT) is:

**QUIET/GIBBERISH**:
- ✅ Confirms: TTT weights learned bad behavior
- Root cause: Training data (LibriLight + DailyTalk) taught it wrong patterns
- Solution: Retrain with different data or different loss weighting

**TALKATIVE/GOOD**:
- ✅ Confirms: TTT learned good features
- Problem was: Gating learned to ignore them
- Solution: Fix gating mechanism or prevent it from collapsing during training

### If alpha=0.0 (pure base Moshi) is:

**TALKATIVE**:
- ✅ Confirms: Base Moshi works fine
- This is your baseline

## Quick Start

```bash
cd /home/alufr/ttt_tests/moshi-finetune

# Test what pure TTT learned
./test_ttt_gating.sh 1.0

# Compare to pure base Moshi  
./test_ttt_gating.sh 0.0

# Try the original training target
./test_ttt_gating.sh 0.3
```

## Files Modified

1. **run_inference_with_ttt.py** - Added `--force-gating-alpha` parameter
2. **test_ttt_gating.sh** - Easy script to test different alpha values

## Command-Line Usage

You can also use it directly:
```bash
python run_inference_with_ttt.py \
    --checkpoint /path/to/checkpoint \
    --hf-repo kyutai/moshiko-pytorch-bf16 \
    --force-gating-alpha 1.0 \
    input.wav output.wav
```

## Expected Output

When you run with `--force-gating-alpha`, you'll see:
```
🔧 FORCING gating_alpha override to: 1.0
   This will replace learned gating values from checkpoint
   ✅ Set transformer.layers.29.seq_modeling_block.forward_ssm_gating = 1.0
   ✅ Set transformer.layers.29.seq_modeling_block.backward_ssm_gating = 1.0
   ...
🔧 Overrode 6 gating parameters
```

Then in the statistics:
```
📍 transformer.layers.29.seq_modeling_block.forward_ssm_gating:
   Effective TTT contribution: 100.00% (avg), 100.00% (max)
```

## Next Steps Based on Results

### If TTT alone (alpha=1.0) is quiet:
1. **Retrain with different data** - Use only LibriLight (no DailyTalk)
2. **Change loss weights** - Penalize silence more
3. **Filter training data** - Remove segments with too much silence
4. **Adjust training objective** - Add diversity penalty

### If TTT alone (alpha=1.0) is good:
1. **Fix gating mechanism** - Prevent it from collapsing to zero
2. **Add gating regularization** - Keep alpha closer to initial value
3. **Use earlier checkpoint** - Before gating learned to turn off
4. **Remove gating** - Just use fixed blend (e.g., always 30%)

This experiment will definitively tell you whether the problem is:
- **A) TTT weights learned silence** (need to retrain)
- **B) Gating learned to ignore good TTT** (fix gating mechanism)
