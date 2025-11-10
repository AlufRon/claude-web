# TTT Model "Not Talkative" Issue - Analysis & Solutions

## Problem Summary
Your TTT-finetuned Moshi model is less talkative than baseline Moshi, generating minimal output like "Well, So" instead of longer responses.

## Root Causes

### 1. **Training Data Distribution Mismatch**
- **Pre-training**: LibriLight (audiobook narration - continuous monologue)
- **Fine-tuning**: DailyTalk (two-speaker conversations with turn-taking)
- **Issue**: Model learned conversation patterns where there are natural pauses and turn-taking between speakers

### 2. **TTT Contribution is Very Low**
From your inference logs:
```
Layer 29: 9.08% TTT contribution (forward), 29.30% (backward)
Layer 30: 0.03% TTT contribution (forward), 29.30% (backward)  
Layer 31: 0.99% TTT contribution (forward), 29.30% (backward)
```

The forward gating alphas are very low, meaning TTT is barely being used during generation. The model is mostly relying on the frozen base model.

### 3. **Limited Fine-tuning**
- Only 500 steps completed (out of 10,000 planned)
- 60-second audio segments
- Model may not have fully adapted to generating conversational responses

### 4. **Conservative Behavior from Conversation Data**
DailyTalk has TWO speakers. During training:
- Model sees many segments where one speaker is quiet while the other talks
- May have learned to be more conservative about when to speak
- Learned "listening" behavior as well as "speaking" behavior

## Solutions

### ✅ **Solution 1: Increase Temperature (QUICK FIX)**

I've added temperature controls to the inference script and created new submission scripts.

**Usage:**
```bash
cd /home/alufr/ttt_tests/moshi-finetune

# Try with increased temperatures (1.2 for audio, 1.1 for text)
./submit_inference_talkative.sh

# Or customize:
./submit_inference_talkative.sh \
    /path/to/checkpoint \
    input.wav \
    output.wav \
    kyutai/moshiko-pytorch-bf16 \
    1.5 \  # audio temp (default 0.8, trying 1.5)
    1.3    # text temp (default 0.9, trying 1.3)
```

**What it does:**
- Higher temperature = more randomness/creativity in generation
- Audio temp 1.2 (vs default 0.8) = more varied audio tokens
- Text temp 1.1 (vs default 0.9) = more varied text
- Should make the model more willing to generate longer responses

### **Solution 2: Continue Training**

The model stopped at step 500. You could:

```bash
# Resume training for more steps
./submit_job.sh example/dailytalk_finetune_from_librilight.yaml
```

Edit the config to:
- Increase `max_steps` from 10000 to 20000+
- Monitor if gating alpha increases (showing more TTT usage)
- Check if loss continues to decrease

### **Solution 3: Try Different Checkpoint**

If you have other checkpoints (step 100, 200, 300, etc.):

```bash
./submit_inference_talkative.sh \
    /sise/eliyanac-group/ron_al/dailytalk_finetune_from_librilight5/checkpoints/checkpoint_000300/consolidated \
    input.wav \
    output.wav
```

Earlier checkpoints might be less "conservative" if the model was learning to be quieter over time.

### **Solution 4: Adjust TTT Gating Alpha**

The low gating alpha (0.09-0.01) suggests TTT isn't contributing much. You could:

1. **During training**: Increase `initial_gating_alpha` from 0.3 to 0.5 or 0.7
2. **During inference**: Manually boost TTT contribution (would require code changes)

### **Solution 5: Different Training Strategy**

For future training:

1. **Use LibriLight only** (audiobooks are monologue, naturally more "talkative")
2. **Filter DailyTalk** to only use segments where one speaker dominates
3. **Augment data** with more "verbose" examples
4. **Adjust loss weighting** to encourage longer generations

## Testing Recommendations

### Test 1: Temperature Sweep
```bash
# Low temp (conservative)
./submit_inference.sh checkpoint input.wav output_temp0.8.wav

# Medium temp (balanced)  
# (use submit_inference_talkative.sh with default temps)

# High temp (very talkative)
./submit_inference_talkative.sh checkpoint input.wav output_temp1.5.wav "" 1.5 1.3
```

### Test 2: Compare with Baseline
Run the same audio through baseline Moshi (without TTT) to see the difference:
```bash
# Run without TTT for comparison
python moshi/run_inference.py input.wav output_baseline.wav
```

### Test 3: Different Input Audio
Try inputs with different characteristics:
- Quiet/ambient audio (does it fill the silence?)
- Question-like intonation (does it respond?)  
- Continuous speech (does it interrupt or wait?)

## Quick Start: Try Higher Temperature Now

```bash
cd /home/alufr/ttt_tests/moshi-finetune
./submit_inference_talkative.sh
```

This will run inference with:
- Audio temp: 1.2 (50% higher than default 0.8)
- Text temp: 1.1 (22% higher than default 0.9)
- Same checkpoint, same input
- Output saved to: `output_talkative.wav`

## Expected Results

With higher temperature:
- ✅ Longer generated responses
- ✅ More varied word choices
- ✅ Less conservative/quiet behavior
- ⚠️  Possible side effect: Slightly less coherent (trade-off for talkativeness)

## If Still Not Talkative Enough

1. **Push temperature even higher**: Try 1.8, 2.0, 2.5
2. **Check gating alpha during training**: If it's decreasing, TTT is being "turned off"
3. **Train for longer**: Current model only saw 500 steps
4. **Consider architectural changes**: Maybe gating should be additive vs. multiplicative

## Files Modified/Created

1. `run_inference_with_ttt.py` - Added temperature controls (--temp, --temp-text)
2. `submit_inference_talkative.sh` - New submission script with higher temps
3. `run_inference_ttt_talkative.slurm` - SLURM script for talkative inference

## Monitoring

Watch the logs for:
```
sampling: True, audio temp: 1.2, text temp: 1.1
```

And compare the text output to see if it generates more tokens.
