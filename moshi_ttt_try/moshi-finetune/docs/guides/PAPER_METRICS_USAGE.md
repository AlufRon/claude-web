# Paper Metrics Evaluation - Usage Guide

## Overview

The paper metrics evaluation system now supports both **TTT checkpoints** and **baseline Moshi** using the exact same evaluation code. This ensures fair comparison between models.

## Fixed Issues

### Critical Bug Fix (October 16, 2025)

**Problem**: The `_compute_likelihood` function had a dimension mismatch error when computing loss:
```
The shape of the mask [240] at index 0 does not match the shape of the indexed tensor [210, 2048]
```

**Root Cause**: The function was not properly using the model's output mask that accounts for delay patterns in the Moshi architecture.

**Solution**: Modified `finetune/paper_metrics.py` to:
1. Use `output.mask` from the model's `LMOutput` object
2. Properly handle the 4D logits tensor `[B, K, T, vocab_size]`
3. Align masks with delay-adjusted sequences

## Usage

### 1. Evaluate TTT Checkpoint

```bash
# Interactive mode
./submit_paper_metrics.sh /path/to/checkpoint_dir 50

# SLURM batch mode
sbatch submit_paper_metrics.sh /path/to/checkpoint_dir 50
```

**Example**:
```bash
./submit_paper_metrics.sh \
  /sise/eliyanac-group/ron_al/dailytalk_finetune_from_librilight6/checkpoints/checkpoint_003000/consolidated/ \
  50
```

### 2. Evaluate Baseline Moshi (NEW!)

```bash
# Interactive mode
./submit_paper_metrics.sh --baseline 50

# SLURM batch mode
sbatch submit_paper_metrics.sh --baseline 50
```

This loads the baseline Moshi model from HuggingFace (`kyutai/moshiko-pytorch-bf16`) without any TTT modifications.

### 3. Python Direct Usage

You can also use the Python script directly:

```bash
# TTT checkpoint
python run_paper_metrics_on_checkpoint.py \
    --checkpoint /path/to/checkpoint_dir \
    --max-samples 50 \
    --config example/moshi_7B_multilayer_with_ttt.yaml \
    --device cuda

# Baseline Moshi
python run_paper_metrics_on_checkpoint.py \
    --baseline \
    --max-samples 50 \
    --config example/moshi_7B_multilayer_with_ttt.yaml \
    --device cuda
```

## Evaluation Metrics

The system evaluates on 4 tasks:

1. **sBLIMP**: Syntactic minimal pairs (grammatical vs ungrammatical)
2. **sWUGGY**: Phonotactic minimal pairs (real words vs non-words)
3. **tStoryCloze**: Textual story completion
4. **sStoryCloze**: Spoken story completion

**Paper Metrics Average**: Mean of all available task accuracies

## Results Location

- **TTT checkpoint**: `{checkpoint_dir}/paper_metrics_results.json`
- **Baseline**: `./baseline_paper_metrics_results.json`

You can also specify custom output path:
```bash
python run_paper_metrics_on_checkpoint.py \
    --baseline \
    --output /path/to/custom_results.json \
    --max-samples 50
```

## Example Results (Oct 16, 2025)

### TTT Checkpoint 003000
```
paper_metrics_avg  :  47.00%
sblimp_accuracy    :  44.00% (50 samples)
swuggy_accuracy    :  50.00% (50 samples)
tstory_accuracy    :   0.00% (0 samples - missing pairs)
sstory_accuracy    :   0.00% (0 samples - missing pairs)
```

### Baseline Moshi
*Run `./submit_paper_metrics.sh --baseline 50` to get baseline results*

## Configuration

The evaluation uses paths from `example/moshi_7B_multilayer_with_ttt.yaml`:

```yaml
paper_metrics:
  sblimp_audio_dir: /sise/eliyanac-group/ron_al/sBLIMP_audio/sBLIMP_audio
  sblimp_gold_csv: /sise/eliyanac-group/ron_al/sBLIMP_audio/gold.csv
  swuggy_audio_dir: /sise/eliyanac-group/ron_al/swuggy_audio/swuggy_audio
  swuggy_gold_csv: /sise/eliyanac-group/ron_al/swuggy_audio/gold.csv
  tstory_audio_dir: /sise/eliyanac-group/ron_al/tSC/tSC/
  sstory_audio_dir: /sise/eliyanac-group/ron_al/sSC/sSC/
```

## Performance

- **Time**: ~30-40 seconds for 50 samples (10 per task)
- **Memory**: ~10GB GPU memory
- **Recommended**: 50-100 samples for quick evaluation, 500+ for paper-ready results

## Technical Details

### Model Forward Pass
The evaluation uses the model's standard forward pass which returns:
```python
LMOutput(
    logits=[B, K, T, vocab],    # Audio predictions
    mask=[B, K, T],              # Valid positions (accounts for delays)
    text_logits=[B, 1, T, text_vocab],
    text_mask=[B, 1, T]
)
```

### Likelihood Computation
```python
# Use model's mask (handles delay patterns correctly)
valid_mask = output.mask & (target != zero_token_id)

# Compute cross-entropy only on valid positions
loss = F.cross_entropy(logits[valid_mask], target[valid_mask])
likelihood = -loss
```

### Key Difference: TTT vs Baseline
- **TTT**: Loads checkpoint → applies TTT integration → loads trained weights
- **Baseline**: Loads pre-trained Moshi from HuggingFace → no modifications
- **Evaluation**: Both use identical `_compute_likelihood` function

## Troubleshooting

### Missing Story Cloze Pairs
If you see warnings like:
```
WARNING - No structured story cloze pairs found
```

This means the audio files don't follow expected naming patterns. The evaluator looks for:
- `*_correct.wav` / `*_incorrect.wav`
- `*_ending1.wav` / `*_ending2.wav`

Check your data directory structure.

### Dimension Mismatch Errors
If you encounter shape errors, ensure you're using the latest version of `finetune/paper_metrics.py` with the October 16, 2025 fix.

## Next Steps

1. **Run baseline evaluation**: `./submit_paper_metrics.sh --baseline 50`
2. **Compare results**: Check how TTT improves (or doesn't) over baseline
3. **Scale up**: Run with more samples for paper-ready results
4. **Analyze**: Look at per-task improvements (syntax vs phonotactics)
