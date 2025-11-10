# Paper Metrics Evaluation for TTT Checkpoints

## Overview

This minimal script evaluates TTT checkpoints on paper metrics benchmarks (sBLIMP, sWUGGY, tStoryCloze, sStoryCloze).

## Files

- **`run_paper_metrics_on_checkpoint.py`**: Main evaluation script
- **`submit_paper_metrics.sh`**: SLURM submission script

## Quick Start

### Option 1: Direct Python Execution

```bash
python run_paper_metrics_on_checkpoint.py \
    --checkpoint /path/to/checkpoint_dir/consolidated \
    --max-samples 50 \
    --device cuda
```

### Option 2: SLURM Submission

```bash
# Use default checkpoint (librilight7/checkpoint_000100)
sbatch submit_paper_metrics.sh

# Or specify custom checkpoint and sample count
sbatch submit_paper_metrics.sh /path/to/checkpoint 100
```

## Arguments

### `run_paper_metrics_on_checkpoint.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | **Required** | Path to checkpoint directory (must contain `training_config.json` and `lora.safetensors`) |
| `--hf-repo` | `kyutai/moshiko-pytorch-bf16` | HuggingFace repo for base Moshi model |
| `--max-samples` | `50` | Max samples per task (50 = ~5-10 min, 200 = ~20-30 min) |
| `--device` | `cuda` | Device to use (`cuda` or `cpu`) |
| `--output` | `checkpoint_dir/paper_metrics_results.json` | Output JSON file path |

## What It Does

1. **Loads checkpoint**: Reads `training_config.json` and `lora.safetensors`
2. **Loads base Moshi**: Downloads/loads base model from HuggingFace
3. **Applies TTT integration**: Adds TTT layers to model
4. **Loads TTT weights**: Loads trained TTT parameters
5. **Runs evaluations**:
   - **sBLIMP**: Syntactic minimal pairs (grammatical vs ungrammatical)
   - **sWUGGY**: Phonotactic minimal pairs (word-like vs non-word-like)
   - **tStoryCloze**: Textual story completion (2 choices)
   - **sStoryCloze**: Spoken story completion (2 choices)
6. **Saves results**: JSON file with accuracies and sample counts

## Output

### Console Output

```
================================================================================
PAPER METRICS EVALUATION
================================================================================
Checkpoint:   /path/to/checkpoint_000100/consolidated
Max samples:  50 per task
Device:       cuda
================================================================================

🏗️  Loading model...
✅ TTT Config:
   Layers: 29,30,31
   Base LR: 0.0001
   Gating alpha: 0.001

✅ Loaded base Moshi model
✅ TTT integration applied
✅ TTT checkpoint loaded successfully

================================================================================
RUNNING PAPER METRICS EVALUATION
================================================================================
This will evaluate on:
  • sBLIMP (syntactic minimal pairs)
  • sWUGGY (phonotactic minimal pairs)
  • tStoryCloze (textual story completion)
  • sStoryCloze (spoken story completion)

⏱️  Expected time: ~5-10 minutes for 50 samples per task

================================================================================
✅ EVALUATION COMPLETE
================================================================================

📊 Results:
   paper_metrics_avg             :  62.50%
   sblimp_accuracy               :  58.00%
   sblimp_samples                :      50
   sstory_accuracy               :  65.00%
   sstory_samples                :      50
   swuggy_accuracy               :  70.00%
   swuggy_samples                :      50
   tstory_accuracy               :  57.00%
   tstory_samples                :      50

💾 Results saved to: /path/to/checkpoint_000100/consolidated/paper_metrics_results.json
```

### JSON Output

File: `checkpoint_dir/paper_metrics_results.json`

```json
{
  "sblimp_accuracy": 0.58,
  "sblimp_samples": 50,
  "swuggy_accuracy": 0.70,
  "swuggy_samples": 50,
  "tstory_accuracy": 0.57,
  "tstory_samples": 50,
  "sstory_accuracy": 0.65,
  "sstory_samples": 50,
  "paper_metrics_avg": 0.625
}
```

## Implementation Details

### Checkpoint Loading

The script follows the same pattern as `run_inference_with_ttt.py`:

1. Load `training_config.json` to get TTT configuration
2. Create `TTTArgs` from config
3. Load base Moshi model from HuggingFace
4. Apply TTT integration (add TTT layers)
5. Load TTT weights from `lora.safetensors`
6. Keep TTT weights in float32 for precision

### Evaluation Method

Uses `PaperMetricsEvaluator.evaluate_all()` which:
- Loads audio files from benchmark datasets
- Encodes to Moshi tokens via MIMI
- Computes log-likelihoods for each choice
- Selects higher likelihood choice
- Computes accuracy per task

### Performance

- **50 samples/task**: ~5-10 minutes total
- **100 samples/task**: ~10-20 minutes total
- **200 samples/task**: ~20-30 minutes total

Actual time depends on:
- Audio file lengths
- GPU speed
- Number of TTT layers

## Examples

### Quick Test (50 samples)

```bash
python run_paper_metrics_on_checkpoint.py \
    --checkpoint /sise/eliyanac-group/ron_al/dailytalk_finetune_from_librilight7/checkpoints/checkpoint_000100/consolidated \
    --max-samples 50
```

### Full Evaluation (200 samples)

```bash
python run_paper_metrics_on_checkpoint.py \
    --checkpoint /sise/eliyanac-group/ron_al/dailytalk_finetune_from_librilight7/checkpoints/checkpoint_000100/consolidated \
    --max-samples 200 \
    --output ./evaluation_results/checkpoint_100_full_metrics.json
```

### Compare Multiple Checkpoints

```bash
for ckpt in checkpoint_000100 checkpoint_000200 checkpoint_000300; do
    python run_paper_metrics_on_checkpoint.py \
        --checkpoint /path/to/$ckpt/consolidated \
        --max-samples 100 \
        --output ./results/${ckpt}_metrics.json
done
```

## Troubleshooting

### Error: "training_config.json not found"

Make sure you're pointing to the **consolidated** directory:
```bash
# ✅ Correct
--checkpoint /path/to/checkpoint_000100/consolidated

# ❌ Wrong
--checkpoint /path/to/checkpoint_000100
```

### Error: "lora.safetensors not found"

The checkpoint must contain trained TTT weights. Check:
```bash
ls /path/to/checkpoint_000100/consolidated/
# Should see: training_config.json, lora.safetensors, config.json
```

### Error: "TTT is not enabled in this checkpoint"

The checkpoint was trained without TTT. Check `training_config.json`:
```bash
cat /path/to/checkpoint_000100/consolidated/training_config.json | grep -A5 '"ttt"'
```

Should show: `"enable": true`

### Low Accuracy (<50%)

This could indicate:
- Model hasn't trained enough
- TTT not learning properly
- Data/benchmark mismatch

Try checking:
1. Training loss curves
2. Inner loop diagnostics (Figure 5)
3. Gating alpha values

## Dependencies

Required packages (already in `moshi_ttt_fixed` environment):
- `torch`
- `safetensors`
- `librosa`
- `numpy`
- Moshi package (in `moshi-finetune/moshi`)

## Notes

- Script is **read-only** - doesn't modify checkpoint
- Can be run multiple times safely
- Results are deterministic (same checkpoint → same results)
- Supports both TTT and LoRA checkpoints (as long as they have correct structure)
