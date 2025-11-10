# Quick Guide: Run Paper Metrics on Checkpoint

## TL;DR - Just Run This:

```bash
cd /home/alufr/ttt_tests/moshi-finetune

# Direct execution (interactive)
python run_paper_metrics_on_checkpoint.py \
    --checkpoint /sise/eliyanac-group/ron_al/dailytalk_finetune_from_librilight7/checkpoints/checkpoint_000100/consolidated \
    --max-samples 50

# OR submit to SLURM (recommended)
sbatch submit_paper_metrics.sh
```

## Output:

Results saved to: `checkpoint_dir/paper_metrics_results.json`

Example:
```json
{
  "paper_metrics_avg": 0.625,  // ← Overall score (62.5%)
  "sblimp_accuracy": 0.58,     // ← Syntax (58%)
  "swuggy_accuracy": 0.70,     // ← Phonotactics (70%)
  "tstory_accuracy": 0.57,     // ← Text stories (57%)
  "sstory_accuracy": 0.65      // ← Speech stories (65%)
}
```

## What It Does:

1. Loads your TTT checkpoint (`lora.safetensors`)
2. Runs 4 language understanding benchmarks
3. Takes ~5-10 minutes (50 samples) or ~20-30 minutes (200 samples)
4. Saves JSON with accuracies

## Key Files:

- **Script**: `run_paper_metrics_on_checkpoint.py` (minimal, clean implementation)
- **Submit**: `submit_paper_metrics.sh` (SLURM wrapper)
- **Full docs**: `PAPER_METRICS_EVALUATION_GUIDE.md`

## Compare Checkpoints:

```bash
# Test checkpoint at step 100
python run_paper_metrics_on_checkpoint.py \
    --checkpoint .../checkpoint_000100/consolidated \
    --max-samples 100 \
    --output results_step100.json

# Test checkpoint at step 500
python run_paper_metrics_on_checkpoint.py \
    --checkpoint .../checkpoint_000500/consolidated \
    --max-samples 100 \
    --output results_step500.json

# Compare
cat results_step100.json | grep paper_metrics_avg
cat results_step500.json | grep paper_metrics_avg
```

Done! 🎉
