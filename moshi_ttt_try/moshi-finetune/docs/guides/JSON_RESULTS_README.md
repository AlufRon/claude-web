# Paper Metrics JSON Results System

This system automatically saves paper metrics evaluation results to JSON files with comprehensive metadata, enabling flexible post-hoc analysis and comparison plots.

## Features

✅ **Automatic JSON Saving**: All evaluation results saved with metadata  
✅ **Run Directory Organization**: Timestamped directories for each evaluation  
✅ **Model Type Detection**: Automatically detects TTT vs Baseline models  
✅ **Comprehensive Metadata**: Full configuration and evaluation context preserved  
✅ **Standalone Plotting**: Generate plots from saved JSON without re-running evaluations  
✅ **Multi-Run Comparisons**: Compare TTT vs Baseline performance across time  

## Configuration

Add to your training config YAML:

```yaml
paper_metrics:
  paper_metrics_eval: true
  paper_metrics_freq: 1000
  
  # JSON results saving (ENABLED BY DEFAULT)
  save_results_json: true
  results_dir: "./evaluation_results"
  
  # Your existing paper metrics config...
  sblimp_max_pairs: 2000
  swuggy_max_pairs: 2000
  # etc...
```

## Directory Structure

After evaluations run, you'll get:

```
evaluation_results/
├── run_20251003_143022_ttt/
│   ├── results.json          # Full structured data
│   └── summary.txt           # Human-readable summary
├── run_20251003_150145_baseline/
│   ├── results.json
│   └── summary.txt
└── analysis/                 # Generated plots
    ├── comparison_plots.png
    └── single_run_analysis.png
```

## JSON Structure

```json
{
  "metadata": {
    "timestamp": "20251003_143022",
    "datetime": "2025-01-03T14:30:22",
    "model_type": "TTT",
    "config": {
      "sblimp_max_pairs": 2000,
      "use_silence_codes": true,
      "ttt_chunking": {...},
      "streaming": {...}
    }
  },
  "results": {
    "sblimp": {"accuracy": 0.5035, "samples": 2000, "correct": 1007},
    "swuggy": {"accuracy": 0.498, "samples": 2000, "correct": 996},
    "tstory": {"accuracy": 0.5494, "samples": 1871},
    "sstory": {"accuracy": 0.5232, "samples": 1871},
    "librilight": {"loss_8k": 2.45, "loss_16k": 2.32, "slope": -0.000012},
    "aggregate": {"paper_metrics_avg": 0.5185, "paper_metrics_f1": 0.518}
  },
  "librilight_detailed": {
    "positions": [0, 100, 200, ...],
    "losses": [2.8, 2.7, 2.6, ...],
    "slopes": {"overall": -0.000012, "early": -0.000008}
  }
}
```

## Using the Plotting Utility

### Single Run Analysis
```bash
python plot_evaluation_results.py --run evaluation_results/run_20251003_143022_ttt/results.json
```

### Compare Multiple Runs
```bash
python plot_evaluation_results.py --compare run1/results.json run2/results.json --output comparison.png
```

### Auto-Discover and Compare All Runs
```bash
python plot_evaluation_results.py --compare-all evaluation_results/ --output all_runs.png
```

### List Available Runs
```bash
python plot_evaluation_results.py --compare-all evaluation_results/ --list-runs
```

## Generated Plots

### Single Run Plot Includes:
- **Benchmark Accuracies**: Bar chart of sBLIMP, sWUGGY, tStory, sStory
- **LibriLight Performance**: Long context loss progression
- **Position Analysis**: Detailed token-by-token loss (if available)
- **Configuration Summary**: Evaluation parameters and settings
- **Performance Summary**: Key metrics and trends
- **Slope Analysis**: Learning progression indicators

### Comparison Plot Includes:
- **Side-by-side Benchmarks**: TTT vs Baseline on each task
- **LibriLight Comparison**: Context length performance curves
- **Overall Performance**: Aggregate accuracy comparison
- **Run Metadata**: Model types, timestamps, configurations

## Benefits

1. **Data Preservation**: Complete evaluation history saved permanently
2. **Flexible Analysis**: Generate any visualization from saved data
3. **Easy Comparisons**: Compare performance across model variants/training stages
4. **Debugging Support**: Full context for reproducing and understanding results
5. **Collaboration**: Share structured results with team members
6. **Minimal Overhead**: JSON saving adds <0.1s to evaluation time

## Integration

The JSON saving is **automatically enabled** when you run paper metrics evaluation. The system:

1. **Detects model type** (TTT vs Baseline) automatically
2. **Creates timestamped directories** for each evaluation run
3. **Saves comprehensive JSON** with all metrics and metadata
4. **Generates human-readable summary** for quick viewing
5. **Preserves plot data** for detailed position analysis

No code changes needed - just ensure `save_results_json: true` in your config!

## Troubleshooting

**Missing plots?** The JSON-first approach means plots are generated on-demand from saved data rather than during evaluation. Use the plotting utility to create visualizations.

**Environment issues?** Make sure to run plotting with the correct conda environment:
```bash
conda activate moshi_ttt_fixed
python plot_evaluation_results.py [options]
```

**Large files?** Position-wise data from LibriLight can be large. The JSON includes detailed position analysis for comprehensive comparisons.

## Example Workflow

1. **Run training with paper metrics enabled**
2. **Evaluation results automatically saved to JSON**
3. **Generate comparison plots**:
   ```bash
   python plot_evaluation_results.py --compare-all evaluation_results/
   ```
4. **Share results**: Send JSON files + generated plots to collaborators
5. **Deep analysis**: Use JSON data for custom analysis scripts

This system enables **reproducible, comprehensive evaluation analysis** while maintaining the flexibility to generate any visualization post-hoc!