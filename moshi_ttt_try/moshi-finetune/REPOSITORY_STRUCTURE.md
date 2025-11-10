# Repository Structure

This document describes the organization of the moshi-finetune-ttt repository.

## Directory Structure

```
moshi-finetune-ttt/
├── finetune/              # Core fine-tuning package
├── moshi_ttt/             # TTT (Test-Time Training) implementation
├── tests/                 # Unit tests
├── example/               # Example configuration files
├── configs/               # Production configuration files
│
├── docs/                  # All documentation
│   ├── guides/           # User guides and quick starts
│   ├── analysis/         # Analysis and investigation reports
│   ├── fixes/            # Bug fixes and solutions
│   └── implementation/   # Implementation details and status
│
├── analysis/              # Analysis scripts
│   ├── librilight/       # LibriLight dataset analysis
│   ├── memory/           # Memory usage analysis
│   ├── training/         # Training metrics analysis
│   └── evaluation/       # Evaluation results analysis
│
├── debug/                 # Debugging scripts
│   ├── ttt/              # TTT-specific debugging
│   ├── checkpoint/       # Checkpoint-related debugging
│   ├── librilight/       # LibriLight debugging
│   └── general/          # General debugging utilities
│
├── evaluation/            # Evaluation tools
│   ├── scripts/          # Evaluation scripts
│   └── figure5/          # Figure 5 reproduction tools
│
├── slurm/                 # SLURM job scripts
│   ├── training/         # Training job scripts
│   ├── inference/        # Inference job scripts
│   └── evaluation/       # Evaluation job scripts
│
├── tools/                 # Utility tools
│   ├── plotting/         # Plotting and visualization
│   ├── data_prep/        # Data preparation utilities
│   └── monitoring/       # Training monitoring tools
│
├── test_scripts/          # Test scripts
│   ├── ttt/              # TTT functionality tests
│   ├── librilight/       # LibriLight integration tests
│   ├── checkpoint/       # Checkpoint system tests
│   └── general/          # General functionality tests
│
├── investigation/         # Research and investigation scripts
├── training/              # Training entry points
├── inference/             # Inference entry points
├── utils/                 # General utilities
├── data/                  # Data files and baselines
├── dashboard/             # Web dashboard for results
└── dataset_prep/          # Dataset preparation scripts
```

## Core Files

- `train.py` - Main training entry point
- `README.md` - Project overview and setup
- `CONTRIBUTING.md` - Contribution guidelines
- `LICENSE` - License information
- `pyproject.toml` - Python package configuration
- `setup.cfg` - Setup configuration
- `uv.lock` - UV dependency lock file

## Key Directories

### `finetune/`
Core fine-tuning implementation including:
- Data loading and processing
- Model wrapping and checkpointing
- Training loop implementation
- TTT integration
- Monitoring and metrics

### `moshi_ttt/`
Test-Time Training (TTT) implementation:
- TTT layer implementations
- Hybrid layer combining Moshi and TTT
- SSM (State Space Model) operations
- Configuration and utilities

### `docs/`
Comprehensive documentation:
- **guides/** - Step-by-step guides for users
- **analysis/** - Deep-dive analysis reports
- **fixes/** - Documentation of bug fixes
- **implementation/** - Implementation details and status updates

### `analysis/`
Scripts for analyzing results:
- Training metrics and convergence
- LibriLight dataset characteristics
- Memory usage patterns
- Evaluation results

### `evaluation/`
Evaluation tools and scripts:
- Figure 5 reproduction from paper
- Checkpoint evaluation
- Inner loop evaluation
- Paper metrics computation

### `slurm/`
SLURM job submission scripts for HPC:
- Training jobs with various configurations
- Inference jobs (baseline and TTT)
- Evaluation and metrics computation

### `tools/`
Utility tools:
- Plotting and visualization scripts
- Data preparation utilities
- Training monitoring tools

## Common Workflows

### Training
```bash
# Basic training
python train.py --config example/moshi_7B.yaml

# TTT training
cd training/
python train_ttt_production.py --config ../configs/production_ttt_dailytalk.yaml
```

### Inference
```bash
# TTT inference
cd inference/
python run_inference_with_ttt.py --checkpoint path/to/checkpoint

# Figure 5 evaluation
python run_inference_with_figure5.py --config figure5_quick.yaml
```

### Evaluation
```bash
# Paper metrics
cd evaluation/scripts/
python run_paper_metrics_on_checkpoint.py --checkpoint path/to/checkpoint

# Figure 5 analysis
cd evaluation/figure5/
python evaluate_checkpoint_figure5.py
```

### Analysis
```bash
# Analyze training results
cd analysis/training/
python analyze_results.py

# Plot evaluation results
cd tools/plotting/
python plot_evaluation_results.py
```

## Documentation Navigation

- **Getting Started**: See `docs/guides/QUICK_START.md`
- **TTT Implementation**: See `docs/implementation/MOSHI_TTT_COMPLETE_IMPLEMENTATION_REPORT.md`
- **LibriLight Setup**: See `docs/implementation/LIBRILIGHT_IMPLEMENTATION_COMPLETE.md`
- **Troubleshooting**: Check `docs/fixes/` directory
- **Paper Metrics**: See `docs/guides/PAPER_METRICS_EVALUATION_GUIDE.md`

## Generated Files (Not in Git)

The following directories contain generated files (excluded by `.gitignore`):
- `evaluation_plots/` - Generated evaluation plots
- `evaluation_results/` - Evaluation result outputs
- `runs/` - Training run outputs
- `*.log`, `*.err` files - Log files
- `*.pkl` files - Cache files
- `*.png` files - Generated plots
- `*.wav` files - Generated audio

These files are regenerated during training/evaluation and are not tracked in version control.
