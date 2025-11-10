#!/usr/bin/env python3
"""
Paper Metrics Dashboard - Data Aggregation Script

Scans checkpoint directories for paper metrics results and training configs,
aggregates them into a single JSON file for the dashboard.

Usage:
    python aggregate_paper_metrics.py --output dashboard_data.json
    python aggregate_paper_metrics.py --checkpoint-dir /path/to/specific/dir
    python aggregate_paper_metrics.py --incremental  # Add new runs only
"""

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class MetricsData:
    """Container for paper metrics results"""
    sblimp_accuracy: float = 0.0
    sblimp_samples: int = 0
    swuggy_accuracy: float = 0.0
    swuggy_samples: int = 0
    tstory_accuracy: float = 0.0
    tstory_samples: int = 0
    sstory_accuracy: float = 0.0
    sstory_samples: int = 0
    librilight_perplexity_8k: float = 0.0
    librilight_perplexity_16k: float = 0.0
    librilight_perplexity_24k: float = 0.0
    librilight_slope: float = 0.0
    librilight_samples: int = 0
    paper_metrics_avg: float = 0.0


@dataclass
class LibriLightProgression:
    """Single position-wise perplexity data point"""
    position: int
    loss: float
    perplexity: float


@dataclass
class CheckpointRun:
    """Complete data for a single checkpoint run"""
    id: str
    name: str
    checkpoint_path: str
    checkpoint_step: int
    evaluation_timestamp: str
    training_config: Dict[str, Any]
    metrics: Dict[str, Any]
    librilight_progression: List[Dict[str, float]]  # Raw data (all samples)
    librilight_progression_mean: List[Dict[str, float]]  # Averaged by position
    librilight_progression_median: List[Dict[str, float]]  # Median by position
    librilight_progression_individual: List[List[Dict[str, float]]]  # Separated by file
    tags: List[str]
    notes: str


class PaperMetricsAggregator:
    """Aggregates paper metrics from multiple checkpoints"""

    def __init__(self, checkpoint_dirs: List[Path], log_dir: Optional[Path] = None):
        """
        Args:
            checkpoint_dirs: List of base directories to search for checkpoints
            log_dir: Directory containing paper_metrics_*.log files (optional)
        """
        self.checkpoint_dirs = [Path(d) for d in checkpoint_dirs]
        self.log_dir = Path(log_dir) if log_dir else None
        self.runs: List[CheckpointRun] = []

    def find_checkpoints_from_registry(self) -> List[Path]:
        """
        Find checkpoints from the runs registry (FAST - no filesystem scanning).

        Returns:
            List of paths to consolidated checkpoint directories
        """
        registry_path = Path(__file__).parent / "runs_registry.json"

        if not registry_path.exists():
            logger.warning("⚠️  No runs registry found, falling back to full directory scan")
            logger.warning("   This will be slow. Future runs will use the registry.")
            return self.find_all_checkpoints()

        try:
            with open(registry_path, 'r') as f:
                registry_data = json.load(f)

            checkpoint_paths = []
            self.registry_runs = {}  # Store mapping of checkpoint_dir -> results_file

            for run in registry_data.get("runs", []):
                checkpoint_path_str = run["checkpoint_path"]
                results_file_str = run["results_file"]

                # Handle baseline runs (checkpoint_path may be "None" or empty)
                if checkpoint_path_str in ["None", "none", "", None]:
                    # For baseline runs, use the directory containing the results file
                    results_file = Path(results_file_str)
                    if not results_file.is_absolute():
                        # Relative path - assume it's relative to project root
                        results_file = Path(__file__).parent.parent / results_file

                    # Only add if the results file actually exists
                    if results_file.exists():
                        checkpoint_dir = results_file.parent
                        checkpoint_paths.append(checkpoint_dir)
                        # Store the results filename for later use
                        self.registry_runs[str(checkpoint_dir)] = results_file.name
                    else:
                        logger.warning(f"⚠️  Baseline results not found (skipping): {results_file}")
                else:
                    checkpoint_path = Path(checkpoint_path_str)
                    if checkpoint_path.exists():
                        checkpoint_paths.append(checkpoint_path)
                        # Standard runs use paper_metrics_results.json
                        self.registry_runs[str(checkpoint_path)] = "paper_metrics_results.json"
                    else:
                        logger.warning(f"⚠️  Checkpoint not found (skipping): {checkpoint_path}")

            logger.info(f"✅ Found {len(checkpoint_paths)} checkpoints from registry")
            return checkpoint_paths

        except Exception as e:
            logger.error(f"❌ Error reading registry: {e}")
            logger.warning("   Falling back to full directory scan")
            return self.find_all_checkpoints()

    def find_all_checkpoints(self) -> List[Path]:
        """
        Find all checkpoint directories containing paper_metrics_results.json
        (SLOW - scans entire directory tree)

        Returns:
            List of paths to consolidated checkpoint directories
        """
        logger.info("🔍 Searching for checkpoints with paper metrics (full scan)...")

        checkpoint_paths = []

        for base_dir in self.checkpoint_dirs:
            if not base_dir.exists():
                logger.warning(f"⚠️  Directory not found: {base_dir}")
                continue

            # Search for paper_metrics_results.json files
            # Typically located at: base_dir/checkpoints/checkpoint_NNNN/consolidated/paper_metrics_results.json
            pattern = "**/consolidated/paper_metrics_results.json"

            for metrics_file in base_dir.glob(pattern):
                checkpoint_dir = metrics_file.parent
                checkpoint_paths.append(checkpoint_dir)
                logger.info(f"   Found: {checkpoint_dir}")

        logger.info(f"✅ Found {len(checkpoint_paths)} checkpoints with paper metrics")
        return checkpoint_paths

    def load_training_config(self, checkpoint_dir: Path) -> Optional[Dict[str, Any]]:
        """Load training_config.json from checkpoint directory"""
        config_path = checkpoint_dir / "training_config.json"

        if not config_path.exists():
            logger.warning(f"⚠️  No training_config.json in {checkpoint_dir}")
            return None

        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config
        except Exception as e:
            logger.error(f"❌ Error loading config from {config_path}: {e}")
            return None

    def load_paper_metrics(self, checkpoint_dir: Path, metrics_filename: str = "paper_metrics_results.json") -> Optional[Dict[str, Any]]:
        """Load paper metrics from checkpoint directory

        Args:
            checkpoint_dir: Directory containing the metrics file
            metrics_filename: Name of the metrics file (default: paper_metrics_results.json)
        """
        metrics_path = checkpoint_dir / metrics_filename

        if not metrics_path.exists():
            logger.warning(f"⚠️  No {metrics_filename} in {checkpoint_dir}")
            return None

        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            return metrics
        except Exception as e:
            logger.error(f"❌ Error loading metrics from {metrics_path}: {e}")
            return None

    def extract_checkpoint_step(self, checkpoint_path: Path) -> int:
        """Extract checkpoint step number from path"""
        # Pattern: checkpoint_002000 -> 2000
        match = re.search(r'checkpoint_(\d+)', str(checkpoint_path))
        if match:
            return int(match.group(1))
        return 0

    def extract_run_name(self, checkpoint_path: Path, config: Dict[str, Any], is_baseline: bool = False) -> str:
        """Generate human-readable name for the run"""
        if is_baseline:
            model = config.get('model', 'unknown')
            return f"Baseline - {model}"

        step = self.extract_checkpoint_step(checkpoint_path)

        # Try to get descriptive info from wandb config
        wandb_name = config.get('wandb', {}).get('run_name', '')
        if wandb_name:
            return f"CP{step:04d} - {wandb_name}"

        # Fallback: use parent directory name
        parent_name = checkpoint_path.parent.parent.parent.name
        return f"CP{step:04d} - {parent_name}"

    def generate_run_id(self, checkpoint_path: Path, is_baseline: bool = False) -> str:
        """Generate unique ID for the run"""
        if is_baseline:
            return "baseline_pretrained"
        step = self.extract_checkpoint_step(checkpoint_path)
        parent_name = checkpoint_path.parent.parent.parent.name
        return f"{parent_name}_cp{step:06d}"

    def classify_training_type(self, config: Dict[str, Any]) -> str:
        """Classify training type: ttt, lora, full, or baseline"""
        if config.get('ttt', {}).get('enable', False):
            return 'ttt'
        elif config.get('lora', {}).get('enable', False):
            return 'lora'
        elif config.get('full_finetuning', False):
            return 'full'
        else:
            return 'baseline'

    def extract_tags(self, config: Dict[str, Any], metrics: Dict[str, Any]) -> List[str]:
        """Extract descriptive tags for filtering"""
        tags = []

        # Training type
        training_type = self.classify_training_type(config)
        tags.append(training_type)

        # TTT-specific tags
        if config.get('ttt', {}).get('enable', False):
            ttt = config['ttt']
            layers = ttt.get('layers', 'none')
            tags.append(f"ttt-layers-{layers.replace(',', '-')}")

            base_lr = ttt.get('base_lr', 0)
            if base_lr >= 0.01:
                tags.append('high-lr')
            elif base_lr >= 0.001:
                tags.append('medium-lr')
            else:
                tags.append('low-lr')

        # Data source
        train_data = config.get('data', {}).get('train_data', '')
        if 'daily' in train_data.lower():
            tags.append('daily-talk')
        elif 'librilight' in train_data.lower():
            tags.append('librilight')
        elif 'librispeech' in train_data.lower():
            tags.append('librispeech')

        # Training duration
        max_steps = config.get('max_steps', 0)
        if max_steps >= 2000:
            tags.append('long-training')
        elif max_steps >= 1000:
            tags.append('medium-training')
        else:
            tags.append('short-training')

        return tags

    def parse_librilight_logs(self, checkpoint_path: Path) -> List[Dict[str, float]]:
        """
        Parse LibriLight position-wise perplexity from log files

        Returns:
            List of {position, loss, perplexity} dicts
        """
        # This is optional - if we can't find logs, return empty list
        if not self.log_dir:
            return []

        # Try to find matching log file
        checkpoint_step = self.extract_checkpoint_step(checkpoint_path)
        parent_name = checkpoint_path.parent.parent.parent.name

        # Normalize checkpoint path for comparison
        checkpoint_path_str = str(checkpoint_path).replace('/consolidated', '')

        # Pattern: paper_metrics_cp002000_*.log or paper_metrics_*.err
        # Try multiple patterns to find position data
        patterns_to_try = [
            f"paper_metrics_cp{checkpoint_step:06d}_*.log",
            f"paper_metrics_cp{checkpoint_step:06d}_*.err",
            f"paper_metrics_*{checkpoint_step}*.log",
            f"paper_metrics_*{checkpoint_step}*.err",
            "paper_metrics_*.log",
            "paper_metrics_*.err"
        ]

        progression = []
        matched_log_file = None

        for log_pattern in patterns_to_try:
            for log_file in self.log_dir.glob(log_pattern):
                try:
                    with open(log_file, 'r') as f:
                        content = f.read()

                    # First, verify this log is for the correct checkpoint
                    # Look for "Checkpoint: /path/to/checkpoint" in the log
                    checkpoint_match = re.search(r'Checkpoint:\s+(.+?)(?:\n|$)', content)
                    if checkpoint_match:
                        log_checkpoint_path = checkpoint_match.group(1).strip().replace('/consolidated', '')
                        # Check if this log is for our checkpoint
                        if checkpoint_path_str not in log_checkpoint_path:
                            # This log is for a different checkpoint, skip it
                            continue
                    elif log_pattern in ["paper_metrics_*.log", "paper_metrics_*.err"]:
                        # Fallback wildcard pattern - verify checkpoint matches
                        # Skip if we can't verify this is the right checkpoint
                        continue

                    # Parse position-wise output - try multiple formats
                    # Format 1: Position XXXXX: Loss X.XXXX | PPL XXXX.XX
                    pattern1 = r'Position\s+(\d+):\s+Loss\s+([\d.]+)\s+\|\s+PPL\s+([\d.]+)'
                    # Format 2: Position XXXXX/XXXXX: loss=X.XXXX, perplexity=XXX.XX
                    pattern2 = r'Position\s+(\d+)/\d+:\s+loss=([\d.]+),\s+perplexity=([\d.]+)'

                    matches = re.findall(pattern1, content)
                    if not matches:
                        matches = re.findall(pattern2, content)

                    if matches:
                        for position, loss, ppl in matches:
                            progression.append({
                                'position': int(position),
                                'loss': float(loss),
                                'perplexity': float(ppl)
                            })

                        matched_log_file = log_file.name
                        logger.info(f"✅ Found {len(matches)} position data points in {log_file.name} for checkpoint {checkpoint_path.name}")
                        break  # Found matching log with data

                except Exception as e:
                    logger.warning(f"⚠️  Error parsing log {log_file}: {e}")

            if progression:
                break  # Found data, stop searching

        if not progression and matched_log_file is None:
            logger.debug(f"ℹ️  No position data found for {checkpoint_path.name}")

        # Sort by position
        progression.sort(key=lambda x: x['position'])

        return progression

    def aggregate_progression_data(self, raw_progression: List[Dict[str, float]]) -> Dict[str, Any]:
        """
        Aggregate progression data by position.

        Returns dict with:
            - raw: original data
            - mean: averaged by position
            - median: median by position
            - individual: separated by file (groups of points with same positions)
        """
        from collections import defaultdict
        import statistics

        if not raw_progression:
            return {
                'raw': [],
                'mean': [],
                'median': [],
                'individual': []
            }

        # Group by position
        by_position = defaultdict(lambda: {'loss': [], 'perplexity': []})
        for point in raw_progression:
            pos = point['position']
            by_position[pos]['loss'].append(point['loss'])
            by_position[pos]['perplexity'].append(point['perplexity'])

        # Calculate mean
        mean_progression = []
        for pos in sorted(by_position.keys()):
            losses = by_position[pos]['loss']
            ppls = by_position[pos]['perplexity']
            mean_progression.append({
                'position': pos,
                'loss': statistics.mean(losses),
                'perplexity': statistics.mean(ppls),
                'loss_std': statistics.stdev(losses) if len(losses) > 1 else 0,
                'perplexity_std': statistics.stdev(ppls) if len(ppls) > 1 else 0,
                'n_samples': len(losses)
            })

        # Calculate median
        median_progression = []
        for pos in sorted(by_position.keys()):
            losses = by_position[pos]['loss']
            ppls = by_position[pos]['perplexity']
            median_progression.append({
                'position': pos,
                'loss': statistics.median(losses),
                'perplexity': statistics.median(ppls),
                'n_samples': len(losses)
            })

        # Separate individual files
        # Find the maximum number of files across all positions
        n_files = max(len(by_position[pos]['perplexity']) for pos in by_position.keys())
        individual = [[] for _ in range(n_files)]

        for pos in sorted(by_position.keys()):
            n_samples_at_pos = len(by_position[pos]['perplexity'])
            for file_idx in range(min(n_files, n_samples_at_pos)):
                individual[file_idx].append({
                    'position': pos,
                    'loss': by_position[pos]['loss'][file_idx],
                    'perplexity': by_position[pos]['perplexity'][file_idx]
                })

        return {
            'raw': raw_progression,
            'mean': mean_progression,
            'median': median_progression,
            'individual': individual
        }

    def process_checkpoint(self, checkpoint_path: Path) -> Optional[CheckpointRun]:
        """Process a single checkpoint and extract all data"""
        logger.info(f"📂 Processing {checkpoint_path}")

        # Check if we have a custom metrics filename from registry
        metrics_filename = getattr(self, 'registry_runs', {}).get(str(checkpoint_path), "paper_metrics_results.json")
        metrics = self.load_paper_metrics(checkpoint_path, metrics_filename)

        if not metrics:
            logger.warning(f"⚠️  Skipping {checkpoint_path} - no metrics found")
            return None

        # Load config (or create baseline config if missing)
        config = self.load_training_config(checkpoint_path)
        is_baseline = False

        if not config:
            # Check if this is a baseline run (has metrics but no config)
            if metrics_filename.startswith("baseline"):
                logger.info(f"📊 Detected baseline run (no training config)")
                is_baseline = True
                # Create minimal config for baseline
                config = {
                    "baseline": True,
                    "model": "kyutai/moshiko-pytorch-bf16",
                    "ttt": {"enable": False},
                    "lora": {"enable": False}
                }
            else:
                logger.warning(f"⚠️  Skipping {checkpoint_path} - missing config")
                return None

        # Extract metadata
        run_id = self.generate_run_id(checkpoint_path, is_baseline)
        run_name = self.extract_run_name(checkpoint_path, config, is_baseline)
        checkpoint_step = self.extract_checkpoint_step(checkpoint_path) if not is_baseline else 0

        # Get evaluation timestamp from metrics file
        metrics_path = checkpoint_path / metrics_filename
        eval_timestamp = datetime.fromtimestamp(metrics_path.stat().st_mtime).isoformat()

        # Extract tags
        tags = self.extract_tags(config, metrics)

        # Parse LibriLight progression
        # First check if metrics already contain progression data (e.g., for baseline)
        raw_progression = metrics.get('librilight_progression', [])

        # If not in metrics, try to parse from log files
        if not raw_progression:
            raw_progression = self.parse_librilight_logs(checkpoint_path)

        # Aggregate progression data
        aggregated = self.aggregate_progression_data(raw_progression)

        # Generate notes
        notes = self.generate_notes(config, checkpoint_path)

        # Create run object
        run = CheckpointRun(
            id=run_id,
            name=run_name,
            checkpoint_path=str(checkpoint_path),
            checkpoint_step=checkpoint_step,
            evaluation_timestamp=eval_timestamp,
            training_config=config,
            metrics=metrics,
            librilight_progression=aggregated['raw'],  # Keep raw for backward compatibility
            librilight_progression_mean=aggregated['mean'],
            librilight_progression_median=aggregated['median'],
            librilight_progression_individual=aggregated['individual'],
            tags=tags,
            notes=notes
        )

        return run

    def generate_notes(self, config: Dict[str, Any], checkpoint_path: Path) -> str:
        """Generate descriptive notes for the run"""
        notes = []

        # Training type
        training_type = self.classify_training_type(config)

        if training_type == 'ttt':
            ttt = config['ttt']
            notes.append(f"TTT training on layers {ttt.get('layers', 'none')}")
            notes.append(f"Base LR: {ttt.get('base_lr', 0)}")
            notes.append(f"Gating alpha: {ttt.get('initial_gating_alpha', 0)}")
        elif training_type == 'lora':
            lora = config['lora']
            notes.append(f"LoRA training (rank {lora.get('rank', 0)})")
        elif training_type == 'full':
            notes.append("Full finetuning")

        # Data source
        train_data = config.get('data', {}).get('train_data', '')
        if train_data:
            data_name = Path(train_data).parent.name
            notes.append(f"Data: {data_name}")

        # Training duration
        max_steps = config.get('max_steps', 0)
        notes.append(f"Trained for {max_steps} steps")

        return " | ".join(notes)

    def aggregate_all(self) -> Dict[str, Any]:
        """
        Aggregate all checkpoints into dashboard data structure

        Returns:
            Complete dashboard data dict
        """
        logger.info("🚀 Starting aggregation...")

        # Find all checkpoints (uses registry by default, falls back to full scan if needed)
        checkpoint_paths = self.find_checkpoints_from_registry()

        if not checkpoint_paths:
            logger.error("❌ No checkpoints found!")
            return {
                'schema_version': '1.0',
                'metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'total_runs': 0,
                    'checkpoint_base_paths': []
                },
                'runs': []
            }

        # Process each checkpoint
        runs = []
        for checkpoint_path in checkpoint_paths:
            run = self.process_checkpoint(checkpoint_path)
            if run:
                runs.append(asdict(run))

        # Sort runs by checkpoint step (descending)
        runs.sort(key=lambda x: x['checkpoint_step'], reverse=True)

        # Create metadata
        metadata = {
            'generated_at': datetime.now().isoformat(),
            'total_runs': len(runs),
            'checkpoint_base_paths': [str(d) for d in self.checkpoint_dirs]
        }

        # Try to identify baseline run
        baseline = None
        for run in runs:
            if run['tags'] and 'baseline' in run['tags']:
                baseline = {
                    'id': run['id'],
                    'name': run['name'],
                    'metrics': run['metrics'],
                    'librilight_progression': run.get('librilight_progression', []),
                    'librilight_progression_mean': run.get('librilight_progression_mean', []),
                    'librilight_progression_median': run.get('librilight_progression_median', []),
                    'librilight_progression_individual': run.get('librilight_progression_individual', [])
                }
                break

        # Build final data structure
        dashboard_data = {
            'schema_version': '1.0',
            'metadata': metadata,
            'baseline': baseline,
            'runs': runs
        }

        logger.info(f"✅ Aggregation complete! Processed {len(runs)} runs")

        return dashboard_data

    def save_to_file(self, data: Dict[str, Any], output_path: Path):
        """Save aggregated data to JSON file"""
        logger.info(f"💾 Saving to {output_path}...")

        try:
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info(f"✅ Dashboard data saved successfully!")
            logger.info(f"📊 File size: {output_path.stat().st_size / 1024:.1f} KB")

        except Exception as e:
            logger.error(f"❌ Error saving file: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate paper metrics results for dashboard'
    )

    parser.add_argument(
        '--checkpoint-dirs',
        nargs='+',
        default=['/sise/eliyanac-group/ron_al'],
        help='Base directories to search for checkpoints'
    )

    parser.add_argument(
        '--log-dir',
        type=str,
        default='/home/alufr/ttt_tests/moshi-finetune',
        help='Directory containing paper_metrics_*.log files'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='dashboard_data.json',
        help='Output JSON file path'
    )

    parser.add_argument(
        '--incremental',
        action='store_true',
        help='Incremental mode: only add new runs (NOT IMPLEMENTED YET)'
    )

    args = parser.parse_args()

    if args.incremental:
        logger.warning("⚠️  Incremental mode not implemented yet, running full aggregation")

    # Create aggregator
    aggregator = PaperMetricsAggregator(
        checkpoint_dirs=args.checkpoint_dirs,
        log_dir=args.log_dir if Path(args.log_dir).exists() else None
    )

    # Run aggregation
    try:
        data = aggregator.aggregate_all()

        # Save to file
        output_path = Path(args.output)
        aggregator.save_to_file(data, output_path)

        logger.info("🎉 Done! You can now open dashboard.html to view results")

    except Exception as e:
        logger.error(f"❌ Aggregation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
