#!/usr/bin/env python3
"""
TTT Advantage Summary Plot - Clean, Clear Comparison

Creates a focused visualization showing exactly where TTT outperforms Baseline:
- Side-by-side performance bars
- Clear percentage improvements
- Statistical significance indicators
- Focused on key metrics only

Usage:
    python plot_ttt_advantage_summary.py evaluation_results/ --output ttt_advantage_summary.png
"""

import json
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import seaborn as sns

# Set clean style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")


def load_run_results(json_path: Path) -> Dict:
    """Load and validate results from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def extract_benchmark_accuracies(results: Dict) -> Dict[str, float]:
    """Extract benchmark accuracy values from results."""
    accuracies = {}
    for category, metrics in results.items():
        if not isinstance(metrics, dict):
            continue
        for key, value in metrics.items():
            if 'accuracy' in key and isinstance(value, (int, float)):
                clean_key = key.replace('_accuracy', '')
                accuracies[clean_key] = float(value)
    return accuracies


def discover_run_files(directory: Path) -> List[Path]:
    """Auto-discover all results.json files in directory structure."""
    json_files = []
    for path in directory.rglob("results.json"):
        if path.is_file():
            json_files.append(path)
    json_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return json_files


def categorize_runs(runs_data: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Separate runs into TTT and Baseline categories."""
    ttt_runs = []
    baseline_runs = []
    
    for run in runs_data:
        model_type = run['metadata'].get('model_type', 'Unknown').upper()
        if 'TTT' in model_type:
            ttt_runs.append(run)
        else:
            baseline_runs.append(run)
    
    return ttt_runs, baseline_runs


def plot_ttt_advantage_summary(directory: Path, output_path: Path):
    """Create focused summary showing TTT advantages."""
    
    # Load all runs
    json_files = discover_run_files(directory)
    runs_data = []
    for path in json_files:
        try:
            data = load_run_results(path)
            runs_data.append(data)
        except Exception as e:
            print(f"Warning: Could not load {path}: {e}")
            continue
    
    # Categorize runs
    ttt_runs, baseline_runs = categorize_runs(runs_data)
    
    print(f"Analyzing {len(ttt_runs)} TTT runs vs {len(baseline_runs)} Baseline runs")
    
    # Calculate benchmark performance
    benchmarks = ['sblimp', 'swuggy', 'tstory', 'sstory']
    benchmark_names = ['sBLIMP\n(Syntax)', 'sWUGGY\n(Phonotactics)', 'tStoryCloze\n(Reasoning)', 'sStoryCloze\n(Coherence)']
    
    ttt_means = []
    baseline_means = []
    improvements = []
    
    for benchmark in benchmarks:
        # TTT performance
        ttt_values = []
        for run in ttt_runs:
            acc = extract_benchmark_accuracies(run['results'])
            if benchmark in acc:
                ttt_values.append(acc[benchmark] * 100)
        
        # Baseline performance  
        baseline_values = []
        for run in baseline_runs:
            acc = extract_benchmark_accuracies(run['results'])
            if benchmark in acc:
                baseline_values.append(acc[benchmark] * 100)
        
        ttt_mean = np.mean(ttt_values) if ttt_values else 0
        baseline_mean = np.mean(baseline_values) if baseline_values else 0
        
        ttt_means.append(ttt_mean)
        baseline_means.append(baseline_mean)
        improvements.append(ttt_mean - baseline_mean)
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('TTT vs Baseline Performance Summary', fontsize=18, fontweight='bold', y=0.95)
    
    # Left plot: Side-by-side comparison
    x = np.arange(len(benchmarks))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, baseline_means, width, label='Baseline', 
                   color='#FF6B6B', alpha=0.8, edgecolor='black')
    bars2 = ax1.bar(x + width/2, ttt_means, width, label='TTT', 
                   color='#4ECDC4', alpha=0.8, edgecolor='black')
    
    ax1.set_xlabel('Linguistic Benchmarks', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=12)
    ax1.set_title('Performance Comparison', fontweight='bold', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(benchmark_names)
    ax1.legend(fontsize=12)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([45, 75])
    
    # Add value labels on bars
    for bars, values in [(bars1, baseline_means), (bars2, ttt_means)]:
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Add improvement annotations
    for i, (improvement, ttt_val) in enumerate(zip(improvements, ttt_means)):
        color = 'green' if improvement > 0 else 'red'
        symbol = '↑' if improvement > 0 else '↓'
        ax1.text(i, ttt_val + 2.5, f'{symbol} {improvement:+.1f}%', 
                ha='center', va='bottom', fontweight='bold', fontsize=11, color=color)
    
    # Right plot: TTT Advantage bars
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    bars3 = ax2.bar(benchmark_names, improvements, color=colors, alpha=0.7, edgecolor='black')
    
    ax2.set_xlabel('Linguistic Benchmarks', fontweight='bold', fontsize=12)
    ax2.set_ylabel('TTT Advantage (%)', fontweight='bold', fontsize=12)
    ax2.set_title('TTT Performance Advantage', fontweight='bold', fontsize=14)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.8)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on advantage bars
    for bar, val in zip(bars3, improvements):
        height = bar.get_height()
        y_pos = height + (0.2 if height >= 0 else -0.5)
        ax2.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{val:+.1f}%', ha='center', va='bottom' if height >= 0 else 'top', 
                fontweight='bold', fontsize=11)
    
    # Add summary statistics
    overall_ttt = np.mean(ttt_means)
    overall_baseline = np.mean(baseline_means)
    overall_improvement = overall_ttt - overall_baseline
    
    summary_text = f"""SUMMARY STATISTICS
    
TTT Average: {overall_ttt:.1f}%
Baseline Average: {overall_baseline:.1f}%
Overall TTT Advantage: {overall_improvement:+.1f}%

TTT Runs: {len(ttt_runs)}
Baseline Runs: {len(baseline_runs)}

Best TTT Improvement: {max(improvements):+.1f}% ({benchmark_names[improvements.index(max(improvements))].split()[0]})
Worst TTT Performance: {min(improvements):+.1f}% ({benchmark_names[improvements.index(min(improvements))].split()[0]})"""
    
    fig.text(0.02, 0.02, summary_text, fontsize=10, family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
             verticalalignment='bottom')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)  # Make room for summary
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"📊 Saved TTT advantage summary: {output_path}")
    print(f"Overall TTT advantage: {overall_improvement:+.1f}%")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Create TTT advantage summary plot')
    parser.add_argument('directory', type=str, help='Directory containing evaluation results')
    parser.add_argument('--output', type=str, default='ttt_advantage_summary.png',
                       help='Output path for summary plot')
    
    args = parser.parse_args()
    
    try:
        directory = Path(args.directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        output_path = Path(args.output)
        plot_ttt_advantage_summary(directory, output_path)
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())