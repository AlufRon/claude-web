#!/usr/bin/env python3
"""
Enhanced Evaluation Results Comparison with Clear TTT vs Baseline Distinction

This creates highly detailed visualizations that clearly distinguish between TTT and Baseline models:
- Color-coded bars (TTT = Blue, Baseline = Red)
- Grouped comparisons with statistical analysis
- Clear model type labeling and legends
- Aggregated performance summaries
- Detailed difference analysis

Usage:
    python plot_enhanced_comparison.py evaluation_results/ --output enhanced_comparison.png
"""

import json
import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
from datetime import datetime
import seaborn as sns
from collections import defaultdict

# Set style for better visualizations
plt.style.use('default')
sns.set_palette("husl")


def load_run_results(json_path: Path) -> Dict:
    """Load and validate results from JSON file."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Validate structure
        required_keys = ['metadata', 'results']
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing required key: {key}")
        
        return data
    
    except Exception as e:
        print(f"Error loading {json_path}: {e}")
        raise


def extract_benchmark_accuracies(results: Dict) -> Dict[str, float]:
    """Extract benchmark accuracy values from results."""
    accuracies = {}
    
    # Look through all categories for accuracy metrics
    for category, metrics in results.items():
        if not isinstance(metrics, dict):
            continue
            
        for key, value in metrics.items():
            if 'accuracy' in key and isinstance(value, (int, float)):
                # Clean up the key name for display
                clean_key = key.replace('_accuracy', '')
                accuracies[clean_key] = float(value)
    
    return accuracies


def extract_librilight_data(results: Dict) -> tuple:
    """Extract LibriLight loss progression data."""
    positions = []
    losses = []
    
    # Look for LibriLight category
    librilight_data = results.get('librilight', {})
    
    # Extract loss at different positions
    for key, value in librilight_data.items():
        if key.startswith('librilight_loss_') and key.endswith('k'):
            try:
                pos = int(key.replace('librilight_loss_', '').replace('k', ''))
                positions.append(pos)
                losses.append(float(value))
            except ValueError:
                continue
    
    # Sort by position
    if positions:
        sorted_data = sorted(zip(positions, losses))
        positions, losses = zip(*sorted_data)
    
    return list(positions), list(losses)


def discover_run_files(directory: Path) -> List[Path]:
    """Auto-discover all results.json files in directory structure."""
    json_files = []
    
    for path in directory.rglob("results.json"):
        if path.is_file():
            json_files.append(path)
    
    # Sort by modification time (newest first)
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
        elif 'BASELINE' in model_type:
            baseline_runs.append(run)
        else:
            # Try to infer from path or other indicators
            print(f"Warning: Unclear model type '{model_type}', categorizing as baseline")
            baseline_runs.append(run)
    
    return ttt_runs, baseline_runs


def calculate_statistics(values: List[float]) -> Dict[str, float]:
    """Calculate basic statistics for a list of values."""
    if not values:
        return {'mean': 0, 'std': 0, 'min': 0, 'max': 0}
    
    return {
        'mean': np.mean(values),
        'std': np.std(values),
        'min': np.min(values),
        'max': np.max(values)
    }


def plot_enhanced_comparison(directory: Path, output_path: Path):
    """Create enhanced comparison plot with clear TTT vs Baseline distinction."""
    
    # Load all runs
    json_files = discover_run_files(directory)
    if not json_files:
        raise ValueError(f"No results.json files found in {directory}")
    
    runs_data = []
    for path in json_files:
        try:
            data = load_run_results(path)
            runs_data.append(data)
        except Exception as e:
            print(f"Warning: Could not load {path}: {e}")
            continue
    
    if not runs_data:
        raise ValueError("No valid run data found")
    
    # Categorize runs
    ttt_runs, baseline_runs = categorize_runs(runs_data)
    
    print(f"Found {len(ttt_runs)} TTT runs and {len(baseline_runs)} Baseline runs")
    
    # Create the figure with more space and better layout
    fig = plt.figure(figsize=(24, 16))
    gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.4, wspace=0.3)
    
    fig.suptitle(f'TTT vs Baseline Model Comparison\n'
                 f'{len(ttt_runs)} TTT Runs vs {len(baseline_runs)} Baseline Runs',
                 fontsize=20, fontweight='bold')
    
    # Define colors for clear distinction
    ttt_color = '#2E86AB'      # Blue
    baseline_color = '#F24236'  # Red
    ttt_alpha = 0.8
    baseline_alpha = 0.8
    
    # 1. Linguistic Benchmarks Comparison (Top Row)
    benchmark_names = ['sblimp', 'swuggy', 'tstory', 'sstory']
    benchmark_titles = ['sBLIMP\n(Syntax)', 'sWUGGY\n(Phonotactics)', 'tStoryCloze\n(Reasoning)', 'sStoryCloze\n(Coherence)']
    
    for idx, (benchmark, title) in enumerate(zip(benchmark_names, benchmark_titles)):
        ax = fig.add_subplot(gs[0, idx])
        
        # Extract benchmark values for each category
        ttt_values = []
        baseline_values = []
        
        for run in ttt_runs:
            acc = extract_benchmark_accuracies(run['results'])
            if benchmark in acc:
                ttt_values.append(acc[benchmark] * 100)
        
        for run in baseline_runs:
            acc = extract_benchmark_accuracies(run['results'])
            if benchmark in acc:
                baseline_values.append(acc[benchmark] * 100)
        
        # Calculate statistics
        ttt_stats = calculate_statistics(ttt_values)
        baseline_stats = calculate_statistics(baseline_values)
        
        # Create grouped bar plot
        x_pos = [0, 1]
        means = [ttt_stats['mean'], baseline_stats['mean']]
        stds = [ttt_stats['std'], baseline_stats['std']]
        colors = [ttt_color, baseline_color]
        labels = ['TTT', 'Baseline']
        
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.8, 
                     color=colors, edgecolor='black', linewidth=1)
        
        # Add individual data points
        if ttt_values:
            ax.scatter([0] * len(ttt_values), ttt_values, 
                      alpha=0.6, color=ttt_color, s=30, label=f'TTT runs (n={len(ttt_values)})')
        if baseline_values:
            ax.scatter([1] * len(baseline_values), baseline_values, 
                      alpha=0.6, color=baseline_color, s=30, label=f'Baseline runs (n={len(baseline_values)})')
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, fontweight='bold', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontweight='bold')
        ax.set_title(title, fontweight='bold', fontsize=14)
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Chance')
        ax.set_ylim([40, 100])
        ax.grid(axis='y', alpha=0.3)
        
        # Add statistical significance annotation
        if ttt_values and baseline_values:
            diff = ttt_stats['mean'] - baseline_stats['mean']
            ax.text(0.5, max(means) + max(stds) + 3, 
                   f'Δ = {diff:+.1f}%', ha='center', va='bottom', 
                   fontweight='bold', fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add value labels on bars
        for bar, val, std in zip(bars, means, stds):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std + 1,
                   f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # 2. LibriLight Long Context Performance (Second Row, Spans 2 columns)
    ax_libri = fig.add_subplot(gs[1, :2])
    
    ttt_libri_data = []
    baseline_libri_data = []
    
    # Collect LibriLight data
    for run in ttt_runs:
        positions, losses = extract_librilight_data(run['results'])
        if positions and losses:
            ttt_libri_data.append((positions, losses))
    
    for run in baseline_runs:
        positions, losses = extract_librilight_data(run['results'])
        if positions and losses:
            baseline_libri_data.append((positions, losses))
    
    # Plot LibriLight comparisons
    if ttt_libri_data:
        # Calculate mean and std for TTT
        all_positions = ttt_libri_data[0][0]  # Assume same positions for all runs
        ttt_losses_matrix = np.array([losses for _, losses in ttt_libri_data])
        ttt_mean_losses = np.mean(ttt_losses_matrix, axis=0)
        ttt_std_losses = np.std(ttt_losses_matrix, axis=0)
        
        ax_libri.plot(all_positions, ttt_mean_losses, 'o-', color=ttt_color, 
                     linewidth=3, markersize=8, label=f'TTT (n={len(ttt_libri_data)})')
        ax_libri.fill_between(all_positions, 
                             ttt_mean_losses - ttt_std_losses,
                             ttt_mean_losses + ttt_std_losses,
                             alpha=0.2, color=ttt_color)
    
    if baseline_libri_data:
        # Calculate mean and std for Baseline
        all_positions = baseline_libri_data[0][0]  # Assume same positions for all runs
        baseline_losses_matrix = np.array([losses for _, losses in baseline_libri_data])
        baseline_mean_losses = np.mean(baseline_losses_matrix, axis=0)
        baseline_std_losses = np.std(baseline_losses_matrix, axis=0)
        
        ax_libri.plot(all_positions, baseline_mean_losses, 's-', color=baseline_color,
                     linewidth=3, markersize=8, label=f'Baseline (n={len(baseline_libri_data)})')
        ax_libri.fill_between(all_positions,
                             baseline_mean_losses - baseline_std_losses,
                             baseline_mean_losses + baseline_std_losses,
                             alpha=0.2, color=baseline_color)
    
    ax_libri.set_xlabel('Context Length (k tokens)', fontweight='bold', fontsize=12)
    ax_libri.set_ylabel('Cross-Entropy Loss', fontweight='bold', fontsize=12)
    ax_libri.set_title('LibriLight: Long Context Performance\n(Mean ± Std)', fontweight='bold', fontsize=14)
    ax_libri.legend(fontsize=12, loc='upper right')
    ax_libri.grid(True, alpha=0.3)
    
    # 3. Overall Performance Summary (Second Row, Right)
    ax_overall = fig.add_subplot(gs[1, 2:])
    
    # Calculate overall performance for each category
    ttt_overall = []
    baseline_overall = []
    
    for run in ttt_runs:
        acc = extract_benchmark_accuracies(run['results'])
        if acc:
            ttt_overall.append(np.mean(list(acc.values())) * 100)
    
    for run in baseline_runs:
        acc = extract_benchmark_accuracies(run['results'])
        if acc:
            baseline_overall.append(np.mean(list(acc.values())) * 100)
    
    # Create box plots for overall performance
    data_to_plot = []
    labels_to_plot = []
    colors_to_plot = []
    
    if ttt_overall:
        data_to_plot.append(ttt_overall)
        labels_to_plot.append(f'TTT\n(n={len(ttt_overall)})')
        colors_to_plot.append(ttt_color)
    
    if baseline_overall:
        data_to_plot.append(baseline_overall)
        labels_to_plot.append(f'Baseline\n(n={len(baseline_overall)})')
        colors_to_plot.append(baseline_color)
    
    if data_to_plot:
        box_plot = ax_overall.boxplot(data_to_plot, labels=labels_to_plot, patch_artist=True)
        
        for patch, color in zip(box_plot['boxes'], colors_to_plot):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Add individual points
        for i, (data, color) in enumerate(zip(data_to_plot, colors_to_plot)):
            y = data
            x = np.random.normal(i+1, 0.04, size=len(y))
            ax_overall.scatter(x, y, alpha=0.6, color=color, s=40)
    
    ax_overall.set_ylabel('Average Accuracy (%)', fontweight='bold', fontsize=12)
    ax_overall.set_title('Overall Performance Distribution', fontweight='bold', fontsize=14)
    ax_overall.grid(axis='y', alpha=0.3)
    ax_overall.set_ylim([45, 75])
    
    # Add performance difference annotation
    if ttt_overall and baseline_overall:
        ttt_mean = np.mean(ttt_overall)
        baseline_mean = np.mean(baseline_overall)
        diff = ttt_mean - baseline_mean
        ax_overall.text(0.5, 0.95, f'TTT Advantage: {diff:+.2f}%', 
                       transform=ax_overall.transAxes, ha='center', va='top',
                       fontweight='bold', fontsize=12,
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # 4. Detailed Statistics Table (Third Row)
    ax_stats = fig.add_subplot(gs[2, :])
    ax_stats.axis('off')
    
    # Create statistics table
    stats_text = "DETAILED PERFORMANCE STATISTICS\n" + "="*60 + "\n\n"
    
    # Individual benchmark statistics
    for benchmark, title in zip(benchmark_names, ['sBLIMP', 'sWUGGY', 'tStoryCloze', 'sStoryCloze']):
        stats_text += f"{title}:\n"
        
        # TTT stats
        ttt_vals = []
        for run in ttt_runs:
            acc = extract_benchmark_accuracies(run['results'])
            if benchmark in acc:
                ttt_vals.append(acc[benchmark] * 100)
        
        # Baseline stats
        baseline_vals = []
        for run in baseline_runs:
            acc = extract_benchmark_accuracies(run['results'])
            if benchmark in acc:
                baseline_vals.append(acc[benchmark] * 100)
        
        if ttt_vals:
            ttt_stats = calculate_statistics(ttt_vals)
            stats_text += f"  TTT:      {ttt_stats['mean']:.1f}% ± {ttt_stats['std']:.1f}% (range: {ttt_stats['min']:.1f}%-{ttt_stats['max']:.1f}%)\n"
        
        if baseline_vals:
            baseline_stats = calculate_statistics(baseline_vals)
            stats_text += f"  Baseline: {baseline_stats['mean']:.1f}% ± {baseline_stats['std']:.1f}% (range: {baseline_stats['min']:.1f}%-{baseline_stats['max']:.1f}%)\n"
        
        if ttt_vals and baseline_vals:
            diff = ttt_stats['mean'] - baseline_stats['mean']
            stats_text += f"  Difference: {diff:+.1f}% {'(TTT Better)' if diff > 0 else '(Baseline Better)'}\n"
        
        stats_text += "\n"
    
    # Overall statistics
    if ttt_overall and baseline_overall:
        stats_text += "OVERALL PERFORMANCE:\n"
        ttt_overall_stats = calculate_statistics(ttt_overall)
        baseline_overall_stats = calculate_statistics(baseline_overall)
        
        stats_text += f"TTT Average:      {ttt_overall_stats['mean']:.2f}% ± {ttt_overall_stats['std']:.2f}%\n"
        stats_text += f"Baseline Average: {baseline_overall_stats['mean']:.2f}% ± {baseline_overall_stats['std']:.2f}%\n"
        stats_text += f"TTT Advantage:    {ttt_overall_stats['mean'] - baseline_overall_stats['mean']:+.2f}%\n"
    
    ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes, ha='left', va='top',
                 fontsize=11, family='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # 5. Model Information Summary (Fourth Row)
    ax_info = fig.add_subplot(gs[3, :])
    ax_info.axis('off')
    
    info_text = "RUN INFORMATION SUMMARY\n" + "="*30 + "\n\n"
    
    info_text += f"Total Runs Analyzed: {len(runs_data)}\n"
    info_text += f"  • TTT Models: {len(ttt_runs)}\n"
    info_text += f"  • Baseline Models: {len(baseline_runs)}\n\n"
    
    info_text += "TTT Runs:\n"
    for i, run in enumerate(ttt_runs[:10]):  # Show first 10
        timestamp = run['metadata'].get('timestamp', 'Unknown')
        info_text += f"  {i+1}. {timestamp}\n"
    if len(ttt_runs) > 10:
        info_text += f"  ... and {len(ttt_runs) - 10} more\n"
    
    info_text += "\nBaseline Runs:\n"
    for i, run in enumerate(baseline_runs[:10]):  # Show first 10
        timestamp = run['metadata'].get('timestamp', 'Unknown')
        info_text += f"  {i+1}. {timestamp}\n"
    if len(baseline_runs) > 10:
        info_text += f"  ... and {len(baseline_runs) - 10} more\n"
    
    # Add configuration information
    info_text += f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    
    ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes, ha='left', va='top',
                fontsize=10, family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"📊 Saved enhanced comparison plot: {output_path}")
    print(f"   TTT runs: {len(ttt_runs)}, Baseline runs: {len(baseline_runs)}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Create enhanced TTT vs Baseline comparison plots')
    parser.add_argument('directory', type=str, help='Directory containing evaluation results')
    parser.add_argument('--output', type=str, default='enhanced_ttt_baseline_comparison.png',
                       help='Output path for enhanced plot')
    
    args = parser.parse_args()
    
    try:
        directory = Path(args.directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        output_path = Path(args.output)
        plot_enhanced_comparison(directory, output_path)
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())