#!/usr/bin/env python3
"""
Plot All Training Runs - Simple and Robust

Creates multiple visualization plots from all evaluation results:
1. LibriLight Loss Curves (for runs with LibriLight data)
2. Paper Metrics Comparison (for runs with paper metrics)
3. Summary Statistics

Usage:
    python plot_all_training_runs.py evaluation_results_from_logs/ --output plots/
"""

import json
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np


def load_all_results(directory: Path) -> List[Dict]:
    """Load all results.json files from subdirectories."""
    results = []
    for results_file in directory.glob("*/results.json"):
        try:
            with open(results_file, 'r') as f:
                data = json.load(f)
                data['_source_file'] = str(results_file)
                results.append(data)
        except Exception as e:
            print(f"⚠️ Warning: Failed to load {results_file}: {e}")
    return results


def extract_librilight_curves(results: List[Dict]) -> List[Tuple[str, List[int], List[float]]]:
    """Extract LibriLight loss curves from all results."""
    curves = []
    
    for data in results:
        librilight = data.get('results', {}).get('librilight', {})
        positions = []
        losses = []
        
        for key, value in librilight.items():
            if key.startswith('librilight_loss_') and key.endswith('k'):
                try:
                    pos = int(key.replace('librilight_loss_', '').replace('k', ''))
                    positions.append(pos)
                    losses.append(float(value))
                except (ValueError, TypeError):
                    continue
        
        if positions:
            sorted_data = sorted(zip(positions, losses))
            positions, losses = zip(*sorted_data)
            
            # Get run identifier
            job_id = data.get('metadata', {}).get('slurm_job_id', 'unknown')
            curves.append((f"Job {job_id}", list(positions), list(losses)))
    
    return curves


def extract_paper_metrics(results: List[Dict]) -> List[Tuple[str, Dict[str, float]]]:
    """Extract paper metrics from all results."""
    metrics = []
    
    for data in results:
        paper_metrics = data.get('results', {}).get('paper_metrics', {})
        if not paper_metrics:
            continue
        
        # Flatten all accuracy metrics
        accuracies = {}
        for category, values in paper_metrics.items():
            if isinstance(values, dict):
                for key, value in values.items():
                    if 'accuracy' in key and isinstance(value, (int, float)):
                        clean_key = key.replace('_accuracy', '')
                        accuracies[clean_key] = float(value)
        
        if accuracies:
            job_id = data.get('metadata', {}).get('slurm_job_id', 'unknown')
            metrics.append((f"Job {job_id}", accuracies))
    
    return metrics


def plot_librilight_curves(curves: List[Tuple[str, List[int], List[float]]], output_path: Path):
    """Plot all LibriLight loss curves."""
    if not curves:
        print("⚠️ No LibriLight curves to plot")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: All curves overlaid
    colors = plt.cm.viridis(np.linspace(0, 1, len(curves)))
    
    for (label, positions, losses), color in zip(curves, colors):
        ax1.plot(positions, losses, '-o', label=label, alpha=0.7, markersize=3, linewidth=1.5)
    
    ax1.set_xlabel('Token Position (k)', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title(f'LibriLight Loss Curves - All {len(curves)} Runs', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Plot 2: Summary statistics
    all_positions = sorted(set(pos for _, positions, _ in curves for pos in positions))
    
    if all_positions:
        losses_by_position = {pos: [] for pos in all_positions}
        
        for _, positions, losses in curves:
            for pos, loss in zip(positions, losses):
                losses_by_position[pos].append(loss)
        
        positions_list = []
        mean_losses = []
        std_losses = []
        min_losses = []
        max_losses = []
        
        for pos in sorted(all_positions):
            if losses_by_position[pos]:
                positions_list.append(pos)
                losses_array = np.array(losses_by_position[pos])
                mean_losses.append(np.mean(losses_array))
                std_losses.append(np.std(losses_array))
                min_losses.append(np.min(losses_array))
                max_losses.append(np.max(losses_array))
        
        ax2.plot(positions_list, mean_losses, 'b-o', linewidth=2, label='Mean Loss', markersize=6)
        ax2.fill_between(positions_list, 
                         np.array(mean_losses) - np.array(std_losses),
                         np.array(mean_losses) + np.array(std_losses),
                         alpha=0.3, label='±1 Std Dev')
        ax2.plot(positions_list, min_losses, 'g--', alpha=0.5, label='Best (Min)', linewidth=1.5)
        ax2.plot(positions_list, max_losses, 'r--', alpha=0.5, label='Worst (Max)', linewidth=1.5)
        
        ax2.set_xlabel('Token Position (k)', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.set_title(f'LibriLight Loss - Aggregate Statistics ({len(curves)} runs)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Saved LibriLight curves: {output_path}")


def plot_paper_metrics(metrics: List[Tuple[str, Dict[str, float]]], output_path: Path):
    """Plot paper metrics comparison."""
    if not metrics:
        print("⚠️ No paper metrics to plot")
        return
    
    # Get all unique metrics
    all_metrics = set()
    for _, metric_dict in metrics:
        all_metrics.update(metric_dict.keys())
    
    all_metrics = sorted(all_metrics)
    
    if not all_metrics:
        print("⚠️ No metrics found")
        return
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(max(12, len(all_metrics) * 0.8), 8))
    
    # Prepare data
    x_positions = np.arange(len(all_metrics))
    bar_width = 0.8 / len(metrics) if metrics else 0.8
    
    for i, (label, metric_dict) in enumerate(metrics):
        values = [metric_dict.get(metric, 0) * 100 for metric in all_metrics]  # Convert to percentage
        offset = (i - len(metrics) / 2) * bar_width
        ax.bar(x_positions + offset, values, bar_width, label=label, alpha=0.7)
    
    ax.set_xlabel('Benchmark', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(f'Paper Metrics Comparison - {len(metrics)} Runs', fontsize=14, fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(all_metrics, rotation=45, ha='right')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 100])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Saved paper metrics: {output_path}")


def create_summary_plot(results: List[Dict], output_path: Path):
    """Create a summary statistics plot."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Collect statistics
    job_ids = []
    final_steps = []
    avg_accuracies = []
    libri_16k_losses = []
    
    for data in results:
        job_id = data.get('metadata', {}).get('slurm_job_id', 'unknown')
        
        # Final step
        training = data.get('results', {}).get('training_progress', {})
        if training and 'final_step' in training:
            job_ids.append(job_id)
            final_steps.append(training['final_step'])
            
            # Average accuracy
            paper_metrics = data.get('results', {}).get('paper_metrics', {})
            aggregate = paper_metrics.get('aggregate', {})
            avg_acc = aggregate.get('paper_metrics_avg', 0) * 100
            avg_accuracies.append(avg_acc)
            
            # LibriLight 16k
            librilight = data.get('results', {}).get('librilight', {})
            libri_16k = librilight.get('librilight_loss_16k', 0)
            libri_16k_losses.append(libri_16k if libri_16k > 0 else np.nan)
    
    # Plot 1: Training Steps
    if final_steps:
        ax1.bar(range(len(job_ids)), final_steps, alpha=0.7, color='skyblue')
        ax1.set_xlabel('Run', fontsize=10)
        ax1.set_ylabel('Final Training Step', fontsize=10)
        ax1.set_title(f'Training Progress - {len(job_ids)} Runs', fontsize=12, fontweight='bold')
        ax1.set_xticks(range(len(job_ids)))
        ax1.set_xticklabels([str(jid) for jid in job_ids], rotation=90, fontsize=6)
        ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Average Accuracy
    if avg_accuracies:
        colors = ['green' if acc > 50 else 'orange' if acc > 0 else 'red' for acc in avg_accuracies]
        ax2.bar(range(len(job_ids)), avg_accuracies, alpha=0.7, color=colors)
        ax2.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% threshold')
        ax2.set_xlabel('Run', fontsize=10)
        ax2.set_ylabel('Average Accuracy (%)', fontsize=10)
        ax2.set_title(f'Paper Metrics Performance - {len(job_ids)} Runs', fontsize=12, fontweight='bold')
        ax2.set_xticks(range(len(job_ids)))
        ax2.set_xticklabels([str(jid) for jid in job_ids], rotation=90, fontsize=6)
        ax2.set_ylim([0, 100])
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.legend()
    
    # Plot 3: LibriLight 16k Loss
    valid_libri = [(i, loss) for i, loss in enumerate(libri_16k_losses) if not np.isnan(loss) and loss > 0]
    if valid_libri:
        indices, losses = zip(*valid_libri)
        ax3.bar(range(len(indices)), losses, alpha=0.7, color='purple')
        ax3.set_xlabel('Run (with LibriLight)', fontsize=10)
        ax3.set_ylabel('Loss at 16k tokens', fontsize=10)
        ax3.set_title(f'LibriLight 16k Loss - {len(valid_libri)} Runs', fontsize=12, fontweight='bold')
        ax3.set_xticks(range(len(indices)))
        ax3.set_xticklabels([str(job_ids[i]) for i in indices], rotation=90, fontsize=6)
        ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Summary Text
    ax4.axis('off')
    summary_text = f"TRAINING RUNS SUMMARY\n" + "="*50 + "\n\n"
    summary_text += f"Total Runs: {len(results)}\n"
    summary_text += f"Runs with training data: {len(final_steps)}\n"
    summary_text += f"Runs with paper metrics: {sum(1 for acc in avg_accuracies if acc > 0)}\n"
    summary_text += f"Runs with LibriLight: {len(valid_libri)}\n\n"
    
    if final_steps:
        summary_text += f"Training Steps:\n"
        summary_text += f"  Min: {min(final_steps)}\n"
        summary_text += f"  Max: {max(final_steps)}\n"
        summary_text += f"  Mean: {np.mean(final_steps):.0f}\n\n"
    
    if avg_accuracies:
        valid_accs = [acc for acc in avg_accuracies if acc > 0]
        if valid_accs:
            summary_text += f"Paper Metrics Accuracy:\n"
            summary_text += f"  Min: {min(valid_accs):.1f}%\n"
            summary_text += f"  Max: {max(valid_accs):.1f}%\n"
            summary_text += f"  Mean: {np.mean(valid_accs):.1f}%\n\n"
    
    if valid_libri:
        losses_only = [loss for _, loss in valid_libri]
        summary_text += f"LibriLight 16k Loss:\n"
        summary_text += f"  Best (min): {min(losses_only):.3f}\n"
        summary_text += f"  Worst (max): {max(losses_only):.3f}\n"
        summary_text += f"  Mean: {np.mean(losses_only):.3f}\n"
    
    ax4.text(0.1, 0.95, summary_text, transform=ax4.transAxes, 
             fontsize=11, family='monospace', va='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Saved summary: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Plot all training runs')
    parser.add_argument('directory', type=str, help='Directory containing evaluation results')
    parser.add_argument('--output', type=str, default='plots/',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    try:
        # Load all results
        directory = Path(args.directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        print(f"🔍 Loading results from {directory}...")
        results = load_all_results(directory)
        print(f"✅ Loaded {len(results)} result files")
        
        if not results:
            print("❌ No results found!")
            return 1
        
        # Create output directory
        output_dir = Path(args.output)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Extract data
        librilight_curves = extract_librilight_curves(results)
        paper_metrics = extract_paper_metrics(results)
        
        print(f"📊 Found {len(librilight_curves)} runs with LibriLight data")
        print(f"📊 Found {len(paper_metrics)} runs with paper metrics")
        
        # Create plots
        if librilight_curves:
            plot_librilight_curves(librilight_curves, output_dir / 'librilight_curves.png')
        
        if paper_metrics:
            plot_paper_metrics(paper_metrics, output_dir / 'paper_metrics_comparison.png')
        
        create_summary_plot(results, output_dir / 'summary_statistics.png')
        
        print(f"\n✅ All plots saved to {output_dir}/")
        print(f"   - librilight_curves.png")
        print(f"   - paper_metrics_comparison.png")
        print(f"   - summary_statistics.png")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
