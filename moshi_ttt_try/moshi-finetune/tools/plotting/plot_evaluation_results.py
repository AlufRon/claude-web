#!/usr/bin/env python3
"""
Standalone Evaluation Results Plotting Utility

This utility creates plots from saved evaluation JSON files, enabling:
- Single-run comprehensive visualizations  
- Multi-run comparison plots
- Flexible post-hoc analysis without re-running evaluations

Usage:
    # Single run plot
    python plot_evaluation_results.py --run path/to/results.json --output single_run.png
    
    # Multi-run comparison 
    python plot_evaluation_results.py --compare run1/results.json run2/results.json --output comparison.png
    
    # Auto-discover and compare all runs
    python plot_evaluation_results.py --compare-all evaluation_results/ --output all_runs.png
"""

import json
import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
from datetime import datetime


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


def plot_single_run(json_path: Path, output_path: Optional[Path] = None):
    """Create comprehensive plot for a single evaluation run."""
    
    data = load_run_results(json_path)
    results = data['results']
    metadata = data['metadata']
    
    # Set up the figure
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    model_type = metadata.get('model_type', 'Unknown')
    timestamp = metadata.get('timestamp', 'Unknown')
    fig.suptitle(f'Paper Metrics Evaluation Results - {model_type} Model ({timestamp})', 
                 fontsize=16, fontweight='bold')
    
    # 1. Benchmark Accuracies Bar Chart
    ax1 = fig.add_subplot(gs[0, :2])
    accuracies = extract_benchmark_accuracies(results)
    
    if accuracies:
        names = list(accuracies.keys())
        values = [accuracies[name] * 100 for name in names]  # Convert to percentage
        
        bars = ax1.bar(names, values, alpha=0.7, color=['blue', 'green', 'orange', 'red'])
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Linguistic Benchmark Performance', fontweight='bold')
        ax1.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Chance Level')
        ax1.set_ylim([0, 100])
        ax1.grid(axis='y', alpha=0.3)
        ax1.legend()
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
    else:
        ax1.text(0.5, 0.5, 'No benchmark data available', ha='center', va='center', 
                transform=ax1.transAxes, fontsize=12)
        ax1.set_title('Linguistic Benchmark Performance', fontweight='bold')
    
    # 2. LibriLight Long Context Performance
    ax2 = fig.add_subplot(gs[0, 2:])
    positions, losses = extract_librilight_data(results)
    
    if positions and losses:
        ax2.plot(positions, losses, 'o-', linewidth=3, markersize=8, color='blue', 
                label=f'{model_type} Model')
        ax2.set_xlabel('Context Length (k tokens)')
        ax2.set_ylabel('Cross-Entropy Loss')
        ax2.set_title('LibriLight: Long Context Performance', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Add slope information if available
        librilight_data = results.get('librilight', {})
        slope = librilight_data.get('librilight_slope', None)
        if slope is not None:
            ax2.text(0.02, 0.98, f'Slope: {slope:.6f}', transform=ax2.transAxes,
                    va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        ax2.text(0.5, 0.5, 'No LibriLight data available', ha='center', va='center',
                transform=ax2.transAxes, fontsize=12)
        ax2.set_title('LibriLight: Long Context Performance', fontweight='bold')
    
    # 3. Detailed Position Analysis (if available)
    ax3 = fig.add_subplot(gs[1, :])
    
    if 'librilight_detailed' in data:
        detailed = data['librilight_detailed']
        positions = detailed.get('positions', [])
        losses = detailed.get('losses', [])
        
        if positions and losses:
            # Sample every 100 positions for cleaner visualization
            step = max(1, len(positions) // 1000)
            sampled_pos = positions[::step]
            sampled_losses = losses[::step]
            
            ax3.plot(sampled_pos, sampled_losses, linewidth=2, alpha=0.8, color='blue')
            ax3.set_xlabel('Token Position')
            ax3.set_ylabel('Cross-Entropy Loss')
            ax3.set_title(f'Position-wise Loss Analysis ({len(positions)} total positions)', 
                         fontweight='bold')
            ax3.grid(True, alpha=0.3)
            
            # Mark key positions
            key_positions = [8000, 16000, 24000, 32000]
            colors = ['red', 'orange', 'green', 'purple']
            for pos, color in zip(key_positions, colors):
                if pos < len(positions):
                    ax3.axvline(x=pos, color=color, linestyle='--', alpha=0.7, 
                               label=f'{pos//1000}k tokens')
            ax3.legend()
    
    if not ('librilight_detailed' in data and data['librilight_detailed'].get('positions')):
        ax3.text(0.5, 0.5, 'No detailed position data available', ha='center', va='center',
                transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Position-wise Loss Analysis', fontweight='bold')
    
    # 4. Configuration and Metadata
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.axis('off')
    
    config_text = "CONFIGURATION\n" + "="*20 + "\n"
    config_text += f"Model Type: {model_type}\n"
    config_text += f"Timestamp: {timestamp}\n"
    
    config = metadata.get('config', {})
    if config:
        config_text += f"\nBenchmark Samples:\n"
        for key in ['sblimp_max_pairs', 'swuggy_max_pairs', 'tstory_max_pairs', 'sstory_max_pairs']:
            if key in config:
                config_text += f"  {key.replace('_max_pairs', '')}: {config[key]}\n"
        
        config_text += f"\nStream Config:\n"
        config_text += f"  Use silence: {config.get('use_silence_codes', 'N/A')}\n"
        config_text += f"  User stream: {config.get('use_user_stream', 'N/A')}\n"
    
    ax4.text(0.05, 0.95, config_text, transform=ax4.transAxes, ha='left', va='top',
            fontsize=9, family='monospace', 
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    
    # 5. Performance Summary
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')
    
    summary_text = "PERFORMANCE SUMMARY\n" + "="*22 + "\n"
    
    # Overall metrics
    for category, metrics in results.items():
        if isinstance(metrics, dict):
            if 'paper_metrics_avg' in metrics:
                avg_acc = metrics['paper_metrics_avg'] * 100
                summary_text += f"Average Accuracy: {avg_acc:.1f}%\n"
            if 'paper_metrics_f1' in metrics:
                f1_score = metrics['paper_metrics_f1'] * 100  
                summary_text += f"F1 Score: {f1_score:.1f}%\n"
    
    # Individual benchmarks
    summary_text += "\nIndividual Benchmarks:\n"
    for name, acc in accuracies.items():
        summary_text += f"  {name}: {acc*100:.1f}%\n"
    
    # LibriLight summary
    if positions and losses:
        summary_text += f"\nLibriLight:\n"
        summary_text += f"  Context: {min(positions)}k - {max(positions)}k tokens\n"
        summary_text += f"  Best loss: {min(losses):.3f}\n"
        if slope is not None:
            improvement = "improving" if slope < 0 else "plateau/declining"
            summary_text += f"  Trend: {improvement}\n"
    
    ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes, ha='left', va='top',
            fontsize=9, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    # 6. Slope Analysis (if available)
    ax6 = fig.add_subplot(gs[2, 2:])
    
    if 'librilight_detailed' in data and 'slopes' in data['librilight_detailed']:
        slopes = data['librilight_detailed']['slopes']
        
        slope_names = []
        slope_values = []
        
        for key, value in slopes.items():
            if isinstance(value, (int, float)):
                slope_names.append(key.replace('_', ' ').title())
                slope_values.append(value)
        
        if slope_names:
            colors = ['red' if v >= 0 else 'green' for v in slope_values]
            bars = ax6.bar(slope_names, slope_values, alpha=0.7, color=colors)
            ax6.set_ylabel('Slope (loss change per token)')
            ax6.set_title('Learning Slope Analysis', fontweight='bold')
            ax6.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax6.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for bar, val in zip(bars, slope_values):
                height = bar.get_height()
                y_pos = height + (max(slope_values) - min(slope_values)) * 0.02
                ax6.text(bar.get_x() + bar.get_width()/2., y_pos,
                        f'{val:.1e}', ha='center', va='bottom', fontsize=8, rotation=45)
    else:
        ax6.text(0.5, 0.5, 'No slope analysis available', ha='center', va='center',
                transform=ax6.transAxes, fontsize=12)
        ax6.set_title('Learning Slope Analysis', fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    if output_path is None:
        output_path = json_path.parent / f'evaluation_plot_{timestamp}.png'
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"📊 Saved single-run plot: {output_path}")
    return output_path


def plot_run_comparison(json_paths: List[Path], output_path: Path):
    """Create comparison plot across multiple evaluation runs with clear TTT vs Baseline distinction."""
    
    runs_data = []
    for path in json_paths:
        try:
            data = load_run_results(path)
            runs_data.append(data)
        except Exception as e:
            print(f"Warning: Could not load {path}: {e}")
            continue
    
    if not runs_data:
        raise ValueError("No valid run data found")
    
    # Create larger figure for better visibility
    fig, axes = plt.subplots(2, 3, figsize=(24, 14))
    fig.suptitle('TTT vs Baseline Model Comparison - All Individual Runs', fontsize=20, fontweight='bold')
    
    # Extract run labels and determine colors based on model type
    run_labels = []
    run_colors = []
    ttt_color = '#2E86AB'      # Blue for TTT
    baseline_color = '#F24236'  # Red for Baseline
    
    for data in runs_data:
        metadata = data['metadata']
        model_type = metadata.get('model_type', 'Unknown').upper()
        timestamp = metadata.get('timestamp', 'Unknown')
        
        # Create cleaner labels
        short_timestamp = timestamp.split('_')[1] if '_' in timestamp else timestamp[-6:]
        if 'TTT' in model_type:
            run_labels.append(f"TTT\n{short_timestamp}")
            run_colors.append(ttt_color)
        else:
            run_labels.append(f"BASE\n{short_timestamp}")
            run_colors.append(baseline_color)
    
    # 1. Compare linguistic benchmarks with color coding
    benchmark_names = ['sblimp', 'swuggy', 'tstory', 'sstory']
    benchmark_titles = ['sBLIMP (Syntax)', 'sWUGGY (Phonotactics)', 'tStoryCloze (Reasoning)', 'sStoryCloze (Coherence)']
    
    for idx, (benchmark, title) in enumerate(zip(benchmark_names, benchmark_titles)):
        row = 0 if idx < 2 else 1
        col = idx % 2
        ax = axes[row, col]
        
        values = []
        colors = []
        
        for i, run in enumerate(runs_data):
            accuracies = extract_benchmark_accuracies(run['results'])
            acc_key = f"{benchmark}_accuracy"
            
            # Look for the accuracy in any category
            found_value = None
            for category, metrics in run['results'].items():
                if isinstance(metrics, dict) and acc_key in metrics:
                    found_value = metrics[acc_key] * 100
                    break
            
            if found_value is None:
                # Try without _accuracy suffix
                for category, metrics in run['results'].items():
                    if isinstance(metrics, dict) and benchmark in metrics:
                        val = metrics[benchmark]
                        if isinstance(val, (int, float)) and 0 <= val <= 1:
                            found_value = val * 100
                            break
            
            values.append(found_value if found_value is not None else 0)
            colors.append(run_colors[i])
        
        # Create bars with distinct colors
        bars = ax.bar(range(len(values)), values, alpha=0.8, color=colors, 
                     edgecolor='black', linewidth=1)
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(run_labels, rotation=45, ha='right', fontsize=10, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontweight='bold', fontsize=14)
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Chance Level')
        ax.set_ylim([45, 75])
        ax.grid(axis='y', alpha=0.3)
        
        # Add legend for first plot only
        if idx == 0:
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor=ttt_color, label='TTT Models'),
                             Patch(facecolor=baseline_color, label='Baseline Models')]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        # Add value labels with better positioning
        for bar, val, color in zip(bars, values, colors):
            if val > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 2. LibriLight comparison with color coding
    ax = axes[0, 2]
    
    ttt_count = 0
    baseline_count = 0
    
    for i, (run, label) in enumerate(zip(runs_data, run_labels)):
        positions, losses = extract_librilight_data(run['results'])
        
        if positions and losses:
            color = run_colors[i]
            model_type = 'TTT' if 'TTT' in label else 'Baseline'
            
            if model_type == 'TTT':
                ttt_count += 1
                line_style = '-'
                marker = 'o'
                alpha = 0.7
            else:
                baseline_count += 1
                line_style = '--'
                marker = 's'
                alpha = 0.7
            
            ax.plot(positions, losses, marker=marker, linestyle=line_style, 
                   color=color, linewidth=2, markersize=5, alpha=alpha, label=label)
    
    ax.set_xlabel('Context Length (k tokens)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cross-Entropy Loss', fontsize=12, fontweight='bold')
    ax.set_title(f'LibriLight: Long Context Performance\n({ttt_count} TTT, {baseline_count} Baseline)', 
                fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Create cleaner legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=ttt_color, linewidth=3, label=f'TTT Models ({ttt_count})'),
                      Line2D([0], [0], color=baseline_color, linewidth=3, linestyle='--', 
                            label=f'Baseline Models ({baseline_count})')]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # 3. Overall performance comparison with enhanced visualization
    ax = axes[1, 2]
    overall_scores = []
    
    for run in runs_data:
        # Look for overall average
        found_avg = None
        for category, metrics in run['results'].items():
            if isinstance(metrics, dict) and 'paper_metrics_avg' in metrics:
                found_avg = metrics['paper_metrics_avg'] * 100
                break
        
        if found_avg is None:
            # Calculate average from individual benchmarks
            accuracies = extract_benchmark_accuracies(run['results'])
            if accuracies:
                found_avg = np.mean(list(accuracies.values())) * 100
            else:
                found_avg = 0
        
        overall_scores.append(found_avg)
    
    # Create bars with model-type colors
    bars = ax.bar(range(len(overall_scores)), overall_scores, alpha=0.8, 
                 color=run_colors, edgecolor='black', linewidth=1)
    ax.set_xticks(range(len(overall_scores)))
    ax.set_xticklabels(run_labels, rotation=45, ha='right', fontsize=10, fontweight='bold')
    ax.set_ylabel('Average Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Overall Performance by Model Type', fontweight='bold', fontsize=14)
    ax.set_ylim([45, 75])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, overall_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
               f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=ttt_color, label='TTT Models'),
                      Patch(facecolor=baseline_color, label='Baseline Models')]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # Add summary statistics in bottom left
    ax_summary = axes[1, 0]
    ax_summary.axis('off')
    
    # Calculate summary stats
    ttt_scores = [score for i, score in enumerate(overall_scores) if 'TTT' in run_labels[i]]
    baseline_scores = [score for i, score in enumerate(overall_scores) if 'BASE' in run_labels[i]]
    
    summary_text = "PERFORMANCE SUMMARY\n" + "="*25 + "\n\n"
    summary_text += f"TTT Models ({len(ttt_scores)} runs):\n"
    if ttt_scores:
        summary_text += f"  Mean: {np.mean(ttt_scores):.1f}%\n"
        summary_text += f"  Best: {max(ttt_scores):.1f}%\n"
        summary_text += f"  Worst: {min(ttt_scores):.1f}%\n"
        summary_text += f"  Std: {np.std(ttt_scores):.1f}%\n"
    
    summary_text += f"\nBaseline Models ({len(baseline_scores)} runs):\n"
    if baseline_scores:
        summary_text += f"  Mean: {np.mean(baseline_scores):.1f}%\n"
        summary_text += f"  Best: {max(baseline_scores):.1f}%\n"
        summary_text += f"  Worst: {min(baseline_scores):.1f}%\n"
        summary_text += f"  Std: {np.std(baseline_scores):.1f}%\n"
    
    if ttt_scores and baseline_scores:
        diff = np.mean(ttt_scores) - np.mean(baseline_scores)
        summary_text += f"\nTTT vs Baseline:\n"
        summary_text += f"  Difference: {diff:+.1f}%\n"
        summary_text += f"  Winner: {'TTT' if diff > 0 else 'Baseline'}"
    
    ax_summary.text(0.05, 0.95, summary_text, transform=ax_summary.transAxes, ha='left', va='top',
                   fontsize=12, family='monospace', fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Add run information in bottom middle
    ax_info = axes[1, 1]
    ax_info.axis('off')
    
    info_text = "RUN DETAILS\n" + "="*15 + "\n\n"
    info_text += "TTT Runs:\n"
    ttt_runs_info = [(i, label) for i, label in enumerate(run_labels) if 'TTT' in label]
    for i, (idx, label) in enumerate(ttt_runs_info[:8]):  # Show first 8
        timestamp = label.split('\n')[1]
        info_text += f"  {i+1}. {timestamp}\n"
    if len(ttt_runs_info) > 8:
        info_text += f"  ... +{len(ttt_runs_info)-8} more\n"
    
    info_text += "\nBaseline Runs:\n"
    baseline_runs_info = [(i, label) for i, label in enumerate(run_labels) if 'BASE' in label]
    for i, (idx, label) in enumerate(baseline_runs_info[:8]):  # Show first 8
        timestamp = label.split('\n')[1]
        info_text += f"  {i+1}. {timestamp}\n"
    if len(baseline_runs_info) > 8:
        info_text += f"  ... +{len(baseline_runs_info)-8} more\n"
    
    ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes, ha='left', va='top',
                fontsize=10, family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"📊 Saved enhanced comparison plot: {output_path}")
    print(f"   TTT runs: {len([l for l in run_labels if 'TTT' in l])}")
    print(f"   Baseline runs: {len([l for l in run_labels if 'BASE' in l])}")
    return output_path


def discover_run_files(directory: Path) -> List[Path]:
    """Auto-discover all results.json files in directory structure."""
    json_files = []
    
    for path in directory.rglob("results.json"):
        if path.is_file():
            json_files.append(path)
    
    # Sort by modification time (newest first)
    json_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    
    return json_files


def main():
    parser = argparse.ArgumentParser(description='Plot evaluation results from JSON files')
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--run', type=str, help='Path to single run JSON file')
    group.add_argument('--compare', nargs='+', help='Paths to multiple run JSON files for comparison')
    group.add_argument('--compare-all', type=str, help='Directory to auto-discover and compare all runs')
    
    parser.add_argument('--output', type=str, help='Output path for plot')
    parser.add_argument('--list-runs', action='store_true', help='List discovered runs without plotting')
    
    args = parser.parse_args()
    
    try:
        if args.run:
            # Single run plot
            json_path = Path(args.run)
            if not json_path.exists():
                raise FileNotFoundError(f"JSON file not found: {json_path}")
            
            output_path = Path(args.output) if args.output else None
            plot_single_run(json_path, output_path)
            
        elif args.compare:
            # Multi-run comparison
            json_paths = [Path(p) for p in args.compare]
            for path in json_paths:
                if not path.exists():
                    raise FileNotFoundError(f"JSON file not found: {path}")
            
            output_path = Path(args.output) if args.output else Path('comparison.png')
            plot_run_comparison(json_paths, output_path)
            
        elif args.compare_all:
            # Auto-discover and compare
            directory = Path(args.compare_all)
            if not directory.exists():
                raise FileNotFoundError(f"Directory not found: {directory}")
            
            json_files = discover_run_files(directory)
            
            if args.list_runs:
                print(f"Discovered {len(json_files)} evaluation runs:")
                for i, path in enumerate(json_files):
                    try:
                        data = load_run_results(path)
                        metadata = data['metadata']
                        model_type = metadata.get('model_type', 'Unknown')
                        timestamp = metadata.get('timestamp', 'Unknown')
                        print(f"  {i+1}. {path} ({model_type}, {timestamp})")
                    except Exception as e:
                        print(f"  {i+1}. {path} (Error: {e})")
                return
            
            if len(json_files) == 0:
                print(f"No results.json files found in {directory}")
                return
            elif len(json_files) == 1:
                print(f"Only one run found, creating single-run plot")
                output_path = Path(args.output) if args.output else None
                plot_single_run(json_files[0], output_path)
            else:
                print(f"Found {len(json_files)} runs, creating comparison plot")
                output_path = Path(args.output) if args.output else Path('all_runs_comparison.png')
                plot_run_comparison(json_files, output_path)
    
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())