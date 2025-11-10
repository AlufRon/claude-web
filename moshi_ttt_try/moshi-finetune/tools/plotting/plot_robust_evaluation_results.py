#!/usr/bin/env python3
"""
Robust Evaluation Results Plotting Utility

This version handles:
- TTT disabled runs (0 layers)
- Missing paper metrics or LibriLight data
- Incomplete evaluation results
- Mixed TTT/Baseline/Failed runs

Usage:
    python plot_robust_evaluation_results.py --run path/to/results.json --output plot.png
    python plot_robust_evaluation_results.py --compare-all directory/ --output comparison.png
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
        
        # Validate basic structure
        if 'metadata' not in data:
            data['metadata'] = {}
        if 'results' not in data:
            data['results'] = {}
        
        return data
    
    except Exception as e:
        print(f"Error loading {json_path}: {e}")
        raise


def determine_model_type(data: Dict) -> str:
    """Determine the actual model type based on configuration."""
    metadata = data.get('metadata', {})
    config = metadata.get('config', {})
    
    # Check for TTT configuration
    ttt_layers = config.get('ttt_layers', {})
    num_layers = ttt_layers.get('num_layers', 0)
    
    if num_layers > 0:
        return 'TTT'
    else:
        # Check if TTT was intended but disabled
        if 'ttt_layers' in config or 'ttt_chunking' in config:
            return 'TTT-Disabled'
        else:
            return 'Baseline'


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


def extract_librilight_data(results: Dict) -> tuple:
    """Extract LibriLight loss progression data."""
    positions = []
    losses = []
    
    librilight_data = results.get('librilight', {})
    
    for key, value in librilight_data.items():
        if key.startswith('librilight_loss_') and key.endswith('k'):
            try:
                pos = int(key.replace('librilight_loss_', '').replace('k', ''))
                positions.append(pos)
                losses.append(float(value))
            except ValueError:
                continue
    
    if positions:
        sorted_data = sorted(zip(positions, losses))
        positions, losses = zip(*sorted_data)
    
    return list(positions), list(losses)


def plot_single_run_robust(json_path: Path, output_path: Optional[Path] = None):
    """Create robust plot for a single evaluation run."""
    
    data = load_run_results(json_path)
    results = data['results']
    metadata = data['metadata']
    
    # Determine actual model type
    model_type = determine_model_type(data)
    
    # Set up the figure
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    timestamp = metadata.get('timestamp', 'Unknown')
    job_id = metadata.get('slurm_job_id', 'Unknown')
    
    fig.suptitle(f'Training Run Analysis - {model_type} Model (Job {job_id}, {timestamp})', 
                 fontsize=16, fontweight='bold')
    
    # 1. Benchmark Accuracies Bar Chart (if available)
    ax1 = fig.add_subplot(gs[0, :2])
    accuracies = extract_benchmark_accuracies(results)
    
    if accuracies:
        names = list(accuracies.keys())
        values = [accuracies[name] * 100 for name in names]
        
        colors = ['blue' if model_type == 'TTT' else 'red' if model_type == 'Baseline' else 'gray']
        bars = ax1.bar(names, values, alpha=0.7, color=colors * len(names))
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
        ax1.text(0.5, 0.5, 'No paper metrics data available\\n(Training may have been incomplete)', 
                ha='center', va='center', transform=ax1.transAxes, fontsize=14, 
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
        ax1.set_title('Linguistic Benchmark Performance', fontweight='bold')
    
    # 2. LibriLight Long Context Performance (if available)
    ax2 = fig.add_subplot(gs[0, 2:])
    positions, losses = extract_librilight_data(results)
    
    if positions and losses:
        color = 'blue' if model_type == 'TTT' else 'red' if model_type == 'Baseline' else 'gray'
        ax2.plot(positions, losses, 'o-', linewidth=3, markersize=8, color=color, 
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
        ax2.text(0.5, 0.5, 'No LibriLight data available\\n(Long context evaluation not run)', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=14,
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
        ax2.set_title('LibriLight: Long Context Performance', fontweight='bold')
    
    # 3. Configuration and Model Details
    ax3 = fig.add_subplot(gs[1, :2])
    ax3.axis('off')
    
    config_text = "MODEL CONFIGURATION\n" + "="*30 + "\n"
    config_text += f"Model Type: {model_type}\n"
    config_text += f"Job ID: {job_id}\n"
    config_text += f"Timestamp: {timestamp}\n"
    config_text += f"Original Log: {metadata.get('original_log_file', 'Unknown')}\n\n"
    
    config = metadata.get('config', {})
    
    # TTT Configuration
    if 'ttt_layers' in config:
        ttt_config = config['ttt_layers']
        config_text += f"TTT Configuration:\n"
        config_text += f"  Layers: {ttt_config.get('num_layers', 0)}\n"
        config_text += f"  Indices: {ttt_config.get('layer_indices', [])}\n"
        config_text += f"  Parameters: {ttt_config.get('parameters', 0):,}\n"
        config_text += f"  Dimension: {ttt_config.get('dim', 'N/A')}\n"
        config_text += f"  Heads: {ttt_config.get('heads', 'N/A')}\n\n"
    
    # Training Configuration
    if 'training' in config:
        training_config = config['training']
        config_text += f"Training Configuration:\n"
        config_text += f"  Learning Rate: {training_config.get('lr', 'N/A')}\n"
        config_text += f"  Max Steps: {training_config.get('max_steps', 'N/A')}\n"
        config_text += f"  Batch Size: {training_config.get('batch_size', 'N/A')}\n"
        config_text += f"  Weight Decay: {training_config.get('weight_decay', 'N/A')}\n\n"
    
    # Benchmark Configuration
    benchmark_counts = []
    for key in ['sblimp_max_pairs', 'swuggy_max_pairs', 'tstory_max_pairs', 'sstory_max_pairs']:
        if key in config:
            benchmark_counts.append(f"{key.replace('_max_pairs', '')}: {config[key]}")
    
    if benchmark_counts:
        config_text += f"Benchmark Samples:\\n  " + "\\n  ".join(benchmark_counts)
    
    ax3.text(0.05, 0.95, config_text, transform=ax3.transAxes, ha='left', va='top',
            fontsize=10, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # 4. Performance Summary
    ax4 = fig.add_subplot(gs[1, 2:])
    ax4.axis('off')
    
    summary_text = "PERFORMANCE SUMMARY\n" + "="*25 + "\n\n"
    
    # Overall metrics
    if 'aggregate' in results:
        agg = results['aggregate']
        if 'paper_metrics_avg' in agg:
            avg_acc = agg['paper_metrics_avg'] * 100
            summary_text += f"Average Accuracy: {avg_acc:.1f}%\\n"
        if 'paper_metrics_f1' in agg:
            f1_score = agg['paper_metrics_f1'] * 100
            summary_text += f"F1 Score: {f1_score:.1f}%\\n"
    
    # Individual benchmarks
    if accuracies:
        summary_text += f"\\nIndividual Benchmarks:\\n"
        for name, acc in accuracies.items():
            summary_text += f"  {name.upper()}: {acc*100:.1f}%\\n"
    else:
        summary_text += f"\\nNo benchmark results available\\n"
    
    # LibriLight summary
    if positions and losses:
        summary_text += f"\\nLibriLight Results:\\n"
        summary_text += f"  Context: {min(positions)}k - {max(positions)}k tokens\\n"
        summary_text += f"  Best loss: {min(losses):.3f}\\n"
        summary_text += f"  Worst loss: {max(losses):.3f}\\n"
        
        slope = results.get('librilight', {}).get('librilight_slope')
        if slope is not None:
            trend = "improving" if slope < 0 else "plateau/declining"
            summary_text += f"  Trend: {trend}\\n"
    else:
        summary_text += f"\\nNo LibriLight results available\\n"
    
    # Training Status
    if not accuracies and not (positions and losses):
        summary_text += f"\\n⚠️ INCOMPLETE TRAINING RUN\\n"
        summary_text += f"This run appears to have failed\\n"
        summary_text += f"or was terminated before evaluation.\\n"
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, ha='left', va='top',
            fontsize=11, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # 5. Status and Diagnostics
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    
    status_text = "RUN STATUS & DIAGNOSTICS\n" + "="*35 + "\n\n"
    
    # Determine run status
    has_benchmarks = bool(accuracies)
    has_librilight = bool(positions and losses)
    
    if has_benchmarks and has_librilight:
        status = "✅ COMPLETE - Full evaluation completed"
        status_color = 'lightgreen'
    elif has_benchmarks:
        status = "⚠️ PARTIAL - Benchmarks only (no LibriLight)"
        status_color = 'lightyellow'
    elif has_librilight:
        status = "⚠️ PARTIAL - LibriLight only (no benchmarks)"
        status_color = 'lightyellow'
    else:
        status = "❌ INCOMPLETE - No evaluation results"
        status_color = 'lightcoral'
    
    status_text += f"Status: {status}\\n\\n"
    
    # Model type explanation
    if model_type == 'TTT':
        status_text += f"✅ TTT Model: {config.get('ttt_layers', {}).get('num_layers', 0)} layers enabled\\n"
    elif model_type == 'TTT-Disabled':
        status_text += f"🔧 TTT Model: TTT layers configured but disabled (0 active layers)\\n"
    else:
        status_text += f"📊 Baseline Model: Standard transformer without TTT\\n"
    
    # Data availability
    status_text += f"\\nData Availability:\\n"
    status_text += f"  Paper Metrics: {'✅ Available' if has_benchmarks else '❌ Missing'}\\n"
    status_text += f"  LibriLight: {'✅ Available' if has_librilight else '❌ Missing'}\\n"
    
    if has_librilight:
        status_text += f"  LibriLight Positions: {len(positions)} measured\\n"
    
    ax5.text(0.05, 0.95, status_text, transform=ax5.transAxes, ha='left', va='top',
            fontsize=11, family='monospace',
            bbox=dict(boxstyle='round', facecolor=status_color, alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    if output_path is None:
        output_path = json_path.parent / f'robust_plot_{timestamp}.png'
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"📊 Saved robust plot: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Create robust plots for evaluation results')
    parser.add_argument('--run', type=str, help='Path to single run JSON file')
    parser.add_argument('--output', type=str, help='Output path for plot')
    
    args = parser.parse_args()
    
    try:
        if args.run:
            json_path = Path(args.run)
            if not json_path.exists():
                raise FileNotFoundError(f"JSON file not found: {json_path}")
            
            output_path = Path(args.output) if args.output else None
            plot_single_run_robust(json_path, output_path)
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())