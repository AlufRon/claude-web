#!/usr/bin/env python3
"""
Memory Pattern Analysis for TTT vs Baseline

This script analyzes memory usage patterns from training logs to identify:
1. Memory peaks and when they occur
2. Average memory usage vs peak usage
3. TTT-specific memory patterns
4. Opportunities for memory optimization

Usage:
    python analyze_memory_patterns.py parsed_logs/ --output memory_analysis/
"""

import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import re
from collections import defaultdict


def parse_log_file_for_memory(log_path: Path) -> Dict:
    """Parse a log file to extract memory usage over time."""
    memory_data = {
        'steps': [],
        'peak_memory': [],
        'allocated_memory': [],
        'loss': [],
        'ttt_enabled': False,
        'num_ttt_layers': 0,
        'job_id': None
    }
    
    try:
        with open(log_path, 'r') as f:
            content = f.read()
        
        # Extract job ID
        job_match = re.search(r'SLURM_JOB_ID:\s*(\d+)', content)
        if job_match:
            memory_data['job_id'] = job_match.group(1)
        
        # Check if TTT is enabled
        ttt_match = re.search(r'Applying TTT to (\d+) layers', content)
        if ttt_match:
            memory_data['ttt_enabled'] = True
            memory_data['num_ttt_layers'] = int(ttt_match.group(1))
        
        # Extract memory data from step logs
        # Pattern: step: 000010 - ... - loss: 2.171 - ... - peak_alloc_mem (GB): 44.2 - alloc_mem (GB): 20.9
        step_pattern = r'step:\s*(\d+).*?loss:\s*([\d.]+).*?peak_alloc_mem \(GB\):\s*([\d.]+).*?alloc_mem \(GB\):\s*([\d.]+)'
        
        for match in re.finditer(step_pattern, content, re.DOTALL):
            step = int(match.group(1))
            loss = float(match.group(2))
            peak_mem = float(match.group(3))
            alloc_mem = float(match.group(4))
            
            memory_data['steps'].append(step)
            memory_data['peak_memory'].append(peak_mem)
            memory_data['allocated_memory'].append(alloc_mem)
            memory_data['loss'].append(loss)
    
    except Exception as e:
        print(f"⚠️ Error parsing {log_path}: {e}")
    
    return memory_data


def calculate_memory_statistics(memory_data: Dict) -> Dict:
    """Calculate memory usage statistics."""
    if not memory_data['peak_memory']:
        return {}
    
    peak_mem = np.array(memory_data['peak_memory'])
    alloc_mem = np.array(memory_data['allocated_memory'])
    
    stats = {
        'peak_memory': {
            'max': float(np.max(peak_mem)),
            'min': float(np.min(peak_mem)),
            'mean': float(np.mean(peak_mem)),
            'std': float(np.std(peak_mem)),
            'median': float(np.median(peak_mem))
        },
        'allocated_memory': {
            'max': float(np.max(alloc_mem)),
            'min': float(np.min(alloc_mem)),
            'mean': float(np.mean(alloc_mem)),
            'std': float(np.std(alloc_mem)),
            'median': float(np.median(alloc_mem))
        },
        'memory_overhead': {
            'max': float(np.max(peak_mem - alloc_mem)),
            'mean': float(np.mean(peak_mem - alloc_mem)),
            'percent_of_peak': float(np.mean((peak_mem - alloc_mem) / peak_mem * 100))
        },
        'total_steps': len(memory_data['steps'])
    }
    
    return stats


def compare_ttt_vs_baseline(all_data: List[Dict]) -> Dict:
    """Compare memory usage between TTT and baseline runs."""
    ttt_runs = [d for d in all_data if d['ttt_enabled']]
    baseline_runs = [d for d in all_data if not d['ttt_enabled']]
    
    comparison = {
        'ttt': {
            'count': len(ttt_runs),
            'peak_memory': [],
            'allocated_memory': [],
            'memory_overhead': []
        },
        'baseline': {
            'count': len(baseline_runs),
            'peak_memory': [],
            'allocated_memory': [],
            'memory_overhead': []
        }
    }
    
    for run in ttt_runs:
        if run['peak_memory']:
            comparison['ttt']['peak_memory'].extend(run['peak_memory'])
            comparison['ttt']['allocated_memory'].extend(run['allocated_memory'])
            comparison['ttt']['memory_overhead'].extend(
                np.array(run['peak_memory']) - np.array(run['allocated_memory'])
            )
    
    for run in baseline_runs:
        if run['peak_memory']:
            comparison['baseline']['peak_memory'].extend(run['peak_memory'])
            comparison['baseline']['allocated_memory'].extend(run['allocated_memory'])
            comparison['baseline']['memory_overhead'].extend(
                np.array(run['peak_memory']) - np.array(run['allocated_memory'])
            )
    
    # Calculate statistics
    for key in ['ttt', 'baseline']:
        if comparison[key]['peak_memory']:
            comparison[key]['stats'] = {
                'peak_mean': np.mean(comparison[key]['peak_memory']),
                'peak_std': np.std(comparison[key]['peak_memory']),
                'alloc_mean': np.mean(comparison[key]['allocated_memory']),
                'alloc_std': np.std(comparison[key]['allocated_memory']),
                'overhead_mean': np.mean(comparison[key]['memory_overhead']),
                'overhead_std': np.std(comparison[key]['memory_overhead'])
            }
    
    return comparison


def plot_memory_timeline(data: List[Dict], output_path: Path):
    """Plot memory usage over time for multiple runs."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Separate TTT and baseline
    ttt_runs = [d for d in data if d['ttt_enabled'] and d['steps']]
    baseline_runs = [d for d in data if not d['ttt_enabled'] and d['steps']]
    
    # Plot 1: Peak Memory - TTT runs
    ax1.set_title(f'Peak Memory Usage - TTT Runs ({len(ttt_runs)} runs)', fontsize=14, fontweight='bold')
    for run in ttt_runs[:10]:  # Limit to 10 for readability
        label = f"Job {run['job_id']} ({run['num_ttt_layers']} layers)"
        ax1.plot(run['steps'], run['peak_memory'], '-o', alpha=0.7, markersize=3, label=label)
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Peak Memory (GB)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)
    
    # Plot 2: Allocated Memory - TTT runs
    ax2.set_title(f'Allocated Memory - TTT Runs ({len(ttt_runs)} runs)', fontsize=14, fontweight='bold')
    for run in ttt_runs[:10]:
        label = f"Job {run['job_id']}"
        ax2.plot(run['steps'], run['allocated_memory'], '-o', alpha=0.7, markersize=3, label=label)
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Allocated Memory (GB)')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)
    
    # Plot 3: Memory Overhead (Peak - Allocated)
    ax3.set_title('Memory Overhead Analysis', fontsize=14, fontweight='bold')
    for run in ttt_runs[:10]:
        overhead = np.array(run['peak_memory']) - np.array(run['allocated_memory'])
        label = f"Job {run['job_id']}"
        ax3.plot(run['steps'], overhead, '-o', alpha=0.7, markersize=3, label=label)
    ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Training Step')
    ax3.set_ylabel('Memory Overhead (GB)')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=8)
    
    # Plot 4: Memory efficiency (Allocated / Peak ratio)
    ax4.set_title('Memory Efficiency (Allocated/Peak Ratio)', fontsize=14, fontweight='bold')
    for run in ttt_runs[:10]:
        efficiency = np.array(run['allocated_memory']) / np.array(run['peak_memory']) * 100
        label = f"Job {run['job_id']}"
        ax4.plot(run['steps'], efficiency, '-o', alpha=0.7, markersize=3, label=label)
    ax4.axhline(y=100, color='green', linestyle='--', alpha=0.5, label='Perfect efficiency')
    ax4.set_xlabel('Training Step')
    ax4.set_ylabel('Efficiency (%)')
    ax4.set_ylim([0, 100])
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Saved memory timeline: {output_path}")


def plot_memory_distribution(comparison: Dict, output_path: Path):
    """Plot memory distribution comparison."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Peak Memory Distribution
    ax1.set_title('Peak Memory Distribution: TTT vs Baseline', fontsize=14, fontweight='bold')
    if comparison['ttt']['peak_memory']:
        ax1.hist(comparison['ttt']['peak_memory'], bins=50, alpha=0.6, label='TTT', color='blue')
    if comparison['baseline']['peak_memory']:
        ax1.hist(comparison['baseline']['peak_memory'], bins=50, alpha=0.6, label='Baseline', color='red')
    ax1.set_xlabel('Peak Memory (GB)')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Allocated Memory Distribution
    ax2.set_title('Allocated Memory Distribution: TTT vs Baseline', fontsize=14, fontweight='bold')
    if comparison['ttt']['allocated_memory']:
        ax2.hist(comparison['ttt']['allocated_memory'], bins=50, alpha=0.6, label='TTT', color='blue')
    if comparison['baseline']['allocated_memory']:
        ax2.hist(comparison['baseline']['allocated_memory'], bins=50, alpha=0.6, label='Baseline', color='red')
    ax2.set_xlabel('Allocated Memory (GB)')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Memory Overhead Distribution
    ax3.set_title('Memory Overhead Distribution: TTT vs Baseline', fontsize=14, fontweight='bold')
    if comparison['ttt']['memory_overhead']:
        ax3.hist(comparison['ttt']['memory_overhead'], bins=50, alpha=0.6, label='TTT', color='blue')
    if comparison['baseline']['memory_overhead']:
        ax3.hist(comparison['baseline']['memory_overhead'], bins=50, alpha=0.6, label='Baseline', color='red')
    ax3.set_xlabel('Memory Overhead (GB)')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Statistics Comparison
    ax4.axis('off')
    stats_text = "MEMORY STATISTICS COMPARISON\n" + "="*50 + "\n\n"
    
    if comparison['ttt'].get('stats'):
        ttt_stats = comparison['ttt']['stats']
        stats_text += f"TTT Runs (n={comparison['ttt']['count']}):\n"
        stats_text += f"  Peak Memory: {ttt_stats['peak_mean']:.2f} ± {ttt_stats['peak_std']:.2f} GB\n"
        stats_text += f"  Allocated Memory: {ttt_stats['alloc_mean']:.2f} ± {ttt_stats['alloc_std']:.2f} GB\n"
        stats_text += f"  Overhead: {ttt_stats['overhead_mean']:.2f} ± {ttt_stats['overhead_std']:.2f} GB\n"
        stats_text += f"  Efficiency: {ttt_stats['alloc_mean']/ttt_stats['peak_mean']*100:.1f}%\n\n"
    
    if comparison['baseline'].get('stats'):
        base_stats = comparison['baseline']['stats']
        stats_text += f"Baseline Runs (n={comparison['baseline']['count']}):\n"
        stats_text += f"  Peak Memory: {base_stats['peak_mean']:.2f} ± {base_stats['peak_std']:.2f} GB\n"
        stats_text += f"  Allocated Memory: {base_stats['alloc_mean']:.2f} ± {base_stats['alloc_std']:.2f} GB\n"
        stats_text += f"  Overhead: {base_stats['overhead_mean']:.2f} ± {base_stats['overhead_std']:.2f} GB\n"
        stats_text += f"  Efficiency: {base_stats['alloc_mean']/base_stats['peak_mean']*100:.1f}%\n\n"
    
    if comparison['ttt'].get('stats') and comparison['baseline'].get('stats'):
        ttt_stats = comparison['ttt']['stats']
        base_stats = comparison['baseline']['stats']
        peak_diff = ttt_stats['peak_mean'] - base_stats['peak_mean']
        alloc_diff = ttt_stats['alloc_mean'] - base_stats['alloc_mean']
        overhead_diff = ttt_stats['overhead_mean'] - base_stats['overhead_mean']
        
        stats_text += f"Difference (TTT - Baseline):\n"
        stats_text += f"  Peak Memory: {peak_diff:+.2f} GB ({peak_diff/base_stats['peak_mean']*100:+.1f}%)\n"
        stats_text += f"  Allocated Memory: {alloc_diff:+.2f} GB ({alloc_diff/base_stats['alloc_mean']*100:+.1f}%)\n"
        stats_text += f"  Overhead: {overhead_diff:+.2f} GB\n\n"
        
        stats_text += f"💡 Key Insights:\n"
        if overhead_diff > 1.0:
            stats_text += f"  ⚠️ TTT has {overhead_diff:.1f} GB more overhead\n"
            stats_text += f"  → Opportunity for memory optimization!\n"
        else:
            stats_text += f"  ✅ TTT overhead is similar to baseline\n"
        
        efficiency_ttt = ttt_stats['alloc_mean']/ttt_stats['peak_mean']*100
        efficiency_base = base_stats['alloc_mean']/base_stats['peak_mean']*100
        if efficiency_ttt < efficiency_base - 5:
            stats_text += f"  ⚠️ TTT efficiency is {efficiency_base - efficiency_ttt:.1f}% lower\n"
            stats_text += f"  → Memory fragmentation or peaks during backward pass\n"
    
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, 
             fontsize=10, family='monospace', va='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Saved memory distribution: {output_path}")


def analyze_memory_peaks(data: List[Dict], output_path: Path):
    """Analyze when memory peaks occur."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    ttt_runs = [d for d in data if d['ttt_enabled'] and d['steps']]
    
    # Collect peak locations
    peak_steps = []
    peak_values = []
    
    for run in ttt_runs:
        if not run['peak_memory']:
            continue
        
        peak_mem = np.array(run['peak_memory'])
        steps = np.array(run['steps'])
        
        # Find local maxima
        for i in range(1, len(peak_mem) - 1):
            if peak_mem[i] > peak_mem[i-1] and peak_mem[i] > peak_mem[i+1]:
                peak_steps.append(steps[i])
                peak_values.append(peak_mem[i])
    
    # Plot 1: Peak occurrence histogram
    if peak_steps:
        ax1.hist(peak_steps, bins=20, alpha=0.7, color='red', edgecolor='black')
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('Number of Peak Occurrences')
        ax1.set_title('When Do Memory Peaks Occur?', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add median line
        median_peak_step = np.median(peak_steps)
        ax1.axvline(median_peak_step, color='blue', linestyle='--', linewidth=2,
                   label=f'Median: {median_peak_step:.0f}')
        ax1.legend()
    
    # Plot 2: Peak magnitude distribution
    if peak_values:
        ax2.hist(peak_values, bins=30, alpha=0.7, color='orange', edgecolor='black')
        ax2.set_xlabel('Peak Memory (GB)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Peak Memory Values', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add statistics
        mean_peak = np.mean(peak_values)
        std_peak = np.std(peak_values)
        ax2.axvline(mean_peak, color='blue', linestyle='--', linewidth=2,
                   label=f'Mean: {mean_peak:.1f} GB')
        ax2.axvline(mean_peak + std_peak, color='red', linestyle=':', linewidth=2,
                   label=f'+1σ: {mean_peak + std_peak:.1f} GB')
        ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Saved peak analysis: {output_path}")


def generate_report(all_data: List[Dict], comparison: Dict, output_path: Path):
    """Generate a comprehensive memory analysis report."""
    report = []
    report.append("=" * 80)
    report.append("MEMORY PATTERN ANALYSIS REPORT")
    report.append("=" * 80)
    report.append("")
    
    # Summary
    ttt_count = len([d for d in all_data if d['ttt_enabled']])
    baseline_count = len([d for d in all_data if not d['ttt_enabled']])
    
    report.append(f"Total Runs Analyzed: {len(all_data)}")
    report.append(f"  TTT Runs: {ttt_count}")
    report.append(f"  Baseline Runs: {baseline_count}")
    report.append("")
    
    # TTT Statistics
    if comparison['ttt'].get('stats'):
        stats = comparison['ttt']['stats']
        report.append("TTT Memory Statistics:")
        report.append(f"  Peak Memory: {stats['peak_mean']:.2f} ± {stats['peak_std']:.2f} GB")
        report.append(f"  Allocated Memory: {stats['alloc_mean']:.2f} ± {stats['alloc_std']:.2f} GB")
        report.append(f"  Memory Overhead: {stats['overhead_mean']:.2f} ± {stats['overhead_std']:.2f} GB")
        report.append(f"  Memory Efficiency: {stats['alloc_mean']/stats['peak_mean']*100:.1f}%")
        report.append("")
    
    # Baseline Statistics
    if comparison['baseline'].get('stats'):
        stats = comparison['baseline']['stats']
        report.append("Baseline Memory Statistics:")
        report.append(f"  Peak Memory: {stats['peak_mean']:.2f} ± {stats['peak_std']:.2f} GB")
        report.append(f"  Allocated Memory: {stats['alloc_mean']:.2f} ± {stats['alloc_std']:.2f} GB")
        report.append(f"  Memory Overhead: {stats['overhead_mean']:.2f} ± {stats['overhead_std']:.2f} GB")
        report.append(f"  Memory Efficiency: {stats['alloc_mean']/stats['peak_mean']*100:.1f}%")
        report.append("")
    
    # Optimization Recommendations
    report.append("=" * 80)
    report.append("OPTIMIZATION RECOMMENDATIONS")
    report.append("=" * 80)
    report.append("")
    
    if comparison['ttt'].get('stats') and comparison['baseline'].get('stats'):
        ttt_stats = comparison['ttt']['stats']
        base_stats = comparison['baseline']['stats']
        overhead_diff = ttt_stats['overhead_mean'] - base_stats['overhead_mean']
        
        if overhead_diff > 1.0:
            report.append(f"1. HIGH PRIORITY: Reduce TTT memory overhead ({overhead_diff:.1f} GB excess)")
            report.append("   Strategies:")
            report.append("   - Implement gradient checkpointing for TTT layers")
            report.append("   - Use memory-efficient attention in TTT")
            report.append("   - Clear TTT state caches more aggressively")
            report.append("")
        
        efficiency_diff = (base_stats['alloc_mean']/base_stats['peak_mean'] - 
                          ttt_stats['alloc_mean']/ttt_stats['peak_mean']) * 100
        
        if efficiency_diff > 5:
            report.append(f"2. MEDIUM PRIORITY: Improve memory efficiency ({efficiency_diff:.1f}% gap)")
            report.append("   Strategies:")
            report.append("   - Profile backward pass for TTT layers")
            report.append("   - Optimize intermediate tensor allocations")
            report.append("   - Consider mixed precision for TTT computations")
            report.append("")
        
        report.append("3. GENERAL OPTIMIZATIONS:")
        report.append("   - Monitor memory peaks during backward pass")
        report.append("   - Implement dynamic batch size adjustment")
        report.append("   - Use memory-mapped datasets if applicable")
        report.append("   - Clear CUDA cache at regular intervals")
    
    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"📄 Saved analysis report: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze memory patterns from training logs')
    parser.add_argument('log_directory', type=str, 
                       help='Directory containing .log files')
    parser.add_argument('--output', type=str, default='memory_analysis',
                       help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    try:
        # Setup paths
        log_dir = Path(args.log_directory)
        output_dir = Path(args.output)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Find all log files
        log_files = list(log_dir.glob("*.log"))
        print(f"🔍 Found {len(log_files)} log files")
        
        if not log_files:
            print("❌ No log files found!")
            return 1
        
        # Parse all log files
        print("📊 Parsing log files for memory data...")
        all_data = []
        for log_file in log_files:
            print(f"  Processing {log_file.name}...")
            memory_data = parse_log_file_for_memory(log_file)
            if memory_data['steps']:
                all_data.append(memory_data)
        
        print(f"✅ Successfully parsed {len(all_data)} logs with memory data")
        
        if not all_data:
            print("❌ No valid memory data found!")
            return 1
        
        # Compare TTT vs Baseline
        print("📊 Comparing TTT vs Baseline...")
        comparison = compare_ttt_vs_baseline(all_data)
        
        # Generate plots
        print("📊 Generating visualization plots...")
        plot_memory_timeline(all_data, output_dir / 'memory_timeline.png')
        plot_memory_distribution(comparison, output_dir / 'memory_distribution.png')
        analyze_memory_peaks(all_data, output_dir / 'memory_peaks.png')
        
        # Generate report
        print("📄 Generating analysis report...")
        generate_report(all_data, comparison, output_dir / 'memory_analysis_report.txt')
        
        # Save raw data
        analysis_data = {
            'comparison': comparison,
            'run_count': len(all_data),
            'ttt_count': len([d for d in all_data if d['ttt_enabled']]),
            'baseline_count': len([d for d in all_data if not d['ttt_enabled']])
        }
        
        with open(output_dir / 'memory_analysis_data.json', 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            def convert_to_json_serializable(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_to_json_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_json_serializable(item) for item in obj]
                return obj
            
            json.dump(convert_to_json_serializable(analysis_data), f, indent=2)
        
        print(f"\n✅ Memory analysis complete!")
        print(f"📁 Results saved to: {output_dir}/")
        print(f"   - memory_timeline.png")
        print(f"   - memory_distribution.png")
        print(f"   - memory_peaks.png")
        print(f"   - memory_analysis_report.txt")
        print(f"   - memory_analysis_data.json")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
