#!/usr/bin/env python3
"""
Analyze evaluation results to compare TTT vs baseline performance
"""

import json
import pandas as pd
from pathlib import Path
import numpy as np

def load_results():
    """Load all evaluation results"""
    results_dir = Path("evaluation_results")
    all_results = []
    
    for results_file in results_dir.glob("*/results.json"):
        try:
            with open(results_file, 'r') as f:
                data = json.load(f)
            
            # Extract run info from path
            run_name = results_file.parent.name
            if "_ttt" in run_name:
                model_type = "TTT"
            elif "_baseline" in run_name:
                model_type = "Baseline"
            else:
                model_type = "Unknown"
            
            # Extract key metrics
            result = {
                'run_name': run_name,
                'model_type': model_type,
                'timestamp': run_name.split('_')[1] + "_" + run_name.split('_')[2],
            }
            
            # Add all evaluation results
            if 'results' in data:
                results_data = data['results']
                
                # Add aggregate metrics
                if 'aggregate' in results_data:
                    for key, value in results_data['aggregate'].items():
                        result[key] = value
                
                # Add task-specific metrics
                for task in ['sblimp', 'swuggy', 'tstory', 'sstory']:
                    if task in results_data:
                        for key, value in results_data[task].items():
                            result[key] = value
                
                # Add LibriLight metrics
                if 'librilight' in results_data:
                    for key, value in results_data['librilight'].items():
                        result[key] = value
            
            all_results.append(result)
            
        except Exception as e:
            print(f"Error loading {results_file}: {e}")
    
    return all_results

def create_comparison_table(results):
    """Create comparison table between TTT and baseline"""
    df = pd.DataFrame(results)
    
    # Separate TTT and baseline results
    ttt_results = df[df['model_type'] == 'TTT'].copy()
    baseline_results = df[df['model_type'] == 'Baseline'].copy()
    
    print(f"\n📊 EVALUATION RESULTS COMPARISON")
    print(f"=" * 60)
    print(f"Total runs: {len(df)}")
    print(f"TTT runs: {len(ttt_results)}")
    print(f"Baseline runs: {len(baseline_results)}")
    print()
    
    # Key metrics to compare
    key_metrics = [
        'paper_metrics_avg', 'paper_metrics_f1',
        'sblimp_accuracy', 'swuggy_accuracy', 'tstory_accuracy', 'sstory_accuracy',
        'librilight_loss_8k', 'librilight_loss_16k', 'librilight_loss_24k',
        'librilight_slope'
    ]
    
    # Create summary table
    summary_data = []
    
    for metric in key_metrics:
        if metric in df.columns:
            ttt_values = ttt_results[metric].dropna()
            baseline_values = baseline_results[metric].dropna()
            
            if len(ttt_values) > 0 and len(baseline_values) > 0:
                ttt_mean = ttt_values.mean()
                ttt_std = ttt_values.std()
                baseline_mean = baseline_values.mean()
                baseline_std = baseline_values.std()
                
                # Calculate improvement (lower is better for loss, higher is better for accuracy)
                if 'loss' in metric or 'slope' in metric:
                    improvement = ((baseline_mean - ttt_mean) / baseline_mean) * 100
                    better = "TTT" if ttt_mean < baseline_mean else "Baseline"
                else:  # accuracy metrics
                    improvement = ((ttt_mean - baseline_mean) / baseline_mean) * 100
                    better = "TTT" if ttt_mean > baseline_mean else "Baseline"
                
                summary_data.append({
                    'Metric': metric,
                    'TTT Mean': f"{ttt_mean:.4f}",
                    'TTT Std': f"{ttt_std:.4f}",
                    'Baseline Mean': f"{baseline_mean:.4f}",
                    'Baseline Std': f"{baseline_std:.4f}",
                    'Improvement %': f"{improvement:+.2f}%",
                    'Better': better
                })
    
    summary_df = pd.DataFrame(summary_data)
    
    print("📈 SUMMARY COMPARISON")
    print(summary_df.to_string(index=False))
    print()
    
    # Detailed results table
    print("📋 DETAILED RESULTS")
    print("-" * 80)
    
    # Select key columns for display
    display_cols = ['run_name', 'model_type', 'timestamp']
    for metric in key_metrics:
        if metric in df.columns:
            display_cols.append(metric)
    
    display_df = df[display_cols].copy()
    
    # Sort by model type and timestamp
    display_df = display_df.sort_values(['model_type', 'timestamp'])
    
    print(display_df.to_string(index=False))
    print()
    
    # Overall winner
    print("🏆 OVERALL PERFORMANCE")
    print("-" * 30)
    
    # Count wins for each model type
    ttt_wins = len([x for x in summary_data if x['Better'] == 'TTT'])
    baseline_wins = len([x for x in summary_data if x['Better'] == 'Baseline'])
    
    print(f"TTT wins: {ttt_wins} metrics")
    print(f"Baseline wins: {baseline_wins} metrics")
    
    if ttt_wins > baseline_wins:
        print("🎯 Winner: TTT performs better overall")
    elif baseline_wins > ttt_wins:
        print("🎯 Winner: Baseline performs better overall")
    else:
        print("🤝 Result: Tie - performance is similar")
    
    # Save results to CSV
    summary_df.to_csv('comparison_summary.csv', index=False)
    display_df.to_csv('detailed_results.csv', index=False)
    print(f"\n💾 Results saved to comparison_summary.csv and detailed_results.csv")

if __name__ == "__main__":
    results = load_results()
    create_comparison_table(results)