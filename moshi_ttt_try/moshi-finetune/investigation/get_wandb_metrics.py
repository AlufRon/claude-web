#!/usr/bin/env python3
"""
Extract Current Metrics from WandB Runs

Fetches the latest metrics from running TTT experiments for comparison.
"""

import wandb
import pandas as pd
from datetime import datetime
import argparse

def get_latest_metrics(project="ttt-moshi-production", limit=5):
    """
    Fetch latest metrics from WandB runs.
    
    Args:
        project: WandB project name
        limit: Number of recent runs to fetch
    """
    # Initialize WandB API
    api = wandb.Api()
    
    # Get runs from the project
    runs = api.runs(f"{project}", per_page=limit)
    
    print(f"📊 Fetching metrics from {project}")
    print("=" * 80)
    
    results = []
    
    for i, run in enumerate(runs):
        print(f"\n🏃 Run {i+1}: {run.name}")
        print(f"   ID: {run.id}")
        print(f"   State: {run.state}")
        print(f"   URL: {run.url}")
        
        # Get the run's config
        config = run.config
        ttt_enabled = config.get('ttt', {}).get('enable', False)
        ttt_layers = config.get('ttt', {}).get('layers', 'N/A')
        max_steps = config.get('max_steps', 'N/A')
        
        print(f"   TTT Enabled: {ttt_enabled}")
        print(f"   TTT Layers: {ttt_layers}")
        print(f"   Max Steps: {max_steps}")
        
        # Get latest metrics
        try:
            # Get history for paper metrics
            history = run.history(keys=[
                'step',
                'eval/paper_metrics/sblimp_accuracy',
                'eval/paper_metrics/swuggy_accuracy', 
                'eval/paper_metrics/tstory_accuracy',
                'eval/paper_metrics/sstory_accuracy',
                'eval/paper_metrics/paper_metrics_avg',
                'eval/paper_metrics/librilight_loss_8k',
                'eval/paper_metrics/librilight_loss_16k', 
                'eval/paper_metrics/librilight_loss_24k',
                'eval/paper_metrics/librilight_slope',
                'train/ttt_gating_alpha',
                'train/ttt_gating_alpha_raw',
                'train/loss',
                'train/lr'
            ])
            
            if not history.empty:
                # Get the latest row with metrics
                latest = history.dropna(subset=['eval/paper_metrics/sblimp_accuracy']).tail(1)
                
                if not latest.empty:
                    latest_row = latest.iloc[-1]
                    
                    step = int(latest_row.get('step', 0))
                    sblimp = latest_row.get('eval/paper_metrics/sblimp_accuracy', 0.0)
                    swuggy = latest_row.get('eval/paper_metrics/swuggy_accuracy', 0.0) 
                    tstory = latest_row.get('eval/paper_metrics/tstory_accuracy', 0.0)
                    sstory = latest_row.get('eval/paper_metrics/sstory_accuracy', 0.0)
                    avg_score = latest_row.get('eval/paper_metrics/paper_metrics_avg', 0.0)
                    libri_8k = latest_row.get('eval/paper_metrics/librilight_loss_8k', 0.0)
                    libri_slope = latest_row.get('eval/paper_metrics/librilight_slope', 0.0)
                    
                    # Get latest training metrics
                    train_history = run.history(keys=['train/ttt_gating_alpha', 'train/loss', 'train/lr'])
                    if not train_history.empty:
                        latest_train = train_history.tail(1).iloc[-1]
                        gating_alpha = latest_train.get('train/ttt_gating_alpha', 0.0)
                        train_loss = latest_train.get('train/loss', 0.0)
                        lr = latest_train.get('train/lr', 0.0)
                    else:
                        gating_alpha = train_loss = lr = 0.0
                    
                    print(f"   📈 Latest Metrics (Step {step}):")
                    print(f"      sBLIMP: {sblimp:.3f}")
                    print(f"      sWUGGY: {swuggy:.3f}")  
                    print(f"      tStory: {tstory:.3f}")
                    print(f"      sStory: {sstory:.3f}")
                    print(f"      Average: {avg_score:.3f}")
                    print(f"      LibriLight 8k: {libri_8k:.4f}")
                    print(f"      LibriLight Slope: {libri_slope:.6f}")
                    print(f"      TTT Gating α: {gating_alpha:.6f}")
                    print(f"      Train Loss: {train_loss:.4f}")
                    print(f"      Learning Rate: {lr:.2e}")
                    
                    # Store for comparison table
                    results.append({
                        'Run Name': run.name,
                        'State': run.state,
                        'TTT': ttt_enabled,
                        'Layers': ttt_layers,
                        'Step': step,
                        'sBLIMP': sblimp,
                        'sWUGGY': swuggy,
                        'tStory': tstory, 
                        'sStory': sstory,
                        'Avg': avg_score,
                        'Libri_8k': libri_8k,
                        'Libri_Slope': libri_slope,
                        'Gating_α': gating_alpha,
                        'Train_Loss': train_loss,
                        'ID': run.id
                    })
                else:
                    print(f"   ⚠️  No paper metrics found yet")
            else:
                print(f"   ⚠️  No history available")
                
        except Exception as e:
            print(f"   ❌ Error fetching metrics: {e}")
    
    # Create comparison table
    if results:
        print(f"\n📋 COMPARISON TABLE")
        print("=" * 120)
        df = pd.DataFrame(results)
        
        # Format for display
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 20)
        
        print(df.to_string(index=False, float_format=lambda x: f'{x:.3f}' if abs(x) > 0.001 else f'{x:.6f}'))
        
        # Save to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = f"/home/alufr/ttt_tests/moshi-finetune/wandb_metrics_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n💾 Metrics saved to: {csv_path}")
        
        return df
    
    return None

def get_specific_runs(run_names, project="ttt-moshi-production"):
    """Get metrics from specific run names."""
    api = wandb.Api()
    
    print(f"📊 Fetching metrics for specific runs from {project}")
    print("=" * 80)
    
    for run_name in run_names:
        try:
            # Find run by name
            runs = api.runs(f"{project}", {"display_name": run_name})
            if runs:
                run = runs[0]  # Get first match
                print(f"\n🎯 Found run: {run.name}")
                
                # Get latest metrics (same logic as above)
                history = run.history(keys=[
                    'step',
                    'eval/paper_metrics/sblimp_accuracy',
                    'eval/paper_metrics/swuggy_accuracy',
                    'eval/paper_metrics/paper_metrics_avg',
                    'eval/paper_metrics/librilight_slope'
                ])
                
                if not history.empty:
                    latest = history.dropna(subset=['eval/paper_metrics/sblimp_accuracy']).tail(1)
                    if not latest.empty:
                        row = latest.iloc[-1]
                        step = int(row.get('step', 0))
                        sblimp = row.get('eval/paper_metrics/sblimp_accuracy', 0.0)
                        swuggy = row.get('eval/paper_metrics/swuggy_accuracy', 0.0)
                        avg_score = row.get('eval/paper_metrics/paper_metrics_avg', 0.0)
                        slope = row.get('eval/paper_metrics/librilight_slope', 0.0)
                        
                        print(f"   Step {step}: sBLIMP={sblimp:.3f}, sWUGGY={swuggy:.3f}, Avg={avg_score:.3f}, Slope={slope:.6f}")
                
            else:
                print(f"❌ Run not found: {run_name}")
                
        except Exception as e:
            print(f"❌ Error fetching {run_name}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract WandB metrics from TTT experiments")
    parser.add_argument("--project", default="ttt-moshi-production", help="WandB project name")
    parser.add_argument("--limit", type=int, default=5, help="Number of recent runs to fetch")
    parser.add_argument("--runs", nargs="+", help="Specific run names to fetch")
    
    args = parser.parse_args()
    
    if args.runs:
        get_specific_runs(args.runs, args.project)
    else:
        get_latest_metrics(args.project, args.limit)