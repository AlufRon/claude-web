#!/usr/bin/env python3
"""
Get specific WandB run metrics by name
"""

import wandb
import pandas as pd

def get_specific_run_metrics():
    """Get metrics for our current running experiments"""
    api = wandb.Api()
    
    # Names of current running experiments
    target_runs = [
        "lora_baseline_2000_samples_1000steps",  # Baseline no TTT
        "ttt_2000_samples_1000steps",           # Single layer TTT
        "ttt_multilayer_2000_1000steps",        # Multi-layer TTT  
        "ttt_aggressive_lr_1000steps"           # Aggressive learning rate TTT
    ]
    
    print("🎯 Fetching metrics for current experiments")
    print("=" * 60)
    
    results = []
    
    for run_name in target_runs:
        print(f"\n📊 Searching for: {run_name}")
        
        try:
            # Get runs with this name
            runs = list(api.runs("ttt-moshi-production", {"display_name": run_name}))
            
            if not runs:
                print(f"   ❌ No run found with name: {run_name}")
                continue
            
            # Get the most recent run with this name  
            run = runs[0]  # Most recent
            print(f"   ✅ Found run: {run.id} (State: {run.state})")
            print(f"   🔗 URL: {run.url}")
            
            # Try to get summary metrics (latest values)
            summary = run.summary
            print(f"   📈 Summary keys available: {list(summary.keys())[:10]}...")  # Show first 10 keys
            
            # Extract latest paper metrics from summary
            latest_step = summary.get('_step', 0)
            sblimp = summary.get('eval/paper_metrics/sblimp_accuracy', 0.0)
            swuggy = summary.get('eval/paper_metrics/swuggy_accuracy', 0.0)
            tstory = summary.get('eval/paper_metrics/tstory_accuracy', 0.0)
            sstory = summary.get('eval/paper_metrics/sstory_accuracy', 0.0)
            avg_score = summary.get('eval/paper_metrics/paper_metrics_avg', 0.0)
            libri_8k = summary.get('eval/paper_metrics/librilight_loss_8k', 0.0)
            libri_16k = summary.get('eval/paper_metrics/librilight_loss_16k', 0.0)  
            libri_24k = summary.get('eval/paper_metrics/librilight_loss_24k', 0.0)
            libri_slope = summary.get('eval/paper_metrics/librilight_slope', 0.0)
            
            # Training metrics
            train_loss = summary.get('train/loss', 0.0)
            lr = summary.get('train/lr', 0.0)
            gating_alpha = summary.get('train/ttt_gating_alpha', 0.0)
            
            # Get config
            config = run.config
            ttt_enabled = config.get('ttt', {}).get('enable', False)
            ttt_layers = config.get('ttt', {}).get('layers', 'N/A')
            ttt_lr = config.get('ttt', {}).get('base_lr', 'N/A')
            
            print(f"   📊 Latest metrics (Step {latest_step}):")
            print(f"      sBLIMP: {sblimp:.3f}")
            print(f"      sWUGGY: {swuggy:.3f}")
            print(f"      tStory: {tstory:.3f}")
            print(f"      sStory: {sstory:.3f}")
            print(f"      Average: {avg_score:.3f}")
            print(f"      LibriLight 8k: {libri_8k:.4f}")
            print(f"      LibriLight 16k: {libri_16k:.4f}")
            print(f"      LibriLight 24k: {libri_24k:.4f}")
            print(f"      LibriLight Slope: {libri_slope:.6f}")
            print(f"      TTT Gating α: {gating_alpha:.6f}")
            print(f"      Train Loss: {train_loss:.4f}")
            print(f"      Learning Rate: {lr:.2e}")
            print(f"   ⚙️  Config:")
            print(f"      TTT Enabled: {ttt_enabled}")
            print(f"      TTT Layers: {ttt_layers}")
            print(f"      TTT Base LR: {ttt_lr}")
            
            # Store for comparison
            results.append({
                'Run Name': run_name,
                'State': run.state,
                'Step': latest_step,
                'TTT': ttt_enabled,
                'Layers': str(ttt_layers),
                'TTT_LR': ttt_lr,
                'sBLIMP': sblimp,
                'sWUGGY': swuggy, 
                'tStory': tstory,
                'sStory': sstory,
                'Avg': avg_score,
                'Libri_8k': libri_8k,
                'Libri_16k': libri_16k,
                'Libri_24k': libri_24k,
                'Libri_Slope': libri_slope,
                'Gating_α': gating_alpha,
                'Train_Loss': train_loss,
                'LR': lr,
                'URL': run.url
            })
            
        except Exception as e:
            print(f"   ❌ Error processing {run_name}: {e}")
    
    # Create comparison table
    if results:
        print(f"\n📋 EXPERIMENT COMPARISON")
        print("=" * 100)
        df = pd.DataFrame(results)
        
        # Sort by TTT enabled (baseline first, then TTT variants)
        df = df.sort_values(['TTT', 'TTT_LR'], ascending=[True, True])
        
        # Format for display 
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 15)
        
        # Select key columns for comparison
        comparison_cols = ['Run Name', 'State', 'Step', 'TTT', 'Layers', 'sBLIMP', 'sWUGGY', 'Avg', 'Libri_Slope', 'Gating_α']
        print(df[comparison_cols].to_string(index=False, float_format=lambda x: f'{x:.3f}' if abs(x) > 0.001 else f'{x:.6f}'))
        
        # Save detailed results
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        csv_path = f"/home/alufr/ttt_tests/moshi-finetune/current_experiments_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n💾 Detailed results saved to: {csv_path}")
        
        return df
    
    return None

if __name__ == "__main__":
    get_specific_run_metrics()