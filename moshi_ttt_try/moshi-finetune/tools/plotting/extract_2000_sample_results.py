#!/usr/bin/env python3
"""
Extract results from all runs that used 2000 sBLIMP pairs
Focus on paper metrics and LibriLight long-context evaluation
"""

import wandb
import pandas as pd
from datetime import datetime
import numpy as np

def extract_2000_sample_runs():
    """Extract all runs with 2000 sBLIMP pairs and their paper metrics"""
    api = wandb.Api()
    
    print("🔍 Searching for runs with 2000 sBLIMP sample size...")
    print("=" * 80)
    
    # Get all runs from the project
    runs = list(api.runs("ttt-moshi-production"))
    
    results = []
    
    for i, run in enumerate(runs):
        # Check if this run used 2000 sBLIMP samples
        config = run.config
        sblimp_max_pairs = config.get('paper_metrics', {}).get('sblimp_max_pairs', 0)
        
        if sblimp_max_pairs != 2000:
            continue  # Skip runs without 2000 sBLIMP samples
            
        print(f"\n🎯 Run {len(results) + 1}: {run.name}")
        print(f"   ID: {run.id}")
        print(f"   State: {run.state}")
        print(f"   URL: {run.url}")
        
        # Extract configuration
        ttt_enabled = config.get('ttt', {}).get('enable', False)
        ttt_layers = config.get('ttt', {}).get('layers', 'N/A')
        ttt_base_lr = config.get('ttt', {}).get('base_lr', 'N/A')
        lora_enabled = config.get('lora', {}).get('enable', False)
        max_steps = config.get('max_steps', 'N/A')
        
        # Determine experiment type
        if not ttt_enabled and not lora_enabled:
            exp_type = "Frozen"
        elif not ttt_enabled and lora_enabled:
            exp_type = "LoRA Only"
        elif ttt_enabled and lora_enabled:
            if ttt_base_lr == 0.1:
                exp_type = "TTT Aggressive"
            elif "15,31" in str(ttt_layers) or "multilayer" in run.name.lower():
                exp_type = "TTT Multi-layer"
            else:
                exp_type = "TTT Single"
        else:
            exp_type = "Other"
        
        print(f"   Type: {exp_type}")
        print(f"   TTT: {ttt_enabled} (layers: {ttt_layers}, lr: {ttt_base_lr})")
        print(f"   LoRA: {lora_enabled}")
        print(f"   Max Steps: {max_steps}")
        
        # Get paper metrics from summary
        summary = run.summary
        
        # Paper metrics
        sblimp_acc = summary.get('eval/paper_metrics/sblimp_accuracy', 0.0)
        swuggy_acc = summary.get('eval/paper_metrics/swuggy_accuracy', 0.0)
        tstory_acc = summary.get('eval/paper_metrics/tstory_accuracy', 0.0)
        sstory_acc = summary.get('eval/paper_metrics/sstory_accuracy', 0.0)
        overall_acc = summary.get('eval/paper_metrics/paper_metrics_avg', 0.0)
        
        # LibriLight metrics
        libri_8k = summary.get('eval/paper_metrics/librilight_loss_8k', 0.0)
        libri_16k = summary.get('eval/paper_metrics/librilight_loss_16k', 0.0)
        libri_24k = summary.get('eval/paper_metrics/librilight_loss_24k', 0.0)
        libri_slope = summary.get('eval/paper_metrics/librilight_slope', 0.0)
        libri_samples = summary.get('eval/paper_metrics/librilight_samples', 0)
        
        # Training metrics
        final_step = summary.get('_step', 0)
        train_loss = summary.get('train/loss', 0.0)
        eval_loss = summary.get('eval/eval_loss', 0.0)
        gating_alpha = summary.get('train/ttt_gating_alpha', 0.0)
        
        print(f"   📊 Final Step: {final_step}")
        print(f"   📈 Paper Metrics:")
        print(f"      sBLIMP: {sblimp_acc:.3f} (syntax)")
        print(f"      sWUGGY: {swuggy_acc:.3f} (lexical)")
        print(f"      tStory: {tstory_acc:.3f} (story completion)")
        print(f"      sStory: {sstory_acc:.3f} (story completion)")
        print(f"      Overall: {overall_acc:.3f}")
        print(f"   🔄 LibriLight Long Context:")
        print(f"      8k Loss: {libri_8k:.4f}")
        print(f"      16k Loss: {libri_16k:.4f}")
        print(f"      24k Loss: {libri_24k:.4f}")
        print(f"      Slope: {libri_slope:.6f} (negative = improving)")
        print(f"      Samples: {libri_samples}")
        print(f"   🎯 Training:")
        print(f"      Train Loss: {train_loss:.4f}")
        print(f"      Eval Loss: {eval_loss:.4f}" if isinstance(eval_loss, (int, float)) else f"      Eval Loss: {eval_loss}")
        if ttt_enabled:
            print(f"      Gating α: {gating_alpha:.6f}")
        
        # Store results
        results.append({
            'Run_Name': run.name,
            'Run_ID': run.id,
            'State': run.state,
            'Exp_Type': exp_type,
            'Final_Step': final_step,
            'TTT_Enabled': ttt_enabled,
            'TTT_Layers': str(ttt_layers),
            'TTT_Base_LR': ttt_base_lr,
            'LoRA_Enabled': lora_enabled,
            'Max_Steps': max_steps,
            
            # Paper metrics
            'sBLIMP_Accuracy': sblimp_acc,
            'sWUGGY_Accuracy': swuggy_acc,
            'tStory_Accuracy': tstory_acc,
            'sStory_Accuracy': sstory_acc,
            'Overall_Accuracy': overall_acc,
            
            # LibriLight long context
            'LibriLight_8k_Loss': libri_8k,
            'LibriLight_16k_Loss': libri_16k,
            'LibriLight_24k_Loss': libri_24k,
            'LibriLight_Slope': libri_slope,
            'LibriLight_Samples': libri_samples,
            
            # Training metrics
            'Train_Loss': train_loss,
            'Eval_Loss': eval_loss,
            'Gating_Alpha': gating_alpha,
            'URL': run.url
        })
    
    if not results:
        print("\n❌ No runs found with 2000 sBLIMP samples")
        return None
    
    # Create comparison DataFrame
    print(f"\n📋 PAPER METRICS COMPARISON (2000 Sample Runs)")
    print("=" * 120)
    
    df = pd.DataFrame(results)
    
    # Sort by experiment type and final step
    sort_order = {'Frozen': 0, 'LoRA Only': 1, 'TTT Single': 2, 'TTT Multi-layer': 3, 'TTT Aggressive': 4, 'Other': 5}
    df['Sort_Order'] = df['Exp_Type'].map(sort_order)
    df = df.sort_values(['Sort_Order', 'Final_Step'], ascending=[True, False])
    
    # Display key columns
    display_cols = [
        'Run_Name', 'Exp_Type', 'State', 'Final_Step',
        'sBLIMP_Accuracy', 'sWUGGY_Accuracy', 'Overall_Accuracy',
        'LibriLight_Slope', 'LibriLight_8k_Loss', 'Train_Loss'
    ]
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None) 
    pd.set_option('display.max_colwidth', 25)
    
    print(df[display_cols].to_string(index=False, float_format=lambda x: f'{x:.4f}' if abs(x) > 0.0001 else f'{x:.6f}'))
    
    # Summary statistics by experiment type
    print(f"\n📊 SUMMARY BY EXPERIMENT TYPE")
    print("=" * 60)
    
    # Group by experiment type and show averages for completed runs
    completed_df = df[df['State'] == 'finished']
    if not completed_df.empty:
        summary = completed_df.groupby('Exp_Type').agg({
            'sBLIMP_Accuracy': ['count', 'mean', 'std'],
            'sWUGGY_Accuracy': ['mean', 'std'],
            'Overall_Accuracy': ['mean', 'std'],
            'LibriLight_Slope': ['mean', 'std'],
            'LibriLight_8k_Loss': ['mean', 'std']
        }).round(4)
        
        print(summary)
    else:
        print("No completed runs found")
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"/home/alufr/ttt_tests/moshi-finetune/paper_metrics_2000samples_{timestamp}.csv"
    df.drop('Sort_Order', axis=1).to_csv(csv_path, index=False)
    print(f"\n💾 Detailed results saved to: {csv_path}")
    
    # Show best performers
    print(f"\n🏆 BEST PERFORMERS")
    print("-" * 40)
    
    if not completed_df.empty:
        # Best overall accuracy
        best_overall = completed_df.loc[completed_df['Overall_Accuracy'].idxmax()]
        print(f"🎯 Best Overall: {best_overall['Run_Name']} ({best_overall['Exp_Type']}) - {best_overall['Overall_Accuracy']:.3f}")
        
        # Best sBLIMP (syntax)
        best_sblimp = completed_df.loc[completed_df['sBLIMP_Accuracy'].idxmax()]
        print(f"📝 Best sBLIMP: {best_sblimp['Run_Name']} ({best_sblimp['Exp_Type']}) - {best_sblimp['sBLIMP_Accuracy']:.3f}")
        
        # Best sWUGGY (lexical)
        best_swuggy = completed_df.loc[completed_df['sWUGGY_Accuracy'].idxmax()]
        print(f"🔤 Best sWUGGY: {best_swuggy['Run_Name']} ({best_swuggy['Exp_Type']}) - {best_swuggy['sWUGGY_Accuracy']:.3f}")
        
        # Best LibriLight slope (most negative = most improvement with context)
        best_slope = completed_df.loc[completed_df['LibriLight_Slope'].idxmin()]
        print(f"📚 Best LibriLight: {best_slope['Run_Name']} ({best_slope['Exp_Type']}) - {best_slope['LibriLight_Slope']:.6f}")
    
    return df

if __name__ == "__main__":
    extract_2000_sample_runs()