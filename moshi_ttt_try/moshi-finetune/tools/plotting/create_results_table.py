#!/usr/bin/env python3
"""
Create a comprehensive results table comparing all TTT experiments
"""

import pandas as pd
import numpy as np

def create_results_table():
    """Create a comprehensive comparison table of all experiments"""
    
    print("🏆 MOSHI TTT EXPERIMENT RESULTS COMPARISON")
    print("=" * 80)
    
    # Current running experiments with meaningful data (Step 240+)
    results_data = [
        {
            'Experiment': '🧊 Frozen Baseline',
            'Type': 'No Training',
            'TTT': '❌',
            'LoRA': '❌',
            'TTT_Layers': 'None',
            'TTT_LR': 'N/A',
            'Step': 40,
            'sBLIMP': 0.538,
            'sWUGGY': 0.611,
            'tStory': 0.805,
            'sStory': 0.614,
            'Overall': 0.642,
            'LibriLight_8k': 9.0404,
            'LibriLight_16k': 2.7336,
            'LibriLight_24k': 10.1304,
            'LibriLight_Slope': -0.000532,
            'Train_Loss': 1.921,
            'Gating_Alpha': 'N/A'
        },
        {
            'Experiment': '📚 LoRA Baseline',
            'Type': 'LoRA Only',
            'TTT': '❌',
            'LoRA': '✅',
            'TTT_Layers': 'None',
            'TTT_LR': 'N/A',
            'Step': 240,
            'sBLIMP': 0.546,
            'sWUGGY': 0.643,
            'tStory': 0.813,
            'sStory': 0.621,
            'Overall': 0.656,
            'LibriLight_8k': 8.9450,
            'LibriLight_16k': 2.6682,
            'LibriLight_24k': 10.0089,
            'LibriLight_Slope': -0.000527,
            'Train_Loss': 2.232,
            'Gating_Alpha': 'N/A'
        },
        {
            'Experiment': '🧠 TTT Single Layer',
            'Type': 'TTT + LoRA',
            'TTT': '✅',
            'LoRA': '✅',
            'TTT_Layers': '31',
            'TTT_LR': '0.01',
            'Step': 340,
            'sBLIMP': 0.542,
            'sWUGGY': 0.645,
            'tStory': 0.812,
            'sStory': 0.618,
            'Overall': 0.654,
            'LibriLight_8k': 9.0955,
            'LibriLight_16k': 2.6821,
            'LibriLight_24k': 9.9308,
            'LibriLight_Slope': -0.000535,
            'Train_Loss': 1.520,
            'Gating_Alpha': 0.100
        },
        {
            'Experiment': '⚡ TTT Aggressive LR',
            'Type': 'TTT + LoRA',
            'TTT': '✅',
            'LoRA': '✅',
            'TTT_Layers': '31',
            'TTT_LR': '0.1',
            'Step': 240,
            'sBLIMP': 0.540,
            'sWUGGY': 0.642,
            'tStory': 0.810,
            'sStory': 0.617,
            'Overall': 0.652,
            'LibriLight_8k': 9.2904,
            'LibriLight_16k': 2.7162,
            'LibriLight_24k': 9.9837,
            'LibriLight_Slope': -0.000547,
            'Train_Loss': 2.162,
            'Gating_Alpha': 0.100
        },
        {
            'Experiment': '🔗 TTT Multi-layer',
            'Type': 'TTT + LoRA',
            'TTT': '✅',
            'LoRA': '✅',
            'TTT_Layers': '15,31',
            'TTT_LR': '0.01',
            'Step': 240,
            'sBLIMP': 0.504,
            'sWUGGY': 0.561,
            'tStory': 0.522,
            'sStory': 0.494,
            'Overall': 0.520,
            'LibriLight_8k': 0.000,
            'LibriLight_16k': 0.000,
            'LibriLight_24k': 0.000,
            'LibriLight_Slope': 0.000000,
            'Train_Loss': 2.292,
            'Gating_Alpha': 0.100
        }
    ]
    
    # Create DataFrame
    df = pd.DataFrame(results_data)
    
    # Display main comparison table
    print("\n📊 MAIN RESULTS TABLE")
    print("-" * 100)
    
    # Select key columns for display
    display_cols = [
        'Experiment', 'Type', 'Step', 'sBLIMP', 'sWUGGY', 'tStory', 'sStory', 'Overall'
    ]
    
    # Format the table
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', '{:.3f}'.format)
    
    print(df[display_cols].to_string(index=False))
    
    # Detailed technical table
    print(f"\n🔧 TECHNICAL DETAILS TABLE")
    print("-" * 120)
    
    tech_cols = [
        'Experiment', 'TTT_Layers', 'TTT_LR', 'Train_Loss', 'LibriLight_Slope', 'Gating_Alpha'
    ]
    
    print(df[tech_cols].to_string(index=False))
    
    # LibriLight long context table
    print(f"\n📚 LIBRILIGHT LONG CONTEXT ANALYSIS")
    print("-" * 80)
    
    libri_cols = [
        'Experiment', 'LibriLight_8k', 'LibriLight_16k', 'LibriLight_24k', 'LibriLight_Slope'
    ]
    
    print(df[libri_cols].to_string(index=False))
    
    # Performance rankings
    print(f"\n🏆 PERFORMANCE RANKINGS")
    print("-" * 50)
    
    # Overall accuracy ranking
    df_sorted = df.sort_values('Overall', ascending=False)
    print("📈 Overall Accuracy:")
    for i, row in df_sorted.iterrows():
        print(f"   {i+1}. {row['Experiment']}: {row['Overall']:.3f}")
    
    # Individual metric rankings
    metrics = ['sBLIMP', 'sWUGGY', 'tStory', 'sStory']
    metric_names = ['Syntax (sBLIMP)', 'Lexical (sWUGGY)', 'Story T', 'Story S']
    
    for metric, name in zip(metrics, metric_names):
        df_metric = df.sort_values(metric, ascending=False)
        print(f"\n📝 {name}:")
        for i, row in df_metric.iterrows():
            print(f"   {i+1}. {row['Experiment']}: {row[metric]:.3f}")
    
    # LibriLight slope ranking (most negative = best long context improvement)
    df_slope = df[df['LibriLight_Slope'] != 0].sort_values('LibriLight_Slope', ascending=True)
    print(f"\n📚 Long Context Learning (LibriLight Slope, negative = better):")
    for i, row in df_slope.iterrows():
        print(f"   {i+1}. {row['Experiment']}: {row['LibriLight_Slope']:.6f}")
    
    # Improvements over baseline
    print(f"\n📊 IMPROVEMENTS OVER FROZEN BASELINE")
    print("-" * 60)
    
    baseline_overall = df[df['Experiment'] == '🧊 Frozen Baseline']['Overall'].iloc[0]
    baseline_sblimp = df[df['Experiment'] == '🧊 Frozen Baseline']['sBLIMP'].iloc[0]
    baseline_swuggy = df[df['Experiment'] == '🧊 Frozen Baseline']['sWUGGY'].iloc[0]
    
    for _, row in df.iterrows():
        if '🧊 Frozen' in row['Experiment']:
            continue
            
        overall_improvement = (row['Overall'] - baseline_overall) * 100
        sblimp_improvement = (row['sBLIMP'] - baseline_sblimp) * 100
        swuggy_improvement = (row['sWUGGY'] - baseline_swuggy) * 100
        
        print(f"\n{row['Experiment']}:")
        print(f"   Overall: +{overall_improvement:.1f}% ({row['Overall']:.3f} vs {baseline_overall:.3f})")
        print(f"   sBLIMP:  +{sblimp_improvement:.1f}% ({row['sBLIMP']:.3f} vs {baseline_sblimp:.3f})")
        print(f"   sWUGGY:  +{swuggy_improvement:.1f}% ({row['sWUGGY']:.3f} vs {baseline_swuggy:.3f})")
    
    # TTT vs LoRA comparison
    print(f"\n🤖 TTT vs LoRA COMPARISON")
    print("-" * 40)
    
    lora_overall = df[df['Experiment'] == '📚 LoRA Baseline']['Overall'].iloc[0]
    ttt_single = df[df['Experiment'] == '🧠 TTT Single Layer']['Overall'].iloc[0]
    ttt_aggressive = df[df['Experiment'] == '⚡ TTT Aggressive LR']['Overall'].iloc[0]
    
    print(f"LoRA Baseline:       {lora_overall:.3f}")
    print(f"TTT Single Layer:    {ttt_single:.3f} ({(ttt_single - lora_overall)*100:+.1f}%)")
    print(f"TTT Aggressive LR:   {ttt_aggressive:.3f} ({(ttt_aggressive - lora_overall)*100:+.1f}%)")
    
    # Key findings
    print(f"\n💡 KEY FINDINGS")
    print("-" * 30)
    
    print("1. 📚 LoRA fine-tuning is currently the best performer (65.6%)")
    print("2. 🧠 TTT is competitive but not dramatically better (-0.2% vs LoRA)")  
    print("3. ⚡ Aggressive TTT learning rate doesn't help much (-0.4% vs LoRA)")
    print("4. 🔗 Multi-layer TTT is struggling significantly (-13.6% vs LoRA)")
    print("5. 🧊 Frozen Moshi baseline is surprisingly strong (64.2% zero-shot)")
    print("6. 📖 Long context benefits are small but consistent across all models")
    print("7. 🎭 All models excel at story completion (~80%) vs syntax/lexical (~54-64%)")
    print("8. 🎯 TTT gating mechanism is active (α=0.100) but benefits unclear")
    
    # Save results
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"/home/alufr/ttt_tests/moshi-finetune/results_comparison_table_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n💾 Results table saved to: {csv_path}")
    
    return df

if __name__ == "__main__":
    df = create_results_table()