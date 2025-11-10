#!/usr/bin/env python3
"""
TTT Advantage Analysis Script

Analyzes WandB logs to compare TTT vs Baseline models using the enhanced 
position-wise loss metrics. Automatically detects TTT advantage and creates
comprehensive comparison visualizations.

Usage:
    python analyze_ttt_advantage.py --project your_wandb_project --runs run1,run2,run3
    python analyze_ttt_advantage.py --csv results.csv  # From exported WandB data
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TTTAdvantageAnalyzer:
    """Analyzes TTT vs Baseline performance using enhanced position metrics"""
    
    def __init__(self):
        self.data = None
        self.ttt_runs = []
        self.baseline_runs = []
        self.position_metrics = []
        
    def extract_position(self, metric_name):
        """Extract position number from metric name (LibriLight_8k_Loss -> 8)"""
        parts = metric_name.split('_')
        for part in parts:
            if part.endswith('k'):
                return int(part[:-1])
        return 0
        
    def load_data_from_csv(self, csv_path: str):
        """Load data from exported WandB CSV"""
        logger.info(f"Loading data from {csv_path}")
        self.data = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(self.data)} rows of data")
        
        # Detect position metrics columns (adapt to your CSV format)
        self.position_metrics = [col for col in self.data.columns 
                               if 'LibriLight_' in col and ('Loss' in col or 'loss' in col)]
        logger.info(f"Found {len(self.position_metrics)} position metrics: {self.position_metrics}")
        
    def load_data_from_wandb(self, project: str, run_names: List[str]):
        """Load data directly from WandB API"""
        try:
            import wandb
            
            logger.info(f"Loading data from WandB project: {project}")
            api = wandb.Api()
            
            all_data = []
            for run_name in run_names:
                logger.info(f"  Fetching run: {run_name}")
                run = api.run(f"{project}/{run_name}")
                
                # Get run history
                history = run.history()
                history['run_name'] = run_name
                history['model_type'] = self._detect_model_type(run_name, run.config)
                all_data.append(history)
                
            self.data = pd.concat(all_data, ignore_index=True)
            logger.info(f"Loaded {len(self.data)} total data points from {len(run_names)} runs")
            
            # Detect position metrics
            self.position_metrics = [col for col in self.data.columns 
                                   if 'librilight_loss_' in col and col.endswith('k')]
            logger.info(f"Found {len(self.position_metrics)} position metrics")
            
        except ImportError:
            logger.error("WandB not available. Use CSV export instead.")
            raise
            
    def _detect_model_type(self, run_name: str, config: dict) -> str:
        """Detect if run is TTT or Baseline based on name and config"""
        # Check run name
        if 'ttt' in run_name.lower():
            return 'TTT'
        if 'baseline' in run_name.lower():
            return 'Baseline'
            
        # Check config
        if config.get('ttt', {}).get('enable', False):
            return 'TTT'
            
        return 'Baseline'
        
    def classify_runs(self):
        """Classify runs into TTT vs Baseline"""
        if 'model_type' not in self.data.columns:
            # Try to detect from TTT_Enabled column or run names
            if 'TTT_Enabled' in self.data.columns:
                self.data['model_type'] = self.data['TTT_Enabled'].apply(
                    lambda x: 'TTT' if x else 'Baseline'
                )
            elif 'Run_Name' in self.data.columns:
                self.data['model_type'] = self.data['Run_Name'].apply(
                    lambda x: 'TTT' if 'ttt' in str(x).lower() else 'Baseline'
                )
            else:
                # Fallback to run_name column
                self.data['model_type'] = self.data.get('run_name', pd.Series()).apply(
                    lambda x: 'TTT' if 'ttt' in str(x).lower() else 'Baseline'
                )
            
        # Get run names from appropriate column
        run_col = 'Run_Name' if 'Run_Name' in self.data.columns else 'run_name'
        if run_col in self.data.columns:
            self.ttt_runs = self.data[self.data['model_type'] == 'TTT'][run_col].unique()
            self.baseline_runs = self.data[self.data['model_type'] == 'Baseline'][run_col].unique()
        else:
            self.ttt_runs = []
            self.baseline_runs = []
        
        logger.info(f"Classified runs:")
        logger.info(f"  TTT runs ({len(self.ttt_runs)}): {list(self.ttt_runs)}")
        logger.info(f"  Baseline runs ({len(self.baseline_runs)}): {list(self.baseline_runs)}")
        
    def compute_ttt_advantage(self) -> Dict[str, float]:
        """Compute TTT advantage metrics"""
        if not self.position_metrics:
            logger.warning("No position metrics found")
            return {}
            
        results = {}
        
        # Get latest values for each model type
        ttt_data = self.data[self.data['model_type'] == 'TTT']
        baseline_data = self.data[self.data['model_type'] == 'Baseline']
        
        if ttt_data.empty or baseline_data.empty:
            logger.warning("Missing TTT or Baseline data")
            return {}
            
        for metric in self.position_metrics:
            if metric in ttt_data.columns and metric in baseline_data.columns:
                # Get mean values (handling NaN)
                ttt_mean = ttt_data[metric].dropna().mean()
                baseline_mean = baseline_data[metric].dropna().mean()
                
                if not (np.isnan(ttt_mean) or np.isnan(baseline_mean)):
                    # Lower is better for loss, so improvement = baseline - ttt
                    improvement = baseline_mean - ttt_mean
                    relative_improvement = improvement / baseline_mean * 100
                    
                    results[metric] = {
                        'ttt_loss': ttt_mean,
                        'baseline_loss': baseline_mean,
                        'absolute_improvement': improvement,
                        'relative_improvement_pct': relative_improvement
                    }
                    
        return results
        
    def analyze_plateau_detection(self) -> Dict[str, int]:
        """Detect where loss improvement plateaus for each model type"""
        plateau_positions = {}
        
        for model_type in ['TTT', 'Baseline']:
            model_data = self.data[self.data['model_type'] == model_type]
            if model_data.empty:
                continue
                
            # Get position losses in order
            position_losses = []
            positions = []
            
            for metric in sorted(self.position_metrics, key=self.extract_position):
                if metric in model_data.columns:
                    loss = model_data[metric].dropna().mean()
                    if not np.isnan(loss):
                        pos_k = self.extract_position(metric)
                        positions.append(pos_k * 1000)  # Convert 8k -> 8000
                        position_losses.append(loss)
                        
            # Detect plateau (where slope becomes close to 0)
            plateau_pos = None
            if len(position_losses) >= 3:
                for i in range(1, len(position_losses) - 1):
                    # Calculate slope over window
                    slope = (position_losses[i+1] - position_losses[i-1]) / (positions[i+1] - positions[i-1])
                    if abs(slope) < 1e-5:  # Very small slope = plateau
                        plateau_pos = positions[i]
                        break
                        
            plateau_positions[model_type] = plateau_pos
            
        return plateau_positions
        
    def create_comparison_plots(self, output_dir: str = "ttt_analysis_plots"):
        """Create comprehensive comparison visualizations"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        if not self.position_metrics:
            logger.warning("No position metrics to plot")
            return
            
        # 1. Main comparison plot - Loss vs Position
        plt.figure(figsize=(14, 8))
        
        for model_type in ['TTT', 'Baseline']:
            model_data = self.data[self.data['model_type'] == model_type]
            if model_data.empty:
                continue
                
            positions = []
            losses = []
            errors = []
            
            for metric in sorted(self.position_metrics, key=self.extract_position):
                if metric in model_data.columns:
                    # Extract position number (8k -> 8)
                    pos_k = self.extract_position(metric)
                    pos = pos_k * 1000
                    loss_values = model_data[metric].dropna()
                    
                    if not loss_values.empty:
                        positions.append(pos)
                        losses.append(loss_values.mean())
                        errors.append(loss_values.std())
                        
            if positions:
                color = 'blue' if model_type == 'TTT' else 'red'
                plt.errorbar(positions, losses, yerr=errors, marker='o', linewidth=3, 
                           markersize=8, label=f'{model_type} Model', color=color, alpha=0.8)
                
        plt.xlabel('Token Position', fontsize=12)
        plt.ylabel('Cross-Entropy Loss', fontsize=12)
        plt.title('TTT vs Baseline: Loss vs Token Position\n(TTT Paper Figure 2 Equivalent)', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / 'ttt_vs_baseline_position_loss.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Advantage heatmap
        advantage_data = self.compute_ttt_advantage()
        if advantage_data:
            plt.figure(figsize=(12, 8))
            
            positions = []
            improvements = []
            
            for metric, data in advantage_data.items():
                pos = self.extract_position(metric)
                positions.append(f'{pos}k')
                improvements.append(data['relative_improvement_pct'])
                
            plt.bar(positions, improvements, color=['green' if x > 0 else 'red' for x in improvements])
            plt.xlabel('Token Position', fontsize=12)
            plt.ylabel('TTT Improvement (%)', fontsize=12)
            plt.title('TTT Advantage by Position\n(Positive = TTT Better)', fontsize=14)
            plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(output_path / 'ttt_advantage_by_position.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        # 3. Training progression
        if 'step' in self.data.columns:
            plt.figure(figsize=(14, 6))
            
            # Pick a representative metric (8k loss)
            metric = 'eval/paper_metrics/librilight_loss_8k'
            if metric not in self.data.columns:
                metric = next((m for m in self.position_metrics if '8k' in m), None)
                
            if metric:
                for model_type in ['TTT', 'Baseline']:
                    model_data = self.data[self.data['model_type'] == model_type]
                    if not model_data.empty and metric in model_data.columns:
                        plt.plot(model_data['step'], model_data[metric], 
                               label=f'{model_type} Model', linewidth=2, alpha=0.8)
                        
                plt.xlabel('Training Step', fontsize=12)
                plt.ylabel('Loss at 8k Position', fontsize=12)
                plt.title('Training Progression: 8k Token Loss', fontsize=14)
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(output_path / 'training_progression_8k.png', dpi=300, bbox_inches='tight')
                plt.close()
                
        logger.info(f"Plots saved to {output_path}")
        
    def generate_report(self, output_file: str = "ttt_analysis_report.json"):
        """Generate comprehensive analysis report"""
        report = {
            'summary': {
                'total_runs': len(self.data['run_name'].unique()) if 'run_name' in self.data.columns else 'unknown',
                'ttt_runs': len(self.ttt_runs),
                'baseline_runs': len(self.baseline_runs),
                'position_metrics_found': len(self.position_metrics)
            },
            'ttt_advantage': self.compute_ttt_advantage(),
            'plateau_analysis': self.analyze_plateau_detection(),
            'position_metrics': self.position_metrics
        }
        
        # Add interpretation
        advantage_data = report['ttt_advantage']
        if advantage_data:
            improvements = [data['relative_improvement_pct'] for data in advantage_data.values()]
            report['interpretation'] = {
                'average_improvement_pct': float(np.mean(improvements)),
                'max_improvement_pct': float(max(improvements)),
                'positions_with_advantage': int(sum(1 for x in improvements if x > 0)),
                'total_positions': int(len(improvements)),
                'ttt_shows_advantage': bool(np.mean(improvements) > 0)
            }
        else:
            report['interpretation'] = {
                'status': 'No valid comparison data found',
                'ttt_shows_advantage': False
            }
            
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"Analysis report saved to {output_file}")
        
        # Print summary
        print("\n🎯 TTT ADVANTAGE ANALYSIS SUMMARY")
        print("=" * 50)
        
        if report['interpretation'].get('ttt_shows_advantage', False):
            avg_improvement = report['interpretation']['average_improvement_pct']
            print(f"✅ TTT shows advantage: {avg_improvement:.2f}% average improvement")
            print(f"📈 Maximum improvement: {report['interpretation']['max_improvement_pct']:.2f}%")
            print(f"🎯 Positions with advantage: {report['interpretation']['positions_with_advantage']}/{report['interpretation']['total_positions']}")
        else:
            print("❌ No clear TTT advantage detected")
            print("   Possible issues: insufficient data, no difference, or evaluation problems")
            
        plateau_data = report['plateau_analysis']
        if plateau_data:
            print(f"\n📊 Plateau Analysis:")
            for model_type, position in plateau_data.items():
                if position:
                    print(f"   {model_type}: plateaus at ~{position:,} tokens")
                else:
                    print(f"   {model_type}: no plateau detected (keeps improving)")
                    
        return report

def main():
    parser = argparse.ArgumentParser(description='Analyze TTT vs Baseline performance')
    parser.add_argument('--csv', type=str, help='Path to exported WandB CSV file')
    parser.add_argument('--project', type=str, help='WandB project name')
    parser.add_argument('--runs', type=str, help='Comma-separated list of run names')
    parser.add_argument('--output-dir', type=str, default='ttt_analysis_plots', help='Output directory for plots')
    parser.add_argument('--report', type=str, default='ttt_analysis_report.json', help='Output report file')
    
    args = parser.parse_args()
    
    analyzer = TTTAdvantageAnalyzer()
    
    # Load data
    if args.csv:
        analyzer.load_data_from_csv(args.csv)
    elif args.project and args.runs:
        run_names = [r.strip() for r in args.runs.split(',')]
        analyzer.load_data_from_wandb(args.project, run_names)
    else:
        print("Error: Provide either --csv or both --project and --runs")
        return
        
    # Classify runs and analyze
    analyzer.classify_runs()
    analyzer.create_comparison_plots(args.output_dir)
    report = analyzer.generate_report(args.report)
    
    print(f"\n📁 Results saved to:")
    print(f"   Plots: {args.output_dir}/")
    print(f"   Report: {args.report}")

if __name__ == '__main__':
    main()