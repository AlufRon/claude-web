#!/usr/bin/env python3
"""
Batch Training Log Parser for Moshi TTT

This script finds and parses all training log files in a directory,
creating comprehensive JSON results for each log.

Usage:
    python batch_parse_logs.py /path/to/logs/ --output-dir parsed_logs/
    python batch_parse_logs.py . --pattern "moshi_ttt.*.log" --output-dir results/
"""

import argparse
import json
from pathlib import Path
from typing import List
import sys
import os

# Import the parser from the main script
sys.path.append(str(Path(__file__).parent))
from parse_training_log import TrainingLogParser


def find_log_files(directory: Path, pattern: str = "*.log") -> List[Path]:
    """Find all log files matching the pattern."""
    log_files = list(directory.glob(pattern))
    log_files.sort()  # Sort for consistent processing order
    return log_files


def process_single_log(log_file: Path, output_dir: Path, force: bool = False) -> bool:
    """Process a single log file and save results."""
    try:
        # Determine output filename
        output_filename = f"{log_file.stem}_parsed.json"
        output_path = output_dir / output_filename
        
        # Skip if already exists and not forcing
        if output_path.exists() and not force:
            print(f"⏭️  Skipping {log_file.name} (already parsed)")
            return True
        
        print(f"📄 Processing {log_file.name}...")
        
        # Parse the log
        parser = TrainingLogParser(log_file)
        results = parser.parse_complete_log()
        
        # Save results
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Print brief summary
        model_type = results['metadata'].get('model_type', 'Unknown')
        job_id = results['metadata'].get('slurm_job_id', 'Unknown')
        
        summary_items = []
        if 'training_progress' in results and results['training_progress']:
            final_step = results['training_progress'].get('final_step', 0)
            summary_items.append(f"steps: {final_step}")
        
        if 'paper_metrics' in results and results['paper_metrics'] and 'aggregate' in results['paper_metrics']:
            avg_acc = results['paper_metrics']['aggregate']['paper_metrics_avg']
            summary_items.append(f"acc: {avg_acc:.1%}")
        
        if 'librilight_results' in results and results['librilight_results']:
            libri = results['librilight_results']
            if 'librilight_loss_16k' in libri:
                loss_16k = libri['librilight_loss_16k']
                summary_items.append(f"16k loss: {loss_16k:.3f}")
        
        summary = ", ".join(summary_items)
        print(f"✅ {log_file.name} → {output_filename} ({model_type}, job {job_id}) [{summary}]")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to process {log_file.name}: {e}")
        return False


def create_summary_report(output_dir: Path) -> None:
    """Create a summary report of all parsed logs."""
    json_files = list(output_dir.glob("*_parsed.json"))
    if not json_files:
        print("⚠️  No parsed JSON files found for summary")
        return
    
    print(f"\n📊 Creating summary report from {len(json_files)} parsed logs...")
    
    summary_data = {
        "summary": {
            "total_logs": len(json_files),
            "parsed_at": TrainingLogParser(Path(__file__))._extract_metadata().get('start_time', 'unknown'),
            "ttt_runs": 0,
            "baseline_runs": 0
        },
        "runs": []
    }
    
    for json_file in sorted(json_files):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            model_type = data['metadata'].get('model_type', 'Unknown')
            if model_type == 'TTT':
                summary_data["summary"]["ttt_runs"] += 1
            else:
                summary_data["summary"]["baseline_runs"] += 1
            
            # Extract key metrics for summary
            run_summary = {
                "log_file": json_file.stem.replace('_parsed', ''),
                "model_type": model_type,
                "job_id": data['metadata'].get('slurm_job_id', 'Unknown'),
                "hostname": data['metadata'].get('hostname', 'Unknown'),
                "start_time": data['metadata'].get('start_time', 'Unknown')
            }
            
            # Training info
            if 'training_progress' in data and data['training_progress']:
                progress = data['training_progress']
                run_summary.update({
                    "final_step": progress.get('final_step', 0),
                    "final_loss": progress.get('final_loss', 0.0),
                    "training_duration": progress.get('training_duration', 'Unknown')
                })
            
            # TTT config
            if 'ttt_config' in data and data['ttt_config']:
                ttt = data['ttt_config']
                run_summary["ttt_config"] = {
                    "num_layers": ttt.get('num_layers', 0),
                    "layer_indices": ttt.get('layer_indices', []),
                    "parameters": ttt.get('parameters', 0)
                }
            
            # Paper metrics
            if 'paper_metrics' in data and data['paper_metrics']:
                metrics = data['paper_metrics']
                paper_metrics = {}
                
                for benchmark in ['sblimp', 'swuggy', 'tstory', 'sstory']:
                    if benchmark in metrics:
                        paper_metrics[f"{benchmark}_accuracy"] = metrics[benchmark]['accuracy']
                
                if 'aggregate' in metrics:
                    paper_metrics['average_accuracy'] = metrics['aggregate']['paper_metrics_avg']
                
                if paper_metrics:
                    run_summary["paper_metrics"] = paper_metrics
            
            # LibriLight results
            if 'librilight_results' in data and data['librilight_results']:
                libri = data['librilight_results']
                librilight_summary = {}
                
                # Key positions
                for pos in ['8k', '16k', '24k']:
                    key = f'librilight_loss_{pos}'
                    if key in libri:
                        librilight_summary[key] = libri[key]
                
                if 'librilight_slope' in libri:
                    librilight_summary['slope'] = libri['librilight_slope']
                
                if 'num_books' in libri:
                    librilight_summary['num_books'] = libri['num_books']
                
                # Count all position measurements
                position_keys = [k for k in libri.keys() if k.startswith('librilight_loss_') and ('k' in k or k.endswith('000'))]
                if position_keys:
                    librilight_summary['total_positions'] = len(position_keys)
                
                if librilight_summary:
                    run_summary["librilight"] = librilight_summary
            
            summary_data["runs"].append(run_summary)
            
        except Exception as e:
            print(f"⚠️  Could not include {json_file.name} in summary: {e}")
    
    # Save summary report
    summary_path = output_dir / "batch_parsing_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    print(f"📋 Summary report saved: {summary_path}")
    
    # Print overview
    print(f"\n📈 BATCH PROCESSING SUMMARY:")
    print(f"   Total logs processed: {summary_data['summary']['total_logs']}")
    print(f"   TTT runs: {summary_data['summary']['ttt_runs']}")
    print(f"   Baseline runs: {summary_data['summary']['baseline_runs']}")
    
    # Show best performers
    runs_with_metrics = [r for r in summary_data["runs"] if "paper_metrics" in r]
    if runs_with_metrics:
        best_avg = max(runs_with_metrics, key=lambda x: x["paper_metrics"].get("average_accuracy", 0))
        print(f"   Best average accuracy: {best_avg['paper_metrics']['average_accuracy']:.1%} ({best_avg['log_file']}, {best_avg['model_type']})")
    
    runs_with_libri = [r for r in summary_data["runs"] if "librilight" in r and "librilight_loss_16k" in r["librilight"]]
    if runs_with_libri:
        best_16k = min(runs_with_libri, key=lambda x: x["librilight"]["librilight_loss_16k"])
        print(f"   Best LibriLight 16k: {best_16k['librilight']['librilight_loss_16k']:.3f} ({best_16k['log_file']}, {best_16k['model_type']})")


def main():
    parser = argparse.ArgumentParser(description='Batch parse Moshi TTT training logs')
    parser.add_argument('directory', type=str, help='Directory containing log files')
    parser.add_argument('--output-dir', type=str, default='parsed_logs', 
                       help='Output directory for parsed JSON files')
    parser.add_argument('--pattern', type=str, default='*.log',
                       help='File pattern to match (default: *.log)')
    parser.add_argument('--force', action='store_true',
                       help='Force reprocessing of existing files')
    parser.add_argument('--no-summary', action='store_true',
                       help='Skip creating summary report')
    
    args = parser.parse_args()
    
    try:
        # Setup paths
        input_dir = Path(args.directory)
        if not input_dir.exists():
            raise ValueError(f"Input directory does not exist: {input_dir}")
        
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Find log files
        log_files = find_log_files(input_dir, args.pattern)
        if not log_files:
            print(f"⚠️  No log files found matching pattern '{args.pattern}' in {input_dir}")
            return 1
        
        print(f"🔍 Found {len(log_files)} log files to process")
        print(f"📁 Output directory: {output_dir}")
        
        # Process each log file
        success_count = 0
        for log_file in log_files:
            if process_single_log(log_file, output_dir, args.force):
                success_count += 1
        
        print(f"\n✅ Successfully processed {success_count}/{len(log_files)} log files")
        
        # Create summary report
        if not args.no_summary and success_count > 0:
            create_summary_report(output_dir)
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == '__main__':
    exit(main())