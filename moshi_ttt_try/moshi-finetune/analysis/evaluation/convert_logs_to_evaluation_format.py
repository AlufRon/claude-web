#!/usr/bin/env python3
"""
Convert Parsed Training Logs to Evaluation Results Format

This script converts the parsed training logs into the exact same format
as the evaluation_results JSON files, making them compatible with all
existing plotting and analysis scripts.

Usage:
    python convert_logs_to_evaluation_format.py parsed_training_logs/ --output evaluation_results_from_logs/
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
import re


def extract_timestamp_from_filename(filename: str) -> str:
    """Extract SLURM job ID as timestamp from filename."""
    match = re.search(r'moshi_ttt\.(\d+)_parsed\.json', filename)
    if match:
        job_id = match.group(1)
        # Create a timestamp-like format: YYYYMMDD_HHMMSS
        # Since we don't have exact time, use job_id as unique identifier
        return f"job_{job_id}"
    return "unknown"


def convert_parsed_log_to_evaluation_format(parsed_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert parsed log data to evaluation results format."""
    
    # Extract metadata
    metadata = parsed_data.get('metadata', {})
    training_config = parsed_data.get('training_config', {})
    ttt_config = parsed_data.get('ttt_config', {})
    paper_metrics = parsed_data.get('paper_metrics', {})
    librilight_results = parsed_data.get('librilight_results', {})
    
    # Build config section in evaluation format
    config = {}
    
    # Paper metrics configuration
    if 'paper_metrics' in training_config:
        pm_config = training_config['paper_metrics']
        config.update({
            'sblimp_max_pairs': pm_config.get('sblimp_max_pairs', 2000),
            'swuggy_max_pairs': pm_config.get('swuggy_max_pairs', 2000),
            'tstory_max_pairs': pm_config.get('tstory_max_pairs', 2000),
            'sstory_max_pairs': pm_config.get('sstory_max_pairs', 2000),
            'librilight_mode': pm_config.get('librilight_evaluation_mode', 'pre_concatenated'),
            'use_user_stream': pm_config.get('paper_metrics_use_user_stream', False),
            'use_silence_codes': pm_config.get('paper_metrics_use_silence', True),
        })
    
    # Training configuration
    config['first_codebook_weight'] = training_config.get('first_codebook_weight_multiplier', 100.0)
    
    # TTT configuration
    if ttt_config:
        config['ttt_chunking'] = {
            'optimize_chunk_size': training_config.get('ttt', {}).get('optimize_chunk_size', True),
            'chunk_size_override': training_config.get('ttt', {}).get('chunk_size'),
            'max_chunk_size': training_config.get('ttt', {}).get('max_chunk_size', 50),
            'prefer_efficiency': training_config.get('ttt', {}).get('prefer_efficiency', True),
            'mini_batch_size': training_config.get('ttt', {}).get('mini_batch_size', 1)
        }
        
        # Add TTT layer information
        config['ttt_layers'] = {
            'num_layers': ttt_config.get('num_layers', 0),
            'layer_indices': ttt_config.get('layer_indices', []),
            'parameters': ttt_config.get('parameters', 0),
            'dim': ttt_config.get('dim'),
            'heads': ttt_config.get('heads'),
            'initial_alpha': ttt_config.get('initial_alpha')
        }
    
    # Streaming configuration
    config['streaming'] = {
        'enabled': True,
        'use_fixed_method': True,
        'memory_check': True,
        'cache_clear_interval': 3000,
        'max_sequence_length': 50000
    }
    
    # Training parameters
    if 'optim' in training_config:
        config['training'] = {
            'lr': training_config['optim'].get('lr'),
            'weight_decay': training_config['optim'].get('weight_decay'),
            'max_steps': training_config.get('max_steps'),
            'batch_size': training_config.get('batch_size'),
            'gradient_checkpointing': training_config.get('gradient_checkpointing')
        }
    
    # Build results section
    results = {}
    
    # Paper metrics results
    if paper_metrics:
        for benchmark in ['sblimp', 'swuggy', 'tstory', 'sstory']:
            if benchmark in paper_metrics:
                bm_data = paper_metrics[benchmark]
                results[benchmark] = {
                    f'{benchmark}_accuracy': bm_data.get('accuracy', 0.0),
                    f'{benchmark}_samples': bm_data.get('total', 2000),
                    f'{benchmark}_correct': bm_data.get('correct', 0)
                }
        
        # Aggregate metrics
        if 'aggregate' in paper_metrics:
            results['aggregate'] = {
                'paper_metrics_avg': paper_metrics['aggregate'].get('paper_metrics_avg', 0.0),
                'paper_metrics_f1': paper_metrics['aggregate'].get('paper_metrics_avg', 0.0)  # Use avg as f1 if not available
            }
    
    # LibriLight results
    if librilight_results:
        results['librilight'] = {}
        
        # Copy all librilight results
        for key, value in librilight_results.items():
            if key.startswith('librilight_'):
                results['librilight'][key] = value
            elif key in ['books_evaluated', 'num_books']:
                results['librilight'][key] = value
    
    # Build final evaluation format
    evaluation_result = {
        'metadata': {
            'timestamp': metadata.get('timestamp', extract_timestamp_from_filename(str(parsed_data.get('parsing_info', {}).get('log_file', '')))),
            'datetime': metadata.get('start_time', datetime.now().isoformat()),
            'model_type': metadata.get('model_type', 'TTT'),
            'config': config,
            'original_log_file': parsed_data.get('parsing_info', {}).get('log_file'),
            'slurm_job_id': metadata.get('slurm_job_id'),
            'hostname': metadata.get('hostname')
        },
        'results': results
    }
    
    return evaluation_result


def convert_single_file(input_path: Path, output_dir: Path) -> bool:
    """Convert a single parsed log file to evaluation format."""
    try:
        # Load parsed data
        with open(input_path, 'r') as f:
            parsed_data = json.load(f)
        
        # Convert to evaluation format
        evaluation_data = convert_parsed_log_to_evaluation_format(parsed_data)
        
        # Create output filename
        job_id = parsed_data.get('metadata', {}).get('slurm_job_id', 'unknown')
        model_type = parsed_data.get('metadata', {}).get('model_type', 'TTT').lower()
        
        # Create run directory name similar to evaluation_results
        run_dir_name = f"run_job_{job_id}_{model_type}"
        run_dir = output_dir / run_dir_name
        run_dir.mkdir(exist_ok=True)
        
        # Save results.json
        output_path = run_dir / "results.json"
        with open(output_path, 'w') as f:
            json.dump(evaluation_data, f, indent=2, ensure_ascii=False)
        
        # Create summary.txt
        summary_path = run_dir / "summary.txt"
        with open(summary_path, 'w') as f:
            f.write(f"Evaluation Results Summary\n")
            f.write(f"========================\n\n")
            f.write(f"Job ID: {job_id}\n")
            f.write(f"Model Type: {evaluation_data['metadata']['model_type']}\n")
            f.write(f"Start Time: {evaluation_data['metadata']['datetime']}\n")
            f.write(f"Original Log: {evaluation_data['metadata'].get('original_log_file', 'Unknown')}\n\n")
            
            # Paper metrics summary
            if 'aggregate' in evaluation_data['results']:
                avg_acc = evaluation_data['results']['aggregate'].get('paper_metrics_avg', 0.0)
                f.write(f"Overall Accuracy: {avg_acc:.1%}\n")
            
            # Individual benchmarks
            f.write(f"\nBenchmark Results:\n")
            for benchmark in ['sblimp', 'swuggy', 'tstory', 'sstory']:
                if benchmark in evaluation_data['results']:
                    acc = evaluation_data['results'][benchmark].get(f'{benchmark}_accuracy', 0.0)
                    f.write(f"  {benchmark.upper()}: {acc:.1%}\n")
            
            # LibriLight summary
            if 'librilight' in evaluation_data['results']:
                libri = evaluation_data['results']['librilight']
                f.write(f"\nLibriLight Results:\n")
                for pos in ['8k', '16k', '24k']:
                    key = f'librilight_loss_{pos}'
                    if key in libri:
                        f.write(f"  {pos}: {libri[key]:.3f}\n")
                
                if 'librilight_slope' in libri:
                    f.write(f"  Slope: {libri['librilight_slope']:.6f}\n")
        
        print(f"✅ Converted {input_path.name} → {run_dir_name}/")
        return True
        
    except Exception as e:
        print(f"❌ Failed to convert {input_path.name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Convert parsed logs to evaluation results format')
    parser.add_argument('input_dir', type=str, help='Directory containing parsed JSON files')
    parser.add_argument('--output', type=str, default='evaluation_results_from_logs',
                       help='Output directory for evaluation format results')
    parser.add_argument('--force', action='store_true',
                       help='Overwrite existing output files')
    
    args = parser.parse_args()
    
    try:
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            raise ValueError(f"Input directory does not exist: {input_dir}")
        
        output_dir = Path(args.output)
        output_dir.mkdir(exist_ok=True)
        
        # Find all parsed JSON files (exclude summary)
        json_files = [f for f in input_dir.glob("*_parsed.json") if not f.name.startswith('batch_')]
        
        if not json_files:
            print(f"⚠️  No parsed JSON files found in {input_dir}")
            return 1
        
        print(f"🔍 Found {len(json_files)} parsed log files to convert")
        print(f"📁 Output directory: {output_dir}")
        
        # Convert each file
        success_count = 0
        for json_file in sorted(json_files):
            if convert_single_file(json_file, output_dir):
                success_count += 1
        
        print(f"\n✅ Successfully converted {success_count}/{len(json_files)} log files")
        print(f"📊 Results saved in: {output_dir}")
        print(f"\n💡 You can now use these with your existing plotting scripts:")
        print(f"   python plot_evaluation_results.py --compare-all {output_dir}/ --output training_logs_comparison.png")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == '__main__':
    exit(main())