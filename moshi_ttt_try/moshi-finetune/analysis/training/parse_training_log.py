#!/usr/bin/env python3
"""
Training Log Parser for Moshi TTT Logs

This script parses training log files and extracts:
- Complete training configuration 
- TTT layer configuration
- Paper metrics results (sBLIMP, sWUGGY, tStoryCloze, sStoryCloze)
- LibriLight results at all positions (1k-41k)
- Training statistics and metadata

Usage:
    python parse_training_log.py moshi_ttt.7007179.log --output results.json
"""

import re
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import ast


class TrainingLogParser:
    """Parser for Moshi TTT training logs."""
    
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_content = self._load_log()
        
    def _load_log(self) -> str:
        """Load log file content."""
        try:
            with open(self.log_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            raise ValueError(f"Failed to load log file {self.log_path}: {e}")
    
    def parse_complete_log(self) -> Dict[str, Any]:
        """Parse complete log and extract all information."""
        print(f"📄 Parsing log file: {self.log_path.name}")
        
        result = {
            "metadata": self._extract_metadata(),
            "training_config": self._extract_training_config(),
            "ttt_config": self._extract_ttt_config(),
            "training_progress": self._extract_training_progress(),
            "paper_metrics": self._extract_paper_metrics(),
            "librilight_results": self._extract_librilight_results(),
            "system_info": self._extract_system_info(),
            "parsing_info": {
                "parsed_at": datetime.now().isoformat(),
                "log_file": str(self.log_path),
                "log_size_bytes": len(self.log_content)
            }
        }
        
        print(f"✅ Successfully parsed {len(self.log_content)} characters")
        return result
    
    def _extract_metadata(self) -> Dict[str, Any]:
        """Extract basic metadata."""
        metadata = {}
        
        # Extract SLURM job ID
        slurm_match = re.search(r'SLURM_JOB_ID: (\d+)', self.log_content)
        if slurm_match:
            metadata['slurm_job_id'] = slurm_match.group(1)
            
        # Extract hostname
        hostname_match = re.search(r'Hostname: (.+)', self.log_content)
        if hostname_match:
            metadata['hostname'] = hostname_match.group(1).strip()
            
        # Extract start time
        start_time_match = re.search(r'🚀 Starting Moshi TTT Training\n.*?(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', self.log_content, re.DOTALL)
        if start_time_match:
            metadata['start_time'] = start_time_match.group(1)
            
        # Extract config file
        config_match = re.search(r'📄 Config file: (.+)', self.log_content)
        if config_match:
            metadata['config_file'] = config_match.group(1).strip()
            
        # Determine model type
        if 'TTT' in self.log_content:
            metadata['model_type'] = 'TTT'
        else:
            metadata['model_type'] = 'Baseline'
            
        # Extract timestamp from filename or content
        timestamp_match = re.search(r'moshi_ttt\.(\d+)\.log', str(self.log_path))
        if timestamp_match:
            metadata['timestamp'] = timestamp_match.group(1)
        
        return metadata
    
    def _extract_training_config(self) -> Dict[str, Any]:
        """Extract complete training configuration."""
        config = {}
        
        # Look for TrainArgs section
        train_args_match = re.search(r'TrainArgs: ({.*?})\n', self.log_content, re.DOTALL)
        if train_args_match:
            try:
                # Clean up the config string
                config_str = train_args_match.group(1)
                # Fix common formatting issues
                config_str = config_str.replace("'", '"')
                config_str = re.sub(r'(\w+):', r'"\1":', config_str)
                config_str = re.sub(r': None', ': null', config_str)
                config_str = re.sub(r': True', ': true', config_str)
                config_str = re.sub(r': False', ': false', config_str)
                
                # Use ast.literal_eval for safer parsing
                train_args_text = train_args_match.group(1)
                try:
                    config = ast.literal_eval(train_args_text)
                except:
                    print("⚠️ Could not parse TrainArgs as literal, extracting key values manually")
                    config = self._manual_config_extraction()
            except Exception as e:
                print(f"⚠️ Error parsing TrainArgs: {e}")
                config = self._manual_config_extraction()
        
        return config
    
    def _manual_config_extraction(self) -> Dict[str, Any]:
        """Manually extract key training configuration values."""
        config = {}
        
        # Extract key values manually
        patterns = {
            'batch_size': r"'batch_size': (\d+)",
            'max_steps': r"'max_steps': (\d+)",
            'lr': r"'lr': ([\d.e-]+)",
            'weight_decay': r"'weight_decay': ([\d.]+)",
            'gradient_checkpointing': r"'gradient_checkpointing': (True|False)",
            'first_codebook_weight_multiplier': r"'first_codebook_weight_multiplier': ([\d.]+)",
            'full_finetuning': r"'full_finetuning': (True|False)"
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, self.log_content)
            if match:
                value = match.group(1)
                if value in ['True', 'False']:
                    config[key] = value == 'True'
                elif '.' in value or 'e' in value:
                    config[key] = float(value)
                else:
                    config[key] = int(value)
        
        # Extract paper metrics config
        paper_metrics_section = re.search(r"'paper_metrics': {(.*?)}", self.log_content, re.DOTALL)
        if paper_metrics_section:
            config['paper_metrics'] = {}
            pm_patterns = {
                'librilight_max_files': r"'librilight_max_files': (\d+)",
                'sblimp_max_pairs': r"'sblimp_max_pairs': (\d+)",
                'swuggy_max_pairs': r"'swuggy_max_pairs': (\d+)",
                'paper_metrics_use_silence': r"'paper_metrics_use_silence': (True|False)",
                'paper_metrics_use_user_stream': r"'paper_metrics_use_user_stream': (True|False)"
            }
            
            for key, pattern in pm_patterns.items():
                match = re.search(pattern, self.log_content)
                if match:
                    value = match.group(1)
                    if value in ['True', 'False']:
                        config['paper_metrics'][key] = value == 'True'
                    else:
                        config['paper_metrics'][key] = int(value)
        
        return config
    
    def _extract_ttt_config(self) -> Dict[str, Any]:
        """Extract TTT-specific configuration."""
        ttt_config = {}
        
        # TTT layers
        layers_match = re.search(r'Applying TTT to (\d+) layers: \[([^\]]+)\]', self.log_content)
        if layers_match:
            ttt_config['num_layers'] = int(layers_match.group(1))
            layer_list = [int(x.strip()) for x in layers_match.group(2).split(',')]
            ttt_config['layer_indices'] = layer_list
        
        # TTT configuration details
        ttt_details_match = re.search(r'TTT config: dim=(\d+), heads=(\d+), lr=([\d.]+)', self.log_content)
        if ttt_details_match:
            ttt_config['dim'] = int(ttt_details_match.group(1))
            ttt_config['heads'] = int(ttt_details_match.group(2))
            ttt_config['ttt_lr'] = float(ttt_details_match.group(3))
        
        # TTT gating
        gating_match = re.search(r'TTT gating: initial_alpha=([\d.]+)', self.log_content)
        if gating_match:
            ttt_config['initial_alpha'] = float(gating_match.group(1))
        
        # TTT parameters
        params_match = re.search(r'TTT parameters: ([\d,]+)', self.log_content)
        if params_match:
            ttt_config['parameters'] = int(params_match.group(1).replace(',', ''))
        
        # State persistence
        if 'TTT state persistence: ENABLED' in self.log_content:
            ttt_config['state_persistence'] = True
        elif 'TTT state persistence: DISABLED' in self.log_content:
            ttt_config['state_persistence'] = False
        
        return ttt_config
    
    def _extract_training_progress(self) -> Dict[str, Any]:
        """Extract training progress information."""
        progress = {}
        
        # Find all step logs
        step_pattern = r'step: (\d+) - done \(%\): ([\d.]+) - loss: ([\d.]+) - lr: ([\d.e-]+)'
        steps = re.findall(step_pattern, self.log_content)
        
        if steps:
            progress['total_steps'] = len(steps)
            progress['final_step'] = int(steps[-1][0])
            progress['final_loss'] = float(steps[-1][2])
            progress['final_lr'] = float(steps[-1][3])
            
            # Extract step history (sample every 100 steps for large logs)
            step_history = []
            for i, (step, pct, loss, lr) in enumerate(steps):
                if i % max(1, len(steps) // 100) == 0 or i == len(steps) - 1:
                    step_history.append({
                        'step': int(step),
                        'loss': float(loss),
                        'lr': float(lr),
                        'progress_pct': float(pct)
                    })
            progress['step_history'] = step_history
        
        # Extract training duration
        duration_match = re.search(r'Training completed.*?(\d+:\d+:\d+)', self.log_content)
        if duration_match:
            progress['training_duration'] = duration_match.group(1)
        
        return progress
    
    def _extract_paper_metrics(self) -> Dict[str, Any]:
        """Extract paper metrics evaluation results."""
        metrics = {}
        
        # sBLIMP results
        sblimp_match = re.search(r'sBLIMP: (\d+)/(\d+) = ([\d.]+)', self.log_content)
        if sblimp_match:
            correct = int(sblimp_match.group(1))
            total = int(sblimp_match.group(2))
            accuracy = float(sblimp_match.group(3))
            metrics['sblimp'] = {
                'correct': correct,
                'total': total,
                'accuracy': accuracy
            }
        
        # sWUGGY results
        swuggy_match = re.search(r'sWUGGY: (\d+)/(\d+) = ([\d.]+)', self.log_content)
        if swuggy_match:
            correct = int(swuggy_match.group(1))
            total = int(swuggy_match.group(2))
            accuracy = float(swuggy_match.group(3))
            metrics['swuggy'] = {
                'correct': correct,
                'total': total,
                'accuracy': accuracy
            }
        
        # tStoryCloze results
        tstory_match = re.search(r'tStoryCloze: (\d+)/(\d+) = ([\d.]+)', self.log_content)
        if tstory_match:
            correct = int(tstory_match.group(1))
            total = int(tstory_match.group(2))
            accuracy = float(tstory_match.group(3))
            metrics['tstory'] = {
                'correct': correct,
                'total': total,
                'accuracy': accuracy
            }
        
        # sStoryCloze results
        sstory_match = re.search(r'sStoryCloze: (\d+)/(\d+) = ([\d.]+)', self.log_content)
        if sstory_match:
            correct = int(sstory_match.group(1))
            total = int(sstory_match.group(2))
            accuracy = float(sstory_match.group(3))
            metrics['sstory'] = {
                'correct': correct,
                'total': total,
                'accuracy': accuracy
            }
        
        # Overall metrics
        if metrics:
            accuracies = [m['accuracy'] for m in metrics.values()]
            metrics['aggregate'] = {
                'paper_metrics_avg': sum(accuracies) / len(accuracies),
                'num_benchmarks': len(accuracies)
            }
        
        return metrics
    
    def _extract_librilight_results(self) -> Dict[str, Any]:
        """Extract LibriLight evaluation results at all positions."""
        librilight = {}
        
        # Extract key position results (8k, 16k, 24k)
        key_results_match = re.search(r'LibriLight results - 8k: ([\d.]+), 16k: ([\d.]+), 24k: ([\d.]+), slope: ([-\d.e]+)', self.log_content)
        if key_results_match:
            librilight['librilight_loss_8k'] = float(key_results_match.group(1))
            librilight['librilight_loss_16k'] = float(key_results_match.group(2))
            librilight['librilight_loss_24k'] = float(key_results_match.group(3))
            librilight['librilight_slope'] = float(key_results_match.group(4))
        
        # Extract all position results (1k, 2k, 3k, ... 41k)
        position_pattern = r'Position (\d+)000/\d+: avg_loss=([\d.]+)'
        positions = re.findall(position_pattern, self.log_content)
        
        if positions:
            print(f"📊 Found LibriLight results for {len(positions)} positions")
            for pos_str, loss_str in positions:
                pos = int(pos_str)
                loss = float(loss_str)
                librilight[f'librilight_loss_{pos}k'] = loss
        
        # Extract final comprehensive results from the end of log
        final_results_pattern = r'🎯 LibriLight evaluation completed: ({.*?})'
        final_match = re.search(final_results_pattern, self.log_content, re.DOTALL)
        if final_match:
            try:
                # Clean up the result string for parsing
                result_str = final_match.group(1)
                result_str = result_str.replace("'", '"')
                
                # Manual parsing for the comprehensive results
                individual_results = re.findall(r"'librilight_loss_(\d+)': ([\d.]+)", result_str)
                for pos, loss in individual_results:
                    librilight[f'librilight_loss_{pos}'] = float(loss)
                
                # Extract other metrics
                slope_match = re.search(r"'librilight_slope': ([-\d.e]+)", result_str)
                if slope_match:
                    librilight['librilight_slope'] = float(slope_match.group(1))
                
                samples_match = re.search(r"'librilight_samples': (\d+)", result_str)
                if samples_match:
                    librilight['librilight_samples'] = int(samples_match.group(1))
                    
            except Exception as e:
                print(f"⚠️ Error parsing final LibriLight results: {e}")
        
        # Extract books evaluated
        books_match = re.search(r"LibriLight books evaluated: \[([^\]]+)\]", self.log_content)
        if books_match:
            books_str = books_match.group(1)
            books = [book.strip().strip("'\"") for book in books_str.split(',')]
            librilight['books_evaluated'] = books
            librilight['num_books'] = len(books)
        
        return librilight
    
    def _extract_system_info(self) -> Dict[str, Any]:
        """Extract system and environment information."""
        system_info = {}
        
        # GPU information
        gpu_match = re.search(r'GPU 0: (.+)', self.log_content)
        if gpu_match:
            system_info['gpu'] = gpu_match.group(1).strip()
        
        # CUDA version
        cuda_match = re.search(r'CUDA Version: ([\d.]+)', self.log_content)
        if cuda_match:
            system_info['cuda_version'] = cuda_match.group(1)
        
        # Python version
        python_match = re.search(r'Python version: (.+)', self.log_content)
        if python_match:
            system_info['python_version'] = python_match.group(1).strip()
        
        # PyTorch version
        torch_match = re.search(r'PyTorch version: (.+)', self.log_content)
        if torch_match:
            system_info['pytorch_version'] = torch_match.group(1).strip()
        
        # Conda environment
        conda_match = re.search(r'Active conda environment: (.+)', self.log_content)
        if conda_match:
            system_info['conda_env'] = conda_match.group(1).strip()
        
        return system_info


def main():
    parser = argparse.ArgumentParser(description='Parse Moshi TTT training logs into JSON')
    parser.add_argument('log_file', type=str, help='Path to the training log file')
    parser.add_argument('--output', type=str, help='Output JSON file path')
    parser.add_argument('--pretty', action='store_true', help='Pretty print JSON output')
    
    args = parser.parse_args()
    
    try:
        # Parse the log
        log_parser = TrainingLogParser(Path(args.log_file))
        results = log_parser.parse_complete_log()
        
        # Determine output path
        if args.output:
            output_path = Path(args.output)
        else:
            log_path = Path(args.log_file)
            output_path = log_path.parent / f"{log_path.stem}_parsed.json"
        
        # Save results
        with open(output_path, 'w') as f:
            if args.pretty:
                json.dump(results, f, indent=2, ensure_ascii=False)
            else:
                json.dump(results, f, ensure_ascii=False)
        
        print(f"💾 Saved parsed results to: {output_path}")
        
        # Print summary
        print("\n📋 PARSING SUMMARY:")
        print(f"   Model Type: {results['metadata'].get('model_type', 'Unknown')}")
        print(f"   Job ID: {results['metadata'].get('slurm_job_id', 'Unknown')}")
        
        if 'ttt_config' in results and results['ttt_config']:
            ttt = results['ttt_config']
            print(f"   TTT Layers: {ttt.get('num_layers', 0)} layers {ttt.get('layer_indices', [])}")
            print(f"   TTT Parameters: {ttt.get('parameters', 0):,}")
        
        if 'training_progress' in results and results['training_progress']:
            progress = results['training_progress']
            print(f"   Training Steps: {progress.get('final_step', 0)}")
            print(f"   Final Loss: {progress.get('final_loss', 0.0):.3f}")
        
        if 'paper_metrics' in results and results['paper_metrics']:
            metrics = results['paper_metrics']
            if 'aggregate' in metrics:
                print(f"   Average Accuracy: {metrics['aggregate']['paper_metrics_avg']:.1%}")
        
        if 'librilight_results' in results and results['librilight_results']:
            libri = results['librilight_results']
            positions = [k for k in libri.keys() if k.startswith('librilight_loss_') and 'k' in k]
            print(f"   LibriLight Positions: {len(positions)} measured")
            if 'librilight_loss_16k' in libri:
                print(f"   LibriLight 16k Loss: {libri['librilight_loss_16k']:.3f}")
        
        print(f"   Log Size: {results['parsing_info']['log_size_bytes']:,} bytes")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())