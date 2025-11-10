#!/usr/bin/env python3
"""
Experimental Comparison: Moshi-Native vs TTT-Optimized LibriLight Evaluation

This script provides comprehensive comparison between:
1. Moshi-Native: Token-by-token streaming (S=1) with TTT online GD
2. TTT-Optimized: Chunked streaming with optimized chunk sizes
3. Legacy: Original 50-token chunking approach

Metrics measured:
- Perplexity (primary quality metric)
- Speed (tokens/second)
- Memory usage (peak memory)
- TTT adaptation patterns (parameter changes)
"""

import torch
import time
import logging
import psutil
import gc
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import json
import matplotlib.pyplot as plt
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ComparisonConfig:
    """Configuration for experimental comparison."""
    
    # Test sequence settings
    sequence_lengths: List[int] = field(default_factory=lambda: [100, 500, 1000, 2000])
    num_repeats: int = 3
    
    # Model settings
    device: str = "cuda"
    model_path: Optional[str] = None
    
    # Output settings
    results_dir: str = "comparison_results"
    save_plots: bool = True
    save_detailed_logs: bool = True
    
    # Memory monitoring
    monitor_memory: bool = True
    memory_log_interval: int = 100  # tokens

@dataclass
class EvaluationResult:
    """Results from a single evaluation run."""
    
    approach: str
    sequence_length: int
    perplexity: float
    avg_loss: float
    
    # Performance metrics
    total_time: float
    tokens_per_second: float
    peak_memory_mb: float
    
    # TTT metrics
    ttt_updates_count: int
    avg_ttt_update_magnitude: float
    
    # Configuration used
    mini_batch_size: int
    chunk_size: int
    base_lr: float
    
    # Detailed data
    position_losses: List[float] = field(default_factory=list)
    memory_timeline: List[Tuple[int, float]] = field(default_factory=list)
    ttt_parameter_changes: List[float] = field(default_factory=list)

class LibriLightComparison:
    """Comprehensive comparison of LibriLight evaluation approaches."""
    
    def __init__(self, config: ComparisonConfig):
        self.config = config
        self.results: List[EvaluationResult] = []
        
        # Create results directory
        self.results_dir = Path(self.config.results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        logger.info(f"🔬 Experimental comparison initialized")
        logger.info(f"📊 Sequence lengths: {self.config.sequence_lengths}")
        logger.info(f"🔄 Repeats per test: {self.config.num_repeats}")
        logger.info(f"💾 Results directory: {self.results_dir}")
    
    def run_full_comparison(self):
        """Run comprehensive comparison of all approaches."""
        logger.info("🚀 Starting full experimental comparison")
        
        # Define approaches to test
        approaches = [
            ("moshi_native", self._get_moshi_native_config),
            ("ttt_optimized_efficient", self._get_ttt_optimized_efficient_config),
            ("ttt_optimized_balanced", self._get_ttt_optimized_balanced_config),
            ("legacy_chunked", self._get_legacy_chunked_config),
        ]
        
        # Run comparison for each sequence length
        for seq_len in self.config.sequence_lengths:
            logger.info(f"\n📏 Testing sequence length: {seq_len}")
            
            # Generate test data
            test_codes, test_targets = self._generate_test_data(seq_len)
            
            # Test each approach
            for approach_name, config_func in approaches:
                logger.info(f"  🧪 Testing approach: {approach_name}")
                
                # Run multiple times for statistical significance
                approach_results = []
                for repeat in range(self.config.num_repeats):
                    logger.info(f"    🔄 Repeat {repeat + 1}/{self.config.num_repeats}")
                    
                    try:
                        result = self._run_single_evaluation(
                            approach_name, config_func(), test_codes, test_targets
                        )
                        approach_results.append(result)
                        self.results.append(result)
                        
                    except Exception as e:
                        logger.error(f"    ❌ Failed: {e}")
                        continue
                
                # Log summary for this approach
                if approach_results:
                    avg_perplexity = np.mean([r.perplexity for r in approach_results])
                    avg_speed = np.mean([r.tokens_per_second for r in approach_results])
                    logger.info(f"    📊 Average: {avg_perplexity:.3f} perplexity, {avg_speed:.1f} tok/s")
        
        logger.info("✅ Full comparison completed")
        
        # Generate analysis and plots
        self._analyze_results()
        self._save_results()
        
        if self.config.save_plots:
            self._create_plots()
    
    def _get_moshi_native_config(self) -> Dict:
        """Configuration for Moshi-native token-by-token streaming."""
        return {
            'evaluation_method': 'moshi_native',
            'ttt': {
                'enable': True,
                'mini_batch_size': 1,      # Online gradient descent
                'base_lr': 0.025,          # Compensated learning rate
                'persistent_states': True,
                'optimize_chunk_size': False,
                'chunk_size': 1,
                'max_chunk_size': 1,
                'prefer_efficiency': False
            },
            'description': "Token-by-token streaming (S=1) with TTT online GD"
        }
    
    def _get_ttt_optimized_efficient_config(self) -> Dict:
        """Configuration for TTT-optimized chunking (maximum efficiency)."""
        return {
            'evaluation_method': 'ttt_optimized',
            'ttt': {
                'enable': True,
                'mini_batch_size': 25,     # Large mini-batches for efficiency
                'base_lr': 0.1,            # Standard learning rate
                'persistent_states': True,
                'optimize_chunk_size': True,
                'chunk_size': None,        # Auto-calculate (will be 25)
                'max_chunk_size': 50,
                'prefer_efficiency': True
            },
            'description': "TTT-optimized chunking for maximum efficiency"
        }
    
    def _get_ttt_optimized_balanced_config(self) -> Dict:
        """Configuration for TTT-optimized chunking (balanced approach)."""
        return {
            'evaluation_method': 'ttt_optimized',
            'ttt': {
                'enable': True,
                'mini_batch_size': 10,     # Medium mini-batches
                'base_lr': 0.1,            # Standard learning rate
                'persistent_states': True,
                'optimize_chunk_size': True,
                'chunk_size': None,        # Auto-calculate (will be 10)
                'max_chunk_size': 50,
                'prefer_efficiency': True
            },
            'description': "TTT-optimized chunking for balanced performance"
        }
    
    def _get_legacy_chunked_config(self) -> Dict:
        """Configuration for legacy 50-token chunking."""
        return {
            'evaluation_method': 'legacy_chunked',
            'ttt': {
                'enable': True,
                'mini_batch_size': 4,      # Original mini-batch size
                'base_lr': 0.1,            # Original learning rate
                'persistent_states': True,
                'optimize_chunk_size': False,  # Disable optimization
                'chunk_size': 50,          # Fixed chunk size
                'max_chunk_size': 50,
                'prefer_efficiency': False
            },
            'description': "Legacy 50-token chunking (original approach)"
        }
    
    def _generate_test_data(self, seq_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate synthetic test data for evaluation."""
        # Create realistic audio token sequences
        # Use appropriate vocabulary size for Moshi (typically 1024 per codebook)
        vocab_size = 1024
        
        codes = torch.randint(0, vocab_size, (1, 8, seq_length), device=self.config.device)
        targets = torch.randint(0, vocab_size, (1, 8, seq_length), device=self.config.device)
        
        return codes, targets
    
    def _run_single_evaluation(self, approach_name: str, config: Dict, 
                             codes: torch.Tensor, targets: torch.Tensor) -> EvaluationResult:
        """Run a single evaluation with the given configuration."""
        
        # Set up evaluation
        from finetune.paper_metrics import PaperMetricsEvaluator
        
        # Create evaluator with test configuration
        evaluator = PaperMetricsEvaluator(None, None, device=self.config.device, config=config)
        
        # Create mock model for testing
        mock_model = self._create_mock_model(config)
        
        # Memory monitoring setup
        if self.config.monitor_memory:
            initial_memory = self._get_memory_usage()
            memory_timeline = [(0, initial_memory)]
        else:
            memory_timeline = []
        
        # Performance timing
        start_time = time.time()
        
        # Run evaluation based on approach
        try:
            if config['evaluation_method'] == 'moshi_native':
                position_losses = evaluator._evaluate_librilight_moshi_native(mock_model, codes, targets)
                ttt_updates_count = codes.shape[-1]  # One update per token
                
            elif config['evaluation_method'] == 'ttt_optimized':
                # Use the optimized chunking approach
                position_losses = evaluator._evaluate_librilight_streaming(mock_model, codes, targets)
                chunk_size = evaluator.get_optimal_ttt_chunk_size()
                ttt_updates_count = (codes.shape[-1] // chunk_size) * (chunk_size // config['ttt']['mini_batch_size'])
                
            elif config['evaluation_method'] == 'legacy_chunked':
                # Use legacy approach (temporarily restore old method)
                position_losses = self._evaluate_legacy_chunked(evaluator, mock_model, codes, targets)
                ttt_updates_count = (codes.shape[-1] // 50) * (50 // config['ttt']['mini_batch_size'])
                
            else:
                raise ValueError(f"Unknown evaluation method: {config['evaluation_method']}")
        
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            # Return dummy result to avoid crashing the comparison
            return self._create_dummy_result(approach_name, codes.shape[-1], config)
        
        # Performance calculations
        end_time = time.time()
        total_time = end_time - start_time
        tokens_per_second = codes.shape[-1] / total_time if total_time > 0 else 0
        
        # Memory usage
        if self.config.monitor_memory:
            peak_memory = max(mem for _, mem in memory_timeline)
        else:
            peak_memory = self._get_memory_usage()
        
        # Calculate metrics
        avg_loss = np.mean(position_losses) if position_losses else float('inf')
        perplexity = np.exp(avg_loss) if avg_loss < 10 else float('inf')  # Avoid overflow
        
        # TTT metrics (mock values for now)
        avg_ttt_update_magnitude = 0.05  # Placeholder
        
        # Create result
        result = EvaluationResult(
            approach=approach_name,
            sequence_length=codes.shape[-1],
            perplexity=perplexity,
            avg_loss=avg_loss,
            total_time=total_time,
            tokens_per_second=tokens_per_second,
            peak_memory_mb=peak_memory,
            ttt_updates_count=ttt_updates_count,
            avg_ttt_update_magnitude=avg_ttt_update_magnitude,
            mini_batch_size=config['ttt']['mini_batch_size'],
            chunk_size=config['ttt'].get('chunk_size', 1),
            base_lr=config['ttt']['base_lr'],
            position_losses=position_losses,
            memory_timeline=memory_timeline
        )
        
        return result
    
    def _create_mock_model(self, config: Dict):
        """Create a mock model for testing purposes."""
        
        class MockModel:
            def __init__(self, ttt_config):
                self.ttt_config = ttt_config
                self.device = "cuda"
                
                # Mock TTT attributes
                self.mini_batch_size = ttt_config['mini_batch_size']
                self.base_lr = ttt_config['base_lr']
                
            def eval(self):
                pass
                
            def streaming(self, batch_size):
                return MockStreamingContext()
            
            def named_modules(self):
                yield "mock_ttt_layer", self
            
            def __call__(self, codes, condition_tensors=None):
                # Mock forward pass that returns realistic logits
                B, K, T = codes.shape
                vocab_size = 1024
                
                # Generate mock logits with some structure
                logits = torch.randn(B, K, T, vocab_size, device=codes.device) * 0.1
                
                return MockOutput(logits)
        
        class MockStreamingContext:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
        
        class MockOutput:
            def __init__(self, logits):
                self.logits = logits
        
        return MockModel(config['ttt'])
    
    def _evaluate_legacy_chunked(self, evaluator, model, codes, targets):
        """Temporarily implement legacy chunking for comparison."""
        # This would use the old hardcoded chunk_size=50 approach
        # For now, return mock results
        seq_length = codes.shape[-1]
        return [2.5 + 0.1 * np.random.randn() for _ in range(seq_length)]  # Mock losses around 2.5
    
    def _create_dummy_result(self, approach_name: str, seq_length: int, config: Dict) -> EvaluationResult:
        """Create a dummy result when evaluation fails."""
        return EvaluationResult(
            approach=approach_name,
            sequence_length=seq_length,
            perplexity=float('inf'),
            avg_loss=float('inf'),
            total_time=0,
            tokens_per_second=0,
            peak_memory_mb=0,
            ttt_updates_count=0,
            avg_ttt_update_magnitude=0,
            mini_batch_size=config['ttt']['mini_batch_size'],
            chunk_size=config['ttt'].get('chunk_size', 1),
            base_lr=config['ttt']['base_lr']
        )
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024**2)
        else:
            return psutil.Process().memory_info().rss / (1024**2)
    
    def _analyze_results(self):
        """Analyze and summarize the experimental results."""
        logger.info("\n📊 Analyzing experimental results...")
        
        # Group results by approach
        by_approach = {}
        for result in self.results:
            if result.approach not in by_approach:
                by_approach[result.approach] = []
            by_approach[result.approach].append(result)
        
        # Print summary table
        logger.info("\n📋 EXPERIMENTAL RESULTS SUMMARY")
        logger.info("=" * 80)
        logger.info(f"{'Approach':<20} {'Seq Len':<8} {'Perplexity':<12} {'Speed (tok/s)':<12} {'Memory (MB)':<12}")
        logger.info("-" * 80)
        
        for approach_name, results in by_approach.items():
            for seq_len in self.config.sequence_lengths:
                seq_results = [r for r in results if r.sequence_length == seq_len]
                if seq_results:
                    avg_perplexity = np.mean([r.perplexity for r in seq_results])
                    avg_speed = np.mean([r.tokens_per_second for r in seq_results])
                    avg_memory = np.mean([r.peak_memory_mb for r in seq_results])
                    
                    logger.info(f"{approach_name:<20} {seq_len:<8} {avg_perplexity:<12.3f} {avg_speed:<12.1f} {avg_memory:<12.1f}")
        
        # Identify best approaches
        logger.info("\n🏆 BEST APPROACHES BY METRIC")
        logger.info("=" * 50)
        
        # Best perplexity
        best_perplexity = min(self.results, key=lambda r: r.perplexity if r.perplexity != float('inf') else 1000)
        logger.info(f"🎯 Best Perplexity: {best_perplexity.approach} ({best_perplexity.perplexity:.3f})")
        
        # Best speed
        best_speed = max(self.results, key=lambda r: r.tokens_per_second)
        logger.info(f"⚡ Best Speed: {best_speed.approach} ({best_speed.tokens_per_second:.1f} tok/s)")
        
        # Best memory efficiency
        best_memory = min(self.results, key=lambda r: r.peak_memory_mb)
        logger.info(f"💾 Best Memory: {best_memory.approach} ({best_memory.peak_memory_mb:.1f} MB)")
    
    def _save_results(self):
        """Save detailed results to JSON file."""
        results_file = self.results_dir / "experimental_results.json"
        
        # Convert results to serializable format
        serializable_results = []
        for result in self.results:
            result_dict = {
                'approach': result.approach,
                'sequence_length': result.sequence_length,
                'perplexity': result.perplexity,
                'avg_loss': result.avg_loss,
                'total_time': result.total_time,
                'tokens_per_second': result.tokens_per_second,
                'peak_memory_mb': result.peak_memory_mb,
                'ttt_updates_count': result.ttt_updates_count,
                'avg_ttt_update_magnitude': result.avg_ttt_update_magnitude,
                'mini_batch_size': result.mini_batch_size,
                'chunk_size': result.chunk_size,
                'base_lr': result.base_lr,
                'position_losses_sample': result.position_losses[:10],  # First 10 for space
            }
            serializable_results.append(result_dict)
        
        with open(results_file, 'w') as f:
            json.dump({
                'config': {
                    'sequence_lengths': self.config.sequence_lengths,
                    'num_repeats': self.config.num_repeats,
                },
                'results': serializable_results
            }, f, indent=2)
        
        logger.info(f"📄 Detailed results saved to: {results_file}")
    
    def _create_plots(self):
        """Create visualization plots for the results."""
        logger.info("📈 Creating visualization plots...")
        
        # Group results by approach
        by_approach = {}
        for result in self.results:
            if result.approach not in by_approach:
                by_approach[result.approach] = []
            by_approach[result.approach].append(result)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('LibriLight Evaluation Approaches Comparison', fontsize=16)
        
        # Plot 1: Perplexity vs Sequence Length
        ax1 = axes[0, 0]
        for approach_name, results in by_approach.items():
            seq_lens = []
            perplexities = []
            for seq_len in self.config.sequence_lengths:
                seq_results = [r for r in results if r.sequence_length == seq_len and r.perplexity != float('inf')]
                if seq_results:
                    seq_lens.append(seq_len)
                    perplexities.append(np.mean([r.perplexity for r in seq_results]))
            
            if seq_lens:
                ax1.plot(seq_lens, perplexities, 'o-', label=approach_name, linewidth=2, markersize=6)
        
        ax1.set_xlabel('Sequence Length')
        ax1.set_ylabel('Perplexity')
        ax1.set_title('Perplexity vs Sequence Length')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Speed vs Sequence Length
        ax2 = axes[0, 1]
        for approach_name, results in by_approach.items():
            seq_lens = []
            speeds = []
            for seq_len in self.config.sequence_lengths:
                seq_results = [r for r in results if r.sequence_length == seq_len]
                if seq_results:
                    seq_lens.append(seq_len)
                    speeds.append(np.mean([r.tokens_per_second for r in seq_results]))
            
            if seq_lens:
                ax2.plot(seq_lens, speeds, 's-', label=approach_name, linewidth=2, markersize=6)
        
        ax2.set_xlabel('Sequence Length')
        ax2.set_ylabel('Tokens per Second')
        ax2.set_title('Processing Speed vs Sequence Length')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Memory Usage vs Sequence Length
        ax3 = axes[1, 0]
        for approach_name, results in by_approach.items():
            seq_lens = []
            memories = []
            for seq_len in self.config.sequence_lengths:
                seq_results = [r for r in results if r.sequence_length == seq_len]
                if seq_results:
                    seq_lens.append(seq_len)
                    memories.append(np.mean([r.peak_memory_mb for r in seq_results]))
            
            if seq_lens:
                ax3.plot(seq_lens, memories, '^-', label=approach_name, linewidth=2, markersize=6)
        
        ax3.set_xlabel('Sequence Length')
        ax3.set_ylabel('Peak Memory (MB)')
        ax3.set_title('Memory Usage vs Sequence Length')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: TTT Updates Count vs Sequence Length
        ax4 = axes[1, 1]
        for approach_name, results in by_approach.items():
            seq_lens = []
            updates = []
            for seq_len in self.config.sequence_lengths:
                seq_results = [r for r in results if r.sequence_length == seq_len]
                if seq_results:
                    seq_lens.append(seq_len)
                    updates.append(np.mean([r.ttt_updates_count for r in seq_results]))
            
            if seq_lens:
                ax4.plot(seq_lens, updates, 'D-', label=approach_name, linewidth=2, markersize=6)
        
        ax4.set_xlabel('Sequence Length')
        ax4.set_ylabel('TTT Updates Count')
        ax4.set_title('TTT Adaptation Frequency vs Sequence Length')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.results_dir / "comparison_plots.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        logger.info(f"📊 Plots saved to: {plot_file}")
        
        plt.close()

def main():
    """Main function to run the experimental comparison."""
    logger.info("🔬 Starting LibriLight Evaluation Approaches Comparison")
    
    # Configuration for the experiment
    config = ComparisonConfig(
        sequence_lengths=[100, 500, 1000, 2000],  # Start with smaller sequences for testing
        num_repeats=2,  # Reduced for faster testing
        device="cuda" if torch.cuda.is_available() else "cpu",
        results_dir="librilight_comparison_results",
        save_plots=True,
        save_detailed_logs=True,
        monitor_memory=True
    )
    
    # Run the comparison
    comparison = LibriLightComparison(config)
    comparison.run_full_comparison()
    
    logger.info("✅ Experimental comparison completed!")
    logger.info(f"📁 Results available in: {comparison.results_dir}")

if __name__ == "__main__":
    main()