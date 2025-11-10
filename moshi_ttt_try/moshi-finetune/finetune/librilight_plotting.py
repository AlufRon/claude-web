#!/usr/bin/env python3
"""
LibriLight Evaluation Plotting Module
Automatically generates visualization plots for LibriLight evaluation results.
"""

import matplotlib.pyplot as plt
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)


def create_librilight_plots(
    results: Dict[str, float],
    position_losses: List[float],
    measurement_positions: List[int],
    model_type: str = "Unknown",
    output_dir: str = "/home/alufr/ttt_tests/moshi-finetune/evaluation_plots/librilight",
    timestamp: Optional[str] = None
) -> Tuple[str, str]:
    """
    Create comprehensive LibriLight evaluation plots.
    
    Args:
        results: Dictionary with LibriLight metrics (loss_8k, loss_16k, etc.)
        position_losses: List of losses at each position
        measurement_positions: List of measurement positions (1k, 2k, etc.)
        model_type: Type of model being evaluated ("Frozen", "TTT", etc.)
        output_dir: Directory to save plots
        timestamp: Optional timestamp string for filename
        
    Returns:
        Tuple of (main_plot_path, detailed_plot_path)
    """
    try:
        # Set up timestamp
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🎨 Creating LibriLight plots for {model_type} model...")
        logger.info(f"📁 Output directory: {output_path}")
        
        # Set up plot style
        plt.style.use('default')
        
        # Create main comparison plot
        main_plot_file = create_main_comparison_plot(
            results, position_losses, measurement_positions, model_type, 
            output_path, timestamp
        )
        
        # Create detailed position plot
        detailed_plot_file = create_detailed_position_plot(
            results, position_losses, measurement_positions, model_type,
            output_path, timestamp
        )
        
        logger.info(f"✅ LibriLight plots created successfully:")
        logger.info(f"   📊 Main plot: {main_plot_file}")
        logger.info(f"   📈 Detailed plot: {detailed_plot_file}")
        
        return str(main_plot_file), str(detailed_plot_file)
        
    except Exception as e:
        logger.error(f"❌ Error creating LibriLight plots: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None, None


def create_main_comparison_plot(
    results: Dict[str, float],
    position_losses: List[float],
    measurement_positions: List[int],
    model_type: str,
    output_path: Path,
    timestamp: str
) -> Path:
    """Create the main 4-panel comparison plot."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'LibriLight Evaluation: {model_type} Model Results', fontsize=16, fontweight='bold')
    
    # Extract key metrics
    loss_8k = results.get('librilight_loss_8k', float('nan'))
    loss_16k = results.get('librilight_loss_16k', float('nan'))
    loss_24k = results.get('librilight_loss_24k', float('nan'))
    slope = results.get('librilight_slope', float('nan'))
    samples = results.get('librilight_samples', 0)
    
    # Plot 1: Current Results
    positions = np.array(measurement_positions)
    if len(position_losses) >= len(positions):
        current_losses = np.array([position_losses[pos-1] if pos <= len(position_losses) else float('nan') 
                                  for pos in positions])
    else:
        # Handle case where we have fewer losses than expected positions
        current_losses = np.array(position_losses + [float('nan')] * (len(positions) - len(position_losses)))
    
    # Check if we have valid data
    has_valid_data = not np.all(np.isnan(current_losses[:len(position_losses)]))
    
    if has_valid_data:
        ax1.plot(positions[:len(position_losses)], position_losses, 'b-o', linewidth=2, markersize=6, 
                label=f'{model_type} Results')
        status_text = f"✅ WORKING\n{len(position_losses)} positions\nSlope: {slope:.6f}"
        status_color = 'green'
    else:
        ax1.axhline(y=float('nan'), color='red', linestyle='--', linewidth=3, label='NaN values')
        ax1.fill_between([0, max(positions)], [0, 0], [5, 5], color='red', alpha=0.2)
        status_text = f"❌ BROKEN\nAll NaN values\nEvaluation failed"
        status_color = 'red'
    
    ax1.text(np.mean(positions), 2.5 if has_valid_data else 2.5, status_text,
             fontsize=12, ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor=status_color, alpha=0.3))
    
    ax1.set_xlim(0, max(positions) * 1.1)
    ax1.set_ylim(1.5, 3.0) 
    ax1.set_xlabel('Token Position')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{model_type} Model Evaluation')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Expected Behavior Comparison
    if model_type.lower() == "frozen":
        # Show flat line for frozen model
        expected_losses = np.full_like(positions[:len(position_losses)], np.mean(position_losses) if position_losses else 2.3026)
        ax2.plot(positions[:len(position_losses)], position_losses, 'b-o', linewidth=2, markersize=6, label='Frozen Baseline')
        ax2.axhline(y=np.mean(position_losses) if position_losses else 2.3026, color='blue', linestyle=':', alpha=0.7, label='Expected Flat')
        comparison_text = "📊 BASELINE ESTABLISHED\nNo adaptation expected\nFlat learning curve"
    else:
        # Show TTT expected improvement
        baseline = 2.3026
        ttt_improvement = baseline - 0.1 * np.log(positions / 1000) / np.log(24)
        ax2.plot(positions, [baseline] * len(positions), 'b-o', linewidth=2, alpha=0.7, label='Frozen Baseline')
        ax2.plot(positions[:len(position_losses)], position_losses, 'g-s', linewidth=3, markersize=8, label=f'{model_type} Actual')
        ax2.fill_between(positions[:len(position_losses)], [baseline] * len(position_losses), position_losses, 
                        color='green', alpha=0.2, label='TTT Effect')
        comparison_text = f"🎯 TTT EVALUATION\nAdaptive learning\nShould improve over time"
    
    ax2.text(np.mean(positions), 2.15, comparison_text,
             fontsize=12, ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor='orange', alpha=0.3))
    
    ax2.set_xlim(0, max(positions) * 1.1)
    ax2.set_ylim(2.1, 2.35)
    ax2.set_xlabel('Token Position')
    ax2.set_ylabel('Loss')
    ax2.set_title('Expected vs Actual Performance')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Learning Curve Analysis
    if has_valid_data and len(position_losses) > 1:
        # Calculate slope and trend
        x_vals = np.array(range(len(position_losses)))
        z = np.polyfit(x_vals, position_losses, 1)
        trend_line = np.poly1d(z)
        
        ax3.plot(positions[:len(position_losses)], position_losses, 'b-o', linewidth=2, markersize=6, label='Actual Losses')
        ax3.plot(positions[:len(position_losses)], trend_line(x_vals), 'r--', linewidth=2, label=f'Trend (slope={slope:.6f})')
        
        # Analyze trend
        if slope < -0.001:
            trend_text = "📈 IMPROVING\nNegative slope\nAdaptive learning"
            trend_color = 'green'
        elif abs(slope) <= 0.001:
            trend_text = "📊 STABLE\nFlat performance\nNo adaptation"
            trend_color = 'blue'
        else:
            trend_text = "📉 DEGRADING\nPositive slope\nPossible overfitting"
            trend_color = 'orange'
    else:
        trend_text = "❌ NO DATA\nCannot analyze\ntrend"
        trend_color = 'red'
    
    ax3.text(np.mean(positions), 2.2, trend_text,
             fontsize=12, ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor=trend_color, alpha=0.3))
    
    ax3.set_xlim(0, max(positions) * 1.1)
    ax3.set_ylim(2.1, 2.35)
    ax3.set_xlabel('Token Position')
    ax3.set_ylabel('Loss')
    ax3.set_title('Learning Curve Analysis')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: Metrics Summary
    ax4.axis('off')
    
    summary_text = f"""
📊 LIBRILIGHT EVALUATION SUMMARY

🔍 MODEL: {model_type}
📅 Timestamp: {timestamp}

📈 KEY METRICS:
   • Loss at 8k:  {loss_8k:.4f}
   • Loss at 16k: {loss_16k:.4f}  
   • Loss at 24k: {loss_24k:.4f}
   • Slope:       {slope:.6f}
   • Samples:     {samples}

📊 EVALUATION QUALITY:
   • Positions measured: {len(position_losses)}
   • Valid data points: {np.sum(~np.isnan(position_losses)) if position_losses else 0}
   • Numerical stability: {'✅ Stable' if has_valid_data else '❌ NaN values'}

🎯 INTERPRETATION:
   • Learning trend: {'Improving' if slope < -0.001 else 'Stable' if abs(slope) <= 0.001 else 'Degrading'}
   • Context utilization: {'Good' if has_valid_data else 'Failed'}
   • TTT effectiveness: {'TBD' if model_type.lower() == 'frozen' else 'Measured'}
"""
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
    
    ax4.set_title('Evaluation Summary', fontsize=14, fontweight='bold')
    
    # Save plot
    plt.tight_layout()
    main_plot_file = output_path / f"librilight_evaluation_{model_type.lower()}_{timestamp}.png"
    plt.savefig(main_plot_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return main_plot_file


def create_detailed_position_plot(
    results: Dict[str, float],
    position_losses: List[float],
    measurement_positions: List[int],
    model_type: str,
    output_path: Path,
    timestamp: str
) -> Path:
    """Create detailed position analysis plot."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Detailed LibriLight Position Analysis: {model_type} Model', fontsize=16, fontweight='bold')
    
    positions = np.array(measurement_positions[:len(position_losses)])
    losses = np.array(position_losses)
    
    # Plot 1: All measured positions
    ax1.plot(positions, losses, 'bo-', linewidth=2, markersize=4, label=f'{model_type} Results')
    
    if len(losses) > 0:
        mean_loss = np.mean(losses)
        ax1.axhline(y=mean_loss, color='blue', linestyle='--', alpha=0.5, label=f'Mean Loss ({mean_loss:.4f})')
        
        # Highlight key positions
        key_positions = [8000, 16000, 24000]
        for pos in key_positions:
            if pos in positions:
                idx = np.where(positions == pos)[0][0]
                ax1.annotate(f'{pos//1000}k', xy=(pos, losses[idx]), xytext=(pos, losses[idx] + 0.01),
                            arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                            ha='center', fontweight='bold', color='red')
    
    ax1.set_xlabel('Token Position')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{model_type}: All {len(losses)} Measurement Positions')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Position-wise analysis
    if len(losses) > 1:
        # Calculate differences from baseline
        baseline = losses[0] if len(losses) > 0 else 2.3026
        improvements = baseline - losses
        
        ax2.bar(positions, improvements, alpha=0.7, color='green' if np.mean(improvements) > 0 else 'red',
                label='Improvement from Start')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5, label='No Change')
        
        # Add trend analysis
        if len(improvements) > 2:
            z = np.polyfit(range(len(improvements)), improvements, 1)
            trend_line = np.poly1d(z)(range(len(improvements)))
            ax2.plot(positions, trend_line, 'r--', linewidth=2, label=f'Trend (slope={z[0]:.6f})')
    
    ax2.set_xlabel('Token Position')
    ax2.set_ylabel('Improvement from Start')
    ax2.set_title('Position-wise Improvement Analysis')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    # Save detailed plot
    detailed_plot_file = output_path / f"librilight_detailed_{model_type.lower()}_{timestamp}.png"
    plt.savefig(detailed_plot_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return detailed_plot_file


def extract_position_losses_from_results(
    results: Dict[str, Any],
    max_positions: int = 24000
) -> Tuple[List[float], List[int]]:
    """
    Extract position-specific losses from results dictionary.
    
    Args:
        results: Dictionary containing librilight results
        max_positions: Maximum position to extract
        
    Returns:
        Tuple of (position_losses, measurement_positions)
    """
    position_losses = []
    measurement_positions = []
    
    # Extract position-specific metrics (every 1k tokens)
    for pos in range(1000, max_positions + 1, 1000):
        key = f"librilight_loss_{pos}"
        if key in results:
            loss_value = results[key]
            if not (np.isnan(loss_value) or np.isinf(loss_value)):
                position_losses.append(float(loss_value))
                measurement_positions.append(pos)
    
    # Fallback to main metrics if no position-specific data
    if not position_losses:
        for pos, key in [(8000, 'librilight_loss_8k'), (16000, 'librilight_loss_16k'), (24000, 'librilight_loss_24k')]:
            if key in results:
                loss_value = results[key]
                if not (np.isnan(loss_value) or np.isinf(loss_value)):
                    position_losses.append(float(loss_value))
                    measurement_positions.append(pos)
    
    return position_losses, measurement_positions


def determine_model_type(results: Dict[str, Any], config: Dict[str, Any] = None) -> str:
    """
    Determine the model type from results and config.
    
    Args:
        results: Evaluation results
        config: Model configuration
        
    Returns:
        String describing the model type
    """
    # Check if TTT is enabled in config
    if config:
        if config.get('ttt_enabled', False) or config.get('use_ttt', False):
            return "TTT"
        elif config.get('frozen_model', True):
            return "Frozen"
    
    # Infer from results - flat slope suggests frozen model
    slope = results.get('librilight_slope', 0)
    if abs(slope) < 1e-6:
        return "Frozen"
    elif slope < -0.001:
        return "TTT"
    else:
        return "Unknown"