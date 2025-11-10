"""
Figure 5 Plotting: Multi-Layer TTT Loss Trajectories

Plots three loss curves per TTT layer:
- ℓ(W₀; xₜ): Loss with frozen initial weights (no learning)
- ℓ(Wₜ₋₁; xₜ): Loss before gradient descent (accumulated learning)
- ℓ(Wₜ; xₜ): Loss after gradient descent (immediate improvement)

This demonstrates:
1. Test-time learning effectiveness (gap between blue and orange/green)
2. Gradient descent benefit (gap between orange and green)
3. Adaptation over sequence (downward trend in orange/green)
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d


def smooth_losses(losses: List[float], window_size: int = 10) -> np.ndarray:
    """
    Apply sliding window average for visual clarity (as mentioned in Figure 5 caption).
    
    Args:
        losses: Raw loss values
        window_size: Sliding window size (default: 10 as per paper)
    
    Returns:
        Smoothed loss values
    """
    if len(losses) < window_size:
        return np.array(losses)
    return uniform_filter1d(losses, size=window_size, mode='nearest')


def create_figure5_single_layer(
    positions: List[int],
    W0_losses: List[float],
    Wt_prev_losses: List[float],
    Wt_losses: List[float],
    layer_id: int,
    ax: plt.Axes,
    smooth_window: int = 10
):
    """
    Create Figure 5 plot for a single TTT layer.
    
    Args:
        positions: Token positions (x-axis)
        W0_losses: ℓ(W₀; xₜ) - frozen initial weights
        Wt_prev_losses: ℓ(Wₜ₋₁; xₜ) - before gradient descent
        Wt_losses: ℓ(Wₜ; xₜ) - after gradient descent
        layer_id: TTT layer ID (e.g., 29, 30, 31)
        ax: Matplotlib axes to plot on
        smooth_window: Sliding window size for smoothing
    """
    # Smooth losses for visual clarity
    W0_smooth = smooth_losses(W0_losses, smooth_window)
    Wt_prev_smooth = smooth_losses(Wt_prev_losses, smooth_window)
    Wt_smooth = smooth_losses(Wt_losses, smooth_window)
    
    # Plot three lines (matching Figure 5 style)
    ax.plot(positions, W0_smooth, '-', color='#4472C4', linewidth=2, 
            label=r'$\ell(W_0; x_t)$', alpha=0.9)
    ax.plot(positions, Wt_prev_smooth, '-', color='#ED7D31', linewidth=2,
            label=r'$\ell(W_{t-1}; x_t)$', alpha=0.9)
    ax.plot(positions, Wt_smooth, '-', color='#70AD47', linewidth=2,
            label=r'$\ell(W_t; x_t)$', alpha=0.9)
    
    # Styling (match Figure 5 appearance)
    ax.set_xlabel('Token index t', fontsize=11)
    ax.set_ylabel('TTT loss ℓ', fontsize=11)
    ax.set_title(f'Layer {layer_id}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_xlim(0, max(positions))
    
    # Compute statistics
    final_W0 = W0_smooth[-1]
    final_Wt = Wt_smooth[-1]
    improvement = ((final_W0 - final_Wt) / final_W0) * 100
    
    # Add statistics annotation
    stats_text = f'Improvement: {improvement:.1f}%'
    ax.text(0.02, 0.02, stats_text,
            transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))


def create_figure5_multilayer(
    losses_by_layer: Dict[int, Dict[str, Dict[int, float]]],
    output_dir: str,
    sequence_id: str = "librilight_seq_0",
    smooth_window: int = 10,
    layer_ids: List[int] = [29, 30, 31]
) -> str:
    """
    Create Figure 5: Multi-layer TTT loss trajectories.
    
    Args:
        losses_by_layer: Nested dict structure:
            {
                layer_id: {
                    'W0': {position: loss_value},      # ℓ(W₀; xₜ)
                    'Wt_prev': {position: loss_value}, # ℓ(Wₜ₋₁; xₜ)
                    'Wt': {position: loss_value}       # ℓ(Wₜ; xₜ)
                }
            }
        output_dir: Directory to save plot
        sequence_id: Identifier for the sequence
        smooth_window: Sliding window size (default: 10 as per paper)
        layer_ids: TTT layer IDs to plot
    
    Returns:
        Path to saved plot
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with 3 subplots (one per layer)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, layer_id in enumerate(layer_ids):
        if layer_id not in losses_by_layer:
            print(f"Warning: Layer {layer_id} not found in losses_by_layer")
            continue
        
        layer_data = losses_by_layer[layer_id]
        
        # Extract positions (should be same for all three loss types)
        positions = sorted(layer_data['W0'].keys())
        
        # Extract loss values
        W0_losses = [layer_data['W0'][pos] for pos in positions]
        Wt_prev_losses = [layer_data['Wt_prev'][pos] for pos in positions]
        Wt_losses = [layer_data['Wt'][pos] for pos in positions]
        
        # Create subplot for this layer
        create_figure5_single_layer(
            positions=positions,
            W0_losses=W0_losses,
            Wt_prev_losses=Wt_prev_losses,
            Wt_losses=Wt_losses,
            layer_id=layer_id,
            ax=axes[idx],
            smooth_window=smooth_window
        )
    
    # Overall title
    fig.suptitle('Figure 5: Test-Time Training Loss Trajectories (Multi-Layer TTT)',
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = f'figure5_multilayer_ttt_{sequence_id}.png'
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved Figure 5 plot: {plot_path}")
    
    # Also save as PDF for publication
    pdf_path = plot_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight', format='pdf')
    print(f"✅ Saved Figure 5 PDF: {pdf_path}")
    
    plt.close()
    
    # Save statistics as JSON
    stats = {}
    for layer_id in layer_ids:
        if layer_id not in losses_by_layer:
            continue
        
        layer_data = losses_by_layer[layer_id]
        positions = sorted(layer_data['W0'].keys())
        
        W0_losses = [layer_data['W0'][pos] for pos in positions]
        Wt_prev_losses = [layer_data['Wt_prev'][pos] for pos in positions]
        Wt_losses = [layer_data['Wt'][pos] for pos in positions]
        
        # Smooth for statistics
        W0_smooth = smooth_losses(W0_losses, smooth_window)
        Wt_prev_smooth = smooth_losses(Wt_prev_losses, smooth_window)
        Wt_smooth = smooth_losses(Wt_losses, smooth_window)
        
        stats[f'layer_{layer_id}'] = {
            'initial_W0_loss': float(W0_smooth[0]),
            'final_W0_loss': float(W0_smooth[-1]),
            'initial_Wt_loss': float(Wt_smooth[0]),
            'final_Wt_loss': float(Wt_smooth[-1]),
            'total_improvement_pct': float(((W0_smooth[-1] - Wt_smooth[-1]) / W0_smooth[-1]) * 100),
            'avg_gradient_step_improvement': float(np.mean(np.array(Wt_prev_losses) - np.array(Wt_losses))),
            'num_positions': len(positions)
        }
    
    stats_path = os.path.join(output_dir, f'figure5_stats_{sequence_id}.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"✅ Saved Figure 5 statistics: {stats_path}")
    
    return plot_path


def create_figure5_stacked_layers(
    losses_by_layer: Dict[int, Dict[str, Dict[int, float]]],
    output_dir: str,
    sequence_id: str = "librilight_seq_0",
    smooth_window: int = 10,
    layer_ids: List[int] = [29, 30, 31]
) -> str:
    """
    Alternative visualization: Stacked plots (vertical arrangement).
    Useful when you want larger individual plots.
    
    Same args as create_figure5_multilayer but with vertical stacking.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with 3 vertically stacked subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    
    for idx, layer_id in enumerate(layer_ids):
        if layer_id not in losses_by_layer:
            continue
        
        layer_data = losses_by_layer[layer_id]
        positions = sorted(layer_data['W0'].keys())
        
        W0_losses = [layer_data['W0'][pos] for pos in positions]
        Wt_prev_losses = [layer_data['Wt_prev'][pos] for pos in positions]
        Wt_losses = [layer_data['Wt'][pos] for pos in positions]
        
        create_figure5_single_layer(
            positions=positions,
            W0_losses=W0_losses,
            Wt_prev_losses=Wt_prev_losses,
            Wt_losses=Wt_losses,
            layer_id=layer_id,
            ax=axes[idx],
            smooth_window=smooth_window
        )
    
    fig.suptitle('Figure 5: Test-Time Training Loss Trajectories (Stacked)',
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    plot_filename = f'figure5_stacked_{sequence_id}.png'
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved Figure 5 stacked plot: {plot_path}")
    
    plt.close()
    return plot_path


def convert_raw_fig5_data_to_plotting_format(raw_data: Dict, max_T: int = 2048) -> Dict:
    """
    Convert raw Figure 5 data from ttt_mlp logging to plotting format.
    
    Args:
        raw_data: Dict from fig5_get() with structure:
            {layer_id: {'l0': [sums], 'lprev': [sums], 'lafter': [sums], 'cnt': [counts]}}
        max_T: Maximum position to include
    
    Returns:
        Dict with structure expected by create_figure5_multilayer():
            {layer_id: {'W0': {pos: loss}, 'Wt_prev': {pos: loss}, 'Wt': {pos: loss}}}
    """
    losses_by_layer = {}
    
    for layer_id, data in raw_data.items():
        counts = data['cnt']
        l0_sums = data['l0']
        lprev_sums = data['lprev']
        lafter_sums = data['lafter']
        
        losses_by_layer[layer_id] = {
            'W0': {},       # ℓ(W₀; xₜ)
            'Wt_prev': {},  # ℓ(Wₜ₋₁; xₜ)
            'Wt': {}        # ℓ(Wₜ; xₜ)
        }
        
        # Convert sums to averages for each position
        for pos in range(min(len(counts), max_T)):
            if counts[pos] > 0:
                losses_by_layer[layer_id]['W0'][pos] = l0_sums[pos] / counts[pos]
                losses_by_layer[layer_id]['Wt_prev'][pos] = lprev_sums[pos] / counts[pos]
                losses_by_layer[layer_id]['Wt'][pos] = lafter_sums[pos] / counts[pos]
    
    return losses_by_layer


# Example usage
if __name__ == "__main__":
    # Example data structure
    losses_by_layer = {
        29: {
            'W0': {0: 1.7, 100: 1.7, 500: 1.75, 1000: 1.75, 2000: 1.75},
            'Wt_prev': {0: 1.2, 100: 1.0, 500: 0.95, 1000: 0.9, 2000: 0.9},
            'Wt': {0: 1.0, 100: 0.9, 500: 0.88, 1000: 0.85, 2000: 0.85}
        },
        30: {
            'W0': {0: 1.0, 100: 1.0, 500: 1.0, 1000: 1.0, 2000: 1.0},
            'Wt_prev': {0: 0.8, 100: 0.7, 500: 0.65, 1000: 0.62, 2000: 0.6},
            'Wt': {0: 0.7, 100: 0.65, 500: 0.62, 1000: 0.6, 2000: 0.6}
        },
        31: {
            'W0': {0: 1.0, 100: 1.0, 500: 1.0, 1000: 1.0, 2000: 1.0},
            'Wt_prev': {0: 0.75, 100: 0.7, 500: 0.66, 1000: 0.64, 2000: 0.62},
            'Wt': {0: 0.7, 100: 0.67, 500: 0.64, 1000: 0.63, 2000: 0.62}
        }
    }
    
    # Create plots
    plot_path = create_figure5_multilayer(
        losses_by_layer,
        output_dir="./evaluation_plots/figure5",
        sequence_id="example"
    )
    print(f"Example plot saved to: {plot_path}")
