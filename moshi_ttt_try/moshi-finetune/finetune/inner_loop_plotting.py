"""
Inner Loop Loss Plotting for TTT

Implements Figure 4 from the TTT paper: visualization of reconstruction loss
convergence during the inner gradient descent loop.

This shows three key metrics:
- ℓ(W₀; xₜ): Loss with initial weights (no learning)
- ℓ(Wₜ₋₁; xₜ): Loss with weights from previous token
- ℓ(Wₜ; xₜ): Loss after gradient descent on current token

The plot demonstrates that:
1. Gradient descent reduces reconstruction loss
2. The loss improves as more tokens are processed (test-time learning)
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np


def create_inner_loop_loss_plot(
    losses_per_position: Dict[int, List[float]],
    output_dir: str,
    sequence_id: Optional[str] = None,
    show_individual_curves: bool = True,
    show_statistics: bool = True,
) -> str:
    """
    Create Figure 4-style plot showing inner loop reconstruction loss convergence.
    
    Args:
        losses_per_position: Dict mapping token position to list of reconstruction losses
                           from each mini-batch iteration at that position
        output_dir: Directory to save the plot
        sequence_id: Optional identifier for the sequence being plotted
        show_individual_curves: If True, show individual position curves (can be noisy)
        show_statistics: If True, show mean/std statistics across positions
    
    Returns:
        Path to the saved plot file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data
    positions = sorted(losses_per_position.keys())
    
    # Figure 4 shows three key loss values:
    # - Initial loss: ℓ(W₀; xₜ) - first mini-batch loss (before any updates)
    # - Previous loss: ℓ(Wₜ₋₁; xₜ) - also first mini-batch (weights from prev token)
    # - Final loss: ℓ(Wₜ; xₜ) - last mini-batch loss (after gradient descent)
    
    initial_losses = []  # ℓ(W₀; xₜ) or ℓ(Wₜ₋₁; xₜ)
    final_losses = []    # ℓ(Wₜ; xₜ)
    
    for pos in positions:
        losses = losses_per_position[pos]
        if len(losses) > 0:
            initial_losses.append(losses[0])   # First mini-batch
            final_losses.append(losses[-1])    # Last mini-batch
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Initial vs Final Loss (main comparison)
    ax = axes[0]
    ax.plot(positions, initial_losses, 'o-', label='ℓ(Wₜ₋₁; xₜ) - Before Update', 
            color='#e74c3c', linewidth=2, markersize=4)
    ax.plot(positions, final_losses, 's-', label='ℓ(Wₜ; xₜ) - After Update', 
            color='#2ecc71', linewidth=2, markersize=4)
    
    ax.set_xlabel('Token Position', fontsize=12)
    ax.set_ylabel('Reconstruction Loss (MSE)', fontsize=12)
    ax.set_title('Inner Loop Loss Convergence (Figure 4)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add improvement annotation
    if len(initial_losses) > 0 and len(final_losses) > 0:
        avg_improvement = np.mean(np.array(initial_losses) - np.array(final_losses))
        pct_improvement = 100 * avg_improvement / np.mean(initial_losses)
        ax.text(0.02, 0.98, f'Avg. Improvement: {pct_improvement:.1f}%',
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Plot 2: Loss Reduction Per Position
    ax = axes[1]
    loss_reductions = np.array(initial_losses) - np.array(final_losses)
    ax.plot(positions, loss_reductions, 'o-', color='#3498db', linewidth=2, markersize=4)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Token Position', fontsize=12)
    ax.set_ylabel('Loss Reduction (Initial - Final)', fontsize=12)
    ax.set_title('Gradient Descent Effectiveness', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    if len(loss_reductions) > 0:
        mean_reduction = np.mean(loss_reductions)
        std_reduction = np.std(loss_reductions)
        ax.text(0.02, 0.98, f'Mean: {mean_reduction:.4f}\nStd: {std_reduction:.4f}',
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    
    # Save plot
    filename = f'inner_loop_losses_{sequence_id}.png' if sequence_id else 'inner_loop_losses.png'
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Also save raw data as JSON
    data_filename = f'inner_loop_losses_{sequence_id}.json' if sequence_id else 'inner_loop_losses.json'
    data_path = os.path.join(output_dir, data_filename)
    
    data = {
        'positions': positions,
        'initial_losses': [float(x) for x in initial_losses],
        'final_losses': [float(x) for x in final_losses],
        'loss_reductions': [float(x) for x in loss_reductions],
        'statistics': {
            'mean_initial': float(np.mean(initial_losses)),
            'mean_final': float(np.mean(final_losses)),
            'mean_reduction': float(np.mean(loss_reductions)),
            'std_reduction': float(np.std(loss_reductions)),
            'improvement_percent': float(100 * np.mean(loss_reductions) / np.mean(initial_losses)),
        }
    }
    
    with open(data_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"[Inner Loop Plotting] Saved plot to: {output_path}")
    print(f"[Inner Loop Plotting] Saved data to: {data_path}")
    print(f"[Inner Loop Plotting] Statistics:")
    print(f"  - Mean initial loss: {data['statistics']['mean_initial']:.4f}")
    print(f"  - Mean final loss: {data['statistics']['mean_final']:.4f}")
    print(f"  - Mean reduction: {data['statistics']['mean_reduction']:.4f}")
    print(f"  - Improvement: {data['statistics']['improvement_percent']:.1f}%")
    
    return output_path


def create_detailed_inner_loop_plot(
    losses_per_position: Dict[int, List[float]],
    output_dir: str,
    sequence_id: Optional[str] = None,
    max_positions_to_show: int = 20,
) -> str:
    """
    Create detailed plot showing full mini-batch iteration curves for selected positions.
    
    This shows how the loss decreases within each position's inner loop (across mini-batches).
    
    Args:
        losses_per_position: Dict mapping token position to list of reconstruction losses
        output_dir: Directory to save the plot
        sequence_id: Optional identifier for the sequence
        max_positions_to_show: Maximum number of position curves to plot (avoid clutter)
    
    Returns:
        Path to the saved plot file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    positions = sorted(losses_per_position.keys())
    
    # Select positions to show (evenly spaced)
    if len(positions) > max_positions_to_show:
        indices = np.linspace(0, len(positions) - 1, max_positions_to_show, dtype=int)
        selected_positions = [positions[i] for i in indices]
    else:
        selected_positions = positions
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot each position's inner loop convergence
    colors = plt.cm.viridis(np.linspace(0, 1, len(selected_positions)))
    
    for pos, color in zip(selected_positions, colors):
        losses = losses_per_position[pos]
        mini_batch_indices = list(range(len(losses)))
        ax.plot(mini_batch_indices, losses, 'o-', label=f'Position {pos}',
                color=color, alpha=0.7, linewidth=1.5, markersize=3)
    
    ax.set_xlabel('Mini-Batch Iteration', fontsize=12)
    ax.set_ylabel('Reconstruction Loss (MSE)', fontsize=12)
    ax.set_title('Inner Loop Convergence per Position', fontsize=14, fontweight='bold')
    ax.legend(fontsize=8, ncol=2, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    filename = f'inner_loop_detailed_{sequence_id}.png' if sequence_id else 'inner_loop_detailed.png'
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[Inner Loop Plotting] Saved detailed plot to: {output_path}")
    
    return output_path


def aggregate_inner_loop_statistics(
    all_sequences_losses: Dict[str, Dict[int, List[float]]],
    output_dir: str,
) -> Dict:
    """
    Aggregate statistics across multiple sequences and create summary plot.
    
    Args:
        all_sequences_losses: Dict mapping sequence_id to losses_per_position dict
        output_dir: Directory to save plots and statistics
    
    Returns:
        Dictionary containing aggregated statistics
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect all initial and final losses across all sequences
    all_initial_losses = []
    all_final_losses = []
    all_reductions = []
    
    for seq_id, losses_per_position in all_sequences_losses.items():
        for pos, losses in losses_per_position.items():
            if len(losses) > 0:
                initial = losses[0]
                final = losses[-1]
                all_initial_losses.append(initial)
                all_final_losses.append(final)
                all_reductions.append(initial - final)
    
    # Compute statistics
    stats = {
        'num_sequences': len(all_sequences_losses),
        'num_positions': len(all_initial_losses),
        'mean_initial_loss': float(np.mean(all_initial_losses)),
        'std_initial_loss': float(np.std(all_initial_losses)),
        'mean_final_loss': float(np.mean(all_final_losses)),
        'std_final_loss': float(np.std(all_final_losses)),
        'mean_reduction': float(np.mean(all_reductions)),
        'std_reduction': float(np.std(all_reductions)),
        'improvement_percent': float(100 * np.mean(all_reductions) / np.mean(all_initial_losses)),
    }
    
    # Create summary plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Histogram of initial losses
    axes[0].hist(all_initial_losses, bins=50, color='#e74c3c', alpha=0.7, edgecolor='black')
    axes[0].axvline(stats['mean_initial_loss'], color='black', linestyle='--', linewidth=2,
                    label=f"Mean: {stats['mean_initial_loss']:.4f}")
    axes[0].set_xlabel('Initial Loss ℓ(Wₜ₋₁; xₜ)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Distribution of Initial Losses', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Histogram of final losses
    axes[1].hist(all_final_losses, bins=50, color='#2ecc71', alpha=0.7, edgecolor='black')
    axes[1].axvline(stats['mean_final_loss'], color='black', linestyle='--', linewidth=2,
                    label=f"Mean: {stats['mean_final_loss']:.4f}")
    axes[1].set_xlabel('Final Loss ℓ(Wₜ; xₜ)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Distribution of Final Losses', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Histogram of loss reductions
    axes[2].hist(all_reductions, bins=50, color='#3498db', alpha=0.7, edgecolor='black')
    axes[2].axvline(stats['mean_reduction'], color='black', linestyle='--', linewidth=2,
                    label=f"Mean: {stats['mean_reduction']:.4f}")
    axes[2].set_xlabel('Loss Reduction (Initial - Final)', fontsize=12)
    axes[2].set_ylabel('Frequency', fontsize=12)
    axes[2].set_title('Distribution of Loss Reductions', fontsize=14, fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'inner_loop_summary.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save statistics
    stats_path = os.path.join(output_dir, 'inner_loop_statistics.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"[Inner Loop Plotting] Saved summary plot to: {output_path}")
    print(f"[Inner Loop Plotting] Saved statistics to: {stats_path}")
    print(f"[Inner Loop Plotting] Aggregate Statistics:")
    print(f"  - Sequences: {stats['num_sequences']}")
    print(f"  - Positions: {stats['num_positions']}")
    print(f"  - Mean improvement: {stats['improvement_percent']:.1f}%")
    
    return stats


def create_per_layer_plots(
    losses_per_layer: Dict[int, List[float]],
    output_dir: str,
    sequence_id: Optional[str] = None,
) -> List[str]:
    """
    Create individual plots for each TTT layer showing reconstruction loss vs token position.
    
    Args:
        losses_per_layer: Dict mapping layer_id to list of reconstruction losses
        output_dir: Directory to save the plots
        sequence_id: Optional identifier for the sequence being plotted
    
    Returns:
        List of paths to the saved plot files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plot_paths = []
    
    for layer_id, losses in losses_per_layer.items():
        if len(losses) == 0:
            continue
            
        # Create figure for this layer
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Token positions (0-indexed)
        positions = list(range(len(losses)))
        
        # Plot reconstruction loss over token positions
        ax.plot(positions, losses, 'o-', color='#2E86AB', linewidth=2, markersize=4, 
               label=f'Layer {layer_id} Reconstruction Loss')
        
        # Calculate improvement if we have enough data points
        if len(losses) >= 10:
            # Compare first 10% vs last 10% of sequence
            early_window = max(1, len(losses) // 10)
            late_window = max(1, len(losses) // 10)
            
            initial_loss = np.mean(losses[:early_window])
            final_loss = np.mean(losses[-late_window:])
            improvement = (initial_loss - final_loss) / initial_loss * 100
            
            # Add improvement annotation
            ax.text(0.02, 0.98, f'Layer {layer_id}\nInitial: {initial_loss:.3f}\nFinal: {final_loss:.3f}\nImprovement: {improvement:.1f}%',
                   transform=ax.transAxes, verticalalignment='top', horizontalalignment='left',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                   fontsize=10)
        
        # Formatting
        ax.set_xlabel('Token Position', fontsize=12)
        ax.set_ylabel('Reconstruction Loss (MSE)', fontsize=12)
        ax.set_title(f'TTT Layer {layer_id}: Reconstruction Loss vs Token Position', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add trend line if we have enough points
        if len(losses) >= 5:
            z = np.polyfit(positions, losses, 1)
            p = np.poly1d(z)
            ax.plot(positions, p(positions), '--', alpha=0.7, color='red', linewidth=1, label='Trend')
            ax.legend()
        
        # Save plot
        seq_suffix = f"_{sequence_id}" if sequence_id else ""
        plot_filename = f'layer_{layer_id}_reconstruction_loss{seq_suffix}.png'
        plot_path = os.path.join(output_dir, plot_filename)
        
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        plot_paths.append(plot_path)
        print(f"[Per-Layer Plotting] Saved Layer {layer_id} plot to: {plot_path}")
    
    return plot_paths


def create_combined_layer_comparison_plot(
    losses_per_layer: Dict[int, List[float]],
    output_dir: str,
    sequence_id: Optional[str] = None,
) -> str:
    """
    Create a single plot comparing all layers' reconstruction losses.
    
    Args:
        losses_per_layer: Dict mapping layer_id to list of reconstruction losses
        output_dir: Directory to save the plot
        sequence_id: Optional identifier for the sequence being plotted
    
    Returns:
        Path to the saved plot file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Color palette for different layers
    colors = plt.cm.tab10(np.linspace(0, 1, len(losses_per_layer)))
    
    for (layer_id, losses), color in zip(losses_per_layer.items(), colors):
        if len(losses) == 0:
            continue
            
        # Token positions (0-indexed)
        positions = list(range(len(losses)))
        
        # Plot reconstruction loss over token positions
        ax.plot(positions, losses, 'o-', color=color, linewidth=2, markersize=3, 
               label=f'Layer {layer_id}', alpha=0.8)
    
    # Formatting
    ax.set_xlabel('Token Position', fontsize=12)
    ax.set_ylabel('Reconstruction Loss (MSE)', fontsize=12)
    ax.set_title('TTT Layers Comparison: Reconstruction Loss vs Token Position', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Save plot
    seq_suffix = f"_{sequence_id}" if sequence_id else ""
    plot_filename = f'all_layers_comparison{seq_suffix}.png'
    plot_path = os.path.join(output_dir, plot_filename)
    
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[Per-Layer Plotting] Saved layer comparison plot to: {plot_path}")
    return plot_path
