"""
TTT-MLP Core Operation

Implements the Test-Time Training MLP operation with analytical gradient computation.
This is the core component that processes mini-batches sequentially, updating internal
states via gradient descent during the forward pass.

Based on the ttt-video-dit implementation, adapted for speech processing in Llama-Omni.

Critical requirements:
- FP32 precision for W1, b1, W2, b2 (prevents numerical instability)
- Analytical gradient computation (no autograd in inner loop)
- Sequential mini-batch processing (states carry forward)
- Returns both final states and outputs

Key differences from ttt-video-dit:
- Removed 3D RoPE (video-specific)
- Removed interleaving logic (video frames)
- Simplified for 1D speech sequences
- Added FP32 enforcement checks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
import math


def ln_fused_l2_bwd(
    Z: torch.Tensor,
    target: torch.Tensor,
    ttt_norm_weight: torch.Tensor,
    ttt_norm_bias: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Fused LayerNorm + L2 loss backward pass.

    Computes gradient of L2 loss w.r.t. pre-LayerNorm activations analytically.

    Args:
        Z: Pre-LayerNorm activations [B, N, H, mb, D]
        target: Target values [B, N, H, mb, D]
        ttt_norm_weight: LayerNorm weight [H, D]
        ttt_norm_bias: LayerNorm bias [H, D]
        eps: LayerNorm epsilon

    Returns:
        grad_Z: Gradient w.r.t. Z [B, N, H, mb, D]
    """

    # LayerNorm forward
    mean = Z.mean(dim=-1, keepdim=True)
    var = Z.var(dim=-1, keepdim=True, unbiased=False)
    std = torch.sqrt(var + eps)
    Z_normalized = (Z - mean) / std

    # Apply weight and bias
    weight = ttt_norm_weight.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, H, 1, D]
    bias = ttt_norm_bias.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    Z_ln = weight * Z_normalized + bias

    # L2 loss gradient w.r.t. LayerNorm output
    grad_Z_ln = 2.0 * (Z_ln - target)

    # Backprop through LayerNorm
    D = Z.shape[-1]
    grad_normalized = grad_Z_ln * weight

    # LayerNorm backward
    grad_var = -0.5 * (grad_normalized * (Z - mean)).sum(dim=-1, keepdim=True) / (var + eps)**(1.5)
    grad_mean = -grad_normalized.sum(dim=-1, keepdim=True) / std - \
                2.0 * grad_var * (Z - mean).sum(dim=-1, keepdim=True) / D

    grad_Z = grad_normalized / std + grad_var * 2.0 * (Z - mean) / D + grad_mean / D

    return grad_Z


def gelu_bwd(Z: torch.Tensor, approximate: str = "tanh") -> torch.Tensor:
    """
    GELU backward pass (derivative only, not full backward).

    Args:
        Z: Pre-GELU activations [B, N, H, mb, D]
        approximate: 'tanh' or 'none'

    Returns:
        grad_multiplier: Element-wise gradient multiplier [B, N, H, mb, D]
    """

    if approximate == "tanh":
        # Tanh approximation: gelu(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
        # Derivative is more complex but faster
        sqrt_2_over_pi = math.sqrt(2.0 / math.pi)
        a = sqrt_2_over_pi * (Z + 0.044715 * Z ** 3)
        tanh_a = torch.tanh(a)
        sech2_a = 1.0 - tanh_a ** 2

        grad = 0.5 * (1.0 + tanh_a) + \
               0.5 * Z * sech2_a * sqrt_2_over_pi * (1.0 + 3.0 * 0.044715 * Z ** 2)
    else:
        # Exact GELU derivative
        # gelu(x) = x * Φ(x) where Φ is standard normal CDF
        # gelu'(x) = Φ(x) + x * φ(x) where φ is standard normal PDF
        cdf = 0.5 * (1.0 + torch.erf(Z / math.sqrt(2.0)))
        pdf = torch.exp(-0.5 * Z ** 2) / math.sqrt(2.0 * math.pi)
        grad = cdf + Z * pdf

    return grad


def compute_mini_batch(
    params_dict: Dict[str, torch.Tensor],
    XK_mini_batch: torch.Tensor,
    XQ_mini_batch: torch.Tensor,
    XV_mini_batch: torch.Tensor,
    eta: torch.Tensor,
    ttt_norm_weight: torch.Tensor,
    ttt_norm_bias: torch.Tensor,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """
    Process one mini-batch with TTT-MLP.

    This is the core TTT operation:
    1. Forward pass: Compute reconstruction loss
    2. Backward pass: Compute gradients analytically
    3. Update states: W1, b1, W2, b2 via gradient descent
    4. Test-time prediction: Use updated states for XQ

    Args:
        params_dict: Current TTT states
            - W1_states: [B, H, D, hidden_dim] FP32
            - b1_states: [B, H, 1, hidden_dim] FP32
            - W2_states: [B, H, hidden_dim, D] FP32
            - b2_states: [B, H, 1, D] FP32
        XK_mini_batch: Key features [B, H, mb, D]
        XQ_mini_batch: Query features [B, H, mb, D]
        XV_mini_batch: Value features (target) [B, H, mb, D]
        eta: Learning rate [B, H, mb, 1]
        ttt_norm_weight: LayerNorm weight [H, D]
        ttt_norm_bias: LayerNorm bias [H, D]

    Returns:
        updated_params: Updated TTT states
        XQW_output: Transformed query features [B, H, mb, D]
    """

    # Extract current states (all FP32!)
    W1_init = params_dict["W1_states"]
    b1_init = params_dict["b1_states"]
    W2_init = params_dict["W2_states"]
    b2_init = params_dict["b2_states"]

    B, H, mb, D = XK_mini_batch.shape
    hidden_dim = W1_init.shape[-1]

    # Verify FP32
    assert W1_init.dtype == torch.float32, f"W1 must be FP32, got {W1_init.dtype}"
    assert b1_init.dtype == torch.float32, f"b1 must be FP32, got {b1_init.dtype}"
    assert W2_init.dtype == torch.float32, f"W2 must be FP32, got {W2_init.dtype}"
    assert b2_init.dtype == torch.float32, f"b2 must be FP32, got {b2_init.dtype}"

    # Ensure inputs are FP32
    XK = XK_mini_batch.float()
    XQ = XQ_mini_batch.float()
    XV = XV_mini_batch.float()

    # =========================================================================
    # FORWARD PASS: Reconstruction
    # =========================================================================

    # First layer: XK @ W1 + b1
    # [B, H, mb, D] @ [B, H, D, hidden_dim] -> [B, H, mb, hidden_dim]
    Z1 = torch.einsum('bhmd,bhdk->bhmk', XK, W1_init) + b1_init

    # GELU activation
    X2 = F.gelu(Z1, approximate="tanh")

    # Second layer: X2 @ W2 + b2
    # [B, H, mb, hidden_dim] @ [B, H, hidden_dim, D] -> [B, H, mb, D]
    Z2 = torch.einsum('bhmk,bhkd->bhmd', X2, W2_init) + b2_init

    # Reconstruction target (normalized)
    reconstruction_target = XV  # Already normalized by caller

    # =========================================================================
    # BACKWARD PASS: Analytical Gradients
    # =========================================================================

    # Gradient of loss w.r.t. Z2 (pre-second-layer output)
    # Fused LayerNorm + L2 backward
    grad_l_wrt_Z2 = ln_fused_l2_bwd(
        Z2,
        reconstruction_target,
        ttt_norm_weight,
        ttt_norm_bias
    )  # [B, H, mb, D]

    # Gradient w.r.t. W2: X2^T @ grad_Z2
    grad_W2 = torch.einsum('bhmk,bhmd->bhkd', X2, grad_l_wrt_Z2)

    # Gradient w.r.t. b2: sum over mini-batch dimension
    grad_b2 = grad_l_wrt_Z2.sum(dim=2, keepdim=True)  # [B, H, 1, D]

    # Backprop through second layer to X2
    grad_X2 = torch.einsum('bhmd,bhkd->bhmk', grad_l_wrt_Z2, W2_init)

    # Backprop through GELU
    gelu_grad = gelu_bwd(Z1, approximate="tanh")  # [B, H, mb, hidden_dim]
    grad_Z1 = grad_X2 * gelu_grad

    # Gradient w.r.t. W1: XK^T @ grad_Z1
    grad_W1 = torch.einsum('bhmd,bhmk->bhdk', XK, grad_Z1)

    # Gradient w.r.t. b1: sum over mini-batch dimension
    grad_b1 = grad_Z1.sum(dim=2, keepdim=True)  # [B, H, 1, hidden_dim]

    # =========================================================================
    # INNER LOOP UPDATE: Gradient Descent
    # =========================================================================

    # Extract last learning rate (for final update)
    last_eta = eta[:, :, -1:, :]  # [B, H, 1, 1]

    # Update W1: W1 -= eta * grad_W1
    W1_last = W1_init - last_eta.squeeze(-1).unsqueeze(-2) * grad_W1

    # Update b1: b1 -= eta * grad_b1
    b1_last = b1_init - last_eta.squeeze(-1) * grad_b1

    # Update W2: W2 -= eta * grad_W2
    W2_last = W2_init - last_eta.squeeze(-1).unsqueeze(-2) * grad_W2

    # Update b2: b2 -= eta * grad_b2
    b2_last = b2_init - last_eta.squeeze(-1) * grad_b2

    # =========================================================================
    # TEST-TIME PREDICTION: Use Updated States for XQ
    # =========================================================================

    # Compute attention-weighted gradients (for smooth prediction)
    # Attention scores: XQ @ XK^T (softmax over mini-batch)
    attn_scores = torch.einsum('bhqd,bhkd->bhqk', XQ, XK)  # [B, H, mb, mb]
    attn_scores = attn_scores / math.sqrt(D)
    attn_weights = F.softmax(attn_scores, dim=-1)

    # Weighted sum of gradients
    # [B, H, mb, mb] @ [B, H, mb, hidden_dim] -> [B, H, mb, hidden_dim]
    Attn1 = torch.einsum('bhqk,bhmk->bhqk', attn_weights, grad_Z1)

    # Compute mean states (averaged over updates)
    W1_bar = (W1_init + W1_last) / 2.0
    b1_bar = (b1_init + b1_last) / 2.0
    W2_bar = (W2_init + W2_last) / 2.0
    b2_bar = (b2_init + b2_last) / 2.0

    # Forward with query using averaged states
    # Z1_bar = XQ @ W1_bar - (eta * Attn1) @ grad_Z1 + b1_bar
    # Simplified version without attention weighting (can add if needed)
    Z1_bar = torch.einsum('bhmd,bhdk->bhmk', XQ, W1_bar) + b1_bar
    X2_bar = F.gelu(Z1_bar, approximate="tanh")
    XQW_output = torch.einsum('bhmk,bhkd->bhmd', X2_bar, W2_bar) + b2_bar

    # Apply LayerNorm to output
    XQW_output = F.layer_norm(
        XQW_output,
        normalized_shape=(D,),
        weight=ttt_norm_weight.unsqueeze(0).unsqueeze(0),
        bias=ttt_norm_bias.unsqueeze(0).unsqueeze(0)
    )

    # Package updated states
    updated_params = {
        "W1_states": W1_last.detach(),  # Detach to prevent gradient accumulation
        "b1_states": b1_last.detach(),
        "W2_states": W2_last.detach(),
        "b2_states": b2_last.detach(),
    }

    return updated_params, XQW_output


def ttt_mlp(
    XK: torch.Tensor,
    XQ: torch.Tensor,
    XV: torch.Tensor,
    eta: torch.Tensor,
    ttt_norm_weight: torch.Tensor,
    ttt_norm_bias: torch.Tensor,
    W1_init: torch.Tensor,
    b1_init: torch.Tensor,
    W2_init: torch.Tensor,
    b2_init: torch.Tensor,
    checkpoint_group_size: int = 4,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """
    TTT-MLP operation over multiple mini-batches.

    Processes sequence in mini-batches, updating TTT states sequentially.
    Uses gradient checkpointing to save memory.

    Args:
        XK: Keys [B, H, num_mb, mb_size, D]
        XQ: Queries [B, H, num_mb, mb_size, D]
        XV: Values (targets) [B, H, num_mb, mb_size, D]
        eta: Learning rates [B, H, num_mb, mb_size, 1]
        ttt_norm_weight: LayerNorm weight [H, D]
        ttt_norm_bias: LayerNorm bias [H, D]
        W1_init: Initial W1 [B, H, D, hidden_dim] FP32
        b1_init: Initial b1 [B, H, 1, hidden_dim] FP32
        W2_init: Initial W2 [B, H, hidden_dim, D] FP32
        b2_init: Initial b2 [B, H, 1, D] FP32
        checkpoint_group_size: Checkpoint every N mini-batches (memory vs speed)

    Returns:
        final_params: Final TTT states after all mini-batches
        XQW_batch: Transformed outputs [B, H, num_mb, mb_size, D]
    """

    B, H, num_mb, mb_size, D = XK.shape

    # Initialize states
    current_params = {
        "W1_states": W1_init,
        "b1_states": b1_init,
        "W2_states": W2_init,
        "b2_states": b2_init,
    }

    # Output buffer
    XQW_list = []

    # Process mini-batches sequentially
    for mb_idx in range(num_mb):
        # Extract current mini-batch
        XK_mb = XK[:, :, mb_idx, :, :]  # [B, H, mb_size, D]
        XQ_mb = XQ[:, :, mb_idx, :, :]
        XV_mb = XV[:, :, mb_idx, :, :]
        eta_mb = eta[:, :, mb_idx, :, :]  # [B, H, mb_size, 1]

        # Process mini-batch (with optional checkpointing)
        if checkpoint_group_size > 0 and mb_idx % checkpoint_group_size == 0:
            # Use gradient checkpointing for memory efficiency
            updated_params, XQW_mb = torch.utils.checkpoint.checkpoint(
                compute_mini_batch,
                current_params,
                XK_mb,
                XQ_mb,
                XV_mb,
                eta_mb,
                ttt_norm_weight,
                ttt_norm_bias,
                use_reentrant=False
            )
        else:
            # Regular forward pass
            updated_params, XQW_mb = compute_mini_batch(
                current_params,
                XK_mb,
                XQ_mb,
                XV_mb,
                eta_mb,
                ttt_norm_weight,
                ttt_norm_bias,
            )

        # Update current states for next mini-batch
        current_params = updated_params

        # Collect output
        XQW_list.append(XQW_mb.unsqueeze(2))  # Add mini-batch dimension back

    # Concatenate all mini-batch outputs
    XQW_batch = torch.cat(XQW_list, dim=2)  # [B, H, num_mb, mb_size, D]

    return current_params, XQW_batch


if __name__ == "__main__":
    """Test the TTT-MLP operation."""
    print("Testing TTT-MLP Operation")
    print("=" * 60)

    # Parameters
    B, H, num_mb, mb_size, D = 2, 8, 4, 64, 128
    hidden_dim = 4 * D

    # Create test inputs
    XK = torch.randn(B, H, num_mb, mb_size, D, dtype=torch.float32)
    XQ = torch.randn(B, H, num_mb, mb_size, D, dtype=torch.float32)
    XV = torch.randn(B, H, num_mb, mb_size, D, dtype=torch.float32)
    eta = torch.ones(B, H, num_mb, mb_size, 1, dtype=torch.float32) * 0.01

    # Initialize states
    W1_init = torch.randn(B, H, D, hidden_dim, dtype=torch.float32) * 0.02
    b1_init = torch.zeros(B, H, 1, hidden_dim, dtype=torch.float32)
    W2_init = torch.randn(B, H, hidden_dim, D, dtype=torch.float32) * 0.02
    b2_init = torch.zeros(B, H, 1, D, dtype=torch.float32)

    # TTT norm parameters
    ttt_norm_weight = torch.ones(H, D, dtype=torch.float32)
    ttt_norm_bias = torch.zeros(H, D, dtype=torch.float32)

    print(f"Input shapes:")
    print(f"  XK: {XK.shape}")
    print(f"  W1_init: {W1_init.shape}")
    print(f"  Mini-batches: {num_mb}, size: {mb_size}")

    # Run TTT-MLP
    final_params, XQW_output = ttt_mlp(
        XK, XQ, XV, eta,
        ttt_norm_weight, ttt_norm_bias,
        W1_init, b1_init, W2_init, b2_init,
        checkpoint_group_size=2
    )

    print(f"\nOutput shapes:")
    print(f"  XQW: {XQW_output.shape}")
    print(f"  W1_final: {final_params['W1_states'].shape}")

    print(f"\nState statistics:")
    print(f"  W1: mean={final_params['W1_states'].mean():.6f}, "
          f"std={final_params['W1_states'].std():.6f}")
    print(f"  b1: mean={final_params['b1_states'].mean():.6f}, "
          f"std={final_params['b1_states'].std():.6f}")

    print(f"\nOutput statistics:")
    print(f"  XQW: mean={XQW_output.mean():.6f}, std={XQW_output.std():.6f}")

    # Check for NaN
    has_nan = torch.isnan(XQW_output).any()
    print(f"\nNumerical health:")
    print(f"  Has NaN: {has_nan}")
    print(f"  Output range: [{XQW_output.min():.6f}, {XQW_output.max():.6f}]")

    print("\n✅ TTT-MLP test complete!")
