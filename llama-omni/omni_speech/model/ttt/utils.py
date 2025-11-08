"""
Core TTT Utility Functions

This module contains essential utility functions for TTT layers:
- LayerNorm forward and fused backward
- Rotary Position Embeddings
- Scanning operations for sequential mini-batch processing

CRITICAL NOTES:
1. All TTT inner states (W1, b1, W2, b2) MUST be float32
2. LayerNorm operations must use float32 for numerical stability
3. Gradients accumulate over thousands of steps - precision is critical
"""

import torch
import torch.nn.functional as F
import logging
from typing import Optional, Tuple

# Setup logger
logger = logging.getLogger(__name__)


def ln_fwd(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    LayerNorm forward pass (used in TTT reconstruction target).

    CRITICAL: This operates on TTT hidden states which should be float32.

    Args:
        x: Input tensor [B*nh, N, f] where:
           - B = batch size
           - nh = number of heads
           - N = sequence length (or mini-batch size)
           - f = feature dimension (head_dim)
        gamma: Scale parameter [1, nh, 1, f]
        beta: Bias parameter [1, nh, 1, f]
        eps: Epsilon for numerical stability

    Returns:
        Normalized tensor [B*nh, N, f]

    Shape transformations:
        Input:  [B*nh, N, f]
        -> Reshape: [B, nh, N, f]
        -> Normalize along f dimension
        -> Scale and shift with per-head gamma/beta
        -> Reshape: [B*nh, N, f]
    """
    B_nh, N, HF = x.shape
    nh = gamma.shape[1]

    # Reshape to separate batch and heads: [B*nh, N, f] -> [B, nh, N, f]
    x = x.reshape(-1, nh, N, HF)

    # Compute mean and variance over feature dimension
    mu = x.mean(dim=-1, keepdim=True)  # [B, nh, N, 1]
    var = x.var(dim=-1, keepdim=True, unbiased=False)  # [B, nh, N, 1]

    # Normalize
    std = torch.sqrt(var + eps)
    x_hat = (x - mu) / std  # [B, nh, N, f]

    # Apply per-head scale and shift
    # gamma, beta: [1, nh, 1, f]
    y = gamma * x_hat + beta  # [B, nh, N, f]

    # Reshape back: [B, nh, N, f] -> [B*nh, N, f]
    y = y.reshape(-1, N, HF)

    return y


def ln_fused_l2_bwd(
    x: torch.Tensor,
    l2_target: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Fused LayerNorm backward with L2 loss gradient computation.

    This is THE CORE of TTT's learning mechanism. It computes:
    1. Forward LayerNorm: z = LN(x)
    2. L2 loss: loss = ||z - target||^2
    3. Backward gradient: grad_x = d(loss)/dx

    CRITICAL: This function is called thousands of times during generation.
    Numerical precision is ESSENTIAL - must use float32.

    The gradient computation uses the analytical derivative of LayerNorm,
    which involves:
    - Mean gradient: affects all elements equally
    - Variance gradient: weighted by normalized values

    Args:
        x: Input tensor [B*nh, N, f] - the pre-LN activations
        l2_target: Target tensor [B*nh, N, f] - what we want LN(x) to match
        gamma: Scale parameter [1, nh, 1, f]
        beta: Bias parameter [1, nh, 1, f]
        eps: Epsilon for numerical stability

    Returns:
        Gradient tensor [B*nh, N, f] - grad of L2 loss w.r.t. x

    Mathematical breakdown:
        z = gamma * (x - mu) / std + beta    [LayerNorm]
        loss = ||z - target||^2               [L2 loss]
        grad_z = 2 * (z - target)             [Loss gradient]
        grad_x = BackwardLN(grad_z)           [LN backward]

    The LN backward is:
        grad_x = (gamma / std) * [
            grad_z
            - mean(grad_z)
            - x_hat * mean(grad_z * x_hat)
        ] / f

    where x_hat = (x - mu) / std
    """
    B_nh, N, HF = x.shape
    nh = gamma.shape[1]

    # Forward pass - compute mean and variance
    mu = x.mean(dim=-1, keepdim=True)  # [B*nh, N, 1]
    var = x.var(dim=-1, keepdim=True, unbiased=False)  # [B*nh, N, 1]

    # Normalize
    std = torch.sqrt(var + eps)
    x_hat = (x - mu) / std  # [B*nh, N, f] - normalized input

    # Apply scale and shift - reshape for per-head parameters
    # x_hat: [B*nh, N, f] -> [B, nh, N, f]
    # gamma, beta: [1, nh, 1, f]
    x_hat_reshaped = x_hat.reshape(-1, nh, N, HF)
    y = gamma * x_hat_reshaped + beta  # [B, nh, N, f]

    # Compute loss gradient: grad_y = 2 * (y - target)
    # For L2 loss ||y - target||^2, gradient is 2 * (y - target)
    # But we typically absorb the factor of 2 into the learning rate
    l2_target_reshaped = l2_target.reshape(-1, nh, N, HF)
    grad_output = y - l2_target_reshaped  # [B, nh, N, f]

    # Backward through scale: grad_x_hat = grad_output * gamma
    grad_x_hat = (grad_output * gamma).reshape(B_nh, N, HF)  # [B*nh, N, f]

    # Backward through normalization
    # This is the analytical derivative of (x - mu) / std
    # The formula accounts for:
    # - Direct contribution: grad_x_hat / std
    # - Mean shift: affects all elements
    # - Variance change: weighted by normalized values

    z = (
        (1.0 / HF) * (
            HF * grad_x_hat  # Direct contribution
            - grad_x_hat.sum(dim=-1, keepdim=True)  # Mean gradient
            - x_hat * (grad_x_hat * x_hat).sum(dim=-1, keepdim=True)  # Variance gradient
        ) / std
    )

    return z


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate half the hidden dims of the input (for RoPE).

    Used in Rotary Position Embeddings (RoPE).
    Takes second half of features and negates them, then swaps halves.

    Args:
        x: Input tensor [..., d]

    Returns:
        Rotated tensor [..., d] where:
        - First half = -second half of input
        - Second half = first half of input

    Example:
        Input: [a, b, c, d]
        Output: [-c, -d, a, b]
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Rotary Position Embeddings to query and key tensors.

    RoPE encodes position information by rotating features in 2D planes.
    This preserves relative position information in a rotation-equivariant way.

    IMPORTANT: For TTT, RoPE is applied WITHIN mini-batches, not globally.
    This means position_ids range from 0 to mini_batch_size-1 for each mini-batch.

    Args:
        q: Query tensor [B, nh, N, f]
        k: Key tensor [B, nh, N, f]
        cos: Cosine values [B, N, f] or [B, 1, N, f]
        sin: Sine values [B, N, f] or [B, 1, N, f]

    Returns:
        Tuple of (q_embed, k_embed) with same shape as inputs

    Math:
        For each 2D plane (feature pairs):
        [q1]   [cos  -sin] [q1]
        [q2] = [sin   cos] [q2]

        Implemented as:
        q_embed = q * cos + rotate_half(q) * sin
    """
    # Ensure cos/sin have correct shape for broadcasting
    if cos.dim() == 3:  # [B, N, f]
        cos = cos.unsqueeze(1)  # [B, 1, N, f]
        sin = sin.unsqueeze(1)  # [B, 1, N, f]

    # Apply rotation
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


def precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float = 10000.0,
    device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute cosine and sine values for Rotary Position Embeddings.

    For TTT, this is computed per mini-batch, NOT for the full sequence.
    So 'end' should be mini_batch_size, not the full sequence length.

    Args:
        dim: Feature dimension (head_dim)
        end: Maximum position index (mini_batch_size for TTT)
        theta: Base for position encoding (10000 for standard RoPE)
        device: Device to create tensors on

    Returns:
        Tuple of (cos, sin) tensors, each of shape [end, dim]

    Math:
        freq[i] = 1 / (theta ^ (2i / dim))  for i in [0, dim/2)
        pos_enc[pos, 2i] = cos(pos * freq[i])
        pos_enc[pos, 2i+1] = sin(pos * freq[i])
    """
    # Compute frequencies
    # freqs[i] = 1 / (theta ^ (2*i / dim))
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))

    # Create position indices [0, 1, 2, ..., end-1]
    t = torch.arange(end, device=device, dtype=torch.float32)

    # Compute outer product: [end, 1] x [1, dim/2] = [end, dim/2]
    freqs = torch.outer(t, freqs)

    # Duplicate for both elements of each 2D plane: [end, dim/2] -> [end, dim]
    freqs_cis = torch.cat([freqs, freqs], dim=-1)

    # Compute cos and sin
    cos = freqs_cis.cos()
    sin = freqs_cis.sin()

    return cos, sin


def scan(
    fn,
    init_params,
    inputs,
    checkpoint_group_size: int = 1
):
    """
    Sequential scan operation over mini-batches with gradient checkpointing.

    This is the core sequential operation in TTT. It applies 'fn' to each
    mini-batch sequentially, carrying forward the TTT state (W1, b1, etc.)
    from one mini-batch to the next.

    CRITICAL: This is where conversation-level state persistence happens.
    - init_params contains initial W1, b1 (from previous conversation or initialization)
    - After processing all mini-batches, final params contain updated W1, b1
    - These updated params should be saved for next batch in same conversation

    Args:
        fn: Function to apply to each mini-batch
            Signature: fn(params_dict, inputs_dict) -> (new_params_dict, output)
        init_params: Initial parameters (dict with W1_states, b1_states, etc.)
        inputs: Input tensors (dict with XK, XQ, XV, eta) - first dim is mini-batch
        checkpoint_group_size: How many mini-batches to process before checkpointing

    Returns:
        Tuple of (final_params, all_outputs)
        - final_params: Updated parameters after processing all mini-batches
        - all_outputs: Concatenated outputs from all mini-batches

    Implementation:
        for each mini-batch:
            params, output = fn(params, mini_batch_inputs)
            outputs.append(output)
        return params, concat(outputs)

    Gradient checkpointing is used to save memory during backward pass.
    """
    num_mini_batches = inputs['XK'].shape[0]  # First dimension is mini-batch

    outputs = []
    current_params = init_params

    for i in range(num_mini_batches):
        # Extract inputs for this mini-batch
        mini_batch_inputs = {
            key: val[i] for key, val in inputs.items()
        }

        # Apply function
        if checkpoint_group_size > 1 and i % checkpoint_group_size == 0:
            # Use gradient checkpointing
            current_params, output = torch.utils.checkpoint.checkpoint(
                fn, current_params, mini_batch_inputs, use_reentrant=False
            )
        else:
            current_params, output = fn(current_params, mini_batch_inputs)

        outputs.append(output)

    # Stack outputs along mini-batch dimension
    all_outputs = torch.stack(outputs, dim=0)

    return current_params, all_outputs


# Logging utilities
def log_ttt_state(
    layer_idx: int,
    step: int,
    W1: torch.Tensor,
    b1: torch.Tensor,
    grad_norm: Optional[float] = None,
    loss: Optional[float] = None,
    log_level: str = "DEBUG"
):
    """
    Log TTT state information for debugging.

    Args:
        layer_idx: Which TTT layer (0-indexed)
        step: Current step/token position
        W1: Weight matrix [nh, f, f]
        b1: Bias vector [nh, 1, f]
        grad_norm: Norm of gradients (if available)
        loss: Reconstruction loss (if available)
        log_level: Logging level
    """
    if log_level == "DEBUG":
        # Compute statistics
        W1_mean = W1.mean().item()
        W1_std = W1.std().item()
        W1_max = W1.abs().max().item()
        b1_mean = b1.mean().item()
        b1_std = b1.std().item()

        msg = (
            f"[TTT Layer {layer_idx} Step {step}] "
            f"W1: mean={W1_mean:.6f} std={W1_std:.6f} max={W1_max:.6f} | "
            f"b1: mean={b1_mean:.6f} std={b1_std:.6f}"
        )

        if grad_norm is not None:
            msg += f" | grad_norm={grad_norm:.6f}"
        if loss is not None:
            msg += f" | loss={loss:.6f}"

        logger.debug(msg)
