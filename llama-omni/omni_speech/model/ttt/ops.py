"""
TTT Operations - Core TTT Algorithm Implementation

This module implements the Test-Time Training algorithms:
- ttt_linear: Linear model as hidden state
- ttt_mlp: 2-layer MLP as hidden state

CRITICAL IMPLEMENTATION DETAILS:
1. TTT updates happen during forward pass (not backprop!)
2. Gradients are computed analytically using ln_fused_l2_bwd
3. State (W1, b1) is updated via gradient descent DURING inference
4. Updates accumulate across mini-batches within a conversation
5. All inner state operations MUST be in float32

The algorithm for each mini-batch:
1. Compute reconstruction target: XV_norm = LN(XV - XK)
2. Compute prediction: Z = f(XK; W) where f is linear or MLP
3. Compute loss: L = ||LN(Z) - XV_norm||^2
4. Compute gradient: grad = d(L)/d(W) using analytical formula
5. Update state: W_new = W_old - eta * grad
6. Compute output: output = f(XQ; W_new)

This happens sequentially for each mini-batch, with state carrying forward.
"""

import torch
import torch.nn.functional as F
from torch.utils._pytree import tree_map
import logging
from typing import Dict, Tuple

from .utils import ln_fwd, ln_fused_l2_bwd, scan

logger = logging.getLogger(__name__)


@torch.compile
def compute_ttt_linear_mini_batch(
    params_dict: Dict[str, torch.Tensor],
    inputs: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """
    Process one mini-batch with TTT-Linear.

    This is the CORE TTT algorithm. It:
    1. Computes reconstruction loss: ||LN(XK @ W1 + b1) - LN(XV - XK)||^2
    2. Computes analytical gradient of this loss w.r.t. W1, b1
    3. Updates W1, b1 using gradient descent: W1 -= eta * grad_W1
    4. Computes output: XQ @ W1_updated + b1_updated

    CRITICAL: All operations on W1, b1 must be in float32.
    The inputs (XK, XQ, XV) can be in bf16/fp16, but we cast internally.

    Args:
        params_dict: Dictionary containing:
            - W1_states: [B*nh, f, f] - MUST be float32
            - b1_states: [B*nh, 1, f] - MUST be float32
            - ttt_norm_weight: [1, nh, 1, f] - LayerNorm scale
            - ttt_norm_bias: [1, nh, 1, f] - LayerNorm bias

        inputs: Dictionary containing:
            - XQ_mini_batch: [B, nh, K, f] - query vectors
            - XV_mini_batch: [B, nh, K, f] - value vectors (reconstruction target)
            - XK_mini_batch: [B, nh, K, f] - key vectors (input to TTT model)
            - eta_mini_batch: [B, nh, K, 1] - learning rates per token

    Returns:
        Tuple of:
        - last_param_dict: Updated parameters (W1_last, b1_last)
        - XQW_mini_batch: Output [B, nh, K, f]

    Shape flow:
        X1 = XK [B, nh, K, f]
        Z1 = X1 @ W1 + b1 = [B, nh, K, f] @ [B*nh, f, f] + [B*nh, 1, f]
           = [B, nh, K, f]  (after proper broadcasting)

        reconstruction_target = XV - XK = [B, nh, K, f]

        grad = ln_fused_l2_bwd(Z1, target) = [B, nh, K, f]

        W1_new = W1 - eta * (X1.T @ grad) = [B*nh, f, f] - [B*nh, f, K] @ [B*nh, K, f]
        b1_new = b1 - eta * sum(grad) = [B*nh, 1, f]

        output = XQ @ W1_new + b1_new = [B, nh, K, f]
    """
    # Extract parameters
    W1_init = params_dict["W1_states"]  # [B*nh, f, f] float32
    b1_init = params_dict["b1_states"]  # [B*nh, 1, f] float32
    ttt_norm_weight = params_dict["ttt_norm_weight"]  # [1, nh, 1, f]
    ttt_norm_bias = params_dict["ttt_norm_bias"]  # [1, nh, 1, f]

    # Extract inputs
    XQ_mini_batch = inputs["XQ"]  # [B, nh, K, f]
    XV_mini_batch = inputs["XV"]  # [B, nh, K, f]
    XK_mini_batch = inputs["XK"]  # [B, nh, K, f]
    eta_mini_batch = inputs["eta"]  # [B, nh, K, 1]

    B, num_heads, K, head_dim = XQ_mini_batch.shape

    # Reshape for batch processing: [B, nh, K, f] -> [B*nh, K, f]
    XQ = XQ_mini_batch.reshape(B * num_heads, K, head_dim)
    XV = XV_mini_batch.reshape(B * num_heads, K, head_dim)
    XK = XK_mini_batch.reshape(B * num_heads, K, head_dim)
    eta = eta_mini_batch.reshape(B * num_heads, K, 1)

    # Ensure W1, b1 are float32 (CRITICAL!)
    assert W1_init.dtype == torch.float32, f"W1 must be float32, got {W1_init.dtype}"
    assert b1_init.dtype == torch.float32, f"b1 must be float32, got {b1_init.dtype}"

    # 1. Forward pass: compute Z1 = XK @ W1 + b1
    # XK: [B*nh, K, f]
    # W1: [B*nh, f, f]
    # Z1: [B*nh, K, f]
    Z1 = XK @ W1_init + b1_init  # Broadcasting: [B*nh, K, f] + [B*nh, 1, f]

    # 2. Compute reconstruction target: XV - XK
    # This is the "residual" that TTT tries to predict
    reconstruction_target = XV - XK  # [B*nh, K, f]

    # 3. Compute loss gradient using fused LayerNorm + L2 backward
    # This computes: grad_l_wrt_Z1 = d(||LN(Z1) - LN(target)||^2) / d(Z1)
    # Shape: [B*nh, K, f]
    ln_weight = ttt_norm_weight.reshape(num_heads, 1, head_dim)  # [nh, 1, f]
    ln_bias = ttt_norm_bias.reshape(num_heads, 1, head_dim)  # [nh, 1, f]

    # We need to expand ln_weight, ln_bias for batch dimension
    # Input to ln_fused_l2_bwd expects: [B*nh, K, f]
    # and gamma/beta: [1, nh, 1, f]
    grad_l_wrt_Z1 = ln_fused_l2_bwd(
        Z1,  # [B*nh, K, f]
        reconstruction_target,  # [B*nh, K, f]
        ttt_norm_weight,  # [1, nh, 1, f]
        ttt_norm_bias  # [1, nh, 1, f]
    )

    # 4. Compute attention-weighted output (before parameter update)
    # Attn1 = XQ @ XK.T = [B*nh, K, f] @ [B*nh, f, K] = [B*nh, K, K]
    Attn1 = XQ @ XK.transpose(-2, -1)

    # 5. Update bias: b1_new = b1 - eta * grad
    # eta: [B*nh, K, 1]
    # grad_l_wrt_Z1: [B*nh, K, f]
    # eta * grad: [B*nh, K, f]
    # sum over K dimension: [B*nh, 1, f]
    b1_bar = b1_init - (eta @ grad_l_wrt_Z1)  # Using matmul to sum over K
    # Actually we need: b1_bar = b1_init - sum(eta * grad_l_wrt_Z1, dim=-2, keepdim=True)
    # Let me fix this:
    b1_bar = b1_init - torch.sum(eta * grad_l_wrt_Z1, dim=-2, keepdim=True)
    # Shape: [B*nh, 1, f]

    # 6. Compute output with attention-weighted update
    # Z1_bar = XQ @ W1 - (eta * Attn1) @ grad + b1_bar
    #
    # XQ @ W1: [B*nh, K, f]
    # eta: [B*nh, K, 1]
    # Attn1: [B*nh, K, K]
    # grad: [B*nh, K, f]
    # (eta * Attn1) @ grad: need to be careful with shapes
    #
    # Actually, eta should broadcast: eta * Attn1 = [B*nh, K, 1] * [B*nh, K, K]
    # But we want element-wise for each query position
    # Let me check the original formula...
    #
    # From ttt-video-dit: Z1_bar = XQ @ W1 - (eta * Attn1) @ grad + b1_bar
    # where eta is [B, nh, K, 1] and Attn1 is implicitly [K, K] for each batch/head
    #
    # I think the formula is:
    # For each position k in K:
    #   Z1_bar[k] = XQ[k] @ W1 - eta[k] * sum_j(Attn1[k,j] * grad[j]) + b1_bar
    #
    # Which is: Z1_bar = XQ @ W1 - diag(eta) @ Attn1 @ grad + b1_bar
    #
    # Since eta is per-token, we need:
    # (eta * Attn1) @ grad where eta multiplies each row of Attn1

    # Correcting: eta_Attn1 = eta * Attn1 where eta: [B*nh, K, 1], Attn1: [B*nh, K, K]
    # This gives [B*nh, K, K] with each row k scaled by eta[k]
    eta_Attn1 = eta * Attn1  # [B*nh, K, 1] * [B*nh, K, K] -> [B*nh, K, K]

    Z1_bar = XQ @ W1_init - (eta_Attn1 @ grad_l_wrt_Z1) + b1_bar
    # Shape: [B*nh, K, f]

    # 7. Apply LayerNorm to output
    Z1_bar = ln_fwd(Z1_bar, ttt_norm_weight, ttt_norm_bias)

    # 8. Add residual connection
    XQW_mini_batch = XQ + Z1_bar  # [B*nh, K, f]

    # 9. Update parameters for next mini-batch
    # We use the LAST eta value for the final update
    # last_eta: [B*nh, 1, 1] (eta for position K-1)
    last_eta = eta[:, -1:, :]  # [B*nh, 1, 1]

    # W1_last = W1_init - last_eta * XK.T @ grad
    # XK: [B*nh, K, f]
    # grad: [B*nh, K, f]
    # XK.T @ grad: [B*nh, f, K] @ [B*nh, K, f] = [B*nh, f, f]
    # last_eta * (...): [B*nh, 1, 1] * [B*nh, f, f] - need to broadcast correctly
    #
    # Actually from the formula: W1_last = W1_init - (last_eta * XK).T @ grad
    # where last_eta multiplies each column of XK

    # Let me check the original code more carefully...
    # From ttt-video-dit ops/ttt_linear.py line 40:
    # W1_last = W1_init - (last_eta_mini_batch * X1).transpose(-1, -2) @ grad_l_wrt_Z1
    #
    # So: (last_eta * XK).T @ grad
    # last_eta: [B*nh, 1, 1]
    # XK: [B*nh, K, f]
    # last_eta * XK: [B*nh, K, f] (broadcasts)
    # (last_eta * XK).T: [B*nh, f, K]
    # @ grad: [B*nh, f, K] @ [B*nh, K, f] = [B*nh, f, f]

    W1_last = W1_init - (last_eta * XK).transpose(-1, -2) @ grad_l_wrt_Z1

    # b1_last = b1_init - sum(last_eta * grad)
    # last_eta: [B*nh, 1, 1]
    # grad: [B*nh, K, f]
    # last_eta * grad: [B*nh, K, f]
    # sum over K: [B*nh, 1, f]
    b1_last = b1_init - torch.sum(last_eta * grad_l_wrt_Z1, dim=-2, keepdim=True)

    # Ensure output dtypes match
    W1_last = W1_last.to(torch.float32)
    b1_last = b1_last.to(torch.float32)

    # Reshape output back to [B, nh, K, f]
    XQW_mini_batch = XQW_mini_batch.reshape(B, num_heads, K, head_dim)

    # Package updated parameters
    last_param_dict = {
        "W1_states": W1_last,
        "b1_states": b1_last,
        "ttt_norm_weight": ttt_norm_weight,
        "ttt_norm_bias": ttt_norm_bias,
    }

    return last_param_dict, XQW_mini_batch


def ttt_linear(
    XK: torch.Tensor,
    XQ: torch.Tensor,
    XV: torch.Tensor,
    eta: torch.Tensor,
    ttt_norm_weight: torch.Tensor,
    ttt_norm_bias: torch.Tensor,
    W1_init: torch.Tensor,
    b1_init: torch.Tensor,
    checkpoint_group_size: int
) -> torch.Tensor:
    """
    Apply TTT-Linear to a sequence of mini-batches.

    This is the main entry point for TTT-Linear. It:
    1. Splits the sequence into mini-batches
    2. Processes each mini-batch sequentially with compute_ttt_linear_mini_batch
    3. Carries TTT state (W1, b1) forward from one mini-batch to the next
    4. Returns concatenated outputs

    CRITICAL: This function processes mini-batches SEQUENTIALLY.
    State from mini-batch i is used as input to mini-batch i+1.
    This is where conversation-level memory happens!

    Args:
        XK: Key tensor [B, nh, nc, K, f] where nc = num_mini_batches
        XQ: Query tensor [B, nh, nc, K, f]
        XV: Value tensor [B, nh, nc, K, f]
        eta: Learning rate tensor [B, nh, nc, K, 1]
        ttt_norm_weight: LayerNorm weight [1, nh, 1, f]
        ttt_norm_bias: LayerNorm bias [1, nh, 1, f]
        W1_init: Initial weight [B*nh, f, f] float32
        b1_init: Initial bias [B*nh, 1, f] float32
        checkpoint_group_size: How many mini-batches per checkpoint

    Returns:
        Output tensor [B, nc, K, nh, f]

    The output is reshaped to [B, nc, K, nh, f] to separate:
    - B: batch dimension
    - nc: mini-batch index
    - K: position within mini-batch
    - nh: head dimension
    - f: feature dimension
    """
    B, nh, nc, K, f = XK.shape

    logger.debug(
        f"[TTT-Linear] Processing {nc} mini-batches of size {K}, "
        f"batch={B}, heads={nh}, features={f}"
    )

    # Package initial parameters
    init_params_dict = {
        "W1_states": W1_init,
        "b1_states": b1_init,
        "ttt_norm_weight": ttt_norm_weight,
        "ttt_norm_bias": ttt_norm_bias,
    }

    # Package inputs
    inputs = {
        "XK": XK,
        "XQ": XQ,
        "XV": XV,
        "eta": eta,
    }

    # Reorder dimensions: [B, nh, nc, K, f] -> [nc, B, nh, K, f]
    # This makes the first dimension the mini-batch index for sequential processing
    inputs = tree_map(lambda x: x.permute(2, 0, 1, 3, 4), inputs)

    # Allocate output tensor
    XQW_batch = torch.empty_like(inputs["XK"])

    # Process mini-batches sequentially with gradient checkpointing
    final_params, XQW_batch = scan(
        compute_ttt_linear_mini_batch,
        init_params_dict,
        inputs,
        checkpoint_group_size,
    )

    # Log final state statistics
    W1_final = final_params["W1_states"]
    b1_final = final_params["b1_states"]
    logger.debug(
        f"[TTT-Linear] Final state: "
        f"W1 mean={W1_final.mean().item():.6f} std={W1_final.std().item():.6f}, "
        f"b1 mean={b1_final.mean().item():.6f} std={b1_final.std().item():.6f}"
    )

    # Reorder output: [nc, B, nh, K, f] -> [B, nc, K, nh, f]
    XQW_batch = XQW_batch.permute(1, 0, 3, 2, 4)

    return XQW_batch


# TODO: Implement ttt_mlp when needed
# For now, we'll start with ttt_linear which is simpler and proven to work
