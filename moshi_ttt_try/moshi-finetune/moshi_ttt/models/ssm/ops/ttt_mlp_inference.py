"""
TTT-MLP operations optimized for inference with cache-based persistence.
Adapted from ttt-lm-kernels to work with Moshi TTT architecture.
"""
import logging
import torch
import torch.nn.functional as F

from .utils import gelu_bwd, ln_fused_l2_bwd, ln_fwd

logger = logging.getLogger(__name__)


def ttt_mlp_decode_token(states, inputs, ttt_norm_weight, ttt_norm_bias):
    """
    Process a single token with TTT updates using cached states.
    
    Follows ttt-lm-kernels/ttt/modeling_ttt.py::ttt_linear_decode_last_token_in_mini_batch
    but adapted for 2-layer MLP.
    
    Args:
        states: Dict with keys W1_init, b1_init, W2_init, b2_init, W1_grad, b1_grad, W2_grad, b2_grad
                All tensors are [B*nh, d_in, d_out] or [B*nh, 1, d]
        inputs: Dict with keys XV, XK, XQ (each [B*nh, 1, head_dim]), token_idx, ttt_lr
        ttt_norm_weight: [1, nh, 1, head_dim]
        ttt_norm_bias: [1, nh, 1, head_dim]
        
    Returns:
        Z2_bar: [B*nh, 1, head_dim] - Output after TTT update
        
    Note: This function modifies states IN-PLACE, enabling persistence across tokens!
    """
    # Get REFERENCES to cache (not copies!)
    W1 = states['W1_init']
    b1 = states['b1_init']
    W2 = states['W2_init']
    b2 = states['b2_init']
    W1_grad = states['W1_grad']
    b1_grad = states['b1_grad']
    W2_grad = states['W2_grad']
    b2_grad = states['b2_grad']
    
    # Extract inputs
    XV, XK, XQ = inputs['XV'], inputs['XK'], inputs['XQ']
    token_idx, ttt_lr = inputs['token_idx'], inputs['ttt_lr']
    
    B_mul_NH, K, HF = XV.shape
    NH = ttt_norm_weight.shape[1]
    
    # === Forward pass with CURRENT weights ===
    Z1 = XK @ W1 + b1
    X2 = F.gelu(Z1, approximate="tanh")
    Z2 = X2 @ W2 + b2
    
    # Compute reconstruction target
    l2_target = XV - XK
    
    # Compute gradients using ln_fused_l2_bwd
    grad_l_wrt_Z2 = ln_fused_l2_bwd(Z2, l2_target, ttt_norm_weight, ttt_norm_bias)
    grad_l_wrt_Z1 = grad_l_wrt_Z2 @ W2.transpose(-2, -1) * gelu_bwd(Z1)
    
    # Scale gradients by learning rate
    ilr_mul_dl_dZ2 = ttt_lr * grad_l_wrt_Z2
    ilr_mul_dl_dZ1 = ttt_lr * grad_l_wrt_Z1
    
    # === Accumulate gradients IN-PLACE ===
    W1_grad.add_(XK.transpose(-1, -2) @ ilr_mul_dl_dZ1)
    b1_grad.add_(ilr_mul_dl_dZ1)
    W2_grad.add_(X2.transpose(-1, -2) @ ilr_mul_dl_dZ2)
    b2_grad.add_(ilr_mul_dl_dZ2)
    
    # === Update weights IN-PLACE ===
    W1.sub_(token_idx * W1_grad)
    b1.sub_(token_idx * b1_grad)
    W2.sub_(token_idx * W2_grad)
    b2.sub_(token_idx * b2_grad)
    
    # === Forward pass with UPDATED weights ===
    Z1_bar = XQ @ W1 + b1
    X2_bar = F.gelu(Z1_bar, approximate="tanh")
    Z2_bar = X2_bar @ W2 + b2
    
    # === Reset gradient accumulators ===
    W1_grad.zero_()
    b1_grad.zero_()
    W2_grad.zero_()
    b2_grad.zero_()
    
    return Z2_bar


def ttt_mlp_prefill_mini_batch(states, inputs, i, ttt_norm_weight, ttt_norm_bias):
    """
    Process a mini-batch during prefill (sequence processing).
    
    Follows ttt-lm-kernels/ttt/modeling_ttt.py but adapted for 2-layer MLP.
    
    Args:
        states: Dict with W1_init, b1_init, W2_init, b2_init
        inputs: Dict with XV[i], XK[i], XQ[i] (each [B*nh, mini_batch_size, head_dim])
                and eta[i], eta_last[i]
        i: Mini-batch index
        ttt_norm_weight: [1, nh, 1, head_dim]
        ttt_norm_bias: [1, nh, 1, head_dim]
        
    Returns:
        Z2_bar: [B*nh, mini_batch_size, head_dim]
        
    Note: Updates states IN-PLACE at the end of the mini-batch!
    """
    W1_init = states['W1_init']
    b1_init = states['b1_init']
    W2_init = states['W2_init']
    b2_init = states['b2_init']
    
    XV_mini_batch = inputs['XV'][i]
    XK_mini_batch = inputs['XK'][i]
    XQ_mini_batch = inputs['XQ'][i]
    eta_mini_batch = inputs['eta'][i]
    eta_mini_batch_last = inputs['eta_last'][i]
    
    # === Layer 1 ===
    Z1 = XK_mini_batch @ W1_init + b1_init
    l2_target_1 = XV_mini_batch - XK_mini_batch
    
    # Compute gradients
    grad_l_wrt_Z1 = ln_fused_l2_bwd(Z1, l2_target_1, ttt_norm_weight, ttt_norm_bias)
    
    # Compute b1_bar across the sequence
    b1_bar = b1_init - torch.tril(eta_mini_batch) @ grad_l_wrt_Z1
    
    # Compute Z1_bar
    Attn1 = torch.tril(XQ_mini_batch @ XK_mini_batch.transpose(-1, -2))
    Z1_bar = XQ_mini_batch @ W1_init - (eta_mini_batch * Attn1) @ grad_l_wrt_Z1 + b1_bar
    
    # === Layer 2 ===
    X2 = F.gelu(Z1, approximate="tanh")
    X2_bar = F.gelu(Z1_bar, approximate="tanh")
    Z2 = X2 @ W2_init + b2_init
    
    # For layer 2, the reconstruction target is the normalized Z1_bar
    l2_target_2 = ln_fwd(Z1_bar, ttt_norm_weight, ttt_norm_bias)
    
    grad_l_wrt_Z2 = ln_fused_l2_bwd(Z2, l2_target_2, ttt_norm_weight, ttt_norm_bias)
    
    b2_bar = b2_init - torch.tril(eta_mini_batch) @ grad_l_wrt_Z2
    
    Attn2 = torch.tril(X2_bar @ X2.transpose(-1, -2))
    Z2_bar = X2_bar @ W2_init - (eta_mini_batch * Attn2) @ grad_l_wrt_Z2 + b2_bar
    
    # === Update weights IN-PLACE (at end of mini-batch) ===
    W1_init.sub_((eta_mini_batch_last * XK_mini_batch.transpose(-1, -2)) @ grad_l_wrt_Z1)
    b1_init.copy_(b1_bar[:, -1:])
    
    W2_init.sub_((eta_mini_batch_last * X2.transpose(-1, -2)) @ grad_l_wrt_Z2)
    b2_init.copy_(b2_bar[:, -1:])
    
    return Z2_bar


def ttt_mlp_multi_layer_decode_token(states, inputs, ttt_norm_weight, ttt_norm_bias, num_layers):
    """
    Process a single token with multi-layer TTT-MLP using cached states.
    
    Args:
        states: Dict with keys W{i}_init, b{i}_init, W{i}_grad, b{i}_grad for i in range(num_layers)
        inputs: Dict with XV, XK, XQ, token_idx, ttt_lr
        ttt_norm_weight: [1, nh, 1, d]
        ttt_norm_bias: [1, nh, 1, d]
        num_layers: Number of layers in the MLP
        
    Returns:
        Output after all layers
    """
    XV, XK, XQ = inputs['XV'], inputs['XK'], inputs['XQ']
    token_idx, ttt_lr = inputs['token_idx'], inputs['ttt_lr']
    
    B_mul_NH, K, _ = XV.shape
    NH = ttt_norm_weight.shape[1]
    
    # First layer: reconstruction target is XV - XK
    reconstruction_target = XV - XK
    X_current = XK
    
    # Forward through all layers with current weights
    layer_outputs = []
    for i in range(num_layers):
        W = states[f'W{i}_init']
        b = states[f'b{i}_init']
        
        Z = X_current @ W + b
        
        if i < num_layers - 1:
            # Hidden layers: apply GELU
            X_next = F.gelu(Z, approximate="tanh")
        else:
            # Last layer: no activation
            X_next = Z
        
        layer_outputs.append((X_current, Z, X_next))
        X_current = X_next
    
    # Compute loss and gradients
    final_Z = layer_outputs[-1][1]
    grad_l_wrt_Z = ln_fused_l2_bwd(final_Z, reconstruction_target, ttt_norm_weight, ttt_norm_bias)
    
    # Backward through layers
    for i in range(num_layers - 1, -1, -1):
        X_in, Z, X_out = layer_outputs[i]
        W = states[f'W{i}_init']
        W_grad = states[f'W{i}_grad']
        b_grad = states[f'b{i}_grad']
        
        # Scale by learning rate
        ilr_mul_grad = ttt_lr * grad_l_wrt_Z
        
        # Accumulate gradients
        W_grad.add_(X_in.transpose(-1, -2) @ ilr_mul_grad)
        b_grad.add_(ilr_mul_grad)
        
        # Update weights
        W.sub_(token_idx * W_grad)
        states[f'b{i}_init'].sub_(token_idx * b_grad)
        
        # Reset gradients
        W_grad.zero_()
        b_grad.zero_()
        
        # Propagate gradient to previous layer
        if i > 0:
            grad_l_wrt_Z = grad_l_wrt_Z @ W.transpose(-2, -1)
            if i < num_layers - 1:  # Apply GELU backward if not first layer
                grad_l_wrt_Z = grad_l_wrt_Z * gelu_bwd(Z)
    
    # Forward through updated weights with XQ
    X_current = XQ
    for i in range(num_layers):
        W = states[f'W{i}_init']
        b = states[f'b{i}_init']
        
        Z = X_current @ W + b
        
        if i < num_layers - 1:
            X_current = F.gelu(Z, approximate="tanh")
        else:
            X_current = Z
    
    return X_current


def ttt_mlp_multi_layer_prefill_mini_batch(states, inputs, i, ttt_norm_weight, ttt_norm_bias, num_layers):
    """
    Process a mini-batch during prefill with multi-layer TTT-MLP.
    
    This is a simplified version - full implementation would need causal attention
    across all layers similar to the 2-layer version.
    """
    # For now, delegate to simpler approach or implement full causal logic
    # This is a placeholder that needs full implementation
    raise NotImplementedError("Multi-layer prefill mini-batch not yet implemented for cache-based inference")
