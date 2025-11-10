import torch
import torch.nn.functional as F
from torch.utils._pytree import tree_map

from .utils import gelu_bwd, ln_fused_l2_bwd, ln_fwd
from ..utils import scan


# Global flag for inner loop loss logging (Figure 4)
_global_log_inner_loop_losses = False
_collected_inner_loop_losses = {}  # Stores losses per layer during evaluation


def set_inner_loop_logging(enabled: bool):
    """Enable/disable inner loop loss logging globally."""
    global _global_log_inner_loop_losses, _collected_inner_loop_losses
    _global_log_inner_loop_losses = enabled
    if enabled:
        _collected_inner_loop_losses = {}


def get_collected_inner_loop_losses():
    """Get all collected inner loop losses and reset."""
    global _collected_inner_loop_losses
    losses = _collected_inner_loop_losses
    _collected_inner_loop_losses = {}
    return losses


def get_collected_inner_loop_losses_for_layer(layer_id: int):
    """Get collected inner loop losses for a specific layer."""
    global _collected_inner_loop_losses
    return _collected_inner_loop_losses.get(layer_id, [])


# ==== Figure 5 logging (per-layer, per-position) ====
_inner_fig5_enabled = False
_inner_fig5_max_T = 2048
_inner_fig5 = {}  # layer_id -> {'l0': [sum]*T, 'lprev':[sum]*T, 'lafter':[sum]*T, 'cnt':[int]*T}


def fig5_set_logging(enabled: bool, max_T: int = 2048):
    """Turn on/off Figure 5 position-indexed logging."""
    global _inner_fig5_enabled, _inner_fig5_max_T, _inner_fig5
    _inner_fig5_enabled = enabled
    _inner_fig5_max_T = max_T
    if enabled:
        _inner_fig5 = {}
        print(f"[Figure 5] Logging enabled: max_T={max_T}")
    else:
        print(f"[Figure 5] Logging disabled")


def fig5_clear():
    """Clear Figure 5 accumulated data."""
    global _inner_fig5
    _inner_fig5.clear()


def fig5_get():
    """Return raw accumulators for plotting."""
    return _inner_fig5


def _fig5_ensure(layer_id: int):
    """Ensure buffers exist for a layer."""
    if layer_id not in _inner_fig5:
        T = _inner_fig5_max_T
        _inner_fig5[layer_id] = {
            'l0':    [0.0] * T,
            'lprev': [0.0] * T,
            'lafter': [0.0] * T,
            'cnt':   [0] * T,
        }


def _fig5_log(layer_id: int, t: int, l0: float, lprev: float, lafter: float):
    """Log three losses for a specific layer and token position."""
    if (not _inner_fig5_enabled) or (t >= _inner_fig5_max_T) or (t < 0):
        return
    _fig5_ensure(layer_id)
    d = _inner_fig5[layer_id]
    d['l0'][t]     += float(l0)
    d['lprev'][t]  += float(lprev)
    d['lafter'][t] += float(lafter)
    d['cnt'][t]    += 1


def _recon_loss_with_params(X1, target, W1, b1, W2, b2, ln_weight, ln_bias):
    """Compute reconstruction loss with arbitrary parameters (no state mutation)."""
    Z1 = X1 @ W1 + b1
    X2 = F.gelu(Z1, approximate="tanh")
    Z2 = X2 @ W2 + b2
    Z2_normalized = ln_fwd(Z2, ln_weight, ln_bias, layer_id=None)
    return F.mse_loss(Z2_normalized, target, reduction='mean')


def _recon_loss_multi_layer(X0, target, weight_list, bias_list, ln_weight, ln_bias):
    """Compute reconstruction loss for multi-layer MLP with arbitrary parameters."""
    num_layers = len(weight_list)
    X = X0

    for i in range(num_layers):
        Z = X @ weight_list[i] + bias_list[i]
        if i < num_layers - 1:
            X = F.gelu(Z, approximate="tanh")
        else:
            Z_normalized = ln_fwd(Z, ln_weight, ln_bias, layer_id=None)
            return F.mse_loss(Z_normalized, target, reduction='mean')


# @torch.compile  # Disabled to prevent CUDA OOM
def compute_mini_batch(params_dict, inputs, log_losses=None, layer_id=None):
    """
    Compute one mini-batch of TTT updates.
    
    Returns:
        If log_losses=False (default): (last_param_dict, XQW_mini_batch)
        If log_losses=True: (last_param_dict, XQW_mini_batch, reconstruction_loss)
    """
    # Cast to float32 for numerical stability
    input_dtype = params_dict["W1_states"].dtype
    
    W1_init = params_dict["W1_states"].to(torch.float32)
    b1_init = params_dict["b1_states"].to(torch.float32)
    W2_init = params_dict["W2_states"].to(torch.float32)
    b2_init = params_dict["b2_states"].to(torch.float32)

    ttt_norm_weight = params_dict["ttt_norm_weight"].to(torch.float32)
    ttt_norm_bias = params_dict["ttt_norm_bias"].to(torch.float32)

    XQ_mini_batch = inputs["XQ"].to(torch.float32)
    XV_mini_batch = inputs["XV"].to(torch.float32)
    XK_mini_batch = inputs["XK"].to(torch.float32)
    eta_mini_batch = inputs["eta"].to(torch.float32)

    num_heads = XQ_mini_batch.size(1)
    head_dim = XQ_mini_batch.size(-1)

    X1 = XK_mini_batch
    reconstruction_target = XV_mini_batch - XK_mini_batch

    ln_weight = ttt_norm_weight.reshape(num_heads, 1, head_dim)
    ln_bias = ttt_norm_bias.reshape(num_heads, 1, head_dim)
    
    # Use global flag if log_losses not explicitly specified
    if log_losses is None:
        log_losses = _global_log_inner_loop_losses
    
    # ==== Figure 5: Compute three losses (l0, lprev, lafter) ====
    if _inner_fig5_enabled and layer_id is not None:
        pos_t = inputs.get("pos_t", -1)
        if isinstance(pos_t, torch.Tensor):
            pos_t = int(pos_t.item())

        if pos_t >= 0:
            # (A) lprev: ℓ(Wₜ₋₁; xₜ) - loss BEFORE update
            lprev = _recon_loss_with_params(
                X1, reconstruction_target,
                W1_init, b1_init, W2_init, b2_init,
                ln_weight, ln_bias
            )
            
            # (B) l0: ℓ(W₀; xₜ) - loss with frozen initial weights
            W1_0 = inputs.get("W1_0", W1_init)
            b1_0 = inputs.get("b1_0", b1_init)
            W2_0 = inputs.get("W2_0", W2_init)
            b2_0 = inputs.get("b2_0", b2_init)
            
            if "W1_0" in inputs:
                W1_0 = W1_0.to(torch.float32)
                b1_0 = b1_0.to(torch.float32)
                W2_0 = W2_0.to(torch.float32)
                b2_0 = b2_0.to(torch.float32)
            
            l0 = _recon_loss_with_params(
                X1, reconstruction_target,
                W1_0, b1_0, W2_0, b2_0,
                ln_weight, ln_bias
            )
    
    # Optionally compute reconstruction loss for Figure 4
    if log_losses:
        Z1 = X1 @ W1_init + b1_init
        X2 = F.gelu(Z1, approximate="tanh")
        Z2 = X2 @ W2_init + b2_init
        Z2_normalized = ln_fwd(Z2, ln_weight, ln_bias, layer_id=layer_id)
        reconstruction_loss = F.mse_loss(Z2_normalized, reconstruction_target).item()
        
        global _collected_inner_loop_losses
        if layer_id is not None:
            if layer_id not in _collected_inner_loop_losses:
                _collected_inner_loop_losses[layer_id] = []
            _collected_inner_loop_losses[layer_id].append(reconstruction_loss)
        else:
            if 'global' not in _collected_inner_loop_losses:
                _collected_inner_loop_losses['global'] = []
            _collected_inner_loop_losses['global'].append(reconstruction_loss)
    
    # ==== Gradient computation and update ====
    Z1 = X1 @ W1_init + b1_init
    X2 = F.gelu(Z1, approximate="tanh")
    Z2 = X2 @ W2_init + b2_init
    
    grad_l_wrt_Z2 = ln_fused_l2_bwd(Z2, reconstruction_target, ln_weight, ln_bias)
    grad_l_wrt_Z1 = grad_l_wrt_Z2 @ W2_init.transpose(-2, -1) * gelu_bwd(Z1)

    Attn1 = XQ_mini_batch @ X1.transpose(-2, -1)
    b1_bar = b1_init - eta_mini_batch @ grad_l_wrt_Z1
    Z1_bar = XQ_mini_batch @ W1_init - (eta_mini_batch * Attn1) @ grad_l_wrt_Z1 + b1_bar
    X2_bar = F.gelu(Z1_bar, approximate="tanh")

    Attn2 = X2_bar @ X2.transpose(-2, -1)
    b2_bar = b2_init - eta_mini_batch @ grad_l_wrt_Z2
    Z2_bar = X2_bar @ W2_init - (eta_mini_batch * Attn2) @ grad_l_wrt_Z2 + b2_bar

    last_eta_mini_batch = eta_mini_batch[:, :, -1, :, None]
    W1_last = W1_init - (last_eta_mini_batch * X1).transpose(-1, -2) @ grad_l_wrt_Z1
    b1_last = b1_init - torch.sum(last_eta_mini_batch * grad_l_wrt_Z1, dim=-2, keepdim=True)
    W2_last = W2_init - (last_eta_mini_batch * X2).transpose(-1, -2) @ grad_l_wrt_Z2
    b2_last = b2_init - torch.sum(last_eta_mini_batch * grad_l_wrt_Z2, dim=-2, keepdim=True)

    # ==== Figure 5: Compute lafter - loss AFTER update ====
    if _inner_fig5_enabled and layer_id is not None:
        pos_t = inputs.get("pos_t", -1)
        if isinstance(pos_t, torch.Tensor):
            pos_t = int(pos_t.item())

        if pos_t >= 0:
            # (C) lafter: ℓ(Wₜ; xₜ) - loss with updated weights
            lafter = _recon_loss_with_params(
                X1, reconstruction_target,
                W1_last, b1_last, W2_last, b2_last,
                ln_weight, ln_bias
            )
            
            # Log all three losses
            _fig5_log(layer_id, pos_t, l0.item(), lprev.item(), lafter.item())

    Z2_bar = ln_fwd(Z2_bar, ln_weight, ln_bias, layer_id=layer_id)
    XQW_mini_batch = XQ_mini_batch + Z2_bar

    # Cast outputs back to original dtype
    last_param_dict = {
        "W1_states": W1_last.to(input_dtype),
        "b1_states": b1_last.to(input_dtype),
        "W2_states": W2_last.to(input_dtype),
        "b2_states": b2_last.to(input_dtype),
        "ttt_norm_weight": ttt_norm_weight.to(input_dtype),
        "ttt_norm_bias": ttt_norm_bias.to(input_dtype),
    }

    if log_losses:
        return last_param_dict, XQW_mini_batch.to(input_dtype), reconstruction_loss
    else:
        return last_param_dict, XQW_mini_batch.to(input_dtype)


def ttt_mlp(XK, XQ, XV, eta, ttt_norm_weight, ttt_norm_bias, W1_init, b1_init, W2_init, b2_init, checkpoint_group_size, log_losses=False, layer_id=None, stream_pos_base=0):
    """
    TTT-MLP forward pass.
    
    Returns:
        If log_losses=False (default): XQW_batch
        If log_losses=True: (XQW_batch, losses_list)
    """
    init_params_dict = {
        "W1_states": W1_init,
        "b1_states": b1_init,
        "W2_states": W2_init,
        "b2_states": b2_init,
        "ttt_norm_weight": ttt_norm_weight,
        "ttt_norm_bias": ttt_norm_bias,
    }

    inputs = {
        "XK": XK,
        "XQ": XQ,
        "XV": XV,
        "eta": eta,
    }

    # Reorder for iteration
    inputs = tree_map(lambda x: x.permute(2, 0, 1, 3, 4), inputs)
    
    # For Figure 5: add frozen W₀ and position tracking
    if _inner_fig5_enabled and layer_id is not None:
        num_mini_batches = inputs["XK"].shape[0]
        inputs["W1_0"] = W1_init.detach().expand(num_mini_batches, -1, -1, -1, -1)
        inputs["b1_0"] = b1_init.detach().expand(num_mini_batches, -1, -1, -1, -1)
        inputs["W2_0"] = W2_init.detach().expand(num_mini_batches, -1, -1, -1, -1)
        inputs["b2_0"] = b2_init.detach().expand(num_mini_batches, -1, -1, -1, -1)
        pos_indices = torch.arange(stream_pos_base, stream_pos_base + num_mini_batches,
                                   device=XK.device, dtype=torch.long)
        inputs["pos_t"] = pos_indices

    XQW_batch = torch.empty_like(inputs["XK"])

    if log_losses:
        losses = []
        
        def compute_mini_batch_with_losses(params_dict, inputs_batch):
            result = compute_mini_batch(params_dict, inputs_batch, log_losses=True, layer_id=layer_id)
            losses.append(result[2])
            return result[0], result[1]
        
        _, XQW_batch = scan(
            compute_mini_batch_with_losses,
            init_params_dict,
            inputs,
            checkpoint_group_size,
        )
        
        return XQW_batch.permute(1, 0, 3, 2, 4), losses
    else:
        _, XQW_batch = scan(
            lambda params_dict, inputs_batch: compute_mini_batch(params_dict, inputs_batch, log_losses=False, layer_id=layer_id),
            init_params_dict,
            inputs,
            checkpoint_group_size,
        )

        return XQW_batch.permute(1, 0, 3, 2, 4)


def ttt_mlp_with_states(XK, XQ, XV, eta, ttt_norm_weight, ttt_norm_bias, W1_init, b1_init, W2_init, b2_init, checkpoint_group_size, log_losses=False, layer_id=None, stream_pos_base=0):
    """
    TTT-MLP that returns both output and updated states for persistence.
    
    Returns:
        If log_losses=False (default): (XQW_batch, final_params_dict)
        If log_losses=True: (XQW_batch, final_params_dict, losses_list)
    """
    init_params_dict = {
        "W1_states": W1_init,
        "b1_states": b1_init,
        "W2_states": W2_init,
        "b2_states": b2_init,
        "ttt_norm_weight": ttt_norm_weight,
        "ttt_norm_bias": ttt_norm_bias,
    }

    inputs = {
        "XK": XK,
        "XQ": XQ,
        "XV": XV,
        "eta": eta,
    }

    # Reorder for iteration
    inputs = tree_map(lambda x: x.permute(2, 0, 1, 3, 4), inputs)
    
    # For Figure 5: add frozen W₀ and position tracking
    if _inner_fig5_enabled and layer_id is not None:
        num_mini_batches = inputs["XK"].shape[0]
        inputs["W1_0"] = W1_init.detach().expand(num_mini_batches, -1, -1, -1, -1)
        inputs["b1_0"] = b1_init.detach().expand(num_mini_batches, -1, -1, -1, -1)
        inputs["W2_0"] = W2_init.detach().expand(num_mini_batches, -1, -1, -1, -1)
        inputs["b2_0"] = b2_init.detach().expand(num_mini_batches, -1, -1, -1, -1)
        pos_indices = torch.arange(stream_pos_base, stream_pos_base + num_mini_batches,
                                   device=XK.device, dtype=torch.long)
        inputs["pos_t"] = pos_indices

    XQW_batch = torch.empty_like(inputs["XK"])

    if log_losses:
        losses = []
        
        def compute_mini_batch_with_losses(params_dict, inputs_batch):
            result = compute_mini_batch(params_dict, inputs_batch, log_losses=True, layer_id=layer_id)
            losses.append(result[2])
            return result[0], result[1]
        
        final_params_dict, XQW_batch = scan(
            compute_mini_batch_with_losses,
            init_params_dict,
            inputs,
            checkpoint_group_size,
        )
        
        return XQW_batch.permute(1, 0, 3, 2, 4), final_params_dict, losses
    else:
        final_params_dict, XQW_batch = scan(
            lambda params_dict, inputs_batch: compute_mini_batch(params_dict, inputs_batch, log_losses=False, layer_id=layer_id),
            init_params_dict,
            inputs,
            checkpoint_group_size,
        )

        return XQW_batch.permute(1, 0, 3, 2, 4), final_params_dict


def compute_multi_layer_mini_batch(params_dict, inputs, log_losses=None, layer_id=None):
    """
    Compute one mini-batch of TTT updates for N-layer MLP.

    Returns:
        If log_losses=False (default): (last_param_dict, XQW_mini_batch)
        If log_losses=True: (last_param_dict, XQW_mini_batch, reconstruction_loss)
    """
    # Extract parameters
    ttt_norm_weight = params_dict["ttt_norm_weight"]
    ttt_norm_bias = params_dict["ttt_norm_bias"]
    weight_states = params_dict["weight_states"]
    bias_states = params_dict["bias_states"]
    num_layers = len(weight_states)
    
    XQ_mini_batch = inputs["XQ"]
    XV_mini_batch = inputs["XV"]
    XK_mini_batch = inputs["XK"]
    eta_mini_batch = inputs["eta"]

    num_heads = XQ_mini_batch.size(1)
    head_dim = XQ_mini_batch.size(-1)
    
    # Forward pass through N layers
    layer_inputs = [XK_mini_batch]
    layer_outputs = []
    activations = []
    
    for i in range(num_layers):
        Zi = layer_inputs[i] @ weight_states[i] + bias_states[i]
        layer_outputs.append(Zi)
        
        if i < num_layers - 1:
            Xi_plus_1 = F.gelu(Zi, approximate="tanh")
            activations.append(Xi_plus_1)
            layer_inputs.append(Xi_plus_1)
    
    final_Z = layer_outputs[-1]
    reconstruction_target = XV_mini_batch - XK_mini_batch
    
    ln_weight = ttt_norm_weight.reshape(num_heads, 1, head_dim)
    ln_bias = ttt_norm_bias.reshape(num_heads, 1, head_dim)
    
    if log_losses is None:
        log_losses = _global_log_inner_loop_losses
    
    if log_losses:
        final_Z_normalized = ln_fwd(final_Z, ln_weight, ln_bias, layer_id=layer_id)
        reconstruction_loss = F.mse_loss(final_Z_normalized, reconstruction_target).item()
        global _collected_inner_loop_losses
        _collected_inner_loop_losses.append(reconstruction_loss)
    
    # Backward pass
    gradients = [None] * num_layers
    gradients[-1] = ln_fused_l2_bwd(final_Z, reconstruction_target, ln_weight, ln_bias)
    
    for i in range(num_layers - 2, -1, -1):
        grad_next = gradients[i + 1]
        weight_next = weight_states[i + 1]
        activation_current = layer_outputs[i]
        gradients[i] = grad_next @ weight_next.transpose(-2, -1) * gelu_bwd(activation_current)
    
    # Update parameters
    updated_weights = []
    updated_biases = []
    last_eta_mini_batch = eta_mini_batch[:, :, -1, :, None]
    
    for i in range(num_layers):
        Xi = layer_inputs[i]
        grad_Zi = gradients[i]
        
        Wi_last = weight_states[i] - (last_eta_mini_batch * Xi).transpose(-1, -2) @ grad_Zi
        bi_last = bias_states[i] - torch.sum(last_eta_mini_batch * grad_Zi, dim=-2, keepdim=True)
        
        updated_weights.append(Wi_last)
        updated_biases.append(bi_last)
    
    # Forward pass with updated parameters
    updated_inputs = [XQ_mini_batch]
    
    for i in range(num_layers):
        if i == 0:
            Attn_i = XQ_mini_batch @ layer_inputs[i].transpose(-2, -1)
            bi_bar = updated_biases[i] - eta_mini_batch @ gradients[i]
            Zi_bar = XQ_mini_batch @ updated_weights[i] - (eta_mini_batch * Attn_i) @ gradients[i] + bi_bar
        else:
            Xi = updated_inputs[i]
            Attn_i = Xi @ activations[i-1].transpose(-2, -1) if i > 0 else None
            bi_bar = updated_biases[i] - eta_mini_batch @ gradients[i]
            
            if Attn_i is not None:
                Zi_bar = Xi @ updated_weights[i] - (eta_mini_batch * Attn_i) @ gradients[i] + bi_bar
            else:
                Zi_bar = Xi @ updated_weights[i] + bi_bar
        
        if i < num_layers - 1:
            Xi_plus_1_bar = F.gelu(Zi_bar, approximate="tanh")
            updated_inputs.append(Xi_plus_1_bar)
        else:
            final_Z_bar = ln_fwd(Zi_bar, ln_weight, ln_bias, layer_id=layer_id)
    
    XQW_mini_batch = XQ_mini_batch + final_Z_bar
    
    last_param_dict = {
        "weight_states": updated_weights,
        "bias_states": updated_biases,
        "ttt_norm_weight": ttt_norm_weight,
        "ttt_norm_bias": ttt_norm_bias,
    }
    
    if log_losses:
        return last_param_dict, XQW_mini_batch, reconstruction_loss
    else:
        return last_param_dict, XQW_mini_batch


def ttt_mlp_multi_layer(XK, XQ, XV, eta, ttt_norm_weight, ttt_norm_bias, weight_states, bias_states, checkpoint_group_size, log_losses=False, layer_id=None, stream_pos_base=0, return_updated_weights=False):
    """
    TTT-MLP forward pass for N layers.

    Returns:
        If log_losses=False and return_updated_weights=False: XQW_batch
        If log_losses=True: (XQW_batch, losses_list)
        If return_updated_weights=True: (XQW_batch, updated_weight_states, updated_bias_states)
    """
    init_params_dict = {
        "weight_states": weight_states,
        "bias_states": bias_states,
        "ttt_norm_weight": ttt_norm_weight,
        "ttt_norm_bias": ttt_norm_bias,
    }

    inputs = {
        "XK": XK,
        "XQ": XQ,
        "XV": XV,
        "eta": eta,
    }

    # Reorder for iteration
    inputs = tree_map(lambda x: x.permute(2, 0, 1, 3, 4), inputs)

    # For Figure 5: add frozen W₀ and position tracking
    if _inner_fig5_enabled and layer_id is not None:
        num_mini_batches = inputs["XK"].shape[0]
        inputs["weight_states_0"] = [w.detach().expand(num_mini_batches, -1, -1, -1, -1) for w in weight_states]
        inputs["bias_states_0"] = [b.detach().expand(num_mini_batches, -1, -1, -1, -1) for b in bias_states]
        pos_indices = torch.arange(stream_pos_base, stream_pos_base + num_mini_batches,
                                   device=XK.device, dtype=torch.long)
        inputs["pos_t"] = pos_indices

    XQW_batch = torch.empty_like(inputs["XK"])

    if log_losses:
        losses = []
        
        def compute_multi_layer_mini_batch_with_losses(params_dict, inputs_batch):
            result = compute_multi_layer_mini_batch(params_dict, inputs_batch, log_losses=True)
            losses.append(result[2])
            return result[0], result[1]
        
        final_params, XQW_batch = scan(
            compute_multi_layer_mini_batch_with_losses,
            init_params_dict,
            inputs,
            checkpoint_group_size,
        )
        
        return XQW_batch.permute(1, 0, 3, 2, 4), losses
    else:
        final_params, XQW_batch = scan(
            compute_multi_layer_mini_batch,
            init_params_dict,
            inputs,
            checkpoint_group_size,
        )

        XQW_batch_reordered = XQW_batch.permute(1, 0, 3, 2, 4)
        
        if return_updated_weights:
            updated_weight_states = final_params["weight_states"]
            updated_bias_states = final_params["bias_states"]
            return XQW_batch_reordered, updated_weight_states, updated_bias_states
        else:
            return XQW_batch_reordered