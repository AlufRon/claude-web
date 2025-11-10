import torch
import os


def ln_fwd(x, gamma, beta, eps=1e-8, layer_id=None):
    "Batch forward for LayerNorm."
    
    # Capture input norm BEFORE normalization (for debugging)
    debug_step = int(os.environ.get('TTT_NORM_DEBUG_STEP', '-1'))
    if debug_step > 0 and layer_id is not None:
        with torch.no_grad():
            pre_norm = x.norm(p=2, dim=-1).mean().item()
            gamma_min = gamma.min().item()
            gamma_max = gamma.max().item()
            gamma_mean = gamma.mean().item()
            x_shape = x.shape
            gamma_shape = gamma.shape
            print(f"[TTT-NORM-LN-SHAPES] Step={debug_step} Layer={layer_id} | x_shape={x_shape}, gamma_shape={gamma_shape}")
            print(f"[TTT-NORM-LN-GAMMA-DEBUG] Step={debug_step} Layer={layer_id} | Gamma: min={gamma_min:.6f}, max={gamma_max:.6f}, mean={gamma_mean:.6f}")

    # Mean and variance computation
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)

    # Normalization
    std = torch.sqrt(var + eps)
    x_hat = (x - mu) / std

    # Scale and shift
    y = gamma * x_hat + beta

    # Log if debug step is set
    if debug_step > 0 and layer_id is not None:
        with torch.no_grad():
            post_norm = y.norm(p=2, dim=-1).mean().item()
            gamma_mean = gamma.mean().item()
            print(f"[TTT-NORM-LN] Step={debug_step} Layer={layer_id} | Pre={pre_norm:.3f} | Post={post_norm:.3f} | Gamma={gamma_mean:.3f}")

    # MINIMAL LOG: Warn if output norm is suspiciously large (should be ~1.0)
    if layer_id is not None:
        with torch.no_grad():
            post_norm = y.norm(p=2, dim=-1).mean().item()
            if post_norm > 10.0:  # Should be ~1.0, warn if >10x
                print(f"⚠️  [TTT-LAYER-NORM-LARGE] Layer={layer_id} | Output_norm={post_norm:.1f} (expected ~1.0)")

    return y


def ln_fused_l2_bwd(x, l2_target, gamma, beta, eps=1e-8):
    "Batch backward for LayerNorm fused with L2 loss."
    D = x.shape[-1]

    # Mean and variance computation
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)

    # Normalization
    std = torch.sqrt(var + eps)
    x_hat = (x - mu) / std

    # Scale and shift
    y = gamma * x_hat + beta

    grad_output = y - l2_target
    grad_x_hat = grad_output * gamma
    z = (
        (1.0 / D)
        * (
            D * grad_x_hat
            - grad_x_hat.sum(dim=-1, keepdim=True)
            - x_hat * (grad_x_hat * x_hat).sum(dim=-1, keepdim=True)
        )
        / std
    )

    return z


def gelu_bwd(x):
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (1 + tanh_out)
    return ff
