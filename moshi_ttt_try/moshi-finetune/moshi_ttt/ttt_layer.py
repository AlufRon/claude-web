"""
TTT Layer implementation using Video-DiT's EXACT approach with proper gradient handling.
This fixes the NaN issues by using Video-DiT's proven scan-based implementation.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
from torch.utils._pytree import tree_map
from .config import TTTConfig
from .format_utils import SequenceMetadata
from .models.ssm.ops.ttt_mlp import ttt_mlp_with_states, _global_log_inner_loop_losses
from .models.ssm.ops.ttt_mlp import ttt_mlp as ttt_mlp_ops

logger = logging.getLogger(__name__)


# ============================================================================
# Rotary Position Embeddings (RoPE) - From ttt-lm-pytorch
# Only used when config.use_rope=True
# Reference: /home/alufr/ttt_tests/ttt-lm-pytorch/ttt.py lines 118-270
# ============================================================================

class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) - Exact copy from ttt-lm-pytorch.

    CRITICAL: Only needs max_position_embeddings=mini_batch_size
    because TTT-LM uses position_ids % mini_batch_size (position modulo trick).

    This prevents extrapolation issues on long sequences by keeping positions
    bounded to [0, mini_batch_size).
    """
    def __init__(self, dim, max_position_embeddings=16, base=10000, device=None, scaling_factor=1.0):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Compute inverse frequencies: inv_freq[i] = 1 / (base^(2i/dim))
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        """Precompute cos/sin cache for all positions"""
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)
        t = t / self.scaling_factor

        # Compute frequencies: freqs[i] = position * inv_freq[i]
        freqs = torch.outer(t, self.inv_freq)

        # Different from paper, but matches ttt-lm-pytorch exactly
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, position_ids):
        """
        Args:
            x: Input tensor (used for device/dtype reference)
            position_ids: Position indices [B, seq_len] or [seq_len]
                         CRITICAL: Should already have modulo applied!

        Returns:
            cos, sin: Cosine and sine values [seq_len, head_dim]
        """
        if position_ids is None:
            # Default: sequential positions
            # Infer sequence length from input shape
            if x.dim() == 5:  # [B, H, NC, C, HD]
                seq_len = x.shape[2] * x.shape[3]  # NC * C
            elif x.dim() == 3:  # [B, seq_len, dim]
                seq_len = x.shape[1]
            else:
                seq_len = x.shape[1]
            position_ids = torch.arange(seq_len, device=x.device, dtype=torch.long)
        else:
            # Flatten position_ids if needed
            if position_ids.dim() > 1:
                position_ids = position_ids.flatten()

        # CRITICAL: Ensure position_ids are within bounds using modulo
        # This is safe because TTT-LM always uses position_ids % mini_batch_size
        # before passing to RoPE
        position_ids = position_ids % self.max_position_embeddings

        # Index into cached cos/sin using bounded position_ids
        cos = self.cos_cached[position_ids].to(dtype=x.dtype)  # [seq_len, head_dim]
        sin = self.sin_cached[position_ids].to(dtype=x.dtype)  # [seq_len, head_dim]

        return cos, sin


def rotate_half(x):
    """
    Rotates half the hidden dims of the input for RoPE.

    For RoPE, we split features into pairs and rotate them.
    Example: [a, b, c, d] -> [-c, -d, a, b]

    Args:
        x: Input tensor [..., dim]

    Returns:
        Rotated tensor [..., dim]
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """
    Apply Rotary Position Embedding to query and key tensors.

    Args:
        q: Query tensor [B, H, L, HD] or similar
        k: Key tensor (same shape as q)
        cos: Cosine values [L, HD]
        sin: Sine values [L, HD]
        position_ids: Not used (for API compatibility with HuggingFace)
        unsqueeze_dim: Controls unsqueezing (1 means unsqueeze at dims 0 and 1)

    Returns:
        q_embed, k_embed: Rotated query and key tensors

    Formula:
        RoPE(x) = x * cos + rotate_half(x) * sin
    """
    # Unsqueeze cos/sin for broadcasting to [B, H, L, HD]
    # From [L, HD] to [1, 1, L, HD]
    if unsqueeze_dim == 1:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
    else:
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)

    # Apply rotation: x * cos + rotate_half(x) * sin
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


# ============================================================================
# End of RoPE Implementation
# ============================================================================


def ln_fwd(x, gamma, beta, eps=1e-8):
    """Batch forward for LayerNorm (from Video-DiT)"""
    # Mean and variance computation
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)

    # Normalization
    std = torch.sqrt(var + eps)
    x_hat = (x - mu) / std

    # Scale and shift
    y = gamma * x_hat + beta

    return y


def ln_fused_l2_bwd(x, l2_target, gamma, beta, eps=1e-8):
    """Batch backward for LayerNorm fused with L2 loss (from Video-DiT)"""
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
    """GELU backward (from Video-DiT)"""
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (1 + tanh_out)
    return ff


def scan(f, init, xs, checkpoint_group=0):
    """Mimic jax.lax.scan function (from Video-DiT)"""
    carry = init
    if isinstance(xs, dict):
        num_items = len(next(iter(xs.values())))
    else:
        num_items = len(xs[0])

    def scan_fn(carry, i_start, i_end):
        sub_out_list = []
        for i in range(i_start, i_end):
            if isinstance(xs, dict):
                x = {key: tensor[i] for key, tensor in xs.items()}
            else:
                x = [x[i] for x in xs]
            carry, y = f(carry, x)
            sub_out_list.append(y)
        sub_out = torch.stack(sub_out_list)
        return carry, sub_out

    if checkpoint_group > 0:
        out_list = []
        for k in range(0, num_items, checkpoint_group):
            carry, sub_out = torch.utils.checkpoint.checkpoint(
                scan_fn,
                carry,
                k,
                min(k + checkpoint_group, num_items),
                use_reentrant=False,
            )
            out_list.append(sub_out)
        out = torch.concatenate(out_list, dim=0)
    else:
        carry, out = scan_fn(carry, 0, num_items)

    return carry, out


@torch.compile
def compute_mini_batch(params_dict, inputs):
    """
    Core TTT computation for one mini-batch (EXACT copy from Video-DiT).
    This is where the TTT magic happens with proper gradient computation.
    """
    W1_init = params_dict["W1_states"]
    b1_init = params_dict["b1_states"]
    W2_init = params_dict["W2_states"]
    b2_init = params_dict["b2_states"]

    ttt_norm_weight = params_dict["ttt_norm_weight"]
    ttt_norm_bias = params_dict["ttt_norm_bias"]

    XQ_mini_batch = inputs["XQ"]
    XV_mini_batch = inputs["XV"]
    XK_mini_batch = inputs["XK"]

    eta_mini_batch = inputs["eta"]

    num_heads = XQ_mini_batch.size(1)
    head_dim = XQ_mini_batch.size(-1)

    # Forward pass through TTT-MLP with current weights
    X1 = XK_mini_batch
    Z1 = X1 @ W1_init + b1_init
    X2 = F.gelu(Z1, approximate="tanh")
    Z2 = X2 @ W2_init + b2_init
    reconstruction_target = XV_mini_batch - XK_mini_batch

    # Compute gradients for TTT updates
    ln_weight = ttt_norm_weight.reshape(num_heads, 1, head_dim)
    ln_bias = ttt_norm_bias.reshape(num_heads, 1, head_dim)
    grad_l_wrt_Z2 = ln_fused_l2_bwd(Z2, reconstruction_target, ln_weight, ln_bias)
    grad_l_wrt_Z1 = grad_l_wrt_Z2 @ W2_init.transpose(-2, -1) * gelu_bwd(Z1)

    # Compute TTT output using attention mechanism (EXACT Video-DiT implementation)
    Attn1 = XQ_mini_batch @ X1.transpose(-2, -1)
    # Fix tensor shape for matrix multiplication - eta should be [B, H, 1, C]
    eta_mini_batch_fixed = eta_mini_batch.transpose(-2, -1)  # [B, H, C, 1] -> [B, H, 1, C]
    b1_bar = b1_init - eta_mini_batch_fixed @ grad_l_wrt_Z1
    Z1_bar = XQ_mini_batch @ W1_init - (eta_mini_batch * Attn1) @ grad_l_wrt_Z1 + b1_bar
    X2_bar = F.gelu(Z1_bar, approximate="tanh")

    Attn2 = X2_bar @ X2.transpose(-2, -1)
    b2_bar = b2_init - eta_mini_batch_fixed @ grad_l_wrt_Z2
    Z2_bar = X2_bar @ W2_init - (eta_mini_batch * Attn2) @ grad_l_wrt_Z2 + b2_bar

    # Update TTT weights for next mini-batch
    last_eta_mini_batch = eta_mini_batch[:, :, -1, :, None]
    W1_last = W1_init - (last_eta_mini_batch * X1).transpose(-1, -2) @ grad_l_wrt_Z1
    b1_last = b1_init - torch.sum(last_eta_mini_batch * grad_l_wrt_Z1, dim=-2, keepdim=True)
    W2_last = W2_init - (last_eta_mini_batch * X2).transpose(-1, -2) @ grad_l_wrt_Z2
    b2_last = b2_init - torch.sum(last_eta_mini_batch * grad_l_wrt_Z2, dim=-2, keepdim=True)

    # Apply layer norm and residual connection
    Z2_bar = ln_fwd(Z2_bar, ln_weight, ln_bias)
    XQW_mini_batch = XQ_mini_batch + Z2_bar

    # Return updated parameters and output
    last_param_dict = {
        "W1_states": W1_last,
        "b1_states": b1_last,
        "W2_states": W2_last,
        "b2_states": b2_last,
        "ttt_norm_weight": ttt_norm_weight,
        "ttt_norm_bias": ttt_norm_bias,
    }

    return last_param_dict, XQW_mini_batch

def ttt_mlp(XK, XQ, XV, eta, ttt_norm_weight, ttt_norm_bias, W1_init, b1_init, W2_init, b2_init, checkpoint_group_size):
    """
    TTT-MLP implementation using scan (EXACT copy from Video-DiT).
    This is the entry point that orchestrates the sequential TTT computation.
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

    # Reorder such that mini-batch is first dimension for iteration
    inputs = tree_map(lambda x: x.permute(2, 0, 1, 3, 4), inputs)

    # Use scan for sequential processing with proper gradient handling
    _, XQW_batch = scan(
        compute_mini_batch,  # Function to iterate over
        init_params_dict,
        inputs,
        checkpoint_group_size,
    )

    return XQW_batch.permute(1, 0, 3, 2, 4)


class TTTBase(nn.Module):
    """Base TTT class using Video-DiT's exact approach"""
    
    def __init__(self, config: TTTConfig, layer_id: int = None):
        super().__init__()
        self.config = config
        self.layer_id = layer_id  # Store layer ID for per-layer loss tracking
        self.width = config.model_dim
        self.num_heads = config.num_heads
        self.head_dim = config.model_dim // config.num_heads
        self.mini_batch_size = config.mini_batch_size
        self.ttt_base_lr = config.ttt_base_lr
        self.scan_checkpoint_group_size = max(config.mini_batch_size // 4, 1)  # Video-DiT default

        # Conditional RoPE initialization (only when enabled)
        if config.use_rope:
            self.rotary_emb = RotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.mini_batch_size,  # CRITICAL: Only mini_batch_size positions needed!
                base=config.rope_theta,
            )
            logger.info(f"[TTTBase Layer {layer_id}] RoPE enabled: theta={config.rope_theta}, max_pos={self.mini_batch_size}")
        else:
            self.rotary_emb = None  # Explicit None when disabled

        self._init_qkvo_proj()
        self._init_ttt_lr_gate()
        self._init_ttt_ln()
        self.post_norm = nn.LayerNorm(self.width, eps=1e-6)

    def _init_qkvo_proj(self):
        """Initialize Q/K/V/O projections (Video-DiT pattern)"""
        self.wq = nn.Linear(self.width, self.num_heads * self.head_dim, bias=True)
        self.wk = nn.Linear(self.width, self.num_heads * self.head_dim, bias=True)
        self.wv = nn.Linear(self.width, self.num_heads * self.head_dim, bias=True)
        self.wo = nn.Linear(self.width, self.num_heads * self.head_dim, bias=True)

    def _init_ttt_lr_gate(self):
        """Initialize TTT learning rate gating (Video-DiT pattern)"""
        # Create temporary Linear layer to get proper initialization
        linear_weight_data = nn.Linear(self.width, 1, bias=True).weight.data
        linear_bias_data = nn.Linear(self.width, 1, bias=True).bias.data
        
        self.learnable_ttt_lr_weight = nn.Parameter(
            torch.stack(
                [torch.normal(0, 0.02, size=linear_weight_data.shape) for _ in range(self.num_heads)],
                dim=0,
            )
        )
        self.learnable_ttt_lr_bias = nn.Parameter(
            torch.stack(
                [torch.normal(0, 0.02, size=linear_bias_data.shape) for _ in range(self.num_heads)],
                dim=0,
            )
        )

    def _init_ttt_ln(self):
        """Initialize TTT layer norm parameters (Video-DiT pattern)"""
        self.ttt_norm_weight = nn.Parameter(torch.ones(self.num_heads, self.head_dim))
        self.ttt_norm_bias = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))

    def init_weights(self):
        """Initialize weights following Video-DiT pattern"""
        for linear in (self.wq, self.wk, self.wv):
            nn.init.normal_(linear.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.wo.weight, mean=0.0, std=0.02)
        self.post_norm.reset_parameters()
        nn.init.ones_(self.ttt_norm_weight.data)
        nn.init.zeros_(self.ttt_norm_bias)
        nn.init.normal_(self.learnable_ttt_lr_weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.learnable_ttt_lr_bias)

    def get_qkv_projections(self, hidden_states):
        """Get Q, K, V projections following Video-DiT pattern"""
        XQ, XK, XV = (
            self.wq(hidden_states),
            self.wk(hidden_states),
            self.wv(hidden_states),
        )
        return XQ, XK, XV

    def get_eta(self, X):
        """Compute TTT learning rate following Video-DiT pattern exactly"""
        B, NC, C, d_model = X.shape
        
        # Reshape for linear projection: [B, NC, C, d_model] -> [B*NC*C, d_model]
        X_flat = X.reshape(-1, d_model)
        
        # Apply learnable learning rate projection for each head
        eta_list = []
        for h in range(self.num_heads):
            # Apply linear transformation: [B*NC*C, d_model] -> [B*NC*C, 1]
            eta_h = F.linear(X_flat, self.learnable_ttt_lr_weight[h], self.learnable_ttt_lr_bias[h])
            eta_list.append(eta_h)
        
        # Stack heads: [num_heads, B*NC*C, 1] -> [B*NC*C, num_heads, 1]
        eta = torch.stack(eta_list, dim=1)
        
        # Reshape back: [B*NC*C, num_heads, 1] -> [B, NC, C, num_heads, 1] -> [B, num_heads, NC, C, 1]
        eta = eta.reshape(B, NC, C, self.num_heads, 1)
        eta = eta.permute(0, 3, 1, 2, 4)  # [B, num_heads, NC, C, 1]
        
        return eta

    def ln_reconstruction_target(self, XV, XK):
        """Layer norm reconstruction target following Video-DiT pattern"""
        B, L, num_heads, head_dim = XV.shape
        
        # Apply layer norm per head
        XV_normed = torch.zeros_like(XV)
        for h in range(num_heads):
            # Get weight and bias for this head
            weight = self.ttt_norm_weight[h]  # [head_dim]
            bias = self.ttt_norm_bias[h]      # [head_dim]
            
            # Apply layer norm: [B, L, head_dim]
            XV_h = XV[:, :, h, :]  
            XV_normed[:, :, h, :] = F.layer_norm(XV_h, (head_dim,), weight, bias, eps=1e-6)
        
        return XV_normed

    def process_input(self, hidden_states: torch.Tensor, seq_metadata: SequenceMetadata):
        """Process input following Video-DiT pattern"""
        B, L = hidden_states.shape[:2]
        mini_batch_size = self.mini_batch_size
        
        # Get Q/K/V projections
        XQ, XK, XV = self.get_qkv_projections(hidden_states)
        XQ = XQ.view(B, L, -1, self.head_dim)
        XK = XK.view(B, L, -1, self.head_dim)
        XV = XV.view(B, L, -1, self.head_dim)
        
        # L2 Norm (Video-DiT pattern)
        XQ = F.normalize(XQ, p=2, dim=-1)
        XK = F.normalize(XK, p=2, dim=-1)

        # ============================================================================
        # CONDITIONAL RoPE APPLICATION (following ttt-lm-pytorch pattern)
        # Applied AFTER L2 norm, BEFORE mini-batch reshaping
        # Reference: ttt-lm-pytorch line 869-874
        # ============================================================================
        if self.config.use_rope and self.rotary_emb is not None:
            # Generate position_ids (from metadata or default sequential)
            if hasattr(seq_metadata, 'position_ids') and seq_metadata.position_ids is not None:
                position_ids = seq_metadata.position_ids[0]  # [L] - take first batch item
            else:
                # Default: sequential positions [0, 1, 2, ..., L-1]
                position_ids = torch.arange(L, device=hidden_states.device, dtype=torch.long)

            # CRITICAL: Apply position modulo to keep positions in [0, mini_batch_size)
            # This prevents extrapolation issues on long sequences
            position_ids_bounded = position_ids % self.mini_batch_size

            # Reshape to [B, H, L, HD] for RoPE application (matching ttt-lm-pytorch)
            XQ_rope = XQ.permute(0, 2, 1, 3)  # [B, L, H, HD] -> [B, H, L, HD]
            XK_rope = XK.permute(0, 2, 1, 3)  # [B, L, H, HD] -> [B, H, L, HD]

            # Compute cos/sin using bounded positions
            # Pass 1D position_ids to avoid flattening issues
            cos, sin = self.rotary_emb(XQ_rope, position_ids_bounded)  # cos, sin: [L, HD]

            # Reshape cos/sin for broadcasting with [B, H, L, HD]
            cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, L, HD]
            sin = sin.unsqueeze(0).unsqueeze(0)  # [1, 1, L, HD]

            # Apply RoPE rotation to Q and K (NOT V!)
            XQ_rope = (XQ_rope * cos) + (rotate_half(XQ_rope) * sin)
            XK_rope = (XK_rope * cos) + (rotate_half(XK_rope) * sin)

            # Reshape back to [B, L, H, HD]
            XQ = XQ_rope.permute(0, 2, 1, 3)
            XK = XK_rope.permute(0, 2, 1, 3)
        # ============================================================================
        # End of RoPE application
        # ============================================================================

        # Apply layer norm reconstruction target
        XV = self.ln_reconstruction_target(XV, XK)

        # Convert to mini-batch format for TTT processing
        # Pad sequence to be divisible by mini_batch_size
        NC = (L + mini_batch_size - 1) // mini_batch_size
        padded_L = NC * mini_batch_size
        
        if padded_L > L:
            pad_len = padded_L - L
            XQ_pad = torch.zeros(B, pad_len, XQ.shape[2], self.head_dim, device=XQ.device, dtype=XQ.dtype)
            XK_pad = torch.zeros(B, pad_len, XK.shape[2], self.head_dim, device=XK.device, dtype=XK.dtype)
            XV_pad = torch.zeros(B, pad_len, XV.shape[2], self.head_dim, device=XV.device, dtype=XV.dtype)
            
            XQ = torch.cat([XQ, XQ_pad], dim=1)
            XK = torch.cat([XK, XK_pad], dim=1)
            XV = torch.cat([XV, XV_pad], dim=1)
        
        # Reshape to mini-batch format: [B, padded_L, H, HD] -> [B, H, NC, C, HD]
        XQ = XQ.view(B, NC, mini_batch_size, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        XK = XK.view(B, NC, mini_batch_size, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        XV = XV.view(B, NC, mini_batch_size, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        
        # Compute learning rate
        X_for_eta = XV.permute(0, 2, 3, 1, 4).reshape(B, NC, mini_batch_size, -1)  # [B, NC, C, H*HD]
        eta = self.get_eta(X_for_eta)  # [B, H, NC, C, 1]
        
        # Scale by mini-batch size and base learning rate
        eta = (self.ttt_base_lr / mini_batch_size) * eta
        
        return {
            "XQ": XQ,
            "XK": XK, 
            "XV": XV,
            "eta": eta,
            "original_length": L,
            "padded_length": padded_L,
            "NC": NC,
            "C": mini_batch_size
        }

    def forward(self, hidden_states: torch.Tensor, seq_metadata: SequenceMetadata):
        """Forward pass following Video-DiT pattern"""
        # Process input and get TTT format
        processed_inputs = self.process_input(hidden_states, seq_metadata)
        
        # Apply TTT computation
        hidden_states = self.ttt(processed_inputs)

        # Post-processing
        hidden_states = self.post_norm(hidden_states)
        hidden_states = self.wo(hidden_states)

        return hidden_states


class TTTMLP(TTTBase):
    """TTT-MLP implementation using Video-DiT's EXACT approach"""
    
    def __init__(self, config: TTTConfig, layer_id: int = None, use_kernel: bool = False):
        super().__init__(config, layer_id)

        # Check if this is multi-layer configuration
        num_layers = getattr(config, 'ttt_mlp_layers', 2)
        expansion_factor = getattr(config, 'ttt_mlp_expansion_factor', 4.0)

        if num_layers >= 3:
            logger.info(f"[TTT-MLP] Layer {layer_id}: Multi-layer config detected but using standard 2-layer implementation")
            logger.info(f"[TTT-MLP] Config has ttt_mlp_layers={num_layers}, but TTTMLP class only supports 2 layers")
            logger.warning(f"[TTT-MLP] ⚠️ To use multi-layer, you need TTTMLPMultiLayer class (not yet integrated)")

        self.W1 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, self.head_dim, 4 * self.head_dim)))
        self.b1 = nn.Parameter(torch.zeros(self.num_heads, 1, 4 * self.head_dim))
        self.W2 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, 4 * self.head_dim, self.head_dim)))
        self.b2 = nn.Parameter(torch.zeros(self.num_heads, 1, self.head_dim))
        self.use_kernel = use_kernel

        # Track streaming position for Figure 5 logging
        self.stream_position = 0

        if not hasattr(self, '_init_logged'):
            logger.info(f"[TTT-MLP] Layer {layer_id}: Initialized with 2-layer MLP (head_dim={self.head_dim}, intermediate={4 * self.head_dim})")
            self._init_logged = True

    def init_weights(self):
        super().init_weights()
        nn.init.normal_(self.W1, mean=0.0, std=0.02)
        nn.init.zeros_(self.b1)
        nn.init.normal_(self.W2, mean=0.0, std=0.02)
        nn.init.zeros_(self.b2)

    def ttt(self, inputs):
        """TTT processing using Video-DiT's EXACT ttt_mlp function"""
        B = inputs["XV"].shape[0]
        num_mini_batch = inputs["XV"].shape[2]
        L = inputs["XV"].shape[2] * inputs["XV"].shape[3]

        # Prepare initial weights (Video-DiT pattern)
        W1_states = torch.tile(self.W1.unsqueeze(0), dims=(B, 1, 1, 1))
        b1_states = torch.tile(self.b1.unsqueeze(0), dims=(B, 1, 1, 1))
        W2_states = torch.tile(self.W2.unsqueeze(0), dims=(B, 1, 1, 1))
        b2_states = torch.tile(self.b2.unsqueeze(0), dims=(B, 1, 1, 1))

        # Disable checkpointing during evaluation to avoid CUDA graph conflicts
        # During training: use checkpointing for memory efficiency
        # During evaluation: disable checkpointing for CUDA graph compatibility
        if self.training:
            checkpoint_group_size = min(max(self.scan_checkpoint_group_size, 1), num_mini_batch)
        else:
            checkpoint_group_size = 0  # No checkpointing during evaluation

        # Check for persistent states support
        if hasattr(self, 'persistent_states') and self.persistent_states:
            # Use state-returning version for persistent TTT states (JAX-style)
            if _global_log_inner_loop_losses:
                XQW_batch, final_states, losses = ttt_mlp_with_states(
                    inputs["XK"],
                    inputs["XQ"],
                    inputs["XV"],
                    inputs["eta"],
                    self.ttt_norm_weight,
                    self.ttt_norm_bias,
                    W1_states,
                    b1_states,
                    W2_states,
                    b2_states,
                    checkpoint_group_size,
                    log_losses=_global_log_inner_loop_losses,
                    layer_id=self.layer_id,
                    stream_pos_base=self.stream_position,  # Pass current stream position for Figure 5
                )
            else:
                XQW_batch, final_states = ttt_mlp_with_states(
                    inputs["XK"],
                    inputs["XQ"],
                    inputs["XV"],
                    inputs["eta"],
                    self.ttt_norm_weight,
                    self.ttt_norm_bias,
                    W1_states,
                    b1_states,
                    W2_states,
                    b2_states,
                    checkpoint_group_size,
                    log_losses=_global_log_inner_loop_losses,
                    layer_id=self.layer_id,
                    stream_pos_base=self.stream_position,  # Pass current stream position for Figure 5
                )
            
            # Update model parameters with persistent states (JAX-style behavior)
            # Extract final states and remove batch dimension for parameter update
            with torch.no_grad():
                # final_states have shape [B, H, ...] - take first batch element [0] to get [H, ...]
                final_W1 = final_states["W1_states"][0]  # [B, H, HD, 4*HD] -> [H, HD, 4*HD]
                final_b1 = final_states["b1_states"][0]  # [B, H, 1, 4*HD] -> [H, 1, 4*HD]
                final_W2 = final_states["W2_states"][0]  # [B, H, 4*HD, HD] -> [H, 4*HD, HD]
                final_b2 = final_states["b2_states"][0]  # [B, H, 1, HD] -> [H, 1, HD]
                
                # Validate shapes before copying to catch dimension mismatches early
                if final_W1.shape != self.W1.shape:
                    raise RuntimeError(f"TTT STATE PERSISTENCE BUG: W1 shape mismatch! Expected {self.W1.shape}, got {final_W1.shape}")
                if final_b1.shape != self.b1.shape:
                    raise RuntimeError(f"TTT STATE PERSISTENCE BUG: b1 shape mismatch! Expected {self.b1.shape}, got {final_b1.shape}")
                if final_W2.shape != self.W2.shape:
                    raise RuntimeError(f"TTT STATE PERSISTENCE BUG: W2 shape mismatch! Expected {self.W2.shape}, got {final_W2.shape}")
                if final_b2.shape != self.b2.shape:
                    raise RuntimeError(f"TTT STATE PERSISTENCE BUG: b2 shape mismatch! Expected {self.b2.shape}, got {final_b2.shape}")
                
                # Copy updated states to model parameters
                self.W1.data.copy_(final_W1)
                self.b1.data.copy_(final_b1) 
                self.W2.data.copy_(final_W2)
                self.b2.data.copy_(final_b2)
        else:
            # Use original non-persistent version (current behavior)
            if _global_log_inner_loop_losses:
                XQW_batch, losses = ttt_mlp_ops(
                    inputs["XK"],
                    inputs["XQ"],
                    inputs["XV"],
                    inputs["eta"],
                    self.ttt_norm_weight,
                    self.ttt_norm_bias,
                    W1_states,
                    b1_states,
                    W2_states,
                    b2_states,
                    checkpoint_group_size,
                    log_losses=_global_log_inner_loop_losses,
                    layer_id=self.layer_id,
                )
            else:
                XQW_batch = ttt_mlp_ops(
                    inputs["XK"],
                    inputs["XQ"],
                    inputs["XV"],
                    inputs["eta"],
                    self.ttt_norm_weight,
                    self.ttt_norm_bias,
                    W1_states,
                    b1_states,
                    W2_states,
                    b2_states,
                    checkpoint_group_size,
                    log_losses=_global_log_inner_loop_losses,
                    layer_id=self.layer_id,
                )

        # Reshape back to sequence format
        original_length = inputs["original_length"]
        XQW_batch = XQW_batch.reshape(B, L, self.width)

        # Trim back to original length
        XQW_batch = XQW_batch[:, :original_length, :]

        # Update stream position for next call (for Figure 5 logging)
        self.stream_position += num_mini_batch

        return XQW_batch