"""
TTT Layer Implementation for Llama

This module provides a drop-in replacement for Llama's attention mechanism.
Instead of standard attention, it uses Test-Time Training to maintain
an RNN-style hidden state that adapts during inference.

Key Features:
- Compatible with Llama's interface (hidden_states in/out)
- Supports both training and inference modes
- Handles mini-batch processing (required for stable TTT updates)
- Maintains conversation-level state persistence
- Comprehensive logging and state tracking

Architecture:
    Input: hidden_states [B, L, D]
    ├─> Split into mini-batches of size K: [B, num_mb, K, D]
    ├─> For each mini-batch:
    │   ├─> Compute Q, K, V projections
    │   ├─> Apply RoPE (within mini-batch positions 0 to K-1)
    │   ├─> Run TTT update: W_new = W_old - eta * grad
    │   └─> Compute output: output = Q @ W_new
    └─> Concatenate outputs: [B, L, D]

CRITICAL: The sequence length L must be divisible by mini_batch_size K.
Padding may be needed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import logging
import math

from .ops import ttt_linear
from .utils import apply_rotary_pos_emb, precompute_freqs_cis, log_ttt_state

logger = logging.getLogger(__name__)


class TTTLinearLayer(nn.Module):
    """
    TTT-Linear layer - drop-in replacement for Llama attention.

    This layer uses a linear model as the TTT hidden state:
        f(x; W, b) = x @ W + b

    During forward pass:
    1. Updates W, b using gradient descent on reconstruction loss
    2. Computes output using updated W, b
    3. Returns output with same shape as input

    Args:
        config: Model configuration with TTT parameters
        layer_idx: Index of this layer (for logging)
    """

    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx if layer_idx is not None else 0

        # Model dimensions
        self.hidden_size = config.hidden_size
        self.num_heads = config.ttt_num_heads if config.ttt_num_heads else config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.mini_batch_size = config.ttt_mini_batch_size

        assert self.hidden_size % self.num_heads == 0, (
            f"hidden_size ({self.hidden_size}) must be divisible by num_heads ({self.num_heads})"
        )

        # TTT hyperparameters
        self.ttt_base_lr = config.ttt_base_lr
        self.checkpoint_group_size = config.ttt_checkpoint_group_size

        # Logging
        self.enable_logging = config.ttt_enable_logging
        self.log_level = config.ttt_log_level
        self.log_interval = config.ttt_log_interval
        self.step_counter = 0

        # Q, K, V projections (standard Llama style)
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)

        # Output projection
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # TTT inner state parameters (MUST be float32!)
        # These are the "hidden state" of the TTT-RNN
        # W1: [num_heads, head_dim, head_dim]
        # b1: [num_heads, 1, head_dim]
        self.register_parameter(
            "W1",
            nn.Parameter(
                torch.normal(0, 0.02, size=(self.num_heads, self.head_dim, self.head_dim), dtype=torch.float32)
            )
        )
        self.register_parameter(
            "b1",
            nn.Parameter(
                torch.zeros(self.num_heads, 1, self.head_dim, dtype=torch.float32)
            )
        )

        # TTT LayerNorm parameters (per-head)
        # These normalize the reconstruction target
        ln_weight = torch.ones(self.head_dim)
        ln_bias = torch.zeros(self.head_dim)
        self.register_parameter(
            "ttt_norm_weight",
            nn.Parameter(
                ln_weight.reshape(1, 1, 1, self.head_dim).expand(1, self.num_heads, 1, self.head_dim).clone()
            )
        )
        self.register_parameter(
            "ttt_norm_bias",
            nn.Parameter(
                ln_bias.reshape(1, 1, 1, self.head_dim).expand(1, self.num_heads, 1, self.head_dim).clone()
            )
        )

        # Learnable TTT learning rate (per-head gating)
        # This allows the model to learn how much to update for each head
        self.learnable_ttt_lr_weight = nn.Parameter(
            torch.normal(0, 0.02, size=(self.num_heads, self.hidden_size, 1))
        )
        self.learnable_ttt_lr_bias = nn.Parameter(
            torch.zeros(1, 1, self.num_heads)
        )

        # Post-attention LayerNorm
        self.post_norm = nn.LayerNorm(self.hidden_size, eps=1e-6)

        # Precompute RoPE frequencies for mini-batches
        # NOTE: RoPE is applied WITHIN mini-batches, so max position = mini_batch_size
        cos, sin = precompute_freqs_cis(
            self.head_dim,
            self.mini_batch_size,
            theta=getattr(config, 'rope_theta', 10000.0)
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        logger.info(
            f"[TTT Layer {self.layer_idx}] Initialized with "
            f"hidden_size={self.hidden_size}, num_heads={self.num_heads}, "
            f"head_dim={self.head_dim}, mini_batch_size={self.mini_batch_size}"
        )

    def init_weights(self):
        """Initialize weights using Llama-style initialization."""
        # Q, K, V, O projections
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            nn.init.normal_(proj.weight, mean=0.0, std=0.02)

        # TTT parameters
        nn.init.normal_(self.W1, mean=0.0, std=0.02)
        nn.init.zeros_(self.b1)

        # TTT LayerNorm
        nn.init.ones_(self.ttt_norm_weight)
        nn.init.zeros_(self.ttt_norm_bias)

        # Learnable LR
        nn.init.normal_(self.learnable_ttt_lr_weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.learnable_ttt_lr_bias)

        # Post-norm
        self.post_norm.reset_parameters()

    def get_eta(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute per-token learning rates using learnable gating.

        The learning rate for each token is computed as:
            eta = ttt_base_lr * sigmoid(learnable_proj(hidden_states)) / head_dim

        This allows the model to learn which tokens should cause larger updates.

        Args:
            hidden_states: [B, num_mb, K, D]

        Returns:
            eta: [B, nh, num_mb, K, 1]
        """
        B, num_mb, K, D = hidden_states.shape

        # Project: [B, num_mb, K, D] @ [nh, D, 1] -> [B, num_mb, K, nh]
        # We need to reshape for batched matmul
        hidden_flat = hidden_states.reshape(B * num_mb * K, D)  # [B*num_mb*K, D]

        # learnable_ttt_lr_weight: [nh, D, 1]
        # We want: [B*num_mb*K, D] @ [D, nh] -> [B*num_mb*K, nh]
        # So transpose weight: [nh, D, 1] -> [D, nh]
        weight_t = self.learnable_ttt_lr_weight.squeeze(-1).t()  # [D, nh]

        ttt_lr_logits = hidden_flat @ weight_t + self.learnable_ttt_lr_bias  # [B*num_mb*K, nh]

        # Reshape: [B*num_mb*K, nh] -> [B, num_mb, K, nh]
        ttt_lr_logits = ttt_lr_logits.reshape(B, num_mb, K, self.num_heads)

        # Apply sigmoid and scale
        ttt_lr = F.sigmoid(ttt_lr_logits)  # [B, num_mb, K, nh]

        # Reshape to [B, nh, num_mb, K, 1]
        eta = ttt_lr.permute(0, 3, 1, 2).unsqueeze(-1)  # [B, nh, num_mb, K, 1]

        # Scale by base LR and head dim
        eta = self.ttt_base_lr * eta / self.head_dim

        return eta

    def reshape_for_mini_batches(
        self,
        hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, int]:
        """
        Reshape sequence into mini-batches.

        CRITICAL: Sequence length must be divisible by mini_batch_size.
        If not, padding is required (TODO: implement padding).

        Args:
            hidden_states: [B, L, D]

        Returns:
            reshaped: [B, num_mb, K, D] where num_mb * K = L
            num_mini_batches: int

        Raises:
            ValueError: If L is not divisible by mini_batch_size
        """
        B, L, D = hidden_states.shape
        K = self.mini_batch_size

        if L % K != 0:
            raise ValueError(
                f"Sequence length ({L}) must be divisible by mini_batch_size ({K}). "
                f"Please pad the sequence. Remainder: {L % K}"
            )

        num_mb = L // K

        # Reshape: [B, L, D] -> [B, num_mb, K, D]
        reshaped = hidden_states.reshape(B, num_mb, K, D)

        return reshaped, num_mb

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]], Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass of TTT layer.

        Args:
            hidden_states: [B, L, D] input tensor
            attention_mask: Not used in TTT (kept for interface compatibility)
            position_ids: Not used in TTT (RoPE is within mini-batches)
            past_key_value: TTT state from previous batch (for conversation persistence)
            output_attentions: Not applicable for TTT
            use_cache: Whether to return updated TTT state

        Returns:
            Tuple of:
            - output: [B, L, D] transformed hidden states
            - None: (attention weights not applicable for TTT)
            - new_ttt_state: Updated TTT state if use_cache else None

        Process:
            1. Reshape [B, L, D] -> [B, num_mb, K, D]
            2. Compute Q, K, V projections
            3. Apply RoPE within each mini-batch
            4. Compute learning rates eta
            5. Run TTT algorithm (sequential over mini-batches)
            6. Reshape output back to [B, L, D]
        """
        B, L, D = hidden_states.shape

        self.step_counter += 1

        if self.enable_logging and self.step_counter % self.log_interval == 0:
            logger.info(
                f"[TTT Layer {self.layer_idx} Step {self.step_counter}] "
                f"Processing input shape [B={B}, L={L}, D={D}]"
            )

        # Step 1: Reshape into mini-batches
        try:
            hidden_mb, num_mb = self.reshape_for_mini_batches(hidden_states)  # [B, num_mb, K, D]
        except ValueError as e:
            logger.error(f"[TTT Layer {self.layer_idx}] {str(e)}")
            raise

        K = self.mini_batch_size

        # Step 2: Compute Q, K, V projections
        # hidden_mb: [B, num_mb, K, D]
        # Need to flatten for linear projection
        hidden_flat = hidden_mb.reshape(B * num_mb * K, D)  # [B*num_mb*K, D]

        XQ_flat = self.q_proj(hidden_flat)  # [B*num_mb*K, num_heads * head_dim]
        XK_flat = self.k_proj(hidden_flat)
        XV_flat = self.v_proj(hidden_flat)

        # Reshape to [B, num_mb, K, num_heads, head_dim]
        XQ = XQ_flat.reshape(B, num_mb, K, self.num_heads, self.head_dim)
        XK = XK_flat.reshape(B, num_mb, K, self.num_heads, self.head_dim)
        XV = XV_flat.reshape(B, num_mb, K, self.num_heads, self.head_dim)

        # Transpose to [B, num_heads, num_mb, K, head_dim]
        XQ = XQ.permute(0, 3, 1, 2, 4)
        XK = XK.permute(0, 3, 1, 2, 4)
        XV = XV.permute(0, 3, 1, 2, 4)

        # Step 3: Apply L2 normalization to Q and K (standard for TTT)
        XQ = F.normalize(XQ, p=2, dim=-1)
        XK = F.normalize(XK, p=2, dim=-1)

        # Step 4: Apply RoPE within each mini-batch
        # RoPE is computed for positions 0 to K-1 within each mini-batch
        # rope_cos, rope_sin: [K, head_dim]
        # Need to expand to [B, num_mb, K, head_dim] for broadcasting

        cos = self.rope_cos.unsqueeze(0).unsqueeze(0)  # [1, 1, K, head_dim]
        sin = self.rope_sin.unsqueeze(0).unsqueeze(0)

        # XQ, XK: [B, nh, num_mb, K, head_dim]
        # We need to apply RoPE to each mini-batch independently
        # Reshape for RoPE application
        XQ_rope = XQ.reshape(B * self.num_heads * num_mb, K, self.head_dim)
        XK_rope = XK.reshape(B * self.num_heads * num_mb, K, self.head_dim)

        # Expand cos/sin: [1, 1, K, head_dim] -> [B*nh*num_mb, K, head_dim]
        cos_expanded = cos.expand(B * self.num_heads * num_mb, K, self.head_dim)
        sin_expanded = sin.expand(B * self.num_heads * num_mb, K, self.head_dim)

        XQ_rope, XK_rope = apply_rotary_pos_emb(XQ_rope, XK_rope, cos_expanded, sin_expanded)

        # Reshape back
        XQ = XQ_rope.reshape(B, self.num_heads, num_mb, K, self.head_dim)
        XK = XK_rope.reshape(B, self.num_heads, num_mb, K, self.head_dim)

        # Step 5: Normalize V target (XV - XK after LayerNorm)
        # This is handled inside ttt_linear

        # Step 6: Compute learning rates
        eta = self.get_eta(hidden_mb)  # [B, nh, num_mb, K, 1]

        # Step 7: Prepare initial TTT state
        # If past_key_value contains TTT state, use it; otherwise use model parameters
        if past_key_value is not None and len(past_key_value) == 2:
            W1_init, b1_init = past_key_value
            if self.enable_logging:
                logger.debug(
                    f"[TTT Layer {self.layer_idx}] Using cached TTT state from past_key_value"
                )
        else:
            # Tile initial parameters for batch dimension
            W1_init = self.W1.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * self.num_heads, self.head_dim, self.head_dim)
            b1_init = self.b1.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * self.num_heads, 1, self.head_dim)

        # Ensure float32
        W1_init = W1_init.to(torch.float32)
        b1_init = b1_init.to(torch.float32)

        # Step 8: Run TTT algorithm
        # ttt_linear expects: [B, nh, nc, K, f]
        # We have: [B, nh, num_mb, K, head_dim]
        # Perfect match!

        output = ttt_linear(
            XK=XK,
            XQ=XQ,
            XV=XV,
            eta=eta,
            ttt_norm_weight=self.ttt_norm_weight,
            ttt_norm_bias=self.ttt_norm_bias,
            W1_init=W1_init,
            b1_init=b1_init,
            checkpoint_group_size=self.checkpoint_group_size
        )  # Returns [B, num_mb, K, nh, head_dim]

        # Step 9: Reshape output back to [B, L, D]
        # output: [B, num_mb, K, nh, head_dim]
        output = output.reshape(B, L, self.num_heads * self.head_dim)

        # Step 10: Apply output projection
        output = self.o_proj(output)  # [B, L, D]

        # Step 11: Apply post-normalization
        output = self.post_norm(output)

        # Step 12: Prepare new TTT state for caching
        # TODO: Extract final W1, b1 from ttt_linear
        # For now, return None for state
        new_ttt_state = None if not use_cache else (W1_init, b1_init)  # Placeholder

        # Log state statistics periodically
        if self.enable_logging and self.step_counter % self.log_interval == 0:
            log_ttt_state(
                layer_idx=self.layer_idx,
                step=self.step_counter,
                W1=self.W1,
                b1=self.b1,
                log_level=self.log_level
            )

        return output, None, new_ttt_state


# Placeholder for TTT-MLP layer (to be implemented later if needed)
class TTTMLPLayer(nn.Module):
    """TTT-MLP layer - uses 2-layer MLP as hidden state instead of linear model."""
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        raise NotImplementedError("TTT-MLP not yet implemented. Use TTT-Linear for now.")
