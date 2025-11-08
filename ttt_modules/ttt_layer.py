"""
TTT Layer - Drop-in Replacement for Llama Attention

Implements the Test-Time Training layer that replaces standard self-attention
in the top 8 layers (24-31) of Llama-Omni.

Key features:
- Compatible with HuggingFace Transformers interface
- Returns cache in format (output, None, new_cache)
- FP32 precision enforcement for TTT states
- Auto-padding to mini-batch boundaries
- RoPE positions reset per mini-batch
- State persistence across forward passes

Based on Doc 13 (Critical Updates) with all corrections applied.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import warnings


class TTTMLP(nn.Module):
    """
    TTT-MLP Layer - Drop-in replacement for LlamaSdpaAttention

    This layer can replace `self_attn` in Llama decoder layers.
    It maintains the same interface as standard attention layers but uses
    Test-Time Training instead of attention mechanism.

    CRITICAL REQUIREMENTS:
    1. All TTT states (W1, b1, W2, b2) MUST be torch.float32
    2. Sequences auto-padded to mini_batch_size multiples
    3. RoPE positions reset every mini_batch_size tokens
    4. Returns cache for state persistence across batches
    """

    def __init__(self, config, layer_idx: Optional[int] = None):
        """
        Args:
            config: Model configuration (LlamaConfig-like object)
                Required attributes:
                - hidden_size: Model dimension
                - num_attention_heads: Number of attention heads
                - mini_batch_size: TTT mini-batch size (default: 64)
                - rope_theta: RoPE theta parameter
                - ttt_base_lr: Base learning rate for TTT (default: 1.0)
            layer_idx: Layer index (for logging/monitoring)
        """
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.model_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.model_dim // self.num_heads
        self.mini_batch_size = getattr(config, 'mini_batch_size', 64)
        self.ttt_base_lr = getattr(config, 'ttt_base_lr', 1.0)

        if self.model_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_size ({self.model_dim}) must be divisible by "
                f"num_attention_heads ({self.num_heads})"
            )

        # Q, K, V, O projections (same as standard attention)
        self.wq = nn.Linear(self.model_dim, self.model_dim, bias=config.attention_bias if hasattr(config, 'attention_bias') else False)
        self.wk = nn.Linear(self.model_dim, self.model_dim, bias=config.attention_bias if hasattr(config, 'attention_bias') else False)
        self.wv = nn.Linear(self.model_dim, self.model_dim, bias=config.attention_bias if hasattr(config, 'attention_bias') else False)
        self.wo = nn.Linear(self.model_dim, self.model_dim, bias=config.attention_bias if hasattr(config, 'attention_bias') else False)

        # TTT parameters - MUST BE FLOAT32!
        hidden_dim = 4 * self.head_dim  # MLP expansion factor
        self.W1 = nn.Parameter(
            torch.zeros(self.num_heads, self.head_dim, hidden_dim, dtype=torch.float32)
        )
        self.b1 = nn.Parameter(
            torch.zeros(self.num_heads, 1, hidden_dim, dtype=torch.float32)
        )
        self.W2 = nn.Parameter(
            torch.zeros(self.num_heads, hidden_dim, self.head_dim, dtype=torch.float32)
        )
        self.b2 = nn.Parameter(
            torch.zeros(self.num_heads, 1, self.head_dim, dtype=torch.float32)
        )

        # Initialize with small random values
        nn.init.normal_(self.W1, mean=0.0, std=0.02)
        nn.init.normal_(self.W2, mean=0.0, std=0.02)

        # TTT LayerNorm parameters
        self.ttt_norm_weight = nn.Parameter(
            torch.ones(self.num_heads, self.head_dim, dtype=torch.float32)
        )
        self.ttt_norm_bias = nn.Parameter(
            torch.zeros(self.num_heads, self.head_dim, dtype=torch.float32)
        )

        # Learnable learning rate gate
        self.learnable_ttt_lr_weight = nn.Parameter(
            torch.normal(0, 0.02, size=(self.num_heads, self.model_dim, 1))
        )
        self.learnable_ttt_lr_bias = nn.Parameter(
            torch.zeros(self.num_heads, 1)
        )

        # Post-norm (output normalization)
        self.post_norm = nn.LayerNorm(self.model_dim, eps=getattr(config, 'rms_norm_eps', 1e-6))

        # RoPE (positions reset per mini-batch!)
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
        self.rotary_emb = LlamaRotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=self.mini_batch_size,  # NOT full sequence!
            base=getattr(config, 'rope_theta', 10000.0),
        )

        # Verify FP32 at initialization
        self._verify_fp32()

    def _verify_fp32(self):
        """CRITICAL: Verify all TTT parameters are FP32."""
        params_to_check = {
            'W1': self.W1,
            'b1': self.b1,
            'W2': self.W2,
            'b2': self.b2,
            'ttt_norm_weight': self.ttt_norm_weight,
            'ttt_norm_bias': self.ttt_norm_bias,
        }

        for name, param in params_to_check.items():
            if param.dtype != torch.float32:
                raise TypeError(
                    f"TTT parameter {name} must be FP32, got {param.dtype}. "
                    f"This is CRITICAL for numerical stability!"
                )

    def forward(
        self,
        hidden_states: torch.Tensor,                      # [B, L, D]
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,  # TTT cache
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, None, Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass with HuggingFace-compatible interface.

        Args:
            hidden_states: Input features [B, L, D]
            attention_mask: Attention mask (not used in TTT, for compatibility)
            position_ids: Position IDs for RoPE
            past_key_value: TTT state cache (W1, b1, W2, b2)
            output_attentions: Return attention weights (always None for TTT)
            use_cache: Whether to return updated cache
            cache_position: Cache position (for compatibility, not used)

        Returns:
            tuple: (output, None, cache)
            - output: [B, L, D] transformed features
            - None: no attention weights (TTT doesn't have them)
            - cache: (W1, b1, W2, b2) if use_cache else None
        """

        # Verify FP32 each forward pass (catches mixed precision issues)
        if self.training:
            self._verify_fp32()

        B, L, D = hidden_states.shape
        original_length = L

        # AUTO-PAD to mini_batch_size multiple
        pad_len = 0
        if L % self.mini_batch_size != 0:
            pad_len = self.mini_batch_size - (L % self.mini_batch_size)
            hidden_states = F.pad(hidden_states, (0, 0, 0, pad_len))
            L = hidden_states.shape[1]

            # Extend position_ids if provided
            if position_ids is not None:
                last_pos = position_ids[:, -1:]
                pad_positions = last_pos + torch.arange(
                    1, pad_len + 1,
                    device=position_ids.device
                )
                position_ids = torch.cat([
                    position_ids,
                    pad_positions.expand(B, -1)
                ], dim=1)

        # Initialize state from cache or model parameters
        if past_key_value is not None:
            # Load state from cache
            W1_init, b1_init, W2_init, b2_init = past_key_value
        else:
            # Initialize from model parameters (expand for batch)
            W1_init = self.W1.unsqueeze(0).expand(B, -1, -1, -1).contiguous()
            b1_init = self.b1.unsqueeze(0).expand(B, -1, -1, -1).contiguous()
            W2_init = self.W2.unsqueeze(0).expand(B, -1, -1, -1).contiguous()
            b2_init = self.b2.unsqueeze(0).expand(B, -1, -1, -1).contiguous()

        # Process with TTT
        output, W1_final, b1_final, W2_final, b2_final = self._forward_ttt(
            hidden_states, W1_init, b1_init, W2_init, b2_init, position_ids
        )

        # TRIM padding from output
        if output.shape[1] > original_length:
            output = output[:, :original_length, :]

        # Warn if attention_mask is used (TTT doesn't support masking directly)
        if attention_mask is not None and attention_mask.sum() != attention_mask.numel():
            warnings.warn(
                "TTT layer received attention_mask but doesn't support masking. "
                "Masking is ignored. Consider padding sequences instead."
            )

        # Return state for next batch (CRITICAL for long context!)
        new_cache = None
        if use_cache:
            new_cache = (
                W1_final.detach(),  # Detach to prevent gradient accumulation
                b1_final.detach(),
                W2_final.detach(),
                b2_final.detach(),
            )

        # Return: (output, attention_weights=None, cache)
        return output, None, new_cache

    def _forward_ttt(
        self,
        hidden_states: torch.Tensor,
        W1_init: torch.Tensor,
        b1_init: torch.Tensor,
        W2_init: torch.Tensor,
        b2_init: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Core TTT processing.

        Args:
            hidden_states: [B, L, D] (already padded to mini-batch multiple)
            W1_init, b1_init, W2_init, b2_init: Initial TTT states

        Returns:
            output: [B, L, D]
            W1_final, b1_final, W2_final, b2_final: Updated states
        """

        B, L, D = hidden_states.shape
        num_mini_batches = L // self.mini_batch_size

        # 1. Get Q, K, V projections
        XQ = self.wq(hidden_states).view(B, L, self.num_heads, self.head_dim)
        XK = self.wk(hidden_states).view(B, L, self.num_heads, self.head_dim)
        XV = self.wv(hidden_states).view(B, L, self.num_heads, self.head_dim)

        # 2. L2 Normalize Q, K (critical for TTT stability)
        XQ = F.normalize(XQ, p=2, dim=-1)
        XK = F.normalize(XK, p=2, dim=-1)

        # 3. Apply RoPE (POSITIONS RESET PER MINI-BATCH!)
        if position_ids is None:
            # Create local positions: [0-63, 0-63, 0-63, ...]
            position_ids = torch.arange(L, device=hidden_states.device)
            position_ids = position_ids % self.mini_batch_size  # CRITICAL!
            position_ids = position_ids.unsqueeze(0).expand(B, -1)

        # Get RoPE embeddings
        cos, sin = self.rotary_emb(XQ, position_ids)

        # Apply rotary embeddings
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
        XQ, XK = apply_rotary_pos_emb(XQ, XK, cos, sin)

        # 4. Prepare reconstruction target (normalized)
        XV_target = self._ln_reconstruction_target(XV, XK)

        # 5. Reshape to mini-batches [B, num_mb, mb_size, num_heads, head_dim]
        XQ = XQ.reshape(B, num_mini_batches, self.mini_batch_size, self.num_heads, self.head_dim)
        XK = XK.reshape(B, num_mini_batches, self.mini_batch_size, self.num_heads, self.head_dim)
        XV_target = XV_target.reshape(B, num_mini_batches, self.mini_batch_size, self.num_heads, self.head_dim)

        # Transpose to [B, num_heads, num_mb, mb_size, head_dim]
        XQ = XQ.permute(0, 3, 1, 2, 4)
        XK = XK.permute(0, 3, 1, 2, 4)
        XV_target = XV_target.permute(0, 3, 1, 2, 4)

        # 6. Get learning rate (eta)
        eta = self._get_eta(hidden_states)

        # 7. Run TTT-MLP over mini-batches
        from ttt_modules.ops.ttt_mlp import ttt_mlp

        # Convert to FP32 for TTT computation
        XQ_fp32 = XQ.float()
        XK_fp32 = XK.float()
        XV_fp32 = XV_target.float()
        eta_fp32 = eta.float()

        final_params, XQW_batch = ttt_mlp(
            XK=XK_fp32,
            XQ=XQ_fp32,
            XV=XV_fp32,
            eta=eta_fp32,
            ttt_norm_weight=self.ttt_norm_weight,
            ttt_norm_bias=self.ttt_norm_bias,
            W1_init=W1_init,
            b1_init=b1_init,
            W2_init=W2_init,
            b2_init=b2_init,
            checkpoint_group_size=4  # Checkpoint every 4 mini-batches
        )

        # Extract final states
        W1_final = final_params["W1_states"]
        b1_final = final_params["b1_states"]
        W2_final = final_params["W2_states"]
        b2_final = final_params["b2_states"]

        # 8. Reshape back to [B, L, num_heads, head_dim]
        XQW_batch = XQW_batch.permute(0, 2, 3, 1, 4)  # [B, num_mb, mb_size, num_heads, head_dim]
        XQW_batch = XQW_batch.reshape(B, L, self.num_heads, self.head_dim)

        # 9. Post-norm and output projection
        XQW_batch = XQW_batch.reshape(B, L, -1)
        XQW_batch = self.post_norm(XQW_batch)
        output = self.wo(XQW_batch)

        # Convert back to input dtype
        output = output.to(hidden_states.dtype)

        return output, W1_final, b1_final, W2_final, b2_final

    def _ln_reconstruction_target(self, XV: torch.Tensor, XK: torch.Tensor) -> torch.Tensor:
        """
        LayerNorm reconstruction target: LN(XV - XK).

        Args:
            XV: Value features [B, L, H, D]
            XK: Key features [B, L, H, D]

        Returns:
            target: Normalized reconstruction target [B, L, H, D]
        """
        target = XV - XK
        eps = 1e-8

        # Normalize over head_dim
        mean = target.mean(dim=-1, keepdim=True)
        std = target.std(dim=-1, keepdim=True)
        target = (target - mean) / (std + eps)

        # Apply per-head weight and bias
        # Expand to [B, L, H, D]
        weight = self.ttt_norm_weight.unsqueeze(0).unsqueeze(0)  # [1, 1, H, D]
        bias = self.ttt_norm_bias.unsqueeze(0).unsqueeze(0)

        target = weight * target + bias

        # Add back XK for reconstruction
        return target + XK

    def _get_eta(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute per-token learning rate.

        Args:
            hidden_states: [B, L, D]

        Returns:
            eta: [B, H, num_mb, mb_size, 1] learning rates
        """
        B, L, D = hidden_states.shape
        num_mini_batches = L // self.mini_batch_size

        # Reshape to mini-batches
        X = hidden_states.reshape(B, num_mini_batches, self.mini_batch_size, D)

        # Compute learnable LR: [B, H, num_mb, mb_size, 1]
        # Einstein: batch, num_mb, mb_size, channels @ heads, channels, 1
        ttt_lr = torch.einsum(
            "bnkc,hdc->bhnk1",
            X,
            self.learnable_ttt_lr_weight
        ) + self.learnable_ttt_lr_bias.reshape(1, -1, 1, 1, 1)

        ttt_lr = torch.sigmoid(ttt_lr)  # Gate in [0, 1]

        # Final eta: base_lr * gated_lr / head_dim
        eta = (self.ttt_base_lr / self.head_dim) * ttt_lr

        return eta


if __name__ == "__main__":
    """Test the TTT layer."""
    print("Testing TTT Layer")
    print("=" * 60)

    # Mock config
    class Config:
        hidden_size = 4096
        num_attention_heads = 32
        mini_batch_size = 64
        rope_theta = 10000.0
        ttt_base_lr = 1.0
        attention_bias = False
        rms_norm_eps = 1e-6

    config = Config()

    # Create layer
    ttt_layer = TTTMLP(config, layer_idx=24)

    print(f"Layer created:")
    print(f"  Model dim: {ttt_layer.model_dim}")
    print(f"  Num heads: {ttt_layer.num_heads}")
    print(f"  Head dim: {ttt_layer.head_dim}")
    print(f"  Mini-batch size: {ttt_layer.mini_batch_size}")

    # Test forward pass
    B, L, D = 2, 256, 4096  # 256 tokens = 4 mini-batches
    hidden_states = torch.randn(B, L, D)

    print(f"\nInput shape: {hidden_states.shape}")

    # Forward without cache
    output1, attn_weights, cache1 = ttt_layer(
        hidden_states,
        use_cache=True
    )

    print(f"\nOutput shapes:")
    print(f"  output: {output1.shape}")
    print(f"  attn_weights: {attn_weights}")
    print(f"  cache: {len(cache1)} tensors")
    print(f"    W1: {cache1[0].shape}")

    # Forward with cache (state persistence)
    output2, _, cache2 = ttt_layer(
        hidden_states,
        past_key_value=cache1,
        use_cache=True
    )

    print(f"\nWith cache:")
    print(f"  output: {output2.shape}")
    print(f"  States changed: {not torch.allclose(cache1[0], cache2[0])}")

    # Check FP32
    print(f"\nDtype checks:")
    print(f"  W1: {ttt_layer.W1.dtype}")
    print(f"  output: {output1.dtype}")

    print("\n✅ TTT layer test complete!")
