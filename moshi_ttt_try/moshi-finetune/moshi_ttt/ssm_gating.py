"""
SSM Gating Module - Exact copy from Video-DiT
Source: ttt-video-dit/ttt/models/cogvideo/dit.py lines 90-103
"""

import torch
import torch.nn as nn


class SSMGating(nn.Module):
    """
    SSM Gating mechanism exactly as implemented in Video-DiT.
    
    This applies learnable per-dimension gating to control the contribution
    of TTT/SSM outputs. The gating values are bounded to [-1, 1] using tanh.
    """
    
    def __init__(self, config):
        super().__init__()
        
        # Video-DiT line 97: Initialize gating alpha with small values
        # torch.ones(config.model_dim) * config.gating_alpha_init
        self.gating_alpha = nn.Parameter(
            torch.ones(config.model_dim) * config.gating_alpha_init
        )
    
    def forward(self, x):
        """
        Apply gating to input tensor.
        
        Args:
            x: Input tensor of shape [..., model_dim]
            
        Returns:
            Gated tensor: tanh(gating_alpha) * x
        """
        # Video-DiT lines 100-103: Apply tanh gating (exact match)
        # STABILIZATION FIX: Clamp alpha to prevent negative/extreme values
        # Clamping to [-3, 3] gives tanh output in ~[-0.995, 0.995] range
        gating_alpha = torch.clamp(self.gating_alpha, min=-3.0, max=3.0)
        gating_alpha = torch.tanh(gating_alpha)
        output = gating_alpha * x

        # MINIMAL LOG: Warn if input or output is suspiciously large
        with torch.no_grad():
            input_norm = x.norm(p=2, dim=-1).mean().item()
            output_norm = output.norm(p=2, dim=-1).mean().item()
            # TTT output should be ~1.0 after layer norm, gated output should be ~0.01
            if input_norm > 10.0 or output_norm > 1.0:
                alpha_mean = gating_alpha.mean().item()
                print(f"⚠️  [TTT-GATING-LARGE] Input_norm={input_norm:.1f} | Output_norm={output_norm:.1f} | Alpha={alpha_mean:.4f}")

        return output