"""
Step 3.1.1: Create hybrid_layer.py file
Following Video-DiT's SeqModelingBlock pattern exactly
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import sys
import os

# Add original Moshi to path
sys.path.append('/home/alufr/ttt_tests/moshi/moshi')

# Moshi imports
from moshi.modules.transformer import StreamingTransformerLayer

# Our TTT imports  
from .models.ssm.ops.ttt_mlp import ttt_mlp, ttt_mlp_with_states
from .format_utils import moshi_to_ttt_format, ttt_to_moshi_format
from .moshi_metadata import MoshiSequenceMetadata
from .config import TTTConfig
from .utils import SequenceMetadata


class HybridSeqModelingBlock(nn.Module):
    """
    Hybrid block that combines Moshi's attention with Video-DiT's TTT processing.
    
    This follows Video-DiT's SeqModelingBlock pattern:
    1. Attention processing (using Moshi's existing attention)
    2. TTT/SSM processing (using Video-DiT's TTT implementation)
    
    The key difference from Video-DiT is that we work with 1D audio sequences
    instead of 3D video sequences.
    """
    
    def __init__(self, original_layer: StreamingTransformerLayer, ttt_config: TTTConfig, persistent_states: bool = False):
        super().__init__()
        
        # Store the original Moshi layer for attention processing
        self.original_layer = original_layer
        
        # TTT configuration
        self.ttt_config = ttt_config
        
        # State persistence configuration
        self.persistent_states = persistent_states
        
        # Store dimensions for format conversion
        self.d_model = ttt_config.model_dim
        self.num_heads = ttt_config.num_heads
        self.head_dim = self.d_model // self.num_heads
        
        # TTT parameters (following Video-DiT's TTTMLP pattern)
        self._init_ttt_parameters()
        
        # SSM Gating (Video-DiT lines 147-150 pattern) - MUST be created before _init_weights()
        from .ssm_gating import SSMGating
        self.forward_ssm_gating = SSMGating(ttt_config)
        self.backward_ssm_gating = SSMGating(ttt_config)
        
        # CRITICAL FIX: Initialize weights properly AFTER all parameters exist
        self._init_weights()
        
    def _init_ttt_parameters(self):
        """Initialize TTT-MLP parameters following Video-DiT's pattern exactly"""
        # Q, K, V projections for TTT (following Video-DiT's _init_qkvo_proj)
        self.wq = nn.Linear(self.d_model, self.num_heads * self.head_dim, bias=True)
        self.wk = nn.Linear(self.d_model, self.num_heads * self.head_dim, bias=True) 
        self.wv = nn.Linear(self.d_model, self.num_heads * self.head_dim, bias=True)
        self.wo = nn.Linear(self.d_model, self.num_heads * self.head_dim, bias=True)
        
        # TTT-MLP parameters (per head) - following Video-DiT TTTMLP exactly
        self.W1 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, self.head_dim, 4 * self.head_dim)))
        self.b1 = nn.Parameter(torch.zeros(self.num_heads, 1, 4 * self.head_dim))
        self.W2 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, 4 * self.head_dim, self.head_dim)))
        self.b2 = nn.Parameter(torch.zeros(self.num_heads, 1, self.head_dim))
        
        # TTT layer normalization (per head) - properly initialized
        self.ttt_norm_weight = nn.Parameter(torch.ones(self.num_heads, self.head_dim))
        self.ttt_norm_bias = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))
        
        # TTT learning rate parameters - following Video-DiT _init_ttt_lr_gate
        linear_weight_data = nn.Linear(self.d_model, 1, bias=True).weight.data
        self.learnable_ttt_lr_weight = nn.Parameter(
            torch.stack(
                [torch.normal(0, 0.02, size=linear_weight_data.shape) for _ in range(self.num_heads)],
                dim=0,
            )
        )
        linear_bias_data = nn.Linear(self.d_model, 1, bias=True).bias.data
        self.learnable_ttt_lr_bias = nn.Parameter(
            torch.stack(
                [torch.zeros_like(linear_bias_data) for _ in range(self.num_heads)],
                dim=0,
            )
        )
        
        # Post normalization
        self.post_norm = nn.LayerNorm(self.d_model, eps=1e-6)
        
        
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
        
        ttt_lr = torch.einsum("bnkc,hdc->bhnkd", X, self.learnable_ttt_lr_weight) + \
                 self.learnable_ttt_lr_bias.reshape(1, -1, 1, 1, 1)
        
        ttt_lr = F.sigmoid(ttt_lr)  # [B, H, NC, C, 1]
        
        ttt_lr = ttt_lr.permute(0, 1, 2, 4, 3)  # [B, H, NC, 1, C]
        
        # CRITICAL: Validate head_dim to prevent division by zero
        if self.head_dim <= 0:
            raise RuntimeError(f"TTT CRITICAL BUG: head_dim is {self.head_dim} but must be > 0!")
        
        return self.ttt_config.ttt_base_lr * ttt_lr / self.head_dim
        
    def ln_reconstruction_target(self, XV, XK):
        """Layer norm reconstruction target following Video-DiT pattern"""
        B, L, num_heads, head_dim = XV.shape
        
        # Reshape for per-head layer norm
        XV = XV.view(-1, head_dim)  # [B*L*num_heads, head_dim]
        
        # Apply layer norm
        XV_norm = F.layer_norm(XV, (head_dim,), weight=None, bias=None, eps=1e-6)
        XV_norm = XV_norm.view(B, L, num_heads, head_dim)
        
        # Apply per-head weight and bias 
        XV = self.ttt_norm_weight.unsqueeze(0).unsqueeze(0) * XV_norm + \
             self.ttt_norm_bias.unsqueeze(0).unsqueeze(0)
             
        return XV + XK
        
    def _init_weights(self):
        """Initialize TTT weights following Video-DiT's pattern"""
        # QKV and output projections
        for linear in (self.wq, self.wk, self.wv):
            nn.init.normal_(linear.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.wo.weight, mean=0.0, std=0.02)
        
        # Post normalization
        self.post_norm.reset_parameters()
        
        # TTT layer norm
        nn.init.ones_(self.ttt_norm_weight.data)
        nn.init.zeros_(self.ttt_norm_bias.data)
        
        # TTT learning rate
        nn.init.normal_(self.learnable_ttt_lr_weight.data, mean=0.0, std=0.02)
        nn.init.zeros_(self.learnable_ttt_lr_bias.data)
        
        # SSM Gating parameters are already initialized by their constructors - DO NOT re-initialize
        
    def reset_ttt_states(self):
        """Reset TTT states to initial values - useful for sequence boundaries"""
        if self.persistent_states:
            with torch.no_grad():
                # Reset to initial parameter values (same as _init_ttt_parameters)
                old_W1_norm = self.W1.data.norm().item()
                old_b1_norm = self.b1.data.norm().item()
                
                nn.init.normal_(self.W1.data, mean=0.0, std=0.02)
                nn.init.zeros_(self.b1.data)
                nn.init.normal_(self.W2.data, mean=0.0, std=0.02) 
                nn.init.zeros_(self.b2.data)
                
                new_W1_norm = self.W1.data.norm().item()
                new_b1_norm = self.b1.data.norm().item()
                print(f"   TTT Reset: W1 norm {old_W1_norm:.6f} → {new_W1_norm:.6f}, b1 norm {old_b1_norm:.6f} → {new_b1_norm:.6f}")
                
    def get_ttt_state_dict(self):
        """Get current TTT state for checkpointing"""
        if self.persistent_states:
            return {
                'W1': self.W1.data.clone(),
                'b1': self.b1.data.clone(),
                'W2': self.W2.data.clone(),
                'b2': self.b2.data.clone(),
            }
        return None
        
    def load_ttt_state_dict(self, state_dict):
        """Load TTT state from checkpoint"""
        if self.persistent_states and state_dict is not None:
            with torch.no_grad():
                self.W1.data.copy_(state_dict['W1'])
                self.b1.data.copy_(state_dict['b1'])
                self.W2.data.copy_(state_dict['W2'])
                self.b2.data.copy_(state_dict['b2'])
        
    def forward(self, x: torch.Tensor, cross_attention_src: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass following Video-DiT's SeqModelingBlock pattern:
        1. Attention processing (equivalent to Video-DiT's _attn_forward)
        2. TTT processing (equivalent to Video-DiT's _ssm_forward)
        """
        
        # CRITICAL: Input validation to catch shape mismatches early
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"TTT INPUT ERROR: Expected torch.Tensor, got {type(x)}")
        if x.dim() != 3:
            raise ValueError(f"TTT INPUT ERROR: Expected 3D tensor [B, seq_len, d_model], got {x.dim()}D tensor {x.shape}")
        if x.shape[-1] != self.d_model:
            raise ValueError(f"TTT INPUT ERROR: Expected d_model={self.d_model}, got {x.shape[-1]}")
        if x.shape[0] == 0 or x.shape[1] == 0:
            raise ValueError(f"TTT INPUT ERROR: Invalid tensor dimensions {x.shape} - batch_size and seq_len must be > 0")
        
        # Step 1: Attention processing (using original Moshi layer components)
        attn_output = self._attn_forward(x, cross_attention_src)
        
        # Step 2: TTT processing (following Video-DiT's _ssm_forward pattern)  
        ttt_output = self._ttt_forward(attn_output)
        
        return ttt_output
        
    def _attn_forward(self, x: torch.Tensor, cross_attention_src: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Attention processing using Moshi's original components.
        Equivalent to Video-DiT's _attn_forward method.
        """
        # Use Moshi's self-attention block
        x = self.original_layer._sa_block(x)
        
        # Add cross-attention if available
        if self.original_layer.cross_attention is not None and cross_attention_src is not None:
            x = self.original_layer._cross_attention_block(x, cross_attention_src)
            
        return x
        
    def _ttt_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        TTT processing with gated residual connections following Video-DiT's _ssm_forward pattern.
        Converts Moshi format to TTT format, applies TTT, returns gated residual in Moshi format.
        Video-DiT reference: dit.py lines 241-245
        """
        B, seq_len, d_model = x.shape
        
        # Store residual before any processing (Video-DiT line 241 pattern)
        residual_emb = x.clone()
        
        # Convert to TTT format for processing: [B, seq_len, d_model] -> [B, H, NC, C, HD]
        x_ttt, conversion_metadata = moshi_to_ttt_format(x, self.ttt_config)
        
        # Apply TTT processing (returns Moshi format directly) - pass metadata for pad_mask
        x_processed = self._apply_ttt_processing(x_ttt, x, conversion_metadata)
        
        # Apply gated residual connection (Video-DiT line 245 pattern)
        gated_output = self.forward_ssm_gating(x_processed)
        
        # With pad+mask approach, output should already have correct shape
        assert gated_output.shape[1] == seq_len, f"Shape mismatch after TTT: expected {seq_len}, got {gated_output.shape[1]}"
        
        return residual_emb + gated_output
        
    def _apply_ttt_processing(self, x_ttt: torch.Tensor, x_original: torch.Tensor, metadata: dict) -> torch.Tensor:
        """Apply TTT processing to tensor in TTT format with pad_mask support"""
        B, H, NC, C, HD = x_ttt.shape
        seq_len = x_original.shape[1]
        
        # Get Q, K, V projections from original input
        XQ = self.wq(x_original)  # [B, seq_len, H*HD]
        XK = self.wk(x_original)  # [B, seq_len, H*HD] 
        XV = self.wv(x_original)  # [B, seq_len, H*HD]
        
        # Reshape projections: [B, seq_len, H*HD] -> [B, seq_len, H, HD]
        XQ = XQ.view(B, seq_len, H, HD)
        XK = XK.view(B, seq_len, H, HD)
        XV = XV.view(B, seq_len, H, HD)
        
        # Apply L2 normalization (following Video-DiT)
        XQ = F.normalize(XQ, p=2, dim=-1)
        XK = F.normalize(XK, p=2, dim=-1)
        
        # Apply layer norm reconstruction target (Video-DiT pattern)
        XV = self.ln_reconstruction_target(XV, XK)
        
        # Convert to TTT format: [B, seq_len, H, HD] -> [B, H, NC, C, HD]
        XQ_ttt, _ = moshi_to_ttt_format(XQ.view(B, seq_len, H*HD), self.ttt_config)
        XK_ttt, _ = moshi_to_ttt_format(XK.view(B, seq_len, H*HD), self.ttt_config)
        XV_ttt, _ = moshi_to_ttt_format(XV.view(B, seq_len, H*HD), self.ttt_config)
        
        # Reshape to separate heads properly: [B, H, NC, C, HD] -> [B, H, NC, C, HD]
        # Use .reshape() to be safe for non-contiguous tensors
        XQ_ttt = XQ_ttt.reshape(B, H, NC, C, HD)
        XK_ttt = XK_ttt.reshape(B, H, NC, C, HD) 
        XV_ttt = XV_ttt.reshape(B, H, NC, C, HD)
        
        # Get pad_mask from metadata for loss computation
        pad_mask = metadata["pad_mask"]  # [B, H, NC, C, 1]
        
        # Compute TTT learning rate (eta) with proper padding handling
        padded_len = metadata["padded_shape"][1] 
        # Use padded input for eta computation since TTT expects padded dimensions
        if padded_len > seq_len:
            # Need to pad x_original to match TTT format
            pad_needed = padded_len - seq_len
            x_pad = torch.zeros(B, pad_needed, self.d_model, device=x_original.device, dtype=x_original.dtype)
            x_padded = torch.cat([x_original, x_pad], dim=1)
        else:
            x_padded = x_original
            
        x_chunked = x_padded.reshape(B, NC, C, self.d_model)
        
        # Compute learning rate using Video-DiT approach
        ttt_lr_eta = self.get_eta(x_chunked)  # [B, H, NC, 1, C] 
        
        # CRITICAL: Validate C to prevent division by zero
        if C <= 0:
            raise RuntimeError(f"TTT CRITICAL BUG: mini_batch_size (C) is {C} but must be > 0!")
        
        # Scale eta by valid token ratio per chunk (critical for pad+mask)
        valid_mask = pad_mask.squeeze(-1)  # [B, H, NC, C]
        valid_per_chunk = valid_mask.sum(dim=-1, keepdim=True).float()  # [B, H, NC, 1] 
        valid_ratio = (valid_per_chunk / C).clamp_min(1e-6)  # Prevent division by zero
        
        # Expand valid_ratio to match ttt_lr_eta shape: [B, H, NC, 1] -> [B, H, NC, 1, 1]
        valid_ratio = valid_ratio.unsqueeze(-1)  # [B, H, NC, 1, 1]
        
        eta = (1.0 / C) * ttt_lr_eta * valid_ratio  # Scale by valid ratio
        
        # Prepare TTT-MLP parameters for batch processing
        W1_states = self.W1.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, HD, 4*HD]
        b1_states = self.b1.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, 1, 4*HD]
        W2_states = self.W2.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, 4*HD, HD]
        b2_states = self.b2.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, 1, HD]
        
        # Apply TTT-MLP processing 
        # CRITICAL FIX: Disable checkpointing to avoid PyTorch device state error
        # Following TTT-Video-DiT's evaluation approach (scan_checkpoint_group_size = 1e6)
        checkpoint_group_size = 0  # 0 = no checkpointing, allows TTT gradients to flow
        
        if self.persistent_states:
            # Use state-returning version for persistent TTT states (JAX-style)
            XQW_batch, final_states = ttt_mlp_with_states(
                XK_ttt,     # K
                XQ_ttt,     # Q  
                XV_ttt,     # V
                eta,        # learning rate
                self.ttt_norm_weight,
                self.ttt_norm_bias,
                W1_states,
                b1_states,
                W2_states, 
                b2_states,
                checkpoint_group_size
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
            XQW_batch = ttt_mlp(
                XK_ttt,     # K
                XQ_ttt,     # Q  
                XV_ttt,     # V
                eta,        # learning rate
                self.ttt_norm_weight,
                self.ttt_norm_bias,
                W1_states,
                b1_states,
                W2_states, 
                b2_states,
                checkpoint_group_size
            )
        
        # TTT-MLP returns shape [B, NC, C, d_model], need to convert to [B, H, NC, C, HD] first
        # Reshape TTT output: [B, NC, C, d_model] -> [B, NC, C, H, HD] -> [B, H, NC, C, HD]
        # Use .reshape() to be safe for non-contiguous tensors
        XQW_batch = XQW_batch.reshape(B, NC, C, H, HD)
        XQW_batch = XQW_batch.permute(0, 3, 1, 2, 4)  # [B, NC, C, H, HD] -> [B, H, NC, C, HD]
        
        # Now convert TTT output back to Moshi format using conversion_metadata (preserves original seq_len)
        XQW_batch = ttt_to_moshi_format(XQW_batch, metadata)
        
        # Apply post normalization and output projection
        # Ensure consistent dtype (TTT-MLP might change dtype)
        if XQW_batch.dtype != x_original.dtype:
            XQW_batch = XQW_batch.to(x_original.dtype)
            
        XQW_batch = self.post_norm(XQW_batch)
        XQW_batch = self.wo(XQW_batch)
        
        # Output now has original sequence length - no padding needed
        return XQW_batch


class HybridStreamingTransformerLayer(nn.Module):
    """
    Hybrid transformer layer that replaces Moshi's StreamingTransformerLayer.
    
    This follows Video-DiT's TransformerLayer pattern:
    - Uses HybridSeqModelingBlock for attention + TTT processing
    - Keeps Moshi's feedforward (MLP) processing unchanged
    - Maintains full compatibility with Moshi's streaming interface
    """
    
    def __init__(self, original_layer: StreamingTransformerLayer, ttt_config: TTTConfig, persistent_states: bool = False):
        super().__init__()
        
        # Store original layer for feedforward processing
        self.original_layer = original_layer
        
        # Create hybrid seq modeling block (attention + TTT)
        self.seq_modeling_block = HybridSeqModelingBlock(original_layer, ttt_config, persistent_states)
        
        # Store streaming state reference for compatibility
        self._streaming_state = original_layer._streaming_state
        
    def reset_ttt_states(self):
        """Reset TTT states to initial values"""
        self.seq_modeling_block.reset_ttt_states()
        
    def get_ttt_state_dict(self):
        """Get current TTT state for checkpointing"""
        return self.seq_modeling_block.get_ttt_state_dict()
        
    def load_ttt_state_dict(self, state_dict):
        """Load TTT state from checkpoint"""
        self.seq_modeling_block.load_ttt_state_dict(state_dict)
        
    def forward(self, x: torch.Tensor, cross_attention_src: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass following Video-DiT's TransformerLayer pattern:
        1. Sequence modeling (attention + TTT) via HybridSeqModelingBlock
        2. Feedforward processing via original Moshi MLP
        """
        
        # Step 1: Sequence modeling (attention + TTT)
        # seq_output already includes proper residuals (attention_output + gated_TTT)
        x = self.seq_modeling_block(x, cross_attention_src)
        
        # Step 2: Feedforward processing (unchanged from Moshi)
        x = self.original_layer._ff_block(x)
        
        # Update streaming state (maintain Moshi compatibility)
        state = self._streaming_state
        if state:
            state.offset_cpu += x.shape[1]
            
        return x
