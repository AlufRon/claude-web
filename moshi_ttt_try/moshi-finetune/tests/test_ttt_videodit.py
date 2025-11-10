#!/usr/bin/env python3
"""
Step 2.3 (Updated): Test Video-DiT based TTT layer 
Verify our TTT-MLP implementation from Video-DiT works correctly
"""

import torch
import sys
import os

# Add the parent directory to sys.path to import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from moshi_ttt.models.ssm.ttt_layer import TTTMLP
from moshi_ttt.config import TTTConfig
from moshi_ttt.utils import SequenceMetadata


def test_ttt_mlp():
    print("Testing Video-DiT based TTT-MLP layer...")
    
    # Configure for 1D sequence (Moshi audio tokens)
    config = TTTConfig(
        model_dim=1024,
        num_heads=8,
        ttt_base_lr=1.0,
        mini_batch_size=16,
        scan_checkpoint_group_size=1,
        ssm_layer="ttt_mlp"
    )
    
    # Create TTT-MLP layer (disable kernel for CPU testing)
    ttt_layer = TTTMLP(config, use_kernel=False)
    ttt_layer.init_weights()
    
    # Test with 1D sequence data (like Moshi audio tokens)
    batch_size = 2
    seq_len = 64  # Sequence length
    
    # Create test input
    hidden_states = torch.randn(batch_size, seq_len, config.model_dim)
    
    print(f"Input shape: {hidden_states.shape}")
    print(f"Config - model_dim: {config.model_dim}, num_heads: {config.num_heads}, head_dim: {config.head_dim}")
    
    # Get QKV projections
    XQ, XK, XV = ttt_layer.get_qkv_projections(hidden_states)
    
    print(f"XQ shape: {XQ.shape}")
    print(f"XK shape: {XK.shape}")  
    print(f"XV shape: {XV.shape}")
    
    # Reshape for multi-head attention
    XQ = XQ.view(batch_size, seq_len, config.num_heads, config.head_dim)
    XK = XK.view(batch_size, seq_len, config.num_heads, config.head_dim)
    XV = XV.view(batch_size, seq_len, config.num_heads, config.head_dim)
    
    # Reshape for TTT: [B, H, NC, C, HD] where NC*C = seq_len
    # For simplicity, let's use NC=4, C=16 (so 4*16=64=seq_len)
    nc, c = 4, seq_len // 4
    XQ = XQ.view(batch_size, config.num_heads, nc, c, config.head_dim)
    XK = XK.view(batch_size, config.num_heads, nc, c, config.head_dim)
    XV = XV.view(batch_size, config.num_heads, nc, c, config.head_dim)
    
    print(f"Reshaped XQ: {XQ.shape} [B, H, NC, C, HD]")
    
    # Get learning rate (eta) - this should be called with original hidden_states
    # not with reshaped XK. get_eta expects [B, NC, C, HD] not [B, H, NC, C, HD]
    hidden_states_reshaped = hidden_states.view(batch_size, nc, c, config.model_dim)
    eta = ttt_layer.get_eta(hidden_states_reshaped)  # Should be [B, NC, C, model_dim]
    print(f"Eta shape: {eta.shape}")
    
    # Prepare inputs for TTT
    inputs = {
        'XQ': XQ,
        'XK': XK, 
        'XV': XV,
        'eta': eta
    }
    
    # Run TTT forward pass
    try:
        output = ttt_layer.ttt(inputs)
        print(f"TTT output shape: {output.shape}")
        
        # Should output [B, seq_len, model_dim]
        expected_shape = (batch_size, seq_len, config.model_dim)
        if output.shape == expected_shape:
            print(f"✅ SUCCESS: Output shape matches expected {expected_shape}")
        else:
            print(f"❌ FAILURE: Expected shape {expected_shape}, got {output.shape}")
            
        # Check for NaN values
        if torch.isnan(output).any():
            print("❌ FAILURE: Output contains NaN values")
        else:
            print("✅ SUCCESS: No NaN values in output")
            
        # Print some statistics
        print(f"Output stats - min: {output.min():.4f}, max: {output.max():.4f}, mean: {output.mean():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILURE: TTT forward pass failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_ttt_mlp()
    if success:
        print("\n🎉 All TTT layer tests passed!")
        sys.exit(0)
    else:
        print("\n💥 TTT layer tests failed!")
        sys.exit(1)
