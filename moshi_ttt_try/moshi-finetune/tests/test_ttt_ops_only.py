#!/usr/bin/env python3
"""
Test core TTT operations (ops only, no full layer)
"""

import torch
import sys
import os

# Add the parent directory to sys.path to import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_ttt_mlp_ops():
    print("Testing core TTT-MLP operations...")
    
    from moshi_ttt.models.ssm.ops.ttt_mlp import ttt_mlp
    
    # Test parameters
    batch_size = 2
    num_heads = 8
    nc, c = 4, 16  # mini_batches=4, chunk_size=16
    head_dim = 128
    hidden_dim = 4 * head_dim  # Standard MLP expansion
    
    print(f"Testing with B={batch_size}, H={num_heads}, NC={nc}, C={c}, HD={head_dim}")
    
    # Create test inputs
    XK = torch.randn(batch_size, num_heads, nc, c, head_dim)
    XQ = torch.randn(batch_size, num_heads, nc, c, head_dim) 
    XV = torch.randn(batch_size, num_heads, nc, c, head_dim)
    eta = torch.ones(batch_size, num_heads, nc, 1, c) * 0.1  # Learning rate
    
    # TTT parameters
    ttt_norm_weight = torch.ones(num_heads, head_dim)
    ttt_norm_bias = torch.zeros(num_heads, head_dim)
    W1_init = torch.randn(batch_size, num_heads, head_dim, hidden_dim) * 0.02
    b1_init = torch.zeros(batch_size, num_heads, 1, hidden_dim)
    W2_init = torch.randn(batch_size, num_heads, hidden_dim, head_dim) * 0.02
    b2_init = torch.zeros(batch_size, num_heads, 1, head_dim)
    
    checkpoint_group_size = 0  # Disable checkpointing completely
    
    print("Input shapes:")
    print(f"  XK: {XK.shape}")
    print(f"  XQ: {XQ.shape}")
    print(f"  XV: {XV.shape}")
    print(f"  eta: {eta.shape}")
    print(f"  W1_init: {W1_init.shape}")
    
    try:
        # Run TTT MLP operation
        output = ttt_mlp(
            XK, XQ, XV, eta,
            ttt_norm_weight, ttt_norm_bias,
            W1_init, b1_init, W2_init, b2_init,
            checkpoint_group_size
        )
        
        print(f"✅ TTT MLP output shape: {output.shape}")
        
        # Video-DiT outputs [B, NC, C, H, HD], let's reshape to final format
        B, NC, C, H, HD = output.shape
        L = NC * C  # sequence length  
        model_dim = H * HD  # total model dimension
        
        output_reshaped = output.permute(0, 3, 1, 2, 4).reshape(B, L, model_dim)
        print(f"Reshaped output: {output_reshaped.shape}")
        
        # Should output [B, seq_len, model_dim]
        expected_shape = (batch_size, nc * c, num_heads * head_dim)
        if output_reshaped.shape == expected_shape:
            print(f"✅ SUCCESS: Final output shape matches expected {expected_shape}")
        else:
            print(f"❌ FAILURE: Expected final shape {expected_shape}, got {output_reshaped.shape}")
            
        # Check for NaN/Inf
        if torch.isnan(output).any():
            print("❌ FAILURE: Output contains NaN values")
            return False
        elif torch.isinf(output).any():
            print("❌ FAILURE: Output contains Inf values")  
            return False
        else:
            print("✅ SUCCESS: Output is finite")
            
        # Print statistics
        print(f"Output stats - min: {output.min():.4f}, max: {output.max():.4f}, mean: {output.mean():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILURE: TTT MLP ops failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_ttt_mlp_ops()
    if success:
        print("\n🎉 TTT MLP ops test passed!")
        sys.exit(0)
    else:
        print("\n💥 TTT MLP ops test failed!")
        sys.exit(1)
