#!/usr/bin/env python3
"""
Debug the TTT shape conversion issue
"""

import torch
import sys
import os

# Disable torch dynamo compilation to get cleaner errors
import torch._dynamo
torch._dynamo.config.suppress_errors = True

# Add paths
sys.path.append('/home/alufr/ttt_tests/moshi')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_shape_conversion():
    """Debug the shape conversion issue"""
    print("🔍 Debugging TTT shape conversion...")
    
    try:
        from moshi_ttt.format_utils import moshi_to_ttt_format, ttt_to_moshi_format
        from moshi_ttt.moshi_metadata import MoshiSequenceMetadata
        
        # Test with the same dimensions as our failing test
        batch_size = 2
        seq_len = 32
        d_model = 512
        num_heads = 8
        mini_batch_size = 16
        
        print(f"Input dimensions: B={batch_size}, seq_len={seq_len}, d_model={d_model}")
        print(f"TTT config: num_heads={num_heads}, mini_batch_size={mini_batch_size}")
        
        # Create input tensor
        x = torch.randn(batch_size, seq_len, d_model)
        print(f"Input shape: {x.shape}")
        
        # Create metadata
        from moshi_ttt.config import TTTConfig
        ttt_config = TTTConfig(model_dim=d_model, num_heads=num_heads, mini_batch_size=mini_batch_size)
        metadata = MoshiSequenceMetadata(batch_size, seq_len, d_model, ttt_config)
        print(f"Metadata: {metadata}")
        
        # Convert to TTT format
        x_ttt, conversion_metadata = moshi_to_ttt_format(x, ttt_config)
        print(f"TTT format shape: {x_ttt.shape}")
        print(f"Expected TTT shape: [B={batch_size}, H={num_heads}, NC={metadata.NC}, C={metadata.C}, HD={d_model//num_heads}]")
        
        # Expected shape breakdown
        expected_B = batch_size
        expected_H = num_heads  
        expected_NC = metadata.NC
        expected_C = metadata.C
        expected_HD = d_model // num_heads
        
        print(f"Expected: [{expected_B}, {expected_H}, {expected_NC}, {expected_C}, {expected_HD}]")
        print(f"Actual:   {list(x_ttt.shape)}")
        
        # Test conversion back
        x_back = ttt_to_moshi_format(x_ttt, conversion_metadata)
        print(f"Converted back shape: {x_back.shape}")
        print(f"Original shape: {x.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ttt_ops_directly():
    """Test the TTT ops with the shapes we're getting"""
    print("\n🔍 Testing TTT ops with our shapes...")
    
    try:
        from moshi_ttt.models.ssm.ops.ttt_mlp import ttt_mlp
        
        # Parameters from our test
        B, H, NC, C, HD = 2, 8, 2, 16, 64  # Based on seq_len=32, mini_batch_size=16
        
        print(f"TTT MLP input dimensions: B={B}, H={H}, NC={NC}, C={C}, HD={HD}")
        
        # Create test tensors
        XQ = torch.randn(B, H, NC, C, HD)
        XK = torch.randn(B, H, NC, C, HD) 
        XV = torch.randn(B, H, NC, C, HD)
        eta = torch.randn(B, H, NC, 1, C)
        
        # TTT MLP parameters
        ttt_norm_weight = torch.randn(H, HD)
        ttt_norm_bias = torch.randn(H, HD)
        W1_init = torch.randn(B, H, HD, 4 * HD)  
        b1_init = torch.randn(B, H, 1, 4 * HD)
        W2_init = torch.randn(B, H, 4 * HD, HD)
        b2_init = torch.randn(B, H, 1, HD)
        checkpoint_group_size = 1
        
        print(f"XQ shape: {XQ.shape}")
        print(f"XK shape: {XK.shape}")
        print(f"XV shape: {XV.shape}")  
        print(f"eta shape: {eta.shape}")
        print(f"W1_init shape: {W1_init.shape}")
        
        # Test TTT MLP
        with torch.no_grad():
            output = ttt_mlp(XK, XQ, XV, eta, ttt_norm_weight, ttt_norm_bias, 
                           W1_init, b1_init, W2_init, b2_init, checkpoint_group_size)
            print(f"TTT MLP output shape: {output.shape}")
            
        return True
        
    except Exception as e:
        print(f"❌ Error in TTT ops test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Debugging TTT Shape Issues")
    print("=" * 50)
    
    success1 = test_shape_conversion()
    success2 = test_ttt_ops_directly()
    
    print("=" * 50)
    if success1 and success2:
        print("🎉 All debugging tests passed!")
    else:
        print("❌ Some debugging tests failed!")
        
    print("🔧 This should help identify the shape mismatch issue")
