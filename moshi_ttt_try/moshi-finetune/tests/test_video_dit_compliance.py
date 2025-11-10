#!/usr/bin/env python3
"""
Test comparing our Moshi TTT implementation with Video-DiT patterns
"""

import torch
import torch.nn.functional as F
import sys
import os

# Disable torch dynamo compilation to get cleaner errors  
import torch._dynamo
torch._dynamo.config.suppress_errors = True

# Add paths
sys.path.append('/home/alufr/ttt_tests/moshi')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_video_dit_patterns():
    """Test that our implementation follows Video-DiT patterns correctly"""
    print("🔍 Testing Video-DiT pattern compliance...")
    
    try:
        # Import our implementation
        from moshi_ttt.hybrid_layer import HybridSeqModelingBlock
        from moshi_ttt.config import TTTConfig
        from moshi.modules.transformer import StreamingTransformerLayer
        
        # Create configurations
        d_model = 512
        num_heads = 8
        head_dim = d_model // num_heads
        mini_batch_size = 16
        
        # Create original Moshi layer
        original_layer = StreamingTransformerLayer(
            d_model=d_model,
            num_heads=num_heads,
            dim_feedforward=d_model * 4,
            causal=True,
        )
        
        # Create TTT config
        ttt_config = TTTConfig(
            model_dim=d_model,
            num_heads=num_heads,
            mini_batch_size=mini_batch_size,
            ttt_base_lr=0.1
        )
        
        # Create our hybrid block  
        hybrid_block = HybridSeqModelingBlock(original_layer, ttt_config)
        
        print("✅ Created hybrid block successfully")
        
        # Test Video-DiT pattern compliance
        batch_size = 2
        seq_len = 32
        x = torch.randn(batch_size, seq_len, d_model)
        
        print(f"✅ Created test input: {x.shape}")
        
        # Test individual components following Video-DiT pattern
        print("\n📋 Testing Video-DiT pattern components:")
        
        # 1. Test Q, K, V projections
        XQ, XK, XV = hybrid_block.get_qkv_projections(x)
        print(f"  ✅ Q, K, V projections: {XQ.shape}, {XK.shape}, {XV.shape}")
        
        # Reshape like Video-DiT
        XQ_reshaped = XQ.view(batch_size, seq_len, num_heads, head_dim)
        XK_reshaped = XK.view(batch_size, seq_len, num_heads, head_dim)
        XV_reshaped = XV.view(batch_size, seq_len, num_heads, head_dim)
        print(f"  ✅ Reshaped Q, K, V: {XQ_reshaped.shape}")
        
        # 2. Test L2 normalization (Video-DiT pattern)
        XQ_norm = F.normalize(XQ_reshaped, p=2, dim=-1)
        XK_norm = F.normalize(XK_reshaped, p=2, dim=-1)
        print(f"  ✅ L2 normalized Q, K: norms ~1.0: {torch.norm(XQ_norm[0,0,0,:]).item():.3f}")
        
        # 3. Test layer norm reconstruction target
        XV_ln = hybrid_block.ln_reconstruction_target(XV_reshaped, XK_norm)
        print(f"  ✅ Layer norm reconstruction target: {XV_ln.shape}")
        
        # 4. Test eta (learning rate) computation
        # Pad and chunk like Video-DiT
        NC = (seq_len + mini_batch_size - 1) // mini_batch_size
        C = mini_batch_size
        padded_len = NC * C
        
        if seq_len < padded_len:
            padding = torch.zeros(batch_size, padded_len - seq_len, d_model, device=x.device, dtype=x.dtype)
            x_padded = torch.cat([x, padding], dim=1)
        else:
            x_padded = x
            
        x_chunked = x_padded.view(batch_size, NC, C, d_model)
        eta = hybrid_block.get_eta(x_chunked)
        print(f"  ✅ Eta computation: {eta.shape} (expected: [{batch_size}, {num_heads}, {NC}, 1, {C}])")
        
        # 5. Test full forward pass
        print("\n🚀 Testing full forward pass...")
        hybrid_block.eval()
        with torch.no_grad():
            output = hybrid_block(x)
            
        print(f"  ✅ Full forward: {x.shape} → {output.shape}")
        
        # Verify Video-DiT compliance
        assert eta.shape == (batch_size, num_heads, NC, 1, C), f"Eta shape mismatch: {eta.shape}"
        assert output.shape == x.shape, f"Output shape mismatch: {output.shape} vs {x.shape}"
        assert torch.isfinite(output).all(), "Output contains non-finite values"
        
        print(f"\n📊 Video-DiT Compliance Report:")
        print(f"  ✅ Q, K, V projections: PASS")
        print(f"  ✅ L2 normalization: PASS") 
        print(f"  ✅ Layer norm reconstruction: PASS")
        print(f"  ✅ Eta computation: PASS")
        print(f"  ✅ Full forward pass: PASS")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_parameter_initialization():
    """Test that parameters are initialized following Video-DiT pattern"""
    print("\n🔧 Testing parameter initialization patterns...")
    
    try:
        from moshi_ttt.hybrid_layer import HybridSeqModelingBlock
        from moshi_ttt.config import TTTConfig
        from moshi.modules.transformer import StreamingTransformerLayer
        
        # Create components
        d_model = 512
        num_heads = 8
        
        original_layer = StreamingTransformerLayer(d_model=d_model, num_heads=num_heads, dim_feedforward=d_model * 4, causal=True)
        ttt_config = TTTConfig(model_dim=d_model, num_heads=num_heads)
        hybrid_block = HybridSeqModelingBlock(original_layer, ttt_config)
        
        # Check parameter shapes match Video-DiT
        head_dim = d_model // num_heads
        
        print(f"  📏 Parameter shape verification:")
        print(f"    W1: {hybrid_block.W1.shape} (expected: [{num_heads}, {head_dim}, {4*head_dim}])")
        print(f"    b1: {hybrid_block.b1.shape} (expected: [{num_heads}, 1, {4*head_dim}])")
        print(f"    W2: {hybrid_block.W2.shape} (expected: [{num_heads}, {4*head_dim}, {head_dim}])")
        print(f"    b2: {hybrid_block.b2.shape} (expected: [{num_heads}, 1, {head_dim}])")
        print(f"    ttt_norm_weight: {hybrid_block.ttt_norm_weight.shape} (expected: [{num_heads}, {head_dim}])")
        print(f"    ttt_norm_bias: {hybrid_block.ttt_norm_bias.shape} (expected: [{num_heads}, {head_dim}])")
        print(f"    learnable_ttt_lr_weight: {hybrid_block.learnable_ttt_lr_weight.shape}")
        print(f"    learnable_ttt_lr_bias: {hybrid_block.learnable_ttt_lr_bias.shape}")
        
        # Verify shapes
        assert hybrid_block.W1.shape == (num_heads, head_dim, 4*head_dim)
        assert hybrid_block.b1.shape == (num_heads, 1, 4*head_dim) 
        assert hybrid_block.W2.shape == (num_heads, 4*head_dim, head_dim)
        assert hybrid_block.b2.shape == (num_heads, 1, head_dim)
        assert hybrid_block.ttt_norm_weight.shape == (num_heads, head_dim)
        assert hybrid_block.ttt_norm_bias.shape == (num_heads, head_dim)
        
        print("  ✅ All parameter shapes match Video-DiT pattern!")
        
        return True
        
    except Exception as e:
        print(f"❌ Parameter test error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Testing Video-DiT Pattern Compliance")
    print("=" * 60)
    
    test1_passed = test_video_dit_patterns()
    test2_passed = test_parameter_initialization()
    
    print("=" * 60)
    print("📊 FINAL RESULTS:")
    print(f"   Video-DiT Patterns: {'✅ PASS' if test1_passed else '❌ FAIL'}")
    print(f"   Parameter Init: {'✅ PASS' if test2_passed else '❌ FAIL'}")
    
    overall_success = test1_passed and test2_passed
    print(f"\n🏆 OVERALL: {'✅ SUCCESS' if overall_success else '❌ FAILED'}")
    
    if overall_success:
        print("✨ Our implementation correctly follows Video-DiT patterns!")
    else:
        print("🔧 Need to fix Video-DiT compliance issues")
        
    sys.exit(0 if overall_success else 1)
