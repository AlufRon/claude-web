#!/usr/bin/env python3
"""
Test the actual hybrid layer functionality
"""

import torch
import sys
import os

# Add paths
sys.path.append('/home/alufr/ttt_tests/moshi')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_hybrid_layer_instantiation():
    """Test that we can create and use the hybrid layer"""
    print("🧪 Testing hybrid layer instantiation and basic usage...")
    
    try:
        # Import Moshi components
        from moshi.modules.transformer import StreamingTransformerLayer
        
        # Import our hybrid components
        from moshi_ttt.hybrid_layer import HybridStreamingTransformerLayer
        from moshi_ttt.config import TTTConfig
        
        print("✅ All imports successful")
        
        # Create a base Moshi layer
        d_model = 512
        num_heads = 8
        
        original_layer = StreamingTransformerLayer(
            d_model=d_model,
            num_heads=num_heads,
            dim_feedforward=d_model * 4,
            causal=True,
        )
        print(f"✅ Created original StreamingTransformerLayer: d_model={d_model}, num_heads={num_heads}")
        
        # Create TTT config
        ttt_config = TTTConfig(
            model_dim=d_model,
            num_heads=num_heads,
            mini_batch_size=16,
            ttt_base_lr=0.1
        )
        print(f"✅ Created TTT config: {ttt_config}")
        
        # Create hybrid layer
        hybrid_layer = HybridStreamingTransformerLayer(
            original_layer=original_layer,
            ttt_config=ttt_config
        )
        print(f"✅ Created hybrid layer: {type(hybrid_layer)}")
        
        # Test forward pass
        batch_size = 2
        seq_len = 32
        x = torch.randn(batch_size, seq_len, d_model)
        
        print(f"✅ Created input tensor: {x.shape}")
        
        # Forward pass
        hybrid_layer.eval()
        with torch.no_grad():
            output = hybrid_layer(x)
            
        print(f"✅ Hybrid layer forward pass: {x.shape} → {output.shape}")
        
        # Verify shapes
        assert output.shape == x.shape, f"Shape mismatch: {output.shape} vs {x.shape}"
        print("✅ Output shape matches input shape")
        
        # Check for NaN/Inf
        assert torch.isfinite(output).all(), "Output contains NaN or Inf values"
        print("✅ Output is finite")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Testing Hybrid Layer Functionality")
    print("=" * 50)
    
    success = test_hybrid_layer_instantiation()
    
    print("=" * 50)
    if success:
        print("🎉 HYBRID LAYER TEST SUCCESS!")
        print("✨ The hybrid layer can be instantiated and used!")
    else:
        print("❌ HYBRID LAYER TEST FAILED!")
        print("🔧 Need to debug and fix issues")
    
    sys.exit(0 if success else 1)
