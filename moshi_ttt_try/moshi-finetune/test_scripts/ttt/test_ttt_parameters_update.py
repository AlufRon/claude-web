#!/usr/bin/env python3
"""
Test script to verify TTT parameters can update properly.
This will create the exact same setup as training but in isolation.
"""

import sys
import os
import torch
import torch.nn as nn

# Set up environment
sys.path.append('/home/alufr/ttt_tests/moshi-finetune')

def test_ttt_parameter_updates():
    """Test that TTT parameters can receive gradients and update"""
    print("🧪 Testing TTT parameter updates...")
    
    # Import our TTT components
    from moshi_ttt.config import TTTConfig
    from moshi_ttt.hybrid_layer import HybridSeqModelingBlock
    
    # Add Moshi path and create a mock StreamingTransformerLayer
    sys.path.append('/home/alufr/ttt_tests/moshi/moshi')
    from moshi.modules.transformer import StreamingTransformerLayer
    
    # Create TTT config
    ttt_config = TTTConfig(
        model_dim=4096,
        num_heads=32,
        ttt_base_lr=0.1,
        mini_batch_size=2
    )
    
    # Create a mock original layer (just need the basic structure)
    class MockStreamingTransformerLayer(nn.Module):
        def __init__(self, d_model=4096):
            super().__init__()
            self.d_model = d_model
            self._streaming_state = None
            self.cross_attention = None  # No cross attention
            
        def _sa_block(self, x, cross_attention_src=None):
            return x + 0.01 * torch.randn_like(x)  # Add small random change
            
        def _attn_block(self, x, cross_attention_src=None):
            return x + 0.01 * torch.randn_like(x)  # Add small random change
            
        def _ff_block(self, x):
            return x + 0.01 * torch.randn_like(x)  # Add small random change
    
    original_layer = MockStreamingTransformerLayer()
    
    # Create hybrid block
    print("📦 Creating HybridSeqModelingBlock...")
    hybrid_block = HybridSeqModelingBlock(original_layer, ttt_config)
    
    # Move to appropriate device
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    hybrid_block = hybrid_block.to(device)
    
    print(f"🎯 Using device: {device}")
    
    # Create test input
    batch_size = 2
    seq_len = 50  # Short sequence for testing
    
    x = torch.randn(batch_size, seq_len, ttt_config.model_dim, device=device, requires_grad=True)
    print(f"📊 Input shape: {x.shape}")
    
    # Create optimizer for TTT parameters only
    ttt_params = []
    for name, param in hybrid_block.named_parameters():
        if any(ttt_pattern in name.lower() for ttt_pattern in ['ttt', 'wq', 'wk', 'wv', 'wo', 'W1', 'W2', 'b1', 'b2', 'learnable_ttt']):
            ttt_params.append(param)
            print(f"   🎯 TTT param: {name} - shape: {param.shape}")
    
    print(f"📋 Found {len(ttt_params)} TTT parameters")
    
    if len(ttt_params) == 0:
        print("❌ No TTT parameters found!")
        return False
    
    optimizer = torch.optim.SGD(ttt_params, lr=0.1)
    
    # Record initial values
    initial_values = {}
    for name, param in hybrid_block.named_parameters():
        if any(ttt_pattern in name.lower() for ttt_pattern in ['learnable_ttt_lr']):
            initial_values[name] = param.data.clone()
            print(f"📋 Initial {name}: {param.data.mean().item():.6f}")
    
    print("\n🔄 Running forward and backward pass...")
    
    # Forward pass
    try:
        output = hybrid_block(x)
        print(f"✅ Forward pass successful - output shape: {output.shape}")
        
        # Create a simple loss
        loss = output.mean()
        print(f"📊 Loss: {loss.item():.6f}")
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        print("✅ Backward pass successful")
        
        # Check gradients
        grad_found = False
        for name, param in hybrid_block.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                if grad_norm > 1e-8:
                    print(f"✅ Gradient found for {name}: norm = {grad_norm:.2e}")
                    grad_found = True
                    
        if not grad_found:
            print("❌ No gradients found!")
            return False
        
        # Optimizer step
        optimizer.step()
        print("✅ Optimizer step successful")
        
        # Check if parameters changed
        changed = False
        for name, param in hybrid_block.named_parameters():
            if name in initial_values:
                initial_val = initial_values[name].mean().item()
                new_val = param.data.mean().item()
                diff = abs(new_val - initial_val)
                print(f"📊 {name}: {initial_val:.6f} → {new_val:.6f} (Δ: {diff:.2e})")
                if diff > 1e-8:
                    changed = True
                    
        if changed:
            print("✅ TTT parameters successfully updated!")
            return True
        else:
            print("❌ TTT parameters did not change!")
            return False
            
    except Exception as e:
        print(f"❌ Error during forward/backward pass: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🚀 TTT Parameter Update Test")
    print("=" * 50)
    
    # Set up CUDA if available
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        print(f"🎯 CUDA available - using GPU {torch.cuda.current_device()}")
    else:
        print("⚠️ CUDA not available - using CPU")
    
    success = test_ttt_parameter_updates()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ CONCLUSION: TTT parameters can update correctly")
        print("   The issue must be in the training setup, not the TTT implementation")
    else:
        print("❌ CONCLUSION: TTT parameters cannot update")
        print("   There's still a fundamental issue with the TTT implementation")

if __name__ == "__main__":
    main()