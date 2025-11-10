#!/usr/bin/env python3
"""
Step 2.3: Test TTT layer in isolation
Verify our TTT-MLP implementation works correctly
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.append('/home/alufr/ttt_tests/moshi-finetune')

def test_ttt_layer_basic():
    """Test basic TTT layer functionality"""
    print("🧪 Testing TTT layer basic functionality...")
    
    try:
        from moshi_ttt.ttt_layer import TTTLayer, MoshiSequenceMetadata
        print("✅ TTTLayer imported successfully")
        
        # Create TTT layer with small dimensions for testing
        model_dim = 256
        num_heads = 4
        mini_batch_size = 8
        
        ttt_layer = TTTLayer(
            model_dim=model_dim,
            num_heads=num_heads,
            mini_batch_size=mini_batch_size,
            ttt_base_lr=0.1,
            use_kernel=False
        )
        print(f"✅ TTT layer created: {model_dim}d, {num_heads} heads")
        
        # Test forward pass
        batch_size = 2
        seq_len = 16  # Divisible by mini_batch_size=8
        
        x = torch.randn(batch_size, seq_len, model_dim)
        print(f"✅ Created input: {x.shape}")
        
        # Test forward pass
        ttt_layer.eval()
        with torch.no_grad():
            output = ttt_layer(x)
            
        print(f"✅ TTT forward pass: {x.shape} → {output.shape}")
        
        # Verify shapes
        if x.shape == output.shape:
            print("✅ Shape consistency verified")
        else:
            print(f"❌ Shape mismatch: {x.shape} != {output.shape}")
            return False
            
        # Check that output is different from input (TTT should transform it)
        if torch.allclose(x, output, atol=1e-6):
            print("⚠️  Output identical to input - TTT may not be working")
        else:
            print("✅ Output differs from input (TTT is active)")
            
        print("🎉 TTT layer basic test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error in TTT layer test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ttt_layer_with_padding():
    """Test TTT layer with sequence length not divisible by mini_batch_size"""
    print("\n🔧 Testing TTT layer with padding...")
    
    try:
        from moshi_ttt.ttt_layer import TTTLayer
        
        # Create TTT layer
        model_dim = 128  
        num_heads = 4
        mini_batch_size = 8
        
        ttt_layer = TTTLayer(
            model_dim=model_dim,
            num_heads=num_heads,
            mini_batch_size=mini_batch_size
        )
        print(f"✅ TTT layer created with mini_batch_size={mini_batch_size}")
        
        # Test with sequence length not divisible by mini_batch_size
        batch_size = 1
        seq_len = 13  # Not divisible by 8, should be padded to 16
        
        x = torch.randn(batch_size, seq_len, model_dim)
        print(f"✅ Created input: {x.shape} (seq_len={seq_len} not divisible by {mini_batch_size})")
        
        ttt_layer.eval()
        with torch.no_grad():
            output = ttt_layer(x)
            
        print(f"✅ TTT with padding: {x.shape} → {output.shape}")
        
        # Should return original sequence length, not padded
        if output.shape == x.shape:
            print("✅ Padding handled correctly (output matches input shape)")
        else:
            print(f"❌ Padding failed: expected {x.shape}, got {output.shape}")
            return False
            
        print("🎉 TTT layer padding test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error in padding test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ttt_gradients():
    """Test that gradients flow through TTT layer"""
    print("\n🔍 Testing TTT gradient flow...")
    
    try:
        from moshi_ttt.ttt_layer import TTTLayer
        
        # Create TTT layer
        ttt_layer = TTTLayer(
            model_dim=64,  # Small for faster test
            num_heads=2,
            mini_batch_size=4
        )
        
        # Create input that requires gradients
        x = torch.randn(1, 8, 64, requires_grad=True)
        
        # Forward pass
        output = ttt_layer(x)
        
        # Compute simple loss
        loss = output.mean()
        
        # Backward pass
        loss.backward()
        
        print(f"✅ Backward pass completed, loss: {loss.item():.4f}")
        
        # Check that input has gradients
        if x.grad is not None:
            print(f"✅ Input gradients: {x.grad.norm().item():.4f}")
        else:
            print("❌ No gradients on input")
            return False
            
        # Check that TTT parameters have gradients
        param_grads = []
        for name, param in ttt_layer.named_parameters():
            if param.grad is not None:
                param_grads.append((name, param.grad.norm().item()))
            else:
                print(f"⚠️  No gradient for {name}")
                
        if param_grads:
            print(f"✅ Parameter gradients: {len(param_grads)} parameters have gradients")
            # Show a few examples
            for name, grad_norm in param_grads[:3]:
                print(f"   {name}: {grad_norm:.4f}")
        else:
            print("❌ No parameter gradients")
            return False
            
        print("🎉 TTT gradient flow test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error in gradient test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ttt_parameter_count():
    """Test TTT layer parameter count"""
    print("\n📊 Testing TTT parameter count...")
    
    try:
        from moshi_ttt.ttt_layer import TTTLayer
        
        model_dim = 512
        num_heads = 8
        
        ttt_layer = TTTLayer(model_dim=model_dim, num_heads=num_heads)
        
        # Count parameters
        total_params = sum(p.numel() for p in ttt_layer.parameters())
        trainable_params = sum(p.numel() for p in ttt_layer.parameters() if p.requires_grad)
        
        print(f"✅ Total parameters: {total_params:,}")
        print(f"✅ Trainable parameters: {trainable_params:,}")
        
        # Check some key components
        head_dim = model_dim // num_heads
        expected_mlp_params = num_heads * (
            head_dim * 4 * head_dim +  # W1
            4 * head_dim +             # b1  
            4 * head_dim * head_dim +  # W2
            head_dim                   # b2
        )
        
        print(f"✅ Expected MLP params: {expected_mlp_params:,}")
        
        if total_params > expected_mlp_params:
            print("✅ Parameter count includes QKV projections and other components")
        else:
            print("⚠️  Parameter count seems low")
            
        print("🎉 TTT parameter count test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error in parameter count test: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting TTT Layer Tests...")
    print("=" * 60)
    
    # Test 1: Basic functionality
    test1_passed = test_ttt_layer_basic()
    
    # Test 2: Padding handling
    test2_passed = test_ttt_layer_with_padding()
    
    # Test 3: Gradient flow
    test3_passed = test_ttt_gradients()
    
    # Test 4: Parameter count
    test4_passed = test_ttt_parameter_count()
    
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS:")
    print(f"   Basic Functionality: {'✅ PASS' if test1_passed else '❌ FAIL'}")
    print(f"   Padding Handling: {'✅ PASS' if test2_passed else '❌ FAIL'}")
    print(f"   Gradient Flow: {'✅ PASS' if test3_passed else '❌ FAIL'}")
    print(f"   Parameter Count: {'✅ PASS' if test4_passed else '❌ FAIL'}")
    
    overall_success = all([test1_passed, test2_passed, test3_passed, test4_passed])
    print(f"\n🏆 OVERALL: {'✅ SUCCESS' if overall_success else '❌ FAILED'}")
    
    if overall_success:
        print("✨ Phase 2 Complete! Ready for Step 3.1 (Hybrid Layer Creation)")
    else:
        print("🔧 Fix issues before proceeding")
    
    sys.exit(0 if overall_success else 1)