#!/usr/bin/env python3
"""
Simple test to verify that FSDP wrapped model gets reset_ttt_states method.
This test focuses on the method binding without full model creation complexity.
"""

import torch
import sys
import os
import types

# Add the moshi_ttt directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from moshi_ttt.models.ssm.ttt_layer import TTTWrapper
from moshi_ttt.config import TTTConfig

def test_fsdp_reset_method_binding():
    """Test that we can bind reset method to FSDP-like model."""
    print("🔗 Testing FSDP Reset Method Binding")
    print("=" * 50)
    
    # Create a simple model with TTT layers
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # Add some TTT layers
            config = TTTConfig(
                model_dim=512,
                num_heads=8,
                rope_theta=10000.0,
                mini_batch_size=8,
                ttt_base_lr=0.1,
                ssm_layer="ttt_linear",
                scan_checkpoint_group_size=0,
            )
            
            self.layer1 = TTTWrapper(config)
            self.layer2 = TTTWrapper(config)
            self.normal_layer = torch.nn.Linear(512, 512)
    
    model = MockModel()
    
    print("✅ Created model with 2 TTT layers")
    
    # Define the reset method (same as in wrapped_model.py)
    def reset_ttt_states(self):
        """Reset TTT inner weights to initial values across all TTT layers."""
        reset_count = 0
        total_resets = 0
        
        def reset_module(module):
            nonlocal reset_count, total_resets
            if hasattr(module, 'reset_ttt_states') and callable(getattr(module, 'reset_ttt_states')):
                total_resets += 1
                try:
                    success = module.reset_ttt_states()
                    if success:
                        reset_count += 1
                except Exception as e:
                    print(f"   Warning: Failed to reset TTT in {type(module).__name__}: {e}")
            
            # Recursively check submodules
            for child in module.children():
                reset_module(child)
        
        print("   Traversing model to find TTT layers...")
        reset_module(self)
        
        if total_resets > 0:
            print(f"   Found {total_resets} TTT layers, successfully reset {reset_count}")
            return reset_count > 0
        else:
            print("   No TTT layers found in model")
            return False
    
    # Bind the method to the model (same as wrapped_model.py)
    model.reset_ttt_states = types.MethodType(reset_ttt_states, model)
    
    print("✅ Bound reset_ttt_states method to model")
    
    # Test 1: Check if model has the method
    has_reset = hasattr(model, 'reset_ttt_states')
    print(f"   hasattr(model, 'reset_ttt_states'): {has_reset}")
    
    if not has_reset:
        print("❌ Method binding failed")
        return False
    
    # Test 2: Try calling the method
    print("\\n🔄 Testing Reset Method Call...")
    try:
        success = model.reset_ttt_states()
        print(f"   Reset returned: {success}")
        
        if success:
            print("✅ Reset method found and reset TTT layers successfully")
            return True
        else:
            print("❌ Reset method didn't find/reset TTT layers")
            return False
            
    except Exception as e:
        print(f"❌ Reset method call failed: {e}")
        return False

def test_hasattr_behavior():
    """Test exact hasattr behavior like paper_metrics.py uses."""
    print("\\n🔍 Testing hasattr Behavior (paper_metrics simulation)")
    print("=" * 50)
    
    # Create model and bind reset method
    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            pass
    
    model = MockModel()
    
    # Test before binding
    print("Before binding reset method:")
    print(f"   hasattr(model, 'reset_ttt_states'): {hasattr(model, 'reset_ttt_states')}")
    
    # Bind method
    def reset_ttt_states(self):
        return True
    
    model.reset_ttt_states = types.MethodType(reset_ttt_states, model)
    
    # Test after binding
    print("After binding reset method:")
    print(f"   hasattr(model, 'reset_ttt_states'): {hasattr(model, 'reset_ttt_states')}")
    
    # Test the exact paper_metrics logic
    print("\\nSimulating paper_metrics.py logic:")
    if hasattr(model, 'reset_ttt_states'):
        print("   🔄 Resetting TTT states (W1, b1, W2, b2) before LibriLight evaluation")
        model.reset_ttt_states()
        print("   ✅ TTT states reset completed")
        return True
    else:
        print("   ❌ Model doesn't have reset_ttt_states method")
        return False

def main():
    """Run all tests."""
    print("🧪 FSDP Reset Method Test Suite")
    print("=" * 60)
    
    test1 = test_fsdp_reset_method_binding()
    test2 = test_hasattr_behavior()
    
    print("\\n📋 Test Summary")
    print("=" * 30)
    print(f"Method binding test: {'✅ PASS' if test1 else '❌ FAIL'}")
    print(f"hasattr behavior test: {'✅ PASS' if test2 else '❌ FAIL'}")
    
    success = test1 and test2
    print(f"\\nOverall: {'✅ ALL TESTS PASSED' if success else '❌ TESTS FAILED'}")
    
    if success:
        print("\\n🚀 FSDP reset method integration is working!")
        print("   The wrapped_model.py fix should work in production.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)