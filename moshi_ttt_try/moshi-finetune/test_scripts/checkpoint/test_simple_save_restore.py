"""
Simple test of save/restore functionality without complex TTT operations.
"""

import torch
import torch.nn as nn
import sys

# Add paths
sys.path.append('/home/alufr/ttt_tests/moshi-finetune')

from moshi_ttt.model_utils import save_ttt_states, restore_ttt_states


class SimpleTTTLayer(nn.Module):
    """Simple layer that mimics TTT save/restore interface"""
    
    def __init__(self, dim=128):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(dim, dim))
        self.b1 = nn.Parameter(torch.zeros(dim))
        self.W2 = nn.Parameter(torch.randn(dim, dim))
        self.b2 = nn.Parameter(torch.zeros(dim))
        
    def save_ttt_states(self):
        """Save TTT parameter states"""
        return {
            'W1': self.W1.clone().detach(),
            'b1': self.b1.clone().detach(),
            'W2': self.W2.clone().detach(),
            'b2': self.b2.clone().detach()
        }
    
    def restore_ttt_states(self, saved_state):
        """Restore TTT parameter states"""
        if saved_state is None:
            return
        
        with torch.no_grad():
            if 'W1' in saved_state:
                self.W1.copy_(saved_state['W1'])
            if 'b1' in saved_state:
                self.b1.copy_(saved_state['b1'])
            if 'W2' in saved_state:
                self.W2.copy_(saved_state['W2'])
            if 'b2' in saved_state:
                self.b2.copy_(saved_state['b2'])
    
    def forward(self, x):
        h = torch.matmul(x, self.W1) + self.b1
        h = torch.relu(h)
        return torch.matmul(h, self.W2) + self.b2


class SimpleModel(nn.Module):
    """Simple model with TTT layers"""
    
    def __init__(self):
        super().__init__()
        self.ttt_layer1 = SimpleTTTLayer(128)
        self.ttt_layer2 = SimpleTTTLayer(128)
        self.normal_layer = nn.Linear(128, 128)
        
    def forward(self, x):
        x = self.ttt_layer1(x)
        x = self.ttt_layer2(x)
        x = self.normal_layer(x)
        return x


def test_model_level_save_restore():
    """Test model-level save/restore functionality"""
    print("🧪 Testing model-level save/restore...")
    
    model = SimpleModel()
    
    # Test save
    saved_states = save_ttt_states(model)
    if saved_states is None:
        print("❌ No TTT states saved")
        return False
    
    print(f"✅ Saved states from {len(saved_states)} layers")
    
    # Store original values
    original_W1_1 = model.ttt_layer1.W1.clone()
    original_W1_2 = model.ttt_layer2.W1.clone()
    
    # Modify parameters
    with torch.no_grad():
        model.ttt_layer1.W1.fill_(999.0)
        model.ttt_layer2.W1.fill_(777.0)
    
    print("✅ Parameters modified")
    
    # Restore
    success = restore_ttt_states(model, saved_states)
    if not success:
        print("❌ Failed to restore states")
        return False
    
    # Verify restoration
    if not torch.allclose(model.ttt_layer1.W1, original_W1_1, atol=1e-7):
        print("❌ Layer 1 not properly restored")
        return False
    
    if not torch.allclose(model.ttt_layer2.W1, original_W1_2, atol=1e-7):
        print("❌ Layer 2 not properly restored")
        return False
    
    print("✅ All parameters properly restored")
    return True


def test_backward_compatibility():
    """Test that save/restore doesn't break backpropagation"""
    print("\n🧪 Testing backward compatibility...")
    
    model = SimpleModel()
    model.train()
    
    # Create input
    x = torch.randn(4, 32, 128, requires_grad=True)
    
    # Forward pass
    output = model(x)
    loss = output.mean()
    
    print(f"Forward pass: loss = {loss.item():.6f}")
    
    # Save/restore cycle
    saved_states = save_ttt_states(model)
    success = restore_ttt_states(model, saved_states)
    
    if not success:
        print("❌ Save/restore failed")
        return False
    
    print("✅ Save/restore completed")
    
    # Try backward pass
    try:
        loss.backward()
        print("✅ Backward pass succeeded!")
        return True
    except Exception as e:
        print(f"❌ Backward pass failed: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("SIMPLE SAVE/RESTORE TEST")
    print("=" * 60)
    
    # Test 1: Model-level save/restore
    test1_passed = test_model_level_save_restore()
    
    # Test 2: Backward compatibility
    test2_passed = test_backward_compatibility()
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"✅ Model-level save/restore: {'PASS' if test1_passed else 'FAIL'}")
    print(f"✅ Backward compatibility: {'PASS' if test2_passed else 'FAIL'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("Save/restore implementation is working correctly")
    else:
        print("\n❌ Some tests failed")
    
    print("=" * 60)