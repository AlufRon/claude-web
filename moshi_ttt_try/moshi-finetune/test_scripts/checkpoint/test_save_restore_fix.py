"""
Test the TTT save/restore fix to ensure it prevents computation graph errors.

This test validates that:
1. save_ttt_states() and restore_ttt_states() work correctly
2. No computation graph errors occur during training
3. TTT states are properly isolated between evaluation and training
"""

import torch
import torch.nn as nn
import sys
import os
import logging

# Add paths for imports
sys.path.append('/home/alufr/ttt_tests/moshi-finetune')
sys.path.append('/home/alufr/ttt_tests/ttt-video-dit')

from moshi_ttt.ttt_layer import TTTMLP
from moshi_ttt.config import TTTConfig
from moshi_ttt.model_utils import save_ttt_states, restore_ttt_states, verify_ttt_state_isolation

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockTTTModel(nn.Module):
    """Mock model containing TTT layers for testing"""
    
    def __init__(self):
        super().__init__()
        config = TTTConfig(
            model_dim=512,
            num_heads=8,
            mini_batch_size=4,
            ttt_base_lr=0.5
        )
        
        # Create multiple TTT layers like a real model
        self.layer1 = TTTMLP(config, use_kernel=False)
        self.layer2 = TTTMLP(config, use_kernel=False)
        self.linear = nn.Linear(512, 512)
        
    def forward(self, x):
        from moshi_ttt.utils import SequenceMetadata
        seq_metadata = SequenceMetadata(seq_text_length=0, is_multiscene=False)
        
        x = self.layer1(x, seq_metadata)
        x = self.layer2(x, seq_metadata)
        x = self.linear(x)
        return x


def test_save_restore_basic():
    """Test basic save and restore functionality"""
    print("🧪 Testing basic save/restore functionality...")
    
    model = MockTTTModel()
    model.train()
    
    # Save initial state
    initial_state = save_ttt_states(model)
    assert initial_state is not None, "Failed to save TTT states"
    
    # Verify we can access parameters
    layer1_W1_initial = model.layer1.W1.clone()
    layer2_W1_initial = model.layer2.W1.clone()
    
    # Modify parameters (simulate evaluation learning)
    with torch.no_grad():
        model.layer1.W1.fill_(999.0)  # Obviously different values
        model.layer2.W1.fill_(777.0)
    
    # Verify parameters changed
    assert not torch.allclose(model.layer1.W1, layer1_W1_initial)
    assert not torch.allclose(model.layer2.W1, layer2_W1_initial)
    print("✅ Parameters successfully modified")
    
    # Restore state
    success = restore_ttt_states(model, initial_state)
    assert success, "Failed to restore TTT states"
    
    # Verify restoration
    assert torch.allclose(model.layer1.W1, layer1_W1_initial, atol=1e-7)
    assert torch.allclose(model.layer2.W1, layer2_W1_initial, atol=1e-7)
    print("✅ Parameters successfully restored")
    
    return True


def test_no_computation_graph_error():
    """Test that save/restore doesn't break computation graph"""
    print("\n🧪 Testing computation graph preservation...")
    
    model = MockTTTModel()
    model.train()
    
    # Create input that requires gradients
    x = torch.randn(2, 8, 512, requires_grad=True)
    
    # Forward pass (builds computation graph)
    output = model(x)
    loss = output.mean()
    
    print(f"Forward pass completed, loss: {loss.item():.6f}")
    
    # Save TTT states (should not affect graph)
    saved_states = save_ttt_states(model)
    print("TTT states saved")
    
    # Restore TTT states (should not affect graph)
    success = restore_ttt_states(model, saved_states)
    assert success
    print("TTT states restored")
    
    # Backward pass should work without errors
    try:
        loss.backward()
        print("✅ Backward pass succeeded - no computation graph errors!")
        return True
    except RuntimeError as e:
        print(f"❌ Backward pass failed: {e}")
        return False


def test_evaluation_isolation():
    """Test that evaluation changes are properly isolated from training"""
    print("\n🧪 Testing evaluation isolation...")
    
    model = MockTTTModel()
    model.train()
    
    # Save training state
    training_state = save_ttt_states(model)
    
    # Simulate evaluation mode
    model.eval()
    
    # Create evaluation input
    eval_input = torch.randn(1, 8, 512)
    
    # Forward pass in eval mode (TTT adapts)
    with torch.no_grad():
        eval_output = model(eval_input)
    
    # Simulate evaluation learning by modifying TTT parameters
    with torch.no_grad():
        model.layer1.W1 += torch.randn_like(model.layer1.W1) * 0.1
        model.layer2.W2 += torch.randn_like(model.layer2.W2) * 0.1
    
    print("Simulated evaluation learning (TTT parameters modified)")
    
    # Save post-evaluation state for verification
    post_eval_state = save_ttt_states(model)
    
    # Return to training mode and restore pre-evaluation state
    model.train()
    success = restore_ttt_states(model, training_state)
    assert success
    
    # Save post-restore state for verification
    post_restore_state = save_ttt_states(model)
    
    # Verify isolation worked
    isolation_verified = verify_ttt_state_isolation(model, training_state, post_restore_state)
    assert isolation_verified, "TTT state isolation verification failed"
    
    print("✅ Evaluation isolation verified - no contamination")
    return True


def test_error_handling():
    """Test error handling with invalid states"""
    print("\n🧪 Testing error handling...")
    
    model = MockTTTModel()
    
    # Test restore with None state
    success = restore_ttt_states(model, None)
    assert success  # Should succeed gracefully
    print("✅ Handled None state correctly")
    
    # Test restore with empty state
    success = restore_ttt_states(model, {})
    assert success  # Should succeed gracefully  
    print("✅ Handled empty state correctly")
    
    return True


def test_multiple_save_restore_cycles():
    """Test multiple save/restore cycles"""
    print("\n🧪 Testing multiple save/restore cycles...")
    
    model = MockTTTModel()
    model.train()
    
    original_W1 = model.layer1.W1.clone()
    
    # Multiple cycles
    for i in range(3):
        # Save state
        saved_state = save_ttt_states(model)
        
        # Modify parameters
        with torch.no_grad():
            model.layer1.W1.add_(torch.randn_like(model.layer1.W1) * 0.1)
        
        # Restore state
        success = restore_ttt_states(model, saved_state)
        assert success
        
        # Verify restoration
        assert torch.allclose(model.layer1.W1, original_W1, atol=1e-6)
        print(f"✅ Cycle {i+1} completed successfully")
    
    return True


if __name__ == "__main__":
    print("=" * 80)
    print("TTT SAVE/RESTORE FIX VALIDATION")
    print("=" * 80)
    
    test_results = []
    
    # Run all tests
    try:
        test_results.append(("Basic save/restore", test_save_restore_basic()))
    except Exception as e:
        print(f"❌ Basic save/restore test failed: {e}")
        test_results.append(("Basic save/restore", False))
    
    try:
        test_results.append(("Computation graph preservation", test_no_computation_graph_error()))
    except Exception as e:
        print(f"❌ Computation graph test failed: {e}")
        test_results.append(("Computation graph preservation", False))
    
    try:
        test_results.append(("Evaluation isolation", test_evaluation_isolation()))
    except Exception as e:
        print(f"❌ Evaluation isolation test failed: {e}")
        test_results.append(("Evaluation isolation", False))
    
    try:
        test_results.append(("Error handling", test_error_handling()))
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        test_results.append(("Error handling", False))
    
    try:
        test_results.append(("Multiple cycles", test_multiple_save_restore_cycles()))
    except Exception as e:
        print(f"❌ Multiple cycles test failed: {e}")
        test_results.append(("Multiple cycles", False))
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST RESULTS:")
    all_passed = True
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 80)
    if all_passed:
        print("🎉 ALL TESTS PASSED - Save/restore fix is working correctly!")
        print("✅ Ready for training pipeline testing")
    else:
        print("❌ Some tests failed - fix needs more work")
    
    print("=" * 80)