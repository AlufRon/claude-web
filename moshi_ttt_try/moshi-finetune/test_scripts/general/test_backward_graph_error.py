"""
Test to replicate the backward graph error:
"Trying to backward through the graph a second time"

This error occurs when we call init_weights() on TTT parameters during training,
which destroys the computation graph needed for backpropagation.
"""

import torch
import torch.nn as nn
import sys
import os

# Add paths for imports
sys.path.append('/home/alufr/ttt_tests/moshi-finetune')
sys.path.append('/home/alufr/ttt_tests/ttt-video-dit')

from moshi_ttt.ttt_layer import TTTMLP
from moshi_ttt.config import TTTConfig


def test_backward_graph_error():
    """
    Replicate the exact scenario that causes the backward graph error:
    1. Create TTT layer with requires_grad=True parameters
    2. Do forward pass that builds computation graph
    3. Call init_weights() which destroys the graph
    4. Try to call backward() -> ERROR
    """
    
    print("🔍 Testing backward graph error replication...")
    
    # Create TTT config matching our Moshi setup
    config = TTTConfig(
        model_dim=4096,
        num_heads=32,
        mini_batch_size=4,
        ttt_base_lr=0.5
    )
    
    # Create TTT layer
    ttt_layer = TTTMLP(config, use_kernel=False)
    ttt_layer.train()  # Important: in training mode
    
    # Create dummy input that requires grad (simulating training data)
    batch_size = 1
    seq_len = 8  # Multiple of mini_batch_size=4
    x = torch.randn(batch_size, seq_len, config.model_dim, requires_grad=True)
    
    # Create sequence metadata
    from moshi_ttt.utils import SequenceMetadata
    seq_metadata = SequenceMetadata(
        seq_text_length=0,
        is_multiscene=False
    )
    
    print(f"✅ Setup complete: TTT layer with {sum(p.numel() for p in ttt_layer.parameters())} parameters")
    
    # Step 1: Forward pass that builds computation graph
    print("📈 Step 1: Forward pass (builds computation graph)")
    output = ttt_layer(x, seq_metadata)
    loss = output.sum()  # Simple loss for testing
    print(f"   Output shape: {output.shape}, Loss: {loss.item():.6f}")
    
    # Verify computation graph exists
    print(f"   Computation graph exists: {x.grad_fn is not None or loss.grad_fn is not None}")
    
    # Step 2: CRITICAL - Call init_weights() like our reset function does
    print("🔄 Step 2: Call init_weights() (this destroys computation graph)")
    
    # This is exactly what our reset function does - and it breaks the graph
    ttt_layer.init_weights()
    print("   init_weights() called - TTT parameters reinitialized")
    
    # Step 3: Try to call backward - this should fail
    print("⬅️  Step 3: Call backward() - this should fail with 'Trying to backward through the graph a second time'")
    
    try:
        loss.backward()
        print("❌ ERROR: backward() succeeded when it should have failed!")
        return False
    except RuntimeError as e:
        if "Trying to backward through the graph a second time" in str(e):
            print(f"✅ SUCCESS: Replicated the exact error!")
            print(f"   Error message: {str(e)}")
            return True
        else:
            print(f"❌ ERROR: Got different error: {str(e)}")
            return False


def test_correct_approach():
    """
    Test the correct approach: using torch.no_grad() around parameter updates
    """
    
    print("\n🔧 Testing correct approach with torch.no_grad()...")
    
    # Create TTT config
    config = TTTConfig(
        model_dim=4096,
        num_heads=32,
        mini_batch_size=4,
        ttt_base_lr=0.5
    )
    
    # Create TTT layer
    ttt_layer = TTTMLP(config, use_kernel=False)
    ttt_layer.train()
    
    # Create dummy input
    batch_size = 1
    seq_len = 8
    x = torch.randn(batch_size, seq_len, config.model_dim, requires_grad=True)
    
    # Create sequence metadata
    from moshi_ttt.utils import SequenceMetadata
    seq_metadata = SequenceMetadata(
        seq_text_length=0,
        is_multiscene=False
    )
    
    # Step 1: Forward pass
    print("📈 Step 1: Forward pass")
    output = ttt_layer(x, seq_metadata)
    loss = output.sum()
    print(f"   Loss: {loss.item():.6f}")
    
    # Step 2: CORRECT - Reset parameters with no_grad()
    print("🔄 Step 2: Reset parameters with torch.no_grad()")
    
    # Store original parameter values for verification
    W1_before = ttt_layer.W1.clone()
    W2_before = ttt_layer.W2.clone()
    
    with torch.no_grad():
        # Reset parameters manually without calling init_weights()
        nn.init.normal_(ttt_layer.W1, mean=0.0, std=0.02)
        nn.init.zeros_(ttt_layer.b1)
        nn.init.normal_(ttt_layer.W2, mean=0.0, std=0.02)  
        nn.init.zeros_(ttt_layer.b2)
    
    # Verify parameters actually changed
    W1_after = ttt_layer.W1.clone()
    W2_after = ttt_layer.W2.clone()
    
    W1_changed = not torch.allclose(W1_before, W1_after)
    W2_changed = not torch.allclose(W2_before, W2_after)
    
    print(f"   Parameters changed: W1={W1_changed}, W2={W2_changed}")
    
    # Step 3: Try backward - this should work
    print("⬅️  Step 3: Call backward() - this should work")
    
    try:
        loss.backward()
        print("✅ SUCCESS: backward() worked correctly!")
        print(f"   Input gradients computed: {x.grad is not None}")
        return True
    except RuntimeError as e:
        print(f"❌ ERROR: backward() failed: {str(e)}")
        return False


if __name__ == "__main__":
    print("=" * 80)
    print("BACKWARD GRAPH ERROR REPLICATION TEST")
    print("=" * 80)
    
    # Test the problematic approach
    error_replicated = test_backward_graph_error()
    
    # Test the correct approach
    correct_approach_works = test_correct_approach()
    
    print("\n" + "=" * 80)
    print("SUMMARY:")
    print(f"✅ Error replicated: {error_replicated}")
    print(f"✅ Correct approach works: {correct_approach_works}")
    
    if error_replicated and correct_approach_works:
        print("\n🎯 CONCLUSION: The problem is calling init_weights() during training.")
        print("   SOLUTION: Use torch.no_grad() around manual parameter reset.")
        print("   Video-DiT never resets TTT weights during training - only at initialization.")
    else:
        print("\n❌ Test inconclusive - need further investigation")
    
    print("=" * 80)