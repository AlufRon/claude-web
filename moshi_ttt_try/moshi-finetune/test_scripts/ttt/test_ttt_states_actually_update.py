#!/usr/bin/env python3
"""
Test to verify TTT states actually update during processing.
The previous test showed 0 changes, which might be because we're in eval mode
or because the test sequences are too simple.
"""

import torch
import sys
import os

# Add the moshi_ttt directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from moshi_ttt.models.ssm.ttt_layer import TTTWrapper
from moshi_ttt.config import TTTConfig
from moshi_ttt.utils import SequenceMetadata

def test_ttt_actually_updates():
    """Test that TTT states actually update when processing sequences."""
    print("🧠 Testing TTT State Updates During Processing")
    print("=" * 60)
    
    # Create TTT layer
    config = TTTConfig(
        model_dim=512,
        num_heads=8,
        rope_theta=10000.0,
        mini_batch_size=8,
        ttt_base_lr=0.1,
        ssm_layer="ttt_linear",
        scan_checkpoint_group_size=0,
    )
    
    ttt_layer = TTTWrapper(config)
    ttt_layer.train()  # Ensure training mode
    
    # Create a longer, more complex sequence that should trigger TTT updates
    batch_size = 2
    seq_len = 64  # Multiple mini-batches
    
    # Use different sequences to create gradient
    seq1 = torch.randn(batch_size, seq_len, config.model_dim) * 2.0  # Larger values
    seq2 = torch.randn(batch_size, seq_len, config.model_dim) * 2.0  # Different data
    
    seq_metadata = SequenceMetadata(
        init_offset=None,
        base_offset=None,
        text_length=0,
        num_chunks=1,
        seq_text_length=0,
        is_multiscene=False,
    )
    
    # Record initial TTT weights
    initial_W1 = ttt_layer.ttt.W1.data.clone()
    initial_b1 = ttt_layer.ttt.b1.data.clone()
    
    print(f"Initial W1 norm: {initial_W1.norm().item():.6f}")
    print(f"Initial b1 norm: {initial_b1.norm().item():.6f}")
    
    # Process first sequence
    print("\n📊 Processing complex sequence 1...")
    with torch.enable_grad():  # Ensure gradients are enabled
        result1 = ttt_layer.forward(seq1, seq_metadata)
    
    after_seq1_W1 = ttt_layer.ttt.W1.data.clone()
    after_seq1_b1 = ttt_layer.ttt.b1.data.clone()
    
    w1_change_1 = (after_seq1_W1 - initial_W1).norm().item()
    b1_change_1 = (after_seq1_b1 - initial_b1).norm().item()
    
    print(f"After Seq1 W1 norm: {after_seq1_W1.norm().item():.6f} (Δ={w1_change_1:.6f})")
    print(f"After Seq1 b1 norm: {after_seq1_b1.norm().item():.6f} (Δ={b1_change_1:.6f})")
    
    # Reset states
    print("\n🔄 Resetting TTT states...")
    reset_worked = ttt_layer.reset_ttt_states()
    
    after_reset_W1 = ttt_layer.ttt.W1.data.clone()
    after_reset_b1 = ttt_layer.ttt.b1.data.clone()
    
    # Process second sequence
    print("\n📚 Processing complex sequence 2...")
    with torch.enable_grad():
        result2 = ttt_layer.forward(seq2, seq_metadata)
    
    after_seq2_W1 = ttt_layer.ttt.W1.data.clone()
    after_seq2_b1 = ttt_layer.ttt.b1.data.clone()
    
    w1_change_2 = (after_seq2_W1 - after_reset_W1).norm().item()
    b1_change_2 = (after_seq2_b1 - after_reset_b1).norm().item()
    
    print(f"After Seq2 W1 norm: {after_seq2_W1.norm().item():.6f} (Δ={w1_change_2:.6f})")
    print(f"After Seq2 b1 norm: {after_seq2_b1.norm().item():.6f} (Δ={b1_change_2:.6f})")
    
    # Test without reset (contamination case)
    print("\n🚨 Testing contamination (no reset between sequences)...")
    
    # Reset to clean state
    ttt_layer.reset_ttt_states()
    clean_W1 = ttt_layer.ttt.W1.data.clone()
    
    # Process seq1 then seq2 without reset
    with torch.enable_grad():
        ttt_layer.forward(seq1, seq_metadata)
        contaminated_W1 = ttt_layer.ttt.W1.data.clone()
        ttt_layer.forward(seq2, seq_metadata)  # This uses contaminated state
        final_contaminated_W1 = ttt_layer.ttt.W1.data.clone()
    
    contamination_change = (final_contaminated_W1 - contaminated_W1).norm().item()
    
    print(f"Seq2 change with contamination: {contamination_change:.6f}")
    print(f"Seq2 change with clean reset: {w1_change_2:.6f}")
    
    # Summary
    print("\n📋 Summary")
    print("=" * 60)
    print(f"TTT states actually update: {w1_change_1 > 1e-6 or b1_change_1 > 1e-6}")
    print(f"Reset method works: {reset_worked}")
    print(f"Reset eliminates contamination: {abs(w1_change_2 - contamination_change) > 1e-6}")
    print(f"Sequence 1 W1 change: {w1_change_1:.6f}")
    print(f"Sequence 2 W1 change (clean): {w1_change_2:.6f}")
    print(f"Sequence 2 W1 change (contaminated): {contamination_change:.6f}")
    
    if w1_change_1 < 1e-6 and b1_change_1 < 1e-6:
        print("\n⚠️  WARNING: TTT states are not updating during processing!")
        print("   This might be expected if TTT updates happen during the scan operation")
        print("   and require multiple mini-batches to show accumulated changes.")
    else:
        print("\n✅ TTT states are updating correctly during processing")

if __name__ == "__main__":
    test_ttt_actually_updates()