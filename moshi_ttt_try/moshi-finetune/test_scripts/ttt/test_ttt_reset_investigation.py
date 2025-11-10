#!/usr/bin/env python3
"""
Test to investigate TTT reset behavior and create a fix.

This test replicates the evaluation flow to see:
1. Does the model actually have reset_ttt_states method?
2. Do TTT inner weights (W1, b1, W2, b2) actually get reset?
3. What happens to TTT states across evaluation tasks?
"""

import torch
import sys
import os

# Add the moshi_ttt directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from moshi_ttt.models.ssm.ttt_layer import TTTWrapper
from moshi_ttt.config import TTTConfig
from moshi_ttt.utils import SequenceMetadata

def test_model_reset_method():
    """Test if model has reset_ttt_states method like paper_metrics.py expects."""
    print("🔍 Testing Model Reset Method")
    print("=" * 50)
    
    # Create TTT layer like the actual training does
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
    
    # Test 1: Does TTTWrapper have reset_ttt_states?
    has_reset = hasattr(ttt_layer, 'reset_ttt_states')
    print(f"1. TTTWrapper has reset_ttt_states: {has_reset}")
    
    # Test 2: Does the underlying TTT layer have reset method?
    has_ttt_reset = hasattr(ttt_layer.ttt, 'reset_ttt_states')
    print(f"2. TTT.ttt has reset_ttt_states: {has_ttt_reset}")
    
    # Test 3: Check what methods TTTWrapper actually has
    print(f"3. TTTWrapper methods: {[m for m in dir(ttt_layer) if not m.startswith('_')]}")
    
    # Test 4: Check what methods the inner TTT has
    print(f"4. TTT.ttt methods: {[m for m in dir(ttt_layer.ttt) if not m.startswith('_')]}")
    
    return ttt_layer, has_reset

def test_ttt_state_persistence():
    """Test what happens to TTT inner weights during processing."""
    print("\n🧠 Testing TTT State Persistence")
    print("=" * 50)
    
    ttt_layer, has_reset = test_model_reset_method()
    
    # Create test sequence
    batch_size = 2
    seq_len = 64
    config = ttt_layer.ttt.config
    
    hidden_states = torch.randn(batch_size, seq_len, config.model_dim)
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
    
    # Process sequence 1 (like sBLIMP)
    print("\n📊 Processing Sequence 1 (simulating sBLIMP)...")
    ttt_layer.train()  # Training mode
    result1 = ttt_layer.forward(hidden_states, seq_metadata)
    
    after_seq1_W1 = ttt_layer.ttt.W1.data.clone()
    after_seq1_b1 = ttt_layer.ttt.b1.data.clone()
    
    w1_change_1 = (after_seq1_W1 - initial_W1).norm().item()
    b1_change_1 = (after_seq1_b1 - initial_b1).norm().item()
    
    print(f"After Seq1 W1 norm: {after_seq1_W1.norm().item():.6f} (Δ={w1_change_1:.6f})")
    print(f"After Seq1 b1 norm: {after_seq1_b1.norm().item():.6f} (Δ={b1_change_1:.6f})")
    
    # Simulate evaluation mode switch (like paper_metrics.py)
    print("\n🔄 Switching to eval mode (simulating paper_metrics)...")
    ttt_layer.eval()
    
    # Try to reset states like paper_metrics.py does
    print("Attempting reset like paper_metrics.py...")
    if has_reset:
        print("   Calling ttt_layer.reset_ttt_states()...")
        ttt_layer.reset_ttt_states()
        after_reset_W1 = ttt_layer.ttt.W1.data.clone()
        after_reset_b1 = ttt_layer.ttt.b1.data.clone()
        
        reset_worked = not torch.equal(after_seq1_W1, after_reset_W1)
        print(f"   Reset worked: {reset_worked}")
        print(f"   After reset W1 norm: {after_reset_W1.norm().item():.6f}")
        print(f"   After reset b1 norm: {after_reset_b1.norm().item():.6f}")
    else:
        print("   ❌ hasattr(model, 'reset_ttt_states') = False")
        print("   ❌ No reset performed - states carry over!")
        after_reset_W1 = after_seq1_W1
        after_reset_b1 = after_seq1_b1
        reset_worked = False
    
    # Process sequence 2 (like LibriLight) 
    print("\n📚 Processing Sequence 2 (simulating LibriLight)...")
    result2 = ttt_layer.forward(hidden_states, seq_metadata)
    
    after_seq2_W1 = ttt_layer.ttt.W1.data.clone()
    after_seq2_b1 = ttt_layer.ttt.b1.data.clone()
    
    # Check what states were used as starting point for seq2
    w1_change_2 = (after_seq2_W1 - after_reset_W1).norm().item()
    b1_change_2 = (after_seq2_b1 - after_reset_b1).norm().item()
    
    print(f"After Seq2 W1 norm: {after_seq2_W1.norm().item():.6f} (Δ={w1_change_2:.6f})")
    print(f"After Seq2 b1 norm: {after_seq2_b1.norm().item():.6f} (Δ={b1_change_2:.6f})")
    
    return {
        'has_reset_method': has_reset,
        'reset_worked': reset_worked,
        'seq1_contaminated_seq2': not reset_worked,
        'w1_changes': [w1_change_1, w1_change_2],
        'b1_changes': [b1_change_1, b1_change_2],
    }

def create_fix():
    """Create a fix for the reset issue."""
    print("\n🔧 Creating Fix for TTT Reset Issue")
    print("=" * 50)
    
    # The issue: TTTWrapper doesn't expose reset_ttt_states method
    # The fix: Add the method to TTTWrapper
    
    fix_code = '''
# Fix: Add to TTTWrapper class in moshi_ttt/models/ssm/ttt_layer.py

def reset_ttt_states(self):
    """Reset TTT inner weights to initial values."""
    if hasattr(self.ttt, 'init_weights'):
        print(f"   Resetting TTT inner weights for {type(self.ttt).__name__}")
        old_W1_norm = self.ttt.W1.data.norm().item()
        old_b1_norm = self.ttt.b1.data.norm().item()
        
        # Reset to initial values
        self.ttt.init_weights()
        
        new_W1_norm = self.ttt.W1.data.norm().item()
        new_b1_norm = self.ttt.b1.data.norm().item()
        
        print(f"   W1 norm: {old_W1_norm:.6f} → {new_W1_norm:.6f}")
        print(f"   b1 norm: {old_b1_norm:.6f} → {new_b1_norm:.6f}")
        
        return True
    else:
        print(f"   Warning: {type(self.ttt).__name__} doesn't have init_weights method")
        return False
'''
    
    print("Fix needed:")
    print(fix_code)
    
    return fix_code

def main():
    """Run the complete investigation and fix."""
    print("🚨 TTT Reset Investigation & Fix")
    print("=" * 70)
    
    # Run investigation
    results = test_ttt_state_persistence()
    
    # Summary
    print("\n📋 Investigation Summary")
    print("=" * 50)
    print(f"Model has reset_ttt_states method: {results['has_reset_method']}")
    print(f"Reset actually works when called: {results['reset_worked']}")
    print(f"States contaminate across evaluations: {results['seq1_contaminated_seq2']}")
    print(f"W1 changes: Seq1={results['w1_changes'][0]:.6f}, Seq2={results['w1_changes'][1]:.6f}")
    print(f"b1 changes: Seq1={results['b1_changes'][0]:.6f}, Seq2={results['b1_changes'][1]:.6f}")
    
    # Create fix
    fix_code = create_fix()
    
    # Verdict
    print("\n🎯 Verdict")
    print("=" * 50)
    if not results['has_reset_method']:
        print("❌ ISSUE CONFIRMED: TTTWrapper missing reset_ttt_states method")
        print("❌ IMPACT: TTT states carry over between evaluation tasks")
        print("✅ FIX AVAILABLE: Add reset_ttt_states method to TTTWrapper")
    else:
        print("✅ Model has reset method")
        if not results['reset_worked']:
            print("❌ But reset doesn't work properly")
        else:
            print("✅ Reset works correctly")
    
    return results, fix_code

if __name__ == "__main__":
    results, fix = main()
    sys.exit(0 if results['has_reset_method'] else 1)