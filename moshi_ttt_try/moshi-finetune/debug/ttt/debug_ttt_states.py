#!/usr/bin/env python3
"""
Debug script to verify TTT state persistence before/during/after evaluation
"""

import torch
from pathlib import Path
import sys
import os

# Change to the correct directory and add to path
os.chdir('/home/alufr/ttt_tests/moshi-finetune')
sys.path.insert(0, '/home/alufr/ttt_tests/moshi-finetune')

try:
    from finetune.train import load_model_and_config
except ImportError:
    print("❌ Could not import load_model_and_config")
    print("Let me try to load a simpler way...")
    
    def load_model_and_config(checkpoint_path):
        """Simple checkpoint loader"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"Checkpoint keys: {list(checkpoint.keys())}")
        return None, None, None

def inspect_ttt_states(model, prefix=""):
    """Inspect all TTT states in the model"""
    print(f"\n🔍 {prefix} TTT State Inspection:")
    ttt_found = False
    
    for name, module in model.named_modules():
        if 'ttt' in name.lower() and hasattr(module, 'W1'):
            ttt_found = True
            w1_norm = torch.norm(module.W1).item()
            b1_norm = torch.norm(module.b1).item() if hasattr(module, 'b1') else 0.0
            w2_norm = torch.norm(module.W2).item() if hasattr(module, 'W2') else 0.0
            b2_norm = torch.norm(module.b2).item() if hasattr(module, 'b2') else 0.0
            
            print(f"  {name}:")
            print(f"    W1 norm: {w1_norm:.6f}")
            print(f"    b1 norm: {b1_norm:.6f}")
            print(f"    W2 norm: {w2_norm:.6f}")
            print(f"    b2 norm: {b2_norm:.6f}")
            
            # Check if states look random (near zero suggests not trained)
            if w1_norm < 0.001:
                print(f"    ⚠️  W1 norm very small - may be uninitialized!")
            elif w1_norm > 10.0:
                print(f"    🔥 W1 norm large - looks trained!")
            else:
                print(f"    ✅ W1 norm reasonable")
    
    if not ttt_found:
        print("  ❌ No TTT modules found!")
    
    return ttt_found

def test_streaming_context_effect(model):
    """Test if streaming context affects TTT states"""
    print("\n🧪 Testing streaming context effect on TTT states...")
    
    # Inspect states before streaming
    inspect_ttt_states(model, "BEFORE streaming context")
    
    # Enter streaming context
    print("\n📡 Entering model.streaming(batch_size=1)...")
    with model.streaming(batch_size=1):
        inspect_ttt_states(model, "INSIDE streaming context")
    
    # After streaming context
    inspect_ttt_states(model, "AFTER streaming context")

def main():
    print("🚨 TTT State Persistence Debug")
    print("=" * 50)
    
    # Load the latest TTT model checkpoint
    model_dir = Path("/home/alufr/ttt_tests/moshi-finetune")
    
    try:
        # Find the latest checkpoint
        checkpoint_dirs = list(model_dir.glob("runs/*/checkpoints"))
        if not checkpoint_dirs:
            print("❌ No checkpoint directories found")
            return
        
        latest_checkpoint_dir = max(checkpoint_dirs, key=lambda p: p.stat().st_mtime)
        checkpoint_files = list(latest_checkpoint_dir.glob("checkpoint_*.pt"))
        
        if not checkpoint_files:
            print("❌ No checkpoint files found")
            return
        
        latest_checkpoint = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
        print(f"📁 Loading checkpoint: {latest_checkpoint}")
        
        # Load model
        model, config, _ = load_model_and_config(str(latest_checkpoint))
        print(f"✅ Model loaded successfully")
        print(f"   Model type: {type(model).__name__}")
        print(f"   Has TTT config: {hasattr(model, 'ttt_config')}")
        
        # Test 1: Inspect initial states
        inspect_ttt_states(model, "INITIAL (after checkpoint load)")
        
        # Test 2: Test streaming context effect
        test_streaming_context_effect(model)
        
        # Test 3: Verify persistent_states setting
        print("\n⚙️ Checking TTT configuration...")
        for name, module in model.named_modules():
            if hasattr(module, 'persistent_states'):
                print(f"  {name}: persistent_states = {module.persistent_states}")
            elif 'ttt' in name.lower():
                print(f"  {name}: no persistent_states attribute")
        
        print("\n🎯 Debug Summary:")
        print("1. If W1/W2 norms are very small (< 0.001), TTT wasn't trained")
        print("2. If W1/W2 norms are large (> 1.0), TTT was trained but may not persist")
        print("3. If streaming context changes norms, that's the bug!")
        
    except Exception as e:
        print(f"❌ Error during debug: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()