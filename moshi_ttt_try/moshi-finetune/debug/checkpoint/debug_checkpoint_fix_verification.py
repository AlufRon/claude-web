#!/usr/bin/env python3
"""
SIMPLE TEST: Verify if the checkpoint fix was actually applied to the source code
"""

import torch
import sys
import os

sys.path.append('/home/alufr/ttt_tests/moshi')
sys.path.insert(0, '.')

def check_checkpoint_fix():
    """Check if checkpoint_group_size = 0 is actually in the source code"""
    print("🔍 VERIFYING CHECKPOINT FIX")
    print("=" * 60)
    
    # Read the hybrid_layer.py file
    hybrid_layer_path = '/home/alufr/ttt_tests/moshi-finetune/moshi_ttt/hybrid_layer.py'
    
    print(f"Reading: {hybrid_layer_path}")
    
    with open(hybrid_layer_path, 'r') as f:
        content = f.read()
    
    # Check for the fix
    if "checkpoint_group_size = 0" in content:
        print("✅ Checkpoint fix is present in source code")
        
        # Find the exact line
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            if "checkpoint_group_size = 0" in line:
                print(f"   Line {i}: {line.strip()}")
                
                # Show context
                print("   Context:")
                for j in range(max(0, i-3), min(len(lines), i+3)):
                    marker = ">>> " if j == i-1 else "    "
                    print(f"   {marker}Line {j+1}: {lines[j].strip()}")
                
    else:
        print("❌ Checkpoint fix NOT found in source code")
        
        # Look for the old line
        if "checkpoint_group_size = min(max(1, NC), NC)" in content:
            print("   Found old checkpoint logic - fix was not applied!")
        else:
            print("   Neither old nor new checkpoint logic found")
    
    return "checkpoint_group_size = 0" in content

def test_runtime_checkpoint():
    """Test if the checkpoint fix works at runtime"""
    print(f"\n🔍 TESTING RUNTIME CHECKPOINT BEHAVIOR")
    print("-" * 50)
    
    try:
        from moshi.modules.transformer import StreamingTransformerLayer
        from moshi_ttt.hybrid_layer import HybridStreamingTransformerLayer
        from moshi_ttt.config import TTTConfig
        
        # Small test setup
        d_model = 256
        num_heads = 4
        
        original_layer = StreamingTransformerLayer(
            d_model=d_model,
            num_heads=num_heads,
            dim_feedforward=d_model * 4,
            causal=True,
            context=50,
            norm='rms_norm'
        )
        
        ttt_config = TTTConfig(
            model_dim=d_model,
            num_heads=num_heads,
            mini_batch_size=8,
            ttt_base_lr=0.1
        )
        
        hybrid_layer = HybridStreamingTransformerLayer(original_layer, ttt_config)
        hybrid_layer.train()
        
        print("✅ Created hybrid layer")
        
        # Check what checkpoint_group_size is actually used
        # We'll monkey-patch to capture the value
        original_ttt_mlp = None
        captured_checkpoint_size = None
        
        def capture_checkpoint_size(*args, **kwargs):
            nonlocal captured_checkpoint_size
            if len(args) >= 11:  # checkpoint_group_size is the 11th argument
                captured_checkpoint_size = args[10]
            print(f"   🎯 CAPTURED checkpoint_group_size: {captured_checkpoint_size}")
            # Don't actually run ttt_mlp to avoid the device error
            # Just return a dummy tensor
            B, H, NC, C, HD = args[0].shape  # XK shape
            return torch.zeros(H, NC, C, B, HD)
        
        # Patch ttt_mlp temporarily
        from moshi_ttt.models.ssm.ops import ttt_mlp
        original_ttt_mlp_func = ttt_mlp.ttt_mlp
        ttt_mlp.ttt_mlp = capture_checkpoint_size
        
        try:
            # Create small input
            x = torch.randn(1, 16, d_model) * 0.1
            
            print("   🔄 Running forward pass to capture checkpoint_group_size...")
            
            # This should call our patched ttt_mlp and capture the checkpoint size
            output = hybrid_layer(x)
            
            print(f"   ✅ Forward pass completed")
            
            if captured_checkpoint_size is not None:
                if captured_checkpoint_size == 0:
                    print(f"   ✅ Fix confirmed: checkpoint_group_size = {captured_checkpoint_size}")
                    return True
                else:
                    print(f"   ❌ Fix not working: checkpoint_group_size = {captured_checkpoint_size}")
                    return False
            else:
                print(f"   ❌ Could not capture checkpoint_group_size")
                return False
                
        finally:
            # Restore original function
            ttt_mlp.ttt_mlp = original_ttt_mlp_func
            
    except Exception as e:
        print(f"   ❌ Runtime test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🧪 CHECKPOINT FIX VERIFICATION")
    print("=" * 60)
    
    # Check 1: Source code verification
    source_fix_present = check_checkpoint_fix()
    
    # Check 2: Runtime verification  
    runtime_fix_working = test_runtime_checkpoint()
    
    print(f"\n🎯 VERIFICATION RESULTS:")
    print("-" * 40)
    print(f"Source code fix present: {source_fix_present}")
    print(f"Runtime fix working: {runtime_fix_working}")
    
    if source_fix_present and runtime_fix_working:
        print("✅ Checkpoint fix is correctly applied and working")
        print("   → Problem must be elsewhere!")
    elif source_fix_present and not runtime_fix_working:
        print("❌ Fix is in source but not working at runtime")
        print("   → Module reload or import issue")
    elif not source_fix_present:
        print("❌ Fix is not in source code")
        print("   → Edit was not saved or applied correctly")
    else:
        print("🤔 Unexpected state")
    
    return source_fix_present and runtime_fix_working

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎯 NEXT: If fix is working, the problem is elsewhere in the training pipeline")
    else:
        print("\n🚨 NEXT: Fix the checkpoint issue first")