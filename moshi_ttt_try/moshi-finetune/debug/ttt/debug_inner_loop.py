#!/usr/bin/env python3
"""
Debug inner loop loss collection to see what's happening.
"""

def check_inner_loop_status():
    """Check if inner loop losses were actually collected."""
    print("🔍 Debugging Inner Loop Loss Collection")
    print("=" * 50)
    
    try:
        from moshi_ttt.models.ssm.ops.ttt_mlp import (
            get_collected_inner_loop_losses, 
            _global_log_inner_loop_losses,
            _collected_inner_loop_losses
        )
        
        print(f"✅ Successfully imported inner loop functions")
        print(f"📊 Global logging enabled: {_global_log_inner_loop_losses}")
        print(f"📦 Collected losses: {len(_collected_inner_loop_losses) if _collected_inner_loop_losses else 0}")
        
        if _collected_inner_loop_losses and len(_collected_inner_loop_losses) > 0:
            print(f"🎯 Sample losses: {_collected_inner_loop_losses[:5]}")
            return True
        else:
            print("❌ No inner loop losses collected")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_ttt_mlp_logging():
    """Test if TTT MLP logging is working."""
    print("\n🧪 Testing TTT MLP Logging")
    print("=" * 50)
    
    try:
        import torch
        from moshi_ttt.models.ssm.ops.ttt_mlp import set_inner_loop_logging, ttt_mlp_with_states
        
        # Enable logging
        set_inner_loop_logging(True)
        print("✅ Enabled inner loop logging")
        
        # Create dummy data
        B, H, NC, C, HD = 1, 8, 4, 1, 128
        x = torch.randn(B, H, NC, C, HD, dtype=torch.bfloat16)
        W1 = torch.randn(B, H, HD, HD*4, dtype=torch.bfloat16)
        W2 = torch.randn(B, H, HD*4, HD, dtype=torch.bfloat16)
        states = torch.zeros(B, H, HD, HD, dtype=torch.bfloat16)
        
        print(f"📊 Test data shapes: x={x.shape}, W1={W1.shape}, W2={W2.shape}")
        
        # Run TTT MLP
        output, new_states = ttt_mlp_with_states(
            x, W1, W2, states,
            mini_batch_size=1,
            persistent_states=True,
            ttt_lr_weight=torch.tensor(0.01),
            ttt_lr_bias=torch.tensor(0.0),
            ttt_norm_weight=torch.ones(HD),
            ttt_norm_bias=torch.zeros(HD)
        )
        
        print(f"✅ TTT MLP completed: output={output.shape}")
        
        # Check if losses were logged
        return check_inner_loop_status()
        
    except Exception as e:
        print(f"❌ TTT MLP test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔍 INNER LOOP DEBUGGING")
    print("=" * 60)
    
    # Check current status
    has_losses = check_inner_loop_status()
    
    if not has_losses:
        print("\n🧪 Testing TTT MLP logging...")
        test_ttt_mlp_logging()
    
    print("\n" + "=" * 60)
    if has_losses:
        print("🎯 Inner loop losses ARE available!")
        print("   The issue might be in the plotting code.")
    else:
        print("❌ Inner loop losses NOT collected during evaluation.")
        print("   The TTT MLP might not be logging properly.")