#!/usr/bin/env python3
"""
Focused diagnostic: Why does TTT increase NaN values in Moshi?
"""

import torch
import sys
import os

sys.path.append('/home/alufr/ttt_tests/moshi')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def check_intermediate_values(hybrid_layer, x, name):
    """Hook to check intermediate values in hybrid layer"""
    print(f"\n🔬 Checking intermediate values for: {name}")
    
    # Get the original layer's attention output
    print("   1. Original attention processing...")
    original_output = hybrid_layer.original_layer(x)
    
    finite_ratio = torch.isfinite(original_output).float().mean().item()
    print(f"      Original attention output: {finite_ratio:.1%} finite")
    
    # Check TTT processing step by step
    print("   2. TTT processing...")
    seq_modeling_block = hybrid_layer.seq_modeling_block
    
    # Step 2a: Attention forward
    attn_output = seq_modeling_block._attn_forward(x)
    finite_ratio = torch.isfinite(attn_output).float().mean().item()
    print(f"      After attention: {finite_ratio:.1%} finite")
    
    # Step 2b: TTT forward (this is where issues likely occur)
    try:
        ttt_output = seq_modeling_block._ttt_forward(attn_output)
        finite_ratio = torch.isfinite(ttt_output).float().mean().item()
        print(f"      After TTT: {finite_ratio:.1%} finite")
        
        # Check if TTT is the culprit
        if torch.isfinite(attn_output).all() and not torch.isfinite(ttt_output).all():
            print("      ⚠️  TTT processing introduced non-finite values!")
            
            # Deeper TTT analysis
            print("      🔍 TTT processing details:")
            try:
                # Check QKV projections
                XQ, XK, XV = seq_modeling_block.get_qkv_projections(attn_output)
                print(f"         Q finite: {torch.isfinite(XQ).float().mean().item():.1%}")
                print(f"         K finite: {torch.isfinite(XK).float().mean().item():.1%}")
                print(f"         V finite: {torch.isfinite(XV).float().mean().item():.1%}")
                
                # Check after L2 normalization
                XQ_norm = torch.nn.functional.normalize(XQ, p=2, dim=-1)
                XK_norm = torch.nn.functional.normalize(XK, p=2, dim=-1)
                print(f"         Q after L2 norm: {torch.isfinite(XQ_norm).float().mean().item():.1%}")
                print(f"         K after L2 norm: {torch.isfinite(XK_norm).float().mean().item():.1%}")
                
            except Exception as e:
                print(f"         ❌ Error in TTT details: {e}")
        
    except Exception as e:
        print(f"      ❌ TTT forward failed: {e}")
        return None
    
    return ttt_output

def main():
    print("🔍 TTT NaN INCREASE DIAGNOSTIC")
    print("Why does TTT processing double the NaN rate?")
    
    try:
        from moshi.modules.transformer import StreamingTransformerLayer
        from moshi_ttt.hybrid_layer import HybridStreamingTransformerLayer
        from moshi_ttt.config import TTTConfig
        
        # Create test layer with correct parameters
        d_model = 512
        num_heads = 8
        
        original_layer = StreamingTransformerLayer(
            d_model=d_model,
            num_heads=num_heads,
            dim_feedforward=int(d_model * 4),
            causal=True,
            context=100,
            layer_scale=None,
            gating='silu',
            norm='rms_norm'
        )
        
        ttt_config = TTTConfig(
            model_dim=d_model,
            num_heads=num_heads,
            mini_batch_size=16,
            ttt_base_lr=0.1
        )
        
        hybrid_layer = HybridStreamingTransformerLayer(original_layer, ttt_config)
        
        print(f"✅ Created test layers: d_model={d_model}, num_heads={num_heads}")
        
        # Test scenarios that might cause NaN
        scenarios = [
            ("normal", torch.randn(2, 32, d_model) * 0.02),
            ("large_values", torch.randn(2, 32, d_model) * 10.0),
            ("very_large", torch.randn(2, 32, d_model) * 100.0),
            ("zeros", torch.zeros(2, 32, d_model)),
            ("small_nonzero", torch.randn(2, 32, d_model) * 1e-6)
        ]
        
        for name, x in scenarios:
            print(f"\n{'='*50}")
            print(f"🧪 SCENARIO: {name}")
            print(f"Input stats: min={x.min():.3e}, max={x.max():.3e}, mean={x.mean():.3e}")
            print(f"{'='*50}")
            
            try:
                with torch.no_grad():
                    result = check_intermediate_values(hybrid_layer, x, name)
                    
                if result is not None:
                    finite_ratio = torch.isfinite(result).float().mean().item()
                    print(f"✅ Final result: {finite_ratio:.1%} finite")
                else:
                    print("❌ Failed to get result")
                    
            except Exception as e:
                print(f"❌ Scenario {name} failed: {e}")
                import traceback
                traceback.print_exc()
        
        # Summary analysis
        print(f"\n{'='*60}")
        print("📊 SUMMARY & ANALYSIS")
        print(f"{'='*60}")
        print("Based on the diagnostic above:")
        print("1. If attention output is finite but TTT output is not → TTT issue")
        print("2. If L2 normalization causes issues → Division by zero in norm")
        print("3. If QKV projections cause issues → Weight initialization problem")
        print("4. If large input values cause issues → TTT processing instability")
        
    except Exception as e:
        print(f"❌ Main diagnostic failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()