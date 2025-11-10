"""
Test to trace where XV and XK are generated and why they have million-scale values.

This test investigates the TTT layer integration to find the source of corruption.
"""

import torch
import sys
sys.path.insert(0, '/home/alufr/ttt_tests/moshi-finetune')


def test_ttt_layer_projections():
    """Test the Q/K/V projections that create XV and XK."""
    print("\n" + "="*80)
    print("TEST: TTT Layer Q/K/V Projections")
    print("="*80)
    
    from moshi_ttt.models.ssm.ttt_layer import TTTMLP
    from moshi_ttt.models.ssm.config import ModelConfig
    
    # Create a TTT layer with typical Moshi config
    config = ModelConfig(
        dim=4096,
        heads=32,
        head_dim=128,
        ttt_base_lr=1.0,
        mini_batch_size=32,
    )
    
    ttt_layer = TTTMLP(config, use_scan=True)
    ttt_layer.eval()
    
    print(f"TTT Layer created:")
    print(f"  dim: {config.dim}")
    print(f"  heads: {config.heads}")
    print(f"  head_dim: {config.head_dim}")
    
    # Create normal input (what should come from transformer)
    B, T, D = 1, 100, 4096
    hidden_states = torch.randn(B, T, D) * 0.1  # Normal scale
    
    print(f"\nInput hidden_states:")
    print(f"  Shape: {hidden_states.shape}")
    print(f"  Min: {hidden_states.min():.4f}")
    print(f"  Max: {hidden_states.max():.4f}")
    print(f"  Mean: {hidden_states.mean():.4f}")
    print(f"  Std: {hidden_states.std():.4f}")
    
    # Check projection weights
    print(f"\nProjection weight statistics:")
    print(f"  wq: min={ttt_layer.wq.weight.min():.4f}, max={ttt_layer.wq.weight.max():.4f}, std={ttt_layer.wq.weight.std():.4f}")
    print(f"  wk: min={ttt_layer.wk.weight.min():.4f}, max={ttt_layer.wk.weight.max():.4f}, std={ttt_layer.wk.weight.std():.4f}")
    print(f"  wv: min={ttt_layer.wv.weight.min():.4f}, max={ttt_layer.wv.weight.max():.4f}, std={ttt_layer.wv.weight.std():.4f}")
    
    # Apply projections
    with torch.no_grad():
        xq = ttt_layer.wq(hidden_states)
        xk = ttt_layer.wk(hidden_states)
        xv = ttt_layer.wv(hidden_states)
    
    print(f"\nAfter Q/K/V projections:")
    print(f"  xq: min={xq.min():.4f}, max={xq.max():.4f}, std={xq.std():.4f}")
    print(f"  xk: min={xk.min():.4f}, max={xk.max():.4f}, std={xk.std():.4f}")
    print(f"  xv: min={xv.min():.4f}, max={xv.max():.4f}, std={xv.std():.4f}")
    
    # Reshape to head dimension
    xq = xq.view(B, T, config.heads, config.head_dim)
    xk = xk.view(B, T, config.heads, config.head_dim)
    xv = xv.view(B, T, config.heads, config.head_dim)
    
    # Apply RoPE if exists
    if hasattr(ttt_layer, 'apply_rope'):
        print(f"\nApplying RoPE...")
        # Create dummy freqs_cis
        freqs_cis = torch.ones(T, config.head_dim // 2, 2)
        xq = ttt_layer.apply_rope(xq, freqs_cis)
        xk = ttt_layer.apply_rope(xk, freqs_cis)
    
    print(f"\nAfter reshape (and RoPE if applied):")
    print(f"  xq: min={xq.min():.4f}, max={xq.max():.4f}, std={xq.std():.4f}")
    print(f"  xk: min={xk.min():.4f}, max={xk.max():.4f}, std={xk.std():.4f}")
    print(f"  xv: min={xv.min():.4f}, max={xv.max():.4f}, std={xv.std():.4f}")
    
    # Check if values explode
    if xk.abs().max() > 1000:
        print(f"\n⚠️  WARNING: XK has exploded to {xk.abs().max():.0f}!")
        return False
    if xv.abs().max() > 1000:
        print(f"\n⚠️  WARNING: XV has exploded to {xv.abs().max():.0f}!")
        return False
    
    print("\n✅ PASS: Projections produce normal-scale outputs")
    return True


def test_ttt_with_large_inputs():
    """Test what happens when TTT layer receives large inputs (like from bug)."""
    print("\n" + "="*80)
    print("TEST: TTT Layer with Large-Scale Inputs")
    print("="*80)
    
    from moshi_ttt.models.ssm.ttt_layer import TTTMLP
    from moshi_ttt.models.ssm.config import ModelConfig
    
    config = ModelConfig(
        dim=4096,
        heads=32,
        head_dim=128,
        ttt_base_lr=1.0,
        mini_batch_size=32,
    )
    
    ttt_layer = TTTMLP(config, use_scan=True)
    ttt_layer.eval()
    
    # Create LARGE input (mimicking the bug)
    B, T, D = 1, 100, 4096
    hidden_states = torch.randn(B, T, D) * 1000000  # MILLION-scale!
    
    print(f"Input hidden_states (LARGE SCALE):")
    print(f"  Min: {hidden_states.min():.0f}")
    print(f"  Max: {hidden_states.max():.0f}")
    print(f"  Std: {hidden_states.std():.0f}")
    
    # Try forward pass
    try:
        with torch.no_grad():
            output = ttt_layer(hidden_states, freqs_cis=None, seq_metadata=None)
        
        print(f"\nOutput:")
        print(f"  Min: {output.min():.0f}")
        print(f"  Max: {output.max():.0f}")
        print(f"  Std: {output.std():.0f}")
        
        if output.abs().max() > 1e10:
            print(f"\n⚠️  WARNING: Output exploded to {output.abs().max():.2e}!")
        
    except Exception as e:
        print(f"\n❌ ERROR during forward pass: {e}")
        return False
    
    print("\n✅ PASS: Large inputs processed (though may produce large outputs)")
    return True


def test_check_depformer_integration():
    """Check how TTT is integrated into the depformer."""
    print("\n" + "="*80)
    print("TEST: Depformer TTT Integration")
    print("="*80)
    
    try:
        from moshi_ttt.hybrid_layer import HybridTransformerLayer
        
        print("Checking HybridTransformerLayer...")
        
        # Look at the source to understand integration
        import inspect
        source = inspect.getsource(HybridTransformerLayer.forward)
        
        # Check if there's normalization before TTT
        if 'norm' in source.lower() or 'layer_norm' in source.lower():
            print("✅ Found normalization in forward pass")
        else:
            print("⚠️  WARNING: No obvious normalization before TTT")
        
        # Check what's passed to TTT
        if 'self.self_attn' in source:
            print("✅ TTT is called via self.self_attn")
        
        print(f"\nForward pass snippet:")
        lines = source.split('\n')[:20]
        for line in lines:
            print(f"  {line}")
        
    except Exception as e:
        print(f"Could not inspect HybridTransformerLayer: {e}")
    
    print("\n✅ PASS: Integration check complete")


def run_projection_tests():
    """Run all projection-focused tests."""
    print("\n" + "#"*80)
    print("# TTT PROJECTION & INTEGRATION TESTS")
    print("#"*80)
    
    try:
        test_ttt_layer_projections()
    except Exception as e:
        print(f"\n❌ Projection test failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        test_ttt_with_large_inputs()
    except Exception as e:
        print(f"\n❌ Large input test failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        test_check_depformer_integration()
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "#"*80)
    print("# END OF PROJECTION TESTS")
    print("#"*80)


if __name__ == "__main__":
    run_projection_tests()
