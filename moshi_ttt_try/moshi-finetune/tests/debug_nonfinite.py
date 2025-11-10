#!/usr/bin/env python3
"""
Debug non-finite values in Moshi forward pass
Investigate when and why NaN/Inf values occur in different scenarios
"""

import torch
import sys
import os

# Disable torch dynamo compilation to get cleaner errors  
import torch._dynamo
torch._dynamo.config.suppress_errors = True

# Add paths
sys.path.append('/home/alufr/ttt_tests/moshi')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def analyze_tensor_health(tensor, name):
    """Analyze tensor for finite/infinite/nan values"""
    if tensor is None:
        print(f"   {name}: None")
        return
    
    total = tensor.numel()
    finite = torch.isfinite(tensor).sum().item()
    inf_count = torch.isinf(tensor).sum().item()
    nan_count = torch.isnan(tensor).sum().item()
    
    print(f"   {name}: shape={tensor.shape}, finite={finite}/{total} ({finite/total:.1%})")
    if inf_count > 0:
        print(f"      ∞ Inf values: {inf_count} ({inf_count/total:.1%})")
    if nan_count > 0:
        print(f"      🚫 NaN values: {nan_count} ({nan_count/total:.1%})")
    
    if finite > 0:
        finite_tensor = tensor[torch.isfinite(tensor)]
        print(f"      📊 Finite stats: min={finite_tensor.min():.3f}, max={finite_tensor.max():.3f}, mean={finite_tensor.mean():.3f}")

def test_case_1_vanilla_moshi():
    """Test Case 1: Vanilla Moshi without any modifications"""
    print("\n" + "="*60)
    print("🧪 TEST CASE 1: Vanilla Moshi (no modifications)")
    print("="*60)
    
    try:
        from moshi.models import loaders
        
        # Load model
        checkpoint_info = loaders.CheckpointInfo.from_hf_repo("kyutai/moshiko-pytorch-bf16")
        lm_config = loaders._lm_kwargs if checkpoint_info.raw_config is None else checkpoint_info.raw_config
        lm_model = loaders.get_moshi_lm(filename=None, lm_kwargs=lm_config, device='cpu', dtype=torch.float32)
        
        print(f"✅ Loaded vanilla Moshi: {lm_config['dim']}d, {lm_config['num_layers']} layers")
        
        # Test different input scenarios
        batch_size = 2
        seq_len = 10
        n_q = lm_config.get('n_q', 8)
        
        # Scenario 1a: Random codes
        print("\n🔬 Scenario 1a: Random codes")
        codes = torch.randint(0, 1024, (batch_size, n_q + 1, seq_len), dtype=torch.int64)
        print(f"   Input codes: shape={codes.shape}, min={codes.min()}, max={codes.max()}")
        
        lm_model.eval()
        with torch.no_grad():
            output = lm_model(codes)
        
        analyze_tensor_health(output.logits, "Output logits")
        if hasattr(output, 'text_logits') and output.text_logits is not None:
            analyze_tensor_health(output.text_logits, "Text logits")
        
        # Scenario 1b: All zeros
        print("\n🔬 Scenario 1b: All zero codes")
        codes_zeros = torch.zeros((batch_size, n_q + 1, seq_len), dtype=torch.int64)
        
        with torch.no_grad():
            output_zeros = lm_model(codes_zeros)
        
        analyze_tensor_health(output_zeros.logits, "Output logits (zeros)")
        if hasattr(output_zeros, 'text_logits') and output_zeros.text_logits is not None:
            analyze_tensor_health(output_zeros.text_logits, "Text logits (zeros)")
        
        return True
        
    except Exception as e:
        print(f"❌ Test Case 1 failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_case_2_hybrid_layer_isolated():
    """Test Case 2: Single hybrid layer in isolation"""
    print("\n" + "="*60)
    print("🧪 TEST CASE 2: Hybrid layer in isolation")
    print("="*60)
    
    try:
        from moshi.modules.transformer import StreamingTransformerLayer
        from moshi_ttt.hybrid_layer import HybridStreamingTransformerLayer
        from moshi_ttt.config import TTTConfig
        
        # Create original layer
        d_model = 512
        num_heads = 8
        
        # Use reasonable Moshi-like config
        original_layer = StreamingTransformerLayer(
            dim=d_model,
            num_heads=num_heads,
            dim_feedforward=int(d_model * 4.125),  # Moshi's hidden_scale
            causal=True,
            layer_scale=None,
            gating='silu',
            norm='rms_norm_f32',
            context=100,
            max_period=10000,
            rope=True
        )
        
        # Create TTT config
        ttt_config = TTTConfig(
            model_dim=d_model,
            num_heads=num_heads,
            mini_batch_size=16,
            ttt_base_lr=0.1
        )
        
        # Create hybrid layer
        hybrid_layer = HybridStreamingTransformerLayer(original_layer, ttt_config)
        
        print(f"✅ Created isolated hybrid layer: d_model={d_model}, num_heads={num_heads}")
        
        # Test different input magnitudes
        batch_size = 2
        seq_len = 32
        
        scenarios = [
            ("normal", torch.randn(batch_size, seq_len, d_model) * 0.1),
            ("small", torch.randn(batch_size, seq_len, d_model) * 0.01),
            ("large", torch.randn(batch_size, seq_len, d_model) * 1.0),
            ("very_large", torch.randn(batch_size, seq_len, d_model) * 10.0),
            ("zeros", torch.zeros(batch_size, seq_len, d_model))
        ]
        
        for name, x in scenarios:
            print(f"\n🔬 Scenario 2{name[0]}: {name} input")
            print(f"   Input stats: min={x.min():.3f}, max={x.max():.3f}, mean={x.mean():.3f}, std={x.std():.3f}")
            
            try:
                with torch.no_grad():
                    output = hybrid_layer(x)
                analyze_tensor_health(output, f"Hybrid output ({name})")
            except Exception as e:
                print(f"   ❌ Failed on {name} input: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test Case 2 failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_case_3_moshi_with_hybrid():
    """Test Case 3: Moshi model with single hybrid layer replaced"""
    print("\n" + "="*60)
    print("🧪 TEST CASE 3: Moshi with hybrid layer replacement")
    print("="*60)
    
    try:
        from moshi.models import loaders
        from moshi_ttt.hybrid_layer import HybridStreamingTransformerLayer
        from moshi_ttt.config import TTTConfig
        
        # Load smaller model configuration to avoid memory issues
        checkpoint_info = loaders.CheckpointInfo.from_hf_repo("kyutai/moshiko-pytorch-bf16")
        lm_config = loaders._lm_kwargs if checkpoint_info.raw_config is None else checkpoint_info.raw_config
        
        # Create smaller test model by modifying config
        small_config = lm_config.copy()
        small_config['num_layers'] = 4  # Reduce layers
        small_config['dim'] = 512  # Reduce dimension
        small_config['num_heads'] = 8  # Reduce heads
        
        print(f"✅ Creating smaller test model: {small_config['dim']}d, {small_config['num_layers']} layers")
        
        # Build smaller model
        lm_model = loaders.get_moshi_lm(
            filename=None,
            lm_kwargs=small_config,
            device='cpu',
            dtype=torch.float32
        )
        
        # Replace middle layer
        target_layer_idx = 2
        original_layer = lm_model.transformer.layers[target_layer_idx]
        
        ttt_config = TTTConfig(
            model_dim=small_config['dim'],
            num_heads=small_config['num_heads'],
            mini_batch_size=16,
            ttt_base_lr=0.1
        )
        
        hybrid_layer = HybridStreamingTransformerLayer(original_layer, ttt_config)
        lm_model.transformer.layers[target_layer_idx] = hybrid_layer
        
        print(f"✅ Replaced layer {target_layer_idx} with hybrid layer")
        
        # Test with different input patterns
        batch_size = 1  # Smaller batch
        seq_len = 5     # Shorter sequence
        n_q = small_config.get('n_q', 8)
        
        scenarios = [
            ("zeros", torch.zeros((batch_size, n_q + 1, seq_len), dtype=torch.int64)),
            ("small_random", torch.randint(0, 10, (batch_size, n_q + 1, seq_len), dtype=torch.int64)),
            ("medium_random", torch.randint(0, 100, (batch_size, n_q + 1, seq_len), dtype=torch.int64)),
            ("full_random", torch.randint(0, 1024, (batch_size, n_q + 1, seq_len), dtype=torch.int64))
        ]
        
        for name, codes in scenarios:
            print(f"\n🔬 Scenario 3{name[0]}: {name}")
            print(f"   Input codes: shape={codes.shape}, min={codes.min()}, max={codes.max()}")
            
            try:
                lm_model.eval()
                with torch.no_grad():
                    output = lm_model(codes)
                
                analyze_tensor_health(output.logits, f"Output logits ({name})")
                if hasattr(output, 'text_logits') and output.text_logits is not None:
                    analyze_tensor_health(output.text_logits, f"Text logits ({name})")
                    
            except Exception as e:
                print(f"   ❌ Failed on {name}: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test Case 3 failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🔍 NON-FINITE VALUES DIAGNOSTIC")
    print("Investigating when and why NaN/Inf occur in Moshi + TTT")
    
    results = {
        'vanilla_moshi': test_case_1_vanilla_moshi(),
        'hybrid_isolated': test_case_2_hybrid_layer_isolated(), 
        'moshi_with_hybrid': test_case_3_moshi_with_hybrid()
    }
    
    print("\n" + "="*60)
    print("📊 DIAGNOSTIC SUMMARY")
    print("="*60)
    
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print("\n💡 ANALYSIS:")
    if results['vanilla_moshi'] and not results['hybrid_isolated']:
        print("   • Issue is in hybrid layer implementation")
    elif not results['vanilla_moshi'] and results['hybrid_isolated']:
        print("   • Issue is in vanilla Moshi model")
    elif not results['vanilla_moshi'] and not results['hybrid_isolated']:
        print("   • Issues in both vanilla Moshi and hybrid layer")
    else:
        print("   • Integration issue when combining Moshi + hybrid layer")

if __name__ == "__main__":
    main()