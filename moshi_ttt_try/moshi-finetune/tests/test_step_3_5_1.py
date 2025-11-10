#!/usr/bin/env python3
"""
Step 3.5.1: Single Layer Replacement Test
Replace one Moshi layer with hybrid layer in actual model and test forward pass
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

def test_single_layer_replacement():
    """Test replacing single layer in actual Moshi model"""
    print("🧪 Step 3.5.1: Single Layer Replacement Test...")
    
    try:
        # Import Moshi components
        from moshi.models import loaders
        from moshi_ttt.hybrid_layer import HybridStreamingTransformerLayer
        from moshi_ttt.config import TTTConfig
        
        print("✅ Imports successful")
        
        # Load Moshi model
        print("📥 Loading Moshi model...")
        checkpoint_info = loaders.CheckpointInfo.from_hf_repo("kyutai/moshiko-pytorch-bf16")
        
        # Get model configuration
        lm_config = (
            loaders._lm_kwargs 
            if checkpoint_info.raw_config is None 
            else checkpoint_info.raw_config
        )
        
        print(f"✅ Model config loaded: {lm_config['dim']}d model with {lm_config['num_layers']} layers")
        
        # Build the model 
        print("🏗️  Building Moshi LM model...")
        lm_model = loaders.get_moshi_lm(
            filename=None,  # Will use default/cached model
            lm_kwargs=lm_config,
            device='cpu', 
            dtype=torch.float32
        )
        
        print(f"✅ Moshi model loaded: {type(lm_model)}")
        print(f"   📊 Model device: {next(lm_model.parameters()).device}")
        print(f"   📏 Model dtype: {next(lm_model.parameters()).dtype}")
        # Check model structure - layers are in transformer.layers
        if hasattr(lm_model, 'transformer') and hasattr(lm_model.transformer, 'layers'):
            layer_list = lm_model.transformer.layers
            total_layers = len(layer_list)
            print(f"   🏗️  Number of layers: {total_layers}")
            target_layer_idx = min(5, total_layers - 1)  # Replace layer 5 or last layer if less than 5
            print(f"🎯 Target layer for replacement: layer {target_layer_idx}")
            
            # Get original layer
            original_layer = layer_list[target_layer_idx]
            print(f"✅ Original layer type: {type(original_layer)}")
            
            # Create TTT config matching the model
            d_model = lm_config['dim']
            num_heads = lm_config.get('num_heads', 8)  # Default to 8 if not specified
            
            ttt_config = TTTConfig(
                model_dim=d_model,
                num_heads=num_heads,
                mini_batch_size=16,
                ttt_base_lr=0.1
            )
            
            print(f"✅ TTT config created: d_model={d_model}, num_heads={num_heads}")
            
            # Create hybrid layer
            print("🔄 Creating hybrid layer...")
            hybrid_layer = HybridStreamingTransformerLayer(original_layer, ttt_config)
            
            # Replace the layer
            print(f"🔁 Replacing layer {target_layer_idx} with hybrid layer...")
            layer_list[target_layer_idx] = hybrid_layer
            
            print("✅ Layer replacement complete!")
            
            # Test forward pass with replaced layer
            print("🚀 Testing forward pass with hybrid layer...")
            
            # Create dummy input (similar to test_moshi_forward.py)
            batch_size = 2
            seq_len = 10
            n_q = lm_config.get('n_q', 8)  # Number of quantization levels
            
            # Create dummy codes tensor
            codes = torch.randint(0, 1024, (batch_size, n_q + 1, seq_len), dtype=torch.int64)
            print(f"✅ Created dummy codes: shape {codes.shape}, dtype {codes.dtype}")
            
            # Forward pass through modified model
            lm_model.eval()
            with torch.no_grad():
                output = lm_model(codes)
                
            print(f"✅ Forward pass successful!")
            print(f"   📊 Output type: {type(output)}")
            print(f"   📏 Logits shape: {output.logits.shape}")
            if hasattr(output, 'text_logits') and output.text_logits is not None:
                print(f"   📝 Text logits shape: {output.text_logits.shape}")
            
            # Verify output is reasonable
            assert torch.isfinite(output.logits).all(), "Output logits contain non-finite values"
            
            print("✅ Output validation passed!")
            
            return True
            
        else:
            print("❌ Model doesn't have transformer.layers - different architecture than expected")
            return False
            
    except Exception as e:
        print(f"❌ Single layer replacement failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_layer_comparison():
    """Compare outputs between original and hybrid layers"""
    print("\n🔍 Step 3.5.1b: Layer Output Comparison...")
    
    try:
        from moshi.modules.transformer import StreamingTransformerLayer
        from moshi_ttt.hybrid_layer import HybridStreamingTransformerLayer  
        from moshi_ttt.config import TTTConfig
        
        # Create matched original and hybrid layers
        d_model = 512
        num_heads = 8
        
        original_layer = StreamingTransformerLayer(
            d_model=d_model,
            num_heads=num_heads,
            dim_feedforward=d_model * 4,
            causal=True,
        )
        
        ttt_config = TTTConfig(
            model_dim=d_model,
            num_heads=num_heads,
            mini_batch_size=16
        )
        
        hybrid_layer = HybridStreamingTransformerLayer(original_layer, ttt_config)
        
        # Test with same input
        batch_size = 2
        seq_len = 32
        x = torch.randn(batch_size, seq_len, d_model)
        
        print(f"✅ Created test layers and input: {x.shape}")
        
        # Forward through both layers
        original_layer.eval()
        hybrid_layer.eval()
        
        with torch.no_grad():
            original_output = original_layer(x)
            hybrid_output = hybrid_layer(x)
            
        print(f"✅ Both forward passes completed")
        print(f"   Original output: {original_output.shape}")
        print(f"   Hybrid output: {hybrid_output.shape}")
        
        # Compare outputs (they should be different due to TTT processing)
        output_diff = torch.norm(hybrid_output - original_output).item()
        print(f"   Output difference (L2 norm): {output_diff:.6f}")
        
        # Verify both are finite and reasonable
        assert torch.isfinite(original_output).all(), "Original output has non-finite values"
        assert torch.isfinite(hybrid_output).all(), "Hybrid output has non-finite values"
        
        # Outputs should be different (TTT adds processing)
        assert output_diff > 1e-6, f"Outputs too similar ({output_diff:.6f}) - TTT may not be active"
        
        print("✅ Layer comparison passed!")
        print("   ✅ Original layer works correctly")  
        print("   ✅ Hybrid layer works correctly")
        print("   ✅ Outputs appropriately different (TTT is active)")
        
        return True
        
    except Exception as e:
        print(f"❌ Layer comparison failed: {e}")
        import traceback  
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Step 3.5.1: Single Layer Replacement Testing")
    print("=" * 60)
    
    test1_passed = test_single_layer_replacement()
    test2_passed = test_layer_comparison()
    
    print("=" * 60)
    print("📊 Step 3.5.1 Results:")
    print(f"   Actual Model Integration: {'✅ PASS' if test1_passed else '❌ FAIL'}")
    print(f"   Layer Output Comparison: {'✅ PASS' if test2_passed else '❌ FAIL'}")
    
    overall_success = test1_passed and test2_passed
    print(f"\n🏆 Step 3.5.1 OVERALL: {'✅ SUCCESS' if overall_success else '❌ FAILED'}")
    
    if overall_success:
        print("✨ Ready for Step 3.5.2: Multiple Layer Replacement!")
    else:
        print("🔧 Need to debug single layer issues before proceeding")
        
    sys.exit(0 if overall_success else 1)
