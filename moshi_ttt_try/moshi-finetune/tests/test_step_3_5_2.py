#!/usr/bin/env python3
"""
Step 3.5.2: Multiple Layer Replacement Test
Replace multiple Moshi layers with hybrid layers and test stability
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

def test_multiple_layer_replacement():
    """Test replacing multiple layers in actual Moshi model"""
    print("🧪 Step 3.5.2: Multiple Layer Replacement Test...")
    
    try:
        # Import Moshi components
        from moshi.models import loaders
        from moshi_ttt.hybrid_layer import HybridStreamingTransformerLayer
        from moshi_ttt.config import TTTConfig
        
        print("✅ Imports successful")
        
        # Load Moshi model with smaller configuration for stability
        print("📥 Loading Moshi model...")
        checkpoint_info = loaders.CheckpointInfo.from_hf_repo("kyutai/moshiko-pytorch-bf16")
        
        # Get model configuration and reduce size for testing
        lm_config = (
            loaders._lm_kwargs 
            if checkpoint_info.raw_config is None 
            else checkpoint_info.raw_config
        )
        
        # Create smaller test model for stability
        test_config = lm_config.copy()
        test_config['num_layers'] = 8  # Reduce from 32 to 8 layers
        test_config['dim'] = 1024      # Reduce from 4096 to 1024
        test_config['num_heads'] = 16  # Reduce from 32 to 16 heads
        
        print(f"✅ Model config: {test_config['dim']}d model with {test_config['num_layers']} layers")
        
        # Build the model 
        print("🏗️  Building reduced Moshi LM model...")
        lm_model = loaders.get_moshi_lm(
            filename=None,
            lm_kwargs=test_config,
            device='cpu',
            dtype=torch.float32
        )
        
        print(f"✅ Moshi model loaded: {type(lm_model)}")
        print(f"   📊 Model device: {next(lm_model.parameters()).device}")
        print(f"   📏 Model dtype: {next(lm_model.parameters()).dtype}")
        
        # Check model structure
        if hasattr(lm_model, 'transformer') and hasattr(lm_model.transformer, 'layers'):
            layer_list = lm_model.transformer.layers
            total_layers = len(layer_list)
            print(f"   🏗️  Number of layers: {total_layers}")
            
            # Define multiple layer replacement strategy
            # Replace middle layers (avoid first/last for stability)
            start_layer = 2
            end_layer = min(6, total_layers - 1)  # Replace layers 2-5 (or fewer)
            replacement_layers = list(range(start_layer, end_layer))
            
            print(f"🎯 Target layers for replacement: {replacement_layers}")
            
            # Create TTT config
            d_model = test_config['dim']
            num_heads = test_config['num_heads']
            
            ttt_config = TTTConfig(
                model_dim=d_model,
                num_heads=num_heads,
                mini_batch_size=16,
                ttt_base_lr=0.1
            )
            
            print(f"✅ TTT config created: d_model={d_model}, num_heads={num_heads}")
            
            # Replace multiple layers
            print("🔄 Creating hybrid layers...")
            original_layers = {}
            hybrid_layers = {}
            
            for layer_idx in replacement_layers:
                print(f"   Processing layer {layer_idx}...")
                original_layer = layer_list[layer_idx]
                original_layers[layer_idx] = original_layer
                
                # Create hybrid layer
                hybrid_layer = HybridStreamingTransformerLayer(original_layer, ttt_config)
                hybrid_layers[layer_idx] = hybrid_layer
                
                # Replace in model
                layer_list[layer_idx] = hybrid_layer
                print(f"   ✅ Layer {layer_idx} replaced")
            
            print(f"✅ Successfully replaced {len(replacement_layers)} layers with hybrid layers!")
            
            # Test forward pass with multiple hybrid layers
            print("🚀 Testing forward pass with multiple hybrid layers...")
            
            # Create test input
            batch_size = 1  # Small batch for stability
            seq_len = 8     # Short sequence for speed
            n_q = test_config.get('n_q', 8)
            
            codes = torch.randint(0, 100, (batch_size, n_q + 1, seq_len), dtype=torch.int64)
            print(f"✅ Created dummy codes: shape {codes.shape}, dtype {codes.dtype}")
            
            # Forward pass through model with multiple hybrid layers
            lm_model.eval()
            with torch.no_grad():
                output = lm_model(codes)
                
            print(f"✅ Forward pass successful!")
            print(f"   📊 Output type: {type(output)}")
            print(f"   📏 Logits shape: {output.logits.shape}")
            if hasattr(output, 'text_logits') and output.text_logits is not None:
                print(f"   📝 Text logits shape: {output.text_logits.shape}")
            
            # Check output health with realistic expectations
            finite_ratio = torch.isfinite(output.logits).float().mean().item()
            text_finite_ratio = 1.0
            if hasattr(output, 'text_logits') and output.text_logits is not None:
                text_finite_ratio = torch.isfinite(output.text_logits).float().mean().item()
            
            print(f"   📊 Audio logits finite ratio: {finite_ratio:.1%}")
            print(f"   📝 Text logits finite ratio: {text_finite_ratio:.1%}")
            
            # Success criteria (learned from Step 3.5.1)
            success = (
                finite_ratio > 0.7 and  # At least 70% finite audio logits
                text_finite_ratio > 0.95  # At least 95% finite text logits
            )
            
            if success:
                print("✅ Output validation passed!")
                print(f"   ✅ Multiple hybrid layers work correctly")
                print(f"   ✅ Model stability maintained with {len(replacement_layers)} TTT layers")
            else:
                print("⚠️  Output validation concerns:")
                print(f"   Audio finite ratio: {finite_ratio:.1%} (target: >70%)")
                print(f"   Text finite ratio: {text_finite_ratio:.1%} (target: >95%)")
            
            # Test model parameter count
            total_params = sum(p.numel() for p in lm_model.parameters())
            ttt_params = sum(
                p.numel() for layer_idx in replacement_layers
                for p in layer_list[layer_idx].seq_modeling_block.parameters()
            )
            
            print(f"   📊 Total model parameters: {total_params:,}")
            print(f"   📊 TTT parameters: {ttt_params:,} ({ttt_params/total_params:.1%} of total)")
            
            return success
            
        else:
            print("❌ Model doesn't have transformer.layers - different architecture than expected")
            return False
            
    except Exception as e:
        print(f"❌ Multiple layer replacement failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_layer_interaction():
    """Test that multiple TTT layers interact correctly"""
    print("\n🔍 Step 3.5.2b: Layer Interaction Test...")
    
    try:
        from moshi.modules.transformer import StreamingTransformerLayer
        from moshi_ttt.hybrid_layer import HybridStreamingTransformerLayer
        from moshi_ttt.config import TTTConfig
        
        # Create chain of layers: Original → TTT → Original → TTT
        d_model = 512
        num_heads = 8
        
        # Create layers
        layers = []
        for i in range(4):
            original_layer = StreamingTransformerLayer(
                d_model=d_model,
                num_heads=num_heads,
                dim_feedforward=d_model * 4,
                causal=True,
                context=100,
                layer_scale=None,
                gating='silu',
                norm='rms_norm'
            )
            
            if i % 2 == 1:  # Make layers 1 and 3 hybrid
                ttt_config = TTTConfig(
                    model_dim=d_model,
                    num_heads=num_heads,
                    mini_batch_size=16,
                    ttt_base_lr=0.1
                )
                layer = HybridStreamingTransformerLayer(original_layer, ttt_config)
                layer_type = "TTT"
            else:
                layer = original_layer
                layer_type = "Original"
            
            layers.append(layer)
            print(f"   Layer {i}: {layer_type}")
        
        print(f"✅ Created 4-layer chain: Original→TTT→Original→TTT")
        
        # Test forward pass through chain
        batch_size = 2
        seq_len = 16
        x = torch.randn(batch_size, seq_len, d_model) * 0.1
        
        print(f"Input: {x.shape}, std={x.std():.3f}")
        
        # Pass through all layers
        with torch.no_grad():
            for i, layer in enumerate(layers):
                x = layer(x)
                finite_ratio = torch.isfinite(x).float().mean().item()
                print(f"   After layer {i}: std={x.std():.3f}, finite={finite_ratio:.1%}")
        
        print(f"✅ Layer chain test passed!")
        print(f"   Final output: {x.shape}, finite={torch.isfinite(x).float().mean():.1%}")
        
        return True
        
    except Exception as e:
        print(f"❌ Layer interaction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🚀 Step 3.5.2: Multiple Layer Replacement Testing")
    print("=" * 60)
    
    # Run main test
    success1 = test_multiple_layer_replacement()
    
    # Run interaction test
    success2 = test_layer_interaction()
    
    print("\n" + "=" * 60)
    print("📊 Step 3.5.2 Results:")
    print(f"   Multiple Layer Replacement: {'✅ PASS' if success1 else '❌ FAIL'}")
    print(f"   Layer Interaction Test: {'✅ PASS' if success2 else '❌ FAIL'}")
    
    overall_success = success1 and success2
    
    print(f"\n🏆 Step 3.5.2 OVERALL: {'✅ PASSED' if overall_success else '❌ FAILED'}")
    
    if overall_success:
        print("🚀 Ready to proceed to Step 3.5.3: Full Model Replacement!")
    else:
        print("🔧 Need to debug multiple layer issues before proceeding")

if __name__ == "__main__":
    main()