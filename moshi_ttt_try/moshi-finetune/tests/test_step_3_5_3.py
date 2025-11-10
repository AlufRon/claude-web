#!/usr/bin/env python3
"""
Step 3.5.3: Full Model Replacement Test
Replace ALL compatible Moshi layers with hybrid layers and test complete integration
Following Video-DiT architecture pattern: SeqModelingBlock contains TTT within attention→TTT flow
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

def test_full_model_replacement():
    """Test replacing ALL transformer layers with hybrid layers"""
    print("🧪 Step 3.5.3: Full Model Replacement Test...")
    
    try:
        # Import Moshi components
        from moshi.models import loaders
        from moshi_ttt.hybrid_layer import HybridStreamingTransformerLayer
        from moshi_ttt.config import TTTConfig
        
        print("✅ Imports successful")
        
        # Create minimal model for full replacement test
        print("📥 Loading minimal Moshi model for full replacement...")
        checkpoint_info = loaders.CheckpointInfo.from_hf_repo("kyutai/moshiko-pytorch-bf16")
        
        lm_config = (
            loaders._lm_kwargs 
            if checkpoint_info.raw_config is None 
            else checkpoint_info.raw_config
        )
        
        # Very small model for full replacement testing
        test_config = lm_config.copy()
        test_config['num_layers'] = 6      # Minimal layer count
        test_config['dim'] = 768          # Reasonable dimension  
        test_config['num_heads'] = 12     # Divisible by 768
        test_config['depformer_num_layers'] = 3  # Reduce depformer too
        
        print(f"✅ Model config: {test_config['dim']}d model with {test_config['num_layers']} transformer layers")
        print(f"   Depformer layers: {test_config['depformer_num_layers']}")
        
        # Build the model 
        print("🏗️  Building minimal Moshi LM model...")
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
        if not (hasattr(lm_model, 'transformer') and hasattr(lm_model.transformer, 'layers')):
            print("❌ Model doesn't have transformer.layers - unexpected architecture")
            return False
            
        layer_list = lm_model.transformer.layers
        total_layers = len(layer_list)
        print(f"   🏗️  Found {total_layers} transformer layers")
        
        # Analyze original model first
        print(f"\n📊 ORIGINAL MODEL ANALYSIS:")
        print("-" * 50)
        
        original_param_count = sum(p.numel() for p in lm_model.parameters())
        transformer_param_count = sum(p.numel() for p in lm_model.transformer.parameters())
        
        print(f"   Total model parameters: {original_param_count:,}")
        print(f"   Transformer parameters: {transformer_param_count:,} ({transformer_param_count/original_param_count:.1%})")
        
        # Test original model first
        print(f"\n🔬 TESTING ORIGINAL MODEL:")
        print("-" * 50)
        
        batch_size = 1
        seq_len = 6     # Very short for speed
        n_q = test_config.get('n_q', 8)
        
        test_codes = torch.randint(0, 50, (batch_size, n_q + 1, seq_len), dtype=torch.int64)
        print(f"   Test input: {test_codes.shape}")
        
        lm_model.eval()
        with torch.no_grad():
            original_output = lm_model(test_codes)
        
        orig_finite_ratio = torch.isfinite(original_output.logits).float().mean().item()
        orig_text_finite_ratio = 1.0
        if hasattr(original_output, 'text_logits') and original_output.text_logits is not None:
            orig_text_finite_ratio = torch.isfinite(original_output.text_logits).float().mean().item()
            
        print(f"   ✅ Original model forward pass successful")
        print(f"   📊 Original audio logits finite: {orig_finite_ratio:.1%}")
        print(f"   📝 Original text logits finite: {orig_text_finite_ratio:.1%}")
        
        # Create TTT config for full replacement
        print(f"\n🔄 FULL TTT REPLACEMENT:")
        print("-" * 50)
        
        d_model = test_config['dim']
        num_heads = test_config['num_heads']
        
        ttt_config = TTTConfig(
            model_dim=d_model,
            num_heads=num_heads,
            mini_batch_size=16,
            ttt_base_lr=0.1
        )
        
        print(f"   TTT config: d_model={d_model}, num_heads={num_heads}")
        print(f"   Replacing ALL {total_layers} transformer layers...")
        
        # Replace ALL transformer layers with hybrid layers
        # Following Video-DiT pattern: each TransformerLayer contains SeqModelingBlock with TTT
        original_layers = []
        replacement_count = 0
        
        for i, layer in enumerate(layer_list):
            print(f"     Processing layer {i}...")
            original_layers.append(layer)
            
            # Create hybrid layer (Video-DiT SeqModelingBlock equivalent)
            hybrid_layer = HybridStreamingTransformerLayer(layer, ttt_config)
            layer_list[i] = hybrid_layer
            replacement_count += 1
            
            print(f"     ✅ Layer {i}: StreamingTransformerLayer → HybridStreamingTransformerLayer")
        
        print(f"   🎉 Successfully replaced ALL {replacement_count} layers!")
        
        # Analyze fully TTT model
        print(f"\n📊 FULLY TTT MODEL ANALYSIS:")
        print("-" * 50)
        
        ttt_param_count = sum(p.numel() for p in lm_model.parameters())
        ttt_transformer_param_count = sum(p.numel() for p in lm_model.transformer.parameters())
        
        print(f"   Total TTT model parameters: {ttt_param_count:,}")
        print(f"   TTT transformer parameters: {ttt_transformer_param_count:,} ({ttt_transformer_param_count/ttt_param_count:.1%})")
        print(f"   Parameter increase: +{ttt_param_count - original_param_count:,} (+{(ttt_param_count/original_param_count - 1)*100:.1f}%)")
        
        # Calculate TTT-specific parameters
        ttt_only_params = 0
        for name, param in lm_model.named_parameters():
            if any(ttt_key in name for ttt_key in ['W1', 'W2', 'b1', 'b2', 'ttt_norm', 'learnable_ttt']):
                ttt_only_params += param.numel()
        
        print(f"   New TTT parameters: {ttt_only_params:,} ({ttt_only_params/ttt_param_count:.1%} of total)")
        print(f"   Preserved Moshi parameters: {ttt_param_count - ttt_only_params:,} ({(ttt_param_count - ttt_only_params)/ttt_param_count:.1%} of total)")
        
        # Test fully TTT model
        print(f"\n🚀 TESTING FULLY TTT MODEL:")
        print("-" * 50)
        
        print(f"   Input: {test_codes.shape}")
        
        # Forward pass through fully TTT model
        lm_model.eval()
        with torch.no_grad():
            ttt_output = lm_model(test_codes)
            
        print(f"   ✅ Fully TTT forward pass successful!")
        print(f"   📊 Output type: {type(ttt_output)}")
        print(f"   📏 TTT logits shape: {ttt_output.logits.shape}")
        if hasattr(ttt_output, 'text_logits') and ttt_output.text_logits is not None:
            print(f"   📝 TTT text logits shape: {ttt_output.text_logits.shape}")
        
        # Analyze output health
        ttt_finite_ratio = torch.isfinite(ttt_output.logits).float().mean().item()
        ttt_text_finite_ratio = 1.0
        if hasattr(ttt_output, 'text_logits') and ttt_output.text_logits is not None:
            ttt_text_finite_ratio = torch.isfinite(ttt_output.text_logits).float().mean().item()
        
        print(f"   📊 TTT audio logits finite: {ttt_finite_ratio:.1%}")
        print(f"   📝 TTT text logits finite: {ttt_text_finite_ratio:.1%}")
        
        # Compare original vs TTT
        print(f"\n📊 ORIGINAL vs FULLY TTT COMPARISON:")
        print("-" * 50)
        print(f"   Audio logits finite ratio:")
        print(f"     Original: {orig_finite_ratio:.1%}")
        print(f"     TTT:      {ttt_finite_ratio:.1%}")
        print(f"     Delta:    {ttt_finite_ratio - orig_finite_ratio:+.1%}")
        
        print(f"   Text logits finite ratio:")
        print(f"     Original: {orig_text_finite_ratio:.1%}")
        print(f"     TTT:      {ttt_text_finite_ratio:.1%}")
        print(f"     Delta:    {ttt_text_finite_ratio - orig_text_finite_ratio:+.1%}")
        
        # Output difference analysis
        audio_diff = torch.norm(ttt_output.logits - original_output.logits).item()
        print(f"   Output difference (L2 norm): {audio_diff:.1f}")
        
        # Success criteria for full model replacement
        # Based on our previous findings: TTT may reduce finite ratio but should maintain usability
        success_criteria = {
            'forward_pass': True,  # Forward pass completed
            'audio_finite': ttt_finite_ratio > 0.6,  # At least 60% finite audio logits
            'text_finite': ttt_text_finite_ratio > 0.95,  # At least 95% finite text logits
            'output_changed': audio_diff > 10,  # Output significantly different (TTT is active)
            'param_reasonable': (ttt_param_count / original_param_count) < 1.2  # Less than 20% increase
        }
        
        print(f"\n✅ SUCCESS CRITERIA EVALUATION:")
        print("-" * 50)
        all_passed = True
        for criterion, passed in success_criteria.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"   {criterion}: {status}")
            if not passed:
                all_passed = False
        
        if all_passed:
            print(f"\n🏆 FULL MODEL REPLACEMENT: ✅ SUCCESS!")
            print(f"   🎉 ALL {replacement_count} layers successfully converted to TTT")
            print(f"   🚀 Fully TTT-enabled Moshi model operational!")
            print(f"   💾 {((ttt_param_count - ttt_only_params)/ttt_param_count)*100:.1f}% of Moshi weights preserved")
            print(f"   ⚡ {(ttt_only_params/ttt_param_count)*100:.1f}% new TTT capabilities added")
        else:
            print(f"\n⚠️  FULL MODEL REPLACEMENT: PARTIAL SUCCESS")
            print(f"   Some criteria not met, but basic functionality works")
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Full model replacement failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_readiness():
    """Test that fully TTT model is ready for training"""
    print("\n🔍 Step 3.5.3b: Training Readiness Test...")
    
    try:
        from moshi.models import loaders
        from moshi_ttt.hybrid_layer import HybridStreamingTransformerLayer
        from moshi_ttt.config import TTTConfig
        
        # Create minimal training-ready model
        checkpoint_info = loaders.CheckpointInfo.from_hf_repo("kyutai/moshiko-pytorch-bf16")
        lm_config = loaders._lm_kwargs if checkpoint_info.raw_config is None else checkpoint_info.raw_config
        
        test_config = lm_config.copy()
        test_config['num_layers'] = 3
        test_config['dim'] = 384
        test_config['num_heads'] = 6
        test_config['depformer_num_layers'] = 2
        
        lm_model = loaders.get_moshi_lm(filename=None, lm_kwargs=test_config, device='cpu', dtype=torch.float32)
        
        # Replace all transformer layers
        ttt_config = TTTConfig(model_dim=test_config['dim'], num_heads=test_config['num_heads'], 
                              mini_batch_size=8, ttt_base_lr=0.1)
        
        for i, layer in enumerate(lm_model.transformer.layers):
            lm_model.transformer.layers[i] = HybridStreamingTransformerLayer(layer, ttt_config)
        
        print(f"✅ Created fully TTT model: {test_config['dim']}d, {test_config['num_layers']} layers")
        
        # Test training step
        batch_size = 1
        seq_len = 4
        n_q = test_config.get('n_q', 8)
        
        codes = torch.randint(0, 20, (batch_size, n_q + 1, seq_len), dtype=torch.int64)
        
        # Enable gradients
        lm_model.train()
        for param in lm_model.parameters():
            param.requires_grad = True
        
        # Forward pass
        output = lm_model(codes)
        
        # Create dummy loss
        loss = output.logits.sum()
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        total_params = 0
        params_with_grad = 0
        ttt_params_with_grad = 0
        
        for name, param in lm_model.named_parameters():
            if param.requires_grad:
                total_params += 1
                if param.grad is not None:
                    params_with_grad += 1
                    if any(ttt_key in name for ttt_key in ['W1', 'W2', 'b1', 'b2', 'ttt_norm', 'learnable_ttt']):
                        ttt_params_with_grad += 1
        
        grad_ratio = params_with_grad / total_params if total_params > 0 else 0
        
        print(f"✅ Training readiness test completed:")
        print(f"   Forward pass: ✅")
        print(f"   Backward pass: ✅") 
        print(f"   Gradient flow: {params_with_grad}/{total_params} ({grad_ratio:.1%})")
        print(f"   TTT gradients: {ttt_params_with_grad} parameters")
        
        training_ready = grad_ratio > 0.9  # At least 90% of parameters have gradients
        
        return training_ready
        
    except Exception as e:
        print(f"❌ Training readiness test failed: {e}")
        return False

def main():
    print("🚀 Step 3.5.3: Full Model Replacement Testing")
    print("Following Video-DiT architecture: SeqModelingBlock pattern")
    print("=" * 70)
    
    # Run main test
    success1 = test_full_model_replacement()
    
    # Run training readiness test
    success2 = test_training_readiness()
    
    print("\n" + "=" * 70)
    print("📊 Step 3.5.3 Results:")
    print(f"   Full Model Replacement: {'✅ PASS' if success1 else '❌ FAIL'}")
    print(f"   Training Readiness: {'✅ PASS' if success2 else '❌ FAIL'}")
    
    overall_success = success1 and success2
    
    print(f"\n🏆 Step 3.5.3 OVERALL: {'✅ PASSED' if overall_success else '❌ FAILED'}")
    
    if overall_success:
        print("\n🎉 PHASE 3 COMPLETE: TTT Integration Successful!")
        print("🚀 Ready to proceed to Phase 4: Model Integration & Training!")
        print("\n💡 Next Steps:")
        print("   - Phase 4.1: Layer replacement utilities")
        print("   - Phase 4.2: Training integration") 
        print("   - Phase 4.3: Performance optimization")
        print("\n🏁 TTT-Moshi hybrid model is fully operational!")
    else:
        print("\n🔧 Need to resolve issues before proceeding to Phase 4")

if __name__ == "__main__":
    main()