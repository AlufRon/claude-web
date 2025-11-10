#!/usr/bin/env python3
"""
Diagnostic: What's different about TTT in full Moshi model context?
Compare layer inputs in isolation vs full model
"""

import torch
import sys
import os

sys.path.append('/home/alufr/ttt_tests/moshi')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def analyze_layer_input_in_model():
    """Analyze what actual layer inputs look like in full Moshi model"""
    print("🔍 ANALYZING LAYER INPUTS IN FULL MOSHI MODEL")
    
    try:
        from moshi.models import loaders
        
        # Load smaller model for analysis
        checkpoint_info = loaders.CheckpointInfo.from_hf_repo("kyutai/moshiko-pytorch-bf16")
        lm_config = loaders._lm_kwargs if checkpoint_info.raw_config is None else checkpoint_info.raw_config
        
        # Create minimal model
        small_config = lm_config.copy()
        small_config['num_layers'] = 2
        small_config['dim'] = 512
        small_config['num_heads'] = 8
        
        lm_model = loaders.get_moshi_lm(
            filename=None,
            lm_kwargs=small_config,
            device='cpu',
            dtype=torch.float32
        )
        
        print(f"✅ Created minimal model: {small_config['dim']}d, {small_config['num_layers']} layers")
        
        # Hook to capture layer inputs
        layer_inputs = {}
        
        def make_input_hook(layer_idx):
            def hook(module, input, output):
                x = input[0]  # First input tensor
                layer_inputs[layer_idx] = {
                    'input': x.clone().detach(),
                    'stats': {
                        'shape': x.shape,
                        'min': x.min().item(),
                        'max': x.max().item(), 
                        'mean': x.mean().item(),
                        'std': x.std().item(),
                        'finite_ratio': torch.isfinite(x).float().mean().item(),
                        'zero_ratio': (x == 0).float().mean().item()
                    }
                }
            return hook
        
        # Attach hooks to transformer layers
        hooks = []
        for i, layer in enumerate(lm_model.transformer.layers):
            hook = layer.register_forward_hook(make_input_hook(i))
            hooks.append(hook)
        
        # Run forward pass
        batch_size = 1
        seq_len = 8
        n_q = small_config.get('n_q', 8)
        codes = torch.randint(0, 100, (batch_size, n_q + 1, seq_len), dtype=torch.int64)
        
        print(f"Input codes: shape={codes.shape}, min={codes.min()}, max={codes.max()}")
        
        lm_model.eval()
        with torch.no_grad():
            output = lm_model(codes)
        
        # Clean up hooks
        for hook in hooks:
            hook.remove()
        
        # Analyze captured inputs
        print(f"\n📊 LAYER INPUT ANALYSIS:")
        print("-" * 60)
        
        for layer_idx in sorted(layer_inputs.keys()):
            stats = layer_inputs[layer_idx]['stats']
            print(f"Layer {layer_idx}:")
            print(f"  Shape: {stats['shape']}")
            print(f"  Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
            print(f"  Mean±Std: {stats['mean']:.3f}±{stats['std']:.3f}")
            print(f"  Finite: {stats['finite_ratio']:.1%}")
            print(f"  Zeros: {stats['zero_ratio']:.1%}")
            
            if stats['finite_ratio'] < 1.0:
                print(f"  ⚠️  Non-finite values present!")
        
        # Test TTT with these realistic inputs
        print(f"\n🧪 TESTING TTT WITH REALISTIC MODEL INPUTS")
        print("-" * 60)
        
        from moshi_ttt.hybrid_layer import HybridStreamingTransformerLayer
        from moshi_ttt.config import TTTConfig
        
        # Create TTT version of layer 0
        original_layer = lm_model.transformer.layers[0]
        ttt_config = TTTConfig(
            model_dim=small_config['dim'],
            num_heads=small_config['num_heads'],
            mini_batch_size=16,
            ttt_base_lr=0.1
        )
        
        hybrid_layer = HybridStreamingTransformerLayer(original_layer, ttt_config)
        
        # Test with captured realistic input
        if 0 in layer_inputs:
            realistic_input = layer_inputs[0]['input']
            print(f"Using realistic input from layer 0: {realistic_input.shape}")
            
            with torch.no_grad():
                # Original layer
                original_output = original_layer(realistic_input)
                orig_finite = torch.isfinite(original_output).float().mean().item()
                
                # Hybrid layer  
                hybrid_output = hybrid_layer(realistic_input)
                hybrid_finite = torch.isfinite(hybrid_output).float().mean().item()
                
                print(f"Original layer output: {orig_finite:.1%} finite")
                print(f"Hybrid layer output: {hybrid_finite:.1%} finite")
                
                if orig_finite > hybrid_finite:
                    print(f"⚠️  TTT reduces finite ratio by {orig_finite - hybrid_finite:.1%}")
                elif hybrid_finite < orig_finite:
                    print(f"✅ TTT improves finite ratio by {hybrid_finite - orig_finite:.1%}")
                else:
                    print(f"✅ TTT maintains finite ratio")
        
        return True
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def compare_embeddings_impact():
    """Check if embedding layers contribute to the NaN issue"""
    print(f"\n🔍 CHECKING EMBEDDING CONTRIBUTION TO NaN")
    print("-" * 60)
    
    try:
        from moshi.models import loaders
        
        checkpoint_info = loaders.CheckpointInfo.from_hf_repo("kyutai/moshiko-pytorch-bf16")
        lm_config = loaders._lm_kwargs if checkpoint_info.raw_config is None else checkpoint_info.raw_config
        
        # Test embedding outputs
        small_config = lm_config.copy()
        small_config['num_layers'] = 1
        small_config['dim'] = 512
        
        lm_model = loaders.get_moshi_lm(
            filename=None,
            lm_kwargs=small_config,
            device='cpu',
            dtype=torch.float32
        )
        
        # Test embedding layer outputs
        codes = torch.randint(0, 100, (1, 9, 5), dtype=torch.int64)
        
        with torch.no_grad():
            # Get embedding outputs
            if hasattr(lm_model, 'emb'):
                emb_outputs = []
                for i, emb_layer in enumerate(lm_model.emb):
                    emb_out = emb_layer(codes[:, i])
                    finite_ratio = torch.isfinite(emb_out).float().mean().item()
                    print(f"Embedding layer {i}: {finite_ratio:.1%} finite")
                    emb_outputs.append(emb_out)
                
                # Check combined embedding
                if emb_outputs:
                    combined_emb = torch.stack(emb_outputs, dim=1).sum(dim=1)
                    finite_ratio = torch.isfinite(combined_emb).float().mean().item()
                    print(f"Combined embeddings: {finite_ratio:.1%} finite")
        
    except Exception as e:
        print(f"❌ Embedding analysis failed: {e}")

def main():
    print("🔍 MOSHI MODEL CONTEXT DIAGNOSTIC")
    print("Understanding NaN source in full model vs isolated layer")
    print("=" * 70)
    
    success1 = analyze_layer_input_in_model()
    compare_embeddings_impact()
    
    print(f"\n📋 KEY INSIGHTS:")
    print("1. Check if layer inputs contain non-finite values")
    print("2. Compare TTT behavior on realistic vs synthetic inputs")  
    print("3. Identify if embeddings contribute to NaN values")
    print("4. Understand propagation of non-finite values through layers")

if __name__ == "__main__":
    main()