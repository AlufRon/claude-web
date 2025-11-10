#!/usr/bin/env python3
"""
CRITICAL INVESTIGATION: Moshi's Depformer vs Transformer
Now that we found both 'transformer' and 'depformer', let's investigate both!
"""

import sys
import torch
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def investigate_both_transformers():
    """Investigate both transformer and depformer in detail"""
    print("🔍 MOSHI TRANSFORMER vs DEPFORMER INVESTIGATION")
    print("=" * 80)
    
    try:
        from moshi.models import loaders
        
        print("1. Loading Moshi model...")
        checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
            hf_repo="kyutai/moshiko-pytorch-bf16",
            moshi_weights=None,
            mimi_weights=None, 
            tokenizer=None,
            config_path=None,
        )
        
        with torch.device("meta"):
            model = checkpoint_info.get_moshi(
                device="meta",
                dtype=torch.bfloat16,
                lm_kwargs_overrides={"gradient_checkpointing": True, "lora": False},
                load_weight=False,
            )
        
        print("\n2. TRANSFORMER ANALYSIS:")
        if hasattr(model, 'transformer'):
            transformer = model.transformer
            print(f"   ✅ Type: {type(transformer)}")
            print(f"   ✅ Has dim: {hasattr(transformer, 'dim')}")
            print(f"   ✅ Has num_heads: {hasattr(transformer, 'num_heads')}")
            print(f"   ✅ Has layers: {hasattr(transformer, 'layers')}")
            
            if hasattr(transformer, 'dim'):
                print(f"   📊 transformer.dim: {transformer.dim}")
            if hasattr(transformer, 'num_heads'):
                print(f"   📊 transformer.num_heads: {transformer.num_heads}")
            if hasattr(transformer, 'layers'):
                layers = transformer.layers
                print(f"   📊 transformer.layers: {len(layers)} layers")
                if len(layers) > 0:
                    first_layer = layers[0]
                    print(f"   📊 First layer type: {type(first_layer)}")
                    
                    # Check layer dimensions
                    if hasattr(first_layer, 'norm1') and hasattr(first_layer.norm1, 'weight'):
                        dim = first_layer.norm1.weight.shape[0]
                        print(f"   📊 Layer dimension (from norm): {dim}")
        else:
            print("   ❌ No transformer found")
        
        print("\n3. DEPFORMER ANALYSIS:")
        if hasattr(model, 'depformer'):
            depformer = model.depformer
            print(f"   ✅ Type: {type(depformer)}")
            print(f"   ✅ Has dim: {hasattr(depformer, 'dim')}")
            print(f"   ✅ Has num_heads: {hasattr(depformer, 'num_heads')}")
            print(f"   ✅ Has layers: {hasattr(depformer, 'layers')}")
            
            if hasattr(depformer, 'dim'):
                print(f"   📊 depformer.dim: {depformer.dim}")
            if hasattr(depformer, 'num_heads'):
                print(f"   📊 depformer.num_heads: {depformer.num_heads}")
            if hasattr(depformer, 'layers'):
                layers = depformer.layers
                print(f"   📊 depformer.layers: {len(layers)} layers")
                if len(layers) > 0:
                    first_layer = layers[0]
                    print(f"   📊 First layer type: {type(first_layer)}")
                    
                    # Check layer dimensions  
                    if hasattr(first_layer, 'norm1') and hasattr(first_layer.norm1, 'weight'):
                        dim = first_layer.norm1.weight.shape[0]
                        print(f"   📊 Layer dimension (from norm): {dim}")
        else:
            print("   ❌ No depformer found")
        
        print("\n4. MODEL OVERVIEW:")
        print(f"   Model type: {type(model)}")
        print(f"   Model dim: {getattr(model, 'dim', 'N/A')}")
        
        # Check all transformer-like attributes
        transformer_attrs = []
        for attr_name in dir(model):
            if 'transform' in attr_name.lower() or 'former' in attr_name.lower():
                if not attr_name.startswith('_'):
                    attr = getattr(model, attr_name)
                    if hasattr(attr, 'layers') or 'transform' in str(type(attr)).lower():
                        transformer_attrs.append((attr_name, attr))
        
        print(f"\n5. ALL TRANSFORMER-LIKE ATTRIBUTES ({len(transformer_attrs)}):")
        for name, attr in transformer_attrs:
            print(f"   {name}: {type(attr)}")
            if hasattr(attr, 'dim'):
                print(f"      dim: {attr.dim}")
            if hasattr(attr, 'layers'):
                print(f"      layers: {len(attr.layers)}")
        
        return model
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_which_transformer_gets_ttt():
    """Analyze which transformer our TTT is currently targeting"""
    print("\n" + "=" * 80)
    print("TTT TARGET ANALYSIS")
    print("=" * 80)
    
    print("Current TTT integration targets:")
    print("   model.transformer.layers  ← Main transformer")
    print("   (NOT model.depformer.layers)")
    
    print("\nFrom training logs:")
    print("   - 32 total layers found")
    print("   - 16 layers converted (middle layers 8-23)")
    print("   - Error tensor had 4096 dimensions")
    
    print("\nThis suggests:")
    print("   1. model.transformer has 32 layers")
    print("   2. model.transformer layers have 4096 dimensions")
    print("   3. We successfully applied TTT to the main transformer")
    print("   4. But what about the depformer?")

def check_moshi_paper_architecture():
    """Check what we know about Moshi architecture from papers/docs"""
    print("\n" + "=" * 80)
    print("MOSHI ARCHITECTURE KNOWLEDGE")
    print("=" * 80)
    
    print("From Moshi documentation/papers:")
    print("   - Moshi has a hierarchical architecture")
    print("   - Main transformer: processes main sequence")
    print("   - Depformer: processes dependency/delay information")
    print("   - Both transformers work together for audio generation")
    
    print("\nKey questions:")
    print("   1. Which transformer is more important for TTT?")
    print("   2. Do they have the same dimensions?")
    print("   3. Should TTT be applied to both?")
    print("   4. What's the interaction between them?")

def main():
    """Run the investigation"""
    model = investigate_both_transformers()
    analyze_which_transformer_gets_ttt()
    check_moshi_paper_architecture()
    
    print("\n" + "=" * 80)
    print("🎯 CRITICAL FINDINGS NEEDED")
    print("=" * 80)
    print("1. Dimensions of transformer vs depformer")
    print("2. Number of layers in each")
    print("3. Which one should get TTT?")
    print("4. Should both get TTT?")
    print("5. Are we missing half the model by only targeting transformer?")

if __name__ == "__main__":
    main()