#!/usr/bin/env python3
"""
DETERMINE EXACT DIMENSIONS: Load actual weights and check dimensions
"""

import sys
import torch
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def determine_exact_dimensions():
    """Load actual model weights and determine exact dimensions"""
    print("🔍 DETERMINING EXACT DIMENSIONS OF BOTH TRANSFORMERS")
    print("=" * 80)
    
    try:
        from moshi.models import loaders
        
        print("1. Loading Moshi model with actual weights...")
        checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
            hf_repo="kyutai/moshiko-pytorch-bf16",
            moshi_weights=None,
            mimi_weights=None,
            tokenizer=None,
            config_path=None,
        )
        
        # Load model with actual weights on CPU to check dimensions
        model = checkpoint_info.get_moshi(
            device="cpu",
            dtype=torch.bfloat16,
            lm_kwargs_overrides={"gradient_checkpointing": True, "lora": False},
            load_weight=True,  # Load actual weights
        )
        
        print("\n2. TRANSFORMER (Main) DIMENSION ANALYSIS:")
        transformer = model.transformer
        print(f"   Type: {type(transformer)}")
        print(f"   Layers: {len(transformer.layers)}")
        
        # Get dimensions from first layer
        first_layer = transformer.layers[0]
        print(f"   First layer type: {type(first_layer)}")
        
        # Check various ways to get dimensions
        dim_sources = []
        
        if hasattr(first_layer, 'norm1') and hasattr(first_layer.norm1, 'weight'):
            norm_dim = first_layer.norm1.weight.shape[0]
            dim_sources.append(('norm1.weight.shape[0]', norm_dim))
            
        if hasattr(first_layer, 'norm2') and hasattr(first_layer.norm2, 'weight'):
            norm2_dim = first_layer.norm2.weight.shape[0]
            dim_sources.append(('norm2.weight.shape[0]', norm2_dim))
            
        if hasattr(first_layer, 'self_attn'):
            attn = first_layer.self_attn
            if hasattr(attn, 'q_proj') and hasattr(attn.q_proj, 'weight'):
                q_in_dim = attn.q_proj.weight.shape[1]  # input dimension
                q_out_dim = attn.q_proj.weight.shape[0]  # output dimension
                dim_sources.append(('self_attn.q_proj.weight.shape[1]', q_in_dim))
                dim_sources.append(('self_attn.q_proj.weight.shape[0]', q_out_dim))
                
        if hasattr(first_layer, 'mlp'):
            mlp = first_layer.mlp
            if hasattr(mlp, 'gate_proj') and hasattr(mlp.gate_proj, 'weight'):
                mlp_in_dim = mlp.gate_proj.weight.shape[1]
                mlp_out_dim = mlp.gate_proj.weight.shape[0]
                dim_sources.append(('mlp.gate_proj.weight.shape[1]', mlp_in_dim))
                dim_sources.append(('mlp.gate_proj.weight.shape[0]', mlp_out_dim))
        
        print("   Dimension sources:")
        for source, dim in dim_sources:
            print(f"     {source}: {dim}")
        
        print("\n3. DEPFORMER DIMENSION ANALYSIS:")
        depformer = model.depformer
        print(f"   Type: {type(depformer)}")
        print(f"   Layers: {len(depformer.layers)}")
        
        # Get dimensions from first depformer layer
        first_dep_layer = depformer.layers[0]
        print(f"   First layer type: {type(first_dep_layer)}")
        
        # Check various ways to get dimensions
        dep_dim_sources = []
        
        if hasattr(first_dep_layer, 'norm1') and hasattr(first_dep_layer.norm1, 'weight'):
            norm_dim = first_dep_layer.norm1.weight.shape[0]
            dep_dim_sources.append(('norm1.weight.shape[0]', norm_dim))
            
        if hasattr(first_dep_layer, 'norm2') and hasattr(first_dep_layer.norm2, 'weight'):
            norm2_dim = first_dep_layer.norm2.weight.shape[0]
            dep_dim_sources.append(('norm2.weight.shape[0]', norm2_dim))
            
        if hasattr(first_dep_layer, 'self_attn'):
            attn = first_dep_layer.self_attn
            if hasattr(attn, 'q_proj') and hasattr(attn.q_proj, 'weight'):
                q_in_dim = attn.q_proj.weight.shape[1]
                q_out_dim = attn.q_proj.weight.shape[0]
                dep_dim_sources.append(('self_attn.q_proj.weight.shape[1]', q_in_dim))
                dep_dim_sources.append(('self_attn.q_proj.weight.shape[0]', q_out_dim))
                
        if hasattr(first_dep_layer, 'mlp'):
            mlp = first_dep_layer.mlp
            if hasattr(mlp, 'gate_proj') and hasattr(mlp.gate_proj, 'weight'):
                mlp_in_dim = mlp.gate_proj.weight.shape[1]
                mlp_out_dim = mlp.gate_proj.weight.shape[0]
                dep_dim_sources.append(('mlp.gate_proj.weight.shape[1]', mlp_in_dim))
                dep_dim_sources.append(('mlp.gate_proj.weight.shape[0]', mlp_out_dim))
        
        print("   Dimension sources:")
        for source, dim in dep_dim_sources:
            print(f"     {source}: {dim}")
        
        print("\n4. MODEL GLOBAL DIMENSIONS:")
        if hasattr(model, 'dim'):
            print(f"   model.dim: {model.dim}")
        
        # Check embedding dimensions
        if hasattr(model, 'emb'):
            emb = model.emb
            if hasattr(emb, 'weight'):
                emb_dim = emb.weight.shape[1]
                print(f"   emb.weight.shape[1]: {emb_dim}")
        
        # Check text embedding
        if hasattr(model, 'text_emb'):
            text_emb = model.text_emb
            if hasattr(text_emb, 'weight'):
                text_emb_dim = text_emb.weight.shape[1]
                print(f"   text_emb.weight.shape[1]: {text_emb_dim}")
        
        return model, dim_sources, dep_dim_sources
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None, [], []

def analyze_ttt_implications(model, transformer_dims, depformer_dims):
    """Analyze implications for TTT integration"""
    print("\n" + "=" * 80)
    print("TTT INTEGRATION IMPLICATIONS")
    print("=" * 80)
    
    if transformer_dims:
        main_dim = transformer_dims[0][1]  # First dimension found
        print(f"Main transformer dimension: {main_dim}")
    
    if depformer_dims:
        dep_dim = depformer_dims[0][1]  # First dimension found
        print(f"Depformer dimension: {dep_dim}")
    
    print("\nCurrent TTT status:")
    print("   - Applied to: model.transformer.layers (32 layers)")
    print("   - Middle layers: 16 layers (8-23)")
    print("   - Using 4096 dimensions (from our fix)")
    
    print("\nQuestions:")
    if transformer_dims and depformer_dims:
        if transformer_dims[0][1] == depformer_dims[0][1]:
            print("   ✅ Both transformers have same dimensions")
            print("   📋 Could apply TTT to both")
        else:
            print("   ⚠️  Transformers have different dimensions")
            print("   📋 Need different TTT configs for each")
    
    print("\nRecommendations:")
    print("   1. Verify which transformer the 4096-dim tensor came from")
    print("   2. Consider applying TTT to depformer too")
    print("   3. Evaluate relative importance of each transformer")

def main():
    """Run the investigation"""
    model, transformer_dims, depformer_dims = determine_exact_dimensions()
    
    if model:
        analyze_ttt_implications(model, transformer_dims, depformer_dims)
    
    print("\n" + "=" * 80)
    print("🎯 SUMMARY")
    print("=" * 80)
    print("This will tell us:")
    print("1. Exact dimensions of both transformers")
    print("2. Whether our 4096 assumption is correct")
    print("3. Which transformer generated the error tensor")
    print("4. Whether we should apply TTT to depformer too")

if __name__ == "__main__":
    main()