#!/usr/bin/env python3
"""
CRITICAL INVESTIGATION: Moshi's Dual Transformer Architecture
- Main Transformer
- Deformer
- Which one has 4096 dimensions?
- Which one should get TTT?
- Are we applying TTT to the right transformer?
"""

import sys
import torch
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def investigate_moshi_dual_architecture():
    """Investigate Moshi's main transformer vs deformer"""
    print("🔍 MOSHI DUAL TRANSFORMER ARCHITECTURE INVESTIGATION")
    print("=" * 80)
    
    try:
        from moshi.models import loaders
        from finetune.args import TrainArgs
        
        print("1. Loading Moshi model to examine architecture...")
        
        # Load exactly like train.py
        checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
            hf_repo="kyutai/moshiko-pytorch-bf16",
            moshi_weights=None,
            mimi_weights=None,
            tokenizer=None,
            config_path=None,
        )
        
        # Load on meta device first
        with torch.device("meta"):
            model = checkpoint_info.get_moshi(
                device="meta", 
                dtype=torch.bfloat16,
                lm_kwargs_overrides={
                    "gradient_checkpointing": True,
                    "lora": False,
                },
                load_weight=False,
            )
        
        print(f"2. Model type: {type(model)}")
        print(f"   Model attributes: {[attr for attr in dir(model) if not attr.startswith('_')]}")
        
        print("\n3. Investigating transformer components...")
        
        # Check for main transformer
        if hasattr(model, 'transformer'):
            print(f"   ✅ Has 'transformer': {type(model.transformer)}")
            transformer = model.transformer
            
            if hasattr(transformer, 'dim'):
                print(f"      transformer.dim: {transformer.dim}")
            if hasattr(transformer, 'num_heads'):
                print(f"      transformer.num_heads: {transformer.num_heads}")
            if hasattr(transformer, 'layers'):
                print(f"      transformer.layers: {len(transformer.layers)} layers")
                print(f"      first layer type: {type(transformer.layers[0])}")
        
        # Check for deformer
        if hasattr(model, 'deformer'):
            print(f"   ✅ Has 'deformer': {type(model.deformer)}")
            deformer = model.deformer
            
            if hasattr(deformer, 'dim'):
                print(f"      deformer.dim: {deformer.dim}")
            if hasattr(deformer, 'num_heads'):
                print(f"      deformer.num_heads: {deformer.num_heads}")
            if hasattr(deformer, 'layers'):
                print(f"      deformer.layers: {len(deformer.layers)} layers")
                print(f"      first layer type: {type(deformer.layers[0])}")
        
        # Check for other transformer-like components
        transformer_like = []
        for attr_name in dir(model):
            if not attr_name.startswith('_'):
                attr = getattr(model, attr_name)
                if hasattr(attr, 'layers') and hasattr(attr, 'dim'):
                    transformer_like.append((attr_name, attr))
        
        print(f"\n4. All transformer-like components found: {len(transformer_like)}")
        for name, component in transformer_like:
            print(f"   {name}:")
            print(f"      type: {type(component)}")
            print(f"      dim: {getattr(component, 'dim', 'N/A')}")
            print(f"      num_heads: {getattr(component, 'num_heads', 'N/A')}")
            print(f"      layers: {len(getattr(component, 'layers', []))}")
        
        return model
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

def check_current_ttt_integration():
    """Check which transformer our TTT integration is targeting"""
    print("\n" + "=" * 80)
    print("CURRENT TTT INTEGRATION ANALYSIS")
    print("=" * 80)
    
    try:
        # Check our TTT integration code
        from finetune.ttt_integration import apply_ttt_to_model
        import inspect
        
        print("1. Examining apply_ttt_to_model function...")
        source = inspect.getsource(apply_ttt_to_model)
        
        # Look for which transformer it targets
        lines = source.split('\n')
        for i, line in enumerate(lines):
            if 'transformer' in line.lower() and ('layers' in line or 'dim' in line):
                print(f"   Line {i}: {line.strip()}")
        
        print("\n2. Key findings from TTT integration:")
        print("   - Targets: model.transformer.layers")
        print("   - Assumes single transformer architecture")
        print("   - May be missing the deformer!")
        
    except Exception as e:
        print(f"ERROR: {e}")

def analyze_training_logs_for_clues():
    """Analyze our training logs for clues about which transformer was used"""
    print("\n" + "=" * 80) 
    print("TRAINING LOG ANALYSIS")
    print("=" * 80)
    
    print("From our successful training run:")
    print("   - 'Applying TTT to 16 layers: [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]'")
    print("   - 'Converted layers: 16/16'")
    print("   - '✅ TTT verification: 16/32 layers are TTT-enabled'")
    
    print("\nQuestions this raises:")
    print("   1. Are these 32 layers from the main transformer only?")
    print("   2. Or are they split between main transformer + deformer?")
    print("   3. Which transformer should get TTT for best results?")
    print("   4. Should we apply TTT to both transformers?")

def investigate_error_tensor_source():
    """Try to determine which transformer produced the 4096-dim tensor"""
    print("\n" + "=" * 80)
    print("ERROR TENSOR SOURCE ANALYSIS") 
    print("=" * 80)
    
    print("From the error:")
    print("   'AssertionError: Model dim mismatch: 4096 != 1024'")
    print("   Location: hybrid_layer.py:200 in _ttt_forward")
    print("   Called from: hybrid_layer.py:170 in forward")
    
    print("\nThis tells us:")
    print("   1. A tensor with 4096 dimensions reached our TTT layer")
    print("   2. Our TTT layer was expecting 1024 (wrong config)")
    print("   3. The 4096-dim tensor came from the transformer we're modifying")
    print("   4. So we ARE targeting the right transformer (the one with 4096 dims)")
    
    print("\nBut we need to verify:")
    print("   1. Is this the main transformer or deformer?")
    print("   2. What dimensions does the other transformer have?")
    print("   3. Should TTT be applied to both?")

def main():
    """Run the investigation"""
    model = investigate_moshi_dual_architecture()
    check_current_ttt_integration()
    analyze_training_logs_for_clues()
    investigate_error_tensor_source()
    
    print("\n" + "=" * 80)
    print("🎯 KEY QUESTIONS TO RESOLVE")
    print("=" * 80)
    print("1. Which transformer has 4096 dimensions - main or deformer?")
    print("2. What dimensions does the other transformer have?") 
    print("3. Are we applying TTT to the correct transformer?")
    print("4. Should TTT be applied to both transformers?")
    print("5. How do main transformer and deformer interact?")

if __name__ == "__main__":
    main()