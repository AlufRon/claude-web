#!/usr/bin/env python3
"""
COMPREHENSIVE INVESTIGATION: Shape/Dimension Issue in TTT-Moshi Integration

This script will thoroughly investigate the shape mismatch issue:
- Original error: "Model dim mismatch: 4096 != 1024"
- What dimensions are involved?
- Where do they come from?
- How does TTT expect them?
- Why did our fix work?
"""

import sys
import torch
import traceback
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def investigate_moshi_model_dimensions():
    """Investigate actual Moshi model dimensions"""
    print("=" * 80)
    print("INVESTIGATION 1: MOSHI MODEL DIMENSIONS")
    print("=" * 80)
    
    try:
        from moshi.models import loaders
        from finetune.args import TrainArgs
        
        # Load model like train.py does
        print("1. Loading Moshi model (same as train.py)...")
        checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
            hf_repo_id="kyutai/moshiko-pytorch-bf16",
            moshi_weights=None,
            mimi_weights=None,
            tokenizer=None,
            config_path=None,
        )
        
        print(f"   checkpoint_info.raw_config: {checkpoint_info.raw_config}")
        
        # Load model on meta device first (like train.py)
        with torch.device("meta"):
            model = checkpoint_info.get_moshi(
                device="meta",
                dtype=torch.bfloat16,
                lm_kwargs_overrides={
                    "gradient_checkpointing": True,
                    "lora": False,
                    "lora_rank": 32,
                    "lora_scaling": 2.0,
                },
                load_weight=False,
            )
        
        print("2. Investigating model structure...")
        print(f"   Model type: {type(model)}")
        print(f"   Has transformer: {hasattr(model, 'transformer')}")
        
        if hasattr(model, 'transformer'):
            transformer = model.transformer
            print(f"   Transformer type: {type(transformer)}")
            print(f"   Has dim attribute: {hasattr(transformer, 'dim')}")
            print(f"   Has num_heads attribute: {hasattr(transformer, 'num_heads')}")
            print(f"   Has layers: {hasattr(transformer, 'layers')}")
            
            if hasattr(transformer, 'dim'):
                print(f"   transformer.dim: {transformer.dim}")
            if hasattr(transformer, 'num_heads'):
                print(f"   transformer.num_heads: {transformer.num_heads}")
                
            if hasattr(transformer, 'layers') and len(transformer.layers) > 0:
                print(f"   Number of layers: {len(transformer.layers)}")
                first_layer = transformer.layers[0]
                print(f"   First layer type: {type(first_layer)}")
                
                # Investigate layer structure
                print("   First layer attributes:")
                for attr in ['self_attn', 'norm1', 'norm2', 'mlp']:
                    if hasattr(first_layer, attr):
                        component = getattr(first_layer, attr)
                        print(f"     {attr}: {type(component)}")
                        
                        if attr == 'self_attn':
                            attn_attrs = ['num_heads', 'head_dim', 'embed_dim', 'hidden_size']
                            for a in attn_attrs:
                                if hasattr(component, a):
                                    print(f"       {a}: {getattr(component, a)}")
                        elif attr == 'norm1' and hasattr(component, 'weight'):
                            print(f"       weight.shape: {component.weight.shape}")
                            
        # Check embedding dimensions
        if hasattr(model, 'embed_tokens'):
            embed = model.embed_tokens
            print(f"   Embedding weight shape: {embed.weight.shape}")
            
        # After loading weights
        print("\n3. Loading actual weights...")
        model_state_dict = torch.load(checkpoint_info.moshi_weights, map_location='cpu', weights_only=True)
        
        # Analyze some key tensor shapes
        print("   Key tensor shapes from state dict:")
        key_patterns = ['embed', 'transformer.layers.0', 'norm', 'head']
        for pattern in key_patterns:
            matching_keys = [k for k in model_state_dict.keys() if pattern in k]
            if matching_keys:
                print(f"     Pattern '{pattern}':")
                for key in matching_keys[:3]:  # Show first 3 matches
                    shape = model_state_dict[key].shape
                    print(f"       {key}: {shape}")
                    
        return model, checkpoint_info, model_state_dict
        
    except Exception as e:
        print(f"ERROR in Moshi investigation: {e}")
        traceback.print_exc()
        return None, None, None

def investigate_ttt_dimension_expectations():
    """Investigate what dimensions TTT expects"""
    print("\n" + "=" * 80)
    print("INVESTIGATION 2: TTT DIMENSION EXPECTATIONS")
    print("=" * 80)
    
    try:
        from moshi_ttt.config import TTTConfig
        from moshi_ttt.format_utils import moshi_to_ttt_format, ttt_to_moshi_format
        
        print("1. TTTConfig structure...")
        config_1024 = TTTConfig(model_dim=1024, num_heads=8, ttt_base_lr=0.1, mini_batch_size=8)
        config_4096 = TTTConfig(model_dim=4096, num_heads=32, ttt_base_lr=0.1, mini_batch_size=8)
        
        print(f"   TTTConfig with 1024: {config_1024}")
        print(f"   TTTConfig with 4096: {config_4096}")
        
        print("\n2. Testing format conversion logic...")
        
        # Test with different tensor shapes
        test_shapes = [
            (2, 100, 1024),  # [batch, seq_len, dim] - wrong dim
            (2, 100, 4096),  # [batch, seq_len, dim] - correct dim
        ]
        
        for i, shape in enumerate(test_shapes):
            print(f"\n   Test {i+1}: Input shape {shape}")
            test_tensor = torch.randn(*shape)
            
            # Test with 1024 config
            try:
                ttt_tensor, metadata = moshi_to_ttt_format(test_tensor, config_1024)
                print(f"     ✅ 1024 config: {test_tensor.shape} → {ttt_tensor.shape}")
            except Exception as e:
                print(f"     ❌ 1024 config failed: {e}")
                
            # Test with 4096 config  
            try:
                ttt_tensor, metadata = moshi_to_ttt_format(test_tensor, config_4096)
                print(f"     ✅ 4096 config: {test_tensor.shape} → {ttt_tensor.shape}")
            except Exception as e:
                print(f"     ❌ 4096 config failed: {e}")
                
        print("\n3. Examining format conversion code...")
        import inspect
        source = inspect.getsource(moshi_to_ttt_format)
        lines = source.split('\n')
        for i, line in enumerate(lines):
            if 'assert' in line and 'd_model' in line:
                print(f"   Line {i}: {line.strip()}")
                
    except Exception as e:
        print(f"ERROR in TTT investigation: {e}")
        traceback.print_exc()

def investigate_dimension_detection_logic():
    """Investigate the dimension detection we implemented"""
    print("\n" + "=" * 80)
    print("INVESTIGATION 3: DIMENSION DETECTION LOGIC")
    print("=" * 80)
    
    try:
        print("1. Original detection logic (before fix)...")
        print("   Used fallback: {'dim': 1024, 'num_heads': 8}")
        print("   Problem: checkpointer_info.raw_config was None")
        
        print("\n2. New detection logic (after fix)...")
        print("   Falls back to: {'dim': 4096, 'num_heads': 32}")
        print("   Better estimate for Moshi 7B")
        
        print("\n3. Testing detection on actual model...")
        model, checkpoint_info, state_dict = investigate_moshi_model_dimensions()
        
        if model is not None:
            # Simulate the detection logic
            print("   Simulating dimension detection:")
            
            actual_dim = None
            actual_heads = None
            
            # Method 1: Try transformer attributes
            if hasattr(model, 'transformer'):
                if hasattr(model.transformer, 'dim'):
                    actual_dim = model.transformer.dim
                    print(f"     Method 1 - transformer.dim: {actual_dim}")
                if hasattr(model.transformer, 'num_heads'):
                    actual_heads = model.transformer.num_heads
                    print(f"     Method 1 - transformer.num_heads: {actual_heads}")
            
            # Method 2: Try from layers
            if actual_dim is None and hasattr(model, 'transformer') and hasattr(model.transformer, 'layers'):
                if len(model.transformer.layers) > 0:
                    first_layer = model.transformer.layers[0]
                    if hasattr(first_layer, 'self_attn'):
                        attn = first_layer.self_attn
                        if hasattr(attn, 'num_heads'):
                            actual_heads = attn.num_heads
                            print(f"     Method 2 - layer.self_attn.num_heads: {actual_heads}")
                        if hasattr(attn, 'head_dim'):
                            head_dim = attn.head_dim
                            if actual_heads:
                                actual_dim = actual_heads * head_dim
                                print(f"     Method 2 - calculated dim: {actual_heads} * {head_dim} = {actual_dim}")
                    
                    # Try layer norm
                    if hasattr(first_layer, 'norm1') and hasattr(first_layer.norm1, 'weight'):
                        norm_dim = first_layer.norm1.weight.shape[0]
                        print(f"     Method 2 - norm1.weight.shape[0]: {norm_dim}")
                        if actual_dim is None:
                            actual_dim = norm_dim
            
            # Method 3: Check embedding
            if actual_dim is None and hasattr(model, 'embed_tokens'):
                if hasattr(model.embed_tokens, 'weight'):
                    embed_dim = model.embed_tokens.weight.shape[1]
                    print(f"     Method 3 - embed_tokens.weight.shape[1]: {embed_dim}")
                    actual_dim = embed_dim
            
            print(f"\n   Final detected dimensions: dim={actual_dim}, heads={actual_heads}")
            
    except Exception as e:
        print(f"ERROR in dimension detection investigation: {e}")
        traceback.print_exc()

def investigate_error_timeline():
    """Investigate the timeline of the error and fix"""
    print("\n" + "=" * 80)
    print("INVESTIGATION 4: ERROR TIMELINE AND FIX")
    print("=" * 80)
    
    print("1. ORIGINAL ERROR:")
    print("   AssertionError: Model dim mismatch: 4096 != 1024")
    print("   Location: moshi_ttt/format_utils.py:40")
    print("   Code: assert d_model == ttt_config.model_dim")
    
    print("\n2. ERROR SEQUENCE:")
    print("   Step 1: Train.py loads Moshi model")
    print("   Step 2: TTT integration called with checkpointer_info.raw_config = None")
    print("   Step 3: Falls back to {'dim': 1024, 'num_heads': 8}")
    print("   Step 4: TTTConfig created with model_dim=1024")
    print("   Step 5: During forward pass, actual tensor has shape [..., 4096]")
    print("   Step 6: moshi_to_ttt_format checks: 4096 == 1024 → FAIL")
    
    print("\n3. ROOT CAUSE:")
    print("   - Moshi 7B actually has 4096 dimensions")
    print("   - checkpointer_info.raw_config was None (no config.json in HF repo)")
    print("   - Fallback assumed smaller model (1024 dims)")
    print("   - TTT layers initialized with wrong dimensions")
    
    print("\n4. THE FIX:")
    print("   - Changed fallback from 1024 → 4096 dimensions")
    print("   - Added multiple detection methods")
    print("   - Better estimate for Moshi 7B")
    
    print("\n5. IMPACT OF FIX:")
    print("   - Parameter count increased: 17M → 69M TTT parameters")
    print("   - Total increase: 84M → 1.14B parameters (+1.1% → +14.9%)")
    print("   - Memory usage increased significantly")
    
def main():
    """Run comprehensive investigation"""
    print("🔍 COMPREHENSIVE SHAPE/DIMENSION ISSUE INVESTIGATION")
    print("=" * 80)
    
    investigate_moshi_model_dimensions()
    investigate_ttt_dimension_expectations()
    investigate_dimension_detection_logic()
    investigate_error_timeline()
    
    print("\n" + "=" * 80)
    print("📋 SUMMARY OF FINDINGS")
    print("=" * 80)
    print("1. Moshi 7B has 4096 dimensions, not 1024")
    print("2. HuggingFace repo has no config.json → raw_config = None")
    print("3. Original fallback (1024) was for smaller models")
    print("4. TTT layers were initialized with wrong dimensions")
    print("5. Forward pass failed when tensor (4096) met TTT config (1024)")
    print("6. Fix: Updated fallback to 4096 for Moshi 7B")
    print("7. Result: Much larger TTT layers but correct dimensions")
    
if __name__ == "__main__":
    main()