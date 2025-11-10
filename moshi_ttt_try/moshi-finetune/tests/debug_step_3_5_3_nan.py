#!/usr/bin/env python3
"""
Debug the NaN in output difference for Step 3.5.3
"""

import torch
import sys
import os

sys.path.append('/home/alufr/ttt_tests/moshi')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def debug_output_difference():
    """Debug why output difference is NaN"""
    print("🔍 Debugging Step 3.5.3 NaN output difference")
    
    try:
        from moshi.models import loaders
        from moshi_ttt.hybrid_layer import HybridStreamingTransformerLayer
        from moshi_ttt.config import TTTConfig
        
        # Create minimal model
        checkpoint_info = loaders.CheckpointInfo.from_hf_repo("kyutai/moshiko-pytorch-bf16")
        lm_config = loaders._lm_kwargs if checkpoint_info.raw_config is None else checkpoint_info.raw_config
        
        test_config = lm_config.copy()
        test_config['num_layers'] = 2
        test_config['dim'] = 256
        test_config['num_heads'] = 4
        
        # Original model
        original_model = loaders.get_moshi_lm(filename=None, lm_kwargs=test_config, device='cpu', dtype=torch.float32)
        
        # TTT model (clone to avoid modifying original)
        ttt_model = loaders.get_moshi_lm(filename=None, lm_kwargs=test_config, device='cpu', dtype=torch.float32)
        
        # Replace layers in TTT model
        ttt_config = TTTConfig(model_dim=test_config['dim'], num_heads=test_config['num_heads'], 
                              mini_batch_size=8, ttt_base_lr=0.1)
        
        for i, layer in enumerate(ttt_model.transformer.layers):
            ttt_model.transformer.layers[i] = HybridStreamingTransformerLayer(layer, ttt_config)
        
        # Test with same input
        codes = torch.randint(0, 20, (1, 9, 4), dtype=torch.int64)
        
        original_model.eval()
        ttt_model.eval()
        
        with torch.no_grad():
            orig_output = original_model(codes)
            ttt_output = ttt_model(codes)
        
        print(f"Original output logits shape: {orig_output.logits.shape}")
        print(f"TTT output logits shape: {ttt_output.logits.shape}")
        
        # Check for NaN in outputs
        orig_has_nan = torch.isnan(orig_output.logits).any().item()
        ttt_has_nan = torch.isnan(ttt_output.logits).any().item()
        
        print(f"Original has NaN: {orig_has_nan}")
        print(f"TTT has NaN: {ttt_has_nan}")
        
        if orig_has_nan or ttt_has_nan:
            print("⚠️ One or both outputs contain NaN - using finite-only comparison")
            
            orig_finite = orig_output.logits[torch.isfinite(orig_output.logits)]
            ttt_finite = ttt_output.logits[torch.isfinite(ttt_output.logits)]
            
            if len(orig_finite) > 0 and len(ttt_finite) > 0:
                # Compare means of finite values
                orig_mean = orig_finite.mean().item()
                ttt_mean = ttt_finite.mean().item()
                difference = abs(orig_mean - ttt_mean)
                print(f"Finite mean difference: {difference:.6f}")
                
                # Or compare finite value counts
                orig_finite_count = torch.isfinite(orig_output.logits).sum().item()
                ttt_finite_count = torch.isfinite(ttt_output.logits).sum().item()
                finite_count_diff = abs(orig_finite_count - ttt_finite_count)
                print(f"Finite count difference: {finite_count_diff}")
                
                # Success if either shows difference
                output_changed = difference > 0.01 or finite_count_diff > 100
            else:
                print("❌ No finite values to compare")
                output_changed = False
        else:
            # Normal L2 norm comparison
            difference = torch.norm(ttt_output.logits - orig_output.logits).item()
            print(f"L2 norm difference: {difference}")
            output_changed = difference > 10
        
        print(f"Output changed (TTT active): {output_changed}")
        
        return True
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_output_difference()