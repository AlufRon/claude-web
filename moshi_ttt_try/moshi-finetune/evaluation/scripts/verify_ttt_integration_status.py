#!/usr/bin/env python3
"""
Verify TTT integration status - check if TTT is actually enabled and working
"""

import torch
import sys
import os

sys.path.append('/home/alufr/ttt_tests/moshi')
sys.path.insert(0, '.')

def verify_integration_status():
    """Check TTT integration status in production training setup"""
    print("🔍 VERIFYING TTT INTEGRATION STATUS")
    print("=" * 60)
    
    try:
        # Check TTT configuration used in production
        print("1. Checking production TTT configuration...")
        
        # Simulate production args
        from argparse import Namespace
        import yaml
        
        # Load the actual config used by training
        config_path = '/home/alufr/ttt_tests/moshi-finetune/configs/dailytalk_ttt_config.yaml'
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"   Config file: {config_path}")
        print(f"   TTT enabled: {config.get('ttt', {}).get('enable', False)}")
        print(f"   TTT layers: {config.get('ttt', {}).get('layers', 'none')}")
        print(f"   TTT base_lr: {config.get('ttt', {}).get('base_lr', 'N/A')}")
        print(f"   TTT mini_batch_size: {config.get('ttt', {}).get('mini_batch_size', 'N/A')}")
        
        # Convert to args format
        args = Namespace()
        args.ttt = Namespace()
        args.ttt.enable = config.get('ttt', {}).get('enable', False)
        args.ttt.layers = config.get('ttt', {}).get('layers', 'none')
        args.ttt.base_lr = config.get('ttt', {}).get('base_lr', 1.0)
        args.ttt.mini_batch_size = config.get('ttt', {}).get('mini_batch_size', 16)
        
        if not args.ttt.enable:
            print("❌ CRITICAL: TTT is DISABLED in config!")
            print("   This explains why TTT parameters don't change")
            return False
        
        print("✅ TTT is enabled in config")
        
        # Check TTT layer specification
        print(f"\n2. Checking TTT layer specification...")
        
        from finetune.ttt_integration import parse_layer_specification
        
        # Mock model with typical Moshi layer count
        total_layers = 32  # Typical Moshi model
        
        try:
            layer_indices = parse_layer_specification(args.ttt.layers, total_layers)
            print(f"   Total model layers: {total_layers}")
            print(f"   TTT layer spec: '{args.ttt.layers}'")
            print(f"   Parsed layer indices: {layer_indices}")
            print(f"   Number of TTT layers: {len(layer_indices)}")
            
            if len(layer_indices) == 0:
                print("❌ CRITICAL: No TTT layers specified!")
                return False
            
            print("✅ TTT layers properly specified")
            
        except Exception as e:
            print(f"❌ CRITICAL: Invalid layer specification: {e}")
            return False
        
        # Test TTT config creation
        print(f"\n3. Testing TTT config creation...")
        
        try:
            from finetune.ttt_integration import create_ttt_config
            
            # Mock model config
            model_config = {
                'dim': 1024,
                'num_layers': 32,
                'num_heads': 8
            }
            
            ttt_config = create_ttt_config(args.ttt, model_config)
            
            print(f"   TTT config created successfully")
            print(f"   Model dim: {ttt_config.model_dim}")
            print(f"   Num heads: {ttt_config.num_heads}")
            print(f"   TTT base lr: {ttt_config.ttt_base_lr}")
            print(f"   Mini batch size: {ttt_config.mini_batch_size}")
            
            print("✅ TTT config creation works")
            
        except Exception as e:
            print(f"❌ CRITICAL: TTT config creation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Test hybrid layer creation
        print(f"\n4. Testing hybrid layer creation...")
        
        try:
            from moshi.modules.transformer import StreamingTransformerLayer
            from moshi_ttt.hybrid_layer import HybridStreamingTransformerLayer
            
            # Create a test layer
            original_layer = StreamingTransformerLayer(
                d_model=1024,
                num_heads=8,
                dim_feedforward=4096,
                causal=True,
                context=100,
                norm='rms_norm'
            )
            
            print(f"   Original layer type: {type(original_layer)}")
            print(f"   Is StreamingTransformerLayer: {isinstance(original_layer, StreamingTransformerLayer)}")
            
            # Create hybrid layer
            hybrid_layer = HybridStreamingTransformerLayer(original_layer, ttt_config)
            
            print(f"   Hybrid layer created: {type(hybrid_layer)}")
            
            # Check parameter count
            original_params = sum(p.numel() for p in original_layer.parameters())
            hybrid_params = sum(p.numel() for p in hybrid_layer.parameters())
            ttt_params = hybrid_params - original_params
            
            print(f"   Original parameters: {original_params:,}")
            print(f"   Hybrid parameters: {hybrid_params:,}")
            print(f"   TTT parameters added: {ttt_params:,}")
            
            if ttt_params == 0:
                print("❌ CRITICAL: No TTT parameters added!")
                return False
            
            print("✅ Hybrid layer creation works")
            
            # Check TTT parameter initialization
            print(f"\n5. Checking TTT parameter initialization...")
            
            gating_params = []
            ttt_weight_params = []
            
            for name, param in hybrid_layer.named_parameters():
                if 'gating_alpha' in name:
                    gating_params.append((name, param.data.mean().item()))
                elif any(x in name.lower() for x in ['ttt', 'W1', 'W2', 'learnable_ttt']):
                    ttt_weight_params.append((name, param.data.norm().item()))
            
            print(f"   Gating parameters:")
            for name, value in gating_params:
                print(f"     {name}: {value:.6f}")
                if abs(value - 0.05) > 0.01:  # Should be close to gating_alpha_init
                    print(f"     ⚠️  Unexpected gating value: {value}")
            
            print(f"   TTT weight parameters (first 5):")
            for name, norm in ttt_weight_params[:5]:
                print(f"     {name}: norm={norm:.6f}")
                if norm < 1e-6:
                    print(f"     ❌ Parameter seems uninitialized: {name}")
            
            print("✅ Parameter initialization check complete")
            
        except Exception as e:
            print(f"❌ CRITICAL: Hybrid layer creation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        print(f"\n🎯 INTEGRATION STATUS SUMMARY:")
        print("-" * 40)
        print("✅ TTT is enabled in config")
        print("✅ TTT layer specification is valid")
        print("✅ TTT config creation works")
        print("✅ Hybrid layer creation works")
        print("✅ TTT parameters are added and initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🧪 TTT INTEGRATION STATUS VERIFICATION")
    print("=" * 60)
    
    success = verify_integration_status()
    
    if success:
        print(f"\n✅ TTT integration appears to be working correctly")
        print(f"   The parameter freeze issue must be elsewhere")
    else:
        print(f"\n❌ TTT integration has critical issues")
        print(f"   Fix these issues before proceeding")

if __name__ == "__main__":
    main()