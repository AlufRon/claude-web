#!/usr/bin/env python3
"""
CRITICAL GRADIENT FLOW DIAGNOSTIC
Check if TTT parameters are receiving gradients during training
"""

import torch
import torch.nn as nn
import sys
import os

sys.path.append('/home/alufr/ttt_tests/moshi')
sys.path.insert(0, '.')

def test_gradient_flow():
    """Check if TTT parameters receive gradients during training"""
    print("🔍 CRITICAL GRADIENT FLOW DIAGNOSTIC")
    print("=" * 60)
    
    try:
        from moshi.modules.transformer import StreamingTransformerLayer
        from moshi_ttt.hybrid_layer import HybridStreamingTransformerLayer
        from moshi_ttt.config import TTTConfig
        
        # Create test setup
        d_model = 512
        num_heads = 8
        batch_size = 2
        seq_len = 32
        
        print(f"📦 Test setup: d_model={d_model}, heads={num_heads}, batch={batch_size}, seq={seq_len}")
        
        # Create original layer
        original_layer = StreamingTransformerLayer(
            d_model=d_model,
            num_heads=num_heads,
            dim_feedforward=d_model * 4,
            causal=True,
            context=100,
            norm='rms_norm'
        )
        
        # Create TTT config
        ttt_config = TTTConfig(
            model_dim=d_model,
            num_heads=num_heads,
            mini_batch_size=16,
            ttt_base_lr=0.1
        )
        
        # Create hybrid layer
        hybrid_layer = HybridStreamingTransformerLayer(original_layer, ttt_config)
        hybrid_layer.train()
        
        print(f"✅ Created hybrid layer with TTT")
        
        # 1. CHECK: Are TTT parameters in the model?
        print(f"\n🔍 STEP 1: PARAMETER INVENTORY")
        print("-" * 40)
        
        total_params = 0
        moshi_params = 0
        ttt_params = 0
        gating_params = 0
        
        print("Parameters in hybrid layer:")
        for name, param in hybrid_layer.named_parameters():
            total_params += 1
            param_count = param.numel()
            print(f"   {name}: {param.shape} ({param_count:,} params, requires_grad={param.requires_grad})")
            
            if any(x in name.lower() for x in ['ttt', 'W1', 'W2', 'b1', 'b2', 'learnable_ttt']):
                ttt_params += 1
            elif 'gating_alpha' in name:
                gating_params += 1
            else:
                moshi_params += 1
        
        print(f"\nParameter summary:")
        print(f"   Total parameters: {total_params}")
        print(f"   Moshi parameters: {moshi_params}")
        print(f"   TTT parameters: {ttt_params}")
        print(f"   Gating parameters: {gating_params}")
        
        # 2. CHECK: Forward pass and gradient computation
        print(f"\n🔍 STEP 2: FORWARD + BACKWARD PASS")
        print("-" * 40)
        
        # Create input
        x = torch.randn(batch_size, seq_len, d_model) * 0.1
        print(f"Input shape: {x.shape}")
        
        # Forward pass
        print("🔄 Forward pass...")
        output = hybrid_layer(x)
        print(f"Output shape: {output.shape}")
        print(f"Output finite: {torch.isfinite(output).float().mean().item():.1%}")
        
        # Create loss
        target = torch.zeros_like(output)
        loss = nn.functional.mse_loss(output, target)
        print(f"Loss: {loss.item():.6f}")
        
        # Backward pass
        print("⬅️  Backward pass...")
        loss.backward()
        print("✅ Backward completed")
        
        # 3. CHECK: Which parameters received gradients?
        print(f"\n🔍 STEP 3: GRADIENT ANALYSIS")
        print("-" * 40)
        
        params_with_grad = 0
        ttt_params_with_grad = 0
        gating_params_with_grad = 0
        moshi_params_with_grad = 0
        
        max_grad_norm = 0.0
        min_grad_norm = float('inf')
        
        print("Gradient status:")
        for name, param in hybrid_layer.named_parameters():
            if param.requires_grad:
                has_grad = param.grad is not None
                
                if has_grad:
                    params_with_grad += 1
                    grad_norm = param.grad.norm().item()
                    max_grad_norm = max(max_grad_norm, grad_norm)
                    min_grad_norm = min(min_grad_norm, grad_norm)
                    
                    # Categorize parameter
                    if any(x in name.lower() for x in ['ttt', 'W1', 'W2', 'b1', 'b2', 'learnable_ttt']):
                        ttt_params_with_grad += 1
                        param_type = "TTT"
                    elif 'gating_alpha' in name:
                        gating_params_with_grad += 1
                        param_type = "GATING"
                    else:
                        moshi_params_with_grad += 1
                        param_type = "MOSHI"
                    
                    print(f"   ✅ {name}: grad_norm={grad_norm:.6f} ({param_type})")
                else:
                    param_type = "UNKNOWN"
                    if any(x in name.lower() for x in ['ttt', 'W1', 'W2', 'b1', 'b2', 'learnable_ttt']):
                        param_type = "TTT"
                    elif 'gating_alpha' in name:
                        param_type = "GATING"
                    else:
                        param_type = "MOSHI"
                    
                    print(f"   ❌ {name}: NO GRADIENT ({param_type})")
        
        # 4. SUMMARY AND DIAGNOSIS
        print(f"\n🎯 GRADIENT FLOW SUMMARY")
        print("-" * 40)
        
        grad_ratio = params_with_grad / total_params if total_params > 0 else 0
        
        print(f"Parameters with gradients: {params_with_grad}/{total_params} ({grad_ratio:.1%})")
        print(f"Moshi params with gradients: {moshi_params_with_grad}/{moshi_params}")
        print(f"TTT params with gradients: {ttt_params_with_grad}/{ttt_params}")
        print(f"Gating params with gradients: {gating_params_with_grad}/{gating_params}")
        
        if params_with_grad > 0:
            print(f"Gradient magnitude range: {min_grad_norm:.6f} to {max_grad_norm:.6f}")
        
        # 5. DIAGNOSIS
        print(f"\n🚨 DIAGNOSIS:")
        print("-" * 40)
        
        if ttt_params_with_grad == 0:
            print("❌ PROBLEM CONFIRMED: TTT parameters are NOT receiving gradients!")
            print("   This explains why ttt_alpha stays constant.")
            print("   Possible causes:")
            print("   1. TTT forward pass is not being called")
            print("   2. TTT output is detached from computational graph")
            print("   3. TTT layers are not properly connected to loss")
            
        elif gating_params_with_grad == 0:
            print("❌ PROBLEM CONFIRMED: Gating parameters are NOT receiving gradients!")
            print("   This explains why ttt_alpha stays constant.")
            
        else:
            print("✅ TTT parameters ARE receiving gradients")
            print("   Problem might be elsewhere (learning rate, parameter updates, etc.)")
        
        # 6. CHECK: TTT forward pass execution
        print(f"\n🔍 STEP 4: TTT EXECUTION CHECK")
        print("-" * 40)
        
        # Add hooks to see if TTT methods are called
        ttt_forward_called = False
        apply_ttt_called = False
        
        def hook_ttt_forward(self, input, output):
            nonlocal ttt_forward_called
            ttt_forward_called = True
            print("   ✅ _ttt_forward was called")
        
        def hook_apply_ttt(self, input, output):
            nonlocal apply_ttt_called
            apply_ttt_called = True
            print("   ✅ _apply_ttt_processing was called")
        
        # Register hooks
        if hasattr(hybrid_layer.seq_modeling_block, '_ttt_forward'):
            hybrid_layer.seq_modeling_block._ttt_forward = \
                hybrid_layer.seq_modeling_block._ttt_forward.__func__.__get__(hybrid_layer.seq_modeling_block, type(hybrid_layer.seq_modeling_block))
        
        # Run another forward pass to check execution
        print("🔄 Forward pass with execution monitoring...")
        with torch.no_grad():
            _ = hybrid_layer(x)
        
        print(f"TTT forward called: {ttt_forward_called}")
        print(f"Apply TTT called: {apply_ttt_called}")
        
        return {
            'total_params': total_params,
            'params_with_grad': params_with_grad,
            'ttt_params_with_grad': ttt_params_with_grad,
            'gating_params_with_grad': gating_params_with_grad,
            'grad_ratio': grad_ratio,
            'max_grad_norm': max_grad_norm,
            'ttt_forward_called': ttt_forward_called
        }
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = test_gradient_flow()
    if results:
        print(f"\n🏁 TEST COMPLETED")
        print(f"Key finding: {results['ttt_params_with_grad']}/{results.get('ttt_params', '?')} TTT parameters received gradients")