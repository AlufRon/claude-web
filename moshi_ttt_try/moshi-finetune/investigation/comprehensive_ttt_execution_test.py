#!/usr/bin/env python3
"""
Comprehensive TTT execution test with fixed initialization
"""

import torch
import torch.nn as nn
import sys
import os

sys.path.append('/home/alufr/ttt_tests/moshi')
sys.path.insert(0, '.')

def test_complete_ttt_execution():
    """Test complete TTT execution with fixed initialization"""
    print("🧪 COMPREHENSIVE TTT EXECUTION TEST")
    print("=" * 60)
    
    # Global execution counter
    execution_stats = {
        'ttt_forward_calls': 0,
        'apply_ttt_calls': 0,
        'ttt_mlp_calls': 0,
        'gating_calls': 0
    }
    
    try:
        from moshi.modules.transformer import StreamingTransformerLayer
        from moshi_ttt.hybrid_layer import HybridStreamingTransformerLayer, HybridSeqModelingBlock
        from moshi_ttt.config import TTTConfig
        from moshi_ttt.ssm_gating import SSMGating
        
        # Hook all TTT methods
        print("1. Installing execution hooks...")
        
        # Hook TTT forward
        original_ttt_forward = HybridSeqModelingBlock._ttt_forward
        def hooked_ttt_forward(self, x):
            execution_stats['ttt_forward_calls'] += 1
            print(f"   🎯 _ttt_forward called (#{execution_stats['ttt_forward_calls']})")
            return original_ttt_forward(self, x)
        HybridSeqModelingBlock._ttt_forward = hooked_ttt_forward
        
        # Hook apply TTT processing
        original_apply_ttt = HybridSeqModelingBlock._apply_ttt_processing
        def hooked_apply_ttt(self, x_ttt, x_original):
            execution_stats['apply_ttt_calls'] += 1
            print(f"   🎯 _apply_ttt_processing called (#{execution_stats['apply_ttt_calls']})")
            result = original_apply_ttt(self, x_ttt, x_original)
            print(f"   ✅ _apply_ttt_processing completed")
            return result
        HybridSeqModelingBlock._apply_ttt_processing = hooked_apply_ttt
        
        # Hook TTT MLP
        from moshi_ttt.models.ssm.ops.ttt_mlp import ttt_mlp
        original_ttt_mlp = ttt_mlp
        def hooked_ttt_mlp(*args, **kwargs):
            execution_stats['ttt_mlp_calls'] += 1
            print(f"   🎯 ttt_mlp called (#{execution_stats['ttt_mlp_calls']})")
            if len(args) >= 11:
                checkpoint_size = args[10]
                print(f"       checkpoint_group_size: {checkpoint_size}")
            result = original_ttt_mlp(*args, **kwargs)
            print(f"   ✅ ttt_mlp completed successfully")
            return result
        
        # Replace in module
        import moshi_ttt.models.ssm.ops.ttt_mlp as ttt_mlp_module
        ttt_mlp_module.ttt_mlp = hooked_ttt_mlp
        
        # Hook gating
        original_gating_forward = SSMGating.forward
        def hooked_gating_forward(self, x):
            execution_stats['gating_calls'] += 1
            print(f"   🎯 SSMGating.forward called (#{execution_stats['gating_calls']})")
            print(f"       gating_alpha value: {self.gating_alpha.data.mean().item():.6f}")
            result = original_gating_forward(self, x)
            return result
        SSMGating.forward = hooked_gating_forward
        
        print("✅ All hooks installed")
        
        # Create test setup with realistic parameters
        print("\n2. Creating test setup...")
        d_model = 1024  # Real Moshi dimensions
        num_heads = 8
        batch_size = 1
        seq_len = 64
        
        original_layer = StreamingTransformerLayer(
            d_model=d_model,
            num_heads=num_heads,
            dim_feedforward=d_model * 4,
            causal=True,
            context=200,
            norm='rms_norm'
        )
        
        ttt_config = TTTConfig(
            model_dim=d_model,
            num_heads=num_heads,
            mini_batch_size=16,
            ttt_base_lr=1.0,
            gating_alpha_init=0.05
        )
        
        hybrid_layer = HybridStreamingTransformerLayer(original_layer, ttt_config)
        hybrid_layer.train()
        
        print("✅ Test setup created")
        
        # Check initial parameter values
        print("\n3. Checking initial parameter values...")
        initial_values = {}
        for name, param in hybrid_layer.named_parameters():
            if 'gating_alpha' in name:
                initial_values[name] = param.data.clone()
                print(f"   Initial {name}: {param.data.mean().item():.6f}")
        
        # Create optimizer
        optimizer = torch.optim.AdamW(hybrid_layer.parameters(), lr=1e-3)
        
        print("\n4. Forward pass with full monitoring...")
        
        # Create input
        x = torch.randn(batch_size, seq_len, d_model) * 0.1
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        output = hybrid_layer(x)
        print(f"   Output shape: {output.shape}")
        print(f"   Output finite: {torch.isfinite(output).all()}")
        
        # Create loss
        target = torch.zeros_like(output)
        loss = nn.functional.mse_loss(output, target)
        print(f"   Loss: {loss.item():.6f}")
        
        print("\n5. Backward pass...")
        loss.backward()
        print("✅ Backward completed")
        
        # Check gradients
        print("\n6. Gradient analysis...")
        ttt_grads = 0
        gating_grads = 0
        total_grad_norm = 0.0
        
        for name, param in hybrid_layer.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                total_grad_norm += grad_norm
                
                if any(x in name.lower() for x in ['ttt', 'W1', 'W2', 'b1', 'b2', 'learnable_ttt']):
                    ttt_grads += 1
                elif 'gating_alpha' in name:
                    gating_grads += 1
                    print(f"   Gating gradient {name}: {grad_norm:.6f}")
        
        print(f"   TTT parameters with gradients: {ttt_grads}")
        print(f"   Gating parameters with gradients: {gating_grads}")
        print(f"   Total gradient norm: {total_grad_norm:.6f}")
        
        print("\n7. Optimizer step...")
        optimizer.step()
        print("✅ Optimizer step completed")
        
        # Check parameter changes
        print("\n8. Parameter change analysis...")
        for name, initial_value in initial_values.items():
            current_param = dict(hybrid_layer.named_parameters())[name]
            current_value = current_param.data
            change = (current_value - initial_value).abs().max().item()
            print(f"   {name}: change = {change:.8f}")
        
        # Execution summary
        print(f"\n🎯 EXECUTION SUMMARY:")
        print("-" * 40)
        for key, count in execution_stats.items():
            print(f"{key}: {count}")
        
        total_calls = sum(execution_stats.values())
        
        if execution_stats['ttt_mlp_calls'] > 0:
            print("✅ TTT MLP is being called!")
            if gating_grads > 0:
                print("✅ Gating parameters have gradients!")
                return True
            else:
                print("❌ Gating parameters have no gradients")
                return False
        else:
            print("❌ TTT MLP is still not being called")
            return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🧪 COMPREHENSIVE TTT EXECUTION TEST")
    print("=" * 60)
    
    success = test_complete_ttt_execution()
    
    if success:
        print(f"\n🎉 SUCCESS! TTT execution is working properly")
        print(f"   The weight initialization fix resolved the issue")
    else:
        print(f"\n❌ TTT execution still has issues")
        print(f"   Need further investigation")

if __name__ == "__main__":
    main()