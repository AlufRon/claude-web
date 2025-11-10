#!/usr/bin/env python3
"""
Debug why TTT parameters freeze even with checkpoint fix working
Check if TTT layers are actually being executed during training
"""

import torch
import sys
import os

sys.path.append('/home/alufr/ttt_tests/moshi')
sys.path.insert(0, '.')

def add_execution_hooks():
    """Add hooks to detect if TTT methods are being called"""
    print("🔍 ADDING EXECUTION MONITORING HOOKS")
    print("=" * 60)
    
    # Global counters
    execution_stats = {
        'ttt_forward_calls': 0,
        'apply_ttt_calls': 0,
        'gating_forward_calls': 0,
        'ttt_mlp_calls': 0
    }
    
    try:
        # Hook 1: TTT forward method
        from moshi_ttt.hybrid_layer import HybridSeqModelingBlock
        original_ttt_forward = HybridSeqModelingBlock._ttt_forward
        
        def monitored_ttt_forward(self, x):
            execution_stats['ttt_forward_calls'] += 1
            print(f"   🎯 _ttt_forward called (#{execution_stats['ttt_forward_calls']})")
            return original_ttt_forward(self, x)
        
        HybridSeqModelingBlock._ttt_forward = monitored_ttt_forward
        print("✅ Hooked _ttt_forward")
        
        # Hook 2: Apply TTT processing method
        original_apply_ttt = HybridSeqModelingBlock._apply_ttt_processing
        
        def monitored_apply_ttt(self, x_ttt, x_original):
            execution_stats['apply_ttt_calls'] += 1
            print(f"   🎯 _apply_ttt_processing called (#{execution_stats['apply_ttt_calls']})")
            return original_apply_ttt(self, x_ttt, x_original)
        
        HybridSeqModelingBlock._apply_ttt_processing = monitored_apply_ttt
        print("✅ Hooked _apply_ttt_processing")
        
        # Hook 3: SSM Gating forward
        from moshi_ttt.ssm_gating import SSMGating
        original_gating_forward = SSMGating.forward
        
        def monitored_gating_forward(self, x):
            execution_stats['gating_forward_calls'] += 1
            print(f"   🎯 SSMGating.forward called (#{execution_stats['gating_forward_calls']})")
            print(f"       gating_alpha value: {self.gating_alpha.data.mean().item():.6f}")
            result = original_gating_forward(self, x)
            print(f"       gating applied: input_norm={x.norm().item():.4f} -> output_norm={result.norm().item():.4f}")
            return result
        
        SSMGating.forward = monitored_gating_forward
        print("✅ Hooked SSMGating.forward")
        
        # Hook 4: TTT MLP calls
        from moshi_ttt.models.ssm.ops.ttt_mlp import ttt_mlp
        original_ttt_mlp = ttt_mlp
        
        def monitored_ttt_mlp(*args, **kwargs):
            execution_stats['ttt_mlp_calls'] += 1
            print(f"   🎯 ttt_mlp called (#{execution_stats['ttt_mlp_calls']})")
            if len(args) >= 11:
                checkpoint_size = args[10]
                print(f"       checkpoint_group_size: {checkpoint_size}")
            return original_ttt_mlp(*args, **kwargs)
        
        # Replace the function in the module
        import moshi_ttt.models.ssm.ops.ttt_mlp as ttt_mlp_module
        ttt_mlp_module.ttt_mlp = monitored_ttt_mlp
        print("✅ Hooked ttt_mlp")
        
        return execution_stats
        
    except Exception as e:
        print(f"❌ Failed to add hooks: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_with_monitoring():
    """Test TTT execution with monitoring hooks"""
    print(f"\n🔍 TESTING TTT EXECUTION WITH MONITORING")
    print("-" * 50)
    
    # Add hooks
    execution_stats = add_execution_hooks()
    if not execution_stats:
        return None
    
    try:
        from moshi.modules.transformer import StreamingTransformerLayer
        from moshi_ttt.hybrid_layer import HybridStreamingTransformerLayer
        from moshi_ttt.config import TTTConfig
        
        # Small test setup
        d_model = 512
        num_heads = 8
        batch_size = 1
        seq_len = 32
        
        print(f"Creating test model (d_model={d_model}, heads={num_heads})")
        
        original_layer = StreamingTransformerLayer(
            d_model=d_model,
            num_heads=num_heads,
            dim_feedforward=d_model * 4,
            causal=True,
            context=100,
            norm='rms_norm'
        )
        
        ttt_config = TTTConfig(
            model_dim=d_model,
            num_heads=num_heads,
            mini_batch_size=16,
            ttt_base_lr=0.1
        )
        
        hybrid_layer = HybridStreamingTransformerLayer(original_layer, ttt_config)
        hybrid_layer.train()
        
        print("✅ Created hybrid layer")
        
        # Test forward pass
        x = torch.randn(batch_size, seq_len, d_model) * 0.1
        
        print(f"\n🔄 FORWARD PASS MONITORING:")
        print("-" * 30)
        
        # Check gating alpha before
        for name, param in hybrid_layer.named_parameters():
            if 'gating_alpha' in name:
                print(f"Before: {name} = {param.data.mean().item():.6f}")
        
        # Forward pass
        output = hybrid_layer(x)
        
        print(f"\n📊 EXECUTION STATISTICS:")
        print("-" * 30)
        for key, count in execution_stats.items():
            print(f"{key}: {count}")
        
        # Create loss and backward pass
        target = torch.zeros_like(output)
        loss = torch.nn.functional.mse_loss(output, target)
        
        print(f"\n⬅️  BACKWARD PASS:")
        print("-" * 30)
        print(f"Loss: {loss.item():.6f}")
        
        loss.backward()
        print("✅ Backward completed")
        
        # Check gradients
        print(f"\n📈 GRADIENT ANALYSIS:")
        print("-" * 30)
        
        for name, param in hybrid_layer.named_parameters():
            if 'gating_alpha' in name and param.grad is not None:
                print(f"Gradient: {name} = {param.grad.norm().item():.6f}")
        
        # Check if ANY TTT method was called
        total_ttt_calls = sum(execution_stats.values())
        
        print(f"\n🎯 DIAGNOSIS:")
        print("-" * 30)
        
        if total_ttt_calls == 0:
            print("❌ NO TTT methods were called!")
            print("   → TTT layers are not being executed at all")
            print("   → Check if hybrid layers are properly integrated")
        elif execution_stats['ttt_mlp_calls'] == 0:
            print("❌ TTT methods called but ttt_mlp never executed!")
            print("   → TTT forward fails before reaching ttt_mlp")
        else:
            print("✅ TTT methods are being called")
            print("   → Problem is elsewhere (gradients, parameter updates, etc.)")
        
        return execution_stats
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print("🧪 TTT EXECUTION MONITORING")
    print("=" * 60)
    
    execution_stats = test_with_monitoring()
    
    if execution_stats:
        total_calls = sum(execution_stats.values())
        print(f"\n🎯 FINAL SUMMARY:")
        print(f"Total TTT method calls: {total_calls}")
        
        if total_calls == 0:
            print("🚨 CRITICAL: TTT layers are not executing!")
            print("   Root cause: Integration problem, not checkpoint issue")
        else:
            print("✅ TTT layers are executing")
            print("   Root cause: Must be gradient flow or parameter update issue")
    
    return execution_stats

if __name__ == "__main__":
    results = main()