#!/usr/bin/env python3
"""
TEST: Does disabling checkpointing fix TTT gradient flow?
Following TTT-Video-DiT's approach of disabling checkpointing during eval.
"""

import torch
import torch.nn as nn
import sys
import os

sys.path.append('/home/alufr/ttt_tests/moshi')
sys.path.insert(0, '.')

def test_checkpoint_fix():
    """Test if disabling checkpointing fixes TTT parameter gradient flow"""
    print("🧪 TESTING CHECKPOINT FIX")
    print("=" * 60)
    
    try:
        from moshi.modules.transformer import StreamingTransformerLayer
        from moshi_ttt.hybrid_layer import HybridStreamingTransformerLayer
        from moshi_ttt.config import TTTConfig
        
        # Test setup
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
        
        print(f"✅ Created layers and config")
        
        # TEST 1: Current implementation (checkpointing enabled)
        print(f"\n🔍 TEST 1: CURRENT IMPLEMENTATION (WITH CHECKPOINTING)")
        print("-" * 50)
        
        try:
            hybrid_layer_with_checkpoint = HybridStreamingTransformerLayer(original_layer, ttt_config)
            hybrid_layer_with_checkpoint.train()
            
            x = torch.randn(batch_size, seq_len, d_model) * 0.1
            
            # Check current checkpoint_group_size calculation
            seq_modeling_block = hybrid_layer_with_checkpoint.seq_modeling_block
            
            # Simulate the calculation from hybrid_layer.py line 261
            from moshi_ttt.format_utils import moshi_to_ttt_format
            x_ttt, conversion_metadata = moshi_to_ttt_format(x, ttt_config)
            B, H, NC, C, HD = x_ttt.shape
            current_checkpoint_size = min(max(1, NC), NC)
            
            print(f"   TTT format shape: {x_ttt.shape}")
            print(f"   NC (num chunks): {NC}")
            print(f"   Current checkpoint_group_size: {current_checkpoint_size}")
            
            # Try forward pass
            print("   🔄 Forward pass with current checkpointing...")
            output = hybrid_layer_with_checkpoint(x)
            print("   ✅ Forward pass succeeded!")
            
            # Test backward pass
            target = torch.zeros_like(output)
            loss = nn.functional.mse_loss(output, target)
            print(f"   Loss: {loss.item():.6f}")
            
            print("   ⬅️  Backward pass...")
            loss.backward()
            print("   ✅ Backward pass succeeded!")
            
            # Check gradients
            ttt_params_with_grad = 0
            gating_params_with_grad = 0
            
            for name, param in hybrid_layer_with_checkpoint.named_parameters():
                if param.requires_grad and param.grad is not None:
                    if any(x in name.lower() for x in ['ttt', 'W1', 'W2', 'b1', 'b2', 'learnable_ttt']):
                        ttt_params_with_grad += 1
                    elif 'gating_alpha' in name:
                        gating_params_with_grad += 1
            
            print(f"   TTT parameters with gradients: {ttt_params_with_grad}")
            print(f"   Gating parameters with gradients: {gating_params_with_grad}")
            
            test1_success = True
            test1_ttt_grads = ttt_params_with_grad
            test1_gating_grads = gating_params_with_grad
            
        except Exception as e:
            print(f"   ❌ Test 1 failed: {e}")
            test1_success = False
            test1_ttt_grads = 0
            test1_gating_grads = 0
        
        # TEST 2: Modified implementation (checkpointing disabled)
        print(f"\n🔍 TEST 2: MODIFIED IMPLEMENTATION (NO CHECKPOINTING)")
        print("-" * 50)
        
        try:
            # Create a fresh hybrid layer
            original_layer2 = StreamingTransformerLayer(
                d_model=d_model,
                num_heads=num_heads,
                dim_feedforward=d_model * 4,
                causal=True,
                context=100,
                norm='rms_norm'
            )
            
            hybrid_layer_no_checkpoint = HybridStreamingTransformerLayer(original_layer2, ttt_config)
            hybrid_layer_no_checkpoint.train()
            
            # MODIFY: Disable checkpointing by patching the checkpoint_group_size
            # We'll monkey-patch the _apply_ttt_processing method
            original_apply_ttt = hybrid_layer_no_checkpoint.seq_modeling_block._apply_ttt_processing
            
            def patched_apply_ttt(self, x_ttt, x_original):
                # Copy the original method but with checkpoint_group_size = 0
                B, H, NC, C, HD = x_ttt.shape
                
                # Get Q, K, V projections from original input
                XQ = self.wq(x_original)
                XK = self.wk(x_original) 
                XV = self.wv(x_original)
                
                # Reshape projections
                seq_len = x_original.shape[1]
                XQ = XQ.view(B, seq_len, H, HD)
                XK = XK.view(B, seq_len, H, HD)
                XV = XV.view(B, seq_len, H, HD)
                
                # Apply L2 normalization
                XQ = torch.nn.functional.normalize(XQ, p=2, dim=-1)
                XK = torch.nn.functional.normalize(XK, p=2, dim=-1)
                
                # Apply layer norm reconstruction target
                XV = self.ln_reconstruction_target(XV, XK)
                
                # Convert to TTT format
                from moshi_ttt.format_utils import moshi_to_ttt_format
                XQ_ttt, _ = moshi_to_ttt_format(XQ.view(B, seq_len, H*HD), self.ttt_config)
                XK_ttt, _ = moshi_to_ttt_format(XK.view(B, seq_len, H*HD), self.ttt_config)
                XV_ttt, _ = moshi_to_ttt_format(XV.view(B, seq_len, H*HD), self.ttt_config)
                
                # Reshape to separate heads properly
                XQ_ttt = XQ_ttt.view(B, H, NC, C, HD)
                XK_ttt = XK_ttt.view(B, H, NC, C, HD) 
                XV_ttt = XV_ttt.view(B, H, NC, C, HD)
                
                # Compute TTT learning rate
                padded_len = NC * C
                x_padded = x_original
                if x_original.shape[1] < padded_len:
                    padding = torch.zeros(B, padded_len - seq_len, self.ttt_config.model_dim, 
                                        device=x_original.device, dtype=x_original.dtype)
                    x_padded = torch.cat([x_original, padding], dim=1)
                    
                x_chunked = x_padded.view(B, NC, C, self.ttt_config.model_dim)
                ttt_lr_eta = self.get_eta(x_chunked)
                eta = 1.0 / C * ttt_lr_eta
                
                # Prepare TTT-MLP parameters
                W1_states = self.W1.unsqueeze(0).expand(B, -1, -1, -1)
                b1_states = self.b1.unsqueeze(0).expand(B, -1, -1, -1)
                W2_states = self.W2.unsqueeze(0).expand(B, -1, -1, -1)
                b2_states = self.b2.unsqueeze(0).expand(B, -1, -1, -1)
                
                # CRITICAL CHANGE: Set checkpoint_group_size = 0 (disable checkpointing)
                checkpoint_group_size = 0  # ← This is the fix!
                
                print(f"       Using checkpoint_group_size: {checkpoint_group_size} (DISABLED)")
                
                # Apply TTT-MLP processing
                from moshi_ttt.models.ssm.ops.ttt_mlp import ttt_mlp
                XQW_batch = ttt_mlp(
                    XK_ttt, XQ_ttt, XV_ttt, eta,
                    self.ttt_norm_weight, self.ttt_norm_bias,
                    W1_states, b1_states, W2_states, b2_states,
                    checkpoint_group_size  # 0 = no checkpointing
                )
                
                # Reshape output back to expected format
                L = NC * C
                XQW_batch = XQW_batch.reshape(B, L, self.ttt_config.model_dim)
                
                # Apply post normalization and output projection
                XQW_batch = self.post_norm(XQW_batch)
                XQW_batch = self.wo(XQW_batch)
                
                # Remove padding if we added any
                if seq_len < padded_len:
                    XQW_batch = XQW_batch[:, :seq_len, :]
                
                return XQW_batch
            
            # Apply the patch
            import types
            hybrid_layer_no_checkpoint.seq_modeling_block._apply_ttt_processing = types.MethodType(
                patched_apply_ttt, hybrid_layer_no_checkpoint.seq_modeling_block
            )
            
            print(f"   ✅ Patched to disable checkpointing")
            
            x = torch.randn(batch_size, seq_len, d_model) * 0.1
            
            # Try forward pass
            print("   🔄 Forward pass without checkpointing...")
            output = hybrid_layer_no_checkpoint(x)
            print("   ✅ Forward pass succeeded!")
            
            # Test backward pass
            target = torch.zeros_like(output)
            loss = nn.functional.mse_loss(output, target)
            print(f"   Loss: {loss.item():.6f}")
            
            print("   ⬅️  Backward pass...")
            loss.backward()
            print("   ✅ Backward pass succeeded!")
            
            # Check gradients
            ttt_params_with_grad = 0
            gating_params_with_grad = 0
            total_grad_norm = 0.0
            
            for name, param in hybrid_layer_no_checkpoint.named_parameters():
                if param.requires_grad and param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    total_grad_norm += grad_norm
                    
                    if any(x in name.lower() for x in ['ttt', 'W1', 'W2', 'b1', 'b2', 'learnable_ttt']):
                        ttt_params_with_grad += 1
                        print(f"       ✅ TTT param '{name}': grad_norm={grad_norm:.6f}")
                    elif 'gating_alpha' in name:
                        gating_params_with_grad += 1
                        print(f"       ✅ Gating param '{name}': grad_norm={grad_norm:.6f}")
            
            print(f"   TTT parameters with gradients: {ttt_params_with_grad}")
            print(f"   Gating parameters with gradients: {gating_params_with_grad}")
            print(f"   Total gradient norm: {total_grad_norm:.6f}")
            
            test2_success = True
            test2_ttt_grads = ttt_params_with_grad
            test2_gating_grads = gating_params_with_grad
            
        except Exception as e:
            print(f"   ❌ Test 2 failed: {e}")
            import traceback
            traceback.print_exc()
            test2_success = False
            test2_ttt_grads = 0
            test2_gating_grads = 0
        
        # RESULTS COMPARISON
        print(f"\n🎯 RESULTS COMPARISON")
        print("=" * 60)
        
        print(f"TEST 1 (WITH CHECKPOINTING):")
        print(f"   Success: {test1_success}")
        print(f"   TTT params with gradients: {test1_ttt_grads}")
        print(f"   Gating params with gradients: {test1_gating_grads}")
        
        print(f"\nTEST 2 (NO CHECKPOINTING):")
        print(f"   Success: {test2_success}")
        print(f"   TTT params with gradients: {test2_ttt_grads}")
        print(f"   Gating params with gradients: {test2_gating_grads}")
        
        # CONCLUSION
        print(f"\n🏆 CONCLUSION:")
        print("-" * 40)
        
        if not test1_success and test2_success:
            print("✅ CHECKPOINTING WAS THE PROBLEM!")
            print("   - With checkpointing: Forward pass fails")
            print("   - Without checkpointing: Everything works")
            print("   → Solution: Disable checkpointing like TTT-Video-DiT")
            
        elif test1_success and test2_success:
            if test2_ttt_grads > test1_ttt_grads or test2_gating_grads > test1_gating_grads:
                print("✅ CHECKPOINTING WAS BLOCKING GRADIENTS!")
                print(f"   - More TTT gradients without checkpointing: {test2_ttt_grads} vs {test1_ttt_grads}")
                print(f"   - More gating gradients without checkpointing: {test2_gating_grads} vs {test1_gating_grads}")
                print("   → Solution: Disable checkpointing for better gradient flow")
            else:
                print("🤔 Both tests work equally well")
                print("   Checkpointing might not be the main issue")
                
        elif test1_success and not test2_success:
            print("❌ Disabling checkpointing made things worse")
            print("   The problem is elsewhere")
            
        else:
            print("❌ Both tests failed - deeper issue exists")
        
        return {
            'test1_success': test1_success,
            'test1_ttt_grads': test1_ttt_grads,
            'test1_gating_grads': test1_gating_grads,
            'test2_success': test2_success,
            'test2_ttt_grads': test2_ttt_grads,
            'test2_gating_grads': test2_gating_grads
        }
        
    except Exception as e:
        print(f"❌ Test setup failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = test_checkpoint_fix()
    if results:
        print(f"\n📊 SUMMARY:")
        print(f"Checkpointing issue confirmed: {results['test2_ttt_grads'] > results['test1_ttt_grads']}")