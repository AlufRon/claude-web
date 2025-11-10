#!/usr/bin/env python3
"""
CRITICAL: Debug why TTT parameters still not changing in production training
Despite fixing checkpointing, ttt_alpha remains constant at 0.050049
"""

import torch
import torch.nn as nn
import sys
import os

sys.path.append('/home/alufr/ttt_tests/moshi')
sys.path.insert(0, '.')

def test_production_setup():
    """Reproduce the exact production training setup to debug TTT parameter freeze"""
    print("🔍 DEBUGGING PRODUCTION TTT TRAINING")
    print("=" * 60)
    
    try:
        # Import the exact same modules as production training
        from finetune.ttt_integration import apply_ttt_to_model
        from moshi.models import loaders
        import torch
        from torch.optim import AdamW
        
        print("✅ Imported production modules")
        
        # TEST 1: Check if the fix was actually applied
        print(f"\n🔍 TEST 1: VERIFY CHECKPOINT FIX IS APPLIED")
        print("-" * 50)
        
        # Load a small model to test
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")
        
        # Create a minimal model setup like production
        print("Loading Moshi model...")
        
        # Use the same loading as production training
        lm_model = loaders.load_lm_model_only_from_checkpoint(
            '/home/alufr/.cache/huggingface/hub/models--kyutai--moshiko-pytorch-bf16/snapshots/29f67a516c1ff45b72ebde5509cd4ae3a9dfa8a8/moshiko-pytorch-bf16.bin'
        )
        lm_model = lm_model.to(device)
        lm_model.train()
        
        print(f"✅ Loaded model: {type(lm_model)}")
        print(f"Model device: {next(lm_model.parameters()).device}")
        
        # Apply TTT exactly like production
        print("Applying TTT integration...")
        
        # Use the same TTT config as production
        from argparse import Namespace
        args = Namespace()
        args.ttt = Namespace()
        args.ttt.enable = True
        args.ttt.layers = "middle"  # Same as production config
        args.ttt.base_lr = 1.0
        args.ttt.mini_batch_size = 16
        
        # Apply TTT
        apply_ttt_to_model(lm_model, args)
        
        print(f"✅ Applied TTT to model")
        
        # TEST 2: Check if checkpoint fix is actually in effect
        print(f"\n🔍 TEST 2: VERIFY CHECKPOINT_GROUP_SIZE = 0")
        print("-" * 50)
        
        checkpoint_sizes = []
        for name, module in lm_model.named_modules():
            if hasattr(module, 'seq_modeling_block'):
                # Check if we can access the checkpoint logic
                print(f"Found hybrid layer: {name}")
                
                # Check the source code of _apply_ttt_processing
                import inspect
                source = inspect.getsource(module.seq_modeling_block._apply_ttt_processing)
                if "checkpoint_group_size = 0" in source:
                    print("   ✅ Checkpoint fix is applied (checkpoint_group_size = 0)")
                else:
                    print("   ❌ Checkpoint fix NOT applied!")
                    print("   Current code snippet:")
                    lines = source.split('\n')
                    for i, line in enumerate(lines):
                        if 'checkpoint_group_size' in line:
                            print(f"   Line {i}: {line.strip()}")
        
        # TEST 3: Check parameter registration and gradient setup
        print(f"\n🔍 TEST 3: CHECK PARAMETER REGISTRATION")
        print("-" * 50)
        
        total_params = 0
        ttt_params = 0
        gating_params = 0
        
        print("Checking all model parameters:")
        for name, param in lm_model.named_parameters():
            total_params += 1
            
            if any(x in name.lower() for x in ['ttt', 'W1', 'W2', 'b1', 'b2', 'learnable_ttt']):
                ttt_params += 1
                print(f"   TTT: {name} - shape: {param.shape} - requires_grad: {param.requires_grad}")
            elif 'gating_alpha' in name:
                gating_params += 1
                print(f"   GATING: {name} - shape: {param.shape} - requires_grad: {param.requires_grad} - value: {param.data.mean().item():.6f}")
        
        print(f"\nParameter summary:")
        print(f"   Total: {total_params}")
        print(f"   TTT: {ttt_params}")  
        print(f"   Gating: {gating_params}")
        
        if ttt_params == 0:
            print("❌ PROBLEM: No TTT parameters found!")
            return None
        
        # TEST 4: Minimal forward/backward pass
        print(f"\n🔍 TEST 4: MINIMAL FORWARD/BACKWARD PASS")
        print("-" * 50)
        
        # Create a small input like production training
        batch_size = 1
        seq_len = 64  # Small sequence
        n_codebooks = 8
        
        # Create input codes [B, K, T] 
        codes = torch.randint(0, 1000, (batch_size, n_codebooks, seq_len), dtype=torch.int64, device=device)
        print(f"Input codes shape: {codes.shape}")
        
        # Create optimizer like production
        optimizer = AdamW(
            lm_model.parameters(),
            lr=5e-4,  # Higher LR for testing
            betas=(0.9, 0.95),
            eps=1e-08,
            weight_decay=0.1,
        )
        
        print("✅ Created optimizer")
        
        # Store initial gating alpha values
        initial_gating_values = {}
        for name, param in lm_model.named_parameters():
            if 'gating_alpha' in name:
                initial_gating_values[name] = param.data.clone()
                print(f"   Initial {name}: {param.data.mean().item():.6f}")
        
        print("\n🔄 Forward pass...")
        
        try:
            optimizer.zero_grad()
            
            # Forward pass
            output = lm_model(codes)
            print(f"✅ Forward pass succeeded")
            print(f"Output shapes: logits={output.logits.shape}, text_logits={output.text_logits.shape}")
            
            # Simple loss calculation
            # Text loss
            text_target = codes[:, :lm_model.audio_offset]  # First codebooks for text
            text_loss = nn.functional.cross_entropy(
                output.text_logits.reshape(-1, output.text_logits.size(-1)),
                text_target.reshape(-1)
            )
            
            # Audio loss  
            audio_target = codes[:, lm_model.audio_offset:lm_model.audio_offset + lm_model.dep_q]
            audio_loss = nn.functional.cross_entropy(
                output.logits.reshape(-1, output.logits.size(-1)),
                audio_target.reshape(-1)
            )
            
            total_loss = text_loss + audio_loss
            print(f"✅ Loss computed: total={total_loss.item():.6f}, text={text_loss.item():.6f}, audio={audio_loss.item():.6f}")
            
            print("\n⬅️  Backward pass...")
            total_loss.backward()
            print("✅ Backward pass succeeded")
            
            print("\n📊 Gradient analysis...")
            
            # Check gradients
            ttt_grads = 0
            gating_grads = 0
            max_grad = 0.0
            
            for name, param in lm_model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    max_grad = max(max_grad, grad_norm)
                    
                    if any(x in name.lower() for x in ['ttt', 'W1', 'W2', 'b1', 'b2', 'learnable_ttt']):
                        ttt_grads += 1
                        print(f"   TTT grad: {name} - norm: {grad_norm:.6f}")
                    elif 'gating_alpha' in name:
                        gating_grads += 1
                        print(f"   GATING grad: {name} - norm: {grad_norm:.6f}")
            
            print(f"\nGradient summary:")
            print(f"   TTT params with gradients: {ttt_grads}")
            print(f"   Gating params with gradients: {gating_grads}")
            print(f"   Max gradient norm: {max_grad:.6f}")
            
            if gating_grads == 0:
                print("❌ PROBLEM: Gating parameters have no gradients!")
                
                # Debug: Check if gating layers are being called
                print("\n🔍 Checking gating layer usage...")
                for name, module in lm_model.named_modules():
                    if 'gating' in name.lower():
                        print(f"   Found gating module: {name} - {type(module)}")
                        if hasattr(module, 'gating_alpha'):
                            print(f"      gating_alpha shape: {module.gating_alpha.shape}")
                            print(f"      gating_alpha value: {module.gating_alpha.data.mean().item():.6f}")
            
            print("\n⏭️  Optimizer step...")
            torch.nn.utils.clip_grad_norm_(lm_model.parameters(), 1.0)
            optimizer.step()
            print("✅ Optimizer step completed")
            
            # Check if gating alpha changed
            print("\n📈 Parameter change analysis...")
            for name, initial_value in initial_gating_values.items():
                current_param = dict(lm_model.named_parameters())[name]
                current_value = current_param.data
                change = (current_value - initial_value).abs().max().item()
                print(f"   {name}: change = {change:.8f}")
                
                if change < 1e-8:
                    print(f"      ❌ No significant change in {name}")
                else:
                    print(f"      ✅ {name} changed by {change:.8f}")
            
            return {
                'forward_success': True,
                'backward_success': True,
                'ttt_grads': ttt_grads,
                'gating_grads': gating_grads,
                'max_grad': max_grad
            }
            
        except Exception as e:
            print(f"❌ Forward/backward failed: {e}")
            import traceback
            traceback.print_exc()
            return None
        
    except Exception as e:
        print(f"❌ Test setup failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("🧪 STARTING PRODUCTION DEBUG")
    results = test_production_setup()
    
    if results:
        print(f"\n🎯 DIAGNOSTIC SUMMARY:")
        print(f"TTT gradients: {results['ttt_grads']}")
        print(f"Gating gradients: {results['gating_grads']}")
        print(f"Max gradient: {results['max_grad']:.6f}")
        
        if results['gating_grads'] == 0:
            print("🚨 ROOT CAUSE: Gating parameters not receiving gradients in production setup!")
    else:
        print("❌ Unable to complete diagnostic")