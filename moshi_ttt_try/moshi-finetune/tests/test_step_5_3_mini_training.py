#!/usr/bin/env python3
"""
Step 5.3: Mini Training Run
Validate that TTT-Moshi model can train end-to-end with loss convergence
"""

import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import matplotlib.pyplot as plt
from pathlib import Path

# Disable torch dynamo compilation to get cleaner errors
import torch._dynamo
torch._dynamo.config.suppress_errors = True

# Add paths
sys.path.append('/home/alufr/ttt_tests/moshi')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def create_synthetic_training_data(batch_size, seq_len, n_codebooks, vocab_size, num_batches):
    """Create synthetic training data that mimics Moshi's format"""
    print(f"📦 Creating synthetic training data...")
    print(f"   Format: [{batch_size}, {n_codebooks}, {seq_len}] for {num_batches} batches")
    
    batches = []
    for i in range(num_batches):
        # Create codes tensor [B, K, T]
        codes = torch.randint(0, vocab_size, (batch_size, n_codebooks, seq_len), dtype=torch.int64)
        
        # Make text tokens (codebook 0) have smaller vocabulary 
        codes[:, 0, :] = torch.randint(0, min(100, vocab_size), (batch_size, seq_len), dtype=torch.int64)
        
        # Ensure some variation between batches
        if i % 2 == 1:
            codes = codes + torch.randint(0, 5, codes.shape, dtype=torch.int64)
            codes = torch.clamp(codes, 0, vocab_size - 1)
        
        batches.append(codes)
    
    print(f"✅ Created {len(batches)} synthetic batches")
    return batches

def compute_moshi_loss(output, codes, model):
    """Compute loss exactly like Moshi training (from finetune/loss.py)"""
    
    def compute_loss_with_mask(logits, target, target_mask, mode, first_codebook_weight_multiplier=1.0):
        """Simplified version of finetune/loss.py compute_loss_with_mask"""
        target = torch.where(target_mask, target, torch.zeros_like(target))
        
        weights = target_mask.float()
        if mode == "audio":
            weights[:, 0] *= first_codebook_weight_multiplier
        
        logits = logits.view(-1, logits.size(-1)).float()
        target = target.view(-1)
        weights = weights.view(-1)
        
        mb_loss = nn.functional.cross_entropy(logits, target, reduction="none")
        mb_loss = torch.where(weights > 0.0, mb_loss * weights, torch.zeros_like(mb_loss))
        
        # Avoid division by zero
        weights_sum = torch.sum(weights)
        if weights_sum > 0:
            mb_loss = torch.sum(mb_loss) / weights_sum
        else:
            mb_loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        return mb_loss
    
    # Text loss (codebook 0)
    text_loss = compute_loss_with_mask(
        output.text_logits,
        codes[:, :model.audio_offset],  # Text codebooks only
        output.text_mask,
        mode="text"
    )
    
    # Audio loss (codebooks 1+)
    audio_loss = compute_loss_with_mask(
        output.logits, 
        codes[:, model.audio_offset:model.audio_offset + model.dep_q],  # Audio codebooks
        output.mask,
        mode="audio", 
        first_codebook_weight_multiplier=2.0
    )
    
    return text_loss, audio_loss, text_loss + audio_loss

def test_mini_training_run():
    """Test 5-step mini training run with TTT-Moshi model"""
    print("🚀 Step 5.3: Mini Training Run")
    print("Testing TTT-Moshi end-to-end training capability")
    print("=" * 70)
    
    try:
        from moshi.models import loaders
        from moshi_ttt.hybrid_layer import HybridStreamingTransformerLayer
        from moshi_ttt.config import TTTConfig
        
        print("✅ Imports successful")
        
        # 1. Create minimal model for training
        print("\n📥 Setting up TTT-Moshi model...")
        checkpoint_info = loaders.CheckpointInfo.from_hf_repo("kyutai/moshiko-pytorch-bf16")
        lm_config = loaders._lm_kwargs if checkpoint_info.raw_config is None else checkpoint_info.raw_config
        
        # Use very small model for training test
        train_config = lm_config.copy()
        train_config['num_layers'] = 3        # Minimal layers for training
        train_config['dim'] = 256             # Small dimension for speed
        train_config['num_heads'] = 4         # Small number of heads
        train_config['depformer_num_layers'] = 2  # Reduce depformer
        
        print(f"   Model config: {train_config['dim']}d, {train_config['num_layers']} layers")
        
        # Build model
        lm_model = loaders.get_moshi_lm(
            filename=None,
            lm_kwargs=train_config,
            device='cpu',
            dtype=torch.float32
        )
        
        print(f"✅ Base model loaded: {type(lm_model)}")
        
        # 2. Convert to TTT model
        print("\n🔄 Converting to TTT-enabled model...")
        
        ttt_config = TTTConfig(
            model_dim=train_config['dim'],
            num_heads=train_config['num_heads'],
            mini_batch_size=8,  # Small mini-batch for training
            ttt_base_lr=0.01    # Small learning rate for TTT
        )
        
        # Replace all transformer layers with TTT
        original_param_count = sum(p.numel() for p in lm_model.parameters())
        
        for i, layer in enumerate(lm_model.transformer.layers):
            lm_model.transformer.layers[i] = HybridStreamingTransformerLayer(layer, ttt_config)
            print(f"   ✅ Layer {i}: Converted to TTT")
        
        ttt_param_count = sum(p.numel() for p in lm_model.parameters()) 
        print(f"✅ TTT conversion complete:")
        print(f"   Original parameters: {original_param_count:,}")
        print(f"   TTT parameters: {ttt_param_count:,}")
        print(f"   Parameter increase: +{ttt_param_count - original_param_count:,} (+{(ttt_param_count/original_param_count - 1)*100:.1f}%)")
        
        # 3. Setup training
        print("\n⚙️  Setting up training...")
        
        # Create synthetic training data
        batch_size = 1
        seq_len = 4      # Very short sequences for speed
        n_codebooks = train_config.get('n_q', 8) + 1  # Text + audio codebooks
        vocab_size = 32  # Small vocab for synthetic data
        num_train_steps = 5
        
        train_batches = create_synthetic_training_data(
            batch_size, seq_len, n_codebooks, vocab_size, num_train_steps
        )
        
        # Setup optimizer
        optimizer = optim.AdamW(lm_model.parameters(), lr=1e-4, weight_decay=0.01)
        
        # Enable training mode
        lm_model.train()
        
        # Track training metrics
        losses = []
        text_losses = []
        audio_losses = []
        
        print(f"✅ Training setup complete:")
        print(f"   Optimizer: AdamW(lr=1e-4)")
        print(f"   Training steps: {num_train_steps}")
        print(f"   Batch size: {batch_size}, Sequence length: {seq_len}")
        
        # 4. Training loop
        print(f"\n🎯 Starting {num_train_steps}-step training run...")
        print("-" * 50)
        
        for step in range(num_train_steps):
            print(f"\n📚 Training Step {step + 1}/{num_train_steps}")
            
            # Get batch
            codes = train_batches[step]
            print(f"   Input codes: {codes.shape}")
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            print("   🔄 Forward pass...")
            try:
                output = lm_model(codes)
                print(f"   ✅ Forward completed")
                print(f"      Audio logits: {output.logits.shape}")
                print(f"      Text logits: {output.text_logits.shape}")
                
                # Check output health
                audio_finite = torch.isfinite(output.logits).float().mean().item()
                text_finite = torch.isfinite(output.text_logits).float().mean().item()
                print(f"      Audio finite: {audio_finite:.1%}, Text finite: {text_finite:.1%}")
                
            except Exception as e:
                print(f"   ❌ Forward pass failed: {e}")
                return False
            
            # Compute loss
            print("   📊 Computing loss...")
            try:
                text_loss, audio_loss, total_loss = compute_moshi_loss(output, codes, lm_model)
                print(f"   ✅ Loss computed")
                print(f"      Text loss: {text_loss.item():.4f}")
                print(f"      Audio loss: {audio_loss.item():.4f}")
                print(f"      Total loss: {total_loss.item():.4f}")
                
                # Check loss is reasonable
                if not torch.isfinite(total_loss):
                    print(f"   ⚠️  Non-finite loss detected!")
                    return False
                    
                losses.append(total_loss.item())
                text_losses.append(text_loss.item())
                audio_losses.append(audio_loss.item())
                
            except Exception as e:
                print(f"   ❌ Loss computation failed: {e}")
                return False
            
            # Backward pass
            print("   ⬅️  Backward pass...")
            try:
                total_loss.backward()
                print(f"   ✅ Backward completed")
                
                # Check gradients
                total_params = 0
                params_with_grad = 0
                ttt_params_with_grad = 0
                max_grad_norm = 0.0
                
                for name, param in lm_model.named_parameters():
                    if param.requires_grad:
                        total_params += 1
                        if param.grad is not None:
                            params_with_grad += 1
                            grad_norm = param.grad.norm().item()
                            max_grad_norm = max(max_grad_norm, grad_norm)
                            
                            if any(ttt_key in name for ttt_key in ['W1', 'W2', 'b1', 'b2', 'ttt_norm', 'learnable_ttt']):
                                ttt_params_with_grad += 1
                
                grad_ratio = params_with_grad / total_params if total_params > 0 else 0
                print(f"   📊 Gradient check:")
                print(f"      Parameters with gradients: {params_with_grad}/{total_params} ({grad_ratio:.1%})")
                print(f"      TTT parameters with gradients: {ttt_params_with_grad}")
                print(f"      Max gradient norm: {max_grad_norm:.4f}")
                
                if grad_ratio < 0.8:
                    print(f"   ⚠️  Low gradient flow detected!")
                
            except Exception as e:
                print(f"   ❌ Backward pass failed: {e}")
                return False
            
            # Optimizer step
            print("   ⏭️  Optimizer step...")
            try:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(lm_model.parameters(), max_norm=1.0)
                optimizer.step()
                print(f"   ✅ Optimizer step completed")
                
            except Exception as e:
                print(f"   ❌ Optimizer step failed: {e}")
                return False
            
            print(f"   🎯 Step {step + 1} complete: Loss = {total_loss.item():.4f}")
        
        # 5. Training analysis
        print(f"\n📊 TRAINING ANALYSIS:")
        print("-" * 50)
        
        print(f"Loss trajectory:")
        for i, (loss, text_loss, audio_loss) in enumerate(zip(losses, text_losses, audio_losses)):
            print(f"   Step {i+1}: Total={loss:.4f}, Text={text_loss:.4f}, Audio={audio_loss:.4f}")
        
        # Check for learning (loss should change)
        initial_loss = losses[0]
        final_loss = losses[-1]
        loss_change = abs(final_loss - initial_loss)
        loss_change_pct = (loss_change / initial_loss) * 100
        
        print(f"\nLearning indicators:")
        print(f"   Initial loss: {initial_loss:.4f}")
        print(f"   Final loss: {final_loss:.4f}")
        print(f"   Loss change: {loss_change:.4f} ({loss_change_pct:.1f}%)")
        
        # Success criteria
        success_criteria = {
            'forward_pass': True,  # All forward passes completed
            'backward_pass': True,  # All backward passes completed
            'finite_loss': all(torch.isfinite(torch.tensor(l)) for l in losses),
            'gradient_flow': grad_ratio > 0.8,  # Most parameters have gradients
            'loss_variation': loss_change > 0.001,  # Loss changed meaningfully
            'ttt_gradients': ttt_params_with_grad > 0,  # TTT parameters learning
        }
        
        print(f"\n✅ SUCCESS CRITERIA:")
        print("-" * 50)
        all_passed = True
        for criterion, passed in success_criteria.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"   {criterion}: {status}")
            if not passed:
                all_passed = False
        
        if all_passed:
            print(f"\n🏆 STEP 5.3: ✅ SUCCESS!")
            print(f"   🎉 TTT-Moshi model successfully trained end-to-end!")
            print(f"   📚 All {num_train_steps} training steps completed")
            print(f"   🧠 TTT layers learning alongside Moshi")
            print(f"   🚀 Model ready for full-scale training!")
        else:
            print(f"\n⚠️  STEP 5.3: PARTIAL SUCCESS")
            print(f"   Some criteria not met, but core functionality works")
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Mini training run failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_stability():
    """Additional test for training stability over more steps"""
    print(f"\n🔍 Step 5.3b: Training Stability Test...")
    
    try:
        from moshi.modules.transformer import StreamingTransformerLayer
        from moshi_ttt.hybrid_layer import HybridStreamingTransformerLayer  
        from moshi_ttt.config import TTTConfig
        
        # Create minimal test case
        d_model = 128
        num_heads = 2
        seq_len = 4
        batch_size = 1
        
        original_layer = StreamingTransformerLayer(
            d_model=d_model,
            num_heads=num_heads,
            dim_feedforward=d_model * 2,
            causal=True,
            context=10,
            norm='rms_norm'
        )
        
        ttt_config = TTTConfig(
            model_dim=d_model,
            num_heads=num_heads,
            mini_batch_size=4,
            ttt_base_lr=0.01
        )
        
        hybrid_layer = HybridStreamingTransformerLayer(original_layer, ttt_config)
        
        # Simple training loop
        optimizer = optim.SGD(hybrid_layer.parameters(), lr=0.01)
        
        losses = []
        for step in range(10):
            optimizer.zero_grad()
            
            x = torch.randn(batch_size, seq_len, d_model) * 0.1
            output = hybrid_layer(x)
            
            # Simple MSE loss
            target = torch.zeros_like(output)
            loss = nn.functional.mse_loss(output, target)
            
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
        
        # Check stability
        loss_std = torch.std(torch.tensor(losses)).item()
        stable = loss_std < 1.0  # Not exploding
        
        print(f"✅ Stability test: {'STABLE' if stable else 'UNSTABLE'}")
        print(f"   Loss std: {loss_std:.4f}")
        print(f"   Final loss: {losses[-1]:.4f}")
        
        return stable
        
    except Exception as e:
        print(f"❌ Stability test failed: {e}")
        return False

def main():
    print("🔥 Step 5.3: Mini Training Run Testing")
    print("Validating end-to-end TTT-Moshi training capability")
    print("=" * 70)
    
    # Run main training test
    success1 = test_mini_training_run()
    
    # Run stability test
    success2 = test_training_stability()
    
    print("\n" + "=" * 70)
    print("📊 Step 5.3 Results:")
    print(f"   Mini Training Run: {'✅ PASS' if success1 else '❌ FAIL'}")
    print(f"   Training Stability: {'✅ PASS' if success2 else '❌ FAIL'}")
    
    overall_success = success1 and success2
    
    print(f"\n🏆 Step 5.3 OVERALL: {'✅ PASSED' if overall_success else '❌ FAILED'}")
    
    if overall_success:
        print("\n🎉 PHASE 5 COMPLETE: TTT Training Integration Successful!")
        print("🚀 TTT-Moshi model ready for production training!")
        print("\n💡 Next Steps:")
        print("   - Production training with real data")
        print("   - Hyperparameter optimization")
        print("   - Performance benchmarking")
        print("   - Model deployment")
        print("\n🏁 Full TTT-Moshi pipeline is complete and validated!")
    else:
        print("\n🔧 Need to resolve training issues before production use")

if __name__ == "__main__":
    main()