"""
Test to isolate and reproduce the TTT billion-scale loss bug.

This test aims to:
1. Verify TTT code works with normal inputs
2. Reproduce the billion-scale loss with real model activations
3. Identify where the corruption occurs
"""

import torch
import torch.nn.functional as F
from moshi_ttt.models.ssm.ops.ttt_mlp import compute_mini_batch, _recon_loss_with_params
from moshi_ttt.models.ssm.ops.utils import ln_fwd


def test_synthetic_inputs_normal_scale():
    """Test 1: TTT should work fine with normal-scale inputs."""
    print("\n" + "="*80)
    print("TEST 1: Synthetic Inputs (Normal Scale)")
    print("="*80)
    
    B, H, C, HD = 1, 32, 32, 128  # mini_batch_size=32
    
    # Create normal-scale inputs (typical neural network activation range)
    torch.manual_seed(42)
    XQ = torch.randn(B, H, C, HD, dtype=torch.float32) * 0.1  # Small values
    XK = torch.randn(B, H, C, HD, dtype=torch.float32) * 0.1
    XV = torch.randn(B, H, C, HD, dtype=torch.float32) * 0.1
    
    # TTT parameters (properly initialized)
    W1 = torch.randn(B, H, HD, HD*4, dtype=torch.float32) * 0.02
    b1 = torch.zeros(B, H, 1, HD*4, dtype=torch.float32)
    W2 = torch.randn(B, H, HD*4, HD, dtype=torch.float32) * 0.02
    b2 = torch.zeros(B, H, 1, HD, dtype=torch.float32)
    
    ttt_norm_weight = torch.ones(H, HD, dtype=torch.float32)
    ttt_norm_bias = torch.zeros(H, HD, dtype=torch.float32)
    
    # Learning rate (similar to actual model)
    eta = torch.ones(B, H, C, HD, 1, dtype=torch.float32) * 0.001
    
    # Prepare params and inputs
    params_dict = {
        "W1_states": W1,
        "b1_states": b1,
        "W2_states": W2,
        "b2_states": b2,
        "ttt_norm_weight": ttt_norm_weight,
        "ttt_norm_bias": ttt_norm_bias,
    }
    
    inputs = {
        "XQ": XQ,
        "XK": XK,
        "XV": XV,
        "eta": eta,
    }
    
    # Compute reconstruction target
    reconstruction_target = XV - XK
    
    print(f"Input Statistics:")
    print(f"  XK: min={XK.min():.4f}, max={XK.max():.4f}, mean={XK.mean():.4f}, std={XK.std():.4f}")
    print(f"  XV: min={XV.min():.4f}, max={XV.max():.4f}, mean={XV.mean():.4f}, std={XV.std():.4f}")
    print(f"  reconstruction_target: min={reconstruction_target.min():.4f}, max={reconstruction_target.max():.4f}, "
          f"mean={reconstruction_target.mean():.4f}, std={reconstruction_target.std():.4f}")
    
    # Compute loss manually
    ln_weight = ttt_norm_weight.reshape(H, 1, HD)
    ln_bias = ttt_norm_bias.reshape(H, 1, HD)
    loss = _recon_loss_with_params(XK, reconstruction_target, W1, b1, W2, b2, ln_weight, ln_bias)
    
    print(f"\nReconstruction Loss: {loss.item():.6f}")
    
    # Run TTT update
    result_params, XQW = compute_mini_batch(params_dict, inputs, log_losses=False, layer_id=None)
    
    print(f"Output XQW: min={XQW.min():.4f}, max={XQW.max():.4f}, mean={XQW.mean():.4f}")
    
    # Verify loss is reasonable
    assert loss.item() < 100.0, f"Loss too large: {loss.item()}"
    print("✅ PASS: Loss is in normal range")
    
    return loss.item()


def test_synthetic_inputs_large_scale():
    """Test 2: Reproduce billion-scale loss with corrupted inputs."""
    print("\n" + "="*80)
    print("TEST 2: Synthetic Inputs (Large Scale - Mimicking Bug)")
    print("="*80)
    
    B, H, C, HD = 1, 32, 32, 128
    
    # Create LARGE-scale inputs (mimicking the bug we saw)
    torch.manual_seed(42)
    XQ = torch.randn(B, H, C, HD, dtype=torch.float32) * 1000000  # Million-scale!
    XK = torch.randn(B, H, C, HD, dtype=torch.float32) * 1000000
    XV = torch.randn(B, H, C, HD, dtype=torch.float32) * 1000000
    
    # TTT parameters (properly initialized)
    W1 = torch.randn(B, H, HD, HD*4, dtype=torch.float32) * 0.02
    b1 = torch.zeros(B, H, 1, HD*4, dtype=torch.float32)
    W2 = torch.randn(B, H, HD*4, HD, dtype=torch.float32) * 0.02
    b2 = torch.zeros(B, H, 1, HD, dtype=torch.float32)
    
    ttt_norm_weight = torch.ones(H, HD, dtype=torch.float32)
    ttt_norm_bias = torch.zeros(H, HD, dtype=torch.float32)
    
    eta = torch.ones(B, H, C, HD, 1, dtype=torch.float32) * 0.001
    
    reconstruction_target = XV - XK
    
    print(f"Input Statistics:")
    print(f"  XK: min={XK.min():.0f}, max={XK.max():.0f}, mean={XK.mean():.0f}, std={XK.std():.0f}")
    print(f"  XV: min={XV.min():.0f}, max={XV.max():.0f}, mean={XV.mean():.0f}, std={XV.std():.0f}")
    print(f"  reconstruction_target: min={reconstruction_target.min():.0f}, max={reconstruction_target.max():.0f}, "
          f"mean={reconstruction_target.mean():.0f}, std={reconstruction_target.std():.0f}")
    
    # Compute loss
    ln_weight = ttt_norm_weight.reshape(H, 1, HD)
    ln_bias = ttt_norm_bias.reshape(H, 1, HD)
    loss = _recon_loss_with_params(XK, reconstruction_target, W1, b1, W2, b2, ln_weight, ln_bias)
    
    print(f"\nReconstruction Loss: {loss.item():.0f}")
    print(f"Loss magnitude: {loss.item():.2e}")
    
    # This should produce billion-scale loss
    assert loss.item() > 1e9, f"Expected billion-scale loss, got {loss.item()}"
    print("✅ PASS: Successfully reproduced billion-scale loss with large inputs")
    
    return loss.item()


def test_float32_vs_bfloat16():
    """Test 3: Compare float32 vs bfloat16 behavior."""
    print("\n" + "="*80)
    print("TEST 3: Float32 vs BFloat16 Comparison")
    print("="*80)
    
    B, H, C, HD = 1, 32, 32, 128
    
    # Create large-scale inputs
    torch.manual_seed(42)
    XK_f32 = torch.randn(B, H, C, HD, dtype=torch.float32) * 1000000
    XV_f32 = torch.randn(B, H, C, HD, dtype=torch.float32) * 1000000
    
    # Convert to bfloat16
    XK_bf16 = XK_f32.to(torch.bfloat16)
    XV_bf16 = XV_f32.to(torch.bfloat16)
    
    print(f"Float32 XK: min={XK_f32.min():.0f}, max={XK_f32.max():.0f}")
    print(f"BFloat16 XK: min={XK_bf16.min():.0f}, max={XK_bf16.max():.0f}")
    
    # Check if bfloat16 causes overflow/clamping
    reconstruction_target_f32 = XV_f32 - XK_f32
    reconstruction_target_bf16 = XV_bf16 - XK_bf16
    
    print(f"\nFloat32 target: min={reconstruction_target_f32.min():.0f}, max={reconstruction_target_f32.max():.0f}")
    print(f"BFloat16 target: min={reconstruction_target_bf16.min():.0f}, max={reconstruction_target_bf16.max():.0f}")
    
    # Compute MSE loss
    loss_f32 = F.mse_loss(reconstruction_target_f32, torch.zeros_like(reconstruction_target_f32))
    loss_bf16 = F.mse_loss(reconstruction_target_bf16.float(), torch.zeros_like(reconstruction_target_bf16).float())
    
    print(f"\nMSE Loss (Float32): {loss_f32.item():.2e}")
    print(f"MSE Loss (BFloat16): {loss_bf16.item():.2e}")
    
    # Check for infinities
    has_inf_f32 = torch.isinf(reconstruction_target_f32).any()
    has_inf_bf16 = torch.isinf(reconstruction_target_bf16).any()
    
    print(f"\nHas infinities (Float32): {has_inf_f32}")
    print(f"Has infinities (BFloat16): {has_inf_bf16}")
    
    print("✅ PASS: Dtype comparison complete")


def test_mini_batch_size_effect():
    """Test 4: Effect of mini_batch_size on loss magnitude."""
    print("\n" + "="*80)
    print("TEST 4: Mini-Batch Size Effect")
    print("="*80)
    
    B, H, HD = 1, 32, 128
    
    # Test different mini_batch sizes
    mini_batch_sizes = [1, 16, 32, 64]
    losses = []
    
    for C in mini_batch_sizes:
        torch.manual_seed(42)
        XK = torch.randn(B, H, C, HD, dtype=torch.float32) * 1000000
        XV = torch.randn(B, H, C, HD, dtype=torch.float32) * 1000000
        
        reconstruction_target = XV - XK
        
        # Compute MSE with reduction='mean'
        loss = F.mse_loss(reconstruction_target, torch.zeros_like(reconstruction_target), reduction='mean')
        
        print(f"\nmini_batch_size={C}:")
        print(f"  Shape: {reconstruction_target.shape}")
        print(f"  Total elements: {reconstruction_target.numel()}")
        print(f"  MSE Loss: {loss.item():.2e}")
        
        losses.append(loss.item())
    
    # Check if loss is independent of mini_batch_size
    loss_variance = torch.tensor(losses).std().item()
    print(f"\nLoss variance across mini_batch_sizes: {loss_variance:.2e}")
    
    # With reduction='mean', losses should be similar regardless of C
    print("✅ PASS: Mini-batch size effect analyzed")
    
    return losses


def test_real_model_forward_pass():
    """Test 5: Extract actual XV, XK from model forward pass."""
    print("\n" + "="*80)
    print("TEST 5: Real Model Forward Pass")
    print("="*80)
    
    try:
        import sys
        sys.path.insert(0, '/home/alufr/ttt_tests/moshi-finetune')
        from finetune.checkpoint_info import CheckpointInfo
        import logging
        
        # Suppress logging
        logging.getLogger().setLevel(logging.ERROR)
        
        print("Loading Moshi model...")
        checkpoint_info = CheckpointInfo(
            hf_repo_id="kyutai/moshiko-pytorch-bf16",
            config_path=None,
            moshi_path=None,
            mimi_path=None,
            tokenizer_path=None
        )
        
        moshi = checkpoint_info.get_moshi(device='cuda', dtype=torch.bfloat16)
        print("✅ Model loaded")
        
        # Create small audio input
        batch_size = 1
        audio_tokens = torch.randint(0, 2048, (batch_size, 8, 100), device='cuda')  # 100 tokens
        
        print(f"\nInput tokens shape: {audio_tokens.shape}")
        
        # Hook to capture TTT layer inputs
        captured_inputs = {}
        
        def capture_hook(name):
            def hook(module, input, output):
                # input is a tuple, get the hidden states
                if len(input) > 0:
                    hidden_states = input[0]
                    captured_inputs[name] = {
                        'hidden_states': hidden_states.detach().cpu(),
                        'min': hidden_states.min().item(),
                        'max': hidden_states.max().item(),
                        'mean': hidden_states.mean().item(),
                        'std': hidden_states.std().item(),
                    }
            return hook
        
        # Find TTT layers and attach hooks
        hooks = []
        for name, module in moshi.depformer.layers.named_children():
            if hasattr(module, 'self_attn') and hasattr(module.self_attn, 'ttt'):
                hook = module.self_attn.register_forward_hook(capture_hook(f"layer_{name}"))
                hooks.append(hook)
                print(f"Attached hook to layer {name}")
        
        # Run forward pass
        print("\nRunning forward pass...")
        with torch.no_grad():
            output = moshi(audio_tokens)
        
        print("✅ Forward pass complete")
        
        # Analyze captured inputs
        print("\nCaptured TTT layer inputs:")
        for layer_name, stats in captured_inputs.items():
            print(f"\n{layer_name}:")
            print(f"  Shape: {stats['hidden_states'].shape}")
            print(f"  Min: {stats['min']:.4f}")
            print(f"  Max: {stats['max']:.4f}")
            print(f"  Mean: {stats['mean']:.4f}")
            print(f"  Std: {stats['std']:.4f}")
            
            # Check for large values
            if abs(stats['max']) > 1000 or abs(stats['min']) > 1000:
                print(f"  ⚠️  WARNING: Large values detected!")
        
        # Cleanup
        for hook in hooks:
            hook.remove()
        
        print("\n✅ PASS: Real model analysis complete")
        
        return captured_inputs
        
    except Exception as e:
        print(f"❌ SKIP: Could not load model - {e}")
        return None


def run_all_tests():
    """Run all tests in sequence."""
    print("\n" + "#"*80)
    print("# TTT RECONSTRUCTION BUG TEST SUITE")
    print("#"*80)
    
    results = {}
    
    # Test 1: Normal inputs
    try:
        results['normal_loss'] = test_synthetic_inputs_normal_scale()
    except Exception as e:
        print(f"❌ FAIL: Test 1 failed - {e}")
        results['normal_loss'] = None
    
    # Test 2: Large inputs
    try:
        results['large_loss'] = test_synthetic_inputs_large_scale()
    except Exception as e:
        print(f"❌ FAIL: Test 2 failed - {e}")
        results['large_loss'] = None
    
    # Test 3: Dtype comparison
    try:
        test_float32_vs_bfloat16()
    except Exception as e:
        print(f"❌ FAIL: Test 3 failed - {e}")
    
    # Test 4: Mini-batch size
    try:
        results['mini_batch_losses'] = test_mini_batch_size_effect()
    except Exception as e:
        print(f"❌ FAIL: Test 4 failed - {e}")
        results['mini_batch_losses'] = None
    
    # Test 5: Real model (optional - skip if model not available)
    try:
        results['real_model'] = test_real_model_forward_pass()
    except Exception as e:
        print(f"⚠️  SKIP: Test 5 skipped - {e}")
        results['real_model'] = None
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    if results.get('normal_loss') and results.get('large_loss'):
        ratio = results['large_loss'] / results['normal_loss']
        print(f"Loss ratio (large/normal): {ratio:.2e}x")
        print(f"Normal input loss: {results['normal_loss']:.6f}")
        print(f"Large input loss: {results['large_loss']:.2e}")
    
    print("\n" + "#"*80)
    print("# END OF TEST SUITE")
    print("#"*80)
    
    return results


if __name__ == "__main__":
    results = run_all_tests()
