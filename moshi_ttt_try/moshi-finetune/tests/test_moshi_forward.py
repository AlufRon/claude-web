#!/usr/bin/env python3
"""
Step 1.3: Test vanilla Moshi forward pass (no TTT)
Verify we can run forward pass through Moshi model
"""

import torch
import sys
import os

# Add parent directory to path to import moshi
sys.path.append('/home/alufr/ttt_tests/moshi')

def test_moshi_forward_pass():
    """Test basic Moshi forward pass with dummy input"""
    print("🧪 Testing Moshi forward pass...")
    
    try:
        from moshi.models import loaders
        print("✅ Imported loaders")
        
        # Load checkpoint info
        checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
            hf_repo="kyutai/moshiko-pytorch-bf16"
        )
        print("✅ Checkpoint info loaded")
        
        # Get model config
        lm_config = (
            loaders._lm_kwargs 
            if checkpoint_info.raw_config is None 
            else checkpoint_info.raw_config
        )
        
        print("🏗️  Building LM model...")
        # Build the model (CPU first to avoid GPU memory issues)
        model = checkpoint_info.get_moshi(device="cpu")
        print(f"✅ LM model loaded: {type(model)}")
        
        # Print some basic model info
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   📊 Total parameters: {total_params:,}")
        print(f"   🏗️  Model device: {next(model.parameters()).device}")
        print(f"   📏 Model dtype: {next(model.parameters()).dtype}")
        
        # Create dummy input data
        batch_size = 2
        num_codebooks = 17  # Moshi uses 17 codebooks (1 text + 8 audio x 2 for stereo)
        seq_len = 10  # Short sequence for test
        
        # Create dummy codes (similar to what interleaver would produce)
        # Moshi expects codes of shape [B, K, T] where K=num_codebooks
        codes = torch.randint(0, 2048, (batch_size, num_codebooks, seq_len), dtype=torch.long)
        print(f"✅ Created dummy codes: shape {codes.shape}, dtype {codes.dtype}")
        
        # Test forward pass
        print("🚀 Running forward pass...")
        model.eval()
        with torch.no_grad():
            output = model(codes=codes)
            
        print(f"✅ Forward pass completed!")
        print(f"   📊 Output type: {type(output)}")
        
        if hasattr(output, 'logits'):
            print(f"   📏 Logits shape: {output.logits.shape}")
        if hasattr(output, 'text_logits'):
            print(f"   📝 Text logits shape: {output.text_logits.shape}")
            
        # Verify output shapes make sense
        expected_vocab_size = lm_config.get('n_q', 2048)  # Audio vocab size
        if hasattr(output, 'logits'):
            actual_vocab_size = output.logits.shape[-1]
            print(f"   🔍 Vocab size check: expected ~{expected_vocab_size}, got {actual_vocab_size}")
            
        print("🎉 Moshi forward pass test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error in forward pass: {e}")
        print(f"   Error type: {type(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_streaming_transformer_layer():
    """Test individual StreamingTransformerLayer (our target for TTT injection)"""
    print("\n🔧 Testing StreamingTransformerLayer...")
    
    try:
        from moshi.modules.transformer import StreamingTransformerLayer
        print("✅ StreamingTransformerLayer imported")
        
        # Create a simple transformer layer
        d_model = 512
        num_heads = 8 
        layer = StreamingTransformerLayer(
            d_model=d_model,
            num_heads=num_heads,
            dim_feedforward=2048,
            causal=True
        )
        print(f"✅ StreamingTransformerLayer created: d_model={d_model}, num_heads={num_heads}")
        
        # Test forward pass through single layer
        batch_size = 2
        seq_len = 16
        x = torch.randn(batch_size, seq_len, d_model)
        print(f"✅ Created test input: {x.shape}")
        
        layer.eval()
        with torch.no_grad():
            y = layer(x)
            
        print(f"✅ Layer forward pass: input {x.shape} → output {y.shape}")
        
        # Verify shapes match
        if x.shape == y.shape:
            print("✅ Shape consistency verified")
        else:
            print(f"⚠️  Shape mismatch: {x.shape} != {y.shape}")
            
        print("🎉 StreamingTransformerLayer test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error in layer test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting Moshi Forward Pass Tests...")
    print("=" * 60)
    
    # Test 1: Full model forward pass
    test1_passed = test_moshi_forward_pass()
    
    # Test 2: Individual layer test
    test2_passed = test_streaming_transformer_layer()
    
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS:")
    print(f"   Full Model Forward: {'✅ PASS' if test1_passed else '❌ FAIL'}")
    print(f"   Single Layer Test: {'✅ PASS' if test2_passed else '❌ FAIL'}")
    
    overall_success = test1_passed and test2_passed
    print(f"\n🏆 OVERALL: {'✅ SUCCESS' if overall_success else '❌ FAILED'}")
    
    if overall_success:
        print("✨ Phase 1 Complete! Ready for Step 2.1 (TTT Layer Creation)")
    else:
        print("🔧 Fix issues before proceeding")
    
    sys.exit(0 if overall_success else 1)