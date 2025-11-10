#!/usr/bin/env python3
"""
Step 1.2: Test base Moshi loading
Verify we can load Moshi model without any TTT modifications
"""

import torch
import sys
import os

# Add parent directory to path to import moshi
sys.path.append('/home/alufr/ttt_tests/moshi')

def test_moshi_loading():
    """Test that we can load base Moshi model"""
    print("🧪 Testing Moshi model loading...")
    
    try:
        from moshi.models import loaders
        print("✅ Successfully imported moshi.models.loaders")
        
        # Load checkpoint info (this should work without downloading if cached)
        print("📥 Loading checkpoint info...")
        checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
            hf_repo="kyutai/moshiko-pytorch-bf16"
        )
        print(f"✅ Checkpoint info loaded: {type(checkpoint_info)}")
        
        # Try to get the model configuration
        print("⚙️  Getting model configuration...")
        lm_config = (
            loaders._lm_kwargs 
            if checkpoint_info.raw_config is None 
            else checkpoint_info.raw_config
        )
        print(f"✅ Model config loaded: {len(lm_config)} parameters")
        print(f"   Key config params: {list(lm_config.keys())[:5]}...")
        
        # Load text tokenizer (lightweight test)
        print("📝 Loading text tokenizer...")
        spm = checkpoint_info.get_text_tokenizer()
        print(f"✅ Text tokenizer loaded: vocab_size = {spm.vocab_size}")
        
        print("🎉 All base Moshi loading tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading Moshi: {e}")
        print(f"   Error type: {type(e)}")
        return False

def test_moshi_basic_components():
    """Test loading individual Moshi components"""
    print("\n🔧 Testing Moshi components...")
    
    try:
        # Test streaming transformer import
        from moshi.modules.transformer import StreamingTransformer, StreamingTransformerLayer
        print("✅ StreamingTransformer classes imported successfully")
        
        # Test LM model import  
        from moshi.models.lm import LMModel
        print("✅ LMModel imported successfully")
        
        print("🎉 All Moshi component tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error importing Moshi components: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting Moshi Base Tests...")
    print("=" * 50)
    
    # Test 1: Basic loading
    test1_passed = test_moshi_loading()
    
    # Test 2: Component imports
    test2_passed = test_moshi_basic_components()
    
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS:")
    print(f"   Base Loading: {'✅ PASS' if test1_passed else '❌ FAIL'}")  
    print(f"   Component Imports: {'✅ PASS' if test2_passed else '❌ FAIL'}")
    
    overall_success = test1_passed and test2_passed
    print(f"\n🏆 OVERALL: {'✅ SUCCESS' if overall_success else '❌ FAILED'}")
    
    if overall_success:
        print("✨ Ready to proceed to Step 1.3!")
    else:
        print("🔧 Fix issues before proceeding")
    
    sys.exit(0 if overall_success else 1)