#!/usr/bin/env python3
"""
Test updated sBLIMP evaluator to confirm non-zero accuracy potential
"""

import sys
import os
sys.path.append('/home/alufr/ttt_tests/moshi-finetune')

# Set environment first
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

import torch
import logging

# Configure logging for verbose output
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_sblimp_data_loading():
    """Test that sBLIMP data loading works with the fixed evaluator"""
    try:
        from finetune.paper_metrics import PaperMetricsEvaluator
        import pandas as pd
        from pathlib import Path
        
        # Test configuration
        config = {
            'sblimp_audio_dir': '/sise/eliyanac-group/ron_al/sblimp_data/sLM21_dataset/syntactic/test/',
            'sblimp_gold_csv': '/sise/eliyanac-group/ron_al/sblimp_data/sLM21_dataset/syntactic/test/gold.csv',
            'sblimp_max_pairs': 5  # Small test
        }
        
        # Create dummy evaluator (we don't need real models for data testing)
        evaluator = PaperMetricsEvaluator(None, None, device="cuda", config=config)
        
        # Test CSV loading
        gold_csv = config['sblimp_gold_csv']
        audio_dir = config['sblimp_audio_dir']
        
        print(f"Testing data loading:")
        print(f"  Gold CSV: {gold_csv}")
        print(f"  Audio dir: {audio_dir}")
        
        if not Path(gold_csv).exists():
            print(f"❌ Gold CSV not found: {gold_csv}")
            return False
            
        if not Path(audio_dir).exists():
            print(f"❌ Audio directory not found: {audio_dir}")
            return False
        
        # Load CSV
        df = pd.read_csv(gold_csv)
        print(f"✅ CSV loaded with {len(df)} rows and columns: {list(df.columns)}")
        
        # Test sentence pair creation
        sentence_pairs = evaluator._create_sblimp_sentence_pairs(df, audio_dir, max_pairs=5)
        print(f"✅ Created {len(sentence_pairs)} sentence pairs")
        
        if len(sentence_pairs) == 0:
            print(f"❌ No valid sentence pairs found")
            return False
        
        # Test that audio files exist
        for i, pair in enumerate(sentence_pairs[:3]):  # Check first 3 pairs
            good_path = pair['good_audio_path']
            bad_path = pair['bad_audio_path']
            
            good_exists = Path(good_path).exists()
            bad_exists = Path(bad_path).exists()
            
            print(f"  Pair {i+1}: good={good_exists}, bad={bad_exists}")
            print(f"    Good: {good_path}")
            print(f"    Bad: {bad_path}")
            
            if not good_exists or not bad_exists:
                print(f"❌ Missing audio files for pair {i+1}")
                return False
        
        print(f"✅ All tested audio files exist")
        print(f"✅ sBLIMP data loading test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ sBLIMP data loading test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_minimal_moshi_loading():
    """Test that we can load Moshi model for evaluation"""
    try:
        import torch
        from moshi.models.loaders import get_moshi_lm, get_mimi, hf_hub_download, DEFAULT_REPO
        
        print(f"Testing minimal Moshi loading...")
        
        # Load MIMI (lighter model)
        print("  Loading MIMI...")
        mimi_weight = hf_hub_download(DEFAULT_REPO, "tokenizer-e351c8d8-checkpoint125.safetensors")
        mimi = get_mimi(mimi_weight, device="cuda").eval().float()
        print(f"  ✅ MIMI loaded")
        
        # Load Moshi LM
        print("  Loading Moshi LM...")
        moshi_weight = hf_hub_download(DEFAULT_REPO, "model.safetensors") 
        moshi = get_moshi_lm(moshi_weight, device="cuda").eval().float()
        print(f"  ✅ Moshi LM loaded")
        
        # Check key attributes
        print(f"  Model attributes:")
        print(f"    num_codebooks: {moshi.num_codebooks}")
        print(f"    dep_q: {moshi.dep_q}")
        print(f"    audio_offset: {moshi.audio_offset}")
        print(f"    zero_token_id: {moshi.zero_token_id}")
        
        print(f"✅ Moshi loading test PASSED")
        return True, moshi, mimi
        
    except Exception as e:
        print(f"❌ Moshi loading test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None

def test_audio_encoding():
    """Test audio encoding with real files"""
    try:
        from finetune.paper_metrics import PaperMetricsEvaluator
        from pathlib import Path
        
        # Test configuration
        config = {
            'sblimp_audio_dir': '/sise/eliyanac-group/ron_al/sblimp_data/sLM21_dataset/syntactic/test/',
            'sblimp_gold_csv': '/sise/eliyanac-group/ron_al/sblimp_data/sLM21_dataset/syntactic/test/gold.csv',
        }
        
        # Load models for encoding test
        success, moshi, mimi = test_minimal_moshi_loading()
        if not success:
            return False
        
        # Create evaluator
        evaluator = PaperMetricsEvaluator(mimi, None, device="cuda", config=config)
        
        # Find a real audio file to test
        audio_dir = Path(config['sblimp_audio_dir'])
        audio_files = list(audio_dir.glob("*.wav"))
        
        if len(audio_files) == 0:
            print(f"❌ No audio files found in {audio_dir}")
            return False
        
        test_file = audio_files[0]
        print(f"Testing audio encoding with: {test_file}")
        
        # Test encoding
        codes = evaluator._encode_audio(str(test_file))
        print(f"✅ Audio encoded successfully")
        print(f"  Codes shape: {codes.shape}")
        print(f"  Codes dtype: {codes.dtype}")
        print(f"  Codes device: {codes.device}")
        
        # Test likelihood computation  
        nll = evaluator._compute_sblimp_likelihood(moshi, codes)
        print(f"✅ Likelihood computed: {nll}")
        
        if nll == float('inf'):
            print(f"❌ Got infinite likelihood - this may indicate an issue")
            return False
        
        print(f"✅ Audio encoding test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Audio encoding test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests to verify the fixed evaluator"""
    print("🧪 Testing fixed sBLIMP evaluator...")
    print("=" * 60)
    
    # Test 1: Data loading
    print("\n📋 Test 1: Data Loading")
    data_ok = test_sblimp_data_loading()
    
    # Test 2: Audio encoding
    print("\n🎵 Test 2: Audio Encoding")
    encoding_ok = test_audio_encoding()
    
    # Summary
    print("\n" + "=" * 60)
    print("🎯 TEST SUMMARY:")
    print(f"  Data Loading: {'✅ PASS' if data_ok else '❌ FAIL'}")
    print(f"  Audio Encoding: {'✅ PASS' if encoding_ok else '❌ FAIL'}")
    
    if data_ok and encoding_ok:
        print(f"\n🎉 ALL TESTS PASSED - Fixed evaluator should produce non-zero accuracy!")
        print(f"   The evaluator can now:")
        print(f"   ✅ Load and parse sBLIMP CSV data correctly")
        print(f"   ✅ Create valid sentence pairs using working methodology")  
        print(f"   ✅ Encode real audio files without errors")
        print(f"   ✅ Compute meaningful likelihoods (not infinite)")
        return 0
    else:
        print(f"\n❌ SOME TESTS FAILED - Need to fix remaining issues")
        return 1

if __name__ == "__main__":
    sys.exit(main())