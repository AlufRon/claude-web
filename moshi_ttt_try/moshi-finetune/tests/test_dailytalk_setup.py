#!/usr/bin/env python3
"""
Test script to validate DailyTalk dataset setup and TTT-Moshi integration
before running full training
"""

import torch
import sys
import os
import json
from pathlib import Path

# Add paths
sys.path.append('/home/alufr/ttt_tests/moshi')
sys.path.insert(0, '.')

def test_dataset_access():
    """Test access to DailyTalk dataset"""
    print("🔍 Testing DailyTalk Dataset Access...")
    
    base_path = "/sise/eliyanac-group/ron_al/daily-talk-contiguous"
    
    # Check paths exist
    paths_to_check = [
        f"{base_path}/train/dailytalk_train.jsonl",
        f"{base_path}/eval/dailytalk_eval.jsonl",
        f"{base_path}/data_stereo"
    ]
    
    for path in paths_to_check:
        if os.path.exists(path):
            print(f"   ✅ {path}")
        else:
            print(f"   ❌ {path} - NOT FOUND")
            return False
    
    # Check sample audio file
    with open(f"{base_path}/train/dailytalk_train.jsonl", 'r') as f:
        first_line = f.readline().strip()
        data = json.loads(first_line)
        audio_path = os.path.join(base_path, "train", data['path'])
        
        if os.path.exists(audio_path):
            print(f"   ✅ Sample audio: {audio_path}")
        else:
            print(f"   ❌ Sample audio: {audio_path} - NOT FOUND")
            return False
    
    print("✅ Dataset access test passed!")
    return True

def test_ttt_imports():
    """Test TTT module imports"""
    print("\n🔍 Testing TTT Module Imports...")
    
    try:
        from moshi_ttt.config import TTTConfig
        from moshi_ttt.hybrid_layer import HybridStreamingTransformerLayer
        print("   ✅ TTT imports successful")
        
        # Test TTT config creation
        config = TTTConfig(
            model_dim=512,
            num_heads=8,
            mini_batch_size=16,
            ttt_base_lr=0.1
        )
        print("   ✅ TTT config creation successful")
        
        return True
    except Exception as e:
        print(f"   ❌ TTT import failed: {e}")
        return False

def test_moshi_model_loading():
    """Test Moshi model loading and TTT integration"""
    print("\n🔍 Testing Moshi Model Loading...")
    
    try:
        from moshi.models import loaders
        from moshi_ttt.hybrid_layer import HybridStreamingTransformerLayer
        from moshi_ttt.config import TTTConfig
        
        # Load small model for testing
        print("   Loading checkpoint info...")
        checkpoint_info = loaders.CheckpointInfo.from_hf_repo("kyutai/moshiko-pytorch-bf16")
        
        lm_config = loaders._lm_kwargs if checkpoint_info.raw_config is None else checkpoint_info.raw_config
        
        # Use very small config for quick test
        test_config = lm_config.copy()
        test_config['num_layers'] = 2
        test_config['dim'] = 256
        test_config['num_heads'] = 4
        test_config['depformer_num_layers'] = 1
        
        print("   Loading model...")
        lm_model = loaders.get_moshi_lm(
            filename=None,
            lm_kwargs=test_config,
            device='cpu',
            dtype=torch.float32
        )
        
        print(f"   ✅ Model loaded: {type(lm_model)}")
        print(f"      Layers: {len(lm_model.transformer.layers)}")
        
        # Test TTT integration
        print("   Testing TTT integration...")
        ttt_config = TTTConfig(
            model_dim=test_config['dim'],
            num_heads=test_config['num_heads'],
            mini_batch_size=8,
            ttt_base_lr=0.1
        )
        
        # Convert first layer to TTT
        original_layer = lm_model.transformer.layers[0]
        hybrid_layer = HybridStreamingTransformerLayer(original_layer, ttt_config)
        lm_model.transformer.layers[0] = hybrid_layer
        
        print("   ✅ TTT integration successful")
        
        # Test forward pass
        print("   Testing forward pass...")
        batch_size = 1
        seq_len = 4
        n_codebooks = test_config.get('n_q', 8) + 1
        
        codes = torch.randint(0, 32, (batch_size, n_codebooks, seq_len), dtype=torch.int64)
        
        lm_model.eval()
        with torch.no_grad():
            output = lm_model(codes)
        
        print(f"   ✅ Forward pass successful")
        print(f"      Output logits: {output.logits.shape}")
        print(f"      Text logits: {output.text_logits.shape}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Model loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_loading():
    """Test data loading pipeline"""
    print("\n🔍 Testing Data Loading Pipeline...")
    
    try:
        # Test basic data loading without full pipeline
        base_path = "/sise/eliyanac-group/ron_al/daily-talk-contiguous"
        train_file = f"{base_path}/train/dailytalk_train.jsonl"
        
        print("   Reading training data...")
        train_samples = []
        with open(train_file, 'r') as f:
            for i, line in enumerate(f):
                if i >= 5:  # Only check first 5 samples
                    break
                data = json.loads(line.strip())
                
                # Build full path
                if data['path'].startswith('../'):
                    audio_path = os.path.join(base_path, data['path'][3:])
                else:
                    audio_path = os.path.join(base_path, "train", data['path'])
                
                if os.path.exists(audio_path):
                    train_samples.append({
                        'path': audio_path,
                        'duration': data['duration']
                    })
                    print(f"   ✅ Sample {i+1}: {os.path.basename(audio_path)} ({data['duration']:.1f}s)")
                else:
                    print(f"   ❌ Sample {i+1}: {audio_path} not found")
                    return False
        
        print(f"   ✅ Found {len(train_samples)} valid training samples")
        
        # Check evaluation data too
        eval_file = f"{base_path}/eval/dailytalk_eval.jsonl"
        with open(eval_file, 'r') as f:
            eval_count = sum(1 for _ in f)
        
        print(f"   ✅ Found {eval_count} evaluation samples")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Data loading test failed: {e}")
        return False

def test_config_loading():
    """Test training configuration"""
    print("\n🔍 Testing Training Configuration...")
    
    try:
        from finetune.args import TrainArgs
        
        config_path = "configs/dailytalk_ttt_config.yaml"
        if not os.path.exists(config_path):
            print(f"   ❌ Config file not found: {config_path}")
            return False
        
        print(f"   Loading config: {config_path}")
        args = TrainArgs.load(config_path, drop_extra_fields=False)
        
        print("   ✅ Config loaded successfully")
        print(f"      Training data: {args.data.train_data}")
        print(f"      Eval data: {args.data.eval_data}")
        print(f"      Max steps: {args.max_steps}")
        print(f"      Batch size: {args.batch_size}")
        print(f"      LoRA enabled: {args.lora.enable}")
        
        # Validate paths in config
        if not os.path.exists(args.data.train_data):
            print(f"   ❌ Training data path invalid: {args.data.train_data}")
            return False
        
        if not os.path.exists(args.data.eval_data):
            print(f"   ❌ Eval data path invalid: {args.data.eval_data}")
            return False
        
        print("   ✅ Config paths validated")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Config test failed: {e}")
        return False

def main():
    print("🔥 DailyTalk TTT-Moshi Setup Validation")
    print("Testing all components before production training")
    print("=" * 60)
    
    tests = [
        ("Dataset Access", test_dataset_access),
        ("TTT Imports", test_ttt_imports), 
        ("Moshi Model Loading", test_moshi_model_loading),
        ("Data Loading", test_data_loading),
        ("Config Loading", test_config_loading),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    print("\n" + "=" * 60)
    print("📊 SETUP VALIDATION RESULTS:")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if not passed:
            all_passed = False
    
    print(f"\n🏆 OVERALL: {'✅ READY FOR TRAINING' if all_passed else '❌ SETUP ISSUES DETECTED'}")
    
    if all_passed:
        print("\n🚀 Next Steps:")
        print("   1. Run training with: python train_ttt.py configs/dailytalk_ttt_config.yaml")
        print("   2. Monitor training logs for TTT integration")
        print("   3. Check model convergence and performance")
        print("\n💡 Training Command Examples:")
        print("   # TTT on middle layers (recommended)")
        print("   python train_ttt.py configs/dailytalk_ttt_config.yaml --ttt_layers=middle")
        print("   # TTT on all layers") 
        print("   python train_ttt.py configs/dailytalk_ttt_config.yaml --ttt_layers=all")
        print("   # Vanilla Moshi baseline")
        print("   python train_ttt.py configs/dailytalk_ttt_config.yaml --ttt_layers=none")
    else:
        print("\n🔧 Please resolve the failing tests before proceeding with training")
    
    return all_passed

if __name__ == "__main__":
    main()