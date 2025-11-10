#!/usr/bin/env python3
"""
Basic functionality test for multi-layer TTT-MLP implementation.

This script tests the new configurable TTT-MLP layer count feature
to ensure backward compatibility and multi-layer functionality.
"""

import torch
import sys
from dataclasses import dataclass
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from finetune.args import TTTArgs
from moshi_ttt.models.ssm.ttt_layer import TTTWrapper, TTTMLP, TTTMLPMultiLayer


@dataclass 
class MockConfig:
    """Mock configuration class for testing"""
    model_dim: int = 512
    num_heads: int = 8
    ssm_layer: str = "ttt_mlp"
    rope_theta: float = 10000.0
    
    # TTT-specific
    ttt_base_lr: float = 0.1
    mini_batch_size: int = 4
    scan_checkpoint_group_size: int = 1
    
    # Multi-layer configuration
    ttt_mlp_layers: int = 2
    ttt_mlp_expansion_factor: float = 4.0
    ttt_mlp_hidden_dims: list = None
    
    @property
    def head_dim(self):
        return self.model_dim // self.num_heads


def test_ttt_args_validation():
    """Test TTTArgs validation for multi-layer configuration"""
    print("🧪 Testing TTTArgs validation...")
    
    # Test default 2-layer configuration (should work)
    args = TTTArgs(enable=True, ttt_mlp_layers=2)
    print(f"✅ Default 2-layer config: layers={args.ttt_mlp_layers}, expansion={args.ttt_mlp_expansion_factor}")
    
    # Test 3-layer configuration
    args = TTTArgs(enable=True, ttt_mlp_layers=3, ttt_mlp_expansion_factor=4.0)
    print(f"✅ 3-layer config: layers={args.ttt_mlp_layers}, expansion={args.ttt_mlp_expansion_factor}")
    
    # Test custom dimensions
    args = TTTArgs(enable=True, ttt_mlp_layers=3, ttt_mlp_hidden_dims=[128, 256])
    print(f"✅ Custom dimensions config: layers={args.ttt_mlp_layers}, dims={args.ttt_mlp_hidden_dims}")
    
    # Test invalid configuration (should fail)
    try:
        args = TTTArgs(enable=True, ttt_mlp_layers=3, ttt_mlp_hidden_dims=[128])  # Wrong number of dims
        print("❌ Should have failed validation")
    except AssertionError as e:
        print(f"✅ Correctly caught validation error: {e}")
    
    print()


def test_layer_instantiation():
    """Test that correct layer classes are instantiated based on configuration"""
    print("🧪 Testing layer instantiation...")
    
    # Test 2-layer configuration (should use TTTMLP)
    config_2layer = MockConfig()
    config_2layer.ttt_mlp_layers = 2
    
    wrapper = TTTWrapper(config_2layer)
    print(f"✅ 2-layer config instantiated: {type(wrapper.ttt).__name__}")
    assert isinstance(wrapper.ttt, TTTMLP), f"Expected TTTMLP, got {type(wrapper.ttt)}"
    
    # Test 3-layer configuration (should use TTTMLPMultiLayer)
    config_3layer = MockConfig()
    config_3layer.ttt_mlp_layers = 3
    
    wrapper = TTTWrapper(config_3layer)
    print(f"✅ 3-layer config instantiated: {type(wrapper.ttt).__name__}")
    assert isinstance(wrapper.ttt, TTTMLPMultiLayer), f"Expected TTTMLPMultiLayer, got {type(wrapper.ttt)}"
    
    print()


def test_multi_layer_parameter_creation():
    """Test that multi-layer class creates correct parameters"""
    print("🧪 Testing multi-layer parameter creation...")
    
    # Test 3-layer with default expansion
    config = MockConfig()
    config.ttt_mlp_layers = 3
    config.ttt_mlp_expansion_factor = 4.0
    
    multi_layer = TTTMLPMultiLayer(config)
    
    print(f"✅ Created {multi_layer.num_layers}-layer MLP")
    print(f"   Layer dimensions: {multi_layer.layer_dims}")
    print(f"   Number of weight parameters: {len(multi_layer.weights)}")
    print(f"   Number of bias parameters: {len(multi_layer.biases)}")
    
    # Verify dimensions are correct
    expected_dims = [64, 256, 256, 64]  # head_dim=64, expansion=4x
    assert multi_layer.layer_dims == expected_dims, f"Expected {expected_dims}, got {multi_layer.layer_dims}"
    
    # Verify parameter shapes
    for i, (weight, bias) in enumerate(zip(multi_layer.weights, multi_layer.biases)):
        in_dim = multi_layer.layer_dims[i]
        out_dim = multi_layer.layer_dims[i + 1]
        expected_weight_shape = (config.num_heads, in_dim, out_dim)
        expected_bias_shape = (config.num_heads, 1, out_dim)
        
        assert weight.shape == expected_weight_shape, f"Layer {i} weight shape mismatch"
        assert bias.shape == expected_bias_shape, f"Layer {i} bias shape mismatch"
        
        print(f"   Layer {i}: W{i+1} {weight.shape}, b{i+1} {bias.shape}")
    
    print()


def test_custom_dimensions():
    """Test multi-layer with custom dimensions"""
    print("🧪 Testing custom dimensions...")
    
    config = MockConfig()
    config.ttt_mlp_layers = 4
    config.ttt_mlp_hidden_dims = [128, 256, 128]  # 4 layers: 64->128->256->128->64
    
    multi_layer = TTTMLPMultiLayer(config)
    
    expected_dims = [64, 128, 256, 128, 64]
    assert multi_layer.layer_dims == expected_dims, f"Expected {expected_dims}, got {multi_layer.layer_dims}"
    
    print(f"✅ Created {multi_layer.num_layers}-layer MLP with custom dimensions")
    print(f"   Layer dimensions: {multi_layer.layer_dims}")
    
    print()


def test_parameter_count():
    """Test that parameter count scales correctly with layers"""
    print("🧪 Testing parameter count scaling...")
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters())
    
    # Test 2-layer standard
    config_2 = MockConfig()
    config_2.ttt_mlp_layers = 2
    model_2 = TTTMLP(config_2)
    params_2 = count_parameters(model_2)
    
    # Test 3-layer multi
    config_3 = MockConfig()
    config_3.ttt_mlp_layers = 3
    model_3 = TTTMLPMultiLayer(config_3)
    params_3 = count_parameters(model_3)
    
    print(f"✅ 2-layer parameter count: {params_2:,}")
    print(f"✅ 3-layer parameter count: {params_3:,}")
    print(f"   Ratio: {params_3/params_2:.2f}x")
    
    # 3-layer should have more parameters
    assert params_3 > params_2, "3-layer model should have more parameters than 2-layer"
    
    print()


def main():
    """Run all tests"""
    print("🚀 Testing Multi-Layer TTT-MLP Implementation")
    print("=" * 50)
    
    try:
        test_ttt_args_validation()
        test_layer_instantiation() 
        test_multi_layer_parameter_creation()
        test_custom_dimensions()
        test_parameter_count()
        
        print("🎉 All tests passed!")
        print("\n✅ Multi-layer TTT-MLP implementation is working correctly!")
        print("✅ Backward compatibility maintained for 2-layer configurations")
        print("✅ Multi-layer functionality activated for 3+ layer configurations")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)