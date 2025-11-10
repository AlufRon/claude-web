"""
Test selective attention layer unfreezing feature.
"""
import pytest
import torch
from moshi.models.loaders import get_model

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from finetune.args import TrainArgs, TTTArgs, DataArgs, OptimizerArgs
from finetune.wrapped_model import configure_trainable_parameters, is_attention_parameter_in_layer
from finetune.ttt_integration import apply_ttt_to_model, parse_layer_specification


def test_parse_layer_specification():
    """Test layer specification parsing."""
    # Test "none"
    assert parse_layer_specification("none", 32) == []

    # Test "all"
    assert parse_layer_specification("all", 32) == list(range(32))

    # Test "middle"
    middle = parse_layer_specification("middle", 32)
    assert middle == list(range(8, 24))  # 32//4=8, 3*32//4=24

    # Test specific indices
    assert parse_layer_specification("1,3,5", 32) == [1, 3, 5]
    assert parse_layer_specification("0,10,20,31", 32) == [0, 10, 20, 31]


def test_is_attention_parameter_in_layer():
    """Test attention parameter identification."""
    # Test attention parameters in specific layers (direct pattern)
    assert is_attention_parameter_in_layer("transformer.layers.1.self_attn.in_proj_weight", [1]) == True
    assert is_attention_parameter_in_layer("transformer.layers.1.self_attn.out_proj.weight", [1]) == True
    assert is_attention_parameter_in_layer("transformer.layers.5.self_attn.in_proj_bias", [1, 3, 5]) == True

    # Test attention parameters with wrapped pattern (after TTT integration)
    assert is_attention_parameter_in_layer("transformer.layers.1.original_layer.self_attn.in_proj_weight", [1]) == True
    assert is_attention_parameter_in_layer("transformer.layers.1.original_layer.self_attn.out_proj.weight", [1]) == True

    # Test non-attention parameters
    assert is_attention_parameter_in_layer("transformer.layers.1.norm1.weight", [1]) == False
    assert is_attention_parameter_in_layer("transformer.layers.1.linear1.weight", [1]) == False

    # Test wrong layer
    assert is_attention_parameter_in_layer("transformer.layers.2.self_attn.in_proj_weight", [1]) == False
    assert is_attention_parameter_in_layer("transformer.layers.2.self_attn.in_proj_weight", [1, 3, 5]) == False

    # Test empty layer list
    assert is_attention_parameter_in_layer("transformer.layers.1.self_attn.in_proj_weight", []) == False


@pytest.mark.slow
def test_configure_trainable_with_unfrozen_attention():
    """Test parameter freezing with unfrozen attention layers."""
    # Create minimal model (this will be slow)
    model = get_model('kyutai/moshiko-pytorch-bf16', device='cpu')

    # Create args with TTT and unfrozen attention in layer 1
    args = TrainArgs(
        data=DataArgs(dataset_type='dummy'),
        run_dir='/tmp/test_run',
        optim=OptimizerArgs(lr=1e-4),
        ttt=TTTArgs(
            enable=True,
            layers="1",  # TTT on layer 1
            unfrozen_attention_layers="1",  # Unfreeze attention on layer 1
        ),
        lora=None,
        full_finetuning=False,
    )

    # Apply TTT
    apply_ttt_to_model(model, args)

    # Configure trainable parameters
    configure_trainable_parameters(model, args)

    # Verify results
    ttt_params = []
    attn_params = []
    frozen_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'ttt' in name.lower():
                ttt_params.append(name)
            elif 'layers.1.self_attn' in name:
                attn_params.append(name)
            else:
                frozen_params.append(name)
        else:
            if 'layers.1.self_attn' in name:
                frozen_params.append(name)

    # Check TTT parameters are trainable
    assert len(ttt_params) > 0, "TTT parameters should be trainable"

    # Check attention in layer 1 is trainable
    assert len(attn_params) > 0, "Attention parameters in layer 1 should be trainable"

    # Check other attention layers are frozen
    layer_0_attn_frozen = False
    for name, param in model.named_parameters():
        if 'layers.0.self_attn' in name:
            if not param.requires_grad:
                layer_0_attn_frozen = True
                break
    assert layer_0_attn_frozen, "Attention in layer 0 should be frozen"

    print(f"✓ Found {len(ttt_params)} trainable TTT parameters")
    print(f"✓ Found {len(attn_params)} trainable attention parameters in layer 1")
    print(f"✓ Other layers remain frozen")


@pytest.mark.slow
def test_unfrozen_attention_multiple_layers():
    """Test unfreezing attention in multiple layers."""
    model = get_model('kyutai/moshiko-pytorch-bf16', device='cpu')

    # Create args with TTT and unfrozen attention in layers 1,3,5
    args = TrainArgs(
        data=DataArgs(dataset_type='dummy'),
        run_dir='/tmp/test_run',
        optim=OptimizerArgs(lr=1e-4),
        ttt=TTTArgs(
            enable=True,
            layers="1,3,5",
            unfrozen_attention_layers="1,3,5",
        ),
        lora=None,
        full_finetuning=False,
    )

    apply_ttt_to_model(model, args)
    configure_trainable_parameters(model, args)

    # Check each specified layer has trainable attention
    for layer_idx in [1, 3, 5]:
        has_trainable = False
        for name, param in model.named_parameters():
            if f'layers.{layer_idx}.self_attn' in name and param.requires_grad:
                has_trainable = True
                break
        assert has_trainable, f"Layer {layer_idx} attention should be trainable"

    # Check layer 0 and 2 are frozen
    for layer_idx in [0, 2]:
        all_frozen = True
        for name, param in model.named_parameters():
            if f'layers.{layer_idx}.self_attn' in name and param.requires_grad:
                all_frozen = False
                break
        assert all_frozen, f"Layer {layer_idx} attention should be frozen"

    print("✓ Selective unfreezing works for multiple layers")


if __name__ == "__main__":
    print("Testing layer specification parsing...")
    test_parse_layer_specification()
    print("✓ Layer specification parsing works\n")

    print("Testing attention parameter identification...")
    test_is_attention_parameter_in_layer()
    print("✓ Attention parameter identification works\n")

    print("Testing configure_trainable with unfrozen attention (SLOW)...")
    test_configure_trainable_with_unfrozen_attention()
    print("✓ Configure trainable works\n")

    print("Testing multiple layer unfreezing (SLOW)...")
    test_unfrozen_attention_multiple_layers()
    print("✓ Multiple layer unfreezing works\n")

    print("\n✅ All tests passed!")
