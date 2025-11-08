"""
Llama-Omni TTT Integration

Utilities for integrating TTT layers into Llama-Omni model.
Replaces standard self-attention in top 8 layers (24-31) with TTT-MLP.

Usage:
    from ttt_modules.llama_omni_integration import convert_llama_to_ttt

    model = OmniSpeechLlamaForCausalLM.from_pretrained(...)
    model = convert_llama_to_ttt(model, ttt_layers=[24, 25, 26, 27, 28, 29, 30, 31])

Based on Doc 13 specifications.
"""

import torch
import torch.nn as nn
from typing import List, Optional
import logging

from ttt_modules.ttt_layer import TTTMLP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_llama_to_ttt(
    model: nn.Module,
    ttt_layers: List[int] = None,
    mini_batch_size: int = 64,
    ttt_base_lr: float = 1.0,
    verify_replacement: bool = True,
) -> nn.Module:
    """
    Convert Llama model to use TTT in specified layers.

    Replaces self_attn in specified layers with TTTMLP layers.

    Args:
        model: Llama model (e.g., OmniSpeechLlamaForCausalLM)
        ttt_layers: List of layer indices to convert (default: [24-31])
        mini_batch_size: TTT mini-batch size (must divide sequence length)
        ttt_base_lr: Base learning rate for TTT updates
        verify_replacement: Verify TTT parameters are FP32 after replacement

    Returns:
        Modified model with TTT layers

    Example:
        >>> from transformers import AutoModel
        >>> model = AutoModel.from_pretrained("ictnlp/Llama-Omni")
        >>> model = convert_llama_to_ttt(model, ttt_layers=[24, 25, 26, 27, 28, 29, 30, 31])
        >>> # Now layers 24-31 use TTT instead of attention
    """

    if ttt_layers is None:
        # Default: Top 8 layers (24-31) for 32-layer Llama
        ttt_layers = list(range(24, 32))

    logger.info(f"Converting {len(ttt_layers)} layers to TTT: {ttt_layers}")

    # Get model config
    config = model.config

    # Add TTT-specific config attributes
    config.mini_batch_size = mini_batch_size
    config.ttt_base_lr = ttt_base_lr

    # Access Llama layers
    # For Llama-Omni: model.model.layers
    # For HuggingFace Llama: model.model.layers
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
    elif hasattr(model, 'layers'):
        layers = model.layers
    else:
        raise AttributeError(
            "Cannot find model layers. Expected model.model.layers or model.layers"
        )

    total_layers = len(layers)
    logger.info(f"Model has {total_layers} layers")

    # Verify layer indices are valid
    for layer_idx in ttt_layers:
        if layer_idx < 0 or layer_idx >= total_layers:
            raise ValueError(
                f"Invalid layer index {layer_idx}. "
                f"Model has {total_layers} layers (0-{total_layers-1})"
            )

    # Replace layers
    replaced_count = 0
    for layer_idx in ttt_layers:
        layer = layers[layer_idx]

        # Verify layer has self_attn
        if not hasattr(layer, 'self_attn'):
            logger.warning(
                f"Layer {layer_idx} has no self_attn attribute. Skipping."
            )
            continue

        # Create TTT layer
        ttt_layer = TTTMLP(config, layer_idx=layer_idx)

        # Copy Q, K, V, O weights from attention layer if available
        if hasattr(layer.self_attn, 'q_proj'):
            logger.info(f"  Copying Q/K/V/O weights from layer {layer_idx}")
            try:
                ttt_layer.wq.weight.data = layer.self_attn.q_proj.weight.data.clone()
                ttt_layer.wk.weight.data = layer.self_attn.k_proj.weight.data.clone()
                ttt_layer.wv.weight.data = layer.self_attn.v_proj.weight.data.clone()
                ttt_layer.wo.weight.data = layer.self_attn.o_proj.weight.data.clone()

                # Copy biases if present
                if hasattr(layer.self_attn.q_proj, 'bias') and layer.self_attn.q_proj.bias is not None:
                    ttt_layer.wq.bias.data = layer.self_attn.q_proj.bias.data.clone()
                    ttt_layer.wk.bias.data = layer.self_attn.k_proj.bias.data.clone()
                    ttt_layer.wv.bias.data = layer.self_attn.v_proj.bias.data.clone()
                    ttt_layer.wo.bias.data = layer.self_attn.o_proj.bias.data.clone()

            except Exception as e:
                logger.warning(f"  Could not copy weights: {e}")

        # Replace self_attn with TTT layer
        layer.self_attn = ttt_layer

        logger.info(f"✅ Replaced layer {layer_idx} with TTT")
        replaced_count += 1

    logger.info(f"Successfully replaced {replaced_count}/{len(ttt_layers)} layers")

    # Verify FP32 precision
    if verify_replacement:
        logger.info("Verifying FP32 precision...")
        verify_ttt_fp32(model, ttt_layers)

    return model


def verify_ttt_fp32(model: nn.Module, ttt_layers: List[int]):
    """
    Verify all TTT parameters are FP32.

    Args:
        model: Model with TTT layers
        ttt_layers: List of TTT layer indices

    Raises:
        TypeError: If any TTT parameter is not FP32
    """

    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
    else:
        layers = model.layers

    all_good = True
    for layer_idx in ttt_layers:
        layer = layers[layer_idx]

        if not isinstance(layer.self_attn, TTTMLP):
            logger.warning(f"Layer {layer_idx} is not a TTT layer!")
            all_good = False
            continue

        ttt_layer = layer.self_attn

        # Check critical parameters
        params_to_check = {
            'W1': ttt_layer.W1,
            'b1': ttt_layer.b1,
            'W2': ttt_layer.W2,
            'b2': ttt_layer.b2,
        }

        for param_name, param in params_to_check.items():
            if param.dtype != torch.float32:
                logger.error(
                    f"❌ Layer {layer_idx} {param_name}: {param.dtype} (expected torch.float32)"
                )
                all_good = False

    if all_good:
        logger.info("✅ All TTT parameters are FP32")
    else:
        raise TypeError(
            "Some TTT parameters are not FP32! This will cause numerical instability. "
            "Check the logs above for details."
        )


def create_ttt_param_groups(
    model: nn.Module,
    ttt_layers: List[int],
    ttt_lr: float = 1e-4,
    other_lr: float = 2e-5,
) -> List[dict]:
    """
    Create optimizer parameter groups with separate learning rates for TTT params.

    CRITICAL: TTT parameters (W1, b1, W2, b2) should have different LR than
    other parameters (Q, K, V, O projections and LR gates).

    Args:
        model: Model with TTT layers
        ttt_layers: List of TTT layer indices
        ttt_lr: Learning rate for W1, b1, W2, b2
        other_lr: Learning rate for all other parameters

    Returns:
        List of parameter groups for optimizer

    Example:
        >>> param_groups = create_ttt_param_groups(model, ttt_layers=[24-31])
        >>> optimizer = torch.optim.AdamW(param_groups)
    """

    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
    else:
        layers = model.layers

    # TTT state parameters (W1, b1, W2, b2) - MUST be FP32
    ttt_state_params = []

    # TTT projection parameters (Q, K, V, O, LR gates)
    ttt_proj_params = []

    # All other parameters
    other_params = []

    # Collect TTT parameters
    ttt_layer_names = set()
    for layer_idx in ttt_layers:
        layer = layers[layer_idx]

        if isinstance(layer.self_attn, TTTMLP):
            ttt_layer = layer.self_attn

            # State parameters (FP32)
            ttt_state_params.extend([
                ttt_layer.W1,
                ttt_layer.b1,
                ttt_layer.W2,
                ttt_layer.b2,
                ttt_layer.ttt_norm_weight,
                ttt_layer.ttt_norm_bias,
            ])

            # Projection parameters
            ttt_proj_params.extend([
                *ttt_layer.wq.parameters(),
                *ttt_layer.wk.parameters(),
                *ttt_layer.wv.parameters(),
                *ttt_layer.wo.parameters(),
                ttt_layer.learnable_ttt_lr_weight,
                ttt_layer.learnable_ttt_lr_bias,
                *ttt_layer.post_norm.parameters(),
            ])

            # Track which layers are TTT
            ttt_layer_names.add(f"model.layers.{layer_idx}.self_attn")

    # Collect all other parameters
    for name, param in model.named_parameters():
        # Skip if already in TTT params
        is_ttt = any(ttt_name in name for ttt_name in ttt_layer_names)
        if not is_ttt:
            other_params.append(param)

    param_groups = [
        {
            'params': ttt_state_params,
            'lr': ttt_lr,
            'name': 'ttt_states',
            'weight_decay': 0.0,  # No weight decay for TTT states
        },
        {
            'params': ttt_proj_params,
            'lr': other_lr,
            'name': 'ttt_projections',
            'weight_decay': 0.01,
        },
        {
            'params': other_params,
            'lr': other_lr,
            'name': 'other',
            'weight_decay': 0.01,
        },
    ]

    logger.info(f"Parameter groups:")
    logger.info(f"  TTT states: {len(ttt_state_params)} params, lr={ttt_lr}")
    logger.info(f"  TTT projections: {len(ttt_proj_params)} params, lr={other_lr}")
    logger.info(f"  Other: {len(other_params)} params, lr={other_lr}")

    return param_groups


def setup_ttt_fp32_hooks(model: nn.Module, ttt_layers: List[int]):
    """
    Setup hooks to ensure TTT parameters stay FP32 during training.

    CRITICAL: Mixed precision training can convert FP32 params to FP16/BF16.
    These hooks prevent that for TTT states.

    Args:
        model: Model with TTT layers
        ttt_layers: List of TTT layer indices

    Example:
        >>> model = convert_llama_to_ttt(model)
        >>> setup_ttt_fp32_hooks(model, ttt_layers=[24-31])
        >>> # Now TTT states will stay FP32 even with mixed precision
    """

    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
    else:
        layers = model.layers

    def check_fp32_hook(module, input, output):
        """Hook to verify FP32 precision."""
        if not isinstance(module, TTTMLP):
            return

        # Check and fix if needed
        if module.W1.dtype != torch.float32:
            logger.warning(f"TTT layer W1 was converted to {module.W1.dtype}! Converting back to FP32...")
            module.W1.data = module.W1.data.float()
            module.b1.data = module.b1.data.float()
            module.W2.data = module.W2.data.float()
            module.b2.data = module.b2.data.float()

    # Register hooks
    for layer_idx in ttt_layers:
        layer = layers[layer_idx]
        if isinstance(layer.self_attn, TTTMLP):
            layer.self_attn.register_forward_hook(check_fp32_hook)
            logger.info(f"Registered FP32 hook for layer {layer_idx}")


if __name__ == "__main__":
    """Test TTT integration."""
    print("Testing Llama-Omni TTT Integration")
    print("=" * 60)

    # Mock Llama model
    class MockConfig:
        hidden_size = 4096
        num_attention_heads = 32
        num_hidden_layers = 32
        intermediate_size = 11008
        rope_theta = 10000.0
        attention_bias = False
        rms_norm_eps = 1e-6

    class MockAttention(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    class MockLayer(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.self_attn = MockAttention(config)
            self.mlp = nn.Linear(config.hidden_size, config.hidden_size)

    class MockModel(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.layers = nn.ModuleList([MockLayer(config) for _ in range(config.num_hidden_layers)])

    class MockLlamaOmni(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.model = MockModel(config)

    # Create mock model
    config = MockConfig()
    model = MockLlamaOmni(config)

    print(f"Created mock model with {len(model.model.layers)} layers")

    # Convert top 8 layers to TTT
    ttt_layer_indices = [24, 25, 26, 27, 28, 29, 30, 31]
    model = convert_llama_to_ttt(
        model,
        ttt_layers=ttt_layer_indices,
        mini_batch_size=64
    )

    print("\nVerifying conversion:")
    for idx in ttt_layer_indices:
        is_ttt = isinstance(model.model.layers[idx].self_attn, TTTMLP)
        print(f"  Layer {idx}: {'TTT ✅' if is_ttt else 'Attention ❌'}")

    # Create parameter groups
    param_groups = create_ttt_param_groups(
        model,
        ttt_layers=ttt_layer_indices,
        ttt_lr=1e-4,
        other_lr=2e-5
    )

    print("\n✅ TTT integration test complete!")
