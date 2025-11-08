"""
TTT Integration with Llama Layers

This module handles the integration of TTT layers into Llama's transformer architecture.
It provides clean, minimal changes that maintain backward compatibility.

Design principles:
1. Non-invasive: Works with existing Llama code
2. Configurable: Easily enable/disable TTT
3. Selective: Replace only specified layers
4. Compatible: Works with existing training/inference code

Integration happens in 3 steps:
1. Check config for TTT settings
2. Replace attention in specified layers with TTT
3. Initialize TTT parameters properly

The replacement is clean - we swap out the self_attn module while keeping
everything else (MLP, LayerNorm, etc.) unchanged.
"""

import logging
from typing import Optional, List
import torch.nn as nn

from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from .ttt_layer import TTTLinearLayer, TTTMLPLayer
from .logger import setup_ttt_logging

logger = logging.getLogger(__name__)


def should_use_ttt_for_layer(config, layer_idx: int) -> bool:
    """
    Determine if a specific layer should use TTT.

    Args:
        config: Model configuration
        layer_idx: 0-indexed layer number

    Returns:
        True if this layer should use TTT

    Logic:
        - If use_ttt is False, return False for all layers
        - If ttt_layer_indices is specified, check if layer_idx is in the list
        - Otherwise, default to top 8 layers
    """
    if not config.use_ttt:
        return False

    if config.ttt_layer_indices is not None:
        return layer_idx in config.ttt_layer_indices

    # Default: top 8 layers
    total_layers = config.num_hidden_layers
    return layer_idx >= (total_layers - 8)


def replace_attention_with_ttt(
    layer: LlamaDecoderLayer,
    layer_idx: int,
    config
) -> LlamaDecoderLayer:
    """
    Replace a Llama layer's attention module with TTT.

    This is the core integration function. It:
    1. Creates a TTT layer with the same dimensions
    2. Replaces the self_attn module
    3. Keeps everything else unchanged (MLP, LayerNorm, etc.)

    Args:
        layer: Original LlamaDecoderLayer
        layer_idx: Index of this layer
        config: Model configuration

    Returns:
        Modified layer with TTT instead of attention

    The modification is clean:
        Before:
            layer.self_attn = LlamaSdpaAttention(...)
            layer.mlp = LlamaMLP(...)
            layer.input_layernorm = RMSNorm(...)
            layer.post_attention_layernorm = RMSNorm(...)

        After:
            layer.self_attn = TTTLinearLayer(...)  # <-- CHANGED
            layer.mlp = LlamaMLP(...)               # unchanged
            layer.input_layernorm = RMSNorm(...)    # unchanged
            layer.post_attention_layernorm = RMSNorm(...)  # unchanged
    """
    logger.info(f"[TTT Integration] Replacing attention in layer {layer_idx} with TTT-{config.ttt_layer_type}")

    # Create TTT layer
    if config.ttt_layer_type == "ttt_linear":
        ttt_layer = TTTLinearLayer(config, layer_idx=layer_idx)
    elif config.ttt_layer_type == "ttt_mlp":
        ttt_layer = TTTMLPLayer(config, layer_idx=layer_idx)
    else:
        raise ValueError(f"Unknown TTT layer type: {config.ttt_layer_type}")

    # Initialize TTT weights
    ttt_layer.init_weights()

    # Replace self_attn with TTT
    # CRITICAL: We keep the name 'self_attn' so that Llama's forward pass works unchanged
    layer.self_attn = ttt_layer

    logger.debug(
        f"[TTT Integration] Layer {layer_idx} now uses {ttt_layer.__class__.__name__}"
    )

    return layer


def integrate_ttt_into_model(model, config) -> None:
    """
    Integrate TTT into a Llama model's specified layers.

    This is the main entry point for TTT integration. It:
    1. Checks if TTT is enabled
    2. Determines which layers to modify
    3. Replaces attention with TTT in those layers
    4. Sets up logging

    Args:
        model: LlamaModel instance (or subclass like OmniSpeechLlamaModel)
        config: Model configuration with TTT settings

    This function modifies the model in-place.

    Usage:
        model = OmniSpeechLlamaForCausalLM(config)
        if config.use_ttt:
            integrate_ttt_into_model(model.model, config)  # model.model is the LlamaModel
    """
    if not config.use_ttt:
        logger.info("[TTT Integration] TTT disabled in config, skipping integration")
        return

    logger.info("=" * 80)
    logger.info("[TTT Integration] Starting TTT integration into Llama layers")
    logger.info("=" * 80)

    # Get the actual Llama layers
    # For OmniSpeechLlamaModel, layers are at model.layers
    if hasattr(model, 'layers'):
        layers = model.layers
    else:
        raise AttributeError(
            f"Model {type(model)} doesn't have 'layers' attribute. "
            "Expected LlamaModel or similar structure."
        )

    # Count how many layers will use TTT
    ttt_layer_count = 0
    ttt_layer_indices = []

    for layer_idx in range(len(layers)):
        if should_use_ttt_for_layer(config, layer_idx):
            ttt_layer_count += 1
            ttt_layer_indices.append(layer_idx)

    logger.info(
        f"[TTT Integration] Will replace {ttt_layer_count}/{len(layers)} layers with TTT"
    )
    logger.info(f"[TTT Integration] TTT layer indices: {ttt_layer_indices}")

    # Replace attention with TTT in specified layers
    for layer_idx in ttt_layer_indices:
        layers[layer_idx] = replace_attention_with_ttt(
            layers[layer_idx],
            layer_idx,
            config
        )

    logger.info("=" * 80)
    logger.info(f"[TTT Integration] Successfully integrated TTT into {ttt_layer_count} layers")
    logger.info("=" * 80)

    # Setup logging if enabled
    if config.ttt_enable_logging:
        log_dir = config.ttt_csv_log_path if config.ttt_csv_log_path else "./ttt_logs"
        ttt_logger, csv_logger = setup_ttt_logging(
            log_dir=log_dir,
            log_level=config.ttt_log_level,
            enable_csv=True,
            csv_flush_interval=config.ttt_log_interval
        )

        # Store loggers in model for access during forward pass
        # (This is a bit hacky, but keeps things simple)
        if not hasattr(model, '_ttt_loggers'):
            model._ttt_loggers = {}
        model._ttt_loggers['logger'] = ttt_logger
        model._ttt_loggers['csv_logger'] = csv_logger

        logger.info(f"[TTT Integration] Logging enabled: log_dir={log_dir}")


def verify_ttt_integration(model) -> dict:
    """
    Verify that TTT integration was successful.

    Checks:
    1. Which layers use TTT
    2. Parameter counts
    3. Dtype of TTT states

    Args:
        model: Integrated model

    Returns:
        Dictionary with verification results
    """
    results = {
        "ttt_layers": [],
        "standard_layers": [],
        "total_ttt_params": 0,
        "w1_b1_dtype": None,
        "integration_successful": True,
        "errors": []
    }

    if not hasattr(model, 'layers'):
        results["integration_successful"] = False
        results["errors"].append("Model doesn't have 'layers' attribute")
        return results

    for layer_idx, layer in enumerate(model.layers):
        # Check if self_attn is a TTT layer
        if isinstance(layer.self_attn, (TTTLinearLayer, TTTMLPLayer)):
            results["ttt_layers"].append(layer_idx)

            # Count parameters
            ttt_params = sum(p.numel() for p in layer.self_attn.parameters())
            results["total_ttt_params"] += ttt_params

            # Check dtype of W1, b1
            if hasattr(layer.self_attn, 'W1'):
                w1_dtype = layer.self_attn.W1.dtype
                if results["w1_b1_dtype"] is None:
                    results["w1_b1_dtype"] = str(w1_dtype)

                # Verify float32
                if w1_dtype != torch.float32:
                    results["integration_successful"] = False
                    results["errors"].append(
                        f"Layer {layer_idx}: W1 is {w1_dtype}, expected torch.float32"
                    )
        else:
            results["standard_layers"].append(layer_idx)

    logger.info("=" * 80)
    logger.info("[TTT Verification] Integration verification results:")
    logger.info(f"  TTT layers: {results['ttt_layers']}")
    logger.info(f"  Standard layers: {results['standard_layers']}")
    logger.info(f"  Total TTT parameters: {results['total_ttt_params']:,}")
    logger.info(f"  W1/b1 dtype: {results['w1_b1_dtype']}")
    logger.info(f"  Integration successful: {results['integration_successful']}")

    if results["errors"]:
        logger.error(f"  Errors: {results['errors']}")

    logger.info("=" * 80)

    return results


# Example usage documentation
_USAGE_EXAMPLE = """
# Example: Integrating TTT into Llama-Omni

from omni_speech.model.language_model.omni_speech_llama import (
    OmniSpeechConfig,
    OmniSpeechLlamaForCausalLM
)
from omni_speech.model.ttt.integration import (
    integrate_ttt_into_model,
    verify_ttt_integration
)

# 1. Create config with TTT enabled
config = OmniSpeechConfig(
    # Standard Llama config
    hidden_size=4096,
    num_hidden_layers=32,
    num_attention_heads=32,

    # TTT config
    use_ttt=True,
    ttt_layer_type="ttt_linear",
    ttt_mini_batch_size=64,
    ttt_layer_indices=[24, 25, 26, 27, 28, 29, 30, 31],  # Top 8 layers
    ttt_base_lr=1.0,
    ttt_enable_logging=True,
    ttt_log_level="INFO",
)

# 2. Create model (standard way)
model = OmniSpeechLlamaForCausalLM(config)

# 3. Integrate TTT (happens automatically in __init__ if use_ttt=True)
# But you can also do it manually:
# integrate_ttt_into_model(model.model, config)

# 4. Verify integration
verification = verify_ttt_integration(model.model)
assert verification["integration_successful"]

# 5. Use model normally
# TTT will be used automatically in specified layers
outputs = model.generate(...)
"""
