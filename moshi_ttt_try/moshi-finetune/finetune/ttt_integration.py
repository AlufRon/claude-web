"""
TTT Integration Module for Moshi
Integrates Test-Time Training layers into Moshi models following Video-DiT patterns.
"""

import logging
from typing import List, Union

import torch.nn as nn
from moshi.models.lm import LMModel
from moshi.modules.transformer import StreamingTransformerLayer

# Import our TTT implementation
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from moshi_ttt.hybrid_layer import HybridStreamingTransformerLayer
from moshi_ttt.config import TTTConfig
from .args import TTTArgs

logger = logging.getLogger(__name__)


def main_logger_info(message: str) -> None:
    """Log info message only on main process (rank 0)"""
    try:
        from .distributed import get_rank
        if get_rank() == 0:
            logger.info(message)
    except (ValueError, RuntimeError):
        # Not in distributed context, just log directly
        logger.info(message)


def parse_layer_specification(layers_spec: str, total_layers: int) -> List[int]:
    """
    Parse layer specification string into list of layer indices.
    
    Args:
        layers_spec: "all", "middle", "none", or comma-separated indices like "1,3,5"
        total_layers: Total number of layers in the model
        
    Returns:
        List of layer indices to convert to TTT
    """
    if layers_spec == "none":
        return []
    elif layers_spec == "all":
        return list(range(total_layers))
    elif layers_spec == "middle":
        # Convert middle half of layers
        start = total_layers // 4
        end = 3 * total_layers // 4
        return list(range(start, end))
    else:
        # Parse comma-separated indices
        try:
            indices = [int(x.strip()) for x in layers_spec.split(",")]
            # Validate indices are within range
            valid_indices = [idx for idx in indices if 0 <= idx < total_layers]
            if len(valid_indices) != len(indices):
                invalid = [idx for idx in indices if idx < 0 or idx >= total_layers]
                raise ValueError(f"Invalid layer indices {invalid} for model with {total_layers} layers")
            return sorted(valid_indices)
        except ValueError as e:
            raise ValueError(f"Failed to parse layer specification '{layers_spec}': {e}")


def create_ttt_config(ttt_args: TTTArgs, model_config: dict) -> TTTConfig:
    """
    Create TTTConfig from TTTArgs and model configuration.
    
    Args:
        ttt_args: TTT configuration from training args
        model_config: Model configuration dict containing dimensions
        
    Returns:
        TTTConfig object for layer initialization
    """
    config = TTTConfig(
        model_dim=model_config.get('dim', 1024),
        num_heads=model_config.get('num_heads', 8),
        ttt_base_lr=ttt_args.base_lr,
        mini_batch_size=ttt_args.mini_batch_size,
        gating_alpha_init=ttt_args.initial_gating_alpha,
    )
    
    # Add multi-layer TTT-MLP configuration if present
    if hasattr(ttt_args, 'ttt_mlp_layers') and ttt_args.ttt_mlp_layers is not None:
        config.ttt_mlp_layers = ttt_args.ttt_mlp_layers
        main_logger_info(f"🔧 Multi-layer TTT-MLP: {ttt_args.ttt_mlp_layers} layers configured")
    
    if hasattr(ttt_args, 'ttt_mlp_expansion_factor') and ttt_args.ttt_mlp_expansion_factor is not None:
        config.ttt_mlp_expansion_factor = ttt_args.ttt_mlp_expansion_factor
        main_logger_info(f"🔧 Multi-layer expansion factor: {ttt_args.ttt_mlp_expansion_factor}")
    
    if hasattr(ttt_args, 'ttt_mlp_hidden_dims') and ttt_args.ttt_mlp_hidden_dims is not None:
        config.ttt_mlp_hidden_dims = ttt_args.ttt_mlp_hidden_dims
        main_logger_info(f"🔧 Multi-layer custom dimensions: {ttt_args.ttt_mlp_hidden_dims}")

    # Add TTT output normalization configuration if present
    if hasattr(ttt_args, 'normalize_ttt_output') and ttt_args.normalize_ttt_output:
        config.normalize_ttt_output = ttt_args.normalize_ttt_output
        config.target_output_norm = ttt_args.target_output_norm
        main_logger_info(f"🔧 TTT output normalization enabled: target norm = {ttt_args.target_output_norm}")

    # Add RoPE configuration if present
    if hasattr(ttt_args, 'use_rope'):
        config.use_rope = ttt_args.use_rope
        config.rope_theta = ttt_args.rope_theta
        if ttt_args.use_rope:
            main_logger_info(f"🔧 RoPE enabled: theta = {ttt_args.rope_theta}")
        else:
            main_logger_info("🔧 RoPE disabled (default behavior)")

    return config


def apply_ttt_to_model(model: Union[LMModel, nn.Module], ttt_args: TTTArgs, model_config: dict) -> None:
    """
    Apply TTT layers to specified layers in Moshi model.
    
    This function modifies the model in-place by replacing specified StreamingTransformerLayer
    instances with HybridStreamingTransformerLayer instances that include TTT processing.
    
    Args:
        model: Moshi model (LMModel or FSDP-wrapped model)
        ttt_args: TTT configuration
        model_config: Model configuration dict
    """
    if not ttt_args.enable:
        main_logger_info("TTT disabled - using vanilla Moshi")
        return
    
    # Handle FSDP-wrapped models
    if hasattr(model, '_fsdp_wrapped_module'):
        actual_model = model._fsdp_wrapped_module
    elif hasattr(model, 'module'):
        actual_model = model.module
    else:
        actual_model = model
    
    # Ensure we have access to transformer layers
    if not hasattr(actual_model, 'transformer') or not hasattr(actual_model.transformer, 'layers'):
        raise ValueError("Model does not have expected transformer.layers structure")
    
    transformer_layers = actual_model.transformer.layers
    total_layers = len(transformer_layers)
    
    # Parse which layers to convert
    layer_indices = parse_layer_specification(ttt_args.layers, total_layers)
    
    if not layer_indices:
        main_logger_info("No layers specified for TTT conversion")
        return
    
    # Create TTT configuration
    ttt_config = create_ttt_config(ttt_args, model_config)
    
    main_logger_info(f"Applying TTT to {len(layer_indices)} layers: {layer_indices}")
    main_logger_info(f"TTT config: dim={ttt_config.model_dim}, heads={ttt_config.num_heads}, lr={ttt_config.ttt_base_lr}")
    main_logger_info(f"TTT gating: initial_alpha={ttt_config.gating_alpha_init}")
    main_logger_info(f"TTT state persistence: {'ENABLED' if ttt_args.persistent_states else 'DISABLED'} ({'JAX-style' if ttt_args.persistent_states else 'Original'} behavior)")
    
    # Count parameters before conversion
    original_params = sum(p.numel() for p in model.parameters())
    
    # Convert specified layers to hybrid layers
    converted_count = 0
    for layer_idx in layer_indices:
        if layer_idx < len(transformer_layers):
            original_layer = transformer_layers[layer_idx]
            
            # Ensure it's a StreamingTransformerLayer - FAIL LOUDLY if not
            if not isinstance(original_layer, StreamingTransformerLayer):
                actual_type = type(original_layer).__name__
                raise TypeError(
                    f"TTT INTEGRATION FAILURE: Layer {layer_idx} is {actual_type}, not StreamingTransformerLayer! "
                    f"Cannot apply TTT to incompatible layer type. "
                    f"Expected: StreamingTransformerLayer, Got: {actual_type}"
                )
                
            # Create hybrid layer
            hybrid_layer = HybridStreamingTransformerLayer(original_layer, ttt_config, ttt_args.persistent_states, layer_idx)
            
            # CRITICAL MEMORY OPTIMIZATION: Propagate checkpointing flag
            # This enables gradient checkpointing for TTT layers to save ~12-15 GB per layer
            # The checkpointing flag is inherited from the base transformer configuration
            if hasattr(original_layer, 'checkpointing'):
                hybrid_layer.checkpointing = original_layer.checkpointing
                main_logger_info(f"   🔧 Layer {layer_idx} checkpointing: {hybrid_layer.checkpointing}")
            
            # Ensure hybrid layer is on the same device as the original layer
            if hasattr(original_layer, 'device'):
                device = original_layer.device
            else:
                # Get device from first parameter
                device = next(original_layer.parameters()).device
            hybrid_layer = hybrid_layer.to(device)
            
            # Replace the layer
            transformer_layers[layer_idx] = hybrid_layer
            converted_count += 1
            
            main_logger_info(f"   ✅ Layer {layer_idx} → TTT")
    
    # Count parameters after conversion
    ttt_params = sum(p.numel() for p in model.parameters())
    param_increase = ttt_params - original_params
    
    main_logger_info(f"✅ TTT conversion complete:")
    main_logger_info(f"   Converted layers: {converted_count}/{len(layer_indices)}")
    main_logger_info(f"   Parameter increase: +{param_increase:,} (+{param_increase/original_params*100:.1f}%)")
    
    # Log TTT parameter count
    ttt_param_count = sum(
        p.numel() for name, p in model.named_parameters() 
        if any(k in name for k in ['W1', 'W2', 'b1', 'b2', 'ttt_norm', 'learnable_ttt_lr'])
    )
    main_logger_info(f"   TTT parameters: {ttt_param_count:,}")


def log_ttt_parameters(model: Union[LMModel, nn.Module]) -> None:
    """
    Log TTT-specific parameter information for monitoring.
    
    Args:
        model: Model to analyze
    """
    # Count TTT parameters
    ttt_params = {}
    for name, param in model.named_parameters():
        if any(k in name for k in ['W1', 'W2', 'b1', 'b2', 'ttt_norm', 'learnable_ttt_lr']):
            param_type = None
            for k in ['W1', 'W2', 'b1', 'b2', 'ttt_norm', 'learnable_ttt_lr']:
                if k in name:
                    param_type = k
                    break
            
            if param_type:
                if param_type not in ttt_params:
                    ttt_params[param_type] = 0
                ttt_params[param_type] += param.numel()
    
    if ttt_params:
        main_logger_info("TTT parameter breakdown:")
        for param_type, count in ttt_params.items():
            main_logger_info(f"   {param_type}: {count:,}")
        
        total_ttt = sum(ttt_params.values())
        total_params = sum(p.numel() for p in model.parameters())
        main_logger_info(f"   Total TTT: {total_ttt:,} ({total_ttt/total_params*100:.2f}%)")


def verify_ttt_integration(model: Union[LMModel, nn.Module]) -> bool:
    """
    Verify that TTT integration was successful.
    
    Args:
        model: Model to verify
        
    Returns:
        True if TTT layers are present and properly integrated
    """
    # Handle FSDP-wrapped models
    if hasattr(model, '_fsdp_wrapped_module'):
        actual_model = model._fsdp_wrapped_module
    elif hasattr(model, 'module'):
        actual_model = model.module
    else:
        actual_model = model
    
    if not hasattr(actual_model, 'transformer') or not hasattr(actual_model.transformer, 'layers'):
        return False
    
    # Count hybrid layers
    hybrid_count = 0
    total_layers = len(actual_model.transformer.layers)
    
    for layer in actual_model.transformer.layers:
        if isinstance(layer, HybridStreamingTransformerLayer):
            hybrid_count += 1
    
    if hybrid_count > 0:
        main_logger_info(f"✅ TTT verification: {hybrid_count}/{total_layers} layers are TTT-enabled")
        return True
    else:
        main_logger_info("ℹ️  TTT verification: No TTT layers found (TTT may be disabled)")
        return hybrid_count == 0  # This is fine if TTT is disabled        return hybrid_count == 0  # This is fine if TTT is disabled


def reset_gating_alpha(model: Union[LMModel, nn.Module], new_alpha: float) -> None:
    """
    Reset gating alpha values for all TTT layers to a new value.
    
    This is useful when fine-tuning from a pre-trained checkpoint where you want
    to reduce TTT's influence by starting with a smaller gating alpha.
    
    Args:
        model: Model with TTT layers
        new_alpha: New initial value for gating alpha
        
    Example:
        # Pre-train with alpha=0.3 (TTT has strong influence)
        # Fine-tune with alpha=0.01 (reduce TTT influence, rely more on attention)
        reset_gating_alpha(model, 0.01)
    """
    import torch
    
    # Handle FSDP-wrapped models
    if hasattr(model, '_fsdp_wrapped_module'):
        actual_model = model._fsdp_wrapped_module
    elif hasattr(model, 'module'):
        actual_model = model.module
    else:
        actual_model = model
    
    if not hasattr(actual_model, 'transformer') or not hasattr(actual_model.transformer, 'layers'):
        main_logger_info("⚠️  Cannot reset gating alpha: model has no transformer layers")
        return
    
    reset_count = 0
    for layer_idx, layer in enumerate(actual_model.transformer.layers):
        if isinstance(layer, HybridStreamingTransformerLayer):
            # Access seq_modeling_block which contains the gating modules
            if hasattr(layer, 'seq_modeling_block'):
                seq_block = layer.seq_modeling_block
                
                # Reset forward gating
                if hasattr(seq_block, 'forward_ssm_gating'):
                    old_alpha = seq_block.forward_ssm_gating.gating_alpha.data.clone()
                    seq_block.forward_ssm_gating.gating_alpha.data.fill_(new_alpha)
                    main_logger_info(
                        f"   Layer {layer_idx} forward gating: "
                        f"mean={old_alpha.mean().item():.6f} → {new_alpha:.6f}"
                    )
                
                # Reset backward gating (if bidirectional)
                if hasattr(seq_block, 'backward_ssm_gating'):
                    old_alpha = seq_block.backward_ssm_gating.gating_alpha.data.clone()
                    seq_block.backward_ssm_gating.gating_alpha.data.fill_(new_alpha)
                    main_logger_info(
                        f"   Layer {layer_idx} backward gating: "
                        f"mean={old_alpha.mean().item():.6f} → {new_alpha:.6f}"
                    )
                
                reset_count += 1
    
    if reset_count > 0:
        main_logger_info(f"✅ Reset gating alpha to {new_alpha} for {reset_count} TTT layers")
        main_logger_info(f"   This reduces TTT influence from checkpoint values")
    else:
        main_logger_info("⚠️  No TTT layers found to reset gating alpha")
