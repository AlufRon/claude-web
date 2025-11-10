"""
TTT Parameter Detection and Initialization Utilities

This module provides centralized utilities for:
- Detecting TTT parameters in model
- Initializing TTT parameters properly
- Managing different training modes
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


def is_ttt_parameter(param_name: str) -> bool:
    """
    Check if parameter belongs to TTT layers.
    
    Args:
        param_name: Full parameter name (e.g., "transformer.layers.31.gating_alpha")
        
    Returns:
        bool: True if parameter is TTT-related
    """
    # Convert to lowercase for case-insensitive matching
    name_lower = param_name.lower()
    
    # TTT-specific parameter patterns
    ttt_patterns = [
        "gating_alpha",           # SSM gating parameter
        "ttt_norm_weight",        # TTT layer norm weight
        "ttt_norm_bias",          # TTT layer norm bias
        "learnable_ttt_lr_weight", # Learnable TTT learning rate weight
        "learnable_ttt_lr_bias",   # Learnable TTT learning rate bias
        "wq.weight",              # TTT query projection weight
        "wq.bias",                # TTT query projection bias
        "wk.weight",              # TTT key projection weight
        "wk.bias",                # TTT key projection bias
        "wv.weight",              # TTT value projection weight
        "wv.bias",                # TTT value projection bias
        "wo.weight",              # TTT output projection weight
        "wo.bias",                # TTT output projection bias
        "w1",                     # TTT MLP first layer (parameter, not nn.Linear)
        "w2",                     # TTT MLP second layer (parameter, not nn.Linear)
        "b1",                     # TTT MLP first bias
        "b2",                     # TTT MLP second bias
        "weights.",               # Multi-layer TTT-MLP weights (weights.0, weights.1, weights.2, etc.)
        "biases.",                # Multi-layer TTT-MLP biases (biases.0, biases.1, biases.2, etc.)
        "post_norm.weight",       # TTT post normalization weight
        "post_norm.bias",         # TTT post normalization bias
    ]
    
    # Check if any TTT pattern matches
    for pattern in ttt_patterns:
        if pattern in name_lower:
            return True
    
    # Additional check for hybrid layer context
    if "hybridseqmodelingblock" in name_lower or "ttt_block" in name_lower:
        return True

    return False


def classify_ttt_parameter(param_name: str) -> str:
    """
    Classify TTT parameter into learning rate group.

    This enables multi-learning-rate optimization where different parameter types
    get different learning rates:
    - 'ttt_alpha': Gating alpha parameters (need high LR to grow from 0.005 → 0.5)
    - 'ttt_weights': TTT projection/MLP weights (need moderate LR, randomly initialized)
    - 'base': Everything else (LoRA, embeddings - need very low LR, pretrained)

    Args:
        param_name: Full parameter name (e.g., "transformer.layers.31.forward_ssm_gating.gating_alpha")

    Returns:
        str: 'ttt_alpha', 'ttt_weights', or 'base'
    """
    name_lower = param_name.lower()

    # Highest priority: Gating alpha (needs 1000x higher LR)
    if 'gating_alpha' in name_lower:
        return 'ttt_alpha'

    # TTT weights (need 10x higher LR than base)
    if is_ttt_parameter(param_name):
        return 'ttt_weights'

    # Everything else (LoRA, embeddings, etc.)
    return 'base'


def get_parameter_groups(model: torch.nn.Module) -> Dict[str, List[torch.nn.Parameter]]:
    """
    Group model parameters by learning rate requirements.

    This is used to create an optimizer with different learning rates for
    different parameter types, solving the TTT training instability problem.

    Args:
        model: The model to extract parameters from

    Returns:
        Dict with keys: 'base', 'ttt_weights', 'ttt_alpha'
        Each value is a list of parameters

    Example:
        >>> groups = get_parameter_groups(model)
        >>> print(f"Base: {len(groups['base'])} params")
        >>> print(f"TTT weights: {len(groups['ttt_weights'])} params")
        >>> print(f"TTT alpha: {len(groups['ttt_alpha'])} params")
    """
    groups = {
        'base': [],
        'ttt_weights': [],
        'ttt_alpha': []
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        group_name = classify_ttt_parameter(name)
        groups[group_name].append(param)

    return groups


def initialize_ttt_parameter(param: torch.nn.Parameter, param_name: str, gating_alpha_init: float = 0.05) -> None:
    """
    Initialize TTT parameter based on its type following Video-DiT patterns.
    
    Args:
        param: Parameter tensor to initialize
        param_name: Full parameter name for type detection
        gating_alpha_init: Initial value for gating alpha parameters
    """
    name_lower = param_name.lower()
    
    if "gating_alpha" in name_lower:
        # Video-DiT initialization: configurable gating alpha
        torch.nn.init.constant_(param, gating_alpha_init)
        logger.debug(f"Initialized {param_name} as gating_alpha with value {gating_alpha_init}")
        
    elif any(x in name_lower for x in ["wq.weight", "wk.weight", "wv.weight", "wo.weight"]):
        # TTT projection weights: small normal distribution
        torch.nn.init.normal_(param, mean=0.0, std=0.02)
        logger.debug(f"Initialized {param_name} as TTT projection weight with normal(0, 0.02)")
        
    elif any(x in name_lower for x in ["wq.bias", "wk.bias", "wv.bias", "wo.bias"]):
        # TTT projection biases: zeros
        torch.nn.init.zeros_(param)
        logger.debug(f"Initialized {param_name} as TTT projection bias with zeros")
        
    elif any(x in name_lower for x in ["w1", "w2"]):
        # TTT MLP parameter tensors: small normal distribution
        torch.nn.init.normal_(param, mean=0.0, std=0.02)
        logger.debug(f"Initialized {param_name} as TTT MLP weight with normal(0, 0.02)")
        
    elif any(x in name_lower for x in ["b1", "b2"]):
        # TTT MLP bias tensors: zeros
        torch.nn.init.zeros_(param)
        logger.debug(f"Initialized {param_name} as TTT MLP bias with zeros")
        
    elif "ttt_norm_weight" in name_lower:
        # Layer norm weight: ones
        torch.nn.init.ones_(param)
        logger.debug(f"Initialized {param_name} as TTT norm weight with ones")
        
    elif "ttt_norm_bias" in name_lower:
        # Layer norm bias: zeros
        torch.nn.init.zeros_(param)
        logger.debug(f"Initialized {param_name} as TTT norm bias with zeros")
        
    elif "learnable_ttt_lr_weight" in name_lower:
        # Learnable learning rate weight: small normal
        torch.nn.init.normal_(param, mean=0.0, std=0.02)
        logger.debug(f"Initialized {param_name} as learnable TTT LR weight with normal(0, 0.02)")
        
    elif "learnable_ttt_lr_bias" in name_lower:
        # Learnable learning rate bias: zeros
        torch.nn.init.zeros_(param)
        logger.debug(f"Initialized {param_name} as learnable TTT LR bias with zeros")
        
    elif "post_norm.weight" in name_lower:
        # Post norm weight: ones (standard layer norm)
        torch.nn.init.ones_(param)
        logger.debug(f"Initialized {param_name} as post norm weight with ones")
        
    elif "post_norm.bias" in name_lower:
        # Post norm bias: zeros (standard layer norm)
        torch.nn.init.zeros_(param)
        logger.debug(f"Initialized {param_name} as post norm bias with zeros")
        
    else:
        # Default for unknown TTT parameters
        torch.nn.init.normal_(param, mean=0.0, std=0.02)
        logger.warning(f"Unknown TTT parameter type {param_name}, using default normal(0, 0.02)")


def count_ttt_parameters(model: torch.nn.Module, trainable_only: bool = True) -> Dict[str, int]:
    """
    Count TTT parameters in the model.
    
    Args:
        model: PyTorch model
        trainable_only: If True, only count trainable parameters
        
    Returns:
        Dict with TTT parameter counts by type
    """
    counts = {
        "gating_alpha": 0,
        "ttt_projections": 0,  # wq, wk, wv, wo
        "ttt_mlp": 0,          # w1, w2, b1, b2
        "ttt_norm": 0,         # ttt_norm_weight, ttt_norm_bias
        "ttt_lr": 0,           # learnable_ttt_lr
        "ttt_other": 0,        # other TTT parameters
        "total_ttt": 0
    }
    
    for name, param in model.named_parameters():
        if trainable_only and not param.requires_grad:
            continue
            
        if is_ttt_parameter(name):
            name_lower = name.lower()
            param_count = param.numel()
            
            if "gating_alpha" in name_lower:
                counts["gating_alpha"] += param_count
            elif any(x in name_lower for x in ["wq", "wk", "wv", "wo"]):
                counts["ttt_projections"] += param_count
            elif any(x in name_lower for x in ["w1", "w2", "b1", "b2"]):
                counts["ttt_mlp"] += param_count
            elif any(x in name_lower for x in ["ttt_norm_weight", "ttt_norm_bias"]):
                counts["ttt_norm"] += param_count
            elif "learnable_ttt_lr" in name_lower:
                counts["ttt_lr"] += param_count
            else:
                counts["ttt_other"] += param_count
                
            counts["total_ttt"] += param_count
    
    return counts


def get_training_mode_from_args(args) -> str:
    """
    Determine training mode from arguments.
    
    Args:
        args: TrainArgs object
        
    Returns:
        Training mode string: "frozen", "lora", "ttt", "lora+ttt", "full"
    """
    # Check if explicit training mode is set
    if hasattr(args, 'training_mode') and args.training_mode != "auto":
        return args.training_mode
    
    # Auto-detect based on configuration
    if args.full_finetuning:
        return "full"
    elif args.lora.enable and args.ttt.enable:
        return "lora+ttt"
    elif args.lora.enable:
        return "lora"
    elif args.ttt.enable:
        return "ttt"
    else:
        return "frozen"


def log_ttt_parameter_status(model: torch.nn.Module, args) -> None:
    """
    Log detailed status of TTT parameters in the model.
    
    Args:
        model: PyTorch model
        args: Training arguments
    """
    training_mode = get_training_mode_from_args(args)
    
    # Count all parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Count by type
    lora_params = sum(p.numel() for name, p in model.named_parameters() 
                     if p.requires_grad and "lora" in name)
    ttt_counts = count_ttt_parameters(model, trainable_only=True)
    ttt_total = ttt_counts["total_ttt"]
    other_params = trainable_params - lora_params - ttt_total
    
    # Log summary
    logger.info(f"🎯 Training mode: {training_mode}")
    logger.info(f"📊 Parameter breakdown:")
    logger.info(f"   Total parameters: {total_params:,}")
    logger.info(f"   Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    logger.info(f"     - LoRA parameters: {lora_params:,}")
    logger.info(f"     - TTT parameters: {ttt_total:,}")
    logger.info(f"     - Other parameters: {other_params:,}")
    
    # Log TTT breakdown if any TTT parameters exist
    if ttt_total > 0:
        logger.info(f"🧠 TTT parameter breakdown:")
        for param_type, count in ttt_counts.items():
            if count > 0 and param_type != "total_ttt":
                logger.info(f"     - {param_type}: {count:,}")


def validate_ttt_parameters(model: torch.nn.Module) -> bool:
    """
    Validate that TTT parameters are properly initialized and configured.
    
    Args:
        model: PyTorch model
        
    Returns:
        bool: True if validation passes
    """
    issues = []
    
    for name, param in model.named_parameters():
        if is_ttt_parameter(name):
            # Check for meta tensors
            if param.is_meta:
                issues.append(f"TTT parameter {name} is still a meta tensor")
                
            # Check for NaN/Inf values
            if torch.isnan(param).any():
                issues.append(f"TTT parameter {name} contains NaN values")
                
            if torch.isinf(param).any():
                issues.append(f"TTT parameter {name} contains Inf values")
                
            # Check gating_alpha values are reasonable
            if "gating_alpha" in name.lower():
                if param.abs().max() > 10.0:
                    issues.append(f"Gating alpha {name} has unreasonably large values: {param.abs().max()}")
    
    if issues:
        logger.error("❌ TTT parameter validation failed:")
        for issue in issues:
            logger.error(f"   {issue}")
        return False
    else:
        logger.info("✅ TTT parameter validation passed")
        return True