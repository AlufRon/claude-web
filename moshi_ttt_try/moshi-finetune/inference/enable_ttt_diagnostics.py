"""
Utility to enable TTT diagnostic logging on a Moshi model.

Usage:
    from inference.enable_ttt_diagnostics import enable_ttt_diagnostics

    # Enable diagnostics on all TTT layers
    enable_ttt_diagnostics(model, log_frequency=100, track_history=False)
"""

import logging
from typing import Optional
import torch.nn as nn

logger = logging.getLogger(__name__)


def enable_ttt_diagnostics(
    model: nn.Module,
    log_frequency: int = 100,
    track_history: bool = False,
    layer_ids: Optional[list] = None
) -> int:
    """
    Enable diagnostic logging on TTT layers in the model.

    Args:
        model: Moshi model (possibly with TTT layers)
        log_frequency: Log diagnostics every N steps (default: 100)
        track_history: Whether to track history for plotting (default: False)
        layer_ids: Specific layer IDs to enable (default: all TTT layers)

    Returns:
        Number of TTT layers with diagnostics enabled
    """
    enabled_count = 0

    # Check if model has transformer attribute
    if not hasattr(model, 'transformer') or not hasattr(model.transformer, 'layers'):
        logger.warning("Model does not have transformer.layers - no TTT layers to enable diagnostics")
        return 0

    # Iterate through all layers
    for idx, layer in enumerate(model.transformer.layers):
        # Check if this is a HybridStreamingTransformerLayer
        if hasattr(layer, 'enable_diagnostics'):
            # If layer_ids specified, only enable for those layers
            if layer_ids is not None and idx not in layer_ids:
                continue

            # Enable diagnostics
            layer.enable_diagnostics(
                log_frequency=log_frequency,
                track_history=track_history
            )
            enabled_count += 1
            logger.info(f"✅ Enabled diagnostics on layer {idx}")

    if enabled_count == 0:
        logger.warning("No TTT layers found in model (or layer_ids filter excluded all layers)")
    else:
        logger.info(f"")
        logger.info(f"{'='*80}")
        logger.info(f"📊 TTT Diagnostics Enabled")
        logger.info(f"{'='*80}")
        logger.info(f"   Enabled layers: {enabled_count}")
        logger.info(f"   Log frequency: every {log_frequency} steps")
        logger.info(f"   Track history: {track_history}")
        logger.info(f"{'='*80}")
        logger.info(f"")

    return enabled_count


def disable_ttt_diagnostics(model: nn.Module) -> int:
    """
    Disable diagnostic logging on all TTT layers.

    Args:
        model: Moshi model with TTT layers

    Returns:
        Number of TTT layers with diagnostics disabled
    """
    disabled_count = 0

    if not hasattr(model, 'transformer') or not hasattr(model.transformer, 'layers'):
        return 0

    for layer in model.transformer.layers:
        if hasattr(layer, 'disable_diagnostics'):
            layer.disable_diagnostics()
            disabled_count += 1

    if disabled_count > 0:
        logger.info(f"❌ Disabled diagnostics on {disabled_count} TTT layers")

    return disabled_count


def get_diagnostic_history(model: nn.Module) -> dict:
    """
    Get diagnostic history from all TTT layers.

    Args:
        model: Moshi model with TTT layers

    Returns:
        Dict mapping layer_id to history summary
    """
    history = {}

    if not hasattr(model, 'transformer') or not hasattr(model.transformer, 'layers'):
        return history

    for idx, layer in enumerate(model.transformer.layers):
        if hasattr(layer, 'get_diagnostic_history'):
            layer_history = layer.get_diagnostic_history()
            if layer_history:
                history[f"layer_{idx}"] = layer_history

    return history
