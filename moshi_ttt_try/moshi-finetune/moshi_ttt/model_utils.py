"""
Model-level utilities for TTT state management.

This module provides functions to save and restore TTT states across all
layers in a model, enabling safe evaluation isolation without computation
graph destruction.
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Ensure INFO level logs are shown


def save_ttt_states(model) -> Optional[Dict[str, Any]]:
    """
    Save TTT states from all hybrid layers in the model.
    
    This function traverses the model and collects TTT states from all layers
    that support the save_ttt_states() method. Use this before evaluation
    to capture the pre-evaluation state.
    
    Args:
        model: PyTorch model containing hybrid TTT layers
        
    Returns:
        Dict mapping layer names to their saved TTT states, or None if no TTT layers found
    """
    saved_states = {}
    ttt_layer_count = 0
    
    try:
        for name, module in model.named_modules():
            if hasattr(module, 'save_ttt_states') and callable(getattr(module, 'save_ttt_states')):
                try:
                    state = module.save_ttt_states()
                    if state is not None:
                        saved_states[name] = state
                        ttt_layer_count += 1
                        logger.debug(f"💾 Saved TTT state for layer: {name}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to save TTT state for layer {name}: {e}")
        
        if ttt_layer_count > 0:
            logger.info(f"💾 TTT states saved from {ttt_layer_count} layers")
            return saved_states
        else:
            logger.debug("No TTT layers found with save_ttt_states() method")
            return None
            
    except Exception as e:
        logger.error(f"❌ Failed to save TTT states: {e}")
        return None


def restore_ttt_states(model, saved_states: Optional[Dict[str, Any]]) -> bool:
    """
    Restore TTT states to all hybrid layers in the model.
    
    This function restores previously saved TTT states to their corresponding
    layers. Use this after evaluation to remove evaluation contamination
    from training state.
    
    Args:
        model: PyTorch model containing hybrid TTT layers
        saved_states: Dict returned by save_ttt_states()
        
    Returns:
        bool: True if restoration was successful, False otherwise
    """
    print("🔍 PRINT: restore_ttt_states function called")  # Backup debug method
    logger.info("🔍 DEBUG: restore_ttt_states function called")
    logger.info(f"🔍 DEBUG: model type: {type(model)}")
    logger.info(f"🔍 DEBUG: saved_states type: {type(saved_states)}")
    
    if saved_states is None:
        logger.info("🔍 DEBUG: No TTT states to restore (saved_states is None)")
        return True
    
    restored_count = 0
    total_states = len(saved_states)
    logger.info(f"🔍 DEBUG: attempting to restore {total_states} TTT states")
    
    try:
        logger.info("🔍 DEBUG: iterating through model modules")
        for name, module in model.named_modules():
            if name in saved_states:
                logger.info(f"🔍 DEBUG: found module {name} in saved_states")
                if hasattr(module, 'restore_ttt_states'):
                    logger.info(f"🔍 DEBUG: module {name} has restore_ttt_states method")
                    try:
                        module.restore_ttt_states(saved_states[name])
                        restored_count += 1
                        logger.info(f"🔄 DEBUG: Successfully restored TTT state for layer: {name}")
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to restore TTT state for layer {name}: {e}")
                else:
                    logger.info(f"🔍 DEBUG: module {name} does NOT have restore_ttt_states method")
        
        logger.info(f"🔍 DEBUG: restore loop completed, restored {restored_count}/{total_states}")
        
        if restored_count == total_states:
            logger.info(f"🔄 TTT states restored to {restored_count}/{total_states} layers - evaluation contamination removed")
            return True
        else:
            logger.warning(f"⚠️ Partial TTT restore: {restored_count}/{total_states} layers restored")
            return False
            
    except Exception as e:
        logger.error(f"❌ Failed to restore TTT states: {e}")
        return False


def get_ttt_layer_info(model) -> Dict[str, Dict[str, Any]]:
    """
    Get information about TTT layers in the model.
    
    This is a diagnostic function to understand which layers support TTT
    state management.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dict mapping layer names to their TTT capabilities
    """
    ttt_info = {}
    
    for name, module in model.named_modules():
        capabilities = {}
        
        if hasattr(module, 'save_ttt_states'):
            capabilities['can_save'] = callable(getattr(module, 'save_ttt_states'))
        if hasattr(module, 'restore_ttt_states'):
            capabilities['can_restore'] = callable(getattr(module, 'restore_ttt_states'))
        if hasattr(module, 'reset_ttt_states'):
            capabilities['can_reset'] = callable(getattr(module, 'reset_ttt_states'))
        
        if capabilities:
            ttt_info[name] = capabilities
    
    return ttt_info


def verify_ttt_state_isolation(model, pre_eval_states: Optional[Dict[str, Any]], 
                              post_eval_states: Optional[Dict[str, Any]]) -> bool:
    """
    Verify that TTT states have been properly isolated from evaluation.
    
    This function compares TTT states before and after evaluation to ensure
    that evaluation contamination has been properly removed.
    
    Args:
        model: PyTorch model
        pre_eval_states: States saved before evaluation
        post_eval_states: States after restore operation
        
    Returns:
        bool: True if isolation is verified, False otherwise
    """
    if pre_eval_states is None or post_eval_states is None:
        logger.debug("Cannot verify TTT isolation: missing state snapshots")
        return False
    
    try:
        # Check that we have the same layers
        if set(pre_eval_states.keys()) != set(post_eval_states.keys()):
            logger.warning("⚠️ TTT isolation verification: layer mismatch")
            return False
        
        # Check that states match for each layer
        import torch
        for layer_name in pre_eval_states.keys():
            pre_state = pre_eval_states[layer_name]
            post_state = post_eval_states[layer_name]
            
            if set(pre_state.keys()) != set(post_state.keys()):
                logger.warning(f"⚠️ TTT isolation verification: parameter mismatch in {layer_name}")
                return False
            
            for param_name in pre_state.keys():
                if not torch.allclose(pre_state[param_name], post_state[param_name], atol=1e-7):
                    logger.warning(f"⚠️ TTT isolation verification: {layer_name}.{param_name} not properly restored")
                    return False
        
        logger.info("✅ TTT isolation verified: all states properly restored")
        return True
        
    except Exception as e:
        logger.error(f"❌ TTT isolation verification failed: {e}")
        return False