"""
TTT Cache for Inference
Adapted from ttt-lm-kernels/ttt/generation.py to work with Moshi architecture.
"""
import logging
from collections import defaultdict
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class TTTCache:
    """
    Cache for TTT weights and states during inference.
    
    This class manages persistent TTT weights across tokens, enabling the test-time
    training behavior where weights are updated during the forward pass and persist
    to the next token.
    
    Based on ttt-lm-kernels implementation.
    """
    
    def __init__(self, max_batch_size: int, model, mini_batch_size: int = 32):
        """
        Initialize TTT cache.
        
        Args:
            max_batch_size: Maximum batch size for inference
            model: The Moshi model containing TTT layers
            mini_batch_size: Size of mini-batches for TTT processing
        """
        self.max_batch_size = max_batch_size
        self.model = model
        self.mini_batch_size = mini_batch_size
        self.seqlen_offset = 0
        self.params_dict = defaultdict(dict)
        
        # Determine dtype from model
        self.dtype = next(model.parameters()).dtype
        self.device = next(model.parameters()).device
        
        logger.info(f"[TTTCache] Initialized with batch_size={max_batch_size}, "
                   f"mini_batch_size={mini_batch_size}, dtype={self.dtype}")
    
    def allocate_inference_cache(self):
        """
        Allocate cache tensors for all TTT layers.
        
        For each TTT layer, allocates:
        - W_init: Initial weights for each layer in the TTT-MLP
        - b_init: Initial biases for each layer in the TTT-MLP
        - W_grad: Gradient accumulator for weights
        - b_grad: Gradient accumulator for biases
        """
        logger.info("[TTTCache] Allocating inference cache...")
        
        # Find all TTT layers in the model
        ttt_layers = []
        for name, module in self.model.named_modules():
            # Look for hybrid layers with TTT
            if hasattr(module, 'ttt_wrapper') and module.ttt_wrapper is not None:
                ttt_layers.append((name, module.ttt_wrapper.ttt))
        
        logger.info(f"[TTTCache] Found {len(ttt_layers)} TTT layers")
        
        for layer_idx, (name, ttt_module) in enumerate(ttt_layers):
            # Determine what type of TTT layer this is
            if hasattr(ttt_module, 'weights'):
                # Multi-layer TTT-MLP
                num_ttt_layers = len(ttt_module.weights)
                logger.info(f"[TTTCache] Layer {layer_idx} ({name}): Multi-layer TTT-MLP with {num_ttt_layers} layers")
                
                # Allocate cache for each TTT layer
                for i, (weight, bias) in enumerate(zip(ttt_module.weights, ttt_module.biases)):
                    # Get dimensions: [num_heads, in_dim, out_dim]
                    nh, in_dim, out_dim = weight.shape
                    
                    # Tile for batch size: [B*nh, in_dim, out_dim]
                    tiled_weight = torch.tile(
                        weight.data, 
                        (self.max_batch_size,) + (1,) * (weight.dim() - 1)
                    )
                    tiled_bias = torch.tile(
                        bias.data,
                        (self.max_batch_size,) + (1,) * (bias.dim() - 1)
                    )
                    
                    # Store initial weights and create gradient accumulators
                    self.params_dict[f"W{i}_init"][layer_idx] = tiled_weight
                    self.params_dict[f"b{i}_init"][layer_idx] = tiled_bias
                    self.params_dict[f"W{i}_grad"][layer_idx] = torch.zeros_like(tiled_weight)
                    self.params_dict[f"b{i}_grad"][layer_idx] = torch.zeros_like(tiled_bias)
                    
                    logger.info(f"[TTTCache]   Layer {i}: W{i} shape {tiled_weight.shape}, "
                               f"b{i} shape {tiled_bias.shape}")
                    
            elif hasattr(ttt_module, 'W1') and hasattr(ttt_module, 'W2'):
                # Standard 2-layer TTT-MLP
                logger.info(f"[TTTCache] Layer {layer_idx} ({name}): Standard 2-layer TTT-MLP")
                
                # W1, b1
                nh, in_dim, out_dim = ttt_module.W1.shape
                tiled_W1 = torch.tile(
                    ttt_module.W1.data,
                    (self.max_batch_size,) + (1,) * (ttt_module.W1.dim() - 1)
                )
                tiled_b1 = torch.tile(
                    ttt_module.b1.data,
                    (self.max_batch_size,) + (1,) * (ttt_module.b1.dim() - 1)
                )
                
                self.params_dict["W1_init"][layer_idx] = tiled_W1
                self.params_dict["b1_init"][layer_idx] = tiled_b1
                self.params_dict["W1_grad"][layer_idx] = torch.zeros_like(tiled_W1)
                self.params_dict["b1_grad"][layer_idx] = torch.zeros_like(tiled_b1)
                
                # W2, b2
                nh, in_dim, out_dim = ttt_module.W2.shape
                tiled_W2 = torch.tile(
                    ttt_module.W2.data,
                    (self.max_batch_size,) + (1,) * (ttt_module.W2.dim() - 1)
                )
                tiled_b2 = torch.tile(
                    ttt_module.b2.data,
                    (self.max_batch_size,) + (1,) * (ttt_module.b2.dim() - 1)
                )
                
                self.params_dict["W2_init"][layer_idx] = tiled_W2
                self.params_dict["b2_init"][layer_idx] = tiled_b2
                self.params_dict["W2_grad"][layer_idx] = torch.zeros_like(tiled_W2)
                self.params_dict["b2_grad"][layer_idx] = torch.zeros_like(tiled_b2)
                
                logger.info(f"[TTTCache]   W1 shape {tiled_W1.shape}, W2 shape {tiled_W2.shape}")
                
            elif hasattr(ttt_module, 'W1'):
                # TTT-Linear (single layer)
                logger.info(f"[TTTCache] Layer {layer_idx} ({name}): TTT-Linear")
                
                nh, in_dim, out_dim = ttt_module.W1.shape
                tiled_W1 = torch.tile(
                    ttt_module.W1.data,
                    (self.max_batch_size,) + (1,) * (ttt_module.W1.dim() - 1)
                )
                tiled_b1 = torch.tile(
                    ttt_module.b1.data,
                    (self.max_batch_size,) + (1,) * (ttt_module.b1.dim() - 1)
                )
                
                self.params_dict["W1_init"][layer_idx] = tiled_W1
                self.params_dict["b1_init"][layer_idx] = tiled_b1
                self.params_dict["W1_grad"][layer_idx] = torch.zeros_like(tiled_W1)
                self.params_dict["b1_grad"][layer_idx] = torch.zeros_like(tiled_b1)
                
                logger.info(f"[TTTCache]   W1 shape {tiled_W1.shape}")
        
        logger.info(f"[TTTCache] ✅ Cache allocation complete for {len(ttt_layers)} layers")
    
    def reset(self, model):
        """
        Reset cache to initial weights from the model.
        
        This should be called at the beginning of each new sequence.
        
        Args:
            model: The Moshi model containing TTT layers
        """
        logger.info("[TTTCache] Resetting cache to initial weights")
        self.seqlen_offset = 0
        self.model = model
        
        # Find all TTT layers again
        ttt_layers = []
        for name, module in self.model.named_modules():
            if hasattr(module, 'ttt_wrapper') and module.ttt_wrapper is not None:
                ttt_layers.append((name, module.ttt_wrapper.ttt))
        
        for layer_idx, (name, ttt_module) in enumerate(ttt_layers):
            if hasattr(ttt_module, 'weights'):
                # Multi-layer TTT-MLP
                for i, (weight, bias) in enumerate(zip(ttt_module.weights, ttt_module.biases)):
                    tiled_weight = torch.tile(
                        weight.data,
                        (self.max_batch_size,) + (1,) * (weight.dim() - 1)
                    )
                    tiled_bias = torch.tile(
                        bias.data,
                        (self.max_batch_size,) + (1,) * (bias.dim() - 1)
                    )
                    
                    self.params_dict[f"W{i}_init"][layer_idx].copy_(tiled_weight)
                    self.params_dict[f"b{i}_init"][layer_idx].copy_(tiled_bias)
                    self.params_dict[f"W{i}_grad"][layer_idx].zero_()
                    self.params_dict[f"b{i}_grad"][layer_idx].zero_()
                    
            elif hasattr(ttt_module, 'W1') and hasattr(ttt_module, 'W2'):
                # Standard 2-layer TTT-MLP
                for param_name in ["W1", "b1", "W2", "b2"]:
                    weight = getattr(ttt_module, param_name).data
                    tiled_weight = torch.tile(
                        weight,
                        (self.max_batch_size,) + (1,) * (weight.dim() - 1)
                    )
                    self.params_dict[f"{param_name}_init"][layer_idx].copy_(tiled_weight)
                    self.params_dict[f"{param_name}_grad"][layer_idx].zero_()
                    
            elif hasattr(ttt_module, 'W1'):
                # TTT-Linear
                for param_name in ["W1", "b1"]:
                    weight = getattr(ttt_module, param_name).data
                    tiled_weight = torch.tile(
                        weight,
                        (self.max_batch_size,) + (1,) * (weight.dim() - 1)
                    )
                    self.params_dict[f"{param_name}_init"][layer_idx].copy_(tiled_weight)
                    self.params_dict[f"{param_name}_grad"][layer_idx].zero_()
        
        logger.info("[TTTCache] ✅ Cache reset complete")
    
    def get_layer_states(self, layer_idx: int, num_layers: Optional[int] = None):
        """
        Get state dict for a specific TTT layer.
        
        Args:
            layer_idx: Index of the TTT layer
            num_layers: Number of layers in the TTT-MLP (if known)
            
        Returns:
            Dictionary containing references to W_init, b_init, W_grad, b_grad tensors
        """
        states = {}
        
        # Try multi-layer first
        if num_layers is not None:
            for i in range(num_layers):
                states[f"W{i}_init"] = self.params_dict[f"W{i}_init"][layer_idx]
                states[f"b{i}_init"] = self.params_dict[f"b{i}_init"][layer_idx]
                states[f"W{i}_grad"] = self.params_dict[f"W{i}_grad"][layer_idx]
                states[f"b{i}_grad"] = self.params_dict[f"b{i}_grad"][layer_idx]
        else:
            # Standard 2-layer or linear
            if "W1_init" in self.params_dict and layer_idx in self.params_dict["W1_init"]:
                states["W1_init"] = self.params_dict["W1_init"][layer_idx]
                states["b1_init"] = self.params_dict["b1_init"][layer_idx]
                states["W1_grad"] = self.params_dict["W1_grad"][layer_idx]
                states["b1_grad"] = self.params_dict["b1_grad"][layer_idx]
                
            if "W2_init" in self.params_dict and layer_idx in self.params_dict["W2_init"]:
                states["W2_init"] = self.params_dict["W2_init"][layer_idx]
                states["b2_init"] = self.params_dict["b2_init"][layer_idx]
                states["W2_grad"] = self.params_dict["W2_grad"][layer_idx]
                states["b2_grad"] = self.params_dict["b2_grad"][layer_idx]
        
        return states
