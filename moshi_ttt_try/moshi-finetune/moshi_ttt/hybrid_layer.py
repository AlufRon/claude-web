"""
Step 3.1.1: Create hybrid_layer.py file
Following Video-DiT's SeqModelingBlock pattern exactly
CLEANED VERSION: Minimal logging, clear weight persistence tracking
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import sys
import os
import logging
from torch.utils.checkpoint import checkpoint

# Add original Moshi to path
sys.path.append('/home/alufr/ttt_tests/moshi/moshi')

# Moshi imports
from moshi.modules.transformer import StreamingTransformerLayer, _LayerState
from moshi.modules.streaming import StreamingModule

# Our TTT imports
from .models.ssm.ops.ttt_mlp import ttt_mlp, ttt_mlp_with_states
from .format_utils import moshi_to_ttt_format, ttt_to_moshi_format
from .moshi_metadata import MoshiSequenceMetadata
from .config import TTTConfig
from .utils import SequenceMetadata
from .diagnostic_logger import TTTDiagnosticLogger

# Setup logger
logger = logging.getLogger(__name__)


class HybridSeqModelingBlock(nn.Module):
    """
    Hybrid block that combines Moshi's attention with Video-DiT's TTT processing.
    
    This follows Video-DiT's SeqModelingBlock pattern:
    1. Attention processing (using Moshi's existing attention)
    2. TTT/SSM processing (using Video-DiT's TTT implementation)
    
    The key difference from Video-DiT is that we work with 1D audio sequences
    instead of 3D video sequences.
    """
    
    def __init__(self, original_layer: StreamingTransformerLayer, ttt_config: TTTConfig, persistent_states: bool = False, layer_id: int = None):
        super().__init__()
        
        # Store the original Moshi layer for attention processing
        self.original_layer = original_layer
        
        # CRITICAL FIX: Detach the original layer from parent streaming management
        # This prevents conflicts when the parent HybridStreamingTransformerLayer enters streaming mode
        # The original layer will manage its own streaming state independently
        self.original_layer.set_streaming_detached(True)
        
        # TTT configuration
        self.ttt_config = ttt_config
        
        # Store layer ID for tracking
        self.layer_id = layer_id
        
        # State persistence configuration
        self.persistent_states = persistent_states
        
        # Store dimensions for format conversion
        self.d_model = ttt_config.model_dim
        self.num_heads = ttt_config.num_heads
        self.head_dim = self.d_model // self.num_heads
        
        # Create Video-DiT TTT layer using TTTWrapper
        num_layers = getattr(ttt_config, 'ttt_mlp_layers', 2)
        logger.info(f"[Hybrid] Layer {layer_id}: Creating TTT-MLP with {num_layers} layers")
        from .models.ssm.ttt_layer import TTTWrapper
        self.ttt_layer = TTTWrapper(ttt_config)

        # CRITICAL: Initialize TTT weights immediately after creation
        self.ttt_layer.ttt.init_weights()
        logger.info(f"[Hybrid] Layer {layer_id}: TTT weights initialized")

        self.ttt_layer.persistent_states = persistent_states

        # SSM Gating
        from .ssm_gating import SSMGating
        self.forward_ssm_gating = SSMGating(ttt_config)
        self.backward_ssm_gating = SSMGating(ttt_config)

        # Save base TTT weights for file-switch resets (when persistent_states=True)
        self._base_ttt_weights = None
        if persistent_states:
            self._save_base_ttt_weights()

    def _save_base_ttt_weights(self):
        """Save the learned base TTT weights for resetting on file switches."""
        try:
            ttt_instance = getattr(self.ttt_layer, 'ttt', self.ttt_layer)
            self._base_ttt_weights = {}

            # Save base weights
            if hasattr(ttt_instance, 'W1'):
                self._base_ttt_weights['W1'] = ttt_instance.W1.data.clone()
            if hasattr(ttt_instance, 'b1'):
                self._base_ttt_weights['b1'] = ttt_instance.b1.data.clone()
            if hasattr(ttt_instance, 'W2'):
                self._base_ttt_weights['W2'] = ttt_instance.W2.data.clone()
            if hasattr(ttt_instance, 'b2'):
                self._base_ttt_weights['b2'] = ttt_instance.b2.data.clone()

            # Multi-layer support
            if hasattr(ttt_instance, 'weights'):
                self._base_ttt_weights['weights'] = [w.data.clone() for w in ttt_instance.weights]
            if hasattr(ttt_instance, 'biases'):
                self._base_ttt_weights['biases'] = [b.data.clone() for b in ttt_instance.biases]

            logger.debug(f"💾 Layer {self.layer_id}: Saved base TTT weights for reset")

        except Exception as e:
            logger.warning(f"⚠️ Layer {self.layer_id}: Failed to save base TTT weights: {e}")

    def save_ttt_states(self):
        """Save current TTT parameter values for later restoration."""
        if not self.persistent_states:
            return None
        
        try:
            saved_state = {}
            ttt_instance = getattr(self.ttt_layer, 'ttt', self.ttt_layer)
            
            # Handle 2-layer TTT-MLP (W1, b1, W2, b2)
            if hasattr(ttt_instance, 'W1'):
                saved_state['W1'] = ttt_instance.W1.clone().detach()
            if hasattr(ttt_instance, 'b1'):
                saved_state['b1'] = ttt_instance.b1.clone().detach()
            if hasattr(ttt_instance, 'W2'):
                saved_state['W2'] = ttt_instance.W2.clone().detach()
            if hasattr(ttt_instance, 'b2'):
                saved_state['b2'] = ttt_instance.b2.clone().detach()
            
            # Handle multi-layer TTT-MLP
            if hasattr(ttt_instance, 'weights'):
                saved_state['weights'] = [w.clone().detach() for w in ttt_instance.weights]
            if hasattr(ttt_instance, 'biases'):
                saved_state['biases'] = [b.clone().detach() for b in ttt_instance.biases]
            
            logger.debug(f"💾 Layer {self.layer_id}: TTT states saved")
            return saved_state
            
        except Exception as e:
            logger.warning(f"⚠️ Layer {self.layer_id}: Failed to save TTT states: {e}")
            return None
    
    def restore_ttt_states(self, saved_state):
        """Restore TTT parameters from previously saved state."""
        if not self.persistent_states or saved_state is None:
            return
        
        try:
            with torch.no_grad():
                ttt_instance = getattr(self.ttt_layer, 'ttt', self.ttt_layer)
                
                # Restore 2-layer TTT-MLP parameters
                if 'W1' in saved_state and hasattr(ttt_instance, 'W1'):
                    ttt_instance.W1.copy_(saved_state['W1'])
                if 'b1' in saved_state and hasattr(ttt_instance, 'b1'):
                    ttt_instance.b1.copy_(saved_state['b1'])
                if 'W2' in saved_state and hasattr(ttt_instance, 'W2'):
                    ttt_instance.W2.copy_(saved_state['W2'])
                if 'b2' in saved_state and hasattr(ttt_instance, 'b2'):
                    ttt_instance.b2.copy_(saved_state['b2'])
                
                # Restore multi-layer TTT-MLP parameters
                if 'weights' in saved_state and hasattr(ttt_instance, 'weights'):
                    for i, w in enumerate(saved_state['weights']):
                        ttt_instance.weights[i].copy_(w)
                if 'biases' in saved_state and hasattr(ttt_instance, 'biases'):
                    for i, b in enumerate(saved_state['biases']):
                        ttt_instance.biases[i].copy_(b)
            
            logger.info(f"🔄 Layer {self.layer_id}: TTT states restored")
            
        except Exception as e:
            logger.warning(f"⚠️ Layer {self.layer_id}: Failed to restore TTT states: {e}")
    
    def reset_ttt_states(self):
        """DEPRECATED: This method breaks the computation graph during training."""
        logger.warning(f"⚠️ Layer {self.layer_id}: reset_ttt_states() is DEPRECATED")
        logger.warning("   Use save_ttt_states() and restore_ttt_states() instead")

        if self.persistent_states:
            logger.error(f"🚫 Layer {self.layer_id}: TTT reset BLOCKED - would break training")

    def reset_ttt_inner_weights_for_new_file(self):
        """
        Reset TTT inner weights to their learned base values when switching files.
        Used when rope_reset_on_new_file=True to reset TTT state for new speaker/context.

        Returns:
            bool: True if reset was successful
        """
        if not self.persistent_states or self._base_ttt_weights is None:
            return False  # Only reset when persistence is enabled and base weights exist

        try:
            # Get the TTT instance (handles both TTTWrapper and direct TTT)
            ttt_instance = getattr(self.ttt_layer, 'ttt', self.ttt_layer)

            # Restore base weights
            with torch.no_grad():
                if 'W1' in self._base_ttt_weights and hasattr(ttt_instance, 'W1'):
                    ttt_instance.W1.data.copy_(self._base_ttt_weights['W1'])
                if 'b1' in self._base_ttt_weights and hasattr(ttt_instance, 'b1'):
                    ttt_instance.b1.data.copy_(self._base_ttt_weights['b1'])
                if 'W2' in self._base_ttt_weights and hasattr(ttt_instance, 'W2'):
                    ttt_instance.W2.data.copy_(self._base_ttt_weights['W2'])
                if 'b2' in self._base_ttt_weights and hasattr(ttt_instance, 'b2'):
                    ttt_instance.b2.data.copy_(self._base_ttt_weights['b2'])

                # Multi-layer support
                if 'weights' in self._base_ttt_weights and hasattr(ttt_instance, 'weights'):
                    for i, base_w in enumerate(self._base_ttt_weights['weights']):
                        ttt_instance.weights[i].data.copy_(base_w)
                if 'biases' in self._base_ttt_weights and hasattr(ttt_instance, 'biases'):
                    for i, base_b in enumerate(self._base_ttt_weights['biases']):
                        ttt_instance.biases[i].data.copy_(base_b)

            logger.debug(f"🔄 Layer {self.layer_id}: TTT inner weights reset to learned base")
            return True

        except Exception as e:
            logger.warning(f"⚠️ Layer {self.layer_id}: Failed to reset TTT inner weights: {e}")
            return False

    def get_ttt_state_dict(self):
        """Get current TTT state for checkpointing"""
        if self.persistent_states:
            return self.ttt_layer.state_dict()
        return None
        
    def load_ttt_state_dict(self, state_dict):
        """Load TTT state from checkpoint"""
        if self.persistent_states and state_dict is not None:
            self.ttt_layer.load_state_dict(state_dict)
        
    def forward(self, x: torch.Tensor, cross_attention_src: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass - NO CHECKPOINTING at this level."""
        # Input validation
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(x)}")
        if x.dim() != 3:
            raise ValueError(f"Expected 3D tensor [B, seq_len, d_model], got {x.dim()}D tensor")
        if x.shape[-1] != self.d_model:
            raise ValueError(f"Expected d_model={self.d_model}, got {x.shape[-1]}")

        return self._forward_impl(x, cross_attention_src)
    
    def _forward_impl(self, x: torch.Tensor, cross_attention_src: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Implementation of the forward pass."""
        # Step 1: Attention processing
        attn_output = self._attn_forward(x, cross_attention_src)
        
        # Step 2: TTT processing
        ttt_output = self._ttt_forward(attn_output)
        
        return ttt_output
        
    def _attn_forward(self, x: torch.Tensor, cross_attention_src: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Attention processing using Moshi's original components."""
        # Use Moshi's self-attention block
        x = self.original_layer._sa_block(x)

        # Add cross-attention if available
        if self.original_layer.cross_attention is not None and cross_attention_src is not None:
            x = self.original_layer._cross_attention_block(x, cross_attention_src)

        # Cache attention output (for diagnostics)
        self._last_attn_output = x.detach()

        return x
        
    def _ttt_forward(self, x: torch.Tensor) -> torch.Tensor:
        """TTT processing with gated residual connections."""
        B, seq_len, d_model = x.shape

        # Store residual before any processing
        residual_emb = x.clone()

        # Convert to TTT format and process
        x_ttt, conversion_metadata = moshi_to_ttt_format(x, self.ttt_config)
        x_processed = self._apply_ttt_processing(x_ttt, x, conversion_metadata)

        # Cache TTT output before gating (for diagnostics)
        self._last_ttt_output = x_processed.detach()

        # Apply gated residual connection
        gated_output = self.forward_ssm_gating(x_processed)

        # Convert to match residual dtype (bfloat16)
        gated_output = gated_output.to(residual_emb.dtype)

        # Verify shape
        assert gated_output.shape[1] == seq_len, f"Shape mismatch: expected {seq_len}, got {gated_output.shape[1]}"

        # ==== WEIGHT PERSISTENCE TRACKING (DISABLED - use diagnostic logger instead) ====
        # Hash-based logging is now handled by the diagnostic logger for better control
        # Keeping baseline establishment for backward compatibility
        if self.persistent_states and not hasattr(self, '_last_weight_hash'):
            # First call - establish baseline (silent)
            ttt_instance = self.ttt_layer.ttt
            if hasattr(ttt_instance, 'W1'):
                with torch.no_grad():
                    self._last_weight_hash = hash((
                        ttt_instance.W1.data.sum().item(),
                        ttt_instance.b1.data.sum().item(),
                        ttt_instance.W2.data.sum().item(),
                        ttt_instance.b2.data.sum().item()
                    ))

        return residual_emb + gated_output
        
    def _apply_ttt_processing(self, x_ttt: torch.Tensor, x_original: torch.Tensor, metadata: dict) -> torch.Tensor:
        """Apply TTT processing - NO CHECKPOINTING at this level."""
        return self._apply_ttt_processing_impl(x_ttt, x_original, metadata)
    
    def _apply_ttt_processing_impl(self, x_ttt: torch.Tensor, x_original: torch.Tensor, metadata: dict) -> torch.Tensor:
        """Implementation of TTT processing."""
        from .format_utils import create_sequence_metadata
        
        seq_metadata = create_sequence_metadata(x_original, self.ttt_config)
        
        # Pad sequence to be multiple of mini_batch_size
        B, seq_len, d_model = x_original.shape
        C = self.ttt_config.mini_batch_size
        
        if seq_len % C != 0:
            pad_len = C - (seq_len % C)
            x_pad = torch.zeros(B, pad_len, d_model, device=x_original.device, dtype=x_original.dtype)
            x_padded = torch.cat([x_original, x_pad], dim=1)
        else:
            x_padded = x_original
        
        # Call TTT layer
        ttt_output = self.ttt_layer(x_padded, seq_metadata, layer_id=self.layer_id)
        
        # Trim back to original length
        ttt_output = ttt_output[:, :seq_len, :]
        
        return ttt_output


class HybridStreamingTransformerLayer(StreamingModule[_LayerState]):
    """Hybrid transformer layer that replaces Moshi's StreamingTransformerLayer."""
    
    def __init__(self, original_layer: StreamingTransformerLayer, ttt_config: TTTConfig, persistent_states: bool = False, layer_id: int = None):
        super().__init__()
        
        self.original_layer = original_layer
        self.layer_id = layer_id

        # Create hybrid seq modeling block
        self.seq_modeling_block = HybridSeqModelingBlock(original_layer, ttt_config, persistent_states, layer_id)

        # Diagnostic logger (disabled by default, enabled via enable_diagnostics())
        self.diagnostic_logger: Optional[TTTDiagnosticLogger] = None

        # Gradient checkpointing configuration
        self.checkpointing = False
        
    def enable_diagnostics(self, log_frequency: int = 100, track_history: bool = False):
        """
        Enable diagnostic logging for this layer.

        Args:
            log_frequency: Log diagnostics every N steps
            track_history: Whether to keep history of metrics for plotting
        """
        self.diagnostic_logger = TTTDiagnosticLogger(
            layer_id=self.layer_id,
            log_frequency=log_frequency,
            track_history=track_history
        )
        logger.info(f"✅ L{self.layer_id}: Diagnostic logging enabled (frequency={log_frequency})")

    def disable_diagnostics(self):
        """Disable diagnostic logging."""
        self.diagnostic_logger = None
        logger.info(f"❌ L{self.layer_id}: Diagnostic logging disabled")

    def get_diagnostic_history(self):
        """Get diagnostic history summary if available."""
        if self.diagnostic_logger and self.diagnostic_logger.track_history:
            return self.diagnostic_logger.get_history_summary()
        return {}

    def _start_original_layer_streaming(self, batch_size: int):
        """Manually start streaming for the detached original layer"""
        original_layer = self.seq_modeling_block.original_layer
        if not hasattr(original_layer, '_streaming_state') or original_layer._streaming_state is None:
            original_layer.streaming_forever(batch_size)

    def _stop_original_layer_streaming(self):
        """Manually stop streaming for the detached original layer"""
        original_layer = self.seq_modeling_block.original_layer
        if hasattr(original_layer, '_streaming_state') and original_layer._streaming_state is not None:
            original_layer._streaming_state = None

    def _init_streaming_state(self, batch_size: int) -> _LayerState:
        """Initialize streaming state."""
        device = next(iter(self.parameters())).device
        
        # Start streaming for the detached original layer
        self._start_original_layer_streaming(batch_size)
        
        return _LayerState(batch_size, device, offset_cpu=0)
        
    def save_ttt_states(self):
        """Save current TTT states"""
        return self.seq_modeling_block.save_ttt_states()
        
    def restore_ttt_states(self, saved_state):
        """Restore TTT states"""
        self.seq_modeling_block.restore_ttt_states(saved_state)
        
    def reset_ttt_states(self):
        """DEPRECATED: Reset TTT states"""
        logger.warning(f"⚠️ L{self.layer_id}: reset_ttt_states() is DEPRECATED")
        self.seq_modeling_block.reset_ttt_states()
        
    def get_ttt_state_dict(self):
        """Get current TTT state for checkpointing"""
        return self.seq_modeling_block.get_ttt_state_dict()
        
    def load_ttt_state_dict(self, state_dict):
        """Load TTT state from checkpoint"""
        self.seq_modeling_block.load_ttt_state_dict(state_dict)
        
    def forward(self, x: torch.Tensor, cross_attention_src: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with optional checkpointing."""
        
        # Step 1: Sequence modeling with optional checkpointing
        if self.checkpointing and x.requires_grad:
            x = checkpoint(
                self._forward_with_seq_modeling,
                x,
                cross_attention_src,
                use_reentrant=False,
            )
        else:
            x = self._forward_with_seq_modeling(x, cross_attention_src)
        
        # Step 2: Feedforward processing
        x = self.original_layer._ff_block(x)
        
        # Update streaming state
        state = self._streaming_state
        if state:
            state.offset_cpu += x.shape[1]
            
        return x
    
    def _forward_with_seq_modeling(self, x: torch.Tensor, cross_attention_src: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Helper for checkpointing: wraps seq_modeling_block forward."""
        # Call seq modeling block
        output = self.seq_modeling_block(x, cross_attention_src)

        # If diagnostics enabled, log intermediate values
        if self.diagnostic_logger is not None:
            # Get intermediate values from seq_modeling_block
            attn_output = getattr(self.seq_modeling_block, '_last_attn_output', None)
            ttt_output = getattr(self.seq_modeling_block, '_last_ttt_output', None)
            gating_alpha = getattr(self.seq_modeling_block.forward_ssm_gating, 'gating_alpha', None)

            # Get TTT weight norm
            ttt_weight_norm = None
            if self.seq_modeling_block.persistent_states:
                ttt_instance = self.seq_modeling_block.ttt_layer.ttt
                if hasattr(ttt_instance, 'W1'):
                    with torch.no_grad():
                        ttt_weight_norm = ttt_instance.W1.data.norm().item()

            # Log diagnostics
            if attn_output is not None and ttt_output is not None:
                self.diagnostic_logger.log_step(
                    attn_output=attn_output,
                    ttt_output=ttt_output,
                    combined_output=output,
                    gating_alpha=gating_alpha,
                    ttt_weight_norm=ttt_weight_norm
                )

        return output