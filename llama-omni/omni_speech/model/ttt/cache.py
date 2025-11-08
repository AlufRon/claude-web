"""
TTT State Management - Conversation-Level Persistence

This module implements state management for TTT layers, ensuring that:
1. TTT state (W1, b1) persists across batches within the same conversation
2. State is reset when a new conversation begins
3. State is properly saved and restored during generation

CRITICAL DESIGN PRINCIPLES:
1. State resets ONLY when conversation_id changes, NOT between batches
2. All inner states stored in float32 (not bf16/fp16)
3. State can be checkpointed and restored
4. Compatible with HuggingFace transformers caching system

The key difference from standard KV cache:
- KV cache: Stores past keys/values (grows linearly with sequence length)
- TTT cache: Stores learned parameters W1, b1 (fixed size, updates over time)
"""

import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class TTTCache:
    """
    Cache for TTT layer states across conversation turns.

    This stores the TTT hidden states (W1, b1) for each layer, allowing
    the model to maintain conversation-level memory.

    Attributes:
        max_batch_size: Maximum batch size to allocate
        num_layers: Number of TTT layers
        num_heads: Number of attention heads per layer
        head_dim: Dimension of each head
        device: Device to store tensors on
        dtype: Data type for computation (bf16/fp16)
        state_dtype: Data type for TTT states (MUST be float32)
        conversation_id: Current conversation ID (for reset detection)
        seqlen_offset: Current position in sequence
        params_dict: Dictionary storing states per layer

    State organization:
        params_dict = {
            "W1_init": {layer_idx: [B, nh, head_dim, head_dim]},  # Initial W1
            "b1_init": {layer_idx: [B, nh, 1, head_dim]},         # Initial b1
            "W1_current": {layer_idx: [B, nh, head_dim, head_dim]},  # Current W1
            "b1_current": {layer_idx: [B, nh, 1, head_dim]},         # Current b1
        }

    Usage:
        # Initialize
        cache = TTTCache(
            max_batch_size=8,
            num_layers=32,
            num_heads=32,
            head_dim=128,
            device='cuda'
        )

        # Before generation
        cache.reset(conversation_id="conv_123")

        # During generation - state automatically updated via past_key_values
        outputs = model(..., past_key_values=cache)

        # To start new conversation
        cache.reset(conversation_id="conv_456")  # This resets states
    """

    max_batch_size: int
    num_layers: int
    num_heads: int
    head_dim: int
    device: torch.device = field(default_factory=lambda: torch.device('cuda'))
    dtype: torch.dtype = torch.bfloat16
    state_dtype: torch.dtype = torch.float32  # MUST be float32 for stability

    # Conversation tracking
    conversation_id: Optional[str] = None
    seqlen_offset: int = 0

    # State storage
    params_dict: Dict = field(default_factory=lambda: defaultdict(dict))

    # Statistics tracking
    total_updates: int = 0
    reset_count: int = 0

    def __post_init__(self):
        """Validate configuration and allocate storage."""
        assert self.state_dtype == torch.float32, (
            f"TTT states MUST be float32 for numerical stability, got {self.state_dtype}"
        )

        logger.info(
            f"[TTTCache] Initializing cache for {self.num_layers} layers, "
            f"{self.num_heads} heads, {self.head_dim} head_dim, "
            f"max_batch_size={self.max_batch_size}"
        )

        # Allocate storage for all layers
        self.allocate()

    def allocate(self):
        """
        Allocate storage for TTT states for all layers.

        Creates tensors for W1 and b1 for each layer, initialized to zero.
        These will be overwritten with actual initial parameters during first use.
        """
        for layer_idx in range(self.num_layers):
            # W1: [B, nh, head_dim, head_dim]
            self.params_dict["W1_init"][layer_idx] = torch.zeros(
                self.max_batch_size,
                self.num_heads,
                self.head_dim,
                self.head_dim,
                dtype=self.state_dtype,
                device=self.device
            )

            # b1: [B, nh, 1, head_dim]
            self.params_dict["b1_init"][layer_idx] = torch.zeros(
                self.max_batch_size,
                self.num_heads,
                1,
                self.head_dim,
                dtype=self.state_dtype,
                device=self.device
            )

            # Current states (updated during forward pass)
            self.params_dict["W1_current"][layer_idx] = torch.zeros_like(
                self.params_dict["W1_init"][layer_idx]
            )
            self.params_dict["b1_current"][layer_idx] = torch.zeros_like(
                self.params_dict["b1_init"][layer_idx]
            )

        logger.debug(f"[TTTCache] Allocated storage for {self.num_layers} layers")

    def reset(
        self,
        conversation_id: Optional[str] = None,
        layer_initial_params: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = None
    ):
        """
        Reset TTT states for a new conversation.

        CRITICAL: This should ONLY be called when starting a new conversation,
        NOT between batches in the same conversation!

        Args:
            conversation_id: New conversation ID (if None, generates new ID)
            layer_initial_params: Optional dict mapping layer_idx to (W1, b1) tuples
                                 If provided, use these as initial parameters
                                 If None, reset to zeros

        The reset process:
        1. Update conversation_id
        2. Reset seqlen_offset to 0
        3. Reset all W1, b1 to initial values
        4. Increment reset counter
        """
        old_conversation_id = self.conversation_id
        self.conversation_id = conversation_id if conversation_id else f"conv_{self.reset_count}"

        logger.info(
            f"[TTTCache] Resetting state: {old_conversation_id} -> {self.conversation_id}"
        )

        self.seqlen_offset = 0
        self.reset_count += 1

        # Reset states for each layer
        for layer_idx in range(self.num_layers):
            if layer_initial_params and layer_idx in layer_initial_params:
                # Use provided initial parameters
                W1_layer, b1_layer = layer_initial_params[layer_idx]

                # Tile for batch dimension if needed
                if W1_layer.shape[0] != self.max_batch_size:
                    W1_layer = W1_layer.unsqueeze(0).expand(
                        self.max_batch_size, -1, -1, -1
                    )
                    b1_layer = b1_layer.unsqueeze(0).expand(
                        self.max_batch_size, -1, -1, -1
                    )

                self.params_dict["W1_init"][layer_idx].copy_(W1_layer.to(self.state_dtype))
                self.params_dict["b1_init"][layer_idx].copy_(b1_layer.to(self.state_dtype))
            else:
                # Reset to zeros (will be initialized from model parameters on first use)
                self.params_dict["W1_init"][layer_idx].zero_()
                self.params_dict["b1_init"][layer_idx].zero_()

            # Copy init to current
            self.params_dict["W1_current"][layer_idx].copy_(
                self.params_dict["W1_init"][layer_idx]
            )
            self.params_dict["b1_current"][layer_idx].copy_(
                self.params_dict["b1_init"][layer_idx]
            )

        logger.debug(
            f"[TTTCache] Reset complete. "
            f"conversation_id={self.conversation_id}, "
            f"total_resets={self.reset_count}"
        )

    def update(
        self,
        layer_idx: int,
        W1_new: torch.Tensor,
        b1_new: torch.Tensor,
        batch_size: Optional[int] = None
    ):
        """
        Update TTT state for a specific layer.

        Called after each forward pass to save the updated state.

        Args:
            layer_idx: Which layer to update
            W1_new: New W1 state [B*nh, head_dim, head_dim] or [B, nh, head_dim, head_dim]
            b1_new: New b1 state [B*nh, 1, head_dim] or [B, nh, 1, head_dim]
            batch_size: Current batch size (for reshaping if needed)

        The state is reshaped to [B, nh, ...] format for storage.
        """
        # Ensure float32
        W1_new = W1_new.to(self.state_dtype)
        b1_new = b1_new.to(self.state_dtype)

        # Reshape if needed: [B*nh, ...] -> [B, nh, ...]
        if W1_new.dim() == 3:  # [B*nh, head_dim, head_dim]
            if batch_size is None:
                raise ValueError("batch_size required when W1 is [B*nh, ...]")
            W1_new = W1_new.reshape(batch_size, self.num_heads, self.head_dim, self.head_dim)
            b1_new = b1_new.reshape(batch_size, self.num_heads, 1, self.head_dim)

        # Copy to cache (only for actual batch size, not full allocation)
        B = W1_new.shape[0]
        self.params_dict["W1_current"][layer_idx][:B].copy_(W1_new)
        self.params_dict["b1_current"][layer_idx][:B].copy_(b1_new)

        self.total_updates += 1

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"[TTTCache Layer {layer_idx}] Updated state: "
                f"W1 mean={W1_new.mean().item():.6f}, "
                f"b1 mean={b1_new.mean().item():.6f}"
            )

    def get(
        self,
        layer_idx: int,
        batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get current TTT state for a layer.

        Args:
            layer_idx: Which layer
            batch_size: Current batch size

        Returns:
            Tuple of (W1, b1):
            - W1: [B, nh, head_dim, head_dim] in float32
            - b1: [B, nh, 1, head_dim] in float32
        """
        W1 = self.params_dict["W1_current"][layer_idx][:batch_size]
        b1 = self.params_dict["b1_current"][layer_idx][:batch_size]

        # Ensure float32
        return W1.to(self.state_dtype), b1.to(self.state_dtype)

    def get_stats(self) -> Dict:
        """
        Get cache statistics for logging/debugging.

        Returns:
            Dictionary with statistics:
            - conversation_id: Current conversation
            - seqlen_offset: Current position
            - total_updates: Total state updates
            - reset_count: Number of resets
            - memory_mb: Approximate memory usage in MB
        """
        # Calculate memory usage
        memory_bytes = 0
        for layer_idx in range(self.num_layers):
            for key in ["W1_init", "b1_init", "W1_current", "b1_current"]:
                tensor = self.params_dict[key][layer_idx]
                memory_bytes += tensor.nelement() * tensor.element_size()

        memory_mb = memory_bytes / (1024 ** 2)

        return {
            "conversation_id": self.conversation_id,
            "seqlen_offset": self.seqlen_offset,
            "total_updates": self.total_updates,
            "reset_count": self.reset_count,
            "memory_mb": memory_mb,
            "num_layers": self.num_layers,
        }

    def save_checkpoint(self, path: str):
        """
        Save TTT cache to disk.

        Useful for:
        - Saving conversation state mid-way
        - Creating checkpoints during long generation
        - Debugging state evolution

        Args:
            path: File path to save to (.pt extension recommended)
        """
        checkpoint = {
            "conversation_id": self.conversation_id,
            "seqlen_offset": self.seqlen_offset,
            "total_updates": self.total_updates,
            "reset_count": self.reset_count,
            "params_dict": self.params_dict,
            "config": {
                "max_batch_size": self.max_batch_size,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "head_dim": self.head_dim,
            }
        }

        torch.save(checkpoint, path)
        logger.info(f"[TTTCache] Saved checkpoint to {path}")

    @classmethod
    def load_checkpoint(cls, path: str, device: Optional[torch.device] = None):
        """
        Load TTT cache from disk.

        Args:
            path: File path to load from
            device: Device to load tensors onto

        Returns:
            TTTCache instance with loaded state
        """
        checkpoint = torch.load(path, map_location=device)

        # Create cache with config from checkpoint
        cache = cls(
            max_batch_size=checkpoint["config"]["max_batch_size"],
            num_layers=checkpoint["config"]["num_layers"],
            num_heads=checkpoint["config"]["num_heads"],
            head_dim=checkpoint["config"]["head_dim"],
            device=device if device else torch.device('cuda'),
        )

        # Restore state
        cache.conversation_id = checkpoint["conversation_id"]
        cache.seqlen_offset = checkpoint["seqlen_offset"]
        cache.total_updates = checkpoint["total_updates"]
        cache.reset_count = checkpoint["reset_count"]
        cache.params_dict = checkpoint["params_dict"]

        logger.info(
            f"[TTTCache] Loaded checkpoint from {path}, "
            f"conversation={cache.conversation_id}, "
            f"offset={cache.seqlen_offset}"
        )

        return cache


def create_ttt_cache_from_config(config, max_batch_size: int = 1) -> TTTCache:
    """
    Helper function to create TTTCache from model config.

    Args:
        config: Model configuration (OmniSpeechConfig)
        max_batch_size: Maximum batch size to support

    Returns:
        Initialized TTTCache
    """
    # Determine which layers use TTT
    if config.ttt_layer_indices is not None:
        num_ttt_layers = len(config.ttt_layer_indices)
    else:
        # Default: top 8 layers
        num_ttt_layers = min(8, config.num_hidden_layers)

    num_heads = config.ttt_num_heads if config.ttt_num_heads else config.num_attention_heads
    head_dim = config.hidden_size // num_heads

    cache = TTTCache(
        max_batch_size=max_batch_size,
        num_layers=num_ttt_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    )

    return cache
