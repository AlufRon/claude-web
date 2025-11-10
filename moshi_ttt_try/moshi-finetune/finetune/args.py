import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from simple_parsing.helpers import Serializable

from .data.args import DataArgs


@dataclass
class LoraArgs(Serializable):
    enable: bool = False
    rank: int = 64
    scaling: float = 2.0
    ft_embed: bool = False

    def __post_init__(self) -> None:
        if self.enable:
            assert self.rank > 0
            assert self.scaling > 0.0


@dataclass
class OptimArgs(Serializable):
    lr: float = 1e-4
    weight_decay: float = 0.1
    pct_start: float = 0.05


@dataclass
class WandbArgs(Serializable):
    project: str | None = None  # Fill this argument to use wandb.
    offline: bool = False
    key: str | None = None
    run_name: str | None = None

    def __post_init__(self) -> None:
        if self.project is not None:
            try:
                import wandb  # noqa: F401
            except ImportError:
                raise ImportError(
                    "`wandb` not installed. Either make sure `wandb` is installed or set `wandb:project` to None."
                )

            if len(self.project) == 0:
                raise ValueError("`wandb.project` must not be an empty string.")


@dataclass
class ModelPaths(Serializable):
    hf_repo_id: str | None = "kyutai/moshiko-pytorch-bf16"
    mimi_path: str | None = None
    moshi_path: str | None = None
    tokenizer_path: str | None = None
    config_path: str | None = None

    def __post_init__(self) -> None:
        if self.hf_repo_id is not None and self.config_path is None:
            print(
                "Warning: `hf_repo_id` is set but `config_path` is None. "
                "This will load default models."
            )


@dataclass
class TTTArgs(Serializable):
    """Configuration for Test-Time Training (TTT) layers in Moshi"""
    enable: bool = False
    layers: str = "middle"  # "all", "middle", "none", or comma-separated indices like "1,3,5"
    base_lr: float = 1.0
    mini_batch_size: int = 16
    persistent_states: bool = True  # Enable TTT state persistence across training steps (JAX-style)
    initial_gating_alpha: float = 0.1  # Initial gating alpha for TTT layers
    override_gating_alpha_on_resume: bool = False  # Reset gating alpha when resuming from checkpoint

    # Attention Context Configuration (division of labor: local attention + global TTT)
    ttt_layer_context: Optional[int] = None  # Attention context for TTT layers (e.g., 750 for 60s local). None = use Moshi default
    non_ttt_layer_context: Optional[int] = None  # Attention context for non-TTT layers. None = use Moshi default

    # Multi-Learning-Rate Configuration
    weight_lr_multiplier: float = 10.0    # TTT weights LR multiplier (relative to base optim.lr)
    alpha_lr_multiplier: float = 100.0   # Gating alpha LR multiplier (relative to base optim.lr)

    # TTT Inner Loop Diagnostics (Figure 4 from paper)
    log_inner_loop_losses: bool = False  # Enable logging of reconstruction losses during mini-batch iterations
    inner_loop_log_interval: int = 1  # Log every N mini-batches (1 = all batches)
    save_inner_loop_plots: bool = False  # Automatically generate Figure 4 plots during evaluation
    inner_loop_plot_dir: str = "./evaluation_plots/inner_loop"  # Directory for inner loop plots
    
    # TTT-Optimized LibriLight evaluation settings
    optimize_chunk_size: bool = True  # Enable TTT-optimized chunking for LibriLight evaluation
    chunk_size: Optional[int] = None  # Override chunk size; if None, calculated from mini_batch_size
    max_chunk_size: int = 50  # Maximum chunk size (for memory constraints)
    prefer_efficiency: bool = True  # Prefer 100% efficiency over smaller chunk sizes
    
    # TTT-MLP Multi-layer Configuration (NEW)
    ttt_mlp_layers: int = 2  # Number of MLP layers (default: 2, use 3+ for multi-layer)
    ttt_mlp_hidden_dims: Optional[list] = None  # Custom layer dimensions [dim1, dim2, ...]; if None, use expansion_factor
    ttt_mlp_expansion_factor: float = 4.0  # Default expansion ratio for auto-sizing layers (head_dim * factor)
    
    # TTT Output Normalization (NEW)
    normalize_ttt_output: bool = False  # Enable learnable output scaling to match attention magnitude
    target_output_norm: float = 25.0  # Target L2 norm for TTT output (default: ~25 to match attention)

    # RoPE Configuration (following ttt-lm-pytorch pattern)
    use_rope: bool = False  # Enable Rotary Position Embeddings (default: False for backward compatibility)
    rope_theta: float = 10000.0  # Base for RoPE frequency computation (standard value)

    # Selective Attention Unfreezing (Video-DiT approach)
    unfrozen_attention_layers: str = "none"  # "all", "middle", "none", or comma-separated indices like "1,3,5"

    # Training Diagnostics (magnitude/gating analysis during training)
    enable_training_diagnostics: bool = False  # Enable TTT diagnostics during training
    training_diagnostic_frequency: int = 100  # Log diagnostics every N training steps

    def __post_init__(self) -> None:
        if self.enable:
            assert self.base_lr > 0.0, "TTT base learning rate must be positive"
            assert self.mini_batch_size > 0, "TTT mini batch size must be positive"
            valid_layers = ["all", "middle", "none"]
            if self.layers not in valid_layers:
                # Check if it's a comma-separated list of integers
                try:
                    layer_indices = [int(x.strip()) for x in self.layers.split(",")]
                    assert all(idx >= 0 for idx in layer_indices), "Layer indices must be non-negative"
                except (ValueError, AssertionError):
                    raise ValueError(f"TTT layers must be one of {valid_layers} or comma-separated layer indices")
        
        # Validate TTT-optimized chunking configuration
        if self.chunk_size is not None:
            assert self.chunk_size > 0, "TTT chunk size must be positive"
            assert self.chunk_size <= self.max_chunk_size, f"TTT chunk size {self.chunk_size} exceeds max {self.max_chunk_size}"
        assert self.max_chunk_size > 0, "TTT max chunk size must be positive"
        
        # Validate TTT-MLP multi-layer configuration
        assert self.ttt_mlp_layers >= 2, "TTT-MLP must have at least 2 layers"
        assert self.ttt_mlp_expansion_factor > 0.0, "TTT-MLP expansion factor must be positive"
        
        if self.ttt_mlp_hidden_dims is not None:
            assert len(self.ttt_mlp_hidden_dims) == self.ttt_mlp_layers - 1, \
                f"TTT-MLP hidden_dims must specify {self.ttt_mlp_layers - 1} dimensions for {self.ttt_mlp_layers} layers"
            assert all(dim > 0 for dim in self.ttt_mlp_hidden_dims), \
                "All TTT-MLP hidden dimensions must be positive"
        
        # Validate TTT output normalization
        if self.normalize_ttt_output:
            assert self.target_output_norm > 0.0, "TTT target output norm must be positive"

        # Validate RoPE configuration
        if self.use_rope:
            assert self.rope_theta > 0.0, "RoPE theta must be positive"

    def get_optimal_chunk_size(self) -> int:
        """Calculate optimal chunk size for TTT evaluation based on configuration."""
        if not self.optimize_chunk_size:
            return self.max_chunk_size  # Use legacy behavior
        
        if self.chunk_size is not None:
            return min(self.chunk_size, self.max_chunk_size)  # Use explicit override
        
        # Calculate optimal chunk size based on mini_batch_size
        mini_batch = self.mini_batch_size
        max_chunk = self.max_chunk_size
        
        if self.prefer_efficiency:
            # Find largest divisor of max_chunk_size that's >= mini_batch_size
            # This gives 100% efficiency with largest possible chunks
            for chunk_size in range(max_chunk, mini_batch - 1, -1):
                if max_chunk % chunk_size == 0:
                    return chunk_size
            # Fallback: use mini_batch_size for perfect efficiency
            return mini_batch
        else:
            # Use mini_batch_size directly for maximum adaptation granularity
            return min(mini_batch, max_chunk)
    
    def calculate_efficiency(self, chunk_size: Optional[int] = None) -> dict:
        """Calculate TTT processing efficiency metrics for a given chunk size."""
        if chunk_size is None:
            chunk_size = self.get_optimal_chunk_size()
            
        mini_batch = self.mini_batch_size
        num_complete_batches = chunk_size // mini_batch
        remaining_tokens = chunk_size % mini_batch
        
        if remaining_tokens > 0:
            # Last mini-batch needs padding
            padding_tokens = mini_batch - remaining_tokens
            total_processed = chunk_size + padding_tokens
            efficiency = chunk_size / total_processed
            num_batches = num_complete_batches + 1
        else:
            # Perfect alignment
            efficiency = 1.0
            num_batches = num_complete_batches
            padding_tokens = 0
        
        return {
            "chunk_size": chunk_size,
            "mini_batch_size": mini_batch,
            "efficiency_percent": efficiency * 100,
            "padding_tokens": padding_tokens,
            "num_mini_batches": num_batches,
            "tokens_per_batch": chunk_size / num_batches if num_batches > 0 else 0,
            "total_processed_tokens": total_processed if remaining_tokens > 0 else chunk_size
        }


@dataclass
class TrainArgs(Serializable):
    data: DataArgs

    run_dir: str  # Path to the directory where everything will be saved. It needs to be empty.
    # Name of the wandb run, if None it will be set to the name of the run_dir.
    moshi_paths: ModelPaths = field(default_factory=ModelPaths)
    first_codebook_weight_multiplier: float = 1.0
    text_padding_weight: float = 0.5

    optim: OptimArgs = field(default_factory=OptimArgs)
    seed: int = 0
    # Number of steps to accumulate gradients before doing an optimizer step.
    num_microbatches: int = 1

    duration_sec: float = 10
    batch_size: int = 1
    max_norm: float = 1.0  # Gradient clipping.
    max_steps: int = 100  # Number of training steps.
    log_freq: int = 1  # Number of steps between each logging.

    # Number of steps between each checkpoint saving. If inferior to 1, only the last checkpoint will be saved.
    ckpt_freq: int = 0
    save_adapters: bool = True
    # If False, no checkpoints will be saved. This is useful for development.
    do_ckpt: bool = True
    num_ckpt_keep: int | None = 3
    eval_freq: int = 0
    do_eval: bool = False

    # Efficiency
    # Determines whether gradient checkpointing should be utilized or not
    # during the training process. Gradient checkpointing can be beneficial in
    # reducing memory usage at the cost of slightly longer training times.
    gradient_checkpointing: bool = True

    world_size: int | None = field(init=False, default=None)

    # logging
    wandb: WandbArgs = field(default_factory=WandbArgs)

    # LoRA
    lora: LoraArgs | None = field(default_factory=LoraArgs)
    full_finetuning: bool = False

    # TTT (Test-Time Training)
    ttt: TTTArgs = field(default_factory=TTTArgs)

    # Paper metrics evaluation
    paper_metrics: dict = field(default_factory=dict)

    param_dtype: str = "bfloat16"

    overwrite_run_dir: bool = False
    
    # Continuous RoPE (applies to all Moshi transformer layers)
    rope_continuous: bool = False  # Enable continuous RoPE positions across chunks from same file
    rope_reset_on_new_file: bool = True  # Reset positions when new file begins

    # Checkpoint resuming for transfer learning or continuing training
    resume_from: str | None = None      # Path to checkpoint/consolidated/ directory to resume from
    reset_optimizer: bool = False       # Start fresh optimizer (for transfer learning)
    reset_step: bool = False            # Reset step counter to 0 (for transfer learning)

    def __post_init__(self) -> None:
        assert getattr(self, "world_size", None) is None
        self.world_size = int(os.environ.get("WORLD_SIZE", -1))

        if self.wandb.offline:
            command = f"cd {self.run_dir}; wandb sync --sync-all"
            logging.info(f"to sync wandb offline, run: {command}")

        assert self.num_microbatches >= 1

        assert self.num_ckpt_keep is None or self.num_ckpt_keep >= 1

        if not self.save_adapters:
            logging.warning(
                "You have disabled `save_adapters` and are thus merging the "
                "trained LoRA checkpoint into the base model upon checkpointing. "
                "This might lead to OOM errors - make sure you have enough CPU "
                "and GPU memory."
            )
