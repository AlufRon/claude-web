import functools
import logging
import math
from typing import Callable, Union

import safetensors
import torch
import torch.distributed.fsdp.wrap as torch_wrap
from moshi.models.lm import LMModel
from moshi.models.loaders import CheckpointInfo, _is_safetensors
from moshi.modules.transformer import StreamingTransformerLayer
from torch.distributed.fsdp import BackwardPrefetch
from torch.distributed.fsdp.api import ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel

from .args import TrainArgs
from .distributed import get_rank, get_world_size
from .ttt_integration import apply_ttt_to_model, log_ttt_parameters, verify_ttt_integration, parse_layer_specification
from .ttt_utils import is_ttt_parameter, initialize_ttt_parameter, get_training_mode_from_args, log_ttt_parameter_status, validate_ttt_parameters

# Import TTT hybrid layer for FSDP policy
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from moshi_ttt.hybrid_layer import HybridStreamingTransformerLayer

logger = logging.getLogger(__name__)


def main_logger_info(message: str) -> None:
    if get_rank() == 0:
        logger.info(message)


def get_fsdp_policy(is_lora: bool, is_ttt: bool = False) -> Callable[[torch.nn.Module], bool]:
    """
    This function instantiates the FSDP wrap policy.
    - Each Transformers block becomes its own FSDP group so that only a single
      Transformer block is sharded at a time
    - If LoRA or TTT is enabled, we additionally create separate FSDP sub-groups for
      every trainable and non-trainable parameter group since this is a
      requirement for mixed requires_grad=True/False training. See:
      https://pytorch.org/docs/stable/fsdp.html
    """

    # Each transformer block becomes a FSDP group, each being sharded separately
    # Include both original and TTT-enhanced transformer layers
    transformer_block_wrap_policy = functools.partial(
        torch_wrap.transformer_auto_wrap_policy,
        transformer_layer_cls=(StreamingTransformerLayer, HybridStreamingTransformerLayer),
    )

    if not is_lora and not is_ttt:
        return transformer_block_wrap_policy

    def fsdp_lora_policy_fn(module):
        return all(p.requires_grad for p in module.parameters())

    # For LoRA/TTT training, trainable and non-trainable parameters need to be put into
    # different FSDP groups
    fsdp_lora_policy = functools.partial(
        torch_wrap.lambda_auto_wrap_policy, lambda_fn=fsdp_lora_policy_fn
    )

    policies = [fsdp_lora_policy, transformer_block_wrap_policy]

    return functools.partial(torch_wrap._or_policy, policies=policies)


def log_train_params(model: Union[torch.nn.Module, FullyShardedDataParallel]):
    world_size = get_world_size()

    num_params = world_size * sum(p.numel() for p in model.parameters())
    num_train_params = world_size * sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    main_logger_info(
        f"{num_train_params:,.0f} out of {num_params:,.0f} parameters are finetuned "
        f"({num_train_params / num_params * 100:.2f}%)."
    )


def initialize_trainable_parameters(model: torch.nn.Module, param_dtype: torch.dtype, args: TrainArgs):
    """
    Initialize LoRA and TTT parameters properly.
    
    LoRA initialization follows the original paper:
    https://arxiv.org/abs/2106.09685
    
    TTT initialization follows Video-DiT patterns.
    """
    initialized_count = 0
    ttt_initialized_count = 0
    lora_initialized_count = 0
    
    for m_name, module in model.named_modules():
        if all(p.is_meta for p in module.parameters()):
            for p_name, param in module.named_parameters():
                # Convert meta tensor to real tensor
                module._parameters[p_name] = torch.nn.Parameter(
                    torch.empty_like(param, device="cpu", dtype=param_dtype)
                )
                param = module._parameters[p_name]
                
                # Build full parameter name for detection
                full_param_name = f"{m_name}.{p_name}" if m_name else p_name
                
                # Initialize based on parameter type
                if m_name.split(".")[-1] == "lora_A":
                    # LoRA A matrices: Kaiming uniform
                    torch.nn.init.kaiming_uniform_(param, a=math.sqrt(5))
                    lora_initialized_count += 1
                    logger.debug(f"Initialized LoRA A parameter: {full_param_name}")
                    
                elif m_name.split(".")[-1] == "lora_B":
                    # LoRA B matrices: zeros (important for LoRA)
                    torch.nn.init.zeros_(param)
                    lora_initialized_count += 1
                    logger.debug(f"Initialized LoRA B parameter: {full_param_name}")
                    
                elif is_ttt_parameter(full_param_name):
                    # TTT parameters: use specialized initialization
                    initialize_ttt_parameter(param, full_param_name, args.ttt.initial_gating_alpha)
                    ttt_initialized_count += 1
                    
                elif args.lora.ft_embed and "emb" in full_param_name.lower():
                    # Embedding parameters (if LoRA fine-tuning is enabled)
                    torch.nn.init.normal_(param, mean=0.0, std=0.02)
                    logger.debug(f"Initialized embedding parameter: {full_param_name}")
                    
                else:
                    # Other trainable parameters: small normal initialization
                    torch.nn.init.normal_(param, mean=0.0, std=0.02)
                    logger.debug(f"Initialized other parameter: {full_param_name}")
                
                initialized_count += 1
    
    main_logger_info(f"✅ Parameter initialization complete:")
    main_logger_info(f"   Total initialized: {initialized_count}")
    main_logger_info(f"   LoRA parameters: {lora_initialized_count}")
    main_logger_info(f"   TTT parameters: {ttt_initialized_count}")
    main_logger_info(f"   Other parameters: {initialized_count - lora_initialized_count - ttt_initialized_count}")


def is_attention_parameter_in_layer(param_name: str, layer_indices: list) -> bool:
    """Check if parameter is attention parameter in specified layers.

    Handles both direct and wrapped layer patterns:
    - Direct: transformer.layers.1.self_attn.in_proj_weight
    - Wrapped: transformer.layers.1.original_layer.self_attn.in_proj_weight
    """
    if not layer_indices:
        return False
    for layer_idx in layer_indices:
        # Check both direct and wrapped patterns
        direct_pattern = f"transformer.layers.{layer_idx}.self_attn."
        wrapped_pattern = f"transformer.layers.{layer_idx}.original_layer.self_attn."
        if direct_pattern in param_name or wrapped_pattern in param_name:
            return True
    return False


def configure_trainable_parameters(model: torch.nn.Module, args: TrainArgs) -> None:
    """
    Configure which parameters should be trainable based on training mode.

    Training modes:
    - frozen: No parameters trainable (baseline)
    - lora: Only LoRA parameters trainable
    - ttt: Only TTT parameters trainable (+ optionally attention in TTT layers)
    - lora+ttt: Both LoRA and TTT parameters trainable (+ optionally attention in TTT layers)
    - full: All parameters trainable
    """
    training_mode = get_training_mode_from_args(args)

    # Parse unfrozen attention layers if TTT enabled
    unfrozen_attn_layers = []
    if args.ttt.enable and args.ttt.unfrozen_attention_layers != "none":
        # Get total number of layers from model
        total_layers = len(model.transformer.layers)
        unfrozen_attn_layers = parse_layer_specification(
            args.ttt.unfrozen_attention_layers,
            total_layers
        )
        if unfrozen_attn_layers:
            main_logger_info(f"🔓 Unfreezing attention in layers: {unfrozen_attn_layers}")
    
    lora_count = 0
    ttt_count = 0
    attention_count = 0
    other_count = 0
    total_trainable = 0

    for name, param in model.named_parameters():
        should_train = False
        param_type = "other"
        
        if training_mode == "frozen":
            should_train = False
            
        elif training_mode == "full":
            should_train = True
            
        elif training_mode == "lora":
            if "lora" in name:
                should_train = True
                param_type = "lora"
            elif args.lora.ft_embed and "emb" in name:
                should_train = True
                param_type = "embedding"
            else:
                should_train = False
                
        elif training_mode == "ttt":
            if is_ttt_parameter(name):
                should_train = True
                param_type = "ttt"
            elif unfrozen_attn_layers and is_attention_parameter_in_layer(name, unfrozen_attn_layers):
                should_train = True
                param_type = "attention"
            else:
                should_train = False

        elif training_mode == "lora+ttt":
            if "lora" in name:
                should_train = True
                param_type = "lora"
            elif is_ttt_parameter(name):
                should_train = True
                param_type = "ttt"
            elif unfrozen_attn_layers and is_attention_parameter_in_layer(name, unfrozen_attn_layers):
                should_train = True
                param_type = "attention"
            elif args.lora.ft_embed and "emb" in name:
                should_train = True
                param_type = "embedding"
            else:
                should_train = False
        
        # Set requires_grad
        param.requires_grad = should_train
        
        # Count by type (order matters: attention before ttt to avoid misclassification)
        if should_train:
            total_trainable += param.numel()
            if param_type == "lora" or "lora" in name:
                lora_count += param.numel()
            elif param_type == "attention":
                attention_count += param.numel()
            elif param_type == "ttt" or is_ttt_parameter(name):
                ttt_count += param.numel()
            else:
                other_count += param.numel()
    
    # Log configuration
    total_params = sum(p.numel() for p in model.parameters())
    main_logger_info(f"🎯 Training mode: {training_mode}")
    main_logger_info(f"📊 Trainable parameters: {total_trainable:,} / {total_params:,} ({total_trainable/total_params*100:.2f}%)")
    if lora_count > 0:
        main_logger_info(f"   LoRA parameters: {lora_count:,}")
    if ttt_count > 0:
        main_logger_info(f"   TTT parameters: {ttt_count:,}")
    if attention_count > 0:
        main_logger_info(f"   Attention parameters: {attention_count:,}")
    if other_count > 0:
        main_logger_info(f"   Other parameters: {other_count:,}")


def get_fsdp_model(
    args: TrainArgs, checkpointer_info: CheckpointInfo
) -> FullyShardedDataParallel | LMModel:
    """
    Initializes and returns a FullyShardedDataParallel (FSDP) LMModel or a non sharded LMModel if one GPU available.
    Args:
        args (TrainArgs): A configuration object containing training arguments
            and settings. Key attributes include:
            - param_dtype: The data type for model parameters (e.g., "bfloat16", "float32").
            - gradient_checkpointing: Whether to enable gradient checkpointing.
            - lora: Configuration for LoRA fine-tuning, including enabling, rank, and scaling.
            - full_finetuning: Whether to enable full model fine-tuning or only LoRA fine-tuning.
        checkpointer_info: provide the initial checkpoint to train from.
    Notes:
        - The function uses meta-device initialization for memory efficiency.
        - Then parameters are initialized on the first GPU (rank=0) only.
    """

    if args.param_dtype == "bfloat16":
        param_dtype = torch.bfloat16
    elif args.param_dtype == "float32":
        param_dtype = torch.float32

    with torch.device("meta"):
        model = checkpointer_info.get_moshi(
            device="meta",
            dtype=param_dtype,
            lm_kwargs_overrides={
                "gradient_checkpointing": args.gradient_checkpointing,
                "lora": args.lora.enable,
                "lora_rank": args.lora.rank,
                "lora_scaling": args.lora.scaling,
            },
            load_weight=False,
        )

    if get_rank() == 0:
        moshi_weight = checkpointer_info.moshi_weights

        assert _is_safetensors(moshi_weight), "Model is not safetensors"
        model_state_dict = safetensors.torch.load_file(moshi_weight)

        logger.info(f"Converting model to dtype {param_dtype} ...")

        for k, v in model_state_dict.items():
            model_state_dict[k] = v.to(param_dtype)

        model.load_state_dict(model_state_dict, strict=False, assign=True)

        if (args.lora.enable or args.ttt.enable) and not args.full_finetuning:
            logger.info("Initializing trainable parameters (LoRA and TTT) ...")
            # initialize LoRA and TTT layers
            initialize_trainable_parameters(model, param_dtype, args)

        assert not any(p.is_meta for p in model.parameters()), (
            "All parameters should be initialized by now"
        )
        assert all(p.dtype == param_dtype for p in model.parameters()), (
            f"All parameters should be on {param_dtype}"
        )

        logger.info("Finished initialization!")
        param_init_fn = None
    else:

        def param_init_fn(m):
            m.to_empty(device=torch.cuda.current_device(), recurse=False)
            m.to(param_dtype)

        assert all(p.is_meta for p in model.parameters()), (
            "All parameters should be on meta"
        )

    torch.distributed.barrier()

    # Apply TTT integration before FSDP wrapping on ALL ranks
    # This ensures model structure is consistent across all processes
    main_logger_info("Applying TTT integration...")
    try:
        # Get LM config for TTT
        if checkpointer_info.raw_config is not None:
            lm_config = checkpointer_info.raw_config
        else:
            # For non-rank-0, parameters are on meta device, so use fallback config
            if get_rank() == 0:
                # Detect actual model dimensions from the loaded model
                try:
                    # Try multiple ways to get the actual dimensions
                    actual_dim = None
                    actual_heads = None
                    
                    # Method 1: Try transformer attributes
                    if hasattr(model, 'transformer'):
                        if hasattr(model.transformer, 'dim'):
                            actual_dim = model.transformer.dim
                        if hasattr(model.transformer, 'num_heads'):
                            actual_heads = model.transformer.num_heads
                    
                    # Method 2: Try from layers if available
                    if actual_dim is None and hasattr(model, 'transformer') and hasattr(model.transformer, 'layers'):
                        if len(model.transformer.layers) > 0:
                            first_layer = model.transformer.layers[0]
                            if hasattr(first_layer, 'self_attn') and hasattr(first_layer.self_attn, 'num_heads'):
                                actual_heads = first_layer.self_attn.num_heads
                            if hasattr(first_layer, 'self_attn') and hasattr(first_layer.self_attn, 'head_dim'):
                                head_dim = first_layer.self_attn.head_dim
                                if actual_heads:
                                    actual_dim = actual_heads * head_dim
                            # Try to get dim from layer norm
                            if hasattr(first_layer, 'norm1') and hasattr(first_layer.norm1, 'weight'):
                                actual_dim = first_layer.norm1.weight.shape[0]
                    
                    # Method 3: Check embedding dimensions
                    if actual_dim is None and hasattr(model, 'embed_tokens') and hasattr(model.embed_tokens, 'weight'):
                        actual_dim = model.embed_tokens.weight.shape[1]
                    
                    if actual_dim and actual_heads:
                        lm_config = {"dim": actual_dim, "num_heads": actual_heads}
                        main_logger_info(f"Detected model dimensions: dim={actual_dim}, heads={actual_heads}")
                    else:
                        # Fallback - try to get a reasonable estimate for Moshi 7B
                        lm_config = {"dim": 4096, "num_heads": 32}  # Typical Moshi 7B dimensions
                        main_logger_info(f"Using estimated Moshi 7B dimensions: dim=4096, heads=32")
                        
                except AttributeError as e:
                    # Fallback if we can't detect
                    lm_config = {"dim": 4096, "num_heads": 32}  # Better fallback for Moshi 7B
                    main_logger_info(f"Using fallback Moshi 7B dimensions: dim=4096, heads=32 (error: {e})")
            else:
                # Non-rank-0: Use fallback dimensions (will be synchronized by FSDP later)
                lm_config = {"dim": 4096, "num_heads": 32}
                
        apply_ttt_to_model(model, args.ttt, lm_config)
        verify_ttt_integration(model)
    except Exception as e:
        # ALWAYS FAIL LOUDLY for TTT integration failures - no silent fallbacks
        raise RuntimeError(
            f"TTT INTEGRATION CRITICAL FAILURE: TTT layer integration completely failed! "
            f"TTT is enabled (args.ttt.enable={args.ttt.enable}) but integration failed. "
            f"This indicates a fundamental problem with TTT layer creation or model compatibility. "
            f"Original error: {e}"
        ) from e

    # Configure attention context (LOCAL for TTT, GLOBAL for non-TTT)
    if args.ttt.ttt_layer_context is not None or args.ttt.non_ttt_layer_context is not None:
        main_logger_info("🔧 Configuring attention context (division of labor: local attention + global TTT)...")
        ttt_context = args.ttt.ttt_layer_context
        non_ttt_context = args.ttt.non_ttt_layer_context
        modified_ttt = 0
        modified_non_ttt = 0

        if hasattr(model, 'transformer') and hasattr(model.transformer, 'layers'):
            for layer in model.transformer.layers:
                # Regular StreamingTransformerLayer (non-TTT)
                if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'context'):
                    if non_ttt_context is not None:
                        layer.self_attn.context = non_ttt_context
                        modified_non_ttt += 1
                # HybridStreamingTransformerLayer (TTT)
                elif hasattr(layer, 'original_layer') and hasattr(layer.original_layer, 'self_attn'):
                    if hasattr(layer.original_layer.self_attn, 'context'):
                        if ttt_context is not None:
                            layer.original_layer.self_attn.context = ttt_context
                            modified_ttt += 1

            main_logger_info(f"✅ Set attention context: TTT layers={ttt_context} ({modified_ttt} layers), Non-TTT={non_ttt_context} ({modified_non_ttt} layers)")

    # Configure continuous RoPE on transformer layers (applies to ALL layers, not just TTT)
    if hasattr(args, 'rope_continuous') and args.rope_continuous:
        main_logger_info(f"Configuring continuous RoPE: rope_continuous={args.rope_continuous}, rope_reset_on_new_file={getattr(args, 'rope_reset_on_new_file', True)}")
        rope_reset = getattr(args, 'rope_reset_on_new_file', True)
        
        # DEBUG: Log model structure
        main_logger_info(f"  DEBUG: Model type: {type(model).__name__}")
        main_logger_info(f"  DEBUG: Model attributes: {[attr for attr in dir(model) if not attr.startswith('_')][:20]}")
        main_logger_info(f"  DEBUG: Has 'transformer': {hasattr(model, 'transformer')}")
        main_logger_info(f"  DEBUG: Has 'depformer': {hasattr(model, 'depformer')}")
        
        # Configure main transformer (it's a StreamingTransformer directly)
        if hasattr(model, 'transformer'):
            main_logger_info(f"  DEBUG: model.transformer type: {type(model.transformer).__name__}")
            main_logger_info(f"  DEBUG: model.transformer has rope_continuous: {hasattr(model.transformer, 'rope_continuous')}")
            model.transformer.rope_continuous = args.rope_continuous
            model.transformer.rope_reset_on_new_file = rope_reset
            main_logger_info(f"  ✓ Configured main transformer RoPE: rope_continuous={model.transformer.rope_continuous}, rope_reset_on_new_file={model.transformer.rope_reset_on_new_file}")
        else:
            main_logger_info("  ⚠️ Main transformer not found")
        
        # Configure depformer (dep_q) - it's a ProjectedTransformer
        if hasattr(model, 'depformer') and hasattr(model.depformer, 'transformer'):
            main_logger_info(f"  DEBUG: model.depformer.transformer type: {type(model.depformer.transformer).__name__}")
            model.depformer.transformer.rope_continuous = args.rope_continuous
            model.depformer.transformer.rope_reset_on_new_file = rope_reset
            main_logger_info(f"  ✓ Configured depformer RoPE: rope_continuous={model.depformer.transformer.rope_continuous}")
        else:
            main_logger_info("  ⚠️ Depformer not found")

    torch.distributed.barrier()

    # Configure trainable parameters based on training mode
    # Configure trainable parameters based on training mode
    configure_trainable_parameters(model, args)

    if get_world_size() == 1:
        return model.cuda()

    auto_wrap_policy = get_fsdp_policy(args.lora.enable, args.ttt.enable)

    main_logger_info(f"Sharding model over {get_world_size()} GPUs ...")

    wrapped_model = FullyShardedDataParallel(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=auto_wrap_policy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        limit_all_gathers=True,
        device_id=torch.cuda.current_device(),
        sync_module_states=True,
        param_init_fn=param_init_fn,
        use_orig_params=True,
    )

    main_logger_info("Model sharded!")

    # Add TTT state management methods to FSDP wrapped model for paper_metrics compatibility
    def save_ttt_states(self):
        """Save TTT states from all layers - safe for use during training."""
        from moshi_ttt.model_utils import save_ttt_states as save_states_impl
        return save_states_impl(self)
    
    def restore_ttt_states(self, saved_states):
        """Restore TTT states to all layers - safe for use during training."""
        from moshi_ttt.model_utils import restore_ttt_states as restore_states_impl
        return restore_states_impl(self, saved_states)
    
    def reset_ttt_states(self):
        """DEPRECATED: Reset TTT inner weights - breaks computation graph during training."""
        print("⚠️  WARNING: reset_ttt_states() is DEPRECATED and breaks computation graph")
        print("   Use save_ttt_states() before evaluation and restore_ttt_states() after")
        print("   This prevents 'Trying to backward through the graph a second time' errors")
        
        reset_count = 0
        total_resets = 0
        visited = set()  # Prevent infinite recursion
        
        def reset_module(module):
            nonlocal reset_count, total_resets
            
            # Prevent infinite recursion
            module_id = id(module)
            if module_id in visited:
                return
            visited.add(module_id)
            
            # Check if this is a TTTWrapper (not the top-level model)
            module_type = type(module).__name__
            if (hasattr(module, 'reset_ttt_states') and 
                callable(getattr(module, 'reset_ttt_states')) and
                module_type == 'TTTWrapper'):  # Only call on actual TTT layers
                total_resets += 1
                try:
                    success = module.reset_ttt_states()
                    if success:
                        reset_count += 1
                except Exception as e:
                    print(f"   Warning: Failed to reset TTT in {type(module).__name__}: {e}")
            
            # Recursively check submodules
            for child in module.children():
                reset_module(child)
        
        print("   🚫 BLOCKED: TTT reset would break training backward pass")
        print("   Switch to save/restore pattern in evaluation code")
        return False
    
    # Bind the methods to the wrapped model instance
    import types
    wrapped_model.save_ttt_states = types.MethodType(save_ttt_states, wrapped_model)
    wrapped_model.restore_ttt_states = types.MethodType(restore_ttt_states, wrapped_model)
    wrapped_model.reset_ttt_states = types.MethodType(reset_ttt_states, wrapped_model)

    log_train_params(wrapped_model)
    
    # Enhanced TTT parameter logging and validation
    log_ttt_parameter_status(wrapped_model, args)
    validate_ttt_parameters(wrapped_model)

    return wrapped_model
