#!/usr/bin/env python3
"""
TTT-Moshi Production Training Script
Train TTT-enhanced Moshi model on real DailyTalk dataset
Following the same structure as the original train.py
"""

import dataclasses
import logging
import os
import pprint
import shutil
from contextlib import ExitStack
from pathlib import Path

import fire
import torch.cuda
import torch.distributed as dist
from torch.optim import AdamW, lr_scheduler

from finetune.args import TrainArgs
from finetune.checkpointing import Checkpointer
from finetune.data.data_loader import build_data_loader
from finetune.data.interleaver import InterleavedTokenizer, Interleaver
from finetune.distributed import (
    BACKEND,
    avg_aggregate,
    get_rank,
    get_world_size,
    is_torchrun,
    set_device,
)
from finetune.eval import evaluate
from finetune.loss import compute_loss_with_mask
from finetune.mixed_precision import (
    downcast_mixed_precision,
    prepare_mixed_precision,
    upcast_mixed_precision,
)
from finetune.monitoring.metrics_logger import (
    MetricsLogger,
    eval_log_msg,
    get_eval_logs,
    get_train_logs,
    train_log_msg,
)
from finetune.monitoring.utils import set_logger
from finetune.utils import TrainState, logged_closing, set_random_seed
from finetune.wrapped_model import get_fsdp_model
from moshi.models import loaders

# TTT Integration imports
import sys
sys.path.insert(0, '.')
from moshi_ttt.hybrid_layer import HybridStreamingTransformerLayer
from moshi_ttt.config import TTTConfig

logger = logging.getLogger("train_ttt")


def main_logger_info(message: str) -> None:
    if get_rank() == 0:
        logger.info(message)


def apply_ttt_to_model(model, ttt_config: TTTConfig, ttt_layers="all"):
    """
    Apply TTT to specified layers of the Moshi model
    
    Args:
        model: The loaded Moshi LM model
        ttt_config: TTT configuration
        ttt_layers: Which layers to convert ("all", "middle", or list of indices)
    """
    main_logger_info("🔄 Applying TTT to Moshi model...")
    
    if not hasattr(model, 'transformer') or not hasattr(model.transformer, 'layers'):
        raise ValueError("Model doesn't have expected transformer.layers structure")
    
    total_layers = len(model.transformer.layers)
    main_logger_info(f"   Found {total_layers} transformer layers")
    
    # Determine which layers to convert
    if ttt_layers == "all":
        layer_indices = list(range(total_layers))
    elif ttt_layers == "middle":
        # Convert middle 50% of layers
        start = total_layers // 4
        end = 3 * total_layers // 4
        layer_indices = list(range(start, end))
    elif isinstance(ttt_layers, (list, tuple)):
        layer_indices = [i for i in ttt_layers if 0 <= i < total_layers]
    else:
        raise ValueError(f"Invalid ttt_layers specification: {ttt_layers}")
    
    main_logger_info(f"   Converting layers: {layer_indices}")
    
    # Count parameters before conversion
    original_params = sum(p.numel() for p in model.parameters())
    
    # Convert specified layers to TTT
    for layer_idx in layer_indices:
        original_layer = model.transformer.layers[layer_idx]
        hybrid_layer = HybridStreamingTransformerLayer(original_layer, ttt_config)
        model.transformer.layers[layer_idx] = hybrid_layer
        main_logger_info(f"   ✅ Layer {layer_idx}: StreamingTransformerLayer → HybridStreamingTransformerLayer")
    
    # Count parameters after conversion
    ttt_params = sum(p.numel() for p in model.parameters())
    param_increase = ttt_params - original_params
    param_increase_pct = (param_increase / original_params) * 100
    
    main_logger_info(f"✅ TTT conversion complete:")
    main_logger_info(f"   Converted {len(layer_indices)}/{total_layers} layers")
    main_logger_info(f"   Original parameters: {original_params:,}")
    main_logger_info(f"   TTT parameters: {ttt_params:,}")
    main_logger_info(f"   Parameter increase: +{param_increase:,} (+{param_increase_pct:.1f}%)")
    
    return model


def train_ttt(config: str, 
              ttt_layers: str = "middle",
              ttt_mini_batch_size: int = 16, 
              ttt_base_lr: float = 0.1):
    """
    Main training function for TTT-enhanced Moshi
    
    Args:
        config: Path to training configuration file
        ttt_layers: Which layers to convert ("all", "middle", or "none")
        ttt_mini_batch_size: TTT mini-batch size
        ttt_base_lr: TTT base learning rate
    """
    args: TrainArgs = TrainArgs.load(config, drop_extra_fields=False)
    set_logger(logging.INFO)
    
    main_logger_info("🚀 Starting TTT-Moshi training...")
    main_logger_info(f"   TTT layers: {ttt_layers}")
    main_logger_info(f"   TTT mini-batch size: {ttt_mini_batch_size}")
    main_logger_info(f"   TTT base LR: {ttt_base_lr}")

    with ExitStack() as exit_stack:
        _train_ttt(args, exit_stack, ttt_layers, ttt_mini_batch_size, ttt_base_lr)
    logger.info("Closed everything!")


def _train_ttt(args: TrainArgs, exit_stack: ExitStack, ttt_layers: str, 
               ttt_mini_batch_size: int, ttt_base_lr: float):
    """Internal training function with TTT integration"""
    
    # 1. Initial setup and checks (same as original)
    set_random_seed(args.seed)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Init NCCL
    if "LOCAL_RANK" in os.environ:
        set_device()
        logger.info("Going to init comms...")
        dist.init_process_group(backend=BACKEND)
    else:
        logger.error(
            "PyTorch environment is not correctly initialized. This message should only be displayed when testing."
        )

    # 2. Init run dir
    main_logger_info(f"Run dir: {args.run_dir}")
    run_dir = Path(args.run_dir)

    if is_torchrun():
        if run_dir.exists() and not args.overwrite_run_dir:
            raise RuntimeError(
                f"Run dir {run_dir} already exists. Make sure to either rename `run_dir` or remove {run_dir}."
            )
        elif run_dir.exists():
            main_logger_info(f"Removing run dir {run_dir}...")
            shutil.rmtree(run_dir)

    if args.full_finetuning:
        assert not args.lora.enable, "LoRA should not be enabled for full finetuning."
    else:
        assert args.lora.enable, "LoRA should be enabled for partial finetuning"

    dist.barrier()
    run_dir.mkdir(exist_ok=True, parents=True)

    args_path = run_dir / "args.yaml"
    if not args_path.exists():
        args.save(args_path)

    main_logger_info(f"TrainArgs: {pprint.pformat(dataclasses.asdict(args))}")

    # 3. Get loggers
    metrics_logger: MetricsLogger = MetricsLogger(
        run_dir,
        tag="train",
        is_master=get_rank() == 0,
        wandb_args=args.wandb,
        config=dataclasses.asdict(args),
    )
    exit_stack.enter_context(logged_closing(metrics_logger, "metrics_logger"))

    eval_logger: MetricsLogger = MetricsLogger(
        run_dir,
        tag="eval",
        is_master=get_rank() == 0,
        wandb_args=args.wandb,
        config=dataclasses.asdict(args),
    )
    exit_stack.enter_context(logged_closing(eval_logger, "eval_logger"))

    # 4.1 Load function calling audio encoder and tokenizer
    main_logger_info("Loading Mimi and Moshi...")
    checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
        hf_repo=args.moshi_paths.hf_repo_id,
        moshi_weights=args.moshi_paths.moshi_path,
        mimi_weights=args.moshi_paths.mimi_path,
        tokenizer=args.moshi_paths.tokenizer_path,
        config_path=args.moshi_paths.config_path,
    )

    lm_config = (
        loaders._lm_kwargs
        if checkpoint_info.raw_config is None
        else checkpoint_info.raw_config
    )
    lm_config["lora"] = args.lora.enable
    lm_config["lora_rank"] = args.lora.rank
    lm_config["lora_scaling"] = args.lora.scaling

    mimi = checkpoint_info.get_mimi(device="cuda")
    mimi.eval()
    for p in mimi.parameters():
        p.requires_grad = False

    # 4.2 Load and shard model, prepare interleaver for audio/text tokens.
    main_logger_info("Loading base Moshi model...")
    model = get_fsdp_model(args, checkpoint_info)
    main_logger_info(f"✅ Base model loaded: {type(model)}")

    # 🔥 TTT INTEGRATION: Apply TTT to the loaded model
    if ttt_layers != "none":
        ttt_config = TTTConfig(
            model_dim=lm_config['dim'],
            num_heads=lm_config['num_heads'],
            mini_batch_size=ttt_mini_batch_size,
            ttt_base_lr=ttt_base_lr
        )
        model = apply_ttt_to_model(model, ttt_config, ttt_layers)
    else:
        main_logger_info("🚫 TTT disabled - using vanilla Moshi")

    spm = checkpoint_info.get_text_tokenizer()

    interleaver = Interleaver(
        spm,
        mimi.frame_rate,
        model.text_padding_token_id,
        model.end_of_text_padding_id,
        model.zero_token_id,
        keep_main_only=True,
    )
    interleaved_tokenizer = InterleavedTokenizer(
        mimi, interleaver, duration_sec=args.duration_sec
    )

    # 5. Load data loaders
    main_logger_info("Loading data loaders...")
    data_loader = build_data_loader(
        instruct_tokenizer=interleaved_tokenizer,
        args=args.data,
        batch_size=args.batch_size,
        seed=args.seed,
        rank=get_rank(),  # DDP rank
        world_size=get_world_size(),  # DDP world_size
        is_eval=False,
    )

    if args.do_eval:
        eval_data_loader = build_data_loader(
            instruct_tokenizer=interleaved_tokenizer,
            args=args.data,
            batch_size=args.batch_size,
            seed=None,
            rank=get_rank(),  # DDP rank
            world_size=get_world_size(),  # DDP world_size
            is_eval=True,
        )

    # 6. Load model (already done above with TTT integration)
    # Define mixed precision
    param_dtype = getattr(torch, args.param_dtype)
    optim_dtype = torch.float32

    assert args.lora is not None, "`args.lora` should be set to a valid value."

    # 7. Load optimizer
    main_logger_info("Setting up optimizer...")
    optimizer = AdamW(
        model.parameters(),
        lr=args.optim.lr,
        betas=(0.9, 0.95),
        eps=1e-08,
        weight_decay=args.optim.weight_decay,
    )

    scheduler = lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.optim.lr,
        total_steps=args.max_steps,
        pct_start=args.optim.pct_start,
    )

    state = TrainState(args.max_steps)

    # 8. Initialize checkpointer
    if args.do_ckpt:
        checkpointer = Checkpointer(
            model=model,
            state=state,
            config=lm_config,
            run_dir=run_dir,
            optimizer=optimizer,
            num_ckpt_keep=args.num_ckpt_keep,
            full_finetuning=args.full_finetuning,
        )

    # 9. Prepare mixed precision
    prepare_mixed_precision(
        model.parameters(), param_dtype=param_dtype, optim_dtype=optim_dtype
    )

    # 10. 🔥 TTT-specific training monitoring
    ttt_metrics = {
        'ttt_params': sum(1 for name, _ in model.named_parameters() 
                         if any(ttt_key in name for ttt_key in ['W1', 'W2', 'b1', 'b2', 'ttt_norm', 'learnable_ttt'])),
        'total_params': sum(1 for _ in model.named_parameters()),
    }
    main_logger_info(f"📊 TTT Training Metrics:")
    main_logger_info(f"   TTT parameters: {ttt_metrics['ttt_params']}")
    main_logger_info(f"   Total parameters: {ttt_metrics['total_params']}")

    # 11. Train! (same as original with TTT monitoring)
    main_logger_info("🚀 Starting training loop...")
    model.train()
    torch.cuda.empty_cache()

    while state.step < args.max_steps:
        state.start_step()
        is_last_step = state.step == args.max_steps

        optimizer.zero_grad()

        loss = torch.tensor([0.0], device="cuda")
        n_batch_tokens: int = 0
        n_real_tokens: int = 0

        for i in range(args.num_microbatches):
            batch = next(data_loader)
            codes = batch.codes

            condition_tensors = None
            if batch.condition_attributes is not None:
                condition_tensors = model.condition_provider.prepare(
                    batch.condition_attributes
                )

            # forward / backward (same as original)
            output = model(codes=codes, condition_tensors=condition_tensors)
            text_loss = compute_loss_with_mask(
                output.text_logits,
                codes[:, : model.audio_offset],
                output.text_mask,
                mode="text",
                text_padding_weight=args.text_padding_weight,
                text_padding_ids={
                    model.text_padding_token_id,
                    model.end_of_text_padding_id,
                },
            )
            audio_loss = compute_loss_with_mask(
                output.logits,
                codes[:, model.audio_offset : model.audio_offset + model.dep_q],
                output.mask,
                mode="audio",
                first_codebook_weight_multiplier=args.first_codebook_weight_multiplier,
            )

            mb_loss = text_loss + audio_loss
            mb_loss.backward()

            loss += mb_loss.detach()
            n_batch_tokens += output.text_mask.numel() + output.mask.numel()
            n_real_tokens += (
                torch.sum(output.text_mask).item() + torch.sum(output.mask).item()
            )

            if i < args.num_microbatches - 1:
                # synchronize CUDA to re-run backward
                assert args.num_microbatches > 1  # should not happen
                torch.cuda.synchronize()

        if args.num_microbatches > 1:
            loss /= args.num_microbatches
            for p in model.parameters():
                if p.requires_grad:
                    assert p.grad is not None
                    p.grad.div_(args.num_microbatches)

        # upcast params for optimizer update
        upcast_mixed_precision(model.parameters(), optim_dtype=optim_dtype)

        # clip grad norm
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)

        # optimizer step
        optimizer.step()

        # downcast params for forward & backward
        downcast_mixed_precision(model.parameters(), param_dtype=param_dtype)

        last_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        # Host sync
        loss_item = loss.item()
        avg_loss = avg_aggregate(loss_item)

        if args.do_eval and (
            (args.eval_freq > 0 and state.step % args.eval_freq == 0) or is_last_step
        ):
            # write perplexity to state
            evaluate(model, eval_data_loader, state, args)

            eval_logs = get_eval_logs(
                state.step,
                avg_loss,
                state.this_eval_perplexity,
                state.this_eval_loss,
            )

            main_logger_info(eval_log_msg(eval_logs))
            eval_logger.log(eval_logs, step=state.step)

        # Timing
        state.end_step(n_batch_tokens)

        if state.step % args.log_freq == 0:
            train_logs = get_train_logs(
                state,
                avg_loss,
                n_real_tokens,
                last_lr,
                torch.cuda.max_memory_allocated(),
                torch.cuda.memory_allocated(),
                args,
                model,  # Pass model to access TTT gating alpha
            )
            
            # 🔥 Add TTT-specific logs
            train_logs['ttt_enabled'] = ttt_layers != "none"
            train_logs['ttt_layers'] = ttt_layers
            
            main_logger_info(train_log_msg(state, logs=train_logs, loss=avg_loss))
            metrics_logger.log(train_logs, step=state.step)

        if args.do_ckpt and (
            (args.ckpt_freq > 0 and state.step % args.ckpt_freq == 0) or is_last_step
        ):
            checkpointer.save_checkpoint(
                save_only_lora=not args.full_finetuning and args.save_adapters,
                dtype=param_dtype,
            )

    main_logger_info("🎉 TTT-Moshi training completed!")


if __name__ == "__main__":
    """Usage:
    
    # Train with TTT on middle layers
    python train_ttt.py config.yaml --ttt_layers=middle
    
    # Train with TTT on all layers  
    python train_ttt.py config.yaml --ttt_layers=all
    
    # Train vanilla Moshi (no TTT)
    python train_ttt.py config.yaml --ttt_layers=none
    """
    fire.Fire(train_ttt)