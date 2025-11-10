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

# from torch.profiler import ProfilerActivity, profile

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
# Lazy import to avoid torchaudio crash if not doing evaluation
# from finetune.paper_metrics import PaperMetricsEvaluator
from inference.enable_ttt_diagnostics import enable_ttt_diagnostics
from moshi.models import loaders

logger = logging.getLogger("train")

# Global flag for one-time debug logging
_logged_file_id_debug = False


def main_logger_info(message: str) -> None:
    if get_rank() == 0:
        logger.info(message)


def train(config: str):
    args: TrainArgs = TrainArgs.load(config, drop_extra_fields=False)
    set_logger(logging.INFO)

    with ExitStack() as exit_stack:
        _train(args, exit_stack)
    logger.info("Closed everything!")


def _train(args: TrainArgs, exit_stack: ExitStack):
    # 1. Initial setup and checks
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
        # For partial finetuning, either LoRA or TTT (or both) should be enabled
        # TEMPORARILY DISABLED: Allow running with both disabled for frozen baseline testing
        # if not args.lora.enable and not args.ttt.enable:
        #     raise ValueError("For partial finetuning, either LoRA or TTT must be enabled")
        pass

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
    model = get_fsdp_model(args, checkpoint_info)

    # 4.3 Enable TTT diagnostics if requested
    if args.ttt.enable and args.ttt.enable_training_diagnostics:
        main_logger_info("")
        main_logger_info("="*80)
        main_logger_info("Enabling TTT Training Diagnostics")
        main_logger_info("="*80)
        num_enabled = enable_ttt_diagnostics(
            model=model,
            log_frequency=args.ttt.training_diagnostic_frequency,
            track_history=False  # Don't track history during training (saves memory)
        )
        if num_enabled == 0:
            main_logger_info("⚠️  No TTT layers found to enable diagnostics")
        else:
            main_logger_info(f"✅ Enabled diagnostics on {num_enabled} TTT layer(s)")
            main_logger_info(f"   Logging every {args.ttt.training_diagnostic_frequency} forward passes")
        main_logger_info("="*80)
        main_logger_info("")

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

    # 6. Load model
    # Define mixed precision
    param_dtype = getattr(torch, args.param_dtype)
    optim_dtype = torch.float32

    assert args.lora is not None, "`args.lora` should be set to a valid value."

    # 7. Load optimizer with multi-learning-rate support
    from finetune.ttt_utils import get_parameter_groups

    # Get parameter groups for different learning rates
    param_groups_dict = get_parameter_groups(model)

    # Create optimizer with parameter-specific learning rates
    optimizer_param_groups = [
        {
            'params': param_groups_dict['base'],
            'lr': args.optim.lr,
            'name': 'base',
        },
        {
            'params': param_groups_dict['ttt_weights'],
            'lr': args.optim.lr * args.ttt.weight_lr_multiplier,
            'name': 'ttt_weights',
        },
        {
            'params': param_groups_dict['ttt_alpha'],
            'lr': args.optim.lr * args.ttt.alpha_lr_multiplier,
            'name': 'ttt_alpha',
        },
    ]

    # Log parameter counts and learning rates per group
    main_logger_info("📊 Parameter Groups and Learning Rates:")
    for group in optimizer_param_groups:
        num_params = sum(p.numel() for p in group['params'])
        lr_multiplier = group['lr'] / args.optim.lr if args.optim.lr > 0 else 1.0
        main_logger_info(
            f"   {group['name']:15s}: {num_params:12,} params, "
            f"lr={group['lr']:.2e} ({lr_multiplier:7.0f}x base)"
        )

    optimizer = AdamW(
        optimizer_param_groups,
        betas=(0.9, 0.95),
        eps=1e-08,
        weight_decay=args.optim.weight_decay,
    )

    # Create scheduler with per-group max learning rates
    scheduler = lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[
            args.optim.lr,                                      # base
            args.optim.lr * args.ttt.weight_lr_multiplier,     # ttt_weights
            args.optim.lr * args.ttt.alpha_lr_multiplier,      # ttt_alpha
        ],
        total_steps=args.max_steps,
        pct_start=args.optim.pct_start,
    )

    state = TrainState(args.max_steps)

    # NEW: Load checkpoint if resuming
    if args.resume_from is not None:
        main_logger_info("=" * 80)
        main_logger_info("📂 RESUMING FROM CHECKPOINT")
        main_logger_info("=" * 80)
        main_logger_info(f"   Checkpoint path: {args.resume_from}")

        from finetune.checkpoint_loader import CheckpointLoader

        load_optimizer = not getattr(args, 'reset_optimizer', False)
        resume_step, checkpoint_config = CheckpointLoader.load_checkpoint(
            checkpoint_path=args.resume_from,
            model=model,
            optimizer=optimizer if load_optimizer else None,
            load_optimizer=load_optimizer,
            strict=False  # Allow missing keys for base model
        )

        # Verify TTT config
        current_config = dataclasses.asdict(args)
        CheckpointLoader.verify_ttt_config(checkpoint_config, current_config)

        # Override gating alpha if requested (AFTER checkpoint loading)
        if args.ttt.enable and args.ttt.override_gating_alpha_on_resume:
            main_logger_info(f"🔄 Overriding gating alpha from checkpoint...")
            main_logger_info(f"   Checkpoint had gating alpha values from training")
            main_logger_info(f"   Resetting to initial_gating_alpha = {args.ttt.initial_gating_alpha}")
            
            from finetune.ttt_integration import reset_gating_alpha
            reset_gating_alpha(model, args.ttt.initial_gating_alpha)
            
            main_logger_info(f"✅ Gating alpha override complete")
            main_logger_info(f"   Use case: Reduce TTT influence when fine-tuning on new data")

        # Set starting step
        if not getattr(args, 'reset_step', False):
            state.step = resume_step
            main_logger_info(f"   Resuming from step {resume_step}")
        else:
            state.step = 0
            main_logger_info(f"   Reset step counter to 0 (transfer learning)")

        if getattr(args, 'reset_optimizer', False):
            main_logger_info(f"   Reset optimizer (fresh start for new task)")

        main_logger_info("✅ Checkpoint loaded successfully")
        main_logger_info("=" * 80)

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
            training_args=args,
        )
    # 9. Prepare mixed precision
    prepare_mixed_precision(
        model.parameters(), param_dtype=param_dtype, optim_dtype=optim_dtype
    )

    # 11. train!
    model.train()
    torch.cuda.empty_cache()

    # Track previous file_id for TTT inner weight reset on file switching
    previous_file_id = None
    ttt_reset_count = 0
    
    while state.step < args.max_steps:
        state.start_step()
        is_last_step = state.step == args.max_steps
        
        # Set debug flag for TTT normalization logging (every 10 steps for better coverage)
        if state.step % 10 == 0 and state.step < 500:
            os.environ['TTT_NORM_DEBUG_STEP'] = str(state.step)
        else:
            os.environ['TTT_NORM_DEBUG_STEP'] = '-1'

        optimizer.zero_grad()

        loss = torch.tensor([0.0], device="cuda")
        n_batch_tokens: int = 0
        n_real_tokens: int = 0

        for i in range(args.num_microbatches):
            batch = next(data_loader)
            codes = batch.codes
            
            # Extract file_id for continuous RoPE (if available)
            file_id = None
            if hasattr(batch, 'file_id') and batch.file_id is not None:
                file_id = batch.file_id[0] if isinstance(batch.file_id, list) else batch.file_id
                
            # DEBUG: Log file_id once
            global _logged_file_id_debug
            if not _logged_file_id_debug:
                logger.info(f"[TRAIN-DEBUG] batch has file_id attr: {hasattr(batch, 'file_id')}")
                logger.info(f"[TRAIN-DEBUG] batch.file_id value: {getattr(batch, 'file_id', 'MISSING')}")
                logger.info(f"[TRAIN-DEBUG] extracted file_id: {file_id}")
                _logged_file_id_debug = True

            # ====================================================================
            # TTT INNER WEIGHT RESET ON FILE SWITCHING
            # ====================================================================
            # When switching to a new file AND rope_reset_on_new_file is enabled,
            # we should also reset TTT inner weights (W1, W2, W3, b1, b2, b3)
            # to prevent catastrophic interference from previous file's adaptations
            if (file_id is not None and 
                previous_file_id is not None and 
                file_id != previous_file_id and
                getattr(args, 'rope_reset_on_new_file', False)):
                
                main_logger_info(f"🔄 [TTT-RESET] File switch detected: {previous_file_id} → {file_id}")
                main_logger_info(f"🔄 [TTT-RESET] Resetting TTT inner weights (W1, W2, W3, b1, b2, b3) to learned initialization")
                
                # Reset TTT inner weights for all TTT layers
                reset_success = False
                reset_layer_count = 0
                
                # Handle FSDP-wrapped model
                actual_model = model
                if hasattr(model, '_fsdp_wrapped_module'):
                    actual_model = model._fsdp_wrapped_module
                elif hasattr(model, 'module'):
                    actual_model = model.module
                
                # Reset each TTT layer
                if hasattr(actual_model, 'transformer') and hasattr(actual_model.transformer, 'layers'):
                    for layer_idx, layer in enumerate(actual_model.transformer.layers):
                        if hasattr(layer, 'seq_modeling_block'):
                            seq_block = layer.seq_modeling_block
                            if hasattr(seq_block, 'reset_ttt_inner_weights_for_new_file'):
                                if seq_block.reset_ttt_inner_weights_for_new_file():
                                    reset_success = True
                                    reset_layer_count += 1
                
                if reset_success:
                    ttt_reset_count += 1
                    main_logger_info(f"✅ [TTT-RESET] Reset complete: {reset_layer_count} TTT layers (total file switches: {ttt_reset_count})")
                else:
                    main_logger_info(f"⚠️  [TTT-RESET] No TTT layers found to reset")
            
            # Update previous_file_id for next iteration
            if file_id is not None:
                previous_file_id = file_id
            # ====================================================================

            condition_tensors = None
            if batch.condition_attributes is not None:
                condition_tensors = model.condition_provider.prepare(
                    batch.condition_attributes
                )

            # forward / backward
            # Note: file_id is extracted but not used - reserved for future continuous RoPE support
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

        # Get learning rates for all parameter groups
        last_lrs = scheduler.get_last_lr()
        last_lr = last_lrs[0]  # Keep backward compatibility with single LR logs
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

        # Paper metrics evaluation
        if hasattr(args, 'paper_metrics') and args.paper_metrics.get('paper_metrics_eval', False) and (
            (args.paper_metrics.get('paper_metrics_freq', 0) > 0 and state.step % args.paper_metrics.get('paper_metrics_freq', 0) == 0) or is_last_step
        ):
            try:
                # Initialize paper metrics evaluator if not already done
                if not hasattr(state, 'paper_metrics_evaluator'):
                    main_logger_info("Initializing paper metrics evaluator...")
                    # Lazy import to avoid torchaudio dependency at startup
                    from finetune.paper_metrics import PaperMetricsEvaluator
                    state.paper_metrics_evaluator = PaperMetricsEvaluator(
                        mimi,  # Fixed: positional argument instead of keyword
                        interleaved_tokenizer,
                        config=args.paper_metrics
                    )
                
                main_logger_info("Running paper metrics evaluation...")
                paper_metrics_results = state.paper_metrics_evaluator.evaluate_all(model)
                
                # Log paper metrics results
                for benchmark_name, results in paper_metrics_results.items():
                    main_logger_info(f"Paper metrics - {benchmark_name}: {results}")
                    
                    # Handle both dict and scalar results
                    if isinstance(results, dict):
                        eval_logger.log({f'paper_metrics/{benchmark_name}_{k}': v for k, v in results.items()}, step=state.step)
                    else:
                        # Handle scalar results (like averages)
                        eval_logger.log({f'paper_metrics/{benchmark_name}': results}, step=state.step)
                    
            except Exception as e:
                logger.error(f"Paper metrics evaluation failed: {e}")
                main_logger_info(f"Paper metrics evaluation failed: {e}")

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
                last_lrs,  # Pass all learning rates for multi-LR logging
            )
            main_logger_info(train_log_msg(state, logs=train_logs, loss=avg_loss))
            metrics_logger.log(train_logs, step=state.step)

        if args.do_ckpt and (
            (args.ckpt_freq > 0 and state.step % args.ckpt_freq == 0) or is_last_step
        ):
            checkpointer.save_checkpoint(
                save_only_lora=not args.full_finetuning and args.save_adapters,
                dtype=param_dtype,
            )

    main_logger_info("done!")


if __name__ == "__main__":
    """See README.md for usage."""
    fire.Fire(train)
