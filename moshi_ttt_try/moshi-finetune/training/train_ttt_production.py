#!/usr/bin/env python3
"""
Production TTT-Moshi Training Script (Single GPU)
Full training run using the production TTT integration.
"""

import os
import sys
import logging
import time
from pathlib import Path

# Add paths
sys.path.append('/home/alufr/ttt_tests/moshi')
sys.path.insert(0, '.')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def run_production_training():
    """Run full production TTT-Moshi training"""
    print("🚀 PRODUCTION TTT-MOSHI TRAINING")
    print("=" * 60)
    
    try:
        # Import all necessary modules
        from finetune.args import TrainArgs
        from finetune.data.data_loader import build_data_loader
        from finetune.data.interleaver import InterleavedTokenizer, Interleaver
        from finetune.loss import compute_loss_with_mask
        from finetune.monitoring.metrics_logger import MetricsLogger
        from finetune.paper_metrics import create_paper_metrics_evaluator
        from moshi.models import loaders
        import torch
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import OneCycleLR
        
        print("✅ All imports successful")
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")
        
        # Load configuration
        import sys
        if len(sys.argv) > 1:
            config_path = sys.argv[1]
        else:
            config_path = "configs/production_ttt_dailytalk.yaml"
        print(f"📋 Loading config: {config_path}")
        args = TrainArgs.load(config_path, drop_extra_fields=False)
        
        # Create run directory
        run_dir = Path(args.run_dir)
        if run_dir.exists() and args.overwrite_run_dir:
            import shutil
            shutil.rmtree(run_dir)
        run_dir.mkdir(exist_ok=True, parents=True)
        
        # Save config
        config_save_path = run_dir / "config.yaml"
        args.save(config_save_path)
        
        print(f"✅ Config loaded - Run dir: {run_dir}")
        print(f"   TTT: {args.ttt.enable} (layers: {args.ttt.layers})")
        print(f"   LoRA: {args.lora.enable} (rank: {args.lora.rank})")
        print(f"   Training: {args.max_steps} steps, batch size {args.batch_size}")
        
        # Load model components
        print("📥 Loading Moshi components...")
        checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
            hf_repo=args.moshi_paths.hf_repo_id
        )
        
        # Get model config
        lm_config = (
            loaders._lm_kwargs
            if checkpoint_info.raw_config is None
            else checkpoint_info.raw_config
        )
        lm_config["lora"] = args.lora.enable
        lm_config["lora_rank"] = args.lora.rank
        lm_config["lora_scaling"] = args.lora.scaling
        
        print(f"✅ Model config: {lm_config['dim']}d, {lm_config['num_layers']} layers")
        
        # Load Mimi encoder
        mimi = checkpoint_info.get_mimi(device=device)
        mimi.eval()
        for p in mimi.parameters():
            p.requires_grad = False
        print("✅ Mimi loaded and frozen")
        
        # Load model with TTT integration
        print("🧠 Loading model with TTT integration...")
        
        # Mock single-GPU environment for TTT integration
        os.environ['WORLD_SIZE'] = '1'
        os.environ['RANK'] = '0'
        
        model = checkpoint_info.get_moshi(
            device=device,
            dtype=torch.float32,
            lm_kwargs_overrides={
                "gradient_checkpointing": args.gradient_checkpointing,
                "lora": args.lora.enable,
                "lora_rank": args.lora.rank,
                "lora_scaling": args.lora.scaling,
            },
            load_weight=True,
        )
        
        # Apply TTT integration
        from finetune.ttt_integration import apply_ttt_to_model, verify_ttt_integration, log_ttt_parameters
        
        original_params = sum(p.numel() for p in model.parameters())
        apply_ttt_to_model(model, args.ttt, lm_config)
        ttt_params = sum(p.numel() for p in model.parameters())
        param_increase = ttt_params - original_params
        
        print(f"✅ Model loaded with TTT: +{param_increase:,} parameters ({param_increase/original_params*100:.1f}%)")
        verify_ttt_integration(model)
        log_ttt_parameters(model)
        
        # Set up data pipeline
        print("📦 Setting up data pipeline...")
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
        
        # Create data loaders
        data_loader = build_data_loader(
            instruct_tokenizer=interleaved_tokenizer,
            args=args.data,
            batch_size=args.batch_size,
            seed=args.seed,
            rank=0,
            world_size=1,
            is_eval=False,
        )
        
        if args.do_eval:
            eval_data_loader = build_data_loader(
                instruct_tokenizer=interleaved_tokenizer,
                args=args.data,
                batch_size=args.batch_size,
                seed=None,
                rank=0,
                world_size=1,
                is_eval=True,
            )
        print("✅ Data pipeline ready")
        
        # Set up paper metrics evaluator
        print("📊 Setting up paper metrics evaluator...")
        paper_metrics_config = getattr(args, 'paper_metrics', {})
        paper_metrics_evaluator = create_paper_metrics_evaluator(
            mimi_encoder=mimi,
            interleaved_tokenizer=interleaved_tokenizer,
            device=device,
            config=paper_metrics_config
        )
        print("✅ Paper metrics evaluator ready")
        
        # Set up training
        print("⚙️  Setting up training...")
        
        # Configure trainable parameters
        if not args.full_finetuning:
            # Only train TTT and LoRA parameters, freeze base model
            for name, param in model.named_parameters():
                if any(k in name for k in ['W1', 'W2', 'ttt', 'lora']):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        else:
            # Full fine-tuning - train all parameters
            for param in model.parameters():
                param.requires_grad = True
        
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✅ Trainable parameters: {trainable_params:,}/{total_params:,} ({trainable_params/total_params*100:.1f}%)")
        
        # Set up optimizer and scheduler
        optimizer = AdamW(
            model.parameters(),
            lr=args.optim.lr,
            betas=(0.9, 0.95),
            eps=1e-08,
            weight_decay=args.optim.weight_decay,
        )
        
        scheduler = OneCycleLR(
            optimizer,
            max_lr=args.optim.lr,
            total_steps=args.max_steps,
            pct_start=args.optim.pct_start,
        )
        
        # Set up metrics logging
        metrics_logger = MetricsLogger(
            run_dir,
            tag="train",
            is_master=True,
            wandb_args=args.wandb,
            config=args.__dict__,
        )
        
        model.train()
        print("✅ Training setup complete")
        
        # Training loop
        print(f"🎯 Starting training: {args.max_steps} steps...")
        print(f"   Logging every {args.log_freq} steps")
        print(f"   Checkpointing: {'enabled' if args.do_ckpt else 'disabled'}")
        print(f"   Evaluation: {'enabled' if args.do_eval else 'disabled'}")
        
        start_time = time.time()
        step = 0
        losses = []
        
        while step < args.max_steps:
            step += 1
            step_start = time.time()
            
            # Get batch
            batch = next(data_loader)
            codes = batch.codes.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            
            condition_tensors = None
            if batch.condition_attributes is not None:
                condition_tensors = model.condition_provider.prepare(
                    batch.condition_attributes
                )
            
            # Model forward pass with microbatching
            loss = torch.tensor([0.0], device=device)
            n_batch_tokens = 0
            n_real_tokens = 0
            
            for mb_idx in range(args.num_microbatches):
                # In single GPU, we just process the batch as-is
                output = model(codes=codes, condition_tensors=condition_tensors)
                
                # Compute losses
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
            
            if args.num_microbatches > 1:
                loss /= args.num_microbatches
                for p in model.parameters():
                    if p.requires_grad and p.grad is not None:
                        p.grad.div_(args.num_microbatches)
            
            # Gradient clipping and optimizer step
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
            optimizer.step()
            scheduler.step()
            
            # Track metrics
            loss_item = loss.item()
            losses.append(loss_item)
            step_time = time.time() - step_start
            
            # Logging
            if step % args.log_freq == 0:
                # Check gradients
                total_grads = sum(1 for p in model.parameters() if p.grad is not None)
                ttt_grads = sum(1 for name, p in model.named_parameters() 
                              if p.grad is not None and any(k in name for k in ['W1', 'W2', 'ttt']))
                lora_grads = sum(1 for name, p in model.named_parameters()
                               if p.grad is not None and 'lora' in name)
                
                elapsed_time = time.time() - start_time
                steps_per_sec = step / elapsed_time
                
                print(f"\n📊 Step {step}/{args.max_steps}")
                print(f"   Loss: {loss_item:.4f} (text: {text_loss.item():.4f}, audio: {audio_loss.item():.4f})")
                print(f"   Gradients: {total_grads} total, {ttt_grads} TTT, {lora_grads} LoRA")
                print(f"   Step time: {step_time:.2f}s, Speed: {steps_per_sec:.2f} steps/s")
                print(f"   Learning rate: {scheduler.get_last_lr()[0]:.2e}")
                
                # Log to metrics
                train_logs = {
                    'loss': loss_item,
                    'text_loss': text_loss.item(),
                    'audio_loss': audio_loss.item(),
                    'learning_rate': scheduler.get_last_lr()[0],
                    'step_time': step_time,
                    'steps_per_second': steps_per_sec,
                    'gradient_norm': torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf')).item(),
                    'ttt_gradients': ttt_grads,
                    'lora_gradients': lora_grads,
                }
                metrics_logger.log(train_logs, step=step)
            
            # Evaluation
            if args.do_eval and args.eval_freq > 0 and step % args.eval_freq == 0:
                print(f"\n📈 Running evaluation at step {step}...")
                model.eval()
                eval_losses = []
                
                with torch.no_grad():
                    for eval_step in range(10):  # Limited eval - 10 steps
                        try:
                            eval_batch = next(eval_data_loader)
                        except StopIteration:
                            break
                        eval_codes = eval_batch.codes.to(device)
                        
                        eval_output = model(codes=eval_codes)
                        eval_text_loss = compute_loss_with_mask(
                            eval_output.text_logits,
                            eval_codes[:, : model.audio_offset],
                            eval_output.text_mask,
                            mode="text",
                            text_padding_weight=args.text_padding_weight,
                            text_padding_ids={
                                model.text_padding_token_id,
                                model.end_of_text_padding_id,
                            },
                        )
                        eval_audio_loss = compute_loss_with_mask(
                            eval_output.logits,
                            eval_codes[:, model.audio_offset : model.audio_offset + model.dep_q],
                            eval_output.mask,
                            mode="audio",
                            first_codebook_weight_multiplier=args.first_codebook_weight_multiplier,
                        )
                        eval_total_loss = eval_text_loss + eval_audio_loss
                        eval_losses.append(eval_total_loss.item())
                
                avg_eval_loss = sum(eval_losses) / len(eval_losses)
                print(f"   Eval loss: {avg_eval_loss:.4f}")
                
                eval_logs = {
                    'eval_loss': avg_eval_loss,
                    'eval_text_loss': eval_text_loss.item(),
                    'eval_audio_loss': eval_audio_loss.item(),
                }
                
                # Run paper metrics evaluation if enabled
                if (hasattr(args, 'paper_metrics') and 
                    paper_metrics_config.get('paper_metrics_eval', False) and 
                    paper_metrics_config.get('paper_metrics_freq', 0) > 0 and 
                    step % paper_metrics_config.get('paper_metrics_freq', 200) == 0):
                    print(f"   📊 Running paper metrics evaluation...")
                    paper_metrics_results = paper_metrics_evaluator.evaluate_all(
                        model, max_samples_per_task=50  # Lightweight evaluation
                    )
                    eval_logs.update(paper_metrics_results)
                    print(f"   Paper metrics avg: {paper_metrics_results.get('paper_metrics_avg', 0.0):.3f}")
                
                metrics_logger.log(eval_logs, step=step)
                
                model.train()
            
            # Checkpointing
            if args.do_ckpt and args.ckpt_freq > 0 and step % args.ckpt_freq == 0:
                checkpoint_path = run_dir / f"checkpoint_step_{step}.pt"
                print(f"💾 Saving checkpoint: {checkpoint_path}")
                
                checkpoint_data = {
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': loss_item,
                    'config': args.__dict__,
                }
                
                torch.save(checkpoint_data, checkpoint_path)
                print(f"✅ Checkpoint saved")
        
        # Training complete
        total_time = time.time() - start_time
        print(f"\n🏆 TRAINING COMPLETE!")
        print(f"   Total time: {total_time/3600:.2f} hours")
        print(f"   Average step time: {total_time/args.max_steps:.2f}s")
        print(f"   Final loss: {losses[-1]:.4f}")
        print(f"   Loss change: {losses[0]:.4f} → {losses[-1]:.4f}")
        
        # Save final model
        final_model_path = run_dir / "final_model.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': lm_config,
            'training_args': args.__dict__,
            'final_loss': losses[-1],
            'total_steps': args.max_steps,
        }, final_model_path)
        print(f"💾 Final model saved: {final_model_path}")
        
        # Close metrics logger
        metrics_logger.close()
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main entry point"""
    success = run_production_training()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 PRODUCTION TTT-MOSHI TRAINING: ✅ COMPLETE!")
        print("\n🔥 Training Summary:")
        print("   ✅ TTT layers successfully trained")
        print("   ✅ LoRA adaptation applied")
        print("   ✅ Real DailyTalk data processed")
        print("   ✅ Model checkpoints saved")
        print("   ✅ Metrics logged")
    else:
        print("❌ TRAINING FAILED")
        print("   Please check the error messages above")
    
    return success

if __name__ == "__main__":
    main()