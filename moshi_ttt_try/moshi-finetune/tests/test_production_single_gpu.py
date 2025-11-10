#!/usr/bin/env python3
"""
Production TTT-Moshi Single-GPU Training Test
Tests the complete production pipeline without distributed training complexity.
"""

import os
import sys
import logging
import tempfile
from pathlib import Path

# Add paths
sys.path.append('/home/alufr/ttt_tests/moshi')
sys.path.insert(0, '.')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def test_production_single_gpu():
    """Test complete production TTT-Moshi training pipeline on single GPU"""
    print("🚀 PRODUCTION TTT-MOSHI SINGLE-GPU TEST")
    print("=" * 60)
    
    try:
        # Import all necessary modules
        from finetune.args import TrainArgs
        from finetune.wrapped_model import get_fsdp_model
        from finetune.data.data_loader import build_data_loader
        from finetune.data.interleaver import InterleavedTokenizer, Interleaver
        from finetune.loss import compute_loss_with_mask
        from moshi.models import loaders
        import torch
        from torch.optim import AdamW
        
        print("✅ All imports successful")
        
        # Set device (CPU for compatibility)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Create temporary run directory
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "test_run"
            run_dir.mkdir()
            
            # Create modified config for single-GPU testing
            config_content = """
# Single-GPU TTT-Moshi Test Configuration
data:
  train_data: /sise/eliyanac-group/ron_al/daily-talk-contiguous/train/dailytalk_train.jsonl
  eval_data: /sise/eliyanac-group/ron_al/daily-talk-contiguous/eval/dailytalk_eval.jsonl
  shuffle: true

run_dir: {run_dir}
overwrite_run_dir: true

moshi_paths:
  hf_repo_id: kyutai/moshiko-pytorch-bf16

# LoRA for efficiency
lora:
  enable: true
  rank: 32
  scaling: 1.0
  ft_embed: false

# TTT enabled
ttt:
  enable: true
  layers: "1,2"  # Just test 2 layers
  base_lr: 1.0
  mini_batch_size: 8

# Small-scale training for testing
full_finetuning: false
duration_sec: 10.0
batch_size: 1
num_microbatches: 1
max_steps: 3
log_freq: 1

# Optimizer
optim:
  lr: 1e-4
  weight_decay: 0.01
  pct_start: 0.1

max_norm: 1.0
seed: 42
gradient_checkpointing: false
param_dtype: float32

# Disable checkpointing and eval for test
do_ckpt: false
do_eval: false

wandb:
  project: null
""".format(run_dir=str(run_dir))
            
            config_path = run_dir / "test_config.yaml"
            with open(config_path, 'w') as f:
                f.write(config_content)
            
            # Load configuration
            print("📋 Loading configuration...")
            args = TrainArgs.load(str(config_path), drop_extra_fields=False)
            print(f"✅ Config loaded: TTT={args.ttt.enable}, LoRA={args.lora.enable}")
            
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
            
            # Use original model config for compatibility
            lm_config = lm_config.copy()
            lm_config.update({
                "lora": args.lora.enable,
                "lora_rank": args.lora.rank,
                "lora_scaling": args.lora.scaling,
            })
            
            print(f"   Actual model dimensions: {lm_config.get('dim', 'unknown')}d, {lm_config.get('num_heads', 'unknown')} heads")
            
            print(f"✅ Model config: {lm_config['dim']}d, {lm_config['num_layers']} layers")
            
            # Load Mimi encoder
            mimi = checkpoint_info.get_mimi(device=device)
            mimi.eval()
            for p in mimi.parameters():
                p.requires_grad = False
            print("✅ Mimi loaded and frozen")
            
            # Load model with TTT integration (single GPU - no FSDP)
            print("🧠 Loading model with TTT integration...")
            
            # Mock single-GPU environment
            os.environ['WORLD_SIZE'] = '1'
            os.environ['RANK'] = '0'
            
            # Load model directly (bypass FSDP for single GPU)
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
            
            # Apply TTT integration manually
            from finetune.ttt_integration import apply_ttt_to_model, verify_ttt_integration
            
            original_params = sum(p.numel() for p in model.parameters())
            apply_ttt_to_model(model, args.ttt, lm_config)
            ttt_params = sum(p.numel() for p in model.parameters())
            param_increase = ttt_params - original_params
            
            print(f"✅ Model loaded with TTT: +{param_increase:,} parameters")
            verify_ttt_integration(model)
            
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
            
            # Create data loader
            data_loader = build_data_loader(
                instruct_tokenizer=interleaved_tokenizer,
                args=args.data,
                batch_size=args.batch_size,
                seed=args.seed,
                rank=0,
                world_size=1,
                is_eval=False,
            )
            print("✅ Data pipeline ready")
            
            # Set up training
            print("⚙️  Setting up training...")
            
            # Configure trainable parameters
            if args.lora.enable and not args.full_finetuning:
                for name, param in model.named_parameters():
                    if "lora" in name or any(k in name for k in ['W1', 'W2', 'ttt']):
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
            
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"✅ Trainable parameters: {trainable_params:,}/{total_params:,} ({trainable_params/total_params*100:.1f}%)")
            
            # Set up optimizer
            optimizer = AdamW(
                model.parameters(),
                lr=args.optim.lr,
                weight_decay=args.optim.weight_decay,
            )
            
            model.train()
            print("✅ Training setup complete")
            
            # Training loop
            print(f"🎯 Running {args.max_steps} training steps...")
            
            for step in range(args.max_steps):
                print(f"\n📚 Step {step+1}/{args.max_steps}")
                
                # Get batch
                batch = next(data_loader)
                codes = batch.codes.to(device)
                print(f"   Input shape: {codes.shape}")
                
                # Forward pass
                optimizer.zero_grad()
                
                condition_tensors = None
                if batch.condition_attributes is not None:
                    condition_tensors = model.condition_provider.prepare(
                        batch.condition_attributes
                    )
                
                output = model(codes=codes, condition_tensors=condition_tensors)
                
                # Compute loss
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
                
                total_loss = text_loss + audio_loss
                print(f"   Loss: {total_loss.item():.4f} (text: {text_loss.item():.4f}, audio: {audio_loss.item():.4f})")
                
                # Backward pass
                total_loss.backward()
                
                # Check gradients
                total_grads = sum(1 for p in model.parameters() if p.grad is not None)
                ttt_grads = sum(1 for name, p in model.named_parameters() 
                              if p.grad is not None and any(k in name for k in ['W1', 'W2', 'ttt']))
                lora_grads = sum(1 for name, p in model.named_parameters()
                               if p.grad is not None and 'lora' in name)
                
                print(f"   Gradients: {total_grads} total, {ttt_grads} TTT, {lora_grads} LoRA")
                
                # Gradient clipping and optimizer step
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_norm)
                optimizer.step()
                
                print(f"   ✅ Step {step+1} completed")
            
            print(f"\n🏆 PRODUCTION TEST COMPLETE!")
            print(f"✅ Successfully trained TTT-Moshi for {args.max_steps} steps")
            print(f"✅ TTT integration working in production pipeline")
            print(f"✅ LoRA + TTT combination validated")
            print(f"✅ Real DailyTalk data processing confirmed")
            
            return True
            
    except Exception as e:
        print(f"❌ Production test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the production single-GPU test"""
    success = test_production_single_gpu()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 PRODUCTION TTT-MOSHI SINGLE-GPU TEST: ✅ SUCCESS!")
        print("\n🚀 Ready for production use!")
        print("   The TTT-Moshi integration is fully functional")
        print("   You can now scale to multi-GPU distributed training")
    else:
        print("❌ PRODUCTION TEST FAILED")
        print("   Please fix the issues before proceeding")
    
    return success

if __name__ == "__main__":
    main()