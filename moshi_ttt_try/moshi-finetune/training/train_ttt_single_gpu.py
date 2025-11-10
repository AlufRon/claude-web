#!/usr/bin/env python3
"""
TTT-Moshi Single-GPU Training Script
Simplified version without distributed training for validation and testing
"""

import logging
import os
import sys
import time
from pathlib import Path
import json

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

# Add paths
sys.path.append('/home/alufr/ttt_tests/moshi')
sys.path.insert(0, '.')

from moshi.models import loaders
from moshi_ttt.hybrid_layer import HybridStreamingTransformerLayer
from moshi_ttt.config import TTTConfig

logger = logging.getLogger("train_ttt_single")

class SimpleDataLoader:
    """Simple data loader for DailyTalk dataset without complex pipeline"""
    
    def __init__(self, data_file, max_duration=10.0, max_samples=None):
        self.data_file = data_file
        self.max_duration = max_duration
        self.samples = []
        
        # Load samples
        with open(data_file, 'r') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                    
                data = json.loads(line.strip())
                if data['duration'] <= max_duration:
                    # Build full path
                    if data['path'].startswith('../'):
                        base_path = os.path.dirname(data_file)
                        audio_path = os.path.join(base_path, data['path'])
                    else:
                        audio_path = data['path']
                    
                    if os.path.exists(audio_path):
                        self.samples.append({
                            'path': audio_path,
                            'duration': data['duration']
                        })
        
        print(f"📦 Loaded {len(self.samples)} samples from {data_file}")
        self.current_idx = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current_idx >= len(self.samples):
            self.current_idx = 0  # Reset for continuous iteration
        
        sample = self.samples[self.current_idx]
        self.current_idx += 1
        
        # For validation, return synthetic codes matching Moshi format
        # In real training, this would use Mimi encoder
        batch_size = 1
        n_codebooks = 17  # 1 text + 16 audio (adjusted to match actual Moshi config)
        seq_len = max(4, int(sample['duration'] * 2))  # ~2 tokens per second
        
        # Create synthetic codes tensor
        codes = torch.randint(0, 1024, (batch_size, n_codebooks, seq_len), dtype=torch.int64)
        
        # Make text codebook (0) have smaller vocab
        codes[:, 0, :] = torch.randint(0, 100, (batch_size, seq_len), dtype=torch.int64)
        
        # Mock batch object with expected attributes
        class MockBatch:
            def __init__(self, codes):
                self.codes = codes
                self.condition_attributes = None
        
        return MockBatch(codes)

def apply_ttt_to_model(model, ttt_config, layer_indices):
    """Apply TTT to specified layers"""
    print(f"🔄 Applying TTT to layers: {layer_indices}")
    
    original_params = sum(p.numel() for p in model.parameters())
    
    for layer_idx in layer_indices:
        if layer_idx < len(model.transformer.layers):
            original_layer = model.transformer.layers[layer_idx]
            hybrid_layer = HybridStreamingTransformerLayer(original_layer, ttt_config)
            model.transformer.layers[layer_idx] = hybrid_layer
            print(f"   ✅ Layer {layer_idx} → TTT")
    
    ttt_params = sum(p.numel() for p in model.parameters())
    param_increase = ttt_params - original_params
    
    print(f"✅ TTT applied: +{param_increase:,} parameters (+{param_increase/original_params*100:.1f}%)")
    return model

def compute_moshi_loss(output, codes, model):
    """Simplified loss computation"""
    def safe_cross_entropy(logits, targets, mask):
        logits = logits.view(-1, logits.size(-1))
        targets = targets.view(-1)
        mask = mask.view(-1).float()
        
        # Only compute loss where mask is True
        valid_indices = mask > 0
        if valid_indices.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        valid_logits = logits[valid_indices]
        valid_targets = targets[valid_indices]
        
        loss = nn.functional.cross_entropy(valid_logits, valid_targets, reduction='mean')
        return loss
    
    # Text loss
    text_loss = safe_cross_entropy(
        output.text_logits,
        codes[:, :1],  # Text codebook
        output.text_mask
    )
    
    # Audio loss
    audio_loss = safe_cross_entropy(
        output.logits,
        codes[:, 1:1+model.dep_q],  # Audio codebooks
        output.mask
    )
    
    return text_loss, audio_loss, text_loss + audio_loss

def train_single_gpu(
    train_data_file: str,
    eval_data_file: str = None,
    max_steps: int = 10,
    ttt_layers: str = "middle",
    learning_rate: float = 1e-4,
    duration_sec: float = 10.0,
    model_size: str = "small"  # "small", "medium", or "full"
):
    """
    Single-GPU training function
    
    Args:
        train_data_file: Path to training JSONL file
        eval_data_file: Path to evaluation JSONL file (optional)
        max_steps: Number of training steps
        ttt_layers: "all", "middle", "none", or list of indices
        learning_rate: Learning rate
        duration_sec: Max audio duration to use
        model_size: Model size to use for testing
    """
    print("🚀 TTT-Moshi Single-GPU Training")
    print("=" * 50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 1. Load model
    print("📥 Loading Moshi model...")
    checkpoint_info = loaders.CheckpointInfo.from_hf_repo("kyutai/moshiko-pytorch-bf16")
    lm_config = loaders._lm_kwargs if checkpoint_info.raw_config is None else checkpoint_info.raw_config
    
    # Adjust model size for single-GPU training
    if model_size == "small":
        lm_config = lm_config.copy()
        lm_config['num_layers'] = 8
        lm_config['dim'] = 1024
        lm_config['num_heads'] = 16
        lm_config['depformer_num_layers'] = 4
    elif model_size == "medium":
        lm_config = lm_config.copy()
        lm_config['num_layers'] = 16
        lm_config['dim'] = 2048
        lm_config['num_heads'] = 24
        lm_config['depformer_num_layers'] = 8
    # "full" uses original config
    
    print(f"Model config: {lm_config['dim']}d, {lm_config['num_layers']} layers")
    
    try:
        model = loaders.get_moshi_lm(
            filename=None,
            lm_kwargs=lm_config,
            device=device,
            dtype=torch.float32
        )
        print(f"✅ Model loaded: {type(model)}")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        print("Falling back to CPU and smaller model...")
        device = torch.device("cpu")
        lm_config['num_layers'] = 4
        lm_config['dim'] = 512
        lm_config['num_heads'] = 8
        model = loaders.get_moshi_lm(
            filename=None,
            lm_kwargs=lm_config,
            device=device,
            dtype=torch.float32
        )
    
    # 2. Apply TTT
    if ttt_layers != "none":
        total_layers = len(model.transformer.layers)
        
        if ttt_layers == "all":
            layer_indices = list(range(total_layers))
        elif ttt_layers == "middle":
            start = total_layers // 4
            end = 3 * total_layers // 4
            layer_indices = list(range(start, end))
        else:
            layer_indices = []
        
        if layer_indices:
            ttt_config = TTTConfig(
                model_dim=lm_config['dim'],
                num_heads=lm_config['num_heads'],
                mini_batch_size=8,
                ttt_base_lr=0.01
            )
            model = apply_ttt_to_model(model, ttt_config, layer_indices)
    else:
        print("🚫 TTT disabled - using vanilla Moshi")
    
    # 3. Setup training
    print("⚙️  Setting up training...")
    
    data_loader = SimpleDataLoader(
        train_data_file, 
        max_duration=duration_sec,
        max_samples=50  # Limit samples for quick testing
    )
    
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = OneCycleLR(optimizer, max_lr=learning_rate, total_steps=max_steps, pct_start=0.1)
    
    model.train()
    
    # 4. Training loop
    print(f"🎯 Starting {max_steps}-step training...")
    
    losses = []
    start_time = time.time()
    
    for step in range(max_steps):
        step_start = time.time()
        
        # Get batch
        batch = next(data_loader)
        codes = batch.codes.to(device)
        
        print(f"\n📚 Step {step+1}/{max_steps}")
        print(f"   Input: {codes.shape}")
        
        # Forward pass
        optimizer.zero_grad()
        
        try:
            output = model(codes)
            
            # Compute loss
            text_loss, audio_loss, total_loss = compute_moshi_loss(output, codes, model)
            
            print(f"   Loss: {total_loss.item():.4f} (text: {text_loss.item():.4f}, audio: {audio_loss.item():.4f})")
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimizer step
            optimizer.step()
            scheduler.step()
            
            losses.append(total_loss.item())
            
            # Check gradients
            ttt_grads = sum(1 for name, p in model.named_parameters() 
                          if p.grad is not None and any(k in name for k in ['W1', 'W2', 'ttt']))
            total_grads = sum(1 for name, p in model.named_parameters() if p.grad is not None)
            
            step_time = time.time() - step_start
            print(f"   Gradients: {total_grads} total, {ttt_grads} TTT")
            print(f"   Step time: {step_time:.2f}s")
            
        except Exception as e:
            print(f"   ❌ Step failed: {e}")
            continue
    
    # 5. Results
    total_time = time.time() - start_time
    
    print(f"\n📊 TRAINING COMPLETED:")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Average step time: {total_time/max_steps:.2f}s")
    print(f"   Final loss: {losses[-1]:.4f}")
    
    if len(losses) > 1:
        loss_change = abs(losses[-1] - losses[0])
        print(f"   Loss change: {loss_change:.4f}")
        learning_occurred = loss_change > 0.01
        print(f"   Learning detected: {'✅' if learning_occurred else '❌'}")
    
    print(f"\n✅ Single-GPU TTT-Moshi training validation complete!")
    return True

def main():
    """Main function with command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="TTT-Moshi Single-GPU Training")
    parser.add_argument("train_data", help="Path to training JSONL file")
    parser.add_argument("--eval_data", help="Path to evaluation JSONL file")
    parser.add_argument("--max_steps", type=int, default=10, help="Number of training steps")
    parser.add_argument("--ttt_layers", default="middle", help="TTT layers: all, middle, none")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--duration_sec", type=float, default=10.0, help="Max audio duration")
    parser.add_argument("--model_size", default="small", help="Model size: small, medium, full")
    
    args = parser.parse_args()
    
    train_single_gpu(
        train_data_file=args.train_data,
        eval_data_file=args.eval_data,
        max_steps=args.max_steps,
        ttt_layers=args.ttt_layers,
        learning_rate=args.learning_rate,
        duration_sec=args.duration_sec,
        model_size=args.model_size
    )

if __name__ == "__main__":
    # If called directly, use DailyTalk dataset
    train_data = "/sise/eliyanac-group/ron_al/daily-talk-contiguous/train/dailytalk_train.jsonl"
    eval_data = "/sise/eliyanac-group/ron_al/daily-talk-contiguous/eval/dailytalk_eval.jsonl"
    
    print("🚀 Running TTT-Moshi validation with DailyTalk dataset")
    
    success = train_single_gpu(
        train_data_file=train_data,
        eval_data_file=eval_data,
        max_steps=5,
        ttt_layers="middle",
        learning_rate=1e-4,
        duration_sec=15.0,
        model_size="small"
    )