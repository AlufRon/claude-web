#!/usr/bin/env python3
"""
Evaluate a trained TTT checkpoint with Figure 5 plotting.

This script:
1. Loads a trained checkpoint with TTT layers
2. Runs evaluation on LibriLight sequences
3. Generates Figure 5 plots showing:    logger.info(f"\n🏗️  Initializing model...")
    
    # Setup distributed environment for FSDP (required even for single GPU)
    import socket
    import os
    
    if "LOCAL_RANK" not in os.environ:
        def find_free_port():
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', 0))
                s.listen(1)
                port = s.getsockname()[1]
            return port
        
        free_port = find_free_port()
        os.environ["LOCAL_RANK"] = "0"
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(free_port)
        logger.info(f"   Using free port for distributed: {free_port}")
    
    # Initialize distributed process group
    from finetune.distributed import set_device, BACKEND
    import torch.distributed as dist
    
    set_device()
    if not dist.is_initialized():
        dist.init_process_group(backend=BACKEND)
        logger.info(f"   Initialized distributed process group")
    
    # Load base model using same method as training
    from moshi.models import loaders- ℓ(W₀; xₜ): Frozen initial weights (blue line)
   - ℓ(Wₜ₋₁; xₜ): Before gradient descent (orange line)
   - ℓ(Wₜ; xₜ): After gradient descent (green line)

Usage:
    python evaluate_checkpoint_figure5.py \
        --checkpoint /path/to/checkpoint.pt \
        --config /path/to/config.yaml \
        --output-dir ./evaluation_plots/figure5 \
        --max-sequences 3
"""

import argparse
import torch
from pathlib import Path
import yaml
import logging
import sys
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from finetune.args import TrainArgs
from finetune.wrapped_model import get_fsdp_model
from finetune.paper_metrics import PaperMetricsEvaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_checkpoint_and_config(checkpoint_path: Path, config_path: Path = None):
    """
    Load checkpoint and config, enabling Figure 5 tracking.
    
    Returns:
        (checkpoint_dict, config_dict)
    """
    logger.info(f"📦 Loading checkpoint from: {checkpoint_path}")
    
    # Load checkpoint - handle both .pt and .safetensors files
    if str(checkpoint_path).endswith('.safetensors'):
        # Load safetensors file
        try:
            from safetensors.torch import load_file
            checkpoint = load_file(checkpoint_path)
            logger.info(f"✅ Checkpoint loaded (safetensors)")
        except ImportError:
            raise ImportError("safetensors package required. Install with: pip install safetensors")
    else:
        # Load .pt file with weights_only=False for backward compatibility
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        logger.info(f"✅ Checkpoint loaded")
    
    logger.info(f"   Keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'state_dict'}")
    
    # Load config
    if config_path is None:
        # Try to find config next to checkpoint
        possible_configs = [
            checkpoint_path.parent / "config.yaml",
            checkpoint_path.parent.parent / "config.yaml",
            checkpoint_path.parent / "moshi_7B_multilayer_with_ttt.yaml",
        ]
        
        for possible_config in possible_configs:
            if possible_config.exists():
                config_path = possible_config
                break
        
        if config_path is None:
            raise FileNotFoundError(
                "Could not find config.yaml. Please specify with --config"
            )
    
    logger.info(f"📄 Loading config from: {config_path}")
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    
    # Don't add figure5_* fields - they don't exist in TTTArgs
    # Plotting will be done manually after evaluation
    
    logger.info(f"✅ Config loaded")
    
    return checkpoint, config_dict


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate TTT checkpoint with Figure 5 plotting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate latest checkpoint
  python evaluate_checkpoint_figure5.py \\
      --checkpoint /path/to/checkpoint_100.pt \\
      --max-sequences 1

  # Evaluate with custom config
  python evaluate_checkpoint_figure5.py \\
      --checkpoint /path/to/checkpoint_100.pt \\
      --config my_config.yaml \\
      --output-dir ./figure5_plots \\
      --max-sequences 3
        """
    )
    
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        required=True,
        help="Path to trained checkpoint (.pt file)"
    )
    parser.add_argument(
        "--config", 
        type=str,
        help="Path to config YAML (auto-detected if not specified)"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="./evaluation_plots/figure5",
        help="Directory to save Figure 5 plots"
    )
    parser.add_argument(
        "--max-sequences", 
        type=int, 
        default=1,
        help="Number of LibriLight sequences to evaluate (1-8)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum tokens per sequence (default: full sequence, ~3000 tokens)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run evaluation on"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    config_path = Path(args.config) if args.config else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"
    
    logger.info(f"\n{'='*80}")
    logger.info("FIGURE 5 EVALUATION CONFIGURATION")
    logger.info(f"{'='*80}")
    logger.info(f"Checkpoint:      {checkpoint_path}")
    logger.info(f"Output dir:      {output_dir}")
    logger.info(f"Max sequences:   {args.max_sequences}")
    logger.info(f"Max tokens:      {args.max_tokens or 'Full sequence (~3000)'}")
    logger.info(f"Device:          {device}")
    logger.info(f"{'='*80}\n")
    
    # Load checkpoint and config
    checkpoint, config_dict = load_checkpoint_and_config(checkpoint_path, config_path)
    
    # Ensure run_dir exists (required by TrainArgs)
    if 'run_dir' not in config_dict:
        config_dict['run_dir'] = str(output_dir)
    
    # Note: Don't add figure5_* fields - they don't exist in TTTArgs
    # The plotting will be done after evaluation using the collected losses
    
    if 'paper_metrics' not in config_dict:
        config_dict['paper_metrics'] = {}
    
    config_dict['paper_metrics']['paper_metrics_eval'] = True
    config_dict['paper_metrics']['librilight_max_files'] = args.max_sequences
    
    if args.max_tokens:
        config_dict['paper_metrics']['librilight_max_tokens'] = args.max_tokens
    
    # Save modified config for reference
    eval_config_path = output_dir / "eval_config.yaml"
    with open(eval_config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)
    logger.info(f"💾 Saved evaluation config to: {eval_config_path}")
    
    # Parse config into TrainArgs
    logger.info(f"\n📋 Parsing configuration...")
    
    # Create temporary config file
    temp_config = Path("/tmp/figure5_eval_config.yaml")
    with open(temp_config, 'w') as f:
        yaml.dump(config_dict, f)
    
    # Use TrainArgs.load() method directly (same as training script)
    # Use drop_extra_fields=True to ignore any custom fields we added
    train_args = TrainArgs.load(temp_config, drop_extra_fields=True)
    
    logger.info(f"✅ Configuration parsed")
    logger.info(f"   TTT layers: {train_args.ttt.layers if train_args.ttt.enable else 'None'}")
    logger.info(f"   TTT enabled: {train_args.ttt.enable}")
    
    # Initialize model
    logger.info(f"\n🏗️  Initializing model...")
    
    # Load base model using same method as training script
    from moshi.models import loaders
    
    logger.info(f"   Loading Moshi checkpoint info from HuggingFace...")
    checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
        hf_repo=train_args.moshi_paths.hf_repo_id,
        moshi_weights=train_args.moshi_paths.moshi_path,
        mimi_weights=train_args.moshi_paths.mimi_path,
        tokenizer=train_args.moshi_paths.tokenizer_path,
        config_path=train_args.moshi_paths.config_path,
    )
    
    # Update LM config for LoRA/TTT
    lm_config = (
        loaders._lm_kwargs
        if checkpoint_info.raw_config is None
        else checkpoint_info.raw_config
    )
    lm_config["lora"] = train_args.lora.enable
    lm_config["lora_rank"] = train_args.lora.rank
    lm_config["lora_scaling"] = train_args.lora.scaling
    
    # Build model with FSDP (same as training)
    logger.info(f"   Building model with TTT layers...")
    model = get_fsdp_model(train_args, checkpoint_info)
    
    # Load checkpoint weights
    logger.info(f"   Loading checkpoint weights...")
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # Load state dict (handle FSDP module wrapper)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        logger.warning(f"Failed strict load: {e}")
        logger.info("Trying strict=False...")
        model.load_state_dict(state_dict, strict=False)
    
    model.eval()
    
    logger.info(f"✅ Model loaded successfully")
    logger.info(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"   Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Initialize MIMI and tokenizer
    logger.info(f"\n🎤 Loading MIMI encoder...")
    mimi_encoder = loaders.get_mimi(
        train_args.moshi_paths.hf_repo_id,
        device=device
    )
    
    logger.info(f"📝 Loading tokenizer...")
    interleaved_tokenizer = loaders.get_interleaved_tokenizer()
    
    # Initialize evaluator
    logger.info(f"\n📊 Initializing Paper Metrics Evaluator...")
    evaluator = PaperMetricsEvaluator(
        mimi_encoder=mimi_encoder,
        interleaved_tokenizer=interleaved_tokenizer,
        device=device,
        config=train_args.paper_metrics
    )
    
    logger.info(f"\n{'='*80}")
    logger.info("STARTING FIGURE 5 EVALUATION")
    logger.info(f"{'='*80}\n")
    
    # Run evaluation
    logger.info(f"🚀 Running LibriLight evaluation with Figure 5 tracking...")
    logger.info(f"   This will generate 3 loss curves per TTT layer:")
    logger.info(f"   🔵 Blue:   ℓ(W₀; xₜ)     - Frozen initial weights")
    logger.info(f"   🟠 Orange: ℓ(Wₜ₋₁; xₜ)   - Before gradient descent")
    logger.info(f"   🟢 Green:  ℓ(Wₜ; xₜ)     - After gradient descent")
    logger.info(f"")
    logger.info(f"⏱️  This may take 10-30 minutes depending on sequence length...")
    logger.info(f"")
    
    try:
        # Run LibriLight evaluation
        with torch.no_grad():
            results = evaluator.evaluate_librilight_only(model)
        
        logger.info(f"\n{'='*80}")
        logger.info("✅ EVALUATION COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"\n📊 Results:")
        for key, value in results.items():
            if isinstance(value, float):
                logger.info(f"   {key}: {value:.4f}")
            else:
                logger.info(f"   {key}: {value}")
        
        logger.info(f"\n📁 Outputs saved to: {output_dir}")
        logger.info(f"   - Figure 5 plots (PNG + PDF)")
        logger.info(f"   - Statistics JSON")
        logger.info(f"   - Evaluation config")
        
        # List generated files
        plot_files = list(output_dir.glob("figure5_*.png"))
        if plot_files:
            logger.info(f"\n🎨 Generated plots:")
            for plot_file in sorted(plot_files):
                logger.info(f"   ✅ {plot_file.name}")
        
        logger.info(f"\n{'='*80}\n")
        
    except Exception as e:
        logger.error(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
