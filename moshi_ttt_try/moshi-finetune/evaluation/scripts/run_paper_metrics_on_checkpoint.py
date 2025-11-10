#!/usr/bin/env python3
"""
Minimal script to run paper metrics on a TTT checkpoint.

Usage:
    python run_paper_metrics_on_checkpoint.py \
        --checkpoint /path/to/checkpoint_dir/consolidated \
        --max-samples 50
"""

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch
import safetensors.torch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def register_run(checkpoint_dir: Path, results_path: Path):
    """
    Register a completed evaluation run in the dashboard registry.

    This allows the dashboard to track runs without scanning the filesystem.

    Args:
        checkpoint_dir: Path to checkpoint directory (consolidated)
        results_path: Path to saved paper_metrics_results.json
    """
    registry_path = Path(__file__).parent.parent.parent / "dashboard" / "runs_registry.json"

    try:
        # Load existing registry
        if registry_path.exists():
            with open(registry_path, 'r') as f:
                registry_data = json.load(f)
        else:
            registry_data = {"runs": []}

        # Add new run entry
        new_entry = {
            "checkpoint_path": str(checkpoint_dir),
            "results_file": str(results_path),
            "added_at": datetime.now().isoformat()
        }

        # Check if already registered (avoid duplicates)
        existing_paths = {run.get("checkpoint_path") for run in registry_data["runs"]}
        if str(checkpoint_dir) not in existing_paths:
            registry_data["runs"].append(new_entry)

            # Save updated registry
            with open(registry_path, 'w') as f:
                json.dump(registry_data, f, indent=2)

            logger.info(f"✅ Registered run in dashboard registry")
        else:
            logger.info(f"ℹ️  Run already registered in dashboard")

    except Exception as e:
        logger.warning(f"⚠️  Failed to register run in dashboard: {e}")
        logger.warning("   Dashboard will still work but may need manual update")


def update_dashboard(results_path: Path):
    """
    Automatically update the dashboard after saving paper metrics results.

    Runs the dashboard aggregation script and regenerates standalone HTML
    to keep the dashboard up-to-date with the latest evaluation results.
    """
    # Dashboard is at project root, not in evaluation/scripts/
    dashboard_dir = Path(__file__).parent.parent.parent / "dashboard"
    aggregation_script = dashboard_dir / "aggregate_paper_metrics.py"
    dashboard_data = dashboard_dir / "dashboard_data.json"
    create_standalone_script = dashboard_dir / "create_standalone.sh"

    # Correct log directory path
    log_dir = Path(__file__).parent.parent.parent / "logs" / "evaluation"

    # Check if dashboard exists
    if not aggregation_script.exists():
        logger.warning(f"⚠️  Dashboard not found at {dashboard_dir}")
        logger.warning("   Skipping dashboard update")
        return

    logger.info("\n📊 Updating dashboard...")

    try:
        # Step 1: Aggregate paper metrics data
        logger.info("   Step 1/2: Aggregating paper metrics data...")
        result = subprocess.run(
            [
                sys.executable,
                str(aggregation_script),
                "--checkpoint-dirs", "/sise/eliyanac-group/ron_al",
                "--log-dir", str(log_dir),
                "--output", str(dashboard_data)
            ],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        if result.returncode != 0:
            logger.warning(f"⚠️  Dashboard aggregation failed (exit code {result.returncode})")
            if result.stderr:
                logger.warning(f"   Error: {result.stderr[:200]}")
            logger.warning("   Run manually: cd dashboard && ./update_dashboard.sh")
            return

        logger.info("   ✅ Dashboard data aggregated!")

        # Step 2: Regenerate standalone HTML
        if create_standalone_script.exists():
            logger.info("   Step 2/2: Regenerating standalone HTML...")
            result2 = subprocess.run(
                [str(create_standalone_script)],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(dashboard_dir),
                shell=True
            )

            if result2.returncode == 0:
                logger.info("   ✅ Dashboard HTML regenerated!")
                logger.info(f"\n📊 Dashboard updated successfully!")
                logger.info(f"   View at: {dashboard_dir / 'dashboard_standalone.html'}")
            else:
                logger.warning("⚠️  HTML regeneration failed")
                if result2.stderr:
                    logger.warning(f"   Error: {result2.stderr[:200]}")
                logger.info(f"   Data updated at: {dashboard_data}")
        else:
            logger.warning(f"⚠️  Standalone script not found at {create_standalone_script}")
            logger.info(f"   Data updated at: {dashboard_data}")

    except subprocess.TimeoutExpired:
        logger.warning("⚠️  Dashboard update timed out (large directory scan)")
        logger.warning("   Run manually: cd dashboard && ./update_dashboard.sh")

    except Exception as e:
        logger.warning(f"⚠️  Dashboard update error: {e}")
        logger.warning("   You can update manually: cd dashboard && ./update_dashboard.sh")


def load_checkpoint_config(checkpoint_dir: Path):
    """Load training config from checkpoint directory."""
    checkpoint_dir = Path(checkpoint_dir)
    
    training_config_path = checkpoint_dir / "training_config.json"
    if not training_config_path.exists():
        raise FileNotFoundError(f"training_config.json not found in {checkpoint_dir}")
    
    with open(training_config_path, 'r') as f:
        training_config = json.load(f)
    
    logger.info(f"✅ Loaded training config from {checkpoint_dir}")
    return training_config


def load_baseline_model(hf_repo: str = "kyutai/moshiko-pytorch-bf16", device: str = "cuda"):
    """
    Load baseline Moshi model without TTT for comparison.
    """
    from moshi.models import loaders
    
    logger.info("🔨 Loading baseline Moshi model (no TTT)...")
    checkpoint_info = loaders.CheckpointInfo.from_hf_repo(hf_repo)
    
    model = checkpoint_info.get_moshi(
        device=device,
        dtype=torch.bfloat16,
        load_weight=True
    )
    
    logger.info(f"✅ Loaded baseline Moshi model")

    # Apply RoPE and KV cache fixes for long-context evaluation
    logger.info("🔧 Applying RoPE fix for long-context evaluation...")
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'rope') and model.transformer.rope is not None:
        model.transformer.rope.max_period = 100_000
        model.transformer.max_period = 100_000
        logger.info(f"✅ Set main transformer RoPE max_period: 10,000 → 100,000")
    if hasattr(model, 'depformer') and hasattr(model.depformer, 'transformer'):
        if hasattr(model.depformer.transformer, 'rope') and model.depformer.transformer.rope is not None:
            model.depformer.transformer.rope.max_period = 100_000
            model.depformer.transformer.max_period = 100_000
            logger.info(f"✅ Set depformer RoPE max_period: 10,000 → 100,000")

    logger.info("🔧 Increasing KV cache context size for long sequences...")
    new_context = 30000  # Increase from default 3000 to 30000 for LibriLight
    modified_layers = 0

    if hasattr(model, 'transformer') and hasattr(model.transformer, 'layers'):
        old_context = None
        for layer in model.transformer.layers:
            if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'context'):
                if old_context is None:
                    old_context = layer.self_attn.context
                layer.self_attn.context = new_context
                modified_layers += 1
        if old_context is not None:
            logger.info(f"✅ Set main transformer context: {old_context} → {new_context} ({modified_layers} layers)")

    return model, checkpoint_info


def load_ttt_model(checkpoint_dir: Path, hf_repo: str = "kyutai/moshiko-pytorch-bf16", device: str = "cuda"):
    """
    Load a finetuned checkpoint for evaluation.

    Supports TTT-only, LoRA-only, full finetuning, or hybrid (TTT+LoRA) checkpoints.

    This follows the same pattern as run_inference_with_ttt.py but simplified.
    """
    from moshi.models import loaders
    from moshi.modules.lora import replace_all_linear_with_lora
    from finetune.ttt_integration import apply_ttt_to_model, verify_ttt_integration
    from finetune.args import TTTArgs
    
    checkpoint_dir = Path(checkpoint_dir)
    
    # Step 1: Load config
    logger.info("📂 Loading checkpoint configuration...")
    training_config = load_checkpoint_config(checkpoint_dir)
    
    # Step 2: Determine checkpoint type
    ttt_config = training_config.get('ttt', {})
    lora_config = training_config.get('lora', {})
    full_finetuning = training_config.get('full_finetuning', False)

    ttt_enabled = ttt_config.get('enable', False)
    lora_enabled = lora_config.get('enable', False)

    logger.info(f"📊 Checkpoint type:")
    logger.info(f"   TTT enabled: {ttt_enabled}")
    logger.info(f"   LoRA enabled: {lora_enabled}")
    logger.info(f"   Full finetuning: {full_finetuning}")

    # Create TTTArgs only if TTT is enabled
    ttt_args = None
    if ttt_enabled:
        logger.info("🔧 Creating TTT configuration...")
        ttt_args = TTTArgs(
            enable=ttt_config.get('enable', False),
            layers=ttt_config.get('layers', 'none'),
            base_lr=ttt_config.get('base_lr', 0.001),
            mini_batch_size=ttt_config.get('mini_batch_size', 32),
            persistent_states=ttt_config.get('persistent_states', True),
            initial_gating_alpha=ttt_config.get('initial_gating_alpha', 0.1),
            ttt_mlp_layers=ttt_config.get('ttt_mlp_layers', 3),
            ttt_mlp_expansion_factor=ttt_config.get('ttt_mlp_expansion_factor', 4.0),
            ttt_mlp_hidden_dims=ttt_config.get('ttt_mlp_hidden_dims', None),
        )

        logger.info(f"✅ TTT Config:")
        logger.info(f"   Layers: {ttt_args.layers}")
        logger.info(f"   Base LR: {ttt_args.base_lr}")
        logger.info(f"   Gating alpha: {ttt_args.initial_gating_alpha}")
    
    # Step 3: Load base Moshi model
    logger.info("🔨 Loading base Moshi model...")
    checkpoint_info = loaders.CheckpointInfo.from_hf_repo(hf_repo)
    
    model = checkpoint_info.get_moshi(
        device=device,
        dtype=torch.bfloat16,
        load_weight=True
    )
    
    logger.info(f"✅ Loaded base Moshi model")

    # Step 4: Apply LoRA integration (if LoRA is enabled)
    if lora_enabled:
        logger.info("🔧 Applying LoRA integration...")
        lora_rank = lora_config.get('rank', 128)
        lora_scaling = lora_config.get('scaling', 2.0)

        logger.info(f"   LoRA rank: {lora_rank}")
        logger.info(f"   LoRA scaling: {lora_scaling}")

        # Replace all linear layers with LoRA
        replace_all_linear_with_lora(model, lora_rank, lora_scaling, device=device)
        logger.info("✅ LoRA integration applied")
    else:
        logger.info("⏭️  Skipping LoRA integration (not enabled in checkpoint)")

    # Step 5: Apply TTT integration (if TTT is enabled)
    if ttt_enabled:
        logger.info("🧠 Applying TTT integration...")

        # Get model config for TTT
        model_config = (
            loaders._lm_kwargs
            if checkpoint_info.raw_config is None
            else checkpoint_info.raw_config
        )

        apply_ttt_to_model(model, ttt_args, model_config)

        if not verify_ttt_integration(model):
            raise RuntimeError("TTT integration verification failed!")

        logger.info("✅ TTT integration applied")
    else:
        logger.info("⏭️  Skipping TTT integration (not enabled in checkpoint)")

    # Step 6: Load finetuned parameters (TTT/LoRA/full)
    logger.info("📥 Loading finetuned parameters from checkpoint...")
    weights_path = checkpoint_dir / "lora.safetensors"

    if not weights_path.exists():
        raise FileNotFoundError(f"lora.safetensors not found in {checkpoint_dir}")

    state_dict = safetensors.torch.load_file(str(weights_path))
    logger.info(f"✅ Loaded {len(state_dict)} parameters from checkpoint")

    # Load weights (will be in their checkpoint dtype - typically float32)
    # Use strict=False since we're loading partial weights (TTT, LoRA, or both)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    # Log loading results
    logger.info(f"✅ Checkpoint loaded successfully")
    if missing:
        logger.info(f"   Missing keys (expected for partial loading): {len(missing)}")
    if unexpected:
        # This is only unexpected if we don't have LoRA or TTT enabled
        if not lora_enabled and not ttt_enabled:
            logger.warning(f"⚠️  Unexpected keys in checkpoint (may indicate mismatch): {len(unexpected)}")
            logger.warning(f"   First few: {unexpected[:5]}")
        else:
            # With LoRA/TTT, strict=False is expected
            logger.debug(f"   Unexpected keys: {len(unexpected)} (expected with strict=False)")

    # Convert TTT parameters to bfloat16 if TTT is enabled
    # This must be done AFTER load_state_dict
    if ttt_enabled:
        for name, param in model.named_parameters():
            if 'ttt' in name.lower() or 'learnable_ttt_lr' in name or 'gating_alpha' in name:
                param.data = param.data.to(torch.bfloat16)
        logger.info("✅ TTT parameters converted to bfloat16")

    # Apply RoPE and KV cache fixes for long-context evaluation
    logger.info("🔧 Applying RoPE fix for long-context evaluation...")
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'rope') and model.transformer.rope is not None:
        model.transformer.rope.max_period = 100_000
        model.transformer.max_period = 100_000
        logger.info(f"✅ Set main transformer RoPE max_period: 10,000 → 100,000")
    if hasattr(model, 'depformer') and hasattr(model.depformer, 'transformer'):
        if hasattr(model.depformer.transformer, 'rope') and model.depformer.transformer.rope is not None:
            model.depformer.transformer.rope.max_period = 100_000
            model.depformer.transformer.max_period = 100_000
            logger.info(f"✅ Set depformer RoPE max_period: 10,000 → 100,000")

    logger.info("🔧 Increasing KV cache context size for long sequences...")
    new_context = 30000  # Increase from default 3000 to 30000 for LibriLight
    modified_layers = 0

    if hasattr(model, 'transformer') and hasattr(model.transformer, 'layers'):
        old_context = None
        for layer in model.transformer.layers:
            # Handle regular StreamingTransformerLayer
            if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'context'):
                if old_context is None:
                    old_context = layer.self_attn.context
                layer.self_attn.context = new_context
                modified_layers += 1
            # Handle HybridStreamingTransformerLayer (TTT layers)
            elif hasattr(layer, 'wrapped_layer') and hasattr(layer.wrapped_layer, 'self_attn'):
                if hasattr(layer.wrapped_layer.self_attn, 'context'):
                    if old_context is None:
                        old_context = layer.wrapped_layer.self_attn.context
                    layer.wrapped_layer.self_attn.context = new_context
                    modified_layers += 1
        if old_context is not None:
            logger.info(f"✅ Set main transformer context: {old_context} → {new_context} (all {modified_layers}/32 layers)")

    return model, checkpoint_info


def main():
    parser = argparse.ArgumentParser(
        description="Run paper metrics on TTT checkpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=False,
        default=None,
        help="Path to checkpoint directory (e.g., /path/to/checkpoint_000100/consolidated). Not needed for --baseline."
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Evaluate baseline Moshi model (no TTT) instead of a checkpoint"
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        default="kyutai/moshiko-pytorch-bf16",
        help="HuggingFace repo for base Moshi model"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=50,
        help="Max samples per task (default: 50 for quick eval)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results (default: checkpoint_dir/paper_metrics_results.json)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config with paper_metrics paths (optional)"
    )
    parser.add_argument(
        "--librilight-max-length",
        type=int,
        default=5000,
        help="Max sequence length for LibriLight evaluation (default: 5000 for quick eval, 24000 for full)"
    )
    parser.add_argument(
        "--skip-librilight",
        action="store_true",
        help="Skip LibriLight long-context evaluation"
    )

    args = parser.parse_args()
    
    # Validate arguments
    if args.baseline and args.checkpoint:
        logger.warning("Both --baseline and --checkpoint specified. Using --baseline mode.")
    
    if not args.baseline and not args.checkpoint:
        raise ValueError("Either --baseline or --checkpoint must be specified!")
    
    logger.info(f"\n{'='*80}")
    logger.info("PAPER METRICS EVALUATION")
    logger.info(f"{'='*80}")
    
    if args.baseline:
        logger.info(f"Mode:         BASELINE (pretrained)")
        logger.info(f"HF repo:      {args.hf_repo}")
    else:
        checkpoint_dir = Path(args.checkpoint)
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
        logger.info(f"Mode:         Finetuned Checkpoint")
        logger.info(f"Checkpoint:   {checkpoint_dir}")
    
    logger.info(f"Max samples:  {args.max_samples} per task")
    logger.info(f"Device:       {args.device}")
    logger.info(f"{'='*80}\n")
    
    # Load model
    logger.info("🏗️  Loading model...")
    if args.baseline:
        model, checkpoint_info = load_baseline_model(args.hf_repo, args.device)
        checkpoint_dir = None  # No checkpoint directory for baseline
    else:
        checkpoint_dir = Path(args.checkpoint)
        model, checkpoint_info = load_ttt_model(checkpoint_dir, args.hf_repo, args.device)
    
    model.eval()
    
    # Load MIMI encoder
    logger.info("🎤 Loading MIMI encoder...")
    mimi = checkpoint_info.get_mimi(device=args.device)

    # Create LMGen for LibriLight streaming evaluation (if not skipped)
    lm_gen = None
    if not args.skip_librilight:
        logger.info("🔄 Creating LMGen for LibriLight streaming evaluation...")
        from moshi.models.lm import LMGen
        lm_gen = LMGen(model, temp=0.8, check=False)
        logger.info("✅ LMGen created")
    else:
        logger.info("⏭️  Skipping LibriLight (LMGen not created)")

    # Load paper metrics config if provided
    paper_metrics_config = {}
    if args.config:
        logger.info(f"📄 Loading paper metrics config from {args.config}")
        import yaml
        with open(args.config) as f:
            full_config = yaml.safe_load(f)
            if 'paper_metrics' in full_config:
                paper_metrics_config = full_config['paper_metrics']
                logger.info(f"✅ Loaded paper metrics config with {len(paper_metrics_config)} settings")
            else:
                logger.warning("No 'paper_metrics' section found in config")

    # Override config with command-line max_samples if provided
    if args.max_samples:
        logger.info(f"📊 Overriding max samples from command line: {args.max_samples}")
        paper_metrics_config['sblimp_max_pairs'] = args.max_samples
        paper_metrics_config['swuggy_max_pairs'] = args.max_samples
        paper_metrics_config['tstory_max_pairs'] = args.max_samples
        paper_metrics_config['sstory_max_pairs'] = args.max_samples

    # Note: Paper metrics evaluator doesn't actually use the tokenizer
    # (it only uses MIMI for audio encoding), so we pass None
    logger.info("📊 Creating paper metrics evaluator...")
    from finetune.paper_metrics import PaperMetricsEvaluator

    evaluator = PaperMetricsEvaluator(
        mimi_encoder=mimi,
        interleaved_tokenizer=None,  # Not used by paper metrics
        device=args.device,
        config=paper_metrics_config
    )
    
    # Run evaluation
    logger.info(f"\n{'='*80}")
    logger.info("RUNNING PAPER METRICS EVALUATION")
    logger.info(f"{'='*80}\n")
    logger.info("This will evaluate on:")
    logger.info("  • sBLIMP (syntactic minimal pairs)")
    logger.info("  • sWUGGY (phonotactic minimal pairs)")
    logger.info("  • tStoryCloze (textual story completion)")
    logger.info("  • sStoryCloze (spoken story completion)")
    if lm_gen is not None:
        logger.info(f"  • LibriLight (long-context, up to {args.librilight_max_length} tokens)")
    logger.info("")
    estimated_time = 5 + (10 if lm_gen is not None else 0)  # +10 min for LibriLight
    logger.info(f"⏱️  Expected time: ~{estimated_time}-{estimated_time+5} minutes for {args.max_samples} samples per task")
    logger.info("")

    try:
        with torch.no_grad():
            results = evaluator.evaluate_all(model)
        
        logger.info(f"\n{'='*80}")
        logger.info("✅ EVALUATION COMPLETE")
        logger.info(f"{'='*80}\n")
        
        # Print results
        logger.info("📊 Results:")
        for key, value in sorted(results.items()):
            if isinstance(value, float):
                if key.endswith('_accuracy') or key == 'paper_metrics_avg':
                    logger.info(f"   {key:30s}: {value*100:6.2f}%")
                else:
                    logger.info(f"   {key:30s}: {value:6.4f}")
            else:
                logger.info(f"   {key:30s}: {value}")
        
        # Save results
        output_path = args.output
        if output_path is None:
            if args.baseline:
                output_path = Path("./baseline_paper_metrics_results.json")
            else:
                output_path = checkpoint_dir / "paper_metrics_results.json"
        else:
            output_path = Path(output_path)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"\n💾 Results saved to: {output_path}")

        # Register this run in the dashboard registry
        register_run(checkpoint_dir, output_path)

        # Update dashboard automatically
        update_dashboard(output_path)

        logger.info(f"\n{'='*80}\n")

        return 0
        
    except Exception as e:
        logger.error(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
