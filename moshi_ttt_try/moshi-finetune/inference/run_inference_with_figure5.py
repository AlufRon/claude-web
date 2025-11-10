#!/usr/bin/env python3
"""
TTT Inference with Figure 5 Diagnostics

This script runs inference and generates Figure 5 plots to diagnose TTT learning:
- Blue (l0): Loss with frozen initial weights W₀ (no adaptation)
- Orange (lprev): Loss before gradient update (accumulated learning)
- Green (lafter): Loss after gradient update (immediate improvement)

Expected healthy TTT behavior:
  Blue > Orange > Green  (learning works)
  Orange/Green decrease over time (adaptation from context)
  
If TTT training failed:
  - Lines overlap (no learning)
  - Lines increase (degenerative learning)
  - Huge loss values (numerical instability)
"""

import sys
import os
import argparse
import logging
from pathlib import Path
import torch
import numpy as np
from collections import deque

# Add moshi to path
sys.path.insert(0, str(Path(__file__).parent / "moshi" / "moshi"))

from moshi.models import loaders, MimiModel, LMModel, LMGen
import sentencepiece
import sphn

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)


def log(level, msg):
    """Colored logging for user-visible messages."""
    colors = {
        "info": "\033[1;34m",  # Blue
        "warning": "\033[1;33m",  # Yellow
        "error": "\033[1;31m",  # Red
        "success": "\033[1;32m",  # Green
    }
    reset = "\033[0m"
    color = colors.get(level, "")
    print(f"{color}[{level.capitalize()}]{reset} {msg}", flush=True)


def load_ttt_model(checkpoint_dir: Path, hf_repo: str, device: str = "cuda") -> torch.nn.Module:
    """Load TTT model from checkpoint."""
    from finetune.ttt_integration import apply_ttt_to_model, TTTArgs
    
    # Load configs
    checkpoint_info = loaders.CheckpointInfo.from_checkpoint_dir(checkpoint_dir)
    log("info", f"✅ Loaded configs from {checkpoint_dir}")
    
    # Extract TTT config
    config_path = checkpoint_dir / "ttt_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"❌ TTT config not found: {config_path}")
    
    import yaml
    with open(config_path) as f:
        ttt_config = yaml.safe_load(f)
    
    # Create TTTArgs
    ttt_args = TTTArgs(
        ttt_layers=ttt_config['ttt_layers'],
        ttt_base_lr=ttt_config['ttt_base_lr'],
        mini_batch_size=ttt_config['mini_batch_size'],
        persistent_state=ttt_config.get('persistent_state', False),
        initial_gating_alpha=ttt_config.get('initial_gating_alpha', 0.3),
        ttt_mlp_num_layers=ttt_config.get('ttt_mlp_num_layers', 3),
        ttt_mlp_expansion_factor=ttt_config.get('ttt_mlp_expansion_factor', 4.0),
    )
    
    log("info", f"✅ Created TTTArgs from checkpoint config:")
    log("info", f"   Layers: {ttt_args.ttt_layers}")
    log("info", f"   Base LR: {ttt_args.ttt_base_lr}")
    log("info", f"   Mini batch size: {ttt_args.mini_batch_size}")
    log("info", f"   Initial gating alpha: {ttt_args.initial_gating_alpha}")
    log("info", f"   TTT-MLP layers: {ttt_args.ttt_mlp_num_layers}")
    log("info", f"   TTT-MLP expansion factor: {ttt_args.ttt_mlp_expansion_factor}")
    
    # Load base model
    log("info", "🔨 Loading base Moshi model...")
    model = loaders.get_moshi_lm(hf_repo, device=device, dtype=torch.bfloat16)
    log("info", f"✅ Loaded base Moshi model from {hf_repo}")
    
    # Apply TTT integration
    log("info", "🧠 Applying TTT integration...")
    apply_ttt_to_model(model, ttt_args, device=device)
    
    # Convert base model to bfloat16
    log("info", "🔄 Converting base model to bfloat16...")
    for name, param in model.named_parameters():
        if "ttt" not in name.lower():
            param.data = param.data.to(torch.bfloat16)
    
    # Load TTT checkpoint
    log("info", "📥 Loading TTT parameters from checkpoint...")
    lora_path = checkpoint_dir / "lora.safetensors"
    if not lora_path.exists():
        raise FileNotFoundError(f"❌ TTT checkpoint not found: {lora_path}")
    
    from safetensors.torch import load_file
    ttt_state_dict = load_file(lora_path)
    log("info", f"✅ Loaded {len(ttt_state_dict)} parameters from lora.safetensors")
    
    # Keep TTT weights in float32 for precision
    log("info", "🔄 Keeping TTT weights in float32 for precision (ttt_norm can be bfloat16)...")
    for key in ttt_state_dict.keys():
        if "ttt_norm" in key:
            ttt_state_dict[key] = ttt_state_dict[key].to(torch.bfloat16)
        else:
            ttt_state_dict[key] = ttt_state_dict[key].to(torch.float32)
    
    # Load TTT parameters (strict=False because base model is already loaded)
    missing, unexpected = model.load_state_dict(ttt_state_dict, strict=False)
    
    if missing:
        logger.warning(f"⚠️  Missing keys (expected for base model): {len(missing)} keys")
    if unexpected:
        logger.error(f"❌ Unexpected keys: {unexpected}")
        raise RuntimeError("Unexpected keys in checkpoint!")
    
    log("info", "✅ TTT checkpoint loaded successfully!")
    
    return model


def run_inference_with_figure5(
    model: LMModel,
    mimi: MimiModel,
    input_audio_path: str,
    output_dir: Path,
    device: str = "cuda",
    max_T: int = 2048,
    ttt_layers: list = [29, 30, 31],
):
    """Run inference with Figure 5 logging enabled."""
    from moshi_ttt.models.ssm.ops.ttt_mlp import fig5_set_logging, fig5_clear, fig5_get
    
    # Enable Figure 5 logging
    log("info", f"📊 Enabling Figure 5 logging (max_T={max_T}, layers={ttt_layers})")
    fig5_set_logging(True, max_T=max_T)
    fig5_clear()
    
    # ============================================================
    # 🔍 ENABLE MAGNITUDE LOGGING for TTT dominance tracking
    # ============================================================
    log("info", "🔍 Enabling TTT magnitude logging to track output dominance...")
    ttt_layer_objects = []
    for layer_idx in ttt_layers:
        try:
            layer = model.transformer.layers[layer_idx]
            if hasattr(layer, 'seq_modeling_block'):
                layer.seq_modeling_block.enable_magnitude_logging()
                ttt_layer_objects.append(layer.seq_modeling_block)
            else:
                log("warning", f"⚠️  Layer {layer_idx} doesn't have seq_modeling_block")
        except Exception as e:
            log("warning", f"⚠️  Failed to enable magnitude logging for layer {layer_idx}: {e}")
    
    log("info", f"✅ Magnitude logging enabled for {len(ttt_layer_objects)} TTT layers")
    
    # Load input audio
    log("info", f"🎤 Loading audio from {input_audio_path}")
    in_pcms, _ = sphn.read(input_audio_path, sample_rate=mimi.sample_rate)
    in_pcms = torch.from_numpy(in_pcms).to(device=device)
    in_pcms = in_pcms[None, 0:1]  # [B=1, C=1, T]
    
    duration = in_pcms.shape[-1] / mimi.sample_rate
    log("info", f"✅ Loaded {duration:.1f}s of audio at {mimi.sample_rate} Hz")
    
    # Setup streaming
    batch_size = 1
    frame_size = int(mimi.sample_rate / mimi.frame_rate)
    
    # Create LMGen
    lm_gen = LMGen(
        model,
        use_sampling=True,
        temp=0.8,
        temp_text=0.9,
        top_k=250,
        top_k_text=25,
    )
    
    log("info", "🏃 Starting streaming inference with Figure 5 logging...")
    
    # Pad audio to frame boundaries
    pad_right = (frame_size - in_pcms.shape[-1] % frame_size) % frame_size
    if pad_right > 0:
        in_pcms = torch.nn.functional.pad(in_pcms, (0, pad_right), mode="constant")
    
    # Split into chunks
    chunks = deque([
        chunk for chunk in in_pcms.split(frame_size, dim=2)
        if chunk.shape[-1] == frame_size
    ])
    
    # Setup streaming mode
    mimi.streaming_forever(batch_size)
    lm_gen.streaming_forever(batch_size)
    
    # Process chunks
    out_pcms = []
    text_tokens = []
    first_frame = True
    
    model.eval()
    with torch.no_grad():
        while chunks:
            chunk = chunks.popleft()
            codes = mimi.encode(chunk)
            
            if first_frame:
                # First frame needs special handling
                tokens = lm_gen.step(codes)
                if max(lm_gen.lm_model.delays) > 0:
                    assert tokens is None
                first_frame = False
            
            tokens = lm_gen.step(codes)
            if tokens is None:
                continue
            
            # Decode audio
            audio_tokens = tokens[:, 1:, :]
            out_pcm = mimi.decode(audio_tokens)
            out_pcms.append(out_pcm)
            
            # Collect text
            text_token = tokens[:, 0, 0].item()
            if text_token not in (0, 3):
                text_tokens.append(text_token)
    
    log("info", "✅ Inference complete!")
    
    # ============================================================
    # 🔍 PRINT MAGNITUDE ANALYSIS - Check if TTT dominates output
    # ============================================================
    log("info", "")
    log("info", "=" * 80)
    log("info", "🔍 TTT MAGNITUDE ANALYSIS - Checking Output Dominance")
    log("info", "=" * 80)
    
    for layer_obj in ttt_layer_objects:
        layer_obj.print_magnitude_summary()
    
    # Save detailed magnitude stats to file
    magnitude_report_path = output_dir / "ttt_magnitude_analysis.txt"
    with open(magnitude_report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("TTT MAGNITUDE ANALYSIS - Output Dominance Check\n")
        f.write("=" * 80 + "\n\n")
        f.write("This report checks if TTT learned to output large values to bypass\n")
        f.write("the gating mechanism and dominate the attention output.\n\n")
        f.write("Key Metrics:\n")
        f.write("  - Ratio (TTT/Residual): >1.0 means TTT is LARGER than attention\n")
        f.write("  - Effective Ratio: Shows actual contribution after gating\n")
        f.write("  - Attenuation: How much gating reduces TTT output\n")
        f.write("\n" + "=" * 80 + "\n\n")
        
        for layer_obj in ttt_layer_objects:
            stats = layer_obj.get_magnitude_stats()
            if not stats:
                f.write(f"Layer {layer_obj.layer_id}: No data collected\n\n")
                continue
            
            import numpy as np
            
            # Aggregate
            ttt_raw_norms = [s['ttt_raw_norm'] for s in stats]
            residual_norms = [s['residual_norm'] for s in stats]
            gated_norms = [s['gated_norm'] for s in stats]
            ratios = [s['ratio_ttt_to_residual'] for s in stats]
            effective_ratios = [s['effective_ratio'] for s in stats]
            attenuations = [s['attenuation'] for s in stats]
            
            f.write(f"LAYER {layer_obj.layer_id}\n")
            f.write(f"{'-'*60}\n")
            f.write(f"Samples: {len(stats)}\n\n")
            f.write(f"TTT Raw Output (before gating):\n")
            f.write(f"  L2 norm: mean={np.mean(ttt_raw_norms):.6f}, max={np.max(ttt_raw_norms):.6f}\n")
            f.write(f"  Max abs: mean={np.mean([s['ttt_raw_max'] for s in stats]):.6f}, max={np.max([s['ttt_raw_max'] for s in stats]):.6f}\n\n")
            f.write(f"Attention Output (residual):\n")
            f.write(f"  L2 norm: mean={np.mean(residual_norms):.6f}\n\n")
            f.write(f"Ratio (TTT/Residual):\n")
            f.write(f"  mean={np.mean(ratios):.4f}x, max={np.max(ratios):.4f}x\n")
            f.write(f"  {'⚠️  TTT DOMINATES' if np.mean(ratios) > 1.0 else '✅ Attention dominates'}\n\n")
            f.write(f"After Gating:\n")
            f.write(f"  L2 norm: mean={np.mean(gated_norms):.6f}\n")
            f.write(f"  Effective contribution: {np.mean(effective_ratios)*100:.2f}% (mean), {np.max(effective_ratios)*100:.2f}% (max)\n")
            f.write(f"  Attenuation: {np.mean(attenuations):.2f}x (mean), {np.max(attenuations):.2f}x (max)\n\n")
            f.write(f"{'-'*60}\n\n")
    
    log("success", f"💾 Magnitude analysis saved: {magnitude_report_path}")
    log("info", "=" * 80)
    log("info", "")
    
    # Disable magnitude logging
    for layer_obj in ttt_layer_objects:
        layer_obj.disable_magnitude_logging()
    
    # Concatenate output
    if out_pcms:
        out_audio = torch.cat(out_pcms, dim=-1)
        output_path = output_dir / "output_with_figure5.wav"
        sphn.write_wav(
            str(output_path),
            out_audio[0, 0].cpu().numpy().astype(np.float32),
            mimi.sample_rate,
        )
        log("success", f"💾 Saved output audio: {output_path}")
    
    # Get Figure 5 data
    log("info", "📈 Collecting Figure 5 data...")
    raw_data = fig5_get()
    
    # Disable logging
    fig5_set_logging(False)
    
    return raw_data, text_tokens


def analyze_figure5_data(raw_data: dict, output_dir: Path, ttt_layers: list):
    """Analyze Figure 5 data and create diagnostic plots + report."""
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log("info", "🔬 Analyzing Figure 5 data...")
    
    # Create diagnostic report
    report_path = output_dir / "ttt_diagnostic_report.txt"
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("TTT DIAGNOSTIC REPORT - Figure 5 Analysis\n")
        f.write("=" * 80 + "\n\n")
        
        for layer_id in ttt_layers:
            if layer_id not in raw_data:
                f.write(f"\n❌ Layer {layer_id}: NO DATA COLLECTED\n")
                continue
            
            data = raw_data[layer_id]
            counts = data['cnt']
            l0_sums = data['l0']
            lprev_sums = data['lprev']
            lafter_sums = data['lafter']
            
            # Find valid positions
            valid_positions = [i for i, cnt in enumerate(counts) if cnt > 0]
            
            if not valid_positions:
                f.write(f"\n❌ Layer {layer_id}: NO VALID POSITIONS\n")
                continue
            
            f.write(f"\n{'='*60}\n")
            f.write(f"LAYER {layer_id} ANALYSIS\n")
            f.write(f"{'='*60}\n\n")
            
            # Compute averaged losses
            l0_avg = [l0_sums[i] / counts[i] if counts[i] > 0 else 0 for i in range(len(counts))]
            lprev_avg = [lprev_sums[i] / counts[i] if counts[i] > 0 else 0 for i in range(len(counts))]
            lafter_avg = [lafter_sums[i] / counts[i] if counts[i] > 0 else 0 for i in range(len(counts))]
            
            # Extract valid values
            positions = valid_positions
            l0_vals = [l0_avg[i] for i in positions]
            lprev_vals = [lprev_avg[i] for i in positions]
            lafter_vals = [lafter_avg[i] for i in positions]
            
            # Statistics
            f.write(f"📊 Data Coverage:\n")
            f.write(f"   Valid positions: {len(positions)} / {len(counts)}\n")
            f.write(f"   Position range: {min(positions)} to {max(positions)}\n\n")
            
            f.write(f"📈 Loss Statistics:\n")
            f.write(f"   l0 (W₀):     mean={np.mean(l0_vals):.6f}, std={np.std(l0_vals):.6f}, min={np.min(l0_vals):.6f}, max={np.max(l0_vals):.6f}\n")
            f.write(f"   lprev (Wₜ₋₁): mean={np.mean(lprev_vals):.6f}, std={np.std(lprev_vals):.6f}, min={np.min(lprev_vals):.6f}, max={np.max(lprev_vals):.6f}\n")
            f.write(f"   lafter (Wₜ):  mean={np.mean(lafter_vals):.6f}, std={np.std(lafter_vals):.6f}, min={np.min(lafter_vals):.6f}, max={np.max(lafter_vals):.6f}\n\n")
            
            # Check for expected ordering
            ordering_violations = 0
            improvement_sum = 0
            for i in range(len(positions)):
                if not (l0_vals[i] >= lprev_vals[i] >= lafter_vals[i]):
                    ordering_violations += 1
                improvement = lprev_vals[i] - lafter_vals[i]
                improvement_sum += improvement
            
            violation_rate = ordering_violations / len(positions) * 100
            avg_improvement = improvement_sum / len(positions)
            
            f.write(f"🎯 Learning Quality:\n")
            f.write(f"   Ordering violations: {ordering_violations}/{len(positions)} ({violation_rate:.1f}%)\n")
            f.write(f"   Avg per-step improvement: {avg_improvement:.6f}\n\n")
            
            # Diagnose issues
            f.write(f"🔍 Diagnosis:\n")
            
            if np.mean(l0_vals) > 10.0:
                f.write(f"   ⚠️  HUGE LOSSES: Average l0={np.mean(l0_vals):.2f} >> 1.0\n")
                f.write(f"       → TTT weights may be diverging or numerically unstable\n")
                f.write(f"       → Check: learning rate too high?\n\n")
            
            if violation_rate > 50:
                f.write(f"   ❌ HIGH VIOLATION RATE: {violation_rate:.1f}% positions violate l0 >= lprev >= lafter\n")
                f.write(f"       → TTT learning is NOT working as expected\n")
                f.write(f"       → Expected: Blue > Orange > Green\n\n")
            
            if avg_improvement < 0.0001:
                f.write(f"   ❌ MINIMAL IMPROVEMENT: {avg_improvement:.6f} per step\n")
                f.write(f"       → TTT gradient updates have negligible effect\n")
                f.write(f"       → Check: learning rate too small? Gradient flow broken?\n\n")
            
            if avg_improvement < 0:
                f.write(f"   ❌ NEGATIVE IMPROVEMENT: {avg_improvement:.6f} per step\n")
                f.write(f"       → TTT updates are INCREASING loss!\n")
                f.write(f"       → Learning is degenerative\n\n")
            
            # Check for normal range
            if np.mean(l0_vals) < 0.001 or np.mean(l0_vals) > 5.0:
                f.write(f"   ⚠️  ABNORMAL LOSS RANGE: l0 mean={np.mean(l0_vals):.6f}\n")
                f.write(f"       → Expected range: 0.01 to 1.0 for typical language modeling\n\n")
            
            # Trend analysis
            if len(positions) > 10:
                early_l0 = np.mean(l0_vals[:len(positions)//4])
                late_l0 = np.mean(l0_vals[-len(positions)//4:])
                l0_trend = late_l0 - early_l0
                
                early_lprev = np.mean(lprev_vals[:len(positions)//4])
                late_lprev = np.mean(lprev_vals[-len(positions)//4:])
                lprev_trend = late_lprev - early_lprev
                
                f.write(f"📉 Trend Analysis (first 25% vs last 25%):\n")
                f.write(f"   l0 trend: {l0_trend:+.6f} {'(improving ✅)' if l0_trend < 0 else '(degrading ❌)' if l0_trend > 0 else '(flat)'}\n")
                f.write(f"   lprev trend: {lprev_trend:+.6f} {'(adapting ✅)' if lprev_trend < 0 else '(degrading ❌)' if lprev_trend > 0 else '(flat)'}\n\n")
                
                if lprev_trend > 0:
                    f.write(f"   ❌ lprev INCREASING over time\n")
                    f.write(f"       → TTT is NOT adapting to context\n")
                    f.write(f"       → Expected: orange line should decrease as context grows\n\n")
    
    log("success", f"📄 Diagnostic report saved: {report_path}")
    
    # Create plots
    for layer_id in ttt_layers:
        if layer_id not in raw_data:
            continue
        
        data = raw_data[layer_id]
        counts = data['cnt']
        l0_sums = data['l0']
        lprev_sums = data['lprev']
        lafter_sums = data['lafter']
        
        # Find valid positions
        valid_positions = [i for i, cnt in enumerate(counts) if cnt > 0]
        if not valid_positions:
            continue
        
        # Compute averaged losses
        positions = valid_positions
        l0_vals = [l0_sums[i] / counts[i] for i in positions]
        lprev_vals = [lprev_sums[i] / counts[i] for i in positions]
        lafter_vals = [lafter_sums[i] / counts[i] for i in positions]
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Raw losses
        ax1.plot(positions, l0_vals, 'b-', label='l0 (W₀) - No adaptation', linewidth=2, alpha=0.8)
        ax1.plot(positions, lprev_vals, 'orange', label='lprev (Wₜ₋₁) - Before update', linewidth=2, alpha=0.8)
        ax1.plot(positions, lafter_vals, 'g-', label='lafter (Wₜ) - After update', linewidth=2, alpha=0.8)
        ax1.set_xlabel('Token Position', fontsize=12)
        ax1.set_ylabel('Reconstruction Loss', fontsize=12)
        ax1.set_title(f'Figure 5: TTT Inner Loop Learning - Layer {layer_id}', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Improvement metrics
        improvements = [lprev_vals[i] - lafter_vals[i] for i in range(len(positions))]
        cumulative_benefit = [l0_vals[i] - lafter_vals[i] for i in range(len(positions))]
        
        ax2.plot(positions, improvements, 'purple', label='Per-step improvement (lprev - lafter)', linewidth=2, alpha=0.8)
        ax2.plot(positions, cumulative_benefit, 'teal', label='Cumulative benefit (l0 - lafter)', linewidth=2, alpha=0.8)
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Token Position', fontsize=12)
        ax2.set_ylabel('Loss Reduction', fontsize=12)
        ax2.set_title(f'TTT Learning Effectiveness - Layer {layer_id}', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = output_dir / f"figure5_layer{layer_id}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        log("success", f"📊 Saved plot: {plot_path}")
    
    log("success", f"✅ Figure 5 analysis complete! Check: {output_dir}")


def main():
    parser = argparser()
    args = parser.parse_args()
    
    # Setup paths
    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log("info", "=" * 80)
    log("info", "TTT Inference with Figure 5 Diagnostics")
    log("info", "=" * 80)
    log("info", f"📁 Checkpoint: {checkpoint_dir}")
    log("info", f"🎤 Input: {args.input_audio}")
    log("info", f"📂 Output: {output_dir}")
    log("info", f"🔬 Figure 5 max_T: {args.max_T}")
    log("info", "=" * 80)
    
    # Load Mimi
    log("info", "🎵 Loading Mimi audio codec...")
    checkpoint_info = loaders.CheckpointInfo.from_checkpoint_dir(checkpoint_dir)
    mimi = checkpoint_info.get_mimi(device=args.device)
    log("info", "✅ Mimi loaded")
    
    # Load TTT model
    log("info", "🧠 Loading TTT model...")
    model = load_ttt_model(checkpoint_dir, args.hf_repo, device=args.device)
    log("info", "✅ TTT model loaded")
    
    # Run inference with Figure 5
    raw_data, text_tokens = run_inference_with_figure5(
        model=model,
        mimi=mimi,
        input_audio_path=args.input_audio,
        output_dir=output_dir,
        device=args.device,
        max_T=args.max_T,
        ttt_layers=args.ttt_layers,
    )
    
    # Analyze Figure 5 data
    if raw_data:
        analyze_figure5_data(raw_data, output_dir, args.ttt_layers)
    else:
        log("warning", "⚠️  No Figure 5 data collected!")
    
    log("success", "=" * 80)
    log("success", "✅ Complete! Check diagnostic report and plots in:")
    log("success", f"   {output_dir}")
    log("success", "=" * 80)


def argparser():
    parser = argparse.ArgumentParser(description="TTT Inference with Figure 5 Diagnostics")
    parser.add_argument("--checkpoint-dir", type=str, required=True,
                        help="Path to TTT checkpoint directory")
    parser.add_argument("--input-audio", type=str, required=True,
                        help="Path to input audio file")
    parser.add_argument("--output-dir", type=str, default="./figure5_diagnostics",
                        help="Output directory for plots and reports")
    parser.add_argument("--hf-repo", type=str, default="kyutai/moshiko-pytorch-bf16",
                        help="HuggingFace repo for base Moshi model")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda/cpu)")
    parser.add_argument("--max-T", type=int, default=2048,
                        help="Maximum token positions to track")
    parser.add_argument("--ttt-layers", type=int, nargs='+', default=[29, 30, 31],
                        help="TTT layer IDs to analyze")
    return parser


if __name__ == "__main__":
    main()
