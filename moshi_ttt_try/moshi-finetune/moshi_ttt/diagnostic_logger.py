"""
Diagnostic logger for TTT-Moshi hybrid layer analysis.

Tracks key metrics to understand TTT's impact on the model:
- Magnitude analysis (attention vs TTT contributions)
- Gating behavior
- Distribution statistics
- Cosine similarity
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


class TTTDiagnosticLogger:
    """
    Tracks and logs diagnostic information about TTT layer behavior.
    Designed to be called every N steps during inference.
    """

    def __init__(
        self,
        layer_id: int,
        log_frequency: int = 100,
        track_history: bool = False,
        history_length: int = 1000
    ):
        """
        Args:
            layer_id: Layer identifier
            log_frequency: Log diagnostics every N steps
            track_history: Whether to keep history of metrics
            history_length: Max history entries to keep
        """
        self.layer_id = layer_id
        self.log_frequency = log_frequency
        self.track_history = track_history
        self.history_length = history_length

        # Step counter
        self.step_count = 0

        # History storage
        if track_history:
            self.history: Dict[str, List[float]] = {
                'attn_magnitude': [],
                'ttt_magnitude': [],
                'combined_magnitude': [],
                'ttt_ratio': [],
                'gating_alpha_mean': [],
                'gating_alpha_std': [],
                'cosine_similarity': [],
            }

        # Previous step cache for delta computation
        self.prev_attn_output: Optional[torch.Tensor] = None
        self.prev_ttt_output: Optional[torch.Tensor] = None

    def log_step(
        self,
        attn_output: torch.Tensor,
        ttt_output: torch.Tensor,
        combined_output: torch.Tensor,
        gating_alpha: Optional[torch.Tensor] = None,
        ttt_weight_norm: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Log diagnostics for current step.

        Args:
            attn_output: Output from attention block [B, seq_len, d_model]
            ttt_output: Output from TTT processing (before gating) [B, seq_len, d_model]
            combined_output: Final output after gating [B, seq_len, d_model]
            gating_alpha: Gating alpha values [B, seq_len, d_model] or [heads, d_model]
            ttt_weight_norm: Norm of TTT internal weights

        Returns:
            Dict of computed metrics (empty if not logging this step)
        """
        self.step_count += 1

        # Only log at specified frequency
        if self.step_count % self.log_frequency != 0:
            # Still cache for delta computation
            self.prev_attn_output = attn_output.detach().clone()
            self.prev_ttt_output = ttt_output.detach().clone()
            return {}

        metrics = {}

        with torch.no_grad():
            # ============================================================
            # 1. MAGNITUDE ANALYSIS
            # ============================================================
            attn_mag = attn_output.norm().item()
            ttt_mag = ttt_output.norm().item()
            combined_mag = combined_output.norm().item()

            # Avoid division by zero
            ttt_ratio = ttt_mag / (attn_mag + 1e-8)

            metrics['attn_magnitude'] = attn_mag
            metrics['ttt_magnitude'] = ttt_mag
            metrics['combined_magnitude'] = combined_mag
            metrics['ttt_ratio'] = ttt_ratio

            # Per-token magnitudes (averaged over batch and features)
            attn_mag_per_token = attn_output.norm(dim=-1).mean().item()
            ttt_mag_per_token = ttt_output.norm(dim=-1).mean().item()
            combined_mag_per_token = combined_output.norm(dim=-1).mean().item()

            metrics['attn_mag_per_token'] = attn_mag_per_token
            metrics['ttt_mag_per_token'] = ttt_mag_per_token
            metrics['combined_mag_per_token'] = combined_mag_per_token

            # ============================================================
            # 2. DISTRIBUTION STATISTICS
            # ============================================================
            metrics['attn_mean'] = attn_output.mean().item()
            metrics['attn_std'] = attn_output.std().item()
            metrics['attn_max'] = attn_output.max().item()
            metrics['attn_min'] = attn_output.min().item()

            metrics['ttt_mean'] = ttt_output.mean().item()
            metrics['ttt_std'] = ttt_output.std().item()
            metrics['ttt_max'] = ttt_output.max().item()
            metrics['ttt_min'] = ttt_output.min().item()

            metrics['combined_mean'] = combined_output.mean().item()
            metrics['combined_std'] = combined_output.std().item()
            metrics['combined_max'] = combined_output.max().item()
            metrics['combined_min'] = combined_output.min().item()

            # Sparsity (percentage of values near zero)
            attn_sparsity = (attn_output.abs() < 0.01).float().mean().item() * 100
            ttt_sparsity = (ttt_output.abs() < 0.01).float().mean().item() * 100
            combined_sparsity = (combined_output.abs() < 0.01).float().mean().item() * 100

            metrics['attn_sparsity'] = attn_sparsity
            metrics['ttt_sparsity'] = ttt_sparsity
            metrics['combined_sparsity'] = combined_sparsity

            # ============================================================
            # 3. COSINE SIMILARITY
            # ============================================================
            # Flatten for cosine similarity
            attn_flat = attn_output.flatten()
            ttt_flat = ttt_output.flatten()
            combined_flat = combined_output.flatten()

            # Attention vs TTT
            cos_attn_ttt = F.cosine_similarity(
                attn_flat.unsqueeze(0),
                ttt_flat.unsqueeze(0)
            ).item()
            metrics['cos_attn_ttt'] = cos_attn_ttt

            # Attention vs Combined (shows how much TTT changed direction)
            cos_attn_combined = F.cosine_similarity(
                attn_flat.unsqueeze(0),
                combined_flat.unsqueeze(0)
            ).item()
            metrics['cos_attn_combined'] = cos_attn_combined

            # TTT vs Combined
            cos_ttt_combined = F.cosine_similarity(
                ttt_flat.unsqueeze(0),
                combined_flat.unsqueeze(0)
            ).item()
            metrics['cos_ttt_combined'] = cos_ttt_combined

            # ============================================================
            # 4. GATING ANALYSIS
            # ============================================================
            if gating_alpha is not None:
                metrics['gating_alpha_mean'] = gating_alpha.mean().item()
                metrics['gating_alpha_std'] = gating_alpha.std().item()
                metrics['gating_alpha_max'] = gating_alpha.max().item()
                metrics['gating_alpha_min'] = gating_alpha.min().item()

                # Effective TTT contribution (1 - alpha for residual formulation)
                # Assuming: output = alpha * residual + (1 - alpha) * ttt_output
                # or: output = alpha * attn + (1 - alpha) * ttt
                effective_ttt_weight = (1 - gating_alpha.mean()).item()
                metrics['effective_ttt_weight'] = effective_ttt_weight

            # ============================================================
            # 5. DELTA ANALYSIS (change from previous step)
            # ============================================================
            if self.prev_attn_output is not None:
                attn_delta = (attn_output - self.prev_attn_output).norm().item()
                metrics['attn_delta'] = attn_delta

            if self.prev_ttt_output is not None:
                ttt_delta = (ttt_output - self.prev_ttt_output).norm().item()
                metrics['ttt_delta'] = ttt_delta

                # Relative change
                if ttt_mag > 1e-8:
                    ttt_relative_change = ttt_delta / ttt_mag
                    metrics['ttt_relative_change'] = ttt_relative_change

            # ============================================================
            # 6. TTT INTERNAL WEIGHTS
            # ============================================================
            if ttt_weight_norm is not None:
                metrics['ttt_weight_norm'] = ttt_weight_norm

        # Cache for next delta computation
        self.prev_attn_output = attn_output.detach().clone()
        self.prev_ttt_output = ttt_output.detach().clone()

        # Store in history
        if self.track_history:
            for key in ['attn_magnitude', 'ttt_magnitude', 'combined_magnitude',
                       'ttt_ratio', 'gating_alpha_mean', 'gating_alpha_std', 'cos_attn_ttt']:
                if key in metrics:
                    self.history[key].append(metrics[key])
                    # Trim history
                    if len(self.history[key]) > self.history_length:
                        self.history[key].pop(0)

        # Log to console
        self._log_metrics(metrics)

        return metrics

    def _log_metrics(self, metrics: Dict[str, float]):
        """Format and log metrics to console."""
        logger.info(f"")
        logger.info(f"{'='*80}")
        logger.info(f"📊 TTT Diagnostics - Layer {self.layer_id} - Step {self.step_count}")
        logger.info(f"{'='*80}")

        # Magnitude Analysis
        logger.info(f"")
        logger.info(f"🔍 MAGNITUDE ANALYSIS:")
        logger.info(f"   Attention:  {metrics['attn_magnitude']:>12.4f}  (per-token: {metrics.get('attn_mag_per_token', 0):>8.4f})")
        logger.info(f"   TTT:        {metrics['ttt_magnitude']:>12.4f}  (per-token: {metrics.get('ttt_mag_per_token', 0):>8.4f})")
        logger.info(f"   Combined:   {metrics['combined_magnitude']:>12.4f}  (per-token: {metrics.get('combined_mag_per_token', 0):>8.4f})")
        logger.info(f"   TTT/Attn Ratio: {metrics['ttt_ratio']:.4f}x")

        # Gating Analysis
        if 'gating_alpha_mean' in metrics:
            logger.info(f"")
            logger.info(f"🎛️  GATING ANALYSIS:")
            logger.info(f"   Alpha (mean):  {metrics['gating_alpha_mean']:>8.6f}")
            logger.info(f"   Alpha (std):   {metrics['gating_alpha_std']:>8.6f}")
            logger.info(f"   Alpha (range): [{metrics['gating_alpha_min']:>8.6f}, {metrics['gating_alpha_max']:>8.6f}]")
            logger.info(f"   Effective TTT weight: {metrics.get('effective_ttt_weight', 0):.6f}")

        # Cosine Similarity
        logger.info(f"")
        logger.info(f"📐 COSINE SIMILARITY:")
        logger.info(f"   Attn ↔ TTT:      {metrics['cos_attn_ttt']:>7.4f}")
        logger.info(f"   Attn ↔ Combined: {metrics['cos_attn_combined']:>7.4f}")
        logger.info(f"   TTT ↔ Combined:  {metrics['cos_ttt_combined']:>7.4f}")

        # Distribution Stats (compact)
        logger.info(f"")
        logger.info(f"📈 DISTRIBUTION STATS:")
        logger.info(f"   Attention:  μ={metrics['attn_mean']:>8.4f}, σ={metrics['attn_std']:>8.4f}, sparsity={metrics['attn_sparsity']:>5.1f}%")
        logger.info(f"   TTT:        μ={metrics['ttt_mean']:>8.4f}, σ={metrics['ttt_std']:>8.4f}, sparsity={metrics['ttt_sparsity']:>5.1f}%")
        logger.info(f"   Combined:   μ={metrics['combined_mean']:>8.4f}, σ={metrics['combined_std']:>8.4f}, sparsity={metrics['combined_sparsity']:>5.1f}%")

        # Delta Analysis
        if 'attn_delta' in metrics or 'ttt_delta' in metrics:
            logger.info(f"")
            logger.info(f"Δ CHANGE FROM PREVIOUS STEP:")
            if 'attn_delta' in metrics:
                logger.info(f"   Attention Δ: {metrics['attn_delta']:>12.6f}")
            if 'ttt_delta' in metrics:
                logger.info(f"   TTT Δ:       {metrics['ttt_delta']:>12.6f}  (relative: {metrics.get('ttt_relative_change', 0):.6f})")

        # TTT Internal Weights
        if 'ttt_weight_norm' in metrics:
            logger.info(f"")
            logger.info(f"⚙️  TTT INTERNAL WEIGHTS:")
            logger.info(f"   Weight norm: {metrics['ttt_weight_norm']:.6f}")

        logger.info(f"{'='*80}")
        logger.info(f"")

    def get_history_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics of tracked history."""
        if not self.track_history:
            return {}

        summary = {}
        for key, values in self.history.items():
            if len(values) > 0:
                summary[key] = {
                    'mean': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'last': values[-1],
                    'samples': len(values)
                }

        return summary

    def reset(self):
        """Reset step counter and clear cache."""
        self.step_count = 0
        self.prev_attn_output = None
        self.prev_ttt_output = None
        if self.track_history:
            for key in self.history:
                self.history[key].clear()
