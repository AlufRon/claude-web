"""
TTT Monitoring and FP32 Enforcement Utilities

Tools for monitoring TTT training and ensuring numerical stability.

Key features:
- FP32 precision verification and enforcement
- TTT state monitoring (W1/b1/W2/b2 statistics)
- Numerical health checks (NaN, explosion, gibberish detection)
- Training metrics logging

Based on Doc 13 monitoring infrastructure specifications.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import csv
import json
from pathlib import Path
import numpy as np
import logging

from ttt_modules.ttt_layer import TTTMLP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FP32Enforcer:
    """
    Enforces FP32 precision for TTT parameters during training.

    CRITICAL: TTT states (W1, b1, W2, b2) MUST be torch.float32.
    BF16/FP16 causes numerical instability leading to gibberish after ~3750 updates.

    Usage:
        enforcer = FP32Enforcer(model, ttt_layers=[24-31])
        # Before each forward pass:
        enforcer.verify_and_fix()
    """

    def __init__(
        self,
        model: nn.Module,
        ttt_layers: List[int],
        auto_fix: bool = True,
        raise_on_violation: bool = False,
    ):
        """
        Args:
            model: Model with TTT layers
            ttt_layers: List of TTT layer indices
            auto_fix: Automatically convert back to FP32 if violated
            raise_on_violation: Raise error if precision violated
        """
        self.model = model
        self.ttt_layers = ttt_layers
        self.auto_fix = auto_fix
        self.raise_on_violation = raise_on_violation

        # Get layers
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            self.layers = model.model.layers
        else:
            self.layers = model.layers

        self.violation_count = 0

    def verify_and_fix(self) -> bool:
        """
        Verify all TTT parameters are FP32, optionally fix.

        Returns:
            True if all parameters are FP32, False if violations found
        """

        all_good = True

        for layer_idx in self.ttt_layers:
            layer = self.layers[layer_idx]

            if not isinstance(layer.self_attn, TTTMLP):
                continue

            ttt_layer = layer.self_attn

            # Check critical parameters
            params = {
                'W1': ttt_layer.W1,
                'b1': ttt_layer.b1,
                'W2': ttt_layer.W2,
                'b2': ttt_layer.b2,
                'ttt_norm_weight': ttt_layer.ttt_norm_weight,
                'ttt_norm_bias': ttt_layer.ttt_norm_bias,
            }

            for param_name, param in params.items():
                if param.dtype != torch.float32:
                    all_good = False
                    self.violation_count += 1

                    msg = (
                        f"❌ Layer {layer_idx} {param_name}: "
                        f"{param.dtype} (expected torch.float32)"
                    )

                    if self.raise_on_violation:
                        raise TypeError(msg)

                    logger.error(msg)

                    if self.auto_fix:
                        # Convert back to FP32
                        param.data = param.data.float()
                        logger.info(f"   Auto-fixed: converted {param_name} back to FP32")

        if not all_good:
            logger.warning(
                f"FP32 violations detected! Total violations so far: {self.violation_count}"
            )

        return all_good

    def setup_hooks(self):
        """Setup forward hooks to automatically verify FP32 before each forward pass."""

        def fp32_check_hook(module, input, output):
            if not isinstance(module, TTTMLP):
                return

            # Quick check
            if module.W1.dtype != torch.float32:
                logger.error(f"TTT layer {module.layer_idx} has non-FP32 states!")
                if self.auto_fix:
                    module.W1.data = module.W1.data.float()
                    module.b1.data = module.b1.data.float()
                    module.W2.data = module.W2.data.float()
                    module.b2.data = module.b2.data.float()
                    logger.info("  Auto-fixed to FP32")

        for layer_idx in self.ttt_layers:
            layer = self.layers[layer_idx]
            if isinstance(layer.self_attn, TTTMLP):
                layer.self_attn.register_forward_pre_hook(fp32_check_hook)

        logger.info(f"FP32 enforcement hooks registered for {len(self.ttt_layers)} layers")


class TTTMonitor:
    """
    Monitor TTT state evolution during training.

    Tracks:
    - W1, b1, W2, b2 statistics (mean, std, max, min)
    - Numerical health (NaN, explosion, underflow)
    - State update magnitudes
    - Training metrics

    Critical for debugging TTT training issues.
    """

    def __init__(
        self,
        log_dir: str = "logs/ttt_monitoring",
        log_every_n_steps: int = 10,
    ):
        """
        Args:
            log_dir: Directory for log files
            log_every_n_steps: Log interval (steps)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_every_n_steps = log_every_n_steps

        # CSV logging
        self.csv_path = self.log_dir / "ttt_stats.csv"
        self.csv_file = open(self.csv_path, "w", newline='')
        self.csv_writer = csv.writer(self.csv_file)

        # CSV header
        self.csv_writer.writerow([
            'step', 'layer_idx',
            'W1_mean', 'W1_std', 'W1_max', 'W1_min',
            'b1_mean', 'b1_std', 'b1_max', 'b1_min',
            'W2_mean', 'W2_std', 'W2_max', 'W2_min',
            'b2_mean', 'b2_std', 'b2_max', 'b2_min',
            'has_nan', 'is_exploding', 'is_underflow',
            'dtype_w1', 'dtype_b1'
        ])

        self.step = 0
        self.stats_history = []

        logger.info(f"TTT Monitor initialized. Logging to {self.log_dir}")

    def log_layer_state(
        self,
        layer_idx: int,
        W1: torch.Tensor,
        b1: torch.Tensor,
        W2: torch.Tensor,
        b2: torch.Tensor,
        step: Optional[int] = None,
    ):
        """
        Log statistics for one TTT layer.

        Args:
            layer_idx: Layer index
            W1, b1, W2, b2: TTT state tensors
            step: Training step (uses self.step if None)
        """

        if step is None:
            step = self.step

        # Compute statistics
        W1_mean = W1.mean().item()
        W1_std = W1.std().item()
        W1_max = W1.abs().max().item()
        W1_min = W1.min().item()

        b1_mean = b1.mean().item()
        b1_std = b1.std().item()
        b1_max = b1.abs().max().item()
        b1_min = b1.min().item()

        W2_mean = W2.mean().item()
        W2_std = W2.std().item()
        W2_max = W2.abs().max().item()
        W2_min = W2.min().item()

        b2_mean = b2.mean().item()
        b2_std = b2.std().item()
        b2_max = b2.abs().max().item()
        b2_min = b2.min().item()

        # Numerical health checks
        has_nan = torch.isnan(W1).any() or torch.isnan(b1).any() or \
                  torch.isnan(W2).any() or torch.isnan(b2).any()

        is_exploding = W1_max > 1e4 or b1_max > 1e4 or W2_max > 1e4 or b2_max > 1e4
        is_underflow = W1_max < 1e-6 and b1_max < 1e-6

        # Write to CSV
        self.csv_writer.writerow([
            step, layer_idx,
            W1_mean, W1_std, W1_max, W1_min,
            b1_mean, b1_std, b1_max, b1_min,
            W2_mean, W2_std, W2_max, W2_min,
            b2_mean, b2_std, b2_max, b2_min,
            int(has_nan), int(is_exploding), int(is_underflow),
            str(W1.dtype), str(b1.dtype)
        ])
        self.csv_file.flush()

        # Store in history
        self.stats_history.append({
            'step': step,
            'layer_idx': layer_idx,
            'W1': {'mean': W1_mean, 'std': W1_std, 'max': W1_max},
            'b1': {'mean': b1_mean, 'std': b1_std, 'max': b1_max},
            'has_nan': has_nan,
            'is_exploding': is_exploding,
        })

        # Alert on issues
        if has_nan:
            logger.error(f"🚨 NaN detected in layer {layer_idx} at step {step}!")

        if is_exploding:
            logger.warning(f"⚠️  Explosion detected in layer {layer_idx} at step {step}")

    def log_model_states(
        self,
        model: nn.Module,
        ttt_layers: List[int],
        step: Optional[int] = None,
    ):
        """
        Log all TTT layer states in the model.

        Args:
            model: Model with TTT layers
            ttt_layers: List of TTT layer indices
            step: Training step
        """

        if step is None:
            step = self.step

        # Skip if not logging this step
        if step % self.log_every_n_steps != 0:
            return

        # Get layers
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = model.model.layers
        else:
            layers = model.layers

        # Log each TTT layer
        for layer_idx in ttt_layers:
            layer = layers[layer_idx]

            if not isinstance(layer.self_attn, TTTMLP):
                continue

            ttt_layer = layer.self_attn

            self.log_layer_state(
                layer_idx=layer_idx,
                W1=ttt_layer.W1,
                b1=ttt_layer.b1,
                W2=ttt_layer.W2,
                b2=ttt_layer.b2,
                step=step,
            )

        self.step = step + 1

    def save_summary(self):
        """Save summary statistics to JSON."""
        if not self.stats_history:
            return

        summary = {
            'total_steps': len(self.stats_history),
            'nan_occurrences': sum(1 for s in self.stats_history if s['has_nan']),
            'explosion_occurrences': sum(1 for s in self.stats_history if s['is_exploding']),
            'final_stats': {
                layer_idx: {
                    'W1_mean': [s['W1']['mean'] for s in self.stats_history if s['layer_idx'] == layer_idx][-1],
                    'W1_std': [s['W1']['std'] for s in self.stats_history if s['layer_idx'] == layer_idx][-1],
                }
                for layer_idx in set(s['layer_idx'] for s in self.stats_history)
            }
        }

        summary_path = self.log_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Summary saved to {summary_path}")

    def close(self):
        """Close CSV file and save summary."""
        self.save_summary()
        self.csv_file.close()
        logger.info(f"TTT Monitor closed. Logs saved to {self.log_dir}")


def check_numerical_health(
    tensor: torch.Tensor,
    name: str = "tensor",
    raise_on_nan: bool = True,
    raise_on_inf: bool = True,
) -> Dict[str, bool]:
    """
    Check numerical health of a tensor.

    Args:
        tensor: Tensor to check
        name: Name for logging
        raise_on_nan: Raise error if NaN detected
        raise_on_inf: Raise error if Inf detected

    Returns:
        Dictionary with health checks: has_nan, has_inf, is_exploding

    Raises:
        ValueError: If NaN/Inf detected and raise_on_* is True
    """

    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    is_exploding = tensor.abs().max().item() > 1e4

    if has_nan:
        msg = f"NaN detected in {name}!"
        if raise_on_nan:
            raise ValueError(msg)
        logger.error(msg)

    if has_inf:
        msg = f"Inf detected in {name}!"
        if raise_on_inf:
            raise ValueError(msg)
        logger.error(msg)

    if is_exploding:
        logger.warning(f"{name} is exploding (max value: {tensor.abs().max().item():.2e})")

    return {
        'has_nan': has_nan,
        'has_inf': has_inf,
        'is_exploding': is_exploding,
    }


if __name__ == "__main__":
    """Test monitoring utilities."""
    print("Testing TTT Monitoring Utilities")
    print("=" * 60)

    # Test monitor
    monitor = TTTMonitor(log_dir="test_logs", log_every_n_steps=1)

    # Simulate training steps
    for step in range(5):
        # Create mock TTT states
        W1 = torch.randn(32, 128, 512, dtype=torch.float32) * 0.1
        b1 = torch.zeros(32, 1, 512, dtype=torch.float32)
        W2 = torch.randn(32, 512, 128, dtype=torch.float32) * 0.1
        b2 = torch.zeros(32, 1, 128, dtype=torch.float32)

        # Log states
        monitor.log_layer_state(
            layer_idx=24,
            W1=W1,
            b1=b1,
            W2=W2,
            b2=b2,
            step=step,
        )

        print(f"Step {step}: Logged layer 24 states")

    # Close monitor
    monitor.close()

    print("\n✅ Monitoring test complete!")
    print(f"Logs saved to {monitor.log_dir}")

    # Test numerical health check
    print("\nTesting numerical health check:")

    # Normal tensor
    normal_tensor = torch.randn(100)
    health = check_numerical_health(normal_tensor, "normal_tensor", raise_on_nan=False)
    print(f"  Normal tensor: {health}")

    # Tensor with NaN
    nan_tensor = torch.tensor([1.0, 2.0, float('nan')])
    health = check_numerical_health(nan_tensor, "nan_tensor", raise_on_nan=False)
    print(f"  NaN tensor: {health}")

    print("\n✅ All tests complete!")
