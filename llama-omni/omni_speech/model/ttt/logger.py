"""
TTT Logging and CSV State Tracking

This module provides comprehensive logging for TTT operations:
1. Standard Python logging (console/file)
2. CSV files for state evolution tracking (for plotting)
3. Periodic statistics summaries
4. Memory-efficient logging (doesn't fill disk in seconds)

The CSV tracking allows us to create organized plots showing:
- How TTT states evolve over time
- Gradient norms per layer
- Loss values
- State statistics (mean, std, max)
- Update frequencies

Design principles:
- Log only when meaningful changes occur
- Sample periodically (not every step)
- Rotate log files when they get large
- Flush buffers periodically
"""

import csv
import logging
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, List
import torch


@dataclass
class TTTStateLog:
    """Single row in TTT state CSV log."""
    timestamp: float
    layer_idx: int
    step: int
    conversation_id: str

    # W1 statistics
    w1_mean: float
    w1_std: float
    w1_max: float
    w1_min: float

    # b1 statistics
    b1_mean: float
    b1_std: float
    b1_max: float
    b1_min: float

    # Gradient statistics (if available)
    grad_norm: Optional[float] = None
    loss: Optional[float] = None

    # Update information
    learning_rate: Optional[float] = None
    update_magnitude: Optional[float] = None


class TTTCSVLogger:
    """
    CSV logger for TTT state tracking.

    Creates a CSV file with columns for all TTT state statistics.
    Logs are written periodically to avoid overwhelming disk I/O.

    Usage:
        logger = TTTCSVLogger(log_path="ttt_states.csv", flush_interval=100)

        # During forward pass
        logger.log_state(
            layer_idx=0,
            step=42,
            conversation_id="conv_123",
            W1=W1_tensor,
            b1=b1_tensor,
            grad_norm=0.123,
            loss=0.456
        )

        # Periodically or at end
        logger.flush()
    """

    def __init__(
        self,
        log_path: str,
        flush_interval: int = 100,
        max_rows_per_file: int = 100000
    ):
        """
        Initialize CSV logger.

        Args:
            log_path: Path to CSV file
            flush_interval: Write to disk every N logs
            max_rows_per_file: Start new file after this many rows
        """
        self.log_path = Path(log_path)
        self.flush_interval = flush_interval
        self.max_rows_per_file = max_rows_per_file

        # Create directory if needed
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        # Buffer for pending logs
        self.buffer: List[TTTStateLog] = []
        self.total_rows_written = 0
        self.file_index = 0

        # Initialize CSV file
        self._init_csv()

        logging.info(f"[TTTCSVLogger] Initialized with log_path={self.log_path}")

    def _get_current_path(self) -> Path:
        """Get current CSV file path (with rotation index if needed)."""
        if self.file_index == 0:
            return self.log_path
        else:
            stem = self.log_path.stem
            suffix = self.log_path.suffix
            return self.log_path.parent / f"{stem}_part{self.file_index}{suffix}"

    def _init_csv(self):
        """Initialize CSV file with header."""
        current_path = self._get_current_path()

        # Check if file exists
        file_exists = current_path.exists()

        if not file_exists:
            # Write header
            with open(current_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(TTTStateLog.__annotations__.keys()))
                writer.writeheader()

            logging.info(f"[TTTCSVLogger] Created new CSV file: {current_path}")

    def log_state(
        self,
        layer_idx: int,
        step: int,
        conversation_id: str,
        W1: torch.Tensor,
        b1: torch.Tensor,
        grad_norm: Optional[float] = None,
        loss: Optional[float] = None,
        learning_rate: Optional[float] = None,
        update_magnitude: Optional[float] = None
    ):
        """
        Log TTT state for one layer at one step.

        Args:
            layer_idx: Which TTT layer
            step: Current step/token position
            conversation_id: Current conversation ID
            W1: Weight tensor [nh, f, f] or [B, nh, f, f]
            b1: Bias tensor [nh, 1, f] or [B, nh, 1, f]
            grad_norm: Gradient norm (if available)
            loss: Reconstruction loss (if available)
            learning_rate: Current learning rate
            update_magnitude: Magnitude of parameter update
        """
        # Compute statistics
        # If batched, take first element
        if W1.dim() == 4:  # [B, nh, f, f]
            W1 = W1[0]  # [nh, f, f]
        if b1.dim() == 4:  # [B, nh, 1, f]
            b1 = b1[0]  # [nh, 1, f]

        w1_stats = {
            "w1_mean": W1.mean().item(),
            "w1_std": W1.std().item(),
            "w1_max": W1.max().item(),
            "w1_min": W1.min().item(),
        }

        b1_stats = {
            "b1_mean": b1.mean().item(),
            "b1_std": b1.std().item(),
            "b1_max": b1.max().item(),
            "b1_min": b1.min().item(),
        }

        # Create log entry
        log_entry = TTTStateLog(
            timestamp=time.time(),
            layer_idx=layer_idx,
            step=step,
            conversation_id=conversation_id,
            **w1_stats,
            **b1_stats,
            grad_norm=grad_norm,
            loss=loss,
            learning_rate=learning_rate,
            update_magnitude=update_magnitude
        )

        # Add to buffer
        self.buffer.append(log_entry)

        # Flush if buffer is full
        if len(self.buffer) >= self.flush_interval:
            self.flush()

    def flush(self):
        """Write buffered logs to disk."""
        if not self.buffer:
            return

        current_path = self._get_current_path()

        # Check if we need to rotate to new file
        if self.total_rows_written >= self.max_rows_per_file:
            self.file_index += 1
            self.total_rows_written = 0
            self._init_csv()
            current_path = self._get_current_path()
            logging.info(f"[TTTCSVLogger] Rotated to new file: {current_path}")

        # Write buffered logs
        with open(current_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(TTTStateLog.__annotations__.keys()))

            for log_entry in self.buffer:
                writer.writerow(asdict(log_entry))

        self.total_rows_written += len(self.buffer)

        logging.debug(
            f"[TTTCSVLogger] Flushed {len(self.buffer)} logs to {current_path} "
            f"(total: {self.total_rows_written})"
        )

        # Clear buffer
        self.buffer.clear()

    def __del__(self):
        """Ensure logs are flushed on cleanup."""
        try:
            self.flush()
        except:
            pass


class TTTStatsTracker:
    """
    Track aggregate statistics across all TTT layers.

    Provides periodic summaries without logging every single step.
    Useful for monitoring training progress without overwhelming logs.
    """

    def __init__(self, num_layers: int, summary_interval: int = 1000):
        """
        Initialize stats tracker.

        Args:
            num_layers: Number of TTT layers
            summary_interval: Print summary every N steps
        """
        self.num_layers = num_layers
        self.summary_interval = summary_interval

        # Per-layer statistics
        self.layer_stats = {
            layer_idx: {
                "total_steps": 0,
                "total_loss": 0.0,
                "total_grad_norm": 0.0,
                "w1_magnitude_sum": 0.0,
                "b1_magnitude_sum": 0.0,
            }
            for layer_idx in range(num_layers)
        }

        # Global statistics
        self.global_step = 0
        self.last_summary_step = 0

    def update(
        self,
        layer_idx: int,
        W1: torch.Tensor,
        b1: torch.Tensor,
        loss: Optional[float] = None,
        grad_norm: Optional[float] = None
    ):
        """Update statistics for one layer."""
        stats = self.layer_stats[layer_idx]

        stats["total_steps"] += 1

        if loss is not None:
            stats["total_loss"] += loss

        if grad_norm is not None:
            stats["total_grad_norm"] += grad_norm

        # Compute parameter magnitudes
        w1_mag = W1.norm().item()
        b1_mag = b1.norm().item()

        stats["w1_magnitude_sum"] += w1_mag
        stats["b1_magnitude_sum"] += b1_mag

        self.global_step += 1

        # Check if time for summary
        if self.global_step - self.last_summary_step >= self.summary_interval:
            self.print_summary()
            self.last_summary_step = self.global_step

    def print_summary(self):
        """Print aggregate statistics summary."""
        logging.info("=" * 80)
        logging.info(f"[TTT Stats Summary] Global Step: {self.global_step}")
        logging.info("=" * 80)

        for layer_idx in range(self.num_layers):
            stats = self.layer_stats[layer_idx]

            if stats["total_steps"] == 0:
                continue

            avg_loss = stats["total_loss"] / stats["total_steps"] if stats["total_loss"] > 0 else 0.0
            avg_grad = stats["total_grad_norm"] / stats["total_steps"] if stats["total_grad_norm"] > 0 else 0.0
            avg_w1_mag = stats["w1_magnitude_sum"] / stats["total_steps"]
            avg_b1_mag = stats["b1_magnitude_sum"] / stats["total_steps"]

            logging.info(
                f"  Layer {layer_idx:2d}: "
                f"steps={stats['total_steps']:6d}, "
                f"loss={avg_loss:.6f}, "
                f"grad={avg_grad:.6f}, "
                f"|W1|={avg_w1_mag:.4f}, "
                f"|b1|={avg_b1_mag:.4f}"
            )

        logging.info("=" * 80)


def setup_ttt_logging(
    log_dir: str,
    log_level: str = "INFO",
    enable_csv: bool = True,
    csv_flush_interval: int = 100
) -> Tuple[logging.Logger, Optional[TTTCSVLogger]]:
    """
    Setup comprehensive TTT logging.

    Creates:
    1. Python logger for console/file output
    2. CSV logger for state tracking (if enabled)

    Args:
        log_dir: Directory for log files
        log_level: Logging level ("DEBUG", "INFO", "WARNING", "ERROR")
        enable_csv: Whether to enable CSV state tracking
        csv_flush_interval: How often to flush CSV logs

    Returns:
        Tuple of (logger, csv_logger)
        csv_logger is None if enable_csv=False
    """
    # Create log directory
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Setup Python logger
    logger = logging.getLogger("ttt")
    logger.setLevel(getattr(logging, log_level.upper()))

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_formatter = logging.Formatter(
        '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)

    # File handler
    log_file = log_dir / "ttt.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)  # Always log DEBUG to file
    file_handler.setFormatter(console_formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # CSV logger (if enabled)
    csv_logger = None
    if enable_csv:
        csv_path = log_dir / "ttt_states.csv"
        csv_logger = TTTCSVLogger(
            log_path=str(csv_path),
            flush_interval=csv_flush_interval
        )

    logger.info(f"TTT logging initialized: log_dir={log_dir}, level={log_level}, csv={enable_csv}")

    return logger, csv_logger


# Convenience function for quick logging
def log_ttt_update(
    csv_logger: Optional[TTTCSVLogger],
    layer_idx: int,
    step: int,
    conversation_id: str,
    W1: torch.Tensor,
    b1: torch.Tensor,
    **kwargs
):
    """
    Convenience function to log TTT update.

    Handles None csv_logger gracefully.
    """
    if csv_logger is not None:
        csv_logger.log_state(
            layer_idx=layer_idx,
            step=step,
            conversation_id=conversation_id,
            W1=W1,
            b1=b1,
            **kwargs
        )
