"""
Checkpoint loading for resuming training or transfer learning.

This module provides functionality to:
- Load checkpoints from previous training runs
- Resume training from where it left off
- Transfer learning: load weights but reset optimizer/step
- Verify TTT configuration compatibility
"""

import json
import logging
from pathlib import Path
import torch
import safetensors.torch
from typing import Optional, Tuple

logger = logging.getLogger("checkpoint_loader")


class CheckpointLoader:
    """Load checkpoints for resuming or transfer learning."""

    @staticmethod
    def load_checkpoint(
        checkpoint_path: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        load_optimizer: bool = True,
        strict: bool = False
    ) -> Tuple[int, dict]:
        """
        Load checkpoint and restore model/optimizer state.

        Args:
            checkpoint_path: Path to checkpoint directory (consolidated/)
            model: Model to load weights into
            optimizer: Optimizer to load state into (optional)
            load_optimizer: Whether to load optimizer state
            strict: Whether to require exact key matching

        Returns:
            (step, training_config): Step number and training config dict

        Raises:
            FileNotFoundError: If checkpoint or required files not found
            RuntimeError: If strict=True and keys don't match
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"=" * 80)
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        logger.info(f"=" * 80)

        # 1. Load training config
        training_config_path = checkpoint_path / "training_config.json"
        if not training_config_path.exists():
            raise FileNotFoundError(
                f"training_config.json not found in {checkpoint_path}"
            )

        with open(training_config_path, 'r') as f:
            training_config = json.load(f)

        logger.info(f"✅ Loaded training config from {training_config_path.name}")

        # 2. Determine which weights file to load
        lora_path = checkpoint_path / "lora.safetensors"
        consolidated_path = checkpoint_path / "consolidated.safetensors"

        if lora_path.exists():
            weights_path = lora_path
            logger.info(f"📥 Loading LoRA/TTT weights from {lora_path.name}")
        elif consolidated_path.exists():
            weights_path = consolidated_path
            logger.info(f"📥 Loading consolidated weights from {consolidated_path.name}")
        else:
            raise FileNotFoundError(
                f"No weights file found in {checkpoint_path}. "
                f"Expected either lora.safetensors or consolidated.safetensors"
            )

        # 3. Load model weights
        state_dict = safetensors.torch.load_file(str(weights_path))
        missing, unexpected = model.load_state_dict(state_dict, strict=strict)

        if missing and strict:
            raise RuntimeError(f"Missing keys in checkpoint: {missing}")
        elif missing:
            logger.warning(
                f"⚠️  Missing keys (expected for partial loading): {len(missing)} keys"
            )
            logger.debug(f"   Missing keys: {missing[:5]}...")  # Show first 5

        if unexpected:
            logger.error(f"❌ Unexpected keys: {unexpected}")
            if strict:
                raise RuntimeError(f"Unexpected keys in checkpoint: {unexpected}")

        logger.info(f"✅ Loaded {len(state_dict)} parameters from checkpoint")

        # 4. Load optimizer state (optional)
        step = 0
        if load_optimizer and optimizer is not None:
            optimizer_path = checkpoint_path / "optimizer_state.pt"
            if optimizer_path.exists():
                try:
                    optimizer_state = torch.load(optimizer_path, weights_only=False)
                    optimizer.load_state_dict(optimizer_state['optimizer'])
                    step = optimizer_state.get('step', 0)
                    logger.info(f"✅ Loaded optimizer state from step {step}")
                except Exception as e:
                    logger.warning(f"⚠️  Failed to load optimizer state: {e}")
                    logger.warning(f"   Starting with fresh optimizer")
            else:
                logger.warning(
                    "⚠️  No optimizer state found (optimizer_state.pt missing)"
                )
                logger.warning("   Starting with fresh optimizer")

        logger.info(f"=" * 80)
        return step, training_config

    @staticmethod
    def verify_ttt_config(checkpoint_config: dict, new_config: dict) -> None:
        """
        Verify TTT configuration matches between checkpoint and new config.

        Warns if configurations differ, as this might indicate:
        - Intentional change (e.g., fine-tuning with different settings)
        - Accidental mismatch (might want to fix)

        Args:
            checkpoint_config: Training config from checkpoint
            new_config: Current training config
        """
        checkpoint_ttt = checkpoint_config.get('ttt', {})
        new_ttt = new_config.get('ttt', {})

        # Important keys that affect training behavior
        important_keys = [
            'layers',
            'base_lr',
            'mini_batch_size',
            'ttt_mlp_layers',
            'ttt_mlp_expansion_factor'
        ]

        mismatches = []
        for key in important_keys:
            ckpt_val = checkpoint_ttt.get(key)
            new_val = new_ttt.get(key)
            if ckpt_val != new_val:
                mismatches.append(
                    f"  {key}: checkpoint={ckpt_val}, new={new_val}"
                )

        if mismatches:
            logger.warning("⚠️  TTT configuration mismatch detected:")
            for mismatch in mismatches:
                logger.warning(mismatch)
            logger.warning(
                "   Continuing with NEW config values (checkpoint config ignored)"
            )
            logger.warning(
                "   This is expected for transfer learning with different settings"
            )
        else:
            logger.info("✅ TTT configuration matches checkpoint")
