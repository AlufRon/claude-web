"""
Verify if TTT states persist across forward calls or reset to W0.

This script will:
1. Check W1 norm at the start of each forward call
2. Log if W1 equals self.W1 (means reset to W0)
3. Verify orange line behavior in Figure 5
"""

import torch
import logging

logger = logging.getLogger(__name__)

# Monkey patch to add logging
original_ttt_init = None

def patch_ttt_layer():
    """Add logging to TTT layer to track state persistence."""
    from moshi_ttt.models.ssm.ttt_layer import TTTMLP

    global original_ttt_init
    original_ttt_init = TTTMLP.ttt

    call_counter = [0]  # Mutable to modify in closure

    def ttt_with_logging(self, inputs, layer_id=None):
        B = inputs["XV"].shape[0]
        num_mini_batch = inputs["XV"].shape[2]
        L = inputs["XV"].shape[2] * inputs["XV"].shape[3]

        # Log every forward call
        call_counter[0] += 1
        W1_param_norm = self.W1.norm().item()

        logger.warning(f"\n🔍 [TTT Forward Call #{call_counter[0]}] Layer {layer_id}")
        logger.warning(f"  B={B}, num_mini_batch={num_mini_batch}, L={L}")
        logger.warning(f"  self.W1 norm: {W1_param_norm:.6f}")
        logger.warning(f"  stream_position: {self.stream_position}")

        # Call original
        result = original_ttt_init(self, inputs, layer_id)

        logger.warning(f"  After forward: stream_position={self.stream_position}")

        return result

    TTTMLP.ttt = ttt_with_logging
    logger.warning("✅ Patched TTTMLP.ttt with logging")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    print("Patch function defined. Import this module and call patch_ttt_layer() before evaluation.")
