# Selective Attention Layer Unfreezing

## Overview

This feature allows you to selectively unfreeze attention layers in Moshi when training with TTT. This addresses the magnitude imbalance problem where frozen attention outputs (~1.6 magnitude) compete with trainable TTT outputs (~63.5 magnitude).

## Problem

When training with TTT on frozen Moshi:
- **Frozen attention**: Cannot adapt, outputs stay at ~1.6 magnitude
- **Trainable TTT**: Grows to ~63.5 magnitude to compensate
- **Result**: TTT dominates (95% contribution via gating α≈0.05)

This differs from Video-DiT which:
- Uses higher initial gating α (0.1 vs 0.005)
- Unfreezes attention layers during TTT training
- Allows attention to adapt alongside TTT

## Solution

Unfreeze attention layers where TTT is applied, allowing them to co-adapt during training.

## Usage

### Configuration

Add `unfrozen_attention_layers` to your TTT config:

```yaml
ttt:
  enable: true
  layers: "1,3,5,7,9"  # Which layers get TTT
  unfrozen_attention_layers: "1,3,5,7,9"  # Which layers have unfrozen attention
  base_lr: 0.8
  initial_gating_alpha: 0.1  # Consider using Video-DiT's value
```

### Layer Specification Syntax

Same as `ttt.layers`:

- `"none"` - Don't unfreeze any attention layers (default, current behavior)
- `"all"` - Unfreeze attention in all layers
- `"middle"` - Unfreeze attention in middle 50% of layers
- `"1,3,5,7,9"` - Unfreeze attention in specific layers (comma-separated indices)

### Examples

#### Conservative: Same layers as TTT
```yaml
ttt:
  enable: true
  layers: "1"
  unfrozen_attention_layers: "1"
```

#### Balanced: Middle layers
```yaml
ttt:
  enable: true
  layers: "middle"
  unfrozen_attention_layers: "middle"
```

#### Aggressive: All TTT layers
```yaml
ttt:
  enable: true
  layers: "1,3,5,7,9,11,13,15"
  unfrozen_attention_layers: "1,3,5,7,9,11,13,15"
```

#### Hybrid: More attention than TTT
```yaml
ttt:
  enable: true
  layers: "5"  # TTT only in layer 5
  unfrozen_attention_layers: "4,5,6"  # Unfreeze attention in neighboring layers too
```

## Training Modes

Works with both training modes:
- **ttt mode**: TTT params + selected attention layers trainable
- **lora+ttt mode**: LoRA + TTT params + selected attention layers trainable

## Parameter Counts

Example output during training:
```
🔓 Unfreezing attention in layers: [1, 3, 5, 7, 9]
🎯 Training mode: ttt
📊 Trainable parameters: 75,234,816 / 7,932,438,528 (0.95%)
   TTT parameters: 139,296
   Attention parameters: 75,095,520
```

## Learning Rate Considerations

Attention parameters will use the base optimizer learning rate. Consider:

1. **Separate learning rate groups** (future enhancement):
   - TTT params: Higher LR (e.g., 10x base)
   - Attention params: Lower LR (e.g., 0.1x base, matching Video-DiT)
   - Currently, all use base LR

2. **Adjust base LR**:
   - Lower base LR if using many unfrozen attention layers
   - Higher base LR if only unfreezing a few layers

## Implementation Details

### Files Modified

1. **`finetune/args.py`**: Added `unfrozen_attention_layers` field to `TTTArgs`
2. **`finetune/wrapped_model.py`**:
   - Added `is_attention_parameter_in_layer()` helper
   - Modified `configure_trainable_parameters()` to handle attention unfreezing
   - Added parameter counting for attention params

### Parameter Detection

Attention parameters are identified by pattern:
```python
f"transformer.layers.{layer_idx}.self_attn.*"
```

Examples:
- `transformer.layers.1.self_attn.in_proj_weight`
- `transformer.layers.1.self_attn.in_proj_bias`
- `transformer.layers.1.self_attn.out_proj.weight`
- `transformer.layers.1.self_attn.out_proj.bias`

### Backward Compatibility

Default value is `"none"`, maintaining current behavior. Existing configs continue to work without changes.

## Testing

Run tests:
```bash
conda activate moshi_ttt_fixed
cd /home/alufr/ttt_tests/moshi-finetune
python tests/test_selective_attention_unfreezing.py
```

## Recommendations

Based on Video-DiT paper and diagnostic analysis:

1. **Start conservative**: Unfreeze attention only in TTT layers
   ```yaml
   layers: "1,5,10"
   unfrozen_attention_layers: "1,5,10"
   ```

2. **Use Video-DiT's gating alpha**: 0.1 instead of 0.005
   ```yaml
   initial_gating_alpha: 0.1
   ```

3. **Monitor magnitude balance**: Enable TTT diagnostics to verify attention and TTT have similar magnitudes
   ```bash
   --enable-ttt-diagnostics --diagnostic-log-frequency 100
   ```

4. **Adjust learning rates**: Consider lowering base LR when unfreezing many layers

## Expected Behavior

With unfrozen attention:
- **Early training**: Attention adapts to work with TTT
- **Mid training**: Magnitudes become balanced (both ~10-30 range)
- **Late training**: Gating α stabilizes at healthier value (~0.3-0.5)
- **Result**: More balanced contribution from both mechanisms

Without unfrozen attention (current):
- TTT dominates with 95% contribution
- Attention contribution limited to 5%
- Works but may not fully utilize pretrained knowledge

## Future Enhancements

1. **Separate learning rate groups**: Different LRs for attention vs TTT
2. **Layer normalization unfreezing**: Allow unfreezing layer norms in addition to attention
3. **Automatic layer selection**: Heuristics to choose which layers to unfreeze
4. **Per-layer gating alpha**: Different initial α for each layer

## References

- Video-DiT paper: Uses α_init=0.1 and unfreezes attention during training
- TTT diagnostic analysis: Shows 40x magnitude imbalance with frozen attention
- See: `/home/alufr/ttt_tests/diagrams/comparisons/15_ttt_normalization_problem_and_solution.md`
