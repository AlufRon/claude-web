## TTT Gating Alpha Added to Training Logs

### Changes Made:

1. **Modified `finetune/monitoring/metrics_logger.py`:**
   - Added torch import
   - Modified `get_train_logs()` to accept optional `model` parameter
   - Added logic to extract TTT gating alpha from model when available
   - Updated `train_log_msg()` to include `ttt_alpha` in the log format
   - Added graceful handling when TTT is not enabled

2. **Modified `train.py`:**
   - Updated `get_train_logs()` call to pass the `model` parameter

3. **Modified `train_ttt.py`:**
   - Updated `get_train_logs()` call to pass the `model` parameter

### How it works:

The system automatically detects TTT layers in the model by looking for modules with `gating_alpha` parameters. When found, it:
- Extracts the gating alpha parameter
- Applies `torch.tanh()` (as per the TTT implementation)
- Takes the mean across all dimensions
- Logs it as `ttt_alpha` in the training output

### Example output:
```
2025-09-17 17:54:18 (IST) - 0:01:29 - train - INFO - step: 000001 - done (%): 0.1 - loss: 3.065 - lr: 8.0e-08 - peak_alloc_mem (GB): 24.8 - alloc_mem (GB): 19.8 - words_per_second: 178.9 - avg_words_per_second: 178.9 - ttt_alpha: 0.0500 - ETA: >2025-09-19 04:49:18
```

### Gating Alpha Details:
- **Initial value**: 0.05 (configured in `moshi_ttt/config.py`)
- **Range**: [-1, 1] (bounded by tanh activation)
- **Purpose**: Controls the strength of TTT/SSM gating in the model
- **Implementation**: Follows Video-DiT exact methodology

The value will appear in logs only when TTT is enabled, and will be gracefully omitted for non-TTT training runs.
