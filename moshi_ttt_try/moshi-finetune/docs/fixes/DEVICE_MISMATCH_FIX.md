# Device Mismatch Fix: RoPE freqs_cis on Wrong Device

## 🔴 Issue: CPU/CUDA Device Mismatch

**Job 7114013** crashed with:
```
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
```

At: `Q_rope_complex = Q_complex * freqs_cis` in `apply_audio_rotary_emb()`

## 🔍 Root Cause

1. **Input data is on CUDA**: `x` tensor (BFloat16) is on `cuda:0`
2. **freqs_cis computed on CPU**: `precompute_audio_rope_1d()` uses `torch.arange()` and `torch.polar()` which default to CPU
3. **No device transfer**: Code didn't move `freqs_cis` to match input device
4. **Multiplication fails**: Can't multiply CUDA tensor with CPU tensor

## ✅ The Fix

**File: `moshi_ttt/models/ssm/ttt_layer.py`** (Line 73-77)

**BEFORE:**
```python
def forward(self, x: torch.Tensor, seq_metadata: SequenceMetadata):
    # Compute RoPE frequencies dynamically based on actual sequence length
    seq_len = x.shape[1]  # B, L, D
    freqs_cis = self._precompute_audio_rope_1d(seq_len)
    return self.ttt(x, freqs_cis, seq_metadata)
```

**AFTER:**
```python
def forward(self, x: torch.Tensor, seq_metadata: SequenceMetadata):
    # Compute RoPE frequencies dynamically based on actual sequence length
    seq_len = x.shape[1]  # B, L, D
    freqs_cis = self._precompute_audio_rope_1d(seq_len)
    # Move freqs_cis to same device as input
    freqs_cis = freqs_cis.to(x.device)
    return self.ttt(x, freqs_cis, seq_metadata)
```

## 📊 Why This Happened

- `precompute_audio_rope_1d()` creates tensors without specifying device
- PyTorch defaults to CPU when no device specified
- Video-DiT likely ran on CPU or had different device management
- Moshi runs on CUDA, exposing the device mismatch

## ✅ Expected Next Run

- ✅ freqs_cis moved to CUDA automatically
- ✅ BFloat16 conversion working (from previous fix)
- ✅ Forward pass completes
- ✅ Training proceeds

This is a simple one-line fix with zero overhead (device transfer is cheap for small RoPE tensors).
