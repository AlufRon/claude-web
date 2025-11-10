"""
Debug script to check LibriLight token generation
"""
import torch
import numpy as np
from moshi.models import loaders
from finetune.data.librilight_interleaver import LibriLightInterleaver, LibriLightTokenizer
import soundfile as sf

# Load Mimi
print("Loading Mimi...")
mimi, moshi = loaders.get_mimi_moshi(hf_repo_id="kyutai/moshiko-pytorch-bf16")
mimi = mimi.cuda().eval()

# Create LibriLight interleaver and tokenizer
interleaver = LibriLightInterleaver(zero_padding=0)
tokenizer = LibriLightTokenizer(mimi, interleaver, duration_sec=5.0)

# Load a test audio file
test_file = "/sise/eliyanac-group/ron_al/librilight/extracted_medium/medium/3681/celebrated_crimes_1_1001_librivox_64kb_mp3/celebratedcrimes_1-25_dumas_64kb.flac"
print(f"\nLoading audio: {test_file}")
wav, sr = sf.read(test_file)

# If stereo, take first channel
if len(wav.shape) > 1:
    wav = wav[:, 0]

# Convert to mono format [1, samples]
wav = wav[np.newaxis, :]
print(f"Audio shape: {wav.shape}, sr: {sr}")

# Process through tokenizer
print("\nProcessing through LibriLight tokenizer...")
sample = tokenizer(wav, start_sec=0.0, path=test_file)

print(f"\nOutput codes shape: {sample.codes.shape}")
print(f"Text stream (first row): min={sample.codes[0, 0, :].min().item()}, max={sample.codes[0, 0, :].max().item()}, unique={torch.unique(sample.codes[0, 0, :]).cpu().numpy()}")
print(f"Audio streams (remaining rows): min={sample.codes[0, 1:, :].min().item()}, max={sample.codes[0, 1:, :].max().item()}")

# Check for NaN or inf
print(f"\nChecking for invalid values:")
print(f"NaN in text stream: {torch.isnan(sample.codes[0, 0, :].float()).any().item()}")
print(f"NaN in audio streams: {torch.isnan(sample.codes[0, 1:, :].float()).any().item()}")
print(f"Inf in text stream: {torch.isinf(sample.codes[0, 0, :].float()).any().item()}")
print(f"Inf in audio streams: {torch.isinf(sample.codes[0, 1:, :].float()).any().item()}")

# Print first few tokens
print(f"\nFirst 10 text tokens: {sample.codes[0, 0, :10].cpu().numpy()}")
print(f"First 10 audio tokens (codebook 0, left): {sample.codes[0, 1, :10].cpu().numpy()}")
print(f"First 10 audio tokens (codebook 0, right): {sample.codes[0, 9, :10].cpu().numpy()}")

print("\n✅ Debug complete!")
