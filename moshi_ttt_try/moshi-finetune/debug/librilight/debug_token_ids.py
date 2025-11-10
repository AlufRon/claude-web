"""
Debug script to check token IDs in the system.
"""
import torch
from moshi.models import loaders

# Load model
print("Loading Moshi...")
checkpoint_info = loaders.CheckpointInfo.from_hf_repo(hf_repo="kyutai/moshiko-pytorch-bf16")
mimi = checkpoint_info.get_mimi(device="cuda")
moshi = checkpoint_info.get_lm(device="cuda")
moshi = moshi.eval()

# Check token IDs
print("\n" + "=" * 80)
print("TOKEN ID VALUES")
print("=" * 80)
print(f"model.zero_token_id: {moshi.zero_token_id}")
print(f"model.text_padding_token_id: {moshi.text_padding_token_id}")
print(f"model.end_of_text_padding_id: {moshi.end_of_text_padding_id}")

# Check what the model expects for masking
print("\n" + "=" * 80)
print("MODEL BEHAVIOR")
print("=" * 80)
print(f"The model masks positions where: codes != {moshi.zero_token_id}")
print(f"This means:")
print(f"  - Token {moshi.zero_token_id} is IGNORED (zero_token_id)")
print(f"  - Token 0 is NOT ignored (it's existing_text_end_padding_id)")
print(f"  - Token {moshi.text_padding_token_id} is text padding (weighted by text_padding_weight)")

# Test what happens with different token values
print("\n" + "=" * 80)
print("TESTING TOKEN MASKING")
print("=" * 80)

test_codes = torch.tensor([
    [[0, -1, 3, 5]],  # Text stream with different tokens
], device="cuda", dtype=torch.long)

print(f"Test codes: {test_codes[0, 0]}")
print(f"Positions != zero_token_id ({moshi.zero_token_id}): {test_codes[0, 0] != moshi.zero_token_id}")
print(f"  Position 0 (token=0): {'INCLUDED' if test_codes[0, 0, 0] != moshi.zero_token_id else 'MASKED'}")
print(f"  Position 1 (token=-1): {'INCLUDED' if test_codes[0, 0, 1] != moshi.zero_token_id else 'MASKED'}")
print(f"  Position 2 (token=3): {'INCLUDED' if test_codes[0, 0, 2] != moshi.zero_token_id else 'MASKED'}")
print(f"  Position 3 (token=5): {'INCLUDED' if test_codes[0, 0, 3] != moshi.zero_token_id else 'MASKED'}")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
print(f"✅ For LibriLight (audio-only), text stream should use: {moshi.zero_token_id} (zero_token_id)")
print(f"❌ Currently LibriLightInterleaver might be using: 0 (which is NOT masked!)")
print("=" * 80)
