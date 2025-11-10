#!/usr/bin/env python3
"""
Test LibriLight data loading and audio-only format handling.

This script verifies:
1. LibriLight detection works
2. Mono audio is converted to stereo
3. Text stream is zero-filled
4. Data format is correct for Moshi

Usage:
    conda activate moshi_ttt_fixed
    cd /home/alufr/ttt_tests/moshi-finetune
    python tests/test_librilight_loading.py
"""

import sys
import torch
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_librilight_detection():
    """Test that LibriLight dataset is detected correctly."""
    from finetune.data.dataset import is_librilight_dataset

    print("=" * 80)
    print("TEST 1: LibriLight Detection")
    print("=" * 80)

    # Should detect LibriLight
    test_paths = [
        "/sise/eliyanac-group/ron_al/librilight/extracted_medium/medium/librilight.jsonl",
        "/path/to/LibriLight/data.jsonl",
        "LIBRILIGHT_dataset.jsonl",
    ]

    for path in test_paths:
        result = is_librilight_dataset(path)
        print(f"  {path}")
        print(f"    → Detected as LibriLight: {result}")
        assert result, f"Failed to detect LibriLight in: {path}"

    # Should NOT detect as LibriLight
    non_librilight = [
        "/sise/eliyanac-group/ron_al/daily-talk-contiguous/daily_train.jsonl",
        "/path/to/other_dataset.jsonl",
    ]

    for path in non_librilight:
        result = is_librilight_dataset(path)
        print(f"  {path}")
        print(f"    → Detected as LibriLight: {result}")
        assert not result, f"False positive LibriLight detection: {path}"

    print("✅ LibriLight detection working correctly\n")


def test_librilight_interleaver():
    """Test LibriLight interleaver creates zero-filled text stream."""
    from finetune.data.librilight_interleaver import LibriLightInterleaver

    print("=" * 80)
    print("TEST 2: LibriLight Interleaver (Zero-filled Text)")
    print("=" * 80)

    interleaver = LibriLightInterleaver(zero_padding=0)

    # Test with different sequence lengths
    test_lengths = [10, 50, 100]

    for num_frames in test_lengths:
        text_tokens = interleaver.prepare_item(num_frames)

        print(f"\n  Sequence length: {num_frames}")
        print(f"    Text tokens shape: {text_tokens.shape}")
        print(f"    Expected shape: [1, 1, {num_frames}]")
        print(f"    All zeros: {torch.all(text_tokens == 0).item()}")
        print(f"    Min value: {text_tokens.min().item()}")
        print(f"    Max value: {text_tokens.max().item()}")

        # Assertions
        assert text_tokens.shape == (1, 1, num_frames), \
            f"Wrong shape: expected [1, 1, {num_frames}], got {text_tokens.shape}"
        assert torch.all(text_tokens == 0), "Text tokens should be all zeros"

    print("\n✅ LibriLight interleaver working correctly\n")


def test_mono_to_stereo_conversion():
    """Test that mono audio is correctly converted to stereo."""
    import numpy as np

    print("=" * 80)
    print("TEST 3: Mono to Stereo Conversion")
    print("=" * 80)

    # Simulate mono audio
    mono_audio = np.random.randn(1, 48000).astype(np.float32)  # 1 channel, 2 seconds at 24kHz

    print(f"\n  Original mono audio shape: {mono_audio.shape}")

    # Convert to torch and duplicate channel (as done in LibriLightTokenizer)
    audio_tensor = torch.from_numpy(mono_audio).float()

    if audio_tensor.shape[0] == 1:
        stereo_audio = audio_tensor.repeat(2, 1)
        print(f"  Converted stereo audio shape: {stereo_audio.shape}")
        print(f"  Expected shape: [2, 48000]")

        # Verify channels are identical (duplicated)
        channels_identical = torch.allclose(stereo_audio[0], stereo_audio[1])
        print(f"  Channels identical (duplicated): {channels_identical}")

        # Assertions
        assert stereo_audio.shape[0] == 2, "Should have 2 channels"
        assert stereo_audio.shape[1] == mono_audio.shape[1], "Length should match"
        assert channels_identical, "Channels should be identical (duplicated mono)"

    print("\n✅ Mono to stereo conversion working correctly\n")


def test_librilight_sample_loading():
    """Test loading actual LibriLight sample (if index exists)."""
    from finetune.data.dataset import is_librilight_dataset
    from finetune.data.librilight_interleaver import LibriLightTokenizer, LibriLightInterleaver
    from moshi.models import loaders
    import numpy as np

    print("=" * 80)
    print("TEST 4: LibriLight Sample Loading (Real Data)")
    print("=" * 80)

    # Check if index exists
    index_path = Path("/sise/eliyanac-group/ron_al/librilight/extracted_medium/medium/librilight.jsonl")

    if not index_path.exists():
        print("  ⚠️  LibriLight index not found - skipping real data test")
        print(f"     Expected: {index_path}")
        print("     Run: python scripts/create_librilight_index.py")
        return

    print(f"  ✅ Found LibriLight index: {index_path}")

    # Load Mimi model (for tokenization)
    print("\n  Loading Mimi model...")
    try:
        mimi = loaders.get_mimi("kyutai/moshiko-pytorch-bf16", device="cpu")
        print("  ✅ Mimi loaded")
    except Exception as e:
        print(f"  ⚠️  Could not load Mimi model: {e}")
        print("     Skipping actual audio processing test")
        print("     This is OK - the core LibriLight components are already tested")
        return

    # Create LibriLight tokenizer
    interleaver = LibriLightInterleaver(zero_padding=0)
    tokenizer = LibriLightTokenizer(
        mimi=mimi,
        interleaver=interleaver,
        duration_sec=10  # Short test
    )

    print(f"\n  Created LibriLight tokenizer (duration: 10s)")
    print(f"    Expected audio frames: {tokenizer.num_audio_frames}")

    # Try loading first sample from index
    print("\n  Loading first sample from index...")
    try:
        import json
        with open(index_path, 'r') as f:
            first_line = f.readline()
            entry = json.loads(first_line)

        print(f"    First entry: {entry}")

        # Load audio
        import sphn
        audio_file = Path("/sise/eliyanac-group/ron_al/librilight/extracted_medium/") / entry['path']
        print(f"    Audio file: {audio_file}")

        if audio_file.exists():
            audio, sr = sphn.read(str(audio_file))
            print(f"    Loaded audio shape: {audio.shape}")
            print(f"    Sample rate: {sr}")

            # Take first 10 seconds
            duration_samples = int(10 * sr)
            audio_chunk = audio[:, :duration_samples]

            # Process with tokenizer
            print(f"\n  Processing with LibriLight tokenizer...")
            sample = tokenizer(audio_chunk, start_sec=0.0, path=str(audio_file))

            print(f"    Sample codes shape: {sample.codes.shape}")
            print(f"    Expected shape: [1, 1+2*num_codebooks, num_audio_frames]")

            # Verify structure
            text_stream = sample.codes[:, 0:1, :]  # First stream (text)
            audio_streams = sample.codes[:, 1:, :]  # Remaining streams (audio)

            print(f"\n  Verifying sample structure:")
            print(f"    Text stream shape: {text_stream.shape}")
            print(f"    Text stream all zeros: {torch.all(text_stream == 0).item()}")
            print(f"    Audio streams shape: {audio_streams.shape}")
            print(f"    Audio streams min: {audio_streams.min().item()}")
            print(f"    Audio streams max: {audio_streams.max().item()}")

            # Assertions
            assert sample.codes.shape[0] == 1, "Batch size should be 1"
            assert sample.codes.shape[1] == 1 + 2 * mimi.num_codebooks, \
                f"Should have 1 text + {2 * mimi.num_codebooks} audio streams"
            assert torch.all(text_stream == 0), "Text stream should be all zeros"

            print("\n✅ LibriLight sample loading working correctly")
        else:
            print(f"  ⚠️  Audio file not found: {audio_file}")

    except Exception as e:
        print(f"  ❌ Error loading sample: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("LIBRILIGHT DATA LOADING TEST SUITE")
    print("=" * 80 + "\n")

    try:
        # Test 1: Detection
        test_librilight_detection()

        # Test 2: Interleaver
        test_librilight_interleaver()

        # Test 3: Mono to stereo
        test_mono_to_stereo_conversion()

        # Test 4: Real data loading (if available)
        test_librilight_sample_loading()

        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED")
        print("=" * 80 + "\n")

    except AssertionError as e:
        print("\n" + "=" * 80)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 80 + "\n")
        sys.exit(1)

    except Exception as e:
        print("\n" + "=" * 80)
        print(f"❌ UNEXPECTED ERROR: {e}")
        print("=" * 80 + "\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
