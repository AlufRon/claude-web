#!/usr/bin/env python3
"""
Generate .jsonl index for LibriLight dataset.

This script scans all .flac files in the LibriLight directory and creates
a .jsonl index file with paths and durations needed for training.

Usage:
    conda activate moshi_ttt_fixed
    python scripts/create_librilight_index.py
"""

import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import sphn
from tqdm import tqdm


def create_librilight_index():
    """Generate .jsonl index for LibriLight dataset."""

    # Configuration
    librilight_dir = Path("/sise/eliyanac-group/ron_al/librilight/extracted_medium/medium/")
    output_jsonl = librilight_dir / "librilight.jsonl"

    print("=" * 80)
    print("LibriLight Index Generator")
    print("=" * 80)
    print(f"Scanning directory: {librilight_dir}")

    # Find all FLAC files
    flac_files = list(librilight_dir.rglob("*.flac"))
    print(f"Found {len(flac_files)} FLAC files")

    if len(flac_files) == 0:
        print("ERROR: No FLAC files found!")
        return False

    successful = 0
    failed = 0
    total_duration = 0.0

    print(f"\nProcessing files...")

    with open(output_jsonl, 'w') as f:
        for flac_file in tqdm(flac_files, desc="Processing"):
            try:
                # Read audio to get duration
                audio, sr = sphn.read(str(flac_file))
                duration = audio.shape[1] / sr
                total_duration += duration

                # Create relative path from librilight_dir itself
                # e.g., "100/emerald_city_librivox_64kb_mp3/emerald_city_01_baum_64kb.flac"
                relative_path = str(flac_file.relative_to(librilight_dir))

                # Write entry
                entry = {
                    "path": relative_path,
                    "duration": float(duration)
                }
                f.write(json.dumps(entry) + "\n")
                successful += 1

            except Exception as e:
                print(f"\nError processing {flac_file}: {e}")
                failed += 1
                continue

    print("\n" + "=" * 80)
    print("Results:")
    print("=" * 80)
    print(f"✅ Created: {output_jsonl}")
    print(f"   Successful: {successful} files")
    print(f"   Failed: {failed} files")
    print(f"   Total duration: {total_duration / 3600:.1f} hours")
    print(f"   Average duration: {total_duration / successful:.1f} seconds per file")
    print("=" * 80)

    return successful > 0


if __name__ == "__main__":
    success = create_librilight_index()
    sys.exit(0 if success else 1)
