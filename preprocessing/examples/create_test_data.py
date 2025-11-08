"""
Create synthetic test data for TTT preprocessing pipeline testing.

Generates minimal DeepDialogue-xtts-like structure for testing without downloading
the full 173GB dataset.
"""

import os
import json
import numpy as np
import torchaudio
import torch
from pathlib import Path
import argparse


def generate_synthetic_audio(duration_sec: float, sample_rate: int = 16000) -> torch.Tensor:
    """Generate synthetic speech-like audio (simple sine waves)."""
    num_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, num_samples)

    # Mix of frequencies to simulate speech
    freq1 = 200 + np.random.randn() * 50   # Base frequency
    freq2 = 800 + np.random.randn() * 200  # Formant 1
    freq3 = 2400 + np.random.randn() * 400 # Formant 2

    signal = (
        0.3 * np.sin(2 * np.pi * freq1 * t) +
        0.2 * np.sin(2 * np.pi * freq2 * t) +
        0.1 * np.sin(2 * np.pi * freq3 * t)
    )

    # Add some amplitude modulation (speech-like envelope)
    envelope = 0.5 * (1 + np.sin(2 * np.pi * 3 * t))
    signal = signal * envelope

    # Add noise
    noise = np.random.randn(num_samples) * 0.02
    signal = signal + noise

    # Normalize
    signal = signal / np.max(np.abs(signal)) * 0.8

    return torch.FloatTensor(signal).unsqueeze(0)  # [1, num_samples]


def generate_dialogue(dialogue_id: str, num_turns: int = 4) -> dict:
    """Generate a synthetic dialogue."""
    speakers = ['A', 'B']

    # Sample conversation templates
    templates = [
        "Hello, how can I help you today?",
        "I need assistance with my account.",
        "Of course, I'd be happy to help. What seems to be the issue?",
        "I'm having trouble logging in.",
        "Let me check that for you.",
        "Thank you for your patience.",
        "Is there anything else I can help with?",
        "No, that's all. Thank you!",
    ]

    turns = []
    for i in range(num_turns):
        speaker = speakers[i % 2]
        text = templates[i % len(templates)]

        turns.append({
            "turn_id": i,
            "speaker": speaker,
            "text": text,
            "audio_file": f"segments/{dialogue_id}_turn_{i}.wav"
        })

    return {
        "dialogue_id": dialogue_id,
        "domain": "customer_service",
        "turns": turns
    }


def create_test_dataset(output_dir: str, num_dialogues: int = 10):
    """Create complete test dataset."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    segments_dir = output_path / "segments"
    segments_dir.mkdir(exist_ok=True)

    print(f"Creating test dataset with {num_dialogues} dialogues...")

    for i in range(num_dialogues):
        dialogue_id = f"dialogue_{i:03d}"

        # Generate dialogue metadata
        num_turns = np.random.randint(3, 8)  # 3-7 turns per dialogue
        dialogue = generate_dialogue(dialogue_id, num_turns)

        # Save dialogue JSON
        dialogue_path = output_path / f"{dialogue_id}.json"
        with open(dialogue_path, 'w') as f:
            json.dump(dialogue, f, indent=2)

        # Generate audio for each turn
        for turn in dialogue['turns']:
            audio_duration = np.random.uniform(2.0, 8.0)  # 2-8 seconds
            audio = generate_synthetic_audio(audio_duration)

            audio_path = output_path / turn['audio_file']
            torchaudio.save(str(audio_path), audio, sample_rate=16000)

        print(f"  Created {dialogue_id}: {num_turns} turns")

    # Create dataset info file
    info = {
        "dataset": "DeepDialogue-xtts (Synthetic Test)",
        "num_dialogues": num_dialogues,
        "total_turns": sum(len(generate_dialogue(f"dialogue_{i:03d}")['turns']) for i in range(num_dialogues)),
        "note": "This is synthetic test data for preprocessing pipeline testing. "
                "For real training, download the full DeepDialogue-xtts dataset."
    }

    info_path = output_path / "dataset_info.json"
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)

    print(f"\n✅ Test dataset created at {output_dir}")
    print(f"   - {num_dialogues} dialogues")
    print(f"   - {info['total_turns']} total turns")
    print(f"   - Synthetic audio files in {segments_dir}")


def main():
    parser = argparse.ArgumentParser(description="Create synthetic test data")
    parser.add_argument("--output_dir", type=str, default="./deepdialogue-xtts-sample",
                       help="Output directory for test data")
    parser.add_argument("--num_dialogues", type=int, default=10,
                       help="Number of dialogues to generate")

    args = parser.parse_args()

    create_test_dataset(args.output_dir, args.num_dialogues)


if __name__ == "__main__":
    main()
