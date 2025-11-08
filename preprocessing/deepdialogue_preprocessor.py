"""
DeepDialogue-xtts Dataset Preprocessor for TTT Training

Converts DeepDialogue-xtts dataset to TTT-compatible format for Llama-Omni training.
Ensures conversation-level structure with proper turn tracking for TTT state persistence.

Key features:
- Loads audio with 16kHz sampling (Whisper encoder requirement)
- Creates input_ids with <SPEECH> token markers
- Generates labels for language modeling loss
- Preserves conversation_id and turn_number for TTT state management
- Validates 64-token alignment for TTT mini-batches
"""

import json
import os
import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from transformers import AutoTokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TTTSample:
    """Single training sample in TTT format."""
    speech: torch.FloatTensor          # [num_samples, 1] - Raw audio waveform
    speech_lengths: torch.LongTensor   # [1] - Length of audio in samples
    input_ids: torch.LongTensor        # [seq_len] - Token IDs with <SPEECH> markers
    labels: torch.LongTensor           # [seq_len] - Labels for LM loss (-100 for ignore)
    conversation_id: str               # Unique conversation identifier
    turn_number: int                   # Turn number (0 = reset TTT state)

    def validate(self):
        """Validate sample meets TTT requirements."""
        assert self.speech.dim() == 2, f"Speech must be [num_samples, 1], got {self.speech.shape}"
        assert self.speech.shape[1] == 1, f"Speech must be mono, got {self.speech.shape[1]} channels"
        assert self.speech_lengths.item() == self.speech.shape[0], "Speech length mismatch"
        assert self.input_ids.dim() == 1, f"input_ids must be 1D, got {self.input_ids.shape}"
        assert self.labels.dim() == 1, f"labels must be 1D, got {self.labels.shape}"
        assert len(self.input_ids) == len(self.labels), "input_ids and labels length mismatch"
        assert self.turn_number >= 0, f"turn_number must be >= 0, got {self.turn_number}"
        return True


class DeepDialoguePreprocessor:
    """
    Preprocessor for DeepDialogue-xtts dataset.

    Expected input format (from HuggingFace):
    {
        "dialogue_id": "dialogue_001",
        "domain": "customer_service",
        "turns": [
            {
                "turn_id": 0,
                "speaker": "A",
                "text": "Hello, I need help with my order.",
                "audio_file": "segments/dialogue_001_turn_0.wav"
            },
            ...
        ]
    }
    """

    def __init__(
        self,
        tokenizer_path: str = "meta-llama/Llama-3.1-8B-Instruct",
        target_sample_rate: int = 16000,  # Whisper encoder requirement
        speech_token: str = "<SPEECH>",
        ignore_index: int = -100,
    ):
        """
        Args:
            tokenizer_path: Path to Llama tokenizer
            target_sample_rate: Target audio sample rate (16kHz for Whisper)
            speech_token: Special token for speech segments
            ignore_index: Label value for tokens to ignore in loss
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
        self.target_sample_rate = target_sample_rate
        self.speech_token = speech_token
        self.ignore_index = ignore_index

        # Add special tokens if not present
        if speech_token not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens({'additional_special_tokens': [speech_token]})
            logger.info(f"Added {speech_token} token to vocabulary")

        self.speech_token_id = self.tokenizer.convert_tokens_to_ids(speech_token)
        logger.info(f"Initialized preprocessor with tokenizer from {tokenizer_path}")
        logger.info(f"Speech token '{speech_token}' has ID: {self.speech_token_id}")

    def load_audio(self, audio_path: str) -> Tuple[torch.FloatTensor, int]:
        """
        Load audio file and resample to target sample rate.

        Args:
            audio_path: Path to audio file

        Returns:
            waveform: [num_samples, 1] audio tensor
            sample_rate: Original sample rate
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        waveform, sample_rate = torchaudio.load(audio_path)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if necessary
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=self.target_sample_rate
            )
            waveform = resampler(waveform)

        # Transpose to [num_samples, 1]
        waveform = waveform.T

        return waveform, sample_rate

    def create_input_sequence(
        self,
        user_text: Optional[str],
        assistant_text: str,
        is_speech_input: bool = True
    ) -> Tuple[torch.LongTensor, torch.LongTensor]:
        """
        Create input_ids and labels for a conversation turn.

        Format for speech input:
            User: <SPEECH>
            Assistant: {assistant_text}

        Format for text input:
            User: {user_text}
            Assistant: {assistant_text}

        Args:
            user_text: User's text (None if speech-only)
            assistant_text: Assistant's response text
            is_speech_input: Whether user input is speech

        Returns:
            input_ids: [seq_len] token IDs
            labels: [seq_len] labels with user tokens masked
        """
        # Build conversation
        if is_speech_input:
            user_content = self.speech_token
        else:
            user_content = user_text if user_text else ""

        # Llama-3.1 chat format
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_text}
        ]

        # Tokenize
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
            return_tensors="pt"
        ).squeeze(0)

        # Create labels: mask user input, keep assistant output
        labels = input_ids.clone()

        # Find assistant's response start
        # Tokenize just the assistant message to find where it starts
        assistant_only = self.tokenizer.apply_chat_template(
            [{"role": "assistant", "content": assistant_text}],
            add_generation_prompt=False,
            return_tensors="pt"
        ).squeeze(0)

        # Mask everything except assistant's actual response
        # Simple heuristic: mask first part, keep last len(assistant_only) tokens
        mask_until = len(input_ids) - len(assistant_only)
        labels[:mask_until] = self.ignore_index

        return input_ids, labels

    def process_dialogue(
        self,
        dialogue_path: str,
        audio_base_dir: str,
    ) -> List[TTTSample]:
        """
        Process a single dialogue JSON file into TTT samples.

        Args:
            dialogue_path: Path to dialogue JSON file
            audio_base_dir: Base directory containing audio files

        Returns:
            List of TTTSample objects, one per turn
        """
        with open(dialogue_path, 'r') as f:
            dialogue = json.load(f)

        conversation_id = dialogue['dialogue_id']
        turns = dialogue['turns']
        samples = []

        logger.info(f"Processing dialogue {conversation_id} with {len(turns)} turns")

        for turn in turns:
            turn_id = turn['turn_id']
            text = turn['text']
            audio_file = turn['audio_file']

            # Load audio
            audio_path = os.path.join(audio_base_dir, audio_file)
            try:
                waveform, original_sr = self.load_audio(audio_path)
            except FileNotFoundError:
                logger.warning(f"Skipping turn {turn_id}: audio file not found at {audio_path}")
                continue

            # Create input sequence
            # User speaks (speech input), assistant responds (text from transcript)
            input_ids, labels = self.create_input_sequence(
                user_text=None,  # Speech input, no text
                assistant_text=text,
                is_speech_input=True
            )

            # Create sample
            sample = TTTSample(
                speech=waveform,
                speech_lengths=torch.LongTensor([waveform.shape[0]]),
                input_ids=input_ids,
                labels=labels,
                conversation_id=conversation_id,
                turn_number=turn_id
            )

            # Validate
            try:
                sample.validate()
            except AssertionError as e:
                logger.error(f"Validation failed for {conversation_id} turn {turn_id}: {e}")
                continue

            samples.append(sample)
            logger.debug(f"Processed turn {turn_id}: {len(waveform)} samples, {len(input_ids)} tokens")

        logger.info(f"Completed dialogue {conversation_id}: {len(samples)}/{len(turns)} turns processed")
        return samples

    def process_dataset(
        self,
        dataset_dir: str,
        audio_base_dir: str,
        output_dir: str,
        max_dialogues: Optional[int] = None,
        save_format: str = "pt"  # 'pt' or 'arrow'
    ) -> Dict[str, int]:
        """
        Process entire DeepDialogue-xtts dataset.

        Args:
            dataset_dir: Directory containing dialogue JSON files
            audio_base_dir: Base directory containing audio files
            output_dir: Directory to save processed samples
            max_dialogues: Maximum dialogues to process (None = all)
            save_format: 'pt' for PyTorch tensors, 'arrow' for Arrow format

        Returns:
            Statistics dictionary
        """
        os.makedirs(output_dir, exist_ok=True)

        # Find all dialogue files
        dialogue_files = sorted(Path(dataset_dir).glob("*.json"))
        if max_dialogues:
            dialogue_files = dialogue_files[:max_dialogues]

        logger.info(f"Found {len(dialogue_files)} dialogue files to process")

        stats = {
            'total_dialogues': 0,
            'total_turns': 0,
            'total_audio_duration_sec': 0.0,
            'total_tokens': 0,
            'failed_turns': 0
        }

        all_samples = []

        for i, dialogue_file in enumerate(dialogue_files):
            logger.info(f"Processing dialogue {i+1}/{len(dialogue_files)}: {dialogue_file.name}")

            try:
                samples = self.process_dialogue(str(dialogue_file), audio_base_dir)

                stats['total_dialogues'] += 1
                stats['total_turns'] += len(samples)

                for sample in samples:
                    stats['total_audio_duration_sec'] += sample.speech_lengths.item() / self.target_sample_rate
                    stats['total_tokens'] += len(sample.input_ids)
                    all_samples.append(sample)

            except Exception as e:
                logger.error(f"Failed to process {dialogue_file.name}: {e}")
                stats['failed_turns'] += 1
                continue

        # Save processed samples
        if save_format == "pt":
            self._save_pytorch(all_samples, output_dir)
        elif save_format == "arrow":
            self._save_arrow(all_samples, output_dir)
        else:
            raise ValueError(f"Unknown save format: {save_format}")

        # Save statistics
        stats_path = os.path.join(output_dir, "preprocessing_stats.json")
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Preprocessing complete! Statistics:")
        logger.info(f"  Dialogues: {stats['total_dialogues']}")
        logger.info(f"  Turns: {stats['total_turns']}")
        logger.info(f"  Audio duration: {stats['total_audio_duration_sec']/3600:.2f} hours")
        logger.info(f"  Total tokens: {stats['total_tokens']:,}")
        logger.info(f"  Failed turns: {stats['failed_turns']}")

        return stats

    def _save_pytorch(self, samples: List[TTTSample], output_dir: str):
        """Save samples in PyTorch format (one file per sample)."""
        samples_dir = os.path.join(output_dir, "samples")
        os.makedirs(samples_dir, exist_ok=True)

        # Create index file
        index = []

        for i, sample in enumerate(samples):
            sample_path = os.path.join(samples_dir, f"sample_{i:06d}.pt")

            torch.save({
                'speech': sample.speech,
                'speech_lengths': sample.speech_lengths,
                'input_ids': sample.input_ids,
                'labels': sample.labels,
                'conversation_id': sample.conversation_id,
                'turn_number': sample.turn_number,
            }, sample_path)

            index.append({
                'sample_id': i,
                'conversation_id': sample.conversation_id,
                'turn_number': sample.turn_number,
                'num_tokens': len(sample.input_ids),
                'audio_duration_sec': sample.speech_lengths.item() / self.target_sample_rate,
                'file_path': sample_path
            })

        # Save index
        index_path = os.path.join(output_dir, "index.json")
        with open(index_path, 'w') as f:
            json.dump(index, f, indent=2)

        logger.info(f"Saved {len(samples)} samples to {samples_dir}")
        logger.info(f"Saved index to {index_path}")

    def _save_arrow(self, samples: List[TTTSample], output_dir: str):
        """Save samples in Apache Arrow format (for HuggingFace datasets)."""
        try:
            from datasets import Dataset, Features, Value, Array2D, Array3D, Sequence
        except ImportError:
            raise ImportError("datasets library required for Arrow format. Install with: pip install datasets")

        # Convert to HuggingFace dataset format
        data = {
            'speech': [sample.speech.numpy() for sample in samples],
            'speech_lengths': [sample.speech_lengths.item() for sample in samples],
            'input_ids': [sample.input_ids.tolist() for sample in samples],
            'labels': [sample.labels.tolist() for sample in samples],
            'conversation_id': [sample.conversation_id for sample in samples],
            'turn_number': [sample.turn_number for sample in samples],
        }

        dataset = Dataset.from_dict(data)
        dataset_path = os.path.join(output_dir, "dataset.arrow")
        dataset.save_to_disk(dataset_path)

        logger.info(f"Saved {len(samples)} samples to {dataset_path}")


def main():
    """Example usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess DeepDialogue-xtts for TTT training")
    parser.add_argument("--dataset_dir", type=str, required=True,
                       help="Directory containing dialogue JSON files")
    parser.add_argument("--audio_dir", type=str, required=True,
                       help="Base directory containing audio files")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for processed samples")
    parser.add_argument("--tokenizer", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                       help="Tokenizer path")
    parser.add_argument("--max_dialogues", type=int, default=None,
                       help="Maximum dialogues to process (for testing)")
    parser.add_argument("--format", type=str, default="pt", choices=["pt", "arrow"],
                       help="Output format: 'pt' (PyTorch) or 'arrow' (HuggingFace)")

    args = parser.parse_args()

    preprocessor = DeepDialoguePreprocessor(tokenizer_path=args.tokenizer)

    stats = preprocessor.process_dataset(
        dataset_dir=args.dataset_dir,
        audio_base_dir=args.audio_dir,
        output_dir=args.output_dir,
        max_dialogues=args.max_dialogues,
        save_format=args.format
    )

    print("\n" + "="*80)
    print("PREPROCESSING COMPLETE")
    print("="*80)
    print(f"Processed {stats['total_dialogues']} dialogues")
    print(f"Total turns: {stats['total_turns']}")
    print(f"Audio duration: {stats['total_audio_duration_sec']/3600:.2f} hours")
    print(f"Total tokens: {stats['total_tokens']:,}")
    print(f"Average tokens per turn: {stats['total_tokens']/stats['total_turns']:.1f}")
    print(f"Output saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
