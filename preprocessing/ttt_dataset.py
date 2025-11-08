"""
TTT Conversation-Level Dataset Loader

Implements conversation-aware data loading for TTT training with state persistence.
Critical for proper TTT behavior: states must persist across turns within a conversation.

Key features:
- Groups samples by conversation_id
- Maintains turn order within conversations
- Resets TTT state when turn_number=0
- Supports curriculum training (8k, 16k, 32k, 64k)
- Handles padding to 64-token boundaries for TTT mini-batches
"""

import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TTTConversationDataset(Dataset):
    """
    Conversation-level dataset for TTT training.

    Organizes samples by conversation, ensuring:
    1. All turns from same conversation are processed sequentially
    2. TTT state persists across turns (reset only when turn_number=0)
    3. Conversations can be filtered by total length (curriculum training)

    Unlike standard datasets that shuffle all samples randomly, this dataset
    maintains conversation structure which is CRITICAL for TTT state persistence.
    """

    def __init__(
        self,
        data_dir: str,
        max_context_length: Optional[int] = None,
        min_turns_per_conversation: int = 2,
        pad_to_multiple: int = 64,  # TTT mini-batch size
        curriculum_stage: Optional[str] = None,  # '8k', '16k', '32k', '64k'
    ):
        """
        Args:
            data_dir: Directory containing preprocessed samples and index.json
            max_context_length: Maximum total tokens per conversation (for curriculum)
            min_turns_per_conversation: Minimum turns required per conversation
            pad_to_multiple: Pad sequences to this multiple (64 for TTT)
            curriculum_stage: Training stage ('8k', '16k', '32k', '64k')
        """
        self.data_dir = data_dir
        self.pad_to_multiple = pad_to_multiple
        self.curriculum_stage = curriculum_stage

        # Set max context based on curriculum stage
        if curriculum_stage:
            curriculum_limits = {
                '8k': 8192,
                '16k': 16384,
                '32k': 32768,
                '64k': 65536,
            }
            self.max_context_length = curriculum_limits[curriculum_stage]
            logger.info(f"Curriculum stage {curriculum_stage}: max context = {self.max_context_length}")
        else:
            self.max_context_length = max_context_length

        # Load index
        index_path = os.path.join(data_dir, "index.json")
        with open(index_path, 'r') as f:
            self.index = json.load(f)

        # Group samples by conversation
        self.conversations = self._build_conversations()

        # Filter conversations
        self.conversations = self._filter_conversations(min_turns_per_conversation)

        logger.info(f"Loaded {len(self.conversations)} conversations")
        logger.info(f"Total turns: {sum(len(c['turns']) for c in self.conversations)}")

        # Compute total context lengths
        total_tokens = sum(c['total_tokens'] for c in self.conversations)
        logger.info(f"Total tokens: {total_tokens:,}")
        logger.info(f"Average tokens per conversation: {total_tokens/len(self.conversations):.1f}")

    def _build_conversations(self) -> List[Dict]:
        """Group index entries by conversation_id."""
        conv_dict = defaultdict(list)

        for entry in self.index:
            conv_id = entry['conversation_id']
            conv_dict[conv_id].append(entry)

        # Convert to list and sort turns
        conversations = []
        for conv_id, turns in conv_dict.items():
            # Sort by turn_number
            turns = sorted(turns, key=lambda x: x['turn_number'])

            # Calculate total tokens
            total_tokens = sum(t['num_tokens'] for t in turns)

            conversations.append({
                'conversation_id': conv_id,
                'turns': turns,
                'total_tokens': total_tokens,
                'num_turns': len(turns),
            })

        return conversations

    def _filter_conversations(self, min_turns: int) -> List[Dict]:
        """Filter conversations by criteria."""
        filtered = []

        for conv in self.conversations:
            # Filter by minimum turns
            if conv['num_turns'] < min_turns:
                continue

            # Filter by max context length (curriculum)
            if self.max_context_length and conv['total_tokens'] > self.max_context_length:
                continue

            filtered.append(conv)

        logger.info(f"Filtered: {len(filtered)}/{len(self.conversations)} conversations")
        if self.max_context_length:
            logger.info(f"  (max context: {self.max_context_length})")
        logger.info(f"  (min turns: {min_turns})")

        return filtered

    def __len__(self) -> int:
        """Number of conversations (NOT total turns)."""
        return len(self.conversations)

    def __getitem__(self, idx: int) -> Dict[str, any]:
        """
        Get a complete conversation with all turns.

        Returns a dictionary containing:
        - conversation_id: str
        - turns: List of turn dictionaries, each with:
            - speech: [num_samples, 1]
            - speech_lengths: [1]
            - input_ids: [seq_len]
            - labels: [seq_len]
            - turn_number: int
        - total_tokens: int
        - num_turns: int
        """
        conversation = self.conversations[idx]

        # Load all turns
        turns = []
        for turn_entry in conversation['turns']:
            sample_data = torch.load(turn_entry['file_path'])

            # Pad input_ids and labels to multiple of 64
            input_ids = sample_data['input_ids']
            labels = sample_data['labels']

            if len(input_ids) % self.pad_to_multiple != 0:
                pad_len = self.pad_to_multiple - (len(input_ids) % self.pad_to_multiple)
                # Pad with tokenizer pad_token_id (usually 0) for input_ids
                # Pad with -100 for labels (ignore in loss)
                input_ids = torch.cat([
                    input_ids,
                    torch.zeros(pad_len, dtype=torch.long)  # Assuming pad_token_id = 0
                ])
                labels = torch.cat([
                    labels,
                    torch.full((pad_len,), -100, dtype=torch.long)
                ])

            turns.append({
                'speech': sample_data['speech'],
                'speech_lengths': sample_data['speech_lengths'],
                'input_ids': input_ids,
                'labels': labels,
                'turn_number': sample_data['turn_number'],
                'original_length': len(sample_data['input_ids']),  # Before padding
            })

        return {
            'conversation_id': conversation['conversation_id'],
            'turns': turns,
            'total_tokens': conversation['total_tokens'],
            'num_turns': conversation['num_turns'],
        }


def collate_conversation_batch(batch: List[Dict]) -> Dict[str, any]:
    """
    Collate function for conversation-level batching.

    Since each item is a full conversation (variable number of turns),
    this collator simply returns a list of conversations.

    For TTT training, we typically use batch_size=1 (one conversation at a time)
    to properly maintain state across turns.

    Args:
        batch: List of conversations from __getitem__

    Returns:
        Dictionary with conversation data (batch_size should be 1 for TTT)
    """
    if len(batch) > 1:
        logger.warning(
            f"Batch size > 1 ({len(batch)}) detected. "
            f"For proper TTT state persistence, use batch_size=1 (one conversation at a time)."
        )

    # For batch_size=1, just return the single conversation
    # For batch_size>1, you'd need to pad conversations to same number of turns
    return {
        'conversations': batch,
        'batch_size': len(batch),
    }


class CurriculumScheduler:
    """
    Manages curriculum training stages for TTT.

    Progressive context length training:
    Stage 1: 8k tokens (warm-up, 25% of data)
    Stage 2: 16k tokens (50% of data)
    Stage 3: 32k tokens (75% of data)
    Stage 4: 64k tokens (100% of data)

    Each stage trains until validation loss plateaus.
    """

    def __init__(
        self,
        data_dir: str,
        stages: List[Tuple[str, int, float]] = None
    ):
        """
        Args:
            data_dir: Directory containing preprocessed data
            stages: List of (stage_name, max_length, data_fraction) tuples
                   Default: [('8k', 8192, 0.25), ('16k', 16384, 0.5), ...]
        """
        self.data_dir = data_dir

        if stages is None:
            stages = [
                ('8k', 8192, 0.25),
                ('16k', 16384, 0.50),
                ('32k', 32768, 0.75),
                ('64k', 65536, 1.00),
            ]

        self.stages = stages
        self.current_stage_idx = 0

        logger.info(f"Initialized curriculum with {len(stages)} stages:")
        for name, max_len, frac in stages:
            logger.info(f"  {name}: max {max_len} tokens ({frac*100:.0f}% of data)")

    def get_dataset(self, stage_idx: Optional[int] = None) -> TTTConversationDataset:
        """
        Get dataset for a specific curriculum stage.

        Args:
            stage_idx: Stage index (None = current stage)

        Returns:
            Dataset configured for the stage
        """
        if stage_idx is None:
            stage_idx = self.current_stage_idx

        stage_name, max_length, data_fraction = self.stages[stage_idx]

        logger.info(f"Loading dataset for stage {stage_idx}: {stage_name}")

        dataset = TTTConversationDataset(
            data_dir=self.data_dir,
            curriculum_stage=stage_name,
            min_turns_per_conversation=2,
            pad_to_multiple=64,
        )

        return dataset

    def advance_stage(self) -> bool:
        """
        Move to next curriculum stage.

        Returns:
            True if advanced, False if already at final stage
        """
        if self.current_stage_idx < len(self.stages) - 1:
            self.current_stage_idx += 1
            stage_name = self.stages[self.current_stage_idx][0]
            logger.info(f"Advanced to curriculum stage {self.current_stage_idx}: {stage_name}")
            return True
        else:
            logger.info("Already at final curriculum stage")
            return False

    def get_current_stage_name(self) -> str:
        """Get name of current stage."""
        return self.stages[self.current_stage_idx][0]


def create_ttt_dataloader(
    data_dir: str,
    curriculum_stage: Optional[str] = None,
    batch_size: int = 1,  # Should be 1 for TTT!
    num_workers: int = 0,  # Multi-processing can break state persistence
    shuffle: bool = False,  # Can shuffle conversations, but not turns within
) -> DataLoader:
    """
    Create DataLoader for TTT training.

    IMPORTANT: For proper TTT state persistence:
    - batch_size should be 1 (one conversation at a time)
    - num_workers should be 0 or 1 (avoid multiprocessing state issues)
    - shuffle=True is OK (shuffles conversations, not turns)

    Args:
        data_dir: Directory with preprocessed data
        curriculum_stage: '8k', '16k', '32k', or '64k'
        batch_size: Batch size (MUST be 1 for TTT)
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle conversations

    Returns:
        DataLoader instance
    """
    if batch_size != 1:
        logger.warning(
            f"batch_size={batch_size} detected. "
            f"For proper TTT state persistence, batch_size=1 is strongly recommended."
        )

    if num_workers > 1:
        logger.warning(
            f"num_workers={num_workers} detected. "
            f"Multi-processing can break TTT state persistence. Use num_workers=0 or 1."
        )

    dataset = TTTConversationDataset(
        data_dir=data_dir,
        curriculum_stage=curriculum_stage,
        min_turns_per_conversation=2,
        pad_to_multiple=64,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_conversation_batch,
        pin_memory=True,
    )

    logger.info(f"Created DataLoader: {len(dataset)} conversations, batch_size={batch_size}")

    return dataloader


if __name__ == "__main__":
    """Test the dataset loader."""
    import argparse

    parser = argparse.ArgumentParser(description="Test TTT dataset loader")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Directory with preprocessed data")
    parser.add_argument("--curriculum_stage", type=str, default=None,
                       choices=['8k', '16k', '32k', '64k'],
                       help="Curriculum stage to test")
    parser.add_argument("--num_samples", type=int, default=3,
                       help="Number of conversations to print")

    args = parser.parse_args()

    # Create dataloader
    dataloader = create_ttt_dataloader(
        data_dir=args.data_dir,
        curriculum_stage=args.curriculum_stage,
        batch_size=1,
        shuffle=False,
    )

    print("\n" + "="*80)
    print("TESTING TTT DATASET LOADER")
    print("="*80)

    # Iterate and print samples
    for i, batch in enumerate(dataloader):
        if i >= args.num_samples:
            break

        conversations = batch['conversations']
        for conv in conversations:
            print(f"\nConversation: {conv['conversation_id']}")
            print(f"  Total tokens: {conv['total_tokens']}")
            print(f"  Number of turns: {conv['num_turns']}")

            for j, turn in enumerate(conv['turns']):
                print(f"\n  Turn {j} (turn_number={turn['turn_number']}):")
                print(f"    Speech shape: {turn['speech'].shape}")
                print(f"    Speech length: {turn['speech_lengths'].item()} samples "
                      f"({turn['speech_lengths'].item()/16000:.2f}s)")
                print(f"    Input IDs: {turn['input_ids'].shape} "
                      f"(original: {turn['original_length']})")
                print(f"    Labels: {turn['labels'].shape}")
                print(f"    Padded to 64 multiple: "
                      f"{len(turn['input_ids']) % 64 == 0}")

                # Check turn_number=0 triggers reset
                if turn['turn_number'] == 0:
                    print(f"    ⚠️  TTT STATE RESET (turn_number=0)")

    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
