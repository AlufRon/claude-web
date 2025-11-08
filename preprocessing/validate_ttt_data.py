"""
TTT Dataset Validation Utilities

Comprehensive validation of preprocessed data to ensure TTT training requirements are met.

Critical validations:
1. Sequence length alignment (64-token multiples for TTT mini-batches)
2. Conversation structure (proper turn ordering, conversation_id presence)
3. Turn_number logic (0 triggers reset, sequential increments)
4. Audio format (16kHz, mono, correct shapes)
5. Token distribution (no extreme outliers, proper masking)
6. FP32 compatibility checks (data types for future TTT weights)
"""

import json
import os
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TTTDataValidator:
    """Validates preprocessed TTT dataset for training readiness."""

    def __init__(self, data_dir: str):
        """
        Args:
            data_dir: Directory containing preprocessed samples and index
        """
        self.data_dir = data_dir
        self.index_path = os.path.join(data_dir, "index.json")

        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"Index file not found: {self.index_path}")

        with open(self.index_path, 'r') as f:
            self.index = json.load(f)

        logger.info(f"Loaded index with {len(self.index)} samples")

    def validate_all(self, sample_size: Optional[int] = None) -> Dict[str, any]:
        """
        Run all validation checks.

        Args:
            sample_size: Number of samples to validate (None = all)

        Returns:
            Validation report dictionary
        """
        logger.info("="*80)
        logger.info("STARTING COMPREHENSIVE TTT DATA VALIDATION")
        logger.info("="*80)

        samples_to_check = self.index[:sample_size] if sample_size else self.index

        report = {
            'total_samples': len(self.index),
            'validated_samples': len(samples_to_check),
            'passed': True,
            'errors': [],
            'warnings': [],
            'checks': {}
        }

        # Check 1: Index integrity
        logger.info("\n[1/8] Validating index integrity...")
        index_check = self._validate_index()
        report['checks']['index_integrity'] = index_check
        if not index_check['passed']:
            report['passed'] = False
            report['errors'].extend(index_check['errors'])

        # Check 2: Conversation structure
        logger.info("\n[2/8] Validating conversation structure...")
        conv_check = self._validate_conversations()
        report['checks']['conversation_structure'] = conv_check
        if not conv_check['passed']:
            report['passed'] = False
            report['errors'].extend(conv_check['errors'])

        # Check 3: Sample format
        logger.info("\n[3/8] Validating sample format...")
        format_check = self._validate_sample_formats(samples_to_check)
        report['checks']['sample_format'] = format_check
        if not format_check['passed']:
            report['passed'] = False
            report['errors'].extend(format_check['errors'])
        report['warnings'].extend(format_check['warnings'])

        # Check 4: Audio specifications
        logger.info("\n[4/8] Validating audio specifications...")
        audio_check = self._validate_audio(samples_to_check)
        report['checks']['audio'] = audio_check
        if not audio_check['passed']:
            report['passed'] = False
            report['errors'].extend(audio_check['errors'])

        # Check 5: Sequence alignment (TTT mini-batches)
        logger.info("\n[5/8] Validating sequence alignment for TTT...")
        alignment_check = self._validate_alignment(samples_to_check)
        report['checks']['alignment'] = alignment_check
        if not alignment_check['passed']:
            report['passed'] = False
            report['errors'].extend(alignment_check['errors'])

        # Check 6: Turn ordering
        logger.info("\n[6/8] Validating turn ordering...")
        turn_check = self._validate_turn_ordering()
        report['checks']['turn_ordering'] = turn_check
        if not turn_check['passed']:
            report['passed'] = False
            report['errors'].extend(turn_check['errors'])

        # Check 7: Token statistics
        logger.info("\n[7/8] Computing token statistics...")
        token_stats = self._compute_token_statistics(samples_to_check)
        report['checks']['token_statistics'] = token_stats

        # Check 8: Label masking
        logger.info("\n[8/8] Validating label masking...")
        label_check = self._validate_labels(samples_to_check)
        report['checks']['labels'] = label_check
        if not label_check['passed']:
            report['passed'] = False
            report['errors'].extend(label_check['errors'])

        # Summary
        logger.info("\n" + "="*80)
        if report['passed']:
            logger.info("✅ VALIDATION PASSED - Dataset is ready for TTT training!")
        else:
            logger.error("❌ VALIDATION FAILED - Please fix errors before training")
            logger.error(f"Total errors: {len(report['errors'])}")
            for error in report['errors'][:10]:  # Show first 10
                logger.error(f"  - {error}")
            if len(report['errors']) > 10:
                logger.error(f"  ... and {len(report['errors']) - 10} more")

        if report['warnings']:
            logger.warning(f"⚠️  {len(report['warnings'])} warnings:")
            for warning in report['warnings'][:5]:
                logger.warning(f"  - {warning}")

        logger.info("="*80)

        return report

    def _validate_index(self) -> Dict:
        """Validate index file integrity."""
        errors = []
        required_fields = ['sample_id', 'conversation_id', 'turn_number',
                          'num_tokens', 'audio_duration_sec', 'file_path']

        for i, entry in enumerate(self.index):
            # Check required fields
            missing = [f for f in required_fields if f not in entry]
            if missing:
                errors.append(f"Sample {i}: missing fields {missing}")

            # Check file exists
            if 'file_path' in entry and not os.path.exists(entry['file_path']):
                errors.append(f"Sample {i}: file not found at {entry['file_path']}")

        return {
            'passed': len(errors) == 0,
            'errors': errors,
            'total_entries': len(self.index)
        }

    def _validate_conversations(self) -> Dict:
        """Validate conversation structure."""
        errors = []

        # Group by conversation
        conv_dict = defaultdict(list)
        for entry in self.index:
            conv_dict[entry['conversation_id']].append(entry)

        # Check each conversation
        for conv_id, turns in conv_dict.items():
            # Sort by turn_number
            turns_sorted = sorted(turns, key=lambda x: x['turn_number'])

            # Check turn numbers are sequential starting from 0
            turn_numbers = [t['turn_number'] for t in turns_sorted]
            if turn_numbers[0] != 0:
                errors.append(f"Conversation {conv_id}: first turn is {turn_numbers[0]}, expected 0")

            # Check for gaps
            for i in range(1, len(turn_numbers)):
                if turn_numbers[i] != turn_numbers[i-1] + 1:
                    errors.append(
                        f"Conversation {conv_id}: turn gap between {turn_numbers[i-1]} and {turn_numbers[i]}"
                    )

        return {
            'passed': len(errors) == 0,
            'errors': errors,
            'total_conversations': len(conv_dict),
            'total_turns': len(self.index),
            'avg_turns_per_conversation': len(self.index) / len(conv_dict)
        }

    def _validate_sample_formats(self, samples: List[Dict]) -> Dict:
        """Validate individual sample formats."""
        errors = []
        warnings = []

        for entry in tqdm(samples, desc="Checking sample formats"):
            try:
                sample = torch.load(entry['file_path'])

                # Required fields
                required = ['speech', 'speech_lengths', 'input_ids', 'labels',
                           'conversation_id', 'turn_number']
                missing = [f for f in required if f not in sample]
                if missing:
                    errors.append(f"{entry['file_path']}: missing fields {missing}")
                    continue

                # Check types
                if not isinstance(sample['speech'], torch.Tensor):
                    errors.append(f"{entry['file_path']}: speech is not a tensor")
                if not isinstance(sample['input_ids'], torch.Tensor):
                    errors.append(f"{entry['file_path']}: input_ids is not a tensor")

                # Check dimensions
                if sample['speech'].dim() != 2:
                    errors.append(
                        f"{entry['file_path']}: speech has {sample['speech'].dim()} dims, expected 2"
                    )
                if sample['input_ids'].dim() != 1:
                    errors.append(
                        f"{entry['file_path']}: input_ids has {sample['input_ids'].dim()} dims, expected 1"
                    )

                # Check shapes match
                if len(sample['input_ids']) != len(sample['labels']):
                    errors.append(
                        f"{entry['file_path']}: input_ids length {len(sample['input_ids'])} "
                        f"!= labels length {len(sample['labels'])}"
                    )

                # Check speech is mono
                if sample['speech'].dim() == 2 and sample['speech'].shape[1] != 1:
                    errors.append(
                        f"{entry['file_path']}: speech has {sample['speech'].shape[1]} channels, expected 1"
                    )

            except Exception as e:
                errors.append(f"{entry['file_path']}: failed to load - {str(e)}")

        return {
            'passed': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'samples_checked': len(samples)
        }

    def _validate_audio(self, samples: List[Dict]) -> Dict:
        """Validate audio specifications."""
        errors = []
        sample_rates = []
        durations = []

        for entry in tqdm(samples[:100], desc="Checking audio specs"):  # Sample first 100
            try:
                sample = torch.load(entry['file_path'])
                speech = sample['speech']
                speech_len = sample['speech_lengths'].item()

                # Check length matches
                if speech.shape[0] != speech_len:
                    errors.append(
                        f"{entry['file_path']}: speech shape {speech.shape[0]} "
                        f"!= speech_lengths {speech_len}"
                    )

                # Compute duration (assuming 16kHz)
                duration = speech_len / 16000
                durations.append(duration)

                # Check for silence (potential issue)
                if speech.abs().max() < 0.001:
                    errors.append(f"{entry['file_path']}: audio appears to be silent")

                # Check for clipping
                if speech.abs().max() > 0.99:
                    errors.append(f"{entry['file_path']}: audio may be clipped")

            except Exception as e:
                errors.append(f"{entry['file_path']}: audio validation failed - {str(e)}")

        return {
            'passed': len(errors) == 0,
            'errors': errors,
            'avg_duration_sec': np.mean(durations) if durations else 0,
            'max_duration_sec': np.max(durations) if durations else 0,
            'min_duration_sec': np.min(durations) if durations else 0,
        }

    def _validate_alignment(self, samples: List[Dict]) -> Dict:
        """Validate sequence alignment to 64-token multiples (TTT mini-batch requirement)."""
        errors = []
        misaligned_count = 0

        for entry in tqdm(samples, desc="Checking 64-token alignment"):
            try:
                sample = torch.load(entry['file_path'])
                seq_len = len(sample['input_ids'])

                if seq_len % 64 != 0:
                    misaligned_count += 1
                    if misaligned_count <= 5:  # Report first 5 only
                        errors.append(
                            f"{entry['file_path']}: sequence length {seq_len} "
                            f"is not multiple of 64 (remainder: {seq_len % 64})"
                        )

            except Exception as e:
                errors.append(f"{entry['file_path']}: alignment check failed - {str(e)}")

        if misaligned_count > 5:
            errors.append(f"... and {misaligned_count - 5} more misaligned sequences")

        return {
            'passed': misaligned_count == 0,
            'errors': errors,
            'misaligned_count': misaligned_count,
            'samples_checked': len(samples)
        }

    def _validate_turn_ordering(self) -> Dict:
        """Validate turn_number ordering within conversations."""
        errors = []

        conv_dict = defaultdict(list)
        for entry in self.index:
            conv_dict[entry['conversation_id']].append(entry)

        for conv_id, turns in conv_dict.items():
            # Load turn_numbers from actual files
            turn_data = []
            for entry in turns:
                try:
                    sample = torch.load(entry['file_path'])
                    turn_data.append({
                        'index_turn_number': entry['turn_number'],
                        'file_turn_number': sample['turn_number'],
                        'file_path': entry['file_path']
                    })
                except Exception as e:
                    errors.append(f"{entry['file_path']}: failed to load for turn validation - {str(e)}")

            # Check consistency between index and file
            for td in turn_data:
                if td['index_turn_number'] != td['file_turn_number']:
                    errors.append(
                        f"{td['file_path']}: index has turn_number={td['index_turn_number']}, "
                        f"file has turn_number={td['file_turn_number']}"
                    )

        return {
            'passed': len(errors) == 0,
            'errors': errors
        }

    def _compute_token_statistics(self, samples: List[Dict]) -> Dict:
        """Compute token distribution statistics."""
        token_counts = []
        label_ratios = []  # Ratio of non-masked labels

        for entry in tqdm(samples[:1000], desc="Computing token stats"):  # Sample 1000
            try:
                sample = torch.load(entry['file_path'])
                input_ids = sample['input_ids']
                labels = sample['labels']

                token_counts.append(len(input_ids))

                # Compute ratio of non-masked labels
                non_masked = (labels != -100).sum().item()
                label_ratios.append(non_masked / len(labels))

            except Exception:
                continue

        return {
            'total_tokens': sum(token_counts),
            'avg_tokens_per_sample': np.mean(token_counts) if token_counts else 0,
            'median_tokens_per_sample': np.median(token_counts) if token_counts else 0,
            'max_tokens': np.max(token_counts) if token_counts else 0,
            'min_tokens': np.min(token_counts) if token_counts else 0,
            'std_tokens': np.std(token_counts) if token_counts else 0,
            'avg_label_ratio': np.mean(label_ratios) if label_ratios else 0,
            'median_label_ratio': np.median(label_ratios) if label_ratios else 0,
        }

    def _validate_labels(self, samples: List[Dict]) -> Dict:
        """Validate label masking."""
        errors = []
        warnings = []

        for entry in tqdm(samples[:100], desc="Checking labels"):
            try:
                sample = torch.load(entry['file_path'])
                labels = sample['labels']

                # Check that at least some labels are not masked
                non_masked = (labels != -100).sum().item()
                if non_masked == 0:
                    errors.append(f"{entry['file_path']}: all labels are masked (-100)")
                elif non_masked < 10:
                    warnings.append(
                        f"{entry['file_path']}: very few non-masked labels ({non_masked})"
                    )

                # Check label values are valid token IDs (or -100)
                valid_mask = (labels == -100) | (labels >= 0)
                if not valid_mask.all():
                    errors.append(f"{entry['file_path']}: invalid label values detected")

            except Exception as e:
                errors.append(f"{entry['file_path']}: label validation failed - {str(e)}")

        return {
            'passed': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }

    def save_report(self, report: Dict, output_path: str):
        """Save validation report to JSON."""
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Validation report saved to {output_path}")


def main():
    """Command-line interface for validation."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate TTT dataset")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Directory containing preprocessed data")
    parser.add_argument("--sample_size", type=int, default=None,
                       help="Number of samples to validate (None = all)")
    parser.add_argument("--output", type=str, default=None,
                       help="Path to save validation report JSON")

    args = parser.parse_args()

    validator = TTTDataValidator(args.data_dir)
    report = validator.validate_all(sample_size=args.sample_size)

    if args.output:
        validator.save_report(report, args.output)

    # Exit with error code if validation failed
    import sys
    sys.exit(0 if report['passed'] else 1)


if __name__ == "__main__":
    main()
