"""
Complete example of TTT dataset preprocessing workflow.

This script demonstrates:
1. Preprocessing DeepDialogue-xtts data
2. Validating the processed data
3. Creating training dataloaders
4. Using curriculum learning scheduler

Usage:
    python preprocessing_example.py --raw_data_dir ./deepdialogue-xtts --output_dir ./processed
"""

import sys
import os
from pathlib import Path

# Add parent directory to path to import preprocessing modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from deepdialogue_preprocessor import DeepDialoguePreprocessor
from validate_ttt_data import TTTDataValidator
from ttt_dataset import (
    create_ttt_dataloader,
    CurriculumScheduler,
    TTTConversationDataset
)

import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def step1_preprocess(raw_data_dir: str, output_dir: str, max_dialogues: int = None):
    """Step 1: Preprocess raw data to TTT format."""
    logger.info("="*80)
    logger.info("STEP 1: PREPROCESSING")
    logger.info("="*80)

    preprocessor = DeepDialoguePreprocessor(
        tokenizer_path="meta-llama/Llama-3.1-8B-Instruct",
        target_sample_rate=16000,
        speech_token="<SPEECH>",
    )

    stats = preprocessor.process_dataset(
        dataset_dir=raw_data_dir,
        audio_base_dir=os.path.join(raw_data_dir, "segments"),
        output_dir=output_dir,
        max_dialogues=max_dialogues,
        save_format="pt"
    )

    logger.info("\nPreprocessing Statistics:")
    logger.info(f"  Dialogues: {stats['total_dialogues']}")
    logger.info(f"  Turns: {stats['total_turns']}")
    logger.info(f"  Audio duration: {stats['total_audio_duration_sec']/3600:.2f} hours")
    logger.info(f"  Total tokens: {stats['total_tokens']:,}")
    logger.info(f"  Failed turns: {stats['failed_turns']}")

    return stats


def step2_validate(data_dir: str, sample_size: int = None):
    """Step 2: Validate processed data."""
    logger.info("\n" + "="*80)
    logger.info("STEP 2: VALIDATION")
    logger.info("="*80)

    validator = TTTDataValidator(data_dir)
    report = validator.validate_all(sample_size=sample_size)

    # Save report
    report_path = os.path.join(data_dir, "validation_report.json")
    validator.save_report(report, report_path)

    if report['passed']:
        logger.info("\n✅ VALIDATION PASSED - Dataset is ready for TTT training!")
    else:
        logger.error("\n❌ VALIDATION FAILED - Please fix errors before training")
        logger.error(f"Total errors: {len(report['errors'])}")
        for error in report['errors'][:5]:
            logger.error(f"  - {error}")
        raise ValueError("Validation failed")

    return report


def step3_test_dataloader(data_dir: str):
    """Step 3: Test dataset loader."""
    logger.info("\n" + "="*80)
    logger.info("STEP 3: TESTING DATALOADER")
    logger.info("="*80)

    # Test basic dataloader
    logger.info("\nCreating dataloader for 8k curriculum stage...")
    dataloader = create_ttt_dataloader(
        data_dir=data_dir,
        curriculum_stage="8k",
        batch_size=1,
        shuffle=False,
    )

    logger.info(f"Dataloader created: {len(dataloader.dataset)} conversations")

    # Test iteration
    logger.info("\nTesting iteration (first 3 conversations):")
    for i, batch in enumerate(dataloader):
        if i >= 3:
            break

        conversations = batch['conversations']
        for conv in conversations:
            logger.info(f"\n  Conversation {i+1}: {conv['conversation_id']}")
            logger.info(f"    Total tokens: {conv['total_tokens']}")
            logger.info(f"    Turns: {conv['num_turns']}")

            for j, turn in enumerate(conv['turns']):
                logger.info(f"      Turn {j}: turn_number={turn['turn_number']}, "
                          f"tokens={len(turn['input_ids'])}, "
                          f"audio={turn['speech_lengths'].item()/16000:.2f}s")

                if turn['turn_number'] == 0:
                    logger.info(f"        ⚠️  TTT STATE RESET (turn_number=0)")

    logger.info("\n✅ Dataloader test passed!")


def step4_curriculum_demo(data_dir: str):
    """Step 4: Demonstrate curriculum learning."""
    logger.info("\n" + "="*80)
    logger.info("STEP 4: CURRICULUM LEARNING DEMO")
    logger.info("="*80)

    # Create curriculum scheduler
    scheduler = CurriculumScheduler(data_dir=data_dir)

    logger.info("\nCurriculum stages:")
    for i, (stage_name, max_len, frac) in enumerate(scheduler.stages):
        logger.info(f"  Stage {i}: {stage_name} - max {max_len} tokens ({frac*100:.0f}% of data)")

    # Get dataset for each stage
    logger.info("\nDataset sizes per stage:")
    for i in range(len(scheduler.stages)):
        dataset = scheduler.get_dataset(stage_idx=i)
        stage_name = scheduler.stages[i][0]
        logger.info(f"  {stage_name}: {len(dataset)} conversations")

    logger.info("\n✅ Curriculum demo complete!")


def step5_sample_training_loop(data_dir: str):
    """Step 5: Demonstrate sample training loop structure."""
    logger.info("\n" + "="*80)
    logger.info("STEP 5: SAMPLE TRAINING LOOP STRUCTURE")
    logger.info("="*80)

    logger.info("\nThis demonstrates the structure of a TTT training loop:")
    logger.info("(No actual training, just showing the data flow)\n")

    # Create dataloader
    dataloader = create_ttt_dataloader(
        data_dir=data_dir,
        curriculum_stage="8k",
        batch_size=1,
        shuffle=True,
    )

    logger.info("Training loop structure:")
    logger.info("-" * 60)

    # Simulate training loop
    for epoch in range(1):  # Just 1 epoch for demo
        logger.info(f"\nEpoch {epoch+1}")

        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 2:  # Just show first 2 batches
                break

            conversations = batch['conversations']

            for conv in conversations:
                logger.info(f"\n  Processing conversation: {conv['conversation_id']}")

                # Initialize TTT state (would be actual model state)
                ttt_state = None

                for turn_idx, turn in enumerate(conv['turns']):
                    # Reset state if turn_number=0
                    if turn['turn_number'] == 0:
                        logger.info(f"    Turn {turn_idx}: RESET TTT state")
                        ttt_state = "RESET"  # Would be actual reset
                    else:
                        logger.info(f"    Turn {turn_idx}: Continue with TTT state")

                    # Forward pass (would be actual model call)
                    # output, ttt_state = model(
                    #     speech=turn['speech'],
                    #     input_ids=turn['input_ids'],
                    #     ttt_state=ttt_state
                    # )

                    logger.info(f"      - Speech: {turn['speech'].shape}")
                    logger.info(f"      - Tokens: {turn['input_ids'].shape}")
                    logger.info(f"      - TTT mini-batches: {len(turn['input_ids']) // 64}")

                    # Backward pass (would be actual backprop)
                    # loss = criterion(output, turn['labels'])
                    # loss.backward()

    logger.info("\n✅ Training loop structure demo complete!")
    logger.info("\nKey points:")
    logger.info("  1. Process one conversation at a time (batch_size=1)")
    logger.info("  2. Maintain TTT state across turns within conversation")
    logger.info("  3. Reset state when turn_number=0")
    logger.info("  4. Sequences auto-padded to 64-token boundaries")


def main():
    parser = argparse.ArgumentParser(
        description="Complete TTT preprocessing workflow example"
    )
    parser.add_argument("--raw_data_dir", type=str, required=True,
                       help="Directory containing raw DeepDialogue-xtts data")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for processed data")
    parser.add_argument("--max_dialogues", type=int, default=None,
                       help="Maximum dialogues to process (None = all)")
    parser.add_argument("--skip_preprocessing", action="store_true",
                       help="Skip preprocessing step (if already done)")
    parser.add_argument("--validation_sample_size", type=int, default=None,
                       help="Validation sample size (None = all)")

    args = parser.parse_args()

    try:
        # Step 1: Preprocess
        if not args.skip_preprocessing:
            stats = step1_preprocess(
                raw_data_dir=args.raw_data_dir,
                output_dir=args.output_dir,
                max_dialogues=args.max_dialogues
            )
        else:
            logger.info("Skipping preprocessing (--skip_preprocessing flag set)")

        # Step 2: Validate
        report = step2_validate(
            data_dir=args.output_dir,
            sample_size=args.validation_sample_size
        )

        # Step 3: Test dataloader
        step3_test_dataloader(data_dir=args.output_dir)

        # Step 4: Curriculum demo
        step4_curriculum_demo(data_dir=args.output_dir)

        # Step 5: Training loop demo
        step5_sample_training_loop(data_dir=args.output_dir)

        logger.info("\n" + "="*80)
        logger.info("✅ ALL STEPS COMPLETE!")
        logger.info("="*80)
        logger.info("\nYour dataset is ready for TTT training.")
        logger.info(f"Processed data location: {args.output_dir}")
        logger.info("\nNext steps:")
        logger.info("  1. Implement TTT modules (see docs/TRAINING_STRATEGY_ANALYSIS.md)")
        logger.info("  2. Setup training infrastructure (DeepSpeed, checkpointing)")
        logger.info("  3. Start curriculum training (8k → 16k → 32k → 64k)")

    except Exception as e:
        logger.error(f"\n❌ ERROR: {str(e)}")
        raise


if __name__ == "__main__":
    main()
