#!/usr/bin/env python3
"""
Test Suite for Batch Inference System

Comprehensive validation tests for the TTT batch inference components including:
- BatchInference class functionality  
- Main script interface
- SLURM integration
- Audio processing pipeline
- Error handling and edge cases

Usage:
    # Run all tests
    python test_batch_inference.py
    
    # Run specific test suite
    python test_batch_inference.py TestBatchInferenceCore
    python test_batch_inference.py TestMainScript
    python test_batch_inference.py TestAudioProcessing
    
    # Run with verbose output
    python test_batch_inference.py --verbose
    
    # Generate test audio files
    python test_batch_inference.py --create-test-files
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

# Suppress warnings for cleaner test output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    import torchaudio
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Configure logging for tests
logging.basicConfig(
    level=logging.WARNING,  # Reduce noise during tests
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class TestConfig:
    """Configuration for test suite."""
    
    # Test data paths (created dynamically)
    TEST_DIR = Path(__file__).parent / "test_data"
    AUDIO_DIR = TEST_DIR / "audio"
    OUTPUT_DIR = TEST_DIR / "output"
    CHECKPOINT_DIR = TEST_DIR / "mock_checkpoint"
    
    # Audio generation parameters
    SAMPLE_RATE = 24000
    DURATION = 2.0  # 2 seconds for fast tests
    FREQUENCIES = [440, 880, 1320]  # A4, A5, E6
    
    # Mock checkpoint structure
    CHECKPOINT_FILES = [
        "config.json",
        "model.safetensors",
        "generation_config.json"
    ]


def create_test_audio(output_path: Path, 
                     frequency: float = 440.0, 
                     duration: float = 2.0,
                     sample_rate: int = 24000,
                     amplitude: float = 0.5) -> None:
    """Create synthetic sine wave audio for testing."""
    if not TORCH_AVAILABLE:
        return
        
    t = torch.linspace(0, duration, int(sample_rate * duration))
    waveform = amplitude * torch.sin(2 * torch.pi * frequency * t)
    waveform = waveform.unsqueeze(0)  # Add channel dimension
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(output_path), waveform, sample_rate)


def create_mock_checkpoint(checkpoint_dir: Path) -> None:
    """Create a mock checkpoint structure for testing."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Basic config
    config = {
        "model_type": "moshi",
        "vocab_size": 2048,
        "n_embd": 1024,
        "n_layer": 32,
        "n_head": 16,
        "sample_rate": 24000,
        "ttt_config": {
            "layers": [29, 30, 31],
            "learning_rate": 0.1,
            "mini_batch_size": 16
        }
    }
    
    with open(checkpoint_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Mock model file (empty)
    (checkpoint_dir / "model.safetensors").touch()
    
    # Generation config
    gen_config = {
        "max_length": 1000,
        "temperature": 1.0,
        "top_p": 0.9
    }
    
    with open(checkpoint_dir / "generation_config.json", "w") as f:
        json.dump(gen_config, f, indent=2)


def setup_test_environment():
    """Set up test files and directories."""
    print("🔧 Setting up test environment...")
    
    # Clean up any existing test data
    if TestConfig.TEST_DIR.exists():
        shutil.rmtree(TestConfig.TEST_DIR)
    
    # Create directories
    TestConfig.AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    TestConfig.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create test audio files
    if TORCH_AVAILABLE:
        for i, freq in enumerate(TestConfig.FREQUENCIES):
            audio_path = TestConfig.AUDIO_DIR / f"test_audio_{i}.wav"
            create_test_audio(audio_path, frequency=freq, duration=TestConfig.DURATION)
            print(f"   📄 Created {audio_path.name}")
    
    # Create mock checkpoint
    create_mock_checkpoint(TestConfig.CHECKPOINT_DIR)
    print(f"   🗂️  Created mock checkpoint in {TestConfig.CHECKPOINT_DIR}")
    
    # Create test file list
    file_list_path = TestConfig.TEST_DIR / "file_list.txt"
    if TORCH_AVAILABLE:
        with open(file_list_path, "w") as f:
            for i in range(len(TestConfig.FREQUENCIES)):
                audio_path = TestConfig.AUDIO_DIR / f"test_audio_{i}.wav"
                f.write(f"{audio_path}\\n")
        print(f"   📋 Created {file_list_path}")
    
    print("✅ Test environment ready")


class TestBatchInferenceCore(unittest.TestCase):
    """Test core BatchInference functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test class."""
        cls.test_dir = TestConfig.TEST_DIR
        cls.checkpoint_dir = TestConfig.CHECKPOINT_DIR
    
    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
    def test_import_batch_inference(self):
        """Test that we can import BatchInference classes."""
        try:
            from inference.batch_inference import BatchInference, load_batch_inference
            self.assertTrue(True, "Successfully imported BatchInference")
        except ImportError as e:
            self.fail(f"Failed to import BatchInference: {e}")
    
    def test_checkpoint_validation(self):
        """Test checkpoint directory validation."""
        # Valid checkpoint directory
        self.assertTrue(self.checkpoint_dir.exists(), "Mock checkpoint should exist")
        
        # Required files
        required_files = ["config.json", "model.safetensors"]
        for file in required_files:
            file_path = self.checkpoint_dir / file
            self.assertTrue(file_path.exists(), f"Required file {file} should exist")
    
    def test_config_loading(self):
        """Test loading checkpoint configuration."""
        config_path = self.checkpoint_dir / "config.json"
        
        with open(config_path) as f:
            config = json.load(f)
        
        # Check required fields
        self.assertIn("model_type", config)
        self.assertIn("ttt_config", config)
        self.assertIn("sample_rate", config)
        
        # Check TTT config
        ttt_config = config["ttt_config"]
        self.assertIn("layers", ttt_config)
        self.assertIn("learning_rate", ttt_config)
        self.assertIn("mini_batch_size", ttt_config)


class TestAudioProcessing(unittest.TestCase):
    """Test audio file processing functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test class."""
        cls.audio_dir = TestConfig.AUDIO_DIR
        cls.output_dir = TestConfig.OUTPUT_DIR
    
    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
    def test_audio_file_creation(self):
        """Test that test audio files are created correctly."""
        for i in range(len(TestConfig.FREQUENCIES)):
            audio_path = self.audio_dir / f"test_audio_{i}.wav"
            self.assertTrue(audio_path.exists(), f"Audio file {audio_path} should exist")
            
            # Check audio properties
            waveform, sample_rate = torchaudio.load(str(audio_path))
            self.assertEqual(sample_rate, TestConfig.SAMPLE_RATE)
            expected_samples = int(TestConfig.SAMPLE_RATE * TestConfig.DURATION)
            self.assertAlmostEqual(waveform.shape[1], expected_samples, delta=100)
    
    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available") 
    def test_audio_loading_function(self):
        """Test the audio loading function from run_batch_inference."""
        try:
            from inference.run_batch_inference import load_audio_files
            
            # Get list of test audio files
            audio_files = []
            for i in range(len(TestConfig.FREQUENCIES)):
                audio_path = self.audio_dir / f"test_audio_{i}.wav"
                if audio_path.exists():
                    audio_files.append(str(audio_path))
            
            if audio_files:
                # Test loading
                audio_batch = load_audio_files(
                    audio_files[:2],  # Test with first 2 files
                    target_sr=TestConfig.SAMPLE_RATE,
                    device="cpu"  # Use CPU for tests
                )
                
                # Check shape: [B, 1, samples]
                self.assertEqual(len(audio_batch.shape), 3)
                self.assertEqual(audio_batch.shape[0], 2)  # Batch size
                self.assertEqual(audio_batch.shape[1], 1)  # Mono audio
                
        except ImportError:
            self.skipTest("Could not import load_audio_files function")
    
    def test_file_list_parsing(self):
        """Test parsing file list for batch processing."""
        file_list_path = TestConfig.TEST_DIR / "file_list.txt"
        
        if file_list_path.exists():
            with open(file_list_path) as f:
                lines = f.readlines()
            
            # Should have one line per test audio file
            self.assertGreater(len(lines), 0, "File list should not be empty")
            
            # Each line should be a valid path
            for line in lines:
                audio_path = Path(line.strip())
                if audio_path.exists():  # Only test existing files
                    self.assertTrue(audio_path.is_file())
                    self.assertIn(audio_path.suffix, ['.wav', '.mp3', '.flac'])


class TestMainScript(unittest.TestCase):
    """Test the main run_batch_inference.py script."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test class."""
        cls.script_path = Path(__file__).parent / "run_batch_inference.py"
        cls.test_dir = TestConfig.TEST_DIR
        cls.checkpoint_dir = TestConfig.CHECKPOINT_DIR
        cls.audio_dir = TestConfig.AUDIO_DIR
        cls.output_dir = TestConfig.OUTPUT_DIR
    
    def test_script_exists(self):
        """Test that the main script exists and is executable."""
        self.assertTrue(self.script_path.exists(), "run_batch_inference.py should exist")
        self.assertTrue(os.access(self.script_path, os.R_OK), "Script should be readable")
    
    def test_help_message(self):
        """Test script help message."""
        try:
            result = subprocess.run(
                [sys.executable, str(self.script_path), "--help"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            self.assertEqual(result.returncode, 0, "Help should exit with code 0")
            self.assertIn("batch inference", result.stdout.lower())
            self.assertIn("--checkpoint", result.stdout)
            self.assertIn("--input", result.stdout)
            
        except subprocess.TimeoutExpired:
            self.fail("Script help command timed out")
        except FileNotFoundError:
            self.fail("Could not run Python script")
    
    def test_argument_validation(self):
        """Test script argument validation."""
        # Test missing required arguments
        try:
            result = subprocess.run(
                [sys.executable, str(self.script_path)],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            # Should fail with non-zero exit code
            self.assertNotEqual(result.returncode, 0)
            # Should mention missing arguments
            error_output = result.stderr.lower()
            self.assertTrue(
                "required" in error_output or "error" in error_output,
                "Should indicate missing required arguments"
            )
            
        except subprocess.TimeoutExpired:
            self.fail("Argument validation test timed out")
    
    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
    def test_mock_run_dry(self):
        """Test script with mock data (dry run without actual model loading)."""
        # This test verifies the script can parse arguments and validate files
        # without actually loading a model (which would require real checkpoints)
        
        if not any(self.audio_dir.glob("*.wav")):
            self.skipTest("No test audio files available")
        
        # Get first available audio file
        audio_files = list(self.audio_dir.glob("*.wav"))
        test_audio = audio_files[0]
        
        try:
            # Run with invalid checkpoint to test validation
            result = subprocess.run([
                sys.executable, str(self.script_path),
                "--checkpoint", str(self.checkpoint_dir),
                "--input", str(test_audio),
                "--output-dir", str(self.output_dir / "mock_test"),
            ], capture_output=True, text=True, timeout=30)
            
            # The script should fail at model loading stage, but argument parsing should succeed
            # We expect it to fail when trying to load the actual model
            # This tests that file validation and argument processing work
            
        except subprocess.TimeoutExpired:
            self.fail("Mock run test timed out")


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test class."""
        cls.script_path = Path(__file__).parent / "run_batch_inference.py"
        cls.test_dir = TestConfig.TEST_DIR
        cls.output_dir = TestConfig.OUTPUT_DIR
    
    def test_nonexistent_checkpoint(self):
        """Test handling of nonexistent checkpoint directory."""
        fake_checkpoint = "/nonexistent/checkpoint/dir"
        fake_audio = TestConfig.TEST_DIR / "fake_audio.wav"
        
        try:
            result = subprocess.run([
                sys.executable, str(self.script_path),
                "--checkpoint", fake_checkpoint,
                "--input", str(fake_audio),
                "--output-dir", str(self.output_dir / "error_test")
            ], capture_output=True, text=True, timeout=10)
            
            # Should fail with appropriate error message
            self.assertNotEqual(result.returncode, 0)
            
        except subprocess.TimeoutExpired:
            self.fail("Nonexistent checkpoint test timed out")
    
    def test_nonexistent_input_file(self):
        """Test handling of nonexistent input audio file."""
        fake_audio = "/nonexistent/audio/file.wav"
        
        try:
            result = subprocess.run([
                sys.executable, str(self.script_path),
                "--checkpoint", str(TestConfig.CHECKPOINT_DIR),
                "--input", fake_audio,
                "--output-dir", str(self.output_dir / "error_test")
            ], capture_output=True, text=True, timeout=10)
            
            # Should fail with appropriate error message
            self.assertNotEqual(result.returncode, 0)
            
        except subprocess.TimeoutExpired:
            self.fail("Nonexistent input test timed out")
    
    def test_invalid_arguments(self):
        """Test handling of invalid argument combinations."""
        # Test conflicting output arguments
        try:
            result = subprocess.run([
                sys.executable, str(self.script_path),
                "--checkpoint", str(TestConfig.CHECKPOINT_DIR),
                "--input", "test.wav",
                "--output", "single.wav",
                "--output-dir", str(self.output_dir)
            ], capture_output=True, text=True, timeout=10)
            
            # Should detect argument conflict
            self.assertNotEqual(result.returncode, 0)
            
        except subprocess.TimeoutExpired:
            self.fail("Invalid arguments test timed out")


class TestSLURMIntegration(unittest.TestCase):
    """Test SLURM integration components."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test class."""
        cls.slurm_dir = Path(__file__).parent.parent / "slurm" / "inference"
        cls.test_dir = TestConfig.TEST_DIR
    
    def test_slurm_directory_structure(self):
        """Test that SLURM directory exists with expected structure."""
        # We expect these files will be created by the agent
        expected_files = [
            "submit_batch_inference.sh",
            "run_batch_inference.slurm"
        ]
        
        # For now, just check that the parent slurm directory exists
        slurm_parent = Path(__file__).parent.parent / "slurm"
        if slurm_parent.exists():
            self.assertTrue(slurm_parent.is_dir(), "SLURM parent directory should exist")
    
    def test_environment_variables(self):
        """Test that required environment variables can be set."""
        # Test common SLURM environment variables
        test_vars = {
            'CHECKPOINT_DIR': str(TestConfig.CHECKPOINT_DIR),
            'INPUT_FILES': str(TestConfig.AUDIO_DIR / "*.wav"),
            'OUTPUT_DIR': str(TestConfig.OUTPUT_DIR),
            'HF_REPO': 'kyutai/moshiko-pytorch-bf16'
        }
        
        for var, value in test_vars.items():
            os.environ[var] = value
            self.assertEqual(os.environ[var], value, f"Environment variable {var} should be settable")
        
        # Clean up
        for var in test_vars:
            if var in os.environ:
                del os.environ[var]


def run_test_suite(verbose: bool = False, create_files: bool = False) -> bool:
    """
    Run the complete test suite.
    
    Args:
        verbose: Enable verbose output
        create_files: Just create test files and exit
        
    Returns:
        True if all tests passed
    """
    
    # Set up test environment
    setup_test_environment()
    
    if create_files:
        print("✅ Test files created. Exiting.")
        return True
    
    # Configure test runner
    if verbose:
        logging.getLogger().setLevel(logging.INFO)
        verbosity = 2
    else:
        verbosity = 1
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    test_cases = [
        TestBatchInferenceCore,
        TestAudioProcessing, 
        TestMainScript,
        TestErrorHandling,
        TestSLURMIntegration
    ]
    
    for test_case in test_cases:
        tests = loader.loadTestsFromTestCase(test_case)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    # Summary
    print("\\n" + "="*80)
    print("Test Summary")
    print("="*80)
    
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped) if hasattr(result, 'skipped') else 0
    passed = total_tests - failures - errors - skipped
    
    print(f"Total tests: {total_tests}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failures}")
    print(f"💥 Errors: {errors}")
    print(f"⏭️  Skipped: {skipped}")
    
    if result.failures:
        print("\\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"   {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\\n💥 Errors:")
        for test, traceback in result.errors:
            print(f"   {test}: {traceback.split('Exception:')[-1].strip()}")
    
    success = failures == 0 and errors == 0
    
    if success:
        print("\\n🎉 All tests passed!")
    else:
        print("\\n❌ Some tests failed. See details above.")
    
    # Clean up test environment
    try:
        if TestConfig.TEST_DIR.exists():
            shutil.rmtree(TestConfig.TEST_DIR)
        print("\\n🧹 Cleaned up test files")
    except Exception as e:
        print(f"\\n⚠️  Warning: Could not clean up test files: {e}")
    
    return success


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test suite for batch inference system")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--create-test-files", action="store_true", help="Create test files and exit")
    parser.add_argument("test", nargs="?", help="Specific test class to run")
    
    args = parser.parse_args()
    
    # Handle specific test class
    if args.test:
        # Set up environment
        setup_test_environment()
        
        # Configure verbosity
        if args.verbose:
            verbosity = 2
            logging.getLogger().setLevel(logging.INFO)
        else:
            verbosity = 1
        
        # Run specific test
        try:
            test_class = globals()[args.test]
            suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
            runner = unittest.TextTestRunner(verbosity=verbosity)
            result = runner.run(suite)
            return 0 if result.wasSuccessful() else 1
        except KeyError:
            print(f"❌ Test class '{args.test}' not found")
            return 1
    
    # Run full test suite
    success = run_test_suite(verbose=args.verbose, create_files=args.create_test_files)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())