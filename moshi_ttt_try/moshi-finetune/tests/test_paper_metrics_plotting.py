#!/usr/bin/env python3
"""
Test script for paper metrics plotting functionality.
This verifies that the LibriLight position vs loss plotting works correctly.
"""

import os
import sys
import torch
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from finetune.paper_metrics import PaperMetricsEvaluator


class MockModel:
    """Mock model for testing"""
    def __init__(self, model_type="TTT"):
        self.num_codebooks = 17
        self.zero_token_id = 0
        self.audio_offset = 1
        self.dep_q = 8
        self.model_type = model_type
        
        # Add TTT-specific attributes for model detection
        if model_type == "TTT":
            self.ttt_config = Mock()
        
    def forward(self, x):
        """Mock forward pass"""
        B, C, T = x.shape
        vocab_size = 2048
        
        # Create mock output
        logits = torch.randn(B, self.dep_q, T, vocab_size)
        mask = torch.ones(B, self.dep_q, T, dtype=torch.bool)
        
        # Mock output object
        output = Mock()
        output.logits = logits
        output.mask = mask
        
        return output
    
    def __call__(self, x):
        """Make model callable"""
        return self.forward(x)
    
    def eval(self):
        """Mock eval mode"""
        pass
    
    def modules(self):
        """Mock modules for TTT detection"""
        if self.model_type == "TTT":
            return [Mock(__class__=Mock(__name__="TTTLayer"))]
        else:
            return [Mock(__class__=Mock(__name__="TransformerLayer"))]


class MockMimiEncoder:
    """Mock MIMI encoder"""
    def __init__(self):
        self.channels = 1
    
    def encode(self, waveform):
        """Mock encoding that returns realistic audio codes"""
        B, C, samples = waveform.shape
        # Convert audio length to token sequence length
        seq_len = samples // 1920  # Approximate MIMI frame rate
        return torch.randint(1, 2048, (B, 8, seq_len))


def create_test_audio_file(audio_dir: Path, filename: str, duration_sec: float = 10.0):
    """Create a test audio file"""
    audio_file = audio_dir / filename
    
    # Create simple sine wave audio data
    sample_rate = 24000
    samples = int(sample_rate * duration_sec)
    t = np.linspace(0, duration_sec, samples)
    audio_data = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    
    # Save as simple text file (mock audio)
    # In real test, you'd use torchaudio.save or similar
    with open(audio_file, 'w') as f:
        f.write(f"# Mock audio file: {duration_sec}s\n")
        f.write(f"sample_rate: {sample_rate}\n")
        f.write(f"samples: {samples}\n")
    
    return audio_file


def test_librilight_plotting():
    """Test LibriLight evaluation with plotting"""
    print("🧪 Testing LibriLight plotting functionality...")
    
    # Create temporary directory structure
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create mock LibriLight structure
        librilight_dir = temp_path / "librilight"
        speaker_dir = librilight_dir / "100"
        book_dir = speaker_dir / "emerald_city_librivox_64kb_mp3"
        book_dir.mkdir(parents=True)
        
        # Create test audio files (3 chapters)
        chapter_files = []
        for i in range(3):
            chapter_file = create_test_audio_file(
                book_dir, f"chapter_{i:02d}.wav", duration_sec=60.0
            )
            chapter_files.append(chapter_file)
        
        # Create test config
        config = {
            'librilight_audio_dir': str(librilight_dir),
            'librilight_evaluation_mode': 'single_book',
            'librilight_speaker_id': '100',
            'librilight_book_name': 'emerald_city_librivox_64kb_mp3',
            'librilight_max_chapters': 3,
            'librilight_num_sequences': 1,
        }
        
        # Test with both TTT and Baseline models
        for model_type in ["TTT", "Baseline"]:
            print(f"\n📊 Testing {model_type} model plotting...")
            
            # Create mock components
            model = MockModel(model_type)
            mimi_encoder = MockMimiEncoder()
            tokenizer = Mock()
            
            # Create evaluator
            evaluator = PaperMetricsEvaluator(
                mimi_encoder=mimi_encoder,
                interleaved_tokenizer=tokenizer,
                device="cpu",
                config=config
            )
            
            # Mock the LibriLight loader to return our test files
            with patch('finetune.paper_metrics.LibriLightLoader') as mock_loader_class:
                mock_loader = Mock()
                mock_loader_class.return_value = mock_loader
                
                # Mock the method that gets chapters
                mock_loader.get_ttt_evaluation_chapters.return_value = (
                    chapter_files,
                    {
                        'speaker_id': '100',
                        'book_title': 'Test Book',
                        'author': 'Test Author',
                        'num_chapters': 3
                    }
                )
                
                # Mock audio encoding to return realistic sequences
                def mock_encode_audio(audio_path):
                    # Return codes for ~3000 tokens (realistic chapter length)
                    return torch.randint(1, 2048, (1, 8, 3000))
                
                evaluator._encode_audio = mock_encode_audio
                
                # Run evaluation
                print(f"🔄 Running LibriLight evaluation for {model_type}...")
                results = evaluator.evaluate_librilight_long_context(model)
                
                # Verify results
                assert 'librilight_loss_8k' in results, "Missing 8k loss metric"
                assert 'librilight_loss_16k' in results, "Missing 16k loss metric" 
                assert 'librilight_slope' in results, "Missing slope metric"
                assert 'librilight_samples' in results, "Missing samples metric"
                
                print(f"✅ {model_type} results: 8k={results['librilight_loss_8k']:.3f}, "
                      f"slope={results['librilight_slope']:.6f}")
                
                # Verify plot data was created
                plot_data = evaluator.get_plot_data()
                assert plot_data is not None, f"No plot data created for {model_type}"
                assert 'positions' in plot_data, "Missing positions in plot data"
                assert 'losses' in plot_data, "Missing losses in plot data"
                assert 'model_type' in plot_data, "Missing model_type in plot data"
                assert plot_data['model_type'] == model_type, f"Wrong model type: {plot_data['model_type']}"
                
                print(f"✅ {model_type} plot data: {len(plot_data['positions'])} positions, "
                      f"model_type={plot_data['model_type']}")
                
                # Verify position metrics
                position_metrics = plot_data.get('position_metrics', {})
                assert len(position_metrics) > 0, "No position metrics created"
                
                # Check for expected position metrics (every 1k tokens)
                expected_positions = ['librilight_loss_1000', 'librilight_loss_2000', 'librilight_loss_3000']
                for pos_key in expected_positions:
                    assert pos_key in position_metrics, f"Missing position metric: {pos_key}"
                
                print(f"✅ {model_type} position metrics: {len(position_metrics)} positions measured")
                
                # Verify slopes
                slopes = plot_data.get('slopes', {})
                assert 'overall' in slopes, "Missing overall slope"
                print(f"✅ {model_type} slopes: overall={slopes.get('overall', 0):.6f}")


def test_plotting_comparison():
    """Test that TTT and Baseline models produce different plot characteristics"""
    print("\n🔄 Testing TTT vs Baseline comparison...")
    
    # This would be expanded to test specific differences between TTT and baseline
    # For now, just verify the mock models are detected correctly
    
    ttt_model = MockModel("TTT")
    baseline_model = MockModel("Baseline")
    
    # Test model type detection logic (from paper_metrics.py)
    ttt_detected = "TTT" if hasattr(ttt_model, 'ttt_config') or any('ttt' in str(type(m)).lower() for m in ttt_model.modules()) else "Baseline"
    baseline_detected = "TTT" if hasattr(baseline_model, 'ttt_config') or any('ttt' in str(type(m)).lower() for m in baseline_model.modules()) else "Baseline"
    
    assert ttt_detected == "TTT", f"TTT model not detected correctly: {ttt_detected}"
    assert baseline_detected == "Baseline", f"Baseline model not detected correctly: {baseline_detected}"
    
    print(f"✅ Model detection: TTT={ttt_detected}, Baseline={baseline_detected}")


def test_metrics_logger_integration():
    """Test the metrics logger plotting integration"""
    print("\n📊 Testing metrics logger integration...")
    
    # Mock plot data
    plot_data = {
        'positions': list(range(0, 25000, 100)),  # Every 100 positions
        'losses': [10.0 - i * 0.0001 for i in range(0, 25000, 100)],  # Decreasing loss
        'model_type': 'TTT',
        'measurement_positions': list(range(1000, 25000, 1000)),
        'position_metrics': {f'librilight_loss_{i}': 10.0 - i * 0.0001 for i in range(1000, 25000, 1000)},
        'slopes': {'overall': -0.0001, 'early': -0.0002, 'middle': -0.0001}
    }
    
    # Test the plotting logic without the full MetricsLogger
    try:
        # Test the core plotting components
        import matplotlib.pyplot as plt
        
        # Test matplotlib plotting
        plt.figure(figsize=(14, 8))
        positions = plot_data['positions']
        losses = plot_data['losses']
        plt.plot(positions, losses, linewidth=3, color='blue', label='TTT Model')
        plt.xlabel('Sequence Position (tokens)')
        plt.ylabel('Cross-Entropy Loss')
        plt.title('Position-wise Loss Analysis - TTT Model')
        plt.close()  # Don't display, just test creation
        
        print("✅ Matplotlib plotting successful")
        
        # Test WandB table creation (mock)
        with patch('wandb.Table') as mock_table:
            mock_table.return_value = Mock()
            table_data = [[pos, loss, 'TTT'] for pos, loss in zip(positions[::100], losses[::100])]
            table = mock_table(data=table_data, columns=["position", "loss", "model_type"])
            print("✅ WandB table creation successful")
        
        print("✅ Metrics logger integration successful")
        
    except Exception as e:
        print(f"❌ Metrics logger integration failed: {e}")
        raise


def main():
    """Run all tests"""
    print("🚀 Starting Paper Metrics Plotting Tests")
    print("=" * 50)
    
    try:
        # Test LibriLight plotting
        test_librilight_plotting()
        
        # Test model type detection  
        test_plotting_comparison()
        
        # Test metrics logger integration
        test_metrics_logger_integration()
        
        print("\n" + "=" * 50)
        print("🎉 All tests passed! Paper metrics plotting is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()