#!/usr/bin/env python3
"""
Test that silence codes integration works correctly in paper metrics.
"""

import torch
from finetune.paper_metrics import PaperMetricsEvaluator


def test_silence_codes_integration():
    """Test that silence codes can be enabled/disabled via config"""
    
    # Mock MIMI encoder
    class MockMimi:
        def __init__(self):
            self.sample_rate = 24000
            self.channels = 1
            
        def encode(self, audio):
            B, C, T = audio.shape
            # Return 8 codebooks, T//1920 frames
            frames = T // 1920
            return torch.randint(0, 2048, (B, 8, frames))
    
    # Mock tokenizer
    class MockTokenizer:
        pass
    
    mimi = MockMimi()
    tokenizer = MockTokenizer()
    
    # Test 1: Disabled by default
    print("Test 1: Silence codes disabled by default")
    evaluator = PaperMetricsEvaluator(mimi, tokenizer, device='cpu', config={})
    assert evaluator.use_silence_codes == False
    assert evaluator.silence_cache is None
    print("✅ Disabled by default")
    
    # Test 2: Explicitly disabled
    print("\nTest 2: Explicitly disabled via config")
    evaluator = PaperMetricsEvaluator(mimi, tokenizer, device='cpu', config={'use_silence_codes': False})
    assert evaluator.use_silence_codes == False
    assert evaluator.silence_cache is None
    print("✅ Explicitly disabled works")
    
    # Test 3: Enabled via config
    print("\nTest 3: Enabled via config")
    evaluator = PaperMetricsEvaluator(mimi, tokenizer, device='cpu', config={'use_silence_codes': True})
    assert evaluator.use_silence_codes == True
    assert evaluator.silence_cache is not None
    print("✅ Enabled via config works")
    print(f"   Cache initialized: {type(evaluator.silence_cache).__name__}")
    
    # Test 4: Silence code generation
    print("\nTest 4: Silence code generation")
    silence = evaluator.silence_cache.get_silence_codes(
        target_shape=(1, 8, 100),
        mimi_model=mimi,
        device='cpu'
    )
    assert silence.shape == (1, 8, 100)
    print(f"✅ Generated silence codes with shape: {silence.shape}")
    
    # Test 5: Cache reuse
    print("\nTest 5: Cache reuse (should be instant)")
    silence2 = evaluator.silence_cache.get_silence_codes(
        target_shape=(1, 8, 100),
        mimi_model=mimi,
        device='cpu'
    )
    assert torch.equal(silence, silence2)
    print("✅ Cache reuse works correctly")
    
    print("\n" + "="*50)
    print("✅ ALL TESTS PASSED")
    print("="*50)
    print("\nSilence codes integration is working correctly!")
    print("To enable in your config, add:")
    print("  paper_metrics:")
    print("    use_silence_codes: true")


if __name__ == '__main__':
    test_silence_codes_integration()
