#!/usr/bin/env python3
"""
Step 3.1.1 Test: File imports without errors
Test that hybrid_layer.py imports successfully
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
# Add original Moshi to path
sys.path.append('/home/alufr/ttt_tests/moshi/moshi')

def test_hybrid_layer_imports():
    """Test that hybrid_layer.py file imports without errors"""
    print("🧪 Testing hybrid_layer.py imports...")
    
    try:
        # Test Moshi import
        print("Testing Moshi import...")
        from moshi.modules.transformer import StreamingTransformerLayer
        print("✅ Moshi StreamingTransformerLayer import successful")
        
        # Test our TTT imports
        print("Testing our TTT imports...")
        from moshi_ttt.config import TTTConfig
        from moshi_ttt.utils import SequenceMetadata
        from moshi_ttt.models.ssm.ttt_layer import TTTMLP
        print("✅ TTT imports successful")
        
        # Test the hybrid layer import
        print("Testing hybrid layer import...")
        from moshi_ttt.hybrid_layer import HybridStreamingTransformerLayer, HybridSeqModelingBlock
        print("✅ Hybrid layer import successful")
        
        print("✅ All imports successful!")
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_hybrid_layer_imports()
    if success:
        print(f"\n🎉 Step 3.1.1 SUCCESS: hybrid_layer.py imports work!")
        print("Next: Step 3.1.2 - Test class instantiation")
        sys.exit(0)
    else:
        print(f"\n💥 Step 3.1.1 FAILED: Import issues need to be resolved")
        sys.exit(1)
