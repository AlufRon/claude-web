#!/usr/bin/env python3
"""
Test to replicate the exact streaming state problem:
"transformer.layers.31.seq_modeling_block.original_layer is already streaming!"

This test reproduces the issue where:
1. HybridStreamingTransformerLayer contains HybridSeqModelingBlock
2. HybridSeqModelingBlock contains original_layer (StreamingTransformerLayer)
3. The nested original_layer retains streaming state
4. Reset logic doesn't properly handle nested streaming modules
"""

import sys
import torch
import torch.nn as nn
import logging
from contextlib import ExitStack

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add Moshi to path
sys.path.append('/home/alufr/ttt_tests/moshi/moshi')

def test_streaming_state_problem():
    """Replicate the exact streaming state architecture problem"""
    logger.info("🔍 Replicating streaming state architecture problem...")
    
    try:
        # Import required classes
        from moshi.modules.transformer import StreamingTransformerLayer, _LayerState
        from moshi.modules.streaming import StreamingModule
        from moshi_ttt.hybrid_layer import HybridStreamingTransformerLayer, HybridSeqModelingBlock
        from moshi_ttt.config import TTTConfig
        
        logger.info("✅ All imports successful")
        
        # Create a minimal TTT config
        ttt_config = TTTConfig(
            model_dim=512,
            num_heads=8,
            mini_batch_size=64,
            ssm_layer="ttt_mlp"
        )
        
        # Create a mock original layer that behaves like StreamingTransformerLayer
        class MockStreamingTransformerLayer(StreamingModule[_LayerState]):
            def __init__(self, d_model=512):
                super().__init__()
                self.d_model = d_model
                self.linear = nn.Linear(d_model, d_model)
                
            def _init_streaming_state(self, batch_size: int) -> _LayerState:
                device = next(iter(self.parameters())).device
                return _LayerState(batch_size, device, offset_cpu=0)
                
            def forward(self, x):
                return self.linear(x)
                
            def _ff_block(self, x):
                return self.linear(x)
        
        # Create original layer
        original_layer = MockStreamingTransformerLayer()
        logger.info("✅ Created mock original layer")
        
        # Test 1: Create HybridSeqModelingBlock (this contains the original_layer)
        logger.info("🧪 Test 1: Creating HybridSeqModelingBlock...")
        seq_modeling_block = HybridSeqModelingBlock(original_layer, ttt_config)
        logger.info(f"   original_layer stored at: seq_modeling_block.original_layer")
        logger.info(f"   original_layer type: {type(seq_modeling_block.original_layer)}")
        
        # Test 2: Create HybridStreamingTransformerLayer (this contains seq_modeling_block)
        logger.info("🧪 Test 2: Creating HybridStreamingTransformerLayer...")
        hybrid_layer = HybridStreamingTransformerLayer(original_layer, ttt_config)
        logger.info(f"   seq_modeling_block stored at: hybrid_layer.seq_modeling_block")
        logger.info(f"   nested original_layer at: hybrid_layer.seq_modeling_block.original_layer")
        
        # Test 3: Verify that streaming detachment was applied
        logger.info("🧪 Test 3: Verifying streaming detachment...")
        detached = getattr(hybrid_layer.seq_modeling_block.original_layer, '_streaming_detached', False)
        logger.info(f"   Original layer detached: {detached}")
        
        if not detached:
            logger.error("   ❌ Streaming detachment not applied - this should be fixed!")
            return False
        else:
            logger.info("   ✅ Streaming detachment applied correctly")
        
        # Test 4: Put the top-level layer in streaming mode (should work now)
        logger.info("🧪 Test 4: Entering streaming mode with detached architecture...")
        with hybrid_layer.streaming(batch_size=1):
            logger.info("   ✅ hybrid_layer entered streaming mode")
            
            # Check streaming states at different levels
            def check_streaming_states():
                logger.info("   🔍 Checking streaming states:")
                logger.info(f"     hybrid_layer._streaming_state: {hybrid_layer._streaming_state is not None}")
                logger.info(f"     seq_modeling_block._streaming_state: {getattr(hybrid_layer.seq_modeling_block, '_streaming_state', 'N/A')}")
                logger.info(f"     nested original_layer._streaming_state: {hybrid_layer.seq_modeling_block.original_layer._streaming_state is not None}")
                
            check_streaming_states()
        
        # Test 5: Try to enter streaming mode again AFTER exiting (should work now with detachment)
        logger.info("🧪 Test 5: Attempting to enter streaming mode again after exiting...")
        try:
            with hybrid_layer.streaming(batch_size=1):
                logger.info("   ✅ Streaming works perfectly with detached architecture!")
                logger.info("   🎉 The streaming detachment fix WORKS!")
        except AssertionError as e:
            logger.error(f"   ❌ Still failing after detachment fix: {e}")
            
            # Check if it's the specific error we're seeing
            if "already streaming" in str(e):
                logger.error("   🚨 Detachment fix didn't work - still getting streaming conflicts!")
                
                # Detailed analysis of what's still streaming
                logger.info("   🔍 Detailed analysis of remaining streaming states:")
                for name, module in hybrid_layer.named_modules():
                    if hasattr(module, '_streaming_state') and module._streaming_state is not None:
                        logger.error(f"     STILL STREAMING: {name} - {type(module)}")
                    if hasattr(module, '_streaming_detached'):
                        logger.info(f"     DETACHED STATUS: {name} - detached={module._streaming_detached}")
                
                return False
        
        logger.info("🎉 Test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_architecture_analysis():
    """Analyze the exact architecture that's causing the problem"""
    logger.info("🏗️ Analyzing problematic architecture...")
    
    try:
        from moshi_ttt.hybrid_layer import HybridStreamingTransformerLayer
        
        # Show the exact architecture path that's failing
        logger.info("📋 Architecture hierarchy that leads to the error:")
        logger.info("   Model")
        logger.info("   └── transformer.layers[31] (HybridStreamingTransformerLayer)")
        logger.info("       └── seq_modeling_block (HybridSeqModelingBlock)")
        logger.info("           └── original_layer (StreamingTransformerLayer) ← PROBLEM HERE")
        logger.info("")
        logger.info("🎯 The error path: transformer.layers.31.seq_modeling_block.original_layer")
        logger.info("   This shows the nested streaming module structure that's causing conflicts")
        
    except Exception as e:
        logger.error(f"❌ Architecture analysis failed: {e}")

if __name__ == "__main__":
    logger.info("🚀 Starting streaming state problem replication test...")
    logger.info("="*80)
    
    # Test the architecture analysis
    test_architecture_analysis()
    logger.info("")
    
    # Test the actual streaming problem
    success = test_streaming_state_problem()
    
    logger.info("="*80)
    if success:
        logger.info("✅ Successfully replicated the streaming state problem!")
        logger.info("🔧 This test shows exactly why our reset logic isn't working.")
    else:
        logger.info("❌ Failed to replicate the problem - more investigation needed.")