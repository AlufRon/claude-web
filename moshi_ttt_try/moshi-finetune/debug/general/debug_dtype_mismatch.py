#!/usr/bin/env python3
"""
Debug script to pinpoint exactly where dtype mismatch occurs in paper metrics.
Traces through the forward pass to find which matrix multiplication fails.
"""

import torch
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add necessary paths
sys.path.insert(0, str(Path(__file__).parent.parent / "moshi"))
sys.path.insert(0, str(Path(__file__).parent))

def check_tensor_dtype(tensor, name):
    """Check and log tensor dtype."""
    if isinstance(tensor, torch.Tensor):
        logger.info(f"  {name}: dtype={tensor.dtype}, shape={tensor.shape}, device={tensor.device}")
        return tensor.dtype
    return None

def check_module_dtypes(module, prefix=""):
    """Recursively check all parameter dtypes in a module."""
    logger.info(f"\n{'='*80}")
    logger.info(f"Checking module: {prefix or 'root'}")
    logger.info(f"{'='*80}")
    
    dtypes = {}
    for name, param in module.named_parameters(recurse=False):
        dtype = param.dtype
        dtypes[name] = dtype
        logger.info(f"  {name}: {dtype}")
    
    return dtypes

def hook_matmul_operations():
    """Add hooks to track matrix multiplication operations."""
    # Hook the @ operator on Tensor class
    original_tensor_matmul = torch.Tensor.__matmul__
    
    def traced_tensor_matmul(self, other):
        try:
            return original_tensor_matmul(self, other)
        except RuntimeError as e:
            if "must have the same dtype" in str(e):
                logger.error(f"\n{'='*80}")
                logger.error(f"DTYPE MISMATCH FOUND IN Tensor.__matmul__ (@)!")
                logger.error(f"{'='*80}")
                logger.error(f"Left tensor: dtype={self.dtype}, shape={self.shape}")
                logger.error(f"Right tensor: dtype={other.dtype}, shape={other.shape}")
                logger.error(f"Error: {e}")
                logger.error(f"{'='*80}")
                import traceback
                logger.error("Call stack:")
                logger.error(traceback.format_exc())
                raise
            raise
    
    torch.Tensor.__matmul__ = traced_tensor_matmul
    
    logger.info("✅ Matrix multiplication operations hooked for tracing")

def main():
    import yaml
    from run_paper_metrics_on_checkpoint import load_ttt_model
    from finetune.paper_metrics import PaperMetricsEvaluator
    
    logger.info("="*80)
    logger.info("DTYPE MISMATCH DEBUGGER")
    logger.info("="*80)
    
    # Install hooks first
    hook_matmul_operations()
    
    # Load config
    config_path = Path("example/moshi_7B_multilayer_with_ttt.yaml")
    logger.info(f"\n📄 Loading config from {config_path}")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Checkpoint path
    checkpoint_path = Path("/sise/eliyanac-group/ron_al/dailytalk_finetune_from_librilight8/checkpoints/checkpoint_000100/consolidated")
    
    logger.info(f"\n🏗️  Loading model from {checkpoint_path}")
    model, checkpoint_info = load_ttt_model(str(checkpoint_path), device="cuda")
    
    # Load MIMI encoder properly
    logger.info("\n🎤 Loading MIMI encoder...")
    mimi = checkpoint_info.get_mimi(device="cuda")
    logger.info(f"✅ MIMI loaded: {type(mimi).__name__}")
    
    # Check model dtypes
    logger.info("\n" + "="*80)
    logger.info("MODEL PARAMETER DTYPES")
    logger.info("="*80)
    
    # Check TTT layers specifically
    for layer_idx in [29, 30, 31]:
        layer = model.transformer.layers[layer_idx]
        logger.info(f"\n{'='*80}")
        logger.info(f"Layer {layer_idx}")
        logger.info(f"{'='*80}")
        
        # Check attention module
        if hasattr(layer, 'attn'):
            attn = layer.attn
            if hasattr(attn, 'ssm'):
                ssm = attn.ssm
                logger.info(f"Found SSM in attn: {type(ssm).__name__}")
                
                # Check if it has TTT parameters
                if hasattr(ssm, 'ttt'):
                    ttt = ssm.ttt
                    logger.info(f"TTT type: {type(ttt).__name__}")
                    
                    # Check multi-layer TTT parameters
                    if hasattr(ttt, 'layers'):
                        for i, mlp_layer in enumerate(ttt.layers):
                            logger.info(f"\n  MLP Layer {i}:")
                            for name, param in mlp_layer.named_parameters(recurse=False):
                                logger.info(f"    {name}: {param.dtype}")
                    
                    # Check normalization parameters
                    if hasattr(ttt, 'ttt_norm_weight'):
                        logger.info(f"\n  ttt_norm_weight: {ttt.ttt_norm_weight.dtype}")
                    if hasattr(ttt, 'ttt_norm_bias'):
                        logger.info(f"  ttt_norm_bias: {ttt.ttt_norm_bias.dtype}")
                    
                    # Check gating parameters
                    if hasattr(ttt, 'gate'):
                        logger.info(f"  gate.alpha: {ttt.gate.alpha.dtype}")
    
    # Create a minimal test case
    logger.info("\n" + "="*80)
    logger.info("RUNNING MINIMAL TEST CASE")
    logger.info("="*80)
    
    # Create evaluator
    evaluator = PaperMetricsEvaluator(
        mimi_encoder=mimi,
        interleaved_tokenizer=None,
        device="cuda",
        config=config.get('paper_metrics', {})
    )
    
    # Try to load and process ONE audio file
    test_audio = "/sise/eliyanac-group/ron_al/sblimp_data/sLM21_dataset/syntactic/test/aAAAZvtMsGyf.wav"
    
    if Path(test_audio).exists():
        logger.info(f"\n🎵 Testing with audio file: {test_audio}")
        
        try:
            # Encode audio
            logger.info("\n1️⃣  Encoding audio...")
            codes = evaluator._encode_audio(test_audio)
            logger.info(f"✅ Encoded codes: shape={codes.shape}, dtype={codes.dtype}")
            
            # Try to compute likelihood
            logger.info("\n2️⃣  Computing likelihood...")
            logger.info("This is where the error should occur...")
            
            likelihood = evaluator._compute_likelihood(model, codes)
            logger.info(f"✅ Likelihood computed: {likelihood}")
            
        except Exception as e:
            logger.error(f"\n❌ Error occurred: {e}")
            logger.error("\nThis should have triggered the detailed dtype mismatch trace above")
            return 1
    else:
        logger.error(f"Test audio file not found: {test_audio}")
        return 1
    
    logger.info("\n" + "="*80)
    logger.info("DEBUG COMPLETE")
    logger.info("="*80)
    return 0

if __name__ == "__main__":
    sys.exit(main())
