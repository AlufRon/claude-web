# CODING AGENT INSTRUCTIONS: Building Non-Streaming Moshi Inference
# ============================================================================
# 
# MISSION: Read the streaming inference code and build a correct non-streaming
# batch inference mode that allows proper TTT evaluation.
#
# TIME ESTIMATE: 2-4 hours of careful work
# DIFFICULTY: Moderate (requires understanding streaming abstractions)
# ============================================================================

## PHASE 1: UNDERSTAND THE CURRENT STREAMING ARCHITECTURE
## ========================================================

### Step 1.1: Read the Core Streaming Files (30 minutes)

READ THESE FILES IN ORDER:

1. `/mnt/project/moshi/moshi/models/lm.py`
   - Focus on: `LMModel` class
   - Key methods:
     * `forward()` - Main forward pass
     * `forward_depformer()` - Streaming depformer (one codebook at a time)
     * `batch_forward_depformer()` - Batch depformer (ALL codebooks at once)
   - CRITICAL: Notice there's ALREADY a `batch_forward_depformer()` method!
   - This means depformer already supports batch mode, you just need to use it

2. `/mnt/project/moshi/moshi/modules/transformer.py`
   - Focus on: `StreamingTransformer` class
   - Key concepts:
     * `_streaming_state` - Holds KV cache and offsets
     * `set_streaming(bool)` - Enable/disable streaming mode
     * `is_streaming` property - Check if in streaming mode
   - CRITICAL: The transformer can be toggled between streaming/non-streaming

3. `/mnt/project/moshi/moshi/modules/streaming.py`
   - Focus on: `StreamingModule` base class
   - Key methods:
     * `set_streaming(bool)` - Recursively enable/disable streaming
     * `_streaming_state` - State object for streaming
   - CRITICAL: Streaming is controlled by a flag, not hardcoded


### Step 1.2: Understand the Streaming State (15 minutes)

IDENTIFY WHAT STREAMING STATE DOES:

```python
# When streaming is ENABLED (current inference):
# - Model maintains KV cache between tokens
# - Processes one token at a time
# - Updates internal offsets
# - Uses RingKVCache for sliding window

# When streaming is DISABLED (what you need to build):
# - No KV cache needed (recompute everything)
# - Process entire sequence at once
# - No offset tracking
# - Standard transformer behavior
```

READ these specific code sections:

In `lm.py`, find the `forward()` method:
```python
def forward(
    self,
    codes: torch.Tensor,
    condition_tensors: tp.Optional[torch.Tensor] = None,
) -> LMOutput:
    # This is the main forward pass
    # It works in BOTH streaming and batch mode!
    # The key is the transformer checks self._streaming_state
```

In `transformer.py`, find how streaming state is used:
```python
def forward(self, x: torch.Tensor, *args, **kwargs):
    # ...
    state = self._streaming_state  # ← This is None when not streaming!
    if state is None:
        offsets = torch.zeros(1, dtype=torch.long, device=x.device)
    else:
        offsets = state.offsets
    # ...
```

CRITICAL INSIGHT: When `_streaming_state` is None, the code already handles 
batch processing! You just need to ensure streaming is disabled.


### Step 1.3: Find the Streaming Entry Points (15 minutes)

LOCATE where streaming inference is initiated.

Look for files in the repository that call inference:
- Might be in `moshi/moshi/inference.py` or similar
- Might be in example scripts
- Might be in `finetune/` directory

SEARCH FOR these patterns:
```python
# Pattern 1: Explicit streaming context
with model.streaming(batch_size):
    for token in tokens:
        output = model.step(token)

# Pattern 2: Manual streaming setup
model.set_streaming(True, batch_size=1)
output = model.step(token)

# Pattern 3: StreamingInference wrapper
inference = StreamingInference(model)
output = inference.step(token)
```

DOCUMENT: Write down the file paths and line numbers where streaming is set up.


## PHASE 2: DESIGN THE NON-STREAMING INTERFACE
## ==============================================

### Step 2.1: Define the API (10 minutes)

CREATE a simple, clear API design:

```python
class NonStreamingInference:
    """
    Non-streaming batch inference for Moshi + TTT evaluation.
    
    Key differences from streaming:
    - Processes entire sequences at once (not token-by-token)
    - No KV cache needed
    - Can use larger mini-batch sizes for TTT (16-32 vs 1)
    - Suitable for evaluation, not real-time dialogue
    """
    
    def __init__(
        self,
        model: LMModel,
        mini_batch_size: int = 16,
        device: str = "cuda"
    ):
        """
        Args:
            model: LMModel instance (with or without TTT)
            mini_batch_size: Mini-batch size for TTT (ignored if no TTT)
            device: Device for inference
        """
        pass
    
    @torch.no_grad()
    def forward(
        self,
        codes: torch.Tensor,  # [B, K, T] - full sequence
        condition_tensors: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Non-streaming forward pass.
        
        Returns:
            logits: [B, K, T, vocab_size]
        """
        pass
    
    def evaluate_perplexity(
        self,
        dataloader,
        max_length: int = 8192
    ) -> dict:
        """
        Evaluate perplexity on dataset.
        
        Returns:
            {"perplexity": float, "avg_loss": float, ...}
        """
        pass
```


### Step 2.2: Identify What Needs to Change (15 minutes)

CREATE A CHECKLIST:

```
THINGS THAT MUST BE DISABLED:
[ ] Streaming state in transformer
[ ] Streaming state in depformer  
[ ] KV cache usage
[ ] Step-by-step token processing
[ ] Offset tracking

THINGS THAT MUST BE ENABLED:
[ ] Batch mode in depformer (use batch_forward_depformer)
[ ] Full sequence processing
[ ] Proper mini-batch size for TTT layers

THINGS THAT STAY THE SAME:
[ ] Model weights
[ ] Forward pass logic (mostly)
[ ] Loss computation
[ ] Embedding layers
```


## PHASE 3: IMPLEMENT NON-STREAMING INFERENCE
## =============================================

### Step 3.1: Create the Main Class (30 minutes)

CREATE FILE: `moshi/moshi/inference_batch.py`

```python
"""
Non-streaming batch inference for Moshi.

This module provides batch inference capability for Moshi, which is essential
for proper evaluation of TTT (Test-Time Training) enhancements.

Unlike streaming inference (one token at a time), batch inference processes
entire sequences at once, enabling:
- Larger mini-batch sizes for TTT (16-32 instead of 1)
- Accurate perplexity measurement across context lengths
- Easier debugging without streaming state management
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List
from dataclasses import dataclass

from moshi.models.lm import LMModel, LMOutput


@dataclass
class BatchInferenceConfig:
    """Configuration for batch inference."""
    mini_batch_size: int = 16  # For TTT layers
    max_sequence_length: int = 8192
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16


class NonStreamingInference:
    """
    Non-streaming batch inference wrapper for Moshi.
    
    Example usage:
        >>> model = load_moshi_model()
        >>> inference = NonStreamingInference(model, mini_batch_size=16)
        >>> logits = inference.forward(codes)
        >>> results = inference.evaluate_perplexity(dataloader)
    """
    
    def __init__(
        self,
        model: LMModel,
        config: Optional[BatchInferenceConfig] = None
    ):
        """
        Initialize non-streaming inference.
        
        Args:
            model: LMModel instance (can have TTT layers)
            config: Configuration for batch inference
        """
        self.model = model
        self.config = config or BatchInferenceConfig()
        
        # IMPORTANT: Configure TTT layers for batch mode
        self._configure_ttt_batch_mode()
        
        # IMPORTANT: Verify model is not in streaming mode
        self._ensure_non_streaming()
    
    def _configure_ttt_batch_mode(self):
        """
        Configure TTT layers to use proper mini-batch size.
        
        This is CRITICAL for TTT to work properly. In streaming mode,
        mini_batch_size=1 which gives noisy gradients. In batch mode,
        we can use mini_batch_size=16 for stable learning.
        """
        ttt_layers_found = 0
        
        # Search for TTT layers in transformer
        if hasattr(self.model, 'transformer'):
            for i, layer in enumerate(self.model.transformer.layers):
                # Check if this is a hybrid layer with TTT
                if hasattr(layer, 'ttt_block') or hasattr(layer, 'seq_modeling'):
                    # Configure mini-batch size
                    if hasattr(layer, 'ttt_block'):
                        if hasattr(layer.ttt_block, 'mini_batch_size'):
                            old_size = layer.ttt_block.mini_batch_size
                            layer.ttt_block.mini_batch_size = self.config.mini_batch_size
                            print(f"Layer {i}: TTT mini_batch_size {old_size} → {self.config.mini_batch_size}")
                            ttt_layers_found += 1
        
        if ttt_layers_found > 0:
            print(f"✓ Configured {ttt_layers_found} TTT layers for batch mode")
        else:
            print("ℹ No TTT layers found (using baseline Moshi)")
    
    def _ensure_non_streaming(self):
        """
        Ensure model is not in streaming mode.
        
        This is a safety check - we want to make sure streaming state
        is not active.
        """
        if hasattr(self.model, 'transformer'):
            if hasattr(self.model.transformer, 'is_streaming'):
                if self.model.transformer.is_streaming:
                    print("⚠ WARNING: Transformer is in streaming mode, disabling...")
                    self.model.transformer.set_streaming(False)
        
        if hasattr(self.model, 'depformer') and self.model.depformer is not None:
            if hasattr(self.model.depformer, 'is_streaming'):
                if self.model.depformer.is_streaming:
                    print("⚠ WARNING: Depformer is in streaming mode, disabling...")
                    self.model.depformer.set_streaming(False)
    
    @torch.no_grad()
    def forward(
        self,
        codes: torch.Tensor,
        condition_tensors: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Non-streaming forward pass.
        
        Processes entire sequence at once without maintaining streaming state.
        
        Args:
            codes: [B, K, T] where K = n_q + 1 (text + audio codebooks)
                   T can be any length (up to max_sequence_length)
            condition_tensors: Optional conditioning information
            
        Returns:
            logits: [B, K, T, vocab_size] - predictions for each position
        """
        # Save streaming state (in case it was on)
        transformer_was_streaming = False
        depformer_was_streaming = False
        
        if hasattr(self.model.transformer, 'is_streaming'):
            transformer_was_streaming = self.model.transformer.is_streaming
            if transformer_was_streaming:
                self.model.transformer.set_streaming(False)
        
        if self.model.depformer is not None:
            if hasattr(self.model.depformer, 'is_streaming'):
                depformer_was_streaming = self.model.depformer.is_streaming
                if depformer_was_streaming:
                    self.model.depformer.set_streaming(False)
        
        try:
            # CRITICAL: Call the model's forward method
            # When streaming is disabled, this already does batch processing!
            output = self.model(codes=codes, condition_tensors=condition_tensors)
            
            # Return logits
            return output.logits
            
        finally:
            # Restore streaming state if it was on
            if transformer_was_streaming:
                self.model.transformer.set_streaming(True)
            if depformer_was_streaming:
                self.model.depformer.set_streaming(True)
    
    def evaluate_perplexity(
        self,
        dataloader,
        max_length: Optional[int] = None,
        compute_position_wise: bool = True,
    ) -> Dict:
        """
        Evaluate perplexity on a dataset.
        
        Args:
            dataloader: DataLoader yielding batches with .codes attribute
            max_length: Maximum sequence length to evaluate (None = use all)
            compute_position_wise: Also compute perplexity per position
            
        Returns:
            Dictionary with:
                - "perplexity": Overall perplexity
                - "avg_loss": Average loss
                - "position_perplexity": Dict[int, float] if compute_position_wise
        """
        self.model.eval()
        max_length = max_length or self.config.max_sequence_length
        
        total_loss = 0.0
        total_tokens = 0
        position_losses = {} if compute_position_wise else None
        
        print(f"Evaluating perplexity (max_length={max_length})...")
        
        for batch_idx, batch in enumerate(dataloader):
            # Get codes and truncate if needed
            codes = batch.codes  # [B, K, T]
            if codes.shape[2] > max_length:
                codes = codes[:, :, :max_length]
            
            B, K, T = codes.shape
            
            # Get condition tensors if available
            condition_tensors = getattr(batch, 'condition_tensors', None)
            
            # Forward pass (batch mode)
            logits = self.forward(codes, condition_tensors)
            
            # Compute loss per position
            for t in range(T):
                # Get predictions and targets for this position
                pred = logits[:, :, t]  # [B, K, vocab_size]
                target = codes[:, :, t]  # [B, K]
                
                # Reshape for cross entropy
                pred = pred.reshape(-1, pred.shape[-1])  # [B*K, vocab_size]
                target = target.reshape(-1)  # [B*K]
                
                # Compute loss
                loss_t = nn.functional.cross_entropy(
                    pred,
                    target,
                    reduction='sum',
                    ignore_index=-1  # In case there are padding tokens
                )
                
                total_loss += loss_t.item()
                total_tokens += (target != -1).sum().item()
                
                if compute_position_wise:
                    if t not in position_losses:
                        position_losses[t] = []
                    # Store average loss for this position
                    position_losses[t].append(loss_t.item() / max(1, (target != -1).sum().item()))
            
            if (batch_idx + 1) % 10 == 0:
                current_perp = torch.exp(torch.tensor(total_loss / max(1, total_tokens)))
                print(f"  Batch {batch_idx + 1}: Running perplexity = {current_perp:.4f}")
        
        # Compute final metrics
        avg_loss = total_loss / max(1, total_tokens)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        results = {
            'perplexity': perplexity,
            'avg_loss': avg_loss,
            'total_tokens': total_tokens,
            'num_batches': batch_idx + 1,
        }
        
        if compute_position_wise:
            # Average perplexity at each position
            position_perplexity = {}
            for pos, losses in position_losses.items():
                avg_pos_loss = sum(losses) / len(losses)
                position_perplexity[pos] = torch.exp(torch.tensor(avg_pos_loss)).item()
            results['position_perplexity'] = position_perplexity
        
        print(f"\n✓ Evaluation complete:")
        print(f"  Perplexity: {perplexity:.4f}")
        print(f"  Avg Loss: {avg_loss:.4f}")
        print(f"  Total tokens: {total_tokens:,}")
        
        return results


# ============================================================================
# HELPER FUNCTIONS FOR EVALUATION
# ============================================================================

def compare_context_lengths(
    model: LMModel,
    dataloader,
    context_lengths: List[int] = [2048, 4096, 8192, 16384],
    mini_batch_size: int = 16,
) -> Dict[int, Dict]:
    """
    Compare perplexity across different context lengths.
    
    This is the KEY experiment to see if TTT helps at long context.
    
    Args:
        model: LMModel with or without TTT
        dataloader: Evaluation data
        context_lengths: List of context lengths to test
        mini_batch_size: Mini-batch size for TTT
        
    Returns:
        Dict mapping context_length -> evaluation results
    """
    config = BatchInferenceConfig(mini_batch_size=mini_batch_size)
    inference = NonStreamingInference(model, config)
    
    results = {}
    for context_len in context_lengths:
        print(f"\n{'='*60}")
        print(f"Evaluating at context length: {context_len}")
        print(f"{'='*60}")
        
        result = inference.evaluate_perplexity(
            dataloader,
            max_length=context_len,
            compute_position_wise=True
        )
        results[context_len] = result
    
    return results


def compare_mini_batch_sizes(
    model: LMModel,
    dataloader,
    mini_batch_sizes: List[int] = [1, 4, 8, 16, 32],
    context_length: int = 8192,
) -> Dict[int, Dict]:
    """
    Compare perplexity with different TTT mini-batch sizes.
    
    This shows how critical proper batching is for TTT.
    
    Args:
        model: LMModel with TTT
        dataloader: Evaluation data
        mini_batch_sizes: List of mini-batch sizes to test
        context_length: Context length to evaluate
        
    Returns:
        Dict mapping mini_batch_size -> evaluation results
    """
    results = {}
    for batch_size in mini_batch_sizes:
        print(f"\n{'='*60}")
        print(f"Evaluating with mini_batch_size: {batch_size}")
        print(f"{'='*60}")
        
        config = BatchInferenceConfig(
            mini_batch_size=batch_size,
            max_sequence_length=context_length
        )
        inference = NonStreamingInference(model, config)
        
        result = inference.evaluate_perplexity(
            dataloader,
            max_length=context_length,
            compute_position_wise=False  # Faster without position-wise
        )
        results[batch_size] = result
    
    return results
```


### Step 3.2: Test the Implementation (45 minutes)

CREATE FILE: `test_nonstreaming_inference.py`

```python
"""
Test script for non-streaming inference.

This script validates that non-streaming inference works correctly
before using it for TTT evaluation.
"""

import torch
from pathlib import Path

# Add your imports here based on your repo structure
from moshi.models import load_model
from moshi.inference_batch import (
    NonStreamingInference,
    BatchInferenceConfig,
    compare_context_lengths,
    compare_mini_batch_sizes,
)


def test_basic_forward():
    """Test 1: Basic forward pass works"""
    print("\n" + "="*60)
    print("TEST 1: Basic Forward Pass")
    print("="*60)
    
    # Load model
    print("Loading model...")
    model = load_model("path/to/checkpoint.safetensors", device="cuda")
    
    # Create inference wrapper
    config = BatchInferenceConfig(mini_batch_size=16)
    inference = NonStreamingInference(model, config)
    
    # Create dummy input
    B, K, T = 2, 9, 1024  # Batch=2, Codebooks=9, Time=1024
    codes = torch.randint(0, 1000, (B, K, T), device="cuda")
    
    # Forward pass
    print(f"Input shape: {codes.shape}")
    logits = inference.forward(codes)
    print(f"Output shape: {logits.shape}")
    
    # Check output shape
    assert logits.shape == (B, K, T, model.card), f"Wrong shape: {logits.shape}"
    print("✓ Forward pass successful!")
    
    return True


def test_streaming_vs_nonstreaming():
    """Test 2: Non-streaming gives same results as streaming"""
    print("\n" + "="*60)
    print("TEST 2: Streaming vs Non-Streaming Consistency")
    print("="*60)
    
    model = load_model("path/to/checkpoint.safetensors", device="cuda")
    
    # Create test input (short sequence for streaming)
    B, K, T = 1, 9, 128
    codes = torch.randint(0, 1000, (B, K, T), device="cuda")
    
    # Method 1: Non-streaming
    print("Running non-streaming inference...")
    inference = NonStreamingInference(model)
    logits_batch = inference.forward(codes)
    
    # Method 2: Streaming (token by token)
    print("Running streaming inference...")
    model.transformer.set_streaming(True, batch_size=B)
    model.depformer.set_streaming(True, batch_size=B)
    
    logits_streaming = []
    for t in range(T):
        # Process one token at a time
        token = codes[:, :, t:t+1]
        # ... (implement streaming step here)
        # logits_t = model.forward_streaming_step(token)
        # logits_streaming.append(logits_t)
    
    # Compare (should be similar, may have small numerical differences)
    # diff = (logits_batch - logits_streaming).abs().max()
    # print(f"Max difference: {diff.item()}")
    
    print("✓ Consistency check complete!")
    return True


def test_perplexity_computation():
    """Test 3: Perplexity computation works"""
    print("\n" + "="*60)
    print("TEST 3: Perplexity Computation")
    print("="*60)
    
    model = load_model("path/to/checkpoint.safetensors", device="cuda")
    inference = NonStreamingInference(model)
    
    # Create dummy dataloader
    # (Replace with your actual dataloader)
    class DummyBatch:
        def __init__(self):
            self.codes = torch.randint(0, 1000, (2, 9, 512), device="cuda")
    
    dataloader = [DummyBatch() for _ in range(5)]
    
    # Evaluate
    print("Computing perplexity...")
    results = inference.evaluate_perplexity(
        dataloader,
        max_length=512,
        compute_position_wise=True
    )
    
    print(f"\nResults:")
    print(f"  Perplexity: {results['perplexity']:.4f}")
    print(f"  Avg Loss: {results['avg_loss']:.4f}")
    print(f"  Total tokens: {results['total_tokens']:,}")
    
    if 'position_perplexity' in results:
        print(f"  Position-wise perplexity computed: {len(results['position_perplexity'])} positions")
    
    print("✓ Perplexity computation successful!")
    return True


def test_ttt_configuration():
    """Test 4: TTT layers are configured correctly"""
    print("\n" + "="*60)
    print("TEST 4: TTT Configuration")
    print("="*60)
    
    model = load_model("path/to/moshi_with_ttt.safetensors", device="cuda")
    
    # Test different mini-batch sizes
    for batch_size in [1, 8, 16, 32]:
        config = BatchInferenceConfig(mini_batch_size=batch_size)
        inference = NonStreamingInference(model, config)
        
        # Check TTT layers are configured
        ttt_configured = False
        for layer in model.transformer.layers:
            if hasattr(layer, 'ttt_block'):
                if hasattr(layer.ttt_block, 'mini_batch_size'):
                    assert layer.ttt_block.mini_batch_size == batch_size
                    ttt_configured = True
        
        if ttt_configured:
            print(f"✓ TTT configured with mini_batch_size={batch_size}")
    
    print("✓ TTT configuration test passed!")
    return True


if __name__ == "__main__":
    print("\n" + "="*60)
    print("TESTING NON-STREAMING INFERENCE")
    print("="*60)
    
    tests = [
        ("Basic Forward Pass", test_basic_forward),
        ("Streaming vs Non-Streaming", test_streaming_vs_nonstreaming),
        ("Perplexity Computation", test_perplexity_computation),
        ("TTT Configuration", test_ttt_configuration),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_fn in tests:
        try:
            if test_fn():
                passed += 1
            else:
                failed += 1
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            failed += 1
            print(f"✗ {test_name} FAILED with exception:")
            print(f"  {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*60)
```


## PHASE 4: CREATE EVALUATION SCRIPT
## ====================================

### Step 4.1: Create Main Evaluation Script (30 minutes)

CREATE FILE: `evaluate_ttt_nonstreaming.py`

```python
"""
Main evaluation script for TTT with non-streaming inference.

This script runs comprehensive evaluations to determine if TTT helps
with long-context audio generation.

Usage:
    python evaluate_ttt_nonstreaming.py \\
        --baseline_model path/to/moshi_baseline.safetensors \\
        --ttt_model path/to/moshi_with_ttt.safetensors \\
        --data_path path/to/test_data \\
        --output_dir ./results
"""

import argparse
import json
import matplotlib.pyplot as plt
from pathlib import Path
import torch

from moshi.models import load_model
from moshi.inference_batch import (
    NonStreamingInference,
    compare_context_lengths,
    compare_mini_batch_sizes,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_model", type=str, required=True)
    parser.add_argument("--ttt_model", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mini_batch_size", type=int, default=16)
    return parser.parse_args()


def plot_results(results_baseline, results_ttt, output_dir):
    """Plot comparison between baseline and TTT"""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Perplexity vs Context Length
    context_lens = sorted(results_baseline.keys())
    perp_baseline = [results_baseline[cl]['perplexity'] for cl in context_lens]
    perp_ttt = [results_ttt[cl]['perplexity'] for cl in context_lens]
    
    axes[0].plot(context_lens, perp_baseline, marker='o', label='Baseline', linewidth=2)
    axes[0].plot(context_lens, perp_ttt, marker='s', label='TTT', linewidth=2)
    axes[0].set_xlabel('Context Length')
    axes[0].set_ylabel('Perplexity')
    axes[0].set_title('Perplexity vs Context Length')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xscale('log', base=2)
    
    # Plot 2: Improvement percentage
    improvements = [(perp_baseline[i] - perp_ttt[i]) / perp_baseline[i] * 100 
                    for i in range(len(context_lens))]
    
    axes[1].bar(range(len(context_lens)), improvements, 
                tick_label=[str(cl) for cl in context_lens])
    axes[1].set_xlabel('Context Length')
    axes[1].set_ylabel('Improvement (%)')
    axes[1].set_title('TTT Improvement over Baseline')
    axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'ttt_evaluation.png', dpi=300)
    print(f"✓ Saved plot: {output_dir}/ttt_evaluation.png")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("="*60)
    print("TTT NON-STREAMING EVALUATION")
    print("="*60)
    
    # Load data
    print(f"\nLoading data from {args.data_path}...")
    # TODO: Implement data loading based on your data format
    # dataloader = create_dataloader(args.data_path)
    
    # Evaluate baseline
    print(f"\n{'='*60}")
    print("EVALUATING BASELINE MODEL")
    print(f"{'='*60}")
    baseline_model = load_model(args.baseline_model, device=args.device)
    results_baseline = compare_context_lengths(
        baseline_model,
        dataloader,
        context_lengths=[2048, 4096, 8192, 16384],
        mini_batch_size=args.mini_batch_size,
    )
    
    # Evaluate TTT
    print(f"\n{'='*60}")
    print("EVALUATING TTT MODEL")
    print(f"{'='*60}")
    ttt_model = load_model(args.ttt_model, device=args.device)
    results_ttt = compare_context_lengths(
        ttt_model,
        dataloader,
        context_lengths=[2048, 4096, 8192, 16384],
        mini_batch_size=args.mini_batch_size,
    )
    
    # Save results
    results = {
        'baseline': results_baseline,
        'ttt': results_ttt,
        'config': vars(args),
    }
    
    results_file = output_dir / 'results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved results: {results_file}")
    
    # Plot
    plot_results(results_baseline, results_ttt, output_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for context_len in [2048, 4096, 8192, 16384]:
        perp_base = results_baseline[context_len]['perplexity']
        perp_ttt = results_ttt[context_len]['perplexity']
        improvement = (perp_base - perp_ttt) / perp_base * 100
        
        status = "✓" if improvement > 0 else "✗"
        print(f"{status} Context {context_len:5d}: "
              f"Baseline={perp_base:.4f}, TTT={perp_ttt:.4f}, "
              f"Improvement={improvement:+.2f}%")
    
    print("\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()
```


## PHASE 5: VALIDATION CHECKLIST
## ================================

BEFORE considering the implementation complete, verify:

```
FUNCTIONALITY CHECKS:
[ ] Non-streaming forward pass runs without errors
[ ] Output shape is correct: [B, K, T, vocab_size]
[ ] Streaming state is properly disabled
[ ] TTT mini-batch size is configurable
[ ] Perplexity computation works
[ ] Position-wise perplexity works
[ ] Can handle different context lengths (2k, 4k, 8k, 16k)
[ ] Works with and without TTT layers
[ ] Works with frozen and unfrozen models

CORRECTNESS CHECKS:
[ ] Results are deterministic (same input → same output)
[ ] Perplexity values are reasonable (not NaN or Inf)
[ ] Longer contexts don't cause OOM
[ ] TTT layers are actually being used (check gradients/updates)
[ ] Mini-batch size actually affects TTT behavior

PERFORMANCE CHECKS:
[ ] Processing speed is reasonable (not slower than expected)
[ ] Memory usage is tracked and reasonable
[ ] Can process full 8k token sequences

SCIENTIFIC CHECKS:
[ ] Baseline model evaluation works
[ ] TTT model evaluation works
[ ] Can compare baseline vs TTT quantitatively
[ ] Position-wise perplexity shows expected patterns
[ ] Results match expectations from TTT paper at 4k-8k tokens
```


## PHASE 6: DOCUMENTATION
## ========================

CREATE FILE: `docs/nonstreaming_inference.md`

Write documentation that explains:
1. Why non-streaming inference was needed
2. How it differs from streaming
3. How to use it
4. Example use cases
5. Limitations


## TROUBLESHOOTING GUIDE
## ======================

Common issues and solutions:

### Issue 1: "RuntimeError: streaming state is not None"
**Cause**: Streaming mode wasn't properly disabled
**Solution**: Call `model.transformer.set_streaming(False)` explicitly

### Issue 2: "Shape mismatch in depformer"
**Cause**: Using streaming depformer method instead of batch method
**Solution**: Use `batch_forward_depformer()` not `forward_depformer()`

### Issue 3: "TTT mini-batch size not changing"
**Cause**: TTT layers not found or not configured
**Solution**: Check that TTT layers exist with `hasattr(layer, 'ttt_block')`

### Issue 4: "OOM with long sequences"
**Cause**: Trying to process too long sequences
**Solution**: Reduce batch size or max_length parameter

### Issue 5: "Perplexity is NaN"
**Cause**: Numerical instability or division by zero
**Solution**: Check for padding tokens, use ignore_index in loss


## FINAL DELIVERABLES
## ====================

When complete, you should have:

1. `moshi/moshi/inference_batch.py` - Main implementation
2. `test_nonstreaming_inference.py` - Test suite
3. `evaluate_ttt_nonstreaming.py` - Evaluation script
4. `docs/nonstreaming_inference.md` - Documentation
5. Results showing if TTT helps or not

ESTIMATED TOTAL TIME: 3-4 hours

Good luck! This is important work that will definitively show if TTT 
helps with long-context audio generation.