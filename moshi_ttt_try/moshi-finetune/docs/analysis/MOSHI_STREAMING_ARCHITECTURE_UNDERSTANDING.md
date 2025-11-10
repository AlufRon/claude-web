# Moshi Streaming Architecture: Complete Understanding

## Executive Summary

Moshi implements a sophisticated streaming architecture that enables real-time audio processing with a **3000-token sliding window context**. The key insight is that Moshi uses **streaming inference** (token-by-token with KV caching) rather than batch processing, which fundamentally changes how we must implement LibriLight evaluation.

## Core Architecture Components

### 1. StreamingModule Base Class (`/home/alufr/ttt_tests/moshi/moshi/moshi/modules/streaming.py`)

**Purpose**: Foundation for all streaming-capable components in Moshi.

**Key Concepts**:
- **Streaming State**: Each module has `_streaming_state` (None when not streaming)
- **Context Manager**: `with module.streaming(batch_size):` enters streaming mode
- **Automatic Propagation**: Streaming mode propagates to all child modules
- **Detached Streaming**: `set_streaming_detached(True)` prevents inheritance from parent
- **State Management**: Automatic state reset on context exit

**Critical Methods**:
```python
def streaming(self, batch_size: int) -> ExitStack:
    """Context manager to enter streaming mode. Reset streaming state on exit."""
    
def streaming_forever(self, batch_size: int):
    """Enters streaming mode permanently (no auto-exit)"""
    self.streaming(batch_size).__enter__()

def reset_streaming(self, reset_mask: torch.Tensor | None = None):
    """Reset the streaming state for specific batch items"""
```

### 2. LMModel vs LMGen Distinction (`/home/alufr/ttt_tests/moshi/moshi/moshi/models/lm.py`)

**LMModel** (Lines 49-514):
- **Purpose**: Core transformer for training and batch inference
- **API**: `forward(codes, condition_tensors)` → LMOutput
- **Usage**: Training and batch evaluation
- **Inherits**: `StreamingContainer` (basic streaming support)

**LMGen** (Lines 550-844):
- **Purpose**: Streaming generation wrapper around LMModel  
- **API**: `step(input_tokens)` → generated tokens
- **Usage**: Real-time inference and streaming evaluation
- **Key Feature**: Manages KV cache, delays, and token generation
- **Inherits**: `StreamingModule[_LMGenState]`

**Critical Architecture Insight**:
```python
# In LMGen._init_streaming_state() (line 646):
state.exit_stack.enter_context(self.lm_model.streaming(batch_size))
```
→ **LMGen automatically puts LMModel into streaming mode!**

### 3. Streaming Context Window Management

**Key Discovery**: Moshi uses a **3000-token sliding window** with KV caching:

**RingKVCache** (`/home/alufr/ttt_tests/moshi/moshi/moshi/modules/transformer.py:187-280`):
- **Capacity**: Set to `context` parameter (3000 tokens)
- **Sliding Window**: Overwrites oldest tokens when capacity exceeded
- **Efficient**: Avoids quadratic attention complexity
- **Memory**: Fixed memory footprint regardless of sequence length

**StreamingMultiheadAttention** (Lines 328-574):
- **Context Limitation**: `self.context` parameter limits attention span
- **Causal Masking**: Only attends to previous tokens within context window
- **Position Tracking**: Maintains absolute positions despite sliding window

### 4. Real-World Streaming Usage Examples

**Server.py** (`/home/alufr/ttt_tests/moshi/moshi/moshi/server.py:59-60`):
```python
self.mimi.streaming_forever(1)        # Permanent streaming mode
self.lm_gen.streaming_forever(1)      # No context manager needed
```

**Run_inference.py** (`/home/alufr/ttt_tests/moshi/moshi/moshi/run_inference.py:89-90`):
```python
self.mimi.streaming_forever(batch_size)
self.lm_gen.streaming_forever(batch_size)
```

**Token-by-Token Processing** (server.py:137-138):
```python
tokens = self.lm_gen.step(codes[:, :, c: c + 1])  # One timestep at a time
if tokens is None:
    continue  # Handle delays in token generation
```

## Key Insights for LibriLight Evaluation Fix

### Problem: Why Current Evaluation Fails

1. **Memory Issue**: LibriLight sequences (24k tokens) → 24k×24k attention = ~37GB
2. **Wrong API**: Using `model.forward()` (training API) instead of `lm_gen.step()` (streaming API)
3. **Batch Processing**: Processing entire sequences instead of token-by-token streaming
4. **No Context Respect**: Not respecting 3000-token context window limitation

### Solution: Streaming Evaluation Architecture

**Core Principle**: Use LMGen.step() for token-by-token evaluation that respects context window.

**Streaming Evaluation Flow**:
1. **Setup**: `lm_gen.streaming_forever(1)` (permanent streaming mode)
2. **Token Loop**: For each audio token in sequence:
   ```python
   codes_input = torch.zeros(1, 17, 1, device=device, dtype=torch.long)
   codes_input[:, 1:9] = current_audio_codes  # Feed audio codes
   tokens = lm_gen.step(codes_input)          # Get predictions
   ```
3. **Loss Computation**: Compare predictions with targets for each position
4. **Memory**: Fixed ~3000-token memory usage regardless of sequence length

### API Corrections from My Failed Attempts

**❌ Wrong (my failed attempts)**:
```python
model.step(inp)                    # LMModel has no 'step' method
model(codes=codes_input)           # Training API, not streaming
```

**✅ Correct (based on actual Moshi code)**:
```python
tokens = lm_gen.step(codes_input)  # LMGen has step() method for streaming
```

## TTT Integration Implications

### Why TTT Helps Moshi

1. **Context Limitation**: Moshi's 3000-token sliding window loses long-term context
2. **TTT Advantage**: TTT accumulates patterns in hidden weights indefinitely
3. **Memory Efficiency**: TTT updates weights (small) vs storing KV cache (large)
4. **Complementary**: TTT provides long-term memory, attention provides short-term

### Training vs Evaluation Modes

**Training Mode**:
- Uses `LMModel.forward()` with full sequences
- Batch processing for efficiency
- No streaming state needed

**Evaluation/Inference Mode**:
- Uses `LMGen.step()` with single tokens
- Streaming state maintained across calls
- Respects context window limitations

## Critical Implementation Requirements

### For Streaming Evaluation Fix

1. **Use LMGen, not LMModel**: `lm_gen.step()` is the correct streaming API
2. **Permanent Streaming**: Use `streaming_forever()` not temporary contexts
3. **Single Token Processing**: Feed one audio token at a time
4. **Proper Input Format**: 
   ```python
   codes_input = torch.zeros(1, 17, 1, device=device, dtype=torch.long)
   codes_input[:, 1:9] = current_audio_codes  # 8 audio codebooks
   ```
5. **Handle Delays**: `tokens = lm_gen.step()` may return None due to delays
6. **Reset Between Sequences**: `lm_gen.reset_streaming()` between evaluation samples

### For TTT Integration

1. **Layer Replacement**: Replace StreamingTransformerLayer with HybridStreamingTransformerLayer
2. **Streaming Compatibility**: TTT layer must inherit from StreamingModule
3. **State Management**: TTT weights need proper streaming state handling
4. **Context Window**: TTT should respect Moshi's streaming context when training

## Moshi Model Structure

**Input Flow**: Audio → Mimi (compression) → codes → LMModel → text + audio predictions
**Streaming Flow**: Single audio token → LMGen.step() → single prediction token

**Layer Hierarchy**:
```
LMModel (training)
├── transformer: StreamingTransformer
│   ├── layers: List[StreamingTransformerLayer]  ← TTT integration point
│   │   ├── self_attn: StreamingMultiheadAttention (KV cache)
│   │   ├── norm1, norm2: LayerNorm
│   │   └── linear1, linear2: FFN
│   └── positional_embedding: sin/rope
└── depformer: StreamingTransformer (parallel prediction)

LMGen (inference)
└── lm_model: LMModel (wrapped in streaming context)
```

## Memory Analysis

**Batch Processing (current LibriLight)**:
- Memory: O(seq_len²) = 24k² = ~37GB for attention
- Speed: Fast (parallel)
- Context: Full sequence context

**Streaming Processing (correct approach)**:
- Memory: O(context_size) = 3k tokens = ~200MB fixed
- Speed: Slower (sequential)  
- Context: Sliding 3k window + TTT indefinite memory

## Next Steps

Based on this understanding, the LibriLight evaluation fix requires:

1. **Replace batch processing with streaming**: Use `lm_gen.step()` token-by-token
2. **Proper model setup**: Use `lm_gen.streaming_forever(1)` 
3. **Correct input format**: Single audio tokens in proper tensor format
4. **Handle generation delays**: Account for None returns from step()
5. **Sequence reset**: Reset streaming state between samples

This will enable:
- ✅ Memory-efficient evaluation (fixed 3k context)
- ✅ TTT vs baseline comparison 
- ✅ Meaningful LibriLight metrics
- ✅ Proper streaming behavior