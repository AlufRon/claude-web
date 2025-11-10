# TTT Attention Context: How Global Memory Works with Local Attention

**Date**: 2025-11-01
**Question**: If we set `context=100` tokens during training with long sequences (e.g., 1000 tokens), can TTT's persistent states still learn the entire sequence?
**Answer**: **YES** - TTT sees the entire sequence through its persistent weight updates, even though it receives attention-filtered features.

---

## Executive Summary

**Key Insight**: TTT persistent states (W1, W2, b1, b2) act as **learned memory** that accumulates information across the entire sequence, similar to RNN hidden states but updated via gradient descent instead of gating functions.

**The Magic**: When you do `output = XQ @ W`, you're not just processing current input - you're multiplying current features by weights that **contain compressed information from all previous tokens**.

---

## Architecture Flow: Both Video-DiT and Moshi-TTT

### Sequential Processing (NOT Parallel)

```
Input [B, seq_len, D]
    ↓
Attention (limited context) → Output [B, seq_len, D]
    ↓
TTT (sequential processing) → Output [B, seq_len, D]
    ↓
Feedforward
```

**CRITICAL**: TTT processes attention output, **NOT** raw input. This is true for **BOTH** Video-DiT and Moshi-TTT.

### Verification from Video-DiT Code

From `ttt-video-dit/ttt/models/cogvideo/dit.py:268-276`:

```python
def forward(self, vid_emb, text_emb, seq_metadata):
    # Step 1: Attention with limited context
    output = self._attn_forward(vid_emb, text_emb, seq_metadata)

    # Step 2: TTT receives attention output (NOT raw input)
    output = self._ssm_forward(output, seq_metadata)

    return output
```

**This is sequential, not parallel!**

---

## How TTT Sees the Entire Sequence Despite Limited Attention

### Example: 1000-token sequence, context=100

#### Position 0-100 (First 100 tokens)

```
Attention Input:  tokens [0-100]
Attention Output: features_0_100 (contains info from tokens 0-100) ✓

TTT Input:        features_0_100
TTT Weights:      W₀, b₀ (initial random weights)
TTT Computation:
    output = features_0_100 @ W₀ + ...

TTT Weight Update (via gradient descent on self-supervised loss):
    W₀ → W₁₀₀
    b₀ → b₁₀₀

    ↑ Weights now "remember" tokens 0-100!
```

**Key Point**: After processing tokens 0-100, the weights W₁₀₀ and b₁₀₀ contain a **compressed representation** of those tokens.

#### Position 101 (Token 101)

```
Attention Input:  tokens [1-101] (rolling window of 100)
                  ❌ Token 0 is no longer visible to attention!
Attention Output: features_1_101 (only has info from tokens 1-101)

TTT Input:        features_1_101 (NO direct info about token 0!)
TTT Weights:      W₁₀₀, b₁₀₀ (← Contains compressed info from token 0!)

TTT Computation:
    Z1 = features_1_101 @ W₁₀₀ + b₁₀₀
         ^^^^^^^^^^^^^^   ^^^^
         Current input    Historical memory!
         (tokens 1-101)   (remembers token 0)

    output = features_1_101 + gradient_descent_update(Z1, ...)

    ✓ Output contains info from tokens [0-101]!
       Token 0 info recovered from W₁₀₀!

TTT Weight Update:
    W₁₀₀ → W₁₀₁ (now remembers tokens 0-101)
    b₁₀₀ → b₁₀₁
```

**Key Point**: The matrix multiplication `features_1_101 @ W₁₀₀` **injects historical information** from token 0 (and all previous tokens) into the current output!

#### Position 500 (Mid-sequence)

```
Attention Input:  tokens [400-500]
                  ❌ Tokens 0-399 NOT visible to attention!
Attention Output: features_400_500

TTT Input:        features_400_500
TTT Weights:      W₄₉₉ (← Contains compressed info from tokens 0-499!)

TTT Computation:
    output = features_400_500 @ W₄₉₉ + ...
             ^^^^^^^^^^^^^^^^   ^^^^
             Current (400-500)  Memory (0-499)

    ✓ Output contains info from tokens [0-500]!

TTT Weight Update:
    W₄₉₉ → W₅₀₀
```

#### Position 1000 (End of sequence)

```
Attention Input:  tokens [900-1000]
                  ❌ Tokens 0-899 NOT visible to attention!
Attention Output: features_900_1000

TTT Input:        features_900_1000
TTT Weights:      W₉₉₉ (← Contains ALL history: tokens 0-999!)

TTT Computation:
    output = features_900_1000 @ W₉₉₉ + ...
             ^^^^^^^^^^^^^^^^^   ^^^^
             Current (900-1000)  Memory (0-999)

    ✓ Output contains info from ENTIRE sequence [0-1000]!
```

---

## The Key Mechanism: Sequential Weight Updates

From `ttt-video-dit/ttt/models/ssm/ops/ttt_mlp.py:48-52`:

```python
# Update weights for next mini-batch using gradient descent
last_eta_mini_batch = eta_mini_batch[:, :, -1, :, None]
W1_last = W1_init - (last_eta_mini_batch * X1).transpose(-1, -2) @ grad_l_wrt_Z1
b1_last = b1_init - torch.sum(last_eta_mini_batch * grad_l_wrt_Z1, dim=-2, keepdim=True)
W2_last = W2_init - (last_eta_mini_batch * X2).transpose(-1, -2) @ grad_l_wrt_Z2
b2_last = b2_init - torch.sum(last_eta_mini_batch * grad_l_wrt_Z2, dim=-2, keepdim=True)
```

**What this does**:
- `W1_init`: Weights from previous mini-batch (contain historical info)
- `grad_l_wrt_Z1`: Gradient based on current input
- `W1_last`: Updated weights (now contain current + historical info)

**The scan loop** (from `utils.py:111-146`) processes tokens **sequentially**:

```python
def scan(f, init, xs, checkpoint_group=0):
    carry = init  # Contains W1, b1, W2, b2 states

    for i in range(num_items):  # SEQUENTIAL!
        x = xs[i]  # Get current mini-batch
        carry, y = f(carry, x)  # Update states, compute output
        # carry = updated weights (W1, b1, W2, b2)
```

**This is exactly like an RNN**:
- RNN: `h_t = tanh(W @ [h_{t-1}, x_t])`
- TTT: `W_t = W_{t-1} - eta * gradient(reconstruction_loss)`

---

## Analogy: TTT Weights as Learned Memory

### Traditional RNN

| Component | Value |
|-----------|-------|
| **Input** | x_t (current token features) |
| **Memory** | h_{t-1} (hidden state vector) |
| **Update** | h_t = tanh(W_h @ h_{t-1} + W_x @ x_t) |
| **Memory Type** | Fixed-size vector |
| **Update Method** | Gating function (tanh/sigmoid) |

### TTT

| Component | Value |
|-----------|-------|
| **Input** | XQ_t (current attention-filtered features) |
| **Memory** | W_{t-1}, b_{t-1} (MLP weight matrices) |
| **Update** | W_t = W_{t-1} - eta * gradient(reconstruction_loss) |
| **Memory Type** | Entire MLP model |
| **Update Method** | Gradient descent (test-time training) |

**Why TTT is better**:
- RNN: Memory is a vector, updated by hand-designed gating
- TTT: Memory is a **learned model**, updated by **learned gradient descent**
- When you do `XQ @ W`, you're running inference through a mini-MLP that was **trained on all previous tokens**!

---

## Video-DiT vs Moshi-TTT Implementation

### Architectural Similarity: IDENTICAL

| Aspect | Video-DiT | Moshi-TTT |
|--------|-----------|-----------|
| **Layer Flow** | Attention → TTT → Feedforward | Attention → TTT → Feedforward ✅ |
| **TTT Input** | Attention-filtered features | Attention-filtered features ✅ |
| **TTT Processing** | Sequential scan with weight updates | Sequential scan with weight updates ✅ |
| **Global Context** | Via persistent weights W, b | Via persistent weights W, b ✅ |
| **Implementation** | `SeqModelingBlock._ssm_forward()` | `HybridStreamingTransformerLayer._ttt_forward()` ✅ |

**Code Evidence**:

**Video-DiT** (`dit.py:268-276`):
```python
def forward(self, vid_emb, text_emb, seq_metadata):
    output = self._attn_forward(vid_emb, text_emb, seq_metadata)  # Attention
    output = self._ssm_forward(output, seq_metadata)              # TTT
    return output
```

**Moshi-TTT** (`hybrid_layer.py:256-264`):
```python
def _forward_impl(self, x, cross_attention_src):
    attn_output = self._attn_forward(x, cross_attention_src)  # Attention
    ttt_output = self._ttt_forward(attn_output)               # TTT
    return ttt_output
```

**Identical architecture!**

### Attention Pattern Differences

| Aspect | Video-DiT | Moshi-TTT |
|--------|-----------|-----------|
| **Segmentation** | Hard 3-second boundaries | Rolling window |
| **Attention Coverage** | 4.8% (3s / 63s) | Configurable (e.g., 10% = 100/1000) |
| **Boundary Type** | Sharp (segment 1 can't see segment 2) | Smooth (gradual aging out) |
| **Training Signal** | Strong (attention MUST not cross boundaries) | Softer (gradual dependency on TTT) |

**Video-DiT Example** (63-second video = 21 × 3-second segments):

```
Segment 1 (0-3s):   Attention sees tokens 0-18048 only     ✓
                    TTT accumulates 0-18048                 ✓

Segment 2 (3-6s):   Attention sees tokens 18048-36096 only ✓ (ISOLATED)
                    TTT accumulates 0-36096                 ✓ (carries Seg 1)

Segment 21 (60-63s): Attention sees tokens 323502-341550   ✓ (ISOLATED)
                     TTT accumulated 0-341550               ✓ (carries ALL)
```

**Attention Coverage**: 3s / 63s = **4.8%**

**Moshi-TTT Example** (80-second sequence = 1000 tokens, context=100):

```
Position 0-100:   Attention sees tokens 0-100     ✓
                  TTT accumulates 0-100           ✓

Position 500:     Attention sees tokens 400-500   ✓
                  TTT accumulates 0-500           ✓ (carries 0-399)

Position 1000:    Attention sees tokens 900-1000  ✓
                  TTT accumulates 0-1000          ✓ (carries ALL)
```

**Attention Coverage**: 100 / 1000 = **10%**

---

## Training with Limited Attention Context

### Question: What happens if we train with context=100 on 1000-token sequences?

**Answer**: TTT learns to be the **global memory**, while attention handles **local feature extraction**.

### Division of Labor

**Attention's Job** (context=100):
- Extract local features from nearby tokens
- Identify local patterns, dependencies
- "What's happening RIGHT NOW in my 100-token window?"

**TTT's Job** (global via persistent states):
- Compress and carry historical information
- Bridge gaps between attention windows
- "What happened BEFORE that attention can't see?"

### Training Dynamics

**Token 0-100**:
- Attention: Learns to extract features from full context (all 100 visible)
- TTT: Learns initial compression

**Token 101**:
- Attention: Extracts features from tokens 1-101 (token 0 invisible)
- TTT: **Forced** to recover token 0 info from weights W₁₀₀
- Gradient signal: "You MUST maintain this information in weights!"

**Token 500**:
- Attention: Only sees 400-500
- TTT: Must recover 0-399 from weights
- **Strong training signal**: Model learns TTT is the memory layer

**Token 1000**:
- Attention: Only sees 900-1000
- TTT: Must recover 0-899 from weights
- By this point, TTT has learned to compress long-range dependencies

### Why This Works

**Gradient Flow**:
```
Loss = reconstruction_loss(XK, XV)

∂Loss/∂W contains signal from:
  - Current tokens (direct gradient)
  - Historical tokens (via chain rule through previous updates)
```

**Self-Supervised Learning**:
- TTT learns to predict `XV` from `XK`
- To predict well at token 1000, it MUST remember context from token 0
- **No choice but to use weights as memory!**

---

## Comparison to Standard Transformer

### Standard Transformer with context=100

```
Position 500: Attention sees tokens 400-500
              Output contains info from 400-500 only ❌
              Tokens 0-399 are LOST FOREVER
```

**Problem**: No memory mechanism - old tokens disappear.

### Moshi-TTT with context=100

```
Position 500: Attention sees tokens 400-500
              TTT sees attention output + W₄₉₉
              Output contains info from 0-500 ✓
              Tokens 0-399 recovered from TTT weights!
```

**Solution**: TTT acts as learned memory.

---

## How Moshi Handles Streaming Inference

### Key Difference: Streaming State Management

**Video-DiT**: Batch processing (entire 63s video at once)
- Processes full sequence in one pass
- TTT weights reset for each new video

**Moshi**: Streaming audio (real-time generation)
- Processes token-by-token in streaming mode
- TTT weights can persist across chunks (if `persistent_states=True`)

### Moshi Streaming with TTT

From `hybrid_layer.py:184-191`:

```python
def reset_ttt_states(self):
    """DEPRECATED: This method breaks the computation graph during training."""
    logger.warning(f"⚠️ Layer {self.layer_id}: reset_ttt_states() is DEPRECATED")
    logger.warning("   Use save_ttt_states() and restore_ttt_states() instead")
```

**Streaming Behavior**:

1. **During Training** (with `persistent_states=True`):
   - Process chunk 1 (e.g., 10s audio): W₀ → W₁₀
   - Process chunk 2 (next 10s): W₁₀ → W₂₀ (carries chunk 1 info)
   - Process chunk 3 (next 10s): W₂₀ → W₃₀ (carries chunks 1-2 info)

2. **During Inference** (streaming):
   - Token 1: W₀ → W₁
   - Token 2: W₁ → W₂ (remembers token 1)
   - Token 1000: W₉₉₉ → W₁₀₀₀ (remembers all previous)

3. **Optional Reset** (for new speaker/context):
   - `save_ttt_states()`: Store current W, b
   - `restore_ttt_states()`: Restore previous W, b
   - `reset_ttt_inner_weights_for_new_file()`: Reset to learned base values

---

## Inference Behavior with Different Context Settings

### Scenario 1: Trained with context=3000, Inference with context=3000 ✅

```
Training: Attention handles 3000 tokens, TTT handles older
Inference: Attention handles 3000 tokens, TTT handles older
Result: Perfect match! TTT expects this division of labor.
```

### Scenario 2: Trained with context=3000, Inference with context=750 ❌

```
Training: TTT learned "attention handles 3000, I handle the rest"
Inference: Attention only handles 750
Result: TTT expects attention to handle 3000!
        Tries to reconstruct missing 2250 tokens → CONFUSED → gibberish
```

**This is why your inference log showed gibberish!**

### Scenario 3: Trained with context=100, Inference with context=100 ✅

```
Training: Attention handles 100, TTT is main memory
Inference: Attention handles 100, TTT is main memory
Result: Perfect match! TTT learned to be aggressive memory.
```

### Scenario 4: Trained with context=100, Inference with context=3000 ❌

```
Training: TTT learned "I'm the primary memory for everything >100 tokens"
Inference: Attention now handles up to 3000 tokens
Result: TTT tries to compress info that attention already handles → redundant/confused
```

---

## Recommended Settings Based on Video-DiT

### Option 1: Match Video-DiT's Absolute Segment Size (3-4 seconds)

```yaml
duration_sec: 80  # Long sequences

ttt:
  ttt_layer_context: 50          # 4 seconds at 12.5 Hz
  non_ttt_layer_context: 100     # 8 seconds for non-TTT layers
```

**Reasoning**:
- Video-DiT uses 3s attention, proven to work
- 4s is close analog for audio
- Coverage: 50/1000 = 5% (similar to Video-DiT's 4.8%)

### Option 2: Match Video-DiT's Coverage Ratio (4.8%)

```yaml
duration_sec: 80  # 1000 tokens at 12.5 Hz

ttt:
  ttt_layer_context: 48          # 4.8% of 1000 ≈ 48
  non_ttt_layer_context: 100     # Keep some context for non-TTT
```

### Option 3: Conservative (Current Default)

```yaml
duration_sec: 80

ttt:
  ttt_layer_context: 100         # 8 seconds, 10% coverage
  non_ttt_layer_context: 100     # Same for both
```

**Issue**: Less aggressive than Video-DiT, might not force TTT hard enough.

---

## Summary: Answers to Key Questions

### Q1: If we set context=100 during training with 1000-token sequences, will TTT learn the entire sequence?

**A: YES!** TTT's persistent weights (W, b) accumulate information via sequential gradient updates. Even though TTT receives attention-filtered features, the weights themselves contain compressed representations of all previous tokens.

### Q2: Is Video-DiT's TTT seeing "raw input" while ours sees "attention output"?

**A: NO!** Both implementations are identical:
- **Video-DiT**: Attention → TTT (processes attention output)
- **Moshi-TTT**: Attention → TTT (processes attention output)

Both receive attention-filtered features, not raw input.

### Q3: How does TTT achieve "global context" if attention is "local"?

**A: Via sequential weight updates!** The key is the `scan` loop:

```python
for i in range(num_tokens):
    W_new = W_old - eta * gradient(current_features)
    output = current_features @ W_new
```

`W_new` contains information from all previous tokens, injected via matrix multiplication.

### Q4: What's the difference between Video-DiT and Moshi-TTT?

**A: Only the attention pattern:**
- **Video-DiT**: Hard 3s segment boundaries (4.8% coverage)
- **Moshi-TTT**: Rolling window with configurable size (e.g., 10% coverage)

Architecture is identical otherwise.

### Q5: What happens during inference if context doesn't match training?

**A: Model confusion and degraded output** (potentially gibberish). TTT learned a specific "division of labor" with attention. Changing context breaks this learned behavior.

**Solution**: Always load context from `training_config.json` (implemented in this PR).

---

## Conclusion

**TTT is fundamentally a learned memory layer**, analogous to RNN hidden states but updated via gradient descent instead of fixed gating functions. When combined with limited-context attention:

1. **Attention** = Local feature extractor (current window)
2. **TTT weights** = Learned memory (all history)
3. **TTT output** = Current features interpreted through historical context

This division of labor emerges naturally during training via the self-supervised reconstruction loss. The model learns to compress long-range dependencies into the TTT weights, enabling global context despite local attention.

**Critical for inference**: Match the attention context used during training. The model learned a specific balance between attention and TTT - changing it breaks the learned behavior.
