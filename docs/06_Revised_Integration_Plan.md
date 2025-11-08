# Revised TTT + Moshi Integration Plan
## Based on Real Implementation Experience

**Status**: Previous attempts failed with gibberish after 5-7 minutes
**Root Causes Identified**: State management + FP32 precision + streaming mismatch
**New Approach**: Fix fundamentals first, then scale up

---

## Phase 0: Debugging Current Implementation (Week 1)

**Goal**: Understand exactly WHY previous attempts produced gibberish.

### Task 0.1: Add Comprehensive Logging

**File**: `moshi/moshi/modules/ttt_debug.py` (NEW)

```python
import torch
import torch.nn as nn
from collections import defaultdict
import json

class TTTDebugMonitor:
    """Monitor TTT behavior during training and inference."""

    def __init__(self, log_path="ttt_debug.jsonl"):
        self.log_path = log_path
        self.reset_events = []
        self.dtype_checks = []
        self.state_norms = []
        self.step = 0

    def log_reset(self, layer_name, reason="unknown"):
        """Log when TTT state resets."""
        event = {
            "step": self.step,
            "event": "state_reset",
            "layer": layer_name,
            "reason": reason
        }
        self.reset_events.append(event)
        print(f"🔄 [{self.step}] State reset in {layer_name}: {reason}")

    def log_dtype_check(self, layer_name, tensor_name, dtype, expected=torch.float32):
        """Check if tensor has expected dtype."""
        is_correct = (dtype == expected)
        event = {
            "step": self.step,
            "event": "dtype_check",
            "layer": layer_name,
            "tensor": tensor_name,
            "dtype": str(dtype),
            "expected": str(expected),
            "correct": is_correct
        }
        self.dtype_checks.append(event)

        if not is_correct:
            print(f"⚠️  [{self.step}] {layer_name}.{tensor_name}: dtype={dtype}, expected={expected}")

    def log_state_norm(self, layer_name, W_norm, grad_norm=None):
        """Log state magnitude."""
        event = {
            "step": self.step,
            "event": "state_norm",
            "layer": layer_name,
            "W_norm": float(W_norm),
            "grad_norm": float(grad_norm) if grad_norm is not None else None
        }
        self.state_norms.append(event)

        # Check for explosion
        if W_norm > 1e4:
            print(f"❌ [{self.step}] {layer_name}: W exploding! norm={W_norm:.2e}")
        if torch.isnan(torch.tensor(W_norm)):
            print(f"❌ [{self.step}] {layer_name}: W is NaN!")

    def save_log(self):
        """Save all logs to file."""
        all_events = self.reset_events + self.dtype_checks + self.state_norms
        all_events.sort(key=lambda x: x['step'])

        with open(self.log_path, 'w') as f:
            for event in all_events:
                f.write(json.dumps(event) + '\n')

        print(f"💾 Saved {len(all_events)} events to {self.log_path}")

    def step_forward(self):
        """Increment step counter."""
        self.step += 1

# Global monitor instance
debug_monitor = TTTDebugMonitor()
```

### Task 0.2: Instrument Existing TTT Layer

**Modify your current TTT implementation** to add logging:

```python
class ExistingTTTLayer(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # Your existing init
        ...

        # ADD: Debug tracking
        self.debug_name = f"ttt_layer_{id(self)}"
        self.reset_count = 0

    def forward(self, x):
        from .ttt_debug import debug_monitor

        # CHECK 1: State reset detection
        if self.W_state is None or hasattr(self, '_should_reset'):
            debug_monitor.log_reset(self.debug_name, reason="state_none_or_forced")
            self.reset_count += 1

        # CHECK 2: Dtype verification
        if hasattr(self, 'W1'):
            debug_monitor.log_dtype_check(self.debug_name, "W1", self.W1.dtype, torch.float32)
            debug_monitor.log_dtype_check(self.debug_name, "b1", self.b1.dtype, torch.float32)

        # CHECK 3: State magnitude
        if hasattr(self, 'W1'):
            W_norm = self.W1.norm().item()
            debug_monitor.log_state_norm(self.debug_name, W_norm)

        # Your existing forward pass
        output = self._existing_forward(x)

        debug_monitor.step_forward()
        return output
```

### Task 0.3: Run Diagnostic Generation Test

**Script**: `test_gibberish_detection.py`

```python
import torch
from moshi.models import loaders
from ttt_debug import debug_monitor
import torchaudio

def generate_long_and_diagnose(duration_minutes=10):
    """Generate long audio and track when gibberish starts."""

    model = loaders.get_lm_model_generator(name="your-ttt-moshi-model")

    outputs = []
    qualities = []

    with model.streaming(batch_size=1):
        frames_per_minute = 750  # 12.5 Hz × 60 sec
        total_frames = duration_minutes * frames_per_minute

        for frame_idx in range(total_frames):
            # Generate next frame
            output = model.step(dummy_input, frame_idx)
            outputs.append(output)

            # Every 30 seconds, evaluate quality
            if frame_idx % 375 == 374:  # 30 sec checkpoints
                minute = frame_idx // 750
                second = (frame_idx % 750) / 12.5

                # Decode audio
                audio_segment = decode_outputs(outputs[-375:])

                # Compute quality metrics
                transcription = whisper.transcribe(audio_segment)
                wer = compute_wer(transcription)

                qualities.append({
                    "time": f"{minute}m {second:.0f}s",
                    "frame": frame_idx,
                    "wer": wer,
                    "transcription": transcription
                })

                print(f"[{minute}m {second:.0f}s] WER={wer:.3f}, Text: {transcription[:50]}...")

                # DETECT GIBBERISH
                if wer > 0.8 or is_gibberish(transcription):
                    print(f"\n❌ GIBBERISH DETECTED at {minute}m {second:.0f}s!")
                    print(f"Total resets so far: {sum(layer.reset_count for layer in model.ttt_layers)}")
                    debug_monitor.save_log()
                    break

    # Save debug log
    debug_monitor.save_log()

    # Analyze reset pattern
    analyze_resets()

def analyze_resets():
    """Analyze when states reset."""
    import json

    with open("ttt_debug.jsonl") as f:
        events = [json.loads(line) for line in f]

    resets = [e for e in events if e["event"] == "state_reset"]

    print(f"\n📊 Reset Analysis:")
    print(f"Total resets: {len(resets)}")

    if len(resets) > 0:
        print(f"First reset at step: {resets[0]['step']}")
        print(f"Last reset at step: {resets[-1]['step']}")
        print(f"Average gap between resets: {resets[-1]['step'] / len(resets):.0f} steps")

        # If resets happening mid-generation → BUG!
        if len(resets) > 1:
            print("⚠️  WARNING: Multiple resets during generation!")
            print("This explains the gibberish problem!")

if __name__ == "__main__":
    generate_long_and_diagnose(duration_minutes=10)
```

**Expected Output**:
```
[0m 30s] WER=0.15, Text: Hello, this is a test of the speech model...
[1m 0s] WER=0.18, Text: The quick brown fox jumps over the...
...
[5m 30s] WER=0.22, Text: As we continue this long conversation...
[6m 0s] WER=0.91, Text: gzxkl jsdhf klwer nmsdf...

❌ GIBBERISH DETECTED at 6m 0s!
Total resets so far: 12

📊 Reset Analysis:
Total resets: 12
First reset at step: 0
Last reset at step: 4800
Average gap between resets: 400 steps

⚠️  WARNING: Multiple resets during generation!
This explains the gibberish problem!
```

**If you see multiple resets → State management bug confirmed!**

---

## Phase 1: Fix State Management (Week 2)

**Goal**: Ensure TTT state persists for entire conversation.

### Task 1.1: Create Conversation-Aware TTT Layer

**File**: `moshi/moshi/modules/ttt_stateful.py` (NEW - 200 lines)

```python
import torch
import torch.nn as nn
from typing import Optional, Tuple
from dataclasses import dataclass

@dataclass
class TTTState:
    """TTT hidden state (conversation memory)."""
    W1: torch.Tensor  # [B, num_heads, head_dim, head_dim] - FP32!
    b1: torch.Tensor  # [B, num_heads, 1, head_dim] - FP32!
    step_count: int = 0
    conversation_id: Optional[str] = None

class StatefulTTTLinear(nn.Module):
    """
    TTT-Linear with conversation-level state persistence.

    Key features:
    - State persists across batches within same conversation
    - State resets only when conversation_id changes
    - FP32 precision for W1, b1 enforced
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        mini_batch_size: int = 16,
        ttt_base_lr: float = 1.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.mini_batch_size = mini_batch_size
        self.ttt_base_lr = ttt_base_lr

        # Q/K/V projections (BF16 for memory efficiency)
        self.q_proj = nn.Linear(d_model, d_model, dtype=torch.bfloat16)
        self.k_proj = nn.Linear(d_model, d_model, dtype=torch.bfloat16)
        self.v_proj = nn.Linear(d_model, d_model, dtype=torch.bfloat16)
        self.out_proj = nn.Linear(d_model, d_model, dtype=torch.bfloat16)

        # TTT parameters - LEARNED INITIAL STATE (FP32!)
        self.W1_init = nn.Parameter(
            torch.normal(0, 0.02, size=(num_heads, self.head_dim, self.head_dim),
                        dtype=torch.float32)  # FP32!
        )
        self.b1_init = nn.Parameter(
            torch.zeros(num_heads, 1, self.head_dim,
                       dtype=torch.float32)  # FP32!
        )

        # TTT LayerNorm (FP32!)
        self.ttt_norm_weight = nn.Parameter(
            torch.ones(num_heads, 1, self.head_dim, dtype=torch.float32)
        )
        self.ttt_norm_bias = nn.Parameter(
            torch.zeros(num_heads, 1, self.head_dim, dtype=torch.float32)
        )

        # Learnable learning rate
        self.learnable_lr = nn.Parameter(torch.zeros(num_heads, 1, 1))

        # Gating (initialized small)
        self.gate_alpha = nn.Parameter(torch.ones(d_model) * 0.1)

        # CONVERSATION STATE (persistent across batches!)
        self.conversation_state: Optional[TTTState] = None

    def reset_conversation(self, conversation_id: Optional[str] = None):
        """Reset state for new conversation."""
        from .ttt_debug import debug_monitor
        debug_monitor.log_reset("stateful_ttt", reason=f"new_conversation:{conversation_id}")

        self.conversation_state = None

    def _init_state(self, batch_size: int, device: torch.device) -> TTTState:
        """Initialize fresh state from learned parameters."""
        # Tile learned init to batch size
        W1 = self.W1_init.unsqueeze(0).expand(batch_size, -1, -1, -1).clone()
        b1 = self.b1_init.unsqueeze(0).expand(batch_size, -1, -1, -1).clone()

        # CRITICAL: Ensure FP32!
        assert W1.dtype == torch.float32, f"W1 must be FP32, got {W1.dtype}"
        assert b1.dtype == torch.float32, f"b1 must be FP32, got {b1.dtype}"

        return TTTState(W1=W1, b1=b1, step_count=0)

    def forward(
        self,
        x: torch.Tensor,
        conversation_id: Optional[str] = None,
        is_new_conversation: bool = False
    ) -> torch.Tensor:
        """
        Forward pass with stateful TTT.

        Args:
            x: Input [batch, seq_len, d_model]
            conversation_id: ID of current conversation
            is_new_conversation: If True, reset state

        Returns:
            Output [batch, seq_len, d_model]
        """
        from .ttt_debug import debug_monitor

        batch_size, seq_len, _ = x.shape

        # CRITICAL: Handle state initialization/reset
        if is_new_conversation or self.conversation_state is None:
            self.reset_conversation(conversation_id)
            self.conversation_state = self._init_state(batch_size, x.device)
        elif conversation_id is not None and self.conversation_state.conversation_id != conversation_id:
            # Different conversation → reset
            self.reset_conversation(conversation_id)
            self.conversation_state = self._init_state(batch_size, x.device)

        # Update conversation ID
        if conversation_id is not None:
            self.conversation_state.conversation_id = conversation_id

        # Q/K/V projections (BF16)
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape for heads
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # L2 normalize (stability)
        Q = torch.nn.functional.normalize(Q, p=2, dim=-1)
        K = torch.nn.functional.normalize(K, p=2, dim=-1)

        # Process with TTT
        output, updated_state = self._ttt_process(
            Q, K, V,
            self.conversation_state.W1,
            self.conversation_state.b1
        )

        # CRITICAL: Update persistent state
        self.conversation_state.W1 = updated_state[0].detach()  # Detach to avoid memory leak
        self.conversation_state.b1 = updated_state[1].detach()
        self.conversation_state.step_count += seq_len

        # Debug logging
        debug_monitor.log_dtype_check("stateful_ttt", "W1", self.conversation_state.W1.dtype)
        debug_monitor.log_state_norm("stateful_ttt", self.conversation_state.W1.norm())

        # Reshape back
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # Gating
        gate_value = torch.tanh(self.gate_alpha)
        output = gate_value * output + (1 - gate_value) * x

        # Output projection
        output = self.out_proj(output)

        return output

    def _ttt_process(self, Q, K, V, W1, b1) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Core TTT processing with mini-batches.

        Returns:
            output: Processed tensor
            updated_state: (W1_final, b1_final)
        """
        batch_size, num_heads, seq_len, head_dim = Q.shape

        # Mini-batch processing
        num_mini_batches = (seq_len + self.mini_batch_size - 1) // self.mini_batch_size
        outputs = []

        for i in range(num_mini_batches):
            start = i * self.mini_batch_size
            end = min((i + 1) * self.mini_batch_size, seq_len)

            Q_mb = Q[:, :, start:end, :]
            K_mb = K[:, :, start:end, :]
            V_mb = V[:, :, start:end, :]

            # Process mini-batch
            output_mb, W1, b1 = self._process_mini_batch(Q_mb, K_mb, V_mb, W1, b1)
            outputs.append(output_mb)

        # Concatenate outputs
        output = torch.cat(outputs, dim=2)

        return output, (W1, b1)

    def _process_mini_batch(self, Q, K, V, W1, b1):
        """Process one mini-batch with TTT update."""
        # Convert to FP32 for computation
        Q_fp32 = Q.float()
        K_fp32 = K.float()
        V_fp32 = V.float()

        # Forward: Z1 = K @ W1 + b1
        Z1 = torch.matmul(K_fp32, W1) + b1

        # Reconstruction target
        reconstruction_target = V_fp32 - K_fp32

        # Compute gradient through LayerNorm + L2 loss
        grad_l_wrt_Z1 = self._ln_fused_l2_bwd(
            Z1, reconstruction_target,
            self.ttt_norm_weight, self.ttt_norm_bias
        )

        # Token-dependent learning rate
        mb_size = Q.shape[2]
        eta = self.ttt_base_lr * torch.sigmoid(self.learnable_lr) / self.head_dim
        eta = eta.expand(-1, -1, mb_size, -1)

        # Causal attention for update
        Attn = torch.matmul(Q_fp32, K_fp32.transpose(-2, -1))
        Attn = torch.tril(Attn)  # Causal mask

        # Update bias
        causal_cumsum_grad = torch.matmul(torch.tril(torch.ones_like(Attn)), grad_l_wrt_Z1)
        b1_bar = b1 - eta * causal_cumsum_grad

        # Compute output
        Z1_bar = torch.matmul(Q_fp32, W1) + b1_bar[:, :, -1:, :]
        Z1_bar = Z1_bar - eta * torch.matmul(Attn, grad_l_wrt_Z1)

        # LayerNorm + residual
        Z1_bar_ln = self._layer_norm(Z1_bar, self.ttt_norm_weight, self.ttt_norm_bias)
        output = Q_fp32 + Z1_bar_ln

        # Update W1, b1 for next mini-batch (using last token)
        last_eta = eta[:, :, -1:, :]
        W1_updated = W1 - (last_eta * K_fp32[:, :, -1:, :]).transpose(-2, -1) @ grad_l_wrt_Z1[:, :, -1:, :]
        b1_updated = b1 - torch.sum(last_eta * grad_l_wrt_Z1[:, :, -1:, :], dim=2, keepdim=True)

        # CRITICAL: Ensure FP32 maintained
        assert W1_updated.dtype == torch.float32
        assert b1_updated.dtype == torch.float32

        # Convert output back to BF16
        return output.bfloat16(), W1_updated, b1_updated

    def _ln_fused_l2_bwd(self, x, target, gamma, beta, eps=1e-6):
        """LayerNorm + L2 loss backward (analytical gradient)."""
        D = x.shape[-1]
        mu = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        std = torch.sqrt(var + eps)
        x_hat = (x - mu) / std
        y = gamma * x_hat + beta

        # Gradient
        grad_output = y - target
        grad_x_hat = grad_output * gamma

        # Backward through LayerNorm
        grad_x = (1.0 / D) * (
            D * grad_x_hat
            - grad_x_hat.sum(dim=-1, keepdim=True)
            - x_hat * (grad_x_hat * x_hat).sum(dim=-1, keepdim=True)
        ) / std

        return grad_x

    def _layer_norm(self, x, gamma, beta, eps=1e-6):
        """LayerNorm forward."""
        mu = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_hat = (x - mu) / torch.sqrt(var + eps)
        return gamma * x_hat + beta
```

### Task 1.2: Modify Moshi Transformer to Use Stateful TTT

**File**: Modify `moshi/moshi/modules/transformer.py`

```python
# Add import
from .ttt_stateful import StatefulTTTLinear

class StreamingTransformerLayer(StreamingModule[_LayerState]):
    def __init__(
        self,
        ...
        use_ttt: bool = False,  # ADD
        ttt_config: Optional[dict] = None,  # ADD
    ):
        super().__init__()

        # Standard attention OR TTT
        if use_ttt:
            self.self_attn = StatefulTTTLinear(
                d_model=d_model,
                num_heads=num_heads,
                **ttt_config
            )
        else:
            self.self_attn = StreamingMultiheadAttention(...)

        # Rest unchanged
        ...

    def forward(self, x, conversation_id=None, is_new_conversation=False):
        # Self-attention block
        if hasattr(self.self_attn, 'conversation_state'):
            # TTT layer
            y = self._sa_block_ttt(
                self.norm1(x, self.step),
                x,
                conversation_id=conversation_id,
                is_new_conversation=is_new_conversation
            )
        else:
            # Standard attention
            y = self._sa_block(self.norm1(x, self.step), x)

        # Feed-forward (unchanged)
        y = self._ff_block(self.norm2(y, self.step), y)

        return y

    def _sa_block_ttt(self, x, residual, conversation_id=None, is_new_conversation=False):
        """Self-attention block with TTT."""
        x = self.self_attn(
            x,
            conversation_id=conversation_id,
            is_new_conversation=is_new_conversation
        )
        x = self.dropout(x)
        x = residual + x
        return x
```

### Task 1.3: Modify Training Loop

**File**: Your training script

```python
def train_with_conversations(model, dataset, ...):
    model.train()

    for epoch in range(num_epochs):
        # Group dataset by conversation
        for conversation_data in dataset.iter_conversations():
            conversation_id = conversation_data['id']

            # Reset conversation state
            for layer in model.transformer.layers:
                if hasattr(layer.self_attn, 'reset_conversation'):
                    layer.self_attn.reset_conversation(conversation_id)

            # Process all batches in this conversation
            for batch_idx, batch in enumerate(conversation_data['batches']):
                is_first_batch = (batch_idx == 0)

                # Forward
                logits = model(
                    batch['input_tokens'],
                    conversation_id=conversation_id,
                    is_new_conversation=is_first_batch  # Only reset on first batch!
                )

                # Loss & backprop
                loss = F.cross_entropy(logits, batch['targets'])
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                print(f"Conv {conversation_id}, Batch {batch_idx}/{len(conversation_data['batches'])}, Loss: {loss:.4f}")

            # End of conversation - state will reset for next one
            print(f"✓ Completed conversation {conversation_id}")
```

**CRITICAL**: `is_new_conversation=True` only on FIRST batch of conversation!

---

## Phase 2: Implement Streaming TTT (Week 3-4)

**Goal**: Make TTT work with Moshi's frame-by-frame generation.

### Task 2.1: Add Streaming Buffer

**Modify `StatefulTTTLinear` to add buffering**:

```python
class StatefulTTTLinear(nn.Module):
    def __init__(self, ..., stream_buffer_size=16):
        super().__init__()
        ...
        self.stream_buffer_size = stream_buffer_size
        self.token_buffer = []

    def streaming_forward(self, x_single_token):
        """
        Process single token in streaming mode.

        Args:
            x_single_token: [batch, 1, d_model] - single frame

        Returns:
            output: [batch, 1, d_model]
        """
        # Add to buffer
        self.token_buffer.append(x_single_token)

        # Check if buffer full
        if len(self.token_buffer) >= self.stream_buffer_size:
            # Process buffer as mini-batch
            x_batch = torch.cat(self.token_buffer, dim=1)  # [batch, buffer_size, d_model]

            # TTT processing
            output_batch = self.forward(x_batch, is_new_conversation=False)

            # Clear buffer
            self.token_buffer = []

            # Return last token
            return output_batch[:, -1:, :]
        else:
            # Buffer not full: use current state without updating
            return self._forward_no_update(x_single_token)

    def _forward_no_update(self, x):
        """Forward pass without TTT update (uses current W, b)."""
        # Q/K/V projections
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Simple forward with current state (no gradient update)
        Z = torch.matmul(K, self.conversation_state.W1) + self.conversation_state.b1

        # LayerNorm
        Z_ln = self._layer_norm(Z, self.ttt_norm_weight, self.ttt_norm_bias)

        # Output
        output = Q + Z_ln
        output = self.out_proj(output)

        return output
```

### Task 2.2: Modify Moshi's Streaming Loop

**File**: Wherever you call `model.step()`

```python
def generate_streaming_with_ttt(model, duration_seconds=600):
    """Generate long audio with buffered TTT updates."""
    model.eval()

    outputs = []
    with torch.no_grad():
        with model.streaming(batch_size=1):
            frames_to_generate = int(duration_seconds * 12.5)  # 12.5 Hz

            for frame_idx in range(frames_to_generate):
                # Get input
                input_codes = ...  # Your input preparation

                # Generate next frame (uses streaming_forward internally)
                output_codes = model.step(input_codes, frame_idx)

                outputs.append(output_codes)

                # Log progress
                if frame_idx % 750 == 0:  # Every minute
                    print(f"Generated {frame_idx/12.5:.0f}s")

    return outputs
```

---

## Phase 3: Validation (Week 5)

### Task 3.1: Long Generation Test

**Script**: `test_long_generation.py`

```python
def test_long_generation():
    """Test if gibberish problem is solved."""
    model = load_model_with_stateful_ttt()

    # Test progressively longer durations
    for duration in [5, 10, 20, 30, 60]:  # minutes
        print(f"\n{'='*50}")
        print(f"Testing {duration}-minute generation")
        print(f"{'='*50}")

        outputs = generate_streaming_with_ttt(model, duration_seconds=duration*60)

        # Evaluate
        audio = decode_outputs(outputs)
        transcription = whisper.transcribe(audio)
        wer = compute_wer(transcription)

        print(f"Duration: {duration}min")
        print(f"WER: {wer:.3f}")
        print(f"Sample: {transcription[:100]}...")

        # Check for gibberish
        if wer > 0.7:
            print(f"❌ FAILED at {duration} minutes (WER={wer:.3f})")
            break
        else:
            print(f"✓ PASSED {duration} minutes")

test_long_generation()
```

### Task 3.2: State Persistence Verification

**Script**: `verify_state_persistence.py`

```python
def verify_state_persistence():
    """Verify that TTT state actually persists."""
    model = load_model_with_stateful_ttt()

    with model.streaming(batch_size=1):
        W1_norms = []

        for frame_idx in range(7500):  # 10 minutes
            output = model.step(dummy_input, frame_idx)

            # Track state evolution
            for layer in model.transformer.layers:
                if hasattr(layer.self_attn, 'conversation_state'):
                    if layer.self_attn.conversation_state is not None:
                        W_norm = layer.self_attn.conversation_state.W1.norm().item()
                        W1_norms.append(W_norm)

        # Plot state evolution
        import matplotlib.pyplot as plt
        plt.plot(W1_norms)
        plt.xlabel("Frame")
        plt.ylabel("W1 Norm")
        plt.title("TTT State Evolution Over Time")
        plt.savefig("ttt_state_evolution.png")

        # Check if state actually changed
        if len(set(W1_norms)) == 1:
            print("❌ FAILED: State never changed (not updating!)")
        else:
            print(f"✓ PASSED: State evolved (range: {min(W1_norms):.2f} - {max(W1_norms):.2f})")

verify_state_persistence()
```

---

## Success Criteria

### Phase 0 Complete When:
- [ ] Debug logging implemented
- [ ] Gibberish problem diagnosed (reset pattern identified)
- [ ] Dtype issues identified (if any)

### Phase 1 Complete When:
- [ ] State persists within conversation during training
- [ ] State resets only between conversations
- [ ] No unexpected state resets during inference
- [ ] FP32 precision verified for W1, b1

### Phase 2 Complete When:
- [ ] Streaming generation works with buffered TTT
- [ ] No latency explosion (buffer size reasonable)
- [ ] State updates happen regularly (every N frames)

### Phase 3 Complete When:
- [ ] 10-minute generation is coherent (no gibberish!)
- [ ] 30-minute generation is coherent
- [ ] State evolution plot shows continuous adaptation
- [ ] WER remains <0.3 throughout long generation

---

## Key Differences from Previous Attempts

| Issue | Previous Approach | New Approach |
|-------|-------------------|--------------|
| **State Reset** | Reset every batch | Reset only per conversation |
| **FP32 Precision** | Maybe not enforced | Explicit FP32 for W1, b1 with asserts |
| **Streaming** | Not properly handled | Buffered mini-batch updates |
| **Debugging** | No visibility | Comprehensive logging |
| **Local+Global** | Tried video-dit pattern | Causal sliding window only |
| **Validation** | Only checked loss | Check loss AND generation quality |

---

## If It Still Doesn't Work

### Fallback Option 1: Hybrid TTT
Only use TTT in top 8 layers (not all 16):
```python
# Layers 0-23: Standard attention
# Layers 24-31: TTT (only top 8)
use_ttt = (layer_idx >= 24)
```

### Fallback Option 2: TTT as Auxiliary Memory
Keep standard attention, add TTT as parallel branch:
```python
# Dual-path architecture
attn_out = self.attention(x)
ttt_out = self.ttt(x)
output = attn_out + 0.1 * ttt_out  # Small TTT contribution
```

### Fallback Option 3: Investigate Moshi-Specific Issues
- Check if Moshi's depformer interferes
- Check if codebook interleaving causes problems
- Try TTT on simpler speech model first (e.g., TTS-only)

---

## Timeline Summary

```
Week 1: Debug current implementation
├─ Day 1-2: Add logging
├─ Day 3-4: Run diagnostic tests
└─ Day 5: Analyze results

Week 2: Fix state management
├─ Day 1-2: Implement StatefulTTTLinear
├─ Day 3-4: Modify training loop
└─ Day 5: Test training with persistence

Week 3-4: Streaming TTT
├─ Week 3: Implement buffering
└─ Week 4: Test long generation

Week 5: Validation
├─ Day 1-3: Long generation tests
├─ Day 4: State persistence verification
└─ Day 5: Final analysis

Total: 5 weeks to either success or clear diagnosis of why it can't work
```

---

## Next Immediate Steps

1. **Add debug logging** to your current implementation (Task 0.1-0.2)
2. **Run diagnostic generation** to confirm state reset bug (Task 0.3)
3. **Review results** - if resets confirmed, proceed to Phase 1
4. **Implement StatefulTTTLinear** with conversation-aware state (Task 1.1)
5. **Test training** with conversation-level persistence

**Don't add complexity until fundamentals are working!**
