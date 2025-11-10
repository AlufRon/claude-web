# Moshi-TTT Final Comprehensive Verification

**Date**: 2025-11-10
**Status**: Complete End-to-End Analysis
**Branch**: `claude/deep-code-review-ttt-011CUzpCD2kNLGH5UnuHN7hF`

---

## Executive Summary

After exhaustive analysis of all code flows (training, inference, data loading, loss computation, state management), **5 CRITICAL ISSUES** have been identified and verified. This document provides complete end-to-end traces and verification of all important flows.

### All Issues Summary

| Issue # | Severity | Component | Status | Impact |
|---------|----------|-----------|--------|--------|
| **#1** | 🔴 Critical | Ring Buffer | Verified | Information loss after 3000 steps |
| **#2** | 🟡 Medium | Normalization | Verified | Incorrect TTT optimization target |
| **#3** | 🟡 Medium | Batch Handling | Verified | Incompatible with batch_size > 1 |
| **#4** | 🔴 Critical | Gradient Flow | Verified | Training corruption (W_base/W_state conflict) |
| **#5** | 🔴 Critical | State Management | **NEW** | Cross-file contamination |

---

## Complete Training Flow Trace

### 1. Data Pipeline (Sequential Processing)

**File**: `moshi_ttt_try/moshi-finetune/finetune/data/dataset.py`

#### 1.1 File Chunking (Lines 43-54)

```python
def maybe_load_local_dataset(...):
    duration = instruct_tokenizer.duration_sec  # e.g., 10.0 seconds
    chunks: list[AudioChunkPath] = []

    for line in lines:  # Each line = one audio file
        data = json.loads(line)
        start_sec = 0

        # Split long file into chunks
        while start_sec < data["duration"]:
            chunks.append((data["path"], start_sec))
            start_sec += duration  # Increment by 10s

    return chunks
```

**Result**: Long audio files split into fixed-duration chunks
- File 1 (60s): [0-10s, 10-20s, 20-30s, 30-40s, 40-50s, 50-60s]
- File 2 (45s): [0-10s, 10-20s, 20-30s, 30-40s, 40-45s]
- File 3 (30s): [0-10s, 10-20s, 20-30s]

#### 1.2 Sequential Dataset Iteration (Lines 250-261)

```python
def get_dataset_iterator(...):
    while True:  # Infinite loop for training
        for jsonl_file in source.jsonl_files:
            dataset = sphn.dataset_jsonl(
                str(jsonl_file),
                duration_sec=instruct_tokenizer.duration_sec,
                ...
            )

            if shuffle_at_epoch:  # FALSE for TTT training!
                dataset = dataset.shuffle(...)
            else:
                # ↓ SEQUENTIAL processing - chunks from same file are consecutive
                dataset = dataset.seq(skip=rank, step_by=world_size)

            for sample in dataset:
                wav = sample["data"][..., : sample["unpadded_len"]]
                result = instruct_tokenizer(wav, sample["start_time_sec"], sample["path"])
                if result is not None:
                    yield result
```

**✅ Sequential processing confirmed**: When `shuffle=False` (TTT training default), chunks from the same file are processed consecutively.

**Processing order**:
```
File1_chunk0 → File1_chunk1 → File1_chunk2 → ... → File1_chunkN →
File2_chunk0 → File2_chunk1 → File2_chunk2 → ... → File2_chunkM →
File3_chunk0 → ...
```

#### 1.3 Sample Creation with File Tracking (Lines 305-315)

**File**: `moshi_ttt_try/moshi-finetune/finetune/data/interleaver.py`

```python
# Track file_id and chunk_index for continuous RoPE
file_id = path  # Use path as unique file identifier
chunk_index = int(start_sec / self.duration_sec)  # Calculate chunk position

# Update chunk tracker (for sequential tracking if needed)
if file_id not in self._file_chunk_map:
    self._file_chunk_map[file_id] = 0
else:
    self._file_chunk_map[file_id] += 1

return Sample(codes, data.get("text_conditions", None),
              file_id=file_id, chunk_index=chunk_index)
```

**✅ File boundaries ARE tracked**: Each sample contains `file_id` (unique per file) and `chunk_index` (position within file).

**Sample structure**:
```python
@dataclass
class Sample:
    codes: torch.Tensor         # Audio/text tokens
    condition_attributes: ...   # Conditioning info
    file_id: str | None         # "/path/to/file1.wav"
    chunk_index: int | None     # 0, 1, 2, 3, ...
```

#### 1.4 Batch Collation (Lines 31-39)

**File**: `moshi_ttt_try/moshi-finetune/finetune/data/data_loader.py`

```python
def build_data_loader(...) -> Iterator[Batch]:
    dataset = build_dataset(...)

    sample_list = []
    for sample in dataset:
        assert sample.codes.dim() == 3
        assert len(sample.codes) == 1  # Each sample has batch_size=1
        sample_list.append(sample)

        if len(sample_list) == batch_size:
            yield Batch.collate(sample_list)  # Combine into batch
            sample_list = []
```

**Batch structure**:
```python
@dataclass
class Batch:
    codes: torch.Tensor                    # (batch_size, codebooks, seq_len)
    condition_attributes: list | None
    file_id: str | None                    # From first sample in batch
    chunk_index: int | None                # From first sample in batch
```

**⚠️ Note**: Batch takes `file_id` from first sample only. If batch_size > 1 and samples come from different files, only the first file_id is preserved!

---

### 2. Training Loop (No State Management)

**File**: `moshi_ttt_try/moshi-finetune/training/train_ttt_production.py`

#### 2.1 Training Setup (Lines 221-232)

```python
model.train()  # Set training mode: model.training = True
print("✅ Training setup complete")

# Training loop
print(f"🎯 Starting training: {args.max_steps} steps...")
print(f"   Logging every {args.log_freq} steps")
print(f"   Checkpointing: {'enabled' if args.do_ckpt else 'disabled'}")
print(f"   Evaluation: {'enabled' if args.do_eval else 'disabled'}")

start_time = time.time()
step = 0
losses = []
```

**Key observation**: No streaming state initialization! Unlike inference (which calls `mimi.streaming_forever()` and `lm_gen.streaming_forever()`), training doesn't explicitly set up streaming state.

#### 2.2 Main Training Loop (Lines 234-298)

```python
while step < args.max_steps:
    step += 1
    step_start = time.time()

    # Get batch
    batch = next(data_loader)  # Contains file_id and chunk_index
    codes = batch.codes.to(device)

    # ❌ NO file_id check!
    # ❌ NO state reset logic!
    # The batch.file_id is completely ignored!

    # Forward pass
    optimizer.zero_grad()

    condition_tensors = None
    if batch.condition_attributes is not None:
        condition_tensors = model.condition_provider.prepare(
            batch.condition_attributes
        )

    # Model forward pass with microbatching
    loss = torch.tensor([0.0], device=device)
    n_batch_tokens = 0
    n_real_tokens = 0

    for mb_idx in range(args.num_microbatches):
        # In single GPU, we just process the batch as-is
        output = model(codes=codes, condition_tensors=condition_tensors)
        # ↑ This calls: forward → transformer → TTT layers
        # ↑ Inside TTT: self.W1.data.copy_(final_states["W1_states"][0])
        # ↑ TTT weights updated during forward pass!

        # Compute losses
        text_loss = compute_loss_with_mask(
            output.text_logits,
            codes[:, : model.audio_offset],
            output.text_mask,
            mode="text",
            text_padding_weight=args.text_padding_weight,
            text_padding_ids={
                model.text_padding_token_id,
                model.end_of_text_padding_id,
            },
        )
        audio_loss = compute_loss_with_mask(
            output.logits,
            codes[:, model.audio_offset : model.audio_offset + model.dep_q],
            output.mask,
            mode="audio",
            first_codebook_weight_multiplier=args.first_codebook_weight_multiplier,
        )

        mb_loss = text_loss + audio_loss
        mb_loss.backward()  # Compute gradients
        # ↑ PROBLEM: Gradients computed for initial W1, but W1 was overwritten!

        loss += mb_loss.detach()
        n_batch_tokens += output.text_mask.numel() + output.mask.numel()
        n_real_tokens += (
            torch.sum(output.text_mask).item() + torch.sum(output.mask).item()
        )

    if args.num_microbatches > 1:
        loss /= args.num_microbatches
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                p.grad.div_(args.num_microbatches)

    # Gradient clipping and optimizer step
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
    optimizer.step()  # Update parameters using gradients
    # ↑ PROBLEM: Optimizer updates W1, conflicting with TTT's copy_()
    scheduler.step()

    # Track metrics
    loss_item = loss.item()
    losses.append(loss_item)
    step_time = time.time() - step_start

    # ... logging, eval, checkpointing ...
```

**🔴 CRITICAL FINDING**: Training loop completely ignores `batch.file_id`! No state reset between files.

---

### 3. Complete Iteration Trace

Let me trace through 5 consecutive training iterations with 2 files:

#### **Iteration 1: File 1, Chunk 0 (0-10s)**

```python
# Data loader yields
batch = {
    codes: Tensor[1, 9, 750],  # batch_size=1, 9 codebooks, 750 tokens
    file_id: "/data/file1.wav",
    chunk_index: 0
}

# Training loop
optimizer.zero_grad()
output = model(codes=batch.codes)

# Inside model → transformer → TTT layer (ttt_layer.py:640-707)
def ttt(self, inputs):
    B, H, NC, C, D = inputs.shape  # B=1, H=8, NC=94, C=8, D=512

    if self.persistent_states:  # TRUE (default)
        # Initialize from current parameters
        W1_states = torch.tile(self.W1.unsqueeze(0), dims=(B, 1, 1, 1))
        # self.W1 = initial weights (from checkpoint or previous update)

        # Run TTT inner loop
        XQW_batch, final_states = ttt_mlp_with_states(
            ..., W1_states, b1_states, W2_states, b2_states, ...
        )
        # TTT performs gradient descent: W1 → W1' → W1'' → ... → W1_final

        # Update parameters with final states
        with torch.no_grad():
            self.W1.data.copy_(final_states["W1_states"][0])  # W1 ← W1_final
            self.b1.data.copy_(final_states["b1_states"][0])  # b1 ← b1_final
            self.W2.data.copy_(final_states["W2_states"][0])  # W2 ← W2_final
            self.b2.data.copy_(final_states["b2_states"][0])  # b2 ← b2_final
        # ↑ TTT weights now hold final states after processing File1_Chunk0

        return XQW_batch

# Back in training loop
text_loss = compute_loss_with_mask(...)  # Cross-entropy loss
audio_loss = compute_loss_with_mask(...)
mb_loss = text_loss + audio_loss  # Total loss

mb_loss.backward()  # Compute gradients
# ↑ PyTorch computes: ∂loss/∂W1_initial
# ↑ But W1 now contains W1_final (overwritten by copy_()!)
# ↑ GRADIENT MISMATCH!

optimizer.step()  # Update: W1 = W1_final - lr × ∂loss/∂W1_initial
# ↑ This is mathematically incorrect!
# ↑ Gradients correspond to initial weights, not final weights
```

**State after iteration 1**:
- `self.W1` = W1_final (from TTT) - lr × grad(W1_initial) (from optimizer)
- TTT state adapted to File 1, Chunk 0
- **CONTAMINATED**: Optimizer update based on wrong gradients

#### **Iteration 2: File 1, Chunk 1 (10-20s)**

```python
# Data loader yields
batch = {
    codes: Tensor[1, 9, 750],
    file_id: "/data/file1.wav",  # Same file
    chunk_index: 1               # Next chunk
}

# Training loop
optimizer.zero_grad()
output = model(codes=batch.codes)

# Inside TTT layer
def ttt(self, inputs):
    if self.persistent_states:
        # Initialize from current parameters (carry over from Iteration 1)
        W1_states = torch.tile(self.W1.unsqueeze(0), dims=(B, 1, 1, 1))
        # ↑ Continues from previous state (INTENDED for same file)

        # Run TTT inner loop
        XQW_batch, final_states = ttt_mlp_with_states(...)
        # TTT adapts further: W1_prev → W1_new

        # Update parameters
        with torch.no_grad():
            self.W1.data.copy_(final_states["W1_states"][0])

        return XQW_batch

mb_loss.backward()
optimizer.step()  # Same gradient mismatch issue
```

**State after iteration 2**:
- TTT state adapted to File 1, Chunks 0-1 (GOOD - maintains continuity)
- Gradient mismatch issue persists (BAD)

#### **Iteration 3: File 1, Chunk 2 (20-30s)**

```python
batch = {
    file_id: "/data/file1.wav",
    chunk_index: 2
}

# Same process, TTT continues adapting to File 1
```

**State after iteration 3**:
- TTT state adapted to File 1, Chunks 0-2 (GOOD)

#### **Iteration 4: File 2, Chunk 0 (0-10s) ⚠️ NEW FILE**

```python
# Data loader yields
batch = {
    codes: Tensor[1, 9, 750],
    file_id: "/data/file2.wav",  # ← DIFFERENT FILE!
    chunk_index: 0                # ← First chunk of new file
}

# Training loop
optimizer.zero_grad()
output = model(codes=batch.codes)

# Inside TTT layer
def ttt(self, inputs):
    if self.persistent_states:
        # Initialize from current parameters
        W1_states = torch.tile(self.W1.unsqueeze(0), dims=(B, 1, 1, 1))
        # ↑ self.W1 contains state from File 1!
        # ↑ NO RESET - File 2 starts with File 1's final state!
        # ↑ CROSS-FILE CONTAMINATION!

        # Run TTT inner loop
        XQW_batch, final_states = ttt_mlp_with_states(...)
        # TTT adapts: File1_state → File2_state
        # ↑ But it should start fresh for File 2!

        with torch.no_grad():
            self.W1.data.copy_(final_states["W1_states"][0])

        return XQW_batch
```

**🔴 CRITICAL PROBLEM**:
- File 2 processing starts with TTT weights adapted to File 1
- No state reset between files
- File 2's TTT adaptation is contaminated by File 1's patterns
- Different speakers, different acoustic conditions, different contexts - all mixed!

**Why this is bad**:
1. File 1 might be a male speaker with background noise
2. File 2 might be a female speaker in a quiet environment
3. TTT should learn to adapt to File 2 independently
4. Instead, File 2 adaptation starts from File 1's final state
5. This corrupts the two-loop learning paradigm

#### **Iteration 5: File 2, Chunk 1 (10-20s)**

```python
batch = {
    file_id: "/data/file2.wav",
    chunk_index: 1
}

# TTT continues adapting File 2, but still contaminated by File 1's initial influence
```

---

### 4. Loss Computation Verification

**File**: `moshi_ttt_try/moshi-finetune/finetune/loss.py`

```python
def compute_loss_with_mask(
    logits: torch.Tensor,      # Model predictions
    target: torch.Tensor,      # Ground truth tokens
    target_mask: torch.Tensor, # Valid token mask
    mode: str,                 # "text" or "audio"
    first_codebook_weight_multiplier: float = 1.0,
    text_padding_weight: float = 1.0,
    text_padding_ids: set[int] | None = None,
):
    # Zero out invalid targets
    target = torch.where(target_mask, target, torch.zeros_like(target))

    # Create loss weights
    weights = target_mask.float()
    if mode == "audio":
        weights[:, 0] *= first_codebook_weight_multiplier  # Weight first codebook more
    elif mode == "text":
        assert text_padding_ids is not None
        for id in text_padding_ids:
            weights[target == id] *= text_padding_weight  # Downweight padding tokens

    # Compute cross-entropy loss
    logits = logits.view(-1, logits.size(-1)).float()
    target = target.view(-1)
    weights = weights.view(-1)
    mb_loss = F.cross_entropy(logits, target, reduction="none")
    mb_loss = torch.where(weights > 0.0, mb_loss * weights, torch.zeros_like(mb_loss))
    mb_loss = torch.sum(mb_loss) / torch.sum(weights)  # Weighted average

    return mb_loss
```

**✅ Loss computation is correct**:
- Standard cross-entropy loss for next-token prediction
- Proper masking for invalid tokens
- Appropriate weighting for different token types
- No issues found

---

### 5. Checkpoint Save/Load Verification

**File**: `moshi_ttt_try/moshi-finetune/training/train_ttt_production.py`

#### 5.1 Checkpoint Saving (Lines 399-413)

```python
if args.do_ckpt and args.ckpt_freq > 0 and step % args.ckpt_freq == 0:
    checkpoint_path = run_dir / f"checkpoint_step_{step}.pt"
    print(f"💾 Saving checkpoint: {checkpoint_path}")

    checkpoint_data = {
        'step': step,
        'model_state_dict': model.state_dict(),  # All model parameters
        'optimizer_state_dict': optimizer.state_dict(),  # Optimizer state
        'scheduler_state_dict': scheduler.state_dict(),  # LR scheduler state
        'loss': loss_item,
        'config': args.__dict__,
    }

    torch.save(checkpoint_data, checkpoint_path)
    print(f"✅ Checkpoint saved")
```

**What `model.state_dict()` includes**:
- All transformer parameters (embeddings, attention, FFN, layer norms)
- All TTT parameters (W1, b1, W2, b2, ttt_norm)
- All LoRA parameters (if enabled)

**✅ Checkpoint saving is correct**: All TTT parameters are included in `model.state_dict()`.

#### 5.2 Final Model Saving (Lines 424-432)

```python
# Save final model
final_model_path = run_dir / "final_model.pt"
torch.save({
    'model_state_dict': model.state_dict(),
    'config': lm_config,
    'training_args': args.__dict__,
    'final_loss': losses[-1],
    'total_steps': args.max_steps,
}, final_model_path)
print(f"💾 Final model saved: {final_model_path}")
```

**✅ Final model saving is correct**: Includes model weights and config.

---

### 6. Inference Flow Verification (Already Correct)

**File**: `moshi_ttt_try/moshi-finetune/inference/run_inference_with_ttt.py`

#### 6.1 Key Differences from Training

| Aspect | Training | Inference |
|--------|----------|-----------|
| **Mode** | `model.train()` | `model.eval()` |
| **Gradients** | Enabled | `torch.no_grad()` |
| **Optimizer** | Running | Not running |
| **TTT Updates** | Conflict with optimizer | Pure test-time training |
| **persistent_states** | Causes Issue #4 | Works correctly |
| **State Reset** | Missing (Issue #5) | Not needed (single file per run) |

#### 6.2 Why Inference is Correct

```python
# main() - line 790
if args.infile:
    model.eval()  # ← CRITICAL: Sets model.training = False
    with torch.no_grad():  # ← CRITICAL: No gradient computation
        success = run_audio_inference(
            model=model,
            mimi=mimi,
            text_tokenizer=text_tokenizer,
            ...,
            audio_path=args.infile
        )
```

**Inside TTT during inference** (`ttt_layer.py:640-707`):
```python
def ttt(self, inputs):
    if self.training:  # ← FALSE during inference
        checkpoint_group_size = min(...)
    else:
        checkpoint_group_size = 0  # ← No checkpointing during eval

    if self.persistent_states:  # ← TRUE (appropriate for streaming)
        W1_states = torch.tile(self.W1.unsqueeze(0), dims=(B, 1, 1, 1))
        XQW_batch, final_states = ttt_mlp_with_states(...)

        with torch.no_grad():  # Already inside torch.no_grad() from line 791
            self.W1.data.copy_(final_states["W1_states"][0])
            # ↑ Test-time training: Adapt to current input
            # ↑ No optimizer to conflict with!
            # ↑ No gradients computed!
            # ↑ This is INTENDED behavior!

        return XQW_batch
```

**✅ Inference is correct** because:
1. No optimizer running → No gradient-based updates
2. `torch.no_grad()` → No autograd graph → No gradient flow issues
3. `model.eval()` → No training-specific behaviors
4. TTT updates are pure test-time training (forward-only adaptation)
5. Single file per script invocation → No cross-file contamination

---

## All Issues Detailed

### Issue #1: Ring Buffer Information Loss (Architectural)

**File**: `moshi/moshi/moshi/modules/transformer.py:187-233`

**Problem**: RingKVCache uses modulo wraparound, permanently discarding old tokens.

```python
class RingKVCache:
    def __init__(self, capacity: int, ...):
        self.capacity = capacity  # e.g., 3000 tokens
        self.buffer = torch.zeros(num_layers, 2, num_heads, capacity, dim)
        self.position = 0

    def update(self, k, v):
        indexes = torch.arange(batch_size, device=k.device) + self.position
        indexes = indexes % self.capacity  # ← OVERWRITES OLD TOKENS!
        self.buffer[layer_idx, 0, :, indexes] = k
        self.buffer[layer_idx, 1, :, indexes] = v
        self.position += batch_size
```

**Impact**:
- After 3000 steps (12.5 seconds at 12Hz), old tokens are overwritten
- TTT can only see recent 3000 tokens, not full history
- Limits long-term adaptation capability

**Severity**: 🔴 Critical (architectural limitation)

---

### Issue #2: Normalization Bug

**File**: `moshi_ttt_try/moshi-finetune/moshi_ttt/ttt_layer.py:476`

**Problem**: LayerNorm applied to XV directly instead of difference (XV - XK).

```python
# CURRENT (INCORRECT):
def ln_reconstruction_target(self, reconstruction_target):
    if self.reconstruction_target_norm:
        return F.layer_norm(
            reconstruction_target,  # ← Should be (XV - XK), but is XV!
            normalized_shape=(reconstruction_target.size(-1),),
            weight=None,
            bias=None,
            eps=1e-5,
        )
    return reconstruction_target

# In compute_mini_batch (ttt_mlp.py:230):
reconstruction_target = XV_mini_batch - XK_mini_batch  # Correct difference
reconstruction_target = ln_reconstruction_target(XV_mini_batch)  # ← WRONG! Uses XV only
```

**Should be**:
```python
reconstruction_target = ln_reconstruction_target(XV_mini_batch - XK_mini_batch)
```

**Impact**:
- TTT optimizes wrong objective
- Reconstruction target includes XK component (should be removed)
- Reduces TTT adaptation effectiveness

**Severity**: 🟡 Medium (correctness issue)

---

### Issue #3: Batch Size > 1 Incompatibility

**File**: `moshi_ttt_try/moshi-finetune/moshi_ttt/ttt_layer.py:686-707`

**Problem**: persistent_states copies state from batch index 0 only.

```python
with torch.no_grad():
    self.W1.data.copy_(final_states["W1_states"][0])  # ← [0] = first batch item
    self.b1.data.copy_(final_states["b1_states"][0])
    self.W2.data.copy_(final_states["W2_states"][0])
    self.b2.data.copy_(final_states["b2_states"][0])
```

**Impact**:
- If batch_size > 1, only first item's state is used
- Other items in batch are ignored
- Incorrect state management for multi-item batches

**Severity**: 🟡 Medium (configuration constraint)

**Workaround**: Use batch_size=1 for training with persistent_states

---

### Issue #4: Gradient Flow Corruption (CRITICAL)

**File**: `moshi_ttt_try/moshi-finetune/moshi_ttt/ttt_layer.py:686-707`
**Config**: `moshi_ttt_try/moshi-finetune/finetune/args.py:74`

**Problem**: `self.W1` serves two conflicting roles:
1. **Trainable parameter** (updated by optimizer based on gradients)
2. **Persistent state** (overwritten by TTT inner loop)

**The conflict**:
```python
# Forward pass
def ttt(self, inputs):
    # 1. Read initial weights
    W1_states = torch.tile(self.W1.unsqueeze(0), ...)  # W1_initial

    # 2. Run TTT inner loop (creates autograd graph)
    XQW_batch, final_states = ttt_mlp_with_states(
        ..., W1_states, ...
    )  # W1_initial → ... → W1_final

    # 3. Overwrite parameters
    with torch.no_grad():
        self.W1.data.copy_(final_states["W1_states"][0])  # W1 ← W1_final

# Backward pass
loss.backward()
# PyTorch computes: ∂loss/∂W1_initial
# But self.W1 now contains W1_final!

# Optimizer step
optimizer.step()
# Updates: W1 = W1_final - lr × ∂loss/∂W1_initial
# ↑ MISMATCH: Gradients for initial, but applied to final!
```

**Why this happens**:
- PyTorch's autograd creates computation graph during forward
- Graph records: loss depends on W1_initial
- But `self.W1.data.copy_()` overwrites W1 in-place (no autograd tracking)
- Gradients computed for W1_initial, but applied to W1_final
- Optimizer and TTT fight over same parameter

**Root cause**: Conflating W_base (trainable initialization) with W_state (ephemeral adaptation).

**Severity**: 🔴 Critical (training corruption)

**Enabled by default**: `persistent_states: bool = True` in `finetune/args.py:74`

---

### Issue #5: Cross-File State Contamination (NEW - CRITICAL)

**Files**:
- Data: `moshi_ttt_try/moshi-finetune/finetune/data/dataset.py:250-261`
- Training: `moshi_ttt_try/moshi-finetune/training/train_ttt_production.py:234-298`

**Problem**: TTT states persist across file boundaries without reset.

**Sequence of events**:

1. **Data pipeline tracks file boundaries**:
```python
# interleaver.py:306-315
file_id = path  # Track which file
chunk_index = int(start_sec / self.duration_sec)  # Track chunk position
return Sample(..., file_id=file_id, chunk_index=chunk_index)
```

2. **Training loop ignores file boundaries**:
```python
# train_ttt_production.py:234-298
while step < args.max_steps:
    batch = next(data_loader)  # Has file_id and chunk_index
    codes = batch.codes.to(device)

    # ❌ No check: if batch.file_id != previous_file_id: reset_ttt_states()
    # ❌ TTT weights carry over from previous file!

    output = model(codes=codes)  # Uses contaminated TTT state
```

3. **Result**: File transitions look like this:
```
File 1:
  Chunk 0: TTT adapts A → B
  Chunk 1: TTT adapts B → C
  Chunk 2: TTT adapts C → D

File 2:  ← DIFFERENT SPEAKER, DIFFERENT CONTEXT
  Chunk 0: TTT adapts D → E  ← Should start from A, but starts from D!
  Chunk 1: TTT adapts E → F  ← Contaminated by File 1
```

**Why this is critical**:

| Scenario | Expected Behavior | Actual Behavior | Impact |
|----------|-------------------|-----------------|--------|
| File 1: Male speaker, noisy | TTT adapts to male + noise | ✅ Correct | ✅ Good |
| File 2: Female speaker, quiet | TTT resets and adapts to female | ❌ Continues from File 1 state | 🔴 Contaminated |
| File 3: Child speaker | TTT resets and adapts to child | ❌ Continues from File 2 state | 🔴 Contaminated |

**Violations of two-loop paradigm**:
- **Outer loop (optimizer)**: Should learn W_base that works across files
- **Inner loop (TTT)**: Should adapt W_base to each specific file
- **Current behavior**: Inner loop adaptation bleeds across file boundaries
- **Result**: W_base never receives clean per-file adaptation signals

**Evidence**:
- ✅ Data pipeline tracks `file_id` (interleaver.py:306)
- ✅ Batch contains `file_id` (interleaver.py:21, 28)
- ❌ Training loop doesn't use `file_id` (train_ttt_production.py - no grep matches)
- ❌ No state reset mechanism exists (grep: no `reset.*state` in training/)

**Severity**: 🔴 Critical (corrupts two-loop learning)

**Affects**: Training with `shuffle=False` and `persistent_states=True` (default configuration)

---

## Verification Checklist

### Training Flow ✅
- [x] Data loading: Verified sequential processing with file tracking
- [x] Chunking: Verified fixed-duration chunks from long files
- [x] Batch creation: Verified file_id and chunk_index populated
- [x] Training loop: Verified no state reset logic
- [x] Loss computation: Verified correct cross-entropy
- [x] Backward pass: Verified gradient computation
- [x] Optimizer step: Verified parameter updates
- [x] Checkpoint saving: Verified all parameters saved

### Inference Flow ✅
- [x] Model eval mode: Verified `model.eval()` called
- [x] No gradients: Verified `torch.no_grad()` wraps inference
- [x] Streaming setup: Verified `streaming_forever()` called
- [x] TTT updates: Verified test-time training (no optimizer)
- [x] State management: Verified single file per run (no contamination)

### Data Pipeline ✅
- [x] File chunking: Verified duration-based splitting
- [x] Sequential processing: Verified `dataset.seq()` when shuffle=False
- [x] File tracking: Verified file_id and chunk_index populated
- [x] Batch collation: Verified file_id preserved in batch

### State Management ❌
- [x] Ring buffer: Verified modulo wraparound (Issue #1)
- [x] TTT parameters: Verified W1, W2, b1, b2 handling
- [x] Persistent states: Verified enabled by default
- [x] Cross-file reset: **Missing** (Issue #5)
- [x] Batch size handling: Verified only index 0 used (Issue #3)

### Gradient Flow ❌
- [x] Training mode: Verified `model.train()` called
- [x] Autograd graph: Verified created during forward
- [x] TTT inner loop: Verified differentiable operations
- [x] Parameter updates: Verified both TTT and optimizer update W1
- [x] Gradient mismatch: **Confirmed** (Issue #4)

### TTT Implementation ⚠️
- [x] Mini-batch processing: Verified sequential mini-batches
- [x] Reconstruction target: **Bug found** (Issue #2)
- [x] State persistence: Verified copy_() mechanism
- [x] Normalization: **Incorrect** (Issue #2)
- [x] Checkpointing: Verified disabled during eval

---

## Summary of Findings

### Critical Issues (Must Fix)

1. **Issue #4: Gradient Flow Corruption**
   - Conflates W_base (trainable) with W_state (ephemeral)
   - Optimizer and TTT fight over same parameters
   - Corrupts training convergence
   - **Fix**: Separate W_base and W_state (see MOSHI_TTT_FIXES.md)

2. **Issue #5: Cross-File State Contamination** (NEW)
   - TTT states bleed across file boundaries
   - Violates two-loop learning paradigm
   - Mixes different speakers/contexts
   - **Fix**: Detect file boundaries and reset TTT states

3. **Issue #1: Ring Buffer Information Loss**
   - Architectural limitation (3000-token capacity)
   - Prevents long-term adaptation
   - **Fix**: Implement chunked attention (see MOSHI_TTT_FIXES.md)

### Medium Priority Issues

4. **Issue #2: Normalization Bug**
   - Incorrect reconstruction target
   - Reduces TTT effectiveness
   - **Fix**: Normalize difference (XV - XK), not XV alone

5. **Issue #3: Batch Size > 1 Incompatibility**
   - Only first batch item's state used
   - **Workaround**: Use batch_size=1 with persistent_states

### Verified Correct

- ✅ Inference implementation (no changes needed)
- ✅ Loss computation (standard cross-entropy)
- ✅ Checkpoint saving/loading (includes all TTT parameters)
- ✅ Data pipeline (tracks file boundaries correctly)
- ✅ TTT inner loop mechanics (differentiable, correct structure)

---

## Next Steps

1. **Add Issue #5 to MOSHI_TTT_CRITICAL_ISSUES.md**
   - Document cross-file contamination
   - Provide evidence from code traces
   - Show impact on two-loop learning

2. **Update MOSHI_TTT_FIXES.md**
   - Add fix for Issue #5
   - Provide state reset implementation
   - Show how to detect file boundaries

3. **Update POST_FIXES_ANALYSIS.md**
   - Include impact of fixing Issue #5
   - Update expected training behavior
   - Revise performance predictions

4. **Implementation Priority**:
   - **Highest**: Issue #4 (gradient flow) + Issue #5 (cross-file contamination)
   - **High**: Issue #1 (ring buffer) or quick fix (increase capacity)
   - **Medium**: Issue #2 (normalization) + Issue #3 (batch size)

---

## Conclusion

This comprehensive verification confirms:

1. **5 issues identified** (4 previously documented + 1 NEW)
2. **Training flow traced end-to-end** with complete data pipeline analysis
3. **Inference verified correct** (no changes needed)
4. **Critical findings**:
   - Gradient flow corruption (Issue #4)
   - Cross-file state contamination (Issue #5) ← NEW
   - Ring buffer information loss (Issue #1)

The most surprising finding is **Issue #5**: Despite the data pipeline correctly tracking file boundaries (`file_id` and `chunk_index`), the training loop completely ignores this information and never resets TTT states between files. This is a fundamental violation of the two-loop learning paradigm and likely a major contributor to training failure.

**Recommendation**: Fix Issues #4 and #5 together as they both relate to state management:
- Issue #4: Separate W_base and W_state
- Issue #5: Reset W_state at file boundaries

Both fixes require modifying the TTT layer's state management and the training loop's batch processing logic.

---

**Analysis Date**: 2025-11-10
**Verification Status**: Complete - All flows traced and verified
**Confidence Level**: Very High (exhaustive end-to-end analysis)
