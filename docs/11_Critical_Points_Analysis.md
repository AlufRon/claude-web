# Critical Points Analysis: Docs vs. Implementation Plan

**Comparison of critical requirements from docs 00-09 against implementation plan (doc 10)**

---

## ✅ ADDRESSED: Critical Points from Docs

### 1. FP32 Precision (MANDATORY)

**From Docs**:
- **Doc 05**: "TTT inner states MUST be FP32" - explicit code showing `dtype=torch.float32`
- **Doc 06**: "FP32 precision verified for W1, b1" - multiple assertions
- **Reason**: Numerical stability during gradient descent in inner loop

**In Plan**:
```python
# Section 2.1: FP32 Precision (MANDATORY)
self.W1 = nn.Parameter(
    torch.zeros(num_heads, head_dim, head_dim,
                dtype=torch.float32)  # ← FP32!
)
assert self.W1.dtype == torch.float32

# Section 5.1: FP32 Enforcement Checklist
# - Parameter initialization ✓
# - Forward pass verification ✓
# - Optimizer setup with separate param groups ✓
```

**Status**: ✅ **FULLY ADDRESSED**
- Explicit FP32 initialization
- Runtime assertions
- Separate optimizer param groups to prevent downcasting
- Verification hooks

---

### 2. State Persistence (NO Reset Mid-Conversation)

**From Docs**:
- **Doc 05**: "State reset bug in training: TTT state resets per batch instead of per conversation"
- **Doc 06**: "CRITICAL: Ensure state persists across batches in SAME conversation"
- **Reason**: Resetting state destroys learned context → gibberish

**In Plan**:
```python
# Section 3.3: State Management
@dataclass
class TTTConversationState:
    W1: torch.Tensor
    b1: torch.Tensor
    conversation_id: str
    turn_number: int

class TTTStateManager:
    def get_or_create_state(self, conversation_id, ...):
        # Only create if doesn't exist

    def reset_conversation(self, conversation_id):
        # Only called at conversation boundaries

# Section 5.2: State Persistence Strategy
for batch in dataloader:
    conversation_id = batch['conversation_id']
    turn_number = batch['turn_number']

    # CRITICAL: Only reset for NEW conversations
    if turn_number == 0:  # First turn only
        model.reset_conversation_state(conversation_id)
```

**Status**: ✅ **FULLY ADDRESSED**
- Explicit state manager with conversation IDs
- Only resets when `turn_number == 0`
- Logging to track unwanted resets
- State checkpointing for recovery

---

### 3. Mini-Batch Size = 64 Tokens

**From Docs**:
- **Doc 07**: "TTT needs mini-batches for stable gradient computation"
- **Doc 09**: "Mini-batch size: 64 (from video-dit)"
- **Reason**: Single-token streaming (Moshi) incompatible with TTT

**In Plan**:
```python
# Section 2.3: Mini-Batch Size
self.mini_batch_size = 64  # From config

# Section 5.3: Mini-Batch Handling
def collate_fn(batch):
    """Custom collator to pad to mini_batch_size"""
    if seq_len % mini_batch_size != 0:
        pad_len = mini_batch_size - (seq_len % mini_batch_size)
        # Pad to nearest multiple of 64

# Runtime check
assert L % self.mini_batch_size == 0, \
    f"Seq len {L} must be multiple of {self.mini_batch_size}"
```

**Status**: ✅ **FULLY ADDRESSED**
- Config parameter: `mini_batch_size = 64`
- Data collator pads sequences
- Runtime assertions
- Test case: `test_mini_batch_size()`

---

### 4. Layer Selection: Replace Layers 24-31 (Top 8)

**From Docs**:
- **Doc 09**: "Replace layers 24-31 (top 8 of 32 Llama layers)"
- **Reason**: Top layers handle abstract reasoning, bottom layers handle low-level features

**In Plan**:
```python
# Section 3.2: Integration into Llama Model
class OmniSpeechTTTConfig(LlamaConfig):
    self.num_ttt_layers = 8
    self.ttt_start_layer = 24  # Layers 24-31

def _replace_attention_with_ttt(self):
    for layer_idx in range(
        self.config.ttt_start_layer,  # 24
        self.config.ttt_start_layer + self.config.num_ttt_layers  # 32
    ):
        ttt_layer = TTTMLP(...)
        original_layer.self_attn = ttt_layer
```

**Status**: ✅ **FULLY ADDRESSED**
- Configurable: `ttt_start_layer` and `num_ttt_layers`
- Default: layers 24-31 (8 layers)
- Can experiment with 4, 8, or 16 layers

---

### 5. No KV Cache Wraparound (Llama-Omni Advantage)

**From Docs**:
- **Doc 05**: "Moshi KV cache wraps @ 3000 tokens → gibberish"
- **Doc 08**: "Llama uses DynamicCache which grows dynamically - no wraparound"
- **Doc 09**: "This perfectly explains the 5-7 minute gibberish problem"
- **Reason**: Root cause of Moshi failure

**In Plan**:
```python
# Section 1.1: Why Llama-Omni?
# "Llama uses transformers DynamicCache - no fixed capacity"

# Phase 5: Testing
def test_long_context():
    """Test with long sequences (32k tokens)"""
    input_ids = torch.randint(0, 1000, (1, 32768)).cuda()
    outputs = model(input_ids=input_ids)

    # Verify no quality degradation
    assert not torch.isnan(outputs.logits).any()
```

**Status**: ✅ **ADDRESSED (via model choice)**
- Llama-Omni chosen specifically to avoid this issue
- Test case verifies no degradation at 32k tokens
- NOTE: This is a model architecture advantage, not something we implement

---

### 6. Copy from ttt-video-dit (PyTorch, not JAX)

**From Docs**:
- **Doc 10 Plan**: Use ttt-video-dit as PRIMARY SOURCE
- **Reason**: Same framework (PyTorch), proven implementation

**In Plan**:
```python
# Section 1.2: TTT Implementation Sources
ttt-video-dit/ttt/models/ssm/  ← PRIMARY SOURCE (PyTorch)
├── ttt_layer.py      # COPY & ADAPT
├── ops/ttt_mlp.py    # COPY 100%
├── ops/ttt_linear.py # COPY 100%
└── ops/utils.py      # COPY utilities

# Section 4: File Copying Matrix
# Detailed source → destination mappings
# What % to copy, what to change
```

**Status**: ✅ **FULLY ADDRESSED**
- Clear source identification
- File-by-file copying matrix
- Specific line counts and change percentages

---

### 7. Streaming vs. Batch Processing

**From Docs**:
- **Doc 07**: "Moshi streams 1 token at a time - incompatible with TTT mini-batches"
- **Doc 08**: "Llama-Omni uses Whisper (batch encoder) - naturally compatible"
- **Reason**: Architectural advantage

**In Plan**:
```python
# Section 1.1: Current Llama-Omni Structure
# Whisper encoder processes full utterance at once (not streaming)
# → Naturally compatible with mini-batch TTT

# Section 2.3: Mini-Batch Size
# "Llama-Omni compatibility: ✅ Whisper processes full utterance"
```

**Status**: ✅ **ADDRESSED (via model choice)**
- Llama-Omni chosen for batch processing compatibility
- Whisper encoder: processes complete utterances
- No need to buffer tokens like in streaming Moshi

---

## ⚠️ PARTIALLY ADDRESSED: Important Points

### 8. Training Data (Long Conversations)

**From Docs**:
- **Doc 09**: "Fine-tune on long conversations (100+ hours)"
- **Doc 10 Appendix B**: Multi-stage training (8k → 16k → 32k)

**In Plan**:
```python
# Section 3.4: Training Infrastructure
# TODO: Implement dataset loading
# Should return long conversations (hours of dialogue)

# Section 10: Open Questions
# Q1: What long-form speech conversations to use for training?
#   - Option A: Generate synthetic with TTS + text dialogues
#   - Option B: Use LibriSpeech/MLS long audiobooks
#   - Option C: Podcast transcripts with speech synthesis
```

**Status**: ⚠️ **ACKNOWLEDGED BUT NOT IMPLEMENTED**
- Recognized as critical requirement
- Listed in "Open Questions"
- **MISSING**: Concrete dataset preparation plan
- **MISSING**: Data pipeline implementation

**What's Needed**:
1. Dataset selection strategy
2. Data preprocessing pipeline
3. Conversation boundary detection
4. Turn number assignment
5. Padding/chunking to 64-token boundaries

---

### 9. Evaluation Metrics

**From Docs**:
- **Doc 09**: Success = "generates coherent text for 10k+ tokens"
- **Doc 10**: "Perplexity, human evaluation, coherence metrics"

**In Plan**:
```python
# Section 8: Success Criteria
- ✅ Generates coherent text for 10k+ tokens
- ✅ No quality degradation at long context
- ✅ Better than baseline on long-form tasks

# Section 10: Open Questions
# Q2: How to measure success?
#   - Perplexity at different sequence lengths
#   - Human evaluation
#   - Coherence metrics (entity, topic consistency)
```

**Status**: ⚠️ **HIGH-LEVEL ONLY**
- General criteria defined
- **MISSING**: Specific evaluation protocol
- **MISSING**: Baseline comparison methodology
- **MISSING**: Automated quality degradation detection

**What's Needed**:
1. Perplexity computation at 1k, 5k, 10k, 30k tokens
2. Human eval protocol (raters, rubric, inter-annotator agreement)
3. Automated metrics: entity tracking, topic coherence, repetition detection
4. Baseline: Llama-Omni without TTT at same sequence lengths

---

### 10. Gradient Checkpointing for Memory

**From Docs**:
- **Doc 05**: Video-DiT uses checkpointing for long sequences
- **Implicit**: Need for 32k+ token sequences

**In Plan**:
```python
# Section 3.1: File: omni_speech/model/ttt/ops/ttt_mlp.py
def ttt_mlp(..., checkpoint_group_size):
    # scan() function handles checkpointing

# Section 6: Risk 4: Memory OOM
# Use gradient checkpointing for scan
# (already implemented via checkpoint_group_size)

# Also enable for Llama layers
model.gradient_checkpointing_enable()
```

**Status**: ✅ **ADDRESSED**
- Checkpointing in scan loop
- Llama gradient checkpointing enabled
- Config parameter: `scan_checkpoint_group_size`

---

## ❌ MISSING: Critical Gaps in Plan

### 11. Inference Strategy for Variable-Length Sequences

**From Docs**:
- **Implicit requirement**: Generation must respect mini-batch boundaries
- **Doc 10**: "Generation must respect mini-batch boundaries"

**In Plan**:
```python
# Section 5.3: Mini-Batch Handling
def collate_fn(batch):
    # Pads training data to mini_batch_size

# Test: test_generation()
# (Important: generation must respect mini-batch boundaries)
```

**Status**: ❌ **INCOMPLETE**

**What's MISSING**:
1. **Inference padding strategy**: How to handle prompts not divisible by 64?
   - Option A: Pad prompt to 64-multiple before generation
   - Option B: Use standard attention for first tokens, TTT after 64-boundary
   - Option C: Start with dummy padding tokens, replace later

2. **Generation loop modification**: Standard `model.generate()` doesn't know about mini-batch constraints
   ```python
   # Current HuggingFace generate:
   for step in range(max_new_tokens):
       logits = model(input_ids)  # Any length
       next_token = sample(logits)
       input_ids = cat([input_ids, next_token])

   # TTT-compatible generate: ???
   # How to handle when input_ids not multiple of 64?
   ```

3. **State initialization for inference**:
   - Training: state persists across batches
   - Inference: when to reset state? Per prompt? Per session?

**What's Needed**:
```python
# File: omni_speech/model/language_model/omni_speech_llama_ttt.py

def generate(
    self,
    input_ids,
    conversation_id=None,
    reset_state=False,
    **kwargs
):
    """
    TTT-aware generation

    Steps:
    1. Pad input_ids to mini_batch_size multiple
    2. Initialize/load conversation state
    3. Generate with state persistence
    4. Handle mini-batch boundaries during sampling
    """

    # MISSING IMPLEMENTATION
    pass

# Also need:
def _pad_to_mini_batch_boundary(self, input_ids):
    """Pad input to nearest 64-multiple"""
    pass

def _prepare_ttt_state_for_generation(self, conversation_id, reset):
    """Get or create state for generation"""
    pass
```

---

### 12. Mixed Precision Training Details

**From Docs**:
- **Doc 05**: "FP32 for W1, b1 but BF16/FP16 for activations"
- **Doc 06**: "Convert to FP32 for computation, back to mixed precision"

**In Plan**:
```python
# Section 5.1: FP32 Enforcement
# Convert inputs to FP32 for TTT computation
hidden_states_fp32 = hidden_states.float()
# ... TTT computation in FP32 ...
return output.to(hidden_states.dtype)

# Optimizer with separate parameter groups
```

**Status**: ⚠️ **HIGH-LEVEL ONLY**

**What's MISSING**:
1. **Explicit autocast handling**:
   ```python
   # How does torch.cuda.amp.autocast interact with TTT?
   with torch.cuda.amp.autocast():
       outputs = model(inputs)  # Are W1, b1 safe?
   ```

2. **Gradient scaler configuration**:
   ```python
   # Do TTT gradients need different scaling?
   scaler = GradScaler()
   # Should TTT params be excluded?
   ```

3. **FSDP mixed precision**:
   ```python
   # Training args mention FSDP
   # Does FSDP try to cast W1, b1 to BF16?
   # Need explicit MixedPrecision policy?
   ```

**What's Needed**:
```python
# File: train_ttt.py

from torch.distributed.fsdp import MixedPrecision

# Explicit mixed precision policy
mp_policy = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16,
    # BUT: exclude TTT parameters
)

# OR: Custom wrapper to protect FP32 params
class FP32ProtectedFSDP(FSDP):
    def __init__(self, module, ...):
        # Mark W1, b1 as no-cast
        for name, param in module.named_parameters():
            if 'W1' in name or 'b1' in name or 'W2' in name or 'b2' in name:
                param._no_mixed_precision = True
```

---

### 13. Hyperparameter Tuning Strategy

**From Docs**:
- **Doc 09**: "TTT learning rate: 1.0 (from video-dit) - may need tuning"
- **Doc 10**: "Mini-batch size: 64 - verify optimal"

**In Plan**:
```python
# Section 10: Open Questions
# Q3: Hyperparameters:
#   - Mini-batch size: 64 - verify optimal
#   - TTT learning rate: 1.0 - may need tuning
#   - Number of TTT layers: 8 - test 4 vs 8 vs 16
```

**Status**: ❌ **NOT ADDRESSED**

**What's MISSING**:
1. **No hyperparameter search plan**
   - Which params to tune? (eta_base, mini_batch_size, num_layers)
   - Search space definition
   - Evaluation metric for comparison

2. **No ablation study design**
   - TTT-Linear vs. TTT-MLP
   - 4 layers vs. 8 layers vs. 16 layers
   - Mini-batch 32 vs. 64 vs. 128

3. **No transfer from video-dit analysis**
   - Video-dit uses 64 tokens - why?
   - Is this optimal for speech?
   - Speech tokens are ~10 Hz, video tokens are different rate

**What's Needed**:
```python
# File: experiments/hyperparameter_search.py

search_space = {
    'ttt_base_lr': [0.1, 0.5, 1.0, 2.0],
    'mini_batch_size': [32, 64, 128],
    'num_ttt_layers': [4, 8, 16],
    'ttt_layer_type': ['linear', 'mlp'],
}

# Run grid search or Bayesian optimization
# Eval metric: Perplexity @ 10k tokens
```

---

### 14. RoPE (Rotary Position Embedding) Handling

**From Docs**:
- **Doc 10**: "Use transformers built-in RoPE instead of 3D version"
- **Video-DiT**: Uses 3D RoPE for video (time, height, width)

**In Plan**:
```python
# Section 3.1: What to copy
# REMOVE (use transformers built-in):
❌ precompute_freqs_cis_3d()  # 3D RoPE (video-specific)
❌ apply_rotary_emb()         # Use transformers version instead

# Section 3.2: TTT Layer Forward
# Apply RoPE (use transformers built-in)
XQ, XK = self._apply_rope(XQ, XK, position_ids)
```

**Status**: ⚠️ **ACKNOWLEDGED, NOT IMPLEMENTED**

**What's MISSING**:
1. **Specific RoPE implementation**:
   ```python
   def _apply_rope(self, XQ, XK, position_ids):
       # Which transformers function to use?
       # apply_rotary_pos_emb() from modeling_llama.py?
       # Or compute freqs_cis ourselves?
       pass
   ```

2. **Position IDs for mini-batches**:
   ```python
   # After reshape to [B, num_mb, mb_size, D]
   # What are the position_ids?
   # Option A: Global positions (0, 1, 2, ..., 8191)
   # Option B: Local positions per mini-batch (0-63, 0-63, ...)
   # Option C: Hybrid?
   ```

3. **RoPE scaling for long context**:
   ```python
   # Llama 3.1 supports 128k context with RoPE scaling
   # Does this apply to TTT layers?
   # Need to pass rope_scaling config?
   ```

**What's Needed**:
```python
# File: omni_speech/model/ttt/ttt_layer.py

from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    LlamaRotaryEmbedding
)

class TTTMLP(nn.Module):
    def __init__(self, config):
        # Initialize RoPE
        self.rotary_emb = LlamaRotaryEmbedding(
            dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

    def _apply_rope(self, XQ, XK, position_ids):
        cos, sin = self.rotary_emb(XQ, position_ids)
        XQ, XK = apply_rotary_pos_emb(XQ, XK, cos, sin)
        return XQ, XK
```

---

### 15. Bi-Directional TTT (Video-DiT Uses This)

**From Docs**:
- **Video-DiT code**: Uses bi-directional TTT for non-causal diffusion
- **Llama-Omni**: Causal autoregressive model

**In Plan**:
```python
# Section 3.1: What to REMOVE
❌ interleave()      # Scene interleaving
❌ undo_interleave() # Scene de-interleaving

# Implicitly removed bi-directional processing
```

**Status**: ⚠️ **ASSUMED CAUSAL, NOT EXPLICIT**

**What's MISSING**:
1. **Explicit decision**: Causal vs. non-causal TTT
   - Video-DiT: Non-causal (diffusion denoising)
   - LLM: Causal (autoregressive generation)
   - Speech-to-speech: Causal for generation, but...?

2. **Impact on implementation**:
   ```python
   # Video-DiT bi-directional:
   TTT_fwd = TTT(X)              # Process forward
   TTT_bwd = TTT(reverse(X))     # Process backward
   output = TTT_fwd + TTT_bwd    # Combine

   # For LLM generation:
   # Can we only use forward direction?
   # Or do we want backward for better context?
   ```

3. **Training vs. Inference**:
   - Training: Could use bi-directional (full sequence available)
   - Inference: Must be causal (can't see future)
   - Need different forward() for train/inference?

**What's Needed**:
```python
# File: omni_speech/model/ttt/ttt_layer.py

class TTTMLP(nn.Module):
    def __init__(self, config):
        self.causal_only = config.causal_ttt  # Config option

    def forward(self, hidden_states, position_ids=None, use_cache=False):
        if self.training and not self.causal_only:
            # Bi-directional for training (better context)
            fwd = self._forward_ttt(hidden_states, position_ids)
            bwd = self._forward_ttt(hidden_states.flip(1), position_ids)
            return fwd + bwd.flip(1)
        else:
            # Causal only for inference
            return self._forward_ttt(hidden_states, position_ids)
```

---

### 16. Speech-Specific Adaptations

**From Docs**:
- **Doc 08**: "Llama-Omni uses Whisper encoder → speech projector → Llama"
- **Doc 10**: Plan focuses on Llama layers only

**In Plan**:
```python
# Section 1.1: Llama-Omni Structure
├── speech_encoder/     ← Whisper encoder
├── speech_projector/   ← Speech->LLM projection
└── language_model/     ← Llama (where we add TTT)

# TTT is added ONLY to Llama layers
# Speech encoder/projector unchanged
```

**Status**: ⚠️ **LLM-ONLY FOCUS**

**What's POTENTIALLY MISSING**:
1. **Speech embedding integration**:
   ```python
   # Input to Llama layer 0:
   # - Text tokens: [B, text_len, D]
   # - Speech tokens: [B, speech_len, D]  # From projector

   # When TTT processes mini-batches:
   # - Mixed text/speech in same mini-batch?
   # - Separate handling needed?
   ```

2. **Speech-specific position encoding**:
   - Text: discrete token positions
   - Speech: continuous time (10 Hz from Whisper)
   - Does RoPE need adjustment for speech tokens?

3. **Speech generator interaction**:
   ```python
   # Llama outputs → Speech generator (CTC + HiFi-GAN)
   # Does TTT affect speech generation quality?
   # Need to test speech output, not just text!
   ```

**What's NEEDED**:
```python
# File: tests/test_speech_integration.py

def test_speech_to_speech_with_ttt():
    """Test full speech-to-speech pipeline"""

    # Load model with TTT
    model = OmniSpeechLlamaForCausalLMTTT(...)

    # Process speech input
    speech_input = load_audio("test.wav")

    # Generate text
    text_output = model.generate(speech=speech_input)

    # Generate speech
    speech_output = model.generate_speech(text_output)

    # Verify speech quality (not just text!)
    # - Intelligibility
    # - Prosody
    # - Voice consistency
```

---

## 📊 Summary: Coverage Analysis

### Fully Addressed (7/16 = 44%)
1. ✅ FP32 Precision
2. ✅ State Persistence
3. ✅ Mini-Batch Size = 64
4. ✅ Layer Selection (24-31)
5. ✅ No KV Wraparound (model choice)
6. ✅ Copy from ttt-video-dit
7. ✅ Batch Processing Compatible

### Partially Addressed (4/16 = 25%)
8. ⚠️ Training Data (acknowledged, not implemented)
9. ⚠️ Evaluation Metrics (high-level only)
10. ⚠️ Gradient Checkpointing (mentioned, not detailed)
11. ⚠️ Mixed Precision (high-level only)

### Missing/Incomplete (5/16 = 31%)
12. ❌ Inference Strategy for Variable-Length
13. ❌ Mixed Precision Training Details
14. ❌ Hyperparameter Tuning
15. ❌ RoPE Handling
16. ❌ Speech-Specific Adaptations

---

## 🔥 CRITICAL: What MUST Be Added to Plan

### Priority 1: Before Implementation Starts

1. **Inference Generation Strategy** (Section 3.2 addition)
   ```python
   # Add to omni_speech_llama_ttt.py
   def generate(...):
       # Pad to mini_batch_size
       # Handle state initialization
       # Mini-batch-aware sampling
   ```

2. **RoPE Implementation** (Section 3.1 addition)
   ```python
   # Specify exact RoPE function to use
   # Position IDs for mini-batches
   # Long-context RoPE scaling
   ```

3. **Mixed Precision Policy** (Section 3.4 addition)
   ```python
   # FSDP MixedPrecision config
   # Autocast handling
   # Gradient scaler setup
   ```

### Priority 2: Before Training Starts

4. **Dataset Preparation Pipeline** (New section 3.5)
   ```python
   # Dataset selection
   # Preprocessing steps
   # Conversation boundary detection
   # Padding to 64-boundaries
   ```

5. **Evaluation Protocol** (New section 4.5)
   ```python
   # Perplexity computation
   # Human eval rubric
   # Baseline comparison
   # Automated quality metrics
   ```

### Priority 3: During Experimentation

6. **Hyperparameter Search** (New section 6.5)
   ```python
   # Search space definition
   # Ablation studies
   # Transfer analysis from video-dit
   ```

7. **Speech Quality Testing** (Add to section 5.5)
   ```python
   # Speech-to-speech pipeline test
   # Voice quality metrics
   # Intelligibility testing
   ```

---

## 📝 Recommended Plan Updates

### Add New Section: 3.6 Inference Implementation

```python
# File: omni_speech/model/language_model/omni_speech_llama_ttt.py

class OmniSpeechLlamaForCausalLMTTT(LlamaForCausalLM):

    def _pad_input_for_ttt(self, input_ids):
        """Pad input to mini_batch_size multiple"""
        B, L = input_ids.shape
        remainder = L % self.config.mini_batch_size

        if remainder != 0:
            pad_len = self.config.mini_batch_size - remainder
            padding = torch.full(
                (B, pad_len),
                self.config.pad_token_id,
                device=input_ids.device
            )
            input_ids = torch.cat([input_ids, padding], dim=1)

        return input_ids, remainder

    def generate(
        self,
        input_ids=None,
        speech=None,
        speech_lengths=None,
        conversation_id=None,
        reset_state=False,
        **kwargs
    ):
        """TTT-aware generation"""

        # Handle speech input
        if speech is not None:
            input_ids, inputs_embeds = self.prepare_inputs_for_speech(
                input_ids, speech, speech_lengths
            )

        # Pad to mini_batch_size
        input_ids, pad_len = self._pad_input_for_ttt(input_ids)

        # Manage TTT state
        if conversation_id:
            if reset_state:
                self.reset_conversation_state(conversation_id)
            # State will persist during generation

        # Standard generation (with TTT in forward pass)
        outputs = super().generate(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

        # Remove padding from outputs if needed
        if pad_len > 0:
            outputs = outputs[:, :-pad_len]

        return outputs
```

### Add New Section: 3.7 Dataset Pipeline

```python
# File: omni_speech/datasets/long_conversation_dataset.py

class LongConversationDataset(Dataset):
    """Dataset for training TTT on long conversations"""

    def __init__(
        self,
        data_path,
        tokenizer,
        mini_batch_size=64,
        max_length=8192,
    ):
        self.conversations = self._load_conversations(data_path)
        self.tokenizer = tokenizer
        self.mini_batch_size = mini_batch_size
        self.max_length = max_length

    def _load_conversations(self, data_path):
        """
        Load conversations with metadata

        Format:
        {
            "conversation_id": "conv_001",
            "turns": [
                {"turn": 0, "speaker": "A", "text": "...", "audio": "..."},
                {"turn": 1, "speaker": "B", "text": "...", "audio": "..."},
                ...
            ]
        }
        """
        # Implementation

    def __getitem__(self, idx):
        conversation = self.conversations[idx]

        # Tokenize all turns
        input_ids = []
        for turn in conversation['turns']:
            turn_ids = self.tokenizer.encode(turn['text'])
            input_ids.extend(turn_ids)

        # Truncate/pad to max_length
        input_ids = input_ids[:self.max_length]

        # Pad to mini_batch_size multiple
        remainder = len(input_ids) % self.mini_batch_size
        if remainder != 0:
            pad_len = self.mini_batch_size - remainder
            input_ids.extend([self.tokenizer.pad_token_id] * pad_len)

        return {
            'input_ids': torch.tensor(input_ids),
            'conversation_id': conversation['conversation_id'],
            'turn_number': 0,  # Start of conversation
        }
```

### Add New Section: 4.6 Evaluation Suite

```python
# File: evaluation/evaluate_long_context.py

class LongContextEvaluator:
    """Comprehensive evaluation for long-context TTT model"""

    def evaluate_perplexity_vs_length(self, model, test_data):
        """
        Measure perplexity at different sequence lengths

        Returns:
            {
                '1k': 12.5,
                '5k': 13.1,
                '10k': 13.8,
                '30k': 14.2,
            }
        """
        results = {}
        for length in [1000, 5000, 10000, 30000]:
            ppl = self._compute_perplexity(model, test_data, max_length=length)
            results[f'{length//1000}k'] = ppl
        return results

    def evaluate_quality_degradation(self, model, prompts):
        """
        Detect quality degradation in generated text

        Metrics:
        - Repetition rate
        - Coherence (semantic similarity between segments)
        - Entity consistency
        - Topic drift
        """
        results = {}
        for prompt in prompts:
            # Generate long text (10k tokens)
            generated = model.generate(prompt, max_new_tokens=10000)

            # Split into segments (1k each)
            segments = self._split_into_segments(generated, segment_size=1000)

            # Measure repetition
            results['repetition'] = self._compute_repetition(segments)

            # Measure coherence
            results['coherence'] = self._compute_coherence(segments)

            # Entity tracking
            results['entity_consistency'] = self._track_entities(segments)

        return results

    def compare_to_baseline(self, ttt_model, baseline_model, test_data):
        """
        Compare TTT model to baseline (Llama-Omni without TTT)

        At same sequence lengths (e.g., 10k tokens):
        - Which has lower perplexity?
        - Which maintains quality better?
        """
        pass
```

---

## 🎯 Final Recommendations

### For the Plan Document (10_TTT_Integration_Plan.md)

1. **Add 3 new sections**:
   - 3.6: Inference Implementation (generation strategy)
   - 3.7: Dataset Pipeline (long conversation data)
   - 4.6: Evaluation Suite (metrics and comparison)

2. **Expand existing sections**:
   - 5.1: Add FSDP/autocast mixed precision details
   - 3.1: Add explicit RoPE implementation
   - 7: Add hyperparameter search to timeline

3. **Create new appendix**:
   - Appendix C: Speech-Specific Considerations
   - Appendix D: Troubleshooting Guide (based on Doc 05-06 issues)

### For Implementation

1. **Start with Priority 1 items** (inference, RoPE, mixed precision)
2. **Validate with small-scale tests** before full training
3. **Keep debug monitoring** from Doc 05-06 (state resets, dtype checks)

---

**Total Coverage**: 7 fully + 4 partially + 5 missing = **68% addressed**

**To reach 100%**: Need to implement the 5 missing items above.
