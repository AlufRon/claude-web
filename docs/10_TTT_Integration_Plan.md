# Comprehensive TTT + Llama-Omni Integration Plan

**Status**: Deep code analysis complete - Ready for implementation
**Date**: 2025-11-08
**Goal**: Add Test-Time Training (TTT) layers to Llama-Omni for unlimited context long-form speech generation

---

## Executive Summary

This document provides a **complete, code-level implementation plan** for integrating TTT into Llama-Omni based on:
- ✅ Deep analysis of both codebases (llama-omni, ttt-video-dit, ttt-lm-jax)
- ✅ Understanding of critical requirements (FP32, state persistence, mini-batches)
- ✅ Review of documentation (docs 00-09)
- ✅ Minimal code changes philosophy (copy existing, adapt minimally)

**Key Strategy**: Copy TTT implementation from `ttt-video-dit` (PyTorch), strip video-specific code, adapt for speech/text sequences.

---

## 1. Repository Structure Overview

### 1.1 Current Llama-Omni Structure

```
llama-omni/
├── omni_speech/
│   ├── model/
│   │   ├── language_model/
│   │   │   ├── omni_speech_llama.py      ← Main model (wraps LlamaForCausalLM)
│   │   ├── omni_speech_arch.py           ← Speech encoder integration
│   │   ├── builder.py                     ← Model loading
│   │   ├── speech_encoder/                ← Whisper encoder
│   │   ├── speech_projector/              ← Speech->LLM projection
│   │   └── speech_generator/              ← Speech synthesis (CTC)
│   ├── infer/                              ← Inference only (no training code!)
│   ├── serve/                              ← Gradio server
│   └── datasets/                           ← Dataset preprocessing
└── predict.py                              ← Inference entry point
```

**Key Finding**: ❌ **No training code in public repo** - need to create from scratch or use HuggingFace Trainer.

### 1.2 TTT Implementation Sources

```
ttt-video-dit/ttt/models/ssm/              ← PRIMARY SOURCE (PyTorch)
├── ttt_layer.py              (473 lines)  ← Base TTT layer (COPY & ADAPT)
├── ops/
│   ├── ttt_linear.py         (85 lines)   ← TTT-Linear core logic (COPY)
│   ├── ttt_mlp.py            (99 lines)   ← TTT-MLP core logic (COPY)
│   └── utils.py                           ← LayerNorm, GELU utils (COPY)
└── utils.py                               ← scan, RoPE (PARTIAL COPY)

ttt-lm-jax/ttt/models/
└── ttt_layer.py              (675 lines)  ← JAX reference (don't copy, use for understanding)
```

---

## 2. Critical Requirements (from Docs Analysis)

### 2.1 FP32 Precision (MANDATORY)

From `docs/05_Critical_Issues_Analysis.md`:

```python
# TTT inner states MUST be FP32
self.W1 = nn.Parameter(
    torch.zeros(num_heads, head_dim, head_dim,
                dtype=torch.float32)  # ← FP32!
)
self.b1 = nn.Parameter(
    torch.zeros(num_heads, 1, head_dim,
                dtype=torch.float32)  # ← FP32!
)

# Verify during forward pass
assert self.W1.dtype == torch.float32, f"W1 must be FP32, got {self.W1.dtype}"
```

**Why**: Numerical stability during gradient descent on W in inner loop.

### 2.2 State Persistence

From `docs/06_Revised_Integration_Plan.md`:

```python
# State persists across batches in SAME conversation
@dataclass
class TTTConversationState:
    W1: torch.Tensor  # [B, num_heads, head_dim, head_dim] - FP32!
    b1: torch.Tensor  # [B, num_heads, 1, head_dim] - FP32!
    conversation_id: str
    turn_number: int
```

**Critical**: Do NOT reset state mid-conversation. Only reset when starting new conversation.

### 2.3 Mini-Batch Size

From `docs/07_Model_Reconsideration.md`:

- **Mini-batch size**: 64 tokens
- **Why**: TTT needs mini-batches for stable gradient computation
- **Llama-Omni compatibility**: ✅ Whisper processes full utterance (not streaming), so naturally compatible

### 2.4 Layer Selection

From `docs/09_Final_Verdict.md`:

- **Llama 3.1 architecture**: 32 layers (0-31)
- **Replace layers 24-31**: Top 8 layers with TTT
- **Keep layers 0-23**: Standard attention (for low-level features)

---

## 3. Implementation Plan

### 3.1 Phase 1: Create TTT Module (Week 1)

**Goal**: Create standalone TTT module adapted from ttt-video-dit

#### File: `omni_speech/model/ttt/ttt_layer.py`

**What to copy**:
```python
# FROM: ttt-video-dit/ttt/models/ssm/ttt_layer.py

class TTTBase(nn.Module):
    # Copy these methods:
    ✅ __init__()                    # Basic setup
    ✅ _init_qkvo_proj()             # Q, K, V, O projections
    ✅ _init_ttt_lr_gate()           # Learnable learning rate
    ✅ _init_ttt_ln()                # LayerNorm for reconstruction
    ✅ get_qkv_projections()         # Process input
    ✅ get_eta()                     # Compute learning rate
    ✅ reshape_to_mini_batch()       # Reshape to [B, num_mb, mb_size, D]

    # REMOVE (video-specific):
    ❌ interleave()                  # Scene interleaving
    ❌ undo_interleave()             # Scene de-interleaving
    ❌ init_device_mesh()            # Tensor parallel for video
    ❌ shard_inputs()                # Distributed video processing
```

**What to adapt**:
```python
# BEFORE (video-dit):
def forward(self, hidden_states, freqs_cis, seq_metadata):
    # freqs_cis: 3D RoPE for video
    # seq_metadata: scene boundaries, text lengths, etc.
    ...

# AFTER (speech):
def forward(self, hidden_states, position_ids=None):
    # position_ids: standard 1D positions
    # No scene metadata needed!
    ...
```

#### File: `omni_speech/model/ttt/ops/ttt_mlp.py`

**What to copy** (ENTIRE FILE - minimal changes):
```python
# FROM: ttt-video-dit/ttt/models/ssm/ops/ttt_mlp.py

# Copy EXACTLY (99 lines):
✅ compute_mini_batch()     # Core TTT-MLP computation
✅ ttt_mlp()                # Scan over mini-batches

# Only change imports:
# BEFORE: from ttt.models.ssm.utils import scan
# AFTER:  from omni_speech.model.ttt.utils import scan
```

#### File: `omni_speech/model/ttt/ops/ttt_linear.py`

**What to copy** (ENTIRE FILE - minimal changes):
```python
# FROM: ttt-video-dit/ttt/models/ssm/ops/ttt_linear.py (85 lines)

✅ compute_mini_batch()     # Core TTT-Linear computation
✅ ttt_linear()             # Scan over mini-batches

# Only change imports
```

#### File: `omni_speech/model/ttt/ops/utils.py`

**What to copy**:
```python
# FROM: ttt-video-dit/ttt/models/ssm/ops/utils.py

✅ ln_fwd()                 # LayerNorm forward
✅ ln_fused_l2_bwd()        # LayerNorm + L2 loss backward
✅ gelu_bwd()               # GELU backward for TTT-MLP
```

#### File: `omni_speech/model/ttt/utils.py`

**What to copy**:
```python
# FROM: ttt-video-dit/ttt/models/ssm/utils.py

✅ scan()                   # Iterate over mini-batches with checkpointing

# REMOVE (use transformers built-in):
❌ precompute_freqs_cis_3d() # 3D RoPE (video-specific)
❌ apply_rotary_emb()        # Use transformers version instead
```

### 3.2 Phase 2: Integrate into Llama Model (Week 1-2)

#### File: `omni_speech/model/language_model/omni_speech_llama_ttt.py`

**Strategy**: Copy `omni_speech_llama.py` → add TTT modifications

```python
# NEW FILE (copy omni_speech_llama.py and modify)

from transformers import LlamaForCausalLM, LlamaModel, LlamaConfig
from omni_speech.model.ttt.ttt_layer import TTTMLP
from omni_speech.model.omni_speech_arch import OmniSpeechMetaModel, OmniSpeechMetaForCausalLM

class OmniSpeechTTTConfig(LlamaConfig):
    model_type = "omni_speech_llama_ttt"

    def __init__(
        self,
        # Llama config params...
        **kwargs
    ):
        super().__init__(**kwargs)

        # TTT-specific config
        self.num_ttt_layers = 8              # Top 8 layers
        self.ttt_start_layer = 24            # Layers 24-31
        self.mini_batch_size = 64            # TTT mini-batch size
        self.ttt_base_lr = 1.0               # Base learning rate for inner loop
        self.use_ttt_mlp = True              # TTT-MLP vs TTT-Linear

        # CRITICAL: FP32 for TTT states
        self.ttt_param_dtype = torch.float32


class OmniSpeechLlamaModelTTT(OmniSpeechMetaModel, LlamaModel):
    """Llama model with TTT in top 8 layers"""

    config_class = OmniSpeechTTTConfig

    def __init__(self, config: OmniSpeechTTTConfig):
        super().__init__(config)

        # Replace attention in layers 24-31 with TTT
        self._replace_attention_with_ttt()

        # State management for conversations
        self.conversation_states = {}  # {conv_id: TTTConversationState}

    def _replace_attention_with_ttt(self):
        """Replace self_attn in layers 24-31 with TTT"""

        for layer_idx in range(
            self.config.ttt_start_layer,
            self.config.ttt_start_layer + self.config.num_ttt_layers
        ):
            # Get original layer
            original_layer = self.layers[layer_idx]

            # Create TTT layer
            ttt_layer = TTTMLP(
                model_dim=self.config.hidden_size,
                num_heads=self.config.num_attention_heads,
                mini_batch_size=self.config.mini_batch_size,
                ttt_base_lr=self.config.ttt_base_lr,
                # CRITICAL: FP32 dtype
                param_dtype=torch.float32
            )

            # Replace self_attn with TTT
            # Keep everything else (mlp, layer_norm, etc.)
            original_layer.self_attn = ttt_layer

            print(f"✅ Replaced layer {layer_idx} attention with TTT")

            # Verify FP32
            assert ttt_layer.W1.dtype == torch.float32
            assert ttt_layer.b1.dtype == torch.float32
```

**Key Design Decision**:

Instead of creating a new layer type, we **replace `self_attn`** in existing `LlamaDecoderLayer`:

```python
# LlamaDecoderLayer structure (from transformers):
class LlamaDecoderLayer(nn.Module):
    def __init__(self, config):
        self.self_attn = LlamaSdpaAttention(config)  # ← Replace this
        self.mlp = LlamaMLP(config)                  # ← Keep this
        self.input_layernorm = LlamaRMSNorm(...)     # ← Keep this
        self.post_attention_layernorm = LlamaRMSNorm(...)  # ← Keep this

    def forward(self, hidden_states, attention_mask=None, ...):
        # Layer norm
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Attention (or TTT!)
        hidden_states, _, _ = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            ...
        )
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states
```

**TTT Layer Interface** (must match attention interface):

```python
class TTTMLP(nn.Module):
    """TTT layer that acts as drop-in replacement for attention"""

    def forward(
        self,
        hidden_states: torch.Tensor,           # [B, L, D]
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Returns:
            tuple: (hidden_states, None, None)
            - hidden_states: [B, L, D]
            - None: no attention weights (TTT doesn't have them)
            - None: no past_key_value (TTT uses internal state)
        """

        # Check sequence length is multiple of mini_batch_size
        B, L, D = hidden_states.shape
        assert L % self.mini_batch_size == 0, \
            f"Seq len {L} must be multiple of mini_batch_size {self.mini_batch_size}"

        # Process with TTT
        hidden_states = self._forward_ttt(hidden_states, position_ids)

        return (hidden_states, None, None)

    def _forward_ttt(self, hidden_states, position_ids):
        """Core TTT processing"""

        # 1. Get Q, K, V projections
        XQ, XK, XV = self.get_qkv_projections(hidden_states)

        # 2. Apply RoPE (use transformers built-in)
        XQ, XK = self._apply_rope(XQ, XK, position_ids)

        # 3. Reshape to mini-batches
        XQ, XK, XV = self.reshape_to_mini_batch(XQ, XK, XV)

        # 4. Get learning rate
        eta = self.get_eta(hidden_states)

        # 5. Run TTT
        from omni_speech.model.ttt.ops.ttt_mlp import ttt_mlp

        XQW = ttt_mlp(
            XK, XQ, XV, eta,
            self.ttt_norm_weight, self.ttt_norm_bias,
            self.W1, self.b1, self.W2, self.b2,
            checkpoint_group_size=self.config.scan_checkpoint_group_size
        )

        # 6. Post-process
        XQW = XQW.reshape(B, L, self.num_heads, self.head_dim)
        XQW = self.post_norm(XQW)
        output = self.wo(XQW.reshape(B, L, -1))

        return output
```

### 3.3 Phase 3: State Management (Week 2)

#### File: `omni_speech/model/ttt/state_manager.py`

**NEW FILE** - for conversation state persistence:

```python
from dataclasses import dataclass
from typing import Dict, Optional
import torch

@dataclass
class TTTConversationState:
    """Persistent state for one conversation"""

    # TTT parameters (FP32!)
    W1: torch.Tensor  # [B, num_heads, head_dim, head_dim]
    b1: torch.Tensor  # [B, num_heads, 1, head_dim]
    W2: Optional[torch.Tensor] = None  # For TTT-MLP
    b2: Optional[torch.Tensor] = None  # For TTT-MLP

    # Metadata
    conversation_id: str = ""
    turn_number: int = 0
    total_tokens_processed: int = 0

    def to_dict(self):
        """Serialize for checkpointing"""
        return {
            'W1': self.W1.cpu(),
            'b1': self.b1.cpu(),
            'W2': self.W2.cpu() if self.W2 is not None else None,
            'b2': self.b2.cpu() if self.b2 is not None else None,
            'conversation_id': self.conversation_id,
            'turn_number': self.turn_number,
            'total_tokens_processed': self.total_tokens_processed,
        }

    @classmethod
    def from_dict(cls, state_dict, device='cuda'):
        """Deserialize from checkpoint"""
        return cls(
            W1=state_dict['W1'].to(device),
            b1=state_dict['b1'].to(device),
            W2=state_dict['W2'].to(device) if state_dict['W2'] is not None else None,
            b2=state_dict['b2'].to(device) if state_dict['b2'] is not None else None,
            conversation_id=state_dict['conversation_id'],
            turn_number=state_dict['turn_number'],
            total_tokens_processed=state_dict['total_tokens_processed'],
        )


class TTTStateManager:
    """Manages TTT states across conversations"""

    def __init__(self):
        self.states: Dict[str, TTTConversationState] = {}

    def get_or_create_state(
        self,
        conversation_id: str,
        num_heads: int,
        head_dim: int,
        device: torch.device,
        use_mlp: bool = True
    ) -> TTTConversationState:
        """Get existing state or create new one"""

        if conversation_id not in self.states:
            # Create initial state (learned parameters from model)
            # These will be set by the model during initialization
            state = TTTConversationState(
                W1=torch.zeros(1, num_heads, head_dim, head_dim,
                              dtype=torch.float32, device=device),
                b1=torch.zeros(1, num_heads, 1, head_dim,
                              dtype=torch.float32, device=device),
                conversation_id=conversation_id,
            )

            if use_mlp:
                # TTT-MLP has 2 layers
                hidden_dim = 4 * head_dim  # Standard 4x expansion
                state.W2 = torch.zeros(1, num_heads, hidden_dim, head_dim,
                                      dtype=torch.float32, device=device)
                state.b2 = torch.zeros(1, num_heads, 1, head_dim,
                                      dtype=torch.float32, device=device)

            self.states[conversation_id] = state

        return self.states[conversation_id]

    def update_state(
        self,
        conversation_id: str,
        W1_new: torch.Tensor,
        b1_new: torch.Tensor,
        W2_new: Optional[torch.Tensor] = None,
        b2_new: Optional[torch.Tensor] = None,
        tokens_processed: int = 0
    ):
        """Update state after processing a batch"""

        if conversation_id in self.states:
            state = self.states[conversation_id]
            state.W1 = W1_new.detach()  # Detach to prevent gradient accumulation
            state.b1 = b1_new.detach()

            if W2_new is not None:
                state.W2 = W2_new.detach()
            if b2_new is not None:
                state.b2 = b2_new.detach()

            state.turn_number += 1
            state.total_tokens_processed += tokens_processed

            # Verify FP32
            assert state.W1.dtype == torch.float32
            assert state.b1.dtype == torch.float32

    def reset_conversation(self, conversation_id: str):
        """Reset state for new conversation"""
        if conversation_id in self.states:
            del self.states[conversation_id]

    def save_checkpoint(self, path: str):
        """Save all conversation states"""
        checkpoint = {
            conv_id: state.to_dict()
            for conv_id, state in self.states.items()
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str, device='cuda'):
        """Load conversation states from checkpoint"""
        checkpoint = torch.load(path, map_location=device)
        self.states = {
            conv_id: TTTConversationState.from_dict(state_dict, device)
            for conv_id, state_dict in checkpoint.items()
        }
```

### 3.4 Phase 4: Training Infrastructure (Week 2-3)

Since llama-omni doesn't provide training code, we need to create it:

#### File: `train_ttt.py`

**NEW FILE** - HuggingFace Trainer-based training:

```python
import torch
from transformers import Trainer, TrainingArguments
from omni_speech.model.builder import load_pretrained_model
from omni_speech.model.language_model.omni_speech_llama_ttt import (
    OmniSpeechLlamaForCausalLMTTT,
    OmniSpeechTTTConfig
)
import argparse

def create_ttt_model_from_pretrained(base_model_path: str):
    """
    Load pretrained Llama-Omni and add TTT layers

    Strategy:
    1. Load pretrained OmniSpeechLlamaForCausalLM
    2. Copy weights to OmniSpeechLlamaForCausalLMTTT
    3. Initialize TTT layers (W1, b1, etc.)
    4. Freeze non-TTT parameters initially
    """

    # Load base model
    tokenizer, base_model, context_len = load_pretrained_model(
        model_path=base_model_path,
        model_base=None,
        s2s=False
    )

    # Create TTT config (copy from base + add TTT params)
    ttt_config = OmniSpeechTTTConfig.from_pretrained(base_model_path)
    ttt_config.num_ttt_layers = 8
    ttt_config.ttt_start_layer = 24
    ttt_config.mini_batch_size = 64
    ttt_config.ttt_base_lr = 1.0

    # Create new model with TTT
    ttt_model = OmniSpeechLlamaForCausalLMTTT(ttt_config)

    # Copy weights from base model
    # (all layers except self_attn in layers 24-31)
    with torch.no_grad():
        # Copy embeddings
        ttt_model.model.embed_tokens.load_state_dict(
            base_model.model.embed_tokens.state_dict()
        )

        # Copy all layers
        for i in range(32):
            if i < 24 or i >= 32:  # Layers not being replaced
                ttt_model.model.layers[i].load_state_dict(
                    base_model.model.layers[i].state_dict()
                )
            else:  # Layers 24-31 (with TTT)
                # Copy everything except self_attn
                ttt_model.model.layers[i].mlp.load_state_dict(
                    base_model.model.layers[i].mlp.state_dict()
                )
                ttt_model.model.layers[i].input_layernorm.load_state_dict(
                    base_model.model.layers[i].input_layernorm.state_dict()
                )
                ttt_model.model.layers[i].post_attention_layernorm.load_state_dict(
                    base_model.model.layers[i].post_attention_layernorm.state_dict()
                )

                # TTT layer (self_attn) is randomly initialized
                # Will be trained from scratch

        # Copy final layer norm and lm_head
        ttt_model.model.norm.load_state_dict(base_model.model.norm.state_dict())
        ttt_model.lm_head.load_state_dict(base_model.lm_head.state_dict())

        # Copy speech components
        ttt_model.model.speech_encoder.load_state_dict(
            base_model.model.speech_encoder.state_dict()
        )
        ttt_model.model.speech_projector.load_state_dict(
            base_model.model.speech_projector.state_dict()
        )

    return ttt_model, tokenizer


def get_trainable_params(model, train_mode='ttt_only'):
    """
    Get trainable parameters based on training mode

    Modes:
    - ttt_only: Only TTT layers (layers 24-31 self_attn)
    - ttt_and_top: TTT + all parameters in layers 24-31
    - full: All parameters
    """

    for name, param in model.named_parameters():
        param.requires_grad = False  # Freeze all first

    if train_mode == 'ttt_only':
        # Only TTT parameters in layers 24-31
        for i in range(24, 32):
            for name, param in model.model.layers[i].self_attn.named_parameters():
                param.requires_grad = True

    elif train_mode == 'ttt_and_top':
        # All parameters in layers 24-31
        for i in range(24, 32):
            for param in model.model.layers[i].parameters():
                param.requires_grad = True

    elif train_mode == 'full':
        # All parameters
        for param in model.parameters():
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./checkpoints/ttt')
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--train_mode', type=str, default='ttt_only',
                       choices=['ttt_only', 'ttt_and_top', 'full'])
    args = parser.parse_args()

    # Create model
    model, tokenizer = create_ttt_model_from_pretrained(args.base_model_path)
    model = get_trainable_params(model, args.train_mode)

    # Load dataset
    # TODO: Implement dataset loading
    # Should return long conversations (hours of dialogue)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        fp16=True,  # Mixed precision for non-TTT parameters
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        gradient_checkpointing=True,
        # CRITICAL: Ensure FP32 for TTT states
        fp16_full_eval=False,
    )

    # Custom Trainer to handle TTT state management
    trainer = TTTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()


class TTTTrainer(Trainer):
    """Custom trainer with TTT state management"""

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Override to handle conversation state persistence

        Key: Don't reset TTT state mid-conversation!
        """

        # Check if this is same conversation as previous batch
        conversation_ids = inputs.get('conversation_id', None)

        if conversation_ids is not None:
            # Ensure state persists for same conversation
            # Reset state for new conversations
            for conv_id in conversation_ids:
                if self._is_new_conversation(conv_id):
                    model.reset_conversation_state(conv_id)

        # Standard forward pass
        outputs = model(**inputs)
        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss

    def _is_new_conversation(self, conversation_id):
        # Track conversation IDs across batches
        # Reset state when conversation changes
        # Implementation depends on dataset format
        pass


if __name__ == '__main__':
    main()
```

### 3.5 Phase 5: Testing & Validation (Week 3-4)

#### File: `tests/test_ttt_integration.py`

**NEW FILE** - comprehensive tests:

```python
import torch
import pytest
from omni_speech.model.language_model.omni_speech_llama_ttt import (
    OmniSpeechLlamaForCausalLMTTT,
    OmniSpeechTTTConfig
)

def test_fp32_precision():
    """CRITICAL: Verify FP32 precision for TTT states"""

    config = OmniSpeechTTTConfig()
    config.num_ttt_layers = 8
    config.ttt_start_layer = 24

    model = OmniSpeechLlamaForCausalLMTTT(config)

    # Check layers 24-31 have FP32 TTT states
    for i in range(24, 32):
        ttt_layer = model.model.layers[i].self_attn

        assert ttt_layer.W1.dtype == torch.float32, \
            f"Layer {i}: W1 must be FP32, got {ttt_layer.W1.dtype}"
        assert ttt_layer.b1.dtype == torch.float32, \
            f"Layer {i}: b1 must be FP32, got {ttt_layer.b1.dtype}"

        if hasattr(ttt_layer, 'W2'):
            assert ttt_layer.W2.dtype == torch.float32
            assert ttt_layer.b2.dtype == torch.float32


def test_state_persistence():
    """Verify state persists across forward passes"""

    config = OmniSpeechTTTConfig()
    model = OmniSpeechLlamaForCausalLMTTT(config)

    # First forward pass
    input_ids = torch.randint(0, 1000, (1, 128))  # 128 tokens (2 mini-batches)
    outputs1 = model(input_ids=input_ids)

    # Get state after first pass
    conv_id = "test_conv_1"
    state1 = model.get_conversation_state(conv_id)

    # Second forward pass (same conversation)
    outputs2 = model(input_ids=input_ids, conversation_id=conv_id)
    state2 = model.get_conversation_state(conv_id)

    # State should have changed (updated by TTT)
    assert not torch.allclose(state1.W1, state2.W1), \
        "State should update between forward passes"

    # State should persist (not reset)
    assert state2.turn_number == 2, \
        "Turn number should increment"


def test_mini_batch_size():
    """Verify sequence length must be multiple of mini_batch_size"""

    config = OmniSpeechTTTConfig()
    config.mini_batch_size = 64
    model = OmniSpeechLlamaForCausalLMTTT(config)

    # Valid: 128 tokens (2 mini-batches)
    input_ids = torch.randint(0, 1000, (1, 128))
    outputs = model(input_ids=input_ids)  # Should work

    # Invalid: 100 tokens (not multiple of 64)
    input_ids = torch.randint(0, 1000, (1, 100))
    with pytest.raises(AssertionError):
        outputs = model(input_ids=input_ids)  # Should fail


def test_long_context():
    """Test with long sequences (32k tokens)"""

    config = OmniSpeechTTTConfig()
    config.mini_batch_size = 64
    model = OmniSpeechLlamaForCausalLMTTT(config).cuda()

    # 32k tokens = 500 mini-batches
    input_ids = torch.randint(0, 1000, (1, 32768)).cuda()

    with torch.no_grad():
        outputs = model(input_ids=input_ids)

    # Verify no OOM, no quality degradation
    assert outputs.logits.shape == (1, 32768, model.config.vocab_size)

    # Check for NaN (indicates numerical instability)
    assert not torch.isnan(outputs.logits).any(), \
        "Found NaN in outputs - FP32 precision issue?"


def test_generation():
    """Test text generation with TTT"""

    from transformers import AutoTokenizer

    config = OmniSpeechTTTConfig()
    model = OmniSpeechLlamaForCausalLMTTT(config).cuda()
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

    # Generate text
    prompt = "Hello, how are you today?"
    input_ids = tokenizer.encode(prompt, return_tensors='pt').cuda()

    # Pad to mini_batch_size multiple
    # (Important: generation must respect mini-batch boundaries)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
        )

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"Generated: {generated_text}")

    assert len(generated_text) > len(prompt), \
        "Should generate new text"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```

---

## 4. File Copying Matrix

| Source File | Destination | Copy % | Changes Needed |
|-------------|-------------|--------|----------------|
| `ttt-video-dit/ttt/models/ssm/ttt_layer.py` | `omni_speech/model/ttt/ttt_layer.py` | 60% | Remove video-specific: interleave, undo_interleave, SequenceMetadata, 3D RoPE. Adapt forward() signature. |
| `ttt-video-dit/ttt/models/ssm/ops/ttt_mlp.py` | `omni_speech/model/ttt/ops/ttt_mlp.py` | 100% | Only change imports |
| `ttt-video-dit/ttt/models/ssm/ops/ttt_linear.py` | `omni_speech/model/ttt/ops/ttt_linear.py` | 100% | Only change imports |
| `ttt-video-dit/ttt/models/ssm/ops/utils.py` | `omni_speech/model/ttt/ops/utils.py` | 100% | Copy ln_fwd, ln_fused_l2_bwd, gelu_bwd |
| `ttt-video-dit/ttt/models/ssm/utils.py` | `omni_speech/model/ttt/utils.py` | 30% | Copy scan() only. Use transformers RoPE instead of 3D version. |

**Total lines to copy**: ~400-500 lines
**Total new code**: ~800-1000 lines (training, state management, tests)

---

## 5. Critical Implementation Notes

### 5.1 FP32 Enforcement Checklist

```python
# 1. Parameter initialization
self.W1 = nn.Parameter(torch.zeros(..., dtype=torch.float32))  # ✓
self.b1 = nn.Parameter(torch.zeros(..., dtype=torch.float32))  # ✓

# 2. Forward pass verification
def forward(self, hidden_states):
    assert self.W1.dtype == torch.float32  # ✓
    assert self.b1.dtype == torch.float32  # ✓

    # Convert inputs to FP32 for TTT computation
    hidden_states_fp32 = hidden_states.float()

    # ... TTT computation in FP32 ...

    # Convert output back to model dtype
    return output.to(hidden_states.dtype)

# 3. Optimizer setup
# Use separate parameter groups
optimizer = torch.optim.AdamW([
    {
        'params': [p for n, p in model.named_parameters()
                   if 'W1' in n or 'b1' in n or 'W2' in n or 'b2' in n],
        'lr': 1e-4,
        # NO fp16/bf16 conversion for these params
    },
    {
        'params': [p for n, p in model.named_parameters()
                   if not ('W1' in n or 'b1' in n or 'W2' in n or 'b2' in n)],
        'lr': 1e-5,
    }
])
```

### 5.2 State Persistence Strategy

```python
# Training loop pseudo-code

for epoch in range(num_epochs):
    for batch in dataloader:
        conversation_id = batch['conversation_id']
        turn_number = batch['turn_number']

        # CRITICAL: Only reset state for new conversations
        if turn_number == 0:  # First turn of conversation
            model.reset_conversation_state(conversation_id)

        # Forward pass (state persists automatically)
        outputs = model(
            input_ids=batch['input_ids'],
            conversation_id=conversation_id
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()

        # State is automatically updated inside model
        # W1, b1 are updated by TTT inner loop
```

### 5.3 Mini-Batch Handling

```python
# Ensure all sequences are multiples of mini_batch_size (64)

def collate_fn(batch):
    """Custom collator to pad to mini_batch_size"""

    mini_batch_size = 64

    for item in batch:
        seq_len = len(item['input_ids'])

        # Pad to nearest multiple of 64
        if seq_len % mini_batch_size != 0:
            pad_len = mini_batch_size - (seq_len % mini_batch_size)
            item['input_ids'] = torch.cat([
                item['input_ids'],
                torch.full((pad_len,), tokenizer.pad_token_id)
            ])
            item['attention_mask'] = torch.cat([
                item['attention_mask'],
                torch.zeros(pad_len)
            ])

    return default_collate(batch)
```

---

## 6. Implementation Timeline

### Week 1: Core TTT Module
- [ ] Day 1-2: Copy and adapt `ttt_layer.py`
- [ ] Day 3: Copy TTT ops (`ttt_mlp.py`, `ttt_linear.py`, `utils.py`)
- [ ] Day 4: Create `state_manager.py`
- [ ] Day 5: Write unit tests for TTT module

### Week 2: Integration
- [ ] Day 1-2: Create `omni_speech_llama_ttt.py`
- [ ] Day 3: Implement layer replacement logic
- [ ] Day 4: Test with small model (verify FP32, interface compatibility)
- [ ] Day 5: Load pretrained Llama-Omni weights

### Week 3: Training Infrastructure
- [ ] Day 1-2: Create `train_ttt.py` and `TTTTrainer`
- [ ] Day 3: Implement dataset loading (long conversations)
- [ ] Day 4-5: Run small-scale training test (1 GPU, small dataset)

### Week 4: Validation & Scale-Up
- [ ] Day 1: Run comprehensive tests (`test_ttt_integration.py`)
- [ ] Day 2-3: Full training run (multi-GPU)
- [ ] Day 4: Evaluate on long-form generation
- [ ] Day 5: Create inference demo

---

## 7. Risk Mitigation

### Risk 1: FP32 Precision Lost During Training

**Mitigation**:
```python
# Add hooks to verify FP32 maintained
def check_fp32_hook(module, input, output):
    if hasattr(module, 'W1'):
        assert module.W1.dtype == torch.float32, \
            f"FP32 lost! W1 dtype: {module.W1.dtype}"

for i in range(24, 32):
    model.model.layers[i].self_attn.register_forward_hook(check_fp32_hook)
```

### Risk 2: State Reset Mid-Conversation

**Mitigation**:
```python
# Add logging to track state resets
class TTTStateManager:
    def reset_conversation(self, conv_id):
        print(f"⚠️ RESETTING state for {conv_id}")
        # Only call this at conversation boundaries!
```

### Risk 3: Sequence Length Not Multiple of 64

**Mitigation**:
```python
# Pad in data collator (shown above)
# Add assertion in model forward pass:
assert L % self.mini_batch_size == 0, \
    f"Seq len {L} must be multiple of {self.mini_batch_size}"
```

### Risk 4: Memory OOM with Long Sequences

**Mitigation**:
```python
# Use gradient checkpointing for scan
# (already implemented in ttt_mlp.py via checkpoint_group_size)

# Also enable gradient checkpointing for Llama layers
model.gradient_checkpointing_enable()
```

---

## 8. Success Criteria

### Phase 1 Success (TTT Module):
- ✅ All tests pass
- ✅ FP32 precision verified
- ✅ Compatible with Llama attention interface

### Phase 2 Success (Integration):
- ✅ Model loads without errors
- ✅ Forward pass works with dummy data
- ✅ Pretrained weights transfer correctly
- ✅ Layers 24-31 use TTT, layers 0-23 use attention

### Phase 3 Success (Training):
- ✅ Training loop runs without crashes
- ✅ Loss decreases over time
- ✅ State persists across batches
- ✅ No NaN in gradients/activations

### Phase 4 Success (Validation):
- ✅ Generates coherent text for 10k+ tokens
- ✅ No quality degradation at long context
- ✅ Better than baseline Llama-Omni on long-form tasks
- ✅ Memory usage scales linearly (not quadratically)

---

## 9. Next Steps

1. **NOW**: Review this plan
2. **Week 1**: Start Phase 1 (copy TTT module files)
3. **Week 2**: Phase 2 (integration)
4. **Week 3**: Phase 3 (training infrastructure)
5. **Week 4**: Phase 4 (validation & experiments)

---

## 10. Open Questions

1. **Dataset**: What long-form speech conversations to use for training?
   - Option A: Generate synthetic with TTS + text dialogues
   - Option B: Use LibriSpeech/MLS long audiobooks
   - Option C: Podcast transcripts with speech synthesis

2. **Evaluation**: How to measure success?
   - Perplexity at different sequence lengths
   - Human evaluation of long-form generation
   - Coherence metrics (entity consistency, topic consistency)

3. **Hyperparameters**:
   - Mini-batch size: 64 (from docs) - verify optimal
   - TTT learning rate: 1.0 (from video-dit) - may need tuning
   - Number of TTT layers: 8 (top layers) - test 4 vs 8 vs 16

---

## Appendix A: Code Snippets Reference

### A.1 Minimal TTT-MLP Forward Pass

```python
def ttt_mlp_forward(XK, XQ, XV, eta, W1, b1, W2, b2, ttt_norm_params):
    """
    Minimal TTT-MLP implementation (single mini-batch)

    Args:
        XK, XQ, XV: [B, num_heads, mini_batch_size, head_dim]
        eta: [B, num_heads, mini_batch_size, 1]  # Learning rate
        W1: [B, num_heads, head_dim, hidden_dim]  # FP32!
        b1: [B, num_heads, 1, hidden_dim]         # FP32!
        W2: [B, num_heads, hidden_dim, head_dim]  # FP32!
        b2: [B, num_heads, 1, head_dim]           # FP32!

    Returns:
        output: [B, num_heads, mini_batch_size, head_dim]
        W1_updated, b1_updated, W2_updated, b2_updated  # Updated states
    """

    # Forward pass (reconstruction)
    Z1 = XK @ W1 + b1                     # [B, H, K, hidden_dim]
    X2 = F.gelu(Z1)                       # [B, H, K, hidden_dim]
    Z2 = X2 @ W2 + b2                     # [B, H, K, head_dim]

    # Reconstruction target
    target = XV - XK                      # [B, H, K, head_dim]

    # Layer norm + L2 loss backward
    grad_Z2 = ln_fused_l2_bwd(Z2, target, ttt_norm_weight, ttt_norm_bias)

    # Backprop through MLP
    grad_Z1 = grad_Z2 @ W2.transpose(-2, -1) * gelu_bwd(Z1)

    # Update parameters (gradient descent)
    last_eta = eta[:, :, -1:, :]  # Last token in mini-batch
    W1_updated = W1 - (last_eta * XK[:, :, -1:, :]).transpose(-2, -1) @ grad_Z1[:, :, -1:, :]
    b1_updated = b1 - (last_eta * grad_Z1).sum(dim=-2, keepdim=True)
    W2_updated = W2 - (last_eta * X2[:, :, -1:, :]).transpose(-2, -1) @ grad_Z2[:, :, -1:, :]
    b2_updated = b2 - (last_eta * grad_Z2).sum(dim=-2, keepdim=True)

    # Output (test-time prediction)
    Attn1 = XQ @ XK.transpose(-2, -1)
    Z1_bar = XQ @ W1 - (eta * Attn1) @ grad_Z1 + b1_updated
    X2_bar = F.gelu(Z1_bar)

    Attn2 = X2_bar @ X2.transpose(-2, -1)
    Z2_bar = X2_bar @ W2 - (eta * Attn2) @ grad_Z2 + b2_updated

    # Apply layer norm
    Z2_bar = ln_fwd(Z2_bar, ttt_norm_weight, ttt_norm_bias)

    output = XQ + Z2_bar  # Residual connection

    return output, W1_updated, b1_updated, W2_updated, b2_updated
```

### A.2 Scan Over Mini-Batches

```python
def scan(f, init_state, inputs, checkpoint_group_size=1):
    """
    Apply function f over sequence of mini-batches

    Args:
        f: Function (state, input) -> (new_state, output)
        init_state: Initial state (e.g., W1, b1, W2, b2)
        inputs: Dict of tensors [num_mini_batches, B, H, K, D]
        checkpoint_group_size: Checkpoint every N iterations

    Returns:
        final_state, outputs
    """

    state = init_state
    outputs = []

    num_mini_batches = inputs[next(iter(inputs))].shape[0]

    for i in range(num_mini_batches):
        # Get mini-batch
        mini_batch = {k: v[i] for k, v in inputs.items()}

        # Apply function (with optional checkpointing)
        if i % checkpoint_group_size == 0:
            state, output = torch.utils.checkpoint.checkpoint(
                f, state, mini_batch, use_reentrant=False
            )
        else:
            state, output = f(state, mini_batch)

        outputs.append(output)

    outputs = torch.stack(outputs, dim=0)  # [num_mb, B, H, K, D]

    return state, outputs
```

---

## Appendix B: Configuration Examples

### B.1 Training Config (Stage 1: 8k context)

```yaml
# config/stage1_8k.yaml

model:
  base_model_path: "ICTNLP/Llama-3.1-8B-Omni"
  num_ttt_layers: 8
  ttt_start_layer: 24
  mini_batch_size: 64
  ttt_base_lr: 1.0
  use_ttt_mlp: true

training:
  num_epochs: 3
  batch_size: 2                    # Effective: 2 * 8 GPUs = 16
  gradient_accumulation_steps: 4   # Effective batch: 64
  learning_rate: 1e-4
  max_sequence_length: 8192        # 128 mini-batches
  train_mode: "ttt_only"           # Only train TTT params

data:
  dataset_path: "data/long_conversations_8k"
  num_workers: 8

optimizer:
  type: "adamw"
  weight_decay: 0.01
  betas: [0.9, 0.95]

scheduler:
  type: "cosine"
  warmup_steps: 100
```

### B.2 Training Config (Stage 2: 32k context)

```yaml
# config/stage2_32k.yaml

model:
  # Load from stage 1 checkpoint
  checkpoint_path: "checkpoints/stage1/final"

training:
  max_sequence_length: 32768       # 512 mini-batches
  train_mode: "ttt_and_top"        # Train TTT + all layer 24-31 params
  learning_rate: 5e-5              # Lower LR for fine-tuning

data:
  dataset_path: "data/long_conversations_32k"
```

---

**End of Plan**

This plan provides:
- ✅ Complete file-by-file copying strategy
- ✅ Detailed code examples for all key components
- ✅ Risk mitigation strategies
- ✅ Clear success criteria
- ✅ 4-week implementation timeline

Ready to implement! 🚀
