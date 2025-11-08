# Can LLaMA-Factory Help with Llama-Omni TTT Fine-Tuning?

**Date**: 2025-11-08
**Analysis**: Comparing LLaMA-Factory vs Custom Training for TTT Integration
**Verdict**: ⚠️ **Partial Help - Critical Gaps Remain**

---

## Executive Summary

**Can LLaMA-Factory be used?** → **Yes, but with significant limitations**
**Should we use it?** → **No for TTT integration, Maybe for standard fine-tuning**
**Recommendation**: **Hybrid approach** (LLaMA-Factory for infrastructure, custom code for TTT)

---

## 1. What LLaMA-Factory Provides

### 1.1 Supported Features ✅

Based on official documentation and code analysis:

```
✅ 100+ LLM support (Llama, Mistral, Qwen, etc.)
✅ Multi-modal models (Vision: LLaVA, Audio: Qwen2-Audio)
✅ Multiple training methods:
   - Supervised Fine-Tuning (SFT)
   - LoRA, QLoRA, DoRA
   - DPO, KTO, ORPO (preference learning)
   - PPO (reinforcement learning)
✅ Custom model registration (via HuggingFace AutoModel API)
✅ Web UI (LlamaBoard) for no-code training
✅ Efficient training (FlashAttention-2, gradient checkpointing)
✅ Distributed training (DeepSpeed, FSDP)
✅ Experiment tracking (TensorBoard, Wandb)
```

### 1.2 Custom Model Registration ✅

```python
# From misc.py - LLaMA-Factory supports custom models via:

# 1. Register your config
class OmniSpeechTTTConfig(LlamaConfig):
    model_type = "omni_speech_llama_ttt"
    auto_map = {
        "AutoConfig": "configuration_omni_speech_ttt.OmniSpeechTTTConfig",
        "AutoModelForCausalLM": "modeling_omni_speech_ttt.OmniSpeechLlamaForCausalLMTTT"
    }

# 2. Register your model
class OmniSpeechLlamaForCausalLMTTT(LlamaForCausalLM):
    config_class = OmniSpeechTTTConfig

# 3. LLaMA-Factory will auto-load via:
# model = AutoModelForCausalLM.from_pretrained("your-model-path")
```

**Status**: ✅ **We can register Llama-Omni with TTT as a custom model**

---

## 2. TTT-Specific Requirements

### 2.1 What TTT Needs ⚠️

From our training strategy analysis:

```
🔴 CRITICAL REQUIREMENTS:
1. FP32 enforcement for W1, b1, W2, b2
2. State persistence across batches (conversation-level)
3. Custom forward pass (state return format)
4. Mini-batch processing (64 tokens)
5. RoPE position reset per mini-batch
6. Curriculum training (8k → 16k → 32k → 64k)
7. Monitoring TTT state statistics (W1_mean, W1_std, etc.)
8. Runtime auto-padding to 64-multiple
```

### 2.2 LLaMA-Factory Compatibility Analysis

| Requirement | LLaMA-Factory Support | Gap Analysis |
|-------------|----------------------|--------------|
| **FP32 Enforcement** | ❌ Partial | Supports mixed precision but no explicit FP32 param groups |
| **State Persistence** | ❌ No | Standard training resets state per batch |
| **Custom Forward** | ✅ Yes | Can use custom model class |
| **Mini-batch Processing** | ❌ No | Standard batching, no mini-batch logic |
| **RoPE Reset** | ❌ No | Uses standard position encoding |
| **Curriculum Training** | ⚠️ Partial | Can manually run stages, no built-in curriculum |
| **TTT Monitoring** | ❌ No | Standard metrics only (loss, accuracy) |
| **Auto-padding** | ⚠️ Partial | Has collators but not TTT-aware |

---

## 3. Gap Analysis

### 3.1 Critical Gap #1: State Persistence ❌

**What TTT Needs**:
```python
# Conversation-level state management
for batch in dataloader:
    conv_id = batch['conversation_id']
    turn = batch['turn_number']

    # DON'T reset state mid-conversation!
    if turn == 0:
        model.reset_conversation_state(conv_id)

    # State persists automatically
    outputs = model(batch, use_cache=True, past_key_values=cache)
```

**What LLaMA-Factory Does**:
```python
# Standard training loop (from LLaMA-Factory internals)
for batch in dataloader:
    # Each batch is independent
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
    optimizer.step()

    # ❌ No state persistence across batches!
```

**Problem**: LLaMA-Factory treats each batch independently. TTT requires state to persist across batches within the same conversation.

---

### 3.2 Critical Gap #2: FP32 Enforcement ❌

**What TTT Needs**:
```python
# Separate optimizer groups with FP32 enforcement
optimizer = AdamW([
    {
        # TTT states - MUST stay FP32
        'params': [p for n, p in model.named_parameters()
                   if 'W1' in n or 'b1' in n or 'W2' in n or 'b2' in n],
        'lr': 1e-4,
    },
    {
        # Other params - can use mixed precision
        'params': [p for n, p in model.named_parameters()
                   if not ('W1' in n or 'b1' in n)],
        'lr': 1e-5,
    }
])

# Verify FP32 every forward
assert model.layers[24].self_attn.W1.dtype == torch.float32
```

**What LLaMA-Factory Does**:
```python
# Standard mixed precision training
# All params treated uniformly
# No explicit dtype enforcement per parameter group
```

**Problem**: LLaMA-Factory's mixed precision training might downcast TTT states to FP16/BF16, causing gibberish after 5-7 minutes.

---

### 3.3 Critical Gap #3: TTT-Specific Monitoring ❌

**What TTT Needs**:
```python
# From TTTMonitor class
monitor.log_layer_state(
    layer_idx=24,
    W1=model.layers[24].self_attn.W1,
    b1=model.layers[24].self_attn.b1,
    W2=model.layers[24].self_attn.W2,
    b2=model.layers[24].self_attn.b2,
)

# Track:
# - W1_mean, W1_std, W1_max (detect explosion)
# - TTT reconstruction loss
# - NaN detection
# - State evolution over time
```

**What LLaMA-Factory Provides**:
```python
# Standard metrics:
# - Training loss
# - Validation loss
# - Accuracy
# - Perplexity

# ❌ No custom TTT state monitoring
```

**Problem**: We need to monitor W1/b1 statistics to catch numerical issues early. LLaMA-Factory doesn't provide hooks for this.

---

### 3.4 Gap #4: Curriculum Training ⚠️

**What TTT Needs**:
```python
# Automated curriculum
curriculum = [
    {'max_length': 8192,  'duration_days': 2},
    {'max_length': 16384, 'duration_days': 3},
    {'max_length': 32768, 'duration_days': 4},
    {'max_length': 65536, 'duration_days': 5},
]

for stage in curriculum:
    # Filter data for this stage
    dataset = prepare_stage_data(dataset, stage['max_length'])

    # Train with progressive context
    train_one_stage(model, dataset, stage)
```

**What LLaMA-Factory Provides**:
```python
# Manual stage training
# 1. Train stage 1:
#    llamafactory-cli train config_8k.yaml
# 2. Train stage 2:
#    llamafactory-cli train config_16k.yaml
# ...

# ⚠️ Can work but requires manual orchestration
```

**Problem**: Need to manually create separate config files and run stages sequentially. No automated curriculum support.

---

## 4. Use Cases: When to Use LLaMA-Factory

### 4.1 ✅ Standard Llama-Omni Fine-Tuning

**If you want to fine-tune Llama-Omni WITHOUT TTT**:

```yaml
# config.yaml for LLaMA-Factory
model_name_or_path: ICTNLP/Llama-3.1-8B-Omni
stage: sft
do_train: true
finetuning_type: lora  # or full

dataset: your_speech_dataset
template: llama3
cutoff_len: 2048

output_dir: checkpoints/llama-omni-lora
num_train_epochs: 3
per_device_train_batch_size: 2
learning_rate: 1e-4
```

**Advantages**:
- ✅ Web UI for easy configuration
- ✅ Automatic LoRA/QLoRA setup
- ✅ Distributed training out of the box
- ✅ No code required

**Limitations**:
- ❌ Still limited to standard context length
- ❌ No TTT (unlimited context not possible)

---

### 4.2 ⚠️ Llama-Omni with TTT (Hybrid Approach)

**If you want TTT integration, use LLaMA-Factory for infrastructure only**:

```python
# 1. Create custom model with TTT
# omni_speech/model/language_model/omni_speech_llama_ttt.py

class OmniSpeechLlamaForCausalLMTTT(OmniSpeechLlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self._replace_attention_with_ttt()  # Custom TTT layers

    def forward(self, **kwargs):
        # Custom forward with state management
        return self._forward_with_ttt_state(**kwargs)

# 2. Register with HuggingFace
AutoConfig.register("omni_speech_llama_ttt", OmniSpeechTTTConfig)
AutoModelForCausalLM.register(OmniSpeechTTTConfig, OmniSpeechLlamaForCausalLMTTT)

# 3. Save model to directory with config.json
model.save_pretrained("omni-speech-ttt")

# 4. Use LLaMA-Factory for basic training loop
# BUT: Override critical components
```

**What to use from LLaMA-Factory**:
- ✅ Data loading infrastructure
- ✅ Distributed training setup (DeepSpeed/FSDP)
- ✅ Basic training loop
- ✅ Logging and experiment tracking

**What to implement custom**:
- ❌ State management (conversation-level)
- ❌ FP32 enforcement (custom hooks)
- ❌ TTT monitoring (custom callback)
- ❌ Curriculum training (custom script)

---

## 5. Recommended Approach

### Option A: Pure Custom Implementation (Recommended for TTT) ⭐

**Pros**:
- ✅ Full control over state management
- ✅ FP32 enforcement guaranteed
- ✅ TTT-specific monitoring built-in
- ✅ Curriculum training automated
- ✅ No surprises from abstraction layers

**Cons**:
- ❌ More upfront coding (~2,900 lines from docs)
- ❌ Need to implement distributed training manually
- ❌ No web UI

**When to use**: For TTT integration (our case)

---

### Option B: LLaMA-Factory + Custom Trainer (Hybrid)

**Pros**:
- ✅ Leverage LLaMA-Factory's infrastructure
- ✅ Less boilerplate code
- ✅ Built-in distributed training
- ✅ Web UI for configuration

**Cons**:
- ⚠️ Need to override critical components
- ⚠️ Fighting against framework assumptions
- ⚠️ Risk of FP32 being lost in mixed precision
- ⚠️ Complex integration with state management

**When to use**: If you want to experiment quickly but still need TTT

**Implementation**:
```python
# Custom trainer inheriting from LLaMA-Factory's trainer
from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer

class TTTTrainer(CustomSeq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ttt_monitor = TTTMonitor()
        self.conversation_states = {}

    def training_step(self, model, inputs):
        # Custom state management
        conv_id = inputs.pop('conversation_id')
        turn = inputs.pop('turn_number')

        if turn == 0:
            # Reset state for new conversation
            self._reset_ttt_state(conv_id)

        # Standard training step
        loss = super().training_step(model, inputs)

        # Custom monitoring
        self._log_ttt_stats(model)

        return loss

    def _log_ttt_stats(self, model):
        # Monitor W1, b1 statistics
        for layer_idx in range(24, 32):
            ttt_layer = model.model.layers[layer_idx].self_attn
            self.ttt_monitor.log_layer_state(
                layer_idx, ttt_layer.W1, ttt_layer.b1,
                ttt_layer.W2, ttt_layer.b2
            )
```

---

### Option C: LLaMA-Factory for Standard Fine-Tuning Only

**Use LLaMA-Factory for**:
- ✅ Fine-tuning Llama-Omni on domain-specific data
- ✅ LoRA adaptation for specific tasks
- ✅ Quick experiments without code

**But NOT for**:
- ❌ TTT integration
- ❌ Long-context training (>8k tokens)
- ❌ State-based models

---

## 6. Comparison Matrix

| Feature | Pure Custom | LLaMA-Factory Hybrid | LLaMA-Factory Only |
|---------|-------------|---------------------|-------------------|
| **TTT Support** | ✅ Full | ⚠️ Partial | ❌ No |
| **State Management** | ✅ Custom | ⚠️ Override needed | ❌ No |
| **FP32 Enforcement** | ✅ Guaranteed | ⚠️ Risky | ❌ No |
| **Curriculum Training** | ✅ Automated | ⚠️ Manual | ⚠️ Manual |
| **TTT Monitoring** | ✅ Built-in | ⚠️ Custom hooks | ❌ No |
| **Development Time** | 2-3 weeks | 1-2 weeks | 1 week |
| **Code Lines** | ~2,900 | ~1,500 | ~100 (config) |
| **Distributed Training** | ⚠️ Manual | ✅ Built-in | ✅ Built-in |
| **Web UI** | ❌ No | ⚠️ Limited | ✅ Yes |
| **Risk Level** | Low | Medium | High (for TTT) |

---

## 7. Practical Recommendations

### For Your Project (TTT Integration):

**🎯 Recommendation: Option A (Pure Custom Implementation)**

**Reasoning**:
1. TTT has **critical requirements** that are hard to guarantee in a framework
2. FP32 enforcement is **non-negotiable** - any framework abstraction risks breaking it
3. State management is **conversation-specific** - frameworks assume batch independence
4. The docs already provide **complete implementation** (~2,900 lines, well-documented)
5. **Control > Convenience** for research/novel architectures

**Timeline**:
```
Week 1-2: Implement TTT modules + integration (~2,900 lines)
Week 3-4: Testing and debugging
Week 5-9: Training (curriculum stages)
Week 10: Validation

Total: 10 weeks
```

---

### If You Still Want to Try LLaMA-Factory:

**Use it for**:
1. **Initial Llama-Omni fine-tuning** (without TTT) to test data pipeline
2. **Baseline comparison** (standard fine-tuned Llama-Omni vs TTT version)
3. **Infrastructure reference** (see how they handle distributed training)

**Example workflow**:
```bash
# Step 1: Fine-tune Llama-Omni with LLaMA-Factory (baseline)
llamafactory-cli train configs/llama_omni_baseline.yaml

# Step 2: Implement TTT with custom code (following docs)
python train_ttt.py --config configs/ttt_stage1.yaml

# Step 3: Compare results
python evaluate.py --baseline checkpoints/baseline --ttt checkpoints/ttt
```

---

## 8. Final Verdict

### Can LLaMA-Factory help?

**Yes, but only for standard fine-tuning, NOT for TTT integration.**

### Should you use it?

**For TTT integration: NO**

**Why not?**
1. ❌ Can't guarantee FP32 enforcement (critical for TTT)
2. ❌ No conversation-level state management (critical for TTT)
3. ❌ No TTT-specific monitoring (important for debugging)
4. ❌ Framework abstraction adds risk with no proportional benefit

**For baseline fine-tuning: YES**

**Why?**
1. ✅ Quick setup for standard Llama-Omni fine-tuning
2. ✅ Good for creating comparison baselines
3. ✅ Useful for testing data pipeline
4. ✅ Web UI makes experimentation easy

---

## 9. Implementation Path Forward

### Recommended Approach:

```
Phase 0 (Optional): Baseline with LLaMA-Factory
├─ Fine-tune Llama-Omni on your data (1 week)
├─ Establish baseline metrics
└─ Validate data pipeline works

Phase 1-4: TTT Implementation (Custom Code)
├─ Follow Doc 13 implementation guide
├─ Use HuggingFace Trainer (not LLaMA-Factory)
├─ Implement custom TTTTrainer with state management
└─ Full control over critical components

Phase 5: Validation
├─ Compare TTT vs baseline
├─ Demonstrate unlimited context advantage
└─ Publish results
```

---

## 10. Code Example: If Using LLaMA-Factory (Not Recommended for TTT)

**Only if you insist on trying the hybrid approach**:

```python
# llamafactory_ttt_adapter.py

from llamafactory.train.sft import run_sft
from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer
from omni_speech.monitoring.ttt_monitor import TTTMonitor
import torch

class TTTAwareTrainer(CustomSeq2SeqTrainer):
    """
    Adapter to make LLaMA-Factory work with TTT
    WARNING: This is a workaround, not ideal!
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TTT-specific additions
        self.ttt_monitor = TTTMonitor(log_dir="logs/ttt")
        self.conversation_states = {}

        # Hook to verify FP32
        self._register_fp32_verification_hooks()

    def _register_fp32_verification_hooks(self):
        """CRITICAL: Verify FP32 every forward pass"""
        def check_fp32(module, input, output):
            if hasattr(module, 'W1'):
                assert module.W1.dtype == torch.float32, \
                    f"FP32 LOST! W1 dtype: {module.W1.dtype}"

        # Register hooks on TTT layers
        for layer_idx in range(24, 32):
            self.model.model.layers[layer_idx].self_attn.register_forward_hook(check_fp32)

    def training_step(self, model, inputs):
        """Override to add state management"""

        # Extract TTT-specific metadata
        conversation_id = inputs.pop('conversation_id', None)
        turn_number = inputs.pop('turn_number', 0)

        # Handle state reset
        if conversation_id and turn_number == 0:
            # Reset TTT state for new conversation
            # (This is hacky - LLaMA-Factory doesn't support this natively)
            for layer_idx in range(24, 32):
                layer = model.model.layers[layer_idx].self_attn
                if hasattr(layer, 'reset_state'):
                    layer.reset_state()

        # Standard training step
        loss = super().training_step(model, inputs)

        # Monitor TTT states
        if self.state.global_step % 100 == 0:
            self._log_ttt_statistics(model)

        return loss

    def _log_ttt_statistics(self, model):
        """Log W1, b1 statistics for monitoring"""
        for layer_idx in range(24, 32):
            ttt_layer = model.model.layers[layer_idx].self_attn
            if hasattr(ttt_layer, 'W1'):
                self.ttt_monitor.log_layer_state(
                    layer_idx=layer_idx,
                    W1=ttt_layer.W1,
                    b1=ttt_layer.b1,
                    W2=getattr(ttt_layer, 'W2', None),
                    b2=getattr(ttt_layer, 'b2', None),
                )

# Usage:
# llamafactory-cli train config.yaml --trainer_class TTTAwareTrainer
```

**Problems with this approach**:
1. ⚠️ Fighting against framework assumptions
2. ⚠️ Fragile (framework updates might break)
3. ⚠️ No guarantee FP32 is maintained through all code paths
4. ⚠️ Conversation state management is hacky

---

## Conclusion

**For Llama-Omni + TTT integration:**

❌ **Don't rely on LLaMA-Factory** for the core training

✅ **Use custom implementation** following Doc 13

⚠️ **Optionally use LLaMA-Factory** for:
- Baseline fine-tuning (comparison)
- Data pipeline testing
- Infrastructure reference

**The TTT requirements are too specialized for a general-purpose framework to handle safely.**

---

**Next Steps**: Proceed with custom implementation as documented in Doc 10 + Doc 13. Estimated timeline: 10 weeks from start to validated model.
