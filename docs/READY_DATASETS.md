# Ready-to-Use Datasets for TTT + Llama-Omni Training

**Question**: Are there datasets already in the correct format?
**Answer**: ✅ **YES - 3 Production-Ready Datasets Found!**

---

## Summary Table

| Dataset | Size | Format | Audio+Transcript | Multi-Turn | TTT-Ready | Download |
|---------|------|--------|-----------------|-----------|-----------|----------|
| **InstructS2S-200K** | 200K samples | JSON + Audio | ✅ | ✅ | ⚠️ Needs conv_id | HuggingFace |
| **DeepDialogue** | 40K dialogues<br>480 hours | JSON + WAV | ✅ | ✅ | ✅ | HuggingFace |
| **MultiDialog** | 9K dialogues<br>340 hours | HF Dataset | ✅ | ✅ | ⚠️ Needs conv_id | HuggingFace |

---

## 1. InstructS2S-200K (Official Llama-Omni Dataset) ⭐

### Overview

**The exact dataset used to train Llama-Omni!**

- **Size**: 200K speech instruction-response pairs
- **Source**: Alpaca (50K) + UltraChat (150K)
- **Format**: JSON files with discrete speech units
- **Location**: https://huggingface.co/datasets/ICTNLP/InstructS2S-200K

### Format

```
Dataset Files:
├── en_part_00.json (10.7 GB)
├── en_part_01.json (10.7 GB)
├── en_part_02.json (10.7 GB)
├── en_part_03.json (10.7 GB)
├── en_part_04.json (10.7 GB)
└── instruct_en_200k_data_cosy2.json
```

### Data Structure

```json
{
  "instruction": "Speech instruction (text)",
  "speech_input": "Discrete speech units (1000 clusters from HuBERT)",
  "response": "Text response",
  "speech_response": "Speech units for response"
}
```

**Note**: Uses discrete units, not raw audio. You may need to:
1. Convert units back to audio (if needed), OR
2. Use as-is with HuBERT feature extractor

### ✅ Has:
- Speech representations
- Text transcripts (instructions + responses)
- Multi-turn structure (from UltraChat)

### ⚠️ Needs:
- **Conversation ID tracking** (for TTT state management)
- May need audio reconstruction from units

### Download

```bash
# Using HuggingFace datasets
from datasets import load_dataset

dataset = load_dataset("ICTNLP/InstructS2S-200K")
```

**Size**: ~50-60 GB total

---

## 2. DeepDialogue-xtts (Best for TTT Training) ⭐⭐⭐

### Overview

**RECOMMENDED: Perfect format for TTT!**

- **Dialogues**: 40,150 high-quality multi-turn conversations
- **Duration**: 480+ hours of audio
- **Domains**: 41 different topics
- **Emotions**: 20 emotion types with coherent progressions
- **Avg Length**: 6.1 turns per dialogue
- **Audio**: High-quality TTS with XTTS-v2 (emotional conditioning)
- **Location**: https://huggingface.co/datasets/SALT-Research/DeepDialogue-xtts

### Format

```
data/
├── dialogues_[model]/
│   ├── dialogue_001.json          ← Metadata
│   └── dialogue_001/
│       ├── dialogue_001_full.wav  ← Full conversation audio
│       ├── metadata.tsv           ← Turn-level info
│       └── segments/
│           ├── turn_0.wav         ← Individual turn audio
│           ├── turn_1.wav
│           └── ...
```

### Data Structure

**dialogue_001.json**:
```json
{
  "dialogue_id": "dialogue_001",
  "domain": "education",
  "turns": [
    {
      "turn_id": 0,
      "speaker": "A",
      "text": "Hello, how can I help you today?",
      "emotion": "neutral",
      "audio_file": "segments/turn_0.wav"
    },
    {
      "turn_id": 1,
      "speaker": "B",
      "text": "I need help understanding this concept.",
      "emotion": "confused",
      "audio_file": "segments/turn_1.wav"
    }
  ]
}
```

**metadata.tsv**:
```
audio_path              text                                    emotion    speaker
segments/turn_0.wav     "Hello, how can I help you today?"     neutral    A
segments/turn_1.wav     "I need help understanding this..."    confused   B
```

### ✅ Has (Perfect for TTT!):
- ✅ WAV files (16kHz, high quality)
- ✅ Text transcripts (in JSON and TSV)
- ✅ **Conversation IDs** (dialogue_id)
- ✅ **Turn numbers** (turn_id, 0-indexed!)
- ✅ Multi-speaker structure
- ✅ Long conversations (6.1 turns average)
- ✅ Emotion annotations (bonus!)

### ⚠️ Minor Adaptation Needed:

Just rename fields to match TTT requirements:

```python
# Preprocessing script
def convert_deepdialogue_to_ttt_format(dialogue):
    return {
        'conversation_id': dialogue['dialogue_id'],  # ✅ Already has!
        'turn_number': turn['turn_id'],              # ✅ Already has!
        'speaker': turn['speaker'],
        'audio_file': turn['audio_file'],
        'transcript': turn['text'],
    }
```

### Download

```bash
# Option 1: HuggingFace datasets
from datasets import load_dataset
dataset = load_dataset("SALT-Research/DeepDialogue-xtts")

# Option 2: Git LFS (for full control)
git lfs install
git clone https://huggingface.co/datasets/SALT-Research/DeepDialogue-xtts
```

**Size**: 173 GB (480 hours of audio)

**Alternative**: **DeepDialogue-orpheus** (same structure, different TTS model)
- https://huggingface.co/datasets/SALT-Research/DeepDialogue-orpheus
- Also 480 hours, more natural speech quality

---

## 3. MultiDialog (Audio-Visual Dialogues)

### Overview

**First large-scale audio-visual dialogue corpus**

- **Dialogues**: 9,920 conversations
- **Duration**: 340 hours
- **Turns**: 106,624 turns, 218,248 utterances
- **Modality**: Audio + Visual (face-to-face)
- **Source**: Based on TopicalChat (open-domain)
- **Location**: https://huggingface.co/datasets/IVLLab/MultiDialog

### Format (HuggingFace Dataset)

```python
# Dataset structure
{
    'file_name': 'path/to/audio.wav',
    'conv_id': 'conversation_001',          # ✅ Conversation ID!
    'utterance_id': 'utt_0',                 # ✅ Turn identifier
    'from': 'speaker_A',
    'value': 'Hello, how are you?',          # ✅ Transcript
    'audio': {
        'path': 'path/to/audio.wav',
        'array': [0.1, 0.2, ...],            # ✅ Decoded audio
        'sampling_rate': 16000
    },
    'original_full_path': 'original/path'
}
```

### ✅ Has:
- ✅ WAV files (embedded in dataset)
- ✅ Transcripts (in 'value' field)
- ✅ **Conversation IDs** (conv_id)
- ✅ Turn identifiers (utterance_id)
- ✅ Multi-speaker
- ✅ Long-form (340 hours)
- ✅ HuggingFace format (easy loading)

### ⚠️ Minor Adaptation:

Convert utterance_id to turn_number:

```python
def convert_multidialog_to_ttt_format(sample):
    # Extract turn number from utterance_id
    # e.g., "utt_0" → 0, "utt_1" → 1
    turn_num = int(sample['utterance_id'].split('_')[1])

    return {
        'conversation_id': sample['conv_id'],
        'turn_number': turn_num,
        'speech': sample['audio']['array'],
        'speech_lengths': len(sample['audio']['array']),
        'transcript': sample['value'],
    }
```

### Download

```python
from datasets import load_dataset

# Load specific split
dataset = load_dataset("IVLLab/MultiDialog", "valid_freq")

# Access data
audio = dataset["valid_freq"][0]["audio"]      # Audio array
transcript = dataset["valid_freq"][0]["value"] # Transcript
conv_id = dataset["valid_freq"][0]["conv_id"]  # Conversation ID
```

**Splits**:
- `train`
- `valid_freq`, `valid_rare`
- `test_freq`, `test_rare`

**Size**: ~150 GB (340 hours audio-visual)

---

## Comparison: Which Dataset to Use?

### For TTT Training (Ranked)

#### 🥇 **DeepDialogue-xtts** - BEST CHOICE

**Pros**:
- ✅ Perfect structure (dialogue_id, turn_id already present)
- ✅ Long conversations (6.1 turns avg)
- ✅ Massive scale (480 hours, 40K dialogues)
- ✅ High-quality audio (XTTS-v2 TTS)
- ✅ Emotion annotations (bonus for rich training)
- ✅ JSON format (easy preprocessing)
- ✅ Minimal preprocessing needed

**Cons**:
- ⚠️ Large download (173 GB)
- ⚠️ Synthetic speech (TTS-generated, not natural)

**Use when**: You want production-quality training with minimal setup

---

#### 🥈 **MultiDialog** - SECOND BEST

**Pros**:
- ✅ HuggingFace dataset format (easiest to load)
- ✅ Natural speech (real conversations, not TTS)
- ✅ Audio-visual (can use video too)
- ✅ Has conv_id and turn tracking
- ✅ Long-form (340 hours)

**Cons**:
- ⚠️ Smaller scale (9K dialogues vs 40K)
- ⚠️ Need to convert utterance_id → turn_number
- ⚠️ Large download (~150 GB)

**Use when**: You want natural speech and don't mind smaller scale

---

#### 🥉 **InstructS2S-200K** - OFFICIAL BUT NEEDS WORK

**Pros**:
- ✅ Official Llama-Omni training data
- ✅ Exactly what was used for the paper
- ✅ Large scale (200K samples)
- ✅ Already proven to work

**Cons**:
- ⚠️ Uses discrete units (not raw audio)
- ⚠️ May need audio reconstruction
- ⚠️ Need to add conversation ID tracking
- ⚠️ Single-turn focus (not multi-turn dialogues)

**Use when**: You want to replicate exact Llama-Omni training, but note this is for standard training, not TTT long-context

---

## Quick Start Guide

### Option A: DeepDialogue-xtts (Recommended)

```bash
# 1. Download dataset
git lfs install
git clone https://huggingface.co/datasets/SALT-Research/DeepDialogue-xtts

# 2. Create preprocessing script
python scripts/preprocess_deepdialogue.py \
    --input DeepDialogue-xtts/data \
    --output dataset/ttt_training \
    --max_length 8192  # For Stage 1 curriculum

# 3. Verify format
python scripts/validate_dataset.py --data dataset/ttt_training
```

**Preprocessing Script** (pseudo-code):
```python
# preprocess_deepdialogue.py

import json
import torchaudio
from transformers import AutoTokenizer

def process_dialogue(dialogue_path):
    # Load dialogue metadata
    with open(f"{dialogue_path}.json") as f:
        metadata = json.load(f)

    samples = []
    for turn in metadata['turns']:
        # Load audio
        audio, sr = torchaudio.load(f"{dialogue_path}/{turn['audio_file']}")

        # Resample to 16kHz if needed
        if sr != 16000:
            audio = torchaudio.functional.resample(audio, sr, 16000)

        # Tokenize transcript
        transcript = turn['text']
        input_ids = tokenizer.encode(f"<SPEECH> {transcript}")

        # Create sample
        sample = {
            'speech': audio.squeeze(0),
            'speech_lengths': audio.shape[1],
            'input_ids': input_ids,
            'labels': create_labels(input_ids),  # IGNORE before transcript
            'conversation_id': metadata['dialogue_id'],  # ✅ Already has!
            'turn_number': turn['turn_id'],              # ✅ Already has!
        }

        # Pad to 64-multiple
        sample = pad_to_mini_batch(sample, mini_batch_size=64)

        samples.append(sample)

    return samples
```

---

### Option B: MultiDialog (Easiest Loading)

```python
# 1. Load dataset
from datasets import load_dataset
dataset = load_dataset("IVLLab/MultiDialog", "train")

# 2. Preprocess
def preprocess_sample(sample):
    turn_num = int(sample['utterance_id'].split('_')[1])

    return {
        'speech': sample['audio']['array'],
        'speech_lengths': len(sample['audio']['array']),
        'transcript': sample['value'],
        'conversation_id': sample['conv_id'],
        'turn_number': turn_num,
    }

# 3. Apply preprocessing
processed = dataset.map(preprocess_sample)

# 4. Save to disk
processed.save_to_disk("dataset/multidialog_processed")
```

---

### Option C: InstructS2S-200K (Advanced)

```python
# 1. Load dataset
from datasets import load_dataset
dataset = load_dataset("ICTNLP/InstructS2S-200K")

# 2. Convert discrete units to audio (if needed)
# This requires HuBERT vocoder or similar

# 3. Add conversation tracking
# (Dataset is single-turn, need to group into conversations)

def create_synthetic_conversations(dataset, turns_per_conv=5):
    conversations = []
    for i in range(0, len(dataset), turns_per_conv):
        conv_id = f"conv_{i // turns_per_conv}"
        for turn_idx in range(turns_per_conv):
            sample = dataset[i + turn_idx]
            sample['conversation_id'] = conv_id
            sample['turn_number'] = turn_idx
            conversations.append(sample)
    return conversations
```

---

## Curriculum Training: Dataset Filtering

### Stage 1: 8k Context

```python
# Filter dialogues by total token length
def filter_for_stage_1(dataset, max_tokens=8192):
    filtered = []
    for dialogue in dataset:
        total_tokens = sum(len(turn['input_ids']) for turn in dialogue['turns'])
        if total_tokens <= max_tokens:
            filtered.append(dialogue)
    return filtered

stage1_data = filter_for_stage_1(deepdialogue_dataset, max_tokens=8192)
print(f"Stage 1: {len(stage1_data)} dialogues")
```

### Apply to All Stages

```python
curriculum = [
    {'name': 'stage_1', 'max_tokens': 8192},
    {'name': 'stage_2', 'max_tokens': 16384},
    {'name': 'stage_3', 'max_tokens': 32768},
    {'name': 'stage_4', 'max_tokens': 65536},
]

for stage in curriculum:
    filtered = filter_by_length(dataset, stage['max_tokens'])
    save_to_disk(filtered, f"data/{stage['name']}")
    print(f"{stage['name']}: {len(filtered)} dialogues")
```

---

## Validation Checklist

Before training, verify your preprocessed dataset:

```python
# validate_dataset.py

def validate_ttt_dataset(dataset_path):
    dataset = load_from_disk(dataset_path)

    print("Validating dataset...")

    # Check 1: Has required fields
    sample = dataset[0]
    required_fields = ['speech', 'speech_lengths', 'input_ids', 'labels',
                      'conversation_id', 'turn_number']
    for field in required_fields:
        assert field in sample, f"Missing field: {field}"
    print("✅ All required fields present")

    # Check 2: Audio format
    assert sample['speech'].dtype == torch.float32
    assert sample['speech'].ndim == 1  # [num_samples]
    print("✅ Audio format correct")

    # Check 3: Padding to 64-multiple
    assert len(sample['input_ids']) % 64 == 0
    print("✅ Padding correct")

    # Check 4: Conversation tracking
    conv_groups = {}
    for sample in dataset:
        conv_id = sample['conversation_id']
        turn = sample['turn_number']
        if conv_id not in conv_groups:
            conv_groups[conv_id] = []
        conv_groups[conv_id].append(turn)

    # Verify turn 0 exists for each conversation
    for conv_id, turns in conv_groups.items():
        assert 0 in turns, f"Conversation {conv_id} missing turn 0 (for state reset)"
    print("✅ Conversation structure valid")

    # Check 5: Label alignment
    assert len(sample['input_ids']) == len(sample['labels'])
    print("✅ Labels aligned with input_ids")

    print(f"\n✅ Dataset valid! {len(dataset)} samples")
    print(f"   Conversations: {len(conv_groups)}")
    print(f"   Avg turns/conv: {len(dataset) / len(conv_groups):.1f}")
```

---

## Final Recommendation

### 🎯 **Use DeepDialogue-xtts**

**Why**:
1. ✅ Already has conversation_id and turn_number (critical for TTT!)
2. ✅ Massive scale (480 hours, 40K dialogues)
3. ✅ Perfect structure for multi-turn learning
4. ✅ Minimal preprocessing needed
5. ✅ JSON format (easy to work with)

**Timeline**:
- Download: 2-4 hours (173 GB)
- Preprocessing: 1 day
- Validation: 2 hours
- **Ready to train**: 2 days total

### 🥈 **Fallback: MultiDialog**

If you prefer natural speech over TTS, or want the easiest HuggingFace integration.

### 🥉 **Advanced: InstructS2S-200K**

Only if you specifically want to replicate Llama-Omni's exact training setup, but note this is for standard training (not optimized for TTT long-context).

---

## Next Steps

1. **Choose dataset** (recommend DeepDialogue-xtts)
2. **Download** (~173 GB, 2-4 hours)
3. **Run preprocessing** script (I can provide complete script)
4. **Validate** with checklist above
5. **Start training** with curriculum (Stage 1: 8k context)

**Want me to create the complete preprocessing script for your chosen dataset?**
