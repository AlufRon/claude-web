# Complete Dataset Requirements for TTT + Llama-Omni Training

**Critical Question**: Are WAV files enough?
**Answer**: ❌ **NO** - You need WAV files + Transcripts + Metadata

---

## 1. Why You Need More Than Just Audio

### The Problem:

Llama-Omni is a **speech-to-text-to-speech** model, not just speech processing:

```
Speech Input (WAV)
    ↓
Whisper Encoder → Extract speech features
    ↓
Speech Projector → Convert to LLM space
    ↓
LLM (Llama 3.1) → Generate TEXT tokens  ← NEEDS TEXT SUPERVISION!
    ↓
(Optional) Speech Decoder → Convert to speech
```

**The LLM generates TEXT tokens**, so you need:
1. ✅ WAV files (for speech encoder input)
2. ✅ **Text transcripts** (for LLM supervision/labels)
3. ✅ **Conversation metadata** (for TTT state management)

---

## 2. Complete Dataset Structure

### 2.1 Minimum Required Components

```
dataset/
├── conversations/
│   ├── conv_001/
│   │   ├── turn_0.wav           ← Audio file
│   │   ├── turn_0.txt           ← Transcript (what was said)
│   │   ├── turn_1.wav
│   │   ├── turn_1.txt
│   │   └── metadata.json        ← Conversation info
│   ├── conv_002/
│   │   └── ...
│   └── conv_N/
└── manifest.jsonl               ← Dataset index
```

### 2.2 File Formats

#### **Audio Files (turn_X.wav)**

```
Format: WAV
Sample Rate: 16 kHz (required by Whisper)
Channels: Mono (1 channel)
Bit Depth: 16-bit
Duration: Variable (typically 10 sec - 5 min per turn)
```

**Example**:
```bash
# Verify audio format
ffprobe turn_0.wav

# Should show:
# - Codec: pcm_s16le
# - Sample rate: 16000 Hz
# - Channels: 1 (mono)
```

---

#### **Transcript Files (turn_X.txt)**

Plain text file containing what was said in the audio.

**Example** (`turn_0.txt`):
```
Hello, how are you doing today? I wanted to ask you about your recent project.
```

**Example** (`turn_1.txt`):
```
I'm doing great, thanks for asking! The project has been going really well. We've made significant progress on the main features.
```

**Important**:
- ✅ Clean text (no timestamps, no special markers)
- ✅ Proper punctuation and capitalization
- ✅ Speaker-separated (one file per turn)
- ❌ Don't include [SPEAKER: ...] tags (handled in metadata)

---

#### **Metadata File (metadata.json)**

Contains conversation structure and TTT-specific information.

**Example**:
```json
{
  "conversation_id": "conv_001",
  "total_turns": 10,
  "total_duration_sec": 1800,
  "speakers": ["A", "B"],
  "turns": [
    {
      "turn_id": 0,
      "speaker": "A",
      "audio_file": "turn_0.wav",
      "transcript_file": "turn_0.txt",
      "duration_sec": 15.3,
      "start_time": 0.0,
      "end_time": 15.3
    },
    {
      "turn_id": 1,
      "speaker": "B",
      "audio_file": "turn_1.wav",
      "transcript_file": "turn_1.txt",
      "duration_sec": 23.7,
      "start_time": 15.3,
      "end_time": 39.0
    },
    {
      "turn_id": 2,
      "speaker": "A",
      "audio_file": "turn_2.wav",
      "transcript_file": "turn_2.txt",
      "duration_sec": 18.5,
      "start_time": 39.0,
      "end_time": 57.5
    }
  ]
}
```

**Critical Fields**:
- `conversation_id`: For TTT state tracking (CRITICAL!)
- `turn_id`: For knowing when to reset state (turn_id=0 → reset)
- `speaker`: For multi-speaker dialogues
- `audio_file`, `transcript_file`: Links to actual data

---

#### **Dataset Manifest (manifest.jsonl)**

Index of all conversations for training.

**Example** (`manifest.jsonl`):
```jsonl
{"conversation_id": "conv_001", "path": "conversations/conv_001", "num_turns": 10, "total_tokens": 4532, "duration_sec": 1800}
{"conversation_id": "conv_002", "path": "conversations/conv_002", "num_turns": 8, "total_tokens": 3891, "duration_sec": 1520}
{"conversation_id": "conv_003", "path": "conversations/conv_003", "num_turns": 15, "total_tokens": 8234, "duration_sec": 3600}
```

**Purpose**: Quick filtering for curriculum training
```python
# For Stage 1 (8k tokens), only use conversations with total_tokens < 8192
# For Stage 2 (16k tokens), use conversations with total_tokens < 16384
# etc.
```

---

## 3. Why Each Component is Needed

### 3.1 WAV Files ✅ (You have this)

**Purpose**: Input to Whisper encoder

```python
# Processing pipeline
speech_waveform = load_audio("turn_0.wav")  # [num_samples, 1]
speech_features = whisper_encoder(speech_waveform)  # [T, 1280]
```

**What happens without WAV?** → Can't encode speech input ❌

---

### 3.2 Text Transcripts ✅ (CRITICAL - You need this!)

**Purpose**: Supervision for LLM training

```python
# Llama-Omni training
input_sequence = ["<SPEECH>", "token_1", "token_2", ...]
labels = [IGNORE, "token_1", "token_2", ...]  # What LLM should predict

# The LLM learns:
# Given <SPEECH> (replaced by speech features), predict "token_1"
# Given "token_1", predict "token_2"
# etc.
```

**From code** (`omni_speech_arch.py`, line 124):
```python
if labels is None:
    labels = torch.full_like(input_ids, IGNORE_INDEX)
```

**What happens without transcripts?** → No labels for LLM training ❌
- LLM can't learn to generate text from speech
- Training loss undefined
- Model learns nothing

---

### 3.3 Conversation Metadata ✅ (CRITICAL for TTT!)

**Purpose**: TTT state management

```python
# From training strategy
for batch in dataloader:
    conv_id = batch['conversation_id']
    turn = batch['turn_number']

    # CRITICAL: Only reset TTT state at conversation start
    if turn == 0:
        model.reset_conversation_state(conv_id)

    # State persists across turns in same conversation
    outputs = model(batch, use_cache=True)
```

**What happens without metadata?** → TTT state resets every batch ❌
- Defeats the purpose of TTT (state-based learning)
- Gibberish output
- No long-context ability

---

## 4. Data Format for Training

### 4.1 PyTorch Dataset Output

After preprocessing, each sample should return:

```python
{
    # Audio (from WAV file)
    'speech': torch.FloatTensor,          # [num_samples, 1] - 16kHz waveform
    'speech_lengths': torch.LongTensor,   # [1] - actual length

    # Text (from transcript)
    'input_ids': torch.LongTensor,        # [seq_len] - tokenized with <SPEECH> markers
    'labels': torch.LongTensor,           # [seq_len] - targets for LM loss
    'attention_mask': torch.BoolTensor,   # [seq_len] - padding mask

    # TTT State Management (from metadata)
    'conversation_id': str,               # "conv_001"
    'turn_number': int,                   # 0, 1, 2, ... (0 = reset state!)

    # Optional Metadata
    'speaker_id': int,                    # 0, 1, 2, ...
    'original_length': int,               # Before padding
}
```

### 4.2 How Transcripts Become input_ids

**Example Conversation**:

```
Turn 0 (User speaks):
  Audio: turn_0.wav
  Transcript: "Hello, how can I help you today?"

Turn 1 (Assistant responds):
  Text: "I need help with my computer."
```

**Tokenization**:

```python
# Llama 3.1 format with speech
input_sequence = [
    "<|begin_of_text|>",
    "<|start_header_id|>", "user", "<|end_header_id|>",
    "<SPEECH>",  # ← SPEECH_TOKEN_INDEX (replaced by speech features)
    "<|eot_id|>",

    "<|start_header_id|>", "assistant", "<|end_header_id|>",
    "I", "need", "help", "with", "my", "computer", ".",
    "<|eot_id|>"
]

# Labels (what to predict)
labels = [
    IGNORE, IGNORE, IGNORE, IGNORE,
    IGNORE,  # Ignore speech token
    IGNORE,
    IGNORE, IGNORE, IGNORE,
    "I", "need", "help", "with", "my", "computer", ".",  # Predict assistant response!
    "<|eot_id|>"
]
```

**The transcript is ESSENTIAL** for creating labels!

---

## 5. Can You Train Without Transcripts?

### Self-Supervised Speech-Only Training?

**Theoretically**: Could train speech encoder in self-supervised way (like Wav2Vec2)

**But for Llama-Omni**: ❌ **NO**

**Why?**
1. Llama-Omni's LLM is **text-based** (generates text tokens)
2. Without transcripts, no supervision for LLM
3. Training loss requires labels (text tokens)
4. You'd be training encoder only, not the full model

**What you'd get**: A speech encoder that extracts features, but no language model that understands/generates

---

## 6. Where to Get Data

### 6.1 Existing Datasets (WAV + Transcripts Available) ✅

#### **LibriSpeech** (Best for Long-Form)
```
Format: Clean audiobooks with transcripts
Duration: 1000+ hours
Speakers: 2,484
URL: https://www.openslr.org/12/

Structure:
LibriSpeech/
├── train-clean-360/
│   ├── 19/
│   │   ├── 198/
│   │   │   ├── 19-198-0000.flac  ← Audio
│   │   │   ├── 19-198-0001.flac
│   │   │   └── 19-198.trans.txt  ← Transcripts (all segments)
```

**Advantages**:
- ✅ Clean audio
- ✅ Long-form (audiobooks = hours of continuous speech)
- ✅ Transcripts included
- ✅ Free & open-source

**Disadvantages**:
- ⚠️ Single speaker (not conversational)
- ⚠️ Need to add conversation structure (can simulate)

---

#### **Common Voice** (Multi-Speaker)
```
Format: Crowdsourced recordings with transcripts
Duration: 20,000+ hours (multiple languages)
Speakers: 100,000+
URL: https://commonvoice.mozilla.org/

Structure:
common_voice/
├── clips/
│   ├── sample_001.mp3  ← Audio
│   ├── sample_002.mp3
└── train.tsv  ← Metadata with transcripts

TSV Format:
path           sentence                    ...
sample_001.mp3 "Hello, how are you?"       ...
sample_002.mp3 "I'm doing great, thanks."  ...
```

**Advantages**:
- ✅ Multi-speaker
- ✅ Transcripts included
- ✅ Diverse accents/styles

**Disadvantages**:
- ⚠️ Short clips (5-10 sec each)
- ⚠️ Need to group into conversations

---

#### **GigaSpeech** (Large-Scale)
```
Format: Podcasts, audiobooks, YouTube
Duration: 10,000 hours
URL: https://github.com/SpeechColab/GigaSpeech

Includes:
- Audio files
- Transcripts (forced alignment)
- Speaker diarization
```

---

### 6.2 Generate Synthetic Data (If No Real Conversations Available)

**Option**: Use TTS to create speech from text conversations

```python
# 1. Get text conversations (e.g., from dialogue datasets)
text_conversations = load_dataset("daily_dialog")

# 2. Use TTS to generate speech
from TTS.api import TTS
tts = TTS("tts_models/en/ljspeech/tacotron2-DDC")

for turn in conversation:
    # Generate WAV from text
    tts.tts_to_file(
        text=turn['text'],
        file_path=f"turn_{turn['id']}.wav"
    )

    # Save transcript
    with open(f"turn_{turn['id']}.txt", 'w') as f:
        f.write(turn['text'])
```

**Advantages**:
- ✅ Can create unlimited data
- ✅ Perfect transcripts (you generate from text)
- ✅ Can control conversation structure

**Disadvantages**:
- ⚠️ Synthetic audio (quality depends on TTS)
- ⚠️ Less natural than real conversations

---

## 7. Preprocessing Pipeline

### 7.1 From Raw Data to Training Format

```python
# preprocess_dataset.py

import torchaudio
from transformers import AutoTokenizer
import json

class LongConversationPreprocessor:
    def __init__(self, tokenizer_path="meta-llama/Llama-3.1-8B-Instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.speech_token = "<SPEECH>"
        self.speech_token_id = 32000  # Custom token

    def process_conversation(self, conv_dir):
        """
        Convert raw conversation to training format

        Input: conversations/conv_001/
        Output: Preprocessed samples ready for DataLoader
        """

        # 1. Load metadata
        with open(f"{conv_dir}/metadata.json") as f:
            metadata = json.load(f)

        samples = []

        # 2. Process each turn
        for turn_idx, turn in enumerate(metadata['turns']):
            # Load audio
            speech_path = f"{conv_dir}/{turn['audio_file']}"
            speech, sr = torchaudio.load(speech_path)

            # Resample to 16kHz if needed
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                speech = resampler(speech)

            # Load transcript
            with open(f"{conv_dir}/{turn['transcript_file']}") as f:
                transcript = f.read().strip()

            # Tokenize transcript
            # Format: <SPEECH> → transcript
            input_text = f"{self.speech_token} {transcript}"
            input_ids = self.tokenizer.encode(input_text, add_special_tokens=True)

            # Create labels (ignore <SPEECH> token, predict transcript)
            labels = input_ids.copy()
            speech_token_idx = labels.index(self.speech_token_id)
            labels[:speech_token_idx+1] = [IGNORE_INDEX] * (speech_token_idx + 1)

            # Pad to mini_batch_size (64) multiple
            seq_len = len(input_ids)
            pad_len = (64 - seq_len % 64) % 64
            if pad_len > 0:
                input_ids.extend([self.tokenizer.pad_token_id] * pad_len)
                labels.extend([IGNORE_INDEX] * pad_len)

            sample = {
                'speech': speech.squeeze(0),  # [num_samples]
                'speech_lengths': torch.tensor([speech.shape[1]]),
                'input_ids': torch.tensor(input_ids),
                'labels': torch.tensor(labels),
                'conversation_id': metadata['conversation_id'],
                'turn_number': turn_idx,
            }

            samples.append(sample)

        return samples

# Usage:
preprocessor = LongConversationPreprocessor()
samples = preprocessor.process_conversation("conversations/conv_001")
```

---

## 8. Minimum Viable Dataset (MVP)

### For Initial Testing:

```
Minimum Requirements:
- 10 conversations
- 5-10 turns each
- 30-60 min total duration
- WAV files (16kHz, mono)
- Transcripts (plain text)
- Metadata (JSON)

This is enough to:
✅ Test data pipeline
✅ Verify model training runs
✅ Debug TTT state management
✅ Validate preprocessing

NOT enough for:
❌ Production model
❌ Long-context evaluation
❌ Generalizable results
```

### For Full Training:

```
Recommended:
- 1,000+ conversations
- 10-50 turns each
- 100-500 hours total duration
- Diverse speakers
- Natural conversations (or high-quality synthetic)

This enables:
✅ Curriculum training (8k → 16k → 32k → 64k)
✅ Generalization
✅ Robust long-context ability
✅ Production-quality model
```

---

## 9. Quick Start: Create Sample Dataset

### 9.1 Using LibriSpeech (Easiest)

```bash
# 1. Download LibriSpeech
wget https://www.openslr.org/resources/12/train-clean-360.tar.gz
tar -xzf train-clean-360.tar.gz

# 2. Convert to conversation format
python scripts/convert_librispeech_to_conversations.py \
    --input LibriSpeech/train-clean-360 \
    --output dataset/conversations \
    --min_duration 600  # 10 minutes minimum per "conversation"
```

**Example conversion**:
```python
# convert_librispeech_to_conversations.py

# LibriSpeech has individual sentences
# Combine into "conversations" (simulated)

# Input: 19-198-0000.flac, 19-198-0001.flac, ...
# Output: Group as conversation turns

# Turn audiobook chapter into conversation:
# - Each paragraph = one turn
# - Speaker ID = "reader"
# - Conversation ID = "book_X_chapter_Y"
```

---

### 9.2 Using TTS (Most Flexible)

```python
# generate_synthetic_conversations.py

from TTS.api import TTS
import json

# 1. Load text conversations
dialogs = [
    {
        "conv_id": "synthetic_001",
        "turns": [
            {"speaker": "A", "text": "Hello, how can I help you today?"},
            {"speaker": "B", "text": "I need help with my project."},
            {"speaker": "A", "text": "Sure, what project are you working on?"},
            # ... more turns
        ]
    }
]

# 2. Generate audio with TTS
tts = TTS("tts_models/en/vctk/vits")  # Multi-speaker TTS

for dialog in dialogs:
    conv_dir = f"dataset/conversations/{dialog['conv_id']}"
    os.makedirs(conv_dir, exist_ok=True)

    metadata = {
        "conversation_id": dialog['conv_id'],
        "turns": []
    }

    for idx, turn in enumerate(dialog['turns']):
        # Generate audio
        wav_path = f"{conv_dir}/turn_{idx}.wav"
        tts.tts_to_file(
            text=turn['text'],
            speaker="p225" if turn['speaker'] == "A" else "p226",
            file_path=wav_path
        )

        # Save transcript
        txt_path = f"{conv_dir}/turn_{idx}.txt"
        with open(txt_path, 'w') as f:
            f.write(turn['text'])

        # Add to metadata
        duration = get_audio_duration(wav_path)
        metadata['turns'].append({
            "turn_id": idx,
            "speaker": turn['speaker'],
            "audio_file": f"turn_{idx}.wav",
            "transcript_file": f"turn_{idx}.txt",
            "duration_sec": duration
        })

    # Save metadata
    with open(f"{conv_dir}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
```

---

## 10. Checklist: Is Your Dataset Ready?

### Before Training:

- [ ] **WAV files exist** (16kHz, mono, WAV format)
- [ ] **Transcripts exist** (one .txt file per audio file)
- [ ] **Metadata exists** (metadata.json per conversation)
- [ ] **Conversation IDs** assigned (unique per conversation)
- [ ] **Turn numbers** assigned (0-indexed, 0 = reset TTT state)
- [ ] **Manifest created** (manifest.jsonl for quick filtering)
- [ ] **Data validated** (run preprocessing on sample)
- [ ] **Curriculum splits** prepared (8k, 16k, 32k, 64k token groups)

### Validation Tests:

```python
# test_dataset.py

def test_dataset_structure():
    """Verify dataset has all required components"""

    # Load sample
    sample = dataset[0]

    # Check audio
    assert 'speech' in sample
    assert sample['speech'].shape[1] > 0  # Has samples
    assert sample['speech'].dtype == torch.float32

    # Check transcripts (via input_ids)
    assert 'input_ids' in sample
    assert 'labels' in sample
    assert len(sample['input_ids']) == len(sample['labels'])

    # Check metadata
    assert 'conversation_id' in sample
    assert 'turn_number' in sample
    assert isinstance(sample['turn_number'], int)

    # Check padding
    assert len(sample['input_ids']) % 64 == 0  # Multiple of mini_batch_size

    print("✅ Dataset structure valid!")
```

---

## Final Answer

### ❌ **NO, WAV files alone are NOT enough**

You need:

1. **WAV files** (16kHz, mono) ← You mentioned having these
2. **Text transcripts** (what was said in each audio) ← REQUIRED!
3. **Metadata** (conversation_id, turn_number) ← REQUIRED for TTT!

Without transcripts and metadata:
- ❌ LLM has no supervision
- ❌ TTT state management broken
- ❌ Training impossible

### Next Steps:

1. **Check if you have transcripts** for your WAV files
   - If YES → Format them as described above
   - If NO → You need to:
     - Use ASR to generate transcripts, OR
     - Use different dataset with transcripts, OR
     - Create synthetic data with TTS

2. **Create metadata.json** for each conversation
   - conversation_id
   - turn_number (0-indexed)
   - Links to WAV + transcript files

3. **Run preprocessing pipeline** to create training samples

Would you like me to:
1. **Create a preprocessing script** for your specific data format?
2. **Show how to generate transcripts** from WAV using ASR?
3. **Help set up a synthetic data generation pipeline** with TTS?
