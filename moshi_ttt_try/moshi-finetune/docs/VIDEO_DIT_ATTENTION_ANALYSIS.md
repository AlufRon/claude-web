# Video-DiT Attention Context Analysis

**Date**: 2025-11-01
**Source**: `/home/alufr/ttt_tests/papers/ttt_dit_video_paper.txt`
**Purpose**: Verify exactly how Video-DiT uses attention context during training vs inference

---

## Executive Summary

**Confirmed**: Video-DiT uses **LOCAL attention limited to 3-second segments** during **BOTH training and inference** for long videos. Attention is deliberately prevented from seeing the full sequence - this forces TTT layers to carry information across segment boundaries.

---

## Key Evidence from Paper

### 1. Explicit Statement on Local vs Global (Lines 365-373)

> "**Local attention, global TTT**. CogVideo-X uses self-attention layers to process the entire input sequence globally for each video of maximum length 3 seconds, but global attention becomes inefficient for long videos. **To avoid increasing the context length of self-attention layers, we make them local to each 3-second segment, attending to each of the n sequence segments independently.** The TTT layers process the entire input sequence globally because they are efficient in long context."

**Analysis**:
- "make them local to each 3-second segment" - attention is restricted
- "attending to each of the n sequence segments independently" - segments are isolated
- "TTT layers process the entire input sequence globally" - only TTT sees full context

### 2. Applies to Both Training and Inference (Lines 312-317)

> "In this subsection, we discuss how to create the input sequence of tokens to our architecture and how each sequence is processed in segments. **Except for the first two text formats in the upcoming discussion, everything applies to both fine-tuning and inference.**"

**Analysis**:
- The local attention mechanism described in Section 3.2 applies to BOTH training and inference
- Not just an inference optimization - this is how they TRAIN the model
- Exception only applies to text format conversion (Format 1→2→3), not attention mechanism

### 3. Architecture Design (Line 256-258, Figure 3)

Figure 3 caption explicitly shows:
> "Right: Our overall pipeline creates input sequences composed of 3-second segments. This structure enables us to **apply self-attention layers locally over segments and TTT layers globally over the entire sequence.**"

### 4. Segment Structure (Lines 318-328)

> "We structure our videos to contain multiple scenes, and each scene contains one or more **3-second segments. We use a 3-second segment as the atomic unit** of text-to-video pairing for three reasons:
> - The maximum length of generation for the original pre-trained CogVideo-X is 3 seconds.
> - The length of most scenes in the Tom and Jerry episodes is at least 3 seconds.
> - Building a dataset with multiple stages (Subsection 3.3) is most convenient given 3-second segments."

**Analysis**:
- 3 seconds is a hard architectural boundary
- Not arbitrary - matches pre-trained model's max length
- Designed around this constraint from the start

---

## Training Progression (Multi-Stage)

From Section 3.3 (Lines 375-392) and Table 2 (Lines 986-994):

| Stage | Video Length | Total Context (tokens) | Attention Context | TTT Context |
|-------|-------------|----------------------|------------------|-------------|
| 1 | 3 seconds | 18,048 | Full (3s) | Full (3s) |
| 2 | 9 seconds | 51,456 | Local (3s segments) | Full (9s) |
| 3 | 18 seconds | 99,894 | Local (3s segments) | Full (18s) |
| 4 | 30 seconds | 168,320 | Local (3s segments) | Full (30s) |
| 5 | 63 seconds | 341,550 | Local (3s segments) | Full (63s) |

**Critical Observations**:

1. **Stage 1 (3 seconds)**: Attention sees full sequence because sequence = 1 segment
   - Fine-tune entire pre-trained model
   - Adapt to Tom & Jerry domain

2. **Stages 2-5 (9s → 63s)**: Attention is LOCAL, TTT is GLOBAL
   - 63-second video = **21 segments** of 3 seconds each
   - Each segment's attention: ~18,000 tokens (3 seconds)
   - Attention **cannot** cross segment boundaries
   - TTT connects all 21 segments = 341,550 tokens total

3. **Line 375-378**: "Following standard practice for LLMs [51], we extend the context length"
   - They extend TTT's effective context
   - Attention remains local (3s segments)

---

## Comparison: Video-DiT vs Our Moshi-TTT Approach

### Video-DiT (Segmented Local Attention)

**63-second video example**:
```
Segment 1 (0-3s):   Attn sees tokens 0-18048     ✓
                    TTT accumulates 0-18048      ✓

Segment 2 (3-6s):   Attn sees tokens 18048-36096 ✓ (ISOLATED from Seg 1)
                    TTT accumulates 0-36096      ✓ (carries Seg 1 info)

Segment 21 (60-63s): Attn sees tokens 323502-341550 ✓ (ISOLATED)
                     TTT accumulated 0-341550       ✓ (carries ALL prior info)
```

**Key Design**:
- **Hard segment boundaries** - attention CANNOT cross
- **Forced dependency** on TTT for cross-segment information
- **Ratio**: Attention sees 3s / 63s = **4.8%** of full sequence at any moment

### Our Moshi-TTT (Ring Buffer Attention)

**80-second sequence example (context=100)**:
```
Position 500 (40s): Attn sees tokens 400-500 (last 100) ✓
                    TTT accumulated 0-500              ✓

Position 1000 (80s): Attn sees tokens 900-1000 ✓
                     TTT accumulated 0-1000     ✓
```

**Key Design**:
- **Rolling window** - attention sees last N tokens continuously
- **Gradual dependency** on TTT for old information
- **Ratio**: Attention sees 100 / 1000 = **10%** of sequence at any moment

---

## Architectural Differences

### 1. Boundary Sharpness

**Video-DiT**:
- Sharp 3-second boundaries
- Attention goes from "sees everything" to "sees nothing" at segment edge
- Clear point where TTT MUST bridge gap

**Moshi-TTT**:
- Smooth rolling window
- Old tokens gradually age out of attention
- TTT must maintain compressed representation continuously

### 2. Training Signal Strength

**Video-DiT**:
- Strong signal: attention literally CANNOT see across segments
- Model MUST use TTT or lose cross-segment information
- Binary: use TTT or fail

**Moshi-TTT**:
- Softer signal: attention loses old tokens gradually
- Model can partially rely on attention overlap
- Gradient: more useful to use TTT for older information

### 3. Context Coverage

**Video-DiT Attention Coverage**:
- 3s per segment × 21 segments = sees 3s at a time
- Coverage ratio: 3 / 63 = **4.8%**
- More aggressive than our approach

**Moshi-TTT Attention Coverage**:
- 8s window in 80s sequence
- Coverage ratio: 8 / 80 = **10%**
- Less aggressive - might be too generous

---

## Critical Insights for Our Implementation

### 1. We Are LESS Restrictive Than Video-DiT

Our `ttt_layer_context: 100` (8 seconds) gives attention **MORE visibility** than Video-DiT's 3-second segments:

```
Video-DiT:   4.8% coverage (3s / 63s)
Our setup:   10% coverage  (8s / 80s)
```

**Implication**: Our TTT might not be forced to work as hard during training.

### 2. Segmented vs Rolling Window

**Video-DiT** benefits from hard boundaries:
- Clear training signal at segment boundaries
- Forces model to explicitly learn cross-boundary coherence
- Works well for video (natural scene boundaries)

**Moshi-TTT** uses continuous rolling:
- More natural for audio (no hard scene boundaries)
- Smoother but potentially weaker training signal
- May need smaller context to compensate

### 3. Video-DiT's Success Suggests Aggressive Is Better

From Table 1 (Lines 556-561):
- TTT-MLP beats baselines by **34 Elo points**
- Biggest wins: temporal consistency (+38), motion naturalness (+39)
- These are exactly the metrics that require long-range modeling

**Hypothesis**: Restricting attention MORE (not less) forces TTT to learn better long-range representations.

---

## Recommendations for Moshi-TTT

### Option 1: Match Video-DiT's Absolute Segment Size (3 seconds)

```yaml
duration_sec: 80
ttt:
  ttt_layer_context: 38        # 3 seconds at 12.5 Hz
  non_ttt_layer_context: 100   # Keep non-TTT with more context
```

**Pros**: Direct analogy to proven Video-DiT approach
**Cons**: Very aggressive, might be too restrictive for audio

### Option 2: Match Video-DiT's Coverage Ratio (4.8%)

```yaml
duration_sec: 80
ttt:
  ttt_layer_context: 48        # 4.8% of 1000 tokens
  non_ttt_layer_context: 100
```

**Pros**: Same "division of labor" ratio as Video-DiT
**Cons**: Still quite aggressive

### Option 3: Conservative Middle Ground (6-7 seconds)

```yaml
duration_sec: 80
ttt:
  ttt_layer_context: 75-88     # 6-7 seconds
  non_ttt_layer_context: 100
```

**Pros**: Halfway between current (8s) and Video-DiT (3s)
**Cons**: Less theoretically grounded

### Option 4: Experiment with Segmented Attention (Like Video-DiT)

Modify our code to use **hard segment boundaries** instead of rolling window:
- Split 80s audio into segments (e.g., 8 × 10s segments)
- Attention confined to segment
- TTT bridges segments

**Pros**: Closer to proven Video-DiT architecture
**Cons**: Requires code changes, might not suit audio well

---

## Current Config Assessment

Looking at [dailytalk_finetune_from_librilight.yaml:64-65](../example/dailytalk_finetune_from_librilight.yaml):

```yaml
ttt_layer_context: 100       # 8 seconds - DOUBLE Video-DiT's 3-4 second coverage
non_ttt_layer_context: 100   # Same as TTT layers
```

**Analysis**:
1. ✅ We DO restrict attention (not using full 3000)
2. ⚠️ But we're LESS restrictive than Video-DiT (10% vs 4.8%)
3. ❌ No differentiation between TTT and non-TTT layers (both 100)
4. ⚠️ Might not force TTT hard enough during training

**Suggested Immediate Change**:

```yaml
ttt_layer_context: 50        # 4 seconds - closer to Video-DiT
non_ttt_layer_context: 100   # Keep non-TTT with broader context
```

This creates:
- TTT layers: 50/1000 = **5% coverage** (matches Video-DiT ~4.8%)
- Non-TTT layers: 100/1000 = **10% coverage** (more context for layers without TTT)
- Clear division of labor between layer types

---

## Conclusion

**Question**: Does Video-DiT let attention see the whole sequence during training?

**Answer**: **NO** - definitively confirmed by:
1. Explicit statement (lines 365-373): "local to each 3-second segment"
2. Applies to training (lines 312-317): "both fine-tuning and inference"
3. Architecture design (Figure 3, Table 2): segmented structure throughout
4. Results (Table 1): 34 Elo improvement proves this approach works

**Our Approach**:
- Currently LESS restrictive than Video-DiT (10% vs 4.8% coverage)
- Could benefit from reducing `ttt_layer_context` to 38-50 tokens (3-4 seconds)
- Should differentiate TTT layer context from non-TTT layer context

**Next Steps**:
1. Consider reducing `ttt_layer_context` to match Video-DiT's ratio
2. Maintain larger context for non-TTT layers
3. Test if more aggressive restriction improves long-range modeling
