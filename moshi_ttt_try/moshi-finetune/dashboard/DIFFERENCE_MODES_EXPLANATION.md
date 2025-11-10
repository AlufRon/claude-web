# Dashboard Difference Modes Explanation

The dashboard now supports **three different ways** to compare checkpoint performance against the baseline.

## Baseline Reference

All comparisons use the **pretrained Moshi model** as the baseline:
- **Model**: `kyutai/moshiko-pytorch-bf16`
- **sBLIMP**: 54.50%
- **sWUGGY**: 63.35%
- **tStory**: 80.06%
- **sStory**: 60.93%

## Three Display Modes

### 1. **Absolute Values** (Default)
Shows the raw accuracy scores.

**Example:**
- Baseline sWUGGY: 63.35%
- Checkpoint sWUGGY: 66.50%
- **Display**: 0.665 (66.50%)

**Use when:** You want to see the actual performance numbers.

---

### 2. **% Difference (Relative)**
Shows the **relative change** as a percentage of the baseline value.

**Formula:** `((checkpoint - baseline) / baseline) × 100`

**Example:**
- Baseline sWUGGY: 63.35% (0.6335)
- Checkpoint sWUGGY: 66.50% (0.665)
- Difference: (0.665 - 0.6335) / 0.6335 × 100
- **Display**: +4.97% (5% relative improvement)

**Use when:** You want to see **proportional improvement** relative to baseline performance.

**⚠️ Warning:** This can be confusing! A "5% improvement" means the model is 5% better **relative to the baseline**, not that accuracy increased by 5 percentage points.

---

### 3. **Difference (Points)** ⭐ **NEW - Less Confusing!**
Shows the **absolute difference** in percentage points.

**Formula:** `(checkpoint - baseline) × 100`

**Example:**
- Baseline sWUGGY: 63.35% (0.6335)
- Checkpoint sWUGGY: 66.50% (0.665)
- Difference: (0.665 - 0.6335) × 100
- **Display**: +3.15 pts (3.15 percentage point improvement)

**Use when:** You want to see the **actual improvement** in percentage points.

**✅ Recommended:** This is the clearest way to understand improvement!

---

## Comparison Example

Let's say:
- **Baseline sWUGGY**: 63.35%
- **Checkpoint sWUGGY**: 66.50%

| Mode | Display | Meaning |
|------|---------|---------|
| **Absolute Values** | 66.50% | The checkpoint achieved 66.50% accuracy |
| **% Difference** | +4.97% | The checkpoint is 4.97% better **relative to baseline** |
| **Difference (Points)** | +3.15 pts | The checkpoint is 3.15 **percentage points** better |

## Which Mode to Use?

### For Research Papers
Use **Difference (Points)** - it's standard to report:
> "We achieved an improvement of 3.15 percentage points on sWUGGY (63.35% → 66.50%)"

### For Quick Comparison
Use **Absolute Values** - you can directly see which model has higher accuracy.

### For Relative Performance
Use **% Difference** - useful for understanding proportional gains, but be careful with interpretation!

---

## How to Change Modes in Dashboard

In the **Checkpoint Comparison** section:
1. Select 2+ checkpoints from the table
2. Find the **"Difference Type"** dropdown (top right of comparison section)
3. Choose:
   - **Absolute Values** - Raw accuracy scores
   - **% Difference (Relative)** - Proportional improvement
   - **Difference (Points)** - Percentage point improvement ⭐ Recommended

The bar chart will update automatically to show the selected mode.

---

## Mathematical Formulas

### % Difference (Relative)
```
relative_diff = ((checkpoint_value - baseline_value) / baseline_value) × 100
```

### Difference (Points)
```
point_diff = (checkpoint_value - baseline_value) × 100
```

### Example Calculation
Given:
- Baseline = 0.6335 (63.35%)
- Checkpoint = 0.665 (66.50%)

**% Difference:**
```
((0.665 - 0.6335) / 0.6335) × 100 = 4.97%
```

**Difference (Points):**
```
(0.665 - 0.6335) × 100 = 3.15 pts
```

---

**Updated:** 2025-10-21
