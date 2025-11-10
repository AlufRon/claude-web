# 📊 LibriLight Individual File Selector

## Overview

The dashboard now allows you to **selectively display individual LibriLight evaluation files** when viewing the long-context performance chart in "Individual Files" mode.

## Why This is Useful

Each checkpoint is evaluated on **3 separate LibriLight audio files** (~1 hour each). Previously, when you selected "Individual Files" mode, all 3 files for all selected checkpoints were shown, which could clutter the chart with 9+ lines.

Now you can **choose which files to display** (e.g., only File 1, or Files 1 and 3), making the chart much clearer when comparing specific file behavior across checkpoints.

## How to Use

### Step 1: Select Checkpoints
Select 2 or more checkpoints from the overview table (the LibriLight chart section will appear below).

### Step 2: Change View Mode to "Individual Files"
In the LibriLight section controls, change **View** to **"Individual Files"**.

### Step 3: Select Which Files to Display
A new **"Files:"** multi-select dropdown will appear. By default, all 3 files are selected.

**To select specific files:**
- **On Windows/Mac**: Hold `Ctrl` (Windows) or `Cmd` (Mac) and click to select/deselect files
- **Select multiple**: Click and drag, or Ctrl/Cmd+Click multiple options
- **Select all**: Click File 1, then Shift+Click File 3

### Step 4: View the Chart
The chart will automatically update to show only the selected files.

## Example Use Cases

### Use Case 1: Compare File 1 Across All Checkpoints
**Goal**: See how different training configurations perform on the same audio file.

1. Select all checkpoints you want to compare
2. Set View to "Individual Files"
3. In the Files selector, select **only File 1**
4. Chart now shows only File 1 for each checkpoint (much clearer!)

### Use Case 2: Check Consistency Across Files
**Goal**: See if a checkpoint performs consistently across different files.

1. Select a single checkpoint
2. Set View to "Individual Files"
3. Select all 3 files (default)
4. Compare the 3 lines - they should be similar if the model is consistent

### Use Case 3: Focus on Specific File Behavior
**Goal**: One file shows interesting behavior, want to zoom in.

1. Select checkpoints showing interesting behavior on File 2
2. Set View to "Individual Files"
3. Select **only File 2**
4. Analyze the specific patterns on this file

## Visual Guide

### Before (All Files Shown - Cluttered)
```
Checkpoint A - File 1 ————————
Checkpoint A - File 2 - - - - -
Checkpoint A - File 3 ·········
Checkpoint B - File 1 ————————
Checkpoint B - File 2 - - - - -
Checkpoint B - File 3 ·········
Baseline - File 1 ————————
Baseline - File 2 - - - - -
Baseline - File 3 ·········
(9 lines - hard to read!)
```

### After (Only File 1 Selected - Clear)
```
Checkpoint A - File 1 ————————
Checkpoint B - File 1 ————————
Baseline - File 1 ————————
(3 lines - easy to compare!)
```

## Technical Details

### File Names
The 3 LibriLight evaluation files are typically:
1. **File 1**: `1hour_speaker100_emerald_city_librivox_64kb_mp3_chapters1-5.flac`
2. **File 2**: Another 1-hour audio file
3. **File 3**: Another 1-hour audio file

Each is ~1 hour long and evaluated up to ~43k tokens.

### Data Structure
The dashboard uses the `librilight_progression_individual` field in the data:
```json
"librilight_progression_individual": [
  [ /* File 1 data points */ ],
  [ /* File 2 data points */ ],
  [ /* File 3 data points */ ]
]
```

### Line Styling
- **Solid line**: File 1
- **Dashed line (short)**: File 2
- **Dashed line (long)**: File 3
- **Baseline files**: Gray with varying dash patterns

## Keyboard Controls

- **Ctrl/Cmd + Click**: Toggle single file selection
- **Shift + Click**: Select range of files
- **Click**: Select only that file (deselects others)

## Tips

1. **Start with all files** to see overall behavior
2. **Narrow down** to specific files when you find interesting patterns
3. **Compare same file** across checkpoints for clearest comparison
4. **Check baseline** - select "Show Baseline" to overlay baseline performance for the same files

## FAQ

**Q: Can I select no files?**
A: No, at least one file must be selected. The chart needs data to display.

**Q: Does this work with baseline?**
A: Yes! When "Show Baseline" is checked and view mode is "Individual Files", the baseline will also respect the file selector.

**Q: What if a checkpoint doesn't have all 3 files?**
A: The selector always shows 3 files, but only files with actual data will be displayed on the chart.

**Q: Can I see this in other view modes (Mean, Median)?**
A: No, the file selector only appears in "Individual Files" mode. The other modes aggregate data across all files.

---

**Updated:** 2025-10-21
