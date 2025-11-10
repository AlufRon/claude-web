# 📊 Paper Metrics Dashboard

**Interactive web-based dashboard for visualizing and comparing TTT-Moshi checkpoint evaluation results.**

## 🎯 Overview

The Paper Metrics Dashboard provides a comprehensive interface for analyzing checkpoint performance across multiple benchmarks. It automatically aggregates evaluation results from multiple checkpoints, tracks training configurations, and visualizes long-context performance with interactive charts.

### Key Features

- 📋 **Overview Table**: Sortable, filterable table of all evaluation runs
- 📊 **Interactive Charts**: Radar charts, bar charts, and line plots for comparison
- 🔍 **Advanced Filtering**: Filter by TTT/LoRA settings, training steps, data source
- 📈 **Long-Context Analysis**: LibriLight perplexity progression (up to 43k tokens)
- 📉 **Statistical Insights**: Best performers, trends, configuration differences
- 💾 **CSV Export**: Export filtered results for further analysis
- 🚀 **Auto-Update**: Automatically updates when new evaluations complete
- 🪟 **Windows-Compatible**: Standalone HTML file for offline viewing

---

## 🚀 Quick Start

### 1. Generate Dashboard Data

The dashboard uses a **runs registry** for fast updates. Data aggregation happens automatically when you run evaluations.

#### Automatic Update (Recommended)
When you run paper metrics evaluation, the dashboard updates automatically:

```bash
python evaluation/scripts/run_paper_metrics_on_checkpoint.py \
    --checkpoint /path/to/checkpoint/consolidated
```

After evaluation completes, you'll see:
```
💾 Results saved to: /path/to/checkpoint/consolidated/paper_metrics_results.json

📊 Updating dashboard...
✅ Dashboard updated successfully!
   View at: /home/alufr/ttt_tests/moshi-finetune/dashboard/dashboard.html
```

#### Manual Update
Force a manual update if needed:

```bash
cd /home/alufr/ttt_tests/moshi-finetune/dashboard
./update_dashboard.sh
```

This will:
1. Aggregate all registered runs from `runs_registry.json`
2. Generate `dashboard_data.json`
3. Create `dashboard_standalone.html` for Windows/offline use

### 2. View Dashboard

#### Option A: On Linux Server (with browser)
```bash
cd /home/alufr/ttt_tests/moshi-finetune/dashboard

# Open in Firefox
firefox dashboard.html &

# Or Chrome
google-chrome dashboard.html &
```

#### Option B: On Windows/Offline
1. Download **only** `dashboard_standalone.html` (single file, ~50-300KB)
2. Double-click to open in any browser
3. All data, CSS, and JavaScript are embedded!

**Note**: Internet connection required for Chart.js CDN (alternatively, can be made fully offline)

---

## 📁 Directory Structure

```
dashboard/
├── aggregate_paper_metrics.py      # Data aggregation script (main engine)
├── dashboard.html                   # Main dashboard HTML (requires server/fetch)
├── dashboard.js                     # Interactive JavaScript logic (~900 lines)
├── dashboard.css                    # Styling and theme (~470 lines)
├── dashboard_data.json              # Aggregated data (generated)
├── dashboard_data_with_progression.json  # Extended data with LibriLight progression
├── dashboard_standalone.html        # Single-file version (generated)
├── runs_registry.json               # Fast lookup registry (auto-updated)
├── update_dashboard.sh              # Convenience update script
├── create_standalone.sh             # Generate standalone HTML
├── test_dashboard.sh                # Test script
├── README.md                        # This file
├── QUICK_START.md                   # Quick reference guide
├── WINDOWS_USAGE.md                 # Windows-specific instructions
└── INTEGRATION_COMPLETE.md          # Implementation details
```

---

## 🔧 How It Works

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  Checkpoint Evaluation                                          │
│  (run_paper_metrics_on_checkpoint.py)                           │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 │ 1. Saves results
                 │    paper_metrics_results.json
                 │
                 │ 2. Registers in runs_registry.json
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│  Dashboard Aggregation                                          │
│  (aggregate_paper_metrics.py)                                   │
│                                                                 │
│  • Reads runs_registry.json (FAST - no filesystem scan)        │
│  • Loads paper_metrics_results.json for each run               │
│  • Loads training_config.json for each checkpoint              │
│  • Parses paper_metrics_*.log for LibriLight progression       │
│  • Aggregates into dashboard_data.json                         │
│  • Generates dashboard_standalone.html                         │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│  Dashboard UI                                                   │
│  (dashboard.html + dashboard.js + dashboard.css)                │
│                                                                 │
│  • Loads dashboard_data.json (or embedded in standalone)       │
│  • Renders interactive tables and charts                       │
│  • Filtering, sorting, comparison                              │
│  • CSV export                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Runs Registry System

The dashboard uses a **registry-based approach** for fast updates:

1. **Registration**: When you run paper metrics, the checkpoint path and results file are added to `runs_registry.json`
2. **Fast Lookup**: Aggregation reads the registry instead of scanning entire directory trees
3. **Incremental**: Only loads data for registered checkpoints
4. **Fallback**: If registry is missing, falls back to full directory scan

**Registry Format** (`runs_registry.json`):
```json
{
  "runs": [
    {
      "checkpoint_path": "/path/to/checkpoint/consolidated",
      "results_file": "/path/to/checkpoint/consolidated/paper_metrics_results.json",
      "added_at": "2025-10-19T13:43:09.135452"
    }
  ]
}
```

### Data Schema

**Input Files Per Checkpoint**:
- `paper_metrics_results.json` - Evaluation metrics (sBLIMP, sWUGGY, tStory, etc.)
- `training_config.json` - Training configuration (TTT settings, LoRA, optimizer, etc.)
- `paper_metrics_*.log` - Optional: Position-wise perplexity data

**Output File** (`dashboard_data.json`):
```json
{
  "schema_version": "1.0",
  "metadata": {
    "generated_at": "2025-10-20T12:00:00",
    "total_runs": 15,
    "checkpoint_base_paths": ["/sise/eliyanac-group/ron_al"]
  },
  "baseline": {
    "id": "baseline_pretrained",
    "name": "Baseline - kyutai/moshiko-pytorch-bf16",
    "metrics": { ... }
  },
  "runs": [
    {
      "id": "dailytalk_finetune_cp002000",
      "name": "CP2000 - experiment_name",
      "checkpoint_path": "/path/to/checkpoint",
      "checkpoint_step": 2000,
      "evaluation_timestamp": "2025-10-18T00:39:26",
      "training_config": {
        "max_steps": 2000,
        "optim": { "lr": 5e-05, ... },
        "ttt": {
          "enable": true,
          "layers": "29,30,31",
          "base_lr": 0.01,
          "mini_batch_size": 8,
          ...
        },
        "lora": { "enable": false, ... }
      },
      "metrics": {
        "sblimp_accuracy": 0.534,
        "swuggy_accuracy": 0.638,
        "tstory_accuracy": 0.795,
        "sstory_accuracy": 0.612,
        "librilight_perplexity_8k": 485.30,
        "librilight_perplexity_16k": 1318.35,
        "librilight_perplexity_24k": 2104.80,
        "librilight_slope": 0.000073,
        "paper_metrics_avg": 0.645
      },
      "librilight_progression": [
        { "position": 1000, "loss": 5.61, "perplexity": 272.17 },
        { "position": 5000, "loss": 5.61, "perplexity": 273.57 },
        ...
      ],
      "librilight_progression_mean": [ ... ],
      "librilight_progression_median": [ ... ],
      "librilight_progression_individual": [ [...], [...], [...] ],
      "tags": ["ttt", "daily-talk", "high-lr", "long-training"],
      "notes": "TTT training on layers 29,30,31 | Base LR: 0.01 | ..."
    }
  ]
}
```

---

## 📊 Dashboard Features

### 1. Overview Table

**Features**:
- ✅ Sortable columns (click headers to sort)
- ✅ Checkbox selection for comparison
- ✅ Key metrics displayed: sBLIMP, sWUGGY, tStory, sStory, LibriLight PPL
- ✅ Training configuration badges (TTT enabled, LoRA enabled)

**Columns**:
- **Name**: Checkpoint identifier (e.g., "CP2000 - experiment_name")
- **Step**: Training checkpoint step number
- **sBLIMP**: Syntactic evaluation accuracy
- **sWUGGY**: Phonotactic well-formedness accuracy
- **tStory**: Text-based story continuation accuracy
- **sStory**: Speech-based story continuation accuracy
- **PPL 8k/16k**: LibriLight perplexity at 8k/16k tokens
- **Avg**: Average of speech metrics (paper_metrics_avg)
- **TTT/LoRA**: Configuration badges

### 2. Filtering System

**Available Filters**:
- **TTT Enabled**: Yes/No/All
- **LoRA Enabled**: Yes/No/All
- **Training Type**: ttt/lora/full/baseline/all
- **Min Training Steps**: Numeric threshold
- **Data Source**: daily-talk/librilight/librispeech/all

**Auto-Generated Tags**:
Tags are automatically extracted from training config for filtering:
- `ttt`, `lora`, `full`, `baseline` - Training type
- `ttt-layers-29-30-31` - Which layers have TTT
- `high-lr`, `medium-lr`, `low-lr` - TTT learning rate range
- `daily-talk`, `librilight`, `librispeech` - Data source
- `long-training`, `medium-training`, `short-training` - Duration category

### 3. Comparison Charts

Select 2+ checkpoints to enable comparison view.

#### Radar Chart
- **Multi-metric comparison** across benchmarks
- Larger polygon area = better overall performance
- Visual pattern matching for strengths/weaknesses
- Auto-scale option to zoom into differences

#### Bar Chart
- **Side-by-side benchmark comparison**
- Color-coded by metric
- Optional: Show **% difference from baseline**
- Auto-scale for focused comparison

**Controls**:
- ☑️ **Auto-scale Y-axis**: Zoom to differences (instead of 0-1.0 range)
- 🔄 **Show % Difference**: Toggle between absolute values and % change vs baseline

#### Configuration Diff
- **Highlights differences** in training configuration
- Side-by-side comparison of key parameters:
  - `max_steps`, `optim.lr`
  - `ttt.enable`, `ttt.layers`, `ttt.base_lr`, `ttt.mini_batch_size`
  - `lora.enable`, `lora.rank`
  - `full_finetuning`
- Yellow highlighting for differing values

### 4. LibriLight Long-Context Visualization

**Chart**: Perplexity vs Token Position (1k - 43k tokens)

**Purpose**: Evaluate how models handle long audio sequences
- Lower perplexity = better
- Flatter curve = better long-context retention
- Rising curve = performance degradation over time

**View Modes**:
- **Mean**: Averaged across all evaluation files (with optional error bars)
- **Median**: Median perplexity at each position
- **Individual Files**: Separate line for each evaluation file
- **All Points (Raw)**: All data points overlaid

**Data Structure**:
Each checkpoint is evaluated on 3 LibriLight files. The dashboard provides:
- `librilight_progression`: Raw data (all samples)
- `librilight_progression_mean`: Position-wise mean with std dev
- `librilight_progression_median`: Position-wise median
- `librilight_progression_individual`: Separated by file (3 arrays)

**Controls**:
- ☑️ **Show Baseline**: Overlay baseline performance
- 📊 **View Mode**: Choose aggregation method
- ☑️ **Show Error Bars**: Display standard deviation (mean mode only)

### 5. Statistics Panel

**Displays**:
- 🏆 Best sBLIMP score and checkpoint
- 🏆 Best sWUGGY score and checkpoint
- 🏆 Best tStory score and checkpoint
- 🏆 Best Overall (paper_metrics_avg) and checkpoint

**Auto-updates** based on current filters.

### 6. CSV Export

Export filtered results for:
- Excel/Google Sheets analysis
- LaTeX table generation
- Sharing with collaborators

**Exported Columns**:
Name, Step, sBLIMP, sWUGGY, tStory, sStory, PPL 8k, PPL 16k, Avg, TTT, LoRA

---

## 🛠️ Usage Guide

### Basic Workflow

1. **Run evaluations** on checkpoints (automatic dashboard update)
2. **Open dashboard** in browser (`firefox dashboard.html`)
3. **Apply filters** to focus on relevant experiments
4. **Select checkpoints** to compare (click checkboxes)
5. **Analyze charts** (radar, bar, long-context)
6. **Export CSV** if needed

### Example: Finding Best TTT Configuration

```
1. Filter: TTT Enabled = Yes
2. Filter: Min Training Steps = 1000
3. Sort by: Avg (click column header twice for descending)
4. Select top 3-5 performers
5. Review:
   - Configuration Diff: What settings differ?
   - LibriLight Chart: Long-context handling
   - Bar Chart: Per-benchmark comparison
```

### Example: Comparing Training Types

```
1. Reset Filters
2. Select:
   - One baseline checkpoint
   - One TTT checkpoint
   - One LoRA checkpoint
   - One full finetuning checkpoint
3. Review Radar Chart: Overall performance comparison
4. Enable "Show % Difference": See relative improvements
5. Review LibriLight Chart: Long-context behavior
```

---

## 🔄 Updating the Dashboard

### Automatic Updates ✅

**Dashboard updates automatically** when you run paper metrics evaluations!

When you execute:
```bash
python evaluation/scripts/run_paper_metrics_on_checkpoint.py \
    --checkpoint /path/to/checkpoint
```

The script automatically:
1. Saves results to `paper_metrics_results.json`
2. Registers run in `runs_registry.json`
3. Calls dashboard aggregation
4. Updates `dashboard_data.json` and `dashboard_standalone.html`

Just **refresh your browser** to see new results!

### Manual Update

Force a manual refresh if needed:

```bash
cd /home/alufr/ttt_tests/moshi-finetune/dashboard

# Option A: Use convenience script
./update_dashboard.sh

# Option B: Run aggregation manually
python aggregate_paper_metrics.py --output dashboard_data.json

# Option C: Create standalone only (after aggregation)
./create_standalone.sh
```

### Adding Baseline Results

To add a baseline (pretrained model evaluation):

1. Run paper metrics on pretrained Moshi (without training)
2. Save results as `baseline_paper_metrics_results.json` in dashboard directory
3. Register in `runs_registry.json`:
   ```json
   {
     "checkpoint_path": "None",
     "results_file": "baseline_paper_metrics_results.json",
     "added_at": "2025-10-19T13:43:09"
   }
   ```
4. Update dashboard: `./update_dashboard.sh`

---

## 💻 Advanced Usage

### Custom Checkpoint Directories

Specify custom directories for aggregation:

```bash
python aggregate_paper_metrics.py \
    --checkpoint-dirs /custom/path1 /custom/path2 \
    --log-dir /path/to/logs \
    --output dashboard_data.json
```

### Custom Metrics

To add new metrics to the dashboard:

1. **Update `aggregate_paper_metrics.py`**:
   ```python
   @dataclass
   class MetricsData:
       # Add your metric
       custom_metric: float = 0.0
   ```

2. **Extract metric** in `load_paper_metrics()`:
   ```python
   custom_metric = metrics.get('custom_metric', 0.0)
   ```

3. **Update `dashboard.js`** to display in table/charts:
   ```javascript
   // Add column in renderTable()
   <td>${this.formatMetric(run.metrics?.custom_metric)}</td>
   ```

4. **Update `dashboard.html`** table header:
   ```html
   <th class="sortable" data-column="metrics.custom_metric">Custom</th>
   ```

### Custom Tags

Add custom filtering tags:

1. **Edit `aggregate_paper_metrics.py`**:
   ```python
   def extract_tags(self, config, metrics):
       tags = []
       # ... existing tags ...

       # Add custom tag
       if config.get('custom_setting', False):
           tags.append('custom-tag')

       return tags
   ```

2. **Update `dashboard.html`** filter dropdown:
   ```html
   <option value="custom-tag">Custom Experiments</option>
   ```

3. Re-run aggregation

---

## 🪟 Windows Usage

### Standalone Version (Recommended)

The standalone version embeds all data, CSS, and JavaScript in a **single HTML file**.

**Advantages**:
- ✅ Single file download
- ✅ No CORS issues
- ✅ Works offline (except Chart.js CDN)
- ✅ Easy sharing

**How to Use**:

1. **On Linux Server**:
   ```bash
   cd /home/alufr/ttt_tests/moshi-finetune/dashboard
   ./update_dashboard.sh  # Generates dashboard_standalone.html
   ```

2. **Download** `dashboard_standalone.html` to Windows (via SCP/WinSCP/etc.)

3. **Open** in any browser (Chrome, Firefox, Edge)

**File Size**: Varies by number of runs (typically 50KB - 500KB)

### Multi-File Version

If you need the multi-file version on Windows:

1. Download all 4 files:
   - `dashboard.html`
   - `dashboard.js`
   - `dashboard.css`
   - `dashboard_data.json`

2. **Option A**: Run local web server
   ```bash
   python -m http.server 8000
   # Then open: http://localhost:8000/dashboard.html
   ```

3. **Option B**: Use browser that allows `file://` JSON loading (not recommended due to security)

**Recommendation**: Use standalone version - much simpler!

---

## 🐛 Troubleshooting

### Dashboard shows "Failed to load dashboard data"

**Cause**: Missing or invalid `dashboard_data.json`

**Solution**:
```bash
cd /home/alufr/ttt_tests/moshi-finetune/dashboard

# Check if file exists
ls -lh dashboard_data.json

# If missing, regenerate
./update_dashboard.sh
```

### "No checkpoints found"

**Cause**: Aggregation script can't find `paper_metrics_results.json` files

**Solutions**:

1. **Check if runs are registered**:
   ```bash
   cat runs_registry.json
   ```

2. **Verify results files exist**:
   ```bash
   find /sise/eliyanac-group/ron_al -name "paper_metrics_results.json" | head -5
   ```

3. **Force full directory scan** (if registry is corrupted):
   ```bash
   rm runs_registry.json
   ./update_dashboard.sh
   ```

### LibriLight chart is empty

**Cause**: No position-wise perplexity data available

**Explanation**: LibriLight progression requires log file parsing. If log files are missing or not in expected location, progression data won't be available.

**Check**:
```bash
ls -lh /home/alufr/ttt_tests/moshi-finetune/paper_metrics_*.log
```

**Note**: Summary metrics (PPL at 8k, 16k, 24k) are still available in the table even without progression data.

### Charts not rendering

**Cause**: Chart.js library not loading

**Solutions**:

1. **Check internet connection** (Chart.js loads from CDN)
2. **Check browser console** (F12 → Console) for errors
3. **Try different browser**
4. **Verify CDN URL** is accessible:
   ```
   https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js
   ```

### CORS error on Windows

**Cause**: Browsers block local JSON file loading for security

**Solution**: Use `dashboard_standalone.html` instead of `dashboard.html`

### Slow aggregation

**Cause**: Large directory tree scan (only happens if registry is missing)

**Solutions**:

1. **Use registry-based approach** (default - fast!)
2. **Specify precise paths**:
   ```bash
   python aggregate_paper_metrics.py \
       --checkpoint-dirs /sise/eliyanac-group/ron_al/specific_experiment
   ```
3. **First scan is slow, subsequent updates are fast** (uses registry)

### Wrong baseline shown

**Cause**: Multiple baseline entries in registry

**Solution**: Edit `runs_registry.json` and `dashboard_data.json` to ensure only one baseline entry exists.

---

## 📚 Technical Details

### Technology Stack

**Backend (Data Aggregation)**:
- Python 3.8+ (standard library only)
- Dependencies: None! (uses only stdlib: `json`, `pathlib`, `re`, `datetime`, `dataclasses`)

**Frontend (Dashboard UI)**:
- HTML5
- CSS3 (Flexbox, Grid, modern features)
- Vanilla JavaScript (ES6+)
- Chart.js 4.4.0 (via CDN)
- Lodash 4.17.21 (via CDN)

**No Build Tools Required**: Just open the HTML file!

### Performance

**Registry-Based Aggregation**:
- **Fast**: 10-100x faster than full directory scan
- **Scalable**: Handles hundreds of checkpoints
- **Incremental**: Only loads data for registered runs

**Full Directory Scan** (fallback):
- Searches entire directory tree
- ~10-60 seconds for large directories
- Only runs if registry is missing

**Dashboard UI**:
- Client-side filtering/sorting (instant)
- Handles 100+ runs smoothly
- Chart rendering: <1 second

### Browser Compatibility

**Tested On**:
- ✅ Chrome 90+
- ✅ Firefox 88+
- ✅ Edge 90+
- ✅ Safari 14+

**Requirements**:
- JavaScript enabled
- Internet connection (for Chart.js CDN)
- Modern ES6+ support

### Security Considerations

**Data Privacy**:
- All data is local (no external API calls except CDN)
- No user tracking or analytics
- Safe for confidential research data

**File Access**:
- Dashboard only reads JSON files
- No write operations from UI
- No code execution from data files

---

## 🎓 Examples

### Example 1: Comparing TTT Layer Configurations

**Goal**: Find optimal number of TTT layers

```bash
# 1. Filter to TTT runs
Filter: TTT Enabled = Yes

# 2. Sort by average metric
Click: "Avg" column header (twice for descending)

# 3. Select top performers
Select checkpoints with:
  - ttt-layers-all
  - ttt-layers-middle
  - ttt-layers-29-30-31
  - ttt-layers-31

# 4. Compare
Review:
  - Radar Chart: Overall performance shape
  - Config Diff: Confirm layer differences
  - LibriLight Chart: Long-context behavior
  - Bar Chart: Individual benchmarks
```

**Insights**:
- More layers doesn't always = better performance
- Check long-context chart for degradation
- Balance performance vs. computational cost

### Example 2: TTT vs LoRA vs Full Finetuning

**Goal**: Compare different finetuning approaches

```bash
# 1. Reset filters
Click: "Reset Filters"

# 2. Select one of each type
Select:
  - 1 baseline (Training Type = baseline)
  - 1 TTT (Training Type = ttt)
  - 1 LoRA (Training Type = lora)
  - 1 Full (Training Type = full)

# 3. Enable percentage difference
Click: "Show % Difference" button

# 4. Analyze
Review:
  - Bar Chart: Which approach improves most over baseline?
  - Radar Chart: Which metrics improve for each approach?
  - LibriLight Chart: Long-context handling comparison
```

**Insights**:
- TTT may excel at long-context tasks
- LoRA may improve specific benchmarks
- Full finetuning provides general improvement but costs more

### Example 3: Training Duration Analysis

**Goal**: Determine optimal training steps

```bash
# 1. Filter to single experiment series
Filter: TTT Enabled = Yes
Filter: Data Source = daily-talk

# 2. Select checkpoints at different steps
Select checkpoints at steps: 100, 500, 1000, 1600, 2000

# 3. Sort by step
Click: "Step" column header

# 4. Analyze progression
Review:
  - Bar Chart: Performance vs training time
  - Radar Chart: When do metrics plateau?
  - Statistics: When is best performance achieved?
```

**Insights**:
- Identify early stopping point
- Check for overfitting (performance decrease)
- Balance training cost vs. performance gain

---

## 📖 File Descriptions

### Core Files

**`aggregate_paper_metrics.py`** (707 lines)
- Main aggregation engine
- Scans checkpoints (registry-based or full scan)
- Loads metrics, configs, and log data
- Extracts tags and metadata
- Aggregates LibriLight progression data (raw, mean, median, individual)
- Generates `dashboard_data.json`

**`dashboard.html`** (220 lines)
- Main HTML structure
- Filter panel
- Overview table
- Comparison section
- LibriLight chart section
- Statistics panel
- Loads Chart.js and Lodash from CDN
- Links to `dashboard.css` and `dashboard.js`

**`dashboard.js`** (~900 lines)
- `PaperMetricsDashboard` class
- Data loading (JSON or embedded)
- Filtering and sorting logic
- Table rendering
- Chart rendering (radar, bar, line)
- Config diff generation
- CSV export
- Event handlers

**`dashboard.css`** (~470 lines)
- Modern responsive design
- Color scheme and theming
- Table styling
- Chart containers
- Filter panel layout
- Button and badge styles
- Responsive breakpoints

### Generated Files

**`dashboard_data.json`**
- Aggregated data from all registered runs
- Generated by `aggregate_paper_metrics.py`
- Size: ~50KB - 500KB (depends on number of runs and progression data)

**`dashboard_standalone.html`**
- Single-file version with embedded data, CSS, and JS
- Generated by `create_standalone.sh`
- Size: ~50KB - 500KB
- Perfect for Windows/offline usage

### Registry

**`runs_registry.json`**
- Fast lookup table for checkpoints
- Auto-updated by evaluation script
- Format: List of `{checkpoint_path, results_file, added_at}`
- Enables fast aggregation without directory scans

### Automation Scripts

**`update_dashboard.sh`** (82 lines)
- Convenience script for full update workflow
- Runs aggregation + standalone generation
- Usage: `./update_dashboard.sh`

**`create_standalone.sh`** (90 lines)
- Generates standalone HTML file
- Embeds data, CSS, and JS into single file
- Usage: `./create_standalone.sh`

**`test_dashboard.sh`**
- Test script for development

### Documentation

**`README.md`** (this file)
- Complete documentation
- Usage guide and troubleshooting

**`QUICK_START.md`**
- Quick reference guide
- Essential commands and workflows

**`WINDOWS_USAGE.md`**
- Windows-specific instructions
- Standalone version guide

**`INTEGRATION_COMPLETE.md`**
- Implementation details
- Architecture documentation

---

## 🚀 Future Enhancements

Potential improvements (not yet implemented):

### Features
- [ ] **Incremental updates**: Only scan for new checkpoints (registry enables this)
- [ ] **Trend analysis**: Performance vs training steps over time (line plots)
- [ ] **Statistical insights**: Correlation analysis, confidence intervals
- [ ] **Pareto frontier**: Multi-objective optimization visualization
- [ ] **LaTeX export**: Auto-generate publication-ready tables
- [ ] **Notebook integration**: Jupyter widget version

### UI/UX
- [ ] **Dark mode**: Toggle for dark theme
- [ ] **Mobile responsive**: Optimize for mobile viewing
- [ ] **Keyboard shortcuts**: Power user features
- [ ] **Save filter presets**: Reusable filter configurations
- [ ] **Annotations**: Add notes to specific runs

### Technical
- [ ] **Fully offline**: Embed Chart.js (no CDN dependency)
- [ ] **Compression**: Gzip data for smaller files
- [ ] **Database backend**: SQLite for very large datasets
- [ ] **Real-time updates**: WebSocket-based live updates
- [ ] **API endpoint**: REST API for programmatic access

---

## 📄 License

Part of the moshi-finetune project. See project license for details.

---

## 🙏 Acknowledgments

Built for the **TTT-Moshi** project to facilitate checkpoint analysis and comparison.

**Key Technologies**:
- [Chart.js](https://www.chartjs.org/) - Beautiful interactive charts
- [Lodash](https://lodash.com/) - Utility functions
- Python standard library - Data aggregation

---

## 📞 Support

**For Issues**:
1. Check this README (especially Troubleshooting section)
2. Review `QUICK_START.md` for quick solutions
3. Check browser console (F12) for errors
4. Verify data files exist and are valid JSON

**For Questions**:
- See `INTEGRATION_COMPLETE.md` for implementation details
- Check code comments in Python/JS files
- Review example workflows above

---

**Happy Analyzing! 📊**

*Last Updated: 2025-10-21*

---

## 🆕 Understanding Difference Modes (Updated 2025-10-21)

The dashboard now supports **three ways** to compare checkpoint performance:

### Quick Reference

| Mode | What it Shows | Example |
|------|---------------|---------|
| **Absolute Values** | Raw accuracy scores | 66.50% |
| **% Difference (Relative)** | Proportional improvement | +4.97% |
| **Difference (Points)** ⭐ | Percentage point improvement | +3.15 pts |

### Example Scenario
If baseline sWUGGY = 63.35% and checkpoint sWUGGY = 66.50%:

- **Absolute**: Shows 66.50% (the actual accuracy)
- **% Difference**: Shows +4.97% (the checkpoint is ~5% better **relative to** baseline)
- **Points**: Shows +3.15 pts (the checkpoint is 3.15 **percentage points** better)

### Which to Use?

- **Research Papers**: Use "Difference (Points)" - clearest and standard in academia
- **Quick Glance**: Use "Absolute Values" - see actual performance
- **Relative Gains**: Use "% Difference" - understand proportional improvement

**⚠️ Important:** "5% improvement" in relative mode means 5% better than baseline, NOT 5 percentage points!

See `DIFFERENCE_MODES_EXPLANATION.md` for detailed examples and mathematical formulas.

