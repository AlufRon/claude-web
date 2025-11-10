# 📊 Paper Metrics Dashboard - Implementation Plan

## 🎯 Project Overview

**Goal:** Create an interactive HTML dashboard that automatically aggregates, visualizes, and compares paper metrics results from all evaluation runs.

**Key Requirements:**
1. Automatic aggregation of new paper metrics runs
2. Interactive filtering by configuration parameters (TTT enabled, LoRA, learning rate, etc.)
3. Visual comparison with charts and graphs
4. Checkpoint selection for head-to-head comparison
5. Historical tracking of all runs
6. Zero maintenance - updates automatically with new runs

---

## 📋 Current State Analysis

### What We Have

1. **Paper Metrics Script:** `/home/alufr/ttt_tests/moshi-finetune/run_paper_metrics_on_checkpoint.py`
   - Evaluates checkpoints on multiple benchmarks
   - Saves results to `paper_metrics_results.json` in checkpoint directory

2. **Results Format:**
```json
{
  "sblimp_accuracy": 0.542,
  "sblimp_samples": 1000,
  "swuggy_accuracy": 0.640,
  "swuggy_samples": 2000,
  "tstory_accuracy": 0.797,
  "tstory_samples": 1871,
  "sstory_accuracy": 0.613,
  "sstory_samples": 1871,
  "librilight_perplexity_8k": 702.27,
  "librilight_perplexity_16k": 1318.30,
  "librilight_perplexity_24k": 0.0,
  "librilight_slope": 0.000073,
  "librilight_samples": 42998,
  "paper_metrics_avg": 0.648
}
```

3. **Training Configuration:**
   - Stored in `checkpoint_*/consolidated/training_config.json`
   - Contains all training parameters:
     - TTT settings (enable, layers, base_lr, mini_batch_size, etc.)
     - LoRA settings (enable, rank, scaling)
     - Optimizer settings (lr, weight_decay)
     - Training duration (max_steps)
     - Full vs adapter finetuning
     - And more...

4. **Log Files:**
   - 86 paper metrics log files with detailed evaluation outputs
   - Contains position-wise LibriLight perplexity progression

---

## 🏗️ Architecture Design

### Option A: Static Site Generator (RECOMMENDED) ⭐

**Technology Stack:**
- **Backend:** Python script to aggregate data
- **Frontend:** Pure HTML + JavaScript (no build step)
- **Charts:** Chart.js or Plotly.js
- **Data:** Single JSON file with all runs
- **Hosting:** Local file system (open HTML in browser)

**Pros:**
- ✅ Simple: No server, no database, no dependencies
- ✅ Fast: Loads instantly, runs locally
- ✅ Portable: Single HTML file + data.json
- ✅ Maintainable: Easy to understand and modify
- ✅ Works offline: No network required

**Cons:**
- ⚠️ Re-generate HTML when data changes
- ⚠️ Limited real-time updates (but we don't need them)

### Option B: Web Application with Backend

**Technology Stack:**
- **Backend:** Flask/FastAPI server
- **Frontend:** React/Vue.js
- **Database:** SQLite
- **Charts:** Recharts/Plotly

**Pros:**
- ✅ Real-time updates
- ✅ More interactive features

**Cons:**
- ❌ Requires server running
- ❌ More complex setup
- ❌ Harder to maintain
- ❌ Overkill for this use case

### **Decision: Option A (Static Site Generator)** ✅

---

## 📊 Dashboard Features

### Core Features (Phase 1)

1. **Overview Table**
   - All runs in sortable table
   - Key metrics: sBLIMP, sWUGGY, tStoryCloze, sStoryCloze, LibriLight PPL
   - Configuration columns: TTT enabled, layers, base LR, steps trained, LoRA
   - Color coding: Best/worst performers

2. **Checkpoint Comparison**
   - Select 2-5 checkpoints to compare side-by-side
   - Radar chart for all metrics
   - Bar charts for each benchmark
   - Configuration diff view

3. **Filtering System**
   - Filter by TTT enabled/disabled
   - Filter by TTT layers (e.g., only 29,30,31)
   - Filter by training steps (e.g., > 1000 steps)
   - Filter by LoRA enabled/disabled
   - Filter by learning rate range

4. **LibriLight Long-Context Visualization**
   - Line chart: Perplexity vs position (1k to 43k tokens)
   - Multiple checkpoints overlaid
   - Baseline comparison
   - Zoom/pan capabilities

### Advanced Features (Phase 2)

5. **Trend Analysis**
   - Performance vs training steps
   - Performance vs TTT base LR
   - Degradation rate analysis

6. **Statistical Insights**
   - Best checkpoint for each metric
   - Pareto frontier (optimal trade-offs)
   - Correlation analysis (e.g., base_lr vs performance)

7. **Export Capabilities**
   - Export filtered data as CSV
   - Export charts as PNG/SVG
   - Generate LaTeX tables for papers

---

## 🔧 Implementation Plan

### Phase 1: Data Aggregation Script (Week 1)

**File:** `aggregate_paper_metrics.py`

**Functionality:**
1. Scan all checkpoint directories for `paper_metrics_results.json`
2. Load corresponding `training_config.json`
3. Extract key parameters from training config
4. Parse log files for LibriLight position-wise data (if needed)
5. Aggregate everything into single `dashboard_data.json`

**Output Structure:**
```json
{
  "runs": [
    {
      "id": "cp_2000_dailytalk_librilight6",
      "checkpoint_path": "/sise/eliyanac-group/.../checkpoint_002000",
      "timestamp": "2025-10-18T00:39:26",
      "training_config": {
        "max_steps": 2000,
        "lr": 5e-05,
        "duration_sec": 200.0,
        "first_codebook_weight_multiplier": 100.0,
        "ttt": {
          "enable": true,
          "layers": "29,30,31",
          "base_lr": 0.01,
          "mini_batch_size": 64,
          "ttt_mlp_layers": 5,
          "initial_gating_alpha": 0.001
        },
        "lora": {
          "enable": false
        },
        "full_finetuning": false
      },
      "metrics": {
        "sblimp_accuracy": 0.534,
        "swuggy_accuracy": 0.638,
        "tstory_accuracy": 0.795,
        "sstory_accuracy": 0.612,
        "librilight_perplexity_8k": 485.30,
        "librilight_perplexity_16k": 1318.35,
        "librilight_slope": 0.000073,
        "paper_metrics_avg": 0.645
      },
      "librilight_progression": [
        {"position": 1000, "loss": 5.61, "perplexity": 272},
        {"position": 2000, "loss": 5.47, "perplexity": 237},
        ...
      ]
    }
  ],
  "metadata": {
    "last_updated": "2025-10-18T12:00:00",
    "total_runs": 15,
    "baseline_run_id": "baseline_moshiko"
  }
}
```

**Implementation Steps:**
```python
# 1. Discovery
def find_all_checkpoints(base_dirs):
    """Find all checkpoint directories with paper_metrics_results.json"""

# 2. Extract config
def extract_training_config(checkpoint_dir):
    """Load and parse training_config.json"""

# 3. Load metrics
def load_paper_metrics(checkpoint_dir):
    """Load paper_metrics_results.json"""

# 4. Parse logs (optional)
def parse_librilight_logs(log_file):
    """Extract position-wise perplexity from logs"""

# 5. Aggregate
def aggregate_all_runs():
    """Combine everything into dashboard_data.json"""
```

### Phase 2: HTML Dashboard (Week 1-2)

**File:** `dashboard.html`

**Structure:**
```html
<!DOCTYPE html>
<html>
<head>
    <title>Paper Metrics Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/lodash@4.17.21"></script>
    <style>/* Modern, clean CSS */</style>
</head>
<body>
    <!-- Header with title and last updated -->
    <header>...</header>

    <!-- Filter Panel -->
    <section id="filters">
        <div class="filter-group">
            <label>TTT Enabled:</label>
            <select id="filter-ttt">...</select>
        </div>
        <!-- More filters... -->
    </section>

    <!-- Overview Table -->
    <section id="overview">
        <table id="results-table">...</table>
    </section>

    <!-- Comparison View -->
    <section id="comparison">
        <div id="checkpoint-selector">...</div>
        <div id="comparison-charts">...</div>
    </section>

    <!-- LibriLight Visualization -->
    <section id="librilight">
        <canvas id="librilight-chart"></canvas>
    </section>

    <script src="dashboard.js"></script>
</body>
</html>
```

**JavaScript (`dashboard.js`):**
```javascript
// 1. Load data
async function loadDashboardData() {
    const response = await fetch('dashboard_data.json');
    return await response.json();
}

// 2. Filter runs
function filterRuns(runs, filters) {
    return runs.filter(run => {
        if (filters.ttt !== 'all' && run.training_config.ttt.enable !== filters.ttt) return false;
        // More filters...
        return true;
    });
}

// 3. Render table
function renderOverviewTable(runs) {
    // Sort, format, and display runs in table
}

// 4. Render comparison
function renderComparison(selectedRuns) {
    // Radar chart, bar charts, config diff
}

// 5. Render LibriLight chart
function renderLibriLightChart(selectedRuns) {
    // Line chart with multiple runs
}

// 6. Event handlers
document.getElementById('filter-ttt').addEventListener('change', applyFilters);
```

### Phase 3: Automation Script (Week 2)

**File:** `update_dashboard.sh`

**Purpose:** Run after each new paper metrics evaluation

```bash
#!/bin/bash
# Update dashboard with new runs

echo "🔄 Updating Paper Metrics Dashboard..."

# 1. Run aggregation script
python aggregate_paper_metrics.py \
    --checkpoint-dirs "/sise/eliyanac-group/ron_al/*/checkpoints" \
    --output dashboard_data.json

# 2. Copy to web-accessible location (optional)
cp dashboard.html dashboard_data.json /path/to/web/server/

echo "✅ Dashboard updated!"
echo "📊 Open dashboard.html in browser to view"
```

### Phase 4: Integration with Existing Workflow (Week 2)

**Modify:** `run_paper_metrics_on_checkpoint.py`

Add at the end:
```python
# After saving paper_metrics_results.json
logger.info("📊 Updating dashboard...")
subprocess.run(["python", "aggregate_paper_metrics.py", "--incremental"])
logger.info("✅ Dashboard updated!")
```

---

## 🎨 UI/UX Design

### Color Scheme
- **Primary:** #4A90E2 (Blue)
- **Success:** #7ED321 (Green) - Best performers
- **Warning:** #F5A623 (Orange) - Middle performers
- **Danger:** #D0021B (Red) - Worst performers
- **Neutral:** #F8F9FA (Light gray background)

### Layout
```
┌─────────────────────────────────────────────────────────────┐
│                  Paper Metrics Dashboard                     │
│              Last Updated: 2025-10-18 12:00                  │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│  Filters:  [TTT: All ▾] [LoRA: All ▾] [Steps: All ▾] ...   │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│  Overview Table (15 runs)                        [Export ▾]  │
├────┬────────┬────────┬────────┬────────┬────────┬──────────┤
│ ☑  │Checkpoint│sBLIMP│sWUGGY│tStory│ sStory│LibriLight│
├────┼────────┼────────┼────────┼────────┼────────┼──────────┤
│ ☑  │CP 2000 │ 0.534 │ 0.638 │ 0.795 │ 0.612 │  485.30  │
│ ☐  │CP 1200 │ 0.542 │ 0.640 │ 0.797 │ 0.613 │  463.21  │
│ ☐  │Baseline│ 0.500 │ 0.600 │ 0.750 │ 0.550 │  712.17  │
└────┴────────┴────────┴────────┴────────┴────────┴──────────┘
┌─────────────────────────────────────────────────────────────┐
│  Comparison View (2 selected)                [Add more ▾]   │
│  ┌─────────────────┐  ┌──────────────────────────────────┐ │
│  │   Radar Chart   │  │       Bar Charts                 │ │
│  │                 │  │  sBLIMP:  ▓▓▓▓▓▓░░  vs ▓▓▓▓░░░░ │ │
│  │    sBLIMP       │  │  sWUGGY:  ▓▓▓▓▓░░░  vs ▓▓▓▓▓░░░ │ │
│  │   ↗     ↖       │  │  ...                             │ │
│  │  sStory tStory  │  │                                  │ │
│  └─────────────────┘  └──────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│  LibriLight Long-Context Performance                         │
│                                                              │
│  Perplexity                                                  │
│  2500 ┤                                            ╭────     │
│  2000 ┤                                    ╭──────╯         │
│  1500 ┤                            ╭──────╯                 │
│  1000 ┤                    ╭──────╯                         │
│   500 ┤           ╭───────╯                                 │
│     0 └───────────────────────────────────────────────────  │
│       0    10k   20k   30k   40k  Tokens                    │
│                                                              │
│  Legend: ─ CP2000  ─ CP1200  ─ Baseline                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 📦 Technology Stack Final Selection

### Python (Data Aggregation)
- **Standard Library:** `json`, `pathlib`, `glob`, `argparse`
- **External:** None needed (keep it simple!)

### Frontend
- **HTML5:** Semantic markup
- **CSS3:** Modern styling with Flexbox/Grid
- **JavaScript (ES6+):** Vanilla JS, no framework needed
- **Chart.js 4.x:** Beautiful, responsive charts
- **DataTables.js:** Interactive, sortable tables

### Why These Choices?
1. **No build step** - Just open HTML in browser
2. **No dependencies** - Everything via CDN
3. **Fast loading** - Single page, minimal assets
4. **Works offline** - Download CDN files if needed
5. **Easy to modify** - Standard web tech, no magic
6. **Future-proof** - Will work for years without updates

---

## 📁 File Structure

```
moshi-finetune/
├── dashboard/
│   ├── aggregate_paper_metrics.py     # Data aggregation script
│   ├── dashboard.html                 # Main dashboard file
│   ├── dashboard.js                   # Dashboard logic
│   ├── dashboard.css                  # Styles
│   ├── dashboard_data.json            # Aggregated data (generated)
│   ├── update_dashboard.sh            # Automation script
│   └── README.md                      # Dashboard usage guide
└── DASHBOARD_PLAN.md                  # This file
```

---

## 🚀 Development Phases

### Phase 1: MVP (1-2 days)
**Goal:** Basic working dashboard

- ✅ Write `aggregate_paper_metrics.py`
- ✅ Create basic HTML table showing all runs
- ✅ Implement simple filtering (TTT yes/no)
- ✅ Add basic sorting

**Deliverable:** Can view and filter paper metrics results

### Phase 2: Visualization (2-3 days)
**Goal:** Interactive charts

- ✅ Add Chart.js integration
- ✅ Implement LibriLight line chart
- ✅ Add radar chart for comparison
- ✅ Add checkpoint selector

**Deliverable:** Can compare checkpoints visually

### Phase 3: Advanced Features (2-3 days)
**Goal:** Full-featured dashboard

- ✅ Advanced filtering (multi-parameter)
- ✅ Configuration diff view
- ✅ Export capabilities
- ✅ Responsive design

**Deliverable:** Production-ready dashboard

### Phase 4: Integration (1 day)
**Goal:** Seamless workflow

- ✅ Integrate with paper metrics script
- ✅ Add automation script
- ✅ Write documentation

**Deliverable:** Fully automated system

---

## 📝 Data Schema

### Dashboard Data Format

```json
{
  "schema_version": "1.0",
  "metadata": {
    "generated_at": "2025-10-18T12:00:00Z",
    "total_runs": 15,
    "checkpoint_base_paths": [
      "/sise/eliyanac-group/ron_al/dailytalk_finetune_from_librilight6",
      "/sise/eliyanac-group/ron_al/dailytalk_finetune_from_librilight7"
    ]
  },
  "baseline": {
    "id": "baseline_moshiko",
    "name": "Baseline (No TTT)",
    "metrics": { ... }
  },
  "runs": [
    {
      "id": "unique_run_id",
      "name": "CP2000 - DailyTalk TTT",
      "checkpoint_path": "/path/to/checkpoint",
      "checkpoint_step": 2000,
      "evaluation_timestamp": "2025-10-18T00:39:26Z",

      "training_config": {
        "training_type": "ttt",  // "ttt", "lora", "full", "baseline"
        "max_steps": 2000,
        "training_duration_sec": 200.0,
        "batch_size": 2,
        "learning_rate": 5e-05,
        "weight_decay": 0.1,
        "first_codebook_weight_multiplier": 100.0,

        "ttt": {
          "enabled": true,
          "layers": "29,30,31",
          "num_layers": 3,
          "base_lr": 0.01,
          "mini_batch_size": 64,
          "mlp_layers": 5,
          "mlp_expansion": 4.0,
          "initial_gating_alpha": 0.001,
          "persistent_states": true
        },

        "lora": {
          "enabled": false,
          "rank": null,
          "scaling": null
        },

        "full_finetuning": false
      },

      "metrics": {
        "sblimp": {
          "accuracy": 0.534,
          "samples": 1000
        },
        "swuggy": {
          "accuracy": 0.638,
          "samples": 2000
        },
        "tstory": {
          "accuracy": 0.795,
          "samples": 1871
        },
        "sstory": {
          "accuracy": 0.612,
          "samples": 1871
        },
        "librilight": {
          "perplexity_8k": 485.30,
          "perplexity_16k": 1318.35,
          "perplexity_24k": 2104.80,
          "slope": 0.000073,
          "samples": 42998
        },
        "average": 0.645
      },

      "librilight_progression": [
        {"position": 1000, "loss": 5.6064, "perplexity": 272.17},
        {"position": 5000, "loss": 5.6116, "perplexity": 273.57},
        {"position": 10000, "loss": 6.5543, "perplexity": 702.27},
        {"position": 20000, "loss": 7.4481, "perplexity": 1716.57},
        {"position": 42000, "loss": 7.6515, "perplexity": 2103.80}
      ],

      "tags": ["ttt", "daily-talk", "3-layers", "high-lr"],
      "notes": "Training from LibriLight pretrained weights"
    }
  ]
}
```

---

## 🔍 Example Queries

The dashboard will support queries like:

1. **"Show me all TTT runs with 3 layers"**
   - Filter: `ttt.enabled=true AND ttt.num_layers=3`

2. **"Compare best checkpoint vs baseline"**
   - Select: `max(paper_metrics_avg)` and `baseline`

3. **"Find checkpoints with best long-context performance"**
   - Sort by: `librilight.perplexity_24k` (ascending)

4. **"Show LoRA vs TTT performance"**
   - Filter: `training_type IN ['lora', 'ttt']`
   - Group by: `training_type`

5. **"Which learning rate works best for TTT?"**
   - Filter: `ttt.enabled=true`
   - X-axis: `ttt.base_lr`
   - Y-axis: `metrics.average`

---

## 🎓 Success Criteria

### Must Have (MVP)
- ✅ View all paper metrics runs in one place
- ✅ Filter by TTT enabled/disabled
- ✅ Compare 2+ checkpoints side-by-side
- ✅ Visualize LibriLight progression
- ✅ Automatic data updates

### Should Have (V1)
- ✅ Advanced filtering (multi-parameter)
- ✅ Sortable columns
- ✅ Radar charts for multi-metric comparison
- ✅ Export filtered data
- ✅ Configuration diff view

### Nice to Have (V2)
- ⭐ Trend analysis over training
- ⭐ Statistical insights
- ⭐ LaTeX table export
- ⭐ Mobile-responsive design
- ⭐ Dark mode

---

## 🚦 Next Steps

1. **Create directory structure**
   ```bash
   mkdir -p dashboard
   cd dashboard
   ```

2. **Start with aggregation script**
   - Begin with simple version that just finds all results
   - Gradually add parsing logic
   - Test with existing checkpoints

3. **Build HTML prototype**
   - Start with static data (hardcoded JSON)
   - Get layout and design right
   - Add interactivity

4. **Connect data to UI**
   - Load from `dashboard_data.json`
   - Implement filters
   - Add charts

5. **Polish and document**
   - Add README
   - Create usage examples
   - Test with real data

---

## 📚 Resources

### Documentation to Write
1. **README.md** - Dashboard usage guide
2. **AGGREGATION_GUIDE.md** - How data aggregation works
3. **CUSTOMIZATION.md** - How to add new metrics/filters

### Code Examples
1. **Example filter:** Show only TTT runs with > 1000 steps
2. **Example comparison:** CP2000 vs Baseline
3. **Example export:** CSV for LaTeX table

### Testing
1. **Test with existing 15+ runs** from logs
2. **Test with new run** (simulate adding checkpoint)
3. **Test filtering** edge cases
4. **Test on different browsers**

---

## 🎯 Timeline

**Total Estimated Time:** 1-2 weeks

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Phase 1: MVP | 1-2 days | Basic dashboard |
| Phase 2: Visualization | 2-3 days | Charts & comparison |
| Phase 3: Advanced | 2-3 days | Full features |
| Phase 4: Integration | 1 day | Automated workflow |
| Testing & Polish | 1-2 days | Production ready |

---

## 💡 Key Design Decisions

### Why Static HTML?
- ✅ Simplest solution that meets requirements
- ✅ No server maintenance
- ✅ Fast and reliable
- ✅ Easy to share (just send file)
- ✅ Works anywhere (laptop, server, USB stick)

### Why Chart.js?
- ✅ Beautiful default styling
- ✅ Responsive out of the box
- ✅ Good documentation
- ✅ Active maintenance
- ✅ Supports all chart types we need

### Why Single JSON File?
- ✅ Simple data model
- ✅ Fast loading (< 1MB for 100 runs)
- ✅ Easy to version control
- ✅ Human-readable
- ✅ Can be queried with `jq` if needed

---

## 🔒 Potential Issues & Solutions

### Issue 1: Large JSON File
**Problem:** With 100+ runs, JSON file gets big (> 5MB)

**Solutions:**
1. Pagination in UI (load data lazily)
2. Compress JSON (gzip, serve with HTTP compression)
3. Split into multiple files (by experiment series)
4. Move old runs to archive

### Issue 2: Log Parsing Complexity
**Problem:** Extracting LibriLight progression from logs

**Solutions:**
1. Make it optional (just use final metrics)
2. Save progression data during evaluation
3. Use regex patterns for robust parsing
4. Cache parsed results

### Issue 3: Config Format Changes
**Problem:** Training config schema evolves

**Solutions:**
1. Schema versioning
2. Fallback defaults for missing fields
3. Config migration script
4. Validate configs on load

---

## ✅ Conclusion

This plan provides a **complete, pragmatic solution** for visualizing and comparing paper metrics results.

**Key Strengths:**
- 🎯 Simple: No server, no build step, no complex dependencies
- 🚀 Fast: Static HTML loads instantly
- 🔧 Maintainable: Standard web tech, easy to modify
- 🎨 Flexible: Easy to add new metrics/filters
- 🤖 Automated: Updates automatically with new runs

**Next Action:** Start implementing Phase 1 (aggregation script)!
