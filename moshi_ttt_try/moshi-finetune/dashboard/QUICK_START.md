# 📊 Paper Metrics Dashboard - Quick Start Guide

## 🎉 Dashboard is Ready!

The Paper Metrics Dashboard has been successfully created and is ready to use.

## 📂 Location

```bash
cd /home/alufr/ttt_tests/moshi-finetune/dashboard
```

## 🚀 Quick Start (3 Steps)

### Step 1: Generate/Update Dashboard Data

```bash
# Option A: Use the convenience script
./update_dashboard.sh

# Option B: Run aggregation manually
python aggregate_paper_metrics.py --output dashboard_data.json
```

### Step 2: Open Dashboard in Browser

```bash
# Open in Firefox
firefox dashboard.html &

# Or Chrome
google-chrome dashboard.html &

# Or just double-click dashboard.html in your file browser
```

### Step 3: Explore!

- **Filter** runs by TTT, LoRA, training steps, etc.
- **Sort** by any metric (click column headers)
- **Compare** checkpoints (select 2+ with checkboxes)
- **Visualize** long-context performance
- **Export** to CSV for further analysis

## 📁 Files Created

```
dashboard/
├── aggregate_paper_metrics.py    # Data aggregation script (16KB)
├── dashboard.html                 # Main dashboard (8KB)
├── dashboard.js                   # Interactive logic (23KB)
├── dashboard.css                  # Styling (8KB)
├── dashboard_data.json            # Sample data (6KB) - UPDATE THIS!
├── update_dashboard.sh            # Automation script (2KB)
├── README.md                      # Full documentation (9KB)
└── QUICK_START.md                 # This file
```

## 🔄 Regular Usage Workflow

### Automatic Updates ✅

**Good news!** The dashboard updates automatically when you run paper metrics evaluations!

When you run:
```bash
python run_paper_metrics_on_checkpoint.py --checkpoint /path/to/checkpoint
```

The dashboard will automatically update after the evaluation completes. Just refresh your browser!

### Manual Update (if needed)

If you want to force a manual update:

```bash
cd /home/alufr/ttt_tests/moshi-finetune/dashboard
./update_dashboard.sh
```

Then refresh your browser to see the new results!

## 📊 Current Status

✅ **Aggregation Script**: Successfully tested with real checkpoint data
✅ **Dashboard HTML**: Created with full interactive features
✅ **Styling**: Modern, clean CSS design
✅ **JavaScript**: Complete filtering, sorting, and visualization logic
✅ **Sample Data**: Working example with 3 runs (TTT, LoRA, Baseline)

## 🎯 What the Dashboard Provides

### Overview Table
- **Sortable columns**: Click any header to sort
- **Key metrics**: sBLIMP, sWUGGY, tStory, sStory, LibriLight PPL
- **Configuration info**: TTT enabled, LoRA enabled
- **Selection**: Checkboxes for comparison

### Filtering System
- TTT Enabled (Yes/No/All)
- LoRA Enabled (Yes/No/All)
- Training Type (ttt, lora, full, baseline)
- Min Training Steps
- Data Source (daily-talk, librilight, etc.)

### Comparison View
- **Radar Chart**: Multi-metric comparison across benchmarks
- **Bar Chart**: Side-by-side benchmark scores
- **Config Diff**: Highlighted differences in training configuration

### LibriLight Long-Context Chart
- Line chart showing perplexity vs token position (1k-43k)
- Multiple checkpoints overlaid
- Optional baseline comparison
- Interactive zoom/pan

### Statistics Panel
- Best performers for each metric
- Quick insights at a glance

## 💡 Tips

1. **First Time**: Run `./update_dashboard.sh` to scan all checkpoints and generate fresh data

2. **Large Scans**: If you have many checkpoints, the initial scan might take a few minutes. Subsequent updates are faster.

3. **Specific Directories**: Target specific experiment directories for faster scans:
   ```bash
   python aggregate_paper_metrics.py \
       --checkpoint-dirs /path/to/experiment1 /path/to/experiment2 \
       --output dashboard_data.json
   ```

4. **Log Parsing**: For LibriLight position-wise data, ensure `paper_metrics_*.log` files are in the log directory

5. **Export**: Use "Export CSV" button to get data for Excel/LaTeX tables

## 🛠️ Customization

Want to add custom metrics or modify the dashboard?

1. **New Metrics**: Edit `aggregate_paper_metrics.py` to extract additional fields
2. **New Charts**: Modify `dashboard.js` to add visualizations
3. **New Filters**: Update both HTML and JS to add filter options
4. **Styling**: Edit `dashboard.css` to change colors/layout

## 📚 Documentation

- **README.md**: Complete documentation with troubleshooting
- **DASHBOARD_PLAN.md**: Original design and architecture
- **Comments**: Extensive inline comments in all code files

## 🎓 Example Commands

### Scan All Checkpoints
```bash
python aggregate_paper_metrics.py \
    --checkpoint-dirs /sise/eliyanac-group/ron_al \
    --output dashboard_data.json
```

### Scan Specific Experiments
```bash
python aggregate_paper_metrics.py \
    --checkpoint-dirs \
        /sise/eliyanac-group/ron_al/dailytalk_finetune_from_librilight6 \
        /sise/eliyanac-group/ron_al/dailytalk_finetune_from_librilight7 \
    --output dashboard_data.json
```

### Quick Update
```bash
./update_dashboard.sh
```

## ✅ Verification

Test that everything works:

```bash
# 1. Check files exist
ls -lh

# 2. Test aggregation on one directory
python aggregate_paper_metrics.py \
    --checkpoint-dirs /sise/eliyanac-group/ron_al/dailytalk_finetune_from_librilight11 \
    --output test.json

# 3. Check output
cat test.json | head -20

# 4. Open dashboard
firefox dashboard.html &
```

## 🐛 Troubleshooting

### "Failed to load dashboard data"
- Make sure `dashboard_data.json` exists
- Run `./update_dashboard.sh` to generate it

### "No runs match the current filters"
- Click "Reset Filters" button
- Or check that your filters aren't too restrictive

### Charts not showing
- Ensure internet connection (loads Chart.js from CDN)
- Check browser console (F12) for errors

### Aggregation is slow
- Normal for first run (scanning large directories)
- Use specific `--checkpoint-dirs` to speed up

## 🎉 Next Steps

1. **Generate real data**: Run `./update_dashboard.sh`
2. **Open dashboard**: `firefox dashboard.html &`
3. **Explore your results**: Filter, compare, export!
4. **Iterate**: After new evaluations, just re-run the update script

---

**Enjoy your new dashboard! 📊**

For detailed documentation, see **README.md**
