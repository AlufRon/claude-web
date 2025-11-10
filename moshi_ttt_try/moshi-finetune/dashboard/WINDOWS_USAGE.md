# 🪟 Using Dashboard on Windows - SOLVED!

## ✅ The Solution: Standalone HTML File

The issue you encountered is that browsers block loading local JSON files for security reasons.

**Solution**: Use the **standalone version** that has everything embedded in a single HTML file!

## 📥 Download Instructions

### Option 1: Single File (EASIEST) ⭐

**Download only this file:**
```
dashboard_standalone.html
```

**Location on server:**
```
/home/alufr/ttt_tests/moshi-finetune/dashboard/dashboard_standalone.html
```

**File size:** ~46 KB

Then:
1. Save it anywhere on Windows
2. Double-click to open in your browser
3. Done! All plots and data are visible

### Option 2: Multi-File Version

If you already downloaded all 4 files but got the error, you have two choices:

**A) Use the standalone file instead** (recommended - see Option 1 above)

**B) Run a local web server:**
```bash
# In the folder with the dashboard files:
python -m http.server 8000

# Then open in browser:
http://localhost:8000/dashboard.html
```

## 🔄 Updating the Standalone File

When you have new paper metrics results, regenerate the standalone file:

```bash
cd /home/alufr/ttt_tests/moshi-finetune/dashboard

# Re-create standalone with latest data
./create_standalone.sh

# Download the updated dashboard_standalone.html to Windows
```

## 📊 What You'll See

Once opened, you'll see:

### 1. Overview Table
- All checkpoint results in a sortable table
- Click column headers to sort
- Checkboxes to select for comparison

### 2. Filters (Top Section)
- TTT Enabled: Yes/No/All
- LoRA Enabled: Yes/No/All
- Training Type: ttt/lora/full/baseline
- Min Training Steps
- Data Source

### 3. Comparison Charts (Select 2+ checkpoints)
- **Radar Chart**: Multi-metric comparison (pentagon-shaped plot)
- **Bar Chart**: Side-by-side benchmark scores

### 4. LibriLight Long-Context Chart
- **Line Chart**: Perplexity vs Token Position
- Shows how models handle long sequences (up to 43k tokens)
- Lower and flatter = better

### 5. Statistics Panel
- Best performers for each metric

## 💡 Tips

### Interactive Features
- **Hover** over charts to see exact values
- **Click** legend items to toggle lines on/off
- **Zoom** on line chart (mouse wheel)
- **Sort** table by clicking column headers
- **Filter** to narrow down results
- **Export CSV** button for further analysis

### Comparing Checkpoints
1. Use filters to find interesting runs
2. Check boxes next to 2-5 checkpoints
3. Scroll down to see comparison charts
4. Uncheck "Clear Selection" to start over

### Understanding the Charts

**Radar Chart:**
- Bigger polygon = better overall performance
- Compare shapes to see strength/weakness patterns

**Bar Chart:**
- Direct side-by-side comparison
- Color-coded by benchmark

**LibriLight Line Chart:**
- Flat line = good long-context handling
- Rising line = degradation over long sequences
- Compare slopes to see which model maintains performance

## ⚠️ Requirements

**You need:**
- ✅ Any modern browser (Chrome, Firefox, Edge)
- ✅ Internet connection (to load Chart.js from CDN)

**You DON'T need:**
- ❌ Python
- ❌ Node.js
- ❌ Web server
- ❌ Any other software

## 🐛 Troubleshooting

### "Charts not showing"
- Check internet connection (Chart.js loads from CDN)
- Try a different browser
- Check browser console (F12) for errors

### "No data displayed"
- Make sure you downloaded `dashboard_standalone.html`
- The regular `dashboard.html` won't work without a server

### "Want offline version"
Let me know and I can create a fully offline version with Chart.js embedded too.

## 🔄 Automatic Updates

The dashboard updates automatically on the server when you run paper metrics evaluations. To see the latest:

1. **On server**, regenerate standalone:
   ```bash
   cd /home/alufr/ttt_tests/moshi-finetune/dashboard
   ./create_standalone.sh
   ```

2. **Download** the updated `dashboard_standalone.html` to Windows

3. **Open** it to see all your latest results!

## 📝 Example Workflow

**First time:**
```bash
# On server
cd /home/alufr/ttt_tests/moshi-finetune/dashboard
./create_standalone.sh

# Download dashboard_standalone.html to Windows
# Open in browser - see sample data
```

**After running evaluations:**
```bash
# Evaluations run automatically updating dashboard_data.json
# Then regenerate standalone:
./create_standalone.sh

# Download updated dashboard_standalone.html
# Open to see new results!
```

## 🎉 You're All Set!

Download **dashboard_standalone.html** and open it in your browser to see:
- ✅ 3 interactive plots (Radar, Bar, Line charts)
- ✅ Filterable/sortable results table
- ✅ Checkpoint comparison
- ✅ Statistics panel
- ✅ CSV export

**File location:** `/home/alufr/ttt_tests/moshi-finetune/dashboard/dashboard_standalone.html`

---

**Questions?** Check README.md for full documentation.
