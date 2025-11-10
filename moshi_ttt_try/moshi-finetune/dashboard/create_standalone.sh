#!/bin/bash
# Create Standalone Dashboard
# This combines HTML, CSS, JS, and data into a single file for easy Windows usage

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "📦 Creating standalone dashboard..."

# Check required files exist
for file in dashboard.html dashboard.js dashboard.css dashboard_data.json; do
    if [ ! -f "$file" ]; then
        echo "❌ Error: $file not found!"
        exit 1
    fi
done

# Create standalone HTML file
cat > dashboard_standalone.html << 'HTMLEOF'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Paper Metrics Dashboard</title>

    <!-- Chart.js for visualizations -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>

    <!-- Lodash for utility functions -->
    <script src="https://cdn.jsdelivr.net/npm/lodash@4.17.21/lodash.min.js"></script>

    <!-- Embedded Data -->
    <script>
    const DASHBOARD_DATA =
HTMLEOF

# Embed data
cat dashboard_data.json >> dashboard_standalone.html

cat >> dashboard_standalone.html << 'HTMLEOF2'
;
    </script>

    <!-- Embedded CSS -->
    <style>
HTMLEOF2

# Embed CSS
cat dashboard.css >> dashboard_standalone.html

cat >> dashboard_standalone.html << 'HTMLEOF3'
    </style>
</head>
<body>
HTMLEOF3

# Extract and embed body content
sed -n '/<body>/,/<\/body>/p' dashboard.html | sed '1d;$d' >> dashboard_standalone.html

cat >> dashboard_standalone.html << 'HTMLEOF4'

    <!-- Embedded JavaScript -->
    <script>
HTMLEOF4

# Embed JavaScript
cat dashboard.js >> dashboard_standalone.html

cat >> dashboard_standalone.html << 'HTMLEOF5'
    </script>
</body>
</html>
HTMLEOF5

echo "✅ Created dashboard_standalone.html"
echo ""
echo "📊 File info:"
ls -lh dashboard_standalone.html
echo ""
echo "📥 For Windows usage:"
echo "   1. Download dashboard_standalone.html only (this single file!)"
echo "   2. Open it in Chrome/Firefox/Edge"
echo "   3. All data, CSS, and JS are embedded - no other files needed!"
echo ""
echo "⚠️  Note: You still need internet connection for Chart.js to load"
echo ""
