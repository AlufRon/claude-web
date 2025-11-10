#!/bin/bash
echo "📊 Dashboard Status Check"
echo ""
echo "Required files:"
for file in dashboard.html dashboard.js dashboard.css dashboard_data.json; do
    if [ -f "$file" ]; then
        echo "  ✅ $file ($(stat -c%s $file) bytes)"
    else
        echo "  ❌ $file MISSING"
    fi
done
echo ""
echo "📂 All files present in: $(pwd)"
echo ""
echo "To view on Linux:"
echo "  firefox dashboard.html &"
echo ""
echo "To download to Windows:"
echo "  1. Download all 4 files above"
echo "  2. Put them in the same folder"
echo "  3. Open dashboard.html"
echo ""
echo "Files ready for download:"
ls -lh dashboard.html dashboard.js dashboard.css dashboard_data.json 2>/dev/null
