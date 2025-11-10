#!/usr/bin/env python3
"""
Create interactive HTML table with all evaluation results
"""

import json
import pandas as pd
from pathlib import Path
import numpy as np

def load_all_results():
    """Load all evaluation results with full detail"""
    results_dir = Path("evaluation_results")
    all_results = []
    
    for results_file in results_dir.glob("*/results.json"):
        try:
            with open(results_file, 'r') as f:
                data = json.load(f)
            
            # Extract run info from path
            run_name = results_file.parent.name
            if "_ttt" in run_name:
                model_type = "TTT"
            elif "_baseline" in run_name:
                model_type = "Baseline"
            else:
                model_type = "Unknown"
            
            # Extract timestamp
            parts = run_name.split('_')
            if len(parts) >= 3:
                date_part = parts[1]  # 20251003
                time_part = parts[2]  # 174546
                # Format as readable datetime
                formatted_date = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}"
                formatted_time = f"{time_part[:2]}:{time_part[2:4]}:{time_part[4:6]}"
                timestamp = f"{formatted_date} {formatted_time}"
            else:
                timestamp = run_name
            
            # Create result record
            result = {
                'Run Name': run_name,
                'Model Type': model_type,
                'Timestamp': timestamp,
                'Date': formatted_date if len(parts) >= 3 else "",
                'Time': formatted_time if len(parts) >= 3 else "",
            }
            
            # Add all evaluation results
            if 'results' in data:
                results_data = data['results']
                
                # Add aggregate metrics
                if 'aggregate' in results_data:
                    for key, value in results_data['aggregate'].items():
                        result[key] = value if not pd.isna(value) else ""
                
                # Add task-specific metrics
                for task in ['sblimp', 'swuggy', 'tstory', 'sstory']:
                    if task in results_data:
                        for key, value in results_data[task].items():
                            result[key] = value if not pd.isna(value) else ""
                
                # Add LibriLight metrics
                if 'librilight' in results_data:
                    for key, value in results_data['librilight'].items():
                        if isinstance(value, (int, float)):
                            result[key] = round(value, 6) if not pd.isna(value) else ""
                        else:
                            result[key] = value if not pd.isna(value) else ""
            
            all_results.append(result)
            
        except Exception as e:
            print(f"Error loading {results_file}: {e}")
    
    return all_results

def create_interactive_html_table(results):
    """Create interactive HTML table with sorting"""
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Sort by timestamp (most recent first)
    df = df.sort_values('Timestamp', ascending=False)
    
    # Replace NaN with empty strings for display
    df = df.fillna("")
    
    # Round numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].apply(lambda x: round(x, 6) if x != "" and not pd.isna(x) else x)
    
    print(f"Creating table with {len(df)} runs")
    print(f"TTT runs: {len(df[df['Model Type'] == 'TTT'])}")
    print(f"Baseline runs: {len(df[df['Model Type'] == 'Baseline'])}")
    
    # Create HTML
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Moshi TTT vs Baseline Evaluation Results</title>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script src="https://cdn.datatables.net/1.11.5/js/jquery.dataTables.min.js"></script>
    <link rel="stylesheet" href="https://cdn.datatables.net/1.11.5/css/jquery.dataTables.min.css">
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }}
        .stats {{
            background-color: white;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        .stat-item {{
            text-align: center;
            padding: 10px;
            background-color: #ecf0f1;
            border-radius: 5px;
        }}
        .stat-number {{
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
        }}
        .stat-label {{
            font-size: 14px;
            color: #7f8c8d;
        }}
        table.dataTable {{
            width: 100% !important;
            background-color: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .model-ttt {{
            background-color: #e8f5e8 !important;
        }}
        .model-baseline {{
            background-color: #f0f8ff !important;
        }}
        .dataTables_wrapper {{
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th {{
            background-color: #34495e !important;
            color: white !important;
            font-weight: bold;
        }}
        .numeric {{
            text-align: right;
        }}
        .model-type-TTT {{
            color: #27ae60;
            font-weight: bold;
        }}
        .model-type-Baseline {{
            color: #3498db;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 Moshi TTT vs Baseline Evaluation Results</h1>
        <p>Comprehensive comparison of all evaluation runs</p>
    </div>
    
    <div class="stats">
        <div class="stats-grid">
            <div class="stat-item">
                <div class="stat-number">{len(df)}</div>
                <div class="stat-label">Total Runs</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">{len(df[df['Model Type'] == 'TTT'])}</div>
                <div class="stat-label">TTT Runs</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">{len(df[df['Model Type'] == 'Baseline'])}</div>
                <div class="stat-label">Baseline Runs</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">{len(df.columns)}</div>
                <div class="stat-label">Metrics Tracked</div>
            </div>
        </div>
    </div>

    <table id="resultsTable" class="display" style="width:100%">
        <thead>
            <tr>
"""
    
    # Add headers
    for col in df.columns:
        html_content += f"                <th>{col}</th>\n"
    
    html_content += """            </tr>
        </thead>
        <tbody>
"""
    
    # Add data rows
    for _, row in df.iterrows():
        model_type = row['Model Type']
        row_class = f"model-{model_type.lower()}" if model_type in ['TTT', 'Baseline'] else ""
        html_content += f"            <tr class='{row_class}'>\n"
        
        for col in df.columns:
            value = row[col]
            
            # Format the cell
            if col == 'Model Type':
                cell_class = f"model-type-{value}" if value in ['TTT', 'Baseline'] else ""
                html_content += f"                <td class='{cell_class}'>{value}</td>\n"
            elif isinstance(value, (int, float)) and value != "":
                html_content += f"                <td class='numeric'>{value}</td>\n"
            else:
                html_content += f"                <td>{value}</td>\n"
        
        html_content += "            </tr>\n"
    
    html_content += """        </tbody>
    </table>

    <script>
        $(document).ready(function() {{
            $('#resultsTable').DataTable({{
                "pageLength": 25,
                "order": [[ 2, "desc" ]], // Sort by timestamp descending
                "scrollX": true,
                "columnDefs": [
                    {{
                        "targets": "_all",
                        "className": "dt-center"
                    }},
                    {{
                        "targets": "numeric",
                        "className": "dt-body-right"
                    }}
                ],
                "dom": 'Bfrtip',
                "buttons": [
                    'copy', 'csv', 'excel'
                ]
            }});
        }});
    </script>
</body>
</html>"""
    
    return html_content

def main():
    print("Loading all evaluation results...")
    results = load_all_results()
    
    print("Creating interactive HTML table...")
    html_content = create_interactive_html_table(results)
    
    # Save HTML file
    output_file = "evaluation_results_interactive.html"
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    print(f"✅ Interactive table saved as: {output_file}")
    print(f"📊 Open {output_file} in your browser to view the sortable table")

if __name__ == "__main__":
    main()