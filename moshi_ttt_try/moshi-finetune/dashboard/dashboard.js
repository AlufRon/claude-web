// Paper Metrics Dashboard - Interactive Logic

class PaperMetricsDashboard {
    constructor() {
        this.data = null;
        this.filteredRuns = [];
        this.selectedRuns = [];
        this.sortColumn = 'checkpoint_step';
        this.sortDirection = 'desc';
        this.diffMode = 'absolute';  // 'absolute', 'percentage', or 'points'

        // Chart instances
        this.charts = {
            radar: null,
            bar: null,
            librilight: null
        };
    }

    async init() {
        console.log('Initializing dashboard...');

        try {
            // Load data
            await this.loadData();

            // Setup event listeners
            this.setupEventListeners();

            // Initial render
            this.applyFilters();
            this.updateMetadata();
            this.renderStatistics();

            console.log('Dashboard initialized successfully!');
        } catch (error) {
            console.error('Error initializing dashboard:', error);
            this.showError('Failed to load dashboard data. Please check that dashboard_data.json exists.');
        }
    }

    async loadData() {
        console.log('Loading dashboard data...');

        // Try to load from embedded data first (for standalone/Windows usage)
        if (typeof DASHBOARD_DATA !== 'undefined') {
            console.log('Using embedded data');
            this.data = DASHBOARD_DATA;
            console.log(`Loaded ${this.data.runs.length} runs from embedded data`);
            return;
        }

        // Fallback to fetching JSON file (for server/Linux usage)
        try {
            const response = await fetch('dashboard_data.json');
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            this.data = await response.json();
            console.log(`Loaded ${this.data.runs.length} runs from JSON file`);
        } catch (error) {
            console.error('Failed to load from JSON file:', error);
            throw new Error('Could not load dashboard data. Please ensure dashboard_data.json exists or data is embedded.');
        }
    }

    setupEventListeners() {
        // Filter inputs
        document.getElementById('filter-ttt').addEventListener('change', () => this.applyFilters());
        document.getElementById('filter-lora').addEventListener('change', () => this.applyFilters());
        document.getElementById('filter-training-type').addEventListener('change', () => this.applyFilters());
        document.getElementById('filter-steps').addEventListener('input', () => this.applyFilters());
        document.getElementById('filter-data').addEventListener('change', () => this.applyFilters());

        // Reset filters button
        document.getElementById('reset-filters').addEventListener('click', () => this.resetFilters());

        // Export CSV button
        document.getElementById('export-csv').addEventListener('click', () => this.exportCSV());

        // Clear selection button
        document.getElementById('clear-selection').addEventListener('click', () => this.clearSelection());

        // Select all checkbox
        document.getElementById('select-all').addEventListener('change', (e) => this.selectAll(e.target.checked));

        // Table sorting
        document.querySelectorAll('.sortable').forEach(th => {
            th.addEventListener('click', () => {
                const column = th.dataset.column;
                this.sortTable(column);
            });
        });

        // Show baseline checkbox
        document.getElementById('show-baseline').addEventListener('change', () => this.updateLibriLightChart());

        // LibriLight view mode
        document.getElementById('librilight-view-mode').addEventListener('change', (e) => {
            // Show/hide file selector based on view mode
            const fileSelector = document.getElementById('file-selector-container');
            if (e.target.value === 'individual') {
                fileSelector.style.display = 'inline-block';
            } else {
                fileSelector.style.display = 'none';
            }
            this.updateLibriLightChart();
        });

        // File selector for individual files
        document.getElementById('file-selector').addEventListener('change', () => this.updateLibriLightChart());

        // Show error bars
        document.getElementById('show-error-bars').addEventListener('change', () => this.updateLibriLightChart());

        // Auto-scale toggle for comparison charts
        document.getElementById('auto-scale-toggle').addEventListener('change', () => this.renderComparisonCharts());

        // Difference mode selector
        document.getElementById('diff-mode').addEventListener('change', (e) => {
            this.diffMode = e.target.value;
            this.renderComparisonCharts();
        });
    }

    applyFilters() {
        const filters = {
            ttt: document.getElementById('filter-ttt').value,
            lora: document.getElementById('filter-lora').value,
            trainingType: document.getElementById('filter-training-type').value,
            minSteps: parseInt(document.getElementById('filter-steps').value) || 0,
            dataSource: document.getElementById('filter-data').value
        };

        this.filteredRuns = this.data.runs.filter(run => {
            // TTT filter
            if (filters.ttt !== 'all') {
                const tttEnabled = run.training_config?.ttt?.enable || false;
                if (filters.ttt === 'true' && !tttEnabled) return false;
                if (filters.ttt === 'false' && tttEnabled) return false;
            }

            // LoRA filter
            if (filters.lora !== 'all') {
                const loraEnabled = run.training_config?.lora?.enable || false;
                if (filters.lora === 'true' && !loraEnabled) return false;
                if (filters.lora === 'false' && loraEnabled) return false;
            }

            // Training type filter
            if (filters.trainingType !== 'all') {
                if (!run.tags.includes(filters.trainingType)) return false;
            }

            // Min steps filter
            if (filters.minSteps > 0) {
                if (run.checkpoint_step < filters.minSteps) return false;
            }

            // Data source filter
            if (filters.dataSource !== 'all') {
                if (!run.tags.includes(filters.dataSource)) return false;
            }

            return true;
        });

        console.log(`Filtered to ${this.filteredRuns.length} runs`);

        // Update display
        this.sortFilteredRuns();
        this.renderTable();
        this.updateFilteredCount();
    }

    sortFilteredRuns() {
        this.filteredRuns.sort((a, b) => {
            let aVal = this.getNestedValue(a, this.sortColumn);
            let bVal = this.getNestedValue(b, this.sortColumn);

            // Handle undefined values
            if (aVal === undefined) aVal = 0;
            if (bVal === undefined) bVal = 0;

            if (this.sortDirection === 'asc') {
                return aVal > bVal ? 1 : -1;
            } else {
                return aVal < bVal ? 1 : -1;
            }
        });
    }

    getNestedValue(obj, path) {
        return path.split('.').reduce((current, key) => current?.[key], obj);
    }

    sortTable(column) {
        if (this.sortColumn === column) {
            // Toggle direction
            this.sortDirection = this.sortDirection === 'asc' ? 'desc' : 'asc';
        } else {
            // New column, default to descending for metrics
            this.sortColumn = column;
            this.sortDirection = 'desc';
        }

        // Update header classes
        document.querySelectorAll('.sortable').forEach(th => {
            th.classList.remove('sorted-asc', 'sorted-desc');
            if (th.dataset.column === column) {
                th.classList.add(this.sortDirection === 'asc' ? 'sorted-asc' : 'sorted-desc');
            }
        });

        this.sortFilteredRuns();
        this.renderTable();
    }

    renderTable() {
        const tbody = document.getElementById('table-body');
        tbody.innerHTML = '';

        if (this.filteredRuns.length === 0) {
            tbody.innerHTML = '<tr><td colspan="12" style="text-align: center; padding: 2rem; color: #999;">No runs match the current filters</td></tr>';
            return;
        }

        this.filteredRuns.forEach(run => {
            const row = document.createElement('tr');

            const isSelected = this.selectedRuns.includes(run.id);
            if (isSelected) row.classList.add('selected');

            row.innerHTML = `
                <td><input type="checkbox" class="row-checkbox" data-run-id="${run.id}" ${isSelected ? 'checked' : ''}></td>
                <td title="${run.checkpoint_path}">${this.truncate(run.name, 40)}</td>
                <td>${run.checkpoint_step}</td>
                <td class="metric-cell">${this.formatMetric(run.metrics?.sblimp_accuracy)}</td>
                <td class="metric-cell">${this.formatMetric(run.metrics?.swuggy_accuracy)}</td>
                <td class="metric-cell">${this.formatMetric(run.metrics?.tstory_accuracy)}</td>
                <td class="metric-cell">${this.formatMetric(run.metrics?.sstory_accuracy)}</td>
                <td class="metric-cell">${this.formatMetric(run.metrics?.librilight_perplexity_8k, 1)}</td>
                <td class="metric-cell">${this.formatMetric(run.metrics?.librilight_perplexity_16k, 1)}</td>
                <td class="metric-cell"><strong>${this.formatMetric(run.metrics?.paper_metrics_avg)}</strong></td>
                <td>${this.renderBadge(run.training_config?.ttt?.enable)}</td>
                <td>${this.renderBadge(run.training_config?.lora?.enable)}</td>
            `;

            // Add checkbox listener
            const checkbox = row.querySelector('.row-checkbox');
            checkbox.addEventListener('change', (e) => {
                if (e.target.checked) {
                    this.selectRun(run.id);
                } else {
                    this.deselectRun(run.id);
                }
            });

            tbody.appendChild(row);
        });
    }

    formatMetric(value, decimals = 3) {
        if (value === undefined || value === null) return '—';
        return typeof value === 'number' ? value.toFixed(decimals) : value;
    }

    truncate(str, maxLen) {
        if (str.length <= maxLen) return str;
        return str.substring(0, maxLen - 3) + '...';
    }

    renderBadge(enabled) {
        if (enabled) {
            return '<span class="badge badge-success">Yes</span>';
        } else {
            return '<span class="badge badge-danger">No</span>';
        }
    }

    selectRun(runId) {
        if (!this.selectedRuns.includes(runId)) {
            this.selectedRuns.push(runId);
            this.updateComparison();
        }
    }

    deselectRun(runId) {
        this.selectedRuns = this.selectedRuns.filter(id => id !== runId);
        this.updateComparison();
    }

    selectAll(checked) {
        if (checked) {
            this.selectedRuns = this.filteredRuns.map(run => run.id);
        } else {
            this.selectedRuns = [];
        }
        this.renderTable();
        this.updateComparison();
    }

    clearSelection() {
        this.selectedRuns = [];
        document.getElementById('select-all').checked = false;
        this.renderTable();
        this.updateComparison();
    }

    updateComparison() {
        const selectedCount = this.selectedRuns.length;
        document.getElementById('selected-count').textContent = `${selectedCount} checkpoint${selectedCount !== 1 ? 's' : ''} selected`;

        if (selectedCount >= 2) {
            document.querySelector('.comparison-placeholder').style.display = 'none';
            document.getElementById('comparison-content').style.display = 'block';
            this.renderComparisonCharts();
            this.renderConfigDiff();
            this.updateLibriLightChart();
        } else {
            document.querySelector('.comparison-placeholder').style.display = 'block';
            document.getElementById('comparison-content').style.display = 'none';
        }
    }

    renderComparisonCharts() {
        const selectedRunsData = this.selectedRuns.map(id =>
            this.data.runs.find(run => run.id === id)
        ).filter(run => run !== undefined);

        // Radar chart
        this.renderRadarChart(selectedRunsData);

        // Bar chart
        this.renderBarChart(selectedRunsData);
    }

    renderRadarChart(runs) {
        const ctx = document.getElementById('radar-chart');

        if (this.charts.radar) {
            this.charts.radar.destroy();
        }

        const autoScale = document.getElementById('auto-scale-toggle').checked;

        // Collect all metric values for auto-scaling
        const allValues = [];
        runs.forEach(run => {
            allValues.push(
                run.metrics?.sblimp_accuracy || 0,
                run.metrics?.swuggy_accuracy || 0,
                run.metrics?.tstory_accuracy || 0,
                run.metrics?.sstory_accuracy || 0,
                run.metrics?.paper_metrics_avg || 0
            );
        });

        // Calculate scale
        let minVal = 0;
        let maxVal = 1.0;
        if (autoScale && allValues.length > 0) {
            minVal = Math.max(0, Math.min(...allValues) - 0.05); // 5% padding below
            maxVal = Math.min(1.0, Math.max(...allValues) + 0.05); // 5% padding above
        }

        const datasets = runs.map((run, idx) => ({
            label: this.truncate(run.name, 30),
            data: [
                run.metrics?.sblimp_accuracy || 0,
                run.metrics?.swuggy_accuracy || 0,
                run.metrics?.tstory_accuracy || 0,
                run.metrics?.sstory_accuracy || 0,
                (run.metrics?.paper_metrics_avg || 0)
            ],
            borderColor: this.getColor(idx),
            backgroundColor: this.getColor(idx, 0.2),
            pointBackgroundColor: this.getColor(idx),
        }));

        this.charts.radar = new Chart(ctx, {
            type: 'radar',
            data: {
                labels: ['sBLIMP', 'sWUGGY', 'tStory', 'sStory', 'Overall'],
                datasets: datasets
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                scales: {
                    r: {
                        beginAtZero: !autoScale,
                        min: minVal,
                        max: maxVal,
                        ticks: {
                            stepSize: autoScale ? (maxVal - minVal) / 5 : 0.2
                        }
                    }
                }
            }
        });
    }

    renderBarChart(runs) {
        const ctx = document.getElementById('bar-chart');

        if (this.charts.bar) {
            this.charts.bar.destroy();
        }

        const autoScale = document.getElementById('auto-scale-toggle').checked;
        const labels = runs.map(run => this.truncate(run.name, 20));

        // Determine what to show based on diffMode
        let datasets;
        let yAxisLabel;
        let minVal = 0;
        let maxVal = 1.0;

        if (this.diffMode === 'percentage' && this.data.baseline) {
            // Calculate percentage difference vs baseline (relative change)
            const baseline = this.data.baseline.metrics;

            datasets = [
                {
                    label: 'sBLIMP % diff',
                    data: runs.map(r => {
                        const val = r.metrics?.sblimp_accuracy || 0;
                        const base = baseline.sblimp_accuracy || 0;
                        return base > 0 ? ((val - base) / base * 100) : 0;
                    }),
                    backgroundColor: 'rgba(74, 144, 226, 0.7)'
                },
                {
                    label: 'sWUGGY % diff',
                    data: runs.map(r => {
                        const val = r.metrics?.swuggy_accuracy || 0;
                        const base = baseline.swuggy_accuracy || 0;
                        return base > 0 ? ((val - base) / base * 100) : 0;
                    }),
                    backgroundColor: 'rgba(126, 211, 33, 0.7)'
                },
                {
                    label: 'tStory % diff',
                    data: runs.map(r => {
                        const val = r.metrics?.tstory_accuracy || 0;
                        const base = baseline.tstory_accuracy || 0;
                        return base > 0 ? ((val - base) / base * 100) : 0;
                    }),
                    backgroundColor: 'rgba(245, 166, 35, 0.7)'
                },
                {
                    label: 'sStory % diff',
                    data: runs.map(r => {
                        const val = r.metrics?.sstory_accuracy || 0;
                        const base = baseline.sstory_accuracy || 0;
                        return base > 0 ? ((val - base) / base * 100) : 0;
                    }),
                    backgroundColor: 'rgba(208, 2, 27, 0.7)'
                }
            ];

            yAxisLabel = '% Difference from Baseline (Relative)';

            // Calculate scale for percentage mode
            const allPercentages = datasets.flatMap(ds => ds.data);
            minVal = Math.min(...allPercentages, 0) - 1; // Add padding
            maxVal = Math.max(...allPercentages, 0) + 1;

        } else if (this.diffMode === 'points' && this.data.baseline) {
            // Calculate absolute difference in percentage points
            const baseline = this.data.baseline.metrics;

            datasets = [
                {
                    label: 'sBLIMP diff (pts)',
                    data: runs.map(r => {
                        const val = r.metrics?.sblimp_accuracy || 0;
                        const base = baseline.sblimp_accuracy || 0;
                        return (val - base) * 100;  // Convert to percentage points
                    }),
                    backgroundColor: 'rgba(74, 144, 226, 0.7)'
                },
                {
                    label: 'sWUGGY diff (pts)',
                    data: runs.map(r => {
                        const val = r.metrics?.swuggy_accuracy || 0;
                        const base = baseline.swuggy_accuracy || 0;
                        return (val - base) * 100;  // Convert to percentage points
                    }),
                    backgroundColor: 'rgba(126, 211, 33, 0.7)'
                },
                {
                    label: 'tStory diff (pts)',
                    data: runs.map(r => {
                        const val = r.metrics?.tstory_accuracy || 0;
                        const base = baseline.tstory_accuracy || 0;
                        return (val - base) * 100;  // Convert to percentage points
                    }),
                    backgroundColor: 'rgba(245, 166, 35, 0.7)'
                },
                {
                    label: 'sStory diff (pts)',
                    data: runs.map(r => {
                        const val = r.metrics?.sstory_accuracy || 0;
                        const base = baseline.sstory_accuracy || 0;
                        return (val - base) * 100;  // Convert to percentage points
                    }),
                    backgroundColor: 'rgba(208, 2, 27, 0.7)'
                }
            ];

            yAxisLabel = 'Difference from Baseline (Percentage Points)';

            // Calculate scale for points mode
            const allPoints = datasets.flatMap(ds => ds.data);
            minVal = Math.min(...allPoints, 0) - 1; // Add padding
            maxVal = Math.max(...allPoints, 0) + 1;

        } else {
            // Show absolute values
            datasets = [
                {
                    label: 'sBLIMP',
                    data: runs.map(r => r.metrics?.sblimp_accuracy || 0),
                    backgroundColor: 'rgba(74, 144, 226, 0.7)'
                },
                {
                    label: 'sWUGGY',
                    data: runs.map(r => r.metrics?.swuggy_accuracy || 0),
                    backgroundColor: 'rgba(126, 211, 33, 0.7)'
                },
                {
                    label: 'tStory',
                    data: runs.map(r => r.metrics?.tstory_accuracy || 0),
                    backgroundColor: 'rgba(245, 166, 35, 0.7)'
                },
                {
                    label: 'sStory',
                    data: runs.map(r => r.metrics?.sstory_accuracy || 0),
                    backgroundColor: 'rgba(208, 2, 27, 0.7)'
                }
            ];

            yAxisLabel = 'Accuracy';

            // Auto-scale for absolute values
            if (autoScale) {
                const allValues = datasets.flatMap(ds => ds.data);
                minVal = Math.max(0, Math.min(...allValues) - 0.05);
                maxVal = Math.min(1.0, Math.max(...allValues) + 0.05);
            }
        }

        this.charts.bar = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: datasets
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                scales: {
                    y: {
                        beginAtZero: !autoScale && this.diffMode === 'absolute',
                        min: minVal,
                        max: maxVal,
                        title: {
                            display: true,
                            text: yAxisLabel
                        }
                    }
                },
                plugins: {
                    tooltip: {
                        callbacks: {
                            label: (context) => {
                                const label = context.dataset.label || '';
                                const value = context.parsed.y;
                                if (this.diffMode === 'percentage') {
                                    return `${label}: ${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
                                } else if (this.diffMode === 'points') {
                                    return `${label}: ${value >= 0 ? '+' : ''}${value.toFixed(2)} pts`;
                                } else {
                                    return `${label}: ${(value * 100).toFixed(2)}%`;
                                }
                            }
                        }
                    }
                }
            }
        });
    }

    updateLibriLightChart() {
        const selectedRunsData = this.selectedRuns.map(id =>
            this.data.runs.find(run => run.id === id)
        ).filter(run => run !== undefined && run.librilight_progression && run.librilight_progression.length > 0);

        if (selectedRunsData.length === 0) {
            return;
        }

        const ctx = document.getElementById('librilight-chart');

        if (this.charts.librilight) {
            this.charts.librilight.destroy();
        }

        const viewMode = document.getElementById('librilight-view-mode').value;
        const showErrorBars = document.getElementById('show-error-bars').checked;

        const datasets = [];

        // Generate datasets based on view mode
        selectedRunsData.forEach((run, idx) => {
            const baseColor = this.getColor(idx);
            const runName = this.truncate(run.name, 30);

            if (viewMode === 'mean' && run.librilight_progression_mean) {
                // Show mean with optional error bars
                const meanData = {
                    label: `${runName} (mean)`,
                    data: run.librilight_progression_mean.map(p => ({
                        x: p.position,
                        y: p.perplexity
                    })),
                    borderColor: baseColor,
                    backgroundColor: this.getColor(idx, 0.1),
                    borderWidth: 2,
                    pointRadius: 3,
                    fill: false,
                    tension: 0.1
                };
                datasets.push(meanData);

                // Add error bars if enabled
                if (showErrorBars && run.librilight_progression_mean[0].perplexity_std !== undefined) {
                    const errorData = {
                        label: `${runName} (±std)`,
                        data: run.librilight_progression_mean.map(p => ({
                            x: p.position,
                            y: p.perplexity,
                            yMin: p.perplexity - p.perplexity_std,
                            yMax: p.perplexity + p.perplexity_std
                        })),
                        borderColor: this.getColor(idx, 0.3),
                        backgroundColor: this.getColor(idx, 0.2),
                        borderWidth: 1,
                        pointRadius: 0,
                        fill: '+1',
                        tension: 0.1,
                        showLine: false
                    };
                    // Note: Full error bar support would require a plugin
                    // For now, we just show the mean line
                }

            } else if (viewMode === 'median' && run.librilight_progression_median) {
                datasets.push({
                    label: `${runName} (median)`,
                    data: run.librilight_progression_median.map(p => ({
                        x: p.position,
                        y: p.perplexity
                    })),
                    borderColor: baseColor,
                    backgroundColor: this.getColor(idx, 0.1),
                    borderWidth: 2,
                    pointRadius: 3,
                    fill: false,
                    tension: 0.1
                });

            } else if (viewMode === 'individual' && run.librilight_progression_individual) {
                // Get selected files from the file selector
                const fileSelector = document.getElementById('file-selector');
                const selectedFiles = Array.from(fileSelector.selectedOptions).map(opt => parseInt(opt.value));

                // Show each selected file as separate line
                run.librilight_progression_individual.forEach((fileData, fileIdx) => {
                    // Only show if this file is selected
                    if (selectedFiles.includes(fileIdx) && fileData && fileData.length > 0) {
                        datasets.push({
                            label: `${runName} - File ${fileIdx + 1}`,
                            data: fileData.map(p => ({
                                x: p.position,
                                y: p.perplexity
                            })),
                            borderColor: this.getColor(idx, 0.5 + fileIdx * 0.15),
                            backgroundColor: this.getColor(idx, 0.05),
                            borderWidth: 2,
                            pointRadius: 2,
                            fill: false,
                            tension: 0.1,
                            borderDash: fileIdx === 0 ? [] : [fileIdx * 3, 3]
                        });
                    }
                });

            } else {
                // Raw: show all points
                datasets.push({
                    label: `${runName} (all points)`,
                    data: run.librilight_progression.map(p => ({
                        x: p.position,
                        y: p.perplexity
                    })),
                    borderColor: baseColor,
                    backgroundColor: this.getColor(idx, 0.1),
                    borderWidth: 2,
                    pointRadius: 1,
                    fill: false,
                    tension: 0.1
                });
            }
        });

        // Add baseline if requested
        const showBaseline = document.getElementById('show-baseline').checked;
        if (showBaseline && this.data.baseline) {
            let baselineData = null;

            if (viewMode === 'mean' && this.data.baseline.librilight_progression_mean) {
                baselineData = this.data.baseline.librilight_progression_mean.map(p => ({
                    x: p.position,
                    y: p.perplexity
                }));
                datasets.push({
                    label: 'Baseline (mean)',
                    data: baselineData,
                    borderColor: '#999',
                    backgroundColor: 'rgba(153, 153, 153, 0.1)',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    pointRadius: 2,
                    fill: false,
                    tension: 0.1
                });
            } else if (viewMode === 'median' && this.data.baseline.librilight_progression_median) {
                baselineData = this.data.baseline.librilight_progression_median.map(p => ({
                    x: p.position,
                    y: p.perplexity
                }));
                datasets.push({
                    label: 'Baseline (median)',
                    data: baselineData,
                    borderColor: '#999',
                    backgroundColor: 'rgba(153, 153, 153, 0.1)',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    pointRadius: 2,
                    fill: false,
                    tension: 0.1
                });
            } else if (viewMode === 'individual' && this.data.baseline.librilight_progression_individual) {
                // Get selected files from the file selector
                const fileSelector = document.getElementById('file-selector');
                const selectedFiles = Array.from(fileSelector.selectedOptions).map(opt => parseInt(opt.value));

                // Show each selected file for baseline
                this.data.baseline.librilight_progression_individual.forEach((fileData, fileIdx) => {
                    // Only show if this file is selected
                    if (selectedFiles.includes(fileIdx) && fileData && fileData.length > 0) {
                        datasets.push({
                            label: `Baseline - File ${fileIdx + 1}`,
                            data: fileData.map(p => ({
                                x: p.position,
                                y: p.perplexity
                            })),
                            borderColor: `rgba(102, 102, 102, ${0.5 + fileIdx * 0.15})`,
                            backgroundColor: 'rgba(102, 102, 102, 0.05)',
                            borderWidth: 2,
                            pointRadius: 2,
                            fill: false,
                            tension: 0.1,
                            borderDash: [5 + fileIdx * 2, 5]
                        });
                    }
                });
            } else if (this.data.baseline.librilight_progression) {
                baselineData = this.data.baseline.librilight_progression.map(p => ({
                    x: p.position,
                    y: p.perplexity
                }));
                datasets.push({
                    label: 'Baseline',
                    data: baselineData,
                    borderColor: '#999',
                    backgroundColor: 'rgba(153, 153, 153, 0.1)',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    pointRadius: 2,
                    fill: false,
                    tension: 0.1
                });
            }
        }

        this.charts.librilight = new Chart(ctx, {
            type: 'line',
            data: { datasets },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                scales: {
                    x: {
                        type: 'linear',
                        title: {
                            display: true,
                            text: 'Token Position'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Perplexity (lower is better)'
                        },
                        beginAtZero: false
                    }
                },
                plugins: {
                    legend: {
                        display: true,
                        position: 'bottom'
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false
                    }
                },
                interaction: {
                    mode: 'nearest',
                    axis: 'x',
                    intersect: false
                }
            }
        });
    }

    renderConfigDiff() {
        const selectedRunsData = this.selectedRuns.map(id =>
            this.data.runs.find(run => run.id === id)
        ).filter(run => run !== undefined);

        if (selectedRunsData.length < 2) {
            return;
        }

        const configDiv = document.getElementById('config-diff');
        configDiv.innerHTML = '';

        // Compare key configuration parameters
        const keysToCompare = [
            'training_config.max_steps',
            'training_config.optim.lr',
            'training_config.ttt.enable',
            'training_config.ttt.layers',
            'training_config.ttt.base_lr',
            'training_config.ttt.mini_batch_size',
            'training_config.ttt.ttt_mlp_layers',
            'training_config.lora.enable',
            'training_config.lora.rank',
            'training_config.full_finetuning'
        ];

        keysToCompare.forEach(key => {
            const values = selectedRunsData.map(run => this.getNestedValue(run, key));
            const allSame = values.every(v => v === values[0]);

            const row = document.createElement('div');
            row.className = 'config-row';

            const keySpan = document.createElement('div');
            keySpan.className = 'config-key';
            keySpan.textContent = key.split('.').pop();

            row.appendChild(keySpan);

            values.forEach(val => {
                const valSpan = document.createElement('div');
                valSpan.className = 'config-value';
                if (!allSame) {
                    valSpan.classList.add('config-diff-highlight');
                }
                valSpan.textContent = val !== undefined ? JSON.stringify(val) : '—';
                row.appendChild(valSpan);
            });

            configDiv.appendChild(row);
        });
    }

    getColor(index, alpha = 1) {
        const colors = [
            `rgba(74, 144, 226, ${alpha})`,   // Blue
            `rgba(126, 211, 33, ${alpha})`,   // Green
            `rgba(245, 166, 35, ${alpha})`,   // Orange
            `rgba(208, 2, 27, ${alpha})`,     // Red
            `rgba(156, 39, 176, ${alpha})`,   // Purple
            `rgba(0, 150, 136, ${alpha})`,    // Teal
        ];
        return colors[index % colors.length];
    }

    resetFilters() {
        document.getElementById('filter-ttt').value = 'all';
        document.getElementById('filter-lora').value = 'all';
        document.getElementById('filter-training-type').value = 'all';
        document.getElementById('filter-steps').value = '';
        document.getElementById('filter-data').value = 'all';
        this.applyFilters();
    }

    updateMetadata() {
        const lastUpdated = new Date(this.data.metadata.generated_at).toLocaleString();
        document.getElementById('last-updated').textContent = `Last Updated: ${lastUpdated}`;
        document.getElementById('total-runs').textContent = `Total Runs: ${this.data.metadata.total_runs}`;
    }

    updateFilteredCount() {
        document.getElementById('filtered-count').textContent = `Showing ${this.filteredRuns.length} run${this.filteredRuns.length !== 1 ? 's' : ''}`;
    }

    renderStatistics() {
        const statsGrid = document.getElementById('stats-grid');

        if (this.filteredRuns.length === 0) {
            statsGrid.innerHTML = '<p>No data to display statistics</p>';
            return;
        }

        const bestSblimp = _.maxBy(this.filteredRuns, r => r.metrics?.sblimp_accuracy || 0);
        const bestSwuggy = _.maxBy(this.filteredRuns, r => r.metrics?.swuggy_accuracy || 0);
        const bestTStory = _.maxBy(this.filteredRuns, r => r.metrics?.tstory_accuracy || 0);
        const bestAvg = _.maxBy(this.filteredRuns, r => r.metrics?.paper_metrics_avg || 0);

        const stats = [
            {
                title: 'Best sBLIMP',
                value: this.formatMetric(bestSblimp.metrics?.sblimp_accuracy),
                label: this.truncate(bestSblimp.name, 25)
            },
            {
                title: 'Best sWUGGY',
                value: this.formatMetric(bestSwuggy.metrics?.swuggy_accuracy),
                label: this.truncate(bestSwuggy.name, 25)
            },
            {
                title: 'Best tStory',
                value: this.formatMetric(bestTStory.metrics?.tstory_accuracy),
                label: this.truncate(bestTStory.name, 25)
            },
            {
                title: 'Best Overall',
                value: this.formatMetric(bestAvg.metrics?.paper_metrics_avg),
                label: this.truncate(bestAvg.name, 25)
            }
        ];

        statsGrid.innerHTML = stats.map(stat => `
            <div class="stat-card">
                <h3>${stat.title}</h3>
                <div class="stat-value">${stat.value}</div>
                <div class="stat-label">${stat.label}</div>
            </div>
        `).join('');
    }

    exportCSV() {
        // Create CSV content
        const headers = ['Name', 'Step', 'sBLIMP', 'sWUGGY', 'tStory', 'sStory', 'PPL 8k', 'PPL 16k', 'Avg', 'TTT', 'LoRA'];

        const rows = this.filteredRuns.map(run => [
            `"${run.name}"`,
            run.checkpoint_step,
            run.metrics?.sblimp_accuracy || '',
            run.metrics?.swuggy_accuracy || '',
            run.metrics?.tstory_accuracy || '',
            run.metrics?.sstory_accuracy || '',
            run.metrics?.librilight_perplexity_8k || '',
            run.metrics?.librilight_perplexity_16k || '',
            run.metrics?.paper_metrics_avg || '',
            run.training_config?.ttt?.enable || false,
            run.training_config?.lora?.enable || false
        ]);

        const csv = [headers.join(','), ...rows.map(r => r.join(','))].join('\n');

        // Download
        const blob = new Blob([csv], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `paper_metrics_${new Date().toISOString().split('T')[0]}.csv`;
        a.click();
        URL.revokeObjectURL(url);
    }

    showError(message) {
        const container = document.querySelector('.container');
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error';
        errorDiv.textContent = message;
        container.insertBefore(errorDiv, container.firstChild);
    }
}

// Initialize dashboard when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    const dashboard = new PaperMetricsDashboard();
    dashboard.init();
});
