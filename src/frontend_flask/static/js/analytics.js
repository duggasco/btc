/**
 * Analytics & Research page functionality
 */

function initializeAnalytics() {
    // Set up tab navigation
    setupTabs();
    
    // Set up form submissions
    setupBacktestForm();
    setupMonteCarloForm();
    setupOptimizationForm();
    
    // Set up range sliders
    setupRangeSliders();
    
    // Load initial data quality metrics
    loadDataQuality();
}

function setupTabs() {
    const tabs = document.querySelectorAll('.tab');
    const panes = document.querySelectorAll('.tab-pane');
    
    tabs.forEach(tab => {
        tab.addEventListener('click', (e) => {
            e.preventDefault();
            
            // Remove active class from all tabs and panes
            tabs.forEach(t => t.classList.remove('active'));
            panes.forEach(p => p.classList.remove('active'));
            
            // Add active class to clicked tab and corresponding pane
            tab.classList.add('active');
            const targetPane = document.querySelector(tab.getAttribute('href'));
            if (targetPane) {
                targetPane.classList.add('active');
            }
        });
    });
}

// Backtesting functionality
function setupBacktestForm() {
    const form = document.getElementById('backtest-form');
    if (form) {
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            await runBacktest();
        });
    }
}

async function runBacktest() {
    const period = document.getElementById('backtest-period').value;
    const optimizeWeights = document.getElementById('optimize-weights').checked;
    const includeMacro = document.getElementById('include-macro').checked;
    
    // Show loading, hide results
    document.getElementById('backtest-loading').style.display = 'block';
    document.getElementById('backtest-results').style.display = 'none';
    
    try {
        const response = await window.apiClient.request('/analytics/api/backtest', {
            method: 'POST',
            body: JSON.stringify({
                period: period,
                optimize_weights: optimizeWeights,
                include_macro: includeMacro
            })
        });
        
        displayBacktestResults(response);
    } catch (error) {
        console.error('Backtest error:', error);
        alert('Failed to run backtest. Please try again.');
    } finally {
        document.getElementById('backtest-loading').style.display = 'none';
    }
}

function displayBacktestResults(results) {
    // Update metrics
    document.getElementById('sortino-ratio').textContent = results.metrics.sortino_ratio.toFixed(2);
    document.getElementById('sharpe-ratio').textContent = results.metrics.sharpe_ratio.toFixed(2);
    document.getElementById('max-drawdown').textContent = `${(results.metrics.max_drawdown * 100).toFixed(1)}%`;
    document.getElementById('win-rate').textContent = `${(results.metrics.win_rate * 100).toFixed(1)}%`;
    document.getElementById('total-return').textContent = `${(results.metrics.total_return * 100).toFixed(1)}%`;
    document.getElementById('profit-factor').textContent = results.metrics.profit_factor.toFixed(2);
    
    // Update trade analysis
    document.getElementById('total-trades').textContent = results.trades.total;
    document.getElementById('long-positions').textContent = results.trades.long;
    document.getElementById('short-positions').textContent = results.trades.short;
    document.getElementById('avg-turnover').textContent = `${results.trades.avg_turnover.toFixed(1)}%`;
    
    // Create charts
    createEquityCurve(results.equity_curve);
    createDrawdownChart(results.drawdown_series);
    
    // Show results
    document.getElementById('backtest-results').style.display = 'block';
}

function createEquityCurve(data) {
    window.chartManager.createLineChart('equity-curve-chart', data, {
        title: '',
        yTitle: 'Portfolio Value ($)',
        xTitle: 'Date',
        color: '#f7931a'
    });
}

function createDrawdownChart(data) {
    const trace = {
        x: data.map(d => d.timestamp),
        y: data.map(d => d.drawdown * 100),
        type: 'scatter',
        mode: 'lines',
        fill: 'tozeroy',
        line: { color: '#ef4444' },
        fillcolor: 'rgba(239, 68, 68, 0.2)'
    };
    
    const layout = {
        ...window.chartManager.darkTheme,
        yaxis: { ...window.chartManager.darkTheme.yaxis, title: 'Drawdown (%)' },
        height: 300
    };
    
    Plotly.newPlot('drawdown-chart', [trace], layout, { displayModeBar: false });
}

// Monte Carlo functionality
function setupMonteCarloForm() {
    const form = document.getElementById('monte-carlo-form');
    if (form) {
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            await runMonteCarlo();
        });
    }
}

async function runMonteCarlo() {
    const numSimulations = parseInt(document.getElementById('num-simulations').value);
    const timeHorizon = parseInt(document.getElementById('time-horizon').value);
    const confidenceLevel = parseFloat(document.getElementById('confidence-level').value);
    
    // Show loading, hide results
    document.getElementById('mc-loading').style.display = 'block';
    document.getElementById('mc-results').style.display = 'none';
    
    try {
        const response = await window.apiClient.request('/analytics/api/monte-carlo', {
            method: 'POST',
            body: JSON.stringify({
                num_simulations: numSimulations,
                time_horizon: timeHorizon,
                confidence_level: confidenceLevel
            })
        });
        
        displayMonteCarloResults(response);
    } catch (error) {
        console.error('Monte Carlo error:', error);
        alert('Failed to run Monte Carlo simulation. Please try again.');
    } finally {
        document.getElementById('mc-loading').style.display = 'none';
    }
}

function displayMonteCarloResults(results) {
    // Update metrics
    document.getElementById('expected-return').textContent = `${(results.projections.expected_return * 100).toFixed(1)}%`;
    document.getElementById('var-value').textContent = `${(results.projections.var * 100).toFixed(1)}%`;
    document.getElementById('expected-volatility').textContent = `${(results.projections.expected_volatility * 100).toFixed(1)}%`;
    document.getElementById('profit-probability').textContent = `${(results.projections.profit_probability * 100).toFixed(1)}%`;
    
    // Create charts
    createSimulationPaths(results.simulation_paths);
    createReturnsDistribution(results.returns_distribution);
    createRiskMetricsChart(results.risk_metrics);
    
    // Show results
    document.getElementById('mc-results').style.display = 'block';
}

function createSimulationPaths(paths) {
    const traces = paths.map(pathData => ({
        x: pathData.data.map(d => d.day),
        y: pathData.data.map(d => d.value),
        type: 'scatter',
        mode: 'lines',
        line: { width: 1, color: 'rgba(156, 163, 175, 0.2)' },
        showlegend: false
    }));
    
    // Add mean path
    const meanPath = calculateMeanPath(paths);
    traces.push({
        x: meanPath.map(d => d.day),
        y: meanPath.map(d => d.value),
        type: 'scatter',
        mode: 'lines',
        line: { width: 3, color: '#f7931a' },
        name: 'Mean Path'
    });
    
    const layout = {
        ...window.chartManager.darkTheme,
        xaxis: { ...window.chartManager.darkTheme.xaxis, title: 'Days' },
        yaxis: { ...window.chartManager.darkTheme.yaxis, title: 'Portfolio Value' },
        height: 400
    };
    
    Plotly.newPlot('simulation-paths-chart', traces, layout, { displayModeBar: false });
}

function calculateMeanPath(paths) {
    if (paths.length === 0) return [];
    
    const days = paths[0].data.length;
    const meanPath = [];
    
    for (let i = 0; i < days; i++) {
        const sum = paths.reduce((acc, path) => acc + path.data[i].value, 0);
        meanPath.push({ day: i, value: sum / paths.length });
    }
    
    return meanPath;
}

function createReturnsDistribution(data) {
    const trace = {
        x: data.map(d => d.bin),
        y: data.map(d => d.count),
        type: 'bar',
        marker: { color: '#f7931a' }
    };
    
    const layout = {
        ...window.chartManager.darkTheme,
        xaxis: { ...window.chartManager.darkTheme.xaxis, title: 'Returns' },
        yaxis: { ...window.chartManager.darkTheme.yaxis, title: 'Frequency' },
        height: 300
    };
    
    Plotly.newPlot('returns-distribution-chart', [trace], layout, { displayModeBar: false });
}

function createRiskMetricsChart(metrics) {
    const trace = {
        x: ['5th', '25th', '50th', '75th', '95th'],
        y: [metrics.percentile_5, metrics.percentile_25, metrics.percentile_50, 
            metrics.percentile_75, metrics.percentile_95].map(v => v * 100),
        type: 'bar',
        marker: {
            color: ['#ef4444', '#f59e0b', '#3b82f6', '#10b981', '#22c55e']
        }
    };
    
    const layout = {
        ...window.chartManager.darkTheme,
        xaxis: { ...window.chartManager.darkTheme.xaxis, title: 'Percentile' },
        yaxis: { ...window.chartManager.darkTheme.yaxis, title: 'Return (%)' },
        height: 300
    };
    
    Plotly.newPlot('risk-metrics-chart', [trace], layout, { displayModeBar: false });
}

// Optimization functionality
function setupOptimizationForm() {
    const form = document.getElementById('optimization-form');
    if (form) {
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            await runOptimization();
        });
    }
}

function setupRangeSliders() {
    // Stop Loss Range
    const slMin = document.getElementById('stop-loss-min');
    const slMax = document.getElementById('stop-loss-max');
    const slDisplay = document.getElementById('sl-range-display');
    
    function updateSLDisplay() {
        slDisplay.textContent = `${slMin.value}% - ${slMax.value}%`;
    }
    
    slMin.addEventListener('input', updateSLDisplay);
    slMax.addEventListener('input', updateSLDisplay);
    
    // Take Profit Range
    const tpMin = document.getElementById('take-profit-min');
    const tpMax = document.getElementById('take-profit-max');
    const tpDisplay = document.getElementById('tp-range-display');
    
    function updateTPDisplay() {
        tpDisplay.textContent = `${tpMin.value}% - ${tpMax.value}%`;
    }
    
    tpMin.addEventListener('input', updateTPDisplay);
    tpMax.addEventListener('input', updateTPDisplay);
}

async function runOptimization() {
    const target = document.getElementById('optimization-target').value;
    const numTrials = parseInt(document.getElementById('num-trials').value);
    const stopLossRange = [
        parseFloat(document.getElementById('stop-loss-min').value) / 100,
        parseFloat(document.getElementById('stop-loss-max').value) / 100
    ];
    const takeProfitRange = [
        parseFloat(document.getElementById('take-profit-min').value) / 100,
        parseFloat(document.getElementById('take-profit-max').value) / 100
    ];
    
    // Show loading, hide results
    document.getElementById('opt-loading').style.display = 'block';
    document.getElementById('opt-results').style.display = 'none';
    
    try {
        const response = await window.apiClient.request('/analytics/api/optimization', {
            method: 'POST',
            body: JSON.stringify({
                target: target,
                num_trials: numTrials,
                stop_loss_range: stopLossRange,
                take_profit_range: takeProfitRange
            })
        });
        
        displayOptimizationResults(response);
    } catch (error) {
        console.error('Optimization error:', error);
        alert('Failed to run optimization. Please try again.');
    } finally {
        document.getElementById('opt-loading').style.display = 'none';
    }
}

function displayOptimizationResults(results) {
    // Update optimal parameters
    document.getElementById('optimal-stop-loss').textContent = `${(results.optimal_params.stop_loss * 100).toFixed(1)}%`;
    document.getElementById('optimal-take-profit').textContent = `${(results.optimal_params.take_profit * 100).toFixed(1)}%`;
    document.getElementById('expected-sortino').textContent = results.optimal_params.expected_sortino.toFixed(2);
    
    // Create charts
    createOptimizationHeatmap(results.heatmap_data);
    createConvergencePlot(results.convergence_data);
    
    // Update top combinations table
    const tbody = document.getElementById('top-combinations-body');
    tbody.innerHTML = results.top_combinations.map(combo => `
        <tr>
            <td>${(combo.stop_loss * 100).toFixed(1)}%</td>
            <td>${(combo.take_profit * 100).toFixed(1)}%</td>
            <td>${combo.sortino.toFixed(2)}</td>
        </tr>
    `).join('');
    
    // Show results
    document.getElementById('opt-results').style.display = 'block';
}

function createOptimizationHeatmap(data) {
    // Prepare data for heatmap
    const stopLosses = [...new Set(data.map(d => d.stop_loss))].sort((a, b) => a - b);
    const takeProfits = [...new Set(data.map(d => d.take_profit))].sort((a, b) => a - b);
    
    const z = [];
    for (let tp of takeProfits) {
        const row = [];
        for (let sl of stopLosses) {
            const point = data.find(d => d.stop_loss === sl && d.take_profit === tp);
            row.push(point ? point.sortino : 0);
        }
        z.push(row);
    }
    
    const trace = {
        x: stopLosses.map(sl => `${(sl * 100).toFixed(1)}%`),
        y: takeProfits.map(tp => `${(tp * 100).toFixed(1)}%`),
        z: z,
        type: 'heatmap',
        colorscale: 'Viridis'
    };
    
    const layout = {
        ...window.chartManager.darkTheme,
        xaxis: { ...window.chartManager.darkTheme.xaxis, title: 'Stop Loss' },
        yaxis: { ...window.chartManager.darkTheme.yaxis, title: 'Take Profit' },
        height: 400
    };
    
    Plotly.newPlot('optimization-heatmap', [trace], layout, { displayModeBar: false });
}

function createConvergencePlot(data) {
    const trace = {
        x: data.map(d => d.trial),
        y: data.map(d => d.best_value),
        type: 'scatter',
        mode: 'lines',
        line: { color: '#f7931a', width: 2 }
    };
    
    const layout = {
        ...window.chartManager.darkTheme,
        xaxis: { ...window.chartManager.darkTheme.xaxis, title: 'Trial' },
        yaxis: { ...window.chartManager.darkTheme.yaxis, title: 'Best Value' },
        height: 300
    };
    
    Plotly.newPlot('convergence-plot', [trace], layout, { displayModeBar: false });
}

// Data Quality functionality
async function loadDataQuality() {
    try {
        const response = await window.apiClient.request('/analytics/api/data-quality');
        displayDataQuality(response);
    } catch (error) {
        console.error('Failed to load data quality metrics:', error);
    }
}

function displayDataQuality(data) {
    // Update overall metrics
    document.getElementById('overall-score').textContent = `${data.overall_metrics.score.toFixed(1)}%`;
    document.getElementById('completeness-score').textContent = `${data.overall_metrics.completeness.toFixed(1)}%`;
    document.getElementById('accuracy-score').textContent = `${data.overall_metrics.accuracy.toFixed(1)}%`;
    document.getElementById('timeliness-score').textContent = `${data.overall_metrics.timeliness.toFixed(1)}%`;
    
    // Update source status table
    const sourceBody = document.getElementById('source-status-body');
    sourceBody.innerHTML = data.source_status.map(source => `
        <tr>
            <td>${source.source}</td>
            <td><span class="status-badge status-${source.status.toLowerCase()}">${source.status}</span></td>
            <td>${source.latency}</td>
            <td>${source.success_rate.toFixed(1)}%</td>
        </tr>
    `).join('');
    
    // Create quality timeline chart
    createQualityTimeline(data.quality_timeline);
    
    // Update missing data table
    const missingBody = document.getElementById('missing-data-body');
    missingBody.innerHTML = data.missing_data.map(field => `
        <tr>
            <td>${field.field}</td>
            <td>${field.missing_count}</td>
            <td>${field.missing_percentage.toFixed(1)}%</td>
        </tr>
    `).join('');
}

function createQualityTimeline(data) {
    const trace = {
        x: data.map(d => d.timestamp),
        y: data.map(d => d.score),
        type: 'scatter',
        mode: 'lines',
        line: { color: '#f7931a', width: 2 }
    };
    
    // Add threshold line
    const thresholdTrace = {
        x: [data[0].timestamp, data[data.length - 1].timestamp],
        y: [90, 90],
        type: 'scatter',
        mode: 'lines',
        line: { color: '#ef4444', width: 1, dash: 'dash' },
        name: 'Threshold'
    };
    
    const layout = {
        ...window.chartManager.darkTheme,
        xaxis: { ...window.chartManager.darkTheme.xaxis, title: 'Time' },
        yaxis: { ...window.chartManager.darkTheme.yaxis, title: 'Quality Score (%)' },
        height: 300
    };
    
    Plotly.newPlot('quality-timeline-chart', [trace, thresholdTrace], layout, { displayModeBar: false });
}