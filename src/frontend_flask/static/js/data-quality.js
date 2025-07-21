/**
 * Data Quality Dashboard
 */

// Global variables
let qualityData = null;
let refreshInterval = null;

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    initializeQualityDashboard();
    loadQualityMetrics();
    
    // Set up auto-refresh every 30 seconds
    refreshInterval = setInterval(loadQualityMetrics, 30000);
});

function initializeQualityDashboard() {
    // Refresh button
    document.getElementById('refresh-quality-btn').addEventListener('click', () => {
        loadQualityMetrics();
    });
    
    // Heatmap type selector
    document.getElementById('heatmap-type').addEventListener('change', (e) => {
        updateCoverageHeatmap(e.target.value);
    });
}

async function loadQualityMetrics() {
    try {
        const response = await fetch('/data/api/quality-metrics');
        const data = await response.json();
        
        if (response.ok) {
            qualityData = data;
            updateQualityDisplay(data);
            updateCompletenessTable(data);
            updateCoverageHeatmap('ohlcv');
            updateGapsAnalysis(data);
            updateSourceQualityChart(data);
            updateCacheMetrics(data);
        } else {
            console.error('Failed to load quality metrics:', data);
        }
    } catch (error) {
        console.error('Error loading quality metrics:', error);
    }
}

function updateQualityDisplay(data) {
    // Update summary metrics
    document.getElementById('total-data-points').textContent = 
        formatNumber(data.summary?.total_records || 0);
    
    document.getElementById('data-coverage').textContent = 
        `${(data.summary?.overall_coverage || 0).toFixed(1)}%`;
    
    document.getElementById('data-sources').textContent = 
        data.summary?.total_sources || 0;
    
    document.getElementById('data-gaps').textContent = 
        data.gaps?.total_gaps || 0;
}

function updateCompletenessTable(data) {
    const tbody = document.querySelector('#completeness-table tbody');
    tbody.innerHTML = '';
    
    if (!data.completeness || data.completeness.length === 0) {
        tbody.innerHTML = '<tr><td colspan="7" class="text-center">No data available</td></tr>';
        return;
    }
    
    data.completeness.forEach(item => {
        const row = document.createElement('tr');
        const qualityClass = item.quality_score >= 90 ? 'text-success' : 
                           item.quality_score >= 70 ? 'text-warning' : 'text-danger';
        
        row.innerHTML = `
            <td>${item.data_type}</td>
            <td>${item.symbol || 'All'}</td>
            <td>${formatNumber(item.record_count)}</td>
            <td>${formatDateRange(item.date_range)}</td>
            <td>
                <div class="progress" style="width: 100px;">
                    <div class="progress-bar" style="width: ${item.coverage}%"></div>
                </div>
                <small>${item.coverage.toFixed(1)}%</small>
            </td>
            <td>${item.gaps || 0}</td>
            <td class="${qualityClass}">${item.quality_score.toFixed(1)}%</td>
        `;
        tbody.appendChild(row);
    });
}

function updateCoverageHeatmap(dataType) {
    if (!qualityData || !qualityData.coverage_heatmap) return;
    
    const heatmapData = qualityData.coverage_heatmap[dataType] || 
                       qualityData.coverage_heatmap.all;
    
    if (!heatmapData) {
        document.getElementById('coverage-heatmap').innerHTML = 
            '<p class="text-center">No coverage data available</p>';
        return;
    }
    
    // Create heatmap using Plotly
    const dates = Object.keys(heatmapData).sort();
    const hours = Array.from({length: 24}, (_, i) => i);
    
    const z = hours.map(hour => 
        dates.map(date => heatmapData[date]?.[hour] || 0)
    );
    
    const data = [{
        z: z,
        x: dates,
        y: hours.map(h => `${h}:00`),
        type: 'heatmap',
        colorscale: 'Viridis',
        hoverongaps: false,
        showscale: true,
        colorbar: {
            title: 'Coverage %',
            titleside: 'right'
        }
    }];
    
    const layout = {
        title: '',
        xaxis: {
            title: 'Date',
            tickangle: -45
        },
        yaxis: {
            title: 'Hour of Day'
        },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#9ca3af' },
        margin: { l: 60, r: 20, t: 20, b: 60 }
    };
    
    Plotly.newPlot('coverage-heatmap', data, layout, {responsive: true});
}

function updateGapsAnalysis(data) {
    const summaryText = document.getElementById('gaps-summary-text');
    const tbody = document.querySelector('#gaps-table tbody');
    
    if (!data.gaps || data.gaps.details.length === 0) {
        summaryText.textContent = 'No data gaps found. All data is continuous.';
        tbody.innerHTML = '<tr><td colspan="6" class="text-center">No gaps found</td></tr>';
        return;
    }
    
    summaryText.textContent = `Found ${data.gaps.total_gaps} gaps totaling ${data.gaps.total_missing_days} missing days.`;
    
    tbody.innerHTML = '';
    data.gaps.details.forEach(gap => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${gap.data_type}</td>
            <td>${gap.symbol}</td>
            <td>${formatDate(gap.gap_start)}</td>
            <td>${formatDate(gap.gap_end)}</td>
            <td>${gap.missing_days}</td>
            <td>
                <button class="btn btn-sm btn-outline-primary" onclick="fillGap('${gap.id}')">
                    <i class="fas fa-fill"></i> Fill Gap
                </button>
            </td>
        `;
        tbody.appendChild(row);
    });
}

function updateSourceQualityChart(data) {
    if (!data.source_quality || data.source_quality.length === 0) {
        document.getElementById('source-quality-chart').innerHTML = 
            '<p class="text-center">No source quality data available</p>';
        return;
    }
    
    const sources = data.source_quality.map(s => s.source);
    const scores = data.source_quality.map(s => s.quality_score);
    const records = data.source_quality.map(s => s.record_count);
    
    const trace1 = {
        x: sources,
        y: scores,
        name: 'Quality Score',
        type: 'bar',
        marker: {
            color: scores.map(score => 
                score >= 90 ? '#22c55e' : 
                score >= 70 ? '#f59e0b' : '#ef4444'
            )
        }
    };
    
    const trace2 = {
        x: sources,
        y: records,
        name: 'Record Count',
        type: 'scatter',
        mode: 'lines+markers',
        yaxis: 'y2',
        line: { color: '#3b82f6' }
    };
    
    const layout = {
        title: 'Quality Score by Data Source',
        xaxis: { title: 'Source' },
        yaxis: { 
            title: 'Quality Score (%)',
            range: [0, 100]
        },
        yaxis2: {
            title: 'Record Count',
            overlaying: 'y',
            side: 'right'
        },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#9ca3af' },
        showlegend: true,
        legend: {
            x: 0,
            y: 1,
            bgcolor: 'rgba(0,0,0,0)'
        }
    };
    
    Plotly.newPlot('source-quality-chart', [trace1, trace2], layout, {responsive: true});
}

function updateCacheMetrics(data) {
    if (!data.cache_performance) {
        return;
    }
    
    const cache = data.cache_performance;
    
    document.getElementById('cache-hit-rate').textContent = 
        `${(cache.hit_rate || 0).toFixed(1)}%`;
    
    document.getElementById('cache-total-requests').textContent = 
        formatNumber(cache.total_requests || 0);
    
    document.getElementById('cache-size').textContent = 
        `${(cache.cache_size_mb || 0).toFixed(1)} MB`;
    
    document.getElementById('cache-response-time').textContent = 
        `${(cache.avg_response_time || 0).toFixed(0)} ms`;
}

// Helper functions
function formatNumber(num) {
    return new Intl.NumberFormat('en-US').format(num);
}

function formatDate(dateStr) {
    if (!dateStr) return 'N/A';
    const date = new Date(dateStr);
    return date.toLocaleDateString('en-US', { 
        year: 'numeric', 
        month: 'short', 
        day: 'numeric' 
    });
}

function formatDateRange(range) {
    if (!range || !range.start || !range.end) return 'N/A';
    return `${formatDate(range.start)} - ${formatDate(range.end)}`;
}

async function fillGap(gapId) {
    // This would trigger a gap-filling process
    console.log('Fill gap:', gapId);
    alert('Gap filling functionality would be implemented here');
}

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (refreshInterval) {
        clearInterval(refreshInterval);
    }
});