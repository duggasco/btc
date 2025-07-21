/**
 * Chart utilities and creators using Plotly
 */

class ChartManager {
    constructor() {
        this.darkTheme = {
            paper_bgcolor: 'rgba(19, 19, 21, 0)',
            plot_bgcolor: 'rgba(19, 19, 21, 0)',
            font: {
                color: '#e5e5e7',
                family: 'Inter, sans-serif',
                size: 12
            },
            margin: { l: 60, r: 30, t: 30, b: 40 },
            showlegend: true,
            legend: {
                bgcolor: 'rgba(26, 26, 29, 0.8)',
                bordercolor: '#27272a',
                borderwidth: 1
            },
            xaxis: {
                gridcolor: '#27272a',
                zerolinecolor: '#27272a',
                tickfont: { size: 11 }
            },
            yaxis: {
                gridcolor: '#27272a',
                zerolinecolor: '#27272a',
                tickfont: { size: 11 }
            }
        };
    }

    createCandlestickChart(elementId, data) {
        if (!data || data.length === 0) {
            this.showNoDataMessage(elementId);
            return;
        }

        const trace = {
            x: data.map(d => d.timestamp),
            open: data.map(d => d.open),
            high: data.map(d => d.high),
            low: data.map(d => d.low),
            close: data.map(d => d.close),
            type: 'candlestick',
            name: 'BTC/USD',
            increasing: { line: { color: '#22c55e' } },
            decreasing: { line: { color: '#ef4444' } }
        };

        const volumeTrace = {
            x: data.map(d => d.timestamp),
            y: data.map(d => d.volume),
            type: 'bar',
            name: 'Volume',
            yaxis: 'y2',
            marker: { color: 'rgba(156, 163, 175, 0.3)' }
        };

        const layout = {
            ...this.darkTheme,
            title: '',
            yaxis: { ...this.darkTheme.yaxis, title: 'Price (USD)' },
            yaxis2: {
                ...this.darkTheme.yaxis,
                title: 'Volume',
                overlaying: 'y',
                side: 'right',
                showgrid: false
            },
            hovermode: 'x unified',
            height: 400
        };

        const config = {
            displayModeBar: false,
            responsive: true
        };

        Plotly.newPlot(elementId, [trace, volumeTrace], layout, config);
    }

    createLineChart(elementId, data, options = {}) {
        if (!data || data.length === 0) {
            this.showNoDataMessage(elementId);
            return;
        }

        const trace = {
            x: data.map(d => d.x || d.timestamp),
            y: data.map(d => d.y || d.value),
            type: 'scatter',
            mode: 'lines',
            name: options.name || 'Value',
            line: {
                color: options.color || '#f7931a',
                width: 2
            }
        };

        const layout = {
            ...this.darkTheme,
            title: options.title || '',
            xaxis: { ...this.darkTheme.xaxis, title: options.xTitle || '' },
            yaxis: { ...this.darkTheme.yaxis, title: options.yTitle || '' },
            height: options.height || 300,
            showlegend: options.showlegend !== false
        };

        const config = {
            displayModeBar: false,
            responsive: true
        };

        Plotly.newPlot(elementId, [trace], layout, config);
    }

    createPortfolioChart(elementId, data) {
        if (!data || data.length === 0) {
            this.showNoDataMessage(elementId);
            return;
        }

        const trace = {
            x: data.map((_, i) => i),
            y: data,
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Portfolio Value',
            line: { color: '#f7931a', width: 2 },
            marker: { size: 4 }
        };

        // Add baseline
        const baseline = {
            x: [0, data.length - 1],
            y: [data[0], data[0]],
            type: 'scatter',
            mode: 'lines',
            name: 'Initial Value',
            line: { color: '#6b7280', width: 1, dash: 'dash' }
        };

        const layout = {
            ...this.darkTheme,
            title: '',
            xaxis: { ...this.darkTheme.xaxis, title: 'Time' },
            yaxis: { ...this.darkTheme.yaxis, title: 'Value ($)' },
            height: 200
        };

        const config = {
            displayModeBar: false,
            responsive: true
        };

        Plotly.newPlot(elementId, [trace, baseline], layout, config);
    }

    updateChart(elementId, data, traceIndex = 0) {
        const update = {
            x: [data.map(d => d.x || d.timestamp)],
            y: [data.map(d => d.y || d.value)]
        };

        Plotly.update(elementId, update, {}, [traceIndex]);
    }

    showNoDataMessage(elementId) {
        const element = document.getElementById(elementId);
        if (element) {
            element.innerHTML = '<div class="no-data-message">No data available</div>';
        }
    }

    clearChart(elementId) {
        Plotly.purge(elementId);
    }
}

// Create global chart manager instance
window.chartManager = new ChartManager();