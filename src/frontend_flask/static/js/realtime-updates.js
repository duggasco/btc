/**
 * Real-time updates manager
 */

class RealtimeUpdates {
    constructor() {
        this.updateInterval = 5000; // 5 seconds for fallback polling
        this.updateTimer = null;
        this.useWebSocket = document.body.dataset.websocketEnabled === 'true';
    }

    start() {
        if (this.useWebSocket && window.wsClient && window.wsClient.connected) {
            // WebSocket is handling real-time updates
            console.log('Using WebSocket for real-time updates');
        } else {
            // Fallback to polling
            this.startPolling();
        }
        
        // Initial update
        this.updateMetrics();
    }

    stop() {
        if (this.updateTimer) {
            clearInterval(this.updateTimer);
            this.updateTimer = null;
        }
    }

    startPolling() {
        this.updateTimer = setInterval(() => {
            this.updateMetrics();
        }, this.updateInterval);
    }

    async updateMetrics() {
        try {
            const data = await window.apiClient.getDashboardData();
            
            console.log('Realtime update - Price data:', data.price); // Debug log
            
            // Update price metrics
            this.updateElement('btc-price', `$${this.formatNumber(data.price.current)}`);
            this.updateElement('btc-change', `${data.price.change_24h > 0 ? '+' : ''}${data.price.change_24h.toFixed(2)}%`, 
                             data.price.change_24h >= 0);
            
            // Update signal
            this.updateSignal(data.signal);
            
            // Update portfolio metrics
            this.updateElement('portfolio-value', `$${this.formatNumber(data.portfolio.total_value)}`);
            this.updateElement('portfolio-change', `${data.portfolio.change_24h > 0 ? '+' : ''}${data.portfolio.change_24h.toFixed(2)}%`,
                             data.portfolio.change_24h >= 0);
            
            // Update performance metrics
            this.updateElement('win-rate', `${data.performance.win_rate.toFixed(1)}%`);
            this.updateElement('total-trades', `${data.performance.total_trades} trades`);
            
            // Update system status
            this.updateSystemStatus(data.system_status);
            
        } catch (error) {
            console.error('Failed to update metrics:', error);
        }
    }

    updateElement(id, value, isPositive = null) {
        const element = document.getElementById(id);
        if (element) {
            element.textContent = value;
            
            if (isPositive !== null) {
                element.classList.remove('positive', 'negative');
                element.classList.add(isPositive ? 'positive' : 'negative');
            }
        }
    }

    updateSignal(signal) {
        const badge = document.getElementById('current-signal');
        if (badge) {
            badge.textContent = signal.current;
            badge.className = `signal-badge signal-${signal.current.toLowerCase()}`;
        }
        
        this.updateElement('signal-confidence', `${signal.confidence}%`);
    }

    updateSystemStatus(status) {
        const updateStatus = (id, isOnline) => {
            const element = document.getElementById(id);
            if (element) {
                element.classList.remove('online', 'offline');
                element.classList.add(isOnline ? 'online' : 'offline');
            }
        };
        
        updateStatus('api-status', status.api);
        updateStatus('database-status', status.database);
        updateStatus('model-status', status.model);
        updateStatus('trading-status', status.trading);
        updateStatus('websocket-status', status.websocket);
    }

    formatNumber(num) {
        return new Intl.NumberFormat('en-US', { 
            minimumFractionDigits: 2,
            maximumFractionDigits: 2 
        }).format(num);
    }
}

// Create global realtime updates instance
window.realtimeUpdates = new RealtimeUpdates();

// Start updates when DOM is loaded
// Only start if we're not on the dashboard page (dashboard has its own updater)
document.addEventListener('DOMContentLoaded', () => {
    if (!document.getElementById('price-chart')) {  // Dashboard has price-chart element
        window.realtimeUpdates.start();
    }
});