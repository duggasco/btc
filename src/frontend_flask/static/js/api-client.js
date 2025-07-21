/**
 * API Client for Flask Frontend
 */

class APIClient {
    constructor(baseUrl = '') {
        this.baseUrl = baseUrl;
    }

    async request(endpoint, options = {}) {
        const url = `${this.baseUrl}${endpoint}`;
        const defaultOptions = {
            headers: {
                'Content-Type': 'application/json',
            },
        };

        try {
            const response = await fetch(url, { ...defaultOptions, ...options });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('API request failed:', error);
            throw error;
        }
    }

    // Dashboard endpoints
    async getDashboardData() {
        return this.request('/api/dashboard-data');
    }

    async getChartData(timeframe = '1D') {
        return this.request(`/api/chart-data?timeframe=${timeframe}`);
    }

    async executeTrade(tradeData) {
        return this.request('/api/execute-trade', {
            method: 'POST',
            body: JSON.stringify(tradeData),
        });
    }

    // Analytics endpoints
    async getBacktestResults(strategy, params) {
        return this.request('/api/backtest', {
            method: 'POST',
            body: JSON.stringify({ strategy, ...params }),
        });
    }

    async getOptimizationResults(strategy) {
        return this.request(`/api/optimization/${strategy}`);
    }

    // Settings endpoints
    async getSettings() {
        return this.request('/api/settings');
    }

    async updateSettings(settings) {
        return this.request('/api/settings', {
            method: 'PUT',
            body: JSON.stringify(settings),
        });
    }

    // System endpoints
    async getSystemStatus() {
        return this.request('/api/system/status');
    }

    async getConfig() {
        return this.request('/api/config');
    }
}

// Create global API client instance
window.apiClient = new APIClient();