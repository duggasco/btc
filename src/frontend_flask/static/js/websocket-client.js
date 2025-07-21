/**
 * WebSocket client for real-time updates
 */

class WebSocketClient {
    constructor() {
        this.socket = null;
        this.connected = false;
        this.subscriptions = new Set();
        this.handlers = {};
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000; // Start with 1 second
    }

    connect() {
        // Connect to the Flask-SocketIO server
        this.socket = io({
            transports: ['websocket', 'polling'],
            reconnection: true,
            reconnectionAttempts: this.maxReconnectAttempts,
            reconnectionDelay: this.reconnectDelay,
            reconnectionDelayMax: 10000
        });

        // Connection event handlers
        this.socket.on('connect', () => {
            console.log('WebSocket connected');
            this.connected = true;
            this.reconnectAttempts = 0;
            this.updateConnectionStatus(true);
            
            // Re-subscribe to previous subscriptions
            if (this.subscriptions.size > 0) {
                this.socket.emit('subscribe', {
                    types: Array.from(this.subscriptions)
                });
            }
        });

        this.socket.on('disconnect', (reason) => {
            console.log('WebSocket disconnected:', reason);
            this.connected = false;
            this.updateConnectionStatus(false);
        });

        this.socket.on('connect_error', (error) => {
            console.error('WebSocket connection error:', error);
            this.reconnectAttempts++;
        });

        // Data event handlers
        this.socket.on('price_update', (data) => {
            this.handleUpdate('price', data);
        });

        this.socket.on('signal_update', (data) => {
            this.handleUpdate('signal', data);
        });

        this.socket.on('portfolio_update', (data) => {
            this.handleUpdate('portfolio', data);
        });

        this.socket.on('trade_update', (data) => {
            this.handleUpdate('trade', data);
        });

        this.socket.on('system_update', (data) => {
            this.handleUpdate('system', data);
        });
    }

    disconnect() {
        if (this.socket) {
            this.socket.disconnect();
            this.socket = null;
            this.connected = false;
        }
    }

    subscribe(updateType, handler) {
        // Add to subscriptions
        this.subscriptions.add(updateType);
        
        // Store handler
        if (!this.handlers[updateType]) {
            this.handlers[updateType] = [];
        }
        this.handlers[updateType].push(handler);
        
        // Send subscription to server if connected
        if (this.connected) {
            this.socket.emit('subscribe', { types: [updateType] });
        }
    }

    unsubscribe(updateType, handler) {
        // Remove from subscriptions
        this.subscriptions.delete(updateType);
        
        // Remove handler
        if (this.handlers[updateType]) {
            const index = this.handlers[updateType].indexOf(handler);
            if (index > -1) {
                this.handlers[updateType].splice(index, 1);
            }
            
            // If no more handlers, unsubscribe from server
            if (this.handlers[updateType].length === 0) {
                delete this.handlers[updateType];
                if (this.connected) {
                    this.socket.emit('unsubscribe', { types: [updateType] });
                }
            }
        }
    }

    handleUpdate(type, data) {
        // Call all registered handlers for this update type
        if (this.handlers[type]) {
            this.handlers[type].forEach(handler => {
                try {
                    handler(data.data, data.timestamp);
                } catch (error) {
                    console.error(`Error in ${type} handler:`, error);
                }
            });
        }
    }

    updateConnectionStatus(connected) {
        // Update UI connection indicators
        const wsIndicator = document.getElementById('websocket-status');
        if (wsIndicator) {
            wsIndicator.classList.toggle('online', connected);
            wsIndicator.classList.toggle('offline', !connected);
        }
        
        // Show notification
        if (connected) {
            this.showNotification('Connected to real-time updates', 'success');
        } else {
            this.showNotification('Real-time updates disconnected', 'warning');
        }
    }

    showNotification(message, type = 'info') {
        // Reuse the notification function from settings if available
        if (typeof showNotification === 'function') {
            showNotification(message, type);
        } else {
            console.log(`[${type.toUpperCase()}] ${message}`);
        }
    }
}

// Create global WebSocket client instance
window.wsClient = new WebSocketClient();

// Auto-connect when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Check if WebSocket is enabled
    const wsEnabled = document.body.dataset.websocketEnabled === 'true';
    if (wsEnabled) {
        window.wsClient.connect();
        
        // Example: Subscribe to price updates
        window.wsClient.subscribe('price', (data, timestamp) => {
            // Update price display
            const priceElement = document.getElementById('btc-price');
            if (priceElement && data.price) {
                priceElement.textContent = `$${formatNumber(data.price)}`;
            }
            
            // Update price change
            const changeElement = document.getElementById('btc-change');
            if (changeElement && data.change_24h !== undefined) {
                const change = data.change_24h;
                changeElement.textContent = `${change > 0 ? '+' : ''}${change.toFixed(2)}%`;
                changeElement.classList.toggle('positive', change >= 0);
                changeElement.classList.toggle('negative', change < 0);
            }
        });
        
        // Example: Subscribe to signal updates
        window.wsClient.subscribe('signal', (data, timestamp) => {
            // Update signal display
            const signalElement = document.getElementById('current-signal');
            if (signalElement && data.signal) {
                signalElement.textContent = data.signal.toUpperCase();
                signalElement.className = `signal-badge signal-${data.signal.toLowerCase()}`;
            }
            
            // Update confidence
            const confidenceElement = document.getElementById('signal-confidence');
            if (confidenceElement && data.confidence !== undefined) {
                confidenceElement.textContent = `${(data.confidence * 100).toFixed(0)}%`;
            }
            
            // Show notification for new signals
            if (data.signal && data.signal !== 'hold') {
                window.wsClient.showNotification(
                    `New ${data.signal.toUpperCase()} signal with ${(data.confidence * 100).toFixed(0)}% confidence`,
                    'info'
                );
            }
        });
        
        // Example: Subscribe to portfolio updates
        window.wsClient.subscribe('portfolio', (data, timestamp) => {
            // Update portfolio value
            const valueElement = document.getElementById('portfolio-value');
            if (valueElement && data.total_value !== undefined) {
                valueElement.textContent = `$${formatNumber(data.total_value)}`;
            }
            
            // Update portfolio change
            const changeElement = document.getElementById('portfolio-change');
            if (changeElement && data.change_24h !== undefined) {
                const change = data.change_24h;
                changeElement.textContent = `${change > 0 ? '+' : ''}${change.toFixed(2)}%`;
                changeElement.classList.toggle('positive', change >= 0);
                changeElement.classList.toggle('negative', change < 0);
            }
        });
        
        // Example: Subscribe to system updates
        window.wsClient.subscribe('system', (data, timestamp) => {
            // Update system status indicators
            if (data.api_status !== undefined) {
                const apiIndicator = document.getElementById('api-status');
                if (apiIndicator) {
                    apiIndicator.classList.toggle('online', data.api_status);
                    apiIndicator.classList.toggle('offline', !data.api_status);
                }
            }
            
            if (data.model_status !== undefined) {
                const modelIndicator = document.getElementById('model-status');
                if (modelIndicator) {
                    modelIndicator.classList.toggle('online', data.model_status);
                    modelIndicator.classList.toggle('offline', !data.model_status);
                }
            }
            
            if (data.trading_status !== undefined) {
                const tradingIndicator = document.getElementById('trading-status');
                if (tradingIndicator) {
                    tradingIndicator.classList.toggle('online', data.trading_status);
                    tradingIndicator.classList.toggle('offline', !data.trading_status);
                }
            }
        });
    }
});

// Helper function
function formatNumber(num) {
    return new Intl.NumberFormat('en-US', { 
        minimumFractionDigits: 2,
        maximumFractionDigits: 2 
    }).format(num);
}