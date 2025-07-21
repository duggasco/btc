/**
 * Dashboard-specific functionality
 */

// Global state
let currentTradingMode = 'live';
let paperTradingEnabled = false;

async function initializeDashboard() {
    // Check paper trading status
    await checkPaperTradingStatus();
    
    // Load initial data
    await loadDashboardData();
    
    // Set up event listeners
    setupEventListeners();
    
    // Initialize charts
    await initializeCharts();
    
    // Set up auto-refresh (every 5 seconds)
    setInterval(async () => {
        await loadDashboardData();
    }, 5000);
}

async function checkPaperTradingStatus() {
    try {
        const response = await fetch('/api/paper-trading/status');
        if (response.ok) {
            const data = await response.json();
            paperTradingEnabled = data.enabled;
            
            // Set the appropriate mode based on paper trading status
            if (paperTradingEnabled) {
                document.getElementById('mode-paper').checked = true;
                currentTradingMode = 'paper';
                updateTradingModeUI('paper');
            }
        }
    } catch (error) {
        console.error('Failed to check paper trading status:', error);
    }
}

async function loadDashboardData() {
    try {
        const data = await window.apiClient.getDashboardData();
        
        console.log('Dashboard data loaded:', data); // Debug log
        console.log('Price data:', data.price); // Debug log
        
        // Update price display
        updatePriceDisplay(data.price);
        
        // Update portfolio values based on trading mode
        if (currentTradingMode === 'paper' && data.paper_trading_enabled) {
            updatePortfolioSection(data.portfolio, true);
        } else {
            updatePortfolioSection(data.portfolio, false);
        }
        
        // Update recent trades
        updateRecentTrades(data.recent_trades);
        
        // Update open positions
        updateOpenPositions(data.positions);
        
        // Update signal history
        updateSignalHistory(data.signal);
        
        // Update current signal display
        updateCurrentSignal(data.signal);
        
        // Update performance metrics in header
        updatePerformanceMetrics(data.performance);
        
    } catch (error) {
        console.error('Failed to load dashboard data:', error);
        showError('Failed to load dashboard data');
    }
}

function updatePerformanceMetrics(performance) {
    // Update header win rate
    const winRate = document.getElementById('win-rate');
    if (winRate) {
        winRate.textContent = `${performance.win_rate.toFixed(1)}%`;
    }
    
    const totalTrades = document.getElementById('total-trades');
    if (totalTrades) {
        totalTrades.textContent = `${performance.total_trades} trades`;
    }
}

function updatePriceDisplay(priceData) {
    // Update header metrics bar (these are always visible)
    const headerPrice = document.getElementById('btc-price');
    if (headerPrice) {
        headerPrice.textContent = `$${formatNumber(priceData.current)}`;
    }
    
    const headerChange = document.getElementById('btc-change');
    if (headerChange) {
        const change = priceData.change_24h || 0;
        headerChange.textContent = `${change > 0 ? '+' : ''}${change.toFixed(2)}%`;
        headerChange.className = `metric-change ${change >= 0 ? 'positive' : 'negative'}`;
    }
    
    // Update main price display
    const currentPrice = document.getElementById('current-price-value');
    if (currentPrice) {
        currentPrice.textContent = `$${formatNumber(priceData.current)}`;
    }
    
    // Update 24h change
    const priceChange = document.getElementById('price-change-24h');
    if (priceChange) {
        const change = priceData.change_24h || 0;
        priceChange.textContent = `${change > 0 ? '+' : ''}${change.toFixed(2)}%`;
        priceChange.className = `price-change ${change >= 0 ? 'positive' : 'negative'}`;
    }
    
    // Update 24h high
    const high24h = document.getElementById('price-high-24h');
    if (high24h) {
        high24h.textContent = `$${formatNumber(priceData.high_24h)}`;
    }
    
    // Update 24h low
    const low24h = document.getElementById('price-low-24h');
    if (low24h) {
        low24h.textContent = `$${formatNumber(priceData.low_24h)}`;
    }
    
    // Update 24h volume
    const volume24h = document.getElementById('price-volume-24h');
    if (volume24h) {
        const volume = priceData.volume_24h || 0;
        if (volume > 1e9) {
            volume24h.textContent = `$${(volume / 1e9).toFixed(1)}B`;
        } else if (volume > 1e6) {
            volume24h.textContent = `$${(volume / 1e6).toFixed(1)}M`;
        } else {
            volume24h.textContent = `$${formatNumber(volume)}`;
        }
    }
    
    // Update timestamp
    const updateTime = document.getElementById('price-update-time');
    if (updateTime) {
        updateTime.textContent = formatTime(new Date());
    }
}

function updatePortfolioSection(portfolio, isPaperTrading = false) {
    // Update header portfolio metrics
    const headerPortfolioValue = document.getElementById('portfolio-value');
    if (headerPortfolioValue) {
        headerPortfolioValue.textContent = `$${formatNumber(portfolio.total_value)}`;
    }
    
    const headerPortfolioChange = document.getElementById('portfolio-change');
    if (headerPortfolioChange) {
        headerPortfolioChange.textContent = `${portfolio.change_24h > 0 ? '+' : ''}${portfolio.change_24h.toFixed(2)}%`;
        headerPortfolioChange.className = `metric-change ${portfolio.change_24h >= 0 ? 'positive' : 'negative'}`;
    }
    
    // Update dashboard portfolio section
    document.getElementById('portfolio-total-value').textContent = `$${formatNumber(portfolio.total_value)}`;
    document.getElementById('portfolio-change-value').textContent = `${portfolio.change_24h > 0 ? '+' : ''}${portfolio.change_24h.toFixed(2)}%`;
    document.getElementById('portfolio-change-value').className = `metric-card-change ${portfolio.change_24h >= 0 ? 'positive' : 'negative'}`;
    
    document.getElementById('portfolio-btc-balance').textContent = `₿${portfolio.btc_balance.toFixed(8)}`;
    document.getElementById('portfolio-usd-balance').textContent = `$${formatNumber(portfolio.usd_balance)}`;
    
    document.getElementById('portfolio-pnl').textContent = `$${formatNumber(portfolio.pnl)}`;
    document.getElementById('portfolio-pnl-percentage').textContent = `${portfolio.pnl_percentage > 0 ? '+' : ''}${portfolio.pnl_percentage.toFixed(2)}%`;
    document.getElementById('portfolio-pnl-percentage').className = `metric-card-change ${portfolio.pnl_percentage >= 0 ? 'positive' : 'negative'}`;
    
    // Show/hide paper trading badge
    const paperBadge = document.getElementById('portfolio-paper-badge');
    if (paperBadge) {
        paperBadge.style.display = isPaperTrading ? 'inline-block' : 'none';
    }
    
    // Update portfolio chart
    // Mock data for now - in production, this would come from the API
    const portfolioHistory = Array.from({length: 30}, (_, i) => 
        portfolio.total_value * (1 + (Math.random() - 0.5) * 0.1)
    );
    window.chartManager.createPortfolioChart('portfolio-chart', portfolioHistory);
}

function updateCurrentSignal(signal) {
    // Update header signal display
    const headerSignal = document.getElementById('current-signal');
    if (headerSignal) {
        headerSignal.textContent = signal.current;
        headerSignal.className = `signal-badge signal-${signal.current.toLowerCase()}`;
    }
    
    const headerConfidence = document.getElementById('signal-confidence');
    if (headerConfidence) {
        headerConfidence.textContent = `${signal.confidence.toFixed(1)}%`;
    }
    
    // Update dashboard signal display
    const signalBadge = document.getElementById('current-signal-badge');
    const confidenceValue = document.getElementById('signal-confidence-value');
    const priceValue = document.getElementById('signal-price-value');
    const timeValue = document.getElementById('signal-time-value');
    
    if (signalBadge) {
        signalBadge.textContent = signal.current;
        signalBadge.className = `signal-badge signal-${signal.current.toLowerCase()}`;
    }
    
    if (confidenceValue) {
        confidenceValue.textContent = `${signal.confidence.toFixed(1)}%`;
    }
    
    if (priceValue) {
        priceValue.textContent = `$${formatNumber(signal.price)}`;
    }
    
    if (timeValue) {
        timeValue.textContent = formatTime(signal.timestamp);
    }
}

function updateRecentTrades(trades) {
    const container = document.getElementById('recent-trades-list');
    container.innerHTML = trades.map(trade => `
        <div class="trade-item">
            <div class="trade-header">
                <span class="trade-type trade-${trade.type.toLowerCase()}">${trade.type}</span>
                <span class="trade-time">${formatTime(trade.timestamp)}</span>
            </div>
            <div class="trade-details">
                <span>₿${trade.quantity.toFixed(8)}</span>
                <span>@$${formatNumber(trade.price)}</span>
            </div>
        </div>
    `).join('');
}

function updateOpenPositions(positions) {
    const container = document.getElementById('open-positions-list');
    
    if (positions.length === 0) {
        container.innerHTML = '<div class="empty-state">No open positions</div>';
        return;
    }
    
    container.innerHTML = positions.map(position => `
        <div class="position-item">
            <div class="position-header">
                <span class="position-type">${position.type}</span>
                <span class="position-pnl ${position.pnl >= 0 ? 'positive' : 'negative'}">
                    ${position.pnl >= 0 ? '+' : ''}$${Math.abs(position.pnl).toFixed(2)}
                </span>
            </div>
            <div class="position-details">
                <div>Entry: $${formatNumber(position.entry_price)}</div>
                <div>Current: $${formatNumber(position.current_price)}</div>
                <div>Qty: ₿${position.quantity.toFixed(8)}</div>
            </div>
        </div>
    `).join('');
}

function updateSignalHistory(currentSignal) {
    // For now, just show the current signal
    // In production, this would show a history of signals
    const container = document.getElementById('signal-history-list');
    container.innerHTML = `
        <div class="signal-history-item">
            <span class="signal-badge signal-${currentSignal.current.toLowerCase()}">${currentSignal.current}</span>
            <span class="signal-time">${formatTime(currentSignal.timestamp)}</span>
        </div>
    `;
}

function setupEventListeners() {
    // Trade form submission
    const tradeForm = document.getElementById('trade-form');
    if (tradeForm) {
        tradeForm.addEventListener('submit', handleTradeSubmit);
    }
    
    // Chart timeframe selector
    const timeframeSelector = document.getElementById('chart-timeframe');
    if (timeframeSelector) {
        timeframeSelector.addEventListener('change', handleTimeframeChange);
    }
    
    // Order type buttons
    document.querySelectorAll('input[name="order-type"]').forEach(input => {
        input.addEventListener('change', updateOrderFormStyle);
    });
    
    // Trading mode toggle
    document.querySelectorAll('input[name="trading-mode"]').forEach(input => {
        input.addEventListener('change', handleTradingModeChange);
    });
}

async function handleTradingModeChange(event) {
    const newMode = event.target.value;
    const previousMode = currentTradingMode;
    
    if (newMode === previousMode) return;
    
    try {
        if (newMode === 'paper') {
            // Enable paper trading
            const response = await fetch('/api/paper-trading/toggle', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ action: 'enable' })
            });
            
            if (response.ok) {
                currentTradingMode = 'paper';
                updateTradingModeUI('paper');
                showSuccess('Paper trading enabled');
                await loadDashboardData();
            } else {
                // Revert the UI if the API call failed
                document.getElementById(`mode-${previousMode}`).checked = true;
                showError('Failed to enable paper trading');
            }
        } else {
            // Disable paper trading
            const response = await fetch('/api/paper-trading/toggle', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ action: 'disable' })
            });
            
            if (response.ok) {
                currentTradingMode = 'live';
                updateTradingModeUI('live');
                showSuccess('Switched to live trading mode');
                await loadDashboardData();
            } else {
                // Revert the UI if the API call failed
                document.getElementById(`mode-${previousMode}`).checked = true;
                showError('Failed to disable paper trading');
            }
        }
    } catch (error) {
        console.error('Trading mode change error:', error);
        document.getElementById(`mode-${previousMode}`).checked = true;
        showError('Failed to change trading mode');
    }
}

function updateTradingModeUI(mode) {
    // Update mode info visibility
    document.getElementById('live-mode-info').style.display = mode === 'live' ? 'block' : 'none';
    document.getElementById('paper-mode-info').style.display = mode === 'paper' ? 'block' : 'none';
    
    // Update order button text
    const orderBtnText = document.getElementById('order-btn-text');
    if (orderBtnText) {
        orderBtnText.textContent = mode === 'paper' ? 'Execute Paper Order' : 'Execute Order';
    }
    
    // Update form styling
    const tradeForm = document.getElementById('trade-form');
    if (tradeForm) {
        tradeForm.classList.toggle('paper-mode', mode === 'paper');
    }
}

async function handleTradeSubmit(event) {
    event.preventDefault();
    
    const orderType = document.querySelector('input[name="order-type"]:checked').value;
    const quantity = parseFloat(document.getElementById('order-quantity').value);
    const price = document.getElementById('order-price').value ? 
                  parseFloat(document.getElementById('order-price').value) : null;
    
    if (!quantity || quantity <= 0) {
        showError('Please enter a valid quantity');
        return;
    }
    
    // Show loading state
    const submitBtn = document.getElementById('execute-order-btn');
    const btnText = document.getElementById('order-btn-text');
    const btnSpinner = document.getElementById('order-btn-spinner');
    
    submitBtn.disabled = true;
    btnSpinner.style.display = 'inline-block';
    
    try {
        let response;
        
        if (currentTradingMode === 'paper') {
            // Execute paper trade
            response = await fetch('/api/paper-trading/trade', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    trade_type: orderType.toLowerCase(),
                    quantity: quantity,
                    price: price
                })
            });
            
            const data = await response.json();
            
            if (response.ok && data.success) {
                showSuccess(`Paper ${orderType} order executed successfully`);
                document.getElementById('trade-form').reset();
                await loadDashboardData();
            } else {
                showError(data.error || 'Paper trade execution failed');
            }
        } else {
            // Execute live trade
            response = await window.apiClient.executeTrade({
                type: orderType,
                quantity: quantity,
                price: price
            });
            
            if (response.success) {
                showSuccess(response.message);
                document.getElementById('trade-form').reset();
                await loadDashboardData();
            } else {
                showError(response.message || 'Trade execution failed');
            }
        }
    } catch (error) {
        console.error('Trade execution error:', error);
        showError('Failed to execute trade');
    } finally {
        // Reset button state
        submitBtn.disabled = false;
        btnSpinner.style.display = 'none';
    }
}

async function handleTimeframeChange(event) {
    const timeframe = event.target.value;
    await loadChartData(timeframe);
}

async function initializeCharts() {
    await loadChartData('1D');
}

async function loadChartData(timeframe) {
    try {
        const data = await window.apiClient.getChartData(timeframe);
        window.chartManager.createCandlestickChart('price-chart', data);
    } catch (error) {
        console.error('Failed to load chart data:', error);
        window.chartManager.showNoDataMessage('price-chart');
    }
}

function updateOrderFormStyle() {
    const orderType = document.querySelector('input[name="order-type"]:checked').value;
    const form = document.getElementById('trade-form');
    
    form.classList.remove('buy-mode', 'sell-mode');
    form.classList.add(orderType.toLowerCase() + '-mode');
}

function showMessage(message, type) {
    const messageDiv = document.getElementById('order-message');
    if (messageDiv) {
        messageDiv.textContent = message;
        messageDiv.className = `alert alert-${type}`;
        messageDiv.style.display = 'block';
        
        setTimeout(() => {
            messageDiv.style.display = 'none';
        }, 5000);
    }
}

function showError(message) {
    showMessage(message, 'danger');
}

function showSuccess(message) {
    showMessage(message, 'success');
}

function formatNumber(num) {
    return new Intl.NumberFormat('en-US', { 
        minimumFractionDigits: 2,
        maximumFractionDigits: 2 
    }).format(num);
}

function formatTime(timestamp) {
    const date = new Date(timestamp);
    return date.toLocaleTimeString('en-US', { 
        hour: '2-digit', 
        minute: '2-digit' 
    });
}