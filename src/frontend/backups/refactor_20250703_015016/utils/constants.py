"""
Application constants and configuration
"""

# Time periods
TIME_PERIODS = {
    "1h": "1 Hour",
    "4h": "4 Hours", 
    "1d": "1 Day",
    "7d": "1 Week",
    "30d": "1 Month",
    "90d": "3 Months",
    "180d": "6 Months",
    "1y": "1 Year",
    "all": "All Time"
}

# Chart colors
CHART_COLORS = {
    "primary": "#1f77b4",
    "success": "#2ecc71",
    "danger": "#e74c3c",
    "warning": "#f39c12",
    "info": "#3498db",
    "bullish": "#00ff00",
    "bearish": "#ff0000",
    "neutral": "#808080"
}

# Trading constants
DEFAULT_POSITION_SIZE = 0.01
DEFAULT_STOP_LOSS_PCT = 5.0
DEFAULT_TAKE_PROFIT_PCT = 10.0
MIN_CONFIDENCE_THRESHOLD = 0.6

# API endpoints
ENDPOINTS = {
    "btc_latest": "/btc/latest",
    "btc_history": "/market/btc-data",
    "signals_latest": "/signals/enhanced/latest",
    "signals_history": "/signals/history",
    "portfolio_metrics": "/portfolio/metrics",
    "portfolio_positions": "/portfolio/positions",
    "trades_all": "/trades/all",
    "backtest_run": "/backtest/enhanced/run",
    "paper_trading_status": "/paper-trading/status"
}

# WebSocket channels
WS_CHANNELS = {
    "prices": "price_updates",
    "signals": "signal_updates",
    "trades": "trade_updates",
    "alerts": "alert_updates"
}
