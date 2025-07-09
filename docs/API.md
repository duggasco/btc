# BTC Trading System API Documentation - UltraThink Enhanced

Base URL: `http://localhost:8080`

## Table of Contents
- [WebSocket Support](#websocket-support)
- [Core Endpoints](#core-endpoints)
- [Trading Signals](#trading-signals)
- [Trading Operations](#trading-operations)
- [Portfolio Management](#portfolio-management)
- [Paper Trading](#paper-trading)
- [Market Data](#market-data)
- [Analytics](#analytics)
- [Backtesting](#backtesting)
- [Configuration](#configuration)
- [System Management](#system-management)

## WebSocket Support

### WS /ws
Real-time updates via WebSocket connection.

**Connection:**
```javascript
const ws = new WebSocket('ws://localhost:8000/ws');
```

**Subscribe to updates:**
```json
// Subscribe to signal updates
{"action": "subscribe_signals"}

// Subscribe to price updates
{"action": "subscribe_prices"}

// Ping to keep alive
{"action": "ping"}
```

**Message types received:**
```json
// Signal update
{
  "type": "signal_update",
  "data": {
    "signal": "buy",
    "confidence": 0.85,
    "predicted_price": 45250.00,
    "timestamp": "2025-01-01T12:00:00"
  }
}

// Price update
{
  "type": "price_update",
  "data": {
    "price": 45000.00,
    "timestamp": "2025-01-01T12:00:00"
  }
}
```

## Core Endpoints

### GET /
Health check endpoint.
```json
{
  "message": "Enhanced BTC Trading System API is running",
  "version": "2.1.0",
  "timestamp": "2025-01-01T12:00:00",
  "status": "healthy",
  "signal_errors": 0,
  "features": ["enhanced_signals", "comprehensive_backtesting", "50+_indicators", "websocket_support", "paper_trading"]
}
```

### GET /health
Detailed system health with component status.
```json
{
  "status": "healthy",
  "timestamp": "2025-01-01T12:00:00",
  "components": {
    "database": "healthy",
    "signal_generator": "healthy",
    "enhanced_signals": "active",
    "signal_update_errors": 0,
    "comprehensive_signals": "active",
    "paper_trading": "enabled",
    "websocket_connections": 3
  },
  "latest_signal": {
    "symbol": "BTC-USD",
    "signal": "buy",
    "confidence": 0.75,
    "predicted_price": 45250.00,
    "timestamp": "2025-01-01T12:00:00"
  },
  "enhanced_features": {
    "macro_indicators": true,
    "sentiment_analysis": true,
    "on_chain_proxies": true,
    "50_plus_signals": true,
    "websocket_streaming": true,
    "paper_trading": true,
    "rate_limiting": true
  }
}
```

## Trading Signals

### GET /signals/enhanced/latest
Get the latest enhanced trading signal with full analysis.
```json
{
  "symbol": "BTC-USD",
  "signal": "buy",
  "confidence": 0.85,
  "predicted_price": 45250.00,
  "timestamp": "2025-01-01T12:00:00",
  "analysis": {
    "consensus_ratio": 0.8,
    "price_confidence_interval": [44500, 46000],
    "signal_distribution": {
      "buy": 8,
      "sell": 1,
      "hold": 1
    },
    "feature_importance": {
      "rsi": 0.152,
      "macd": 0.143,
      "volume_spike": 0.098
    }
  },
  "comprehensive_signals": {
    "technical": {
      "rsi": 35.2,
      "macd_signal": true,
      "bb_position": 0.25
    },
    "sentiment": {
      "fear_proxy": 0.3,
      "greed_proxy": 0.7
    },
    "on_chain": {
      "nvt_proxy": 0.8,
      "accumulation_proxy": 0.65
    }
  }
}
```

### GET /signals/comprehensive
Get all 50+ calculated signals categorized by type.
```json
{
  "timestamp": "2025-01-01T12:00:00",
  "total_signals": 52,
  "categorized_signals": {
    "technical": {
      "rsi": 35.2,
      "macd": 0.15,
      "bb_position": 0.25,
      "atr_normalized": 0.03,
      "stoch_k": 22.5,
      "adx": 35.0
    },
    "momentum": {
      "roc": 5.2,
      "tsi": 0.3,
      "williams_r": -75
    },
    "volatility": {
      "bb_width": 0.05,
      "atr": 1250,
      "volatility_ratio": 1.2
    },
    "volume": {
      "obv_trend": 1.05,
      "cmf": 0.15,
      "mfi": 45
    },
    "sentiment": {
      "fear_greed_index": 45,
      "twitter_sentiment": 0.6,
      "reddit_sentiment": 0.55
    },
    "on_chain": {
      "active_addresses": 850000,
      "nvt_ratio": 65,
      "exchange_net_flow": -5000
    }
  }
}
```

### GET /signals/latest
Get the latest basic trading signal (backward compatibility).
```json
{
  "symbol": "BTC-USD",
  "signal": "buy",
  "confidence": 0.75,
  "predicted_price": 45250.00,
  "timestamp": "2025-01-01T12:00:00"
}
```

### GET /signals/history?hours=24&limit=50
Get historical signals.
```json
[
  {
    "id": 1,
    "symbol": "BTC-USD",
    "signal": "buy",
    "confidence": 0.75,
    "price_prediction": 45250.00,
    "timestamp": "2025-01-01T12:00:00"
  }
]
```

## Trading Operations

### POST /trades/
Create a new trade.
```json
// Request
{
  "symbol": "BTC-USD",
  "trade_type": "buy",
  "price": 45000.00,
  "size": 0.1,
  "lot_id": "optional_lot_id",
  "notes": "AI signal triggered"
}

// Response
{
  "trade_id": "uuid",
  "status": "success",
  "message": "Trade created successfully",
  "pnl": 0
}
```

### GET /trades/?symbol=BTC-USD&limit=100
Get trading history.
```json
[
  {
    "id": "uuid",
    "symbol": "BTC-USD",
    "trade_type": "buy",
    "price": 45000.00,
    "size": 0.1,
    "timestamp": "2025-01-01T12:00:00",
    "status": "executed",
    "pnl": 0,
    "notes": "AI signal triggered"
  }
]
```

### POST /trades/execute
Execute a trade based on current signal.
```json
// Request
{
  "signal": "buy",
  "size": 0.001,
  "reason": "manual"
}

// Response
{
  "trade_id": "uuid",
  "status": "success",
  "message": "Trade executed"
}
```

### GET /trades/recent?limit=10
Get recent trades with formatted data.

### POST /limits/
Create a limit order.
```json
// Request
{
  "symbol": "BTC-USD",
  "limit_type": "stop_loss",
  "price": 40000.00,
  "size": 0.1,
  "lot_id": "optional_lot_id"
}

// Response
{
  "limit_id": "uuid",
  "status": "success",
  "message": "Limit order created"
}
```

### GET /limits/
Get active limit orders.

## Portfolio Management

### GET /portfolio/metrics
Get comprehensive portfolio metrics.
```json
{
  "total_trades": 25,
  "total_pnl": 1250.75,
  "positions_count": 3,
  "total_invested": 10000.00,
  "current_btc_price": 45000.00,
  "signal_confidence": 0.75,
  "consensus_ratio": 0.8,
  "total_realized_pnl": 1000.00,
  "total_unrealized_pnl": 250.75,
  "trades_per_day": 2.5,
  "win_rate": 0.65,
  "avg_signal_confidence": 0.72,
  "signal_distribution": {
    "buy": 15,
    "sell": 8,
    "hold": 2
  }
}
```

### GET /portfolio/positions
Get current positions with P&L calculations.

### GET /analytics/portfolio-comprehensive
Get enhanced portfolio analytics with additional metrics.

## Paper Trading

### GET /paper-trading/status
Get paper trading status and portfolio.
```json
{
  "enabled": true,
  "portfolio": {
    "id": 1,
    "btc_balance": 0.12345,
    "usd_balance": 5432.10,
    "total_pnl": 432.10,
    "trades": [
      {
        "id": "uuid",
        "timestamp": "2025-01-01T12:00:00",
        "type": "buy",
        "price": 45000,
        "amount": 0.1,
        "value": 4500
      }
    ]
  },
  "performance": {
    "total_return": 0.0432,
    "win_rate": 0.65,
    "sharpe_ratio": 1.85,
    "max_drawdown": -0.12,
    "trades_count": 25
  },
  "timestamp": "2025-01-01T12:00:00"
}
```

### POST /paper-trading/toggle
Toggle paper trading on/off.
```json
// Response
{
  "enabled": true,
  "message": "Paper trading enabled"
}
```

### POST /paper-trading/reset
Reset paper trading portfolio to initial $10,000.
```json
// Response
{
  "status": "success",
  "message": "Paper trading portfolio reset"
}
```

### GET /paper-trading/history?days=30
Get paper trading performance history.
```json
[
  {
    "timestamp": "2025-01-01T12:00:00",
    "total_value": 10432.10,
    "daily_pnl": 125.50,
    "win_rate": 0.65,
    "sharpe_ratio": 1.85,
    "max_drawdown": -0.12
  }
]
```

## Market Data

### GET /market/btc-data?period=1mo&include_indicators=true
Get BTC market data with optional enhanced indicators.
```json
{
  "symbol": "BTC-USD",
  "period": "1mo",
  "data": [
    {
      "timestamp": "2025-01-01T00:00:00",
      "Date": "2025-01-01T00:00:00",
      "open": 44800.00,
      "high": 45200.00,
      "low": 44500.00,
      "close": 45000.00,
      "volume": 1500.25,
      "sma_20": 44750.00,
      "rsi": 65.5,
      "macd": 125.75,
      "bb_position": 0.65,
      "fear_proxy": 0.4,
      "momentum_sentiment": 0.7
    }
  ],
  "total_records": 30,
  "enhanced": true
}
```

### GET /btc/latest
Get latest BTC price.
```json
{
  "latest_price": 45000.00,
  "timestamp": "2025-01-01T12:00:00"
}
```

### GET /indicators/all
Get all calculated technical indicators.
```json
{
  "rsi": 35.2,
  "macd": 0.15,
  "bb_position": 0.25,
  "atr_normalized": 0.03,
  "stoch_k": 22.5,
  "adx": 35.0,
  "fear_greed_value": 45,
  "nvt_ratio": 65,
  // ... 50+ more indicators
}
```

## Analytics

### GET /analytics/performance
Get comprehensive performance metrics.
```json
{
  "total_return": 0.125,
  "sharpe_ratio": 1.85,
  "sortino_ratio": 2.15,
  "max_drawdown": -0.18,
  "win_rate": 0.65,
  "profit_factor": 1.8,
  "total_trades": 125,
  "calmar_ratio": 0.69,
  "omega_ratio": 1.5,
  "equity_curve": [10000, 10125, 10089, ...]
}
```

### GET /analytics/risk
Get risk analysis metrics.
```json
{
  "var_95": -0.025,
  "var_99": -0.045,
  "cvar_95": -0.032,
  "cvar_99": -0.058,
  "returns_distribution": [-0.01, 0.02, -0.005, ...]
}
```

### GET /analytics/correlations
Get market correlations analysis.
```json
{
  "correlation_matrix": {
    "rsi": {"macd": 0.65, "volume": 0.32},
    "macd": {"rsi": 0.65, "bb_position": 0.45}
  },
  "key_correlations": {
    "sp500_returns": 0.42,
    "gold_returns": -0.15,
    "dxy_returns": -0.38
  }
}
```

### GET /analytics/optimization
Get strategy optimization results.
```json
{
  "optimal_weights": {
    "technical_weight": 0.40,
    "onchain_weight": 0.35,
    "sentiment_weight": 0.15,
    "macro_weight": 0.10,
    "momentum_weight": 0.30,
    "trend_weight": 0.40,
    "volatility_weight": 0.15,
    "volume_weight": 0.15
  },
  "optimization_history": [
    {"trial": 1, "value": 1.5},
    {"trial": 2, "value": 1.8}
  ]
}
```

### GET /analytics/feature-importance
Get ML feature importance analysis.
```json
{
  "feature_importance": {
    "rsi": 0.152,
    "macd": 0.143,
    "volume_spike": 0.098,
    "fear_greed": 0.087
  },
  "top_10_features": {
    "rsi": 0.152,
    "macd": 0.143
  },
  "timestamp": "2025-01-01T12:00:00"
}
```

### POST /analytics/monte-carlo
Run Monte Carlo simulation for risk assessment.
```json
// Request
{
  "num_simulations": 1000,
  "time_horizon_days": 30
}

// Response
{
  "current_price": 45000.00,
  "simulations": 1000,
  "time_horizon_days": 30,
  "statistics": {
    "mean": 46500.00,
    "std": 2500.00,
    "min": 38000.00,
    "max": 55000.00,
    "percentiles": {
      "5%": 41000.00,
      "25%": 44000.00,
      "50%": 46500.00,
      "75%": 49000.00,
      "95%": 52000.00
    }
  },
  "probability_profit": 0.68
}
```

## Backtesting

### POST /backtest/enhanced/run
Run enhanced backtest with all features.
```json
// Request
{
  "period": "1y",
  "optimize_weights": true,
  "include_macro": true,
  "use_enhanced_weights": true,
  "n_optimization_trials": 50,
  "force": false,
  "settings": {
    "training_window_days": 1008,
    "test_window_days": 90,
    "transaction_cost": 0.0025
  }
}

// Response
{
  "status": "success",
  "backtest_id": 123,
  "summary": {
    "composite_score": 0.75,
    "confidence_score": 0.82,
    "key_metrics": {
      "sortino_ratio": 2.15,
      "max_drawdown": -0.18,
      "total_return": 0.125
    }
  }
}
```

### GET /backtest/enhanced/results/latest
Get the most recent enhanced backtest results with full details.

### GET /backtest/status
Get current backtest status.
```json
{
  "in_progress": false,
  "has_results": true,
  "timestamp": "2025-01-01T12:00:00",
  "system_type": "enhanced"
}
```

### GET /backtest/results/history?limit=10
Get historical backtest results.

### GET /backtest/walk-forward/results
Get walk-forward analysis results.
```json
{
  "total_windows": 10,
  "avg_return": 0.025,
  "consistency_score": 0.85,
  "window_results": [
    {"window": 1, "return": 0.03, "sharpe": 1.8}
  ],
  "stability_metrics": {
    "return_std": 0.008,
    "sharpe_std": 0.3,
    "max_window_dd": -0.12,
    "signal_consistency": 0.78,
    "weight_stability": 0.92,
    "feature_var": 0.05
  }
}
```

### GET /backtest/optimization/results
Get optimization results from Bayesian optimization.

## Configuration

### GET /config/signal-weights/enhanced
Get current enhanced signal weights including sub-categories.
```json
{
  "main_categories": {
    "technical": 0.40,
    "onchain": 0.35,
    "sentiment": 0.15,
    "macro": 0.10
  },
  "technical_sub": {
    "momentum": 0.30,
    "trend": 0.40,
    "volatility": 0.15,
    "volume": 0.15
  },
  "onchain_sub": {
    "flow": 0.40,
    "network": 0.30,
    "holder": 0.30
  },
  "sentiment_sub": {
    "social": 0.50,
    "derivatives": 0.30,
    "fear_greed": 0.20
  }
}
```

### POST /config/signal-weights
Update signal weights.
```json
// Request
{
  "technical_weight": 0.40,
  "onchain_weight": 0.35,
  "sentiment_weight": 0.15,
  "macro_weight": 0.10,
  "momentum_weight": 0.30,
  "trend_weight": 0.40
}
```

### GET /config/model
Get model configuration.

### POST /config/model
Update model configuration.

### GET /config/trading-rules
Get trading rules configuration.

### POST /config/trading-rules
Update trading rules.

### GET /config/backtest-settings
Get backtest configuration settings.

## System Management

### GET /system/status
Get detailed system status.
```json
{
  "api_status": "running",
  "api_version": "2.1.0",
  "timestamp": "2025-01-01T12:00:00",
  "signal_update_errors": 0,
  "latest_signal_time": "2025-01-01T12:00:00",
  "enhanced_signal_time": "2025-01-01T12:00:00",
  "data_cache_status": "available",
  "comprehensive_signals_status": "available",
  "database_status": "connected",
  "signal_generator_status": "enhanced",
  "backtest_system_status": "enhanced",
  "enhanced_features": {
    "50_plus_signals": true,
    "macro_indicators": true,
    "sentiment_analysis": true,
    "on_chain_proxies": true,
    "enhanced_backtesting": true,
    "bayesian_optimization": true,
    "feature_importance": true,
    "confidence_intervals": true,
    "websocket_support": true,
    "paper_trading": true,
    "monte_carlo": true
  },
  "paper_trading_status": {
    "enabled": true,
    "portfolio_value": 10432.10
  }
}
```

### POST /model/retrain/enhanced
Manually trigger enhanced model retraining.
```json
// Response
{
  "status": "success",
  "message": "Enhanced model retraining completed",
  "timestamp": "2025-01-01T12:00:00",
  "results": {
    "training_time_seconds": 125.5,
    "data_points": 1000
  }
}
```

### GET /model/info
Get model information.

### GET /database/stats
Get database statistics.

### GET /database/export
Export database data.

### POST /trading/start
Start automated trading.

### POST /trading/stop
Stop automated trading.

### GET /trading/status
Get trading status.

## Error Responses

### 400 Bad Request
```json
{
  "detail": "Invalid trade data"
}
```

### 404 Not Found
```json
{
  "detail": "Endpoint not found"
}
```

### 429 Too Many Requests
```json
{
  "detail": "Rate limit exceeded"
}
```

### 500 Internal Server Error
```json
{
  "detail": "Database connection failed"
}
```

## Rate Limits
- Default: 100 requests per minute
- WebSocket: No rate limit
- Backtest endpoints: 10 requests per hour

## Authentication
No authentication required for local deployment. Add authentication before deploying to production.

## Data Sources
The system fetches real data from multiple sources:
- **Crypto prices**: CoinGecko, Binance, CryptoCompare
- **On-chain data**: Blockchain.info, Blockchair
- **Sentiment**: Alternative.me (Fear & Greed), Reddit, NewsAPI
- **Macro data**: Yahoo Finance, FRED (with API key)

## Example Usage

### Python
```python
import requests

# Get latest enhanced signal
response = requests.get("http://localhost:8080/signals/enhanced/latest")
signal_data = response.json()

# Execute a trade
trade_data = {
    "symbol": "BTC-USD",
    "trade_type": "buy",
    "price": 45000.00,
    "size": 0.1
}
response = requests.post("http://localhost:8080/trades/", json=trade_data)
```

### JavaScript (with WebSocket)
```javascript
// REST API
fetch('http://localhost:8080/signals/enhanced/latest')
  .then(response => response.json())
  .then(data => console.log(data));

// WebSocket for real-time updates
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onopen = () => {
  ws.send(JSON.stringify({action: 'subscribe_signals'}));
  ws.send(JSON.stringify({action: 'subscribe_prices'}));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Received:', data);
};
```

### cURL
```bash
# Get enhanced signal
curl http://localhost:8080/signals/enhanced/latest

# Create trade
curl -X POST http://localhost:8080/trades/ \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "BTC-USD",
    "trade_type": "buy",
    "price": 45000.00,
    "size": 0.1
  }'

# Toggle paper trading
curl -X POST http://localhost:8080/paper-trading/toggle
```

## Cache Management

The system includes a comprehensive SQLite-based caching layer to reduce API calls and improve performance.

### GET /cache/stats
Get detailed cache statistics including hit rates, size, and performance metrics.

**Response:**
```json
{
  "total_entries": 150,
  "total_size_mb": 2.5,
  "session_stats": {
    "hits": 1250,
    "misses": 45,
    "writes": 195,
    "hit_rate": 0.965
  },
  "cache_efficiency": {
    "total_api_calls_saved": 1250,
    "estimated_time_saved_seconds": 625.0
  }
}
```

### GET /cache/entries
Query cache entries with optional filtering.

**Query Parameters:**
- `data_type` (optional): Filter by data type (e.g., 'ohlcv', 'price', 'sentiment')
- `api_source` (optional): Filter by API source (e.g., 'binance', 'coingecko')
- `limit` (optional): Maximum entries to return (default: 100)

### POST /cache/invalidate
Invalidate cache entries matching specified criteria.

**Query Parameters:**
- `pattern` (optional): Pattern to match cache keys
- `data_type` (optional): Invalidate by data type
- `api_source` (optional): Invalidate by API source
- `reason` (optional): Reason for invalidation

### POST /cache/clear-expired
Remove all expired cache entries.

### POST /cache/optimize
Optimize cache by removing low-value entries and analyzing usage patterns.

### GET /cache/metrics/{format}
Export cache metrics in JSON or Prometheus format.

**Path Parameters:**
- `format`: Either 'json' or 'prometheus'

### POST /cache/warm
Pre-populate cache with commonly requested data.

**Request Body:**
```json
{
  "symbols": ["BTC", "BTCUSDT"],
  "periods": ["1h", "1d", "7d"],
  "sources": ["binance", "coingecko"]
}
```

### GET /cache/info
Get comprehensive cache information including data distribution and performance metrics.

## Cache Maintenance

The system includes automated cache maintenance that starts with the API and runs periodic jobs.

### GET /cache/maintenance/status
Get current cache maintenance status including scheduled jobs and configuration.

**Response:**
```json
{
  "is_running": true,
  "config": {
    "warm_on_startup": true,
    "warm_interval_minutes": 30,
    "optimize_interval_hours": 6,
    "clear_expired_interval_hours": 1,
    "monitor_interval_minutes": 5,
    "hit_rate_threshold": 0.7,
    "cache_size_limit_mb": 1000
  },
  "cache_stats": {
    "total_entries": 150,
    "session_stats": {
      "hit_rate": 0.965
    }
  },
  "scheduled_jobs": [
    {
      "name": "warm_cache",
      "interval_minutes": 30,
      "next_run": "2025-01-01T12:30:00"
    },
    {
      "name": "optimize_cache",
      "interval_hours": 6,
      "next_run": "2025-01-01T18:00:00"
    },
    {
      "name": "clear_expired",
      "interval_hours": 1,
      "next_run": "2025-01-01T13:00:00"
    }
  ]
}
```

### POST /cache/maintenance/start
Manually start cache maintenance tasks (normally starts automatically with API).

### POST /cache/maintenance/stop
Stop cache maintenance tasks.

### POST /cache/maintenance/warm
Manually trigger cache warming.

**Query Parameters:**
- `aggressive` (optional): If true, performs comprehensive warming of all data sources

### PUT /cache/maintenance/config
Update cache maintenance configuration.

**Request Body:**
```json
{
  "warm_interval_minutes": 60,
  "hit_rate_threshold": 0.8,
  "cache_size_limit_mb": 2000
}
```

## Interactive Documentation
Visit http://localhost:8080/docs for interactive Swagger UI documentation with try-it-out functionality for all endpoints.
