# BTC Trading Signals API Documentation

Base URL: `http://localhost:8080`

## Authentication
No authentication required for local deployment.

## Endpoints

### Health & Status

#### GET /
Health check endpoint.
```json
{
  "message": "BTC Trading System API is running",
  "timestamp": "2025-01-01T12:00:00",
  "status": "healthy"
}
```

#### GET /health
Detailed system health.
```json
{
  "status": "healthy",
  "components": {
    "database": "healthy",
    "signal_generator": "healthy"
  },
  "latest_signal": {
    "signal": "buy",
    "confidence": 0.75
  }
}
```

### Trading Signals

#### GET /signals/latest
Get the most recent trading signal.
```json
{
  "symbol": "BTC-USD",
  "signal": "buy",
  "confidence": 0.75,
  "predicted_price": 45250.00,
  "timestamp": "2025-01-01T12:00:00"
}
```

#### GET /signals/history?limit=50
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

### Trading Operations

#### POST /trades/
Create a new trade.
```json
// Request
{
  "symbol": "BTC-USD",
  "trade_type": "buy",
  "price": 45000.00,
  "size": 0.1,
  "lot_id": "optional_lot_id"
}

// Response
{
  "trade_id": "uuid",
  "status": "success",
  "message": "Trade created successfully"
}
```

#### GET /trades/?symbol=BTC-USD&limit=100
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
    "status": "executed"
  }
]
```

### Portfolio Management

#### GET /portfolio/metrics
Get portfolio performance metrics.
```json
{
  "total_trades": 25,
  "total_pnl": 1250.75,
  "positions_count": 3,
  "total_invested": 10000.00,
  "current_btc_price": 45000.00
}
```

#### GET /positions/
Get current positions.
```json
[
  {
    "lot_id": "uuid",
    "symbol": "BTC-USD",
    "total_size": 0.5,
    "available_size": 0.3,
    "avg_buy_price": 42000.00
  }
]
```

### Limit Orders

#### POST /limits/
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

#### GET /limits/
Get active limit orders.
```json
[
  {
    "id": "uuid",
    "symbol": "BTC-USD",
    "limit_type": "stop_loss",
    "price": 40000.00,
    "size": 0.1,
    "active": true,
    "created_at": "2025-01-01T12:00:00"
  }
]
```

### Market Data

#### GET /market/btc-data?period=1mo
Get BTC market data.
```json
{
  "symbol": "BTC-USD",
  "period": "1mo",
  "data": [
    {
      "timestamp": "2025-01-01T00:00:00",
      "open": 44800.00,
      "high": 45200.00,
      "low": 44500.00,
      "close": 45000.00,
      "volume": 1500.25,
      "sma_20": 44750.00,
      "rsi": 65.5,
      "macd": 125.75
    }
  ],
  "total_records": 30
}
```

### Analytics

#### GET /analytics/pnl
Get P&L analytics data.
```json
{
  "daily_pnl": [
    {
      "date": "2025-01-01",
      "pnl": 150.75
    }
  ],
  "cumulative_pnl": [
    {
      "date": "2025-01-01", 
      "pnl": 1250.75
    }
  ]
}
```

#### GET /system/status
Get system status.
```json
{
  "api_status": "running",
  "signal_update_errors": 0,
  "latest_signal_time": "2025-01-01T12:00:00",
  "data_cache_status": "available",
  "database_status": "connected"
}
```

## Signal Types
- `buy` - Bullish signal, price expected to rise
- `sell` - Bearish signal, price expected to fall  
- `hold` - Neutral signal, no clear direction

## Trade Types
- `buy` - Purchase BTC
- `sell` - Sell BTC
- `hold` - No action

## Limit Types
- `stop_loss` - Sell if price drops below threshold
- `take_profit` - Sell if price rises above threshold
- `buy_limit` - Buy if price drops to threshold

## Error Responses

### 400 Bad Request
```json
{
  "detail": "Invalid trade data"
}
```

### 500 Internal Server Error
```json
{
  "detail": "Database connection failed"
}
```

## Rate Limits
No rate limits for local deployment.

## WebSocket Support
Not currently implemented.

## Example Usage

### Get Latest Signal
```bash
curl http://localhost:8080/signals/latest
```

### Create Trade
```bash
curl -X POST http://localhost:8080/trades/ \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "BTC-USD",
    "trade_type": "buy", 
    "price": 45000.00,
    "size": 0.1
  }'
```

### Get Portfolio Metrics
```bash
curl http://localhost:8080/portfolio/metrics
```

## Interactive Documentation
Visit http://localhost:8080/docs for Swagger UI documentation.
