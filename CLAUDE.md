# Claude Code Instructions for BTC Trading System

This document provides context and guidelines for AI assistants working on the BTC Trading System project.

## Project Overview

This is a comprehensive Bitcoin trading system with AI-powered signals, real-time updates, and paper trading capabilities. The system uses:
- **Backend**: FastAPI with WebSocket support for real-time data
- **Frontend**: Streamlit multi-page application
- **AI/ML**: LSTM neural networks with attention mechanism for trading signals
- **Data**: 50+ technical, on-chain, sentiment, and macro indicators from multiple sources
- **Storage**: SQLite database with persistent paper trading portfolio

## Key Commands

### Deployment and Management
```bash
# Deploy the system (from project root)
./init_deploy.sh deploy

# Or use Docker Compose directly
docker compose up -d

# Management commands
./init_deploy.sh start    # Start services
./init_deploy.sh stop     # Stop services
./init_deploy.sh restart  # Restart services
./init_deploy.sh status   # Check service status
./init_deploy.sh logs     # View logs
./init_deploy.sh build    # Rebuild containers
./init_deploy.sh clean    # Clean up resources

# Direct docker commands
docker compose down       # Stop all services
docker compose logs -f    # Follow logs
docker compose ps         # Check service status
```

### Testing
```bash
# Run comprehensive test suite (92 unit tests)
./run_tests.py

# Run tests in Docker
docker compose -f docker-compose.test.yml up

# Run specific test categories
docker build -f Dockerfile.test-simple -t btc-test .
docker run --rm btc-test pytest tests/unit/ -v

# Quick system check
./init_deploy.sh test
```

### Development
```bash
# No linting or formatting tools are currently configured in this project
# Consider adding flake8, black, or ruff for Python code quality
```

## Architecture Overview

### Service Architecture
The system follows a microservices pattern with clear separation:
1. **Backend API** (port 8080): FastAPI service handling all business logic, data fetching, ML predictions
2. **Frontend UI** (port 8501): Streamlit application for user interaction
3. **WebSocket** (ws://localhost:8080/ws): Real-time updates for price and signals
4. **Storage**: Persistent volumes for data, models, logs, and configuration

### Core Components

#### Backend (`/src/backend/`)
- **api/main.py**: FastAPI application entry point with WebSocket endpoints
- **models/lstm.py**: LSTM implementation with attention mechanism for price predictions
- **models/paper_trading.py**: Paper trading logic with persistent portfolio tracking
- **services/integration.py**: Advanced signal generation combining 50+ indicators
- **services/data_fetcher.py**: Multi-source data aggregation (CoinGecko, Binance, etc.)
- **services/backtesting.py**: Walk-forward analysis and Monte Carlo simulations
- **services/notifications.py**: Discord webhook integration for alerts

#### Frontend (`/src/frontend/`)
- **app.py**: Main Streamlit application with sidebar navigation
- **pages/**: Six main pages (Dashboard, Signals, Portfolio, Paper Trading, Analytics, Settings)
- **components/**: Reusable UI components (charts.py, metrics.py, api_client.py, websocket_client.py)
- **utils/**: Helper functions for the frontend

### Data Flow
1. External APIs → Backend data_fetcher → Database storage
2. Stored data → LSTM model → Signal generation → WebSocket broadcast
3. Frontend connects to WebSocket → Real-time UI updates
4. User actions → API calls → Backend processing → Database updates

### Key Design Patterns
- **Real-time Communication**: WebSocket for live price/signal updates
- **Ensemble Predictions**: Multiple LSTM models with consensus voting
- **Fallback Mechanisms**: Graceful handling when external APIs fail
- **Persistent State**: SQLite for paper trading portfolio across sessions
- **Event-driven Notifications**: Discord webhooks for important events

## Configuration

### Main Configuration Files
- `/storage/config/trading_config.json`: Trading rules, thresholds, and weights
- `.env`: Environment variables including Discord webhook and API keys
- `docker-compose.yml`: Service orchestration and networking

### Important Environment Variables
```bash
DATABASE_PATH=/app/data/trading_system.db
MODEL_PATH=/app/models
API_BASE_URL=http://backend:8080  # For frontend-backend communication
DISCORD_WEBHOOK_URL=<webhook_url>  # Optional, for notifications
```

## Common Development Tasks

### Adding New Indicators
1. Add calculation logic in `/src/backend/services/integration.py`
2. Update the `calculate_all_signals()` function
3. Add to feature importance analysis if needed
4. Update frontend to display new indicator

### Modifying Trading Strategy
1. Edit signal weights in `/storage/config/trading_config.json`
2. Adjust thresholds in the configuration
3. Run backtests to validate changes
4. Test with paper trading before live deployment

### Debugging WebSocket Issues
1. Check WebSocket endpoint in `/src/backend/api/main.py`
2. Verify frontend connection in `/src/frontend/components/websocket_client.py`
3. Monitor browser console for connection errors
4. Check CORS settings if connection fails

### Database Schema Changes
1. Database models are in `/src/backend/models/database.py`
2. Paper trading state in `/src/backend/models/paper_trading.py`
3. SQLite database at `/storage/data/trading_system.db`

## Testing Guidelines

### API Testing
- All endpoints documented at http://localhost:8080/docs
- Test script validates core functionality: `scripts/test_system.py`
- WebSocket testing requires manual connection verification

### Paper Trading Testing
1. Reset portfolio: POST `/paper-trading/reset`
2. Enable trading: POST `/paper-trading/toggle`
3. Monitor trades: GET `/paper-trading/history`
4. Check performance: GET `/portfolio/metrics`

## Test Suite

The project includes a comprehensive test suite with 92 unit tests achieving 100% pass rate:
- **Backend Models**: 22 tests (database, paper trading)
- **Backend Services**: 30 tests (data fetcher, notifications)
- **Frontend Components**: 40 tests (API client, charts, WebSocket)
- **Test Framework**: pytest with fixtures and mocking
- **Docker Testing**: Isolated test environment via Dockerfile.test-simple

### Running Tests
```bash
# All unit tests
./run_tests.py

# Specific component
docker run --rm btc-test pytest tests/unit/backend/models/ -v

# With coverage report
docker run --rm btc-test pytest --cov=src tests/unit/
```

## Important Notes

1. **Comprehensive Test Coverage**: 92 unit tests with 100% pass rate
2. **Docker-based Development**: All services run in containers; direct Python execution may fail due to dependencies
3. **Real-time Data**: System depends on external APIs; some features may degrade if APIs are unavailable
4. **Paper Trading Only**: No real trading functionality is implemented for safety
5. **WebSocket Critical**: Many frontend features depend on active WebSocket connection
6. **Test-Driven Development**: All new features should have tests first

## Security Considerations

1. Never commit API keys or Discord webhooks to the repository
2. All sensitive configuration should be in `.env` file (gitignored)
3. Paper trading uses virtual funds only
4. No real exchange connections implemented