# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Development Commands

### Primary Development Commands
```bash
# Deploy the system (recommended starting point)
./init_deploy.sh deploy

# Run comprehensive test suite (92 unit tests)
./run_tests.py

# Run specific test types
./run_tests.py unit        # Unit tests only
./run_tests.py integration # Integration tests only
./run_tests.py --path tests/unit/backend/  # Specific directory

# Docker commands for development
docker compose up -d       # Start all services
docker compose logs -f     # Follow logs
docker compose down        # Stop services
docker compose ps          # Check status
```

### Testing Commands
```bash
# Run tests in Docker (isolated environment)
docker compose -f docker-compose.test.yml up

# Run with coverage
docker run --rm btc-test pytest --cov=src tests/unit/

# Run by test markers
pytest -m unit      # Unit tests only
pytest -m websocket # WebSocket tests
pytest -m ml        # Machine learning tests
```

### Service Management
```bash
# Using init_deploy.sh wrapper
./init_deploy.sh start    # Start services
./init_deploy.sh stop     # Stop services
./init_deploy.sh restart  # Restart services
./init_deploy.sh status   # Check status
./init_deploy.sh logs     # View logs
./init_deploy.sh clean    # Clean up resources
```

**Note**: No linting/formatting tools are configured. Consider adding flake8, black, or ruff.

## High-Level Architecture

### Core Architecture Pattern
The system implements a **microservices architecture with real-time communication**:

```
External APIs → Backend (FastAPI) → SQLite → Frontend (Streamlit)
                    ↓                ↑
                WebSocket ← Real-time Updates
```

### Service Architecture
1. **Backend Service** (port 8090, container port 8000)
   - FastAPI with WebSocket server
   - LSTM models (original + enhanced)
   - Multi-source data fetching with fallbacks
   - Paper trading engine
   - Signal generation from 50+ indicators

2. **Frontend Service** (port 8501)
   - Streamlit multi-page application
   - WebSocket client for real-time updates
   - API client with caching and retry logic
   - Interactive Plotly charts

### Key Architectural Decisions

#### Data Flow & Integration
- **Multi-source pattern**: Primary (Binance) → Fallback (Yahoo) → Enrichment (Fear & Greed, On-chain)
- **50+ indicators**: Technical (21), On-chain (15), Sentiment (14), calculated in `services/integration.py`
- **Feature engineering**: Adaptive selection in `services/feature_engineering.py`

#### ML Model Hierarchy
1. **Enhanced LSTM** (if trained): Ensemble of 3 models with attention
2. **Original LSTM** (fallback): Basic 2-layer with attention
3. **Rule-based** (last resort): Simple threshold-based signals

Training enhanced model: `POST /enhanced-lstm/train`

#### Real-time Communication
- **WebSocket channels**: Price updates, signal broadcasts
- **ConnectionManager**: Handles multiple concurrent connections
- **Auto-reconnection**: Frontend client retries on disconnect

#### State Management
- **SQLite persistence**: Paper trading, signals, backtest results
- **Volume mounts**: Data survives container restarts
- **Audit trail**: All trades and signals logged

### Critical Integration Points

1. **Backend ↔ Frontend Communication**
   - REST API: `http://backend:8080` (internal network)
   - WebSocket: `ws://localhost:8090/ws` (external)
   - Frontend API client handles retries and caching

2. **Data Fetcher Coordination**
   - `enhanced_data_fetcher.py`: 730 days historical data
   - `data_fetcher.py`: Real-time updates
   - Fallback chain prevents single point of failure

3. **Signal Generation Flow**
   ```python
   # In services/enhanced_integration.py
   1. Fetch multi-source data
   2. Calculate 50+ indicators
   3. Get LSTM predictions
   4. Generate consensus signal
   5. Broadcast via WebSocket
   ```

4. **Paper Trading State Machine**
   - Persistent portfolio in SQLite
   - Position sizing based on confidence
   - Real-time P&L calculation
   - Trade history with metrics

### Key Files for Understanding Architecture

- **Backend entry**: `src/backend/api/main.py` - FastAPI routes, WebSocket endpoints
- **Frontend entry**: `src/frontend/app.py` - Streamlit config, page navigation
- **Integration hub**: `src/backend/services/enhanced_integration.py` - Orchestrates all components
- **WebSocket flow**: `src/backend/api/main.py` + `src/frontend/components/websocket_client.py`
- **Data persistence**: `src/backend/models/database.py` - Schema and operations

### Configuration & Environment

Key configuration files:
- `/storage/config/trading_config.json` - Trading rules, thresholds, model params
- `.env` - API keys, Discord webhook (optional)
- `docker-compose.yml` - Service definitions, networking

Environment variables:
```bash
DATABASE_PATH=/app/data/trading_system.db
MODEL_PATH=/app/models
API_BASE_URL=http://backend:8080  # Frontend→Backend communication
DISCORD_WEBHOOK_URL=<optional>     # For notifications

# Optional API Keys (system works without these using fallbacks)
FINNHUB_API_KEY=      # Free tier at finnhub.io
FRED_API_KEY=         # Free at fred.stlouisfed.org
TWELVE_DATA_API_KEY=  # Free tier at twelvedata.com
ALPHA_VANTAGE_API_KEY=# Free at alphavantage.co
GLASSNODE_API_KEY=    # Paid service for on-chain data
```

Copy `.env.example` to `.env` for API key configuration.

### Testing Strategy

The codebase includes 92 unit tests with 100% pass rate, organized by component:
- Backend models: 22 tests
- Backend services: 30 tests
- Frontend components: 40 tests

Test isolation via Docker ensures consistent environment.

## Important Development Notes

1. **Docker-first development**: Direct Python execution may fail due to dependencies
2. **WebSocket critical**: Many features depend on active WebSocket connection
3. **Data source fallbacks**: System degrades gracefully when APIs unavailable
4. **Paper trading only**: No real trading implemented for safety
5. **Enhanced LSTM requires training**: Use `/enhanced-lstm/train` endpoint (5-10 min)
6. **Persistent volumes**: Data, models, logs survive container restarts