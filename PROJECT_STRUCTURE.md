# Project Structure

This document provides a detailed overview of the BTC Trading System project structure after the January 2025 cleanup.

## Directory Structure

```
btc/
├── src/                             # Source code
│   ├── backend/                     # FastAPI backend application
│   │   ├── api/
│   │   │   └── main.py             # FastAPI app, endpoints, WebSocket handlers
│   │   ├── models/
│   │   │   ├── database.py         # SQLite database models and operations
│   │   │   ├── lstm.py             # Base LSTM neural network model
│   │   │   ├── enhanced_lstm.py    # Enhanced LSTM with additional features
│   │   │   ├── enhanced_lstm_returns.py  # LSTM optimized for returns prediction
│   │   │   ├── intel_optimized_lstm.py   # Intel CPU/GPU optimized variant
│   │   │   └── paper_trading.py    # Paper trading portfolio management
│   │   └── services/
│   │       ├── backtesting.py      # Comprehensive backtesting framework
│   │       ├── cache_integration.py # Cache decorators and helpers
│   │       ├── cache_maintenance.py # Automated cache optimization
│   │       ├── cache_service.py    # SQLite-based API caching
│   │       ├── data_fetcher.py     # External API data fetching
│   │       ├── enhanced_data_fetcher.py  # Enhanced data fetching with more sources
│   │       ├── enhanced_integration.py   # Enhanced signal generation
│   │       ├── feature_engineering.py    # Feature calculation and engineering
│   │       ├── integration.py      # Core signal generation logic
│   │       └── notifications.py    # Discord webhook notifications
│   │
│   └── frontend/                    # Streamlit frontend application
│       ├── app.py                  # Main Streamlit app entry point
│       ├── config.py               # Centralized configuration (NEW)
│       ├── components/
│       │   ├── api_client.py       # HTTP client with caching and retry logic
│       │   ├── charts.py           # Interactive Plotly chart components
│       │   ├── metrics.py          # Metric display components
│       │   └── websocket_client.py # WebSocket client for real-time updates
│       ├── pages/                  # Multi-page Streamlit navigation
│       │   ├── 1_📊_Dashboard.py   # Real-time trading dashboard
│       │   ├── 2_📈_Signals.py     # Signal analysis and indicators
│       │   ├── 3_💰_Portfolio.py   # Portfolio management
│       │   ├── 4_📄_Paper_Trading.py # Paper trading interface
│       │   ├── 5_🔬_Analytics.py   # Backtesting and analytics
│       │   └── 6_⚙️_Settings.py    # Configuration management
│       └── utils/
│           ├── constants.py        # Application constants
│           └── helpers.py          # Utility functions
│
├── tests/                          # Test suite
│   ├── conftest.py                # Pytest configuration
│   ├── pytest.ini                 # Pytest settings
│   ├── run_tests.py               # Main test runner script
│   ├── unit/                      # Unit tests
│   │   ├── backend/
│   │   │   ├── models/           # Model unit tests
│   │   │   └── services/         # Service unit tests
│   │   └── frontend/
│   │       └── components/       # Component unit tests
│   ├── integration/               # Integration tests
│   │   ├── test_api_endpoints.py
│   │   ├── test_enhanced_lstm_integration.py
│   │   ├── test_prediction_endpoints.py
│   │   └── test_websocket.py
│   └── e2e/                       # End-to-end tests
│       ├── test_prediction_workflows.py
│       └── test_trading_workflows.py
│
├── docker/                        # Docker configuration
│   ├── docker-compose.yml        # Main compose file
│   ├── docker-compose.intel.yml  # Intel optimization variant
│   ├── docker-compose.test*.yml  # Test configurations
│   ├── Dockerfile.backend        # Backend container
│   ├── Dockerfile.backend.intel  # Intel-optimized backend
│   ├── Dockerfile.frontend       # Frontend container
│   └── Dockerfile.test*          # Test containers
│
├── storage/                      # Persistent storage (mounted volume)
│   ├── config/                   # Configuration files
│   │   └── trading_config.json   # Trading rules and settings
│   ├── data/                     # Data storage
│   │   ├── trading_system.db     # Main SQLite database
│   │   ├── api_cache.db         # API response cache
│   │   └── cache/               # File-based cache
│   ├── logs/                    # Application logs
│   │   ├── backend/
│   │   ├── frontend/
│   │   └── system/
│   └── models/                  # Trained ML models
│       ├── lstm_btc_model.pth   # Current production model
│       ├── best_lstm_model.pth  # Best performing model
│       ├── feature_scaler.pkl   # Feature scaling parameters
│       └── target_scaler.pkl    # Target scaling parameters
│
├── docs/                        # Documentation
│   ├── API.md                   # API endpoint documentation
│   ├── CACHE_MONITORING.md      # Cache system documentation
│   ├── INTEL_OPTIMIZATION.md    # Intel optimization guide
│   └── TEST_WORKFLOWS.md        # Testing documentation
│
├── scripts/                     # Utility scripts
│   ├── create_gitkeeps.sh      # Create .gitkeep files
│   └── run_api_tests_docker.sh # Run API tests in Docker
│
├── init_deploy.sh              # One-click deployment script
├── README.md                   # Project documentation
├── CLAUDE.md                   # Claude AI guidance
├── CLEANUP_SUMMARY.md          # Cleanup changes documentation
├── requirements.txt            # Python dependencies
└── .env.template              # Environment variables template
```

## Key Files and Their Purpose

### Backend Core Files

- **`api/main.py`**: FastAPI application with all endpoints, WebSocket handlers, and middleware
- **`models/lstm.py`**: Core LSTM implementation with configurable architecture
- **`services/data_fetcher.py`**: Fetches data from multiple sources (CoinGecko, Binance, etc.)
- **`services/integration.py`**: Combines indicators to generate trading signals
- **`services/cache_service.py`**: SQLite-based caching to reduce API calls

### Frontend Core Files

- **`app.py`**: Streamlit configuration and main navigation
- **`config.py`**: Centralized configuration for API URLs, timeouts, and settings
- **`components/api_client.py`**: Handles all API communication with retry logic
- **`components/websocket_client.py`**: Manages WebSocket connections for real-time updates
- **`pages/1_📊_Dashboard.py`**: Main trading interface with live price updates

### Configuration Files

- **`storage/config/trading_config.json`**: Trading rules, thresholds, and weights
- **`.env`**: Environment variables (Discord webhook, API keys)
- **`src/frontend/config.py`**: Frontend-specific configuration

### Data Storage

- **`storage/data/trading_system.db`**: Main database storing trades, signals, and portfolio
- **`storage/data/api_cache.db`**: Cached API responses with TTL
- **`storage/models/*.pth`**: Trained PyTorch LSTM models

## Import Patterns

### Backend Imports
```python
# Absolute imports for cross-module references
from src.backend.models.database import get_db_session
from src.backend.services.data_fetcher import ExternalDataFetcher
```

### Frontend Imports
```python
# Relative imports within frontend package
from components.api_client import APIClient
from utils.helpers import format_currency
# Config import
import config
```

## Data Flow

1. **External APIs** → `data_fetcher.py` → `api_cache.db`
2. **Cached Data** → `feature_engineering.py` → **50+ Indicators**
3. **Indicators** → `lstm.py` → **Predictions**
4. **Predictions** → `integration.py` → **Trading Signals**
5. **Signals** → `paper_trading.py` → **Portfolio Updates**
6. **All Updates** → **WebSocket** → **Frontend UI**

## Key Design Decisions

1. **SQLite for Everything**: Both main database and cache use SQLite for simplicity
2. **Centralized Config**: All settings in one place for easy updates
3. **WebSocket First**: Real-time updates prioritized over polling
4. **Graceful Degradation**: System continues working even if external APIs fail
5. **Paper Trading Default**: Safe practice mode before real trading

## Recent Changes (January 2025)

- Removed ~15 redundant files and duplicates
- Created centralized `config.py` for frontend
- Fixed import issues and circular dependencies
- Consolidated utility functions to single location
- Cleaned up project structure for better maintainability