# BTC Trading System - Project Structure

This document provides a detailed overview of the project structure after cleanup and optimization.

## Directory Structure

```
btc/
├── src/                                # Source code
│   ├── backend/                        # FastAPI backend application
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── main.py                # FastAPI app with WebSocket support
│   │   │   └── routes/                # API route definitions
│   │   │       └── __init__.py
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── database.py            # SQLite database management
│   │   │   ├── lstm.py                # LSTM neural network implementation
│   │   │   └── paper_trading.py       # Paper trading with P&L tracking
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── backtesting.py         # Comprehensive backtesting system
│   │   │   ├── data_fetcher.py        # Multi-source data integration
│   │   │   ├── integration.py         # Signal generation & analysis
│   │   │   └── notifications.py       # Discord webhook notifications
│   │   ├── __init__.py
│   │   ├── import_compat.py           # Import compatibility layer
│   │   └── requirements.txt           # Backend dependencies
│   │
│   └── frontend/                       # Streamlit frontend application
│       ├── components/                 # Reusable UI components
│       │   ├── __init__.py
│       │   ├── api_client.py          # API client with caching
│       │   ├── charts.py              # Interactive Plotly charts
│       │   ├── metrics.py             # Metrics display components
│       │   └── websocket_client.py    # WebSocket client for real-time data
│       ├── pages/                      # Multi-page navigation
│       │   ├── __init__.py
│       │   ├── 1_📊_Dashboard.py      # Main trading dashboard
│       │   ├── 2_📈_Signals.py        # Trading signals analysis
│       │   ├── 3_💰_Portfolio.py      # Portfolio management
│       │   ├── 4_📄_Paper_Trading.py  # Paper trading interface
│       │   ├── 5_🔬_Analytics.py      # Advanced analytics
│       │   └── 6_⚙️_Settings.py       # System settings
│       ├── utils/                      # Helper functions
│       │   ├── __init__.py
│       │   ├── constants.py           # Application constants
│       │   └── helpers.py             # Utility functions
│       ├── __init__.py
│       ├── app.py                     # Main Streamlit application
│       ├── MIGRATION_GUIDE.md         # Frontend migration guide
│       └── requirements.txt           # Frontend dependencies
│
├── tests/                             # Test suite
│   ├── unit/                          # Unit tests (92 tests total)
│   │   ├── backend/
│   │   │   ├── models/
│   │   │   │   ├── test_database.py          # 9 tests
│   │   │   │   └── test_paper_trading.py     # 13 tests
│   │   │   └── services/
│   │   │       ├── test_data_fetcher.py      # 12 tests
│   │   │       └── test_notifications.py     # 18 tests
│   │   └── frontend/
│   │       └── components/
│   │           ├── test_api_client.py        # 16 tests
│   │           ├── test_charts.py            # 16 tests
│   │           └── test_websocket_client.py  # 16 tests
│   ├── integration/                   # Integration tests
│   │   ├── test_api_endpoints.py     # API endpoint validation
│   │   └── test_websocket.py         # WebSocket connection tests
│   ├── e2e/                          # End-to-end tests
│   │   └── test_trading_workflows.py # Complete workflow testing
│   ├── __init__.py
│   ├── conftest.py                   # pytest configuration and fixtures
│   ├── requirements.txt              # Test dependencies
│   └── test_data_fetcher.py         # Additional data fetcher tests
│
├── scripts/                          # Utility scripts
│   ├── create_gitkeeps.sh           # Create .gitkeep files
│   ├── init_deploy.sh               # Deployment script (symlink)
│   ├── run_api_tests_docker.sh      # Run API tests in Docker
│   └── test_system.py               # System integration tests
│
├── docs/                            # Documentation
│   └── API.md                       # API endpoint documentation
│
├── Docker Configuration
│   ├── docker-compose.yml           # Main container orchestration
│   ├── docker-compose.test.yml      # Test environment setup
│   ├── Dockerfile.backend           # Backend container definition
│   ├── Dockerfile.frontend          # Frontend container definition
│   ├── Dockerfile.test              # Full test environment
│   └── Dockerfile.test-simple       # Simplified test environment
│
├── Configuration Files
│   ├── .env.template               # Environment variable template
│   ├── .gitignore                  # Git ignore rules
│   ├── pytest.ini                  # pytest configuration
│   └── requirements.txt            # Root level dependencies
│
├── Project Files
│   ├── init_deploy.sh              # One-click deployment script
│   ├── run_tests.py                # Test runner script
│   ├── cleanup_project.sh          # Project cleanup script
│   ├── README.md                   # Project documentation
│   ├── CLAUDE.md                   # AI assistant instructions
│   ├── PROJECT_STRUCTURE.md        # This file
│   ├── DOCKER_MIGRATION.md         # Docker migration guide
│   └── LICENSE                     # MIT License
│
└── Runtime Directories (created by init_deploy.sh)
    └── /storage/                   # Persistent storage (outside project)
        ├── data/                   # SQLite databases
        ├── models/                 # Trained ML models
        ├── logs/                   # Application logs
        ├── config/                 # Runtime configuration
        ├── backups/                # Data backups
        └── exports/                # Export files

```

## Key Files Overview

### Core Application Files
- `src/backend/api/main.py` - FastAPI application with all endpoints and WebSocket
- `src/frontend/app.py` - Main Streamlit application entry point
- `init_deploy.sh` - One-click deployment and management script
- `run_tests.py` - Test suite runner

### Configuration Files
- `.env.template` - Template for environment variables
- `docker-compose.yml` - Container orchestration configuration
- `pytest.ini` - Test framework configuration

### Test Files
- 92 unit tests across backend and frontend components
- Integration tests for API endpoints
- E2E tests for complete workflows

### Documentation
- `README.md` - Main project documentation
- `CLAUDE.md` - Instructions for AI assistants
- `docs/API.md` - API endpoint reference

## File Count Summary

- **Python files**: 62
- **Test files**: 16 
- **Docker files**: 6
- **Shell scripts**: 7
- **Documentation**: 6
- **Configuration**: 5

## Storage Layout

The `/storage` directory is created outside the project directory by `init_deploy.sh`:

```
/storage/
├── data/
│   └── trading_system.db      # Main SQLite database
├── models/
│   └── lstm_model_*.pth       # Trained LSTM models
├── logs/
│   ├── backend/               # Backend application logs
│   ├── frontend/              # Frontend application logs
│   └── system/                # System-level logs
├── config/
│   └── trading_rules.json     # Trading configuration
├── backups/
│   └── *.db.backup           # Database backups
└── exports/
    └── *.csv                 # Data exports
```

## Removed During Cleanup

The following files and directories were removed as they were temporary or no longer needed:

- Refactoring scripts (enhanced_streamlit_refactor.sh, refactor.sh, etc.)
- Backup directories (./backups/, ./src/frontend/backups/)
- Temporary log files
- Old project summaries
- Navigation fix scripts

Total cleanup: ~1.1MB of unnecessary files removed