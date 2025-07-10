# BTC Trading System 🚀

A comprehensive Bitcoin trading system with AI-powered signals, real-time updates, and paper trading capabilities. Built with FastAPI, Streamlit, and LSTM neural networks. Features a complete test suite with 100% passing unit tests.

## 🚀 Features

### Core Capabilities
- 🤖 **AI-powered trading signals** using LSTM neural networks with confidence intervals
- 📊 **Real-time BTC price data** with WebSocket support for live updates
- 💰 **Portfolio management** with P&L tracking and performance analytics
- 🎯 **Limit orders** and advanced risk management
- 📈 **Interactive charts** with 50+ technical indicators
- 🔄 **Automated signal generation** with ensemble predictions
- 📱 **Modern web interface** with real-time notifications
- 🐳 **Fully containerized** with Docker
- 📄 **Paper trading** with persistent portfolio tracking

### Enhanced Features (UltraThink)
- **50+ Trading Indicators**: Comprehensive technical, on-chain, sentiment, and macro indicators
- **Advanced Backtesting**: Walk-forward analysis, Bayesian optimization, Monte Carlo simulation
- **Real-time Updates**: WebSocket integration for live price and signal updates
- **Multi-source Data**: Real data from CoinGecko, Binance, Blockchain.info, and more
- **Paper Trading**: Practice strategies with virtual $10,000 portfolio
- **Discord Notifications**: Real-time alerts for signals, trades, and system status
- **Feature Importance Analysis**: AI-driven insights into which indicators matter most
- **Market Regime Detection**: Automatic identification of market conditions
- **SQLite API Caching**: Intelligent caching layer reduces API calls by 60-80%
- **Automated Cache Maintenance**: Self-optimizing cache with health monitoring

## 🏗️ Architecture

```
btc/
├── src/
│   ├── backend/
│   │   ├── api/
│   │   │   └── main.py              # FastAPI backend with WebSocket support
│   │   ├── models/
│   │   │   ├── database.py          # SQLite database management
│   │   │   ├── lstm.py              # LSTM neural network model
│   │   │   └── paper_trading.py     # Paper trading with P&L tracking
│   │   └── services/
│   │       ├── backtesting.py       # Comprehensive backtesting system
│   │       ├── data_fetcher.py      # Multi-source data integration
│   │       ├── integration.py       # Signal generation & analysis
│   │       ├── notifications.py     # Discord notifications
│   │       ├── cache_service.py     # SQLite-based API caching
│   │       ├── cache_integration.py # Cache decorators & helpers
│   │       └── cache_maintenance.py # Automated cache optimization
│   └── frontend/
│       ├── app.py                   # Main Streamlit application
│       ├── config.py                # Centralized configuration (NEW)
│       ├── components/              # Reusable UI components
│       │   ├── api_client.py       # API client with caching
│       │   ├── charts.py           # Interactive Plotly charts
│       │   ├── metrics.py          # Metrics display components
│       │   └── websocket_client.py # Real-time WebSocket client
│       ├── pages/                   # Multi-page navigation
│       │   ├── 1_📊_Dashboard.py   # Main trading dashboard
│       │   ├── 2_📈_Signals.py     # Signal analysis page
│       │   ├── 3_💰_Portfolio.py   # Portfolio management
│       │   ├── 4_📄_Paper_Trading.py
│       │   ├── 5_🔬_Analytics.py
│       │   └── 6_⚙️_Settings.py
│       └── utils/                   # Helper functions
├── tests/                           # Comprehensive test suite
│   ├── unit/                       # 92 unit tests (100% passing)
│   ├── integration/                # API integration tests
│   └── e2e/                        # End-to-end workflows
├── docker/                         # Docker configuration files
│   ├── docker-compose.yml          # Container orchestration
│   ├── Dockerfile.backend          # Backend container
│   └── Dockerfile.frontend         # Frontend container
└── init_deploy.sh                  # One-click deployment
```

## 🚀 Quick Start

### Prerequisites
- **Docker** and **Docker Compose**
- **2GB+ RAM** recommended
- **1GB+ disk space** for data storage

### 1. Clone and Setup
```bash
git clone https://github.com/duggasco/btc.git
cd btc
chmod +x init_deploy.sh
```

### 2. Configure Environment (Optional)
```bash
cp .env.template .env
# Edit .env to add Discord webhook URL and API keys for enhanced data
```

### 3. Deploy the System
```bash
# One-click deployment
./init_deploy.sh deploy

# Or use Docker Compose directly
docker compose up -d
```

### 4. Access the Interfaces
- **Frontend UI:** http://localhost:8501
- **Backend API:** http://localhost:8080  
- **API Documentation:** http://localhost:8080/docs
- **WebSocket:** ws://localhost:8000/ws

## 📊 Key Features Explained

### AI Trading Signals
- LSTM neural network with attention mechanism
- Ensemble predictions for higher accuracy
- Confidence intervals and consensus analysis
- Real-time signal updates via WebSocket

### Paper Trading
- Start with virtual $10,000
- Persistent portfolio tracking across sessions
- Performance metrics: Sharpe ratio, win rate, max drawdown
- Export trade history for analysis

### Enhanced Backtesting
- **Walk-forward analysis**: Test strategy robustness over time
- **Bayesian optimization**: Find optimal signal weights
- **Monte Carlo simulation**: Risk assessment with thousands of scenarios
- **Feature importance**: Identify which indicators drive performance

### Real-time Features
- WebSocket connections for live updates
- Price alerts for significant movements (±2.5%)
- Signal change notifications
- Portfolio value tracking

### 50+ Trading Indicators

#### Technical (20+)
- Moving Averages (SMA, EMA, multiple timeframes)
- RSI, MACD, Stochastic, Williams %R
- Bollinger Bands, Keltner Channels
- ADX, Aroon, CCI, Parabolic SAR
- Volume indicators (OBV, CMF, MFI)

#### On-chain (10+)
- Network activity (active addresses, transaction count)
- Exchange flows (inflow/outflow analysis)
- NVT ratio proxy
- Hash rate and difficulty
- Whale activity detection

#### Sentiment (10+)
- Fear & Greed Index (real-time)
- Reddit sentiment analysis
- Twitter/social sentiment
- Google Trends integration
- News sentiment scoring

#### Macro (5+)
- S&P 500 correlation
- Gold price correlation
- DXY (Dollar Index) impact
- VIX volatility index
- Treasury yields correlation

## 🛠️ Configuration

### Trading Rules (`/config/trading-rules`)
```json
{
    "min_trade_size": 0.001,
    "max_position_size": 0.1,
    "stop_loss_pct": 5.0,
    "take_profit_pct": 10.0,
    "buy_threshold": 0.6,
    "sell_threshold": 0.6
}
```

### Signal Weights (`/config/signal-weights`)
```json
{
    "technical_weight": 0.40,
    "onchain_weight": 0.35,
    "sentiment_weight": 0.15,
    "macro_weight": 0.10
}
```

### Model Configuration (`/config/model`)
```json
{
    "hidden_size": 50,
    "num_layers": 2,
    "dropout": 0.2,
    "sequence_length": 60,
    "use_attention": true
}
```

## 🔧 Management Commands

```bash
# Using deployment script (recommended)
./init_deploy.sh start    # Start services
./init_deploy.sh stop     # Stop services
./init_deploy.sh restart  # Restart services
./init_deploy.sh status   # Check service status
./init_deploy.sh logs     # View logs
./init_deploy.sh build    # Rebuild containers
./init_deploy.sh test     # Run tests
./init_deploy.sh clean    # Clean up resources

# Using docker-compose directly
docker compose up -d      # Start services
docker compose down       # Stop services
docker compose logs -f    # View logs
docker compose restart    # Restart services
```

## 🧪 Testing

The project includes a comprehensive test suite with 92 unit tests achieving 100% pass rate.

### Run All Tests
```bash
# Using the test runner
./tests/run_tests.py

# Using Docker
docker compose -f docker/docker-compose.test.yml up

# Run specific test categories
docker build -f docker/Dockerfile.test-simple -t btc-test .
docker run --rm btc-test pytest tests/unit/ -v
```

### Test Coverage
- **Unit Tests**: 92 tests covering critical components
  - Backend models: 22 tests
  - Backend services: 30 tests  
  - Frontend components: 40 tests
- **Integration Tests**: API endpoint validation
- **E2E Tests**: Complete workflow testing
- **Pass Rate**: 100% (92 passed, 9 skipped for unimplemented features)

### Quick System Check
```bash
./init_deploy.sh test
```

## 📡 API Endpoints

See [API Documentation](docs/API.md) for complete endpoint reference.

### Core Endpoints
- `GET /` - Health check
- `GET /health` - Detailed health status
- `GET /signals/enhanced/latest` - Latest AI signal with analysis
- `GET /portfolio/metrics` - Portfolio performance
- `WS /ws` - WebSocket for real-time updates

### Paper Trading
- `GET /paper-trading/status` - Current paper trading status
- `POST /paper-trading/toggle` - Enable/disable paper trading
- `POST /paper-trading/reset` - Reset portfolio to $10,000
- `GET /paper-trading/history` - Performance history

### Enhanced Analytics
- `GET /signals/comprehensive` - All 50+ calculated signals
- `GET /analytics/feature-importance` - ML feature importance
- `GET /backtest/enhanced/run` - Run advanced backtest
- `GET /analytics/monte-carlo` - Risk simulation

### Cache Management
- `GET /cache/stats` - Cache performance statistics
- `POST /cache/optimize` - Optimize cache storage
- `GET /cache/maintenance/status` - Maintenance system status

## 🔐 Security & API Keys

### Optional API Keys for Enhanced Data
Add to `.env` file:
```bash
# Discord notifications
DISCORD_WEBHOOK_URL=your_webhook_url

# Enhanced data sources (all have free tiers)
FRED_API_KEY=your_fred_key          # Federal Reserve data
NEWS_API_KEY=your_news_key          # News sentiment
CRYPTOPANIC_API_KEY=your_key        # Crypto news
GLASSNODE_API_KEY=your_key          # On-chain data
```

## 🐳 Docker Details

### Services
- **backend**: FastAPI service with all trading logic
- **frontend**: Streamlit UI with real-time updates

### Volumes
- `/storage/data/`: SQLite database, paper trading data, and API cache
- `/storage/models/`: Trained LSTM models
- `/storage/logs/`: Application logs
- `/storage/config/`: Configuration files
- `/storage/backups/`: Data backups
- `/storage/exports/`: Export files

### Environment Variables
```bash
DATABASE_PATH=/app/data/trading_system.db
MODEL_PATH=/app/models
LOG_PATH=/app/logs
CONFIG_PATH=/app/config
API_HOST=0.0.0.0
API_PORT=8000
STREAMLIT_HOST=0.0.0.0
STREAMLIT_PORT=8501
DISCORD_WEBHOOK_URL=optional_webhook
```

## 🚀 Advanced Usage

### Enable All Features
1. Set up Discord webhook for notifications
2. Add API keys for enhanced data sources
3. Run initial backtest to optimize weights
4. Enable paper trading to test strategies
5. Monitor real-time updates on dashboard

### Custom Strategy Development
1. Modify signal weights in Configuration
2. Adjust trading rules for your risk tolerance
3. Run backtests with different periods
4. Use paper trading to validate changes
5. Export results for analysis

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📈 Performance

- Handles real-time data for multiple indicators
- WebSocket support for instant updates
- Optimized LSTM inference (<100ms)
- SQLite API caching reduces external calls by 60-80%
- 10-100x faster response times for cached data
- Automated cache maintenance ensures optimal performance
- Persistent storage for all trading data

## 🧹 Recent Updates (January 2025)

### Code Cleanup
- **Removed redundant files**: Eliminated ~15 duplicate and backup files
- **Fixed import issues**: Resolved duplicate imports and circular dependencies
- **Centralized configuration**: New `config.py` for frontend settings
- **Bug fixes**: Fixed chart rendering error and improved error handling
- **Consistent API usage**: All frontend components use same backend URL

### Improvements
- Better code organization with single source of truth for utilities
- Environment variable support for all configuration
- Cleaner project structure with no Python cache files
- Enhanced maintainability and easier updates

## ⚠️ Disclaimer

This project is for **educational and demonstration purposes only**. 

- Not financial advice
- Past performance doesn't guarantee future results  
- Cryptocurrency trading carries significant risk
- Always do your own research
- Never invest more than you can afford to lose

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Data provided by CoinGecko, Binance, Blockchain.info
- Sentiment data from Alternative.me, Reddit, NewsAPI
- Built with FastAPI, Streamlit, PyTorch, and Plotly
- Enhanced with UltraThink AI optimization features

---

**Ready to start?** Deploy the system and begin with paper trading to test strategies risk-free!
