# BTC Trading System

A comprehensive Bitcoin trading system with AI-powered signals, built with FastAPI, Streamlit, and PyTorch LSTM models.

## Features

- 🤖 **AI-powered trading signals** using LSTM neural networks
- 📊 **Real-time BTC price data** and technical indicators  
- 💰 **Portfolio management** with P&L tracking
- 🎯 **Limit orders** and risk management
- 📈 **Interactive charts** and analytics
- 🔄 **Automated signal generation**
- 📱 **Modern web interface** with Streamlit
- 🐳 **Fully containerized** with Docker

## Quick Start

1. **Make scripts executable:**
   ```bash
   chmod +x init_and_deploy.sh create_gitkeeps.sh test_system.py
   ```

2. **Deploy the system:**
   ```bash
   ./init_and_deploy.sh deploy
   ```

3. **Access the interfaces:**
   - **Frontend UI:** http://localhost:8501
   - **Backend API:** http://localhost:8080  
   - **API Documentation:** http://localhost:8080/docs

## Management Commands

```bash
./init_and_deploy.sh deploy    # Full deployment
./init_and_deploy.sh start     # Start services
./init_and_deploy.sh stop      # Stop services  
./init_and_deploy.sh restart   # Restart services
./init_and_deploy.sh logs      # View logs
./init_and_deploy.sh test      # Run tests
./init_and_deploy.sh clean     # Cleanup
```

## Testing

Run comprehensive tests:
```bash
python3 test_system.py
```

Quick connectivity test:
```bash
python3 test_system.py quick
```

## API Endpoints

### Core Endpoints
- `GET /` - Health check
- `GET /health` - Detailed health status
- `GET /portfolio/metrics` - Portfolio metrics
- `GET /signals/latest` - Latest trading signal

### Trading
- `POST /trades/` - Create trade
- `GET /trades/` - Get trade history
- `POST /limits/` - Create limit order
- `GET /limits/` - Get active limits

### Market Data
- `GET /market/btc-data` - BTC price data
- `GET /signals/history` - Signal history
- `GET /analytics/pnl` - P&L analytics

## Configuration

Edit `storage/config/trading_config.json`:
```json
{
    "trading": {
        "risk_tolerance": 0.02,
        "max_position_size": 1.0,
        "stop_loss_percentage": 0.05
    },
    "model": {
        "sequence_length": 60,
        "confidence_threshold": 0.7
    }
}
```

## File Structure

```
btc-trading-system/
├── backend_api.py              # FastAPI backend
├── database_models.py          # Database management
├── lstm_model.py              # AI trading model
├── streamlit_app.py           # Frontend interface
├── test_system.py             # Test suite
├── docker-compose.yml         # Container orchestration
├── init_and_deploy.sh         # Deployment script
└── storage/                   # Persistent data
    ├── data/                  # Database files
    ├── models/                # AI model files
    ├── logs/                  # Application logs
    └── config/                # Configuration files
```

## Requirements

- **Docker** and **Docker Compose**
- **2GB+ RAM** recommended
- **1GB+ disk space** for data storage

## Development

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run backend
uvicorn backend_api:app --reload --port 8080

# Run frontend (separate terminal)
streamlit run streamlit_app.py --server.port 8501
```

### Environment Variables
```bash
DATABASE_PATH=/app/data/trading_system.db
MODEL_PATH=/app/models
API_BASE_URL=http://localhost:8080
```

## Architecture

- **Backend:** FastAPI with SQLite database
- **Frontend:** Streamlit web interface
- **AI Model:** PyTorch LSTM for signal generation
- **Data:** Real-time BTC price feeds via yfinance
- **Deployment:** Docker containers with health checks

## Monitoring

- **Logs:** `./init_and_deploy.sh logs`
- **Health:** http://localhost:8080/health
- **Metrics:** Available in frontend dashboard
- **Tests:** Automated test suite included

## Support

- Check logs for errors: `docker compose logs -f`
- Run tests: `python3 test_system.py`
- Restart services: `./init_and_deploy.sh restart`

## License

This project is for **educational and demonstration purposes only**.
