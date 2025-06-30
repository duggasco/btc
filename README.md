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

1. **Make the deployment script executable:**
   ```bash
   chmod +x deploy.sh
   ```

2. **Deploy the system:**
   ```bash
   ./deploy.sh setup
   ```

3. **Access the interfaces:**
   - **Frontend UI:** http://localhost:8501
   - **Backend API:** http://localhost:8000  
   - **API Documentation:** http://localhost:8000/docs

## Management Commands

```bash
./deploy.sh setup     # Initial setup and start
./deploy.sh start     # Start services
./deploy.sh stop      # Stop services  
./deploy.sh restart   # Restart services
./deploy.sh logs      # View logs
./deploy.sh build     # Rebuild containers
./deploy.sh clean     # Stop and cleanup
```

## Requirements

- **Docker** and **Docker Compose**
- **2GB+ RAM** recommended
- **1GB+ disk space** for data storage

## File Structure

```
btc-trading-system/
├── deploy.sh                  # Deployment script
├── docker-compose.yml         # Docker orchestration
├── Dockerfile.backend         # Backend container config
├── Dockerfile.frontend        # Frontend container config
├── requirements-backend.txt   # Backend dependencies
├── requirements-frontend.txt  # Frontend dependencies
├── database_models.py         # Database management
├── lstm_model.py             # AI trading model
├── backend_api.py            # FastAPI backend
├── streamlit_app.py          # Streamlit frontend
├── README.md                 # This file
└── storage/                  # Persistent data (auto-created)
```

## Support

This system creates a complete BTC trading environment with AI-powered signals, real-time data, portfolio tracking, and persistent storage.

For issues, check the logs: `./deploy.sh logs`

## License

This project is for **educational and demonstration purposes only**.
