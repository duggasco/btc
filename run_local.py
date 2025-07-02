#!/usr/bin/env python3
"""Run the API locally with proper imports"""

import sys
import os

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src/backend'))

# Set up module aliases for main.py
import models.paper_trading
import models.database
import models.lstm
import services.data_fetcher
import services.integration
import services.backtesting
import services.notifications

sys.modules['paper_trading_persistence'] = models.paper_trading
sys.modules['database_models'] = models.database
sys.modules['lstm_model'] = models.lstm
sys.modules['external_data_fetcher'] = services.data_fetcher
sys.modules['integration'] = services.integration
sys.modules['backtesting_system'] = services.backtesting
sys.modules['discord_notifications'] = services.notifications

# Now import and run the main app
from api.main import app
import uvicorn

if __name__ == "__main__":
    print("Starting BTC Trading System API (Local Development)...")
    print("API will be available at: http://localhost:8000")
    print("API docs: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
