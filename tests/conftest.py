"""
Global pytest fixtures and configuration for BTC Trading System tests
"""
import pytest
import asyncio
import tempfile
import shutil
import os
import sys
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np

# Add src directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'backend'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'frontend'))

# Import after path setup
from fastapi.testclient import TestClient
from models.database import DatabaseManager
from models.paper_trading import PersistentPaperTrading


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="function")
def temp_dir():
    """Create a temporary directory for test data."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_db_path(temp_dir):
    """Create a temporary database path."""
    return os.path.join(temp_dir, "test_trading_system.db")


@pytest.fixture
def mock_model_path(temp_dir):
    """Create a temporary model directory."""
    model_dir = os.path.join(temp_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    return model_dir


@pytest.fixture
def mock_config_path(temp_dir):
    """Create a temporary config directory with test config."""
    config_dir = os.path.join(temp_dir, "config")
    os.makedirs(config_dir, exist_ok=True)
    
    # Create test trading config
    import json
    config_data = {
        "min_trade_size": 0.001,
        "max_position_size": 0.1,
        "stop_loss_pct": 5.0,
        "take_profit_pct": 10.0,
        "buy_threshold": 0.6,
        "sell_threshold": 0.6,
        "enable_discord": False,
        "enable_limit_orders": True,
        "paper_trading_enabled": True,
        "initial_balance": 10000.0
    }
    
    with open(os.path.join(config_dir, "trading_config.json"), "w") as f:
        json.dump(config_data, f)
    
    return config_dir


@pytest.fixture
def db_manager(mock_db_path):
    """Create a test database manager."""
    manager = DatabaseManager(mock_db_path)
    manager.initialize_database()
    yield manager
    # Cleanup handled by temp_dir fixture


@pytest.fixture
def paper_trading(mock_db_path):
    """Create a test paper trading instance."""
    return PersistentPaperTrading(db_path=mock_db_path)


@pytest.fixture
def sample_price_data():
    """Generate sample price data for testing."""
    dates = pd.date_range(end=datetime.now(), periods=100, freq='H')
    prices = 50000 + np.random.randn(100).cumsum() * 1000
    
    return pd.DataFrame({
        'timestamp': dates,
        'open': prices + np.random.randn(100) * 100,
        'high': prices + abs(np.random.randn(100) * 200),
        'low': prices - abs(np.random.randn(100) * 200),
        'close': prices,
        'volume': np.random.randint(1000, 10000, 100)
    })


@pytest.fixture
def sample_signals():
    """Generate sample trading signals."""
    return {
        'signal': 'buy',
        'confidence': 0.75,
        'predicted_price': 51000.0,
        'current_price': 50000.0,
        'indicators': {
            'rsi': 45.5,
            'macd': 0.02,
            'bb_position': 0.3,
            'volume_ratio': 1.2
        }
    }


@pytest.fixture
def mock_discord_notifier():
    """Mock Discord notifier to prevent actual notifications."""
    with patch('services.notifications.DiscordNotifier') as mock:
        notifier = Mock()
        notifier.enabled = False
        mock.return_value = notifier
        yield notifier


@pytest.fixture
def mock_external_apis():
    """Mock all external API calls."""
    with patch('requests.get') as mock_get, \
         patch('requests.post') as mock_post:
        
        # Mock successful responses
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            'price': 50000.0,
            'volume': 1000000,
            'change_24h': 2.5
        }
        
        mock_post.return_value.status_code = 200
        
        yield {'get': mock_get, 'post': mock_post}


@pytest.fixture
def api_client(mock_db_path, mock_model_path, mock_config_path):
    """Create a test FastAPI client."""
    # Set environment variables for test
    os.environ['DATABASE_PATH'] = mock_db_path
    os.environ['MODEL_PATH'] = mock_model_path
    os.environ['CONFIG_PATH'] = mock_config_path
    os.environ['DISCORD_WEBHOOK_URL'] = ''  # Disable Discord
    
    # Import app after setting env vars
    from api.main import app
    
    with TestClient(app) as client:
        yield client


@pytest.fixture
def websocket_url():
    """WebSocket URL for testing."""
    return "ws://localhost:8000/ws"


@pytest.fixture
def mock_lstm_model(mock_model_path):
    """Mock LSTM model for testing."""
    with patch('models.lstm.TradingSignalGenerator') as mock_model:
        instance = Mock()
        instance.predict.return_value = {
            'signal': 'buy',
            'confidence': 0.75,
            'predicted_price': 51000.0
        }
        mock_model.return_value = instance
        yield instance


# Utility functions for tests
def assert_valid_response(response, expected_status=200):
    """Helper to assert API response is valid."""
    assert response.status_code == expected_status
    if expected_status == 200:
        assert response.json() is not None
    return response.json() if expected_status == 200 else None


def create_test_trade(trade_type="buy", price=50000, size=0.001):
    """Create a test trade object."""
    return {
        'type': trade_type,
        'price': price,
        'size': size,
        'timestamp': datetime.now().isoformat(),
        'value': price * size
    }