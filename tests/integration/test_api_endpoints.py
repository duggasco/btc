"""
Integration tests for API endpoints
"""
import pytest
import json
from datetime import datetime
from unittest.mock import patch, Mock
from fastapi.testclient import TestClient


class TestAPIEndpoints:
    """Integration tests for all API endpoints"""
    
    @pytest.mark.integration
    def test_root_endpoint(self, api_client):
        """Test root endpoint"""
        response = api_client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Enhanced BTC Trading System API is running"
        assert "version" in data
        assert "status" in data
    
    @pytest.mark.integration
    def test_health_endpoint(self, api_client):
        """Test health check endpoint"""
        response = api_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "components" in data
        assert "database" in data["components"]
        assert "signal_generator" in data["components"]
        assert "timestamp" in data
    
    @pytest.mark.integration
    @patch('services.data_fetcher.DataFetcher.fetch_current_price')
    def test_current_price_endpoint(self, mock_fetch, api_client):
        """Test current price endpoint"""
        mock_fetch.return_value = {
            'price': 50000.0,
            'volume': 1000000000,
            'change_24h': 2.5,
            'timestamp': datetime.now().isoformat()
        }
        
        response = api_client.get("/price/current")
        assert response.status_code == 200
        data = response.json()
        assert data["price"] == 50000.0
        assert data["volume"] == 1000000000
        assert data["change_24h"] == 2.5
    
    @pytest.mark.integration
    @patch('services.data_fetcher.DataFetcher.fetch_historical_data')
    def test_historical_prices_endpoint(self, mock_fetch, api_client):
        """Test historical prices endpoint"""
        import pandas as pd
        mock_df = pd.DataFrame({
            'timestamp': pd.date_range(end=datetime.now(), periods=5, freq='D'),
            'price': [48000, 49000, 50000, 51000, 52000]
        })
        mock_fetch.return_value = mock_df
        
        response = api_client.get("/price/history?days=5")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 5
        assert all('timestamp' in item and 'price' in item for item in data)
    
    @pytest.mark.integration
    def test_latest_signal_endpoint(self, api_client, mock_lstm_model):
        """Test latest signal endpoint"""
        response = api_client.get("/signals/latest")
        assert response.status_code == 200
        data = response.json()
        assert "signal" in data
        assert "confidence" in data
        assert "predicted_price" in data
        assert "timestamp" in data
    
    @pytest.mark.integration
    def test_signal_history_endpoint(self, api_client):
        """Test signal history endpoint"""
        # Import the db instance from the API
        from api.main import db
        
        # Add some test signals using the API's db instance
        for i in range(3):
            db.add_model_signal(
                symbol='BTC-USD',
                signal=['buy', 'hold', 'sell'][i],
                confidence=0.7 + i * 0.1,
                price_prediction=50000 + i * 1000
            )
        
        response = api_client.get("/signals/history?limit=3")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 3
        assert data[0]['signal'] == 'sell'  # Most recent
    
    @pytest.mark.integration
    def test_portfolio_metrics_endpoint(self, api_client):
        """Test portfolio metrics endpoint"""
        response = api_client.get("/portfolio/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "total_trades" in data
        assert "total_pnl" in data
        assert "positions_count" in data
        assert "total_invested" in data
    
    @pytest.mark.integration
    def test_execute_trade_endpoint(self, api_client):
        """Test trade execution endpoint"""
        trade_data = {
            "trade_type": "buy",
            "size": 0.01
        }
        
        with patch('services.data_fetcher.DataFetcher.fetch_current_price') as mock_price:
            mock_price.return_value = {'price': 50000.0}
            
            response = api_client.post("/trades/execute", json=trade_data)
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert "trade_id" in data
            assert "message" in data
            assert "pnl" in data
    
    @pytest.mark.integration
    def test_trade_history_endpoint(self, api_client):
        """Test trade history endpoint"""
        # Import the db instance from the API
        from api.main import db
        
        # Add some test trades using the API's db instance
        db.add_trade(
            symbol='BTC-USD',
            trade_type='buy',
            price=50000,
            size=0.1
        )
        db.add_trade(
            symbol='BTC-USD',
            trade_type='sell',
            price=51000,
            size=0.05
        )
        
        response = api_client.get("/trades/history")
        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 2
        assert data[0]['type'] == 'sell'  # Most recent
    
    @pytest.mark.integration
    def test_limit_orders_endpoints(self, api_client):
        """Test limit order endpoints"""
        # Create limit order
        order_data = {
            "type": "stop_loss",
            "trigger_price": 48000,
            "amount": 0.05
        }
        
        response = api_client.post("/limits/create", json=order_data)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "created"
        order_id = data["order_id"]
        
        # Get active limits
        response = api_client.get("/limits/active")
        assert response.status_code == 200
        data = response.json()
        assert len(data) > 0
        assert any(order['id'] == order_id for order in data)
        
        # Cancel limit order
        response = api_client.delete(f"/limits/{order_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "cancelled"
    
    @pytest.mark.integration
    def test_paper_trading_endpoints(self, api_client):
        """Test paper trading endpoints"""
        # Get status
        response = api_client.get("/paper-trading/status")
        assert response.status_code == 200
        data = response.json()
        assert "enabled" in data
        assert "portfolio" in data
        assert "btc_balance" in data["portfolio"]
        assert "usd_balance" in data["portfolio"]
        
        # Toggle paper trading
        response = api_client.post("/paper-trading/toggle")
        assert response.status_code == 200
        
        # Reset portfolio
        response = api_client.post("/paper-trading/reset")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["message"] == "Paper trading portfolio reset"
        
        # Get history
        response = api_client.get("/paper-trading/history")
        assert response.status_code == 200
        data = response.json()
        assert "trades" in data
        assert "metrics" in data
    
    @pytest.mark.integration
    @patch('services.data_fetcher.DataFetcher.fetch_market_data')
    def test_market_data_endpoint(self, mock_fetch, api_client):
        """Test market data endpoint"""
        mock_fetch.return_value = {
            'price': {'price': 50000, 'volume': 1000000, 'change_24h': 2.5},
            'fear_greed': 65,
            'network_stats': {'daily_transactions': 300000}
        }
        
        response = api_client.get("/market/data")
        assert response.status_code == 200
        data = response.json()
        assert "price" in data
        assert "fear_greed" in data
        assert "network_stats" in data
    
    @pytest.mark.integration
    def test_config_endpoints(self, api_client):
        """Test configuration endpoints"""
        # Get config
        response = api_client.get("/config/trading-rules")
        assert response.status_code == 200
        data = response.json()
        assert "min_trade_size" in data
        assert "max_position_size" in data
        
        # Update config
        new_config = {
            "min_trade_size": 0.001,
            "max_position_size": 0.2,
            "stop_loss_pct": 3.0
        }
        response = api_client.post("/config/trading-rules", json=new_config)
        assert response.status_code == 200
        
        # Get signal weights
        response = api_client.get("/config/signal-weights")
        assert response.status_code == 200
    
    @pytest.mark.integration
    def test_error_handling(self, api_client):
        """Test API error handling"""
        # Test 404
        response = api_client.get("/nonexistent")
        assert response.status_code == 404
        
        # Test invalid trade
        response = api_client.post("/trades/execute", json={"type": "invalid"})
        assert response.status_code == 422
        
        # Test invalid limit order
        response = api_client.post("/limits/create", json={})
        assert response.status_code == 422
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_websocket_endpoint(self, api_client):
        """Test WebSocket endpoint"""
        with api_client.websocket_connect("/ws") as websocket:
            # Should receive initial connection message
            data = websocket.receive_json()
            assert data["type"] == "connection"
            assert data["status"] == "connected"
            
            # Should receive periodic updates
            data = websocket.receive_json()
            assert data["type"] in ["price_update", "signal_update"]
    
    @pytest.mark.integration
    def test_cors_headers(self, api_client):
        """Test CORS headers are properly set"""
        response = api_client.get("/", headers={"Origin": "http://localhost:3000"})
        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers
    
    @pytest.mark.integration
    def test_rate_limiting(self, api_client):
        """Test that API handles high request volume"""
        # Make multiple rapid requests
        responses = []
        for _ in range(50):
            response = api_client.get("/price/current")
            responses.append(response)
        
        # All should succeed (no rate limiting implemented yet)
        assert all(r.status_code == 200 for r in responses)
    
    @pytest.mark.integration
    def test_concurrent_requests(self, api_client):
        """Test handling of concurrent requests"""
        import threading
        
        results = []
        
        def make_request():
            response = api_client.get("/signals/latest")
            results.append(response.status_code)
        
        threads = []
        for _ in range(20):
            t = threading.Thread(target=make_request)
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        assert all(status == 200 for status in results)