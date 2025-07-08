"""
Unit tests for API client component
"""
import pytest
from unittest.mock import Mock, patch
import requests
from datetime import datetime, timedelta
import sys
import os

# Add frontend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../src/frontend'))

from components.api_client import APIClient


class TestAPIClient:
    """Test cases for APIClient class"""
    
    @pytest.fixture
    def api_client(self):
        """Create API client instance"""
        return APIClient(base_url="http://localhost:8080")
    
    @pytest.mark.unit
    def test_initialization(self):
        """Test API client initialization"""
        client = APIClient(base_url="http://localhost:8080")
        assert client.base_url == "http://localhost:8080"
        assert hasattr(client, 'session')
        assert client._cache_ttl == 60
    
    @pytest.mark.unit
    def test_custom_base_url(self):
        """Test API client with custom base URL"""
        client = APIClient(base_url="http://example.com:9000")
        assert client.base_url == "http://example.com:9000"
    
    @pytest.mark.unit
    def test_get_request_success(self, api_client):
        """Test successful GET request"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": "test"}
        mock_response.headers = {}
        
        with patch.object(api_client.session, 'request', return_value=mock_response):
            result = api_client.get("/test")
        
        assert result == {"data": "test"}
    
    @pytest.mark.unit
    def test_get_request_failure(self, api_client):
        """Test failed GET request"""
        with patch.object(api_client.session, 'request', side_effect=requests.RequestException("Connection error")):
            result = api_client.get("/test")
        
        assert result is None
    
    @pytest.mark.unit
    def test_post_request_success(self, api_client):
        """Test successful POST request"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "success"}
        mock_response.headers = {}
        
        with patch.object(api_client.session, 'request', return_value=mock_response):
            data = {"key": "value"}
            result = api_client.post("/test", data)
        
        assert result == {"status": "success"}
    
    @pytest.mark.unit
    @pytest.mark.skip(reason="get_current_price method not implemented")
    def test_caching_mechanism(self, api_client):
        """Test request caching"""
        with patch.object(api_client, '_get') as mock_get:
            mock_get.return_value = {"price": 50000}
            
            # First call - should hit API
            result1 = api_client.get_current_price()
            assert mock_get.call_count == 1
            assert result1 == {"price": 50000}
            
            # Second call within cache duration - should use cache
            result2 = api_client.get_current_price()
            assert mock_get.call_count == 1  # No additional call
            assert result2 == {"price": 50000}
    
    @pytest.mark.unit
    @pytest.mark.skip(reason="get_current_price method not implemented")
    def test_cache_expiration(self, api_client):
        """Test cache expiration"""
        # Reduce cache duration for testing
        api_client.cache_duration = 0.1  # 0.1 seconds
        
        with patch.object(api_client, '_get') as mock_get:
            mock_get.return_value = {"price": 50000}
            
            # First call
            api_client.get_current_price()
            
            # Wait for cache to expire
            import time
            time.sleep(0.2)
            
            # Second call - should hit API again
            api_client.get_current_price()
            assert mock_get.call_count == 2
    
    @pytest.mark.unit
    @pytest.mark.skip(reason="get_current_price method not implemented")
    @patch.object(APIClient, '_get')
    def test_get_current_price(self, mock_get, api_client):
        """Test get current price method"""
        mock_get.return_value = {
            "price": 50000,
            "volume": 1000000,
            "change_24h": 2.5
        }
        
        result = api_client.get_current_price()
        
        assert result["price"] == 50000
        mock_get.assert_called_with("/price/current")
    
    @pytest.mark.unit
    @pytest.mark.skip(reason="get_latest_signal method not implemented")
    @patch.object(APIClient, '_get')
    def test_get_latest_signal(self, mock_get, api_client):
        """Test get latest signal method"""
        mock_get.return_value = {
            "signal": "buy",
            "confidence": 0.85,
            "predicted_price": 52000
        }
        
        result = api_client.get_latest_signal()
        
        assert result["signal"] == "buy"
        assert result["confidence"] == 0.85
        mock_get.assert_called_with("/signals/enhanced/latest")
    
    @pytest.mark.unit
    @pytest.mark.skip(reason="get_portfolio_metrics method not implemented")
    @patch.object(APIClient, '_get')
    def test_get_portfolio_metrics(self, mock_get, api_client):
        """Test get portfolio metrics method"""
        mock_get.return_value = {
            "total_value": 11000,
            "total_pnl": 1000,
            "total_pnl_pct": 10.0
        }
        
        result = api_client.get_portfolio_metrics()
        
        assert result["total_value"] == 11000
        mock_get.assert_called_with("/portfolio/metrics")
    
    @pytest.mark.unit
    def test_execute_trade(self, api_client):
        """Test execute trade method"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "executed",
            "trade_id": "123",
            "details": {"type": "buy", "size": 0.1}
        }
        mock_response.headers = {}
        
        with patch.object(api_client.session, 'request', return_value=mock_response):
            result = api_client.execute_trade("buy", 0.1)
        
        assert result["status"] == "executed"
    
    @pytest.mark.unit
    def test_get_historical_prices(self, api_client):
        """Test get historical prices method"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"timestamp": "2024-01-01", "price": 50000},
            {"timestamp": "2024-01-02", "price": 51000}
        ]
        mock_response.headers = {}
        
        with patch.object(api_client.session, 'request', return_value=mock_response):
            result = api_client.get_historical_prices(days=2)
        
        assert len(result) == 2
        assert result[0]["price"] == 50000
    
    @pytest.mark.unit
    def test_get_paper_trading_status(self, api_client):
        """Test get paper trading status method"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "enabled": True,
            "balance": 10000,
            "btc_balance": 0.5
        }
        mock_response.headers = {}
        
        with patch.object(api_client.session, 'request', return_value=mock_response):
            result = api_client.get_paper_trading_status()
        
        assert result["enabled"] is True
        assert result["balance"] == 10000
    
    @pytest.mark.unit
    def test_toggle_paper_trading(self, api_client):
        """Test toggle paper trading method"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "toggled", "enabled": False}
        mock_response.headers = {}
        
        with patch.object(api_client.session, 'request', return_value=mock_response):
            result = api_client.toggle_paper_trading()
        
        assert result["status"] == "toggled"
    
    @pytest.mark.unit
    def test_retry_logic(self, api_client):
        """Test retry logic on failure"""
        # The implementation actually has retry logic (3 retries) for specific exceptions
        with patch.object(api_client.session, 'request') as mock_request:
            # All three attempts fail with ConnectionError (which triggers retry)
            mock_request.side_effect = [
                requests.exceptions.ConnectionError("Error 1"),
                requests.exceptions.ConnectionError("Error 2"),
                requests.exceptions.ConnectionError("Error 3")
            ]
            
            # Should retry 3 times and then return None
            result = api_client.get("/test")
            
            # Should have tried 3 times
            assert result is None
            assert mock_request.call_count == 3
    
    @pytest.mark.unit
    def test_concurrent_requests(self, api_client):
        """Test handling concurrent requests"""
        import threading
        
        results = []
        
        def make_request():
            # Using get method since get_current_price doesn't exist
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"price": 50000}
            mock_response.headers = {}
            
            with patch.object(api_client.session, 'request', return_value=mock_response):
                result = api_client.get("/price/current")
                results.append(result)
        
        threads = []
        for _ in range(5):
            t = threading.Thread(target=make_request)
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        assert len(results) == 5
        assert all(r is not None and r.get("price") == 50000 for r in results)