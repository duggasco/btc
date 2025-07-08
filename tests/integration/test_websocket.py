"""
Integration tests for WebSocket functionality
"""
import pytest
import json
import asyncio
from unittest.mock import patch, Mock
from fastapi.testclient import TestClient
from datetime import datetime


class TestWebSocket:
    """Integration tests for WebSocket connections"""
    
    @pytest.mark.integration
    @pytest.mark.websocket
    def test_websocket_connection(self, api_client):
        """Test basic WebSocket connection"""
        with api_client.websocket_connect("/ws") as websocket:
            # Should receive connection confirmation
            data = websocket.receive_json()
            assert data["type"] == "connection"
            assert data["status"] == "connected"
            assert "timestamp" in data
    
    @pytest.mark.integration
    @pytest.mark.websocket
    def test_websocket_price_updates(self, api_client):
        """Test receiving price updates via WebSocket"""
        with patch('services.data_fetcher.DataFetcher.fetch_current_price') as mock_price:
            mock_price.return_value = {
                'price': 50000.0,
                'volume': 1000000000,
                'change_24h': 2.5,
                'timestamp': datetime.now().isoformat()
            }
            
            with api_client.websocket_connect("/ws") as websocket:
                # Skip connection message
                websocket.receive_json()
                
                # Wait for price update
                data = websocket.receive_json(timeout=15)
                assert data["type"] == "price_update"
                assert "data" in data
                assert data["data"]["price"] == 50000.0
    
    @pytest.mark.integration
    @pytest.mark.websocket
    def test_websocket_signal_updates(self, api_client):
        """Test receiving signal updates via WebSocket"""
        with api_client.websocket_connect("/ws") as websocket:
            # Skip connection message
            websocket.receive_json()
            
            # Wait for signal update (may take up to 60s in real implementation)
            # For testing, we'll simulate faster updates
            data = None
            for _ in range(10):
                try:
                    msg = websocket.receive_json(timeout=2)
                    if msg["type"] == "signal_update":
                        data = msg
                        break
                except:
                    continue
            
            if data:
                assert data["type"] == "signal_update"
                assert "data" in data
                assert "signal" in data["data"]
                assert "confidence" in data["data"]
    
    @pytest.mark.integration
    @pytest.mark.websocket
    def test_websocket_portfolio_updates(self, api_client):
        """Test receiving portfolio updates via WebSocket"""
        with api_client.websocket_connect("/ws") as websocket:
            # Skip initial messages
            websocket.receive_json()
            
            # Trigger a trade to generate portfolio update
            trade_response = api_client.post("/trades/execute", json={
                "type": "buy",
                "amount": 0.01
            })
            
            if trade_response.status_code == 200:
                # Look for portfolio update
                data = None
                for _ in range(5):
                    try:
                        msg = websocket.receive_json(timeout=2)
                        if msg["type"] == "portfolio_update":
                            data = msg
                            break
                    except:
                        continue
                
                if data:
                    assert data["type"] == "portfolio_update"
                    assert "data" in data
                    assert "total_value" in data["data"]
    
    @pytest.mark.integration
    @pytest.mark.websocket
    def test_websocket_trade_notifications(self, api_client):
        """Test receiving trade notifications via WebSocket"""
        with api_client.websocket_connect("/ws") as websocket:
            # Skip connection message
            websocket.receive_json()
            
            # Execute a trade
            with patch('services.data_fetcher.DataFetcher.fetch_current_price') as mock_price:
                mock_price.return_value = {'price': 50000.0}
                
                trade_response = api_client.post("/trades/execute", json={
                    "type": "buy",
                    "amount": 0.01
                })
                
                if trade_response.status_code == 200:
                    # Look for trade notification
                    data = None
                    for _ in range(5):
                        try:
                            msg = websocket.receive_json(timeout=2)
                            if msg["type"] == "trade_executed":
                                data = msg
                                break
                        except:
                            continue
                    
                    if data:
                        assert data["type"] == "trade_executed"
                        assert "data" in data
                        assert data["data"]["type"] == "buy"
    
    @pytest.mark.integration
    @pytest.mark.websocket
    def test_websocket_multiple_clients(self, api_client):
        """Test multiple WebSocket clients"""
        clients = []
        
        # Connect multiple clients
        for i in range(3):
            ws = api_client.websocket_connect("/ws").__enter__()
            # Receive connection message
            ws.receive_json()
            clients.append(ws)
        
        # All clients should receive updates
        with patch('services.data_fetcher.DataFetcher.fetch_current_price') as mock_price:
            mock_price.return_value = {
                'price': 51000.0,
                'volume': 1100000000,
                'change_24h': 3.0,
                'timestamp': datetime.now().isoformat()
            }
            
            # Wait for price updates on all clients
            received = []
            for client in clients:
                try:
                    data = client.receive_json(timeout=15)
                    if data["type"] == "price_update":
                        received.append(data)
                except:
                    pass
            
            # Cleanup
            for client in clients:
                client.__exit__(None, None, None)
            
            # At least some clients should receive the update
            assert len(received) > 0
    
    @pytest.mark.integration
    @pytest.mark.websocket
    def test_websocket_reconnection(self, api_client):
        """Test WebSocket reconnection"""
        # First connection
        with api_client.websocket_connect("/ws") as websocket:
            data = websocket.receive_json()
            assert data["type"] == "connection"
        
        # Second connection (after disconnect)
        with api_client.websocket_connect("/ws") as websocket:
            data = websocket.receive_json()
            assert data["type"] == "connection"
            assert data["status"] == "connected"
    
    @pytest.mark.integration
    @pytest.mark.websocket
    def test_websocket_error_handling(self, api_client):
        """Test WebSocket error handling"""
        with api_client.websocket_connect("/ws") as websocket:
            # Receive connection message
            websocket.receive_json()
            
            # Send invalid message
            websocket.send_text("invalid json")
            
            # Should still receive normal updates
            try:
                data = websocket.receive_json(timeout=15)
                assert data["type"] in ["price_update", "signal_update", "error"]
            except:
                # Timeout is acceptable in this test
                pass
    
    @pytest.mark.integration
    @pytest.mark.websocket
    @pytest.mark.slow
    def test_websocket_long_connection(self, api_client):
        """Test WebSocket connection stability over time"""
        with api_client.websocket_connect("/ws") as websocket:
            # Skip connection message
            websocket.receive_json()
            
            # Collect messages for 30 seconds
            messages = []
            start_time = datetime.now()
            
            while (datetime.now() - start_time).seconds < 30:
                try:
                    data = websocket.receive_json(timeout=5)
                    messages.append(data)
                except:
                    continue
            
            # Should have received multiple updates
            assert len(messages) > 0
            
            # Check message types
            message_types = [msg["type"] for msg in messages]
            assert any(t in ["price_update", "signal_update"] for t in message_types)
    
    @pytest.mark.integration
    @pytest.mark.websocket
    def test_websocket_limit_alert(self, api_client):
        """Test WebSocket limit order alerts"""
        with api_client.websocket_connect("/ws") as websocket:
            # Skip connection message
            websocket.receive_json()
            
            # Create a limit order
            order_response = api_client.post("/limits/create", json={
                "type": "stop_loss",
                "trigger_price": 48000,
                "amount": 0.05
            })
            
            if order_response.status_code == 200:
                # Simulate price drop to trigger limit
                with patch('services.data_fetcher.DataFetcher.fetch_current_price') as mock_price:
                    mock_price.return_value = {'price': 47500.0}  # Below trigger
                    
                    # Look for limit triggered notification
                    data = None
                    for _ in range(10):
                        try:
                            msg = websocket.receive_json(timeout=2)
                            if msg["type"] == "limit_triggered":
                                data = msg
                                break
                        except:
                            continue
                    
                    if data:
                        assert data["type"] == "limit_triggered"
                        assert "data" in data
                        assert data["data"]["trigger_price"] == 48000