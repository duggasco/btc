"""
End-to-end tests for complete trading workflows
"""
import pytest
import time
from datetime import datetime
from unittest.mock import patch, Mock
import threading


class TestTradingWorkflows:
    """Test complete trading workflows from user perspective"""
    
    @pytest.mark.e2e
    @pytest.mark.requires_docker
    def test_complete_trading_cycle(self, api_client, mock_external_apis):
        """Test complete trading cycle: signal -> trade -> portfolio update"""
        # Step 1: Get initial portfolio state
        initial_portfolio = api_client.get("/portfolio/metrics").json()
        initial_balance = initial_portfolio["positions"]["usd_balance"]
        
        # Step 2: Wait for or trigger a buy signal
        with patch('models.lstm.TradingSignalGenerator.predict') as mock_predict:
            mock_predict.return_value = {
                'signal': 'buy',
                'confidence': 0.85,
                'predicted_price': 52000.0
            }
            
            # Get the signal
            signal = api_client.get("/signals/latest").json()
            assert signal["signal"] == "buy"
            assert signal["confidence"] >= 0.8
        
        # Step 3: Execute trade based on signal
        with patch('services.data_fetcher.DataFetcher.fetch_current_price') as mock_price:
            mock_price.return_value = {'price': 50000.0}
            
            trade_response = api_client.post("/trades/execute", json={
                "type": "buy",
                "amount": 0.1
            })
            assert trade_response.status_code == 200
            trade = trade_response.json()
            assert trade["status"] == "executed"
        
        # Step 4: Verify portfolio updated
        updated_portfolio = api_client.get("/portfolio/metrics").json()
        assert updated_portfolio["positions"]["btc_balance"] > 0
        assert updated_portfolio["positions"]["usd_balance"] < initial_balance
        
        # Step 5: Wait for price change and sell signal
        with patch('models.lstm.TradingSignalGenerator.predict') as mock_predict:
            mock_predict.return_value = {
                'signal': 'sell',
                'confidence': 0.9,
                'predicted_price': 48000.0
            }
            
            signal = api_client.get("/signals/latest").json()
            assert signal["signal"] == "sell"
        
        # Step 6: Execute sell trade
        with patch('services.data_fetcher.DataFetcher.fetch_current_price') as mock_price:
            mock_price.return_value = {'price': 55000.0}  # Price went up
            
            trade_response = api_client.post("/trades/execute", json={
                "type": "sell",
                "amount": 0.05
            })
            assert trade_response.status_code == 200
        
        # Step 7: Verify profit
        final_portfolio = api_client.get("/portfolio/metrics").json()
        assert final_portfolio["total_pnl"] > 0  # Should have profit
    
    @pytest.mark.e2e
    def test_paper_trading_workflow(self, api_client):
        """Test complete paper trading workflow"""
        # Step 1: Enable paper trading
        response = api_client.get("/paper-trading/status")
        status = response.json()
        
        if not status["enabled"]:
            api_client.post("/paper-trading/toggle")
        
        # Step 2: Reset portfolio to start fresh
        reset_response = api_client.post("/paper-trading/reset")
        assert reset_response.status_code == 200
        assert reset_response.json()["balance"] == 10000
        
        # Step 3: Execute multiple trades
        with patch('services.data_fetcher.DataFetcher.fetch_current_price') as mock_price:
            # Buy trades
            mock_price.return_value = {'price': 50000.0}
            api_client.post("/trades/execute", json={"type": "buy", "amount": 0.1})
            
            mock_price.return_value = {'price': 48000.0}
            api_client.post("/trades/execute", json={"type": "buy", "amount": 0.1})
            
            # Sell trade at profit
            mock_price.return_value = {'price': 52000.0}
            api_client.post("/trades/execute", json={"type": "sell", "amount": 0.15})
        
        # Step 4: Check performance metrics
        history = api_client.get("/paper-trading/history").json()
        metrics = history["metrics"]
        
        assert metrics["total_trades"] >= 3
        assert metrics["profitable_trades"] >= 1
        assert "win_rate" in metrics
        assert "sharpe_ratio" in metrics
    
    @pytest.mark.e2e
    def test_limit_order_workflow(self, api_client):
        """Test limit order creation and execution workflow"""
        # Step 1: Create stop loss order
        with patch('services.data_fetcher.DataFetcher.fetch_current_price') as mock_price:
            mock_price.return_value = {'price': 50000.0}
            
            order_response = api_client.post("/limits/create", json={
                "type": "stop_loss",
                "trigger_price": 48000,
                "amount": 0.05
            })
            assert order_response.status_code == 200
            order_id = order_response.json()["order_id"]
        
        # Step 2: Verify order is active
        active_orders = api_client.get("/limits/active").json()
        assert any(order["id"] == order_id for order in active_orders)
        
        # Step 3: Simulate price drop to trigger order
        with patch('services.data_fetcher.DataFetcher.fetch_current_price') as mock_price:
            mock_price.return_value = {'price': 47500.0}  # Below trigger
            
            # Wait for order processing (in real system)
            time.sleep(2)
            
            # Check if order was executed
            active_orders = api_client.get("/limits/active").json()
            # Order should no longer be active
            assert not any(order["id"] == order_id for order in active_orders)
    
    @pytest.mark.e2e
    @pytest.mark.websocket
    def test_real_time_updates_workflow(self, api_client):
        """Test real-time updates via WebSocket"""
        received_updates = []
        
        def collect_updates():
            with api_client.websocket_connect("/ws") as websocket:
                # Collect updates for 10 seconds
                start_time = time.time()
                while time.time() - start_time < 10:
                    try:
                        data = websocket.receive_json(timeout=1)
                        received_updates.append(data)
                    except:
                        continue
        
        # Start WebSocket collection in background
        ws_thread = threading.Thread(target=collect_updates)
        ws_thread.start()
        
        # Generate some activity
        time.sleep(1)  # Let WebSocket connect
        
        # Execute a trade
        with patch('services.data_fetcher.DataFetcher.fetch_current_price') as mock_price:
            mock_price.return_value = {'price': 50000.0}
            api_client.post("/trades/execute", json={"type": "buy", "amount": 0.01})
        
        # Wait for WebSocket collection to finish
        ws_thread.join()
        
        # Verify updates were received
        assert len(received_updates) > 0
        
        update_types = [update["type"] for update in received_updates]
        assert "connection" in update_types
        assert any(t in update_types for t in ["price_update", "trade_executed", "portfolio_update"])
    
    @pytest.mark.e2e
    def test_configuration_update_workflow(self, api_client):
        """Test configuration update and its effects"""
        # Step 1: Get current config
        original_config = api_client.get("/config/trading-rules").json()
        
        # Step 2: Update configuration
        new_config = original_config.copy()
        new_config["min_trade_size"] = 0.002
        new_config["buy_threshold"] = 0.7
        
        update_response = api_client.post("/config/trading-rules", json=new_config)
        assert update_response.status_code == 200
        
        # Step 3: Verify trade validation uses new config
        # Try to trade below old minimum but above new minimum
        with patch('services.data_fetcher.DataFetcher.fetch_current_price') as mock_price:
            mock_price.return_value = {'price': 50000.0}
            
            trade_response = api_client.post("/trades/execute", json={
                "type": "buy",
                "amount": 0.0015  # Below old min (0.001) but above new min (0.002)
            })
            
            # Should fail with new config
            assert trade_response.status_code in [400, 422]
        
        # Step 4: Restore original config
        api_client.post("/config/trading-rules", json=original_config)
    
    @pytest.mark.e2e
    def test_market_data_analysis_workflow(self, api_client):
        """Test complete market analysis workflow"""
        # Step 1: Fetch comprehensive market data
        market_data = api_client.get("/market/data").json()
        
        assert "price" in market_data
        assert "fear_greed" in market_data
        assert "network_stats" in market_data
        
        # Step 2: Get historical data for analysis
        history = api_client.get("/price/history?days=7").json()
        assert len(history) > 0
        
        # Step 3: Get comprehensive signals
        signals = api_client.get("/signals/comprehensive").json()
        
        assert "technical_signals" in signals
        assert "on_chain_signals" in signals
        assert "sentiment_signals" in signals
        assert "combined_signal" in signals
        
        # Step 4: Run backtest
        backtest_response = api_client.get("/backtest/enhanced/run?days=30")
        if backtest_response.status_code == 200:
            backtest_results = backtest_response.json()
            assert "total_return" in backtest_results
            assert "sharpe_ratio" in backtest_results
            assert "max_drawdown" in backtest_results
    
    @pytest.mark.e2e
    def test_error_recovery_workflow(self, api_client):
        """Test system behavior and recovery from errors"""
        # Step 1: Test handling of invalid trades
        invalid_trades = [
            {"type": "buy", "amount": -1},  # Negative amount
            {"type": "buy", "amount": 1000},  # Too large
            {"type": "invalid", "amount": 0.1},  # Invalid type
            {"type": "sell", "amount": 100}  # More than we have
        ]
        
        for trade in invalid_trades:
            response = api_client.post("/trades/execute", json=trade)
            assert response.status_code in [400, 422]
        
        # Step 2: Verify system still functional
        health = api_client.get("/health").json()
        assert health["status"] == "healthy"
        
        # Step 3: Execute valid trade
        with patch('services.data_fetcher.DataFetcher.fetch_current_price') as mock_price:
            mock_price.return_value = {'price': 50000.0}
            
            valid_response = api_client.post("/trades/execute", json={
                "type": "buy",
                "amount": 0.01
            })
            assert valid_response.status_code == 200
    
    @pytest.mark.e2e
    @pytest.mark.slow
    def test_long_running_session(self, api_client):
        """Test system stability over extended period"""
        start_time = time.time()
        errors = []
        requests_made = 0
        
        # Run for 60 seconds
        while time.time() - start_time < 60:
            try:
                # Make various API calls
                api_client.get("/health")
                api_client.get("/price/current")
                api_client.get("/signals/latest")
                api_client.get("/portfolio/metrics")
                
                requests_made += 4
                time.sleep(2)  # Don't overwhelm the system
                
            except Exception as e:
                errors.append(str(e))
        
        # System should remain stable
        assert len(errors) == 0
        assert requests_made > 20  # Made multiple requests
        
        # Final health check
        health = api_client.get("/health").json()
        assert health["status"] == "healthy"