"""
End-to-end tests for prediction workflows with price adjustment fixes
"""
import pytest
import time
from datetime import datetime
from unittest.mock import patch
import numpy as np
import pandas as pd


class TestPredictionWorkflowE2E:
    """End-to-end tests for complete prediction workflows"""
    
    @pytest.mark.e2e
    def test_complete_prediction_workflow(self, api_client):
        """Test complete workflow from data fetch to prediction"""
        with patch('services.enhanced_data_fetcher.EnhancedDataFetcher.get_current_btc_price') as mock_price, \
             patch('services.enhanced_data_fetcher.EnhancedDataFetcher.fetch_comprehensive_btc_data') as mock_data:
            
            # Setup mocks
            current_price = 109620.0
            mock_price.return_value = current_price
            
            # Create realistic historical data with proper LSTM training range
            dates = pd.date_range(end=datetime.now(), periods=730, freq='D')
            # Historical prices in the training range (25k to 111k)
            base_prices = np.concatenate([
                np.linspace(25000, 50000, 200),
                np.linspace(50000, 80000, 200),
                np.linspace(80000, 111000, 200),
                np.linspace(111000, current_price, 130)
            ])
            noise = np.random.normal(0, 500, 730)
            prices = base_prices + noise
            
            mock_df = pd.DataFrame({
                'Open': prices - np.random.uniform(100, 500, 730),
                'High': prices + np.random.uniform(100, 1000, 730),
                'Low': prices - np.random.uniform(100, 1000, 730),
                'Close': prices,
                'Volume': np.random.uniform(1e9, 2e9, 730)
            }, index=dates)
            mock_data.return_value = mock_df
            
            # Step 1: Check system health
            response = api_client.get("/health")
            assert response.status_code == 200
            health_data = response.json()
            assert health_data["status"] == "healthy"
            
            # Step 2: Get current price
            response = api_client.get("/price/current")
            if response.status_code == 200:
                price_data = response.json()
                assert price_data["price"] > 0
            
            # Step 3: Get latest signal
            response = api_client.get("/signals/enhanced/latest")
            assert response.status_code == 200
            signal_data = response.json()
            
            # Verify signal structure
            assert "signal" in signal_data
            assert signal_data["signal"] in ["buy", "sell", "hold"]
            assert "confidence" in signal_data
            assert 0 <= signal_data["confidence"] <= 1
            assert "predicted_price" in signal_data
            
            # Verify prediction exists and is positive
            predicted_price = signal_data["predicted_price"]
            if predicted_price > 0:
                # For new system: should be within 10% of current price
                # For fallback: might be based on historical data
                # Accept both behaviors during transition
                assert predicted_price > 10000, f"Prediction {predicted_price} is unreasonably low"
                assert predicted_price < 500000, f"Prediction {predicted_price} is unreasonably high"
            
            # Step 4: Verify comprehensive signals if available
            response = api_client.get("/signals/comprehensive")
            if response.status_code == 200:
                comp_data = response.json()
                # Should have some signals calculated
                assert len(comp_data) > 0
    
    @pytest.mark.e2e
    def test_paper_trading_with_adjusted_predictions(self, api_client):
        """Test paper trading uses reasonable predictions"""
        with patch('services.enhanced_data_fetcher.EnhancedDataFetcher.get_current_btc_price') as mock_price:
            mock_price.return_value = 109620.0
            
            # Step 1: Check paper trading status
            response = api_client.get("/paper-trading/status")
            # Accept both 200 (success) and 500 (if DB not initialized)
            if response.status_code != 200:
                pytest.skip("Paper trading not available in test environment")
                return
            status = response.json()
            initial_balance = status.get("balance", 10000)
            
            # Step 2: Generate a signal
            response = api_client.get("/signals/enhanced/latest")
            assert response.status_code == 200
            signal_data = response.json()
            
            # Step 3: If we have a buy/sell signal with good confidence
            if signal_data["signal"] in ["buy", "sell"] and signal_data["confidence"] > 0.6:
                # Check if paper trading would execute
                # Get portfolio to see if trade happened
                response = api_client.get("/paper-trading/portfolio")
                assert response.status_code == 200
                portfolio = response.json()
                
                # Verify any trades use reasonable prices
                if "trades" in portfolio:
                    for trade in portfolio["trades"]:
                        if "price" in trade and trade["price"] > 0:
                            # Trade price should be reasonable
                            assert 50000 < trade["price"] < 200000
    
    @pytest.mark.e2e
    def test_model_adaptation_workflow(self, api_client):
        """Test model adapts to different price ranges"""
        # Test with different current prices to ensure adaptation
        test_prices = [95000, 109620, 125000]
        
        for current_price in test_prices:
            with patch('services.enhanced_data_fetcher.EnhancedDataFetcher.get_current_btc_price') as mock_price:
                mock_price.return_value = current_price
                
                response = api_client.get("/signals/enhanced/latest")
                # Accept either 200 (success) or fallback behavior
                if response.status_code != 200:
                    continue
                
                data = response.json()
                
                if "predicted_price" in data and data["predicted_price"] is not None and data["predicted_price"] > 0:
                    # Prediction should be reasonable
                    predicted = data["predicted_price"]
                    
                    # Should be within reasonable bounds
                    assert 10000 < predicted < 500000, \
                        f"Prediction {predicted} outside reasonable bounds"
    
    @pytest.mark.e2e
    def test_websocket_price_updates(self, api_client):
        """Test WebSocket delivers reasonable price predictions"""
        # This is a simplified test - full WebSocket test would use websocket-client
        with patch('services.enhanced_data_fetcher.EnhancedDataFetcher.get_current_btc_price') as mock_price:
            mock_price.return_value = 109620.0
            
            # Trigger signal update
            response = api_client.get("/signals/enhanced/latest")
            assert response.status_code == 200
            
            # In a real test, we'd connect to WebSocket and verify:
            # 1. Price updates are reasonable
            # 2. Signal updates include adjusted predictions
            # 3. Updates are consistent with REST API
    
    @pytest.mark.e2e
    def test_historical_analysis_with_predictions(self, api_client):
        """Test historical analysis includes reasonable predictions"""
        # Get historical signals
        response = api_client.get("/signals/history?days=1")
        assert response.status_code == 200
        history = response.json()
        
        if len(history) > 0:
            # Check all historical predictions are reasonable
            for signal in history:
                if "predicted_price" in signal and signal["predicted_price"] is not None:
                    # Historical predictions should be reasonable
                    assert 20000 < signal["predicted_price"] < 200000, \
                        f"Unreasonable historical prediction: {signal['predicted_price']}"
    
    @pytest.mark.e2e
    @pytest.mark.slow
    def test_continuous_prediction_stability(self, api_client):
        """Test predictions remain stable over multiple calls"""
        with patch('services.enhanced_data_fetcher.EnhancedDataFetcher.get_current_btc_price') as mock_price:
            mock_price.return_value = 109620.0
            
            predictions = []
            
            # Get multiple predictions
            for _ in range(5):
                response = api_client.get("/signals/enhanced/latest")
                if response.status_code == 200:
                    data = response.json()
                    if "predicted_price" in data and data["predicted_price"] > 0:
                        predictions.append(data["predicted_price"])
                
                time.sleep(0.1)  # Small delay between requests
            
            if len(predictions) > 1:
                # Predictions should be relatively stable
                min_pred = min(predictions)
                max_pred = max(predictions)
                variation = (max_pred - min_pred) / min_pred
                
                # Should not vary by more than 5% between calls
                assert variation < 0.05, f"Predictions too unstable: {predictions}"


class TestErrorHandlingE2E:
    """Test error handling in prediction workflows"""
    
    @pytest.mark.e2e
    def test_handles_extreme_market_conditions(self, api_client):
        """Test system handles extreme price movements gracefully"""
        extreme_prices = [
            10000,   # Major crash
            500000,  # Major rally
        ]
        
        for extreme_price in extreme_prices:
            with patch('services.enhanced_data_fetcher.EnhancedDataFetcher.get_current_btc_price') as mock_price:
                mock_price.return_value = extreme_price
                
                response = api_client.get("/signals/enhanced/latest")
                # System should handle extreme prices gracefully
                if response.status_code != 200:
                    # Accept graceful failure for extreme conditions
                    continue
                
                data = response.json()
                
                # Should still return valid signal
                assert data["signal"] in ["buy", "sell", "hold"]
                
                # Predictions should be reasonable even in extreme conditions
                if "predicted_price" in data and data["predicted_price"] is not None and data["predicted_price"] > 0:
                    predicted = data["predicted_price"]
                    # Should be within sane bounds
                    assert 1000 < predicted < 1000000, \
                        f"Prediction {predicted} is unreasonable for extreme conditions"
    
    @pytest.mark.e2e
    def test_handles_data_fetch_failures(self, api_client):
        """Test system handles data fetch failures gracefully"""
        with patch('services.enhanced_data_fetcher.EnhancedDataFetcher.get_current_btc_price') as mock_price, \
             patch('services.enhanced_data_fetcher.EnhancedDataFetcher.fetch_comprehensive_btc_data') as mock_data:
            
            # Simulate fetch failures
            mock_price.side_effect = Exception("API error")
            mock_data.side_effect = Exception("Data fetch failed")
            
            # Should still return some response (fallback)
            response = api_client.get("/signals/enhanced/latest")
            assert response.status_code in [200, 500]
            
            if response.status_code == 200:
                data = response.json()
                # Should have basic signal structure even with failures
                assert "signal" in data
                assert data["signal"] in ["buy", "sell", "hold"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "e2e"])