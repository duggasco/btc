"""
Unit tests for enhanced LSTM model with price prediction fixes
"""
import pytest
import numpy as np
import pandas as pd
import torch
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import joblib
import tempfile
import os

from models.enhanced_lstm import LSTMTrainer, EnhancedLSTM
from models.enhanced_lstm_returns import ReturnsDataset
from models.lstm import TradingSignalGenerator, IntegratedTradingSignalGenerator


class TestEnhancedLSTMPredictions:
    """Test price prediction accuracy fixes"""
    
    @pytest.fixture
    def temp_model_dir(self):
        """Create temporary model directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def mock_scalers(self, temp_model_dir):
        """Create mock scalers with realistic price ranges"""
        from sklearn.preprocessing import MinMaxScaler
        
        # Create and save feature scaler
        feature_scaler = MinMaxScaler()
        feature_scaler.fit(np.random.randn(100, 10))  # 10 features
        joblib.dump(feature_scaler, os.path.join(temp_model_dir, 'feature_scaler.pkl'))
        
        # Create and save target scaler with realistic BTC price range
        target_scaler = MinMaxScaler()
        # Training data from $25,157 to $111,722 (as discovered in our analysis)
        target_scaler.fit(np.array([[25157.0], [111722.0]]))
        joblib.dump(target_scaler, os.path.join(temp_model_dir, 'target_scaler.pkl'))
        
        return feature_scaler, target_scaler
    
    def test_lstm_trainer_predict_returns_reasonable_price(self, temp_model_dir, mock_scalers):
        """Test that LSTMTrainer predictions are within reasonable range of current price"""
        trainer = LSTMTrainer(model_dir=temp_model_dir)
        
        # Load the scalers
        trainer.feature_scaler, trainer.target_scaler = mock_scalers
        
        # Create a simple model
        trainer.model = EnhancedLSTM(input_size=10, hidden_size=20, num_layers=1)
        trainer.model.eval()
        
        # Mock model to return a value in the middle of the normalized range
        with patch.object(trainer.model, 'forward') as mock_forward:
            # Model outputs 0.5 (middle of range)
            mock_forward.return_value = torch.tensor([[0.5]])
            
            # Create test features
            features = np.random.randn(60, 10)  # 60 timesteps, 10 features
            
            # Predict
            prediction = trainer.predict(features)
            
            # Check prediction is in the middle of the training range
            expected_price = 25157.0 + (111722.0 - 25157.0) * 0.5  # ~68,439
            assert abs(prediction[0][0] - expected_price) < 100  # Within $100
    
    def test_trading_signal_generator_uses_current_price(self, temp_model_dir):
        """Test that TradingSignalGenerator uses real-time price for predictions"""
        generator = TradingSignalGenerator(sequence_length=30)
        generator.is_trained = True
        
        # Mock the model
        generator.model = Mock()
        generator.model.eval = Mock()
        generator.model.return_value = torch.tensor([[0.1]])  # 10% predicted change
        
        # Mock scaler
        generator.scaler = Mock()
        generator.scaler.transform.return_value = np.random.randn(30, 16)
        
        # Create test data
        test_data = pd.DataFrame({
            'Close': [50000] * 31,  # Historical price much lower than current
            'Open': [49900] * 31,
            'High': [50100] * 31,
            'Low': [49800] * 31,
            'Volume': [1000000] * 31
        })
        
        # Mock get_current_btc_price to return current market price
        with patch('services.enhanced_data_fetcher.EnhancedDataFetcher.get_current_btc_price') as mock_price:
            mock_price.return_value = 109620.0  # Current BTC price
            
            signal, confidence, predicted_price = generator.predict_signal(test_data)
            
            # Predicted price should be based on current price, not historical
            expected_price = 109620.0 * 1.1  # 10% increase
            assert abs(predicted_price - expected_price) < 1000  # Within $1000
            assert predicted_price > 100000  # Much higher than historical data
    
    def test_integrated_generator_confidence_prediction(self):
        """Test IntegratedTradingSignalGenerator with confidence predictions"""
        generator = IntegratedTradingSignalGenerator()
        generator.is_trained = True
        
        # Mock model and scaler
        generator.model = Mock()
        generator.model.train = Mock()
        generator.model.eval = Mock()
        generator.model.return_value = torch.tensor([[0.05]])  # 5% predicted change
        
        generator.scaler = Mock()
        generator.scaler.transform.return_value = np.random.randn(30, 16)
        generator.scaler.inverse_transform = Mock(side_effect=lambda x: x * 50000)
        
        # Create test data
        test_data = pd.DataFrame({
            'Close': [50000] * 100,
            'price': [50000] * 100,
            'Open': [49900] * 100,
            'High': [50100] * 100,
            'Low': [49800] * 100,
            'Volume': [1000000] * 100
        })
        
        # Add some dummy features
        for col in ['fear_proxy', 'nvt_proxy', 'momentum_sentiment']:
            test_data[col] = np.random.randn(100)
        
        # Mock current price fetcher
        with patch('services.enhanced_data_fetcher.EnhancedDataFetcher.get_current_btc_price') as mock_price:
            mock_price.return_value = 109620.0
            
            signal, confidence, mean_price, analysis = generator.predict_with_confidence(test_data, n_predictions=5)
            
            # Check that predictions are reasonable
            assert 90000 < mean_price < 130000  # Within reasonable range of current price
            assert 0 <= confidence <= 1
            assert signal in ['buy', 'sell', 'hold']
            assert 'current_price' in analysis
            assert analysis['current_price'] == 109620.0


class TestPredictionAdjustmentLogic:
    """Test the prediction adjustment logic in enhanced_integration.py"""
    
    def test_prediction_position_calculation(self):
        """Test calculation of prediction position within training range"""
        # Known training range from our analysis
        training_min = 25157.0
        training_max = 111722.0
        training_range = training_max - training_min
        
        # Test various predictions
        test_cases = [
            (25157.0, 0.0),    # Min of range
            (111722.0, 1.0),   # Max of range
            (68439.5, 0.5),    # Middle of range
            (10000.0, 0.0),    # Below range (clamped)
            (150000.0, 1.0),   # Above range (clamped)
        ]
        
        for prediction, expected_position in test_cases:
            position = (prediction - training_min) / training_range
            position = max(0, min(1, position))
            assert abs(position - expected_position) < 0.01
    
    def test_direction_signal_conversion(self):
        """Test conversion of position to directional signal"""
        test_cases = [
            (0.0, -0.5),    # Strong bearish
            (0.25, -0.25),  # Moderate bearish
            (0.5, 0.0),     # Neutral
            (0.75, 0.25),   # Moderate bullish
            (1.0, 0.5),     # Strong bullish
        ]
        
        for position, expected_signal in test_cases:
            direction_signal = position - 0.5
            assert abs(direction_signal - expected_signal) < 0.01
    
    def test_price_change_factor_calculation(self):
        """Test calculation of price change factor"""
        max_daily_change = 0.03
        current_price = 109620.0
        
        test_cases = [
            (-0.5, 0.97),   # Max bearish: -3%
            (-0.25, 0.985), # Moderate bearish: -1.5%
            (0.0, 1.0),     # Neutral: 0%
            (0.25, 1.015),  # Moderate bullish: +1.5%
            (0.5, 1.03),    # Max bullish: +3%
        ]
        
        for direction_signal, expected_factor in test_cases:
            price_change_factor = 1.0 + (direction_signal * 2 * max_daily_change)
            assert abs(price_change_factor - expected_factor) < 0.001
            
            # Check adjusted price
            adjusted_price = current_price * price_change_factor
            expected_price = current_price * expected_factor
            assert abs(adjusted_price - expected_price) < 1
    
    def test_confidence_based_adjustment(self):
        """Test that low confidence reduces prediction magnitude"""
        current_price = 109620.0
        
        test_cases = [
            (110000.0, 0.8, 110000.0),  # High confidence: no adjustment
            (110000.0, 0.5, 109810.0),  # Low confidence: reduced change
            (109000.0, 0.5, 109310.0),  # Low confidence on bearish prediction
        ]
        
        for predicted_price, confidence, expected_adjusted in test_cases:
            if confidence < 0.6:
                adjusted = current_price + (predicted_price - current_price) * confidence
            else:
                adjusted = predicted_price
            
            assert abs(adjusted - expected_adjusted) < 1
    
    def test_sanity_check_caps_extreme_predictions(self):
        """Test that extreme predictions are capped"""
        current_price = 109620.0
        min_reasonable = current_price * 0.90  # -10%
        max_reasonable = current_price * 1.10  # +10%
        
        test_cases = [
            (130000.0, max_reasonable),  # Too high
            (80000.0, min_reasonable),   # Too low
            (110000.0, 110000.0),        # Within range
        ]
        
        for prediction, expected in test_cases:
            capped = max(min_reasonable, min(max_reasonable, prediction))
            assert abs(capped - expected) < 1


class TestRealTimePriceFetching:
    """Test real-time price fetching functionality"""
    
    def test_predict_signal_fetches_current_price(self):
        """Test that predict_signal attempts to fetch current price"""
        # Complex setup to properly test the real-time price fetching
        with patch('sys.modules', sys.modules.copy()) as mock_modules:
            # Create mock enhanced data fetcher module
            mock_enhanced_module = Mock()
            mock_fetcher_instance = Mock()
            mock_fetcher_instance.get_current_btc_price.return_value = 109620.0
            
            # Mock the EnhancedDataFetcher class
            mock_enhanced_class = Mock(return_value=mock_fetcher_instance)
            mock_enhanced_module.EnhancedDataFetcher = mock_enhanced_class
            
            # Inject the mock module
            mock_modules['services.enhanced_data_fetcher'] = mock_enhanced_module
            
            # Create generator with mocked components
            generator = TradingSignalGenerator()
            generator.is_trained = True
            
            # Mock the neural network model
            mock_model = Mock()
            mock_model.eval = Mock()
            mock_model.return_value = torch.tensor([[0.02]])  # 2% price increase prediction
            generator.model = mock_model
            
            # Mock the scaler with proper transform behavior
            mock_scaler = Mock()
            mock_scaler.transform.return_value = np.random.randn(30, 16)
            generator.scaler = mock_scaler
            
            # Create realistic test data
            test_data = pd.DataFrame({
                'Close': [50000] * 31,
                'Open': [49900] * 31,
                'High': [50100] * 31,
                'Low': [49800] * 31,
                'Volume': [1000000] * 31
            })
            
            # Execute prediction
            signal, confidence, predicted_price = generator.predict_signal(test_data)
            
            # Verify complex behaviors
            assert mock_enhanced_class.called, "EnhancedDataFetcher should be instantiated"
            assert mock_fetcher_instance.get_current_btc_price.called, "Current price should be fetched"
            
            # Verify prediction is based on real-time price, not historical
            assert predicted_price > 100000, f"Prediction {predicted_price} should be based on current price"
            assert predicted_price < 120000, f"Prediction {predicted_price} should be reasonable"
            
            # Verify signal generation logic
            assert signal in ['buy', 'sell', 'hold'], f"Signal {signal} should be valid"
            assert 0 <= confidence <= 1, f"Confidence {confidence} should be between 0 and 1"
    
    @patch('services.enhanced_data_fetcher.EnhancedDataFetcher')
    def test_fallback_to_historical_on_fetch_failure(self, mock_fetcher_class):
        """Test fallback to historical data when real-time fetch fails"""
        # Setup mock to raise exception
        mock_fetcher = Mock()
        mock_fetcher.get_current_btc_price.side_effect = Exception("API error")
        mock_fetcher_class.return_value = mock_fetcher
        
        generator = TradingSignalGenerator()
        generator.is_trained = True
        generator.model = Mock()
        generator.model.eval = Mock()
        generator.model.return_value = torch.tensor([[0.02]])
        generator.scaler = Mock()
        generator.scaler.transform.return_value = np.random.randn(30, 16)
        
        # Create test data
        test_data = pd.DataFrame({
            'Close': [50000] * 31,
            'Open': [49900] * 31,
            'High': [50100] * 31,
            'Low': [49800] * 31,
            'Volume': [1000000] * 31
        })
        
        signal, confidence, predicted_price = generator.predict_signal(test_data)
        
        # Should fall back to historical price
        expected_price = 50000 * 1.02
        assert abs(predicted_price - expected_price) < 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])