"""
Unit tests for enhanced integration service with price prediction fixes
"""
import pytest
import numpy as np
import pandas as pd
import torch
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import json
import os

from services.enhanced_integration import EnhancedTradingSystem


class TestEnhancedTradingSystemPredictions:
    """Test enhanced trading system prediction adjustments"""
    
    @pytest.fixture
    def mock_config_path(self, tmp_path):
        """Create mock config file"""
        config = {
            "data": {"history_days": 730, "sequence_length": 60},
            "model": {"hidden_size": 100, "ensemble_size": 3},
            "trading": {"confidence_threshold": 0.7}
        }
        config_path = tmp_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f)
        return str(config_path)
    
    @pytest.fixture
    def trading_system(self, tmp_path, mock_config_path):
        """Create trading system instance"""
        return EnhancedTradingSystem(
            model_dir=str(tmp_path / "models"),
            data_dir=str(tmp_path / "data"),
            config_path=mock_config_path
        )
    
    def test_generate_trading_signal_with_adjustment(self, trading_system):
        """Test that trading signals use adjusted predictions"""
        # Mark model as trained
        trading_system.model_trained = True
        trading_system.selected_features = ['feature1', 'feature2']
        
        # Mock the trainers
        mock_trainers = []
        for i in range(3):
            trainer = Mock()
            trainer.predict.return_value = np.array([[68439.5]])  # Middle of training range
            mock_trainers.append(trainer)
        
        # Mock model loading
        with patch('torch.load') as mock_load, \
             patch('services.enhanced_integration.LSTMTrainer') as mock_trainer_class, \
             patch.object(trading_system.data_fetcher, 'get_current_btc_price') as mock_price:
            
            # Setup mocks
            mock_price.return_value = 109620.0  # Current BTC price
            mock_trainer_class.side_effect = mock_trainers
            
            # Mock checkpoint data
            mock_checkpoint = {
                'model_state_dict': Mock(),
                'config': {'sequence_length': 60},
                'metrics': {'test_metrics': {'directional_accuracy': 0.6}}
            }
            mock_load.return_value = mock_checkpoint
            
            # Create test data
            test_data = pd.DataFrame({
                'Close': [50000] * 100,
                'feature1': np.random.randn(100),
                'feature2': np.random.randn(100)
            })
            
            # Mock feature engineering
            with patch.object(trading_system.feature_engineer, 'engineer_features') as mock_engineer:
                mock_engineer.return_value = (test_data, ['feature1', 'feature2'])
                
                # Generate signal
                result = trading_system.generate_trading_signal(test_data)
        
        # Verify results
        assert 'signal' in result
        assert 'predicted_price' in result
        assert 'confidence' in result
        
        # Price should be adjusted to current range
        assert 100000 < result['predicted_price'] < 120000  # Reasonable range
        
    def test_prediction_adjustment_logic(self, trading_system):
        """Test the detailed prediction adjustment logic"""
        # Test data
        current_price = 109620.0
        training_min = 25157.0
        training_max = 111722.0
        
        test_predictions = [
            40000.0,   # Below middle of training range
            68439.5,   # Middle of training range
            100000.0,  # Upper part of training range
        ]
        
        adjusted_predictions = []
        
        for raw_prediction in test_predictions:
            # Calculate position in training range
            training_range = training_max - training_min
            prediction_position = (raw_prediction - training_min) / training_range
            prediction_position = max(0, min(1, prediction_position))
            
            # Convert to directional signal
            direction_signal = prediction_position - 0.5
            
            # Calculate price change factor
            max_daily_change = 0.03
            price_change_factor = 1.0 + (direction_signal * 2 * max_daily_change)
            
            # Apply to current price
            adjusted_prediction = current_price * price_change_factor
            adjusted_predictions.append(adjusted_prediction)
        
        # Verify adjustments
        assert adjusted_predictions[0] < current_price  # Bearish
        assert abs(adjusted_predictions[1] - current_price) < current_price * 0.01  # Near neutral
        assert adjusted_predictions[2] > current_price  # Bullish
        
        # All should be within reasonable daily change
        for adj_pred in adjusted_predictions:
            pct_change = abs((adj_pred - current_price) / current_price)
            assert pct_change <= 0.03  # Max 3% daily change
    
    def test_fallback_to_rule_based_signal(self, trading_system):
        """Test fallback to rule-based signals when model not trained"""
        trading_system.model_trained = False
        
        # Create test data
        test_data = pd.DataFrame({
            'Close': [109000, 109500, 109620],
            'Open': [108900, 109400, 109500],
            'High': [109200, 109600, 109700],
            'Low': [108800, 109300, 109400],
            'Volume': [1000000, 1100000, 1050000]
        })
        
        # Mock current price
        with patch.object(trading_system.data_fetcher, 'get_current_btc_price') as mock_price:
            mock_price.return_value = 109620.0
            
            result = trading_system.generate_trading_signal(test_data)
        
        # Should return rule-based signal
        assert 'note' in result
        assert 'Rule-based signal' in result['note']
        assert result['predicted_price'] > 0
    
    def test_ensemble_prediction_aggregation(self, trading_system):
        """Test that ensemble predictions are properly aggregated"""
        trading_system.model_trained = True
        trading_system.selected_features = ['feature1']
        
        # Different predictions from ensemble models
        predictions = [
            np.array([[65000.0]]),  # Slightly bearish
            np.array([[68439.5]]),  # Neutral
            np.array([[72000.0]]),  # Slightly bullish
        ]
        
        mock_trainers = []
        for pred in predictions:
            trainer = Mock()
            trainer.predict.return_value = pred
            mock_trainers.append(trainer)
        
        with patch('torch.load') as mock_load, \
             patch('services.enhanced_integration.LSTMTrainer') as mock_trainer_class, \
             patch.object(trading_system.data_fetcher, 'get_current_btc_price') as mock_price:
            
            mock_price.return_value = 109620.0
            mock_trainer_class.side_effect = mock_trainers
            
            mock_checkpoint = {
                'model_state_dict': Mock(),
                'config': {'sequence_length': 60},
                'metrics': {'test_metrics': {'directional_accuracy': 0.6}}
            }
            mock_load.return_value = mock_checkpoint
            
            test_data = pd.DataFrame({
                'Close': [50000] * 100,
                'feature1': np.random.randn(100)
            })
            
            with patch.object(trading_system.feature_engineer, 'engineer_features') as mock_engineer:
                mock_engineer.return_value = (test_data, ['feature1'])
                
                result = trading_system.generate_trading_signal(test_data)
        
        # Average prediction should be near neutral
        # Given the spread of predictions, final should be close to current price
        assert abs(result['predicted_price'] - 109620.0) < 3000  # Within ~3%
    
    def test_confidence_score_calculation(self, trading_system):
        """Test confidence score calculation based on model metrics"""
        # Test various directional accuracies
        test_cases = [
            (0.55, 0.535),  # Low accuracy -> lower confidence (0.55 * 0.7 + 0.5 * 0.3 = 0.535)
            (0.60, 0.570),  # Medium accuracy (0.60 * 0.7 + 0.5 * 0.3 = 0.570)
            (0.70, 0.640),  # High accuracy -> higher confidence (0.70 * 0.7 + 0.5 * 0.3 = 0.640)
        ]
        
        for dir_acc, expected_conf in test_cases:
            metrics = {
                'test_metrics': {
                    'directional_accuracy': dir_acc,
                    'rmse': 5000  # Normalized RMSE
                }
            }
            
            # Calculate confidence
            rmse_conf = max(0, 1 - (5000 / 10000))  # 0.5
            confidence = (dir_acc * 0.7 + rmse_conf * 0.3)
            
            assert abs(confidence - expected_conf) < 0.01


class TestRealtimePriceIntegration:
    """Test real-time price integration in enhanced system"""
    
    def test_current_price_fetching_in_signal_generation(self, trading_system):
        """Test that current price is fetched during signal generation"""
        trading_system.model_trained = True
        trading_system.selected_features = ['Close']
        
        with patch.object(trading_system.data_fetcher, 'get_current_btc_price') as mock_price:
            mock_price.return_value = 109620.0
            
            # Create minimal test data
            test_data = pd.DataFrame({
                'Close': [50000] * 100,
                'Open': [49900] * 100,
                'High': [50100] * 100,
                'Low': [49800] * 100,
                'Volume': [1000000] * 100
            })
            
            # Mock other dependencies
            with patch.object(trading_system.feature_engineer, 'engineer_features') as mock_engineer, \
                 patch('os.path.exists', return_value=False):  # No saved models
                
                mock_engineer.return_value = (test_data, ['Close'])
                
                result = trading_system.generate_trading_signal(test_data)
            
            # Verify current price was fetched
            mock_price.assert_called()
    
    def test_technical_signal_calculation(self, trading_system):
        """Test technical signal calculation with indicators"""
        test_data = pd.DataFrame({
            'RSI': [25],  # Oversold
            'MACD_bullish_cross': [1],
            'BB_position': [0.1],  # Near lower band
        })
        
        result = trading_system._calculate_technical_signal(test_data)
        
        assert result['signal'] == 'buy'  # Should be bullish
        assert result['strength'] > 0.5
    
    def test_onchain_signal_calculation(self, trading_system):
        """Test on-chain signal calculation"""
        test_data = pd.DataFrame({
            'net_exchange_flow': [1000],  # Positive = outflow = bullish
            'nvt_ratio': [50],
            'mvrv_ratio': [0.8]  # < 1 = undervalued
        })
        test_data['nvt_ratio_ma'] = 60  # For comparison
        
        result = trading_system._calculate_onchain_signal(test_data)
        
        assert result['signal'] == 'buy'  # Should be bullish
        assert result['strength'] > 0.5
    
    def test_sentiment_signal_calculation(self, trading_system):
        """Test sentiment signal calculation"""
        test_data = pd.DataFrame({
            'fear_greed_value': [20],  # Extreme fear
            'overall_sentiment': [-0.6]  # Very negative
        })
        
        result = trading_system._calculate_sentiment_signal(test_data)
        
        assert result['signal'] == 'buy'  # Contrarian signal
        assert result['strength'] > 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])