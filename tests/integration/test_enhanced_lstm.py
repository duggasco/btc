"""Integration tests for the enhanced LSTM trading system"""

import pytest
import os
import tempfile
import json
from datetime import datetime
import pandas as pd
import numpy as np

# Import enhanced modules
from services.enhanced_data_fetcher import EnhancedDataFetcher
from services.feature_engineering import FeatureEngineer
from models.enhanced_lstm import LSTMTrainer, EnhancedLSTM
from services.enhanced_integration import EnhancedTradingSystem


class TestEnhancedLSTMIntegration:
    """Test suite for enhanced LSTM system integration"""
    
    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = os.path.join(temp_dir, 'models')
            data_dir = os.path.join(temp_dir, 'data')
            config_dir = os.path.join(temp_dir, 'config')
            
            os.makedirs(model_dir)
            os.makedirs(data_dir)
            os.makedirs(config_dir)
            
            yield {
                'model_dir': model_dir,
                'data_dir': data_dir,
                'config_dir': config_dir,
                'temp_dir': temp_dir
            }
    
    @pytest.fixture
    def sample_btc_data(self):
        """Create sample BTC data for testing"""
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        
        # Generate realistic OHLCV data
        np.random.seed(42)
        close_prices = 50000 + np.cumsum(np.random.randn(100) * 500)
        
        data = pd.DataFrame({
            'Open': close_prices + np.random.uniform(-100, 100, 100),
            'High': close_prices + np.random.uniform(0, 200, 100),
            'Low': close_prices - np.random.uniform(0, 200, 100),
            'Close': close_prices,
            'Volume': np.random.uniform(1e9, 2e9, 100)
        }, index=dates)
        
        return data
    
    def test_enhanced_data_fetcher_init(self, temp_dirs):
        """Test EnhancedDataFetcher initialization"""
        fetcher = EnhancedDataFetcher(cache_dir=temp_dirs['data_dir'])
        
        assert fetcher is not None
        assert fetcher.cache_dir == temp_dirs['data_dir']
        assert hasattr(fetcher, 'free_apis')
        assert 'binance_klines' in fetcher.free_apis
    
    def test_feature_engineer_init(self):
        """Test FeatureEngineer initialization"""
        engineer = FeatureEngineer(min_periods_ratio=0.8)
        
        assert engineer is not None
        assert engineer.min_periods_ratio == 0.8
        assert hasattr(engineer, 'engineer_features')
    
    def test_feature_engineering(self, sample_btc_data):
        """Test feature engineering on sample data"""
        engineer = FeatureEngineer()
        
        # Engineer features
        enhanced_data, features = engineer.engineer_features(sample_btc_data, adaptive=True)
        
        assert enhanced_data is not None
        assert len(features) > 0
        assert len(enhanced_data) > 0
        
        # Check for basic technical indicators
        expected_indicators = ['RSI', 'MACD', 'SMA_20']
        for indicator in expected_indicators:
            assert any(indicator in f for f in features), f"{indicator} not found in features"
    
    def test_lstm_trainer_init(self, temp_dirs):
        """Test LSTMTrainer initialization"""
        trainer = LSTMTrainer(model_dir=temp_dirs['model_dir'])
        
        assert trainer is not None
        assert trainer.model_dir == temp_dirs['model_dir']
        assert trainer.device is not None
    
    def test_enhanced_trading_system_init(self, temp_dirs):
        """Test EnhancedTradingSystem initialization"""
        # Create config file
        config_path = os.path.join(temp_dirs['config_dir'], 'trading_config.json')
        config = {
            'data': {'history_days': 100},
            'model': {'hidden_size': 50, 'num_layers': 1}
        }
        with open(config_path, 'w') as f:
            json.dump(config, f)
        
        system = EnhancedTradingSystem(
            model_dir=temp_dirs['model_dir'],
            data_dir=temp_dirs['data_dir'],
            config_path=config_path
        )
        
        assert system is not None
        assert system.model_trained == False
        assert system.config['data']['history_days'] == 100
    
    def test_signal_generation_untrained(self, temp_dirs, sample_btc_data):
        """Test signal generation with untrained model (should use rule-based)"""
        config_path = os.path.join(temp_dirs['config_dir'], 'trading_config.json')
        config = {'data': {'history_days': 100}}
        with open(config_path, 'w') as f:
            json.dump(config, f)
        
        system = EnhancedTradingSystem(
            model_dir=temp_dirs['model_dir'],
            data_dir=temp_dirs['data_dir'],
            config_path=config_path
        )
        
        # Generate signal without training
        signal = system.generate_trading_signal(sample_btc_data)
        
        assert signal is not None
        assert 'signal' in signal
        assert signal['signal'] in ['buy', 'sell', 'hold']
        assert 'confidence' in signal
        assert 'note' in signal  # Should indicate rule-based
        assert 'Rule-based signal' in signal.get('note', '')
    
    @pytest.mark.slow
    def test_data_preparation(self, temp_dirs, sample_btc_data):
        """Test data preparation for training"""
        config_path = os.path.join(temp_dirs['config_dir'], 'trading_config.json')
        config = {'data': {'history_days': 100}}
        with open(config_path, 'w') as f:
            json.dump(config, f)
        
        system = EnhancedTradingSystem(
            model_dir=temp_dirs['model_dir'],
            data_dir=temp_dirs['data_dir'],
            config_path=config_path
        )
        
        # Mock the data fetcher to return sample data
        system.data_fetcher.fetch_comprehensive_btc_data = lambda days: sample_btc_data
        
        # Prepare data
        success = system.fetch_and_prepare_data()
        
        assert success == True
        assert system.engineered_data is not None
        assert system.selected_features is not None
        assert len(system.selected_features) > 0
    
    def test_feature_selection(self):
        """Test feature selection functionality"""
        engineer = FeatureEngineer()
        
        # Create sample data with features
        df = pd.DataFrame({
            'Close': np.random.randn(100),
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100)
        })
        
        features = ['feature1', 'feature2', 'feature3']
        selected = engineer.select_features(df, features, target_col='Close', max_features=2)
        
        assert len(selected) <= 2
        assert all(f in features for f in selected)
    
    def test_enhanced_lstm_model(self):
        """Test EnhancedLSTM model creation"""
        model = EnhancedLSTM(
            input_size=10,
            hidden_size=20,
            num_layers=2,
            dropout=0.2,
            use_attention=True
        )
        
        assert model is not None
        assert model.input_size == 10
        assert model.hidden_size == 20
        assert model.num_layers == 2
        assert hasattr(model, 'attention')
        
        # Test forward pass with dummy data
        import torch
        dummy_input = torch.randn(1, 10, 10)  # batch_size=1, seq_len=10, features=10
        output = model(dummy_input)
        
        assert output is not None
        assert output.shape == (1, 1)  # batch_size=1, output_size=1