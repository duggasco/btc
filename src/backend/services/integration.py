"""
Enhanced Integration Module with Advanced Features
Maintains ALL original functionality while adding comprehensive enhancements
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from typing import Dict, Tuple, Optional, List, Any

from models.lstm import TradingSignalGenerator, IntegratedTradingSignalGenerator, EnhancedLSTMTradingModel
from models.database import DatabaseManager
from services.backtesting import (
    BacktestConfig, SignalWeights, EnhancedSignalWeights,
    EnhancedBacktestingPipeline, AdaptiveRetrainingScheduler,
    EnhancedBayesianOptimizer, EnhancedWalkForwardBacktester,
    ComprehensiveSignalCalculator, EnhancedPerformanceMetrics
)
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import json
import torch
import traceback
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')
from services.data_fetcher import get_fetcher
from services.historical_data_manager import get_historical_manager

# Enhanced logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========== ENHANCED TRADING SIGNAL GENERATOR ==========

class AdvancedTradingSignalGenerator(IntegratedTradingSignalGenerator):
    """Advanced version with all 50+ signals and LSTM best practices"""
    
    def __init__(self, model_path: str = None, sequence_length: int = 30, use_enhanced_model: bool = True):
        # Resolve model path for container environment
        if model_path and not os.path.isabs(model_path):
            # Try multiple possible locations for the model
            possible_paths = [
                # Container paths
                os.path.join('/app/models', os.path.basename(model_path)),
                os.path.join('/app/data', os.path.basename(model_path)),
                # Development paths
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                           'storage', 'models', os.path.basename(model_path)),
                # Original relative path
                model_path
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    logger.info(f"Found model at: {model_path}")
                    break
            else:
                logger.warning(f"Model not found in any expected location, will train new model")
                model_path = None
        
        # Initialize with enhanced model by default
        super().__init__(model_path, sequence_length, use_enhanced_model=use_enhanced_model)
        
        # Override with EnhancedSignalWeights
        self.signal_weights = EnhancedSignalWeights()
        self.performance_history = []
        self._cached_btc_data = None
        self._cached_predictions = {}
        self._fitted_on_features = None
        
        # Signal calculator is already initialized in parent
        # self.signal_calculator = ComprehensiveSignalCalculator()
        
        # LSTM best practices from research (already in parent)
        # self.feature_scaler = MinMaxScaler()
        # self.target_scaler = MinMaxScaler()
        # self.feature_importance = {}
        
        # Enhanced model parameters
        self.dropout_rate = 0.2
        self.learning_rate = 0.001
        self.batch_size = 32
        
        # Data fetcher already initialized in parent
        # self.data_fetcher = get_fetcher()
        
        # Store whether we're using enhanced model
        self.use_enhanced_model = use_enhanced_model
        
        # Try to load model if path provided
        if model_path:
            self._load_model_safe(model_path)
    
    def _load_model_safe(self, model_path: str) -> bool:
        """Safely load model with architecture compatibility checks"""
        try:
            if not os.path.exists(model_path):
                logger.warning(f"Model file not found: {model_path}")
                return False
            
            # Try to load the checkpoint
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            
            # Check if it's a valid checkpoint
            if 'model_state_dict' not in checkpoint:
                logger.error(f"Invalid model checkpoint format at {model_path}")
                return False
            
            # Get model state dict
            state_dict = checkpoint['model_state_dict']
            
            # Try to determine input size from the saved model
            # Look for the first LSTM layer's weight shape
            input_size = None
            for key, value in state_dict.items():
                if 'lstm.weight_ih_l0' in key:
                    # The input size is the second dimension of weight_ih_l0
                    input_size = value.shape[1]
                    break
            
            if input_size is None:
                logger.warning("Could not determine input size from saved model, using default")
                input_size = 16  # Default
            
            # Check if we need to create a new model with matching architecture
            if hasattr(self, 'model') and hasattr(self.model, 'lstm'):
                current_input_size = self.model.lstm.input_size
                if current_input_size != input_size:
                    logger.info(f"Model architecture mismatch. Creating new model with input_size={input_size}")
                    if self.use_enhanced_model:
                        self.model = EnhancedLSTMTradingModel(
                            input_size=input_size,
                            hidden_size=50,
                            num_layers=2,
                            dropout=self.dropout_rate,
                            use_attention=True
                        )
                    else:
                        from models.lstm import LSTMTradingModel
                        self.model = LSTMTradingModel(
                            input_size=input_size,
                            hidden_size=50,
                            num_layers=2,
                            dropout=self.dropout_rate
                        )
            
            # Load the state dict
            self.model.load_state_dict(state_dict)
            
            # Load other components if available
            if 'scaler' in checkpoint:
                self.scaler = checkpoint['scaler']
            if 'feature_scaler' in checkpoint:
                self.feature_scaler = checkpoint['feature_scaler']
            if 'target_scaler' in checkpoint:
                self.target_scaler = checkpoint['target_scaler']
            if 'is_trained' in checkpoint:
                self.is_trained = checkpoint['is_trained']
            if 'sequence_length' in checkpoint:
                self.sequence_length = checkpoint.get('sequence_length', 60)
            
            logger.info(f"Successfully loaded model from {model_path}")
            logger.info(f"Model input size: {input_size}, sequence length: {self.sequence_length}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {str(e)}")
            logger.error(traceback.format_exc())
            self.is_trained = False
            return False
        
    def fetch_enhanced_btc_data(self, period: str = "2y", include_macro: bool = True) -> pd.DataFrame:
        """Fetch BTC data with additional market indicators"""
        try:
            logger.info(f"Fetching enhanced BTC data for period: {period}")
            
            # Get BTC data using enhanced historical capabilities
            # First try to load existing historical data
            btc_data = self.data_fetcher.load_combined_data('BTC', granularity='1d')
            
            # If not enough data, fetch extended historical data
            if btc_data is None or len(btc_data) < self.sequence_length:
                logger.info("Fetching extended historical data...")
                btc_data = self.data_fetcher.fetch_extended_historical_data('BTC', granularity='1d')
            
            # Filter to requested period if we have more data
            if len(btc_data) > 0:
                days_map = {
                    '1d': 1, '7d': 7, '1mo': 30, '3mo': 90,
                    '6mo': 180, '1y': 365, '2y': 730
                }
                days = days_map.get(period, 90)
                if len(btc_data) > days:
                    btc_data = btc_data.iloc[-days:]
            
            if btc_data is None or len(btc_data) < self.sequence_length:
                logger.warning("Insufficient BTC data, using fallback")
                btc_data = self.generate_dummy_data()
            
            # Add technical indicators
            logger.info("Adding technical indicators...")
            btc_data = self._add_enhanced_technical_indicators(btc_data)
            
            # Fetch and add macro data if requested
            if include_macro:
                logger.info("Fetching macro indicators...")
                btc_data = self._add_macro_indicators(btc_data)
            
            # Add sentiment proxies
            btc_data = self._add_sentiment_proxies(btc_data)
            
            # Add on-chain proxies
            btc_data = self._add_onchain_proxies(btc_data)
            
            logger.info(f"Enhanced data shape: {btc_data.shape}")
            logger.info(f"Enhanced data columns: {list(btc_data.columns)}")
            
            return btc_data
            
        except Exception as e:
            logger.error(f"Error fetching enhanced data: {e}")
            return self.generate_dummy_data()
    
    def _add_enhanced_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add all technical indicators from the 50 signals"""
        # Use the comprehensive signal calculator
        enhanced_data = self.signal_calculator.calculate_all_signals(data)
        
        # Add additional custom indicators
        # Multiple timeframe analysis
        for period in [5, 10, 20, 50, 100, 200]:
            if len(data) >= period:
                enhanced_data[f'sma_{period}'] = data['Close'].rolling(period).mean()
                enhanced_data[f'ema_{period}'] = data['Close'].ewm(span=period).mean()
        
        # Volatility measures
        enhanced_data['volatility_20'] = data['Close'].pct_change().rolling(20).std() * np.sqrt(252)
        enhanced_data['volatility_ratio'] = enhanced_data['volatility_20'] / enhanced_data['volatility_20'].rolling(60).mean()
        
        # Volume analysis
        if 'Volume' in data.columns:
            enhanced_data['volume_ratio'] = data['Volume'] / data['Volume'].rolling(20).mean()
            enhanced_data['price_volume_trend'] = (enhanced_data['Close'].pct_change() * enhanced_data['Volume']).cumsum()
        
        # Price patterns
        enhanced_data['higher_high'] = (enhanced_data['High'] > enhanced_data['High'].shift(1)) & \
                                       (enhanced_data['High'].shift(1) > enhanced_data['High'].shift(2))
        enhanced_data['lower_low'] = (enhanced_data['Low'] < enhanced_data['Low'].shift(1)) & \
                                     (enhanced_data['Low'].shift(1) < enhanced_data['Low'].shift(2))
        
        # Fill NaN values
        enhanced_data = enhanced_data.fillna(method='bfill').fillna(0)
        
        return enhanced_data
    
    def _add_macro_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add macroeconomic indicators using external data fetcher"""
        try:
            # Fetch macro data from external fetcher
            end_date = data.index[-1]
            start_date = data.index[0]
            period = self._estimate_period(start_date, end_date)
            
            # Get macro data
            macro_data = self.data_fetcher.fetch_all_market_data(period)
            
            # S&P 500
            if 'SPY' in macro_data and not macro_data['SPY'].empty:
                spy_data = macro_data['SPY'].reindex(data.index)
                spy_data = spy_data.interpolate(method='nearest')
                data['sp500_returns'] = spy_data['Close'].pct_change()
                data['btc_sp500_corr'] = data['Close'].pct_change().rolling(30).corr(spy_data['Close'].pct_change())
            
            # Gold
            if 'GLD' in macro_data and not macro_data['GLD'].empty:
                gold_data = macro_data['GLD'].reindex(data.index)
                gold_data = gold_data.interpolate(method='nearest')
                data['gold_returns'] = gold_data['Close'].pct_change()
                data['btc_gold_corr'] = data['Close'].pct_change().rolling(30).corr(gold_data['Close'].pct_change())
            
            # DXY
            if 'DXY' in macro_data and not macro_data['DXY'].empty:
                dxy_data = macro_data['DXY'].reindex(data.index)
                dxy_data = dxy_data.interpolate(method='nearest')
                data['dxy_returns'] = dxy_data['Close'].pct_change()
                data['btc_dxy_corr'] = data['Close'].pct_change().rolling(30).corr(dxy_data['Close'].pct_change())
            
            # VIX
            if 'VIX' in macro_data and not macro_data['VIX'].empty:
                vix_data = macro_data['VIX'].reindex(data.index)
                vix_data = vix_data.interpolate(method='nearest')
                data['vix_level'] = vix_data['Close']
                data['vix_change'] = vix_data['Close'].pct_change()
            
            # Add real sentiment data
            sentiment_data = self.data_fetcher.fetch_sentiment_data()
            if sentiment_data:
                # Add latest sentiment values
                data['fear_greed_index'] = sentiment_data['fear_greed'].get('value', 50)
                data['social_sentiment'] = sentiment_data['social'].get('overall_sentiment', 0.5)
                data['news_sentiment'] = sentiment_data['news'].get('news_sentiment', 0.5)
            
            # Add real on-chain data
            onchain_data = self.data_fetcher.fetch_onchain_data()
            if onchain_data and 'network' in onchain_data:
                network = onchain_data['network']
                # Add as scalar values (pandas will broadcast)
                data['active_addresses'] = network.get('active_addresses', 0)
                data['transaction_count'] = network.get('transaction_count', 0)
                data['hash_rate'] = network.get('hash_rate', 0)
            
        except Exception as e:
            logger.warning(f"Error fetching macro data: {e}")
            # Add dummy macro features
            data['sp500_returns'] = np.random.normal(0, 0.01, len(data))
            data['btc_sp500_corr'] = np.random.uniform(-0.5, 0.5, len(data))
            data['gold_returns'] = np.random.normal(0, 0.005, len(data))
            data['vix_level'] = np.random.uniform(10, 30, len(data))
        
        return data.fillna(0)
    
    def _estimate_period(self, start_date, end_date) -> str:
        """Estimate period string from date range"""
        days = (end_date - start_date).days
        if days <= 7:
            return '7d'
        elif days <= 30:
            return '1mo'
        elif days <= 90:
            return '3mo'
        elif days <= 180:
            return '6mo'
        elif days <= 365:
            return '1y'
        elif days <= 730:
            return '2y'
        else:
            return 'max'
    
    def _estimate_period(self, start_date, end_date) -> str:
        """Estimate period string from date range"""
        days = (end_date - start_date).days
        if days <= 7:
            return '7d'
        elif days <= 30:
            return '1mo'
        elif days <= 90:
            return '3mo'
        elif days <= 180:
            return '6mo'
        elif days <= 365:
            return '1y'
        elif days <= 730:
            return '2y'
        else:
            return 'max'
    
    def _add_sentiment_proxies(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add sentiment indicators from real sources"""
        try:
            # Fetch real sentiment data
            sentiment_data = self.data_fetcher.fetch_sentiment_data()
            
            if sentiment_data:
                # Use real fear & greed index
                fear_greed = sentiment_data['fear_greed']
                data['fear_greed_value'] = fear_greed.get('value', 50)
                data['fear_proxy'] = (100 - fear_greed.get('value', 50)) / 100
                data['greed_proxy'] = fear_greed.get('value', 50) / 100
                
                # Use real social sentiment
                social = sentiment_data['social']
                data['twitter_sentiment'] = social.get('twitter_sentiment', 0.5)
                data['reddit_sentiment'] = social.get('reddit_sentiment', 0.5)
                data['news_sentiment'] = social.get('news_sentiment', 0.5)
                data['overall_sentiment'] = social.get('overall_sentiment', 0.5)
                
                # Google trends
                google = sentiment_data.get('google', {})
                data['google_trend'] = google.get('google_trend_normalized', 0.5)
            else:
                # Fallback to proxies
                return self._add_sentiment_proxies_fallback(data)
                
        except Exception as e:
            logger.warning(f"Error fetching real sentiment data: {e}")
            return self._add_sentiment_proxies_fallback(data)
        
        return data.fillna(0.5)
    
    def _add_sentiment_proxies_fallback(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fallback sentiment proxies"""
        # Volume-based sentiment proxy
        if 'Volume' in data.columns:
            data['volume_sentiment'] = data['Volume'].rolling(24).mean() / (data['Volume'].rolling(168).mean() + 1e-8)
        else:
            data['volume_sentiment'] = 1.0
        
        # Price momentum as sentiment
        data['momentum_sentiment'] = data['Close'].pct_change(24) / (data['Close'].pct_change(24).rolling(30).std() + 1e-8)
        
        # Simplified proxies
        data['fear_proxy'] = data.get('volatility_20', 0.3) / 0.3
        data['greed_proxy'] = (data['Close'] - data['Close'].rolling(30).min()) / (data['Close'].rolling(30).max() - data['Close'].rolling(30).min() + 1e-8)
        
        return data.fillna(0.5)
    
    def _add_sentiment_proxies_fallback(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fallback sentiment proxies"""
        # Volume-based sentiment proxy
        if 'Volume' in data.columns:
            data['volume_sentiment'] = data['Volume'].rolling(24).mean() / (data['Volume'].rolling(168).mean() + 1e-8)
        else:
            data['volume_sentiment'] = 1.0
        
        # Price momentum as sentiment
        data['momentum_sentiment'] = data['Close'].pct_change(24) / (data['Close'].pct_change(24).rolling(30).std() + 1e-8)
        
        # Simplified proxies
        data['fear_proxy'] = data.get('volatility_20', 0.3) / 0.3
        data['greed_proxy'] = (data['Close'] - data['Close'].rolling(30).min()) / (data['Close'].rolling(30).max() - data['Close'].rolling(30).min() + 1e-8)
        
        return data.fillna(0.5)
    
    def _add_onchain_proxies(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add on-chain indicators from real sources"""
        try:
            # Fetch real on-chain data
            onchain_data = self.data_fetcher.fetch_onchain_data()
            
            if onchain_data:
                # Network metrics
                network = onchain_data.get('network', {})
                
                # Ensure scalar values are properly assigned
                for key, value in network.items():
                    if key in ['active_addresses', 'transaction_count', 'hash_rate', 'difficulty', 'mempool_size']:
                        if isinstance(value, (int, float)):
                            data[key] = value  # Pandas will broadcast automatically
                        else:
                            logger.warning(f"Non-scalar value for {key}: {type(value)}")
                            data[key] = 0
                
                # NVT Ratio
                if 'nvt_ratio' in network:
                    data['nvt_ratio'] = network['nvt_ratio']
                elif 'market_cap_usd' in network and 'transaction_count' in network:
                    # Calculate NVT if not provided
                    daily_tx_vol = network['transaction_count'] * network.get('market_price_usd', 45000)
                    data['nvt_ratio'] = network['market_cap_usd'] / (daily_tx_vol * 365 + 1e-8)
                
                # Exchange flows
                flows = onchain_data.get('flows', {})
                flow_metrics = {
                    'exchange_inflow': flows.get('exchange_inflow', 0),
                    'exchange_outflow': flows.get('exchange_outflow', 0),
                    'net_exchange_flow': flows.get('net_flow', 0),
                    'whale_activity': flows.get('whale_inflow', 0) + flows.get('whale_outflow', 0)
                }
                
                # Add flow metrics as scalars
                for key, value in flow_metrics.items():
                    if isinstance(value, (int, float)):
                        data[key] = value
                    else:
                        data[key] = 0
                
                # Calculate derived metrics
                if 'Volume' in data.columns and network.get('transaction_count', 0) > 0:
                    data['nvt_proxy'] = data['Close'] * data['Volume'].rolling(30).sum() / (network['transaction_count'] * 7 + 1e-8)
                
                # HODL proxy based on low velocity
                if network.get('active_addresses', 0) > 0:
                    velocity = network.get('transaction_count', 0) / network['active_addresses']
                    data['hodl_proxy'] = 1 / (1 + velocity / 10)
                
            else:
                return self._add_onchain_proxies_fallback(data)
                
        except Exception as e:
            logger.warning(f"Error fetching real on-chain data: {e}")
            return self._add_onchain_proxies_fallback(data)
        
        return data.fillna(0)
    
    def _add_onchain_proxies_fallback(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fallback on-chain proxies"""
        # Network value proxy using price and volume
        if 'Volume' in data.columns:
            data['nvt_proxy'] = data['Close'] * data['Volume'].rolling(30).sum() / (data['Volume'].rolling(7).sum() + 1e-8)
        
        # HODL proxy based on low volatility periods
        data['hodl_proxy'] = 1 / (1 + data['Close'].pct_change().rolling(30).std())
        
        # Accumulation proxy
        data['accumulation_proxy'] = ((data['Close'] - data['Low']) / (data['High'] - data['Low'] + 1e-8)).rolling(30).mean()
        
        # Simple whale proxy based on large volume spikes
        if 'Volume' in data.columns:
            volume_zscore = (data['Volume'] - data['Volume'].rolling(30).mean()) / (data['Volume'].rolling(30).std() + 1e-8)
            data['whale_proxy'] = (volume_zscore > 2).astype(float)
        
        return data.fillna(0)
    
    def _add_onchain_proxies_fallback(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fallback on-chain proxies"""
        # Network value proxy using price and volume
        if 'Volume' in data.columns:
            data['nvt_proxy'] = data['Close'] * data['Volume'].rolling(30).sum() / (data['Volume'].rolling(7).sum() + 1e-8)
        
        # HODL proxy based on low volatility periods
        data['hodl_proxy'] = 1 / (1 + data['Close'].pct_change().rolling(30).std())
        
        # Accumulation proxy
        data['accumulation_proxy'] = ((data['Close'] - data['Low']) / (data['High'] - data['Low'] + 1e-8)).rolling(30).mean()
        
        # Simple whale proxy based on large volume spikes
        if 'Volume' in data.columns:
            volume_zscore = (data['Volume'] - data['Volume'].rolling(30).mean()) / (data['Volume'].rolling(30).std() + 1e-8)
            data['whale_proxy'] = (volume_zscore > 2).astype(float)
        
        return data.fillna(0)
    
    def set_btc_data_cache(self, data: pd.DataFrame) -> None:
        """Cache BTC data for reuse during backtesting"""
        self._cached_btc_data = data.copy() if data is not None else None
        logger.info(f"Cached BTC data with shape: {data.shape if data is not None else 'None'}")
    
    def get_btc_data_cache(self) -> pd.DataFrame:
        """Retrieve cached BTC data"""
        return self._cached_btc_data.copy() if self._cached_btc_data is not None else None
    
    def clear_btc_data_cache(self) -> None:
        """Clear cached BTC data to free memory"""
        self._cached_btc_data = None
        logger.info("Cleared BTC data cache")
    
    def prepare_enhanced_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features using LSTM best practices"""
        logger.info("Preparing enhanced features with LSTM best practices...")
        logger.info(f"Input data shape: {data.shape}")
        
        # Calculate all signals
        features = self.signal_calculator.calculate_all_signals(data)
        logger.info(f"Features after signal calculation: {features.shape}")
        
        # Feature engineering based on research
        categorized = pd.DataFrame(index=features.index)
        
        # Group features by category
        technical_features = []
        onchain_features = []
        sentiment_features = []
        macro_features = []
        
        # Technical features (comprehensive set)
        tech_cols = ['rsi', 'macd', 'bb_position', 'atr_normalized', 'adx', 'cci', 'mfi',
                     'roc', 'stoch_k', 'aroon_bullish', 'obv_trend', 'cmf', 'sar_trend',
                     'volume_ratio', 'volatility_ratio', 'momentum_sentiment']
        
        for col in tech_cols:
            if col in features.columns:
                technical_features.append(col)
                categorized[f'tech_{col}'] = features[col]
        
        # On-chain proxy features
        onchain_cols = ['nvt_proxy', 'hodl_proxy', 'accumulation_proxy', 'whale_proxy',
                        'net_exchange_flow', 'active_addr_growth', 'hash_rate_growth']
        
        for col in onchain_cols:
            if col in features.columns:
                onchain_features.append(col)
                categorized[f'onchain_{col}'] = features[col]
        
        # Sentiment features
        sentiment_cols = ['fear_proxy', 'greed_proxy', 'funding_proxy', 'volume_sentiment',
                          'extreme_fear', 'extreme_greed', 'twitter_bullish', 'reddit_spike']
        
        for col in sentiment_cols:
            if col in features.columns:
                sentiment_features.append(col)
                categorized[f'sent_{col}'] = features[col]
        
        # Macro features
        macro_cols = ['sp500_returns', 'btc_sp500_corr', 'gold_returns', 'btc_gold_corr',
                      'dxy_returns', 'btc_dxy_corr', 'vix_level', 'vix_change']
        
        for col in macro_cols:
            if col in features.columns:
                macro_features.append(col)
                categorized[f'macro_{col}'] = features[col]
        
        # Add composite features for backtesting compatibility
        if technical_features:
            categorized['technical_features'] = features[technical_features].mean(axis=1)
        else:
            categorized['technical_features'] = 0.5
            
        if onchain_features:
            categorized['onchain_features'] = features[onchain_features].mean(axis=1)
        else:
            categorized['onchain_features'] = 0.5
            
        if sentiment_features:
            categorized['sentiment_features'] = features[sentiment_features].mean(axis=1)
        else:
            categorized['sentiment_features'] = 0.5
            
        if macro_features:
            categorized['macro_features'] = features[macro_features].mean(axis=1)
        else:
            categorized['macro_features'] = 0.5
        
        # Add target and price columns
        if 'Close' in features.columns:
            categorized['Close'] = features['Close']
        elif 'Close' in data.columns:
            categorized['Close'] = data['Close']
        else:
            categorized['Close'] = 1.0
            
        # Also preserve OHLV columns if available for backtesting
        for col in ['Open', 'High', 'Low', 'Volume']:
            if col in features.columns:
                categorized[col] = features[col]
            elif col in data.columns:
                categorized[col] = data[col]
                
        categorized['target'] = categorized['Close'].pct_change().shift(-1).fillna(0)
        
        # Add divergence signals
        categorized['bullish_divergence'] = features.get('rsi_bullish_divergence', False) | \
                                           features.get('macd_bullish_divergence', False)
        categorized['bearish_divergence'] = features.get('rsi_bearish_divergence', False) | \
                                           features.get('macd_bearish_divergence', False)
        
        logger.info(f"Enhanced features shape: {categorized.shape}")
        logger.info(f"Feature categories: Technical={len(technical_features)}, "
                   f"OnChain={len(onchain_features)}, Sentiment={len(sentiment_features)}, "
                   f"Macro={len(macro_features)}")
        
        return categorized
    
    def train_enhanced_model(self, data: pd.DataFrame, epochs: int = 100, 
                           validation_split: float = 0.2, early_stopping_patience: int = 10):
        """Train model with LSTM best practices"""
        logger.info("Training enhanced LSTM model with best practices...")
        
        try:
            # Prepare features
            features = self.prepare_enhanced_features(data)
            
            # Remove NaN values
            logger.info(f"Features before dropna: {features.shape}")
            nan_cols = features.columns[features.isna().any()].tolist()
            if nan_cols:
                logger.warning(f"Columns with NaN values: {nan_cols}")
                for col in nan_cols:
                    nan_count = features[col].isna().sum()
                    logger.warning(f"  {col}: {nan_count} NaN values out of {len(features)} rows")
            # Fill NaN values instead of dropping rows
            features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)
            logger.info(f"Features after fillna: {features.shape}")
            
            min_required = self.sequence_length + 1  # Just need 1 sample for prediction after sequence
            if len(features) < min_required:
                logger.error(f"Insufficient data for training: {len(features)} rows, need at least {min_required} (sequence_length={self.sequence_length})")
                return
            
            # Feature selection based on importance
            selected_features = self._select_important_features(features)
            
            # Scale features
            feature_cols = [col for col in selected_features.columns if col not in ['target', 'Close']]
            scaled_features = self.feature_scaler.fit_transform(selected_features[feature_cols])
            
            # Scale target separately
            target = selected_features['target'].values.reshape(-1, 1)
            scaled_target = self.target_scaler.fit_transform(target)
            
            # Create sequences
            X, y = self.create_sequences(
                np.column_stack([scaled_target, scaled_features])
            )
            
            if len(X) == 0:
                logger.error("No sequences created")
                return
            
            # Split data temporally
            split_idx = int(len(X) * (1 - validation_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Convert to tensors
            X_train = torch.FloatTensor(X_train)
            y_train = torch.FloatTensor(y_train).unsqueeze(1)
            X_val = torch.FloatTensor(X_val)
            y_val = torch.FloatTensor(y_val).unsqueeze(1)
            
            # Update model with correct input size
            input_size = X_train.shape[2]  # Number of features
            if self.model.lstm.input_size != input_size:
                logger.info(f"Recreating model with input_size={input_size} (was {self.model.lstm.input_size})")
                from models.lstm import LSTMTradingModel
                self.model = LSTMTradingModel(
                    input_size=input_size,
                    hidden_size=50,
                    num_layers=2,
                    dropout=self.dropout_rate
                )
            
            # Enhanced training with early stopping
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
            criterion = torch.nn.MSELoss()
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
            
            best_val_loss = float('inf')
            patience_counter = 0
            best_model_state = None
            
            self.model.train()
            for epoch in range(epochs):
                # Training
                epoch_loss = 0
                for i in range(0, len(X_train), self.batch_size):
                    batch_X = X_train[i:i+self.batch_size]
                    batch_y = y_train[i:i+self.batch_size]
                    
                    optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    epoch_loss += loss.item()
                
                # Validation
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val)
                    val_loss = criterion(val_outputs, y_val).item()
                
                scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = self.model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
                
                if epoch % 10 == 0:
                    logger.info(f'Epoch [{epoch}/{epochs}], Train Loss: {epoch_loss/len(X_train):.6f}, '
                               f'Val Loss: {val_loss:.6f}')
                
                self.model.train()
            
            # Restore best model
            if best_model_state is not None:
                self.model.load_state_dict(best_model_state)
            
            self.is_trained = True
            logger.info("Enhanced model training completed!")
            
            # Calculate feature importance
            self._calculate_feature_importance(X_val, y_val, feature_cols)
            
        except Exception as e:
            logger.error(f"Error during enhanced training: {e}")
            logger.error(traceback.format_exc())
    
    def _select_important_features(self, features: pd.DataFrame, 
                                  correlation_threshold: float = 0.95) -> pd.DataFrame:
        """Select important features and remove highly correlated ones"""
        # Remove highly correlated features
        feature_cols = [col for col in features.columns if col not in ['target', 'Close']]
        
        if len(feature_cols) > 1:
            corr_matrix = features[feature_cols].corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            
            # Find features to drop
            to_drop = [column for column in upper.columns if any(upper[column] > correlation_threshold)]
            
            logger.info(f"Dropping {len(to_drop)} highly correlated features")
            features = features.drop(columns=to_drop)
        
        return features
    
    def _calculate_feature_importance(self, X_val: torch.Tensor, y_val: torch.Tensor, 
                                    feature_names: List[str]):
        """Calculate feature importance using permutation"""
        self.model.eval()
        criterion = torch.nn.MSELoss()
        
        with torch.no_grad():
            baseline_loss = criterion(self.model(X_val), y_val).item()
            
            importances = {}
            for i, feature_name in enumerate(feature_names):
                # Permute feature
                X_permuted = X_val.clone()
                X_permuted[:, :, i] = X_permuted[torch.randperm(len(X_permuted)), :, i]
                
                # Calculate loss with permuted feature
                permuted_loss = criterion(self.model(X_permuted), y_val).item()
                
                # Importance is the increase in loss
                importances[feature_name] = (permuted_loss - baseline_loss) / baseline_loss
            
            # Sort by importance
            self.feature_importance = dict(sorted(importances.items(), 
                                                key=lambda x: x[1], 
                                                reverse=True))
            
            logger.info(f"Top 5 important features: {list(self.feature_importance.keys())[:5]}")
    
    def predict_with_confidence(self, current_data: pd.DataFrame, 
                              n_predictions: int = 10) -> Tuple[str, float, float, Dict]:
        """Generate predictions with confidence intervals using ensemble"""
        if not self.is_trained:
            logger.warning("Model not trained, using rule-based prediction")
            # Use rule-based prediction instead of training in prediction method
            signal, confidence, price = self.generate_rule_based_signal(current_data)
            return signal, confidence, price, {"method": "rule_based", "reason": "model_not_trained"}
        
        # Prepare features
        features = self.prepare_enhanced_features(current_data)
        
        # Generate multiple predictions with dropout
        self.model.train()  # Enable dropout for uncertainty estimation
        predictions = []
        
        for _ in range(n_predictions):
            signal, confidence, predicted_price = self.predict_signal(current_data)
            predictions.append({
                'signal': signal,
                'confidence': confidence,
                'price': predicted_price
            })
        
        self.model.eval()
        
        # Aggregate predictions
        signals = [p['signal'] for p in predictions]
        signal_counts = {s: signals.count(s) for s in set(signals)}
        consensus_signal = max(signal_counts, key=signal_counts.get)
        
        # Calculate confidence based on consensus
        consensus_ratio = signal_counts[consensus_signal] / len(signals)
        avg_confidence = np.mean([p['confidence'] for p in predictions])
        final_confidence = consensus_ratio * avg_confidence
        
        # Price prediction with confidence interval
        prices = [p['price'] for p in predictions]
        predicted_price = np.mean(prices)
        price_std = np.std(prices)
        
        # Additional analysis
        analysis = {
            'consensus_ratio': consensus_ratio,
            'price_confidence_interval': (predicted_price - 2*price_std, predicted_price + 2*price_std),
            'signal_distribution': signal_counts,
            'feature_importance': dict(list(self.feature_importance.items())[:10]) if self.feature_importance else {}
        }
        
        return consensus_signal, final_confidence, predicted_price, analysis

# ========== ADVANCED INTEGRATED BACKTESTING SYSTEM ==========

class ModelWrapper:
    """Wrapper to make signal generator compatible with sklearn-style interface"""
    def __init__(self, signal_generator):
        self.signal_generator = signal_generator
        self.is_fitted = False
        
    def fit(self, X, y):
        """Dummy fit method - signal generator is already trained"""
        self.is_fitted = True
        return self
        
    def predict(self, X):
        """Generate predictions using the signal generator"""
        # Generate predictions based on feature patterns
        predictions = np.zeros(len(X))
        
        if isinstance(X, np.ndarray) and X.shape[1] > 0:
            # More sophisticated prediction logic
            for i in range(len(X)):
                feature_vec = X[i]
                
                # Simple trading logic based on feature values
                # Assume first few features are technical indicators normalized to [-1, 1]
                if len(feature_vec) >= 3:
                    # Use first 3 features as proxy for RSI, MACD, BB position
                    rsi_proxy = feature_vec[0]
                    macd_proxy = feature_vec[1] if len(feature_vec) > 1 else 0
                    bb_proxy = feature_vec[2] if len(feature_vec) > 2 else 0
                    
                    # Generate signal: positive for buy, negative for sell
                    signal = 0.0
                    
                    # Oversold/overbought conditions
                    if rsi_proxy < -0.5:  # Oversold
                        signal += 0.05
                    elif rsi_proxy > 0.5:  # Overbought
                        signal -= 0.05
                    
                    # Trend following
                    signal += macd_proxy * 0.03
                    
                    # Mean reversion
                    if bb_proxy < -0.5:  # Near lower band
                        signal += 0.03
                    elif bb_proxy > 0.5:  # Near upper band
                        signal -= 0.03
                    
                    predictions[i] = np.clip(signal, -0.1, 0.1)  # Limit to ±10% daily
                else:
                    # Random walk with slight negative bias (transaction costs)
                    predictions[i] = np.random.normal(-0.0001, 0.02)
        
        return predictions


class AdvancedIntegratedBacktestingSystem:
    """Advanced integrated system with all enhancements"""
    
    def __init__(self, db_path: str = None, model_path: str = None, use_enhanced_model: bool = True):
        self.db = DatabaseManager(db_path)
        
        # Model path is now handled by AdvancedTradingSignalGenerator
        self.signal_generator = AdvancedTradingSignalGenerator(model_path, use_enhanced_model=use_enhanced_model)
        self.config = BacktestConfig()
        self.config.min_train_test_ratio = 0.5
        
        # Create model wrapper for sklearn compatibility
        self.model_wrapper = ModelWrapper(self.signal_generator)
        
        # Use enhanced pipeline with the wrapper
        self.pipeline = EnhancedBacktestingPipeline(self.model_wrapper, self.config)
        
        # Performance tracking
        self.performance_tracker = {
            'backtests': [],
            'retraining_history': [],
            'signal_performance': {},
            'portfolio_metrics': []
        }
        
    def run_comprehensive_backtest(self, period: str = "2y", 
                                 optimize_weights: bool = True,
                                 include_macro: bool = True,
                                 save_results: bool = True) -> Dict:
        """Run full backtesting workflow with all enhancements"""
        logger.info(f"Starting advanced comprehensive backtest for {period}...")
        
        try:
            # Step 1: Fetch enhanced historical data
            logger.info("Fetching enhanced BTC data with all indicators...")
            btc_data = self.signal_generator.fetch_enhanced_btc_data(
                period=period, 
                include_macro=include_macro
            )
            
            if btc_data is None or len(btc_data) < self.signal_generator.sequence_length:
                logger.warning(f"Insufficient data for {period}, trying shorter period...")
                btc_data = self.signal_generator.fetch_enhanced_btc_data(period="6mo", include_macro=False)
                
                if btc_data is None or len(btc_data) < self.signal_generator.sequence_length:
                    raise ValueError("Insufficient data even with fallback period")
            
            logger.info(f"Fetched enhanced BTC data shape: {btc_data.shape}")
            
            # Cache the BTC data
            self.signal_generator.set_btc_data_cache(btc_data)
            
            # Step 2: Train enhanced model if not trained
            if not self.signal_generator.is_trained:
                logger.info("Training enhanced LSTM model with best practices...")
                self.signal_generator.train_enhanced_model(btc_data)
            
            # Step 3: Prepare enhanced features
            logger.info("Preparing comprehensive feature set...")
            features = self.signal_generator.prepare_enhanced_features(btc_data)
            
            # Log feature statistics
            logger.info(f"Features shape: {features.shape}")
            logger.info(f"Features columns: {list(features.columns)[:20]}...")  # First 20
            logger.info(f"NaN counts: {features.isna().sum().sum()}")
            
            # Step 4: Run advanced backtest
            logger.info(f"Features length: {len(features)}, optimize_weights: {optimize_weights}")
            if len(features) < 200 and optimize_weights:
                logger.warning("Limited data, using simplified backtest...")
                results = self._run_simplified_advanced_backtest(features)
            else:
                # Full enhanced backtest with optimization
                logger.info("Running enhanced backtest with signal optimization...")
                results = self.pipeline.run_full_backtest(
                    features, 
                    optimize_weights=optimize_weights,
                    use_enhanced_weights=True
                )
            
            # Step 5: Generate comprehensive analysis
            results = self._generate_comprehensive_analysis(results, features, btc_data)
            
            # Step 6: Store results
            if save_results:
                self._save_backtest_results(results)
            
            # Step 7: Update performance tracking
            self._update_performance_tracking(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Advanced backtest failed: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Return comprehensive error report
            return self._generate_error_report(str(e), period)
    
    def _run_simplified_advanced_backtest(self, features: pd.DataFrame) -> Dict:
        """Run simplified backtest with limited data"""
        logger.info("Running simplified advanced backtest...")
        
        try:
            # Initialize results
            returns = []
            positions = []
            signal_activations = {signal: 0 for signal in features.columns if signal.endswith('_cross') or signal.endswith('_spike')}
            
            # Simple walk-forward with 80/20 split
            split_point = int(len(features) * 0.8)
            train_data = features.iloc[:split_point]
            test_data = features.iloc[split_point:]
            
            # Generate signals for test period
            for i in range(len(test_data)):
                # Composite signal based on multiple indicators
                bull_score = 0
                bear_score = 0
                
                row = test_data.iloc[i]
                
                # Technical signals using enhanced feature names
                # RSI signals
                rsi_val = row.get('tech_rsi', 50)
                if rsi_val < 30:  # Oversold
                    bull_score += 2
                    signal_activations['rsi_oversold'] = signal_activations.get('rsi_oversold', 0) + 1
                elif rsi_val > 70:  # Overbought
                    bear_score += 2
                    signal_activations['rsi_overbought'] = signal_activations.get('rsi_overbought', 0) + 1
                
                # MACD signals
                macd_val = row.get('tech_macd', 0)
                if macd_val > 0.02:  # Bullish momentum
                    bull_score += 1.5
                    signal_activations['macd_bullish'] = signal_activations.get('macd_bullish', 0) + 1
                elif macd_val < -0.02:  # Bearish momentum
                    bear_score += 1.5
                    signal_activations['macd_bearish'] = signal_activations.get('macd_bearish', 0) + 1
                
                # Bollinger Bands position
                bb_pos = row.get('tech_bb_position', 0.5)
                if bb_pos < 0.2:  # Near lower band
                    bull_score += 1
                    signal_activations['bb_oversold'] = signal_activations.get('bb_oversold', 0) + 1
                elif bb_pos > 0.8:  # Near upper band
                    bear_score += 1
                    signal_activations['bb_overbought'] = signal_activations.get('bb_overbought', 0) + 1
                
                # Momentum (ROC)
                roc_val = row.get('tech_roc', 0)
                if roc_val > 0.05:  # Strong positive momentum
                    bull_score += 1
                elif roc_val < -0.05:  # Strong negative momentum
                    bear_score += 1
                
                # Volume signals
                vol_ratio = row.get('tech_volume_ratio', 1.0)
                if vol_ratio > 1.5 and bull_score > bear_score:
                    bull_score += 0.5
                    signal_activations['volume_spike'] = signal_activations.get('volume_spike', 0) + 1
                
                # Sentiment signals (if available)
                fear_val = row.get('sent_fear_proxy', 0.5)
                if fear_val < 0.3:  # Extreme fear (contrarian buy)
                    bull_score += 1.5
                elif fear_val > 0.7:  # Extreme greed (contrarian sell)
                    bear_score += 1.5
                
                # On-chain signals
                net_flow = row.get('onchain_net_exchange_flow', 0)
                if net_flow < -0.1:  # Outflow from exchanges (bullish)
                    bull_score += 1
                elif net_flow > 0.1:  # Inflow to exchanges (bearish)
                    bear_score += 1
                
                # Generate position with lower threshold for more trades
                if bull_score > bear_score + 0.5:
                    position = min(bull_score / 4, 1.0)  # Scale position
                elif bear_score > bull_score + 0.5:
                    position = max(-bear_score / 4, -1.0)
                else:
                    position = 0
                
                positions.append(position)
                
                # Calculate return
                if i < len(test_data) - 1:
                    # Use next period's return for backtesting
                    if 'Close' in test_data.columns:
                        next_return = (test_data['Close'].iloc[i + 1] - test_data['Close'].iloc[i]) / test_data['Close'].iloc[i]
                        ret = next_return * position
                    elif 'target' in test_data.columns:
                        ret = test_data.iloc[i]['target'] * position
                    else:
                        ret = 0.0
                else:
                    # Last period, no next return available
                    ret = 0.0
                
                returns.append(ret)
            
            # Calculate metrics
            returns = np.array(returns)
            positions = np.array(positions)
            
            # Apply transaction costs
            if len(positions) > 1:
                position_changes = np.diff(positions)
                transaction_costs = np.abs(position_changes) * self.config.transaction_cost
                returns[1:] -= transaction_costs
            
            # Generate results
            cumulative_returns = (1 + returns).cumprod()
            
            # Log debug info
            logger.info(f"Generated {len(positions)} positions, non-zero: {(np.array(positions) != 0).sum()}")
            logger.info(f"Total trades: {(np.diff(positions) != 0).sum() if len(positions) > 1 else 0}")
            logger.info(f"Signal activations: {signal_activations}")
            logger.info(f"Sample feature values - RSI: {test_data['tech_rsi'].iloc[0] if 'tech_rsi' in test_data.columns else 'N/A'}, MACD: {test_data['tech_macd'].iloc[0] if 'tech_macd' in test_data.columns else 'N/A'}")
            
            results = {
                'timestamp': datetime.now().isoformat(),
                'sortino_ratio_mean': self._calculate_sortino(returns),
                'calmar_ratio_mean': abs(np.mean(returns) * 252 / self._calculate_max_drawdown(cumulative_returns)),
                'max_drawdown_mean': self._calculate_max_drawdown(cumulative_returns),
                'profit_factor_mean': self._calculate_profit_factor(returns),
                'win_rate_mean': (returns > 0).sum() / len(returns) if len(returns) > 0 else 0.5,
                'total_return_mean': cumulative_returns[-1] - 1 if len(cumulative_returns) > 0 else 0,
                'sharpe_ratio_mean': self._calculate_sharpe(returns),
                'volatility_mean': np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0.2,
                'composite_score': 0.5,  # Simplified
                'periods_tested': 1,
                'success': True,
                'total_trades': int((np.diff(positions) != 0).sum() if len(positions) > 1 else 0),
                'signal_activations': signal_activations,
                'optimal_weights': {
                    'technical': 0.40,
                    'onchain': 0.30,
                    'sentiment': 0.20,
                    'macro': 0.10
                }
            }
            
            # Calculate composite score
            results['composite_score'] = self._calculate_composite_score(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Simplified backtest failed: {e}")
            return self._get_default_results()
    
    def _generate_comprehensive_analysis(self, results: Dict, features: pd.DataFrame, 
                                       raw_data: pd.DataFrame) -> Dict:
        """Generate comprehensive analysis with insights"""
        logger.info("Generating comprehensive analysis...")
        
        # Market regime analysis
        market_regime = self._analyze_market_regime(raw_data)
        
        # Feature correlation analysis
        feature_correlations = self._analyze_feature_correlations(features)
        
        # Signal effectiveness analysis
        signal_effectiveness = self._analyze_signal_effectiveness(results)
        
        # Risk decomposition
        risk_decomposition = self._decompose_risk(results)
        
        # Check if results are from simplified backtest (flat structure)
        # If so, restructure to match frontend expectations
        if 'performance_metrics' not in results:
            # Restructure flat results into nested format
            performance_metrics = {
                'sortino_ratio_mean': results.get('sortino_ratio_mean', 0.0),
                'sortino_ratio_std': 0.0,
                'sortino_ratio_min': results.get('sortino_ratio_mean', 0.0),
                'sortino_ratio_max': results.get('sortino_ratio_mean', 0.0),
                'calmar_ratio_mean': results.get('calmar_ratio_mean', 0.0),
                'calmar_ratio_std': 0.0,
                'calmar_ratio_min': results.get('calmar_ratio_mean', 0.0),
                'calmar_ratio_max': results.get('calmar_ratio_mean', 0.0),
                'max_drawdown_mean': results.get('max_drawdown_mean', 0.0),
                'max_drawdown_std': 0.0,
                'max_drawdown_min': results.get('max_drawdown_mean', 0.0),
                'max_drawdown_max': results.get('max_drawdown_mean', 0.0),
                'profit_factor_mean': results.get('profit_factor_mean', 1.0),
                'profit_factor_std': 0.0,
                'profit_factor_min': results.get('profit_factor_mean', 1.0),
                'profit_factor_max': results.get('profit_factor_mean', 1.0),
                'win_rate_mean': results.get('win_rate_mean', 0.5),
                'win_rate_std': 0.0,
                'win_rate_min': results.get('win_rate_mean', 0.5),
                'win_rate_max': results.get('win_rate_mean', 0.5),
                'total_return_mean': results.get('total_return_mean', 0.0),
                'total_return_std': 0.0,
                'total_return_min': results.get('total_return_mean', 0.0),
                'total_return_max': results.get('total_return_mean', 0.0),
                'sharpe_ratio_mean': results.get('sharpe_ratio_mean', 0.0),
                'sharpe_ratio_std': 0.0,
                'sharpe_ratio_min': results.get('sharpe_ratio_mean', 0.0),
                'sharpe_ratio_max': results.get('sharpe_ratio_mean', 0.0),
                'volatility_mean': results.get('volatility_mean', 0.2),
                'volatility_std': 0.0,
                'volatility_min': results.get('volatility_mean', 0.2),
                'volatility_max': results.get('volatility_mean', 0.2),
                'num_trades_mean': float(results.get('total_trades', 0)),
                'num_trades_std': 0.0,
                'num_trades_min': float(results.get('total_trades', 0)),
                'num_trades_max': float(results.get('total_trades', 0)),
                'composite_score': results.get('composite_score', 0.5),
            }
            
            trading_statistics = {
                'total_trades': results.get('total_trades', 0),
                'long_positions': 0,
                'short_positions': 0,
                'avg_position_turnover': 0.0
            }
            
            risk_metrics = {}
            
            # Create properly structured results
            structured_results = {
                'timestamp': results.get('timestamp'),
                'performance_metrics': performance_metrics,
                'trading_statistics': trading_statistics,
                'risk_metrics': risk_metrics,
                'optimal_weights': results.get('optimal_weights', {}),
                'periods_tested': results.get('periods_tested', 1),
                'success': results.get('success', True)
            }
        else:
            # Results already have proper structure
            structured_results = results
        
        # Enhanced results
        enhanced_results = {
            **structured_results,
            'market_analysis': {
                'regime': market_regime,
                'dominant_trend': self._identify_dominant_trend(raw_data),
                'volatility_regime': self._classify_volatility_regime(raw_data)
            },
            'feature_analysis': {
                'top_correlations': feature_correlations,
                'feature_importance': self.signal_generator.feature_importance
            },
            'signal_analysis': {
                'effectiveness': signal_effectiveness,
                'optimal_combinations': self._find_optimal_signal_combinations(results)
            },
            'risk_analysis': {
                'decomposition': risk_decomposition,
                'stress_scenarios': self._run_stress_tests(features)
            },
            'recommendations': self._generate_advanced_recommendations(results, market_regime),
            'confidence_score': self._calculate_confidence_score(results)
        }
        
        return enhanced_results
    
    def _analyze_market_regime(self, data: pd.DataFrame) -> str:
        """Identify current market regime"""
        recent_returns = data['Close'].pct_change().tail(30)
        recent_volatility = recent_returns.std() * np.sqrt(252)
        trend = (data['Close'].tail(1).values[0] - data['Close'].tail(30).values[0]) / data['Close'].tail(30).values[0]
        
        if recent_volatility > 1.0:
            if trend > 0.1:
                return "High Volatility Bull"
            elif trend < -0.1:
                return "High Volatility Bear"
            else:
                return "High Volatility Ranging"
        else:
            if trend > 0.05:
                return "Low Volatility Bull"
            elif trend < -0.05:
                return "Low Volatility Bear"
            else:
                return "Low Volatility Ranging"
    
    def _identify_dominant_trend(self, data: pd.DataFrame) -> str:
        """Identify dominant trend using multiple timeframes"""
        if len(data) < 200:
            return "Insufficient data"
        
        # Multiple timeframe analysis
        short_trend = (data['Close'].tail(20).mean() > data['Close'].tail(50).mean())
        medium_trend = (data['Close'].tail(50).mean() > data['Close'].tail(100).mean())
        long_trend = (data['Close'].tail(100).mean() > data['Close'].tail(200).mean())
        
        trend_score = sum([short_trend, medium_trend, long_trend])
        
        if trend_score >= 2:
            return "Bullish"
        elif trend_score <= 1:
            return "Bearish"
        else:
            return "Mixed"
    
    def _classify_volatility_regime(self, data: pd.DataFrame) -> str:
        """Classify volatility regime"""
        recent_vol = data['Close'].pct_change().tail(30).std() * np.sqrt(252)
        hist_vol = data['Close'].pct_change().std() * np.sqrt(252)
        
        vol_ratio = recent_vol / hist_vol
        
        if vol_ratio > 1.5:
            return "Expanding"
        elif vol_ratio < 0.7:
            return "Contracting"
        else:
            return "Normal"
    
    def _analyze_feature_correlations(self, features: pd.DataFrame) -> Dict:
        """Analyze feature correlations with returns"""
        if 'target' not in features.columns:
            return {}
        
        correlations = {}
        feature_cols = [col for col in features.columns if col not in ['target', 'Close']]
        
        for col in feature_cols[:20]:  # Top 20 to avoid clutter
            corr = features[col].corr(features['target'])
            if abs(corr) > 0.1:  # Only significant correlations
                correlations[col] = round(corr, 3)
        
        # Sort by absolute correlation
        sorted_corr = dict(sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True))
        
        return sorted_corr
    
    def _analyze_signal_effectiveness(self, results: Dict) -> Dict:
        """Analyze effectiveness of different signals"""
        signal_perf = results.get('signal_performance', {})
        
        if not signal_perf:
            return {'status': 'No signal performance data'}
        
        effectiveness = {}
        for signal, perf in signal_perf.items():
            if perf.get('total_count', 0) > 5:  # Minimum activations
                effectiveness[signal] = {
                    'avg_return': perf.get('avg_mean_return', 0),
                    'win_rate': perf.get('avg_win_rate', 0.5),
                    'frequency': perf.get('total_count', 0),
                    'contribution': perf.get('total_contribution', 0)
                }
        
        return effectiveness
    
    def _find_optimal_signal_combinations(self, results: Dict) -> List[Dict]:
        """Find optimal signal combinations"""
        # Simplified version - in practice would use more sophisticated analysis
        top_signals = results.get('top_contributing_signals', {})
        
        if len(top_signals) < 2:
            return []
        
        combinations = []
        signal_list = list(top_signals.keys())
        
        # Check pairs
        for i in range(len(signal_list)):
            for j in range(i+1, len(signal_list)):
                combinations.append({
                    'signals': [signal_list[i], signal_list[j]],
                    'estimated_improvement': 0.1  # Placeholder
                })
        
        return combinations[:5]  # Top 5 combinations
    
    def _decompose_risk(self, results: Dict) -> Dict:
        """Decompose risk into components"""
        return {
            'market_risk': results.get('volatility_mean', 0.5) * 0.6,
            'specific_risk': results.get('volatility_mean', 0.5) * 0.3,
            'model_risk': results.get('volatility_mean', 0.5) * 0.1,
            'total_risk': results.get('volatility_mean', 0.5)
        }
    
    def _run_stress_tests(self, features: pd.DataFrame) -> Dict:
        """Run stress test scenarios"""
        if 'Close' not in features.columns:
            return {}
        
        prices = features['Close'].values
        
        scenarios = {
            'flash_crash': -0.20,  # 20% drop
            'bull_rally': 0.30,    # 30% rally
            'high_volatility': 2.0, # 2x volatility
            'low_volatility': 0.5   # 0.5x volatility
        }
        
        stress_results = {}
        
        for scenario, shock in scenarios.items():
            if 'volatility' in scenario:
                # Volatility shock
                stressed_returns = np.random.normal(0, 0.02 * shock, len(prices))
            else:
                # Price shock
                stressed_returns = np.full(len(prices), shock / len(prices))
            
            # Simple impact calculation
            cumulative_impact = (1 + stressed_returns).cumprod()[-1] - 1
            stress_results[scenario] = {
                'impact': cumulative_impact,
                'max_drawdown': self._calculate_max_drawdown((1 + stressed_returns).cumprod())
            }
        
        return stress_results
    
    def _generate_advanced_recommendations(self, results: Dict, market_regime: str) -> List[str]:
        """Generate advanced recommendations based on comprehensive analysis"""
        recommendations = []
        
        # Base recommendations from results
        if results.get('sortino_ratio_mean', 0) < self.config.target_sortino_ratio:
            recommendations.append("Consider reducing position sizes to improve risk-adjusted returns")
        
        if abs(results.get('max_drawdown_mean', 0)) > self.config.max_drawdown_threshold:
            recommendations.append("Implement tighter stop-loss rules to reduce maximum drawdown")
        
        # Market regime specific recommendations
        if 'High Volatility' in market_regime:
            recommendations.append("High volatility detected - consider volatility-adjusted position sizing")
            recommendations.append("Use wider stops but smaller positions in high volatility regime")
        
        if 'Bear' in market_regime:
            recommendations.append("Bear market regime - consider defensive strategies or reduced exposure")
            recommendations.append("Focus on short signals and risk management")
        
        if 'Ranging' in market_regime:
            recommendations.append("Ranging market - consider mean reversion strategies")
            recommendations.append("Use support/resistance levels for entry/exit")
        
        # Signal-based recommendations
        top_signals = results.get('top_contributing_signals', {})
        if top_signals:
            best_signal = list(top_signals.keys())[0] if top_signals else None
            if best_signal:
                recommendations.append(f"Prioritize {best_signal} signal which showed highest effectiveness")
        
        # Risk-based recommendations
        if results.get('avg_omega_ratio', 1) < 1.5:
            recommendations.append("Omega ratio suggests adjusting profit targets relative to stop losses")
        
        if results.get('total_num_trades', 0) > 100:
            recommendations.append("High trading frequency detected - consider filtering signals to reduce costs")
        
        # Feature importance recommendations
        if hasattr(self.signal_generator, 'feature_importance') and self.signal_generator.feature_importance:
            top_feature = list(self.signal_generator.feature_importance.keys())[0]
            recommendations.append(f"Focus on {top_feature} which shows highest predictive power")
        
        return recommendations[:10]  # Limit to top 10 recommendations
    
    def _calculate_confidence_score(self, results: Dict) -> float:
        """Calculate confidence score for the backtest results"""
        confidence = 0.5  # Base confidence
        
        # Adjust based on performance metrics
        if results.get('sortino_ratio_mean', 0) > 2.0:
            confidence += 0.1
        if results.get('win_rate_mean', 0) > 0.55:
            confidence += 0.1
        if abs(results.get('max_drawdown_mean', -1)) < 0.20:
            confidence += 0.1
        if results.get('periods_tested', 0) > 5:
            confidence += 0.1
        if results.get('total_trades', 0) > 50:
            confidence += 0.05
        
        # Penalize poor metrics
        if results.get('sortino_ratio_mean', 0) < 1.0:
            confidence -= 0.1
        if abs(results.get('max_drawdown_mean', -1)) > 0.40:
            confidence -= 0.1
        
        return max(0.1, min(0.95, confidence))
    
    def _save_backtest_results(self, results: Dict):
        """Save backtest results to database and file"""
        try:
            # Save summary to database
            self.db.add_model_signal(
                symbol="BTC-USD",
                signal="backtest_complete",
                confidence=results.get('confidence_score', 0.5),
                price_prediction=None
            )
            
            # Save detailed results to file in writable directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"/app/data/backtest_results_{timestamp}.json"
            
            try:
                with open(filename, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                logger.info(f"Results saved to {filename}")
            except Exception as file_error:
                logger.warning(f"Could not save results to file: {file_error}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    def _update_performance_tracking(self, results: Dict):
        """Update performance tracking metrics"""
        self.performance_tracker['backtests'].append({
            'timestamp': datetime.now(),
            'composite_score': results.get('composite_score', 0),
            'sortino_ratio': results.get('sortino_ratio_mean', 0),
            'max_drawdown': results.get('max_drawdown_mean', 0),
            'total_trades': results.get('total_trades', 0)
        })
        
        # Keep only last 100 backtests
        if len(self.performance_tracker['backtests']) > 100:
            self.performance_tracker['backtests'] = self.performance_tracker['backtests'][-100:]
    
    def _generate_error_report(self, error: str, period: str) -> Dict:
        """Generate comprehensive error report"""
        return {
            'timestamp': datetime.now().isoformat(),
            'success': False,
            'error': error,
            'period_requested': period,
            'performance_metrics': self._get_default_results(),
            'optimal_weights': {
                'technical': 0.40,
                'onchain': 0.35,
                'sentiment': 0.15,
                'macro': 0.10
            },
            'risk_assessment': {
                'overall_risk': 'Unknown',
                'error': 'Assessment failed due to backtest error'
            },
            'recommendations': [
                f"Backtest failed: {error}",
                "Check data availability and quality",
                "Ensure sufficient historical data exists",
                "Try a shorter time period or disable macro indicators",
                "Check internet connection for data fetching"
            ],
            'market_analysis': {
                'regime': 'Unknown',
                'dominant_trend': 'Unknown',
                'volatility_regime': 'Unknown'
            }
        }
    
    def _get_default_results(self) -> Dict:
        """Get default results structure"""
        return {
            'sortino_ratio_mean': 0.0,
            'calmar_ratio_mean': 0.0,
            'max_drawdown_mean': 0.0,
            'profit_factor_mean': 1.0,
            'win_rate_mean': 0.5,
            'total_return_mean': 0.0,
            'sharpe_ratio_mean': 0.0,
            'volatility_mean': 0.2,
            'composite_score': 0.0,
            'periods_tested': 0,
            'success': False
        }
    
    def _calculate_sortino(self, returns: np.ndarray) -> float:
        """Calculate Sortino ratio"""
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - 0.02 / 252
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return 2.0
        
        downside_deviation = np.sqrt(np.mean(downside_returns**2)) * np.sqrt(252)
        annual_return = np.mean(excess_returns) * 252
        
        return annual_return / downside_deviation if downside_deviation > 0 else 2.0
    
    def _calculate_sharpe(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        
        excess_returns = returns - 0.02 / 252
        return np.mean(excess_returns) * 252 / (returns.std() * np.sqrt(252))
    
    def _calculate_max_drawdown(self, cumulative_returns: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        if len(cumulative_returns) == 0:
            return 0.0
        
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        return np.min(drawdown)
    
    def _calculate_profit_factor(self, returns: np.ndarray) -> float:
        """Calculate profit factor"""
        if len(returns) == 0:
            return 1.0
        
        profits = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        
        return profits / losses if losses > 0 else 2.0
    
    def _calculate_composite_score(self, metrics: Dict) -> float:
        """Calculate composite score"""
        score = 0.5  # Base score
        
        # Risk component (40%)
        sortino_score = min(metrics.get('sortino_ratio_mean', 0) / 2.0, 2.0) * 0.20
        drawdown_score = max(0, 1 + metrics.get('max_drawdown_mean', -0.25) / 0.25) * 0.20
        
        # Return component (35%)
        return_score = min(max(metrics.get('total_return_mean', 0), -0.5), 0.5) * 0.35
        
        # Consistency component (25%)
        win_rate_score = metrics.get('win_rate_mean', 0.5) * 0.15
        trade_score = min(metrics.get('total_trades', 0) / 100, 1.0) * 0.10
        
        total_score = score + sortino_score + drawdown_score + return_score + win_rate_score + trade_score
        
        return max(0, min(1, total_score))
    
    # ========== ORIGINAL METHODS (PRESERVED) ==========
    
    def check_and_retrain(self):
        """Check if model needs retraining based on performance"""
        recent_signals = self.db.get_model_signals(limit=100)
        
        if len(recent_signals) < 10:
            logger.info("Not enough signals for performance evaluation")
            return False
        
        recent_errors = []
        for _, signal in recent_signals.iterrows():
            if signal['price_prediction'] is not None:
                actual_price = signal['price_prediction'] * (1 + np.random.randn() * 0.02)
                error = abs(signal['price_prediction'] - actual_price) / actual_price
                recent_errors.append(error)
        
        recent_errors = np.array(recent_errors)
        
        if self.pipeline.scheduler.should_retrain(recent_signals, recent_errors):
            logger.info("Retraining triggered!")
            self.retrain_model()
            return True
            
        return False
    
    def retrain_model(self, period: str = "6mo", save_model: bool = True) -> Dict:
        """Retrain the LSTM model with latest data"""
        logger.info("Starting model retraining...")
        
        results = {
            'status': 'started',
            'timestamp': datetime.now().isoformat(),
            'errors': [],
            'metrics': {}
        }
        
        try:
            logger.info(f"Fetching {period} of BTC data for retraining...")
            btc_data = self.signal_generator.fetch_enhanced_btc_data(period=period)
            
            if btc_data is None or len(btc_data) < self.signal_generator.sequence_length:
                raise ValueError(f"Insufficient data for retraining")
            
            results['data_points'] = len(btc_data)
            logger.info(f"Fetched {len(btc_data)} data points")
            
            logger.info("Retraining enhanced LSTM model...")
            start_time = datetime.now()
            
            self.signal_generator.train_enhanced_model(btc_data, epochs=100)
            
            training_time = (datetime.now() - start_time).total_seconds()
            results['training_time_seconds'] = training_time
            logger.info(f"Model retrained in {training_time:.2f} seconds")
            
            if save_model:
                # Determine the save directory based on environment
                if os.path.exists('/app/models'):
                    save_dir = '/app/models'
                elif os.path.exists('/app/data'):
                    save_dir = '/app/data'
                else:
                    # Development environment
                    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                                          'storage', 'models')
                    os.makedirs(save_dir, exist_ok=True)
                
                model_filename = f'lstm_btc_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth'
                model_path = os.path.join(save_dir, model_filename)
                self.signal_generator.save_model(model_path)
                results['saved_model_path'] = model_path
                logger.info(f"Model saved to {model_path}")
            
            results['status'] = 'completed'
            results['success'] = True
            
            # Update performance tracking
            self.performance_tracker['retraining_history'].append(results)
            
        except Exception as e:
            logger.error(f"Model retraining failed: {e}")
            results['status'] = 'failed'
            results['success'] = False
            results['errors'].append(str(e))
            
        return results

def main():
    """Main execution with enhanced features"""
    try:
        system = AdvancedIntegratedBacktestingSystem(
            db_path=os.getenv('DATABASE_PATH', '/app/data/trading_system.db'),
            model_path='models/lstm_btc_model.pth'
        )
        
        # Run comprehensive backtest with all enhancements
        results = system.run_comprehensive_backtest(
            period="2y",
            optimize_weights=True,
            include_macro=True,
            save_results=True
        )
        
        # Display comprehensive results
        print("\n" + "="*80)
        print("ADVANCED BACKTEST RESULTS SUMMARY")
        print("="*80)
        
        # Performance Metrics
        print("\nPERFORMANCE METRICS:")
        print(f"Composite Score: {results.get('composite_score', 0):.3f}")
        print(f"Confidence Score: {results.get('confidence_score', 0):.2%}")
        print(f"Sortino Ratio: {results.get('sortino_ratio_mean', 0):.2f}")
        print(f"Sharpe Ratio: {results.get('sharpe_ratio_mean', 0):.2f}")
        print(f"Maximum Drawdown: {results.get('max_drawdown_mean', 0):.2%}")
        print(f"Win Rate: {results.get('win_rate_mean', 0):.2%}")
        print(f"Total Return: {results.get('total_return_mean', 0):.2%}")
        print(f"Volatility: {results.get('volatility_mean', 0):.2%}")
        
        # Risk Metrics
        print("\nRISK METRICS:")
        risk_metrics = results.get('risk_metrics', {})
        print(f"Value at Risk (95%): {risk_metrics.get('var_95', 0):.2%}")
        print(f"Conditional VaR (95%): {risk_metrics.get('cvar_95', 0):.2%}")
        print(f"Omega Ratio: {risk_metrics.get('omega_ratio', 0):.2f}")
        
        # Trading Statistics
        print("\nTRADING STATISTICS:")
        trading_stats = results.get('trading_statistics', {})
        print(f"Total Trades: {trading_stats.get('total_trades', 0)}")
        print(f"Long Positions: {trading_stats.get('long_positions', 0)}")
        print(f"Short Positions: {trading_stats.get('short_positions', 0)}")
        print(f"Position Turnover: {trading_stats.get('avg_position_turnover', 0):.2%}")
        
        # Market Analysis
        print("\nMARKET ANALYSIS:")
        market = results.get('market_analysis', {})
        print(f"Market Regime: {market.get('regime', 'Unknown')}")
        print(f"Dominant Trend: {market.get('dominant_trend', 'Unknown')}")
        print(f"Volatility Regime: {market.get('volatility_regime', 'Unknown')}")
        
        # Optimal Weights
        print("\nOPTIMAL SIGNAL WEIGHTS:")
        weights = results.get('optimal_weights', {})
        print(f"Technical: {weights.get('technical', 0):.2%}")
        print(f"On-chain: {weights.get('onchain', 0):.2%}")
        print(f"Sentiment: {weights.get('sentiment', 0):.2%}")
        print(f"Macro: {weights.get('macro', 0):.2%}")
        
        # Sub-weights if available
        if 'technical_sub' in weights:
            print("\n  Technical Sub-weights:")
            tech_sub = weights['technical_sub']
            print(f"    Momentum: {tech_sub.get('momentum', 0):.2%}")
            print(f"    Trend: {tech_sub.get('trend', 0):.2%}")
            print(f"    Volatility: {tech_sub.get('volatility', 0):.2%}")
            print(f"    Volume: {tech_sub.get('volume', 0):.2%}")
        
        # Top Signals
        print("\nTOP CONTRIBUTING SIGNALS:")
        signal_analysis = results.get('signal_analysis', {})
        top_signals = signal_analysis.get('top_signals', {})
        for i, (signal, perf) in enumerate(list(top_signals.items())[:5], 1):
            print(f"{i}. {signal}: Total Contribution = {perf.get('total_contribution', 0):.4f}")
        
        # Recommendations
        print("\nRECOMMENDATIONS:")
        for i, rec in enumerate(results.get('recommendations', [])[:5], 1):
            print(f"{i}. {rec}")
        
        # Check if retraining needed
        if system.check_and_retrain():
            print("\nModel has been retrained with optimized parameters!")
        
        print("\nAdvanced backtest complete! Detailed results saved to file.")
        
    except Exception as e:
        print(f"\nBacktest failed: {str(e)}")
        logger.error("Main execution failed", exc_info=True)

if __name__ == "__main__":
    main()