"""
Enhanced LSTM Model Integration
Extends the base lstm_model.py with enhanced functionality while preserving all original features
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List, Dict, Optional
from datetime import datetime, timedelta
import time
import random
import requests
import json
import logging
import warnings
warnings.filterwarnings('ignore')

# Import enhanced components from backtesting_system (not enhanced_backtesting_system)
from backtesting_system import (
    SignalWeights, EnhancedSignalWeights, 
    ComprehensiveSignalCalculator, BacktestConfig
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========== ORIGINAL LSTM MODEL (PRESERVED) ==========

class LSTMTradingModel(nn.Module):
    def __init__(self, input_size: int = 16, hidden_size: int = 50, 
                 num_layers: int = 2, output_size: int = 1, dropout: float = 0.2):
        super(LSTMTradingModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        
        return out

# ========== ORIGINAL TRADING SIGNAL GENERATOR (PRESERVED) ==========

class TradingSignalGenerator:
    def __init__(self, model_path: str = None, sequence_length: int = 60):
        self.sequence_length = sequence_length
        self.model = LSTMTradingModel()
        self.scaler = MinMaxScaler()
        self.is_trained = False
        self.cached_data = None
        self.last_fetch_time = None
        
        if model_path:
            self.load_model(model_path)
    
    def fetch_btc_data(self, period: str = "3mo") -> pd.DataFrame:
        """Fetch BTC price data from Yahoo Finance"""
        try:
            logger.info(f"Fetching BTC data for period: {period}")
            import yfinance as yf
            
            btc = yf.Ticker("BTC-USD")
            df = btc.history(period=period)
            
            if df.empty:
                logger.warning("Empty dataframe received from yfinance")
                return self.generate_dummy_data()
            
            # Add technical indicators
            df = self.add_technical_indicators(df)
            
            logger.info(f"Fetched {len(df)} days of BTC data")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching BTC data: {e}")
            return self.generate_dummy_data()
    
    def generate_dummy_data(self) -> pd.DataFrame:
        """Generate dummy data for testing"""
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        
        # Generate realistic-looking price data
        np.random.seed(42)
        base_price = 40000
        price_changes = np.random.randn(100) * 0.02
        prices = base_price * (1 + price_changes).cumprod()
        
        df = pd.DataFrame({
            'Open': prices * (1 + np.random.randn(100) * 0.001),
            'High': prices * (1 + abs(np.random.randn(100)) * 0.005),
            'Low': prices * (1 - abs(np.random.randn(100)) * 0.005),
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, 100)
        }, index=dates)
        
        # Add technical indicators
        df = self.add_technical_indicators(df)
        
        return df
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to dataframe"""
        # Simple Moving Averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # Exponential Moving Averages
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # RSI
        df['RSI'] = self.calculate_rsi(df['Close'])
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Bollinger Bands
        bb_window = 20
        bb_std = 2
        df['BB_Middle'] = df['Close'].rolling(window=bb_window).mean()
        bb_std_dev = df['Close'].rolling(window=bb_window).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * bb_std_dev)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * bb_std_dev)
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        
        # Volatility
        df['Volatility'] = df['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)
        
        # Additional indicators
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['Close_Open_Ratio'] = df['Close'] / df['Open']
        
        # Price position
        df['Price_Position'] = (df['Close'] - df['Low'].rolling(20).min()) / \
                              (df['High'].rolling(20).max() - df['Low'].rolling(20).min())
        
        # Momentum
        df['ROC'] = df['Close'].pct_change(periods=10) * 100
        
        # ATR
        df['ATR'] = self.calculate_atr(df)
        
        # Stochastic
        df['Stoch_K'] = self.calculate_stochastic(df)
        df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
        
        # Normalize volume
        df['Volume_Norm'] = df['Volume'] / df['Volume'].rolling(window=50).mean()
        
        # Fill NaN values
        df = df.fillna(method='bfill')
        
        return df
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        
        return true_range.rolling(period).mean()
    
    def calculate_stochastic(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Stochastic Oscillator"""
        low_min = df['Low'].rolling(window=period).min()
        high_max = df['High'].rolling(window=period).max()
        
        k_percent = 100 * ((df['Close'] - low_min) / (high_max - low_min))
        
        return k_percent
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for model input"""
        feature_columns = [
            'Close', 'Volume_Norm', 'RSI', 'MACD', 'MACD_Histogram',
            'BB_Position', 'BB_Width', 'Volume_Ratio', 'OBV',
            'Volatility', 'High_Low_Ratio', 'Close_Open_Ratio',
            'Price_Position', 'ROC', 'ATR', 'Stoch_K'
        ]
        
        # Select available features
        available_features = [col for col in feature_columns if col in data.columns]
        features = data[available_features].copy()
        
        # Add price change as target
        features['Price_Change'] = features['Close'].pct_change()
        
        # Remove NaN values
        features = features.dropna()
        
        return features
    
    def create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM input"""
        X, y = [], []
        
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i])
            y.append(data[i, 0])  # Predict next price
        
        return np.array(X), np.array(y)
    
    def train_model(self, data: pd.DataFrame, epochs: int = 50, batch_size: int = 32):
        """Train the LSTM model"""
        logger.info("Training LSTM model...")
        
        # Prepare features
        features = self.prepare_features(data)
        
        if len(features) < self.sequence_length:
            logger.error("Insufficient data for training")
            return
        
        # Scale data
        scaled_data = self.scaler.fit_transform(features.values)
        
        # Create sequences
        X, y = self.create_sequences(scaled_data)
        
        if len(X) == 0:
            logger.error("No sequences created")
            return
        
        # Convert to tensors
        X = torch.FloatTensor(X)
        y = torch.FloatTensor(y).unsqueeze(1)
        
        # Train-test split
        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Training loop
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            
            # Mini-batch training
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                avg_loss = total_loss / (len(X_train) / batch_size)
                logger.info(f'Epoch [{epoch}/{epochs}], Loss: {avg_loss:.6f}')
        
        # Evaluate on test set
        self.model.eval()
        with torch.no_grad():
            test_outputs = self.model(X_test)
            test_loss = criterion(test_outputs, y_test)
            logger.info(f'Test Loss: {test_loss.item():.6f}')
        
        self.is_trained = True
        logger.info("Model training completed!")
    
    def predict_signal(self, current_data: pd.DataFrame) -> Tuple[str, float, float]:
        """Generate trading signal based on current data"""
        if not self.is_trained:
            logger.warning("Model not trained, using rule-based signals")
            return self.generate_rule_based_signal(current_data)
        
        # Prepare features
        features = self.prepare_features(current_data)
        
        if len(features) < self.sequence_length:
            logger.warning("Insufficient data for prediction")
            return "hold", 0.5, current_data['Close'].iloc[-1]
        
        # Get last sequence
        last_sequence = features.tail(self.sequence_length).values
        scaled_sequence = self.scaler.transform(last_sequence)
        
        # Reshape for LSTM
        sequence_tensor = torch.FloatTensor(scaled_sequence).unsqueeze(0)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(sequence_tensor)
        
        # Convert prediction to signal
        predicted_change = prediction.item()
        current_price = features['Close'].iloc[-1]
        predicted_price = current_price * (1 + predicted_change)
        
        # Calculate confidence based on prediction magnitude
        confidence = min(abs(predicted_change) * 10, 0.95)
        
        # Generate signal
        if predicted_change > 0.02:  # 2% threshold
            signal = "buy"
        elif predicted_change < -0.02:
            signal = "sell"
        else:
            signal = "hold"
        
        return signal, confidence, predicted_price
    
    def generate_rule_based_signal(self, data: pd.DataFrame) -> Tuple[str, float, float]:
        """Generate signal based on technical indicators"""
        if len(data) < 50:
            return "hold", 0.5, data['Close'].iloc[-1]
        
        latest = data.iloc[-1]
        
        # Signal scoring
        buy_score = 0
        sell_score = 0
        
        # RSI signals
        if 'RSI' in data.columns:
            if latest['RSI'] < 30:
                buy_score += 2
            elif latest['RSI'] > 70:
                sell_score += 2
        
        # MACD signals
        if 'MACD' in data.columns and 'MACD_Signal' in data.columns:
            if latest['MACD'] > latest['MACD_Signal']:
                buy_score += 1
            else:
                sell_score += 1
        
        # Moving average signals
        if 'SMA_20' in data.columns and 'SMA_50' in data.columns:
            if latest['Close'] > latest['SMA_20'] > latest['SMA_50']:
                buy_score += 2
            elif latest['Close'] < latest['SMA_20'] < latest['SMA_50']:
                sell_score += 2
        
        # Bollinger Band signals
        if 'BB_Position' in data.columns:
            if latest['BB_Position'] < 0.2:
                buy_score += 1
            elif latest['BB_Position'] > 0.8:
                sell_score += 1
        
        # Volume signals
        if 'Volume_Ratio' in data.columns:
            if latest['Volume_Ratio'] > 1.5:
                if latest['Close'] > latest['Open']:
                    buy_score += 1
                else:
                    sell_score += 1
        
        # Determine signal
        if buy_score > sell_score + 2:
            signal = "buy"
            confidence = min(0.5 + (buy_score - sell_score) * 0.1, 0.8)
        elif sell_score > buy_score + 2:
            signal = "sell"
            confidence = min(0.5 + (sell_score - buy_score) * 0.1, 0.8)
        else:
            signal = "hold"
            confidence = 0.5
        
        return signal, confidence, latest['Close']
    
    def save_model(self, path: str):
        """Save model state"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'is_trained': self.is_trained,
            'sequence_length': self.sequence_length
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model state"""
        try:
            checkpoint = torch.load(path, map_location=torch.device('cpu'))
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.scaler = checkpoint['scaler']
            self.is_trained = checkpoint['is_trained']
            self.sequence_length = checkpoint.get('sequence_length', 60)
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")

# ========== ENHANCED CLASSES ==========

class EnhancedLSTMTradingModel(LSTMTradingModel):
    """Enhanced LSTM model with additional features while preserving base functionality"""
    
    def __init__(self, input_size: int = 16, hidden_size: int = 50, 
                 num_layers: int = 2, output_size: int = 1, dropout: float = 0.2,
                 use_attention: bool = False):
        # Initialize base model
        super().__init__(input_size, hidden_size, num_layers, output_size, dropout)
        
        # Add optional attention mechanism
        self.use_attention = use_attention
        if use_attention:
            self.attention = nn.MultiheadAttention(hidden_size, num_heads=4, dropout=dropout)
            
    def forward(self, x):
        """Forward pass with optional attention"""
        if not self.use_attention:
            # Use base implementation
            return super().forward(x)
        
        # Enhanced forward with attention
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        
        if self.use_attention:
            # Apply attention to LSTM outputs
            # Reshape for attention: (seq_len, batch, features)
            lstm_out = lstm_out.transpose(0, 1)
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
            # Take the last timestep after attention
            out = attn_out[-1]
        else:
            out = lstm_out[:, -1, :]
            
        out = self.dropout(out)
        out = self.fc(out)
        
        return out

class IntegratedTradingSignalGenerator(TradingSignalGenerator):
    """
    Integrated signal generator that combines base functionality with enhancements
    Preserves ALL original methods while adding enhanced capabilities
    """
    
    def __init__(self, model_path: str = None, sequence_length: int = 60, 
                 use_enhanced_model: bool = False):
        # Initialize base class
        super().__init__(model_path, sequence_length)
        
        # Override model with enhanced version if requested
        if use_enhanced_model:
            self.model = EnhancedLSTMTradingModel(
                input_size=16,  # Keep compatible with base
                use_attention=True
            )
        
        # Initialize enhanced components
        self.signal_calculator = ComprehensiveSignalCalculator()
        self.enhanced_weights = EnhancedSignalWeights()
        self.enhanced_weights.normalize()
        self.enhanced_weights.normalize_subcategories()
        
        # Performance tracking
        self.performance_history = []
        self.feature_importance = {}
        
        # Enhanced scalers for multi-feature support
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
        
        logger.info("Integrated Trading Signal Generator initialized")
    
    def fetch_enhanced_btc_data(self, period: str = "1y", include_macro: bool = False) -> pd.DataFrame:
        """
        Enhanced data fetching that builds on base fetch_btc_data
        Adds comprehensive signals while preserving original functionality
        """
        # First, use base method to get BTC data
        logger.info(f"Fetching base BTC data for period: {period}")
        base_data = self.fetch_btc_data(period=period)
        
        if base_data is None or len(base_data) < self.sequence_length:
            logger.warning("Insufficient base data, using dummy data")
            base_data = self.generate_dummy_data()
        
        # Now enhance with comprehensive signals
        logger.info("Calculating comprehensive signals...")
        enhanced_data = self.signal_calculator.calculate_all_signals(base_data)
        
        # Add macro indicators if requested
        if include_macro:
            logger.info("Adding macro indicators...")
            enhanced_data = self._add_macro_indicators(enhanced_data)
        
        # Add sentiment proxies
        enhanced_data = self._add_sentiment_proxies(enhanced_data)
        
        # Add on-chain proxies
        enhanced_data = self._add_onchain_proxies(enhanced_data)
        
        logger.info(f"Enhanced data shape: {enhanced_data.shape}")
        return enhanced_data
    
    def _add_macro_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add macroeconomic indicators"""
        try:
            # This is a simplified version - in production, fetch real macro data
            # For now, generate synthetic macro indicators
            data['sp500_returns'] = np.random.normal(0, 0.01, len(data))
            data['btc_sp500_corr'] = data['Close'].pct_change().rolling(30).corr(
                pd.Series(np.random.normal(0, 0.01, len(data)))
            ).fillna(0)
            data['gold_returns'] = np.random.normal(0, 0.005, len(data))
            data['vix_level'] = np.random.uniform(10, 30, len(data))
            data['dxy_returns'] = np.random.normal(0, 0.003, len(data))
        except Exception as e:
            logger.warning(f"Error adding macro indicators: {e}")
            # Add dummy values
            data['sp500_returns'] = 0
            data['gold_returns'] = 0
            data['vix_level'] = 20
        
        return data
    
    def _add_sentiment_proxies(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add sentiment indicators"""
        if 'Volume' in data.columns:
            data['volume_sentiment'] = data['Volume'].rolling(24).mean() / (
                data['Volume'].rolling(168).mean() + 1e-8
            )
        else:
            data['volume_sentiment'] = 1.0
        
        # Price momentum as sentiment
        data['momentum_sentiment'] = data['Close'].pct_change(24) / (
            data['Close'].pct_change(24).rolling(30).std() + 1e-8
        )
        
        # Fear and greed proxies
        if 'volatility_20' in data.columns:
            data['fear_proxy'] = data['volatility_20'] / 0.3  # Normalized to typical BTC vol
        else:
            data['fear_proxy'] = data['Close'].pct_change().rolling(20).std() * np.sqrt(252) / 0.3
        
        data['greed_proxy'] = (data['Close'] - data['Close'].rolling(30).min()) / (
            data['Close'].rolling(30).max() - data['Close'].rolling(30).min() + 1e-8
        )
        
        return data.fillna(0.5)
    
    def _add_onchain_proxies(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add on-chain proxy indicators"""
        if 'Volume' in data.columns:
            # NVT proxy
            data['nvt_proxy'] = data['Close'] * data['Volume'].rolling(30).sum() / (
                data['Volume'].rolling(7).sum() + 1e-8
            )
            
            # Whale activity proxy
            volume_zscore = (data['Volume'] - data['Volume'].rolling(30).mean()) / (
                data['Volume'].rolling(30).std() + 1e-8
            )
            data['whale_proxy'] = (volume_zscore > 2).astype(float)
        else:
            data['nvt_proxy'] = 1.0
            data['whale_proxy'] = 0.0
        
        # HODL proxy
        data['hodl_proxy'] = 1 / (1 + data['Close'].pct_change().rolling(30).std() + 1e-8)
        
        # Accumulation proxy
        data['accumulation_proxy'] = ((data['Close'] - data['Low']) / (
            data['High'] - data['Low'] + 1e-8
        )).rolling(30).mean()
        
        return data.fillna(0)
    
    def prepare_enhanced_features(self, data: pd.DataFrame, 
                                use_weights: bool = True) -> pd.DataFrame:
        """
        Enhanced feature preparation that maintains compatibility with base prepare_features
        Can be called with use_weights=False to get base behavior
        """
        if not use_weights:
            # Use base implementation
            return self.prepare_features(data)
        
        # Enhanced feature preparation
        logger.info("Preparing enhanced features...")
        
        # Ensure we have all necessary columns
        if 'technical_features' not in data.columns:
            # Create aggregate features for compatibility
            tech_cols = ['RSI', 'MACD', 'Stoch_K', 'ATR', 'ROC', 'OBV']
            tech_vals = []
            for col in tech_cols:
                if col in data.columns:
                    tech_vals.append(data[col])
            
            if tech_vals:
                data['technical_features'] = pd.concat(tech_vals, axis=1).mean(axis=1)
            else:
                data['technical_features'] = 0.5
        
        # Prepare comprehensive feature set
        features = pd.DataFrame(index=data.index)
        
        # Core features (compatible with base model)
        features['price'] = data['Close']
        features['volume_norm'] = data.get('Volume_Norm', 1.0)
        features['obv'] = data.get('OBV', 0.0)
        features['rsi'] = data.get('RSI', 50.0)
        features['stoch_k'] = data.get('Stoch_K', 50.0)
        features['roc'] = data.get('ROC', 0.0)
        features['macd'] = data.get('MACD', 0.0)
        features['macd_histogram'] = data.get('MACD_Histogram', 0.0)
        features['sma_ratio'] = data['Close'] / (data.get('SMA_20', data['Close']) + 1e-8)
        features['ema_ratio'] = data.get('EMA_12', data['Close']) / (
            data.get('EMA_26', data['Close']) + 1e-8
        )
        features['bb_position'] = data.get('BB_Position', 0.5)
        features['bb_width'] = data.get('BB_Width', 0.1)
        features['atr'] = data.get('ATR', 1.0)
        features['volatility'] = data.get('Volatility', 1.0)
        
        # Add sentiment features
        features['fear_greed'] = data.get('fear_proxy', 0.5)
        features['btc_dominance'] = data.get('btc_dominance', 50) / 100
        
        # Add enhanced features if available
        if 'momentum_sentiment' in data.columns:
            features['momentum_sentiment'] = data['momentum_sentiment']
        
        if 'nvt_proxy' in data.columns:
            features['nvt_proxy'] = data['nvt_proxy']
        
        # Apply weights if using enhanced weights
        if hasattr(self, 'enhanced_weights') and use_weights:
            features = self._apply_enhanced_weights(features, data)
        
        # Clean and fill
        features = features.fillna(method='bfill').fillna(method='ffill').fillna(0)
        
        return features
    
    def _apply_enhanced_weights(self, features: pd.DataFrame, 
                               original_data: pd.DataFrame) -> pd.DataFrame:
        """Apply enhanced signal weights to features"""
        # This maintains the structure but applies weights
        weighted_features = features.copy()
        
        # Apply main category weights
        tech_cols = ['rsi', 'macd', 'stoch_k', 'bb_position', 'atr', 'roc']
        for col in tech_cols:
            if col in weighted_features.columns:
                weighted_features[col] *= self.enhanced_weights.technical_weight
        
        # Apply sentiment weights
        sent_cols = ['fear_greed', 'momentum_sentiment']
        for col in sent_cols:
            if col in weighted_features.columns:
                weighted_features[col] *= self.enhanced_weights.sentiment_weight
        
        # Apply on-chain weights
        chain_cols = ['nvt_proxy', 'obv']
        for col in chain_cols:
            if col in weighted_features.columns:
                weighted_features[col] *= self.enhanced_weights.onchain_weight
        
        return weighted_features
    
    def train_enhanced_model(self, data: pd.DataFrame, epochs: int = 50,
                           batch_size: int = 32, use_validation: bool = True):
        """
        Enhanced training that maintains compatibility with base train_model
        """
        logger.info("Training enhanced model...")
        
        try:
            # Prepare features (enhanced or base depending on data)
            if 'fear_proxy' in data.columns or 'nvt_proxy' in data.columns:
                features = self.prepare_enhanced_features(data, use_weights=True)
            else:
                features = self.prepare_features(data)
            
            # Ensure minimum data
            if len(features) < self.sequence_length + 10:
                logger.warning("Limited data, reducing epochs")
                epochs = min(epochs, 20)
            
            # Scale features
            scaled_data = self.scaler.fit_transform(features.values)
            X, y = self.create_sequences(scaled_data)
            
            if len(X) == 0:
                raise ValueError("No sequences created")
            
            # Convert to tensors
            X = torch.FloatTensor(X)
            y = torch.FloatTensor(y).unsqueeze(1)
            
            # Split data
            if use_validation:
                train_size = int(0.8 * len(X))
                X_train, X_val = X[:train_size], X[train_size:]
                y_train, y_val = y[:train_size], y[train_size:]
            else:
                X_train, y_train = X, y
                X_val, y_val = None, None
            
            # Training setup
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            
            # Training loop
            self.model.train()
            best_val_loss = float('inf')
            
            for epoch in range(epochs):
                total_loss = 0
                batch_count = 0
                
                for i in range(0, len(X_train), batch_size):
                    batch_X = X_train[i:i+batch_size]
                    batch_y = y_train[i:i+batch_size]
                    
                    if len(batch_X) == 0:
                        continue
                    
                    optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    batch_count += 1
                
                avg_loss = total_loss / batch_count if batch_count > 0 else 0
                
                # Validation
                if use_validation and X_val is not None:
                    self.model.eval()
                    with torch.no_grad():
                        val_outputs = self.model(X_val)
                        val_loss = criterion(val_outputs, y_val).item()
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                    
                    self.model.train()
                    
                    if epoch % 10 == 0:
                        logger.info(f'Epoch [{epoch}/{epochs}], Loss: {avg_loss:.6f}, Val Loss: {val_loss:.6f}')
                else:
                    if epoch % 10 == 0:
                        logger.info(f'Epoch [{epoch}/{epochs}], Loss: {avg_loss:.6f}')
            
            self.is_trained = True
            logger.info("Enhanced model training completed!")
            
        except Exception as e:
            logger.error(f"Error during enhanced training: {e}")
            # Fall back to base training
            logger.info("Falling back to base training method...")
            super().train_model(data, epochs=epochs, batch_size=batch_size)
    
    def predict_with_confidence(self, current_data: pd.DataFrame, 
                              n_predictions: int = 10) -> Tuple[str, float, float, Dict]:
        """
        Enhanced prediction with confidence intervals
        Returns: (signal, confidence, predicted_price, analysis_dict)
        """
        if not self.is_trained:
            logger.info("Model not trained, training now...")
            self.train_enhanced_model(current_data)
        
        # Prepare features
        if 'fear_proxy' in current_data.columns:
            features = self.prepare_enhanced_features(current_data)
        else:
            features = self.prepare_features(current_data)
        
        if len(features) < self.sequence_length:
            # Use base prediction method
            signal, confidence, price = self.predict_signal(current_data)
            return signal, confidence, price, {"method": "base", "n_predictions": 1}
        
        # Generate multiple predictions for confidence
        predictions = []
        self.model.train()  # Enable dropout for uncertainty
        
        for _ in range(n_predictions):
            with torch.no_grad():
                # Get last sequence
                last_sequence = features.tail(self.sequence_length).values
                scaled_sequence = self.scaler.transform(last_sequence)
                sequence_tensor = torch.FloatTensor(scaled_sequence).unsqueeze(0)
                
                # Predict
                pred = self.model(sequence_tensor)
                
                # Inverse transform
                dummy_array = np.zeros((1, features.shape[1]))
                dummy_array[0, 0] = pred.item()
                predicted_price = self.scaler.inverse_transform(dummy_array)[0, 0]
                
                predictions.append(predicted_price)
        
        self.model.eval()
        
        # Analyze predictions
        pred_array = np.array(predictions)
        mean_price = np.mean(pred_array)
        std_price = np.std(pred_array)
        
        # Determine signal
        current_price = features['price'].iloc[-1]
        price_change_pct = (mean_price - current_price) / current_price
        
        if price_change_pct > 0.02:
            signal = "buy"
        elif price_change_pct < -0.02:
            signal = "sell"
        else:
            signal = "hold"
        
        # Calculate confidence based on prediction consistency
        confidence = max(0.3, min(0.95, 1 - (std_price / mean_price)))
        
        # Build analysis
        analysis = {
            "method": "enhanced",
            "n_predictions": n_predictions,
            "price_mean": mean_price,
            "price_std": std_price,
            "confidence_interval": (mean_price - 2*std_price, mean_price + 2*std_price),
            "price_change_pct": price_change_pct,
            "current_price": current_price
        }
        
        return signal, confidence, mean_price, analysis
    
    # Override base methods to add compatibility
    def train_model(self, data: pd.DataFrame, epochs: int = 50, batch_size: int = 32):
        """Override base train_model to use enhanced version"""
        self.train_enhanced_model(data, epochs=epochs, batch_size=batch_size, use_validation=False)
    
    def predict_signal(self, current_data: pd.DataFrame) -> Tuple[str, float, float]:
        """Override base predict_signal to maintain compatibility"""
        # Check if we have enhanced data
        if any(col in current_data.columns for col in ['fear_proxy', 'nvt_proxy', 'momentum_sentiment']):
            # Use enhanced prediction
            signal, confidence, price, _ = self.predict_with_confidence(current_data, n_predictions=5)
            return signal, confidence, price
        else:
            # Use base implementation
            return super().predict_signal(current_data)
    
    def set_btc_data_cache(self, data: pd.DataFrame):
        """Set cached data for backtesting"""
        self.cached_data = data
        self.last_fetch_time = datetime.now()
    
    def set_signal_weights(self, weights: EnhancedSignalWeights):
        """Update signal weights"""
        self.enhanced_weights = weights
        self.enhanced_weights.normalize()
        self.enhanced_weights.normalize_subcategories()


def test_integration():
    """Test the integration of all components"""
    logger.info("Testing Enhanced LSTM Integration...")
    
    # Test 1: Basic functionality preservation
    logger.info("\n=== Test 1: Basic Functionality ===")
    basic_gen = IntegratedTradingSignalGenerator(use_enhanced_model=False)
    
    # Fetch basic data
    basic_data = basic_gen.fetch_btc_data(period="1mo")
    logger.info(f"Basic data shape: {basic_data.shape}")
    logger.info(f"Basic data columns: {list(basic_data.columns)[:10]}...")
    
    # Generate basic signal
    signal, conf, price = basic_gen.predict_signal(basic_data)
    logger.info(f"Basic signal: {signal}, confidence: {conf:.2%}, price: ${price:.2f}")
    
    # Test 2: Enhanced functionality
    logger.info("\n=== Test 2: Enhanced Functionality ===")
    enhanced_gen = IntegratedTradingSignalGenerator(use_enhanced_model=True)
    
    # Fetch enhanced data
    enhanced_data = enhanced_gen.fetch_enhanced_btc_data(period="1mo", include_macro=True)
    logger.info(f"Enhanced data shape: {enhanced_data.shape}")
    logger.info(f"Enhanced columns sample: {[col for col in enhanced_data.columns if 'proxy' in col or 'sentiment' in col]}")
    
    # Generate enhanced signal
    signal, conf, price, analysis = enhanced_gen.predict_with_confidence(enhanced_data)
    logger.info(f"Enhanced signal: {signal}, confidence: {conf:.2%}, price: ${price:.2f}")
    logger.info(f"Analysis: {analysis}")
    
    # Test 3: Weight application
    logger.info("\n=== Test 3: Signal Weights ===")
    custom_weights = EnhancedSignalWeights(
        technical_weight=0.5,
        onchain_weight=0.3,
        sentiment_weight=0.15,
        macro_weight=0.05
    )
    enhanced_gen.set_signal_weights(custom_weights)
    logger.info(f"Applied custom weights: tech={custom_weights.technical_weight}, onchain={custom_weights.onchain_weight}")
    
    # Test 4: Model training
    logger.info("\n=== Test 4: Model Training ===")
    enhanced_gen.train_enhanced_model(enhanced_data, epochs=10)
    
    # Test 5: Compatibility with base methods
    logger.info("\n=== Test 5: Base Method Compatibility ===")
    # Should work with base train_model method
    enhanced_gen.train_model(basic_data, epochs=5)
    
    # Should work with base predict_signal
    basic_signal, basic_conf, basic_price = enhanced_gen.predict_signal(basic_data)
    logger.info(f"Base method signal: {basic_signal}, confidence: {basic_conf:.2%}")
    
    logger.info("\n✅ All integration tests completed successfully!")
    
    return enhanced_gen, enhanced_data


if __name__ == "__main__":
    # Run integration test
    generator, data = test_integration()
    
    # Show feature importance if available
    if hasattr(generator, 'feature_importance') and generator.feature_importance:
        logger.info("\n=== Feature Importance ===")
        for feat, imp in list(generator.feature_importance.items())[:5]:
            logger.info(f"{feat}: {imp:.4f}")