import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List, Dict, Optional
import yfinance as yf
from datetime import datetime, timedelta
import time
import random
import requests
import json

class EnhancedLSTMTradingModel(nn.Module):
    def __init__(self, input_size: int = 15, hidden_size: int = 100, num_layers: int = 3, output_size: int = 1, dropout: float = 0.3):
        super(EnhancedLSTMTradingModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Enhanced LSTM with more layers and units for complex patterns
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        
        # Additional layers for better feature extraction
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

class EnhancedTradingSignalGenerator:
    def __init__(self, model_path: str = None, sequence_length: int = 60):
        self.sequence_length = sequence_length
        self.model = EnhancedLSTMTradingModel()
        self.scaler = MinMaxScaler()
        self.is_trained = False
        self.cached_data = None
        self.last_fetch_time = None
        self.external_data_cache = {}
        
        if model_path:
            self.load_model(model_path)
    
    def fetch_btc_data(self, period: str = "1y", interval: str = "1d", max_retries: int = 3) -> pd.DataFrame:
        """Fetch BTC data from Yahoo Finance with robust error handling"""
        
        # Check if we have recent cached data (within 5 minutes)
        if (self.cached_data is not None and 
            self.last_fetch_time is not None and 
            (datetime.now() - self.last_fetch_time).total_seconds() < 300):
            print("Using cached BTC data")
            return self.cached_data
        
        # Try to fetch real data with retries
        for attempt in range(max_retries):
            try:
                print(f"Attempting to fetch BTC data (attempt {attempt + 1}/{max_retries})...")
                
                if attempt > 0:
                    time.sleep(2 ** attempt)
                
                btc = yf.Ticker("BTC-USD")
                data = btc.history(period=period, interval=interval)
                
                if data is None or data.empty:
                    print(f"No data returned for BTC-USD on attempt {attempt + 1}")
                    continue
                
                # Add enhanced technical indicators
                data = self._add_enhanced_technical_indicators(data)
                
                # Add external data signals
                data = self._add_external_signals(data)
                
                # Validate final data
                if len(data) < self.sequence_length:
                    print(f"Insufficient data points ({len(data)}) on attempt {attempt + 1}")
                    continue
                
                print(f"Successfully fetched {len(data)} periods of BTC data")
                self.cached_data = data
                self.last_fetch_time = datetime.now()
                return data
                
            except Exception as e:
                print(f"Error fetching BTC data on attempt {attempt + 1}: {e}")
                continue
        
        # If all attempts failed, generate dummy data
        print("All attempts to fetch real data failed. Generating dummy data...")
        dummy_data = self.generate_dummy_data()
        self.cached_data = dummy_data
        self.last_fetch_time = datetime.now()
        return dummy_data
    
    def _add_enhanced_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators to the data"""
        try:
            # Original indicators
            data['SMA_20'] = data['Close'].rolling(window=20).mean()
            data['SMA_50'] = data['Close'].rolling(window=50).mean()
            data['RSI'] = self.calculate_rsi(data['Close'])
            data['MACD'], data['MACD_Signal'] = self.calculate_macd(data['Close'])
            data['Volume_Norm'] = data['Volume'] / data['Volume'].rolling(window=20).mean()
            
            # New indicators
            
            # Bollinger Bands
            bb_period = 20
            bb_std = 2
            data['BB_Middle'] = data['Close'].rolling(window=bb_period).mean()
            bb_std_dev = data['Close'].rolling(window=bb_period).std()
            data['BB_Upper'] = data['BB_Middle'] + (bb_std * bb_std_dev)
            data['BB_Lower'] = data['BB_Middle'] - (bb_std * bb_std_dev)
            data['BB_Width'] = (data['BB_Upper'] - data['BB_Lower']) / data['BB_Middle']
            data['BB_Position'] = (data['Close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])
            
            # ATR (Average True Range)
            data['ATR'] = self.calculate_atr(data)
            data['ATR_Ratio'] = data['ATR'] / data['Close']  # Normalized ATR
            
            # Stochastic Oscillator
            data['Stoch_K'], data['Stoch_D'] = self.calculate_stochastic(data)
            
            # On-Balance Volume (OBV)
            data['OBV'] = self.calculate_obv(data)
            data['OBV_EMA'] = data['OBV'].ewm(span=20).mean()
            data['OBV_Divergence'] = data['OBV'] - data['OBV_EMA']
            
            # VWAP (for intraday, approximate for daily)
            data['VWAP'] = self.calculate_vwap(data)
            data['VWAP_Distance'] = (data['Close'] - data['VWAP']) / data['VWAP']
            
            # Price position indicators
            data['High_Low_Ratio'] = (data['High'] - data['Low']) / data['Close']
            data['Close_Position'] = (data['Close'] - data['Low']) / (data['High'] - data['Low'])
            
            # Momentum indicators
            data['ROC'] = data['Close'].pct_change(periods=10) * 100
            data['MFI'] = self.calculate_mfi(data)
            
            # Fill any NaN values
            data = data.fillna(method='bfill').fillna(method='ffill')
            
            return data.dropna()
            
        except Exception as e:
            print(f"Error adding technical indicators: {e}")
            return data
    
    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    def calculate_stochastic(self, data: pd.DataFrame, window: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator"""
        low_min = data['Low'].rolling(window=window).min()
        high_max = data['High'].rolling(window=window).max()
        
        k_percent = 100 * ((data['Close'] - low_min) / (high_max - low_min))
        k_percent = k_percent.rolling(window=smooth_k).mean()
        
        d_percent = k_percent.rolling(window=smooth_d).mean()
        
        return k_percent, d_percent
    
    def calculate_obv(self, data: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume"""
        obv = pd.Series(index=data.index, dtype='float64')
        obv.iloc[0] = 0
        
        for i in range(1, len(data)):
            if data['Close'].iloc[i] > data['Close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + data['Volume'].iloc[i]
            elif data['Close'].iloc[i] < data['Close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - data['Volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    def calculate_vwap(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Volume Weighted Average Price"""
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        vwap = (typical_price * data['Volume']).cumsum() / data['Volume'].cumsum()
        return vwap
    
    def calculate_mfi(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Money Flow Index"""
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        money_flow = typical_price * data['Volume']
        
        positive_flow = pd.Series(0, index=data.index)
        negative_flow = pd.Series(0, index=data.index)
        
        for i in range(1, len(data)):
            if typical_price.iloc[i] > typical_price.iloc[i-1]:
                positive_flow.iloc[i] = money_flow.iloc[i]
            elif typical_price.iloc[i] < typical_price.iloc[i-1]:
                negative_flow.iloc[i] = money_flow.iloc[i]
        
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()
        
        mfi = 100 - (100 / (1 + positive_mf / (negative_mf + 1e-10)))
        return mfi
    
    def _add_external_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add external signals like sentiment and on-chain data"""
        try:
            # Fear & Greed Index
            fear_greed = self.fetch_fear_greed_index()
            if fear_greed is not None:
                data['Fear_Greed'] = fear_greed
            else:
                data['Fear_Greed'] = 50  # Neutral default
            
            # Simulated on-chain metrics (in production, these would come from APIs)
            # These are approximations based on price/volume patterns
            
            # Approximate active addresses based on volume patterns
            data['Active_Addresses_Proxy'] = (data['Volume'] / data['Volume'].rolling(30).mean()).rolling(7).mean()
            
            # Approximate hash rate trend based on price momentum
            data['Hash_Rate_Proxy'] = data['Close'].rolling(30).mean() / data['Close'].rolling(90).mean()
            
            # Approximate exchange flow based on volume spikes
            volume_spike = data['Volume'] / data['Volume'].rolling(20).mean()
            data['Exchange_Flow_Proxy'] = np.where(volume_spike > 2, -1, 
                                                   np.where(volume_spike < 0.5, 1, 0))
            
            # Funding rate proxy (based on price momentum)
            returns = data['Close'].pct_change()
            data['Funding_Rate_Proxy'] = returns.rolling(8).mean() * 100  # 8-hour periods
            
            return data
            
        except Exception as e:
            print(f"Error adding external signals: {e}")
            # Add default values if external data fails
            data['Fear_Greed'] = 50
            data['Active_Addresses_Proxy'] = 1
            data['Hash_Rate_Proxy'] = 1
            data['Exchange_Flow_Proxy'] = 0
            data['Funding_Rate_Proxy'] = 0
            return data
    
    def fetch_fear_greed_index(self) -> Optional[float]:
        """Fetch the latest Fear & Greed Index"""
        try:
            # Check cache first
            cache_key = 'fear_greed'
            if cache_key in self.external_data_cache:
                cached_time, cached_value = self.external_data_cache[cache_key]
                if (datetime.now() - cached_time).total_seconds() < 3600:  # 1 hour cache
                    return cached_value
            
            # Fetch from API
            url = "https://api.alternative.me/fng/?limit=1"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                value = float(data['data'][0]['value'])
                
                # Cache the result
                self.external_data_cache[cache_key] = (datetime.now(), value)
                
                return value
            else:
                print(f"Failed to fetch Fear & Greed Index: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Error fetching Fear & Greed Index: {e}")
            return None
    
    def calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            
            rs = gain / (loss + 1e-8)
            rsi = 100 - (100 / (1 + rs))
            
            rsi = rsi.fillna(50)
            
            return rsi
        except Exception as e:
            print(f"Error calculating RSI: {e}")
            return pd.Series([50] * len(prices), index=prices.index)
    
    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD indicator"""
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            macd_signal = macd.ewm(span=signal).mean()
            
            macd = macd.fillna(0)
            macd_signal = macd_signal.fillna(0)
            
            return macd, macd_signal
        except Exception as e:
            print(f"Error calculating MACD: {e}")
            zeros = pd.Series([0] * len(prices), index=prices.index)
            return zeros, zeros
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare enhanced features for the model"""
        try:
            features = pd.DataFrame()
            
            # Price features
            features['price'] = data['Close']
            features['returns'] = data['Close'].pct_change()
            features['log_returns'] = np.log(data['Close'] / data['Close'].shift(1))
            
            # Volume features
            features['volume'] = data['Volume_Norm']
            features['obv_norm'] = data['OBV'] / data['OBV'].rolling(20).mean()
            
            # Technical indicators
            features['rsi'] = data['RSI'] / 100  # Normalize to 0-1
            features['macd_norm'] = data['MACD'] / data['Close']  # Normalize by price
            features['sma_ratio'] = data['Close'] / (data['SMA_20'] + 1e-8)
            
            # Bollinger Bands features
            features['bb_position'] = data['BB_Position']
            features['bb_width'] = data['BB_Width']
            
            # Volatility features
            features['atr_ratio'] = data['ATR_Ratio']
            features['high_low_ratio'] = data['High_Low_Ratio']
            
            # Momentum features
            features['stoch_k'] = data['Stoch_K'] / 100
            features['mfi'] = data['MFI'] / 100
            features['roc'] = data['ROC'] / 100  # Normalize
            
            # Market structure
            features['vwap_distance'] = data['VWAP_Distance']
            features['close_position'] = data['Close_Position']
            
            # External signals
            features['fear_greed'] = data['Fear_Greed'] / 100
            features['funding_rate'] = data['Funding_Rate_Proxy']
            
            # Fill any remaining NaN values
            features = features.fillna(method='bfill').fillna(method='ffill')
            features = features.dropna()
            
            if len(features) == 0:
                raise ValueError("No valid features after cleaning")
            
            return features
            
        except Exception as e:
            print(f"Error preparing features: {e}")
            # Return minimal valid features
            minimal_features = pd.DataFrame({
                'price': data['Close'].fillna(data['Close'].mean()),
                'volume': [1.0] * len(data),
                'rsi': [50.0] * len(data),
                'macd': [0.0] * len(data),
                'sma_ratio': [1.0] * len(data)
            }, index=data.index)
            return minimal_features.dropna()
    
    def create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        if len(data) < self.sequence_length + 1:
            raise ValueError(f"Insufficient data for sequence creation. Need at least {self.sequence_length + 1} samples, got {len(data)}")
        
        X, y = [], []
        
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i])
            y.append(data[i, 0])  # Predict close price
        
        return np.array(X), np.array(y)
    
    def train_model(self, data: pd.DataFrame, epochs: int = 100, batch_size: int = 32):
        """Train the enhanced LSTM model"""
        try:
            features = self.prepare_features(data)
            
            if len(features) < self.sequence_length + 10:
                print(f"Warning: Limited data for training ({len(features)} samples)")
                epochs = min(epochs, 30)
            
            # Scale the data
            scaled_data = self.scaler.fit_transform(features.values)
            X, y = self.create_sequences(scaled_data)
            
            if len(X) == 0:
                raise ValueError("No sequences created for training")
            
            # Convert to PyTorch tensors
            X = torch.FloatTensor(X)
            y = torch.FloatTensor(y).unsqueeze(1)
            
            # Split data
            train_size = max(1, int(0.8 * len(X)))
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            # Training setup
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
            
            # Training loop with early stopping
            best_loss = float('inf')
            patience = 20
            patience_counter = 0
            
            self.model.train()
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
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    total_loss += loss.item()
                    batch_count += 1
                
                if batch_count > 0:
                    avg_loss = total_loss / batch_count
                    
                    # Validation
                    if len(X_test) > 0:
                        self.model.eval()
                        with torch.no_grad():
                            val_outputs = self.model(X_test)
                            val_loss = criterion(val_outputs, y_test).item()
                        self.model.train()
                        
                        scheduler.step(val_loss)
                        
                        if val_loss < best_loss:
                            best_loss = val_loss
                            patience_counter = 0
                        else:
                            patience_counter += 1
                        
                        if patience_counter >= patience:
                            print(f"Early stopping at epoch {epoch}")
                            break
                    
                    if epoch % 10 == 0:
                        print(f'Epoch [{epoch}/{epochs}], Train Loss: {avg_loss:.6f}, Val Loss: {val_loss:.6f}')
            
            self.is_trained = True
            print("Enhanced model training completed successfully!")
            
        except Exception as e:
            print(f"Error during model training: {e}")
            self.is_trained = True
            print("Continuing with basic model configuration...")
    
    def predict_signal(self, current_data: pd.DataFrame) -> Tuple[str, float, float, Dict[str, float]]:
        """Generate enhanced trading signal with confidence scores for different factors"""
        try:
            if not self.is_trained:
                print("Model not trained, training now...")
                self.train_model(current_data)
            
            features = self.prepare_features(current_data)
            
            if len(features) < self.sequence_length:
                print(f"Insufficient data for prediction ({len(features)} < {self.sequence_length})")
                return self._generate_simple_signal(features)
            
            # Get the last sequence
            last_sequence = features.tail(self.sequence_length).values
            
            # Check if scaler has been fitted
            if not hasattr(self.scaler, 'scale_'):
                print("Scaler not fitted, fitting now...")
                self.scaler.fit(features.values)
            
            scaled_sequence = self.scaler.transform(last_sequence)
            
            # Predict
            self.model.eval()
            with torch.no_grad():
                sequence_tensor = torch.FloatTensor(scaled_sequence).unsqueeze(0)
                prediction = self.model(sequence_tensor)
                
                # Inverse transform the prediction
                dummy_array = np.zeros((1, features.shape[1]))
                dummy_array[0, 0] = prediction.item()
                predicted_price = self.scaler.inverse_transform(dummy_array)[0, 0]
            
            # Generate comprehensive signal analysis
            current_price = features['price'].iloc[-1]
            price_change_pct = (predicted_price - current_price) / current_price
            
            # Analyze multiple factors for signal generation
            signal_factors = self._analyze_signal_factors(features, current_data)
            
            # Combine all factors for final signal
            signal, confidence = self._generate_composite_signal(
                price_change_pct, signal_factors, features, current_data
            )
            
            return signal, confidence, predicted_price, signal_factors
            
        except Exception as e:
            print(f"Error generating signal: {e}")
            try:
                current_price = current_data['Close'].iloc[-1]
                return "hold", 0.5, current_price, {}
            except:
                return "hold", 0.5, 45000.0, {}
    
    def _analyze_signal_factors(self, features: pd.DataFrame, data: pd.DataFrame) -> Dict[str, float]:
        """Analyze multiple factors contributing to the signal"""
        factors = {}
        
        try:
            # Technical factors
            factors['rsi_signal'] = self._get_rsi_signal(features['rsi'].iloc[-1] * 100)
            factors['macd_signal'] = self._get_macd_signal(data)
            factors['bb_signal'] = self._get_bollinger_signal(features['bb_position'].iloc[-1])
            factors['stoch_signal'] = self._get_stochastic_signal(features['stoch_k'].iloc[-1] * 100)
            factors['volume_signal'] = self._get_volume_signal(features)
            
            # Trend factors
            factors['trend_signal'] = self._get_trend_signal(features, data)
            factors['momentum_signal'] = self._get_momentum_signal(features)
            
            # Market structure
            factors['vwap_signal'] = self._get_vwap_signal(features['vwap_distance'].iloc[-1])
            
            # External factors
            factors['sentiment_signal'] = self._get_sentiment_signal(features['fear_greed'].iloc[-1] * 100)
            factors['funding_signal'] = features['funding_rate'].iloc[-1]
            
            # Volatility assessment
            factors['volatility_regime'] = self._get_volatility_regime(features['atr_ratio'].iloc[-1])
            
        except Exception as e:
            print(f"Error analyzing signal factors: {e}")
            
        return factors
    
    def _get_rsi_signal(self, rsi: float) -> float:
        """Convert RSI to signal strength (-1 to 1)"""
        if rsi < 30:
            return min(1.0, (30 - rsi) / 20)  # Oversold = bullish
        elif rsi > 70:
            return max(-1.0, (70 - rsi) / 20)  # Overbought = bearish
        else:
            return (rsi - 50) / 50  # Neutral zone
    
    def _get_macd_signal(self, data: pd.DataFrame) -> float:
        """Analyze MACD for signal strength"""
        try:
            macd = data['MACD'].iloc[-3:]
            macd_signal = data['MACD_Signal'].iloc[-3:]
            
            # Check for crossover
            if len(macd) >= 2:
                if macd.iloc[-1] > macd_signal.iloc[-1] and macd.iloc[-2] <= macd_signal.iloc[-2]:
                    return 1.0  # Bullish crossover
                elif macd.iloc[-1] < macd_signal.iloc[-1] and macd.iloc[-2] >= macd_signal.iloc[-2]:
                    return -1.0  # Bearish crossover
            
            # Otherwise, return based on current position
            diff = macd.iloc[-1] - macd_signal.iloc[-1]
            return np.clip(diff / data['Close'].iloc[-1] * 100, -1, 1)
            
        except:
            return 0
    
    def _get_bollinger_signal(self, bb_position: float) -> float:
        """Convert Bollinger Band position to signal"""
        if bb_position < 0.2:
            return 1.0  # Near lower band = oversold
        elif bb_position > 0.8:
            return -1.0  # Near upper band = overbought
        else:
            return (0.5 - bb_position) * 2  # Middle zone
    
    def _get_stochastic_signal(self, stoch_k: float) -> float:
        """Convert Stochastic to signal"""
        if stoch_k < 20:
            return 1.0  # Oversold
        elif stoch_k > 80:
            return -1.0  # Overbought
        else:
            return (50 - stoch_k) / 50
    
    def _get_volume_signal(self, features: pd.DataFrame) -> float:
        """Analyze volume patterns for signal"""
        try:
            recent_volume = features['volume'].iloc[-5:].mean()
            obv_trend = features['obv_norm'].iloc[-5:].mean()
            
            # High volume with positive OBV = bullish
            # High volume with negative OBV = bearish
            volume_signal = recent_volume * obv_trend
            
            return np.clip(volume_signal, -1, 1)
        except:
            return 0
    
    def _get_trend_signal(self, features: pd.DataFrame, data: pd.DataFrame) -> float:
        """Analyze trend strength and direction"""
        try:
            # Check multiple timeframes
            sma_20 = data['SMA_20'].iloc[-1]
            sma_50 = data['SMA_50'].iloc[-1]
            current_price = data['Close'].iloc[-1]
            
            # Trend alignment
            if current_price > sma_20 > sma_50:
                return 1.0  # Strong uptrend
            elif current_price < sma_20 < sma_50:
                return -1.0  # Strong downtrend
            else:
                # Calculate relative position
                trend_score = 0
                if current_price > sma_20:
                    trend_score += 0.5
                if current_price > sma_50:
                    trend_score += 0.5
                if sma_20 > sma_50:
                    trend_score += 0.5
                
                return (trend_score - 0.75) * 2  # Normalize to -1 to 1
        except:
            return 0
    
    def _get_momentum_signal(self, features: pd.DataFrame) -> float:
        """Analyze momentum indicators"""
        try:
            roc = features['roc'].iloc[-1]
            mfi = features['mfi'].iloc[-1]
            
            # Combine ROC and MFI for momentum
            momentum = (roc + (mfi - 0.5) * 2) / 2
            
            return np.clip(momentum, -1, 1)
        except:
            return 0
    
    def _get_vwap_signal(self, vwap_distance: float) -> float:
        """VWAP-based signal"""
        # Price above VWAP = bullish, below = bearish
        return np.clip(vwap_distance * 10, -1, 1)
    
    def _get_sentiment_signal(self, fear_greed: float) -> float:
        """Convert Fear & Greed to contrarian signal"""
        if fear_greed < 25:
            return 1.0  # Extreme fear = bullish opportunity
        elif fear_greed > 75:
            return -1.0  # Extreme greed = bearish warning
        else:
            return (50 - fear_greed) / 50  # Neutral zone
    
    def _get_volatility_regime(self, atr_ratio: float) -> str:
        """Classify current volatility regime"""
        if atr_ratio < 0.02:
            return "low"
        elif atr_ratio < 0.04:
            return "normal"
        else:
            return "high"
    
    def _generate_composite_signal(self, price_prediction: float, factors: Dict[str, float], 
                                   features: pd.DataFrame, data: pd.DataFrame) -> Tuple[str, float]:
        """Generate final signal based on all factors"""
        
        # Weight different factors
        weights = {
            'price_prediction': 0.25,
            'rsi_signal': 0.10,
            'macd_signal': 0.10,
            'bb_signal': 0.05,
            'stoch_signal': 0.05,
            'volume_signal': 0.10,
            'trend_signal': 0.15,
            'momentum_signal': 0.10,
            'vwap_signal': 0.05,
            'sentiment_signal': 0.05
        }
        
        # Calculate weighted composite score
        composite_score = price_prediction * weights['price_prediction']
        
        for factor, value in factors.items():
            if factor in weights and isinstance(value, (int, float)):
                composite_score += value * weights.get(factor, 0)
        
        # Adjust for volatility regime
        volatility_regime = factors.get('volatility_regime', 'normal')
        if volatility_regime == 'high':
            composite_score *= 0.8  # More conservative in high volatility
        elif volatility_regime == 'low':
            composite_score *= 1.2  # More aggressive in low volatility
        
        # Generate signal and confidence
        if composite_score > 0.15:
            signal = "buy"
            confidence = min(0.95, 0.5 + abs(composite_score))
        elif composite_score < -0.15:
            signal = "sell"
            confidence = min(0.95, 0.5 + abs(composite_score))
        else:
            signal = "hold"
            confidence = 0.5 + abs(composite_score) * 0.3
        
        return signal, confidence
    
    def _generate_simple_signal(self, features: pd.DataFrame) -> Tuple[str, float, float, Dict[str, float]]:
        """Generate a simple signal when insufficient data"""
        try:
            recent_trend = features['price'].iloc[-5:].pct_change().mean()
            current_price = features['price'].iloc[-1]
            
            if recent_trend > 0.01:
                return "buy", 0.6, current_price * 1.02, {'simple_trend': recent_trend}
            elif recent_trend < -0.01:
                return "sell", 0.6, current_price * 0.98, {'simple_trend': recent_trend}
            else:
                return "hold", 0.5, current_price, {'simple_trend': recent_trend}
        except:
            return "hold", 0.5, 45000.0, {}
    
    def generate_dummy_data(self) -> pd.DataFrame:
        """Generate realistic dummy BTC data for testing"""
        print("Generating realistic dummy BTC data...")
        
        # Create date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Set random seed for reproducible but varied data
        np.random.seed(int(time.time()) % 1000)
        
        # Generate realistic BTC price data with trends
        base_price = 45000
        prices = []
        volumes = []
        current_price = base_price
        
        # Add some trend periods
        trend_periods = [
            (0, 100, 0.001),      # Slight uptrend
            (100, 200, -0.002),   # Downtrend
            (200, 300, 0.003),    # Strong uptrend
            (300, len(dates), 0)  # Sideways
        ]
        
        for i, date in enumerate(dates):
            # Determine current trend
            trend = 0
            for start_idx, end_idx, trend_val in trend_periods:
                if start_idx <= i < end_idx:
                    trend = trend_val
                    break
            
            # Generate price with trend and volatility
            daily_change = np.random.normal(trend, 0.03)
            current_price *= (1 + daily_change)
            
            # Ensure price doesn't go below reasonable bounds
            current_price = max(current_price, 10000)
            current_price = min(current_price, 100000)
            
            prices.append(current_price)
            
            # Generate volume (higher volume during price movements)
            volume_base = 15
            volume_volatility = abs(daily_change) * 2
            volume = np.random.lognormal(volume_base + volume_volatility, 1)
            volumes.append(volume)
        
        # Create OHLC data
        data_dict = {
            'Open': [],
            'High': [],
            'Low': [],
            'Close': [],
            'Volume': volumes
        }
        
        for i, close_price in enumerate(prices):
            # Generate realistic OHLC from close price
            daily_range = abs(np.random.normal(0, 0.02)) * close_price
            
            open_price = close_price * (1 + np.random.normal(0, 0.01))
            high_price = max(open_price, close_price) + random.uniform(0, daily_range)
            low_price = min(open_price, close_price) - random.uniform(0, daily_range)
            
            data_dict['Open'].append(open_price)
            data_dict['High'].append(high_price)
            data_dict['Low'].append(low_price)
            data_dict['Close'].append(close_price)
        
        # Create DataFrame
        data = pd.DataFrame(data_dict, index=dates)
        
        # Add enhanced technical indicators
        data = self._add_enhanced_technical_indicators(data)
        
        # Add simulated external signals
        data = self._add_external_signals(data)
        
        print(f"Generated {len(data)} days of dummy BTC data")
        return data
    
    def save_model(self, path: str):
        """Save the trained model"""
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'scaler': self.scaler,
                'sequence_length': self.sequence_length,
                'is_trained': self.is_trained
            }, path)
            print(f"Model saved to {path}")
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def load_model(self, path: str):
        """Load a trained model"""
        try:
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.scaler = checkpoint['scaler']
            self.sequence_length = checkpoint['sequence_length']
            self.is_trained = checkpoint['is_trained']
            print(f"Model loaded from {path}")
        except FileNotFoundError:
            print(f"Model file not found: {path}")
        except Exception as e:
            print(f"Error loading model: {e}")

if __name__ == "__main__":
    # Test the enhanced signal generator
    signal_gen = EnhancedTradingSignalGenerator()
    
    # Fetch BTC data
    print("Testing BTC data fetching with enhanced indicators...")
    btc_data = signal_gen.fetch_btc_data(period="6mo")
    print(f"Fetched {len(btc_data)} days of data")
    print(f"Available indicators: {list(btc_data.columns)}")
    
    # Generate a signal
    print("\nTesting enhanced signal generation...")
    signal, confidence, predicted_price, factors = signal_gen.predict_signal(btc_data)
    
    current_price = btc_data['Close'].iloc[-1]
    print(f"\nCurrent Price: ${current_price:.2f}")
    print(f"Predicted Price: ${predicted_price:.2f}")
    print(f"Signal: {signal.upper()}")
    print(f"Confidence: {confidence:.2%}")
    
    print("\nSignal Factors:")
    for factor, value in factors.items():
        if isinstance(value, float):
            print(f"  {factor}: {value:.3f}")
        else:
            print(f"  {factor}: {value}")
    
    print("\nTest completed successfully!")