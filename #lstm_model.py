import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List
import yfinance as yf
from datetime import datetime, timedelta

class LSTMTradingModel(nn.Module):
    def __init__(self, input_size: int = 5, hidden_size: int = 50, num_layers: int = 2, output_size: int = 1, dropout: float = 0.2):
        super(LSTMTradingModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        
        return out

class TradingSignalGenerator:
    def __init__(self, model_path: str = None, sequence_length: int = 60):
        self.sequence_length = sequence_length
        self.model = LSTMTradingModel()
        self.scaler = MinMaxScaler()
        self.is_trained = False
        
        if model_path:
            self.load_model(model_path)
    
    def fetch_btc_data(self, period: str = "1y") -> pd.DataFrame:
        """Fetch BTC data from Yahoo Finance"""
        try:
            btc = yf.Ticker("BTC-USD")
            data = btc.history(period=period)
            
            data['SMA_20'] = data['Close'].rolling(window=20).mean()
            data['SMA_50'] = data['Close'].rolling(window=50).mean()
            data['RSI'] = self.calculate_rsi(data['Close'])
            data['MACD'], data['MACD_Signal'] = self.calculate_macd(data['Close'])
            data['Volume_Norm'] = data['Volume'] / data['Volume'].rolling(window=20).mean()
            
            return data.dropna()
        except Exception as e:
            print(f"Error fetching BTC data: {e}")
            return self.generate_dummy_data()
    
    def generate_dummy_data(self) -> pd.DataFrame:
        """Generate dummy BTC data for testing"""
        dates = pd.date_range(start=datetime.now() - timedelta(days=365), end=datetime.now(), freq='D')
        
        np.random.seed(42)
        base_price = 45000
        price_changes = np.random.normal(0, 0.03, len(dates))
        prices = []
        current_price = base_price
        
        for change in price_changes:
            current_price *= (1 + change)
            prices.append(current_price)
        
        volumes = np.random.lognormal(15, 1, len(dates))
        
        data = pd.DataFrame({
            'Open': prices,
            'High': [p * (1 + abs(np.random.normal(0, 0.02))) for p in prices],
            'Low': [p * (1 - abs(np.random.normal(0, 0.02))) for p in prices],
            'Close': prices,
            'Volume': volumes
        }, index=dates)
        
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['RSI'] = self.calculate_rsi(data['Close'])
        data['MACD'], data['MACD_Signal'] = self.calculate_macd(data['Close'])
        data['Volume_Norm'] = data['Volume'] / data['Volume'].rolling(window=20).mean()
        
        return data.dropna()
    
    def calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        return macd, macd_signal
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for the model"""
        features = pd.DataFrame()
        features['price'] = data['Close']
        features['volume'] = data['Volume_Norm']
        features['rsi'] = data['RSI']
        features['macd'] = data['MACD']
        features['sma_ratio'] = data['Close'] / data['SMA_20']
        
        return features.dropna()
    
    def create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        X, y = [], []
        
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i])
            y.append(data[i, 0])
        
        return np.array(X), np.array(y)
    
    def train_model(self, data: pd.DataFrame, epochs: int = 100, batch_size: int = 32):
        """Train the LSTM model"""
        features = self.prepare_features(data)
        scaled_data = self.scaler.fit_transform(features.values)
        X, y = self.create_sequences(scaled_data)
        
        X = torch.FloatTensor(X)
        y = torch.FloatTensor(y).unsqueeze(1)
        
        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        self.model.train()
        for epoch in range(epochs):
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            
            if epoch % 20 == 0:
                print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')
        
        self.is_trained = True
        print("Model training completed!")
    
    def predict_signal(self, current_data: pd.DataFrame) -> Tuple[str, float, float]:
        """Generate trading signal based on current data"""
        if not self.is_trained:
            self.train_model(current_data)
        
        features = self.prepare_features(current_data)
        
        if len(features) < self.sequence_length:
            return "hold", 0.5, features['price'].iloc[-1]
        
        last_sequence = features.tail(self.sequence_length).values
        scaled_sequence = self.scaler.transform(last_sequence)
        
        self.model.eval()
        with torch.no_grad():
            sequence_tensor = torch.FloatTensor(scaled_sequence).unsqueeze(0)
            prediction = self.model(sequence_tensor)
            
            dummy_array = np.zeros((1, features.shape[1]))
            dummy_array[0, 0] = prediction.item()
            predicted_price = self.scaler.inverse_transform(dummy_array)[0, 0]
        
        current_price = features['price'].iloc[-1]
        price_change_pct = (predicted_price - current_price) / current_price
        
        if price_change_pct > 0.02:
            signal = "buy"
            confidence = min(abs(price_change_pct) * 10, 0.95)
        elif price_change_pct < -0.02:
            signal = "sell"
            confidence = min(abs(price_change_pct) * 10, 0.95)
        else:
            signal = "hold"
            confidence = 0.6
        
        return signal, confidence, predicted_price
    
    def save_model(self, path: str):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'sequence_length': self.sequence_length,
            'is_trained': self.is_trained
        }, path)
    
    def load_model(self, path: str):
        """Load a trained model"""
        try:
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.scaler = checkpoint['scaler']
            self.sequence_length = checkpoint['sequence_length']
            self.is_trained = checkpoint['is_trained']
        except FileNotFoundError:
            print(f"Model file not found: {path}")
        except Exception as e:
            print(f"Error loading model: {e}")
