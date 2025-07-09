"""
Enhanced LSTM Model for Returns Prediction
This model predicts price changes (returns) instead of absolute prices,
making it more robust to changing price ranges.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import logging
from typing import Dict, List, Tuple, Optional, Any
import json
import os
from datetime import datetime
import joblib

logger = logging.getLogger(__name__)

class ReturnsDataset(Dataset):
    """Dataset for returns-based predictions"""
    
    def __init__(self, features: np.ndarray, returns: np.ndarray, sequence_length: int = 60):
        self.features = features
        self.returns = returns
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.features) - self.sequence_length
    
    def __getitem__(self, idx):
        # Get sequence of features
        X = self.features[idx:idx + self.sequence_length]
        # Get next return (not price)
        y = self.returns[idx + self.sequence_length]
        
        return torch.FloatTensor(X), torch.FloatTensor([y])

class ReturnsLSTMTrainer:
    """LSTM trainer that predicts returns instead of prices"""
    
    def __init__(self, model_dir: str = '/app/models', device: str = None):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        logger.info(f"Using device: {self.device}")
        
        # Model components
        self.model = None
        self.feature_scaler = None
        self.returns_scaler = None
        self.feature_names = None
        self.config = None
        
    def prepare_data(self, 
                    df: pd.DataFrame,
                    feature_cols: List[str],
                    target_col: str = 'Close',
                    sequence_length: int = 60,
                    train_split: float = 0.7,
                    val_split: float = 0.15,
                    return_window: int = 1) -> Dict:
        """
        Prepare data for returns prediction
        """
        logger.info(f"Preparing data for returns prediction")
        
        # Store configuration
        self.feature_names = feature_cols
        self.config = {
            'sequence_length': sequence_length,
            'target_col': target_col,
            'feature_names': feature_cols,
            'return_window': return_window
        }
        
        # Calculate returns
        prices = df[target_col].values
        returns = np.diff(prices) / prices[:-1]  # Simple returns
        
        # Add returns as a feature
        df_with_returns = df.copy()
        df_with_returns['returns'] = np.concatenate([[0], returns])  # Pad first value
        
        # Extract features
        features = df_with_returns[feature_cols].values
        
        # Handle missing values
        features = pd.DataFrame(features, columns=feature_cols).fillna(method='ffill').fillna(0).values
        
        # Scale features (but not returns target)
        self.feature_scaler = StandardScaler()
        features_scaled = self.feature_scaler.fit_transform(features)
        
        # Use returns directly without scaling (they're already normalized by nature)
        target_returns = df_with_returns['returns'].values
        
        # Create sequences
        X, y = self._create_sequences(features_scaled, target_returns, sequence_length)
        
        # Temporal split
        n_samples = len(X)
        train_size = int(n_samples * train_split)
        val_size = int(n_samples * val_split)
        
        train_end = train_size
        val_end = train_end + val_size
        
        # Create datasets
        data_splits = {
            'train': {
                'X': X[:train_end],
                'y': y[:train_end],
                'dataset': ReturnsDataset(features_scaled[:train_end + sequence_length], 
                                        target_returns[:train_end + sequence_length], 
                                        sequence_length)
            },
            'val': {
                'X': X[train_end:val_end],
                'y': y[train_end:val_end],
                'dataset': ReturnsDataset(features_scaled[train_end:val_end + sequence_length], 
                                        target_returns[train_end:val_end + sequence_length], 
                                        sequence_length)
            },
            'test': {
                'X': X[val_end:],
                'y': y[val_end:],
                'dataset': ReturnsDataset(features_scaled[val_end:], 
                                        target_returns[val_end:], 
                                        sequence_length)
            }
        }
        
        logger.info(f"Data split - Train: {len(data_splits['train']['X'])}, "
                   f"Val: {len(data_splits['val']['X'])}, "
                   f"Test: {len(data_splits['test']['X'])}")
        
        return data_splits
    
    def _create_sequences(self, features: np.ndarray, returns: np.ndarray, 
                         sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM input"""
        X, y = [], []
        
        for i in range(len(features) - sequence_length):
            X.append(features[i:i + sequence_length])
            y.append(returns[i + sequence_length])
            
        return np.array(X), np.array(y)
    
    def predict_returns(self, features: np.ndarray, current_price: float) -> Dict:
        """
        Predict returns and convert to price prediction
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
            
        if self.feature_scaler is None:
            raise ValueError("Feature scaler not found")
            
        self.model.eval()
        
        # Scale features
        features_scaled = self.feature_scaler.transform(features)
        
        # Convert to tensor
        X = torch.FloatTensor(features_scaled).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Predict return
            predicted_return = self.model(X).cpu().numpy()[0][0]
        
        # Convert return to price
        predicted_price = current_price * (1 + predicted_return)
        
        # Cap extreme predictions
        max_daily_change = 0.10  # 10% max daily change
        predicted_return = np.clip(predicted_return, -max_daily_change, max_daily_change)
        predicted_price_capped = current_price * (1 + predicted_return)
        
        return {
            'predicted_return': predicted_return,
            'predicted_price': predicted_price_capped,
            'confidence': self._calculate_confidence(predicted_return),
            'current_price': current_price
        }
    
    def _calculate_confidence(self, predicted_return: float) -> float:
        """Calculate confidence based on prediction magnitude"""
        # Small changes are more confident
        # Large changes are less confident
        magnitude = abs(predicted_return)
        
        if magnitude < 0.01:  # Less than 1% change
            confidence = 0.8
        elif magnitude < 0.03:  # 1-3% change
            confidence = 0.7
        elif magnitude < 0.05:  # 3-5% change
            confidence = 0.6
        else:  # Greater than 5% change
            confidence = 0.5
            
        return confidence