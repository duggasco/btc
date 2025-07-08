"""
Enhanced LSTM Model Implementation following whitepaper best practices
for BTC/USD price forecasting with proper architecture and training procedures
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import logging
from typing import Dict, List, Tuple, Optional, Any
import json
import os
from datetime import datetime
import joblib

logger = logging.getLogger(__name__)

class BTCDataset(Dataset):
    """Custom dataset for BTC time series following whitepaper recommendations"""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray, sequence_length: int = 60):
        self.features = features
        self.labels = labels
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.features) - self.sequence_length
    
    def __getitem__(self, idx):
        # Get sequence of features
        X = self.features[idx:idx + self.sequence_length]
        # Get label (next value after sequence)
        y = self.labels[idx + self.sequence_length]
        
        return torch.FloatTensor(X), torch.FloatTensor([y])

class AttentionLayer(nn.Module):
    """Attention mechanism as mentioned in whitepaper for LSTM enhancement"""
    
    def __init__(self, hidden_size: int):
        super(AttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(hidden_size, 1)
        
    def forward(self, lstm_output):
        # lstm_output shape: (batch, seq_len, hidden_size)
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        # Weighted sum of LSTM outputs
        context = torch.sum(attention_weights * lstm_output, dim=1)
        return context, attention_weights

class EnhancedLSTM(nn.Module):
    """
    Enhanced LSTM architecture following whitepaper best practices:
    - Multiple LSTM layers with dropout
    - Attention mechanism
    - Proper initialization
    - Batch normalization option
    """
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int = 100,
                 num_layers: int = 2,
                 dropout: float = 0.2,
                 use_attention: bool = True,
                 use_batch_norm: bool = True):
        super(EnhancedLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_attention = use_attention
        
        # Batch normalization for input
        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(input_size)
        else:
            self.batch_norm = None
            
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        
        # Attention layer
        if use_attention:
            self.attention = AttentionLayer(hidden_size)
            
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size // 2, 1)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using Xavier/He initialization"""
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
                
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        batch_size = x.size(0)
        
        # Apply batch normalization if enabled
        if self.batch_norm is not None:
            # Reshape for batch norm: (batch * seq_len, features)
            x_reshaped = x.view(-1, self.input_size)
            x_normed = self.batch_norm(x_reshaped)
            x = x_normed.view(batch_size, -1, self.input_size)
            
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply attention or use last hidden state
        if self.use_attention:
            context, _ = self.attention(lstm_out)
            out = context
        else:
            out = lstm_out[:, -1, :]
            
        # Final layers
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

class LSTMTrainer:
    """
    LSTM training class following whitepaper best practices:
    - Proper train/val/test splits
    - Early stopping
    - Learning rate scheduling
    - Model checkpointing
    """
    
    def __init__(self, 
                 model_dir: str = '/app/models',
                 device: str = None):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        logger.info(f"Using device: {self.device}")
        
        # Model components
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.config = None
        
    def prepare_data(self, 
                    df: pd.DataFrame,
                    feature_cols: List[str],
                    target_col: str = 'Close',
                    sequence_length: int = 60,
                    train_split: float = 0.7,
                    val_split: float = 0.15,
                    scaling_method: str = 'minmax') -> Dict:
        """
        Prepare data following whitepaper recommendations:
        - Proper temporal splits
        - Normalization
        - Sequence formation
        """
        logger.info(f"Preparing data with {len(feature_cols)} features")
        
        # Store configuration
        self.feature_names = feature_cols
        self.config = {
            'sequence_length': sequence_length,
            'target_col': target_col,
            'scaling_method': scaling_method,
            'feature_names': feature_cols
        }
        
        # Check data sufficiency
        min_required = sequence_length + 100  # Need at least 100 samples after sequences
        if len(df) < min_required:
            raise ValueError(f"Insufficient data: {len(df)} rows, need at least {min_required}")
            
        # Extract features and target
        features = df[feature_cols].values
        target = df[target_col].values
        
        # Handle missing values
        features = pd.DataFrame(features, columns=feature_cols).fillna(method='ffill').fillna(0).values
        
        # Scale features
        if scaling_method == 'minmax':
            self.scaler = MinMaxScaler(feature_range=(0, 1))
        else:
            self.scaler = StandardScaler()
            
        features_scaled = self.scaler.fit_transform(features)
        
        # Scale target separately to preserve scale for inverse transform
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        target_scaled = self.target_scaler.fit_transform(target.reshape(-1, 1)).flatten()
        
        # Create sequences
        X, y = self._create_sequences(features_scaled, target_scaled, sequence_length)
        
        # Temporal split (no shuffling!)
        n_samples = len(X)
        train_size = int(n_samples * train_split)
        val_size = int(n_samples * val_split)
        
        # Split indices
        train_end = train_size
        val_end = train_end + val_size
        
        # Create datasets
        data_splits = {
            'train': {
                'X': X[:train_end],
                'y': y[:train_end],
                'dataset': BTCDataset(features_scaled[:train_end + sequence_length], 
                                    target_scaled[:train_end + sequence_length], 
                                    sequence_length)
            },
            'val': {
                'X': X[train_end:val_end],
                'y': y[train_end:val_end],
                'dataset': BTCDataset(features_scaled[train_end:val_end + sequence_length], 
                                    target_scaled[train_end:val_end + sequence_length], 
                                    sequence_length)
            },
            'test': {
                'X': X[val_end:],
                'y': y[val_end:],
                'dataset': BTCDataset(features_scaled[val_end:], 
                                    target_scaled[val_end:], 
                                    sequence_length)
            }
        }
        
        logger.info(f"Data split - Train: {len(data_splits['train']['X'])}, "
                   f"Val: {len(data_splits['val']['X'])}, "
                   f"Test: {len(data_splits['test']['X'])}")
        
        return data_splits
    
    def _create_sequences(self, features: np.ndarray, target: np.ndarray, 
                         sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM input"""
        X, y = [], []
        
        for i in range(len(features) - sequence_length):
            X.append(features[i:i + sequence_length])
            y.append(target[i + sequence_length])
            
        return np.array(X), np.array(y)
    
    def train(self,
             data_splits: Dict,
             input_size: int,
             hidden_size: int = 100,
             num_layers: int = 2,
             dropout: float = 0.2,
             learning_rate: float = 0.001,
             batch_size: int = 32,
             epochs: int = 100,
             patience: int = 10,
             use_attention: bool = True) -> Dict:
        """
        Train LSTM model following whitepaper best practices
        """
        logger.info("Starting LSTM training")
        
        # Create model
        self.model = EnhancedLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            use_attention=use_attention
        ).to(self.device)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5, verbose=True
        )
        
        # Data loaders
        train_loader = DataLoader(
            data_splits['train']['dataset'],
            batch_size=batch_size,
            shuffle=False  # Don't shuffle time series!
        )
        
        val_loader = DataLoader(
            data_splits['val']['dataset'],
            batch_size=batch_size,
            shuffle=False
        )
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': []
        }
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
                
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    
            avg_val_loss = val_loss / len(val_loader)
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            # Record history
            training_history['train_loss'].append(avg_train_loss)
            training_history['val_loss'].append(avg_val_loss)
            training_history['learning_rates'].append(optimizer.param_groups[0]['lr'])
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                
                # Save best model
                self._save_model(epoch, avg_val_loss)
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
                
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, "
                           f"Val Loss = {avg_val_loss:.4f}")
                
        # Load best model
        self._load_best_model()
        
        # Evaluate on test set
        test_metrics = self._evaluate_test_set(data_splits['test']['dataset'])
        
        return {
            'training_history': training_history,
            'test_metrics': test_metrics,
            'best_epoch': epoch - patience,
            'final_val_loss': best_val_loss
        }
    
    def _save_model(self, epoch: int, val_loss: float):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'val_loss': val_loss,
            'config': self.config,
            'feature_names': self.feature_names
        }
        
        path = os.path.join(self.model_dir, 'best_lstm_model.pth')
        torch.save(checkpoint, path)
        
        # Save scalers
        joblib.dump(self.scaler, os.path.join(self.model_dir, 'feature_scaler.pkl'))
        joblib.dump(self.target_scaler, os.path.join(self.model_dir, 'target_scaler.pkl'))
        
    def _load_best_model(self):
        """Load best model from checkpoint"""
        path = os.path.join(self.model_dir, 'best_lstm_model.pth')
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded best model from epoch {checkpoint['epoch']}")
            
    def _evaluate_test_set(self, test_dataset: Dataset) -> Dict:
        """Evaluate model on test set"""
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        self.model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                
                predictions.extend(outputs.cpu().numpy())
                actuals.extend(batch_y.numpy())
                
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Calculate metrics
        mse = np.mean((predictions - actuals) ** 2)
        mae = np.mean(np.abs(predictions - actuals))
        
        # Directional accuracy
        if len(predictions) > 1:
            pred_direction = np.diff(predictions.flatten()) > 0
            actual_direction = np.diff(actuals.flatten()) > 0
            directional_accuracy = np.mean(pred_direction == actual_direction)
        else:
            directional_accuracy = 0
            
        return {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(np.sqrt(mse)),
            'directional_accuracy': float(directional_accuracy)
        }
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Make predictions with trained model"""
        if self.model is None:
            raise ValueError("Model not trained yet")
            
        self.model.eval()
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Convert to tensor
        X = torch.FloatTensor(features_scaled).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            prediction_scaled = self.model(X).cpu().numpy()
            
        # Inverse transform
        prediction = self.target_scaler.inverse_transform(prediction_scaled)
        
        return prediction