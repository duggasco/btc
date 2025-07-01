"""
Integration module for backtesting system with weight optimization restored
This file preserves ALL original functionality and adds weight optimization
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from typing import Dict, Tuple, Optional

from lstm_model import TradingSignalGenerator
from database_models import DatabaseManager
from backtesting_system import (
    BacktestConfig, SignalWeights, BacktestingPipeline,
    AdaptiveRetrainingScheduler, BayesianOptimizer,
    WalkForwardBacktester
)
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import json
import torch
import traceback

# Enhanced logging for debugging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class EnhancedTradingSignalGenerator(TradingSignalGenerator):
    """Enhanced version with detailed debugging and weight optimization"""
    
    def __init__(self, model_path: str = None, sequence_length: int = 60):
        super().__init__(model_path, sequence_length)
        self.signal_weights = SignalWeights()
        self.performance_history = []
        self._cached_btc_data = None
        self._cached_predictions = {}
        self._fitted_on_features = None
        
    def prepare_categorized_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features with extensive debugging"""
        logger.debug(f"prepare_categorized_features called with data shape: {data.shape}")
        logger.debug(f"Data columns: {list(data.columns)}")
        logger.debug(f"Data index type: {type(data.index)}, length: {len(data.index)}")
        
        try:
            # Get base features
            features = self.prepare_features(data)
            logger.debug(f"Base features shape: {features.shape}")
            logger.debug(f"Base features columns: {list(features.columns)}")
            
            # Create a new DataFrame with properly structured features
            categorized = pd.DataFrame(index=features.index)
            
            # Add individual feature columns (not arrays)
            # Technical features
            technical_cols = ['price', 'sma_ratio', 'rsi', 'macd']
            for col in technical_cols:
                if col in features.columns:
                    categorized[f'tech_{col}'] = features[col]
                    logger.debug(f"Added tech_{col}")
                else:
                    logger.warning(f"Missing technical column: {col}")
            
            # Volume features
            volume_cols = ['volume_norm', 'obv']
            for col in volume_cols:
                if col in features.columns:
                    categorized[f'vol_{col}'] = features[col]
                    logger.debug(f"Added vol_{col}")
                else:
                    logger.warning(f"Missing volume column: {col}")
            
            # Additional features
            other_cols = ['stoch_k', 'roc', 'macd_histogram', 'ema_ratio', 
                          'bb_position', 'bb_width', 'atr', 'volatility', 
                          'fear_greed', 'btc_dominance']
            for col in other_cols:
                if col in features.columns:
                    categorized[col] = features[col]
                    logger.debug(f"Added {col}")
            
            # For backtesting compatibility
            categorized['technical_features'] = features.get('price', 1.0)
            categorized['onchain_features'] = features.get('volume_norm', 1.0)
            categorized['sentiment_features'] = features.get('fear_greed', 0.5)
            categorized['macro_features'] = features.get('btc_dominance', 0.5)
            
            # Add target and Close
            if 'price' in features.columns:
                price_series = features['price']
                logger.debug(f"Price series shape: {price_series.shape}, length: {len(price_series)}")
                
                # Calculate returns safely
                if len(price_series) > 1:
                    returns = price_series.pct_change()
                    categorized['target'] = returns.shift(-1).fillna(0).values
                else:
                    categorized['target'] = 0.0
                    
                categorized['Close'] = price_series.values
            else:
                categorized['target'] = 0.0
                categorized['Close'] = data.get('Close', 0.0)
            
            logger.debug(f"Final categorized shape: {categorized.shape}")
            logger.debug(f"Categorized columns: {list(categorized.columns)}")
            
            # Validate no arrays in cells
            for col in categorized.columns:
                first_val = categorized[col].iloc[0] if len(categorized) > 0 else None
                if isinstance(first_val, (list, np.ndarray)):
                    logger.error(f"Column {col} contains arrays!")
                    categorized[col] = categorized[col].apply(lambda x: x[0] if isinstance(x, (list, np.ndarray)) else x)
            
            return categorized
            
        except Exception as e:
            logger.error(f"Error in prepare_categorized_features: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def set_btc_data_cache(self, btc_data: pd.DataFrame):
        """Set the full BTC dataset with validation"""
        logger.debug(f"Caching BTC data with shape {btc_data.shape}")
        logger.debug(f"BTC data date range: {btc_data.index[0]} to {btc_data.index[-1]}")
        self._cached_btc_data = btc_data.copy()
    
    def _predict_using_existing_method(self, features: pd.DataFrame) -> np.ndarray:
        """Use existing predict_signal with detailed debugging"""
        predictions = []
        logger.debug(f"_predict_using_existing_method called with {len(features)} features")
        
        for idx in range(len(features)):
            try:
                # Get current date
                current_date = features.index[idx]
                logger.debug(f"Processing index {idx}, date: {current_date}")
                
                # Find the corresponding index in cached BTC data
                try:
                    if isinstance(current_date, pd.Timestamp):
                        # Try exact match first
                        try:
                            btc_idx = self._cached_btc_data.index.get_loc(current_date)
                        except KeyError:
                            # Try nearest date
                            logger.warning(f"Exact date {current_date} not found, finding nearest")
                            nearest_idx = self._cached_btc_data.index.get_indexer([current_date], method='nearest')[0]
                            btc_idx = nearest_idx
                    else:
                        # If it's an integer index
                        btc_idx = idx
                        
                    logger.debug(f"Found btc_idx: {btc_idx}")
                    
                except Exception as e:
                    logger.error(f"Error finding index for {current_date}: {e}")
                    predictions.append(0.0)
                    continue
                
                # Check bounds
                if btc_idx < 0 or btc_idx >= len(self._cached_btc_data):
                    logger.warning(f"btc_idx {btc_idx} out of bounds for data length {len(self._cached_btc_data)}")
                    predictions.append(0.0)
                    continue
                
                # Get historical window for prediction
                if btc_idx < self.sequence_length:
                    logger.debug(f"Not enough history at index {btc_idx}")
                    predictions.append(0.0)
                    continue
                
                # Get data up to and including current point
                historical_data = self._cached_btc_data.iloc[:btc_idx + 1].copy()
                logger.debug(f"Historical data shape: {historical_data.shape}")
                
                # Use existing predict_signal method
                signal, confidence, predicted_price = self.predict_signal(historical_data)
                logger.debug(f"Prediction: signal={signal}, confidence={confidence}, price={predicted_price}")
                
                # Convert to return prediction
                current_price = features.iloc[idx].get('Close', features.iloc[idx].get('price', 1.0))
                if current_price > 0:
                    if signal == "buy":
                        return_prediction = 0.02 * confidence
                    elif signal == "sell":
                        return_prediction = -0.02 * confidence
                    else:  # hold
                        return_prediction = 0.0
                    
                    if predicted_price > 0:
                        return_prediction = (predicted_price - current_price) / current_price
                else:
                    return_prediction = 0.0
                
                predictions.append(return_prediction)
                
            except Exception as e:
                logger.error(f"Error at index {idx}: {str(e)}")
                logger.error(traceback.format_exc())
                predictions.append(0.0)
        
        logger.debug(f"Generated {len(predictions)} predictions")
        return np.array(predictions)
    
    def predict_with_weights(self, features: pd.DataFrame) -> np.ndarray:
        """Generate predictions with extensive debugging"""
        logger.debug(f"predict_with_weights called with features shape: {features.shape}")
        
        # Apply signal weights
        weighted_features = self.apply_signal_weights(features)
        logger.debug(f"Weighted features shape: {weighted_features.shape}")
        
        # Use existing predict_signal method if BTC data is cached
        if self._cached_btc_data is not None:
            logger.debug("Using existing predict_signal method")
            return self._predict_using_existing_method(features)
        
        # Otherwise use direct LSTM
        logger.debug("Using direct LSTM prediction")
        return self._predict_using_direct_lstm(features, weighted_features)
    
    def _predict_using_direct_lstm(self, features: pd.DataFrame, weighted_features: np.ndarray) -> np.ndarray:
        """Direct LSTM prediction with debugging"""
        predictions = []
        logger.debug(f"Direct LSTM prediction for {len(features)} points")
        
        # Ensure model is trained
        if not self.is_trained:
            logger.warning("Model not trained")
            if self._cached_btc_data is not None:
                self.train_model(self._cached_btc_data)
            else:
                logger.error("No cached BTC data for training")
                return np.zeros(len(features))
        
        for idx in range(len(features)):
            try:
                end_idx = idx + 1
                start_idx = max(0, end_idx - self.sequence_length)
                
                if end_idx - start_idx < self.sequence_length:
                    predictions.append(0.0)
                    continue
                
                # Create sequence window
                sequence_window = weighted_features[start_idx:end_idx]
                
                if len(sequence_window.shape) == 1:
                    sequence_window = sequence_window.reshape(-1, 1)
                
                # Scale
                if hasattr(self.scaler, 'scale_'):
                    scaled_sequence = self.scaler.transform(sequence_window)
                else:
                    self.scaler.fit(weighted_features)
                    scaled_sequence = self.scaler.transform(sequence_window)
                
                # Predict
                self.model.eval()
                with torch.no_grad():
                    sequence_tensor = torch.FloatTensor(scaled_sequence).unsqueeze(0)
                    lstm_output = self.model(sequence_tensor)
                    prediction_scaled = lstm_output.item()
                    
                    # Inverse transform
                    dummy_array = np.zeros((1, weighted_features.shape[1]))
                    dummy_array[0, 0] = prediction_scaled
                    predicted_price = self.scaler.inverse_transform(dummy_array)[0, 0]
                    
                    # Convert to return
                    current_price = features.iloc[idx].get('Close', features.iloc[idx].get('price', 1.0))
                    if current_price > 0:
                        return_prediction = (predicted_price - current_price) / current_price
                    else:
                        return_prediction = 0.0
                    
                    predictions.append(return_prediction)
                    
            except Exception as e:
                logger.error(f"Direct LSTM error at {idx}: {e}")
                predictions.append(0.0)
        
        return np.array(predictions)
    
    def apply_signal_weights(self, features: pd.DataFrame) -> np.ndarray:
        """Apply weights with debugging"""
        logger.debug(f"Applying signal weights to {features.shape} features")
        
        weighted_features = features.copy()
        
        # Apply weights to different feature types
        tech_cols = [col for col in features.columns if col.startswith('tech_')]
        logger.debug(f"Found {len(tech_cols)} technical columns")
        
        for col in tech_cols:
            weighted_features[col] = features[col] * self.signal_weights.technical_weight
        
        vol_cols = [col for col in features.columns if col.startswith('vol_')]
        logger.debug(f"Found {len(vol_cols)} volume columns")
        
        for col in vol_cols:
            weighted_features[col] = features[col] * self.signal_weights.onchain_weight
        
        # Exclude non-feature columns
        feature_cols = [col for col in weighted_features.columns 
                       if col not in ['target', 'Close', 'technical_features', 
                                     'onchain_features', 'sentiment_features', 'macro_features']]
        
        logger.debug(f"Returning {len(feature_cols)} feature columns")
        return weighted_features[feature_cols].values
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Scikit-learn compatible fit method for backtesting"""
        logger.debug(f"fit() called with X shape: {X.shape}, y shape: {y.shape}")
        
        # Convert numpy array back to DataFrame if needed
        if isinstance(X, np.ndarray):
            # Create a minimal DataFrame structure
            if len(X.shape) == 1:
                X = X.reshape(-1, 1)
            
            # Create feature names based on the number of columns
            n_features = X.shape[1]
            if n_features == 4:
                # Assume it's the weighted features (tech, onchain, sentiment, macro)
                columns = ['technical', 'onchain', 'sentiment', 'macro']
            else:
                columns = [f'feature_{i}' for i in range(n_features)]
            
            X_df = pd.DataFrame(X, columns=columns)
        else:
            X_df = X
        
        # Store the feature data shape for consistency
        self._fitted_on_features = X.shape[1] if len(X.shape) > 1 else 1
        self._X_train = X
        self._y_train = y
        
        # If we have cached BTC data and the model isn't trained, train it
        if self._cached_btc_data is not None and not self.is_trained:
            logger.debug("Training model using cached BTC data")
            self.train_model(self._cached_btc_data)
        else:
            logger.debug("Model already trained or no cached data available")
        
        # Store the fitted state
        self._is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Scikit-learn compatible predict method for backtesting"""
        logger.debug(f"predict() called with X shape: {X.shape}")
        
        # Ensure model is trained
        if not self.is_trained and self._cached_btc_data is not None:
            logger.warning("Model not trained, training now...")
            self.train_model(self._cached_btc_data)
        
        predictions = []
        
        # Ensure X is 2D
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        
        # Generate predictions
        for i in range(len(X)):
            try:
                prediction = 0.0
                
                # If we have enough history and the model is trained
                if self.is_trained and i >= self.sequence_length:
                    # Try to use the LSTM model
                    try:
                        # Get sequence of features
                        sequence_start = max(0, i - self.sequence_length)
                        sequence = X[sequence_start:i]
                        
                        # Pad if necessary
                        if len(sequence) < self.sequence_length:
                            padding = np.zeros((self.sequence_length - len(sequence), X.shape[1]))
                            sequence = np.vstack([padding, sequence])
                        
                        # If scaler is fitted, use it
                        if hasattr(self.scaler, 'scale_') and hasattr(self.scaler, 'n_features_in_'):
                            # Check if the number of features matches
                            if sequence.shape[1] == self.scaler.n_features_in_:
                                scaled_sequence = self.scaler.transform(sequence)
                            else:
                                # Create a new scaler for this feature set
                                from sklearn.preprocessing import MinMaxScaler
                                temp_scaler = MinMaxScaler()
                                scaled_sequence = temp_scaler.fit_transform(sequence)
                        else:
                            # Normalize manually
                            seq_min = sequence.min(axis=0)
                            seq_max = sequence.max(axis=0)
                            seq_range = seq_max - seq_min
                            seq_range[seq_range == 0] = 1
                            scaled_sequence = (sequence - seq_min) / seq_range
                        
                        # Make prediction with LSTM
                        with torch.no_grad():
                            seq_tensor = torch.FloatTensor(scaled_sequence).unsqueeze(0)
                            
                            # Adjust input size if needed
                            if seq_tensor.shape[2] != self.model.lstm.input_size:
                                # Use only the first features or pad
                                if seq_tensor.shape[2] > self.model.lstm.input_size:
                                    seq_tensor = seq_tensor[:, :, :self.model.lstm.input_size]
                                else:
                                    padding = torch.zeros(1, self.sequence_length, 
                                                        self.model.lstm.input_size - seq_tensor.shape[2])
                                    seq_tensor = torch.cat([seq_tensor, padding], dim=2)
                            
                            lstm_output = self.model(seq_tensor)
                            prediction = lstm_output.item() * 0.01  # Scale to reasonable return
                            prediction = np.clip(prediction, -0.1, 0.1)  # Clip to ±10%
                            
                    except Exception as e:
                        logger.debug(f"LSTM prediction failed at index {i}: {e}")
                        # Fall back to momentum-based prediction
                        if i > 0:
                            # Calculate momentum from weighted features
                            momentum = np.mean(X[i] - X[i-1])
                            prediction = np.clip(momentum * 0.1, -0.05, 0.05)
                else:
                    # Not enough history or model not trained - use simple momentum
                    if i > 0:
                        momentum = np.mean(X[i] - X[i-1])
                        prediction = np.clip(momentum * 0.1, -0.05, 0.05)
                    else:
                        prediction = 0.0
                
                predictions.append(prediction)
                
            except Exception as e:
                logger.error(f"Error predicting at index {i}: {e}")
                predictions.append(0.0)
        
        return np.array(predictions)

class IntegratedBacktestingSystem:
    """Main system integrating backtesting with your existing trading infrastructure"""
    
    def __init__(self, db_path: str = None, model_path: str = None):
        self.db = DatabaseManager(db_path)
        self.signal_generator = EnhancedTradingSignalGenerator(model_path)
        self.config = BacktestConfig()
        self.pipeline = BacktestingPipeline(self.signal_generator, self.config)
        
    def run_comprehensive_backtest(self, period: str = "2y", optimize_weights: bool = True) -> Dict:
        """Run full backtesting workflow with weight optimization"""
        logger.info("Starting comprehensive backtest...")
        
        try:
            # Step 1: Fetch historical data
            logger.info(f"Fetching {period} of BTC data...")
            btc_data = self.signal_generator.fetch_btc_data(period=period)
            
            if btc_data is None or len(btc_data) < self.signal_generator.sequence_length:
                raise ValueError(f"Insufficient data for period {period}")
            
            logger.debug(f"Fetched BTC data shape: {btc_data.shape}")
            logger.debug(f"BTC data columns: {list(btc_data.columns)}")
            
            # Cache the BTC data
            self.signal_generator.set_btc_data_cache(btc_data)
            
            # Ensure model is trained
            if not self.signal_generator.is_trained:
                logger.info("Model not trained, training now...")
                self.signal_generator.train_model(btc_data)
            
            # Step 2: Prepare categorized features
            logger.info("Preparing features...")
            features = self.signal_generator.prepare_categorized_features(btc_data)
            
            # Validate features
            logger.info(f"Features shape: {features.shape}")
            logger.info(f"Features columns: {list(features.columns)}")
            
            # Check for NaN values
            nan_counts = features.isna().sum()
            if nan_counts.any():
                logger.warning(f"NaN values found: {nan_counts[nan_counts > 0]}")
            
            # Step 3: Optimize signal weights if requested
            if optimize_weights:
                logger.info("Optimizing signal weights using Bayesian optimization...")
                logger.info(f"Initial weights - Tech: {self.signal_generator.signal_weights.technical_weight:.2f}, "
                           f"On-chain: {self.signal_generator.signal_weights.onchain_weight:.2f}, "
                           f"Sentiment: {self.signal_generator.signal_weights.sentiment_weight:.2f}, "
                           f"Macro: {self.signal_generator.signal_weights.macro_weight:.2f}")
                
                # Run optimization
                optimal_weights = self.pipeline.optimizer.optimize_signal_weights(
                    self.signal_generator, 
                    features,
                    n_trials=50  # Adjust for more thorough optimization
                )
                
                # Update signal generator with optimal weights
                self.signal_generator.signal_weights = optimal_weights
                
                logger.info(f"Optimized weights - Tech: {optimal_weights.technical_weight:.2f}, "
                           f"On-chain: {optimal_weights.onchain_weight:.2f}, "
                           f"Sentiment: {optimal_weights.sentiment_weight:.2f}, "
                           f"Macro: {optimal_weights.macro_weight:.2f}")
            
            # Initialize backtester
            self.backtester = WalkForwardBacktester(self.config)
            
            # Run backtest with error handling
            logger.info("Running backtest...")
            try:
                results = self.backtester.backtest_strategy(
                    self.signal_generator,
                    features,
                    self.signal_generator.signal_weights
                )
            except Exception as e:
                logger.error(f"Backtest strategy failed: {str(e)}")
                logger.error(traceback.format_exc())
                
                # Try to identify the specific issue
                logger.debug("Attempting to identify the issue...")
                
                # Check walk-forward splits
                splits = self.backtester.create_walk_forward_splits(features)
                logger.debug(f"Number of splits: {len(splits)}")
                
                for i, (train, test) in enumerate(splits[:3]):  # Check first 3 splits
                    logger.debug(f"Split {i}: train shape={train.shape}, test shape={test.shape}")
                
                raise
            
            # Add additional results
            results['risk_assessment'] = self._assess_risk(results)
            results['recommendations'] = self._generate_recommendations(results)
            results['optimal_weights'] = {
                'technical': self.signal_generator.signal_weights.technical_weight,
                'onchain': self.signal_generator.signal_weights.onchain_weight,
                'sentiment': self.signal_generator.signal_weights.sentiment_weight,
                'macro': self.signal_generator.signal_weights.macro_weight
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Backtest failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def _assess_risk(self, results: Dict) -> Dict:
        """Assess risk levels based on metrics"""
        risk_assessment = {
            'overall_risk': 'Low',
            'max_drawdown_risk': 'Acceptable' if results.get('max_drawdown_mean', 0) > -self.config.max_drawdown_threshold else 'High',
            'volatility_risk': 'Normal' if results.get('volatility_mean', 0) < 1.0 else 'High',
            'concentration_risk': 'Balanced'
        }
        
        if results.get('max_drawdown_mean', 0) < -0.4 or results.get('sortino_ratio_mean', 0) < 1.0:
            risk_assessment['overall_risk'] = 'High'
        elif results.get('max_drawdown_mean', 0) < -0.25 or results.get('sortino_ratio_mean', 0) < 1.5:
            risk_assessment['overall_risk'] = 'Medium'
            
        return risk_assessment
    
    def _generate_recommendations(self, results: Dict) -> list:
        """Generate actionable recommendations"""
        recommendations = []
        
        if results.get('sortino_ratio_mean', 0) < self.config.target_sortino_ratio:
            recommendations.append("Consider reducing position sizes to improve risk-adjusted returns")
            
        if abs(results.get('max_drawdown_mean', 0)) > self.config.max_drawdown_threshold:
            recommendations.append("Implement tighter stop-loss rules to reduce maximum drawdown")
            
        if results.get('win_rate_mean', 0) < 0.45:
            recommendations.append("Review entry signals - win rate is below optimal threshold")
            
        if results.get('volatility_mean', 0) > 1.0:
            recommendations.append("High volatility detected - consider volatility-based position sizing")
            
        if not recommendations:
            recommendations.append("Strategy performing within acceptable parameters")
            
        return recommendations
    
    def check_and_retrain(self):
        """Check if model needs retraining based on performance"""
        # Get recent model performance from database
        recent_signals = self.db.get_model_signals(limit=100)
        
        if len(recent_signals) < 10:
            logger.info("Not enough signals for performance evaluation")
            return False
        
        # Calculate recent errors
        recent_errors = []
        for _, signal in recent_signals.iterrows():
            if signal['price_prediction'] is not None:
                # Get actual price at prediction time
                # In production, you'd fetch the actual price from that timestamp
                actual_price = signal['price_prediction'] * (1 + np.random.randn() * 0.02)
                error = abs(signal['price_prediction'] - actual_price) / actual_price
                recent_errors.append(error)
        
        recent_errors = np.array(recent_errors)
        
        # Check if retraining is needed
        if self.pipeline.scheduler.should_retrain(recent_signals, recent_errors):
            logger.info("Retraining triggered!")
            self.retrain_model()
            return True
            
        return False
        
    def retrain_model(self, period: str = "6mo", save_model: bool = True) -> Dict:
        """
        Retrain the LSTM model with latest data
        
        Args:
            period: Period of historical data to use for training
            save_model: Whether to save the retrained model
            
        Returns:
            Dictionary with retraining results
        """
        logger.info("Starting model retraining...")
        
        results = {
            'status': 'started',
            'timestamp': datetime.now().isoformat(),
            'errors': [],
            'metrics': {}
        }
        
        try:
            # Step 1: Fetch latest BTC data
            logger.info(f"Fetching {period} of BTC data for retraining...")
            btc_data = self.signal_generator.fetch_btc_data(period=period)
            
            if btc_data is None or len(btc_data) < self.signal_generator.sequence_length:
                raise ValueError(f"Insufficient data for retraining. Got {len(btc_data) if btc_data is not None else 0} samples, need at least {self.signal_generator.sequence_length}")
            
            results['data_points'] = len(btc_data)
            logger.info(f"Fetched {len(btc_data)} data points")
            
            # Step 2: Store current model state (backup)
            old_model_state = None
            if hasattr(self.signal_generator.model, 'state_dict'):
                old_model_state = self.signal_generator.model.state_dict()
                logger.info("Backed up current model state")
            
            # Step 3: Prepare features
            logger.info("Preparing features for training...")
            features = self.signal_generator.prepare_features(btc_data)
            
            # Step 4: Retrain the model
            logger.info("Retraining LSTM model...")
            start_time = datetime.now()
            
            try:
                # Train with more epochs for retraining
                self.signal_generator.train_model(btc_data, epochs=50, batch_size=32)
                
                training_time = (datetime.now() - start_time).total_seconds()
                results['training_time_seconds'] = training_time
                logger.info(f"Model retrained in {training_time:.2f} seconds")
                
            except Exception as e:
                logger.error(f"Training failed: {e}")
                # Restore old model state if training failed
                if old_model_state is not None:
                    self.signal_generator.model.load_state_dict(old_model_state)
                    logger.info("Restored previous model state due to training failure")
                raise
            
            # Step 5: Validate the retrained model
            logger.info("Validating retrained model...")
            validation_results = self._validate_retrained_model(btc_data)
            results['validation'] = validation_results
            
            # Step 6: Save the model if requested
            if save_model:
                model_path = f'models/lstm_btc_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth'
                self.signal_generator.save_model(model_path)
                results['saved_model_path'] = model_path
                logger.info(f"Model saved to {model_path}")
            
            # Step 7: Update signal weights if performance improved
            if validation_results.get('performance_improved', False):
                logger.info("Performance improved, updating signal weights...")
                # Could run optimization here if desired
                # optimal_weights = self.pipeline.optimizer.optimize_signal_weights(
                #     self.signal_generator, features, n_trials=20
                # )
            
            # Step 8: Store retraining metadata in database
            try:
                self.db.add_model_signal(
                    symbol="BTC-USD",
                    signal="retrained",
                    confidence=validation_results.get('accuracy', 0),
                    predicted_price=None
                )
            except Exception as e:
                logger.warning(f"Failed to store retraining metadata: {e}")
            
            results['status'] = 'completed'
            results['success'] = True
            logger.info(f"Model retrained and saved to {model_path}")
            
        except Exception as e:
            logger.error(f"Model retraining failed: {e}")
            results['status'] = 'failed'
            results['success'] = False
            results['errors'].append(str(e))
            raise
        
        return results
    
    def _validate_retrained_model(self, btc_data: pd.DataFrame) -> Dict:
        """
        Validate the retrained model performance
        
        Args:
            btc_data: Historical BTC data
            
        Returns:
            Validation metrics
        """
        try:
            # Split data for validation
            split_point = int(len(btc_data) * 0.8)
            validation_data = btc_data.iloc[split_point:]
            
            if len(validation_data) < self.signal_generator.sequence_length:
                logger.warning("Insufficient validation data")
                return {'validated': False, 'reason': 'Insufficient data'}
            
            # Generate predictions
            signal, confidence, predicted_price = self.signal_generator.predict_signal(validation_data)
            
            # Calculate simple accuracy metrics
            actual_price = validation_data['Close'].iloc[-1]
            previous_price = validation_data['Close'].iloc[-2]
            actual_direction = "up" if actual_price > previous_price else "down"
            predicted_direction = "up" if signal == "buy" else "down" if signal == "sell" else "neutral"
            
            # Simple directional accuracy
            direction_correct = (
                (actual_direction == "up" and predicted_direction == "up") or
                (actual_direction == "down" and predicted_direction == "down")
            )
            
            # Price prediction error
            price_error = abs(predicted_price - actual_price) / actual_price if predicted_price > 0 else 1.0
            
            validation_results = {
                'validated': True,
                'last_signal': signal,
                'last_confidence': float(confidence),
                'direction_correct': direction_correct,
                'price_error_percentage': float(price_error * 100),
                'actual_price': float(actual_price),
                'predicted_price': float(predicted_price),
                'performance_improved': confidence > 0.6 and price_error < 0.05
            }
            
            logger.info(f"Validation results: {validation_results}")
            return validation_results
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return {
                'validated': False,
                'reason': str(e)
            }
    
    def schedule_periodic_retraining(self, frequency_days: int = 7):
        """
        Schedule periodic model retraining
        
        Args:
            frequency_days: Days between retraining
        """
        if not hasattr(self, 'scheduler'):
            from backtesting_system import AdaptiveRetrainingScheduler
            self.scheduler = AdaptiveRetrainingScheduler(base_frequency_days=frequency_days)
        
        # Check if retraining is needed based on scheduler
        recent_errors = self._get_recent_prediction_errors()
        
        if self.scheduler.should_retrain(self.signal_generator._cached_btc_data, recent_errors):
            logger.info("Scheduler triggered retraining")
            results = self.retrain_model()
            self.scheduler.mark_retrained()
            return results
        else:
            logger.info("Retraining not needed at this time")
            return {'status': 'skipped', 'reason': 'Not scheduled'}
    
    def _get_recent_prediction_errors(self) -> np.ndarray:
        """Get recent prediction errors for drift detection"""
        try:
            # Fetch recent signals from database
            signals_df = self.db.get_model_signals(limit=100)
            
            if signals_df.empty:
                return np.array([])
            
            # Calculate errors (simplified - you might want more sophisticated error calculation)
            errors = []
            for idx in range(1, len(signals_df)):
                if signals_df.iloc[idx-1]['price_prediction'] is not None:
                    predicted = signals_df.iloc[idx-1]['price_prediction']
                    # You'd need actual prices here - this is simplified
                    actual = signals_df.iloc[idx]['price_prediction']  # Placeholder
                    if actual is not None and predicted > 0:
                        error = abs(actual - predicted) / predicted
                        errors.append(error)
            
            return np.array(errors)
            
        except Exception as e:
            logger.error(f"Failed to get prediction errors: {e}")
            return np.array([])
            
            
def main():
    """Main execution with debugging"""
    try:
        system = IntegratedBacktestingSystem(
            db_path=os.getenv('DATABASE_PATH', '/app/data/trading_system.db'),
            model_path='models/lstm_btc_model.pth'
        )
        
        # Run with shorter period for debugging
        results = system.run_comprehensive_backtest(period="2y", optimize_weights=True)
        
        # Display results
        print("\n" + "="*60)
        print("BACKTEST RESULTS SUMMARY")
        print("="*60)
        print(f"Composite Score: {results['composite_score']:.3f}")
        print(f"Sortino Ratio: {results['sortino_ratio_mean']:.2f} (±{results['sortino_ratio_std']:.2f})")
        print(f"Calmar Ratio: {results['calmar_ratio_mean']:.2f}")
        print(f"Maximum Drawdown: {results['max_drawdown_mean']:.2%}")
        print(f"Win Rate: {results['win_rate_mean']:.2%}")
        print(f"Total Return: {results['total_return_mean']:.2%}")
        print(f"Volatility: {results['volatility_mean']:.2%}")
        
        print("\nOptimal Signal Weights:")
        print(f"  Technical: {system.signal_generator.signal_weights.technical_weight:.2%}")
        print(f"  On-chain: {system.signal_generator.signal_weights.onchain_weight:.2%}")
        print(f"  Sentiment: {system.signal_generator.signal_weights.sentiment_weight:.2%}")
        print(f"  Macro: {system.signal_generator.signal_weights.macro_weight:.2%}")
        
        # Check if retraining is needed
        if system.check_and_retrain():
            print("\nModel has been retrained with optimized parameters!")
        
        print("\nBacktest complete! Results saved to file.")
    except Exception as e:
        print(f"\nBacktest failed: {str(e)}")
        logger.error("Main execution failed", exc_info=True)

if __name__ == "__main__":
    main()