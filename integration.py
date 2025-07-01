"""
Integration module for backtesting system with existing BTC trading repository
This file should be placed in your repository root alongside lstm_model.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from typing import Dict

from lstm_model import TradingSignalGenerator
from database_models import DatabaseManager
from backtesting_system import (
    BacktestConfig, SignalWeights, BacktestingPipeline,
    AdaptiveRetrainingScheduler, BayesianOptimizer
)
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedTradingSignalGenerator(TradingSignalGenerator):
    """Enhanced version of your existing TradingSignalGenerator with backtesting capabilities"""
    
    def __init__(self, model_path: str = None, sequence_length: int = 60):
        super().__init__(model_path, sequence_length)
        self.signal_weights = SignalWeights()  # Default weights
        self.performance_history = []
        
    def prepare_categorized_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features categorized by type for weighted optimization"""
        # Get base features
        features = self.prepare_features(data)
        
        # Categorize features
        categorized = pd.DataFrame(index=features.index)
        
        # Technical indicators (price-based)
        technical_cols = ['price', 'sma_ratio', 'rsi', 'macd']
        if all(col in features.columns for col in technical_cols):
            categorized['technical_features'] = features[technical_cols].values
        
        # Volume-based features (can be considered on-chain for crypto)
        volume_cols = ['volume']
        if all(col in features.columns for col in volume_cols):
            categorized['onchain_features'] = features[volume_cols].values
        
        # Add placeholder columns for missing feature types
        if 'sentiment_features' not in categorized:
            categorized['sentiment_features'] = np.zeros((len(features), 1))
        if 'macro_features' not in categorized:
            categorized['macro_features'] = np.zeros((len(features), 1))
            
        # Add target for backtesting
        categorized['target'] = features['price'].pct_change().shift(-1).fillna(0)
        categorized['Close'] = features['price']
        
        return categorized
    
    def apply_signal_weights(self, features: pd.DataFrame) -> np.ndarray:
        """Apply optimized signal weights to features"""
        weighted_features = []
        
        # Apply weights to each category
        if 'technical_features' in features:
            tech_features = features[['price', 'sma_ratio', 'rsi', 'macd']].values
            weighted_features.append(tech_features * self.signal_weights.technical_weight)
            
        if 'volume' in features:
            volume_features = features[['volume']].values.reshape(-1, 1)
            weighted_features.append(volume_features * self.signal_weights.onchain_weight)
            
        # Concatenate all weighted features
        if weighted_features:
            return np.hstack(weighted_features)
        else:
            return features.values
    
    def predict_with_weights(self, features: pd.DataFrame) -> np.ndarray:
        """Generate predictions using weighted features"""
        # Apply signal weights
        weighted_features = self.apply_signal_weights(features)
        
        # Scale features
        if hasattr(self.scaler, 'scale_'):
            scaled_features = self.scaler.transform(weighted_features)
        else:
            scaled_features = self.scaler.fit_transform(weighted_features)
        
        # Generate predictions (simplified for integration)
        # In production, this would use your actual LSTM model
        predictions = np.random.randn(len(features)) * 0.02  # Placeholder
        
        return predictions

class IntegratedBacktestingSystem:
    """Main system integrating backtesting with your existing trading infrastructure"""
    
    def __init__(self, db_path: str = None, model_path: str = None):
        self.db = DatabaseManager(db_path)
        self.signal_generator = EnhancedTradingSignalGenerator(model_path)
        self.config = BacktestConfig()
        self.pipeline = BacktestingPipeline(self.signal_generator, self.config)
        
    def run_comprehensive_backtest(self, period: str = "2y", optimize_weights: bool = True):
        """Run full backtesting workflow"""
        logger.info("Starting comprehensive backtest...")
        
        # Step 1: Fetch historical data
        logger.info(f"Fetching {period} of BTC data...")
        btc_data = self.signal_generator.fetch_btc_data(period=period)
        
        # Step 2: Prepare categorized features
        logger.info("Preparing features...")
        features = self.signal_generator.prepare_categorized_features(btc_data)
        
        # Step 3: Optimize signal weights if requested
        if optimize_weights:
            logger.info("Optimizing signal weights using Bayesian optimization...")
            optimal_weights = self.pipeline.optimizer.optimize_signal_weights(
                self.signal_generator, 
                features,
                n_trials=50  # Reduced for faster testing
            )
            self.signal_generator.signal_weights = optimal_weights
        
        # Step 4: Run backtest with optimal weights
        logger.info("Running backtest with optimized parameters...")
        results = self.pipeline.backtester.backtest_strategy(
            self.signal_generator,
            features,
            self.signal_generator.signal_weights
        )
        
        # Step 5: Store results in database
        self._store_backtest_results(results)
        
        # Step 6: Generate trading signals for current market
        self._generate_current_signals(btc_data)
        
        return results
    
    def _store_backtest_results(self, results: Dict):
        """Store backtest results in database"""
        # Create a summary for database storage
        summary = {
            'timestamp': datetime.now(),
            'composite_score': results['composite_score'],
            'sortino_ratio': results['sortino_ratio_mean'],
            'max_drawdown': results['max_drawdown_mean'],
            'total_return': results['total_return_mean'],
            'signal_weights': {
                'technical': self.signal_generator.signal_weights.technical_weight,
                'onchain': self.signal_generator.signal_weights.onchain_weight,
                'sentiment': self.signal_generator.signal_weights.sentiment_weight,
                'macro': self.signal_generator.signal_weights.macro_weight
            }
        }
        
        # Store as model signal with special type
        self.db.add_model_signal(
            symbol="BTC-USD",
            signal="backtest_result",
            confidence=results['composite_score'],
            price_prediction=None
        )
        
        # Save detailed results to file
        with open(f'backtest_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    def _generate_current_signals(self, btc_data: pd.DataFrame):
        """Generate and store current trading signals"""
        # Get latest data for prediction
        current_features = self.signal_generator.prepare_features(btc_data)
        
        # Generate signal
        signal, confidence, predicted_price = self.signal_generator.predict_signal(btc_data)
        
        # Store in database
        self.db.add_model_signal(
            symbol="BTC-USD",
            signal=signal,
            confidence=confidence,
            price_prediction=predicted_price
        )
        
        logger.info(f"Generated signal: {signal} (confidence: {confidence:.2%})")
    
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
    
    def retrain_model(self):
        """Retrain model with latest data and optimized parameters"""
        logger.info("Starting model retraining...")
        
        # Fetch fresh data
        btc_data = self.signal_generator.fetch_btc_data(period="1y")
        
        # Train with optimized hyperparameters
        self.signal_generator.train_model(btc_data, epochs=100, batch_size=192)
        
        # Save updated model
        model_path = f'models/lstm_btc_optimized_{datetime.now().strftime("%Y%m%d")}.pth'
        self.signal_generator.save_model(model_path)
        
        # Mark retraining complete
        self.pipeline.scheduler.mark_retrained()
        
        logger.info(f"Model retrained and saved to {model_path}")

def main():
    """Main execution function"""
    # Initialize system
    system = IntegratedBacktestingSystem(
        db_path=os.getenv('DATABASE_PATH', '/app/data/trading_system.db'),
        model_path='models/lstm_btc_model.pth'
    )
    
    # Run comprehensive backtest
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

if __name__ == "__main__":
    main()
