import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
from datetime import datetime, timedelta
import optuna
from sklearn.preprocessing import MinMaxScaler
import logging
from dataclasses import dataclass
import json
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BacktestConfig:
    """Configuration for backtesting parameters"""
    training_window_days: int = 1008  # 3-4 years as per research
    test_window_days: int = 90  # 3 months
    purge_days: int = 2  # Prevent information leakage
    retraining_frequency_days: int = 90  # Quarterly retraining
    min_train_test_ratio: float = 0.7  # 70% training minimum
    transaction_cost: float = 0.0025  # 0.25% per trade
    max_drawdown_threshold: float = 0.25  # 25% max acceptable drawdown
    target_sortino_ratio: float = 2.0  # Target Sortino ratio
    
@dataclass
class SignalWeights:
    """Optimizable signal weights for feature importance"""
    technical_weight: float = 0.40  # 40% technical indicators
    onchain_weight: float = 0.35   # 35% on-chain metrics
    sentiment_weight: float = 0.15  # 15% sentiment data
    macro_weight: float = 0.10      # 10% macroeconomic factors
    
    def normalize(self):
        """Ensure weights sum to 1.0"""
        total = self.technical_weight + self.onchain_weight + self.sentiment_weight + self.macro_weight
        self.technical_weight /= total
        self.onchain_weight /= total
        self.sentiment_weight /= total
        self.macro_weight /= total

class PerformanceMetrics:
    """Calculate crypto-specific performance metrics"""
    
    @staticmethod
    def sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio focusing on downside deviation"""
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf')
        
        downside_deviation = np.sqrt(np.mean(downside_returns**2)) * np.sqrt(252)
        annual_return = np.mean(excess_returns) * 252
        
        return annual_return / downside_deviation if downside_deviation > 0 else float('inf')
    
    @staticmethod
    def calmar_ratio(returns: np.ndarray, max_drawdown: float) -> float:
        """Calculate Calmar ratio (annual return / max drawdown)"""
        annual_return = np.mean(returns) * 252
        return annual_return / abs(max_drawdown) if max_drawdown != 0 else float('inf')
    
    @staticmethod
    def maximum_drawdown(cumulative_returns: np.ndarray) -> float:
        """Calculate maximum drawdown from cumulative returns"""
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        return np.min(drawdown)
    
    @staticmethod
    def profit_factor(returns: np.ndarray) -> float:
        """Calculate profit factor (gross profits / gross losses)"""
        profits = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        return profits / losses if losses > 0 else float('inf')
    
    @staticmethod
    def win_rate(returns: np.ndarray) -> float:
        """Calculate win rate percentage"""
        return (returns > 0).sum() / len(returns) if len(returns) > 0 else 0
    
    @staticmethod
    def information_ratio(returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
        """Calculate information ratio"""
        active_returns = returns - benchmark_returns
        if active_returns.std() == 0:
            return 0
        return (active_returns.mean() * 252) / (active_returns.std() * np.sqrt(252))

class WalkForwardBacktester:
    """Implements walk-forward analysis for LSTM crypto trading"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.results_history = []
        
    def create_walk_forward_splits(self, data: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Create walk-forward train/test splits with adaptive sizing"""
        splits = []
        total_days = len(data)
        
        # Adaptive parameters based on data size
        if total_days < 200:
            training_window = max(60, int(total_days * 0.6))
            test_window = max(20, int(total_days * 0.2))
        else:
            training_window = min(self.config.training_window_days, int(total_days * 0.7))
            test_window = min(self.config.test_window_days, int(total_days * 0.2))
        
        # FIXED: Move start_idx and while loop outside the if/else
        start_idx = 0
        
        while start_idx + training_window + self.config.purge_days + test_window <= total_days:
            # Training data
            train_end = start_idx + training_window
            train_data = data.iloc[start_idx:train_end]
            
            # Purge period to prevent information leakage
            test_start = train_end + self.config.purge_days
            test_end = test_start + test_window
            test_data = data.iloc[test_start:test_end]
            
            splits.append((train_data, test_data))
            
            # Move forward by retraining frequency
            start_idx += self.config.retraining_frequency_days
                
        logger.info(f"Created {len(splits)} walk-forward splits")
        return splits
    
    def backtest_strategy(self, model, data: pd.DataFrame, signal_weights: SignalWeights) -> Dict:
        """Run complete walk-forward backtest"""
        splits = self.create_walk_forward_splits(data)
        all_results = []
        
        for i, (train_data, test_data) in enumerate(splits):
            logger.info(f"Processing split {i+1}/{len(splits)}")
            
            # Apply signal weights to features
            weighted_features = self._apply_signal_weights(train_data, signal_weights)
            
            # Train model on this split
            # FIX: Ensure target is 1D by using .ravel()
            target = train_data['target'].values.ravel() if 'target' in train_data else np.zeros(len(train_data))
            model.fit(weighted_features, target)
            
            # Generate predictions
            test_features = self._apply_signal_weights(test_data, signal_weights)
            predictions = model.predict(test_features)
            
            # Calculate returns with transaction costs
            returns = self._calculate_returns(test_data, predictions)
            
            # Store results
            split_metrics = self._calculate_split_metrics(returns)
            all_results.append(split_metrics)
        
        # Aggregate results
        return self._aggregate_results(all_results)
    
    def _apply_signal_weights(self, data: pd.DataFrame, weights: SignalWeights) -> np.ndarray:
        """Apply optimized weights to different signal categories"""
        weighted_features_list = []
        
        # Process each row individually
        for idx, row in data.iterrows():
            row_features = []
            
            # Handle technical features
            if 'technical_features' in data.columns:
                tech_feat = row['technical_features']
                if isinstance(tech_feat, (list, np.ndarray)):
                    tech_feat = np.array(tech_feat).flatten()
                    weighted_tech = tech_feat * weights.technical_weight
                    row_features.extend(weighted_tech)
                else:
                    row_features.append(float(tech_feat) * weights.technical_weight)
            
            # Handle onchain features
            if 'onchain_features' in data.columns:
                onchain_feat = row['onchain_features']
                if isinstance(onchain_feat, (list, np.ndarray)):
                    onchain_feat = np.array(onchain_feat).flatten()
                    weighted_onchain = onchain_feat * weights.onchain_weight
                    row_features.extend(weighted_onchain)
                else:
                    row_features.append(float(onchain_feat) * weights.onchain_weight)
            
            # Handle sentiment features
            if 'sentiment_features' in data.columns:
                sent_feat = row['sentiment_features']
                if isinstance(sent_feat, (list, np.ndarray)):
                    sent_feat = np.array(sent_feat).flatten()
                    weighted_sent = sent_feat * weights.sentiment_weight
                    row_features.extend(weighted_sent)
                else:
                    row_features.append(float(sent_feat) * weights.sentiment_weight)
            
            # Handle macro features
            if 'macro_features' in data.columns:
                macro_feat = row['macro_features']
                if isinstance(macro_feat, (list, np.ndarray)):
                    macro_feat = np.array(macro_feat).flatten()
                    weighted_macro = macro_feat * weights.macro_weight
                    row_features.extend(weighted_macro)
                else:
                    row_features.append(float(macro_feat) * weights.macro_weight)
            
            weighted_features_list.append(row_features)
        
        # Convert to numpy array
        weighted_features = np.array(weighted_features_list)
        
        # Ensure we return a 2D array
        if weighted_features.ndim == 1:
            weighted_features = weighted_features.reshape(-1, 1)
        
        return weighted_features
    
    def _calculate_returns(self, data: pd.DataFrame, predictions: np.ndarray) -> np.ndarray:
        """Calculate returns including transaction costs"""
        # Convert predictions to positions (-1, 0, 1)
        positions = np.where(predictions > 0.02, 1, 
                           np.where(predictions < -0.02, -1, 0))
        
        # Calculate raw returns
        price_returns = data['Close'].pct_change().fillna(0).values
        strategy_returns = positions[:-1] * price_returns[1:]
        
        # Apply transaction costs on position changes
        position_changes = np.diff(positions)
        transaction_costs = np.abs(position_changes) * self.config.transaction_cost
        
        # Net returns
        net_returns = strategy_returns - transaction_costs
        
        return net_returns
    
    def _calculate_split_metrics(self, returns: np.ndarray) -> Dict:
        """Calculate all performance metrics for a split"""
        cumulative_returns = (1 + returns).cumprod()
        
        metrics = {
            'sortino_ratio': PerformanceMetrics.sortino_ratio(returns),
            'calmar_ratio': PerformanceMetrics.calmar_ratio(
                returns, 
                PerformanceMetrics.maximum_drawdown(cumulative_returns)
            ),
            'max_drawdown': PerformanceMetrics.maximum_drawdown(cumulative_returns),
            'profit_factor': PerformanceMetrics.profit_factor(returns),
            'win_rate': PerformanceMetrics.win_rate(returns),
            'total_return': cumulative_returns[-1] - 1 if len(cumulative_returns) > 0 else 0,
            'sharpe_ratio': self._calculate_sharpe_ratio(returns),
            'avg_daily_return': np.mean(returns),
            'volatility': np.std(returns) * np.sqrt(252)
        }
        
        return metrics
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        excess_returns = returns - risk_free_rate / 252
        if returns.std() == 0:
            return 0
        return np.mean(excess_returns) * 252 / (np.std(returns) * np.sqrt(252))
    
    def _aggregate_results(self, results: List[Dict]) -> Dict:
        """Aggregate metrics across all splits"""
        if not results or len(results) == 0:
            logger.warning("No results to aggregate - insufficient data")
            return {
                'sortino_ratio_mean': 0.0,
                'calmar_ratio_mean': 0.0,
                'max_drawdown_mean': 0.0,
                'profit_factor_mean': 0.0,
                'win_rate_mean': 0.0,
                'total_return_mean': 0.0,
                'sharpe_ratio_mean': 0.0,
                'volatility_mean': 0.0,
                'composite_score': 0.0,
                'periods_tested': 0,
                'success': False,
                'error_message': 'Insufficient data for backtesting'
            }
            
        aggregated = {}
        
        for metric in results[0].keys():
            values = [r[metric] for r in results]
            aggregated[f'{metric}_mean'] = np.mean(values)
            aggregated[f'{metric}_std'] = np.std(values)
            aggregated[f'{metric}_min'] = np.min(values)
            aggregated[f'{metric}_max'] = np.max(values)
        
        # Calculate composite score as per research
        aggregated['composite_score'] = self._calculate_composite_score(aggregated)
        
        return aggregated
    
    def _calculate_composite_score(self, metrics: Dict) -> float:
        """Calculate composite score based on research recommendations"""
        # Risk Component (40%)
        risk_score = (
            min(metrics['sortino_ratio_mean'] / self.config.target_sortino_ratio, 2.0) * 0.20 +
            max(0, 1 - metrics['max_drawdown_mean'] / self.config.max_drawdown_threshold) * 0.20
        )
        
        # Return Component (35%)
        return_score = (
            min(metrics['calmar_ratio_mean'] / 3.0, 1.0) * 0.20 +
            min(metrics['profit_factor_mean'] / 2.0, 1.0) * 0.15
        )
        
        # Consistency Component (25%)
        consistency_score = (
            metrics['win_rate_mean'] * 0.15 +
            min(max(metrics.get('information_ratio_mean', 0) / 1.0, 0), 1.0) * 0.10
        )
        
        return risk_score + return_score + consistency_score

class BayesianOptimizer:
    """Implements Bayesian optimization for hyperparameter and signal weight tuning"""
    
    def __init__(self, backtester: WalkForwardBacktester):
        self.backtester = backtester
        self.study = None
        
    def optimize_signal_weights(self, model, data: pd.DataFrame, n_trials: int = 100) -> SignalWeights:
        """Optimize signal weights using Optuna"""
        
        def objective(trial):
            # Suggest signal weights
            weights = SignalWeights(
                technical_weight=trial.suggest_float('technical_weight', 0.1, 0.6),
                onchain_weight=trial.suggest_float('onchain_weight', 0.1, 0.5),
                sentiment_weight=trial.suggest_float('sentiment_weight', 0.05, 0.3),
                macro_weight=trial.suggest_float('macro_weight', 0.05, 0.3)
            )
            weights.normalize()
            
            # Run backtest with these weights
            results = self.backtester.backtest_strategy(model, data, weights)
            
            # Return negative composite score (Optuna minimizes)
            return -results['composite_score']
        
        # Create and run study
        self.study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        self.study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        # Get best weights
        best_params = self.study.best_params
        best_weights = SignalWeights(
            technical_weight=best_params['technical_weight'],
            onchain_weight=best_params['onchain_weight'],
            sentiment_weight=best_params['sentiment_weight'],
            macro_weight=best_params['macro_weight']
        )
        best_weights.normalize()
        
        logger.info(f"Best signal weights found: {best_weights}")
        return best_weights
    
    def optimize_lstm_architecture(self, data: pd.DataFrame, n_trials: int = 50) -> Dict:
        """Optimize LSTM architecture hyperparameters"""
        
        def objective(trial):
            # Suggest hyperparameters based on research
            params = {
                'num_layers': trial.suggest_int('num_layers', 1, 4),
                'hidden_size': trial.suggest_categorical('hidden_size', [50, 128, 256, 512]),
                'dropout': trial.suggest_float('dropout', 0.1, 0.5),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128, 192])
            }
            
            # Create and train model with these parameters
            model = self._create_lstm_model(params)
            
            # Simplified training and evaluation
            train_loss = self._train_model(model, data, params)
            
            return train_loss
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        return study.best_params
    
    def _create_lstm_model(self, params: Dict) -> nn.Module:
        """Create LSTM model with given parameters"""
        # This should be integrated with your existing LSTM model
        # Placeholder for model creation
        pass
    
    def _train_model(self, model, data, params) -> float:
        """Train model and return validation loss"""
        # Placeholder for training logic
        pass

class ConceptDriftDetector:
    """Detect concept drift and trigger retraining"""
    
    def __init__(self, window_size: int = 100, threshold: float = 3.0):
        self.window_size = window_size
        self.threshold = threshold
        self.baseline_stats = None
        
    def detect_drift(self, recent_errors: np.ndarray) -> bool:
        """Detect if concept drift has occurred using ADWIN-inspired approach"""
        if len(recent_errors) < self.window_size:
            return False
            
        if self.baseline_stats is None:
            self.baseline_stats = {
                'mean': np.mean(recent_errors[:self.window_size]),
                'std': np.std(recent_errors[:self.window_size])
            }
            return False
        
        # Calculate recent statistics
        recent_mean = np.mean(recent_errors[-self.window_size:])
        
        # Check if drift occurred (simplified Page-Hinkley test)
        z_score = abs(recent_mean - self.baseline_stats['mean']) / (self.baseline_stats['std'] + 1e-8)
        
        if z_score > self.threshold:
            logger.warning(f"Concept drift detected! Z-score: {z_score:.2f}")
            # Update baseline
            self.baseline_stats = {
                'mean': recent_mean,
                'std': np.std(recent_errors[-self.window_size:])
            }
            return True
            
        return False

class AdaptiveRetrainingScheduler:
    """Manages adaptive retraining based on market conditions"""
    
    def __init__(self, base_frequency_days: int = 90):
        self.base_frequency = base_frequency_days
        self.last_retrain_date = datetime.now()
        self.volatility_threshold = 0.5  # 50% annualized volatility
        self.drift_detector = ConceptDriftDetector()
        
    def should_retrain(self, recent_data: pd.DataFrame, recent_errors: np.ndarray) -> bool:
        """Determine if model should be retrained"""
        
        # Check base frequency
        days_since_retrain = (datetime.now() - self.last_retrain_date).days
        if days_since_retrain >= self.base_frequency:
            logger.info("Retraining due to scheduled frequency")
            return True
        
        # Check volatility-based retraining
        if self._check_high_volatility(recent_data):
            logger.info("Retraining due to high volatility")
            return True
        
        # Check concept drift
        if self.drift_detector.detect_drift(recent_errors):
            logger.info("Retraining due to concept drift")
            return True
        
        return False
    
    def _check_high_volatility(self, data: pd.DataFrame) -> bool:
        """Check if recent volatility exceeds threshold"""
        returns = data['Close'].pct_change().dropna()
        recent_volatility = returns.tail(30).std() * np.sqrt(252)
        return recent_volatility > self.volatility_threshold
    
    def mark_retrained(self):
        """Mark that retraining has occurred"""
        self.last_retrain_date = datetime.now()

class BacktestingPipeline:
    """Main pipeline for backtesting and optimization"""
    
    def __init__(self, model, config: BacktestConfig = None):
        self.model = model
        self.config = config or BacktestConfig()
        self.backtester = WalkForwardBacktester(self.config)
        self.optimizer = BayesianOptimizer(self.backtester)
        self.scheduler = AdaptiveRetrainingScheduler()
        
    def run_full_backtest(self, data: pd.DataFrame) -> Dict:
        """Run complete backtesting pipeline with optimization"""
        logger.info("Starting full backtesting pipeline")
        
        # Step 1: Optimize signal weights
        logger.info("Optimizing signal weights...")
        optimal_weights = self.optimizer.optimize_signal_weights(self.model, data)
        
        # Step 2: Run final backtest with optimal weights
        logger.info("Running final backtest with optimal weights...")
        final_results = self.backtester.backtest_strategy(self.model, data, optimal_weights)
        
        # Step 3: Generate report
        report = self._generate_report(final_results, optimal_weights)
        
        return report
    
    def _generate_report(self, results: Dict, weights: SignalWeights) -> Dict:
        """Generate comprehensive backtest report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'performance_metrics': results,
            'optimal_weights': {
                'technical': weights.technical_weight,
                'onchain': weights.onchain_weight,
                'sentiment': weights.sentiment_weight,
                'macro': weights.macro_weight
            },
            'risk_assessment': self._assess_risk(results),
            'recommendations': self._generate_recommendations(results)
        }
        
        return report
    
    def _assess_risk(self, results: Dict) -> Dict:
        """Assess risk levels based on metrics"""
        risk_assessment = {
            'overall_risk': 'Low',
            'max_drawdown_risk': 'Acceptable' if results['max_drawdown_mean'] < self.config.max_drawdown_threshold else 'High',
            'volatility_risk': 'Normal' if results['volatility_mean'] < 1.0 else 'High',
            'concentration_risk': 'Balanced'  # Based on signal weights
        }
        
        if results['max_drawdown_mean'] > 0.4 or results['sortino_ratio_mean'] < 1.0:
            risk_assessment['overall_risk'] = 'High'
        elif results['max_drawdown_mean'] > 0.25 or results['sortino_ratio_mean'] < 1.5:
            risk_assessment['overall_risk'] = 'Medium'
            
        return risk_assessment
    
    def _generate_recommendations(self, results: Dict) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if results['sortino_ratio_mean'] < self.config.target_sortino_ratio:
            recommendations.append("Consider reducing position sizes to improve risk-adjusted returns")
            
        if results['max_drawdown_mean'] > self.config.max_drawdown_threshold:
            recommendations.append("Implement tighter stop-loss rules to reduce maximum drawdown")
            
        if results['win_rate_mean'] < 0.45:
            recommendations.append("Review entry signals - win rate is below optimal threshold")
            
        if results['volatility_mean'] > 1.0:
            recommendations.append("High volatility detected - consider volatility-based position sizing")
            
        return recommendations

# Integration with existing LSTM model
def integrate_backtesting_with_existing_model(lstm_model_path: str, data_path: str):
    """Example integration with your existing LSTM model"""
    
    # Load your existing model
    from lstm_model import TradingSignalGenerator  # Your existing model
    
    # Initialize model
    signal_generator = TradingSignalGenerator(model_path=lstm_model_path)
    
    # Load data
    btc_data = signal_generator.fetch_btc_data(period="2y")
    
    # Prepare features (ensure they're categorized)
    features = signal_generator.prepare_features(btc_data)
    
    # Run backtesting pipeline
    pipeline = BacktestingPipeline(signal_generator)
    results = pipeline.run_full_backtest(features)
    
    # Save results
    with open('backtest_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Check if retraining is needed
    if pipeline.scheduler.should_retrain(btc_data, results['performance_metrics']):
        logger.info("Retraining model with optimized parameters...")
        signal_generator.train_model(btc_data, epochs=100)
        signal_generator.save_model('optimized_model.pth')
        pipeline.scheduler.mark_retrained()
    
    return results

if __name__ == "__main__":
    # Example usage
    results = integrate_backtesting_with_existing_model(
        lstm_model_path='models/lstm_btc_model.pth',
        data_path='data/btc_data.csv'
    )
    
    print("Backtest Complete!")
    print(f"Composite Score: {results['performance_metrics']['composite_score']:.3f}")
    print(f"Sortino Ratio: {results['performance_metrics']['sortino_ratio_mean']:.2f}")
    print(f"Max Drawdown: {results['performance_metrics']['max_drawdown_mean']:.2%}")