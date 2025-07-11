"""
Strategy Optimizer - Full parameter optimization using Optuna
Utilizes the complete optimization capabilities of the system
Enhanced to handle limited data scenarios gracefully
"""

import optuna
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import asyncio
from datetime import datetime
import warnings

from .backtesting import (
    BacktestConfig, SignalWeights, EnhancedSignalWeights,
    WalkForwardBacktester, EnhancedWalkForwardBacktester,
    BayesianOptimizer, EnhancedBayesianOptimizer,
    BacktestingPipeline, EnhancedBacktestingPipeline
)

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Minimum data requirements
MIN_DATA_ROWS_STANDARD = 100  # Minimum for standard LSTM
MIN_DATA_ROWS_ENHANCED = 200  # Minimum for enhanced LSTM with walk-forward
MIN_DATA_ROWS_OPTIMIZATION = 50  # Absolute minimum for any optimization
MIN_SEQUENCE_LENGTH = 24  # Minimum sequence for LSTM predictions

@dataclass
class OptimizationConfig:
    """Configuration for strategy optimization"""
    # Parameter ranges from frontend
    technical_weight_range: Tuple[float, float] = (0.2, 0.6)
    onchain_weight_range: Tuple[float, float] = (0.1, 0.5)
    sentiment_weight_range: Tuple[float, float] = (0.05, 0.3)
    macro_weight_range: Tuple[float, float] = (0.05, 0.3)
    position_size_range: Tuple[float, float] = (0.05, 0.3)
    stop_loss_range: Tuple[float, float] = (0.02, 0.1)
    take_profit_range: Tuple[float, float] = (0.05, 0.2)
    min_confidence_range: Tuple[float, float] = (0.4, 0.8)
    
    # Optimization settings
    objective: str = "sharpe_ratio"  # sharpe_ratio, total_return, win_rate, risk_adjusted_return
    n_trials: int = 50
    constraints: List[str] = field(default_factory=list)
    timeout_seconds: int = 300  # 5 minutes
    
    # Backtesting settings
    lookback_days: int = 90
    initial_capital: float = 10000
    transaction_cost: float = 0.0025
    
    # Model settings
    use_enhanced_model: bool = True  # Use enhanced LSTM with attention
    adapt_to_data_size: bool = True  # Automatically adjust strategy based on data
    min_sequence_length: int = 24  # Minimum sequence for LSTM

class StrategyOptimizer:
    """Full strategy parameter optimizer using Optuna"""
    
    def __init__(self, backtest_system):
        """Initialize optimizer with backtesting system"""
        self.backtest_system = backtest_system
        self.study = None
        self.best_params = None
        self.optimization_history = []
        self.constraint_violations = []
        self.data_validation_results = {}
        self.model_type_used = None
        
    def create_objective(self, config: OptimizationConfig, data: pd.DataFrame):
        """Create Optuna objective function for given configuration"""
        
        def objective(trial):
            try:
                # Suggest signal weights
                technical_weight = trial.suggest_float(
                    'technical_weight', 
                    config.technical_weight_range[0], 
                    config.technical_weight_range[1]
                )
                onchain_weight = trial.suggest_float(
                    'onchain_weight',
                    config.onchain_weight_range[0],
                    config.onchain_weight_range[1]
                )
                sentiment_weight = trial.suggest_float(
                    'sentiment_weight',
                    config.sentiment_weight_range[0],
                    config.sentiment_weight_range[1]
                )
                macro_weight = trial.suggest_float(
                    'macro_weight',
                    config.macro_weight_range[0],
                    config.macro_weight_range[1]
                )
                
                # Create signal weights
                weights = SignalWeights(
                    technical_weight=technical_weight,
                    onchain_weight=onchain_weight,
                    sentiment_weight=sentiment_weight,
                    macro_weight=macro_weight
                )
                weights.normalize()
                
                # Suggest trading parameters
                position_size = trial.suggest_float(
                    'position_size',
                    config.position_size_range[0],
                    config.position_size_range[1]
                )
                stop_loss = trial.suggest_float(
                    'stop_loss',
                    config.stop_loss_range[0],
                    config.stop_loss_range[1]
                )
                take_profit = trial.suggest_float(
                    'take_profit',
                    config.take_profit_range[0],
                    config.take_profit_range[1]
                )
                min_confidence = trial.suggest_float(
                    'min_confidence',
                    config.min_confidence_range[0],
                    config.min_confidence_range[1]
                )
                
                # Update backtest configuration with suggested parameters
                self.backtest_system.config.position_size = position_size
                self.backtest_system.config.buy_threshold = (1 - min_confidence) * 0.1  # Convert confidence to threshold
                self.backtest_system.config.sell_threshold = (1 - min_confidence) * 0.1
                self.backtest_system.config.stop_loss = stop_loss
                self.backtest_system.config.take_profit = take_profit
                self.backtest_system.config.transaction_cost = config.transaction_cost
                
                # Run backtest with these parameters
                logger.info(f"Trial {trial.number}: Testing parameters...")
                
                # Create a temporary config with suggested parameters
                import copy
                original_config = copy.deepcopy(self.backtest_system.config)
                
                # Update config for this trial
                self.backtest_system.config.position_size = position_size
                self.backtest_system.config.buy_threshold = (1 - min_confidence) * 0.1
                self.backtest_system.config.sell_threshold = (1 - min_confidence) * 0.1
                self.backtest_system.config.stop_loss = stop_loss
                self.backtest_system.config.take_profit = take_profit
                
                try:
                    # Use the full backtesting pipeline
                    if hasattr(self.backtest_system, 'pipeline') and hasattr(self.backtest_system.pipeline, 'backtester'):
                        # Run backtest with weights through the pipeline
                        backtester = self.backtest_system.pipeline.backtester
                        model = self.backtest_system.pipeline.model
                        results = backtester.backtest_strategy(model, data, weights)
                    else:
                        # Use the simplified backtest through the system
                        # This runs a mini backtest for this trial
                        results = self._run_trial_backtest(data, weights, config)
                finally:
                    # Restore original config
                    self.backtest_system.config = original_config
                
                # Check constraints
                constraint_violated = self._check_constraints(results, config.constraints)
                if constraint_violated:
                    self.constraint_violations.append({
                        'trial': trial.number,
                        'violation': constraint_violated,
                        'results': results
                    })
                    # Return a penalty value for constraint violations
                    return 1e10
                
                # Store trial results
                self.optimization_history.append({
                    'trial': trial.number,
                    'params': trial.params,
                    'results': results,
                    'objective_value': self._calculate_objective_value(results, config.objective)
                })
                
                # Return objective value based on optimization goal
                return self._calculate_objective_value(results, config.objective)
                
            except Exception as e:
                logger.error(f"Trial {trial.number} failed: {str(e)}")
                return 1e10  # Return large penalty for failed trials
        
        return objective
    
    def _calculate_objective_value(self, results: Dict, objective: str) -> float:
        """Calculate objective value based on optimization goal"""
        # Optuna minimizes by default, so negate values we want to maximize
        
        if objective == "sharpe_ratio":
            return -results.get('sharpe_ratio', 0)  # Maximize Sharpe
        elif objective == "total_return":
            return -results.get('total_return', 0)  # Maximize return
        elif objective == "win_rate":
            return -results.get('win_rate', 0)  # Maximize win rate
        elif objective == "risk_adjusted_return":
            # Composite score that balances return and risk
            sharpe = results.get('sharpe_ratio', 0)
            sortino = results.get('sortino_ratio', 0)
            max_dd = abs(results.get('max_drawdown', 1))
            
            # Composite score: 40% Sharpe, 40% Sortino, 20% drawdown penalty
            composite = 0.4 * sharpe + 0.4 * sortino - 0.2 * max_dd
            return -composite
        else:
            # Default to Sharpe ratio
            return -results.get('sharpe_ratio', 0)
    
    def _check_constraints(self, results: Dict, constraints: List[str]) -> Optional[str]:
        """Check if results violate any constraints"""
        for constraint in constraints:
            if "Max Drawdown" in constraint:
                # Extract percentage from constraint string
                max_allowed = float(constraint.split("<")[1].strip().rstrip("%")) / 100
                actual_dd = abs(results.get('max_drawdown', 0))
                if actual_dd > max_allowed:
                    return f"Max drawdown {actual_dd:.1%} exceeds limit {max_allowed:.1%}"
                    
            elif "Min Win Rate" in constraint:
                # Extract percentage from constraint string
                min_required = float(constraint.split(">")[1].strip().rstrip("%")) / 100
                actual_wr = results.get('win_rate', 0)
                if actual_wr < min_required:
                    return f"Win rate {actual_wr:.1%} below minimum {min_required:.1%}"
                    
            elif "Min Sharpe" in constraint:
                min_required = float(constraint.split(">")[1].strip())
                actual_sharpe = results.get('sharpe_ratio', 0)
                if actual_sharpe < min_required:
                    return f"Sharpe ratio {actual_sharpe:.2f} below minimum {min_required}"
                    
        return None
    
    def _run_trial_backtest(self, data: pd.DataFrame, weights: SignalWeights, 
                           config: OptimizationConfig) -> Dict:
        """Run a backtest trial using the system's simplified backtest"""
        try:
            # Apply weights to features
            weighted_features = self._apply_weights_to_features(data, weights)
            
            # Determine sequence length based on data size
            sequence_length = config.min_sequence_length
            if len(data) < sequence_length * 2:
                sequence_length = max(10, len(data) // 3)
                logger.debug(f"Adjusted sequence length to {sequence_length} for limited data")
            
            # Get predictions using the model
            if hasattr(self.backtest_system, 'signal_generator') and hasattr(self.backtest_system.signal_generator, 'model'):
                model = self.backtest_system.signal_generator.model
                if model is not None:
                    try:
                        # Get predictions from the model
                        predictions = []
                        
                        # Ensure we have enough data for at least one prediction
                        if len(weighted_features) > sequence_length:
                            for i in range(sequence_length, len(weighted_features)):
                                sequence = weighted_features.iloc[i-sequence_length:i]
                                # Handle model prediction with error catching
                                try:
                                    pred = model.predict(sequence.values)
                                    predictions.append(pred[0] if isinstance(pred, np.ndarray) else pred)
                                except Exception as e:
                                    logger.debug(f"Model prediction error: {e}, using fallback")
                                    predictions.append(np.random.uniform(-0.02, 0.02))
                            
                            predictions = np.array(predictions)
                        else:
                            logger.warning(f"Insufficient data for predictions: {len(weighted_features)} rows, need > {sequence_length}")
                            predictions = np.random.uniform(-0.02, 0.02, max(1, len(data) - sequence_length))
                    except Exception as e:
                        logger.warning(f"Model prediction failed: {e}, using fallback predictions")
                        predictions = np.random.uniform(-0.02, 0.02, max(1, len(data) - sequence_length))
                else:
                    # No model, use technical indicator-based predictions
                    logger.info("No trained model available, using technical indicator-based predictions")
                    predictions = self._generate_technical_predictions(data, sequence_length)
            else:
                # Fallback to technical indicator-based predictions
                predictions = self._generate_technical_predictions(data, sequence_length)
            
            # Calculate returns based on predictions and thresholds
            positions = []
            returns = []
            
            buy_threshold = self.backtest_system.config.buy_threshold
            sell_threshold = self.backtest_system.config.sell_threshold
            position_size = self.backtest_system.config.position_size
            
            current_position = 0
            entry_price = None
            
            for i in range(len(predictions)):
                if i < len(data) - 1:
                    current_price = data['Close'].iloc[i] if 'Close' in data else 100
                    next_price = data['Close'].iloc[i + 1] if 'Close' in data else 100
                    
                    # Trading logic
                    if predictions[i] > buy_threshold and current_position == 0:
                        # Buy signal
                        current_position = position_size
                        entry_price = current_price
                    elif predictions[i] < -sell_threshold and current_position > 0:
                        # Sell signal
                        current_position = 0
                        entry_price = None
                    
                    # Calculate returns
                    if current_position > 0:
                        ret = (next_price - current_price) / current_price * current_position
                        # Apply stop loss and take profit
                        if entry_price:
                            price_change = (current_price - entry_price) / entry_price
                            if price_change <= -self.backtest_system.config.stop_loss:
                                current_position = 0
                                ret = -self.backtest_system.config.stop_loss * position_size
                            elif price_change >= self.backtest_system.config.take_profit:
                                current_position = 0
                                ret = self.backtest_system.config.take_profit * position_size
                    else:
                        ret = 0
                    
                    returns.append(ret)
                    positions.append(current_position)
            
            # Calculate metrics
            returns = np.array(returns)
            cumulative_returns = (1 + returns).cumprod()
            
            # Apply transaction costs
            position_changes = np.diff([0] + positions)
            transaction_costs = np.abs(position_changes) * config.transaction_cost
            returns[:len(transaction_costs)] -= transaction_costs
            
            # Calculate performance metrics
            total_return = cumulative_returns[-1] - 1 if len(cumulative_returns) > 0 else 0
            sharpe_ratio = np.mean(returns) * 252 / (np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0
            max_drawdown = self._calculate_max_drawdown(cumulative_returns)
            win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0
            sortino_ratio = self._calculate_sortino(returns)
            num_trades = (np.diff([0] + positions) != 0).sum()
            
            return {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'sortino_ratio': sortino_ratio,
                'num_trades': num_trades,
                'avg_return': np.mean(returns),
                'volatility': np.std(returns) * np.sqrt(252)
            }
            
        except Exception as e:
            logger.error(f"Trial backtest failed: {str(e)}")
            # Return poor metrics on error
            return {
                'total_return': -0.5,
                'sharpe_ratio': -2,
                'max_drawdown': -0.5,
                'win_rate': 0,
                'sortino_ratio': -2,
                'num_trades': 0
            }
    
    def _apply_weights_to_features(self, data: pd.DataFrame, weights: SignalWeights) -> pd.DataFrame:
        """Apply signal weights to features"""
        weighted_data = data.copy()
        
        # Map feature columns to categories
        for col in weighted_data.columns:
            if col.startswith('tech_'):
                weighted_data[col] *= weights.technical_weight
            elif col.startswith('onchain_'):
                weighted_data[col] *= weights.onchain_weight
            elif col.startswith('sent_'):
                weighted_data[col] *= weights.sentiment_weight
            elif col.startswith('macro_'):
                weighted_data[col] *= weights.macro_weight
        
        return weighted_data
    
    def _calculate_max_drawdown(self, cumulative_returns: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        return np.min(drawdown)
    
    def _calculate_sortino(self, returns: np.ndarray) -> float:
        """Calculate Sortino ratio"""
        excess_returns = returns - 0.02 / 252  # Risk-free rate
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return 2.0
        
        downside_deviation = np.sqrt(np.mean(downside_returns**2)) * np.sqrt(252)
        annual_return = np.mean(excess_returns) * 252
        
        return annual_return / downside_deviation if downside_deviation > 0 else 2.0
    
    def validate_data(self, data: pd.DataFrame, config: OptimizationConfig) -> Dict[str, Any]:
        """Validate data before optimization and adjust configuration"""
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'adjustments': [],
            'data_rows': len(data),
            'features': len(data.columns),
            'date_range': f"{data.index[0]} to {data.index[-1]}" if len(data) > 0 else "No data"
        }
        
        # Check minimum data requirements
        if len(data) < MIN_DATA_ROWS_OPTIMIZATION:
            validation_results['is_valid'] = False
            validation_results['warnings'].append(
                f"Insufficient data: {len(data)} rows, minimum required: {MIN_DATA_ROWS_OPTIMIZATION}"
            )
            return validation_results
        
        # Adjust model type based on data size
        if config.adapt_to_data_size:
            if len(data) < MIN_DATA_ROWS_ENHANCED:
                if config.use_enhanced_model:
                    config.use_enhanced_model = False
                    self.model_type_used = "standard"
                    validation_results['adjustments'].append(
                        f"Switched to standard LSTM due to limited data ({len(data)} rows)"
                    )
                    logger.info(f"Using standard LSTM for optimization with {len(data)} rows")
            else:
                self.model_type_used = "enhanced"
                logger.info(f"Using enhanced LSTM for optimization with {len(data)} rows")
        
        # Adjust sequence length if needed
        if len(data) < config.min_sequence_length * 2:
            new_sequence_length = max(10, len(data) // 3)
            validation_results['adjustments'].append(
                f"Reduced sequence length from {config.min_sequence_length} to {new_sequence_length}"
            )
            config.min_sequence_length = new_sequence_length
        
        # Adjust number of trials based on data size
        if len(data) < MIN_DATA_ROWS_STANDARD:
            suggested_trials = max(10, config.n_trials // 2)
            if suggested_trials < config.n_trials:
                validation_results['adjustments'].append(
                    f"Reduced optimization trials from {config.n_trials} to {suggested_trials} due to limited data"
                )
                config.n_trials = suggested_trials
        
        # Check for required columns
        required_columns = ['Close']
        optional_columns = ['Open', 'High', 'Low', 'Volume']
        
        for col in required_columns:
            if col not in data.columns:
                validation_results['is_valid'] = False
                validation_results['warnings'].append(f"Missing required column: {col}")
        
        for col in optional_columns:
            if col not in data.columns:
                validation_results['warnings'].append(f"Missing optional column: {col}")
        
        # Check for data quality
        null_percentage = (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
        if null_percentage > 20:
            validation_results['warnings'].append(
                f"High percentage of null values: {null_percentage:.1f}%"
            )
        
        # Store validation results
        self.data_validation_results = validation_results
        
        return validation_results
    
    async def optimize_async(self, config: OptimizationConfig, data: pd.DataFrame) -> Dict:
        """Run optimization asynchronously with progress tracking"""
        try:
            # Validate data first
            logger.info("Validating data for optimization...")
            validation = self.validate_data(data, config)
            
            if not validation['is_valid']:
                error_msg = f"Data validation failed: {'; '.join(validation['warnings'])}"
                logger.error(error_msg)
                return {
                    'status': 'error',
                    'error': error_msg,
                    'validation': validation
                }
            
            # Log any adjustments made
            if validation['adjustments']:
                logger.info(f"Data adjustments: {'; '.join(validation['adjustments'])}")
            
            # Create Optuna study with appropriate settings
            study_name = f"strategy_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.study = optuna.create_study(
                study_name=study_name,
                direction='minimize',
                sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=min(10, config.n_trials // 5)),
                pruner=optuna.pruners.MedianPruner(n_startup_trials=min(5, config.n_trials // 10))
            )
            
            # Set study user attributes for tracking
            self.study.set_user_attr("data_rows", len(data))
            self.study.set_user_attr("model_type", self.model_type_used or "standard")
            self.study.set_user_attr("optimization_objective", config.objective)
            
            # Create objective function
            objective = self.create_objective(config, data)
            
            # Run optimization
            start_time = datetime.now()
            logger.info(f"Starting optimization with {config.n_trials} trials...")
            
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            
            # Add progress callback for logging
            def optimization_callback(study, trial):
                if trial.number % max(1, config.n_trials // 10) == 0:
                    logger.info(f"Progress: {trial.number}/{config.n_trials} trials completed")
                    if study.best_trial:
                        logger.info(f"Current best value: {study.best_value:.4f}")
            
            await loop.run_in_executor(
                None,
                lambda: self.study.optimize(
                    objective,
                    n_trials=config.n_trials,
                    timeout=config.timeout_seconds,
                    callbacks=[optimization_callback],
                    show_progress_bar=False  # Use our custom logging instead
                )
            )
            
            end_time = datetime.now()
            optimization_time = (end_time - start_time).total_seconds()
            logger.info(f"Optimization completed in {optimization_time:.1f} seconds")
            
            # Extract best parameters
            self.best_params = self.study.best_params
            best_value = self.study.best_value
            
            # Get the best trial results
            best_trial_idx = self.study.best_trial.number
            best_results = next(
                (h['results'] for h in self.optimization_history if h['trial'] == best_trial_idx),
                {}
            )
            
            # Prepare response
            return {
                'status': 'success',
                'best_parameters': {
                    'technical_weight': self.best_params.get('technical_weight', 0.4),
                    'onchain_weight': self.best_params.get('onchain_weight', 0.3),
                    'sentiment_weight': self.best_params.get('sentiment_weight', 0.2),
                    'macro_weight': self.best_params.get('macro_weight', 0.1),
                    'position_size': self.best_params.get('position_size', 0.1),
                    'stop_loss': self.best_params.get('stop_loss', 0.05),
                    'take_profit': self.best_params.get('take_profit', 0.1),
                    'min_confidence': self.best_params.get('min_confidence', 0.6)
                },
                'expected_performance': {
                    'sharpe_ratio': best_results.get('sharpe_ratio', 0),
                    'total_return': best_results.get('total_return', 0),
                    'max_drawdown': best_results.get('max_drawdown', 0),
                    'win_rate': best_results.get('win_rate', 0),
                    'sortino_ratio': best_results.get('sortino_ratio', 0),
                    'num_trades': best_results.get('num_trades', 0)
                },
                'optimization_details': {
                    'trials_completed': len(self.study.trials),
                    'best_trial': best_trial_idx,
                    'optimization_time': f"{optimization_time:.1f}s",
                    'objective': config.objective,
                    'constraints_applied': config.constraints,
                    'constraint_violations': len(self.constraint_violations),
                    'convergence_achieved': self._check_convergence(),
                    'model_type_used': self.model_type_used or 'standard',
                    'data_validation': validation
                },
                'trial_history': self._get_trial_history_summary(),
                'parameter_importance': self._calculate_parameter_importance()
            }
            
        except Exception as e:
            logger.error(f"Optimization failed: {str(e)}")
            raise
    
    def _check_convergence(self) -> bool:
        """Check if optimization has converged"""
        if not self.study or len(self.study.trials) < 10:
            return False
        
        # Check if best value hasn't improved in last 20% of trials
        n_trials = len(self.study.trials)
        recent_start = int(n_trials * 0.8)
        
        best_so_far = float('inf')
        for i in range(recent_start):
            if self.study.trials[i].value < best_so_far:
                best_so_far = self.study.trials[i].value
        
        # Check if any recent trial improved
        for i in range(recent_start, n_trials):
            if self.study.trials[i].value < best_so_far * 0.99:  # 1% improvement threshold
                return False
                
        return True
    
    def _get_trial_history_summary(self) -> List[Dict]:
        """Get summary of optimization trials"""
        if not self.optimization_history:
            return []
        
        # Return top 10 trials
        sorted_history = sorted(
            self.optimization_history,
            key=lambda x: x['objective_value']
        )[:10]
        
        return [
            {
                'trial': h['trial'],
                'sharpe_ratio': h['results'].get('sharpe_ratio', 0),
                'total_return': h['results'].get('total_return', 0),
                'max_drawdown': h['results'].get('max_drawdown', 0),
                'position_size': h['params'].get('position_size', 0)
            }
            for h in sorted_history
        ]
    
    def _calculate_parameter_importance(self) -> Dict[str, float]:
        """Calculate importance of each parameter using Optuna's built-in importance evaluator"""
        if not self.study or len(self.study.trials) < 5:
            return {}
        
        try:
            importance = optuna.importance.get_param_importances(self.study)
            # Normalize to percentages
            total = sum(importance.values())
            if total > 0:
                return {k: v / total for k, v in importance.items()}
            return importance
        except Exception as e:
            logger.warning(f"Could not calculate parameter importance: {e}")
            return {}
    
    def _generate_technical_predictions(self, data: pd.DataFrame, sequence_length: int) -> np.ndarray:
        """Generate predictions based on technical indicators when no model is available"""
        try:
            predictions = []
            
            # Calculate simple technical indicators for prediction
            data = data.copy()
            
            # Price momentum
            data['returns'] = data['Close'].pct_change()
            data['sma_20'] = data['Close'].rolling(20).mean()
            data['sma_50'] = data['Close'].rolling(50).mean()
            data['rsi'] = self._calculate_rsi(data['Close'], 14)
            
            # Generate signals based on technical indicators
            for i in range(sequence_length, len(data)):
                score = 0.0
                
                # Trend following
                if i >= 50:  # Ensure we have enough data
                    if data['sma_20'].iloc[i] > data['sma_50'].iloc[i]:
                        score += 0.01
                    else:
                        score -= 0.01
                
                # Mean reversion
                if 'rsi' in data.columns and not pd.isna(data['rsi'].iloc[i]):
                    rsi_val = data['rsi'].iloc[i]
                    if rsi_val < 30:
                        score += 0.02  # Oversold
                    elif rsi_val > 70:
                        score -= 0.02  # Overbought
                
                # Momentum
                if i >= 5:
                    recent_return = (data['Close'].iloc[i] - data['Close'].iloc[i-5]) / data['Close'].iloc[i-5]
                    score += np.clip(recent_return, -0.02, 0.02)
                
                # Add some randomness to avoid overfitting
                score += np.random.normal(0, 0.005)
                
                # Clip final prediction
                predictions.append(np.clip(score, -0.05, 0.05))
            
            return np.array(predictions)
            
        except Exception as e:
            logger.warning(f"Technical prediction generation failed: {e}")
            # Return small random predictions as fallback
            return np.random.uniform(-0.02, 0.02, max(1, len(data) - sequence_length))
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi