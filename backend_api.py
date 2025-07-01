from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import pandas as pd
from datetime import datetime
import threading
import time
import os
import logging
import json
import glob
import asyncio
import numpy as np

from database_models import DatabaseManager
from lstm_model import TradingSignalGenerator
from integration import IntegratedBacktestingSystem
from backtesting_system import BacktestConfig, SignalWeights

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradeRequest(BaseModel):
    symbol: str
    trade_type: str
    price: float
    size: float
    lot_id: Optional[str] = None

class LimitOrder(BaseModel):
    symbol: str
    limit_type: str
    price: float
    size: Optional[float] = None
    lot_id: Optional[str] = None

app = FastAPI(title="BTC Trading System API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components with error handling
try:
    db_path = os.getenv('DATABASE_PATH', '/app/data/trading_system.db')
    db = DatabaseManager(db_path)
    logger.info(f"Database initialized at {db_path}")
except Exception as e:
    logger.error(f"Failed to initialize database: {e}")
    raise

try:
    signal_generator = TradingSignalGenerator()
    logger.info("Signal generator initialized")
except Exception as e:
    logger.error(f"Failed to initialize signal generator: {e}")
    raise

# Global variables for caching
latest_btc_data = None
latest_signal = None
signal_update_errors = 0
max_signal_errors = 5

# Backtesting global variables
backtest_system = None
backtest_in_progress = False
latest_backtest_results = None

class SignalUpdater:
    def __init__(self):
        self.running = False
        self.thread = None
        
    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self.update_signals_loop, daemon=True)
            self.thread.start()
            logger.info("Signal updater thread started")
            
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("Signal updater stopped")
        
    def update_signals_loop(self):
        """Continuously update trading signals with error handling"""
        global signal_update_errors
        
        while self.running:
            try:
                self.update_signals()
                signal_update_errors = 0  # Reset error count on success
                time.sleep(300)  # Update every 5 minutes
            except Exception as e:
                signal_update_errors += 1
                logger.error(f"Error updating signals (attempt {signal_update_errors}): {e}")
                
                # Exponential backoff with max errors check
                if signal_update_errors >= max_signal_errors:
                    logger.warning(f"Too many signal update errors ({signal_update_errors}), extending sleep time")
                    time.sleep(600)  # Wait 10 minutes after too many errors
                else:
                    time.sleep(60 * signal_update_errors)  # Gradually increase wait time
                
    def update_signals(self):
        """Update trading signals and store in database"""
        global latest_btc_data, latest_signal
        
        try:
            logger.info("Fetching BTC data for signal update...")
            btc_data = signal_generator.fetch_btc_data(period="3mo")
            
            if btc_data is None or len(btc_data) == 0:
                logger.warning("No BTC data received, using cached data if available")
                if latest_btc_data is not None:
                    btc_data = latest_btc_data
                else:
                    raise ValueError("No BTC data available and no cached data")
            
            latest_btc_data = btc_data
            logger.info(f"Successfully fetched {len(btc_data)} days of BTC data")
            
            # Generate signal
            logger.info("Generating trading signal...")
            signal, confidence, predicted_price = signal_generator.predict_signal(btc_data)
            
            # Store signal in database
            try:
                db.add_model_signal("BTC-USD", signal, confidence, predicted_price)
                logger.info("Signal stored in database")
            except Exception as e:
                logger.warning(f"Failed to store signal in database: {e}")
            
            # Update global latest signal
            latest_signal = {
                "symbol": "BTC-USD",
                "signal": signal,
                "confidence": confidence,
                "predicted_price": predicted_price,
                "timestamp": datetime.now()
            }
            
            logger.info(f"Signal updated: {signal} (confidence: {confidence:.2%}, price: ${predicted_price:.2f})")
            
        except Exception as e:
            logger.error(f"Error in update_signals: {e}")
            # Don't re-raise, let the loop continue
            
            # If we have no signal at all, create a default one
            if latest_signal is None:
                logger.info("Creating default signal due to errors")
                latest_signal = {
                    "symbol": "BTC-USD",
                    "signal": "hold",
                    "confidence": 0.5,
                    "predicted_price": 45000.0,
                    "timestamp": datetime.now()
                }

signal_updater = SignalUpdater()

def load_latest_backtest_results():
    """Load the most recent backtest results from file"""
    global latest_backtest_results
    try:
        # Find the most recent backtest results file
        result_files = glob.glob('backtest_results_*.json')
        if result_files:
            latest_file = max(result_files, key=os.path.getctime)
            with open(latest_file, 'r') as f:
                latest_backtest_results = json.load(f)
                logger.info(f"Loaded backtest results from {latest_file}")
    except Exception as e:
        logger.error(f"Failed to load backtest results: {e}")

@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    logger.info("Starting BTC Trading System API...")
    
    # Start signal updater
    try:
        signal_updater.start()
        logger.info("Signal updater started successfully")
    except Exception as e:
        logger.error(f"Failed to start signal updater: {e}")
    
    # Generate initial signal
    try:
        logger.info("Generating initial signal...")
        initial_data = signal_generator.fetch_btc_data(period="1mo")
        signal, confidence, predicted_price = signal_generator.predict_signal(initial_data)
        
        global latest_signal, latest_btc_data
        latest_btc_data = initial_data
        latest_signal = {
            "symbol": "BTC-USD",
            "signal": signal,
            "confidence": confidence,
            "predicted_price": predicted_price,
            "timestamp": datetime.now()
        }
        logger.info(f"Initial signal generated: {signal}")
        
    except Exception as e:
        logger.warning(f"Failed to generate initial signal: {e}")
        # Create a default signal
        latest_signal = {
            "symbol": "BTC-USD",
            "signal": "hold",
            "confidence": 0.5,
            "predicted_price": 45000.0,
            "timestamp": datetime.now()
        }
    
    # Initialize backtest system
    global backtest_system
    try:
        backtest_system = IntegratedBacktestingSystem(
            db_path=db_path,
            model_path='models/lstm_btc_model.pth'
        )
        logger.info("Backtest system initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize backtest system: {e}")
    
    # Load latest backtest results if available
    load_latest_backtest_results()
    
    logger.info("BTC Trading System API startup complete")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down BTC Trading System API...")
    signal_updater.stop()
    logger.info("API shutdown complete")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "BTC Trading System API is running", 
        "timestamp": datetime.now(),
        "status": "healthy",
        "signal_errors": signal_update_errors
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    try:
        # Check database
        db_status = "healthy"
        try:
            metrics = db.get_portfolio_metrics()
            db_status = "healthy"
        except Exception as e:
            db_status = f"error: {str(e)}"
        
        # Check signal generator
        signal_status = "healthy" if latest_signal else "no_signal"
        
        return {
            "status": "healthy",
            "timestamp": datetime.now(),
            "components": {
                "database": db_status,
                "signal_generator": signal_status,
                "signal_update_errors": signal_update_errors
            },
            "latest_signal": latest_signal
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.post("/trades/")
async def create_trade(trade: TradeRequest):
    """Create a new trade"""
    try:
        trade_id = db.add_trade(
            symbol=trade.symbol,
            trade_type=trade.trade_type,
            price=trade.price,
            size=trade.size,
            lot_id=trade.lot_id
        )
        logger.info(f"Trade created: {trade_id}")
        return {"trade_id": trade_id, "status": "success", "message": "Trade created successfully"}
    except Exception as e:
        logger.error(f"Failed to create trade: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/trades/")
async def get_trades(symbol: Optional[str] = None, limit: Optional[int] = 100):
    """Get trading history"""
    try:
        trades_df = db.get_trades(symbol=symbol, limit=limit)
        return trades_df.to_dict('records')
    except Exception as e:
        logger.error(f"Failed to get trades: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/positions/")
async def get_positions():
    """Get current positions"""
    try:
        positions_df = db.get_positions()
        return positions_df.to_dict('records')
    except Exception as e:
        logger.error(f"Failed to get positions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/limits/")
async def create_limit_order(limit_order: LimitOrder):
    """Create a limit order"""
    try:
        limit_id = db.add_trading_limit(
            symbol=limit_order.symbol,
            limit_type=limit_order.limit_type,
            price=limit_order.price,
            size=limit_order.size,
            lot_id=limit_order.lot_id
        )
        logger.info(f"Limit order created: {limit_id}")
        return {"limit_id": limit_id, "status": "success", "message": "Limit order created"}
    except Exception as e:
        logger.error(f"Failed to create limit order: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/limits/")
async def get_limits():
    """Get active limit orders"""
    try:
        limits_df = db.get_trading_limits()
        return limits_df.to_dict('records')
    except Exception as e:
        logger.error(f"Failed to get limits: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/signals/latest")
async def get_latest_signal():
    """Get the latest trading signal"""
    global latest_signal
    
    try:
        if latest_signal is None:
            logger.info("No cached signal, generating new one...")
            try:
                btc_data = signal_generator.fetch_btc_data(period="1mo")
                signal, confidence, predicted_price = signal_generator.predict_signal(btc_data)
                
                latest_signal = {
                    "symbol": "BTC-USD",
                    "signal": signal,
                    "confidence": confidence,
                    "predicted_price": predicted_price,
                    "timestamp": datetime.now()
                }
                logger.info(f"Generated new signal: {signal}")
            except Exception as e:
                logger.error(f"Failed to generate signal: {e}")
                # Return a default signal
                latest_signal = {
                    "symbol": "BTC-USD",
                    "signal": "hold",
                    "confidence": 0.5,
                    "predicted_price": 45000.0,
                    "timestamp": datetime.now(),
                    "error": "Signal generation failed, using default"
                }
        
        return latest_signal
        
    except Exception as e:
        logger.error(f"Failed to get latest signal: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting signal: {str(e)}")

@app.get("/signals/history")
async def get_signal_history(limit: int = 50):
    """Get historical trading signals"""
    try:
        signals_df = db.get_model_signals(limit=limit)
        return signals_df.to_dict('records')
    except Exception as e:
        logger.error(f"Failed to get signal history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/portfolio/metrics")
async def get_portfolio_metrics():
    """Get portfolio performance metrics"""
    try:
        metrics = db.get_portfolio_metrics()
        
        # Add current BTC price if available
        if latest_btc_data is not None and len(latest_btc_data) > 0:
            try:
                metrics['current_btc_price'] = float(latest_btc_data['Close'].iloc[-1])
            except Exception as e:
                logger.warning(f"Failed to get current BTC price: {e}")
                metrics['current_btc_price'] = None
        else:
            metrics['current_btc_price'] = None
        
        return metrics
    except Exception as e:
        logger.error(f"Failed to get portfolio metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/market/btc-data")
async def get_btc_data(period: str = "1mo"):
    """Get BTC market data"""
    try:
        logger.info(f"Fetching BTC data for period: {period}")
        btc_data = signal_generator.fetch_btc_data(period=period)
        
        if btc_data is None or len(btc_data) == 0:
            raise ValueError("No BTC data available")
        
        data_records = []
        for idx, row in btc_data.iterrows():
            try:
                record = {
                    'timestamp': idx.isoformat(),
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close']),
                    'volume': float(row['Volume']),
                    'sma_20': float(row['SMA_20']) if not pd.isna(row['SMA_20']) else None,
                    'sma_50': float(row['SMA_50']) if not pd.isna(row['SMA_50']) else None,
                    'rsi': float(row['RSI']) if not pd.isna(row['RSI']) else None,
                    'macd': float(row['MACD']) if not pd.isna(row['MACD']) else None
                }
                data_records.append(record)
            except Exception as e:
                logger.warning(f"Error processing data row {idx}: {e}")
                continue
        
        # Return last 100 records to avoid large responses
        return {
            "symbol": "BTC-USD",
            "period": period,
            "data": data_records[-100:] if len(data_records) > 100 else data_records,
            "total_records": len(data_records)
        }
        
    except Exception as e:
        logger.error(f"Failed to get BTC data: {e}")
        
        # Return dummy data as fallback
        try:
            dummy_data = signal_generator.generate_dummy_data()
            dummy_records = []
            
            for idx, row in dummy_data.tail(50).iterrows():  # Last 50 days
                record = {
                    'timestamp': idx.isoformat(),
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close']),
                    'volume': float(row['Volume']),
                    'sma_20': float(row['SMA_20']) if not pd.isna(row['SMA_20']) else None,
                    'sma_50': float(row['SMA_50']) if not pd.isna(row['SMA_50']) else None,
                    'rsi': float(row['RSI']) if not pd.isna(row['RSI']) else None,
                    'macd': float(row['MACD']) if not pd.isna(row['MACD']) else None
                }
                dummy_records.append(record)
            
            return {
                "symbol": "BTC-USD",
                "period": period,
                "data": dummy_records,
                "total_records": len(dummy_records),
                "note": "Using simulated data due to data fetch error"
            }
            
        except Exception as dummy_error:
            logger.error(f"Failed to generate dummy data: {dummy_error}")
            raise HTTPException(status_code=500, detail="Unable to fetch or generate BTC data")

@app.get("/analytics/pnl")
async def get_pnl_data():
    """Get P&L analytics data"""
    try:
        trades_df = db.get_trades()
        
        if trades_df.empty:
            return {"daily_pnl": [], "cumulative_pnl": []}
        
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
        trades_df['date'] = trades_df['timestamp'].dt.date
        trades_df['trade_value'] = trades_df['price'] * trades_df['size']
        trades_df['signed_value'] = trades_df['trade_value'] * trades_df['trade_type'].map({
            'buy': -1, 
            'sell': 1, 
            'hold': 0
        })
        
        daily_pnl = trades_df.groupby('date')['signed_value'].sum().reset_index()
        daily_pnl['cumulative_pnl'] = daily_pnl['signed_value'].cumsum()
        
        return {
            "daily_pnl": [
                {"date": str(row['date']), "pnl": float(row['signed_value'])}
                for _, row in daily_pnl.iterrows()
            ],
            "cumulative_pnl": [
                {"date": str(row['date']), "pnl": float(row['cumulative_pnl'])}
                for _, row in daily_pnl.iterrows()
            ]
        }
    except Exception as e:
        logger.error(f"Failed to get P&L data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/system/status")
async def get_system_status():
    """Get detailed system status"""
    try:
        return {
            "api_status": "running",
            "timestamp": datetime.now(),
            "signal_update_errors": signal_update_errors,
            "latest_signal_time": latest_signal.get('timestamp') if latest_signal else None,
            "data_cache_status": "available" if latest_btc_data is not None else "empty",
            "database_status": "connected",
            "signal_generator_status": "initialized"
        }
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/backtest/")
async def run_backtest(config: BacktestConfig):
    """Run backtesting simulation"""
    try:
        logger.info(f"Running backtest with config: {config}")
        
        # Fetch historical data
        btc_data = signal_generator.fetch_btc_data(period=config.period)
        if btc_data is None or len(btc_data) < signal_generator.sequence_length:
            raise ValueError("Insufficient data for backtesting")
        
        # Prepare features for the model
        features_df = signal_generator.prepare_features(btc_data)
        logger.info(f"Features shape: {features_df.shape}")  # Debug log
        
        # Extract just the price series for return calculations
        prices = features_df['price'].values
        logger.info(f"Prices shape: {prices.shape}")  # Debug log
        
        # Ensure prices is 1D
        if len(prices.shape) > 1:
            logger.warning(f"Prices array is not 1D: {prices.shape}, flattening...")
            prices = prices.flatten()
        
        # Initialize backtest results
        results = {
            'trades': [],
            'equity_curve': [float(config.initial_capital)],  # Ensure float
            'timestamps': [btc_data.index[signal_generator.sequence_length]],
            'positions': [],
            'signals': []
        }
        
        # Backtest variables
        capital = float(config.initial_capital)
        position = 0.0
        entry_price = 0.0
        
        # Run through historical data
        for i in range(signal_generator.sequence_length, len(features_df)):
            current_price = float(prices[i])  # Ensure float
            current_time = btc_data.index[i]
            
            # Get model signal using the feature window
            try:
                # Create a subset of data up to current point
                historical_data = btc_data.iloc[:i+1].copy()
                signal, confidence, predicted_price = signal_generator.predict_signal(historical_data)
            except Exception as e:
                logger.warning(f"Signal generation failed at index {i}: {e}")
                signal = "hold"
                confidence = 0.5
                predicted_price = current_price
            
            results['signals'].append({
                'timestamp': current_time,
                'signal': signal,
                'confidence': float(confidence),
                'predicted_price': float(predicted_price),
                'actual_price': float(current_price)
            })
            
            # Execute trading logic
            if position == 0 and signal == "buy" and confidence >= config.confidence_threshold:
                # Open long position
                position_value = capital * config.position_size
                position = position_value / current_price
                entry_price = current_price
                capital -= position_value
                
                results['trades'].append({
                    'timestamp': current_time,
                    'type': 'buy',
                    'price': float(current_price),
                    'size': float(position),
                    'value': float(position_value),
                    'capital_after': float(capital)
                })
                
            elif position > 0:
                # Check exit conditions
                current_value = position * current_price
                pnl_pct = (current_price - entry_price) / entry_price
                
                should_sell = False
                exit_reason = ""
                
                if signal == "sell" and confidence >= config.confidence_threshold:
                    should_sell = True
                    exit_reason = "signal"
                elif pnl_pct <= -config.stop_loss:
                    should_sell = True
                    exit_reason = "stop_loss"
                elif pnl_pct >= config.take_profit:
                    should_sell = True
                    exit_reason = "take_profit"
                
                if should_sell:
                    # Close position
                    capital += current_value
                    
                    results['trades'].append({
                        'timestamp': current_time,
                        'type': 'sell',
                        'price': float(current_price),
                        'size': float(position),
                        'value': float(current_value),
                        'pnl': float(current_value - (position * entry_price)),
                        'pnl_pct': float(pnl_pct),
                        'exit_reason': exit_reason,
                        'capital_after': float(capital)
                    })
                    
                    position = 0
                    entry_price = 0
            
            # Update equity curve
            total_equity = capital + (position * current_price if position > 0 else 0)
            results['equity_curve'].append(float(total_equity))
            results['timestamps'].append(current_time)
            results['positions'].append(float(position))
        
        # Calculate final metrics
        final_equity = results['equity_curve'][-1]
        total_return = (final_equity - config.initial_capital) / config.initial_capital
        
        # Calculate performance metrics with careful array handling
        equity_array = np.array(results['equity_curve'], dtype=np.float64)
        logger.info(f"Equity array shape: {equity_array.shape}")  # Debug log
        
        # Ensure equity_array is 1D
        if len(equity_array.shape) > 1:
            logger.warning(f"Equity array is not 1D: {equity_array.shape}, flattening...")
            equity_array = equity_array.flatten()
        
        if len(equity_array) > 1:
            # Calculate returns carefully
            returns = np.diff(equity_array) / equity_array[:-1]
            logger.info(f"Returns shape: {returns.shape}")  # Debug log
            
            # Ensure returns is 1D
            if len(returns.shape) > 1:
                logger.warning(f"Returns array is not 1D: {returns.shape}, flattening...")
                returns = returns.flatten()
            
            # Handle any potential NaN or inf values
            returns = returns[np.isfinite(returns)]
            
            if len(returns) > 0:
                sharpe_ratio = np.sqrt(252) * (np.mean(returns) / np.std(returns)) if np.std(returns) > 0 else 0
                
                # Calculate max drawdown carefully
                cummax = np.maximum.accumulate(equity_array)
                drawdown = (equity_array - cummax) / cummax
                max_drawdown = np.min(drawdown)
                
                # Calculate win rate
                sell_trades = [t for t in results['trades'] if t.get('type') == 'sell']
                if len(sell_trades) > 0:
                    winning_trades = [t for t in sell_trades if t.get('pnl', 0) > 0]
                    win_rate = len(winning_trades) / len(sell_trades)
                else:
                    win_rate = 0
            else:
                sharpe_ratio = 0
                max_drawdown = 0
                win_rate = 0
        else:
            sharpe_ratio = 0
            max_drawdown = 0
            win_rate = 0
        
        # Summary statistics
        results['summary'] = {
            'initial_capital': float(config.initial_capital),
            'final_equity': float(final_equity),
            'total_return': float(total_return),
            'total_return_pct': float(total_return * 100),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown),
            'max_drawdown_pct': float(max_drawdown * 100),
            'total_trades': len([t for t in results['trades'] if t['type'] == 'buy']),
            'win_rate': float(win_rate),
            'avg_trade_return': float(np.mean([t.get('pnl_pct', 0) for t in results['trades'] if t.get('type') == 'sell' and 'pnl_pct' in t])) if any(t.get('type') == 'sell' for t in results['trades']) else 0
        }
        
        logger.info(f"Backtest completed: {results['summary']}")
        return results
        
    except Exception as e:
        logger.error(f"Backtest failed with full traceback: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Backtest failed: {str(e)}")

@app.post("/backtest/run")
async def run_backtest_advanced(request: dict):
    """Run advanced backtest with IntegratedBacktestingSystem"""
    global backtest_in_progress, latest_backtest_results
    
    # Extract parameters from request
    period = request.get("period", "1y")
    optimize_weights = request.get("optimize_weights", True)
    force = request.get("force", False)
    
    # Check if backtest is already running
    if backtest_in_progress and not force:
        return {
            "status": "error",
            "message": "Backtest already in progress. Set 'force': true to override."
        }
    
    if not backtest_system:
        return {
            "status": "error",
            "message": "Backtest system not initialized. Please restart the API."
        }
    
    try:
        backtest_in_progress = True
        logger.info(f"Starting advanced backtest with period={period}, optimize_weights={optimize_weights}")
        
        # Fetch and validate data first
        try:
            test_data = signal_generator.fetch_btc_data(period=period)
            if test_data is None or len(test_data) < signal_generator.sequence_length:
                raise ValueError(f"Insufficient data for period {period}. Got {len(test_data) if test_data is not None else 0} records, need at least {signal_generator.sequence_length}")
            
            # Test feature preparation to catch any shape issues early
            test_features = signal_generator.prepare_features(test_data)
            logger.info(f"Test features shape: {test_features.shape}")
            
            # Ensure the model is properly initialized
            if not signal_generator.is_trained:
                logger.info("Model not trained, training before backtest...")
                signal_generator.train_model(test_data)
                
        except Exception as e:
            logger.error(f"Data preparation failed: {e}", exc_info=True)
            raise ValueError(f"Failed to prepare data: {str(e)}")
        
        # Run backtest in background to avoid blocking
        loop = asyncio.get_event_loop()
        
        # Wrap the backtest execution with additional error handling
        async def run_with_timeout():
            try:
                # Set a reasonable timeout (e.g., 30 minutes for comprehensive backtest)
                return await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        backtest_system.run_comprehensive_backtest,
                        period,
                        optimize_weights
                    ),
                    timeout=1800  # 30 minutes
                )
            except asyncio.TimeoutError:
                raise TimeoutError("Backtest timed out after 30 minutes")
            except Exception as e:
                logger.error(f"Backtest execution error: {e}", exc_info=True)
                raise
        
        results = await run_with_timeout()
        
        # Validate results structure
        if not isinstance(results, dict):
            raise ValueError(f"Invalid results type: {type(results)}")
        
        # Store results
        latest_backtest_results = results
        
        # Ensure all numeric values are properly formatted
        summary = {
            "composite_score": float(results.get('composite_score', 0)),
            "sortino_ratio": float(results.get('performance_metrics', {}).get('sortino_ratio_mean', 0)),
            "max_drawdown": float(results.get('performance_metrics', {}).get('max_drawdown_mean', 0)),
            "total_return": float(results.get('performance_metrics', {}).get('total_return_mean', 0)),
            "sharpe_ratio": float(results.get('performance_metrics', {}).get('sharpe_ratio_mean', 0)),
            "win_rate": float(results.get('performance_metrics', {}).get('win_rate_mean', 0)),
            "total_trades": int(results.get('performance_metrics', {}).get('total_trades', 0)),
            "timestamp": datetime.now().isoformat()
        }
        
        # Save results to file for persistence
        try:
            filename = f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump({
                    "timestamp": summary["timestamp"],
                    "parameters": {
                        "period": period,
                        "optimize_weights": optimize_weights
                    },
                    "performance_metrics": results.get('performance_metrics', {}),
                    "optimal_weights": results.get('optimal_weights', {}),
                    "risk_assessment": results.get('risk_assessment', {}),
                    "recommendations": results.get('recommendations', []),
                    "summary": summary
                }, f, indent=2)
            logger.info(f"Backtest results saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save backtest results: {e}")
        
        return {
            "status": "success",
            "message": "Backtest completed successfully",
            "summary": summary,
            "details": {
                "risk_assessment": results.get('risk_assessment', {}),
                "recommendations": results.get('recommendations', []),
                "optimal_weights": results.get('optimal_weights', {}) if optimize_weights else None
            }
        }
        
    except TimeoutError as e:
        logger.error(f"Backtest timeout: {e}")
        return {
            "status": "error",
            "message": str(e),
            "suggestion": "Try with a shorter period or disable weight optimization"
        }
    except ValueError as e:
        logger.error(f"Backtest validation error: {e}")
        return {
            "status": "error", 
            "message": f"Validation error: {str(e)}"
        }
    except Exception as e:
        logger.error(f"Backtest failed with unexpected error: {e}", exc_info=True)
        return {
            "status": "error",
            "message": f"Backtest failed: {str(e)}",
            "error_type": type(e).__name__
        }
    finally:
        backtest_in_progress = False
        logger.info("Backtest process completed (success or failure)")

# Also update the Streamlit interface to use the correct endpoint
# In streamlit_app.py, update the show_run_backtest() function to pass data correctly:

def show_run_backtest():
    """Interface to run new backtests"""
    st.subheader("🚀 Run New Backtest")
    
    # Check if backtest is in progress
    status = fetch_api_data("/backtest/status")
    
    if status and status.get('in_progress'):
        st.warning("⏳ Backtest is currently in progress. Please wait...")
        
        # Add a progress bar
        progress_bar = st.progress(0)
        placeholder = st.empty()
        
        # Poll for completion
        for i in range(100):
            time.sleep(3)  # Check every 3 seconds
            new_status = fetch_api_data("/backtest/status")
            if not new_status.get('in_progress'):
                progress_bar.progress(100)
                placeholder.success("✅ Backtest completed!")
                st.rerun()
                break
            progress_bar.progress(min(i + 1, 99))
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Backtest Parameters")
            
            period = st.selectbox(
                "Data Period",
                ["1mo", "3mo", "6mo", "1y", "2y", "3y"],
                index=3,
                help="Historical data period for backtesting"
            )
            
            optimize_weights = st.checkbox(
                "Optimize Signal Weights",
                value=True,
                help="Use Bayesian optimization to find optimal feature weights"
            )
            
            # Advanced settings in expander
            with st.expander("Advanced Settings"):
                force_run = st.checkbox(
                    "Force Run",
                    value=False,
                    help="Force backtest even if one is in progress"
                )
        
        with col2:
            st.markdown("### Expected Outcomes")
            st.info("""
            **What the backtest will do:**
            
            1. **Walk-Forward Analysis**
               - Train on historical windows
               - Test on future data
               - Prevent look-ahead bias
            
            2. **Optimization** (if enabled)
               - Find optimal signal weights
               - Balance risk vs return
               - ~50 optimization trials
            
            3. **Performance Evaluation**
               - Calculate Sortino ratio
               - Measure maximum drawdown
               - Generate recommendations
            
            ⏱️ **Estimated time**: 5-15 minutes
            """)
        
        # Run backtest button
        if st.button("🎯 Run Backtest", type="primary", use_container_width=True):
            with st.spinner("Initializing backtest..."):
                # Use the correct data structure for the request
                request_data = {
                    "period": period,
                    "optimize_weights": optimize_weights,
                    "force": force_run if 'force_run' in locals() else False
                }
                
                result = post_api_data("/backtest/run", request_data)
                
                if result:
                    if result.get('status') == 'success':
                        st.success("✅ Backtest completed successfully!")
                        st.balloons()
                        
                        # Display summary
                        summary = result.get('summary', {})
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Composite Score", f"{summary.get('composite_score', 0):.3f}")
                        with col2:
                            st.metric("Sortino Ratio", f"{summary.get('sortino_ratio', 0):.2f}")
                        with col3:
                            st.metric("Max Drawdown", f"{summary.get('max_drawdown', 0):.2%}")
                        
                        # Show additional details if available
                        details = result.get('details', {})
                        if details.get('recommendations'):
                            st.markdown("### 💡 Recommendations")
                            for rec in details['recommendations']:
                                st.info(f"• {rec}")
                        
                        st.info("View detailed results in the 'Backtest Results' tab")
                    else:
                        error_msg = result.get('message', 'Unknown error')
                        st.error(f"❌ Backtest failed: {error_msg}")
                        
                        # Show suggestion if available
                        if result.get('suggestion'):
                            st.warning(f"💡 Suggestion: {result['suggestion']}")
                else:
                    st.error("❌ Failed to connect to backtest service")

@app.get("/backtest/status")
async def get_backtest_status():
    """Get current backtest status"""
    return {
        "in_progress": backtest_in_progress,
        "has_results": latest_backtest_results is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/backtest/results/latest")
async def get_latest_backtest_results():
    """Get the most recent backtest results"""
    if latest_backtest_results is None:
        load_latest_backtest_results()
    
    if latest_backtest_results is None:
        raise HTTPException(status_code=404, detail="No backtest results available")
    
    # Format results for frontend
    results = {
        "timestamp": latest_backtest_results.get('timestamp', 'Unknown'),
        "performance_metrics": latest_backtest_results.get('performance_metrics', {}),
        "optimal_weights": latest_backtest_results.get('optimal_weights', {}),
        "risk_assessment": latest_backtest_results.get('risk_assessment', {}),
        "recommendations": latest_backtest_results.get('recommendations', [])
    }
    
    return results

@app.get("/backtest/results/history")
async def get_backtest_history(limit: int = 10):
    """Get historical backtest results"""
    try:
        # Find all backtest result files
        result_files = glob.glob('backtest_results_*.json')
        result_files.sort(key=os.path.getctime, reverse=True)
        
        history = []
        for file_path in result_files[:limit]:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    history.append({
                        "filename": os.path.basename(file_path),
                        "timestamp": data.get('timestamp', 'Unknown'),
                        "composite_score": data.get('performance_metrics', {}).get('composite_score', 0),
                        "sortino_ratio": data.get('performance_metrics', {}).get('sortino_ratio_mean', 0),
                        "max_drawdown": data.get('performance_metrics', {}).get('max_drawdown_mean', 0)
                    })
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")
                
        return history
        
    except Exception as e:
        logger.error(f"Failed to get backtest history: {e}")
        return []

@app.get("/config/signal-weights")
async def get_signal_weights():
    """Get current signal weights"""
    try:
        if backtest_system and hasattr(backtest_system.signal_generator, 'signal_weights'):
            weights = backtest_system.signal_generator.signal_weights
            return {
                "technical": weights.technical_weight,
                "onchain": weights.onchain_weight,
                "sentiment": weights.sentiment_weight,
                "macro": weights.macro_weight
            }
        else:
            # Return default weights
            return {
                "technical": 0.40,
                "onchain": 0.35,
                "sentiment": 0.15,
                "macro": 0.10
            }
    except Exception as e:
        logger.error(f"Failed to get signal weights: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/config/signal-weights")
async def update_signal_weights(weights: Dict[str, float]):
    """Manually update signal weights"""
    try:
        if backtest_system:
            sw = SignalWeights(
                technical_weight=weights.get('technical', 0.40),
                onchain_weight=weights.get('onchain', 0.35),
                sentiment_weight=weights.get('sentiment', 0.15),
                macro_weight=weights.get('macro', 0.10)
            )
            sw.normalize()
            backtest_system.signal_generator.signal_weights = sw
            
            logger.info(f"Signal weights updated: {weights}")
            return {"status": "success", "message": "Signal weights updated"}
        else:
            raise HTTPException(status_code=500, detail="Backtest system not initialized")
            
    except Exception as e:
        logger.error(f"Failed to update signal weights: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/model/retrain")
async def trigger_model_retrain():
    """Manually trigger model retraining"""
    try:
        if not backtest_system:
            raise HTTPException(status_code=500, detail="Backtest system not initialized")
            
        # Run retraining in background
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, backtest_system.retrain_model)
        
        return {
            "status": "success",
            "message": "Model retraining completed",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Model retraining failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/config/backtest-settings")
async def get_backtest_settings():
    """Get current backtest configuration"""
    try:
        if backtest_system:
            config = backtest_system.config
            return {
                "training_window_days": config.training_window_days,
                "test_window_days": config.test_window_days,
                "purge_days": config.purge_days,
                "retraining_frequency_days": config.retraining_frequency_days,
                "transaction_cost": config.transaction_cost,
                "max_drawdown_threshold": config.max_drawdown_threshold,
                "target_sortino_ratio": config.target_sortino_ratio
            }
        else:
            # Return defaults
            return {
                "training_window_days": 1008,
                "test_window_days": 90,
                "purge_days": 2,
                "retraining_frequency_days": 90,
                "transaction_cost": 0.0025,
                "max_drawdown_threshold": 0.25,
                "target_sortino_ratio": 2.0
            }
    except Exception as e:
        logger.error(f"Failed to get backtest settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting BTC Trading System API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)