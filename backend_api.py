from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import threading
import time
import os
import logging

from database_models import DatabaseManager
from lstm_model import EnhancedTradingSignalGenerator

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

app = FastAPI(title="BTC Trading System API", version="2.0.0")

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
    signal_generator = EnhancedTradingSignalGenerator()
    logger.info("Enhanced signal generator initialized")
except Exception as e:
    logger.error(f"Failed to initialize signal generator: {e}")
    raise

# Global variables for caching
latest_btc_data = None
latest_signal = None
signal_update_errors = 0
max_signal_errors = 5
portfolio_analytics_cache = {}

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
            btc_data = signal_generator.fetch_btc_data(period="3mo", interval="1h")
            
            if btc_data is None or len(btc_data) == 0:
                logger.warning("No BTC data received, using cached data if available")
                if latest_btc_data is not None:
                    btc_data = latest_btc_data
                else:
                    raise ValueError("No BTC data available and no cached data")
            
            latest_btc_data = btc_data
            logger.info(f"Successfully fetched {len(btc_data)} periods of BTC data")
            
            # Generate signal
            logger.info("Generating trading signal...")
            signal, confidence, predicted_price, factors = signal_generator.predict_signal(btc_data)
            
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
                "factors": factors,
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
                    "factors": {},
                    "timestamp": datetime.now()
                }

signal_updater = SignalUpdater()

def calculate_portfolio_metrics(trades_df: pd.DataFrame, positions_df: pd.DataFrame, 
                              current_price: float = None) -> Dict[str, Any]:
    """Calculate comprehensive portfolio metrics"""
    metrics = {}
    
    if trades_df.empty:
        return {
            'total_trades': 0,
            'total_volume': 0,
            'total_pnl': 0,
            'realized_pnl': 0,
            'unrealized_pnl': 0,
            'positions_count': 0,
            'total_invested': 0,
            'win_rate': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'current_value': 0
        }
    
    # Basic metrics
    metrics['total_trades'] = len(trades_df)
    metrics['total_volume'] = trades_df['size'].sum()
    
    # Calculate PnL
    buy_trades = trades_df[trades_df['trade_type'] == 'buy']
    sell_trades = trades_df[trades_df['trade_type'] == 'sell']
    
    total_bought = (buy_trades['price'] * buy_trades['size']).sum()
    total_sold = (sell_trades['price'] * sell_trades['size']).sum()
    
    metrics['realized_pnl'] = total_sold - total_bought
    
    # Calculate unrealized PnL for open positions
    metrics['unrealized_pnl'] = 0
    metrics['current_value'] = 0
    
    if not positions_df.empty and current_price:
        total_position_size = positions_df['total_size'].sum()
        avg_cost_basis = (positions_df['total_size'] * positions_df['avg_buy_price']).sum() / total_position_size
        metrics['current_value'] = total_position_size * current_price
        metrics['unrealized_pnl'] = metrics['current_value'] - (total_position_size * avg_cost_basis)
    
    metrics['total_pnl'] = metrics['realized_pnl'] + metrics['unrealized_pnl']
    metrics['positions_count'] = len(positions_df)
    metrics['total_invested'] = total_bought
    
    # Calculate win rate and average win/loss
    closed_trades = []
    for lot_id in trades_df['lot_id'].unique():
        lot_trades = trades_df[trades_df['lot_id'] == lot_id]
        lot_buys = lot_trades[lot_trades['trade_type'] == 'buy']
        lot_sells = lot_trades[lot_trades['trade_type'] == 'sell']
        
        if not lot_sells.empty and not lot_buys.empty:
            buy_avg = (lot_buys['price'] * lot_buys['size']).sum() / lot_buys['size'].sum()
            sell_avg = (lot_sells['price'] * lot_sells['size']).sum() / lot_sells['size'].sum()
            pnl_pct = (sell_avg - buy_avg) / buy_avg
            closed_trades.append(pnl_pct)
    
    if closed_trades:
        wins = [t for t in closed_trades if t > 0]
        losses = [t for t in closed_trades if t < 0]
        
        metrics['win_rate'] = len(wins) / len(closed_trades)
        metrics['avg_win'] = np.mean(wins) if wins else 0
        metrics['avg_loss'] = np.mean(losses) if losses else 0
    else:
        metrics['win_rate'] = 0
        metrics['avg_win'] = 0
        metrics['avg_loss'] = 0
    
    # Calculate Sharpe ratio and max drawdown
    if len(trades_df) > 10:
        # Create daily returns series
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
        daily_pnl = trades_df.set_index('timestamp').resample('D').apply(
            lambda x: ((x['price'] * x['size'] * x['trade_type'].map({'buy': -1, 'sell': 1})).sum())
        )
        
        if len(daily_pnl) > 5:
            returns = daily_pnl.pct_change().dropna()
            if len(returns) > 0:
                metrics['sharpe_ratio'] = np.sqrt(252) * returns.mean() / (returns.std() + 1e-8)
            else:
                metrics['sharpe_ratio'] = 0
                
            # Calculate max drawdown
            cumulative = daily_pnl.cumsum()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / (running_max + 1)
            metrics['max_drawdown'] = drawdown.min()
        else:
            metrics['sharpe_ratio'] = 0
            metrics['max_drawdown'] = 0
    else:
        metrics['sharpe_ratio'] = 0
        metrics['max_drawdown'] = 0
    
    return metrics

@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    logger.info("Starting Enhanced BTC Trading System API...")
    
    # Start signal updater
    try:
        signal_updater.start()
        logger.info("Signal updater started successfully")
    except Exception as e:
        logger.error(f"Failed to start signal updater: {e}")
    
    # Generate initial signal
    try:
        logger.info("Generating initial signal...")
        initial_data = signal_generator.fetch_btc_data(period="1mo", interval="1h")
        signal, confidence, predicted_price, factors = signal_generator.predict_signal(initial_data)
        
        global latest_signal, latest_btc_data
        latest_btc_data = initial_data
        latest_signal = {
            "symbol": "BTC-USD",
            "signal": signal,
            "confidence": confidence,
            "predicted_price": predicted_price,
            "factors": factors,
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
            "factors": {},
            "timestamp": datetime.now()
        }
    
    logger.info("Enhanced BTC Trading System API startup complete")

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
        "message": "Enhanced BTC Trading System API is running", 
        "version": "2.0.0",
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
        
        # Add current value and unrealized PnL
        if not positions_df.empty and latest_btc_data is not None:
            current_price = latest_btc_data['Close'].iloc[-1]
            positions_df['current_value'] = positions_df['total_size'] * current_price
            positions_df['unrealized_pnl'] = positions_df['current_value'] - (positions_df['total_size'] * positions_df['avg_buy_price'])
            positions_df['unrealized_pnl_pct'] = positions_df['unrealized_pnl'] / (positions_df['total_size'] * positions_df['avg_buy_price'])
        
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
    """Get the latest trading signal with detailed factors"""
    global latest_signal
    
    try:
        if latest_signal is None:
            logger.info("No cached signal, generating new one...")
            try:
                btc_data = signal_generator.fetch_btc_data(period="1mo", interval="1h")
                signal, confidence, predicted_price, factors = signal_generator.predict_signal(btc_data)
                
                latest_signal = {
                    "symbol": "BTC-USD",
                    "signal": signal,
                    "confidence": confidence,
                    "predicted_price": predicted_price,
                    "factors": factors,
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
                    "factors": {},
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
    """Get comprehensive portfolio performance metrics"""
    try:
        trades_df = db.get_trades()
        positions_df = db.get_positions()
        
        # Get current BTC price
        current_price = None
        if latest_btc_data is not None and len(latest_btc_data) > 0:
            try:
                current_price = float(latest_btc_data['Close'].iloc[-1])
            except Exception as e:
                logger.warning(f"Failed to get current BTC price: {e}")
        
        # Calculate comprehensive metrics
        metrics = calculate_portfolio_metrics(trades_df, positions_df, current_price)
        metrics['current_btc_price'] = current_price
        
        return metrics
        
    except Exception as e:
        logger.error(f"Failed to get portfolio metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/market/btc-data")
async def get_btc_data(period: str = "1mo", interval: str = "1h"):
    """Get BTC market data with technical indicators"""
    try:
        logger.info(f"Fetching BTC data for period: {period}, interval: {interval}")
        btc_data = signal_generator.fetch_btc_data(period=period, interval=interval)
        
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
                    # Technical indicators
                    'sma_20': float(row['SMA_20']) if not pd.isna(row['SMA_20']) else None,
                    'sma_50': float(row['SMA_50']) if not pd.isna(row['SMA_50']) else None,
                    'rsi': float(row['RSI']) if not pd.isna(row['RSI']) else None,
                    'macd': float(row['MACD']) if not pd.isna(row['MACD']) else None,
                    'macd_signal': float(row['MACD_Signal']) if not pd.isna(row['MACD_Signal']) else None,
                    # New indicators
                    'bb_upper': float(row['BB_Upper']) if not pd.isna(row['BB_Upper']) else None,
                    'bb_lower': float(row['BB_Lower']) if not pd.isna(row['BB_Lower']) else None,
                    'bb_position': float(row['BB_Position']) if not pd.isna(row['BB_Position']) else None,
                    'atr': float(row['ATR']) if not pd.isna(row['ATR']) else None,
                    'stoch_k': float(row['Stoch_K']) if not pd.isna(row['Stoch_K']) else None,
                    'stoch_d': float(row['Stoch_D']) if not pd.isna(row['Stoch_D']) else None,
                    'obv': float(row['OBV']) if not pd.isna(row['OBV']) else None,
                    'vwap': float(row['VWAP']) if not pd.isna(row['VWAP']) else None,
                    'mfi': float(row['MFI']) if not pd.isna(row['MFI']) else None,
                    'fear_greed': float(row['Fear_Greed']) if not pd.isna(row['Fear_Greed']) else None,
                }
                data_records.append(record)
            except Exception as e:
                logger.warning(f"Error processing data row {idx}: {e}")
                continue
        
        # Return last 100 records to avoid large responses
        return {
            "symbol": "BTC-USD",
            "period": period,
            "interval": interval,
            "data": data_records[-100:] if len(data_records) > 100 else data_records,
            "total_records": len(data_records),
            "indicators_available": True
        }
        
    except Exception as e:
        logger.error(f"Failed to get BTC data: {e}")
        
        # Return dummy data as fallback
        try:
            dummy_data = signal_generator.generate_dummy_data()
            dummy_records = []
            
            for idx, row in dummy_data.tail(50).iterrows():  # Last 50 periods
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
                "interval": interval,
                "data": dummy_records,
                "total_records": len(dummy_records),
                "note": "Using simulated data due to data fetch error"
            }
            
        except Exception as dummy_error:
            logger.error(f"Failed to generate dummy data: {dummy_error}")
            raise HTTPException(status_code=500, detail="Unable to fetch or generate BTC data")

@app.get("/analytics/pnl")
async def get_pnl_data():
    """Get detailed P&L analytics data"""
    try:
        trades_df = db.get_trades()
        
        if trades_df.empty:
            return {"daily_pnl": [], "cumulative_pnl": [], "by_lot": {}}
        
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
        trades_df['date'] = trades_df['timestamp'].dt.date
        trades_df['trade_value'] = trades_df['price'] * trades_df['size']
        trades_df['signed_value'] = trades_df['trade_value'] * trades_df['trade_type'].map({
            'buy': -1, 
            'sell': 1, 
            'hold': 0
        })
        
        # Daily PnL
        daily_pnl = trades_df.groupby('date')['signed_value'].sum().reset_index()
        daily_pnl['cumulative_pnl'] = daily_pnl['signed_value'].cumsum()
        
        # PnL by lot
        lot_pnl = {}
        for lot_id in trades_df['lot_id'].unique():
            lot_trades = trades_df[trades_df['lot_id'] == lot_id]
            lot_pnl[lot_id] = lot_trades['signed_value'].sum()
        
        return {
            "daily_pnl": [
                {"date": str(row['date']), "pnl": float(row['signed_value'])}
                for _, row in daily_pnl.iterrows()
            ],
            "cumulative_pnl": [
                {"date": str(row['date']), "pnl": float(row['cumulative_pnl'])}
                for _, row in daily_pnl.iterrows()
            ],
            "by_lot": lot_pnl
        }
    except Exception as e:
        logger.error(f"Failed to get P&L data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/performance")
async def get_performance_analytics():
    """Get detailed performance analytics"""
    try:
        trades_df = db.get_trades()
        positions_df = db.get_positions()
        
        if trades_df.empty:
            return {
                "returns": [],
                "volatility": 0,
                "sharpe_ratio": 0,
                "sortino_ratio": 0,
                "calmar_ratio": 0,
                "max_drawdown": 0,
                "win_rate": 0,
                "profit_factor": 0
            }
        
        # Calculate returns
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
        trades_df = trades_df.sort_values('timestamp')
        
        # Group by day and calculate daily returns
        daily_values = trades_df.groupby(trades_df['timestamp'].dt.date).apply(
            lambda x: (x['price'] * x['size'] * x['trade_type'].map({'buy': -1, 'sell': 1})).sum()
        )
        
        if len(daily_values) > 1:
            daily_returns = daily_values.pct_change().dropna()
            
            # Performance metrics
            avg_return = daily_returns.mean()
            volatility = daily_returns.std()
            
            # Sharpe ratio (annualized)
            sharpe_ratio = np.sqrt(252) * avg_return / (volatility + 1e-8)
            
            # Sortino ratio (only downside volatility)
            downside_returns = daily_returns[daily_returns < 0]
            downside_vol = downside_returns.std() if len(downside_returns) > 0 else volatility
            sortino_ratio = np.sqrt(252) * avg_return / (downside_vol + 1e-8)
            
            # Maximum drawdown
            cumulative_returns = (1 + daily_returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown_series = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown_series.min()
            
            # Calmar ratio
            annual_return = (1 + avg_return) ** 252 - 1
            calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # Win rate and profit factor from individual trades
            trade_returns = []
            for lot_id in trades_df['lot_id'].unique():
                lot_trades = trades_df[trades_df['lot_id'] == lot_id]
                if len(lot_trades) > 1:
                    entry_price = lot_trades[lot_trades['trade_type'] == 'buy']['price'].mean()
                    exit_price = lot_trades[lot_trades['trade_type'] == 'sell']['price'].mean()
                    if entry_price > 0 and exit_price > 0:
                        trade_returns.append((exit_price - entry_price) / entry_price)
            
            if trade_returns:
                wins = [r for r in trade_returns if r > 0]
                losses = [r for r in trade_returns if r < 0]
                
                win_rate = len(wins) / len(trade_returns)
                
                if wins and losses:
                    avg_win = np.mean(wins)
                    avg_loss = abs(np.mean(losses))
                    profit_factor = (len(wins) * avg_win) / (len(losses) * avg_loss)
                else:
                    profit_factor = 0
            else:
                win_rate = 0
                profit_factor = 0
            
            return {
                "returns": daily_returns.tolist(),
                "volatility": float(volatility),
                "sharpe_ratio": float(sharpe_ratio),
                "sortino_ratio": float(sortino_ratio),
                "calmar_ratio": float(calmar_ratio),
                "max_drawdown": float(max_drawdown),
                "win_rate": float(win_rate),
                "profit_factor": float(profit_factor)
            }
        else:
            return {
                "returns": [],
                "volatility": 0,
                "sharpe_ratio": 0,
                "sortino_ratio": 0,
                "calmar_ratio": 0,
                "max_drawdown": 0,
                "win_rate": 0,
                "profit_factor": 0
            }
            
    except Exception as e:
        logger.error(f"Failed to get performance analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/technical")
async def get_technical_analytics():
    """Get current technical indicator readings and signals"""
    try:
        if latest_btc_data is None or len(latest_btc_data) == 0:
            raise ValueError("No BTC data available")
        
        # Get latest readings
        latest = latest_btc_data.iloc[-1]
        
        # Calculate signal strengths
        rsi = latest['RSI']
        rsi_signal = "oversold" if rsi < 30 else "overbought" if rsi > 70 else "neutral"
        
        bb_position = latest['BB_Position']
        bb_signal = "oversold" if bb_position < 0.2 else "overbought" if bb_position > 0.8 else "neutral"
        
        stoch_k = latest['Stoch_K']
        stoch_signal = "oversold" if stoch_k < 20 else "overbought" if stoch_k > 80 else "neutral"
        
        # Trend analysis
        sma_20 = latest['SMA_20']
        sma_50 = latest['SMA_50']
        close = latest['Close']
        
        trend = "bullish" if close > sma_20 > sma_50 else "bearish" if close < sma_20 < sma_50 else "neutral"
        
        return {
            "current_indicators": {
                "price": float(close),
                "rsi": float(rsi),
                "rsi_signal": rsi_signal,
                "macd": float(latest['MACD']),
                "macd_signal_line": float(latest['MACD_Signal']),
                "bollinger_position": float(bb_position),
                "bollinger_signal": bb_signal,
                "stochastic_k": float(stoch_k),
                "stochastic_d": float(latest['Stoch_D']),
                "stochastic_signal": stoch_signal,
                "atr": float(latest['ATR']),
                "mfi": float(latest['MFI']),
                "obv": float(latest['OBV']),
                "vwap": float(latest['VWAP']),
                "sma_20": float(sma_20),
                "sma_50": float(sma_50),
                "fear_greed": float(latest['Fear_Greed'])
            },
            "trend_analysis": {
                "overall_trend": trend,
                "price_vs_sma20": "above" if close > sma_20 else "below",
                "price_vs_sma50": "above" if close > sma_50 else "below",
                "sma20_vs_sma50": "above" if sma_20 > sma_50 else "below"
            },
            "momentum_analysis": {
                "roc": float(latest['ROC']),
                "volume_trend": "high" if latest['Volume_Norm'] > 1.5 else "low" if latest['Volume_Norm'] < 0.5 else "normal"
            },
            "timestamp": latest.name.isoformat() if hasattr(latest.name, 'isoformat') else str(latest.name)
        }
        
    except Exception as e:
        logger.error(f"Failed to get technical analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/system/status")
async def get_system_status():
    """Get detailed system status"""
    try:
        return {
            "api_status": "running",
            "api_version": "2.0.0",
            "timestamp": datetime.now(),
            "signal_update_errors": signal_update_errors,
            "latest_signal_time": latest_signal.get('timestamp') if latest_signal else None,
            "data_cache_status": "available" if latest_btc_data is not None else "empty",
            "database_status": "connected",
            "signal_generator_status": "enhanced",
            "available_indicators": [
                "RSI", "MACD", "Bollinger Bands", "Stochastic", "ATR", 
                "OBV", "VWAP", "MFI", "SMA", "Fear & Greed"
            ]
        }
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Enhanced BTC Trading System API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)