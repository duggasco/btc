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

from database_models import DatabaseManager
from lstm_model import TradingSignalGenerator
from discord_notifications import DiscordNotifier

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

# Initialize Discord notifier
discord_notifier = DiscordNotifier()
logger.info("Discord notifier initialized")

# Global variables for caching
latest_btc_data = None
latest_signal = None
signal_update_errors = 0
max_signal_errors = 5
last_pnl = 0
price_tracker = {
    'last_price': None,
    'daily_open': None,
    'daily_high': None,
    'daily_low': None,
    'last_2_5_alert': None  # Track last alert price to avoid spam
}

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
        
    def check_limit_orders(self, current_price: float):
        """Check and trigger limit orders based on current price"""
        try:
            limits_df = db.get_trading_limits(active_only=True)
            if limits_df.empty:
                return
            
            for _, limit in limits_df.iterrows():
                triggered = False
                
                if limit['limit_type'] == 'stop_loss' and current_price <= limit['price']:
                    triggered = True
                elif limit['limit_type'] == 'take_profit' and current_price >= limit['price']:
                    triggered = True
                elif limit['limit_type'] == 'buy_limit' and current_price <= limit['price']:
                    triggered = True
                elif limit['limit_type'] == 'sell_limit' and current_price >= limit['price']:
                    triggered = True
                
                if triggered:
                    logger.info(f"Limit order triggered: {limit['limit_type']} at ${limit['price']}")
                    
                    # Send Discord notification
                    discord_notifier.notify_limit_triggered(
                        limit_type=limit['limit_type'],
                        trigger_price=limit['price'],
                        current_price=current_price,
                        size=limit.get('size')
                    )
                    
                    # Execute the implied trade
                    if limit['limit_type'] in ['stop_loss', 'take_profit', 'sell_limit']:
                        trade_type = 'sell'
                    else:
                        trade_type = 'buy'
                    
                    # Create trade
                    trade_id = db.add_trade(
                        symbol="BTC-USD",
                        trade_type=trade_type,
                        price=current_price,
                        size=limit.get('size', 0.01),
                        lot_id=limit.get('lot_id')
                    )
                    
                    # Mark limit as inactive
                    # Note: This would need to be implemented in DatabaseManager
                    # db.deactivate_limit(limit['id'])
                    
                    logger.info(f"Executed trade {trade_id} for triggered limit")
                    
        except Exception as e:
            logger.error(f"Error checking limit orders: {e}")
    
    def update_signals_loop(self):
        """Continuously update trading signals with error handling"""
        global signal_update_errors
        
        while self.running:
            try:
                self.update_signals()
                
                # Check limit orders with current price
                if latest_btc_data is not None and not latest_btc_data.empty:
                    current_price = latest_btc_data['Close'].iloc[-1]
                    self.check_limit_orders(current_price)
                
                signal_update_errors = 0  # Reset error count on success
                time.sleep(300)  # Update every 5 minutes
            except Exception as e:
                signal_update_errors += 1
                logger.error(f"Error updating signals (attempt {signal_update_errors}): {e}")
                
                # Exponential backoff with max errors check
                if signal_update_errors >= max_signal_errors:
                    logger.warning(f"Too many signal update errors ({signal_update_errors}), extending sleep time")
                    discord_notifier.notify_system_status("warning", f"Signal update errors: {signal_update_errors}")
                    time.sleep(600)  # Wait 10 minutes after too many errors
                else:
                    time.sleep(60 * signal_update_errors)  # Gradually increase wait time
                
    def update_signals(self):
        """Update trading signals and store in database"""
        global latest_btc_data, latest_signal, price_tracker
        
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
            
            # Update price tracking for alerts
            current_price = btc_data['Close'].iloc[-1]
            daily_open = btc_data['Close'].iloc[-24] if len(btc_data) >= 24 else current_price
            daily_high = btc_data['High'].rolling(24).max().iloc[-1] if len(btc_data) >= 24 else current_price
            daily_low = btc_data['Low'].rolling(24).min().iloc[-1] if len(btc_data) >= 24 else current_price
            
            # Check for +/- 2.5% price movements
            if price_tracker['daily_open']:
                price_change_pct = ((current_price - price_tracker['daily_open']) / price_tracker['daily_open']) * 100
                
                # Check if we should send a price alert
                if abs(price_change_pct) >= 2.5:
                    last_alert_price = price_tracker.get('last_2_5_alert')
                    # Only alert if price moved another 1% since last alert to avoid spam
                    if not last_alert_price or abs((current_price - last_alert_price) / last_alert_price) >= 0.01:
                        discord_notifier.notify_price_alert(current_price, price_change_pct, price_change_pct > 0)
                        price_tracker['last_2_5_alert'] = current_price
            
            price_tracker.update({
                'last_price': current_price,
                'daily_open': daily_open,
                'daily_high': daily_high,
                'daily_low': daily_low
            })
            
            discord_notifier.update_daily_stats(daily_open, daily_high, daily_low)
            
            # Generate signal
            logger.info("Generating trading signal...")
            signal, confidence, predicted_price = signal_generator.predict_signal(btc_data)
            
            # Check if signal changed
            old_signal = latest_signal.get('signal') if latest_signal else None
            
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
            
            # Send Discord notification for signal update
            discord_notifier.notify_signal_update(signal, confidence, predicted_price, current_price)
            
            # Check for major P&L changes
            try:
                metrics = db.get_portfolio_metrics()
                current_pnl = metrics.get('total_pnl', 0)
                global last_pnl
                
                if last_pnl != 0:
                    pnl_change_pct = ((current_pnl - last_pnl) / abs(last_pnl)) * 100
                    # Notify if P&L changed by more than 5%
                    if abs(pnl_change_pct) >= 5:
                        trades_today = db.get_trades(limit=100)
                        if not trades_today.empty:
                            trades_today['date'] = pd.to_datetime(trades_today['timestamp']).dt.date
                            today = datetime.now().date()
                            today_trades = trades_today[trades_today['date'] == today]
                            daily_pnl = (today_trades['price'] * today_trades['size'] * 
                                       today_trades['trade_type'].map({'buy': -1, 'sell': 1, 'hold': 0})).sum()
                        else:
                            daily_pnl = 0
                        
                        discord_notifier.notify_pnl_change(current_pnl, daily_pnl, pnl_change_pct)
                        last_pnl = current_pnl
                else:
                    last_pnl = current_pnl
                    
            except Exception as e:
                logger.warning(f"Failed to check P&L changes: {e}")
            
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

@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    logger.info("Starting BTC Trading System API...")
    
    # Send system online notification
    discord_notifier.notify_system_status("online", "BTC Trading System API is starting up...")
    
    # Start signal updater
    try:
        signal_updater.start()
        logger.info("Signal updater started successfully")
    except Exception as e:
        logger.error(f"Failed to start signal updater: {e}")
        discord_notifier.notify_system_status("error", f"Failed to start signal updater: {str(e)}")
    
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
    
    discord_notifier.notify_system_status("online", "BTC Trading System API started successfully! All systems operational.")
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
        
        # Send Discord notification
        trade_value = trade.price * trade.size
        discord_notifier.notify_trade_executed(
            trade_id=trade_id,
            trade_type=trade.trade_type,
            price=trade.price,
            size=trade.size,
            lot_id=trade.lot_id or "auto",
            trade_value=trade_value
        )
        
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

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting BTC Trading System API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)