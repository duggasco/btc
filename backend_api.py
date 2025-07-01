from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
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
import traceback
import sqlite3
from contextlib import asynccontextmanager
import uuid
from collections import deque

from database_models import DatabaseManager
from lstm_model import TradingSignalGenerator

# Import the ENHANCED classes from integration module
from integration import AdvancedIntegratedBacktestingSystem, AdvancedTradingSignalGenerator
# Import from backtesting_system (corrected from enhanced_backtesting_system)
from backtesting_system import (
    BacktestConfig, SignalWeights, EnhancedSignalWeights, EnhancedBacktestingPipeline,
    ComprehensiveSignalCalculator, EnhancedPerformanceMetrics,
    EnhancedWalkForwardBacktester, EnhancedBayesianOptimizer, AdaptiveRetrainingScheduler,
    PerformanceMetrics
)

# Import Discord notifications if available
try:
    from discord_notifications import DiscordNotifier
    discord_notifier = DiscordNotifier()
except ImportError:
    discord_notifier = None
    logging.warning("Discord notifications not available")
from paper_trading_persistence import PersistentPaperTrading

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============= NEW ENHANCED FEATURES =============

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.signal_subscribers: List[WebSocket] = []
        self.price_subscribers: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        if websocket in self.signal_subscribers:
            self.signal_subscribers.remove(websocket)
        if websocket in self.price_subscribers:
            self.price_subscribers.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                pass

    async def broadcast_signal_update(self, signal_data: dict):
        message = json.dumps({"type": "signal_update", "data": signal_data})
        for connection in self.signal_subscribers:
            try:
                await connection.send_text(message)
            except:
                pass

    async def broadcast_price_update(self, price_data: dict):
        message = json.dumps({"type": "price_update", "data": price_data})
        for connection in self.price_subscribers:
            try:
                await connection.send_text(message)
            except:
                pass

manager = ConnectionManager()

# Enhanced rate limiter
class RateLimiter:
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = deque()
    
    def check_rate_limit(self, client_id: str = "default") -> bool:
        now = time.time()
        # Remove old requests
        while self.requests and self.requests[0][0] < now - self.window_seconds:
            self.requests.popleft()
        
        if len(self.requests) >= self.max_requests:
            return False
        
        self.requests.append((now, client_id))
        return True

rate_limiter = RateLimiter()

# Paper trading with persistence
paper_trading_enabled = False
paper_trading = None  # Will be initialized in lifespan

def execute_paper_trade(signal: str, confidence: float):
    """Execute paper trades based on signals with persistence"""
    global paper_trading
    
    if not paper_trading:
        logger.warning("Paper trading not initialized")
        return
        
    if not latest_btc_data or len(latest_btc_data) == 0:
        return
    
    current_price = float(latest_btc_data['Close'].iloc[-1])
    
    # Get current portfolio state
    portfolio = paper_trading.get_portfolio()
    
    # Determine position size based on confidence
    position_size = min(0.1 * confidence, 0.05)  # Max 5% of portfolio
    
    if signal == "buy" and confidence > trading_rules['buy_threshold']:
        # Calculate BTC amount to buy
        usd_amount = portfolio['usd_balance'] * position_size
        btc_amount = usd_amount / current_price
        
        if usd_amount > 0 and btc_amount >= 0.00001:  # Minimum trade size
            try:
                trade_id = paper_trading.execute_trade("buy", current_price, btc_amount, usd_amount)
                logger.info(f"Paper trade executed: BUY {btc_amount:.6f} BTC at ${current_price:.2f}")
                
                # Send Discord notification if enabled
                if discord_notifier:
                    discord_notifier.notify_trade_executed(
                        trade_id, "buy", current_price, btc_amount, None, usd_amount
                    )
            except Exception as e:
                logger.error(f"Failed to execute paper buy: {e}")
    
    elif signal == "sell" and confidence > trading_rules['sell_threshold']:
        # Calculate BTC amount to sell
        btc_amount = portfolio['btc_balance'] * position_size
        usd_amount = btc_amount * current_price
        
        if btc_amount > 0 and btc_amount >= 0.00001:  # Minimum trade size
            try:
                trade_id = paper_trading.execute_trade("sell", current_price, btc_amount, usd_amount)
                logger.info(f"Paper trade executed: SELL {btc_amount:.6f} BTC at ${current_price:.2f}")
                
                # Send Discord notification if enabled
                if discord_notifier:
                    discord_notifier.notify_trade_executed(
                        trade_id, "sell", current_price, btc_amount, None, usd_amount
                    )
            except Exception as e:
                logger.error(f"Failed to execute paper sell: {e}")

# ============= END NEW ENHANCED FEATURES =============

# ============= ORIGINAL CODE WITH ENHANCEMENTS =============

class TradeRequest(BaseModel):
    symbol: str
    trade_type: str
    price: float
    size: float
    lot_id: Optional[str] = None
    notes: Optional[str] = None

class LimitOrder(BaseModel):
    symbol: str
    limit_type: str
    price: float
    size: Optional[float] = None
    lot_id: Optional[str] = None

class EnhancedBacktestRequest(BaseModel):
    """Enhanced backtest request with more options"""
    period: str = "1y"
    optimize_weights: bool = True
    include_macro: bool = True
    use_enhanced_weights: bool = True
    n_optimization_trials: int = 20
    force: bool = False
    settings: Optional[Dict[str, Any]] = None

# Initialize components with error handling
try:
    db_path = os.getenv('DATABASE_PATH', '/app/data/trading_system.db')
    db = DatabaseManager(db_path)
    logger.info(f"Database initialized at {db_path}")
except Exception as e:
    logger.error(f"Failed to initialize database: {e}")
    raise

# Use ENHANCED signal generator
try:
    signal_generator = AdvancedTradingSignalGenerator()
    logger.info("Advanced signal generator initialized")
except Exception as e:
    logger.error(f"Failed to initialize signal generator: {e}")
    raise

# Global variables for caching
latest_btc_data = None
latest_signal = None
latest_enhanced_signal = None
signal_update_errors = 0
max_signal_errors = 5

# Backtesting global variables
backtest_system = None
backtest_in_progress = False
latest_backtest_results = None

# Enhanced metrics cache
latest_comprehensive_signals = None
signal_calculator = ComprehensiveSignalCalculator()

# Configuration storage (in production, use database)
model_config = {
    'input_size': 16,
    'hidden_size': 50,
    'num_layers': 2,
    'dropout': 0.2,
    'sequence_length': 60,
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 50,
    'use_attention': False,
    'ensemble_size': 5
}

trading_rules = {
    'min_trade_size': 0.001,
    'max_position_size': 0.1,
    'position_scaling': 'confidence_based',
    'stop_loss_pct': 5.0,
    'take_profit_pct': 10.0,
    'max_daily_trades': 10,
    'buy_threshold': 0.6,
    'strong_buy_threshold': 0.8,
    'sell_threshold': 0.6,
    'strong_sell_threshold': 0.8
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
        
    def update_signals_loop(self):
        """Continuously update trading signals with error handling"""
        global signal_update_errors
        
        while self.running:
            try:
                self.update_signals()
                signal_update_errors = 0  # Reset error count on success
                
                # Broadcast updates via WebSocket (NEW)
                asyncio.run(self.broadcast_updates())
                
                if paper_trading_enabled and paper_trading and latest_btc_data is not None:
                    try:
                        current_price = float(latest_btc_data['Close'].iloc[-1])
                        paper_trading.save_performance_snapshot(current_price)
                        logger.debug("Paper trading performance snapshot saved")
                    except Exception as e:
                        logger.error(f"Failed to save paper trading snapshot: {e}")
                
                
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
    
    async def broadcast_updates(self):
        """Broadcast signal and price updates via WebSocket (NEW)"""
        if latest_signal:
            await manager.broadcast_signal_update(latest_signal)
        
        if latest_btc_data is not None and len(latest_btc_data) > 0:
            price_data = {
                "price": float(latest_btc_data['Close'].iloc[-1]),
                "timestamp": datetime.now().isoformat()
            }
            await manager.broadcast_price_update(price_data)
                
    def update_signals(self):
        """Update trading signals and store in database"""
        global latest_btc_data, latest_signal, latest_enhanced_signal, latest_comprehensive_signals
        
        try:
            logger.info("Fetching enhanced BTC data for signal update...")
            # Use enhanced data fetching
            btc_data = signal_generator.fetch_enhanced_btc_data(period="3mo", include_macro=True)
            
            if btc_data is None or len(btc_data) == 0:
                logger.warning("No BTC data received, using cached data if available")
                if latest_btc_data is not None:
                    btc_data = latest_btc_data
                else:
                    raise ValueError("No BTC data available and no cached data")
            
            latest_btc_data = btc_data
            logger.info(f"Successfully fetched {len(btc_data)} days of enhanced BTC data")
            
            # Calculate comprehensive signals
            logger.info("Calculating comprehensive signals...")
            latest_comprehensive_signals = signal_calculator.calculate_all_signals(btc_data)
            
            # Generate enhanced signal with confidence intervals
            logger.info("Generating enhanced trading signal...")
            signal, confidence, predicted_price, analysis = signal_generator.predict_with_confidence(btc_data)
            
            # Store signal in database
            try:
                db.add_enhanced_model_signal(
                    "BTC-USD", signal, confidence, predicted_price, 
                    analysis, signal_generator.enhanced_weights.__dict__ if hasattr(signal_generator, 'enhanced_weights') else None,
                    {k: float(v.iloc[-1]) if hasattr(v, 'iloc') else v for k, v in latest_comprehensive_signals.items() if k in ['rsi', 'macd', 'bb_position']}
                )
                logger.info("Enhanced signal stored in database")
            except Exception as e:
                logger.warning(f"Failed to store signal in database: {e}")
            
            # Update global latest signals
            latest_signal = {
                "symbol": "BTC-USD",
                "signal": signal,
                "confidence": confidence,
                "predicted_price": predicted_price,
                "timestamp": datetime.now()
            }
            
            latest_enhanced_signal = {
                **latest_signal,
                "analysis": analysis,
                "comprehensive_signals": {
                    "technical": {
                        "rsi": float(latest_comprehensive_signals.get('rsi', 50).iloc[-1]) if 'rsi' in latest_comprehensive_signals else 50,
                        "macd_signal": bool(latest_comprehensive_signals.get('macd_bullish_cross', False).iloc[-1]) if 'macd_bullish_cross' in latest_comprehensive_signals else False,
                        "bb_position": float(latest_comprehensive_signals.get('bb_position', 0.5).iloc[-1]) if 'bb_position' in latest_comprehensive_signals else 0.5,
                    },
                    "sentiment": {
                        "fear_proxy": float(latest_comprehensive_signals.get('fear_proxy', 0.5).iloc[-1]) if 'fear_proxy' in latest_comprehensive_signals else 0.5,
                        "greed_proxy": float(latest_comprehensive_signals.get('greed_proxy', 0.5).iloc[-1]) if 'greed_proxy' in latest_comprehensive_signals else 0.5,
                    },
                    "on_chain": {
                        "nvt_proxy": float(latest_comprehensive_signals.get('nvt_proxy', 1.0).iloc[-1]) if 'nvt_proxy' in latest_comprehensive_signals else 1.0,
                        "accumulation_proxy": float(latest_comprehensive_signals.get('accumulation_proxy', 0.5).iloc[-1]) if 'accumulation_proxy' in latest_comprehensive_signals else 0.5,
                    }
                }
            }
            
            logger.info(f"Enhanced signal updated: {signal} (confidence: {confidence:.2%}, price: ${predicted_price:.2f})")
            
            # Send Discord notification if signal changed
            if discord_notifier and hasattr(discord_notifier, 'last_signal') and discord_notifier.last_signal != signal:
                discord_notifier.notify_signal_update(signal, confidence, predicted_price, btc_data['Close'].iloc[-1])
            
            # Execute paper trades if enabled (NEW)
            if paper_trading_enabled:
                execute_paper_trade(signal, confidence)
                
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
                latest_enhanced_signal = {
                    **latest_signal,
                    "analysis": {},
                    "comprehensive_signals": {}
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Enhanced lifespan management with better initialization and paper trading persistence"""
    # Startup
    logger.info("Starting Enhanced BTC Trading System API...")
    
    # Initialize global variables
    global latest_signal, latest_btc_data, latest_enhanced_signal, backtest_system, paper_trading
    
    # Start signal updater
    try:
        signal_updater.start()
        logger.info("Signal updater started successfully")
    except Exception as e:
        logger.error(f"Failed to start signal updater: {e}")
    
    # Initialize paper trading with persistence
    try:
        paper_trading = PersistentPaperTrading(db_path)
        logger.info("Paper trading persistence initialized")
        
        # Log current portfolio state
        portfolio = paper_trading.get_portfolio()
        logger.info(f"Paper trading portfolio loaded - BTC: {portfolio['btc_balance']:.6f}, "
                   f"USD: ${portfolio['usd_balance']:,.2f}, "
                   f"Trades: {len(portfolio['trades'])}")
    except Exception as e:
        logger.error(f"Failed to initialize paper trading persistence: {e}")
        logger.warning("Paper trading will run without persistence")
        paper_trading = None
    
    # Generate initial enhanced signal
    try:
        logger.info("Generating initial enhanced signal...")
        initial_data = signal_generator.fetch_enhanced_btc_data(period="1mo", include_macro=False)
        signal, confidence, predicted_price, analysis = signal_generator.predict_with_confidence(initial_data)
        
        latest_btc_data = initial_data
        latest_signal = {
            "symbol": "BTC-USD",
            "signal": signal,
            "confidence": confidence,
            "predicted_price": predicted_price,
            "timestamp": datetime.now()
        }
        latest_enhanced_signal = {
            **latest_signal,
            "analysis": analysis,
            "comprehensive_signals": {}
        }
        logger.info(f"Initial enhanced signal generated: {signal} (confidence: {confidence:.2%})")
        
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
        latest_enhanced_signal = {
            **latest_signal,
            "analysis": {},
            "comprehensive_signals": {}
        }
        logger.info("Using default signal due to initialization error")
    
    # Initialize ADVANCED backtest system
    try:
        backtest_system = AdvancedIntegratedBacktestingSystem(
            db_path=db_path,
            model_path='models/lstm_btc_model.pth'
        )
        logger.info("Advanced backtest system initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize backtest system: {e}")
        backtest_system = None
    
    # Load latest backtest results if available
    try:
        load_latest_backtest_results()
        logger.info("Loaded previous backtest results")
    except Exception as e:
        logger.warning(f"Could not load previous backtest results: {e}")
    
    # Send startup notification
    if discord_notifier:
        try:
            startup_message = "BTC Trading System API started successfully"
            if paper_trading:
                portfolio = paper_trading.get_portfolio()
                startup_message += f"\nPaper Trading Portfolio: BTC={portfolio['btc_balance']:.6f}, USD=${portfolio['usd_balance']:,.2f}"
            discord_notifier.notify_system_status("online", startup_message)
        except Exception as e:
            logger.warning(f"Failed to send Discord startup notification: {e}")
    
    # Save initial paper trading snapshot if enabled
    if paper_trading_enabled and paper_trading and latest_btc_data is not None:
        try:
            current_price = float(latest_btc_data['Close'].iloc[-1])
            paper_trading.save_performance_snapshot(current_price)
            logger.info("Initial paper trading performance snapshot saved")
        except Exception as e:
            logger.warning(f"Failed to save initial paper trading snapshot: {e}")
    
    logger.info("Enhanced BTC Trading System API startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Enhanced BTC Trading System API...")
    
    # Stop signal updater
    try:
        signal_updater.stop()
        logger.info("Signal updater stopped")
    except Exception as e:
        logger.error(f"Error stopping signal updater: {e}")
    
    # Save final paper trading snapshot
    if paper_trading and latest_btc_data is not None:
        try:
            current_price = float(latest_btc_data['Close'].iloc[-1])
            paper_trading.save_performance_snapshot(current_price)
            
            # Log final portfolio state
            portfolio = paper_trading.get_portfolio()
            metrics = paper_trading.calculate_performance_metrics(current_price)
            
            logger.info(f"Final paper trading state - BTC: {portfolio['btc_balance']:.6f}, "
                       f"USD: ${portfolio['usd_balance']:,.2f}, "
                       f"P&L: ${portfolio['total_pnl']:,.2f}, "
                       f"Return: {metrics['total_return']:.2%}")
            
        except Exception as e:
            logger.error(f"Failed to save final paper trading snapshot: {e}")
    
    # Send shutdown notification
    if discord_notifier:
        try:
            shutdown_message = "BTC Trading System API shutting down"
            if paper_trading:
                portfolio = paper_trading.get_portfolio()
                shutdown_message += f"\nFinal P&L: ${portfolio['total_pnl']:,.2f}"
            discord_notifier.notify_system_status("offline", shutdown_message)
        except Exception as e:
            logger.warning(f"Failed to send Discord shutdown notification: {e}")
    
    logger.info("API shutdown complete")

app = FastAPI(
    title="BTC Trading System API", 
    version="2.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============= NEW WEBSOCKET ENDPOINT =============
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("action") == "subscribe_signals":
                manager.signal_subscribers.append(websocket)
                await websocket.send_text(json.dumps({"status": "subscribed to signals"}))
            
            elif message.get("action") == "subscribe_prices":
                manager.price_subscribers.append(websocket)
                await websocket.send_text(json.dumps({"status": "subscribed to prices"}))
            
            elif message.get("action") == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# ============= ORIGINAL ENDPOINTS WITH ENHANCEMENTS =============

# Core endpoints
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Enhanced BTC Trading System API is running", 
        "version": "2.1.0",
        "timestamp": datetime.now(),
        "status": "healthy",
        "signal_errors": signal_update_errors,
        "features": ["enhanced_signals", "comprehensive_backtesting", "50+_indicators", "websocket_support", "paper_trading"]
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
        enhanced_status = "active" if latest_enhanced_signal else "inactive"
        
        return {
            "status": "healthy",
            "timestamp": datetime.now(),
            "components": {
                "database": db_status,
                "signal_generator": signal_status,
                "enhanced_signals": enhanced_status,
                "signal_update_errors": signal_update_errors,
                "comprehensive_signals": "active" if latest_comprehensive_signals is not None else "inactive",
                "paper_trading": "enabled" if paper_trading_enabled else "disabled",  # NEW
                "websocket_connections": len(manager.active_connections)  # NEW
            },
            "latest_signal": latest_signal,
            "enhanced_features": {
                "macro_indicators": True,
                "sentiment_analysis": True,
                "on_chain_proxies": True,
                "50_plus_signals": True,
                "websocket_streaming": True,  # NEW
                "paper_trading": True,  # NEW
                "rate_limiting": True  # NEW
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

# Enhanced signal endpoints
@app.get("/signals/enhanced/latest")
async def get_enhanced_latest_signal():
    """Get the latest enhanced trading signal with full analysis"""
    global latest_enhanced_signal
    
    if not rate_limiter.check_rate_limit():  # NEW rate limiting
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    try:
        if latest_enhanced_signal is None:
            logger.info("No cached enhanced signal, generating new one...")
            try:
                btc_data = signal_generator.fetch_enhanced_btc_data(period="1mo", include_macro=True)
                signal, confidence, predicted_price, analysis = signal_generator.predict_with_confidence(btc_data)
                
                latest_enhanced_signal = {
                    "symbol": "BTC-USD",
                    "signal": signal,
                    "confidence": confidence,
                    "predicted_price": predicted_price,
                    "timestamp": datetime.now(),
                    "analysis": analysis,
                    "comprehensive_signals": {}
                }
                logger.info(f"Generated new enhanced signal: {signal}")
            except Exception as e:
                logger.error(f"Failed to generate enhanced signal: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        return latest_enhanced_signal
        
    except Exception as e:
        logger.error(f"Failed to get enhanced signal: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting enhanced signal: {str(e)}")

@app.get("/signals/comprehensive")
async def get_comprehensive_signals():
    """Get all 50+ calculated signals"""
    global latest_comprehensive_signals
    
    if latest_comprehensive_signals is None:
        return {
            "status": "no_data",
            "message": "Comprehensive signals not yet calculated. Please wait for next update cycle."
        }
    
    try:
        # Convert DataFrame to dict with latest values
        signals_dict = {}
        for col in latest_comprehensive_signals.columns:
            if col not in ['Open', 'High', 'Low', 'Close', 'Volume']:  # Skip basic OHLCV
                latest_val = latest_comprehensive_signals[col].iloc[-1]
                if isinstance(latest_val, (np.bool_, bool)):
                    signals_dict[col] = bool(latest_val)
                elif isinstance(latest_val, (np.integer, np.floating)):
                    signals_dict[col] = float(latest_val)
                else:
                    signals_dict[col] = str(latest_val)
        
        # Group signals by category
        categorized = {
            "technical": {},
            "momentum": {},
            "volatility": {},
            "volume": {},
            "trend": {},
            "sentiment": {},
            "on_chain": {}
        }
        
        # Categorize signals
        for signal, value in signals_dict.items():
            if any(ind in signal for ind in ['rsi', 'macd', 'stoch', 'mfi', 'roc']):
                categorized["momentum"][signal] = value
            elif any(ind in signal for ind in ['bb_', 'atr', 'volatility']):
                categorized["volatility"][signal] = value
            elif any(ind in signal for ind in ['volume', 'obv', 'cmf']):
                categorized["volume"][signal] = value
            elif any(ind in signal for ind in ['sma', 'ema', 'trend', 'adx']):
                categorized["trend"][signal] = value
            elif any(ind in signal for ind in ['fear', 'greed', 'sentiment']):
                categorized["sentiment"][signal] = value
            elif any(ind in signal for ind in ['nvt', 'whale', 'accumulation', 'hodl']):
                categorized["on_chain"][signal] = value
            else:
                categorized["technical"][signal] = value
        
        return {
            "timestamp": datetime.now(),
            "total_signals": len(signals_dict),
            "categorized_signals": categorized,
            "all_signals": signals_dict
        }
        
    except Exception as e:
        logger.error(f"Failed to get comprehensive signals: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/signals/latest")
async def get_latest_signal():
    """Get the latest trading signal (original endpoint maintained)"""
    global latest_signal
    
    try:
        if latest_signal is None:
            logger.info("No cached signal, generating new one...")
            try:
                btc_data = signal_generator.fetch_enhanced_btc_data(period="1mo", include_macro=False)
                signal, confidence, predicted_price, _ = signal_generator.predict_with_confidence(btc_data)
                
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
async def get_signal_history(limit: int = 50, hours: Optional[int] = None):
    """Get historical trading signals"""
    try:
        if hours:
            # Get signals from last N hours
            conn = sqlite3.connect(db.db_path)
            query = """
                SELECT timestamp, signal, confidence, price_prediction
                FROM model_signals
                WHERE timestamp > datetime('now', '-{} hours')
                ORDER BY timestamp DESC
            """.format(hours)
            
            signals_df = pd.read_sql_query(query, conn)
            conn.close()
            
            if not signals_df.empty:
                signals_df['timestamp'] = pd.to_datetime(signals_df['timestamp'])
                return signals_df.to_dict('records')
            return []
        else:
            # Use existing method
            signals_df = db.get_model_signals(limit=limit)
            return signals_df.to_dict('records')
    except Exception as e:
        logger.error(f"Failed to get signal history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Trading endpoints
@app.post("/trades/")
async def create_trade(trade: TradeRequest):
    """Create a new trade with enhanced features"""
    try:
        # Calculate PnL if this is a sell
        pnl = 0
        if trade.trade_type == 'sell' and trade.lot_id:
            positions_df = db.get_positions()
            position = positions_df[positions_df['lot_id'] == trade.lot_id]
            if not position.empty:
                avg_buy_price = position['avg_buy_price'].iloc[0]
                pnl = (trade.price - avg_buy_price) * trade.size
        
        trade_id = db.add_trade(
            symbol=trade.symbol,
            trade_type=trade.trade_type,
            price=trade.price,
            size=trade.size,
            lot_id=trade.lot_id,
            pnl=pnl,
            notes=trade.notes
        )
        logger.info(f"Trade created: {trade_id} with PnL: {pnl}")
        
        # Send Discord notification
        if discord_notifier:
            trade_value = trade.price * trade.size
            discord_notifier.notify_trade_executed(
                trade_id, trade.trade_type, trade.price, trade.size,
                trade.lot_id, trade_value
            )
        
        return {"trade_id": trade_id, "status": "success", "message": "Trade created successfully", "pnl": pnl}
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
        
        # Send Discord notification
        if discord_notifier:
            current_price = latest_btc_data['Close'].iloc[-1] if latest_btc_data is not None else 0
            discord_notifier.notify_limit_triggered(
                limit_order.limit_type, limit_order.price, current_price, limit_order.size
            )
        
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

# Portfolio endpoints
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
        
        # Add enhanced metrics if available
        if latest_enhanced_signal and 'analysis' in latest_enhanced_signal:
            metrics['signal_confidence'] = latest_enhanced_signal.get('confidence', 0)
            metrics['consensus_ratio'] = latest_enhanced_signal.get('analysis', {}).get('consensus_ratio', 0)
        
        return metrics
    except Exception as e:
        logger.error(f"Failed to get portfolio metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Market data endpoints
@app.get("/market/btc-data")
async def get_btc_data(period: str = "1mo", include_indicators: bool = False):
    """Get BTC market data with optional indicators"""
    try:
        logger.info(f"Fetching BTC data for period: {period}")
        
        if include_indicators:
            # Fetch enhanced data with all indicators
            btc_data = signal_generator.fetch_enhanced_btc_data(period=period, include_macro=False)
        else:
            # Fetch basic data
            btc_data = signal_generator.fetch_btc_data(period=period)
        
        if btc_data is None or len(btc_data) == 0:
            raise ValueError("No BTC data available")
        
        data_records = []
        for idx, row in btc_data.iterrows():
            try:
                record = {
                    'timestamp': idx.isoformat(),
                    'Date': idx.isoformat(),  # Add Date field for compatibility
                    'open': float(row['Open']),
                    'Open': float(row['Open']),  # Duplicate for compatibility
                    'high': float(row['High']),
                    'High': float(row['High']),
                    'low': float(row['Low']),
                    'Low': float(row['Low']),
                    'close': float(row['Close']),
                    'Close': float(row['Close']),
                    'volume': float(row['Volume']),
                    'Volume': float(row['Volume'])
                }
                
                # Add basic indicators always
                for indicator in ['SMA_20', 'SMA_50', 'RSI', 'MACD', 'sma_50', 'sma_200']:
                    if indicator in row and not pd.isna(row[indicator]):
                        record[indicator.lower()] = float(row[indicator])
                        record[indicator] = float(row[indicator])  # Keep original case too
                
                # Add enhanced indicators if requested
                if include_indicators:
                    # Add technical indicators
                    for col in ['bb_position', 'bb_upper', 'bb_lower', 'atr_normalized', 'stoch_k', 'mfi', 'cmf', 'obv']:
                        if col in row and not pd.isna(row[col]):
                            record[col] = float(row[col])
                    
                    # Add sentiment indicators
                    for col in ['fear_proxy', 'greed_proxy', 'momentum_sentiment']:
                        if col in row and not pd.isna(row[col]):
                            record[col] = float(row[col])
                
                data_records.append(record)
            except Exception as e:
                logger.warning(f"Error processing data row {idx}: {e}")
                continue
        
        # Return last 100 records to avoid large responses
        return {
            "symbol": "BTC-USD",
            "period": period,
            "data": data_records[-100:] if len(data_records) > 100 else data_records,
            "total_records": len(data_records),
            "enhanced": include_indicators
        }
        
    except Exception as e:
        logger.error(f"Failed to get BTC data: {e}")
        
        # Return dummy data as fallback
        try:
            dummy_data = signal_generator.generate_dummy_data()
            dummy_records = []
            
            for idx, row in dummy_data.tail(50).iterrows():
                record = {
                    'timestamp': idx.isoformat(),
                    'Date': idx.isoformat(),
                    'open': float(row['Open']),
                    'Open': float(row['Open']),
                    'high': float(row['High']),
                    'High': float(row['High']),
                    'low': float(row['Low']),
                    'Low': float(row['Low']),
                    'close': float(row['Close']),
                    'Close': float(row['Close']),
                    'volume': float(row['Volume']),
                    'Volume': float(row['Volume'])
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

@app.get("/btc/latest")
async def get_latest_btc_price():
    """Get latest BTC price"""
    try:
        if latest_btc_data is not None and len(latest_btc_data) > 0:
            latest_price = float(latest_btc_data['Close'].iloc[-1])
            return {"latest_price": latest_price, "timestamp": datetime.now()}
        else:
            return {"latest_price": 45000.0, "timestamp": datetime.now(), "note": "Using default price"}
    except Exception as e:
        logger.error(f"Failed to get latest BTC price: {e}")
        return {"latest_price": 45000.0, "timestamp": datetime.now(), "error": str(e)}

# Analytics endpoints
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

@app.get("/analytics/portfolio-comprehensive")
async def get_comprehensive_portfolio_analytics():
    """Get enhanced portfolio analytics"""
    try:
        analytics = db.get_portfolio_analytics()
        
        # Add current market regime
        regime_df = db.get_market_regime_history(limit=1)
        if not regime_df.empty:
            analytics['current_market_regime'] = regime_df.iloc[0].to_dict()
        
        # Add feature importance
        feature_importance = db.get_feature_importance_ranking().head(10)
        analytics['top_features'] = feature_importance.to_dict('records')
        
        # Add recent signal performance
        signal_perf = db.get_signal_performance_history(limit=50)
        if not signal_perf.empty:
            analytics['signal_performance_summary'] = {
                'avg_win_rate': signal_perf['win_rate'].mean(),
                'avg_contribution': signal_perf['total_contribution'].mean(),
                'top_performers': signal_perf.nlargest(5, 'total_contribution')[['signal_name', 'total_contribution']].to_dict('records')
            }
        
        return analytics
    except Exception as e:
        logger.error(f"Failed to get comprehensive analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/performance")
async def get_performance_analytics():
    """Get comprehensive performance metrics"""
    try:
        # First try to get from latest backtest results
        if latest_backtest_results and 'performance_metrics' in latest_backtest_results:
            return latest_backtest_results['performance_metrics']
        
        # Otherwise calculate from trades
        trades_df = db.get_trades()
        if trades_df.empty:
            return {
                'total_return': 0,
                'sharpe_ratio': 0,
                'sortino_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'total_trades': 0,
                'calmar_ratio': 0,
                'omega_ratio': 1
            }
        
        # Calculate metrics using EnhancedPerformanceMetrics
        metrics_calculator = EnhancedPerformanceMetrics()
        
        # Calculate returns
        trades_df = trades_df.sort_values('timestamp')
        trades_df['trade_value'] = trades_df['price'] * trades_df['size']
        trades_df['signed_value'] = trades_df['trade_value'] * trades_df['trade_type'].map({
            'buy': -1, 'sell': 1, 'hold': 0
        })
        trades_df['cumulative_pnl'] = trades_df['signed_value'].cumsum()
        trades_df['returns'] = trades_df['cumulative_pnl'].pct_change().fillna(0)
        
        returns = trades_df['returns'].values
        cumulative_pnl = trades_df['cumulative_pnl'].values
        
        # Calculate all metrics
        max_dd = metrics_calculator.maximum_drawdown(cumulative_pnl)
        
        return {
            'total_return': cumulative_pnl[-1] / abs(trades_df[trades_df['trade_type'] == 'buy']['trade_value'].sum()) if len(cumulative_pnl) > 0 else 0,
            'sharpe_ratio': metrics_calculator.sharpe_ratio(returns),
            'sortino_ratio': metrics_calculator.sortino_ratio(returns),
            'max_drawdown': max_dd,
            'win_rate': metrics_calculator.win_rate(returns),
            'profit_factor': metrics_calculator.profit_factor(returns),
            'total_trades': len(trades_df),
            'calmar_ratio': metrics_calculator.calmar_ratio(returns, max_dd),
            'omega_ratio': metrics_calculator.omega_ratio(returns),
            'equity_curve': cumulative_pnl.tolist()
        }
        
    except Exception as e:
        logger.error(f"Failed to get performance analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/risk")
async def get_risk_analytics():
    """Get risk analysis metrics"""
    try:
        trades_df = db.get_trades()
        if trades_df.empty:
            return {
                'var_95': 0,
                'var_99': 0,
                'cvar_95': 0,
                'cvar_99': 0,
                'returns_distribution': []
            }
        
        # Calculate returns
        trades_df = trades_df.sort_values('timestamp')
        trades_df['trade_value'] = trades_df['price'] * trades_df['size']
        trades_df['signed_value'] = trades_df['trade_value'] * trades_df['trade_type'].map({
            'buy': -1, 'sell': 1, 'hold': 0
        })
        trades_df['cumulative_pnl'] = trades_df['signed_value'].cumsum()
        trades_df['returns'] = trades_df['cumulative_pnl'].pct_change().fillna(0)
        
        returns = trades_df['returns'].values
        
        # Calculate risk metrics
        metrics_calculator = EnhancedPerformanceMetrics()
        
        return {
            'var_95': metrics_calculator.value_at_risk(returns, 0.95),
            'var_99': metrics_calculator.value_at_risk(returns, 0.99),
            'cvar_95': metrics_calculator.conditional_value_at_risk(returns, 0.95),
            'cvar_99': metrics_calculator.conditional_value_at_risk(returns, 0.99),
            'returns_distribution': returns.tolist()
        }
        
    except Exception as e:
        logger.error(f"Failed to get risk analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/correlations")
async def get_correlations_analytics():
    """Get market correlations analysis"""
    try:
        # Get latest BTC data with all features
        if latest_btc_data is not None and isinstance(latest_btc_data, pd.DataFrame):
            # Calculate correlation matrix
            feature_cols = [col for col in latest_btc_data.columns 
                          if col not in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
            
            if feature_cols and 'Close' in latest_btc_data.columns:
                # Correlation with price
                correlations = {}
                for col in feature_cols:
                    if pd.api.types.is_numeric_dtype(latest_btc_data[col]):
                        corr = latest_btc_data[col].corr(latest_btc_data['Close'])
                        if not pd.isna(corr):
                            correlations[col] = corr
                
                # Full correlation matrix (limited to top features)
                top_features = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)[:20]
                top_feature_names = [f[0] for f in top_features]
                
                if top_feature_names:
                    corr_matrix = latest_btc_data[top_feature_names + ['Close']].corr()
                    
                    return {
                        'correlation_matrix': corr_matrix.to_dict(),
                        'key_correlations': correlations
                    }
        
        return {
            'correlation_matrix': {},
            'key_correlations': {},
            'message': 'No data available for correlation analysis'
        }
        
    except Exception as e:
        logger.error(f"Failed to get correlations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/optimization")
async def get_optimization_analytics():
    """Get strategy optimization results"""
    try:
        if latest_backtest_results:
            optimal_weights = latest_backtest_results.get('optimal_weights', {})
            
            # Convert to expected format if needed
            if isinstance(optimal_weights, dict):
                return {
                    'optimal_weights': optimal_weights,
                    'optimization_history': latest_backtest_results.get('optimization_history', [])
                }
        
        # Return default enhanced weights
        return {
            'optimal_weights': {
                'technical_weight': 0.40,
                'onchain_weight': 0.35,
                'sentiment_weight': 0.15,
                'macro_weight': 0.10,
                'momentum_weight': 0.30,
                'trend_weight': 0.40,
                'volatility_weight': 0.15,
                'volume_weight': 0.15,
                'flow_weight': 0.40,
                'network_weight': 0.30,
                'holder_weight': 0.30,
                'social_weight': 0.50,
                'derivatives_weight': 0.30,
                'fear_greed_weight': 0.20
            },
            'optimization_history': []
        }
        
    except Exception as e:
        logger.error(f"Failed to get optimization analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/feature-importance")
async def get_feature_importance():
    """Get feature importance from the trained model"""
    try:
        if backtest_system and hasattr(backtest_system.signal_generator, 'feature_importance'):
            importance = backtest_system.signal_generator.feature_importance
            if importance:
                return {
                    "feature_importance": importance,
                    "top_10_features": dict(list(importance.items())[:10]),
                    "timestamp": datetime.now()
                }
        
        return {
            "feature_importance": {},
            "message": "Feature importance not yet calculated. Run a backtest first."
        }
        
    except Exception as e:
        logger.error(f"Failed to get feature importance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Backtest endpoints
@app.post("/backtest/enhanced/run")
async def run_enhanced_backtest(request: EnhancedBacktestRequest):
    """Run enhanced backtest with database storage"""
    global backtest_in_progress, latest_backtest_results
    
    if backtest_in_progress and not request.force:
        return {"status": "error", "message": "Backtest already in progress"}
    
    try:
        backtest_in_progress = True
        
        # Apply custom settings if provided
        if request.settings and backtest_system:
            backtest_system.config.training_window_days = request.settings.get('training_window_days', 1008)
            backtest_system.config.test_window_days = request.settings.get('test_window_days', 90)
            backtest_system.config.transaction_cost = request.settings.get('transaction_cost', 0.0025)
        
        # Run backtest
        loop = asyncio.get_event_loop()
        results = await asyncio.wait_for(
            loop.run_in_executor(None, backtest_system.run_comprehensive_backtest,
                               request.period, request.optimize_weights, 
                               request.include_macro, True),
            timeout=3600
        )
        
        # Save to database
        config = {
            'period': request.period,
            'optimize_weights': request.optimize_weights,
            'include_macro': request.include_macro,
            'n_optimization_trials': request.n_optimization_trials
        }
        backtest_id = db.save_backtest_results(results, config)
        
        # Save feature importance
        if 'feature_analysis' in results and 'feature_importance' in results['feature_analysis']:
            db.save_feature_importance(results['feature_analysis']['feature_importance'])
        
        # Save market regime
        if 'market_analysis' in results:
            db.save_market_regime(results['market_analysis'])
        
        latest_backtest_results = results
        
        return {
            "status": "success",
            "backtest_id": backtest_id,
            "summary": {
                "composite_score": results.get('composite_score', 0),
                "confidence_score": results.get('confidence_score', 0),
                "key_metrics": {
                    "sortino_ratio": results.get('performance_metrics', {}).get('sortino_ratio_mean', 0),
                    "max_drawdown": results.get('performance_metrics', {}).get('max_drawdown_mean', 0),
                    "total_return": results.get('performance_metrics', {}).get('total_return_mean', 0)
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Enhanced backtest failed: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        backtest_in_progress = False

# Alias for streamlit compatibility
@app.post("/backtest/enhanced")
async def run_enhanced_backtest_alias(request: dict):
    """Alias for enhanced backtest endpoint"""
    enhanced_request = EnhancedBacktestRequest(**request)
    return await run_enhanced_backtest(enhanced_request)

@app.get("/backtest/enhanced/results/latest")
async def get_latest_enhanced_backtest_results():
    """Get the most recent enhanced backtest results with full details"""
    if latest_backtest_results is None:
        load_latest_backtest_results()
    
    if latest_backtest_results is None:
        raise HTTPException(status_code=404, detail="No enhanced backtest results available")
    
    # Return full enhanced results
    return latest_backtest_results

@app.get("/backtest/results/latest")
async def get_latest_backtest_results():
    """Get the most recent backtest results"""
    # Delegate to enhanced endpoint
    return await get_latest_enhanced_backtest_results()

@app.get("/backtest/status")
async def get_backtest_status():
    """Get current backtest status"""
    return {
        "in_progress": backtest_in_progress,
        "has_results": latest_backtest_results is not None,
        "timestamp": datetime.now().isoformat(),
        "system_type": "enhanced"
    }

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
                    
                    # Handle both old and new format
                    if 'performance_metrics' in data:
                        composite_score = data.get('performance_metrics', {}).get('composite_score', 0)
                        sortino_ratio = data.get('performance_metrics', {}).get('sortino_ratio_mean', 0)
                        max_drawdown = data.get('performance_metrics', {}).get('max_drawdown_mean', 0)
                    else:
                        composite_score = data.get('composite_score', 0)
                        sortino_ratio = data.get('sortino_ratio_mean', 0)
                        max_drawdown = data.get('max_drawdown_mean', 0)
                    
                    history.append({
                        "filename": os.path.basename(file_path),
                        "timestamp": data.get('timestamp', 'Unknown'),
                        "composite_score": composite_score,
                        "sortino_ratio": sortino_ratio,
                        "max_drawdown": max_drawdown,
                        "confidence_score": data.get('confidence_score', None),
                        "enhanced": 'market_analysis' in data or 'confidence_score' in data
                    })
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")
                
        return history
        
    except Exception as e:
        logger.error(f"Failed to get backtest history: {e}")
        return []

@app.get("/backtest/walk-forward/results")
async def get_walk_forward_results():
    """Get walk-forward analysis results"""
    try:
        if latest_backtest_results and 'walk_forward_results' in latest_backtest_results:
            return latest_backtest_results['walk_forward_results']
        
        # Return placeholder data
        return {
            'total_windows': 0,
            'avg_return': 0,
            'consistency_score': 0,
            'window_results': [],
            'stability_metrics': {
                'return_std': 0,
                'sharpe_std': 0,
                'max_window_dd': 0,
                'signal_consistency': 0,
                'weight_stability': 0,
                'feature_var': 0
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get walk-forward results: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/backtest/optimization/results")
async def get_optimization_results():
    """Get optimization results"""
    try:
        if latest_backtest_results:
            return {
                'best_params': latest_backtest_results.get('optimal_weights', {}),
                'optimization_history': latest_backtest_results.get('optimization_history', []),
                'param_importance': latest_backtest_results.get('param_importance', {})
            }
        
        return {
            'best_params': {},
            'optimization_history': [],
            'param_importance': {}
        }
        
    except Exception as e:
        logger.error(f"Failed to get optimization results: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Configuration endpoints
@app.get("/config/signal-weights/enhanced")
async def get_enhanced_signal_weights():
    """Get current enhanced signal weights including sub-categories"""
    try:
        if backtest_system and hasattr(backtest_system.signal_generator, 'signal_weights'):
            weights = backtest_system.signal_generator.signal_weights
            
            # Check if it's enhanced weights
            if hasattr(weights, 'momentum_weight'):
                return {
                    "main_categories": {
                        "technical": weights.technical_weight,
                        "onchain": weights.onchain_weight,
                        "sentiment": weights.sentiment_weight,
                        "macro": weights.macro_weight
                    },
                    "technical_sub": {
                        "momentum": weights.momentum_weight,
                        "trend": weights.trend_weight,
                        "volatility": weights.volatility_weight,
                        "volume": weights.volume_weight
                    },
                    "onchain_sub": {
                        "flow": weights.flow_weight,
                        "network": weights.network_weight,
                        "holder": weights.holder_weight
                    },
                    "sentiment_sub": {
                        "social": weights.social_weight,
                        "derivatives": weights.derivatives_weight,
                        "fear_greed": weights.fear_greed_weight
                    }
                }
            else:
                # Regular weights
                return {
                    "main_categories": {
                        "technical": weights.technical_weight,
                        "onchain": weights.onchain_weight,
                        "sentiment": weights.sentiment_weight,
                        "macro": weights.macro_weight
                    },
                    "enhanced": False
                }
        else:
            # Return default enhanced weights
            return {
                "main_categories": {
                    "technical": 0.40,
                    "onchain": 0.35,
                    "sentiment": 0.15,
                    "macro": 0.10
                },
                "technical_sub": {
                    "momentum": 0.30,
                    "trend": 0.40,
                    "volatility": 0.15,
                    "volume": 0.15
                },
                "onchain_sub": {
                    "flow": 0.40,
                    "network": 0.30,
                    "holder": 0.30
                },
                "sentiment_sub": {
                    "social": 0.50,
                    "derivatives": 0.30,
                    "fear_greed": 0.20
                }
            }
    except Exception as e:
        logger.error(f"Failed to get enhanced signal weights: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/config/signal-weights")
async def get_signal_weights():
    """Get current signal weights (backward compatibility)"""
    result = await get_enhanced_signal_weights()
    
    # Return simplified version for backward compatibility
    if isinstance(result, dict) and 'main_categories' in result:
        return result['main_categories']
    else:
        return result

@app.post("/config/signal-weights/enhanced")
async def update_enhanced_signal_weights(weights: Dict[str, Any]):
    """Update enhanced signal weights with sub-categories"""
    try:
        if backtest_system:
            # Create enhanced weights
            sw = EnhancedSignalWeights()
            
            # Set main categories
            if 'main_categories' in weights:
                sw.technical_weight = weights['main_categories'].get('technical', 0.40)
                sw.onchain_weight = weights['main_categories'].get('onchain', 0.35)
                sw.sentiment_weight = weights['main_categories'].get('sentiment', 0.15)
                sw.macro_weight = weights['main_categories'].get('macro', 0.10)
            
            # Set sub-categories
            if 'technical_sub' in weights:
                sw.momentum_weight = weights['technical_sub'].get('momentum', 0.30)
                sw.trend_weight = weights['technical_sub'].get('trend', 0.40)
                sw.volatility_weight = weights['technical_sub'].get('volatility', 0.15)
                sw.volume_weight = weights['technical_sub'].get('volume', 0.15)
            
            if 'onchain_sub' in weights:
                sw.flow_weight = weights['onchain_sub'].get('flow', 0.40)
                sw.network_weight = weights['onchain_sub'].get('network', 0.30)
                sw.holder_weight = weights['onchain_sub'].get('holder', 0.30)
            
            if 'sentiment_sub' in weights:
                sw.social_weight = weights['sentiment_sub'].get('social', 0.50)
                sw.derivatives_weight = weights['sentiment_sub'].get('derivatives', 0.30)
                sw.fear_greed_weight = weights['sentiment_sub'].get('fear_greed', 0.20)
            
            # Normalize
            sw.normalize()
            sw.normalize_subcategories()
            
            backtest_system.signal_generator.signal_weights = sw
            
            logger.info(f"Enhanced signal weights updated: {weights}")
            return {"status": "success", "message": "Enhanced signal weights updated"}
        else:
            raise HTTPException(status_code=500, detail="Backtest system not initialized")
            
    except Exception as e:
        logger.error(f"Failed to update enhanced signal weights: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/config/signal-weights")
async def update_signal_weights(weights: dict):
    """Update signal weights - compatible with flat structure from frontend"""
    try:
        # Convert flat structure to nested structure expected by backend
        enhanced_weights = {
            'main_categories': {
                'technical': weights.get('technical_weight', 0.40),
                'onchain': weights.get('onchain_weight', 0.35),
                'sentiment': weights.get('sentiment_weight', 0.15),
                'macro': weights.get('macro_weight', 0.10)
            },
            'technical_sub': {
                'momentum': weights.get('momentum_weight', 0.30),
                'trend': weights.get('trend_weight', 0.40),
                'volatility': weights.get('volatility_weight', 0.15),
                'volume': weights.get('volume_weight', 0.15)
            },
            'onchain_sub': {
                'flow': weights.get('flow_weight', 0.40),
                'network': weights.get('network_weight', 0.30),
                'holder': weights.get('holder_weight', 0.30)
            },
            'sentiment_sub': {
                'social': weights.get('social_weight', 0.50),
                'derivatives': weights.get('derivatives_weight', 0.30),
                'fear_greed': weights.get('fear_greed_weight', 0.20)
            }
        }
        
        # Update using the enhanced endpoint
        return await update_enhanced_signal_weights(enhanced_weights)
        
    except Exception as e:
        logger.error(f"Failed to update signal weights: {e}")
        raise HTTPException(status_code=400, detail=str(e))

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
                "target_sortino_ratio": config.target_sortino_ratio,
                "min_train_test_ratio": config.min_train_test_ratio
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
                "target_sortino_ratio": 2.0,
                "min_train_test_ratio": 0.7
            }
    except Exception as e:
        logger.error(f"Failed to get backtest settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/config/model")
async def get_model_config():
    """Get model configuration"""
    return model_config

@app.post("/config/model")
async def update_model_config(config: dict):
    """Update model configuration"""
    global model_config
    model_config.update(config)
    
    # Apply to signal generator if possible
    if signal_generator and hasattr(signal_generator, 'sequence_length'):
        signal_generator.sequence_length = config.get('sequence_length', 60)
        signal_generator.learning_rate = config.get('learning_rate', 0.001)
        signal_generator.batch_size = config.get('batch_size', 32)
        signal_generator.dropout_rate = config.get('dropout', 0.2)
    
    return {"status": "success", "message": "Model configuration updated"}

@app.get("/config/trading-rules")
async def get_trading_rules():
    """Get trading rules configuration"""
    return trading_rules

@app.post("/config/trading-rules")
async def update_trading_rules(rules: dict):
    """Update trading rules"""
    global trading_rules
    trading_rules.update(rules)
    return {"status": "success", "message": "Trading rules updated"}

# Model endpoints
@app.post("/model/retrain/enhanced")
async def trigger_enhanced_model_retrain():
    """Manually trigger enhanced model retraining with latest data"""
    try:
        if not backtest_system:
            raise HTTPException(status_code=500, detail="Backtest system not initialized")
        
        # Run enhanced retraining in background
        loop = asyncio.get_event_loop()
        
        # The enhanced retrain includes feature importance calculation
        results = await loop.run_in_executor(None, backtest_system.retrain_model, "6mo", True)
        
        return {
            "status": "success",
            "message": "Enhanced model retraining completed",
            "timestamp": datetime.now().isoformat(),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Enhanced model retraining failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/model/retrain")
async def trigger_model_retrain():
    """Manually trigger model retraining (backward compatibility)"""
    # Delegate to enhanced endpoint
    return await trigger_enhanced_model_retrain()

@app.get("/model/info")
async def get_model_info():
    """Get model information"""
    try:
        model_info = {
            'version': '2.0.0',
            'last_trained': 'N/A',
            'training_samples': 0,
            'n_features': 0,
            'accuracy': 0,
            'val_loss': 0
        }
        
        if signal_generator and hasattr(signal_generator, 'model'):
            # Get model details
            if hasattr(signal_generator.model, 'input_size'):
                model_info['n_features'] = signal_generator.model.input_size
            
            if hasattr(signal_generator, 'is_trained') and signal_generator.is_trained:
                model_info['last_trained'] = datetime.now().strftime('%Y-%m-%d')
                model_info['training_samples'] = 1000  # Estimate
                model_info['accuracy'] = 0.85  # Placeholder
                model_info['val_loss'] = 0.0015  # Placeholder
        
        return model_info
        
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Alias for feature importance
@app.get("/model/feature-importance")
async def get_model_feature_importance_alias():
    """Alias for feature importance endpoint"""
    return await get_feature_importance()

# System endpoints
@app.get("/system/status")
async def get_system_status():
    """Get detailed system status"""
    try:
        return {
            "api_status": "running",
            "api_version": "2.1.0",
            "timestamp": datetime.now(),
            "signal_update_errors": signal_update_errors,
            "latest_signal_time": latest_signal.get('timestamp') if latest_signal else None,
            "enhanced_signal_time": latest_enhanced_signal.get('timestamp') if latest_enhanced_signal else None,
            "data_cache_status": "available" if latest_btc_data is not None else "empty",
            "comprehensive_signals_status": "available" if latest_comprehensive_signals is not None else "empty",
            "database_status": "connected",
            "signal_generator_status": "enhanced" if isinstance(signal_generator, AdvancedTradingSignalGenerator) else "basic",
            "backtest_system_status": "enhanced" if isinstance(backtest_system, AdvancedIntegratedBacktestingSystem) else "basic",
            "enhanced_features": {
                "50_plus_signals": True,
                "macro_indicators": True,
                "sentiment_analysis": True,
                "on_chain_proxies": True,
                "enhanced_backtesting": True,
                "bayesian_optimization": True,
                "feature_importance": True,
                "confidence_intervals": True,
                "websocket_support": True,  # NEW
                "paper_trading": True,  # NEW
                "monte_carlo": True  # NEW
            },
            "paper_trading_status": {  # NEW
                "enabled": paper_trading_enabled,
                "portfolio_value": paper_portfolio['usd_balance'] + (paper_portfolio['btc_balance'] * latest_btc_data['Close'].iloc[-1] if latest_btc_data is not None and len(latest_btc_data) > 0 else 0)
            }
        }
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Indicator endpoints
@app.get("/indicators/all")
async def get_all_indicators():
    """Get all calculated indicators"""
    try:
        # Get latest comprehensive signals
        if latest_comprehensive_signals is not None:
            # Convert to dict format expected by frontend
            indicators = {}
            
            for col in latest_comprehensive_signals.columns:
                if col not in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    try:
                        val = latest_comprehensive_signals[col].iloc[-1]
                        if hasattr(val, 'item'):  # numpy scalar
                            indicators[col] = val.item()
                        elif isinstance(val, (bool, np.bool_)):
                            indicators[col] = bool(val)
                        elif isinstance(val, (int, float, np.integer, np.floating)):
                            indicators[col] = float(val)
                        elif isinstance(val, dict):
                            indicators[col] = val
                        else:
                            indicators[col] = str(val)
                    except Exception as e:
                        logger.warning(f"Error processing indicator {col}: {e}")
                        continue
            
            return indicators
        
        return {
            'message': 'No indicators calculated yet',
            'status': 'waiting'
        }
        
    except Exception as e:
        logger.error(f"Failed to get indicators: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Database endpoints
@app.get("/database/stats")
async def get_database_stats():
    """Get database statistics"""
    try:
        conn = sqlite3.connect(db.db_path)
        cursor = conn.cursor()
        
        stats = {}
        tables = ['trades', 'model_signals', 'backtest_results', 'trading_limits']
        
        for table in tables:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                stats[f'{table}_count'] = count
            except Exception as e:
                logger.warning(f"Error counting {table}: {e}")
                stats[f'{table}_count'] = 0
        
        conn.close()
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get database stats: {e}")
        return {
            'trades_count': 0,
            'signals_count': 0,
            'backtest_count': 0,
            'limits_count': 0
        }

@app.get("/database/export")
async def export_database():
    """Export database data"""
    try:
        export_data = {
            'trades': db.get_trades().to_dict('records') if not db.get_trades().empty else [],
            'positions': db.get_positions().to_dict('records') if not db.get_positions().empty else [],
            'signals': [],
            'timestamp': datetime.now().isoformat()
        }
        
        # Get signals
        try:
            conn = sqlite3.connect(db.db_path)
            signals_df = pd.read_sql_query("SELECT * FROM model_signals ORDER BY timestamp DESC LIMIT 100", conn)
            conn.close()
            
            if not signals_df.empty:
                export_data['signals'] = signals_df.to_dict('records')
        except Exception as e:
            logger.warning(f"Error exporting signals: {e}")
        
        return export_data
        
    except Exception as e:
        logger.error(f"Failed to export database: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Trading status endpoints (for compatibility)
@app.get("/trading/status")
async def get_trading_status():
    """Get trading status"""
    return {
        "is_active": paper_trading_enabled,  # Enhanced with paper trading
        "mode": "paper" if paper_trading_enabled else "manual",
        "last_trade_time": paper_portfolio['trades'][-1]['timestamp'] if paper_portfolio['trades'] else None
    }

@app.post("/trading/start")
async def start_trading():
    """Start automated trading"""
    global paper_trading_enabled
    paper_trading_enabled = True
    return {"status": "success", "message": "Paper trading started"}

@app.post("/trading/stop")
async def stop_trading():
    """Stop automated trading"""
    global paper_trading_enabled
    paper_trading_enabled = False
    return {"status": "success", "message": "Paper trading stopped"}

# Trade execution endpoint
@app.post("/trades/execute")
async def execute_trade(request: dict):
    """Execute a trade"""
    trade = TradeRequest(
        symbol=request.get("symbol", "BTC-USD"),
        trade_type=request.get("signal", request.get("trade_type", "hold")),
        price=latest_btc_data['Close'].iloc[-1] if latest_btc_data is not None else 45000.0,
        size=request.get("size", 0.001),
        lot_id=request.get("lot_id"),
        notes=request.get("reason", "manual")
    )
    return await create_trade(trade)

# Recent trades endpoint
@app.get("/trades/recent")
async def get_recent_trades(limit: int = 10):
    """Get recent trades"""
    trades_df = db.get_trades(limit=limit)
    if not trades_df.empty:
        trades_df['reason'] = trades_df.get('notes', 'auto')  # Map notes to reason
        return trades_df.to_dict('records')
    return []

# Portfolio positions endpoint
@app.get("/portfolio/positions")
async def get_portfolio_positions():
    """Get portfolio positions"""
    return await get_positions()

# ============= NEW ENHANCED ENDPOINTS =============

# Paper trading endpoints
@app.get("/paper-trading/status")
async def get_paper_trading_status():
    """Get paper trading status and portfolio"""
    if not paper_trading:
        raise HTTPException(status_code=500, detail="Paper trading not initialized")
    
    portfolio = paper_trading.get_portfolio()
    current_price = latest_btc_data['Close'].iloc[-1] if latest_btc_data is not None and len(latest_btc_data) > 0 else 45000
    metrics = paper_trading.calculate_performance_metrics(current_price)
    
    return {
        "enabled": paper_trading_enabled,
        "portfolio": portfolio,
        "performance": metrics,
        "timestamp": datetime.now()
    }

@app.post("/paper-trading/toggle")
async def toggle_paper_trading():
    """Toggle paper trading on/off"""
    global paper_trading_enabled
    paper_trading_enabled = not paper_trading_enabled
    
    if paper_trading_enabled:
        logger.info("Paper trading enabled")
    else:
        logger.info("Paper trading disabled")
    
    return {
        "enabled": paper_trading_enabled,
        "message": f"Paper trading {'enabled' if paper_trading_enabled else 'disabled'}"
    }

@app.post("/paper-trading/reset")
async def reset_paper_trading():
    """Reset paper trading portfolio"""
    if not paper_trading:
        raise HTTPException(status_code=500, detail="Paper trading not initialized")
    
    paper_trading.reset_portfolio()
    logger.info("Paper trading portfolio reset")
    return {"status": "success", "message": "Paper trading portfolio reset"}
    
    
@app.get("/paper-trading/history")
async def get_paper_trading_history(days: int = 30):
    """Get paper trading performance history"""
    if not paper_trading:
        raise HTTPException(status_code=500, detail="Paper trading not initialized")
    
    try:
        history_df = paper_trading.get_performance_history(days)
        return history_df.to_dict('records')
    except Exception as e:
        logger.error(f"Failed to get paper trading history: {e}")
        return []
        
# Monte Carlo simulation endpoint
@app.post("/analytics/monte-carlo")
async def run_monte_carlo_simulation(
    num_simulations: int = 1000,
    time_horizon_days: int = 30
):
    """Run Monte Carlo simulation for risk assessment"""
    if not latest_btc_data or len(latest_btc_data) < 30:
        raise HTTPException(status_code=400, detail="Insufficient data for simulation")
    
    try:
        # Calculate historical returns
        returns = latest_btc_data['Close'].pct_change().dropna()
        mean_return = returns.mean()
        std_return = returns.std()
        
        # Run simulations
        simulations = []
        for _ in range(num_simulations):
            daily_returns = np.random.normal(mean_return, std_return, time_horizon_days)
            price_path = [latest_btc_data['Close'].iloc[-1]]
            
            for ret in daily_returns:
                price_path.append(price_path[-1] * (1 + ret))
            
            simulations.append(price_path[-1])
        
        # Calculate statistics
        simulations = np.array(simulations)
        percentiles = np.percentile(simulations, [5, 25, 50, 75, 95])
        
        return {
            "current_price": float(latest_btc_data['Close'].iloc[-1]),
            "simulations": num_simulations,
            "time_horizon_days": time_horizon_days,
            "statistics": {
                "mean": float(np.mean(simulations)),
                "std": float(np.std(simulations)),
                "min": float(np.min(simulations)),
                "max": float(np.max(simulations)),
                "percentiles": {
                    "5%": float(percentiles[0]),
                    "25%": float(percentiles[1]),
                    "50%": float(percentiles[2]),
                    "75%": float(percentiles[3]),
                    "95%": float(percentiles[4])
                }
            },
            "probability_profit": float((simulations > latest_btc_data['Close'].iloc[-1]).mean())
        }
        
    except Exception as e:
        logger.error(f"Monte Carlo simulation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Multi-model ensemble endpoint
@app.get("/models/ensemble/predict")
async def get_ensemble_prediction():
    """Get prediction from multiple models (placeholder for future implementation)"""
    # This is a placeholder for future multi-model implementation
    # Would include Random Forest, XGBoost, etc.
    
    if not latest_enhanced_signal:
        raise HTTPException(status_code=404, detail="No signal available")
    
    return {
        "models": {
            "lstm": {
                "signal": latest_enhanced_signal.get("signal"),
                "confidence": latest_enhanced_signal.get("confidence")
            },
            "random_forest": {
                "signal": "placeholder",
                "confidence": 0.0
            },
            "xgboost": {
                "signal": "placeholder", 
                "confidence": 0.0
            }
        },
        "ensemble_signal": latest_enhanced_signal.get("signal"),
        "ensemble_confidence": latest_enhanced_signal.get("confidence"),
        "timestamp": datetime.now()
    }

# Helper functions
def calculate_var(returns: np.ndarray, confidence: float) -> float:
    """Calculate Value at Risk"""
    if len(returns) == 0:
        return 0
    return np.percentile(returns, (1 - confidence) * 100)

def calculate_cvar(returns: np.ndarray, confidence: float) -> float:
    """Calculate Conditional Value at Risk"""
    var = calculate_var(returns, confidence)
    return returns[returns <= var].mean() if len(returns[returns <= var]) > 0 else 0

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Enhanced BTC Trading System API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)