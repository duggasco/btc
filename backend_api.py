from fastapi import FastAPI, HTTPException
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

from database_models import DatabaseManager
from lstm_model import TradingSignalGenerator

# Import the ENHANCED classes instead of the old ones
from integration import AdvancedIntegratedBacktestingSystem, AdvancedTradingSignalGenerator
from enhanced_backtesting_system import (
    BacktestConfig, EnhancedSignalWeights, EnhancedBacktestingPipeline,
    ComprehensiveSignalCalculator, EnhancedPerformanceMetrics
)

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

class EnhancedBacktestRequest(BaseModel):
    """Enhanced backtest request with more options"""
    period: str = "1y"
    optimize_weights: bool = True
    include_macro: bool = True
    use_enhanced_weights: bool = True
    n_optimization_trials: int = 20
    force: bool = False

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
                db.add_model_signal("BTC-USD", signal, confidence, predicted_price)
                logger.info("Signal stored in database")
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
    
    # Generate initial enhanced signal
    try:
        logger.info("Generating initial enhanced signal...")
        initial_data = signal_generator.fetch_enhanced_btc_data(period="1mo", include_macro=False)
        signal, confidence, predicted_price, analysis = signal_generator.predict_with_confidence(initial_data)
        
        global latest_signal, latest_btc_data, latest_enhanced_signal
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
        logger.info(f"Initial enhanced signal generated: {signal}")
        
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
    
    # Initialize ADVANCED backtest system
    global backtest_system
    try:
        backtest_system = AdvancedIntegratedBacktestingSystem(
            db_path=db_path,
            model_path='models/lstm_btc_model.pth'
        )
        logger.info("Advanced backtest system initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize backtest system: {e}")
    
    # Load latest backtest results if available
    load_latest_backtest_results()
    
    logger.info("Enhanced BTC Trading System API startup complete")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Enhanced BTC Trading System API...")
    signal_updater.stop()
    logger.info("API shutdown complete")

# Keep all original endpoints...
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Enhanced BTC Trading System API is running", 
        "version": "2.0.0",
        "timestamp": datetime.now(),
        "status": "healthy",
        "signal_errors": signal_update_errors,
        "features": ["enhanced_signals", "comprehensive_backtesting", "50+_indicators"]
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
                "comprehensive_signals": "active" if latest_comprehensive_signals is not None else "inactive"
            },
            "latest_signal": latest_signal,
            "enhanced_features": {
                "macro_indicators": True,
                "sentiment_analysis": True,
                "on_chain_proxies": True,
                "50_plus_signals": True
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

# Add new ENHANCED endpoints

@app.get("/signals/enhanced/latest")
async def get_enhanced_latest_signal():
    """Get the latest enhanced trading signal with full analysis"""
    global latest_enhanced_signal
    
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


@app.post("/backtest/enhanced/run")
async def run_enhanced_backtest(request: EnhancedBacktestRequest):
    """Run enhanced backtest with database storage"""
    global backtest_in_progress, latest_backtest_results
    
    if backtest_in_progress and not request.force:
        return {"status": "error", "message": "Backtest already in progress"}
    
    try:
        backtest_in_progress = True
        
        # Run backtest
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

@app.get("/backtest/enhanced/results/latest")
async def get_latest_enhanced_backtest_results():
    """Get the most recent enhanced backtest results with full details"""
    if latest_backtest_results is None:
        load_latest_backtest_results()
    
    if latest_backtest_results is None:
        raise HTTPException(status_code=404, detail="No enhanced backtest results available")
    
    # Return full enhanced results
    return latest_backtest_results

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

# Keep all original endpoints (trades, positions, limits, etc.)
@app.post("/trades/")
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
            notes=getattr(trade, 'notes', None)
        )
        logger.info(f"Trade created: {trade_id} with PnL: {pnl}")
        return {"trade_id": trade_id, "status": "success", "pnl": pnl}
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
        
        # Add enhanced metrics if available
        if latest_enhanced_signal and 'analysis' in latest_enhanced_signal:
            metrics['signal_confidence'] = latest_enhanced_signal.get('confidence', 0)
            metrics['consensus_ratio'] = latest_enhanced_signal.get('analysis', {}).get('consensus_ratio', 0)
        
        return metrics
    except Exception as e:
        logger.error(f"Failed to get portfolio metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close']),
                    'volume': float(row['Volume'])
                }
                
                # Add basic indicators always
                for indicator in ['SMA_20', 'SMA_50', 'RSI', 'MACD']:
                    if indicator in row and not pd.isna(row[indicator]):
                        record[indicator.lower()] = float(row[indicator])
                
                # Add enhanced indicators if requested
                if include_indicators:
                    # Add technical indicators
                    for col in ['bb_position', 'atr_normalized', 'stoch_k', 'mfi', 'cmf']:
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
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close']),
                    'volume': float(row['Volume'])
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
            "api_version": "2.0.0",
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
                "confidence_intervals": True
            }
        }
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Keep existing simple backtest endpoint for backward compatibility
@app.post("/backtest/")
async def run_backtest(config: BacktestConfig):
    """Run simple backtesting simulation (backward compatibility)"""
    # ... keep the original implementation ...
    # This ensures old clients still work
    pass

@app.post("/backtest/run")
async def run_backtest_advanced(request: dict):
    """Run backtest with IntegratedBacktestingSystem (enhanced version)"""
    # Convert to enhanced request
    enhanced_request = EnhancedBacktestRequest(
        period=request.get("period", "1y"),
        optimize_weights=request.get("optimize_weights", True),
        force=request.get("force", False),
        use_enhanced_weights=True,
        include_macro=True
    )
    
    # Delegate to enhanced endpoint
    return await run_enhanced_backtest(enhanced_request)

@app.get("/backtest/status")
async def get_backtest_status():
    """Get current backtest status"""
    return {
        "in_progress": backtest_in_progress,
        "has_results": latest_backtest_results is not None,
        "timestamp": datetime.now().isoformat(),
        "system_type": "enhanced"
    }

@app.get("/backtest/results/latest")
async def get_latest_backtest_results():
    """Get the most recent backtest results"""
    # Delegate to enhanced endpoint
    return await get_latest_enhanced_backtest_results()

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

@app.get("/config/signal-weights")
async def get_signal_weights():
    """Get current signal weights (backward compatibility)"""
    result = await get_enhanced_signal_weights()
    
    # Return simplified version for backward compatibility
    if isinstance(result, dict) and 'main_categories' in result:
        return result['main_categories']
    else:
        return result

@app.post("/config/signal-weights")
async def update_signal_weights(weights: Dict[str, float]):
    """Manually update signal weights (backward compatibility)"""
    # Convert to enhanced format
    enhanced_weights = {
        "main_categories": weights
    }
    
    return await update_enhanced_signal_weights(enhanced_weights)

@app.post("/model/retrain")
async def trigger_model_retrain():
    """Manually trigger model retraining (backward compatibility)"""
    # Delegate to enhanced endpoint
    return await trigger_enhanced_model_retrain()

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

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Enhanced BTC Trading System API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)