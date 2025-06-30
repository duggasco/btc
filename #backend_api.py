from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import pandas as pd
from datetime import datetime
import threading
import time
import os

from database_models import DatabaseManager
from lstm_model import TradingSignalGenerator

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

db_path = os.getenv('DATABASE_PATH', '/app/data/trading_system.db')
db = DatabaseManager(db_path)
signal_generator = TradingSignalGenerator()

latest_btc_data = None
latest_signal = None

class SignalUpdater:
    def __init__(self):
        self.running = False
        
    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self.update_signals_loop, daemon=True)
            self.thread.start()
            
    def stop(self):
        self.running = False
        
    def update_signals_loop(self):
        while self.running:
            try:
                self.update_signals()
                time.sleep(300)
            except Exception as e:
                print(f"Error updating signals: {e}")
                time.sleep(60)
                
    def update_signals(self):
        global latest_btc_data, latest_signal
        
        try:
            btc_data = signal_generator.fetch_btc_data(period="3mo")
            latest_btc_data = btc_data
            
            signal, confidence, predicted_price = signal_generator.predict_signal(btc_data)
            
            db.add_model_signal("BTC-USD", signal, confidence, predicted_price)
            
            latest_signal = {
                "symbol": "BTC-USD",
                "signal": signal,
                "confidence": confidence,
                "predicted_price": predicted_price,
                "timestamp": datetime.now()
            }
            
            print(f"Signal updated: {signal} (confidence: {confidence:.2%})")
            
        except Exception as e:
            print(f"Error in update_signals: {e}")

signal_updater = SignalUpdater()

@app.on_event("startup")
async def startup_event():
    print("Starting BTC Trading System API...")
    signal_updater.start()
    print("Signal updater started")

@app.on_event("shutdown")
async def shutdown_event():
    signal_updater.stop()
    print("API shutdown complete")

@app.get("/")
async def root():
    return {"message": "BTC Trading System API is running", "timestamp": datetime.now()}

@app.post("/trades/")
async def create_trade(trade: TradeRequest):
    try:
        trade_id = db.add_trade(
            symbol=trade.symbol,
            trade_type=trade.trade_type,
            price=trade.price,
            size=trade.size,
            lot_id=trade.lot_id
        )
        return {"trade_id": trade_id, "status": "success", "message": "Trade created successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/trades/")
async def get_trades(symbol: Optional[str] = None, limit: Optional[int] = 100):
    try:
        trades_df = db.get_trades(symbol=symbol, limit=limit)
        return trades_df.to_dict('records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/positions/")
async def get_positions():
    try:
        positions_df = db.get_positions()
        return positions_df.to_dict('records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/limits/")
async def create_limit_order(limit_order: LimitOrder):
    try:
        limit_id = db.add_trading_limit(
            symbol=limit_order.symbol,
            limit_type=limit_order.limit_type,
            price=limit_order.price,
            size=limit_order.size,
            lot_id=limit_order.lot_id
        )
        return {"limit_id": limit_id, "status": "success", "message": "Limit order created"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/limits/")
async def get_limits():
    try:
        limits_df = db.get_trading_limits()
        return limits_df.to_dict('records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/signals/latest")
async def get_latest_signal():
    global latest_signal
    
    if latest_signal is None:
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
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error generating signal: {str(e)}")
    
    return latest_signal

@app.get("/signals/history")
async def get_signal_history(limit: int = 50):
    try:
        signals_df = db.get_model_signals(limit=limit)
        return signals_df.to_dict('records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/portfolio/metrics")
async def get_portfolio_metrics():
    try:
        metrics = db.get_portfolio_metrics()
        
        if latest_btc_data is not None:
            metrics['current_btc_price'] = float(latest_btc_data['Close'].iloc[-1])
        
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/market/btc-data")
async def get_btc_data(period: str = "1mo"):
    try:
        btc_data = signal_generator.fetch_btc_data(period=period)
        
        data_records = []
        for idx, row in btc_data.iterrows():
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
        
        return {
            "symbol": "BTC-USD",
            "period": period,
            "data": data_records[-100:]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/pnl")
async def get_pnl_data():
    try:
        trades_df = db.get_trades()
        
        if trades_df.empty:
            return {"daily_pnl": [], "cumulative_pnl": []}
        
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
        trades_df['date'] = trades_df['timestamp'].dt.date
        trades_df['trade_value'] = trades_df['price'] * trades_df['size']
        trades_df['signed_value'] = trades_df['trade_value'] * trades_df['trade_type'].map({'buy': -1, 'sell': 1, 'hold': 0})
        
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
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
