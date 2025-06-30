import sqlite3
import pandas as pd
import os
from datetime import datetime
from typing import List, Dict, Optional
import uuid

class DatabaseManager:
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = os.getenv('DATABASE_PATH', '/app/data/trading_system.db')
        self.db_path = db_path
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        self.init_database()
    
    def init_database(self):
        """Initialize database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create trades table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                trade_type TEXT NOT NULL,
                price REAL NOT NULL,
                size REAL NOT NULL,
                lot_id TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'executed'
            )
        ''')
        
        # Create model signals table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                signal TEXT NOT NULL,
                confidence REAL NOT NULL,
                price_prediction REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                model_version TEXT DEFAULT 'v1.0'
            )
        ''')
        
        # Create limits table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trading_limits (
                id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                limit_type TEXT NOT NULL,
                price REAL NOT NULL,
                size REAL,
                lot_id TEXT,
                active BOOLEAN DEFAULT 1,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create portfolio positions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS positions (
                lot_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                total_size REAL NOT NULL,
                available_size REAL NOT NULL,
                avg_buy_price REAL NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_trade(self, symbol: str, trade_type: str, price: float, size: float, lot_id: str = None) -> str:
        """Add a new trade to the database"""
        if lot_id is None:
            lot_id = str(uuid.uuid4())
        
        trade_id = str(uuid.uuid4())
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO trades (id, symbol, trade_type, price, size, lot_id)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (trade_id, symbol, trade_type, price, size, lot_id))
        
        # Update positions
        self._update_position(cursor, symbol, trade_type, price, size, lot_id)
        
        conn.commit()
        conn.close()
        return trade_id
    
    def _update_position(self, cursor, symbol: str, trade_type: str, price: float, size: float, lot_id: str):
        """Update position based on trade"""
        cursor.execute('SELECT * FROM positions WHERE lot_id = ?', (lot_id,))
        position = cursor.fetchone()
        
        if trade_type == 'buy':
            if position:
                current_size = position[2]
                current_avg_price = position[4]
                
                new_total_size = current_size + size
                new_avg_price = ((current_size * current_avg_price) + (size * price)) / new_total_size
                
                cursor.execute('''
                    UPDATE positions 
                    SET total_size = ?, available_size = ?, avg_buy_price = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE lot_id = ?
                ''', (new_total_size, new_total_size, new_avg_price, lot_id))
            else:
                cursor.execute('''
                    INSERT INTO positions (lot_id, symbol, total_size, available_size, avg_buy_price)
                    VALUES (?, ?, ?, ?, ?)
                ''', (lot_id, symbol, size, size, price))
        
        elif trade_type == 'sell' and position:
            current_total = position[2]
            current_available = position[3]
            
            if current_available >= size:
                new_available = current_available - size
                new_total = current_total - size
                
                cursor.execute('''
                    UPDATE positions 
                    SET total_size = ?, available_size = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE lot_id = ?
                ''', (new_total, new_available, lot_id))
    
    def add_model_signal(self, symbol: str, signal: str, confidence: float, price_prediction: float = None):
        """Add model signal to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO model_signals (symbol, signal, confidence, price_prediction)
            VALUES (?, ?, ?, ?)
        ''', (symbol, signal, confidence, price_prediction))
        
        conn.commit()
        conn.close()
    
    def add_trading_limit(self, symbol: str, limit_type: str, price: float, size: float = None, lot_id: str = None) -> str:
        """Add trading limit"""
        limit_id = str(uuid.uuid4())
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO trading_limits (id, symbol, limit_type, price, size, lot_id)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (limit_id, symbol, limit_type, price, size, lot_id))
        
        conn.commit()
        conn.close()
        return limit_id
    
    def get_trades(self, symbol: str = None, limit: int = None) -> pd.DataFrame:
        """Get trades data"""
        conn = sqlite3.connect(self.db_path)
        
        query = "SELECT * FROM trades"
        params = []
        
        if symbol:
            query += " WHERE symbol = ?"
            params.append(symbol)
        
        query += " ORDER BY timestamp DESC"
        
        if limit:
            query += f" LIMIT {limit}"
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        return df
    
    def get_positions(self) -> pd.DataFrame:
        """Get current positions"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM positions WHERE total_size > 0", conn)
        conn.close()
        return df
    
    def get_model_signals(self, symbol: str = None, limit: int = 10) -> pd.DataFrame:
        """Get recent model signals"""
        conn = sqlite3.connect(self.db_path)
        
        query = "SELECT * FROM model_signals"
        params = []
        
        if symbol:
            query += " WHERE symbol = ?"
            params.append(symbol)
        
        query += " ORDER BY timestamp DESC"
        
        if limit:
            query += f" LIMIT {limit}"
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        return df
    
    def get_trading_limits(self, active_only: bool = True) -> pd.DataFrame:
        """Get trading limits"""
        conn = sqlite3.connect(self.db_path)
        
        query = "SELECT * FROM trading_limits"
        if active_only:
            query += " WHERE active = 1"
        
        query += " ORDER BY created_at DESC"
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    
    def get_portfolio_metrics(self) -> Dict:
        """Calculate portfolio metrics"""
        conn = sqlite3.connect(self.db_path)
        
        trades_df = pd.read_sql_query("SELECT * FROM trades", conn)
        positions_df = pd.read_sql_query("SELECT * FROM positions WHERE total_size > 0", conn)
        
        conn.close()
        
        if trades_df.empty:
            return {
                'total_trades': 0,
                'total_volume': 0,
                'total_pnl': 0,
                'positions_count': 0,
                'total_invested': 0
            }
        
        total_trades = len(trades_df)
        total_volume = trades_df['size'].sum()
        
        buy_trades = trades_df[trades_df['trade_type'] == 'buy']
        sell_trades = trades_df[trades_df['trade_type'] == 'sell']
        
        total_bought = (buy_trades['price'] * buy_trades['size']).sum()
        total_sold = (sell_trades['price'] * sell_trades['size']).sum()
        
        total_pnl = total_sold - total_bought
        
        return {
            'total_trades': total_trades,
            'total_volume': total_volume,
            'total_pnl': total_pnl,
            'positions_count': len(positions_df),
            'total_invested': total_bought
        }
