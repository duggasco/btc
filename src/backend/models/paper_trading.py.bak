# paper_trading_persistence.py
"""
Persistent Paper Trading Module
Stores paper trading state in SQLite database
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional
import sqlite3
import uuid
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class PersistentPaperTrading:
    """Persistent paper trading with database storage"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_database()
        self.portfolio = self._load_portfolio()
        
    def _init_database(self):
        """Initialize paper trading tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Paper trading portfolio state
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS paper_portfolio (
                id INTEGER PRIMARY KEY,
                btc_balance REAL DEFAULT 0,
                usd_balance REAL DEFAULT 10000,
                total_pnl REAL DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Paper trading transactions
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS paper_trades (
                id TEXT PRIMARY KEY,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                trade_type TEXT NOT NULL,
                price REAL NOT NULL,
                amount REAL NOT NULL,
                value REAL NOT NULL,
                portfolio_id INTEGER,
                FOREIGN KEY (portfolio_id) REFERENCES paper_portfolio(id)
            )
        ''')
        
        # Paper trading performance metrics
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS paper_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                total_value REAL,
                daily_pnl REAL,
                win_rate REAL,
                sharpe_ratio REAL,
                max_drawdown REAL
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Paper trading database tables initialized")
        
    def _load_portfolio(self) -> Dict:
        """Load portfolio from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get latest portfolio state
        cursor.execute('''
            SELECT id, btc_balance, usd_balance, total_pnl 
            FROM paper_portfolio 
            ORDER BY created_at DESC 
            LIMIT 1
        ''')
        
        row = cursor.fetchone()
        if row:
            portfolio_id, btc_balance, usd_balance, total_pnl = row
            
            # Load trades
            cursor.execute('''
                SELECT id, timestamp, trade_type, price, amount, value
                FROM paper_trades
                WHERE portfolio_id = ?
                ORDER BY timestamp DESC
                LIMIT 100
            ''', (portfolio_id,))
            
            trades = []
            for trade_row in cursor.fetchall():
                trades.append({
                    'id': trade_row[0],
                    'timestamp': trade_row[1],
                    'type': trade_row[2],
                    'price': trade_row[3],
                    'amount': trade_row[4],
                    'value': trade_row[5]
                })
            
            conn.close()
            
            logger.info(f"Loaded portfolio ID {portfolio_id} with {len(trades)} trades")
            
            return {
                'id': portfolio_id,
                'btc_balance': btc_balance,
                'usd_balance': usd_balance,
                'total_pnl': total_pnl,
                'trades': trades
            }
        else:
            # Create new portfolio
            cursor.execute('''
                INSERT INTO paper_portfolio (btc_balance, usd_balance, total_pnl)
                VALUES (0, 10000, 0)
            ''')
            portfolio_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            logger.info(f"Created new portfolio ID {portfolio_id}")
            
            return {
                'id': portfolio_id,
                'btc_balance': 0.0,
                'usd_balance': 10000.0,
                'total_pnl': 0.0,
                'trades': []
            }
    
    def execute_trade(self, trade_type: str, price: float, amount: float, value: float) -> str:
        """Execute and persist a paper trade"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        trade_id = str(uuid.uuid4())
        
        try:
            # Insert trade
            cursor.execute('''
                INSERT INTO paper_trades (id, trade_type, price, amount, value, portfolio_id)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (trade_id, trade_type, price, amount, value, self.portfolio['id']))
            
            # Update portfolio
            if trade_type == 'buy':
                self.portfolio['btc_balance'] += amount
                self.portfolio['usd_balance'] -= value
            else:  # sell
                self.portfolio['btc_balance'] -= amount
                self.portfolio['usd_balance'] += value
            
            # Calculate total value and P&L
            total_value = self.portfolio['usd_balance'] + (self.portfolio['btc_balance'] * price)
            self.portfolio['total_pnl'] = total_value - 10000.0
            
            # Update portfolio in database
            cursor.execute('''
                UPDATE paper_portfolio
                SET btc_balance = ?, usd_balance = ?, total_pnl = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (self.portfolio['btc_balance'], self.portfolio['usd_balance'], 
                  self.portfolio['total_pnl'], self.portfolio['id']))
            
            # Add to trades list
            self.portfolio['trades'].append({
                'id': trade_id,
                'timestamp': datetime.now().isoformat(),
                'type': trade_type,
                'price': price,
                'amount': amount,
                'value': value
            })
            
            # Keep only last 100 trades in memory
            if len(self.portfolio['trades']) > 100:
                self.portfolio['trades'] = self.portfolio['trades'][-100:]
            
            conn.commit()
            logger.info(f"Paper trade executed: {trade_type} {amount:.6f} BTC at ${price:.2f}")
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to execute paper trade: {e}")
            raise
        finally:
            conn.close()
        
        return trade_id
    
    def get_portfolio(self) -> Dict:
        """Get current portfolio state"""
        return self.portfolio.copy()
    
    def calculate_performance_metrics(self, current_price: float) -> Dict:
        """Calculate performance metrics"""
        if not self.portfolio['trades']:
            return {
                'total_return': 0.0,
                'win_rate': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'trades_count': 0
            }
        
        # Calculate returns for each trade
        returns = []
        for i, trade in enumerate(self.portfolio['trades']):
            if trade['type'] == 'sell' and i > 0:
                # Find matching buy
                for j in range(i-1, -1, -1):
                    if self.portfolio['trades'][j]['type'] == 'buy':
                        buy_price = self.portfolio['trades'][j]['price']
                        sell_price = trade['price']
                        return_pct = (sell_price - buy_price) / buy_price
                        returns.append(return_pct)
                        break
        
        if not returns:
            return {
                'total_return': self.portfolio['total_pnl'] / 10000.0,
                'win_rate': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'trades_count': len(self.portfolio['trades'])
            }
        
        # Calculate metrics
        returns_array = np.array(returns)
        win_rate = (returns_array > 0).sum() / len(returns_array)
        
        # Sharpe ratio (simplified)
        if returns_array.std() > 0:
            sharpe_ratio = (returns_array.mean() * 252) / (returns_array.std() * np.sqrt(252))
        else:
            sharpe_ratio = 0.0
        
        # Max drawdown (simplified)
        cumulative_returns = (1 + returns_array).cumprod()
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0.0
        
        return {
            'total_return': self.portfolio['total_pnl'] / 10000.0,
            'win_rate': win_rate,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'trades_count': len(self.portfolio['trades'])
        }
    
    def save_performance_snapshot(self, current_price: float):
        """Save performance snapshot to database"""
        metrics = self.calculate_performance_metrics(current_price)
        total_value = self.portfolio['usd_balance'] + (self.portfolio['btc_balance'] * current_price)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO paper_performance (total_value, daily_pnl, win_rate, sharpe_ratio, max_drawdown)
            VALUES (?, ?, ?, ?, ?)
        ''', (total_value, self.portfolio['total_pnl'], metrics['win_rate'], 
              metrics['sharpe_ratio'], metrics['max_drawdown']))
        
        conn.commit()
        conn.close()
    
    def get_performance_history(self, days: int = 30) -> pd.DataFrame:
        """Get historical performance data"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT timestamp, total_value, daily_pnl, win_rate, sharpe_ratio, max_drawdown
            FROM paper_performance
            WHERE timestamp > datetime('now', '-{} days')
            ORDER BY timestamp DESC
        '''.format(days)
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        return df
    
    def reset_portfolio(self):
        """Reset portfolio to initial state"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create new portfolio
        cursor.execute('''
            INSERT INTO paper_portfolio (btc_balance, usd_balance, total_pnl)
            VALUES (0, 10000, 0)
        ''')
        
        new_portfolio_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        self.portfolio = {
            'id': new_portfolio_id,
            'btc_balance': 0.0,
            'usd_balance': 10000.0,
            'total_pnl': 0.0,
            'trades': []
        }
        
        logger.info("Paper trading portfolio reset")
