import sqlite3
import pandas as pd
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import uuid
import json
import numpy as np
import time

class DatabaseManager:
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = os.getenv('DATABASE_PATH', '/app/data/trading_system.db')
        self.db_path = db_path
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        self.init_database()
    
    def initialize_database(self):
        """Alias for init_database for test compatibility"""
        self.init_database()
        
        # Also create test tables
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create test tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signals (
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                signal TEXT NOT NULL,
                confidence REAL NOT NULL,
                predicted_price REAL,
                current_price REAL,
                indicators TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS prices (
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                price REAL NOT NULL,
                volume REAL,
                high REAL,
                low REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS portfolio (
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                type TEXT NOT NULL,
                price REAL NOT NULL,
                amount REAL NOT NULL,
                total_value REAL NOT NULL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def init_database(self):
        """Initialize database with required tables including enhanced features"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Original tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                trade_type TEXT NOT NULL,
                price REAL NOT NULL,
                size REAL NOT NULL,
                lot_id TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'executed',
                pnl REAL DEFAULT 0,
                notes TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                signal TEXT NOT NULL,
                confidence REAL NOT NULL,
                price_prediction REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                model_version TEXT DEFAULT 'v1.0',
                analysis_data TEXT,
                signal_weights TEXT,
                comprehensive_signals TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trading_limits (
                id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                limit_type TEXT NOT NULL,
                price REAL NOT NULL,
                size REAL,
                lot_id TEXT,
                active BOOLEAN DEFAULT 1,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                triggered_at DATETIME,
                trigger_price REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS positions (
                lot_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                total_size REAL NOT NULL,
                available_size REAL NOT NULL,
                avg_buy_price REAL NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                realized_pnl REAL DEFAULT 0,
                unrealized_pnl REAL DEFAULT 0
            )
        ''')
        
        # New enhanced tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS backtest_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                period TEXT,
                composite_score REAL,
                confidence_score REAL,
                sortino_ratio REAL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                total_return REAL,
                win_rate REAL,
                total_trades INTEGER,
                results_json TEXT,
                config_json TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signal_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_name TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                mean_return REAL,
                total_contribution REAL,
                win_rate REAL,
                activation_count INTEGER,
                backtest_id INTEGER,
                FOREIGN KEY (backtest_id) REFERENCES backtest_results(id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feature_importance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                feature_name TEXT NOT NULL,
                importance_score REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                model_version TEXT,
                category TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_regime (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                regime_type TEXT,
                volatility_regime TEXT,
                dominant_trend TEXT,
                confidence REAL,
                indicators_json TEXT
            )
        ''')
        
        # Create indices for performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON model_signals(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol)')
        
        conn.commit()
        conn.close()
    
    # Enhanced methods for new functionality
    def add_enhanced_model_signal(self, symbol: str, signal: str, confidence: float, 
                                 price_prediction: float = None, analysis: Dict = None,
                                 signal_weights: Dict = None, comprehensive_signals: Dict = None):
        """Add enhanced model signal with analysis data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO model_signals (symbol, signal, confidence, price_prediction, 
                                     analysis_data, signal_weights, comprehensive_signals)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (symbol, signal, confidence, price_prediction,
              json.dumps(analysis) if analysis else None,
              json.dumps(signal_weights) if signal_weights else None,
              json.dumps(comprehensive_signals) if comprehensive_signals else None))
        
        conn.commit()
        conn.close()
    
    def save_backtest_results(self, results: Dict, config: Dict = None):
        """Save comprehensive backtest results"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Extract key metrics
        perf_metrics = results.get('performance_metrics', {})
        
        cursor.execute('''
            INSERT INTO backtest_results (
                period, composite_score, confidence_score, sortino_ratio,
                sharpe_ratio, max_drawdown, total_return, win_rate,
                total_trades, results_json, config_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            results.get('period', 'unknown'),
            results.get('composite_score', perf_metrics.get('composite_score', 0)),
            results.get('confidence_score', 0),
            perf_metrics.get('sortino_ratio_mean', 0),
            perf_metrics.get('sharpe_ratio_mean', 0),
            perf_metrics.get('max_drawdown_mean', 0),
            perf_metrics.get('total_return_mean', 0),
            perf_metrics.get('win_rate_mean', 0),
            perf_metrics.get('total_trades', 0),
            json.dumps(results),
            json.dumps(config) if config else None
        ))
        
        backtest_id = cursor.lastrowid
        
        # Save signal performance
        signal_perf = results.get('signal_analysis', {}).get('top_signals', {})
        for signal_name, perf in signal_perf.items():
            cursor.execute('''
                INSERT INTO signal_performance (
                    signal_name, mean_return, total_contribution,
                    win_rate, activation_count, backtest_id
                ) VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                signal_name,
                perf.get('avg_mean_return', 0),
                perf.get('total_contribution', 0),
                perf.get('avg_win_rate', 0),
                perf.get('total_count', 0),
                backtest_id
            ))
        
        conn.commit()
        conn.close()
        return backtest_id
    
    def save_feature_importance(self, importance_dict: Dict, model_version: str = 'v1.0'):
        """Save feature importance scores"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for feature, score in importance_dict.items():
            # Determine category
            category = 'technical'
            if any(x in feature.lower() for x in ['sentiment', 'fear', 'greed']):
                category = 'sentiment'
            elif any(x in feature.lower() for x in ['volume', 'transaction', 'nvt']):
                category = 'on_chain'
            elif any(x in feature.lower() for x in ['macro', 'sp500', 'gold', 'vix']):
                category = 'macro'
            
            cursor.execute('''
                INSERT INTO feature_importance (
                    feature_name, importance_score, model_version, category
                ) VALUES (?, ?, ?, ?)
            ''', (feature, score, model_version, category))
        
        conn.commit()
        conn.close()
    
    def save_market_regime(self, regime_data: Dict):
        """Save market regime analysis"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO market_regime (
                regime_type, volatility_regime, dominant_trend,
                confidence, indicators_json
            ) VALUES (?, ?, ?, ?, ?)
        ''', (
            regime_data.get('regime', 'Unknown'),
            regime_data.get('volatility_regime', 'Unknown'),
            regime_data.get('dominant_trend', 'Unknown'),
            regime_data.get('confidence', 0),
            json.dumps(regime_data.get('indicators', {}))
        ))
        
        conn.commit()
        conn.close()
    
    def get_latest_backtest_results(self) -> Dict:
        """Get the most recent backtest results"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM backtest_results
            ORDER BY timestamp DESC
            LIMIT 1
        ''')
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            columns = [desc[0] for desc in cursor.description]
            result = dict(zip(columns, row))
            # Parse JSON fields
            if result.get('results_json'):
                result['full_results'] = json.loads(result['results_json'])
            if result.get('config_json'):
                result['config'] = json.loads(result['config_json'])
            return result
        return None
    
    def get_signal_performance_history(self, signal_name: str = None, limit: int = 100) -> pd.DataFrame:
        """Get historical performance of signals"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT sp.*, br.timestamp as backtest_time, br.period
            FROM signal_performance sp
            JOIN backtest_results br ON sp.backtest_id = br.id
        '''
        
        if signal_name:
            query += ' WHERE sp.signal_name = ?'
            df = pd.read_sql_query(query + ' ORDER BY sp.timestamp DESC LIMIT ?', 
                                 conn, params=(signal_name, limit))
        else:
            df = pd.read_sql_query(query + ' ORDER BY sp.timestamp DESC LIMIT ?', 
                                 conn, params=(limit,))
        
        conn.close()
        return df
    
    def get_feature_importance_ranking(self, category: str = None) -> pd.DataFrame:
        """Get ranked feature importance"""
        conn = sqlite3.connect(self.db_path)
        
        query = 'SELECT * FROM feature_importance'
        if category:
            query += ' WHERE category = ?'
            df = pd.read_sql_query(query + ' ORDER BY importance_score DESC', 
                                 conn, params=(category,))
        else:
            df = pd.read_sql_query(query + ' ORDER BY importance_score DESC', conn)
        
        conn.close()
        return df
    
    def get_market_regime_history(self, limit: int = 30) -> pd.DataFrame:
        """Get market regime history"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(
            'SELECT * FROM market_regime ORDER BY timestamp DESC LIMIT ?',
            conn, params=(limit,)
        )
        conn.close()
        return df
    
    def update_position_pnl(self, lot_id: str, current_price: float):
        """Update position with current unrealized PnL"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE positions 
            SET unrealized_pnl = (? - avg_buy_price) * available_size,
                updated_at = CURRENT_TIMESTAMP
            WHERE lot_id = ?
        ''', (current_price, lot_id))
        
        conn.commit()
        conn.close()
    
    def execute_limit_order(self, limit_id: str, execution_price: float):
        """Mark a limit order as triggered"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE trading_limits
            SET active = 0,
                triggered_at = CURRENT_TIMESTAMP,
                trigger_price = ?
            WHERE id = ?
        ''', (execution_price, limit_id))
        
        conn.commit()
        conn.close()
    
    def get_portfolio_analytics(self) -> Dict:
        """Get comprehensive portfolio analytics"""
        conn = sqlite3.connect(self.db_path)
        
        # Get enhanced metrics
        trades_df = pd.read_sql_query("SELECT * FROM trades", conn)
        positions_df = pd.read_sql_query("SELECT * FROM positions WHERE total_size > 0", conn)
        signals_df = pd.read_sql_query("SELECT * FROM model_signals ORDER BY timestamp DESC LIMIT 100", conn)
        
        analytics = self.get_portfolio_metrics()  # Original metrics
        
        # Add enhanced analytics
        if not trades_df.empty:
            trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
            
            # Calculate additional metrics
            analytics['total_realized_pnl'] = positions_df['realized_pnl'].sum() if not positions_df.empty else 0
            analytics['total_unrealized_pnl'] = positions_df['unrealized_pnl'].sum() if not positions_df.empty else 0
            
            # Trading frequency
            if len(trades_df) > 1:
                time_diff = (trades_df['timestamp'].max() - trades_df['timestamp'].min()).days
                analytics['trades_per_day'] = len(trades_df) / max(time_diff, 1)
            
            # Win rate from trades with PnL
            profitable_trades = trades_df[trades_df['pnl'] > 0]
            analytics['win_rate'] = len(profitable_trades) / len(trades_df) if len(trades_df) > 0 else 0
            
        # Signal accuracy
        if not signals_df.empty:
            recent_signals = signals_df.head(20)
            analytics['avg_signal_confidence'] = recent_signals['confidence'].mean()
            analytics['signal_distribution'] = recent_signals['signal'].value_counts().to_dict()
        
        conn.close()
        return analytics
    
    # Maintain all original methods
    def add_trade(self, symbol: str, trade_type: str, price: float, size: float, 
                  lot_id: str = None, pnl: float = 0, notes: str = None) -> str:
        """Enhanced add_trade with PnL and notes"""
        if lot_id is None:
            lot_id = str(uuid.uuid4())
        
        trade_id = str(uuid.uuid4())
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO trades (id, symbol, trade_type, price, size, lot_id, pnl, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (trade_id, symbol, trade_type, price, size, lot_id, pnl, notes))
        
        # Update positions
        self._update_position_enhanced(cursor, symbol, trade_type, price, size, lot_id, pnl)
        
        conn.commit()
        conn.close()
        return trade_id
    
    def _update_position_enhanced(self, cursor, symbol: str, trade_type: str, price: float, 
                                 size: float, lot_id: str, pnl: float = 0):
        """Enhanced position update with PnL tracking"""
        cursor.execute('SELECT * FROM positions WHERE lot_id = ?', (lot_id,))
        position = cursor.fetchone()
        
        if trade_type == 'buy':
            if position:
                current_size = position[2]
                current_avg_price = position[4]
                current_realized_pnl = position[7]
                
                new_total_size = current_size + size
                new_avg_price = ((current_size * current_avg_price) + (size * price)) / new_total_size
                
                cursor.execute('''
                    UPDATE positions 
                    SET total_size = ?, available_size = ?, avg_buy_price = ?, 
                        updated_at = CURRENT_TIMESTAMP, realized_pnl = ?
                    WHERE lot_id = ?
                ''', (new_total_size, new_total_size, new_avg_price, current_realized_pnl, lot_id))
            else:
                cursor.execute('''
                    INSERT INTO positions (lot_id, symbol, total_size, available_size, 
                                         avg_buy_price, realized_pnl, unrealized_pnl)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (lot_id, symbol, size, size, price, 0, 0))
        
        elif trade_type == 'sell' and position:
            current_total = position[2]
            current_available = position[3]
            current_avg_price = position[4]
            current_realized_pnl = position[7]
            
            if current_available >= size:
                new_available = current_available - size
                new_total = current_total - size
                
                # Calculate realized PnL for this sell
                sell_pnl = (price - current_avg_price) * size
                new_realized_pnl = current_realized_pnl + sell_pnl
                
                cursor.execute('''
                    UPDATE positions 
                    SET total_size = ?, available_size = ?, updated_at = CURRENT_TIMESTAMP,
                        realized_pnl = ?
                    WHERE lot_id = ?
                ''', (new_total, new_available, new_realized_pnl, lot_id))
    
    # Keep all original methods
    def add_model_signal(self, symbol: str, signal: str, confidence: float, price_prediction: float = None):
        """Original method maintained for compatibility"""
        self.add_enhanced_model_signal(symbol, signal, confidence, price_prediction)
    
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
    
    # Additional methods for test compatibility
    def get_connection(self):
        """Get database connection for test compatibility"""
        return sqlite3.connect(self.db_path)
    
    def save_signal(self, signal: str, confidence: float, predicted_price: float, 
                   current_price: float, indicators: Dict = None):
        """Save signal (test compatibility wrapper)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # First ensure signals table exists
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signals (
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                signal TEXT NOT NULL,
                confidence REAL NOT NULL,
                predicted_price REAL,
                current_price REAL,
                indicators TEXT
            )
        ''')
        
        # Add explicit timestamp to ensure proper ordering in tests
        timestamp = datetime.now().isoformat()
        cursor.execute('''
            INSERT INTO signals (timestamp, signal, confidence, predicted_price, current_price, indicators)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (timestamp, signal, confidence, predicted_price, current_price, 
              json.dumps(indicators) if indicators else None))
        
        conn.commit()
        conn.close()
        
        # Small delay to ensure unique timestamps in tests
        time.sleep(0.01)
    
    def save_price(self, price: float, volume: float, high: float, low: float):
        """Save price data (test compatibility)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Ensure prices table exists
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS prices (
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                price REAL NOT NULL,
                volume REAL,
                high REAL,
                low REAL
            )
        ''')
        
        timestamp = datetime.now().isoformat()
        cursor.execute('''
            INSERT INTO prices (timestamp, price, volume, high, low)
            VALUES (?, ?, ?, ?, ?)
        ''', (timestamp, price, volume, high, low))
        
        conn.commit()
        conn.close()
        
        # Small delay to ensure unique timestamps in tests
        time.sleep(0.01)
    
    def get_recent_signals(self, limit: int = 10) -> pd.DataFrame:
        """Get recent signals (test compatibility)"""
        conn = sqlite3.connect(self.db_path)
        
        # Ensure table exists
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signals (
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                signal TEXT NOT NULL,
                confidence REAL NOT NULL,
                predicted_price REAL,
                current_price REAL,
                indicators TEXT
            )
        ''')
        conn.commit()
        
        df = pd.read_sql_query(
            "SELECT * FROM signals ORDER BY timestamp DESC LIMIT ?",
            conn, params=(limit,)
        )
        conn.close()
        return df
    
    def get_recent_prices(self, limit: int = 100) -> pd.DataFrame:
        """Get recent prices (test compatibility)"""
        conn = sqlite3.connect(self.db_path)
        
        # Ensure table exists
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS prices (
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                price REAL NOT NULL,
                volume REAL,
                high REAL,
                low REAL
            )
        ''')
        conn.commit()
        
        df = pd.read_sql_query(
            "SELECT * FROM prices ORDER BY timestamp DESC LIMIT ?",
            conn, params=(limit,)
        )
        conn.close()
        return df
    
    def save_trade(self, trade_type: str, price: float, amount: float, total_value: float):
        """Save trade to portfolio (test compatibility)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Ensure portfolio table exists
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS portfolio (
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                type TEXT NOT NULL,
                price REAL NOT NULL,
                amount REAL NOT NULL,
                total_value REAL NOT NULL
            )
        ''')
        
        timestamp = datetime.now().isoformat()
        cursor.execute('''
            INSERT INTO portfolio (timestamp, type, price, amount, total_value)
            VALUES (?, ?, ?, ?, ?)
        ''', (timestamp, trade_type, price, amount, total_value))
        
        conn.commit()
        conn.close()
        
        # Small delay to ensure unique timestamps in tests
        time.sleep(0.01)
    
    def get_portfolio_history(self) -> pd.DataFrame:
        """Get portfolio history (test compatibility)"""
        conn = sqlite3.connect(self.db_path)
        
        # Ensure table exists
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS portfolio (
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                type TEXT NOT NULL,
                price REAL NOT NULL,
                amount REAL NOT NULL,
                total_value REAL NOT NULL
            )
        ''')
        conn.commit()
        
        df = pd.read_sql_query(
            "SELECT * FROM portfolio ORDER BY timestamp DESC",
            conn
        )
        conn.close()
        return df
    
    def cleanup_old_data(self, days: int = 30):
        """Clean up old data (test compatibility)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        # Clean up old signals
        cursor.execute("DELETE FROM signals WHERE timestamp < ?", (cutoff_date,))
        
        # Clean up old prices
        cursor.execute("DELETE FROM prices WHERE timestamp < ?", (cutoff_date,))
        
        # Clean up old portfolio entries
        cursor.execute("DELETE FROM portfolio WHERE timestamp < ?", (cutoff_date,))
        
        conn.commit()
        conn.close()