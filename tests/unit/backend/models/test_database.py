"""
Unit tests for DatabaseManager
"""
import pytest
from datetime import datetime, timedelta
import pandas as pd
from models.database import DatabaseManager


class TestDatabaseManager:
    """Test cases for DatabaseManager class"""
    
    @pytest.mark.unit
    def test_initialization(self, mock_db_path):
        """Test database initialization"""
        db = DatabaseManager(mock_db_path)
        db.initialize_database()
        
        # Check if tables are created
        with db.get_connection() as conn:
            cursor = conn.cursor()
            
            # Check signals table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='signals'")
            assert cursor.fetchone() is not None
            
            # Check prices table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='prices'")
            assert cursor.fetchone() is not None
            
            # Check portfolio table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='portfolio'")
            assert cursor.fetchone() is not None
    
    @pytest.mark.unit
    def test_save_signal(self, db_manager):
        """Test saving trading signals"""
        # Create test signal
        signal_data = {
            'signal': 'buy',
            'confidence': 0.85,
            'predicted_price': 52000.0,
            'current_price': 50000.0,
            'rsi': 45.5,
            'macd': 0.02,
            'bb_position': 0.3,
            'volume_ratio': 1.2
        }
        
        # Save signal
        db_manager.save_signal(
            signal=signal_data['signal'],
            confidence=signal_data['confidence'],
            predicted_price=signal_data['predicted_price'],
            current_price=signal_data['current_price'],
            indicators={
                'rsi': signal_data['rsi'],
                'macd': signal_data['macd'],
                'bb_position': signal_data['bb_position'],
                'volume_ratio': signal_data['volume_ratio']
            }
        )
        
        # Verify signal was saved
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM signals ORDER BY timestamp DESC LIMIT 1")
            row = cursor.fetchone()
            
            assert row is not None
            assert row[1] == 'buy'  # signal
            assert row[2] == 0.85   # confidence
            assert row[3] == 52000.0  # predicted_price
            assert row[4] == 50000.0  # current_price
    
    @pytest.mark.unit
    def test_save_price(self, db_manager):
        """Test saving price data"""
        # Save price
        db_manager.save_price(
            price=50000.0,
            volume=1000000.0,
            high=51000.0,
            low=49000.0
        )
        
        # Verify price was saved
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM prices ORDER BY timestamp DESC LIMIT 1")
            row = cursor.fetchone()
            
            assert row is not None
            assert row[1] == 50000.0  # price
            assert row[2] == 1000000.0  # volume
            assert row[3] == 51000.0  # high
            assert row[4] == 49000.0  # low
    
    @pytest.mark.unit
    def test_get_recent_signals(self, db_manager):
        """Test retrieving recent signals"""
        # Save multiple signals
        signals = [
            ('buy', 0.8, 52000, 50000),
            ('hold', 0.6, 50500, 50000),
            ('sell', 0.9, 48000, 50000)
        ]
        
        for signal in signals:
            db_manager.save_signal(
                signal=signal[0],
                confidence=signal[1],
                predicted_price=signal[2],
                current_price=signal[3],
                indicators={}
            )
        
        # Get recent signals
        df = db_manager.get_recent_signals(limit=2)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert df.iloc[0]['signal'] == 'sell'  # Most recent
        assert df.iloc[1]['signal'] == 'hold'
    
    @pytest.mark.unit
    def test_get_recent_prices(self, db_manager):
        """Test retrieving recent prices"""
        # Save multiple prices
        prices = [50000, 50500, 51000, 50800]
        
        for price in prices:
            db_manager.save_price(price, 1000000, price + 100, price - 100)
        
        # Get recent prices
        df = db_manager.get_recent_prices(limit=3)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert df.iloc[0]['price'] == 50800  # Most recent
    
    @pytest.mark.unit
    def test_save_trade(self, db_manager):
        """Test saving trades to portfolio"""
        # Save a buy trade
        db_manager.save_trade(
            trade_type='buy',
            price=50000.0,
            amount=0.1,
            total_value=5000.0
        )
        
        # Verify trade was saved
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM portfolio ORDER BY timestamp DESC LIMIT 1")
            row = cursor.fetchone()
            
            assert row is not None
            assert row[1] == 'buy'
            assert row[2] == 50000.0
            assert row[3] == 0.1
            assert row[4] == 5000.0
    
    @pytest.mark.unit
    def test_get_portfolio_history(self, db_manager):
        """Test retrieving portfolio history"""
        # Save multiple trades
        trades = [
            ('buy', 50000, 0.1, 5000),
            ('sell', 51000, 0.05, 2550),
            ('buy', 49000, 0.2, 9800)
        ]
        
        for trade in trades:
            db_manager.save_trade(*trade)
        
        # Get portfolio history
        df = db_manager.get_portfolio_history()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert df.iloc[0]['type'] == 'buy'  # Most recent
        assert df.iloc[0]['price'] == 49000
    
    @pytest.mark.unit
    def test_cleanup_old_data(self, db_manager):
        """Test cleanup of old data"""
        # Save old and new data
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            # Insert old signal (40 days ago)
            old_date = (datetime.now() - timedelta(days=40)).isoformat()
            cursor.execute(
                "INSERT INTO signals (timestamp, signal, confidence, predicted_price, current_price) VALUES (?, ?, ?, ?, ?)",
                (old_date, 'buy', 0.8, 50000, 49000)
            )
            
            # Insert recent signal
            cursor.execute(
                "INSERT INTO signals (signal, confidence, predicted_price, current_price) VALUES (?, ?, ?, ?)",
                ('sell', 0.9, 48000, 50000)
            )
            
            conn.commit()
        
        # Run cleanup
        db_manager.cleanup_old_data(days=30)
        
        # Verify old data was removed
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM signals")
            count = cursor.fetchone()[0]
            
            assert count == 1  # Only recent signal remains
    
    @pytest.mark.unit
    def test_concurrent_access(self, db_manager):
        """Test concurrent database access"""
        import threading
        
        def save_signal(signal_type, confidence):
            db_manager.save_signal(signal_type, confidence, 50000, 49000, {})
        
        # Create multiple threads
        threads = []
        for i in range(10):
            signal = 'buy' if i % 2 == 0 else 'sell'
            confidence = 0.5 + (i * 0.05)
            t = threading.Thread(target=save_signal, args=(signal, confidence))
            threads.append(t)
            t.start()
        
        # Wait for all threads
        for t in threads:
            t.join()
        
        # Verify all signals were saved
        with db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM signals")
            count = cursor.fetchone()[0]
            
            assert count == 10