"""
Unit tests for PersistentPaperTrading
"""
import pytest
from datetime import datetime
import uuid
from models.paper_trading import PersistentPaperTrading, InsufficientFundsError, InsufficientBTCError


class TestPersistentPaperTrading:
    """Test cases for PersistentPaperTrading class"""
    
    @pytest.mark.unit
    def test_initialization(self, mock_db_path):
        """Test paper trading initialization"""
        pt = PersistentPaperTrading(db_path=mock_db_path)
        
        assert pt.initial_balance == 10000
        assert pt.enabled is True
        assert pt.current_balance == 10000
        assert pt.btc_balance == 0
        assert len(pt.trades) == 0
    
    @pytest.mark.unit
    def test_enable_disable(self, paper_trading):
        """Test enabling and disabling paper trading"""
        # Initially enabled
        assert paper_trading.enabled is True
        
        # Disable
        paper_trading.disable()
        assert paper_trading.enabled is False
        
        # Enable again
        paper_trading.enable()
        assert paper_trading.enabled is True
    
    @pytest.mark.unit
    def test_execute_buy_trade(self, paper_trading):
        """Test executing a buy trade"""
        initial_balance = paper_trading.current_balance
        btc_price = 50000
        btc_amount = 0.1
        
        # Execute buy
        trade = paper_trading.execute_trade('buy', btc_price, btc_amount)
        
        # Verify trade details
        assert trade['type'] == 'buy'
        assert trade['price'] == btc_price
        assert trade['size'] == btc_amount
        assert trade['value'] == btc_price * btc_amount
        assert trade['balance_after'] < initial_balance
        
        # Verify balances updated
        assert paper_trading.current_balance == initial_balance - (btc_price * btc_amount)
        assert paper_trading.btc_balance == btc_amount
        assert len(paper_trading.trades) == 1
    
    @pytest.mark.unit
    def test_execute_sell_trade(self, paper_trading):
        """Test executing a sell trade"""
        # First buy some BTC
        paper_trading.execute_trade('buy', 50000, 0.2)
        initial_btc = paper_trading.btc_balance
        initial_usd = paper_trading.current_balance
        
        # Execute sell
        sell_price = 51000
        sell_amount = 0.1
        trade = paper_trading.execute_trade('sell', sell_price, sell_amount)
        
        # Verify trade details
        assert trade['type'] == 'sell'
        assert trade['price'] == sell_price
        assert trade['size'] == sell_amount
        assert trade['value'] == sell_price * sell_amount
        
        # Verify balances updated
        assert paper_trading.current_balance == initial_usd + (sell_price * sell_amount)
        assert paper_trading.btc_balance == initial_btc - sell_amount
        assert len(paper_trading.trades) == 2
    
    @pytest.mark.unit
    def test_insufficient_funds_error(self, paper_trading):
        """Test error when insufficient funds for buy"""
        with pytest.raises(InsufficientFundsError):
            # Try to buy more than balance allows
            paper_trading.execute_trade('buy', 50000, 1000)  # Would cost $50M
    
    @pytest.mark.unit
    def test_insufficient_btc_error(self, paper_trading):
        """Test error when insufficient BTC for sell"""
        with pytest.raises(InsufficientBTCError):
            # Try to sell BTC we don't have
            paper_trading.execute_trade('sell', 50000, 1)
    
    @pytest.mark.unit
    def test_reset_portfolio(self, paper_trading):
        """Test resetting portfolio"""
        # Make some trades
        paper_trading.execute_trade('buy', 50000, 0.1)
        paper_trading.execute_trade('sell', 51000, 0.05)
        
        # Reset
        paper_trading.reset()
        
        # Verify reset
        assert paper_trading.current_balance == 10000
        assert paper_trading.btc_balance == 0
        assert len(paper_trading.trades) == 0
        assert paper_trading.initial_balance == 10000
    
    @pytest.mark.unit
    def test_get_metrics(self, paper_trading):
        """Test portfolio metrics calculation"""
        # Make profitable trades
        paper_trading.execute_trade('buy', 50000, 0.2)    # Buy 0.2 BTC for $10k
        paper_trading.execute_trade('sell', 55000, 0.1)   # Sell 0.1 BTC for $5.5k
        paper_trading.execute_trade('sell', 60000, 0.1)   # Sell 0.1 BTC for $6k
        
        metrics = paper_trading.get_metrics(current_btc_price=60000)
        
        # Verify metrics
        assert metrics['total_trades'] == 3
        assert metrics['profitable_trades'] == 2
        assert metrics['win_rate'] == pytest.approx(2/3, 0.01)
        assert metrics['total_value'] > paper_trading.initial_balance
        assert metrics['total_pnl'] > 0
        assert metrics['total_pnl_pct'] > 0
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
    
    @pytest.mark.unit
    def test_persistence(self, mock_db_path):
        """Test that trades persist across instances"""
        # Create instance and make trades
        pt1 = PersistentPaperTrading(db_path=mock_db_path)
        pt1.execute_trade('buy', 50000, 0.1)
        pt1.execute_trade('sell', 51000, 0.05)
        
        # Create new instance
        pt2 = PersistentPaperTrading(db_path=mock_db_path)
        
        # Verify state was loaded
        assert pt2.current_balance == pt1.current_balance
        assert pt2.btc_balance == pt1.btc_balance
        assert len(pt2.trades) == 2
        assert pt2.enabled == pt1.enabled
    
    @pytest.mark.unit
    def test_trade_with_lot_id(self, paper_trading):
        """Test trades with specific lot IDs"""
        lot_id = str(uuid.uuid4())
        
        # Buy with lot ID
        trade = paper_trading.execute_trade('buy', 50000, 0.1, lot_id=lot_id)
        assert trade['lot_id'] == lot_id
        
        # Sell from specific lot
        sell_trade = paper_trading.execute_trade('sell', 51000, 0.05, lot_id=lot_id)
        assert sell_trade['lot_id'] == lot_id
    
    @pytest.mark.unit
    def test_pnl_calculation(self, paper_trading):
        """Test P&L calculation for trades"""
        # Buy at 50k
        buy_trade = paper_trading.execute_trade('buy', 50000, 0.1)
        assert buy_trade['pnl'] == 0  # No P&L on buy
        
        # Sell at 55k (10% profit)
        sell_trade = paper_trading.execute_trade('sell', 55000, 0.1)
        expected_pnl = (55000 - 50000) * 0.1
        assert sell_trade['pnl'] == expected_pnl
        assert sell_trade['pnl_pct'] == pytest.approx(10.0, 0.01)
    
    @pytest.mark.unit
    def test_multiple_buy_average_price(self, paper_trading):
        """Test average price calculation with multiple buys"""
        # Multiple buys at different prices (smaller amounts to fit in $10k budget)
        paper_trading.execute_trade('buy', 50000, 0.06)  # 0.06 BTC at 50k = $3000
        paper_trading.execute_trade('buy', 48000, 0.06)  # 0.06 BTC at 48k = $2880
        paper_trading.execute_trade('buy', 52000, 0.06)  # 0.06 BTC at 52k = $3120
        
        # Sell all - should use weighted average buy price
        sell_trade = paper_trading.execute_trade('sell', 51000, 0.18)
        
        # Average buy price = (50k + 48k + 52k) / 3 = 50k
        # Profit = (51k - 50k) * 0.18 = 180
        assert sell_trade['pnl'] == pytest.approx(180, 1)
        assert sell_trade['pnl_pct'] == pytest.approx(2.0, 0.1)
    
    @pytest.mark.unit
    def test_trade_size_validation(self, paper_trading):
        """Test trade size validation"""
        # Test minimum trade size
        with pytest.raises(ValueError):
            paper_trading.execute_trade('buy', 50000, 0.00001)  # Too small
        
        # Test negative size
        with pytest.raises(ValueError):
            paper_trading.execute_trade('buy', 50000, -0.1)
        
        # Test zero size
        with pytest.raises(ValueError):
            paper_trading.execute_trade('buy', 50000, 0)
    
    @pytest.mark.unit
    def test_concurrent_trades(self, paper_trading):
        """Test handling concurrent trade execution"""
        import threading
        
        results = []
        
        def execute_buy():
            try:
                trade = paper_trading.execute_trade('buy', 50000, 0.05)
                results.append(('success', trade))
            except Exception as e:
                results.append(('error', str(e)))
        
        # Execute multiple trades concurrently
        threads = []
        for _ in range(5):
            t = threading.Thread(target=execute_buy)
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # Verify results
        successful_trades = [r for r in results if r[0] == 'success']
        assert len(successful_trades) > 0
        
        # Verify final state is consistent
        total_btc_bought = sum(r[1]['size'] for r in successful_trades)
        assert paper_trading.btc_balance == pytest.approx(total_btc_bought, 0.00001)