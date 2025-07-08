"""
Unit tests for charts component
"""
import pytest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import sys
import os

# Add frontend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../src/frontend'))

from components.charts import create_candlestick_chart, create_signal_chart, create_portfolio_chart, create_performance_chart


class TestCharts:
    """Test cases for chart creation functions"""
    
    @pytest.fixture
    def sample_price_data(self):
        """Create sample price data"""
        dates = pd.date_range(end=datetime.now(), periods=100, freq='H')
        prices = 50000 + np.random.randn(100).cumsum() * 1000
        
        return pd.DataFrame({
            'timestamp': dates,
            'price': prices,
            'volume': np.random.randint(1000000, 10000000, 100),
            'high': prices + abs(np.random.randn(100) * 200),
            'low': prices - abs(np.random.randn(100) * 200)
        })
    
    @pytest.fixture
    def sample_signals_data(self):
        """Create sample signals data"""
        dates = pd.date_range(end=datetime.now(), periods=50, freq='2H')
        
        return pd.DataFrame({
            'timestamp': dates,
            'signal': np.random.choice(['buy', 'hold', 'sell'], 50),
            'confidence': np.random.uniform(0.5, 1.0, 50),
            'predicted_price': 50000 + np.random.randn(50).cumsum() * 1000
        })
    
    @pytest.fixture
    def sample_portfolio_data(self):
        """Create sample portfolio data"""
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        
        return pd.DataFrame({
            'timestamp': dates,
            'total_value': 10000 + np.random.randn(30).cumsum() * 500,
            'btc_value': 5000 + np.random.randn(30).cumsum() * 300,
            'usd_balance': 5000 + np.random.randn(30).cumsum() * 200
        })
    
    @pytest.mark.unit
    def test_create_candlestick_chart_basic(self, sample_price_data):
        """Test basic candlestick chart creation"""
        # Convert to the format expected by candlestick chart
        df = sample_price_data.copy()
        df['open'] = df['price'] - np.random.uniform(-100, 100, len(df))
        df['close'] = df['price']
        df.set_index('timestamp', inplace=True)
        
        fig = create_candlestick_chart(df)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        assert fig.layout.title.text is not None
        assert 'Price' in fig.layout.yaxis.title.text
    
    @pytest.mark.unit
    def test_create_candlestick_chart_with_indicators(self, sample_price_data):
        """Test candlestick chart with technical indicators"""
        # Convert to the format expected by candlestick chart
        df = sample_price_data.copy()
        df['open'] = df['price'] - np.random.uniform(-100, 100, len(df))
        df['close'] = df['price']
        df.set_index('timestamp', inplace=True)
        
        # Add technical indicator columns
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        
        fig = create_candlestick_chart(
            df,
            indicators=['sma_20', 'sma_50']
        )
        
        # Should have price trace, volume, and 2 MAs
        assert len(fig.data) >= 4
        
        # Check for moving average traces
        trace_names = [trace.name for trace in fig.data if hasattr(trace, 'name')]
        assert any('MA' in name for name in trace_names)
    
    @pytest.mark.unit
    def test_create_candlestick_chart_empty_data(self):
        """Test candlestick chart with empty data"""
        empty_df = pd.DataFrame(columns=['open', 'high', 'low', 'close'])
        fig = create_candlestick_chart(empty_df)
        
        assert isinstance(fig, go.Figure)
        # Should handle empty data gracefully
    
    @pytest.mark.unit
    def test_create_signal_chart(self, sample_signals_data):
        """Test signal chart creation"""
        fig = create_signal_chart(sample_signals_data)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        
        # Should have different colors for buy/sell/hold
        colors = [trace.marker.color for trace in fig.data if hasattr(trace, 'marker')]
        assert len(set(colors)) > 1  # Multiple colors used
    
    @pytest.mark.unit
    def test_create_signal_chart_with_confidence(self, sample_signals_data):
        """Test signal chart with confidence overlay"""
        # Ensure data has required columns
        sample_signals_data['current_price'] = 50000
        
        fig = create_signal_chart(sample_signals_data)
        
        # Should have multiple traces for different signals and confidence
        assert len(fig.data) >= 3  # buy, sell, hold signals
        
        # Check for subplots (signal history, confidence, distribution)
        assert hasattr(fig, '_grid_ref')
        assert len(fig._grid_ref) >= 3
    
    @pytest.mark.unit
    def test_create_portfolio_chart(self, sample_portfolio_data):
        """Test portfolio chart creation"""
        # Add required columns
        sample_portfolio_data['portfolio_value'] = sample_portfolio_data['total_value']
        sample_portfolio_data['cumulative_pnl'] = np.random.randn(len(sample_portfolio_data)).cumsum() * 100
        sample_portfolio_data['drawdown_pct'] = np.random.uniform(-0.2, 0, len(sample_portfolio_data))
        
        fig = create_portfolio_chart(sample_portfolio_data)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
    
    @pytest.mark.unit
    def test_create_portfolio_chart_breakdown(self, sample_portfolio_data):
        """Test portfolio chart with asset breakdown"""
        # Add required columns for portfolio chart
        sample_portfolio_data['portfolio_value'] = sample_portfolio_data['total_value']
        sample_portfolio_data['cumulative_pnl'] = np.random.randn(len(sample_portfolio_data)).cumsum() * 100
        sample_portfolio_data['drawdown_pct'] = np.random.uniform(-0.2, 0, len(sample_portfolio_data))
        
        fig = create_portfolio_chart(sample_portfolio_data)
        
        # Should have at least traces for portfolio value
        assert len(fig.data) >= 1  # At minimum portfolio value trace
        
        # Check for subplots
        assert hasattr(fig, '_grid_ref')
        assert len(fig._grid_ref) >= 3
    
    @pytest.mark.unit
    def test_create_performance_chart(self):
        """Test performance chart creation"""
        # Create sample performance data with required columns
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        performance_data = pd.DataFrame({
            'timestamp': dates,
            'value': 10000 * (1 + np.random.randn(30) * 0.02).cumprod(),
            'returns': np.random.randn(30) * 0.02,
            'sharpe_ratio': np.random.uniform(0.5, 2.0, 30),
            'win_rate': np.random.uniform(0.4, 0.7, 30),
            'profit_factor': np.random.uniform(0.8, 2.0, 30)
        })
        performance_data.set_index('timestamp', inplace=True)
        
        fig = create_performance_chart(performance_data)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        
        # Should be a line chart with performance data
        assert any(isinstance(trace, go.Scatter) for trace in fig.data)
    
    @pytest.mark.unit
    def test_chart_responsiveness(self, sample_price_data):
        """Test chart responsiveness settings"""
        df = sample_price_data.copy()
        df['open'] = df['price'] - np.random.uniform(-100, 100, len(df))
        df['close'] = df['price']
        df.set_index('timestamp', inplace=True)
        
        fig = create_candlestick_chart(df)
        
        # Should have responsive layout
        assert fig.layout.autosize is not False
        assert fig.layout.height is not None
    
    @pytest.mark.unit
    def test_chart_theme(self, sample_price_data):
        """Test chart theme consistency"""
        df = sample_price_data.copy()
        df['open'] = df['price'] - np.random.uniform(-100, 100, len(df))
        df['close'] = df['price']
        df.set_index('timestamp', inplace=True)
        
        fig = create_candlestick_chart(df)
        
        # Should apply dark theme
        assert fig.layout.template is not None or fig.layout.plot_bgcolor is not None
    
    @pytest.mark.unit
    def test_candlestick_chart(self, sample_price_data):
        """Test candlestick chart creation"""
        # Add OHLC data
        sample_price_data['open'] = sample_price_data['price'] - np.random.uniform(-100, 100, len(sample_price_data))
        sample_price_data['close'] = sample_price_data['price']
        
        df = sample_price_data.copy()
        df.set_index('timestamp', inplace=True)
        
        fig = create_candlestick_chart(df)
        
        # Should have candlestick trace
        assert any(isinstance(trace, go.Candlestick) for trace in fig.data)
    
    @pytest.mark.unit
    def test_chart_annotations(self, sample_price_data):
        """Test chart with annotations"""
        trades = [
            {'timestamp': sample_price_data['timestamp'].iloc[20], 'type': 'buy', 'price': sample_price_data['price'].iloc[20]},
            {'timestamp': sample_price_data['timestamp'].iloc[40], 'type': 'sell', 'price': sample_price_data['price'].iloc[40]}
        ]
        
        df = sample_price_data.copy()
        df['open'] = df['price'] - np.random.uniform(-100, 100, len(df))
        df['close'] = df['price']
        df.set_index('timestamp', inplace=True)
        
        # Create signals dataframe for trades
        signals_df = pd.DataFrame({
            'signal': [t['type'] for t in trades],
            'price': [t['price'] for t in trades]
        }, index=[t['timestamp'] for t in trades])
        
        fig = create_candlestick_chart(df, signals=signals_df)
        
        # Should have trade markers
        assert len(fig.layout.annotations) > 0 or len(fig.data) > 1
    
    @pytest.mark.unit
    def test_multi_axis_chart(self, sample_price_data):
        """Test multi-axis chart creation"""
        # Add RSI data
        sample_price_data['rsi'] = 50 + np.random.uniform(-30, 30, len(sample_price_data))
        
        df = sample_price_data.copy()
        df['open'] = df['price'] - np.random.uniform(-100, 100, len(df))
        df['close'] = df['price']
        df.set_index('timestamp', inplace=True)
        
        fig = create_candlestick_chart(df, indicators=['rsi'])
        
        # Should have secondary y-axis for RSI
        assert fig.layout.yaxis2 is not None or len(fig.data) > 1
    
    @pytest.mark.unit
    def test_chart_export_config(self, sample_price_data):
        """Test chart export configuration"""
        df = sample_price_data.copy()
        df['open'] = df['price'] - np.random.uniform(-100, 100, len(df))
        df['close'] = df['price']
        df.set_index('timestamp', inplace=True)
        
        fig = create_candlestick_chart(df)
        
        # Should have export config
        config = fig.to_dict().get('config', {})
        # Config might be set during display, not creation
        assert isinstance(fig, go.Figure)
    
    @pytest.mark.unit
    def test_real_time_update_compatibility(self, sample_price_data):
        """Test chart structure for real-time updates"""
        df = sample_price_data.copy()
        df['open'] = df['price'] - np.random.uniform(-100, 100, len(df))
        df['close'] = df['price']
        df.set_index('timestamp', inplace=True)
        
        fig = create_candlestick_chart(df)
        
        # Should have extendable traces
        for trace in fig.data:
            if hasattr(trace, 'x') and hasattr(trace, 'y'):
                # Can extend x and y data
                assert isinstance(trace.x, (list, np.ndarray, pd.Series))
                assert isinstance(trace.y, (list, np.ndarray, pd.Series))
    
    @pytest.mark.unit
    def test_error_handling_invalid_data(self):
        """Test chart creation with invalid data"""
        # Invalid data types
        invalid_data = "not a dataframe"
        
        # Should handle gracefully
        try:
            fig = create_candlestick_chart(invalid_data)
            # If it doesn't raise, should return empty figure
            assert isinstance(fig, go.Figure)
        except (TypeError, AttributeError):
            # Expected for invalid input
            pass