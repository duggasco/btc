import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def create_candlestick_chart(df: pd.DataFrame, indicators: list = None) -> go.Figure:
    """Create an enhanced candlestick chart with optional indicators"""
    
    # Create subplots
    rows = 2 if 'volume' in df.columns else 1
    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3] if rows == 2 else [1]
    )
    
    # Add candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='OHLC'
        ),
        row=1, col=1
    )
    
    # Add indicators
    if indicators:
        for indicator in indicators:
            if indicator in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df[indicator],
                        mode='lines',
                        name=indicator.upper()
                    ),
                    row=1, col=1
                )
    
    # Add volume
    if 'volume' in df.columns and rows == 2:
        colors = ['red' if close < open else 'green' 
                 for close, open in zip(df['close'], df['open'])]
        
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['volume'],
                marker_color=colors,
                name='Volume'
            ),
            row=2, col=1
        )
    
    # Update layout
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        template='plotly_dark',
        height=600
    )
    
    return fig

def create_portfolio_chart(trades_df: pd.DataFrame) -> go.Figure:
    """Create portfolio performance chart"""
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=("Portfolio Value", "Drawdown")
    )
    
    # Portfolio value
    fig.add_trace(
        go.Scatter(
            x=trades_df['timestamp'],
            y=trades_df['portfolio_value'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    
    # Calculate drawdown
    rolling_max = trades_df['portfolio_value'].expanding().max()
    drawdown = (trades_df['portfolio_value'] - rolling_max) / rolling_max
    
    # Drawdown
    fig.add_trace(
        go.Scatter(
            x=trades_df['timestamp'],
            y=drawdown * 100,
            mode='lines',
            name='Drawdown %',
            line=dict(color='red', width=1),
            fill='tozeroy'
        ),
        row=2, col=1
    )
    
    fig.update_layout(height=600, showlegend=True)
    
    return fig

def create_signal_chart(signal_history: pd.DataFrame) -> go.Figure:
    """Create signal history visualization"""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=("Signals Over Time", "Confidence Levels")
    )
    
    # Signal scatter plot
    colors = {'buy': 'green', 'sell': 'red', 'hold': 'gray'}
    
    for signal_type in ['buy', 'sell', 'hold']:
        mask = signal_history['signal'] == signal_type
        if mask.any():
            fig.add_trace(
                go.Scatter(
                    x=signal_history[mask]['timestamp'],
                    y=signal_history[mask]['predicted_price'],
                    mode='markers',
                    name=signal_type.upper(),
                    marker=dict(
                        color=colors[signal_type],
                        size=10,
                        symbol='circle'
                    )
                ),
                row=1, col=1
            )
    
    # Confidence line
    fig.add_trace(
        go.Scatter(
            x=signal_history['timestamp'],
            y=signal_history['confidence'],
            mode='lines',
            name='Confidence',
            line=dict(color='blue', width=2)
        ),
        row=2, col=1
    )
    
    fig.update_layout(height=600, showlegend=True)
    return fig

def create_correlation_heatmap(correlation_matrix: pd.DataFrame) -> go.Figure:
    """Create correlation heatmap"""
    
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.index,
        colorscale='RdBu',
        zmid=0,
        text=correlation_matrix.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title="Feature Correlation Matrix",
        height=600,
        width=800
    )
    
    return fig

def create_performance_chart(equity_curve: pd.DataFrame, benchmark: pd.DataFrame = None) -> go.Figure:
    """Create performance comparison chart"""
    fig = go.Figure()
    
    # Portfolio equity curve
    fig.add_trace(go.Scatter(
        x=equity_curve.index,
        y=equity_curve['value'],
        mode='lines',
        name='Portfolio',
        line=dict(color='blue', width=2)
    ))
    
    # Benchmark if provided
    if benchmark is not None:
        fig.add_trace(go.Scatter(
            x=benchmark.index,
            y=benchmark['value'],
            mode='lines',
            name='Benchmark',
            line=dict(color='gray', width=1, dash='dash')
        ))
    
    fig.update_layout(
        title="Portfolio Performance",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        height=500,
        hovermode='x unified'
    )
    
    return fig
