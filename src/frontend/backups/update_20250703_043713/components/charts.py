
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any

def create_candlestick_chart(df: pd.DataFrame, indicators: List[str] = None, 
                           signals: pd.DataFrame = None) -> go.Figure:
    """Create an enhanced candlestick chart with optional indicators and signals"""
    
    # Determine subplot configuration
    has_volume = "volume" in df.columns
    has_rsi = any(ind.startswith("rsi") for ind in (indicators or []))
    has_macd = any(ind.startswith("macd") for ind in (indicators or []))
    
    rows = 1 + (1 if has_volume else 0) + (1 if has_rsi else 0) + (1 if has_macd else 0)
    heights = [0.6]
    if has_volume:
        heights.append(0.1)
    if has_rsi:
        heights.append(0.15)
    if has_macd:
        heights.append(0.15)
    
    # Create subplots
    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=heights,
        subplot_titles=None
    )
    
    # Add candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="BTC/USD",
            increasing_line_color="#00ff88",
            decreasing_line_color="#ff3366"
        ),
        row=1, col=1
    )
    
    # Add technical indicators
    if indicators:
        indicator_colors = {
            "sma": "#3498db",
            "ema": "#e74c3c",
            "bb": "#f39c12",
            "vwap": "#9b59b6"
        }
        
        for indicator in indicators:
            if indicator in df.columns:
                color = next((c for prefix, c in indicator_colors.items() 
                            if indicator.startswith(prefix)), "#95a5a6")
                
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df[indicator],
                        mode="lines",
                        name=indicator.upper(),
                        line=dict(color=color, width=1.5)
                    ),
                    row=1, col=1
                )
        
        # Bollinger Bands
        if all(col in df.columns for col in ["bb_upper", "bb_middle", "bb_lower"]):
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["bb_upper"],
                    mode="lines",
                    name="BB Upper",
                    line=dict(color="#f39c12", width=1, dash="dash")
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["bb_lower"],
                    mode="lines",
                    name="BB Lower",
                    line=dict(color="#f39c12", width=1, dash="dash"),
                    fill="tonexty",
                    fillcolor="rgba(243, 156, 18, 0.1)"
                ),
                row=1, col=1
            )
    
    # Add buy/sell signals
    if signals is not None and not signals.empty:
        buy_signals = signals[signals["signal"] == "buy"]
        sell_signals = signals[signals["signal"] == "sell"]
        
        if not buy_signals.empty:
            fig.add_trace(
                go.Scatter(
                    x=buy_signals.index,
                    y=buy_signals["price"],
                    mode="markers",
                    name="Buy Signal",
                    marker=dict(
                        symbol="triangle-up",
                        size=12,
                        color="#00ff88",
                        line=dict(color="white", width=1)
                    )
                ),
                row=1, col=1
            )
        
        if not sell_signals.empty:
            fig.add_trace(
                go.Scatter(
                    x=sell_signals.index,
                    y=sell_signals["price"],
                    mode="markers",
                    name="Sell Signal",
                    marker=dict(
                        symbol="triangle-down",
                        size=12,
                        color="#ff3366",
                        line=dict(color="white", width=1)
                    )
                ),
                row=1, col=1
            )
    
    # Add volume
    current_row = 2
    if has_volume:
        colors = ["#ff3366" if close < open else "#00ff88" 
                 for close, open in zip(df["close"], df["open"])]
        
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df["volume"],
                marker_color=colors,
                name="Volume",
                opacity=0.7
            ),
            row=current_row, col=1
        )
        fig.update_yaxes(title_text="Volume", row=current_row, col=1)
        current_row += 1
    
    # Add RSI
    if has_rsi:
        rsi_col = next((col for col in df.columns if col.startswith("rsi")), None)
        if rsi_col:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[rsi_col],
                    mode="lines",
                    name="RSI",
                    line=dict(color="#8e44ad", width=2)
                ),
                row=current_row, col=1
            )
            
            # Add overbought/oversold lines
            fig.add_hline(y=70, line_dash="dash", line_color="red", 
                         row=current_row, col=1, opacity=0.5)
            fig.add_hline(y=30, line_dash="dash", line_color="green", 
                         row=current_row, col=1, opacity=0.5)
            
            fig.update_yaxes(title_text="RSI", row=current_row, col=1)
            current_row += 1
    
    # Add MACD
    if has_macd:
        if all(col in df.columns for col in ["macd", "macd_signal", "macd_diff"]):
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["macd"],
                    mode="lines",
                    name="MACD",
                    line=dict(color="#3498db", width=2)
                ),
                row=current_row, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["macd_signal"],
                    mode="lines",
                    name="Signal",
                    line=dict(color="#e74c3c", width=2)
                ),
                row=current_row, col=1
            )
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df["macd_diff"],
                    name="MACD Histogram",
                    marker_color=["#ff3366" if val < 0 else "#00ff88" 
                                 for val in df["macd_diff"]]
                ),
                row=current_row, col=1
            )
            fig.update_yaxes(title_text="MACD", row=current_row, col=1)
    
    # Update layout
    fig.update_layout(
        title="BTC/USD Price Chart",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=800,
        hovermode="x unified",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(0,0,0,0.5)"
        )
    )
    
    # Update axes
    fig.update_xaxes(title_text="Date", row=rows, col=1)
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    
    return fig

def create_portfolio_chart(trades_df: pd.DataFrame) -> go.Figure:
    """Create comprehensive portfolio performance chart"""
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=("Portfolio Value", "P&L", "Drawdown %")
    )
    
    # Portfolio value
    fig.add_trace(
        go.Scatter(
            x=trades_df["timestamp"],
            y=trades_df["portfolio_value"],
            mode="lines",
            name="Portfolio Value",
            line=dict(color="#3498db", width=2),
            fill="tozeroy",
            fillcolor="rgba(52, 152, 219, 0.1)"
        ),
        row=1, col=1
    )
    
    # Add initial value reference line
    if len(trades_df) > 0:
        initial_value = trades_df["portfolio_value"].iloc[0]
        fig.add_hline(
            y=initial_value,
            line_dash="dash",
            line_color="gray",
            annotation_text=f"Initial: ${initial_value:,.2f}",
            row=1, col=1
        )
    
    # P&L bars
    if "pnl" in trades_df.columns:
        colors = ["#ff3366" if pnl < 0 else "#00ff88" for pnl in trades_df["pnl"]]
        fig.add_trace(
            go.Bar(
                x=trades_df["timestamp"],
                y=trades_df["pnl"],
                marker_color=colors,
                name="P&L",
                opacity=0.7
            ),
            row=2, col=1
        )
    
    # Calculate and plot drawdown
    if "portfolio_value" in trades_df.columns:
        rolling_max = trades_df["portfolio_value"].expanding().max()
        drawdown = ((trades_df["portfolio_value"] - rolling_max) / rolling_max * 100)
        
        fig.add_trace(
            go.Scatter(
                x=trades_df["timestamp"],
                y=drawdown,
                mode="lines",
                name="Drawdown %",
                line=dict(color="#e74c3c", width=1),
                fill="tozeroy",
                fillcolor="rgba(231, 76, 60, 0.2)"
            ),
            row=3, col=1
        )
        
        # Add max drawdown line
        max_dd = drawdown.min()
        fig.add_hline(
            y=max_dd,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Max DD: {max_dd:.2f}%",
            row=3, col=1
        )
    
    # Update layout
    fig.update_layout(
        height=800,
        template="plotly_dark",
        showlegend=True,
        hovermode="x unified"
    )
    
    # Update axes
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_yaxes(title_text="Value ($)", row=1, col=1)
    fig.update_yaxes(title_text="P&L ($)", row=2, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=3, col=1)
    
    return fig

def create_signal_chart(signal_history: pd.DataFrame) -> go.Figure:
    """Create comprehensive signal analysis chart"""
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.4, 0.3, 0.3],
        subplot_titles=("Signal History", "Confidence Levels", "Signal Distribution")
    )
    
    # Signal scatter plot with predicted prices
    colors = {"buy": "#00ff88", "sell": "#ff3366", "hold": "#ffaa00"}
    
    for signal_type in ["buy", "sell", "hold"]:
        mask = signal_history["signal"] == signal_type
        if mask.any():
            fig.add_trace(
                go.Scatter(
                    x=signal_history[mask]["timestamp"],
                    y=signal_history[mask].get("predicted_price", signal_history[mask].get("current_price", 0)),
                    mode="markers",
                    name=signal_type.upper(),
                    marker=dict(
                        color=colors[signal_type],
                        size=8 + signal_history[mask]["confidence"] * 20,
                        symbol="circle",
                        line=dict(color="white", width=1)
                    ),
                    text=[f"Confidence: {conf:.1%}" for conf in signal_history[mask]["confidence"]],
                    hovertemplate="%{text}<br>Price: $%{y:,.2f}<extra></extra>"
                ),
                row=1, col=1
            )
    
    # Confidence line chart
    fig.add_trace(
        go.Scatter(
            x=signal_history["timestamp"],
            y=signal_history["confidence"] * 100,
            mode="lines",
            name="Confidence %",
            line=dict(color="#3498db", width=2),
            fill="tozeroy",
            fillcolor="rgba(52, 152, 219, 0.1)"
        ),
        row=2, col=1
    )
    
    # Add confidence threshold line
    fig.add_hline(
        y=60,
        line_dash="dash",
        line_color="yellow",
        annotation_text="Min Threshold",
        row=2, col=1
    )
    
    # Signal distribution over time (stacked area)
    signal_counts = signal_history.groupby([pd.Grouper(key="timestamp", freq="1H"), "signal"]).size().unstack(fill_value=0)
    
    if not signal_counts.empty:
        for signal_type in ["buy", "hold", "sell"]:
            if signal_type in signal_counts.columns:
                fig.add_trace(
                    go.Scatter(
                        x=signal_counts.index,
                        y=signal_counts[signal_type],
                        mode="lines",
                        name=f"{signal_type.upper()} Count",
                        line=dict(color=colors[signal_type], width=0),
                        stackgroup="signals",
                        fillcolor=colors[signal_type]
                    ),
                    row=3, col=1
                )
    
    # Update layout
    fig.update_layout(
        height=900,
        template="plotly_dark",
        showlegend=True,
        hovermode="x unified"
    )
    
    # Update axes
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Confidence (%)", row=2, col=1)
    fig.update_yaxes(title_text="Signal Count", row=3, col=1)
    
    return fig

def create_correlation_heatmap(correlation_matrix: pd.DataFrame) -> go.Figure:
    """Create an interactive correlation heatmap"""
    
    # Create custom colorscale
    colorscale = [
        [0.0, "#ff3366"],    # Strong negative
        [0.25, "#ff8866"],   # Weak negative
        [0.5, "#ffffff"],    # Neutral
        [0.75, "#66ff88"],   # Weak positive
        [1.0, "#00ff88"]     # Strong positive
    ]
    
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.index,
        colorscale=colorscale,
        zmid=0,
        text=correlation_matrix.values,
        texttemplate="%{text:.2f}",
        textfont={"size": 10},
        colorbar=dict(
            title="Correlation",
            tickmode="linear",
            tick0=-1,
            dtick=0.5
        ),
        hoverongaps=False,
        hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>"
    ))
    
    # Add diagonal line
    fig.add_shape(
        type="line",
        x0=0, y0=0,
        x1=len(correlation_matrix.columns)-1,
        y1=len(correlation_matrix.index)-1,
        line=dict(color="white", width=2, dash="dash")
    )
    
    fig.update_layout(
        title="Feature Correlation Matrix",
        height=800,
        width=800,
        template="plotly_dark",
        xaxis=dict(tickangle=-45),
        yaxis=dict(autorange="reversed")
    )
    
    return fig

def create_performance_chart(equity_curve: pd.DataFrame, 
                           benchmark: pd.DataFrame = None,
                           show_metrics: bool = True) -> go.Figure:
    """Create detailed performance comparison chart"""
    
    # Create figure with secondary y-axis
    fig = make_subplots(
        rows=2 if show_metrics else 1,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3] if show_metrics else [1],
        subplot_titles=("Cumulative Returns", "Rolling Metrics") if show_metrics else None
    )
    
    # Calculate returns
    equity_returns = (equity_curve["value"] / equity_curve["value"].iloc[0] - 1) * 100
    
    # Portfolio equity curve
    fig.add_trace(go.Scatter(
        x=equity_curve.index,
        y=equity_returns,
        mode="lines",
        name="Portfolio",
        line=dict(color="#3498db", width=2),
        fill="tozeroy",
        fillcolor="rgba(52, 152, 219, 0.1)",
        hovertemplate="Portfolio<br>Return: %{y:.2f}%<extra></extra>"
    ), row=1, col=1)
    
    # Benchmark if provided
    if benchmark is not None and not benchmark.empty:
        benchmark_returns = (benchmark["value"] / benchmark["value"].iloc[0] - 1) * 100
        
        fig.add_trace(go.Scatter(
            x=benchmark.index,
            y=benchmark_returns,
            mode="lines",
            name="BTC Buy & Hold",
            line=dict(color="#95a5a6", width=2, dash="dash"),
            hovertemplate="Benchmark<br>Return: %{y:.2f}%<extra></extra>"
        ), row=1, col=1)
        
        # Add relative performance
        relative_perf = equity_returns - benchmark_returns
        fig.add_trace(go.Scatter(
            x=equity_curve.index,
            y=relative_perf,
            mode="lines",
            name="Relative Performance",
            line=dict(color="#e74c3c" if relative_perf.iloc[-1] < 0 else "#27ae60", width=1),
            visible="legendonly",
            hovertemplate="Relative<br>Performance: %{y:.2f}%<extra></extra>"
        ), row=1, col=1)
    
    # Add zero line
    fig.add_hline(y=0, line_dash="solid", line_color="white", opacity=0.3, row=1, col=1)
    
    # Rolling metrics
    if show_metrics and len(equity_curve) > 30:
        # Calculate rolling Sharpe ratio (30-day)
        returns = equity_curve["value"].pct_change()
        rolling_sharpe = returns.rolling(30).apply(
            lambda x: np.sqrt(252) * x.mean() / x.std() if x.std() != 0 else 0
        )
        
        # Calculate rolling volatility
        rolling_vol = returns.rolling(30).std() * np.sqrt(252) * 100
        
        # Plot rolling Sharpe
        fig.add_trace(go.Scatter(
            x=equity_curve.index,
            y=rolling_sharpe,
            mode="lines",
            name="30d Sharpe Ratio",
            line=dict(color="#9b59b6", width=2),
            yaxis="y3",
            hovertemplate="Sharpe: %{y:.2f}<extra></extra>"
        ), row=2, col=1)
        
        # Plot rolling volatility
        fig.add_trace(go.Scatter(
            x=equity_curve.index,
            y=rolling_vol,
            mode="lines",
            name="30d Volatility %",
            line=dict(color="#e67e22", width=2),
            yaxis="y4",
            hovertemplate="Volatility: %{y:.1f}%<extra></extra>"
        ), row=2, col=1)
        
        # Update y-axes for metrics
        fig.update_yaxes(title_text="Sharpe Ratio", secondary_y=False, row=2, col=1)
        fig.update_layout(
            yaxis3=dict(
                title="Sharpe Ratio",
                overlaying="y",
                side="left",
                showgrid=False
            ),
            yaxis4=dict(
                title="Volatility %",
                overlaying="y",
                side="right",
                showgrid=False
            )
        )
    
    # Update layout
    fig.update_layout(
        title="Portfolio Performance Analysis",
        height=700 if show_metrics else 500,
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(0,0,0,0.5)"
        )
    )
    
    # Update axes
    fig.update_xaxes(title_text="Date", row=2 if show_metrics else 1, col=1)
    fig.update_yaxes(title_text="Cumulative Return (%)", row=1, col=1)
    
    return fig

def create_monte_carlo_chart(simulations: List[np.ndarray], 
                           percentiles: Dict[str, np.ndarray],
                           final_values: np.ndarray) -> go.Figure:
    """Create Monte Carlo simulation visualization"""
    
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.7, 0.3],
        subplot_titles=("Simulation Paths", "Final Value Distribution")
    )
    
    # Plot sample paths (max 100 for performance)
    sample_size = min(100, len(simulations))
    for i in range(sample_size):
        fig.add_trace(
            go.Scatter(
                x=list(range(len(simulations[i]))),
                y=simulations[i],
                mode="lines",
                line=dict(width=0.5, color="lightgray"),
                showlegend=False,
                opacity=0.3,
                hoverinfo="skip"
            ),
            row=1, col=1
        )
    
    # Plot percentile lines
    percentile_colors = {
        "p95": {"color": "#27ae60", "name": "95th Percentile"},
        "p50": {"color": "#3498db", "name": "Median"},
        "p5": {"color": "#e74c3c", "name": "5th Percentile"}
    }
    
    for key, values in percentiles.items():
        if key in percentile_colors:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(values))),
                    y=values,
                    mode="lines",
                    name=percentile_colors[key]["name"],
                    line=dict(color=percentile_colors[key]["color"], width=2),
                    hovertemplate=f"{percentile_colors[key][name]}<br>Value: $%{{y:,.0f}}<extra></extra>"
                ),
                row=1, col=1
            )
    
    # Final value distribution histogram
    fig.add_trace(
        go.Histogram(
            x=final_values,
            nbinsx=50,
            marker_color="#3498db",
            opacity=0.7,
            name="Final Values",
            showlegend=False,
            hovertemplate="Range: $%{x}<br>Count: %{y}<extra></extra>"
        ),
        row=1, col=2
    )
    
    # Add VaR line
    var_95 = np.percentile(final_values, 5)
    fig.add_vline(
        x=var_95,
        line_dash="dash",
        line_color="red",
        annotation_text=f"VaR (95%): ${var_95:,.0f}",
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        title="Monte Carlo Risk Simulation",
        height=600,
        template="plotly_dark",
        showlegend=True
    )
    
    # Update axes
    fig.update_xaxes(title_text="Days", row=1, col=1)
    fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
    fig.update_xaxes(title_text="Final Value ($)", row=1, col=2)
    fig.update_yaxes(title_text="Frequency", row=1, col=2)
    
    return fig

