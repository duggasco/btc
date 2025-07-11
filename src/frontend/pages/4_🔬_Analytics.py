import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime, timedelta
import time
import json
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components.api_client import APIClient
from components.charts import create_candlestick_chart, create_portfolio_chart
from components.page_styling import setup_page
from utils.helpers import format_currency, format_percentage
from utils.constants import CHART_COLORS

# Set up logging
logger = logging.getLogger(__name__)

# Setup page with consistent styling
api_client = setup_page(
    page_name="Analytics",
    page_title="Analytics & Research",
    page_subtitle="Advanced backtesting, optimization, and performance analysis"
)

# Additional page-specific CSS
st.markdown("""
<style>
/* Analytics specific styling */
.analytics-card {
    background: var(--bg-secondary);
    padding: 1.5rem;
    border-radius: 10px;
    border: 1px solid var(--border-subtle);
    transition: all 0.3s ease;
}

.analytics-card:hover {
    transform: translateY(-2px);
    border-color: var(--border-focus);
}

.result-card {
    background: var(--bg-tertiary);
    padding: 1.2rem;
    border-radius: 8px;
    border-left: 4px solid var(--accent-primary);
    margin: 0.8rem 0;
}

/* Position cards */
.position-card {
    background: var(--bg-secondary);
    padding: 1.2rem;
    border-radius: 8px;
    margin: 0.8rem 0;
    border: 1px solid var(--border-subtle);
    transition: all 0.2s ease;
}

.position-card:hover {
    background: var(--bg-tertiary);
    border-color: var(--accent-primary);
}

/* Profit/Loss styling */
.profit-positive { 
    color: var(--accent-success); 
    font-weight: 600; 
}

.profit-negative { 
    color: var(--accent-danger); 
    font-weight: 600; 
}

/* Form styling */
.order-form {
    background: var(--bg-secondary);
    padding: 1.8rem;
    border-radius: 12px;
    box-shadow: 0 3px 12px rgba(0,0,0,0.08);
    border: 1px solid var(--border-subtle);
}

/* Regime indicators */
.regime-indicator {
    padding: 0.8rem 1.5rem;
    border-radius: 25px;
    text-align: center;
    font-weight: 600;
    color: white;
    display: inline-block;
}

.regime-bullish { background: linear-gradient(135deg, var(--accent-success) 0%, #00d68f 100%); }
.regime-bearish { background: linear-gradient(135deg, var(--accent-danger) 0%, #ee4758 100%); }
.regime-neutral { background: linear-gradient(135deg, var(--text-muted) 0%, var(--text-secondary) 100%); }

/* Quick action buttons */
.quick-action {
    background: var(--bg-tertiary);
    border: 2px solid transparent;
    padding: 0.8rem 1.2rem;
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.2s ease;
    text-align: center;
    margin: 0.5rem 0;
}

.quick-action:hover {
    background: var(--bg-secondary);
    border-color: var(--accent-primary);
    transform: translateY(-2px);
}

/* Parameter preset styling */
.preset-button {
    background: var(--bg-tertiary);
    padding: 0.6rem 1rem;
    border-radius: 6px;
    border: 1px solid var(--border-subtle);
    cursor: pointer;
    transition: all 0.2s ease;
    display: inline-block;
    margin: 0.3rem;
    color: var(--text-primary);
}

.preset-button:hover {
    background: var(--accent-primary);
    color: white;
    border-color: var(--accent-primary);
}

/* Progress indicator */
.progress-bar {
    background: var(--border-subtle);
    border-radius: 10px;
    height: 8px;
    overflow: hidden;
    margin: 1rem 0;
}

.progress-fill {
    background: linear-gradient(90deg, var(--accent-primary) 0%, #e6851a 100%);
    height: 100%;
    transition: width 0.3s ease;
}
</style>
""", unsafe_allow_html=True)

# The api_client is already initialized by setup_page

# Helper functions
def format_pnl(value: float, percentage: float = None) -> str:
    """Format P&L with color coding"""
    color_class = "profit-positive" if value >= 0 else "profit-negative"
    pnl_str = f"${value:,.2f}"
    if percentage is not None:
        pnl_str += f" ({percentage:+.2f}%)"
    return f'<span class="{color_class}">{pnl_str}</span>'

def create_metrics_chart(metrics: dict) -> go.Figure:
    """Create a professional metrics visualization"""
    fig = go.Figure()
    
    metric_names = []
    metric_values = []
    colors = []
    
    metric_map = {
        'total_return_mean': ('Total Return', 100, 'return'),
        'sharpe_ratio_mean': ('Sharpe Ratio', 1, 'ratio'),
        'win_rate_mean': ('Win Rate', 100, 'percentage'),
        'max_drawdown_mean': ('Max Drawdown', -100, 'risk'),
        'sortino_ratio_mean': ('Sortino Ratio', 1, 'ratio'),
        'calmar_ratio_mean': ('Calmar Ratio', 1, 'ratio')
    }
    
    for key, (display_name, multiplier, metric_type) in metric_map.items():
        if key in metrics:
            metric_names.append(display_name)
            value = metrics[key] * multiplier
            metric_values.append(value)
            
            # Color based on metric type and value
            if metric_type == 'return':
                colors.append('#0ecb81' if value > 0 else '#f6465d')
            elif metric_type == 'risk':
                colors.append('#f6465d' if value < -10 else '#ffd700')
            elif metric_type == 'ratio':
                colors.append('#0ecb81' if value > 1 else '#f6465d')
            else:
                colors.append('#1e3c72')
    
    if metric_names:
        fig.add_trace(go.Bar(
            x=metric_names,
            y=metric_values,
            marker_color=colors,
            text=[f'{v:.2f}' for v in metric_values],
            textposition='outside',
            hovertemplate='%{x}: %{y:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title={
                'text': "Key Performance Metrics",
                'font': {'size': 20, 'color': '#1e3c72'}
            },
            xaxis_title="Metric",
            yaxis_title="Value",
            height=400,
            template="plotly_white",
            showlegend=False,
            margin=dict(t=60, b=40, l=40, r=40),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        fig.update_xaxes(tickfont=dict(size=12))
        fig.update_yaxes(tickfont=dict(size=12), gridcolor='rgba(0,0,0,0.1)')
    
    return fig

def create_comprehensive_backtest_chart(results: dict) -> go.Figure:
    """Create comprehensive backtest visualization with professional styling"""
    if not results:
        return go.Figure().add_annotation(
            text="No backtest results available", 
            xref="paper", yref="paper", 
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color='#6b7280')
        )
    
    # Create subplots with custom spacing
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Performance Metrics", 
            "Feature Importance (Top 10)", 
            "Risk Decomposition", 
            "Signal Contributions"
        ),
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "pie"}, {"type": "bar"}]
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.10
    )
    
    # 1. Performance Metrics
    perf_metrics = results.get('performance_metrics', {})
    trading_stats = results.get('trading_statistics', {})
    
    metric_data = [
        ('Total Return %', perf_metrics.get('total_return_mean', 0) * 100, '#0ecb81' if perf_metrics.get('total_return_mean', 0) > 0 else '#f6465d'),
        ('Win Rate %', perf_metrics.get('win_rate_mean', 0) * 100, '#3b82f6'),
        ('Sharpe Ratio', perf_metrics.get('sharpe_ratio_mean', 0), '#8b5cf6'),
        ('Max Drawdown %', abs(perf_metrics.get('max_drawdown_mean', 0) * 100), '#f6465d')
    ]
    
    if metric_data:
        names, values, colors = zip(*metric_data)
        fig.add_trace(go.Bar(
            x=list(names),
            y=list(values),
            marker_color=list(colors),
            text=[f'{v:.2f}' for v in values],
            textposition='outside',
            name='Metrics',
            hovertemplate='%{x}: %{y:.2f}<extra></extra>'
        ), row=1, col=1)
    
    # 2. Feature Importance
    if 'feature_analysis' in results and 'feature_importance' in results['feature_analysis']:
        features = results['feature_analysis']['feature_importance']
        numeric_features = {}
        for k, v in features.items():
            try:
                numeric_features[k] = float(v)
            except (TypeError, ValueError):
                continue
        
        sorted_features = sorted(numeric_features.items(), 
                               key=lambda x: abs(x[1]), 
                               reverse=True)[:10]
        
        if sorted_features:
            feature_names = [f[0] for f in sorted_features]
            importance_values = [f[1] for f in sorted_features]
            colors = ['#0ecb81' if v > 0 else '#f6465d' for v in importance_values]
            
            fig.add_trace(go.Bar(
                y=feature_names,
                x=importance_values,
                orientation='h',
                marker_color=colors,
                text=[f'{v:.4f}' for v in importance_values],
                textposition='outside',
                name='Feature Importance',
                hovertemplate='%{y}: %{x:.4f}<extra></extra>'
            ), row=1, col=2)
    
    # 3. Risk Decomposition
    if 'risk_analysis' in results and 'decomposition' in results['risk_analysis']:
        risk_data = results['risk_analysis']['decomposition']
        
        risk_labels = []
        risk_values = []
        risk_colors = ['#f6465d', '#3b82f6', '#ffd700']
        
        risk_map = {
            'market_risk': 'Market Risk',
            'specific_risk': 'Specific Risk',
            'model_risk': 'Model Risk'
        }
        
        for key, label in risk_map.items():
            if key in risk_data and risk_data[key] > 0:
                risk_labels.append(label)
                risk_values.append(risk_data[key])
        
        if risk_labels:
            fig.add_trace(go.Pie(
                labels=risk_labels,
                values=risk_values,
                hole=0.4,
                marker_colors=risk_colors[:len(risk_labels)],
                textinfo='label+percent',
                textfont=dict(size=12),
                name='Risk Decomposition',
                hovertemplate='%{label}: %{percent}<extra></extra>'
            ), row=2, col=1)
    
    # 4. Signal Contributions
    if 'performance_metrics' in results and 'top_contributing_signals' in results['performance_metrics']:
        signals = results['performance_metrics']['top_contributing_signals']
        if signals:
            signal_names = []
            signal_contributions = []
            
            for sig_name, sig_data in list(signals.items())[:10]:
                signal_names.append(sig_name)
                signal_contributions.append(sig_data.get('total_contribution', 0) * 100)
            
            sorted_pairs = sorted(zip(signal_names, signal_contributions), 
                                key=lambda x: abs(x[1]), reverse=True)
            
            if sorted_pairs:
                signal_names, signal_contributions = zip(*sorted_pairs)
                colors = ['#0ecb81' if v > 0 else '#f6465d' for v in signal_contributions]
                
                fig.add_trace(go.Bar(
                    x=list(signal_names),
                    y=list(signal_contributions),
                    marker_color=colors,
                    text=[f'{v:.2f}%' for v in signal_contributions],
                    textposition='outside',
                    name='Signal Contributions',
                    hovertemplate='%{x}: %{y:.2f}%<extra></extra>'
                ), row=2, col=2)
    
    # Update layout with professional styling
    total_return = perf_metrics.get('total_return_mean', 0) * 100 if 'performance_metrics' in results else 0
    fig.update_layout(
        title={
            'text': f"Backtest Analysis - Total Return: {total_return:.2f}%",
            'font': {'size': 22, 'color': '#1e3c72'},
            'x': 0.5,
            'xanchor': 'center'
        },
        height=800,
        showlegend=False,
        template="plotly_white",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Arial, sans-serif", size=12, color="#495057")
    )
    
    # Update subplot titles
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=16, color='#1e3c72', family='Arial, sans-serif')
    
    # Update axes
    fig.update_xaxes(tickfont=dict(size=11), gridcolor='rgba(0,0,0,0.1)')
    fig.update_yaxes(tickfont=dict(size=11), gridcolor='rgba(0,0,0,0.1)')
    
    return fig

def create_monte_carlo_visualization(results: dict) -> go.Figure:
    """Create professional Monte Carlo simulation visualization"""
    if not results or 'simulations' not in results:
        return go.Figure().add_annotation(
            text="No Monte Carlo results available", 
            xref="paper", yref="paper", 
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color='#6b7280')
        )
    
    simulations = results['simulations']
    percentiles = results.get('percentiles', {})
    
    fig = go.Figure()
    
    # Add gradient background for confidence bands
    if percentiles and 'p95' in percentiles and 'p5' in percentiles:
        x_range = list(range(len(percentiles['p95'])))
        
        # 90% confidence band
        fig.add_trace(go.Scatter(
            x=x_range + x_range[::-1],
            y=percentiles['p95'] + percentiles['p5'][::-1],
            fill='toself',
            fillcolor='rgba(30, 60, 114, 0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=True,
            name='90% Confidence Band',
            hoverinfo='skip'
        ))
        
        # 50% confidence band
        if 'p75' in percentiles and 'p25' in percentiles:
            fig.add_trace(go.Scatter(
                x=x_range + x_range[::-1],
                y=percentiles['p75'] + percentiles['p25'][::-1],
                fill='toself',
                fillcolor='rgba(30, 60, 114, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=True,
                name='50% Confidence Band',
                hoverinfo='skip'
            ))
    
    # Plot sample paths
    if isinstance(simulations, list) and len(simulations) > 0:
        sample_paths = simulations[:min(50, len(simulations))]
        for i, path in enumerate(sample_paths):
            fig.add_trace(go.Scatter(
                y=path,
                mode='lines',
                line=dict(width=0.5, color='rgba(150,150,150,0.2)'),
                showlegend=False,
                hoverinfo='skip'
            ))
    
    # Add percentile lines
    if percentiles:
        # Median
        if 'p50' in percentiles:
            fig.add_trace(go.Scatter(
                y=percentiles['p50'],
                mode='lines',
                name='Median (50th)',
                line=dict(width=3, color='#1e3c72'),
                hovertemplate='Day %{x}<br>Value: $%{y:,.0f}<extra></extra>'
            ))
        
        # Extremes
        if 'p95' in percentiles:
            fig.add_trace(go.Scatter(
                y=percentiles['p95'],
                mode='lines',
                name='95th Percentile',
                line=dict(width=2, color='#0ecb81', dash='dash'),
                hovertemplate='Day %{x}<br>Value: $%{y:,.0f}<extra></extra>'
            ))
        
        if 'p5' in percentiles:
            fig.add_trace(go.Scatter(
                y=percentiles['p5'],
                mode='lines',
                name='5th Percentile',
                line=dict(width=2, color='#f6465d', dash='dash'),
                hovertemplate='Day %{x}<br>Value: $%{y:,.0f}<extra></extra>'
            ))
    
    fig.update_layout(
        title={
            'text': "Monte Carlo Simulation - Portfolio Projections",
            'font': {'size': 20, 'color': '#1e3c72'}
        },
        xaxis_title="Days",
        yaxis_title="Portfolio Value ($)",
        height=600,
        template="plotly_white",
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#e9ecef",
            borderwidth=1
        )
    )
    
    # Add grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.05)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.05)')
    
    return fig

def create_equity_curve(history: list) -> go.Figure:
    """Create portfolio equity curve with professional styling"""
    if not history:
        return go.Figure().add_annotation(
            text="No trading history yet", 
            xref="paper", yref="paper", 
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color='#6b7280')
        )
    
    df = pd.DataFrame(history)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    fig = go.Figure()
    
    # Portfolio value line with gradient fill
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['portfolio_value'],
        mode='lines',
        name='Portfolio Value',
        line=dict(color='#1e3c72', width=3),
        fill='tozeroy',
        fillcolor='rgba(30, 60, 114, 0.1)',
        hovertemplate='%{x}<br>Value: $%{y:,.2f}<extra></extra>'
    ))
    
    # Starting balance reference
    fig.add_hline(
        y=10000, 
        line_dash="dash", 
        line_color="#6b7280",
        annotation_text="Initial: $10,000",
        annotation_position="left"
    )
    
    # Add trade markers
    trades_df = df[df['action'].isin(['buy', 'sell'])]
    if not trades_df.empty:
        buy_trades = trades_df[trades_df['action'] == 'buy']
        sell_trades = trades_df[trades_df['action'] == 'sell']
        
        if not buy_trades.empty:
            fig.add_trace(go.Scatter(
                x=buy_trades['timestamp'],
                y=buy_trades['portfolio_value'],
                mode='markers',
                name='Buy',
                marker=dict(
                    size=10,
                    color='#0ecb81',
                    symbol='triangle-up',
                    line=dict(width=2, color='white')
                ),
                hovertemplate='Buy<br>%{x}<br>Value: $%{y:,.2f}<extra></extra>'
            ))
        
        if not sell_trades.empty:
            fig.add_trace(go.Scatter(
                x=sell_trades['timestamp'],
                y=sell_trades['portfolio_value'],
                mode='markers',
                name='Sell',
                marker=dict(
                    size=10,
                    color='#f6465d',
                    symbol='triangle-down',
                    line=dict(width=2, color='white')
                ),
                hovertemplate='Sell<br>%{x}<br>Value: $%{y:,.2f}<extra></extra>'
            ))
    
    # Calculate performance metrics
    initial_value = df['portfolio_value'].iloc[0]
    final_value = df['portfolio_value'].iloc[-1]
    total_return = ((final_value - initial_value) / initial_value) * 100
    
    fig.update_layout(
        title={
            'text': f"Portfolio Performance - Return: {total_return:+.2f}%",
            'font': {'size': 20, 'color': '#1e3c72'}
        },
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        template="plotly_white",
        height=450,
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#e9ecef",
            borderwidth=1
        )
    )
    
    # Style axes
    fig.update_xaxes(
        showgrid=True, 
        gridwidth=1, 
        gridcolor='rgba(0,0,0,0.05)',
        tickformat='%Y-%m-%d'
    )
    fig.update_yaxes(
        showgrid=True, 
        gridwidth=1, 
        gridcolor='rgba(0,0,0,0.05)',
        tickformat='$,.0f'
    )
    
    return fig

def show_analytics():
    """Main analytics interface with consolidated functionality"""
    
    st.title("📊 Analytics & Trading Hub")
    
    # Professional header
    st.markdown("""
    <div class="analytics-header">
        <h1 style="margin:0;">Comprehensive Analytics & Trading Platform</h1>
        <p style="margin:0.5rem 0 0 0; opacity:0.9;">Backtest strategies • Run simulations • Execute paper trades • Optimize performance</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main tabs
    main_tabs = st.tabs([
        "📈 Backtesting",
        "🎲 Monte Carlo",
        "💼 Paper Trading",
        "⚡ Optimization",
        "📊 Performance"
    ])
    
    # Tab 1: Backtesting
    with main_tabs[0]:
        st.subheader("Strategy Backtesting")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Parameter presets
            st.markdown("#### Quick Start Presets")
            preset_col1, preset_col2, preset_col3, preset_col4 = st.columns(4)
            
            preset = None
            with preset_col1:
                if st.button("🎯 Conservative", use_container_width=True):
                    preset = "conservative"
            with preset_col2:
                if st.button("⚖️ Balanced", use_container_width=True):
                    preset = "balanced"
            with preset_col3:
                if st.button("🚀 Aggressive", use_container_width=True):
                    preset = "aggressive"
            with preset_col4:
                if st.button("🤖 AI Optimized", use_container_width=True):
                    preset = "ai_optimized"
            
            # Apply presets
            if preset:
                if preset == "conservative":
                    st.session_state['bt_initial_capital'] = 10000
                    st.session_state['bt_position_size'] = 5.0
                    st.session_state['bt_stop_loss'] = 3.0
                    st.session_state['bt_take_profit'] = 5.0
                elif preset == "balanced":
                    st.session_state['bt_initial_capital'] = 10000
                    st.session_state['bt_position_size'] = 10.0
                    st.session_state['bt_stop_loss'] = 5.0
                    st.session_state['bt_take_profit'] = 10.0
                elif preset == "aggressive":
                    st.session_state['bt_initial_capital'] = 10000
                    st.session_state['bt_position_size'] = 20.0
                    st.session_state['bt_stop_loss'] = 10.0
                    st.session_state['bt_take_profit'] = 20.0
                elif preset == "ai_optimized":
                    st.session_state['bt_optimize_weights'] = True
                    st.session_state['bt_position_size'] = 15.0
                    st.session_state['bt_stop_loss'] = 7.0
                    st.session_state['bt_take_profit'] = 15.0
            
            # Backtest configuration
            with st.expander("📋 Backtest Configuration", expanded=True):
                config_tabs = st.tabs(["Basic", "Trading", "Advanced"])
                
                with config_tabs[0]:
                    col_b1, col_b2, col_b3 = st.columns(3)
                    
                    with col_b1:
                        period_days = st.number_input(
                            "Test Period (days)", 
                            min_value=30, 
                            max_value=730, 
                            value=180,
                            help="Historical period to test"
                        )
                        initial_capital = st.number_input(
                            "Initial Capital ($)", 
                            min_value=1000, 
                            max_value=1000000, 
                            value=st.session_state.get('bt_initial_capital', 10000),
                            key='bt_initial_capital'
                        )
                    
                    with col_b2:
                        strategy_type = st.selectbox(
                            "Strategy Type", 
                            ["AI Signals", "Technical Only", "Sentiment Based", "Custom"],
                            help="Trading strategy to backtest"
                        )
                        position_sizing = st.selectbox(
                            "Position Sizing", 
                            ["Fixed", "Kelly Criterion", "Risk Parity", "Volatility Adjusted"],
                            help="Position sizing method"
                        )
                    
                    with col_b3:
                        max_positions = st.number_input(
                            "Max Positions", 
                            min_value=1, 
                            max_value=10, 
                            value=3,
                            help="Maximum concurrent positions"
                        )
                        transaction_cost = st.number_input(
                            "Transaction Cost %", 
                            min_value=0.0, 
                            max_value=1.0, 
                            value=0.25,
                            step=0.01,
                            help="Trading fees per transaction"
                        )
                
                with config_tabs[1]:
                    col_t1, col_t2, col_t3 = st.columns(3)
                    
                    with col_t1:
                        position_size_pct = st.number_input(
                            "Position Size %", 
                            min_value=1.0, 
                            max_value=100.0, 
                            value=st.session_state.get('bt_position_size', 10.0),
                            key='bt_position_size',
                            help="Capital per trade"
                        )
                        buy_threshold = st.number_input(
                            "Buy Threshold %", 
                            min_value=0.1, 
                            max_value=10.0, 
                            value=2.0,
                            step=0.1,
                            help="Signal threshold for buys"
                        )
                    
                    with col_t2:
                        sell_threshold = st.number_input(
                            "Sell Threshold %", 
                            min_value=0.1, 
                            max_value=10.0, 
                            value=2.0,
                            step=0.1,
                            help="Signal threshold for sells"
                        )
                        sell_percentage = st.number_input(
                            "Sell Percentage %", 
                            min_value=10.0, 
                            max_value=100.0, 
                            value=50.0,
                            help="Holdings to sell on signal"
                        )
                    
                    with col_t3:
                        stop_loss = st.number_input(
                            "Stop Loss %", 
                            min_value=0.0, 
                            max_value=50.0, 
                            value=st.session_state.get('bt_stop_loss', 5.0),
                            key='bt_stop_loss',
                            help="Maximum loss per trade"
                        )
                        take_profit = st.number_input(
                            "Take Profit %", 
                            min_value=0.0, 
                            max_value=100.0, 
                            value=st.session_state.get('bt_take_profit', 10.0),
                            key='bt_take_profit',
                            help="Profit target per trade"
                        )
                
                with config_tabs[2]:
                    col_a1, col_a2 = st.columns(2)
                    
                    with col_a1:
                        use_walk_forward = st.checkbox(
                            "Use Walk-Forward Analysis", 
                            value=True, 
                            key="walk_forward",
                            help="Adaptive parameter optimization"
                        )
                        include_macro = st.checkbox(
                            "Include Macro Indicators", 
                            value=True, 
                            key="include_macro",
                            help="Use macroeconomic data"
                        )
                    
                    with col_a2:
                        optimize_weights = st.checkbox(
                            "Optimize Signal Weights", 
                            value=st.session_state.get('bt_optimize_weights', False), 
                            key="bt_optimize_weights",
                            help="Find optimal signal weights"
                        )
                        include_transaction_costs = st.checkbox(
                            "Include Transaction Costs", 
                            value=True, 
                            key="transaction_costs",
                            help="Apply trading fees"
                        )
            
            # Run backtest button with progress
            if st.button("🚀 Run Backtest", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                with st.spinner("Initializing backtest..."):
                    progress_bar.progress(10)
                    status_text.text("Preparing data...")
                    
                    # Clear previous results
                    if 'backtest_results' in st.session_state:
                        del st.session_state['backtest_results']
                    
                    # Prepare parameters
                    period_map = {30: "1mo", 90: "3mo", 180: "6mo", 365: "1y", 730: "2y"}
                    period_str = period_map.get(period_days, "1y")
                    
                    enhanced_params = {
                        "period": period_str,
                        "optimize_weights": st.session_state.get('bt_optimize_weights', False),
                        "include_macro": st.session_state.get('include_macro', True),
                        "use_enhanced_weights": True,
                        "n_optimization_trials": 20,
                        "settings": {
                            "initial_capital": initial_capital,
                            "strategy": strategy_type.lower().replace(" ", "_"),
                            "position_size": position_size_pct / 100,
                            "buy_threshold": buy_threshold / 100,
                            "sell_threshold": sell_threshold / 100,
                            "sell_percentage": sell_percentage / 100,
                            "stop_loss": stop_loss / 100,
                            "take_profit": take_profit / 100,
                            "transaction_cost": transaction_cost / 100 if include_transaction_costs else 0.0
                        }
                    }
                    
                    progress_bar.progress(30)
                    status_text.text("Running backtest simulation...")
                    
                    # Run backtest
                    result = api_client.post("/backtest/enhanced", enhanced_params)
                    
                    progress_bar.progress(90)
                    
                    if result and 'error' not in result and result.get('status') != 'error':
                        progress_bar.progress(100)
                        status_text.text("Backtest completed!")
                        st.success("✅ Backtest completed successfully!")
                        st.session_state['backtest_results'] = result
                        time.sleep(1)
                        st.rerun()
                    else:
                        progress_bar.empty()
                        status_text.empty()
                        error_msg = result.get('error', result.get('message', 'Unknown error')) if result else 'Connection error'
                        st.error(f"❌ Backtest failed: {error_msg}")
        
        with col2:
            # Quick stats
            if 'backtest_results' in st.session_state:
                results = st.session_state['backtest_results']
                perf_metrics = results.get('performance_metrics', {})
                trading_stats = results.get('trading_statistics', {})
                
                st.markdown("### 📊 Quick Stats")
                
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                
                total_return = perf_metrics.get('total_return_mean', 0) * 100
                return_color = "profit-positive" if total_return > 0 else "profit-negative"
                st.markdown(f"<h3 class='{return_color}'>{total_return:+.2f}%</h3>", unsafe_allow_html=True)
                st.markdown("<p style='margin:0;color:#6b7280;'>Total Return</p>", unsafe_allow_html=True)
                
                st.markdown("---")
                
                col_s1, col_s2 = st.columns(2)
                with col_s1:
                    st.metric("Trades", f"{trading_stats.get('total_trades', 0)}")
                    st.metric("Win Rate", f"{perf_metrics.get('win_rate_mean', 0)*100:.1f}%")
                with col_s2:
                    st.metric("Sharpe", f"{perf_metrics.get('sharpe_ratio_mean', 0):.2f}")
                    st.metric("Max DD", f"{perf_metrics.get('max_drawdown_mean', 0)*100:.1f}%")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Export button
                if st.button("📥 Export Results", use_container_width=True):
                    # Convert results to JSON for download
                    json_str = json.dumps(results, indent=2)
                    st.download_button(
                        label="Download JSON",
                        data=json_str,
                        file_name=f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
        
        # Display results
        if 'backtest_results' in st.session_state:
            results = st.session_state['backtest_results']
            
            # Main visualization
            st.plotly_chart(create_comprehensive_backtest_chart(results), use_container_width=True)
            
            # Detailed results tabs
            result_tabs = st.tabs(["📊 Metrics", "🔍 Analysis", "⚠️ Risk", "💡 Insights"])
            
            with result_tabs[0]:
                # Detailed metrics
                perf_metrics = results.get('performance_metrics', {})
                trading_stats = results.get('trading_statistics', {})
                risk_metrics = results.get('risk_metrics', {})
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown("**📈 Returns**")
                    st.markdown('<div class="result-card">', unsafe_allow_html=True)
                    st.write(f"Total Return: {perf_metrics.get('total_return_mean', 0)*100:.2f}%")
                    st.write(f"Sharpe Ratio: {perf_metrics.get('sharpe_ratio_mean', 0):.2f}")
                    st.write(f"Sortino Ratio: {perf_metrics.get('sortino_ratio_mean', 0):.2f}")
                    st.write(f"Calmar Ratio: {perf_metrics.get('calmar_ratio_mean', 0):.2f}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown("**📊 Trading**")
                    st.markdown('<div class="result-card">', unsafe_allow_html=True)
                    st.write(f"Total Trades: {trading_stats.get('total_trades', 0)}")
                    st.write(f"Win Rate: {perf_metrics.get('win_rate_mean', 0)*100:.1f}%")
                    st.write(f"Profit Factor: {perf_metrics.get('profit_factor_mean', 0):.2f}")
                    st.write(f"Avg Trade: {trading_stats.get('avg_trade_return', 0)*100:.2f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col3:
                    st.markdown("**⚠️ Risk**")
                    st.markdown('<div class="result-card">', unsafe_allow_html=True)
                    st.write(f"Max Drawdown: {perf_metrics.get('max_drawdown_mean', 0)*100:.2f}%")
                    st.write(f"VaR (95%): {risk_metrics.get('var_95', 0)*100:.2f}%")
                    st.write(f"CVaR (95%): {risk_metrics.get('cvar_95', 0)*100:.2f}%")
                    st.write(f"Volatility: {risk_metrics.get('volatility', 0)*100:.2f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col4:
                    st.markdown("**🎯 Efficiency**")
                    st.markdown('<div class="result-card">', unsafe_allow_html=True)
                    st.write(f"Long: {trading_stats.get('long_positions', 0)}")
                    st.write(f"Short: {trading_stats.get('short_positions', 0)}")
                    st.write(f"Turnover: {trading_stats.get('avg_position_turnover', 0)*100:.2f}%")
                    st.write(f"Avg Hold: {trading_stats.get('avg_holding_days', 'N/A')} days")
                    st.markdown('</div>', unsafe_allow_html=True)
            
            with result_tabs[1]:
                # Feature and signal analysis
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'optimal_weights' in results:
                        st.markdown("**📊 Optimal Signal Weights**")
                        weights = results['optimal_weights']
                        
                        weight_df = pd.DataFrame(list(weights.items()), columns=['Signal Type', 'Weight'])
                        weight_df['Weight %'] = weight_df['Weight'] * 100
                        
                        fig = go.Figure(go.Pie(
                            labels=weight_df['Signal Type'],
                            values=weight_df['Weight'],
                            hole=0.4,
                            marker_colors=['#1e3c72', '#2a5298', '#3b82f6', '#60a5fa']
                        ))
                        fig.update_layout(
                            height=300, 
                            title="Optimal Weight Distribution",
                            showlegend=True,
                            margin=dict(t=40, b=0, l=0, r=0)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    if 'market_analysis' in results:
                        st.markdown("**🌍 Market Analysis**")
                        market = results['market_analysis']
                        
                        st.markdown('<div class="result-card">', unsafe_allow_html=True)
                        
                        regime = market.get('regime', 'Unknown')
                        regime_class = f"regime-{regime.lower()}"
                        st.markdown(f"<div class='{regime_class} regime-indicator'>{regime.upper()}</div>", unsafe_allow_html=True)
                        
                        st.write(f"**Trend:** {market.get('dominant_trend', 'Unknown')}")
                        st.write(f"**Volatility:** {market.get('volatility_regime', 'Unknown')}")
                        st.write(f"**Momentum:** {market.get('momentum_state', 'Unknown')}")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
            
            with result_tabs[2]:
                # Risk analysis
                if 'risk_analysis' in results:
                    risk_data = results['risk_analysis']
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.markdown("**📊 Risk Breakdown**")
                        decomp = risk_data.get('decomposition', {})
                        
                        st.markdown('<div class="result-card">', unsafe_allow_html=True)
                        st.metric("Total Risk", f"{decomp.get('total_risk', 0)*100:.2f}%")
                        st.metric("Market Risk", f"{decomp.get('market_risk', 0)*100:.2f}%")
                        st.metric("Specific Risk", f"{decomp.get('specific_risk', 0)*100:.2f}%")
                        st.metric("Model Risk", f"{decomp.get('model_risk', 0)*100:.2f}%")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2:
                        if 'stress_scenarios' in risk_data:
                            st.markdown("**🎭 Stress Test Results**")
                            
                            scenarios = risk_data['stress_scenarios']
                            scenario_df = pd.DataFrame([
                                {
                                    'Scenario': 'Flash Crash',
                                    'Impact': scenarios.get('flash_crash', {}).get('impact', 0) * 100,
                                    'Max DD': scenarios.get('flash_crash', {}).get('max_drawdown', 0) * 100
                                },
                                {
                                    'Scenario': 'Bull Rally',
                                    'Impact': scenarios.get('bull_rally', {}).get('impact', 0) * 100,
                                    'Max DD': scenarios.get('bull_rally', {}).get('max_drawdown', 0) * 100
                                },
                                {
                                    'Scenario': 'High Volatility',
                                    'Impact': scenarios.get('high_volatility', {}).get('impact', 0) * 100,
                                    'Max DD': scenarios.get('high_volatility', {}).get('max_drawdown', 0) * 100
                                }
                            ])
                            
                            st.dataframe(
                                scenario_df.style.format({
                                    'Impact': '{:+.1f}%',
                                    'Max DD': '{:.1f}%'
                                }).map(
                                    lambda x: 'color: #0ecb81' if x > 0 else 'color: #f6465d' if x < 0 else '',
                                    subset=['Impact']
                                ),
                                use_container_width=True
                            )
            
            with result_tabs[3]:
                # Recommendations and insights
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    if 'recommendations' in results:
                        st.markdown("**💡 Strategy Recommendations**")
                        st.markdown('<div class="result-card">', unsafe_allow_html=True)
                        for i, rec in enumerate(results['recommendations'], 1):
                            st.write(f"{i}. {rec}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    if 'signal_analysis' in results and 'optimal_combinations' in results['signal_analysis']:
                        st.markdown("**🎯 Optimal Signal Combinations**")
                        st.markdown('<div class="result-card">', unsafe_allow_html=True)
                        for combo in results['signal_analysis']['optimal_combinations']:
                            st.write(f"• {combo}")
                        st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown("**📌 Key Takeaways**")
                    st.markdown('<div class="result-card">', unsafe_allow_html=True)
                    
                    perf_metrics = results.get('performance_metrics', {})
                    total_return = perf_metrics.get('total_return_mean', 0)
                    sharpe = perf_metrics.get('sharpe_ratio_mean', 0)
                    
                    if total_return > 0.1 and sharpe > 1:
                        st.success("✅ Strong performance")
                    elif total_return > 0 and sharpe > 0.5:
                        st.info("📊 Moderate performance")
                    else:
                        st.warning("⚠️ Needs optimization")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
    
    # Tab 2: Monte Carlo
    with main_tabs[1]:
        st.subheader("Monte Carlo Risk Analysis")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Simulation configuration
            with st.expander("🎲 Simulation Parameters", expanded=True):
                col_mc1, col_mc2, col_mc3 = st.columns(3)
                
                with col_mc1:
                    num_simulations = st.slider(
                        "Simulations", 
                        min_value=100, 
                        max_value=10000, 
                        value=1000,
                        step=100,
                        help="Number of simulation paths"
                    )
                    time_horizon = st.slider(
                        "Time Horizon (days)", 
                        min_value=30, 
                        max_value=365, 
                        value=90,
                        help="Projection period"
                    )
                
                with col_mc2:
                    confidence_level = st.slider(
                        "Confidence Level", 
                        min_value=80, 
                        max_value=99, 
                        value=95,
                        help="Confidence interval"
                    )
                    use_historical = st.checkbox(
                        "Use Historical Data", 
                        value=True,
                        help="Base on historical patterns"
                    )
                
                with col_mc3:
                    volatility_regime = st.selectbox(
                        "Volatility Regime", 
                        ["Current", "High", "Low", "Custom"],
                        help="Market volatility assumption"
                    )
                    if volatility_regime == "Custom":
                        custom_vol = st.number_input(
                            "Annual Volatility %", 
                            min_value=10, 
                            max_value=200, 
                            value=50
                        )
            
            # Quick scenarios
            st.markdown("#### 🎯 Quick Scenarios")
            scenario_col1, scenario_col2, scenario_col3 = st.columns(3)
            
            with scenario_col1:
                if st.button("📈 Bull Market", use_container_width=True):
                    st.session_state['mc_volatility'] = "Low"
                    st.session_state['mc_horizon'] = 180
            
            with scenario_col2:
                if st.button("📉 Bear Market", use_container_width=True):
                    st.session_state['mc_volatility'] = "High"
                    st.session_state['mc_horizon'] = 90
            
            with scenario_col3:
                if st.button("🌊 Normal Market", use_container_width=True):
                    st.session_state['mc_volatility'] = "Current"
                    st.session_state['mc_horizon'] = 120
            
            # Run simulation
            if st.button("🎲 Run Monte Carlo Simulation", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                with st.spinner("Running simulations..."):
                    progress_bar.progress(20)
                    status_text.text(f"Generating {num_simulations} paths...")
                    
                    params = {
                        "num_simulations": num_simulations,
                        "time_horizon": time_horizon,
                        "confidence_level": confidence_level,
                        "use_historical": use_historical,
                        "volatility_regime": st.session_state.get('mc_volatility', volatility_regime).lower()
                    }
                    
                    if volatility_regime == "Custom":
                        params["custom_volatility"] = custom_vol / 100
                    
                    progress_bar.progress(60)
                    status_text.text("Calculating statistics...")
                    
                    result = api_client.post("/analytics/monte-carlo", params)
                    
                    progress_bar.progress(100)
                    
                    if result and result.get('status') == 'success':
                        status_text.text("Simulation complete!")
                        st.success("✅ Monte Carlo simulation completed!")
                        st.session_state['monte_carlo_results'] = result.get('results', {})
                        time.sleep(0.5)
                        progress_bar.empty()
                        status_text.empty()
                    else:
                        progress_bar.empty()
                        status_text.empty()
                        error_msg = result.get('error', 'Unknown error') if result else 'Connection error'
                        st.error(f"❌ Simulation failed: {error_msg}")
        
        with col2:
            # Risk metrics
            if 'monte_carlo_results' in st.session_state:
                mc_results = st.session_state['monte_carlo_results']
                risk_metrics = mc_results.get('risk_metrics', {})
                
                st.markdown("### 📊 Risk Metrics")
                
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                
                var_95 = risk_metrics.get('var_95', 0)
                cvar_95 = risk_metrics.get('cvar_95', 0)
                prob_loss = risk_metrics.get('prob_loss', 0)
                
                st.metric("VaR (95%)", f"${var_95:,.0f}")
                st.metric("CVaR (95%)", f"${cvar_95:,.0f}")
                
                prob_color = "profit-negative" if prob_loss > 0.5 else "profit-positive"
                st.markdown(f"<p>Prob. of Loss</p>", unsafe_allow_html=True)
                st.markdown(f"<h3 class='{prob_color}'>{prob_loss:.1%}</h3>", unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Display results
        if 'monte_carlo_results' in st.session_state:
            mc_results = st.session_state['monte_carlo_results']
            
            # Main visualization
            st.plotly_chart(create_monte_carlo_visualization(mc_results), use_container_width=True)
            
            # Statistical analysis
            with st.expander("📊 Statistical Summary", expanded=True):
                stats = mc_results.get('statistics', {})
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown("**📈 Expected Returns**")
                    st.markdown('<div class="result-card">', unsafe_allow_html=True)
                    st.write(f"Mean: {stats.get('mean_return', 0):.2%}")
                    st.write(f"Median: {stats.get('median_return', 0):.2%}")
                    st.write(f"Std Dev: {stats.get('std_dev', 0):.2%}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown("**📊 Percentiles**")
                    st.markdown('<div class="result-card">', unsafe_allow_html=True)
                    st.write(f"5th: ${stats.get('p5', 0):,.0f}")
                    st.write(f"25th: ${stats.get('p25', 0):,.0f}")
                    st.write(f"75th: ${stats.get('p75', 0):,.0f}")
                    st.write(f"95th: ${stats.get('p95', 0):,.0f}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col3:
                    st.markdown("**📉 Extremes**")
                    st.markdown('<div class="result-card">', unsafe_allow_html=True)
                    st.write(f"Max Loss: ${stats.get('max_loss', 0):,.0f}")
                    st.write(f"Max Gain: ${stats.get('max_gain', 0):,.0f}")
                    st.write(f"Range: ${stats.get('max_gain', 0) - stats.get('max_loss', 0):,.0f}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col4:
                    st.markdown("**📐 Distribution**")
                    st.markdown('<div class="result-card">', unsafe_allow_html=True)
                    st.write(f"Skewness: {stats.get('skewness', 0):.2f}")
                    st.write(f"Kurtosis: {stats.get('kurtosis', 0):.2f}")
                    
                    # Interpretation
                    if stats.get('skewness', 0) > 0:
                        st.write("Bias: Positive")
                    else:
                        st.write("Bias: Negative")
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # Scenario analysis
            if 'scenarios' in mc_results:
                with st.expander("🎭 Scenario Analysis", expanded=False):
                    scenarios = mc_results['scenarios']
                    
                    # Create scenario cards
                    scenario_cols = st.columns(len(scenarios))
                    
                    for idx, (scenario_name, scenario_data) in enumerate(scenarios.items()):
                        with scenario_cols[idx]:
                            st.markdown(f"**{scenario_name.replace('_', ' ').title()}**")
                            st.markdown('<div class="result-card">', unsafe_allow_html=True)
                            
                            prob = scenario_data.get('probability', 0)
                            ret = scenario_data.get('return', 0)
                            impact = scenario_data.get('impact', 'N/A')
                            
                            st.metric("Probability", f"{prob:.1%}")
                            
                            ret_color = "profit-positive" if ret > 0 else "profit-negative"
                            st.markdown(f"<p>Expected Return</p>", unsafe_allow_html=True)
                            st.markdown(f"<p class='{ret_color}'>{ret:.2%}</p>", unsafe_allow_html=True)
                            
                            st.write(f"Impact: {impact}")
                            
                            st.markdown('</div>', unsafe_allow_html=True)
    
    # Tab 3: Paper Trading
    with main_tabs[2]:
        st.subheader("Paper Trading Simulator")
        
        # Fetch current data
        try:
            pt_status = api_client.get_paper_trading_status() or {}
            current_price_data = api_client.get_current_price() or {}
            latest_signal = api_client.get_latest_signal() or {}
            
            # Try enhanced LSTM
            if latest_signal.get('source') != 'enhanced_lstm':
                enhanced_signal = api_client.get("/signals/enhanced/latest")
                if enhanced_signal and enhanced_signal.get('source') == 'enhanced_lstm':
                    latest_signal = enhanced_signal
                    
        except Exception as e:
            st.error(f"Error connecting to backend: {str(e)}")
            return
        
        # Extract data
        is_enabled = pt_status.get('enabled', False)
        portfolio = pt_status.get('portfolio', {})
        positions = pt_status.get('positions', [])
        performance = pt_status.get('performance', {})
        trade_history = pt_status.get('trades', [])
        current_price = current_price_data.get('price', 0)
        
        # Header with controls
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            status_text = "🟢 Active" if is_enabled else "🔴 Inactive"
            st.markdown(f"### Paper Trading Status: {status_text}")
        
        with col2:
            if st.button("⚡ Toggle Trading", use_container_width=True):
                result = api_client.toggle_paper_trading()
                if result:
                    st.success(f"Paper trading {'disabled' if is_enabled else 'enabled'}")
                    st.rerun()
        
        with col3:
            if st.button("🔄 Reset Portfolio", use_container_width=True):
                if st.session_state.get('confirm_reset'):
                    result = api_client.post("/paper-trading/reset", {})
                    if result:
                        st.success("Portfolio reset!")
                        del st.session_state['confirm_reset']
                        st.rerun()
                else:
                    st.session_state['confirm_reset'] = True
                    st.warning("Click again to confirm reset")
        
        # Trading interface
        trading_tabs = st.tabs(["📊 Overview", "💱 Trading", "📋 Positions", "📜 History"])
        
        with trading_tabs[0]:
            # Portfolio overview
            col1, col2, col3, col4 = st.columns(4)
            
            balance = portfolio.get('balance', 10000)
            btc_holdings = portfolio.get('btc_holdings', 0)
            btc_value = btc_holdings * current_price if current_price else 0
            total_value = balance + btc_value
            initial_balance = 10000
            total_pnl = total_value - initial_balance
            pnl_pct = (total_pnl / initial_balance) * 100
            
            with col1:
                st.metric("💵 Cash Balance", f"${balance:,.2f}")
            
            with col2:
                st.metric("₿ BTC Holdings", f"{btc_holdings:.6f}", f"${btc_value:,.2f}")
            
            with col3:
                st.metric("💼 Total Portfolio", f"${total_value:,.2f}", 
                         f"{total_pnl:+,.2f} ({pnl_pct:+.2f}%)")
            
            with col4:
                win_rate = performance.get('win_rate', 0)
                st.metric("📈 Win Rate", f"{win_rate:.1f}%")
            
            # Performance chart
            if 'history' in pt_status:
                st.plotly_chart(create_equity_curve(pt_status['history']), 
                               use_container_width=True)
            
            # Performance metrics
            with st.expander("📊 Detailed Performance", expanded=True):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown("**Trading Activity**")
                    st.metric("Total Trades", performance.get('total_trades', 0))
                    st.metric("Open Positions", len(positions))
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown("**Returns**")
                    st.metric("Sharpe Ratio", f"{performance.get('sharpe_ratio', 0):.2f}")
                    st.metric("Profit Factor", f"{performance.get('profit_factor', 0):.2f}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col3:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown("**Risk**")
                    st.metric("Max Drawdown", f"{performance.get('max_drawdown', 0):.2f}%")
                    st.metric("Risk/Reward", f"{performance.get('risk_reward_ratio', 0):.2f}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col4:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown("**Efficiency**")
                    st.metric("Avg Trade", f"${performance.get('avg_trade_pnl', 0):.2f}")
                    st.metric("Best Trade", f"${performance.get('best_trade', 0):.2f}")
                    st.markdown('</div>', unsafe_allow_html=True)
        
        with trading_tabs[1]:
            # Trading interface
            if not is_enabled:
                st.warning("⚠️ Enable paper trading to place orders")
            else:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown('<div class="order-form">', unsafe_allow_html=True)
                    
                    # Market info
                    col_price, col_signal = st.columns(2)
                    with col_price:
                        st.metric("Current BTC Price", f"${current_price:,.2f}")
                    
                    with col_signal:
                        if latest_signal:
                            signal = latest_signal.get('signal', 'hold')
                            confidence = latest_signal.get('confidence', 0)
                            signal_color = "profit-positive" if signal == 'buy' else "profit-negative" if signal == 'sell' else ""
                            source = "Enhanced LSTM" if latest_signal.get('source') == 'enhanced_lstm' else "LSTM"
                            
                            st.metric(
                                f"{source} Signal", 
                                f"{signal.upper()}", 
                                f"Confidence: {confidence:.1%}"
                            )
                    
                    st.markdown("---")
                    
                    # Quick trade buttons
                    st.markdown("#### ⚡ Quick Trade")
                    quick_col1, quick_col2, quick_col3 = st.columns(3)
                    
                    with quick_col1:
                        if st.button("🟢 Buy 0.001 BTC", use_container_width=True):
                            order_data = {
                                "type": "buy",
                                "amount": 0.001,
                                "order_type": "market"
                            }
                            result = api_client.post("/paper-trading/trade", order_data)
                            if result and result.get('success'):
                                st.success("Buy order executed!")
                                st.balloons()
                                time.sleep(1)
                                st.rerun()
                    
                    with quick_col2:
                        if st.button("🔴 Sell 50%", use_container_width=True, disabled=btc_holdings == 0):
                            order_data = {
                                "type": "sell",
                                "amount": btc_holdings * 0.5,
                                "order_type": "market"
                            }
                            result = api_client.post("/paper-trading/trade", order_data)
                            if result and result.get('success'):
                                st.success("Sell order executed!")
                                st.balloons()
                                time.sleep(1)
                                st.rerun()
                    
                    with quick_col3:
                        if st.button("🏁 Close All", use_container_width=True, disabled=btc_holdings == 0):
                            order_data = {
                                "type": "sell",
                                "amount": btc_holdings,
                                "order_type": "market"
                            }
                            result = api_client.post("/paper-trading/trade", order_data)
                            if result and result.get('success'):
                                st.success("All positions closed!")
                                time.sleep(1)
                                st.rerun()
                    
                    st.markdown("---")
                    
                    # Custom order form
                    st.markdown("#### 📝 Custom Order")
                    
                    order_type = st.radio("Order Type", ["Market Order", "Limit Order"], horizontal=True)
                    trade_direction = st.radio("Direction", ["Buy", "Sell"], horizontal=True)
                    
                    if trade_direction == "Buy":
                        max_amount = balance / current_price if current_price > 0 else 0
                        amount = st.number_input(
                            "BTC Amount", 
                            min_value=0.0001, 
                            max_value=max_amount,
                            value=min(0.001, max_amount),
                            format="%.6f"
                        )
                        cost = amount * current_price
                        st.info(f"💰 Cost: ${cost:,.2f}")
                    else:
                        amount = st.number_input(
                            "BTC Amount", 
                            min_value=0.0001, 
                            max_value=btc_holdings,
                            value=min(0.001, btc_holdings),
                            format="%.6f"
                        )
                        proceeds = amount * current_price
                        st.info(f"💵 Proceeds: ${proceeds:,.2f}")
                    
                    limit_price = None
                    if order_type == "Limit Order":
                        limit_price = st.number_input(
                            "Limit Price", 
                            min_value=1.0,
                            value=float(current_price),
                            format="%.2f"
                        )
                    
                    if st.button(f"Place {trade_direction} Order", use_container_width=True, type="primary"):
                        order_data = {
                            "type": trade_direction.lower(),
                            "amount": amount,
                            "order_type": "limit" if order_type == "Limit Order" else "market"
                        }
                        
                        if limit_price:
                            order_data["limit_price"] = limit_price
                        
                        result = api_client.post("/paper-trading/trade", order_data)
                        
                        if result and result.get('success'):
                            st.success(f"✅ {trade_direction} order executed!")
                            st.balloons()
                            time.sleep(1)
                            st.rerun()
                        else:
                            error_msg = result.get('error', 'Unknown error') if result else 'Connection error'
                            st.error(f"❌ Order failed: {error_msg}")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    # Trading suggestions
                    st.markdown("### 💡 AI Suggestions")
                    
                    if latest_signal:
                        signal = latest_signal.get('signal', 'hold')
                        confidence = latest_signal.get('confidence', 0)
                        
                        st.markdown('<div class="result-card">', unsafe_allow_html=True)
                        
                        if signal == 'buy' and confidence > 0.6:
                            st.success("🟢 Strong Buy Signal")
                            suggested_amount = min(0.01, (balance * 0.1) / current_price)
                            st.write(f"Amount: {suggested_amount:.6f} BTC")
                            st.write(f"Risk: ${suggested_amount * current_price:,.2f}")
                        elif signal == 'sell' and confidence > 0.6:
                            st.error("🔴 Strong Sell Signal")
                            suggested_amount = min(btc_holdings * 0.5, btc_holdings)
                            st.write(f"Amount: {suggested_amount:.6f} BTC")
                        else:
                            st.info("⏸️ Hold Position")
                            st.write("Wait for stronger signals")
                        
                        st.markdown("---")
                        
                        # Risk management tips
                        st.markdown("**Risk Management**")
                        st.write("• Max 2% risk per trade")
                        st.write("• Use stop losses")
                        st.write("• Take partial profits")
                        st.write("• Keep trading journal")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
        
        with trading_tabs[2]:
            # Current positions
            st.markdown("### 📋 Current Positions")
            
            if positions:
                for pos in positions:
                    entry_price = pos.get('entry_price', 0)
                    size = pos.get('size', 0)
                    entry_value = entry_price * size
                    current_value = current_price * size
                    pnl = current_value - entry_value
                    pnl_pct = (pnl / entry_value) * 100 if entry_value > 0 else 0
                    
                    st.markdown('<div class="position-card">', unsafe_allow_html=True)
                    
                    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                    
                    with col1:
                        position_type = pos.get('type', 'long').upper()
                        st.write(f"**{position_type} Position**")
                        st.write(f"Entry: ${entry_price:,.2f}")
                        st.write(f"Size: {size:.6f} BTC")
                    
                    with col2:
                        st.write("**Current Value**")
                        st.write(f"${current_value:,.2f}")
                        st.write(f"Price: ${current_price:,.2f}")
                    
                    with col3:
                        st.write("**P&L**")
                        st.markdown(format_pnl(pnl, pnl_pct), unsafe_allow_html=True)
                    
                    with col4:
                        if st.button("Close", key=f"close_{pos.get('id', '')}", use_container_width=True):
                            result = api_client.post("/paper-trading/close-position", 
                                                   {"position_id": pos.get('id')})
                            if result:
                                st.success("Position closed!")
                                st.rerun()
                    
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("📭 No open positions")
                st.write("Start trading to build your portfolio!")
        
        with trading_tabs[3]:
            # Trade history
            st.markdown("### 📜 Trade History")
            
            if trade_history:
                # Convert to DataFrame
                df = pd.DataFrame(trade_history)
                
                # Calculate P&L
                if 'exit_price' in df.columns:
                    df['pnl'] = (df['exit_price'] - df['entry_price']) * df['size']
                    df['pnl_pct'] = (df['pnl'] / (df['entry_price'] * df['size'])) * 100
                
                # Format timestamp
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
                
                # Display filters
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    search = st.text_input("🔍 Search trades", placeholder="Type, date, etc...")
                with col2:
                    sort_by = st.selectbox("Sort by", ["timestamp", "pnl", "size"])
                with col3:
                    sort_order = st.radio("Order", ["Desc", "Asc"], horizontal=True)
                
                # Apply filters
                if search:
                    mask = df.astype(str).apply(lambda x: x.str.contains(search, case=False)).any(axis=1)
                    df = df[mask]
                
                df = df.sort_values(sort_by, ascending=(sort_order == "Asc"))
                
                # Display table
                st.dataframe(
                    df[['timestamp', 'type', 'entry_price', 'exit_price', 'size', 'pnl', 'pnl_pct']].style.format({
                        'entry_price': '${:,.2f}',
                        'exit_price': '${:,.2f}',
                        'size': '{:.6f}',
                        'pnl': '${:,.2f}',
                        'pnl_pct': '{:+.2f}%'
                    }).map(
                        lambda x: 'color: #0ecb81' if x > 0 else 'color: #f6465d' if x < 0 else '',
                        subset=['pnl', 'pnl_pct']
                    ),
                    use_container_width=True,
                    height=400
                )
                
                # Summary stats
                st.markdown("---")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    total_pnl = df['pnl'].sum() if 'pnl' in df.columns else 0
                    color = "profit-positive" if total_pnl > 0 else "profit-negative"
                    st.markdown(f"**Total P&L**")
                    st.markdown(f"<h3 class='{color}'>${total_pnl:,.2f}</h3>", unsafe_allow_html=True)
                
                with col2:
                    avg_win = df[df['pnl'] > 0]['pnl'].mean() if 'pnl' in df.columns else 0
                    st.metric("Avg Win", f"${avg_win:,.2f}")
                
                with col3:
                    avg_loss = abs(df[df['pnl'] < 0]['pnl'].mean()) if 'pnl' in df.columns else 0
                    st.metric("Avg Loss", f"${avg_loss:,.2f}")
                
                with col4:
                    best_trade = df['pnl'].max() if 'pnl' in df.columns else 0
                    st.metric("Best Trade", f"${best_trade:,.2f}")
            else:
                st.info("📭 No trade history yet")
                st.write("Execute some trades to see your performance!")
    
    # Tab 4: Optimization
    with main_tabs[3]:
        st.subheader("Strategy Optimization")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Optimization setup
            with st.expander("⚙️ Optimization Configuration", expanded=True):
                st.markdown("#### Parameter Ranges")
                
                col_opt1, col_opt2 = st.columns(2)
                
                with col_opt1:
                    st.markdown("**Signal Weights**")
                    technical_range = st.slider(
                        "Technical Weight", 
                        min_value=0.0, 
                        max_value=1.0, 
                        value=(0.2, 0.6),
                        help="Weight for technical indicators"
                    )
                    onchain_range = st.slider(
                        "On-chain Weight", 
                        min_value=0.0, 
                        max_value=1.0, 
                        value=(0.2, 0.5),
                        help="Weight for on-chain metrics"
                    )
                    sentiment_range = st.slider(
                        "Sentiment Weight", 
                        min_value=0.0, 
                        max_value=1.0, 
                        value=(0.1, 0.3),
                        help="Weight for sentiment analysis"
                    )
                
                with col_opt2:
                    st.markdown("**Trading Parameters**")
                    confidence_range = st.slider(
                        "Min Confidence", 
                        min_value=0.3, 
                        max_value=0.9, 
                        value=(0.5, 0.7),
                        help="Confidence threshold range"
                    )
                    position_size_range = st.slider(
                        "Position Size %", 
                        min_value=1, 
                        max_value=20, 
                        value=(5, 15),
                        help="Position size range"
                    )
                    stop_loss_range = st.slider(
                        "Stop Loss %", 
                        min_value=1, 
                        max_value=10, 
                        value=(3, 7),
                        help="Stop loss range"
                    )
                
                st.markdown("---")
                
                col_obj1, col_obj2 = st.columns(2)
                
                with col_obj1:
                    st.markdown("**Optimization Objective**")
                    objective = st.selectbox(
                        "Primary Goal", 
                        ["Sharpe Ratio", "Total Return", "Win Rate", "Risk-Adjusted Return"],
                        help="What to optimize for"
                    )
                
                with col_obj2:
                    st.markdown("**Constraints**")
                    constraints = st.multiselect(
                        "Apply Constraints", 
                        ["Max Drawdown < 20%", "Min Win Rate > 50%", 
                         "Min Trades > 100", "Max Volatility < 30%"],
                        default=["Max Drawdown < 20%"],
                        help="Optimization constraints"
                    )
                
                num_iterations = st.slider(
                    "Optimization Iterations", 
                    min_value=10, 
                    max_value=100, 
                    value=50,
                    help="More iterations = better results but slower"
                )
            
            # Run optimization
            if st.button("⚡ Run Optimization", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                with st.spinner(f"Running {num_iterations} optimization trials..."):
                    progress_bar.progress(10)
                    status_text.text("Initializing optimization engine...")
                    
                    params = {
                        "ranges": {
                            "technical_weight": technical_range,
                            "onchain_weight": onchain_range,
                            "sentiment_weight": sentiment_range,
                            "min_confidence": confidence_range,
                            "position_size": [x/100 for x in position_size_range],
                            "stop_loss": [x/100 for x in stop_loss_range]
                        },
                        "objective": objective.lower().replace(" ", "_"),
                        "constraints": constraints,
                        "iterations": num_iterations
                    }
                    
                    # Simulate progress
                    for i in range(num_iterations):
                        progress_bar.progress(10 + int(80 * i / num_iterations))
                        status_text.text(f"Testing configuration {i+1}/{num_iterations}...")
                        time.sleep(0.05)  # Small delay for visual effect
                    
                    progress_bar.progress(90)
                    status_text.text("Analyzing results...")
                    
                    result = api_client.post("/analytics/optimize", params)
                    
                    progress_bar.progress(100)
                    
                    if result and result.get('status') == 'success':
                        status_text.text("Optimization complete!")
                        st.success("✅ Optimization completed successfully!")
                        st.session_state['optimization_results'] = result.get('results', {})
                        time.sleep(0.5)
                        progress_bar.empty()
                        status_text.empty()
                    else:
                        progress_bar.empty()
                        status_text.empty()
                        error_msg = result.get('error', 'Unknown error') if result else 'Connection error'
                        st.error(f"❌ Optimization failed: {error_msg}")
        
        with col2:
            # Best parameters
            if 'optimization_results' in st.session_state:
                opt_results = st.session_state['optimization_results']
                best_params = opt_results.get('best_parameters', {})
                
                st.markdown("### 🏆 Optimal Parameters")
                
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                
                if best_params:
                    for param, value in best_params.items():
                        param_name = param.replace("_", " ").title()
                        if isinstance(value, float):
                            if 'weight' in param or 'confidence' in param:
                                st.write(f"**{param_name}**")
                                st.write(f"{value:.1%}")
                            else:
                                st.write(f"**{param_name}**")
                                st.write(f"{value:.2f}")
                        else:
                            st.write(f"**{param_name}**")
                            st.write(f"{value}")
                
                # Expected performance
                expected_perf = opt_results.get('expected_performance', {})
                if expected_perf:
                    st.markdown("---")
                    st.markdown("**Expected Performance**")
                    st.write(f"Return: {expected_perf.get('return', 0):.2%}")
                    st.write(f"Sharpe: {expected_perf.get('sharpe', 0):.2f}")
                    st.write(f"Max DD: {expected_perf.get('max_dd', 0):.1%}")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Apply button
                if st.button("📝 Apply Parameters", use_container_width=True):
                    # Apply optimal parameters
                    config_update = {
                        "signal_weights": {
                            "technical": best_params.get('technical_weight', 0.4),
                            "onchain": best_params.get('onchain_weight', 0.3),
                            "sentiment": best_params.get('sentiment_weight', 0.2),
                            "macro": 1 - sum([
                                best_params.get('technical_weight', 0.4),
                                best_params.get('onchain_weight', 0.3),
                                best_params.get('sentiment_weight', 0.2)
                            ])
                        },
                        "trading_rules": {
                            "min_confidence": best_params.get('min_confidence', 0.6),
                            "position_size": best_params.get('position_size', 0.1),
                            "stop_loss": best_params.get('stop_loss', 0.05)
                        }
                    }
                    
                    result = api_client.post("/config/update", config_update)
                    
                    if result and result.get('status') == 'success':
                        st.success("✅ Parameters applied!")
                    else:
                        st.error("❌ Failed to apply parameters")
        
        # Display optimization results
        if 'optimization_results' in st.session_state:
            opt_results = st.session_state['optimization_results']
            
            # Convergence plot
            if 'convergence' in opt_results:
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    y=opt_results['convergence'],
                    mode='lines+markers',
                    name='Best Score',
                    line=dict(color='#1e3c72', width=3),
                    marker=dict(size=6)
                ))
                
                fig.update_layout(
                    title="Optimization Convergence",
                    xaxis_title="Iteration",
                    yaxis_title="Objective Value",
                    height=400,
                    template="plotly_white"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Parameter importance
            if 'parameter_importance' in opt_results:
                with st.expander("📊 Parameter Importance", expanded=True):
                    importance = opt_results['parameter_importance']
                    
                    imp_df = pd.DataFrame(
                        list(importance.items()), 
                        columns=['Parameter', 'Importance']
                    )
                    imp_df = imp_df.sort_values('Importance', ascending=False)
                    
                    fig = go.Figure(go.Bar(
                        x=imp_df['Importance'],
                        y=imp_df['Parameter'],
                        orientation='h',
                        marker_color='#1e3c72'
                    ))
                    
                    fig.update_layout(
                        title="Parameter Importance",
                        xaxis_title="Importance Score",
                        yaxis_title="Parameter",
                        height=400,
                        template="plotly_white"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    # Tab 5: Performance
    with main_tabs[4]:
        st.subheader("Performance Analytics")
        
        # Time period selector
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            time_period = st.select_slider(
                "Analysis Period",
                options=["1D", "1W", "1M", "3M", "6M", "1Y", "All"],
                value="1M"
            )
        
        # Get performance data
        perf_data = api_client.get(f"/analytics/performance?period={time_period}")
        
        if perf_data:
            # Performance overview
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown("**🎯 Accuracy**")
                st.metric("Signal Accuracy", f"{perf_data.get('signal_accuracy', 0):.1%}")
                st.metric("Direction Accuracy", f"{perf_data.get('direction_accuracy', 0):.1%}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown("**📈 Returns**")
                st.metric("Strategy Return", f"{perf_data.get('strategy_return', 0):.2%}")
                st.metric("Market Return", f"{perf_data.get('market_return', 0):.2%}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown("**📊 Efficiency**")
                st.metric("Alpha", f"{perf_data.get('alpha', 0):.2%}")
                st.metric("Beta", f"{perf_data.get('beta', 0):.2f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col4:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown("**⚡ Activity**")
                st.metric("Total Signals", perf_data.get('total_signals', 0))
                st.metric("Avg Daily Signals", f"{perf_data.get('avg_daily_signals', 0):.1f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Performance charts
            perf_tabs = st.tabs(["Returns", "Signals", "Risk", "Attribution"])
            
            with perf_tabs[0]:
                # Cumulative returns chart
                if 'returns_data' in perf_data:
                    returns_df = pd.DataFrame(perf_data['returns_data'])
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=returns_df['date'],
                        y=returns_df['strategy_cumulative'],
                        name='Strategy',
                        line=dict(color='#1e3c72', width=3)
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=returns_df['date'],
                        y=returns_df['market_cumulative'],
                        name='Market',
                        line=dict(color='#6b7280', width=2, dash='dash')
                    ))
                    
                    fig.update_layout(
                        title="Cumulative Returns Comparison",
                        xaxis_title="Date",
                        yaxis_title="Cumulative Return (%)",
                        height=500,
                        template="plotly_white",
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            with perf_tabs[1]:
                # Signal analysis
                if 'signal_data' in perf_data:
                    signal_df = pd.DataFrame(perf_data['signal_data'])
                    
                    # Signal distribution
                    fig = make_subplots(
                        rows=1, cols=2,
                        subplot_titles=("Signal Distribution", "Signal Accuracy by Type")
                    )
                    
                    # Distribution pie chart
                    signal_counts = signal_df['signal_type'].value_counts()
                    fig.add_trace(go.Pie(
                        labels=signal_counts.index,
                        values=signal_counts.values,
                        hole=0.4
                    ), row=1, col=1)
                    
                    # Accuracy by type
                    accuracy_by_type = signal_df.groupby('signal_type')['accurate'].mean()
                    fig.add_trace(go.Bar(
                        x=accuracy_by_type.index,
                        y=accuracy_by_type.values,
                        marker_color=['#0ecb81', '#f6465d', '#6b7280']
                    ), row=1, col=2)
                    
                    fig.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
            
            with perf_tabs[2]:
                # Risk metrics
                st.markdown("### Risk Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Drawdown chart
                    if 'drawdown_data' in perf_data:
                        dd_df = pd.DataFrame(perf_data['drawdown_data'])
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=dd_df['date'],
                            y=dd_df['drawdown'],
                            fill='tozeroy',
                            fillcolor='rgba(246, 70, 93, 0.2)',
                            line=dict(color='#f6465d', width=2),
                            name='Drawdown'
                        ))
                        
                        fig.update_layout(
                            title="Drawdown Analysis",
                            xaxis_title="Date",
                            yaxis_title="Drawdown (%)",
                            height=350,
                            template="plotly_white"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Risk metrics table
                    risk_metrics = perf_data.get('risk_metrics', {})
                    
                    risk_df = pd.DataFrame([
                        {'Metric': 'Volatility', 'Value': f"{risk_metrics.get('volatility', 0):.2%}"},
                        {'Metric': 'Downside Deviation', 'Value': f"{risk_metrics.get('downside_deviation', 0):.2%}"},
                        {'Metric': 'Max Drawdown', 'Value': f"{risk_metrics.get('max_drawdown', 0):.2%}"},
                        {'Metric': 'VaR (95%)', 'Value': f"{risk_metrics.get('var_95', 0):.2%}"},
                        {'Metric': 'CVaR (95%)', 'Value': f"{risk_metrics.get('cvar_95', 0):.2%}"}
                    ])
                    
                    st.dataframe(risk_df, use_container_width=True, hide_index=True)
            
            with perf_tabs[3]:
                # Performance attribution
                if 'attribution_data' in perf_data:
                    attr_data = perf_data['attribution_data']
                    
                    # Create attribution waterfall chart
                    fig = go.Figure(go.Waterfall(
                        name="Performance Attribution",
                        orientation="v",
                        measure=["relative", "relative", "relative", "relative", "total"],
                        x=["Market", "Timing", "Selection", "Interaction", "Total"],
                        y=[
                            attr_data.get('market', 0),
                            attr_data.get('timing', 0),
                            attr_data.get('selection', 0),
                            attr_data.get('interaction', 0),
                            None
                        ],
                        connector={"line": {"color": "rgb(63, 63, 63)"}},
                    ))
                    
                    fig.update_layout(
                        title="Performance Attribution Analysis",
                        xaxis_title="Component",
                        yaxis_title="Contribution (%)",
                        height=400,
                        template="plotly_white"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📊 No performance data available for the selected period")

# Auto-refresh option
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    
    auto_refresh = st.checkbox("Auto-refresh", value=False)
    if auto_refresh:
        refresh_interval = st.slider("Refresh interval (seconds)", 5, 60, 30)
        time.sleep(refresh_interval)
        st.rerun()
    
    st.markdown("---")
    
    # Quick actions
    st.markdown("### ⚡ Quick Actions")
    
    if st.button("📥 Export All Data", use_container_width=True):
        # Gather all data
        export_data = {
            "timestamp": datetime.now().isoformat(),
            "backtest_results": st.session_state.get('backtest_results'),
            "monte_carlo_results": st.session_state.get('monte_carlo_results'),
            "optimization_results": st.session_state.get('optimization_results'),
            "paper_trading_status": api_client.get_paper_trading_status()
        }
        
        json_str = json.dumps(export_data, indent=2)
        st.download_button(
            label="Download JSON",
            data=json_str,
            file_name=f"analytics_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    if st.button("🔄 Clear Cache", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Cache cleared!")
    
    st.markdown("---")
    
    # System status
    st.markdown("### 📡 System Status")
    
    try:
        health = api_client.get("/health")
        if health:
            st.success("🟢 Backend: Online")
        else:
            st.error("🔴 Backend: Offline")
    except:
        st.error("🔴 Backend: Offline")

# Show the page
show_analytics()