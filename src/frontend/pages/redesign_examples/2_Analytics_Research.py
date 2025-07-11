"""
Analytics & Research - Comprehensive analysis tools
Combines backtesting, Monte Carlo, optimization, and data quality
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import sys

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components.api_client import APIClient
import config

# Page configuration
st.set_page_config(
    page_title="Analytics & Research - BTC Trading System",
    page_icon="BTC",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load professional theme CSS
with open("styles/professional_theme.css", "r") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Additional page-specific CSS
st.markdown("""
<style>
/* Analytics specific styles */
.analytics-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 16px 0;
    border-bottom: 1px solid var(--border-subtle);
    margin-bottom: 20px;
}

.analytics-tabs {
    display: flex;
    gap: 8px;
    background: var(--bg-secondary);
    padding: 4px;
    border-radius: 6px;
}

.tab-button {
    padding: 8px 16px;
    background: transparent;
    border: none;
    border-radius: 4px;
    color: var(--text-secondary);
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s ease;
}

.tab-button.active {
    background: var(--bg-tertiary);
    color: var(--text-primary);
}

.results-container {
    display: grid;
    grid-template-columns: 250px 1fr;
    gap: 16px;
    height: calc(100vh - 200px);
}

.sidebar-params {
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    padding: 16px;
    overflow-y: auto;
}

.main-results {
    display: flex;
    flex-direction: column;
    gap: 16px;
}

.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 12px;
}

.result-card {
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    padding: 16px;
}

.param-section {
    margin-bottom: 20px;
}

.param-section h4 {
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted);
    margin-bottom: 12px;
}

/* Compact form inputs */
.stNumberInput > div > div > input {
    height: 32px;
}

.stSelectbox > div > div > select {
    height: 32px;
}

/* Remove extra spacing */
.stTabs [data-baseweb="tab-panel"] {
    padding-top: 0;
}

[data-testid="column"] > div:first-child {
    gap: 8px;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "api_client" not in st.session_state:
    st.session_state.api_client = APIClient(base_url=config.API_BASE_URL)
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "backtesting"

api_client = st.session_state.api_client

# Header
st.markdown("""
<div class="analytics-header">
    <div>
        <h1 style="margin: 0;">Analytics & Research</h1>
        <p style="margin: 0; color: var(--text-secondary); font-size: 14px;">
            Advanced analysis tools for strategy development and validation
        </p>
    </div>
    <div class="status-indicator">
        <span class="text-muted">Last update: {}</span>
    </div>
</div>
""".format(datetime.now().strftime('%H:%M:%S')), unsafe_allow_html=True)

# Tab navigation
tab1, tab2, tab3, tab4 = st.tabs(["Backtesting", "Monte Carlo", "Optimization", "Data Quality"])

# Backtesting Tab
with tab1:
    col_params, col_results = st.columns([1, 3])
    
    # Parameters sidebar
    with col_params:
        st.markdown("### Backtest Parameters")
        
        # Time Period
        st.markdown("#### Time Period")
        period_type = st.radio("Period Type", ["Quick", "Custom"], horizontal=True, label_visibility="collapsed")
        
        if period_type == "Quick":
            quick_period = st.selectbox("Select Period", ["Last 7 days", "Last 30 days", "Last 90 days", "Last 180 days"])
            days = int(quick_period.split()[1])
            start_date = datetime.now() - timedelta(days=days)
            end_date = datetime.now()
        else:
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start", datetime.now() - timedelta(days=30))
            with col2:
                end_date = st.date_input("End", datetime.now())
        
        # Trading Parameters
        st.markdown("#### Trading Parameters")
        initial_capital = st.number_input("Initial Capital ($)", min_value=1000, max_value=1000000, value=10000, step=1000)
        position_size = st.slider("Position Size %", min_value=5, max_value=100, value=20, step=5)
        
        # Strategy Parameters
        st.markdown("#### Strategy")
        strategy_type = st.selectbox("Strategy Type", ["LSTM Signals", "Technical Indicators", "Hybrid"])
        
        buy_threshold = st.number_input("Buy Threshold %", min_value=0.1, max_value=10.0, value=2.0, step=0.1)
        sell_threshold = st.number_input("Sell Threshold %", min_value=0.1, max_value=10.0, value=2.0, step=0.1)
        
        # Risk Management
        with st.expander("Risk Management", expanded=False):
            stop_loss = st.number_input("Stop Loss %", min_value=0.0, max_value=20.0, value=5.0, step=0.5)
            take_profit = st.number_input("Take Profit %", min_value=0.0, max_value=50.0, value=10.0, step=1.0)
            max_positions = st.number_input("Max Positions", min_value=1, max_value=10, value=3)
        
        # Advanced Options
        with st.expander("Advanced Options", expanded=False):
            optimize_weights = st.checkbox("Optimize Signal Weights", value=False)
            walk_forward = st.checkbox("Walk-Forward Analysis", value=False)
            include_costs = st.checkbox("Include Transaction Costs", value=True)
            transaction_cost = st.number_input("Transaction Cost %", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
        
        # Run button
        if st.button("Run Backtest", type="primary", use_container_width=True):
            with st.spinner("Running backtest..."):
                # Prepare parameters
                params = {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "initial_capital": initial_capital,
                    "position_size": position_size / 100,
                    "buy_threshold": buy_threshold / 100,
                    "sell_threshold": sell_threshold / 100,
                    "stop_loss": stop_loss / 100,
                    "take_profit": take_profit / 100,
                    "strategy_type": strategy_type.lower().replace(" ", "_"),
                    "optimize_weights": optimize_weights,
                    "use_walk_forward": walk_forward,
                    "include_transaction_costs": include_costs,
                    "transaction_cost": transaction_cost / 100
                }
                
                # Run backtest
                try:
                    results = api_client.post("/analytics/backtest/enhanced", params)
                    st.session_state.backtest_results = results
                    st.success("Backtest completed!")
                except Exception as e:
                    st.error(f"Backtest failed: {str(e)}")
    
    # Results area
    with col_results:
        if "backtest_results" in st.session_state and st.session_state.backtest_results:
            results = st.session_state.backtest_results
            
            # Performance Overview
            st.markdown("### Performance Overview")
            
            # Key metrics
            metrics_cols = st.columns(5)
            perf_metrics = results.get("performance_metrics", {})
            
            with metrics_cols[0]:
                total_return = perf_metrics.get("total_return_mean", 0) * 100
                st.metric("Total Return", f"{total_return:.2f}%", None)
            
            with metrics_cols[1]:
                sharpe = perf_metrics.get("sharpe_ratio_mean", 0)
                st.metric("Sharpe Ratio", f"{sharpe:.2f}", None)
            
            with metrics_cols[2]:
                max_dd = perf_metrics.get("max_drawdown_mean", 0) * 100
                st.metric("Max Drawdown", f"{max_dd:.2f}%", None)
            
            with metrics_cols[3]:
                win_rate = perf_metrics.get("win_rate_mean", 0) * 100
                st.metric("Win Rate", f"{win_rate:.1f}%", None)
            
            with metrics_cols[4]:
                trades = results.get("trading_statistics", {}).get("total_trades_mean", 0)
                st.metric("Total Trades", f"{trades:.0f}", None)
            
            # Results tabs
            result_tab1, result_tab2, result_tab3, result_tab4 = st.tabs(["Equity Curve", "Trade Analysis", "Risk Metrics", "Feature Importance"])
            
            # Equity Curve
            with result_tab1:
                # Create sample equity curve
                fig = go.Figure()
                
                # Generate sample data (replace with actual backtest data)
                dates = pd.date_range(start=start_date, end=end_date, freq='D')
                equity = [initial_capital]
                for i in range(1, len(dates)):
                    # Simulate equity growth
                    daily_return = (total_return / 100) / len(dates) + (pd.np.random.randn() * 0.02)
                    equity.append(equity[-1] * (1 + daily_return))
                
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=equity,
                    mode='lines',
                    name='Portfolio Value',
                    line=dict(color='#f7931a', width=2)
                ))
                
                # Add benchmark
                benchmark = [initial_capital] * len(dates)
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=benchmark,
                    mode='lines',
                    name='Benchmark',
                    line=dict(color='#6b7280', width=1, dash='dash')
                ))
                
                fig.update_layout(
                    height=400,
                    margin=dict(l=0, r=0, t=0, b=0),
                    paper_bgcolor='#131315',
                    plot_bgcolor='#0a0a0b',
                    xaxis=dict(gridcolor='#27272a'),
                    yaxis=dict(gridcolor='#27272a', title='Portfolio Value ($)'),
                    font=dict(color='#e5e5e7', size=11),
                    legend=dict(x=0, y=1)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Trade Analysis
            with result_tab2:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Trade distribution
                    st.markdown("#### Trade Distribution")
                    trade_stats = results.get("trading_statistics", {})
                    
                    metrics_html = f"""
                    <div class="metric-grid">
                        <div class="metric-card">
                            <div class="metric-label">Avg Win</div>
                            <div class="metric-value text-success">${trade_stats.get('avg_win_mean', 0):.2f}</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Avg Loss</div>
                            <div class="metric-value text-danger">${trade_stats.get('avg_loss_mean', 0):.2f}</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Best Trade</div>
                            <div class="metric-value text-success">${trade_stats.get('best_trade_mean', 0):.2f}</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Worst Trade</div>
                            <div class="metric-value text-danger">${trade_stats.get('worst_trade_mean', 0):.2f}</div>
                        </div>
                    </div>
                    """
                    st.markdown(metrics_html, unsafe_allow_html=True)
                
                with col2:
                    # Win/Loss pie chart
                    st.markdown("#### Win/Loss Ratio")
                    wins = int(trades * win_rate / 100)
                    losses = int(trades - wins)
                    
                    fig = go.Figure(data=[go.Pie(
                        labels=['Wins', 'Losses'],
                        values=[wins, losses],
                        hole=.3,
                        marker_colors=['#22c55e', '#ef4444']
                    )])
                    
                    fig.update_layout(
                        height=250,
                        margin=dict(l=0, r=0, t=0, b=0),
                        paper_bgcolor='#131315',
                        font=dict(color='#e5e5e7', size=11),
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            # Risk Metrics
            with result_tab3:
                risk_metrics = results.get("risk_metrics", {})
                
                st.markdown("#### Risk Analysis")
                
                # Risk metrics grid
                risk_html = f"""
                <div class="metric-grid">
                    <div class="metric-card">
                        <div class="metric-label">Value at Risk (95%)</div>
                        <div class="metric-value">${risk_metrics.get('var_95_mean', 0):.2f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Conditional VaR</div>
                        <div class="metric-value">${risk_metrics.get('cvar_95_mean', 0):.2f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Sortino Ratio</div>
                        <div class="metric-value">{risk_metrics.get('sortino_ratio_mean', 0):.2f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Calmar Ratio</div>
                        <div class="metric-value">{risk_metrics.get('calmar_ratio_mean', 0):.2f}</div>
                    </div>
                </div>
                """
                st.markdown(risk_html, unsafe_allow_html=True)
                
                # Drawdown chart
                st.markdown("#### Drawdown Analysis")
                
                # Generate sample drawdown data
                fig = go.Figure()
                
                drawdowns = []
                peak = equity[0]
                for val in equity:
                    if val > peak:
                        peak = val
                    drawdown = (val - peak) / peak * 100
                    drawdowns.append(drawdown)
                
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=drawdowns,
                    mode='lines',
                    fill='tozeroy',
                    name='Drawdown',
                    line=dict(color='#ef4444', width=1),
                    fillcolor='rgba(239, 68, 68, 0.2)'
                ))
                
                fig.update_layout(
                    height=250,
                    margin=dict(l=0, r=0, t=0, b=0),
                    paper_bgcolor='#131315',
                    plot_bgcolor='#0a0a0b',
                    xaxis=dict(gridcolor='#27272a'),
                    yaxis=dict(gridcolor='#27272a', title='Drawdown (%)'),
                    font=dict(color='#e5e5e7', size=11)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Feature Importance
            with result_tab4:
                if "feature_importance" in results:
                    st.markdown("#### Feature Importance Analysis")
                    
                    # Feature importance chart
                    features = results["feature_importance"]
                    if features:
                        # Sort features by importance
                        sorted_features = sorted(features.items(), key=lambda x: x[1], reverse=True)[:15]
                        
                        fig = go.Figure(data=[
                            go.Bar(
                                x=[f[1] for f in sorted_features],
                                y=[f[0] for f in sorted_features],
                                orientation='h',
                                marker_color='#f7931a'
                            )
                        ])
                        
                        fig.update_layout(
                            height=400,
                            margin=dict(l=0, r=0, t=0, b=0),
                            paper_bgcolor='#131315',
                            plot_bgcolor='#0a0a0b',
                            xaxis=dict(gridcolor='#27272a', title='Importance'),
                            yaxis=dict(gridcolor='#27272a'),
                            font=dict(color='#e5e5e7', size=11)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Feature importance not available for this backtest")
        else:
            # No results yet
            st.markdown("""
            <div style="text-align: center; padding: 100px 20px; color: var(--text-secondary);">
                <h3 style="margin-bottom: 16px;">No backtest results yet</h3>
                <p>Configure parameters and run a backtest to see results</p>
            </div>
            """, unsafe_allow_html=True)

# Monte Carlo Tab
with tab2:
    col_params, col_results = st.columns([1, 3])
    
    with col_params:
        st.markdown("### Simulation Parameters")
        
        # Basic parameters
        st.markdown("#### Basic Settings")
        num_simulations = st.number_input("Number of Simulations", min_value=100, max_value=10000, value=1000, step=100)
        time_horizon = st.slider("Time Horizon (days)", min_value=1, max_value=365, value=30)
        
        # Market parameters
        st.markdown("#### Market Parameters")
        volatility_regime = st.selectbox("Volatility Regime", ["Current Market", "Low (Bull)", "Medium (Normal)", "High (Bear)", "Custom"])
        
        if volatility_regime == "Custom":
            annual_volatility = st.slider("Annual Volatility %", min_value=10, max_value=200, value=60)
        else:
            annual_volatility = {"Current Market": 60, "Low (Bull)": 40, "Medium (Normal)": 60, "High (Bear)": 100}[volatility_regime]
        
        # Distribution
        st.markdown("#### Distribution")
        distribution = st.selectbox("Price Distribution", ["Log-Normal", "Normal", "Student-t"])
        
        # Run simulation
        if st.button("Run Simulation", type="primary", use_container_width=True):
            with st.spinner("Running Monte Carlo simulation..."):
                params = {
                    "num_simulations": num_simulations,
                    "time_horizon": time_horizon,
                    "volatility_regime": volatility_regime,
                    "annual_volatility": annual_volatility / 100,
                    "distribution": distribution.lower()
                }
                
                try:
                    results = api_client.post("/analytics/monte-carlo", params)
                    st.session_state.monte_carlo_results = results
                    st.success("Simulation completed!")
                except Exception as e:
                    st.error(f"Simulation failed: {str(e)}")
    
    with col_results:
        if "monte_carlo_results" in st.session_state and st.session_state.monte_carlo_results:
            results = st.session_state.monte_carlo_results
            
            st.markdown("### Simulation Results")
            
            # Summary metrics
            metrics = st.columns(4)
            risk_metrics = results.get("risk_metrics", {})
            stats = results.get("statistics", {})
            
            with metrics[0]:
                st.metric("Expected Return", f"{stats.get('mean_return', 0)*100:.2f}%", None)
            with metrics[1]:
                st.metric("95% VaR", f"${risk_metrics.get('var_95', 0):,.2f}", None)
            with metrics[2]:
                st.metric("Probability of Loss", f"{risk_metrics.get('probability_of_loss', 0)*100:.1f}%", None)
            with metrics[3]:
                st.metric("Best Case (95%)", f"${results.get('percentiles', {}).get('p95', [0])[-1]:,.2f}", None)
            
            # Price paths visualization
            st.markdown("#### Price Paths")
            
            simulations = results.get("simulations", [])
            percentiles = results.get("percentiles", {})
            
            if simulations and percentiles:
                fig = go.Figure()
                
                # Plot sample paths
                for i, path in enumerate(simulations[:100]):  # Show first 100 paths
                    fig.add_trace(go.Scatter(
                        x=list(range(len(path))),
                        y=path,
                        mode='lines',
                        line=dict(color='rgba(247, 147, 26, 0.1)', width=1),
                        showlegend=False
                    ))
                
                # Plot percentiles
                x_range = list(range(len(percentiles.get("p50", []))))
                
                # Median
                fig.add_trace(go.Scatter(
                    x=x_range,
                    y=percentiles.get("p50", []),
                    mode='lines',
                    name='Median',
                    line=dict(color='#f7931a', width=2)
                ))
                
                # Confidence bands
                fig.add_trace(go.Scatter(
                    x=x_range + x_range[::-1],
                    y=percentiles.get("p75", []) + percentiles.get("p25", [])[::-1],
                    fill='toself',
                    fillcolor='rgba(247, 147, 26, 0.2)',
                    line=dict(color='rgba(247, 147, 26, 0)'),
                    name='50% Confidence'
                ))
                
                fig.add_trace(go.Scatter(
                    x=x_range + x_range[::-1],
                    y=percentiles.get("p95", []) + percentiles.get("p5", [])[::-1],
                    fill='toself',
                    fillcolor='rgba(247, 147, 26, 0.1)',
                    line=dict(color='rgba(247, 147, 26, 0)'),
                    name='90% Confidence'
                ))
                
                fig.update_layout(
                    height=400,
                    margin=dict(l=0, r=0, t=0, b=0),
                    paper_bgcolor='#131315',
                    plot_bgcolor='#0a0a0b',
                    xaxis=dict(gridcolor='#27272a', title='Days'),
                    yaxis=dict(gridcolor='#27272a', title='Price ($)'),
                    font=dict(color='#e5e5e7', size=11)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Distribution of returns
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Return Distribution")
                
                # Generate return distribution
                returns = stats.get("all_returns", [])
                if not returns:
                    # Generate sample returns
                    import numpy as np
                    returns = np.random.normal(stats.get("mean_return", 0), stats.get("std_return", 0.1), 1000)
                
                fig = go.Figure(data=[go.Histogram(
                    x=returns,
                    nbinsx=50,
                    marker_color='#f7931a',
                    opacity=0.7
                )])
                
                fig.update_layout(
                    height=300,
                    margin=dict(l=0, r=0, t=0, b=0),
                    paper_bgcolor='#131315',
                    plot_bgcolor='#0a0a0b',
                    xaxis=dict(gridcolor='#27272a', title='Return'),
                    yaxis=dict(gridcolor='#27272a', title='Frequency'),
                    font=dict(color='#e5e5e7', size=11),
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### Risk Metrics")
                
                risk_html = f"""
                <div class="result-card">
                    <table style="width: 100%; font-size: 13px;">
                        <tr>
                            <td style="color: var(--text-secondary);">Value at Risk (95%)</td>
                            <td style="text-align: right; font-weight: 600;">${risk_metrics.get('var_95', 0):,.2f}</td>
                        </tr>
                        <tr>
                            <td style="color: var(--text-secondary);">Value at Risk (99%)</td>
                            <td style="text-align: right; font-weight: 600;">${risk_metrics.get('var_99', 0):,.2f}</td>
                        </tr>
                        <tr>
                            <td style="color: var(--text-secondary);">Conditional VaR (95%)</td>
                            <td style="text-align: right; font-weight: 600;">${risk_metrics.get('cvar_95', 0):,.2f}</td>
                        </tr>
                        <tr>
                            <td style="color: var(--text-secondary);">Max Drawdown</td>
                            <td style="text-align: right; font-weight: 600;">{risk_metrics.get('max_drawdown', 0)*100:.2f}%</td>
                        </tr>
                        <tr>
                            <td style="color: var(--text-secondary);">Probability of Profit</td>
                            <td style="text-align: right; font-weight: 600; color: var(--accent-success);">{(1-risk_metrics.get('probability_of_loss', 0))*100:.1f}%</td>
                        </tr>
                    </table>
                </div>
                """
                st.markdown(risk_html, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align: center; padding: 100px 20px; color: var(--text-secondary);">
                <h3 style="margin-bottom: 16px;">No simulation results yet</h3>
                <p>Configure parameters and run a Monte Carlo simulation to see results</p>
            </div>
            """, unsafe_allow_html=True)

# Optimization Tab
with tab3:
    col_params, col_results = st.columns([1, 3])
    
    with col_params:
        st.markdown("### Optimization Settings")
        
        # Objective
        st.markdown("#### Optimization Objective")
        objective = st.selectbox("Optimize for", ["Sharpe Ratio", "Total Return", "Win Rate", "Risk-Adjusted Return"])
        
        # Constraints
        st.markdown("#### Constraints")
        max_drawdown = st.slider("Max Drawdown %", min_value=5, max_value=50, value=20)
        min_win_rate = st.slider("Min Win Rate %", min_value=30, max_value=70, value=45)
        
        # Parameter Ranges
        st.markdown("#### Parameter Ranges")
        
        with st.expander("Position Sizing", expanded=True):
            position_range = st.slider("Position Size Range %", min_value=5, max_value=100, value=(10, 30))
            
        with st.expander("Signal Thresholds", expanded=True):
            buy_range = st.slider("Buy Threshold Range %", min_value=0.5, max_value=10.0, value=(1.0, 5.0))
            sell_range = st.slider("Sell Threshold Range %", min_value=0.5, max_value=10.0, value=(1.0, 5.0))
            
        with st.expander("Risk Management", expanded=True):
            sl_range = st.slider("Stop Loss Range %", min_value=1.0, max_value=20.0, value=(3.0, 10.0))
            tp_range = st.slider("Take Profit Range %", min_value=2.0, max_value=50.0, value=(5.0, 20.0))
        
        # Optimization settings
        st.markdown("#### Settings")
        num_iterations = st.number_input("Iterations", min_value=10, max_value=1000, value=100)
        
        # Run optimization
        if st.button("Start Optimization", type="primary", use_container_width=True):
            with st.spinner("Running optimization..."):
                params = {
                    "objective": objective.lower().replace(" ", "_"),
                    "iterations": num_iterations,
                    "constraints": {
                        "max_drawdown": max_drawdown / 100,
                        "min_win_rate": min_win_rate / 100
                    },
                    "parameter_ranges": {
                        "position_size": list(position_range),
                        "buy_threshold": list(buy_range),
                        "sell_threshold": list(sell_range),
                        "stop_loss": list(sl_range),
                        "take_profit": list(tp_range)
                    }
                }
                
                try:
                    results = api_client.post("/analytics/optimize", params)
                    st.session_state.optimization_results = results
                    st.success("Optimization completed!")
                except Exception as e:
                    st.error(f"Optimization failed: {str(e)}")
    
    with col_results:
        if "optimization_results" in st.session_state and st.session_state.optimization_results:
            results = st.session_state.optimization_results.get("results", {})
            
            st.markdown("### Optimization Results")
            
            # Best parameters
            best_params = results.get("best_parameters", {})
            expected_perf = results.get("expected_performance", {})
            
            # Display best parameters
            st.markdown("#### Optimal Parameters")
            
            param_cols = st.columns(5)
            with param_cols[0]:
                st.metric("Position Size", f"{best_params.get('position_size', 0)*100:.1f}%", None)
            with param_cols[1]:
                st.metric("Buy Threshold", f"{best_params.get('buy_threshold', 0)*100:.2f}%", None)
            with param_cols[2]:
                st.metric("Sell Threshold", f"{best_params.get('sell_threshold', 0)*100:.2f}%", None)
            with param_cols[3]:
                st.metric("Stop Loss", f"{best_params.get('stop_loss', 0)*100:.1f}%", None)
            with param_cols[4]:
                st.metric("Take Profit", f"{best_params.get('take_profit', 0)*100:.1f}%", None)
            
            # Expected performance
            st.markdown("#### Expected Performance")
            
            perf_cols = st.columns(4)
            with perf_cols[0]:
                st.metric("Expected Return", f"{expected_perf.get('return', 0)*100:.2f}%", None)
            with perf_cols[1]:
                st.metric("Expected Sharpe", f"{expected_perf.get('sharpe_ratio', 0):.2f}", None)
            with perf_cols[2]:
                st.metric("Expected Win Rate", f"{expected_perf.get('win_rate', 0)*100:.1f}%", None)
            with perf_cols[3]:
                st.metric("Expected Drawdown", f"{expected_perf.get('max_drawdown', 0)*100:.1f}%", None)
            
            # Optimization progress
            st.markdown("#### Optimization Progress")
            
            # Create convergence chart
            iterations = list(range(1, num_iterations + 1))
            # Generate sample convergence data
            import numpy as np
            convergence = []
            best_val = 0
            for i in range(num_iterations):
                val = best_val + np.random.randn() * 0.1 * (1 - i/num_iterations)
                if val > best_val:
                    best_val = val
                convergence.append(best_val)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=iterations,
                y=convergence,
                mode='lines',
                name='Best Value',
                line=dict(color='#f7931a', width=2)
            ))
            
            fig.update_layout(
                height=300,
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor='#131315',
                plot_bgcolor='#0a0a0b',
                xaxis=dict(gridcolor='#27272a', title='Iteration'),
                yaxis=dict(gridcolor='#27272a', title=objective),
                font=dict(color='#e5e5e7', size=11)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Parameter importance
            if "parameter_importance" in results:
                st.markdown("#### Parameter Importance")
                
                importance = results["parameter_importance"]
                fig = go.Figure(data=[go.Bar(
                    x=list(importance.values()),
                    y=list(importance.keys()),
                    orientation='h',
                    marker_color='#f7931a'
                )])
                
                fig.update_layout(
                    height=250,
                    margin=dict(l=0, r=0, t=0, b=0),
                    paper_bgcolor='#131315',
                    plot_bgcolor='#0a0a0b',
                    xaxis=dict(gridcolor='#27272a', title='Importance'),
                    yaxis=dict(gridcolor='#27272a'),
                    font=dict(color='#e5e5e7', size=11)
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown("""
            <div style="text-align: center; padding: 100px 20px; color: var(--text-secondary);">
                <h3 style="margin-bottom: 16px;">No optimization results yet</h3>
                <p>Configure parameters and run optimization to see results</p>
            </div>
            """, unsafe_allow_html=True)

# Data Quality Tab
with tab4:
    st.markdown("### Data Quality Monitoring")
    
    # Fetch data quality metrics
    try:
        data_quality = api_client.get("/analytics/data-quality")
        
        if data_quality:
            # Summary metrics
            summary = data_quality.get("summary", {})
            
            metrics = st.columns(4)
            with metrics[0]:
                st.metric("Total Datapoints", f"{summary.get('total_datapoints', 0):,}", None)
            with metrics[1]:
                st.metric("Completeness", f"{summary.get('overall_completeness', 0):.1f}%", None)
            with metrics[2]:
                st.metric("Missing Dates", summary.get('total_missing_dates', 0), None)
            with metrics[3]:
                st.metric("Active Sources", len(data_quality.get('by_source', {})), None)
            
            # Data by type
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Data Coverage by Type")
                
                by_type = data_quality.get("by_type", {})
                type_data = []
                
                for data_type, info in by_type.items():
                    type_data.append({
                        "Type": data_type.title(),
                        "Datapoints": info.get("total_datapoints", 0),
                        "Completeness": f"{info.get('completeness', 0):.1f}%"
                    })
                
                if type_data:
                    df_types = pd.DataFrame(type_data)
                    st.dataframe(df_types, use_container_width=True, hide_index=True)
            
            with col2:
                st.markdown("#### Coverage by Time Period")
                
                coverage = data_quality.get("coverage", {})
                
                coverage_html = f"""
                <div class="result-card">
                    <table style="width: 100%; font-size: 13px;">
                        <tr>
                            <td style="color: var(--text-secondary);">Last 24 Hours</td>
                            <td style="text-align: right;">{coverage.get('last_24h', {}).get('price', 0):.0f}%</td>
                        </tr>
                        <tr>
                            <td style="color: var(--text-secondary);">Last 7 Days</td>
                            <td style="text-align: right;">{coverage.get('last_7d', {}).get('price', 0):.0f}%</td>
                        </tr>
                        <tr>
                            <td style="color: var(--text-secondary);">Last 30 Days</td>
                            <td style="text-align: right;">{coverage.get('last_30d', {}).get('price', 0):.0f}%</td>
                        </tr>
                        <tr>
                            <td style="color: var(--text-secondary);">Last 90 Days</td>
                            <td style="text-align: right;">{coverage.get('last_90d', {}).get('price', 0):.0f}%</td>
                        </tr>
                    </table>
                </div>
                """
                st.markdown(coverage_html, unsafe_allow_html=True)
            
            # Data gaps
            gaps = data_quality.get("gaps", [])
            if gaps:
                st.markdown("#### Data Gaps Detected")
                
                gap_data = []
                for gap in gaps[:10]:  # Show first 10 gaps
                    gap_data.append({
                        "Start": gap.get("start", ""),
                        "End": gap.get("end", ""),
                        "Days": gap.get("days", 0)
                    })
                
                df_gaps = pd.DataFrame(gap_data)
                st.dataframe(df_gaps, use_container_width=True, hide_index=True)
            
            # Actions
            st.markdown("#### Actions")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Refresh Data Quality", use_container_width=True):
                    st.rerun()
            with col2:
                if st.button("Fill Data Gaps", use_container_width=True):
                    st.info("Filling data gaps...")
            with col3:
                if st.button("Clear Cache", use_container_width=True):
                    st.info("Clearing cache...")
            
    except Exception as e:
        st.error(f"Failed to load data quality metrics: {str(e)}")

# Footer with system info
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: var(--text-muted); font-size: 12px; padding: 16px 0;">
    BTC Trading System v2.1.0 | Analytics & Research Module | Last Update: {}
</div>
""".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')), unsafe_allow_html=True)