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
from components.charts import create_candlestick_chart
from utils.helpers import format_currency, format_percentage
from utils.constants import CHART_COLORS

# Set up logging
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Analytics", page_icon="🔬", layout="wide")

# Custom CSS for analytics
st.markdown("""
<style>
.analytics-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 20px;
    border-radius: 10px;
    color: white;
    margin-bottom: 20px;
}
.metric-card {
    background: #f0f2f6;
    padding: 20px;
    border-radius: 10px;
    margin: 10px 0;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}
.result-card {
    background: #ffffff;
    padding: 15px;
    border-radius: 8px;
    border-left: 4px solid #667eea;
    margin: 10px 0;
}
.feature-card {
    background: #f8f9fa;
    padding: 10px;
    border-radius: 5px;
    margin: 5px 0;
}
.regime-indicator {
    padding: 10px 20px;
    border-radius: 20px;
    text-align: center;
    font-weight: bold;
    color: white;
}
.regime-bullish { background: #00ff88; }
.regime-bearish { background: #ff3366; }
.regime-neutral { background: #8b92a8; }
</style>
""", unsafe_allow_html=True)

# Initialize API client
@st.cache_resource
def get_api_client():
    return APIClient(os.getenv("API_BASE_URL", "http://backend:8000"))

api_client = get_api_client()

def create_metrics_chart(metrics: dict) -> go.Figure:
    """Create a simple metrics visualization"""
    fig = go.Figure()
    
    # Extract key metrics
    metric_names = []
    metric_values = []
    
    # Common metric mappings
    metric_map = {
        'sortino_ratio_mean': 'Sortino Ratio',
        'sharpe_ratio_mean': 'Sharpe Ratio',
        'total_return_mean': 'Total Return (%)',
        'win_rate_mean': 'Win Rate (%)',
        'max_drawdown_mean': 'Max Drawdown (%)',
        'calmar_ratio_mean': 'Calmar Ratio'
    }
    
    for key, display_name in metric_map.items():
        if key in metrics:
            metric_names.append(display_name)
            value = metrics[key]
            if 'return' in key or 'rate' in key or 'drawdown' in key:
                value *= 100  # Convert to percentage
            metric_values.append(value)
    
    if metric_names:
        fig.add_trace(go.Bar(
            x=metric_names,
            y=metric_values,
            marker_color=['green' if v > 0 else 'red' for v in metric_values],
            text=[f'{v:.2f}' for v in metric_values],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Backtest Performance Metrics",
            xaxis_title="Metric",
            yaxis_title="Value",
            height=400,
            template="plotly_white"
        )
    else:
        fig.add_annotation(text="No metrics available", 
                         xref="paper", yref="paper", 
                         x=0.5, y=0.5, showarrow=False)
    
    return fig

def create_backtest_chart(results: dict) -> go.Figure:
    """Create comprehensive backtest results visualization"""
    if not results:
        return go.Figure().add_annotation(text="No backtest results available", 
                                        xref="paper", yref="paper", 
                                        x=0.5, y=0.5, showarrow=False)
    
    # For the actual API response structure, create a multi-chart visualization
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Performance Metrics", "Feature Importance (Top 10)", 
                       "Risk Decomposition", "Signal Activations"),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "pie"}, {"type": "bar"}]],
        vertical_spacing=0.15,
        horizontal_spacing=0.15
    )
    
    # 1. Performance Metrics Bar Chart
    metric_names = []
    metric_values = []
    metric_colors = []
    
    # Extract key metrics from actual API response structure
    perf_metrics = results.get('performance_metrics', {})
    trading_stats = results.get('trading_statistics', {})
    
    # Add metrics from performance_metrics
    if perf_metrics.get('total_return_mean') is not None:
        metric_names.append('Total Return %')
        metric_values.append(perf_metrics['total_return_mean'] * 100)
        metric_colors.append('green' if perf_metrics['total_return_mean'] > 0 else 'red')
    
    if perf_metrics.get('win_rate_mean') is not None:
        metric_names.append('Win Rate %')
        metric_values.append(perf_metrics['win_rate_mean'] * 100)
        metric_colors.append('blue')
    
    if perf_metrics.get('sharpe_ratio_mean') is not None:
        metric_names.append('Sharpe Ratio')
        metric_values.append(perf_metrics['sharpe_ratio_mean'])
        metric_colors.append('purple')
    
    if perf_metrics.get('max_drawdown_mean') is not None:
        metric_names.append('Max Drawdown %')
        metric_values.append(perf_metrics['max_drawdown_mean'] * 100)
        metric_colors.append('red')
    
    if metric_names:
        fig.add_trace(go.Bar(
            x=metric_names,
            y=metric_values,
            marker_color=metric_colors,
            text=[f'{v:.2f}' for v in metric_values],
            textposition='auto',
            name='Metrics'
        ), row=1, col=1)
    
    # 2. Feature Importance (Top 10)
    if 'feature_analysis' in results and 'feature_importance' in results['feature_analysis']:
        features = results['feature_analysis']['feature_importance']
        # Filter and convert values to ensure they're numeric
        numeric_features = {}
        for k, v in features.items():
            try:
                numeric_features[k] = float(v)
            except (TypeError, ValueError):
                continue
        # Sort and get top 10 by absolute value
        sorted_features = sorted(numeric_features.items(), 
                               key=lambda x: abs(x[1]), 
                               reverse=True)[:10]
        
        if sorted_features:
            feature_names = [f[0] for f in sorted_features]
            importance_values = [f[1] for f in sorted_features]
            
            fig.add_trace(go.Bar(
                y=feature_names,
                x=importance_values,
                orientation='h',
                marker_color=['green' if v > 0 else 'red' for v in importance_values],
                text=[f'{v:.4f}' for v in importance_values],
                textposition='auto',
                name='Feature Importance'
            ), row=1, col=2)
    
    # 3. Risk Decomposition Pie Chart
    if 'risk_analysis' in results and 'decomposition' in results['risk_analysis']:
        risk_data = results['risk_analysis']['decomposition']
        
        risk_labels = []
        risk_values = []
        
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
                hole=0.3,
                marker_colors=['#FF6B6B', '#4ECDC4', '#45B7D1'],
                textinfo='label+percent',
                name='Risk Decomposition'
            ), row=2, col=1)
    
    # 4. Signal Performance or Top Contributing Signals
    if 'performance_metrics' in results and 'top_contributing_signals' in results['performance_metrics']:
        signals = results['performance_metrics']['top_contributing_signals']
        if signals:
            signal_names = []
            signal_contributions = []
            
            for sig_name, sig_data in signals.items():
                signal_names.append(sig_name)
                signal_contributions.append(sig_data.get('total_contribution', 0) * 100)
            
            # Sort by absolute contribution
            sorted_pairs = sorted(zip(signal_names, signal_contributions), 
                                key=lambda x: abs(x[1]), reverse=True)[:10]
            
            if sorted_pairs:
                signal_names, signal_contributions = zip(*sorted_pairs)
                
                fig.add_trace(go.Bar(
                    x=list(signal_names),
                    y=list(signal_contributions),
                    marker_color=['green' if v > 0 else 'red' for v in signal_contributions],
                    text=[f'{v:.2f}%' for v in signal_contributions],
                    textposition='auto',
                    name='Signal Contributions'
                ), row=2, col=2)
    
    # Update layout
    total_return = perf_metrics.get('total_return_mean', 0) * 100 if 'performance_metrics' in results else 0
    fig.update_layout(
        title=f"Backtest Results - Total Return: {total_return:.2f}%",
        height=800,
        showlegend=False,
        template="plotly_white"
    )
    
    # Update axes
    fig.update_xaxes(title_text="Metric", row=1, col=1)
    fig.update_yaxes(title_text="Value", row=1, col=1)
    
    fig.update_xaxes(title_text="Importance", row=1, col=2)
    fig.update_yaxes(title_text="Feature", row=1, col=2)
    
    fig.update_xaxes(title_text="Signal Type", row=2, col=2)
    fig.update_yaxes(title_text="Count", row=2, col=2)
    
    return fig

def create_monte_carlo_chart(results: dict) -> go.Figure:
    """Create Monte Carlo simulation visualization"""
    if not results or 'simulations' not in results:
        return go.Figure().add_annotation(text="No Monte Carlo results available", 
                                        xref="paper", yref="paper", 
                                        x=0.5, y=0.5, showarrow=False)
    
    simulations = results['simulations']
    percentiles = results.get('percentiles', {})
    
    fig = go.Figure()
    
    # Check if simulations is a list of paths or just a number
    if isinstance(simulations, (int, float)):
        # If it's just a number, we can't plot paths
        return go.Figure().add_annotation(
            text=f"Monte Carlo simulation completed with {simulations} simulations", 
            xref="paper", yref="paper", 
            x=0.5, y=0.5, showarrow=False)
    
    # Plot sample paths (limit to 100 for performance)
    if isinstance(simulations, list) and len(simulations) > 0:
        sample_paths = simulations[:min(100, len(simulations))]
        for i, path in enumerate(sample_paths):
            fig.add_trace(go.Scatter(
                y=path,
                mode='lines',
                line=dict(width=0.5, color='lightgray'),
                showlegend=False,
                opacity=0.3
            ))
    
    # Plot percentiles
    if percentiles:
        # Median (50th percentile)
        if 'p50' in percentiles:
            fig.add_trace(go.Scatter(
                y=percentiles['p50'],
                mode='lines',
                name='Median',
                line=dict(width=3, color='blue')
            ))
        
        # Confidence bands
        if 'p95' in percentiles and 'p5' in percentiles:
            x_range = list(range(len(percentiles['p95'])))
            
            fig.add_trace(go.Scatter(
                x=x_range + x_range[::-1],
                y=percentiles['p95'] + percentiles['p5'][::-1],
                fill='toself',
                fillcolor='rgba(102, 126, 234, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False,
                name='90% Confidence'
            ))
            
            fig.add_trace(go.Scatter(
                y=percentiles['p95'],
                mode='lines',
                name='95th Percentile',
                line=dict(width=2, color='green', dash='dash')
            ))
            
            fig.add_trace(go.Scatter(
                y=percentiles['p5'],
                mode='lines',
                name='5th Percentile',
                line=dict(width=2, color='red', dash='dash')
            ))
    
    fig.update_layout(
        title="Monte Carlo Simulation - Portfolio Paths",
        xaxis_title="Days",
        yaxis_title="Portfolio Value ($)",
        height=600,
        template="plotly_white",
        hovermode='x unified'
    )
    
    return fig

def create_feature_importance_chart(features: dict) -> go.Figure:
    """Create feature importance visualization"""
    if not features:
        return go.Figure().add_annotation(text="No feature importance data available", 
                                        xref="paper", yref="paper", 
                                        x=0.5, y=0.5, showarrow=False)
    
    # Filter and convert values to ensure they're numeric
    numeric_features = {}
    for k, v in features.items():
        try:
            numeric_features[k] = float(v)
        except (TypeError, ValueError):
            # Skip non-numeric values
            continue
    
    if not numeric_features:
        return go.Figure().add_annotation(text="No valid numeric feature importance data available", 
                                        xref="paper", yref="paper", 
                                        x=0.5, y=0.5, showarrow=False)
    
    # Sort features by absolute importance value
    sorted_features = sorted(numeric_features.items(), key=lambda x: abs(x[1]), reverse=True)[:20]
    
    feature_names = [f[0] for f in sorted_features]
    importance_values = [f[1] for f in sorted_features]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=importance_values,
        y=feature_names,
        orientation='h',
        marker=dict(
            color=['green' if v > 0 else 'red' for v in importance_values],
            line=dict(color='rgba(0,0,0,0.3)', width=1)
        ),
        text=[f'{v:.4f}' for v in importance_values],
        textposition='auto'
    ))
    
    fig.update_layout(
        title="Top 20 Feature Importance (by Absolute Value)",
        xaxis_title="Importance Score",
        yaxis_title="Feature",
        height=600,
        template="plotly_white",
        yaxis=dict(autorange="reversed")
    )
    
    return fig

def create_stress_test_chart(stress_scenarios: dict) -> go.Figure:
    """Create stress test scenarios visualization"""
    if not stress_scenarios:
        return go.Figure().add_annotation(text="No stress test data available", 
                                        xref="paper", yref="paper", 
                                        x=0.5, y=0.5, showarrow=False)
    
    scenarios = []
    impacts = []
    max_dds = []
    
    scenario_map = {
        'flash_crash': 'Flash Crash',
        'bull_rally': 'Bull Rally',
        'high_volatility': 'High Volatility',
        'low_volatility': 'Low Volatility'
    }
    
    for key, display_name in scenario_map.items():
        if key in stress_scenarios:
            scenarios.append(display_name)
            impacts.append(stress_scenarios[key].get('impact', 0) * 100)
            max_dds.append(stress_scenarios[key].get('max_drawdown', 0) * 100)
    
    fig = go.Figure()
    
    # Impact bars
    fig.add_trace(go.Bar(
        name='Portfolio Impact (%)',
        x=scenarios,
        y=impacts,
        marker_color=['red' if v < 0 else 'green' for v in impacts],
        text=[f'{v:.1f}%' for v in impacts],
        textposition='auto'
    ))
    
    # Max drawdown bars
    fig.add_trace(go.Bar(
        name='Max Drawdown (%)',
        x=scenarios,
        y=max_dds,
        marker_color=['red' if v < 0 else 'lightblue' for v in max_dds],
        text=[f'{v:.1f}%' for v in max_dds],
        textposition='auto'
    ))
    
    fig.update_layout(
        title="Stress Test Scenarios",
        xaxis_title="Scenario",
        yaxis_title="Impact (%)",
        barmode='group',
        height=400,
        template="plotly_white",
        legend=dict(x=0.7, y=1)
    )
    
    return fig

def show_analytics():
    """Main analytics interface"""
    
    st.title("🔬 Advanced Analytics")
    
    # Header
    st.markdown("""
    <div class="analytics-header">
        <h2>Strategy Analysis & Optimization</h2>
        <p>Backtest strategies, run simulations, and analyze feature importance</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Backtesting",
        "🎲 Monte Carlo",
        "📈 Feature Importance",
        "🌡️ Market Regime",
        "🔍 Strategy Optimization"
    ])
    
    with tab1:
        st.subheader("Strategy Backtesting")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.info("Test your trading strategy on historical data with walk-forward analysis")
            
            # Backtest parameters
            with st.expander("⚙️ Backtest Configuration", expanded=True):
                col_param1, col_param2, col_param3 = st.columns(3)
                
                with col_param1:
                    period_days = st.number_input("Test Period (days)", 
                                                min_value=30, 
                                                max_value=730, 
                                                value=180)
                    initial_capital = st.number_input("Initial Capital ($)", 
                                                    min_value=1000, 
                                                    max_value=1000000, 
                                                    value=10000)
                
                with col_param2:
                    strategy_type = st.selectbox("Strategy Type", 
                                               ["AI Signals", "Technical Only", 
                                                "Sentiment Based", "Custom"])
                    position_sizing = st.selectbox("Position Sizing", 
                                                 ["Fixed", "Kelly Criterion", 
                                                  "Risk Parity", "Volatility Adjusted"])
                
                with col_param3:
                    max_positions = st.number_input("Max Positions", 
                                                  min_value=1, 
                                                  max_value=10, 
                                                  value=3)
                    stop_loss = st.number_input("Stop Loss %", 
                                              min_value=0.0, 
                                              max_value=50.0, 
                                              value=5.0)
                
                # Trading Parameters
                st.markdown("##### Trading Parameters")
                col_trade1, col_trade2, col_trade3 = st.columns(3)
                
                with col_trade1:
                    position_size_pct = st.number_input("Position Size %", 
                                                       min_value=1.0, 
                                                       max_value=100.0, 
                                                       value=10.0,
                                                       help="Percentage of capital to use per trade")
                    buy_threshold = st.number_input("Buy Threshold %", 
                                                   min_value=0.1, 
                                                   max_value=10.0, 
                                                   value=2.0,
                                                   step=0.1,
                                                   help="Model prediction threshold to trigger buy signal")
                
                with col_trade2:
                    sell_threshold = st.number_input("Sell Threshold %", 
                                                    min_value=0.1, 
                                                    max_value=10.0, 
                                                    value=2.0,
                                                    step=0.1,
                                                    help="Model prediction threshold to trigger sell signal")
                    sell_percentage = st.number_input("Sell Percentage %", 
                                                     min_value=10.0, 
                                                     max_value=100.0, 
                                                     value=50.0,
                                                     help="Percentage of holdings to sell on signal")
                
                with col_trade3:
                    take_profit = st.number_input("Take Profit %", 
                                                 min_value=0.0, 
                                                 max_value=100.0, 
                                                 value=10.0,
                                                 help="Profit target for partial position exit")
                    transaction_cost = st.number_input("Transaction Cost %", 
                                                      min_value=0.0, 
                                                      max_value=1.0, 
                                                      value=0.25,
                                                      step=0.01,
                                                      help="Trading fee percentage per transaction")
                
                # Advanced options
                use_walk_forward = st.checkbox("Use Walk-Forward Analysis", value=True, key="walk_forward")
                include_transaction_costs = st.checkbox("Include Transaction Costs", value=True, key="transaction_costs")
                optimize_weights = st.checkbox("Optimize Signal Weights", value=False, key="optimize_weights")
                include_macro = st.checkbox("Include Macro Indicators", value=True, key="include_macro")
                
            # Run backtest button
            if st.button("🚀 Run Backtest", type="primary", use_container_width=True):
                # Clear previous results to ensure fresh state
                if 'backtest_results' in st.session_state:
                    del st.session_state['backtest_results']
                
                with st.spinner("Running backtest... This may take a few minutes"):
                    # Prepare backtest parameters for enhanced endpoint
                    params = {
                        "symbol": "BTC-USD",
                        "days": period_days,
                        "initial_capital": initial_capital,
                        "settings": {
                            "strategy": strategy_type.lower().replace(" ", "_"),
                            "position_sizing": position_sizing.lower().replace(" ", "_"),
                            "max_positions": max_positions,
                            "stop_loss": stop_loss / 100,
                            "walk_forward": st.session_state.get('walk_forward', True),
                            "transaction_costs": 0.0025 if st.session_state.get('transaction_costs', True) else 0
                        }
                    }
                    
                    # Create enhanced backtest parameters
                    # Map period to the correct format
                    period_map = {
                        30: "1mo",
                        90: "3mo",
                        180: "6mo",
                        365: "1y",
                        730: "2y"
                    }
                    period_str = period_map.get(period_days, "1y")
                    
                    enhanced_params = {
                        "period": period_str,
                        "optimize_weights": st.session_state.get('optimize_weights', False),
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
                    
                    # Use enhanced backtest endpoint for comprehensive results
                    result = api_client.post("/backtest/enhanced", enhanced_params)
                    
                    if result:
                        if 'error' in result:
                            error_msg = result.get('error', 'Unknown error')
                            st.error(f"❌ Backtest failed: {error_msg}")
                        elif 'status' in result and result['status'] == 'error':
                            error_msg = result.get('message', 'Unknown error')
                            st.error(f"❌ Backtest failed: {error_msg}")
                        else:
                            st.success("✅ Backtest completed successfully!")
                            st.session_state['backtest_results'] = result
                            # Force a rerun to display new results
                            st.rerun()
                    else:
                        st.error("❌ Connection error: Failed to reach backend API")
        
        with col2:
            # Quick stats if results available
            if 'backtest_results' in st.session_state:
                results = st.session_state['backtest_results']
                
                st.markdown("### 📊 Quick Stats")
                
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                
                col_m1, col_m2 = st.columns(2)
                
                # Handle the actual API response format - metrics are in performance_metrics
                perf_metrics = results.get('performance_metrics', {})
                trading_stats = results.get('trading_statistics', {})
                with col_m1:
                    st.metric("Total Return", 
                            f"{perf_metrics.get('total_return_mean', 0)*100:.2f}%")
                    st.metric("Win Rate", 
                            f"{perf_metrics.get('win_rate_mean', 0)*100:.1f}%")
                with col_m2:
                    st.metric("Total Trades", 
                            f"{trading_stats.get('total_trades', 0)}")
                    st.metric("Sharpe Ratio", 
                            f"{perf_metrics.get('sharpe_ratio_mean', 0):.2f}")
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Display results
        if 'backtest_results' in st.session_state:
            results = st.session_state['backtest_results']
            # Extract metrics from actual API response structure
            perf_metrics = results.get('performance_metrics', {})
            trading_stats = results.get('trading_statistics', {})
            risk_metrics = results.get('risk_metrics', {})
            
            # Main visualization
            st.plotly_chart(create_backtest_chart(results), use_container_width=True)
            
            # Additional visualizations in tabs
            detail_tab1, detail_tab2, detail_tab3, detail_tab4 = st.tabs([
                "📊 Detailed Metrics", "🔬 Feature Analysis", "⚠️ Risk Analysis", "📋 Recommendations"
            ])
            
            with detail_tab1:
                # Detailed metrics display
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown("**Returns**")
                    st.write(f"Total Return: {perf_metrics.get('total_return_mean', 0)*100:.2f}%")
                    st.write(f"Sharpe Ratio: {perf_metrics.get('sharpe_ratio_mean', 0):.2f}")
                    st.write(f"Sortino Ratio: {perf_metrics.get('sortino_ratio_mean', 0):.2f}")
                
                with col2:
                    st.markdown("**Trading Stats**")
                    st.write(f"Total Trades: {trading_stats.get('total_trades', 0)}")
                    st.write(f"Win Rate: {perf_metrics.get('win_rate_mean', 0)*100:.1f}%")
                    st.write(f"Profit Factor: {perf_metrics.get('profit_factor_mean', 0):.2f}")
                
                with col3:
                    st.markdown("**Risk Metrics**")
                    st.write(f"Max Drawdown: {perf_metrics.get('max_drawdown_mean', 0)*100:.2f}%")
                    st.write(f"VaR (95%): {risk_metrics.get('var_95', 0)*100:.2f}%")
                    st.write(f"CVaR (95%): {risk_metrics.get('cvar_95', 0)*100:.2f}%")
                
                with col4:
                    st.markdown("**Positions**")
                    st.write(f"Long: {trading_stats.get('long_positions', 0)}")
                    st.write(f"Short: {trading_stats.get('short_positions', 0)}")
                    st.write(f"Turnover: {trading_stats.get('avg_position_turnover', 0)*100:.2f}%")
                
                # Market analysis if available
                if 'market_analysis' in results:
                    st.markdown("---")
                    st.markdown("**Market Analysis**")
                    market = results['market_analysis']
                    
                    col_m1, col_m2, col_m3 = st.columns(3)
                    with col_m1:
                        st.info(f"**Regime:** {market.get('regime', 'Unknown')}")
                    with col_m2:
                        st.info(f"**Trend:** {market.get('dominant_trend', 'Unknown')}")
                    with col_m3:
                        st.info(f"**Volatility:** {market.get('volatility_regime', 'Unknown')}")
                
                # Optimal weights if available
                if 'optimal_weights' in results:
                    st.markdown("---")
                    st.markdown("**Optimal Signal Weights**")
                    weights = results['optimal_weights']
                    
                    weight_df = pd.DataFrame(list(weights.items()), columns=['Signal Type', 'Weight'])
                    fig_weights = go.Figure(go.Pie(
                        labels=weight_df['Signal Type'],
                        values=weight_df['Weight'],
                        hole=0.3
                    ))
                    fig_weights.update_layout(height=300, title="Optimal Weight Distribution")
                    st.plotly_chart(fig_weights, use_container_width=True)
            
            with detail_tab2:
                # Feature analysis
                if 'feature_analysis' in results:
                    feature_data = results['feature_analysis']
                    
                    # Feature importance chart
                    if 'feature_importance' in feature_data:
                        st.subheader("Feature Importance Analysis")
                        fig_features = create_feature_importance_chart(feature_data['feature_importance'])
                        st.plotly_chart(fig_features, use_container_width=True)
                    
                    # Top correlations
                    if 'top_correlations' in feature_data:
                        st.subheader("Top Feature Correlations with Returns")
                        corr_df = pd.DataFrame(
                            list(feature_data['top_correlations'].items()),
                            columns=['Feature', 'Correlation']
                        )
                        corr_df = corr_df.sort_values('Correlation', key=abs, ascending=False)
                        
                        st.dataframe(
                            corr_df.style.format({'Correlation': '{:.3f}'})
                            .background_gradient(cmap='RdYlGn', vmin=-1, vmax=1, subset=['Correlation']),
                            use_container_width=True
                        )
                else:
                    st.info("No feature analysis data available")
            
            with detail_tab3:
                # Risk analysis
                if 'risk_analysis' in results:
                    risk_data = results['risk_analysis']
                    
                    # Risk decomposition
                    if 'decomposition' in risk_data:
                        st.subheader("Risk Decomposition")
                        decomp = risk_data['decomposition']
                        
                        col_r1, col_r2 = st.columns([1, 2])
                        with col_r1:
                            st.metric("Total Risk", f"{decomp.get('total_risk', 0)*100:.2f}%")
                            st.metric("Market Risk", f"{decomp.get('market_risk', 0)*100:.2f}%")
                            st.metric("Specific Risk", f"{decomp.get('specific_risk', 0)*100:.2f}%")
                            st.metric("Model Risk", f"{decomp.get('model_risk', 0)*100:.2f}%")
                        
                        with col_r2:
                            # Risk pie chart
                            risk_labels = ['Market Risk', 'Specific Risk', 'Model Risk']
                            risk_values = [
                                decomp.get('market_risk', 0),
                                decomp.get('specific_risk', 0),
                                decomp.get('model_risk', 0)
                            ]
                            
                            fig_risk = go.Figure(go.Pie(
                                labels=risk_labels,
                                values=risk_values,
                                hole=0.4,
                                marker_colors=['#FF6B6B', '#4ECDC4', '#45B7D1']
                            ))
                            fig_risk.update_layout(height=300, title="Risk Components")
                            st.plotly_chart(fig_risk, use_container_width=True)
                    
                    # Stress scenarios
                    if 'stress_scenarios' in risk_data:
                        st.subheader("Stress Test Scenarios")
                        fig_stress = create_stress_test_chart(risk_data['stress_scenarios'])
                        st.plotly_chart(fig_stress, use_container_width=True)
                else:
                    st.info("No risk analysis data available")
            
            with detail_tab4:
                # Recommendations
                if 'recommendations' in results:
                    st.subheader("Strategy Recommendations")
                    for i, rec in enumerate(results['recommendations'], 1):
                        st.write(f"{i}. {rec}")
                else:
                    st.info("No recommendations available")
                
                # Signal analysis
                if 'signal_analysis' in results:
                    st.markdown("---")
                    st.subheader("Signal Analysis")
                    
                    signal_data = results['signal_analysis']
                    if 'effectiveness' in signal_data:
                        st.write(f"**Status:** {signal_data['effectiveness'].get('status', 'Unknown')}")
                    
                    if 'optimal_combinations' in signal_data and signal_data['optimal_combinations']:
                        st.write("**Optimal Signal Combinations:**")
                        for combo in signal_data['optimal_combinations']:
                            st.write(f"- {combo}")
        
        # Historical Backtest Results Section
        st.markdown("---")
        st.subheader("📜 Historical Backtest Results")
        
        col_hist1, col_hist2 = st.columns([3, 1])
        
        with col_hist1:
            # Get backtest history
            history = api_client.get_backtest_history(limit=20)
            
            if history and len(history) > 0:
                # Create history dataframe
                history_df = pd.DataFrame(history)
                
                # Format the dataframe for display
                display_df = history_df[['timestamp', 'composite_score', 'sortino_ratio', 
                                       'max_drawdown', 'confidence_score']].copy()
                
                # Handle timestamp conversion with error handling
                try:
                    display_df['timestamp'] = pd.to_datetime(display_df['timestamp'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M')
                    # Replace any NaT values with a fallback
                    display_df['timestamp'] = display_df['timestamp'].fillna('Unknown')
                except Exception as e:
                    logger.error(f"Error converting timestamps: {e}")
                    display_df['timestamp'] = display_df['timestamp'].astype(str)
                
                display_df.columns = ['Date', 'Composite Score', 'Sortino Ratio', 'Max Drawdown', 'Confidence']
                
                # Display as interactive table
                st.dataframe(
                    display_df.style.format({
                        'Composite Score': '{:.3f}',
                        'Sortino Ratio': '{:.3f}',
                        'Max Drawdown': '{:.3f}',
                        'Confidence': '{:.3f}'
                    }),
                    use_container_width=True,
                    height=400
                )
                
                # Option to load a specific backtest
                if st.button("📊 Load Selected Backtest Results"):
                    selected_idx = st.selectbox("Select backtest to load", 
                                              range(len(history)), 
                                              format_func=lambda x: f"{history[x]['timestamp']} - Score: {history[x]['composite_score']:.3f}")
                    
                    if selected_idx is not None:
                        # Load the full results for the selected backtest
                        st.session_state['backtest_results'] = history[selected_idx]
                        st.success("✅ Loaded historical backtest results")
                        st.rerun()
            else:
                st.info("No historical backtest results available")
        
        with col_hist2:
            # Summary statistics
            if history and len(history) > 0:
                st.markdown("### 📊 History Stats")
                
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                
                avg_score = np.mean([h.get('composite_score', 0) for h in history])
                best_score = max([h.get('composite_score', 0) for h in history])
                avg_sortino = np.mean([h.get('sortino_ratio', 0) for h in history])
                
                st.metric("Avg Score", f"{avg_score:.3f}")
                st.metric("Best Score", f"{best_score:.3f}")
                st.metric("Avg Sortino", f"{avg_sortino:.3f}")
                st.metric("Total Runs", len(history))
                
                st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.subheader("Monte Carlo Risk Analysis")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.info("Simulate thousands of possible portfolio paths to assess risk")
            
            # Simulation parameters
            with st.expander("⚙️ Simulation Parameters", expanded=True):
                col_sim1, col_sim2, col_sim3 = st.columns(3)
                
                with col_sim1:
                    num_simulations = st.number_input("Number of Simulations", 
                                                    min_value=100, 
                                                    max_value=10000, 
                                                    value=1000)
                    time_horizon = st.number_input("Time Horizon (days)", 
                                                 min_value=30, 
                                                 max_value=365, 
                                                 value=90)
                
                with col_sim2:
                    confidence_level = st.slider("Confidence Level", 
                                               min_value=80, 
                                               max_value=99, 
                                               value=95)
                    use_historical = st.checkbox("Use Historical Data", value=True)
                
                with col_sim3:
                    volatility_regime = st.selectbox("Volatility Regime", 
                                                   ["Current", "High", "Low", "Custom"])
                    if volatility_regime == "Custom":
                        custom_vol = st.number_input("Annual Volatility %", 
                                                   min_value=10, 
                                                   max_value=200, 
                                                   value=50)
            
            # Run simulation button
            if st.button("🎲 Run Monte Carlo Simulation", type="primary", use_container_width=True):
                with st.spinner("Running simulations..."):
                    # Prepare simulation parameters
                    params = {
                        "num_simulations": num_simulations,
                        "time_horizon": time_horizon,
                        "confidence_level": confidence_level,
                        "use_historical": use_historical,
                        "volatility_regime": volatility_regime.lower()
                    }
                    
                    if volatility_regime == "Custom":
                        params["custom_volatility"] = custom_vol / 100
                    
                    # Run simulation
                    result = api_client.post("/analytics/monte-carlo", params)
                    
                    if result and result.get('status') == 'success':
                        st.success("✅ Simulation completed!")
                        st.session_state['monte_carlo_results'] = result.get('results', {})
                    else:
                        error_msg = result.get('error', 'Unknown error') if result else 'Connection error'
                        st.error(f"❌ Simulation failed: {error_msg}")
        
        with col2:
            # Risk metrics if available
            if 'monte_carlo_results' in st.session_state:
                mc_results = st.session_state['monte_carlo_results']
                risk_metrics = mc_results.get('risk_metrics', {})
                
                st.markdown("### 🎯 Risk Metrics")
                
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                
                st.metric("VaR (95%)", 
                        f"${risk_metrics.get('var_95', 0):,.0f}")
                st.metric("CVaR (95%)", 
                        f"${risk_metrics.get('cvar_95', 0):,.0f}")
                st.metric("Prob. of Loss", 
                        f"{risk_metrics.get('prob_loss', 0):.1%}")
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Display simulation results
        if 'monte_carlo_results' in st.session_state:
            mc_results = st.session_state['monte_carlo_results']
            
            # Main visualization
            st.plotly_chart(create_monte_carlo_chart(mc_results), use_container_width=True)
            
            # Statistical summary
            with st.expander("📊 Statistical Summary", expanded=True):
                stats = mc_results.get('statistics', {})
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Expected Outcomes**")
                    st.write(f"Mean Return: {stats.get('mean_return', 0):.2%}")
                    st.write(f"Median Return: {stats.get('median_return', 0):.2%}")
                    st.write(f"Std Deviation: {stats.get('std_dev', 0):.2%}")
                
                with col2:
                    st.markdown("**Percentiles**")
                    st.write(f"5th: ${stats.get('p5', 0):,.0f}")
                    st.write(f"25th: ${stats.get('p25', 0):,.0f}")
                    st.write(f"75th: ${stats.get('p75', 0):,.0f}")
                    st.write(f"95th: ${stats.get('p95', 0):,.0f}")
                
                with col3:
                    st.markdown("**Risk Analysis**")
                    st.write(f"Max Loss: ${stats.get('max_loss', 0):,.0f}")
                    st.write(f"Max Gain: ${stats.get('max_gain', 0):,.0f}")
                    st.write(f"Skewness: {stats.get('skewness', 0):.2f}")
                    st.write(f"Kurtosis: {stats.get('kurtosis', 0):.2f}")
            
            # Scenario analysis
            if 'scenarios' in mc_results:
                with st.expander("🎭 Scenario Analysis", expanded=False):
                    scenarios = mc_results['scenarios']
                    
                    scenario_df = pd.DataFrame({
                        'Scenario': scenarios.keys(),
                        'Probability': [s.get('probability', 0) for s in scenarios.values()],
                        'Expected Return': [s.get('return', 0) for s in scenarios.values()],
                        'Impact': [s.get('impact', 'N/A') for s in scenarios.values()]
                    })
                    
                    st.dataframe(scenario_df.style.format({
                        'Probability': '{:.1%}',
                        'Expected Return': '{:.2%}'
                    }))
    
    with tab3:
        st.subheader("Feature Importance Analysis")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.info("Understand which indicators drive the AI model's predictions")
            
            # Analysis options
            with st.expander("⚙️ Analysis Options", expanded=True):
                col_opt1, col_opt2 = st.columns(2)
                
                with col_opt1:
                    analysis_type = st.selectbox("Analysis Method", 
                                               ["SHAP Values", "Permutation Importance", 
                                                "Correlation Analysis", "Mutual Information"])
                    time_period = st.selectbox("Time Period", 
                                             ["Last 30 days", "Last 90 days", 
                                              "Last 180 days", "All time"])
                
                with col_opt2:
                    feature_category = st.multiselect("Feature Categories", 
                                                    ["Technical", "On-chain", "Sentiment", "Macro"],
                                                    default=["Technical", "On-chain"])
                    min_importance = st.slider("Min Importance Threshold", 
                                             min_value=0.0, 
                                             max_value=0.5, 
                                             value=0.05)
            
            # Run analysis button
            if st.button("📊 Analyze Feature Importance", type="primary", use_container_width=True):
                with st.spinner("Analyzing features..."):
                    # Prepare parameters
                    params = {
                        "method": analysis_type.lower().replace(" ", "_"),
                        "period": time_period.lower().replace(" ", "_"),
                        "categories": feature_category,
                        "threshold": min_importance
                    }
                    
                    # Get feature importance
                    result = api_client.get("/analytics/feature-importance", params)
                    
                    if result and 'feature_importance' in result:
                        st.success("✅ Analysis completed!")
                        st.session_state['feature_importance'] = result.get('feature_importance', {})
                    else:
                        # Fallback to basic feature importance
                        result = api_client.get("/ml/feature-importance")
                        if result and 'features' in result:
                            # Convert list of feature objects to dict
                            features_dict = {}
                            for item in result['features']:
                                if 'feature' in item and 'importance' in item:
                                    try:
                                        # Ensure importance is numeric
                                        features_dict[item['feature']] = float(item['importance'])
                                    except (TypeError, ValueError):
                                        # Skip items with non-numeric importance
                                        continue
                            st.session_state['feature_importance'] = features_dict
                        else:
                            st.error("❌ Analysis failed")
        
        with col2:
            # Top features preview
            if 'feature_importance' in st.session_state:
                features = st.session_state['feature_importance']
                
                st.markdown("### 🏆 Top Features")
                
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                
                # Show top 5 features
                # Filter and convert values to ensure they're numeric
                numeric_features = {}
                for k, v in features.items():
                    try:
                        numeric_features[k] = float(v)
                    except (TypeError, ValueError):
                        # Skip non-numeric values
                        continue
                
                sorted_features = sorted(numeric_features.items(), 
                                       key=lambda x: x[1], 
                                       reverse=True)[:5]
                
                for feature, importance in sorted_features:
                    st.write(f"**{feature}**")
                    st.progress(importance)
                    st.write(f"{importance:.3f}")
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Display feature importance
        if 'feature_importance' in st.session_state:
            features = st.session_state['feature_importance']
            
            # Main visualization
            st.plotly_chart(create_feature_importance_chart(features), 
                           use_container_width=True)
            
            # Category breakdown
            with st.expander("📊 Category Analysis", expanded=True):
                # Group features by category
                categories = {
                    'Technical': ['rsi', 'macd', 'bb_', 'sma', 'ema', 'atr', 'adx'],
                    'On-chain': ['active_addresses', 'transaction', 'nvt', 'hash', 'difficulty'],
                    'Sentiment': ['fear_greed', 'sentiment', 'social', 'reddit', 'twitter'],
                    'Macro': ['sp500', 'gold', 'dxy', 'vix', 'treasury']
                }
                
                category_importance = {}
                for cat, keywords in categories.items():
                    cat_features = {k: v for k, v in features.items() 
                                  if any(kw in k.lower() for kw in keywords)}
                    if cat_features:
                        # Convert values to numeric, filtering out non-numeric ones
                        numeric_values = []
                        numeric_cat_features = {}
                        for k, v in cat_features.items():
                            try:
                                numeric_val = float(v)
                                numeric_values.append(numeric_val)
                                numeric_cat_features[k] = numeric_val
                            except (TypeError, ValueError):
                                continue
                        
                        if numeric_values:
                            category_importance[cat] = {
                                'total': sum(numeric_values),
                                'average': np.mean(numeric_values),
                                'count': len(numeric_values),
                                'features': numeric_cat_features
                            }
                
                # Display category summary
                if category_importance:
                    cat_df = pd.DataFrame({
                        'Category': category_importance.keys(),
                        'Total Importance': [v['total'] for v in category_importance.values()],
                        'Avg Importance': [v['average'] for v in category_importance.values()],
                        'Feature Count': [v['count'] for v in category_importance.values()]
                    })
                    
                    st.dataframe(cat_df.style.format({
                        'Total Importance': '{:.3f}',
                        'Avg Importance': '{:.3f}'
                    }))
                    
                    # Detailed breakdown
                    selected_cat = st.selectbox("Select category for details", 
                                              list(category_importance.keys()))
                    
                    if selected_cat:
                        st.write(f"**{selected_cat} Features:**")
                        cat_features = category_importance[selected_cat]['features']
                        
                        # Filter numeric values before sorting
                        numeric_cat_features = {}
                        for k, v in cat_features.items():
                            try:
                                numeric_cat_features[k] = float(v)
                            except (TypeError, ValueError):
                                continue
                        
                        for feature, importance in sorted(numeric_cat_features.items(), 
                                                        key=lambda x: x[1], 
                                                        reverse=True):
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.write(f"• {feature}")
                            with col2:
                                st.write(f"{importance:.3f}")
    
    with tab4:
        st.subheader("Market Regime Detection")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.info("Identify current market conditions and adapt strategy accordingly")
            
            # Get current regime
            regime_data = api_client.get("/analytics/market-regime")
            
            if regime_data:
                current_regime = regime_data.get('current_regime', 'Unknown')
                confidence = regime_data.get('confidence', 0)
                
                # Display current regime
                regime_class = f"regime-{current_regime.lower()}"
                st.markdown(f"""
                <div class="{regime_class} regime-indicator">
                    <h3>Current Market Regime: {current_regime.upper()}</h3>
                    <p>Confidence: {confidence:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Regime characteristics
                with st.expander("📊 Regime Characteristics", expanded=True):
                    characteristics = regime_data.get('characteristics', {})
                    
                    col_char1, col_char2, col_char3 = st.columns(3)
                    
                    with col_char1:
                        st.markdown("**Volatility**")
                        vol_level = characteristics.get('volatility', 'Normal')
                        st.write(f"Level: {vol_level}")
                        st.write(f"30d: {characteristics.get('vol_30d', 0):.1%}")
                        st.write(f"90d: {characteristics.get('vol_90d', 0):.1%}")
                    
                    with col_char2:
                        st.markdown("**Trend**")
                        trend = characteristics.get('trend', 'Neutral')
                        st.write(f"Direction: {trend}")
                        st.write(f"Strength: {characteristics.get('trend_strength', 0):.1%}")
                        st.write(f"Duration: {characteristics.get('trend_days', 0)} days")
                    
                    with col_char3:
                        st.markdown("**Market Structure**")
                        structure = characteristics.get('market_structure', {})
                        st.write(f"Support: ${structure.get('support', 0):,.0f}")
                        st.write(f"Resistance: ${structure.get('resistance', 0):,.0f}")
                        st.write(f"Range: {structure.get('range_pct', 0):.1%}")
                
                # Historical regimes
                if 'regime_history' in regime_data:
                    with st.expander("📅 Historical Regime Analysis", expanded=False):
                        history = regime_data['regime_history']
                        
                        # Create regime timeline
                        fig = go.Figure()
                        
                        regime_colors = {
                            'bullish': 'green',
                            'bearish': 'red',
                            'neutral': 'gray',
                            'volatile': 'orange'
                        }
                        
                        for regime in history:
                            fig.add_trace(go.Scatter(
                                x=[regime['start'], regime['end']],
                                y=[1, 1],
                                mode='lines',
                                line=dict(
                                    color=regime_colors.get(regime['type'].lower(), 'gray'),
                                    width=20
                                ),
                                name=regime['type'],
                                hovertext=f"{regime['type']}: {regime['duration']} days"
                            ))
                        
                        fig.update_layout(
                            title="Market Regime Timeline",
                            xaxis_title="Date",
                            yaxis_visible=False,
                            height=300,
                            showlegend=False
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Regime statistics
                        regime_stats = pd.DataFrame(history).groupby('type').agg({
                            'duration': ['mean', 'max', 'count'],
                            'avg_return': 'mean',
                            'volatility': 'mean'
                        }).round(2)
                        
                        st.write("**Regime Statistics**")
                        st.dataframe(regime_stats)
        
        with col2:
            # Strategy recommendations
            st.markdown("### 📋 Strategy Recommendations")
            
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            
            if regime_data:
                recommendations = regime_data.get('recommendations', [])
                
                if recommendations:
                    for rec in recommendations:
                        st.write(f"• {rec}")
                else:
                    st.write("No specific recommendations available")
                
                # Optimal parameters for current regime
                optimal_params = regime_data.get('optimal_parameters', {})
                
                if optimal_params:
                    st.markdown("**Optimal Parameters:**")
                    st.write(f"Position Size: {optimal_params.get('position_size', 0):.1%}")
                    st.write(f"Stop Loss: {optimal_params.get('stop_loss', 0):.1%}")
                    st.write(f"Take Profit: {optimal_params.get('take_profit', 0):.1%}")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    with tab5:
        st.subheader("Strategy Optimization")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.info("Find optimal parameters for your trading strategy using Bayesian optimization")
            
            # Optimization setup
            with st.expander("⚙️ Optimization Setup", expanded=True):
                # Parameter ranges
                st.markdown("**Parameter Ranges to Optimize**")
                
                col_param1, col_param2 = st.columns(2)
                
                with col_param1:
                    st.markdown("**Signal Weights**")
                    technical_range = st.slider("Technical Weight", 
                                              min_value=0.0, 
                                              max_value=1.0, 
                                              value=(0.2, 0.6))
                    onchain_range = st.slider("On-chain Weight", 
                                            min_value=0.0, 
                                            max_value=1.0, 
                                            value=(0.2, 0.5))
                    sentiment_range = st.slider("Sentiment Weight", 
                                              min_value=0.0, 
                                              max_value=1.0, 
                                              value=(0.1, 0.3))
                
                with col_param2:
                    st.markdown("**Trading Parameters**")
                    confidence_range = st.slider("Min Confidence", 
                                               min_value=0.3, 
                                               max_value=0.9, 
                                               value=(0.5, 0.7))
                    position_size_range = st.slider("Position Size %", 
                                                  min_value=1, 
                                                  max_value=20, 
                                                  value=(5, 15))
                    stop_loss_range = st.slider("Stop Loss %", 
                                              min_value=1, 
                                              max_value=10, 
                                              value=(3, 7))
                
                # Optimization goals
                st.markdown("**Optimization Goals**")
                objective = st.selectbox("Primary Objective", 
                                       ["Sharpe Ratio", "Total Return", 
                                        "Win Rate", "Risk-Adjusted Return"])
                
                constraints = st.multiselect("Constraints", 
                                           ["Max Drawdown < 20%", "Min Win Rate > 50%", 
                                            "Min Trades > 100", "Max Volatility < 30%"],
                                           default=["Max Drawdown < 20%"])
                
                num_iterations = st.number_input("Optimization Iterations", 
                                               min_value=10, 
                                               max_value=100, 
                                               value=50)
            
            # Run optimization
            if st.button("🎯 Run Optimization", type="primary", use_container_width=True):
                with st.spinner(f"Running {num_iterations} iterations... This may take several minutes"):
                    # Prepare optimization parameters
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
                    
                    # Run optimization
                    result = api_client.post("/analytics/optimize", params)
                    
                    if result and result.get('status') == 'success':
                        st.success("✅ Optimization completed!")
                        st.session_state['optimization_results'] = result.get('results', {})
                    else:
                        error_msg = result.get('error', 'Unknown error') if result else 'Connection error'
                        st.error(f"❌ Optimization failed: {error_msg}")
        
        with col2:
            # Best parameters if available
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
                                st.write(f"**{param_name}:** {value:.1%}")
                            else:
                                st.write(f"**{param_name}:** {value:.2f}")
                        else:
                            st.write(f"**{param_name}:** {value}")
                
                # Expected performance
                expected_perf = opt_results.get('expected_performance', {})
                if expected_perf:
                    st.markdown("**Expected Performance:**")
                    st.write(f"Return: {expected_perf.get('return', 0):.2%}")
                    st.write(f"Sharpe: {expected_perf.get('sharpe', 0):.2f}")
                    st.write(f"Max DD: {expected_perf.get('max_dd', 0):.1%}")
                
                st.markdown('</div>', unsafe_allow_html=True)
        
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
                    line=dict(color=CHART_COLORS['primary'], width=2)
                ))
                
                fig.update_layout(
                    title="Optimization Convergence",
                    xaxis_title="Iteration",
                    yaxis_title="Objective Value",
                    height=400,
                    template="plotly_white"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Parameter evolution
            if 'parameter_history' in opt_results:
                with st.expander("📊 Parameter Evolution", expanded=True):
                    param_history = opt_results['parameter_history']
                    
                    # Create parameter evolution chart
                    fig = make_subplots(
                        rows=2, cols=3,
                        subplot_titles=list(param_history.keys())[:6]
                    )
                    
                    for idx, (param, values) in enumerate(list(param_history.items())[:6]):
                        row = idx // 3 + 1
                        col = idx % 3 + 1
                        
                        fig.add_trace(go.Scatter(
                            y=values,
                            mode='lines',
                            name=param,
                            showlegend=False
                        ), row=row, col=col)
                    
                    fig.update_layout(height=600, title="Parameter Evolution During Optimization")
                    st.plotly_chart(fig, use_container_width=True)
            
            # Apply optimal parameters button
            if st.button("📝 Apply Optimal Parameters", use_container_width=True):
                best_params = opt_results.get('best_parameters', {})
                
                # Convert parameters to config format
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
                
                # Update configuration
                result = api_client.post("/config/update", config_update)
                
                if result and result.get('status') == 'success':
                    st.success("✅ Optimal parameters applied to trading configuration!")
                else:
                    st.error("❌ Failed to apply parameters")

# Auto-refresh option
if st.sidebar.checkbox("Auto-refresh (30s)", value=False):
    time.sleep(30)
    st.rerun()

# Show the page
show_analytics()