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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components.api_client import APIClient
from components.charts import create_candlestick_chart
from utils.helpers import format_currency, format_percentage
from utils.constants import CHART_COLORS

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

def create_backtest_chart(results: dict) -> go.Figure:
    """Create comprehensive backtest results chart"""
    if not results or 'equity_curve' not in results:
        return go.Figure().add_annotation(text="No backtest results available", 
                                        xref="paper", yref="paper", 
                                        x=0.5, y=0.5, showarrow=False)
    
    equity_data = results['equity_curve']
    df = pd.DataFrame(equity_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=("Portfolio Value", "Drawdown %", "Trade Signals"),
        row_heights=[0.5, 0.25, 0.25]
    )
    
    # Portfolio value
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['portfolio_value'],
        mode='lines',
        name='Portfolio Value',
        line=dict(color=CHART_COLORS['primary'], width=2),
        fill='tonexty',
        fillcolor='rgba(102, 126, 234, 0.1)'
    ), row=1, col=1)
    
    # Benchmark (if available)
    if 'benchmark_value' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['benchmark_value'],
            mode='lines',
            name='Buy & Hold',
            line=dict(color='gray', width=2, dash='dash')
        ), row=1, col=1)
    
    # Drawdown
    if 'drawdown' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['drawdown'],
            mode='lines',
            name='Drawdown',
            line=dict(color='red', width=1),
            fill='tozeroy',
            fillcolor='rgba(255, 0, 0, 0.1)'
        ), row=2, col=1)
    
    # Trade signals
    if 'signal' in df.columns:
        buys = df[df['signal'] == 'buy']
        sells = df[df['signal'] == 'sell']
        
        if not buys.empty:
            fig.add_trace(go.Scatter(
                x=buys['timestamp'],
                y=buys['price'],
                mode='markers',
                name='Buy Signal',
                marker=dict(color='green', size=10, symbol='triangle-up')
            ), row=3, col=1)
        
        if not sells.empty:
            fig.add_trace(go.Scatter(
                x=sells['timestamp'],
                y=sells['price'],
                mode='markers',
                name='Sell Signal',
                marker=dict(color='red', size=10, symbol='triangle-down')
            ), row=3, col=1)
    
    # Update layout
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_yaxes(title_text="Value ($)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown %", row=2, col=1)
    fig.update_yaxes(title_text="Price ($)", row=3, col=1)
    
    fig.update_layout(
        title="Backtest Results",
        height=800,
        hovermode='x unified',
        template="plotly_white"
    )
    
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
    
    # Plot sample paths (limit to 100 for performance)
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
    
    # Sort features by importance
    sorted_features = sorted(features.items(), key=lambda x: x[1], reverse=True)[:20]
    
    feature_names = [f[0] for f in sorted_features]
    importance_values = [f[1] for f in sorted_features]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=importance_values,
        y=feature_names,
        orientation='h',
        marker=dict(
            color=importance_values,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Importance")
        )
    ))
    
    fig.update_layout(
        title="Top 20 Feature Importance",
        xaxis_title="Importance Score",
        yaxis_title="Feature",
        height=600,
        template="plotly_white",
        yaxis=dict(autorange="reversed")
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
                
                # Advanced options
                use_walk_forward = st.checkbox("Use Walk-Forward Analysis", value=True)
                include_transaction_costs = st.checkbox("Include Transaction Costs", value=True)
                
            # Run backtest button
            if st.button("🚀 Run Backtest", type="primary", use_container_width=True):
                with st.spinner("Running backtest... This may take a few minutes"):
                    # Prepare backtest parameters
                    params = {
                        "period_days": period_days,
                        "initial_capital": initial_capital,
                        "strategy": strategy_type.lower().replace(" ", "_"),
                        "position_sizing": position_sizing.lower().replace(" ", "_"),
                        "max_positions": max_positions,
                        "stop_loss": stop_loss / 100,
                        "walk_forward": use_walk_forward,
                        "transaction_costs": include_transaction_costs
                    }
                    
                    # Try enhanced backtest first
                    result = api_client.post("/backtest/enhanced/run", params)
                    if not result or result.get('status') == 'error':
                        # Fallback to regular backtest
                        result = api_client.post("/backtest/run", params)
                    
                    if result and result.get('status') == 'success':
                        st.success("✅ Backtest completed successfully!")
                        st.session_state['backtest_results'] = result.get('results', {})
                    else:
                        error_msg = result.get('error', 'Unknown error') if result else 'Connection error'
                        st.error(f"❌ Backtest failed: {error_msg}")
        
        with col2:
            # Quick stats if results available
            if 'backtest_results' in st.session_state:
                results = st.session_state['backtest_results']
                metrics = results.get('metrics', {})
                
                st.markdown("### 📊 Quick Stats")
                
                metric_card = '<div class="metric-card">'
                st.markdown(metric_card, unsafe_allow_html=True)
                
                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    st.metric("Total Return", 
                            f"{metrics.get('total_return', 0):.2%}")
                    st.metric("Sharpe Ratio", 
                            f"{metrics.get('sharpe_ratio', 0):.2f}")
                with col_m2:
                    st.metric("Win Rate", 
                            f"{metrics.get('win_rate', 0):.1f}%")
                    st.metric("Max Drawdown", 
                            f"{metrics.get('max_drawdown', 0):.2%}")
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Display results
        if 'backtest_results' in st.session_state:
            results = st.session_state['backtest_results']
            
            # Main chart
            st.plotly_chart(create_backtest_chart(results), use_container_width=True)
            
            # Detailed metrics
            with st.expander("📈 Detailed Performance Metrics", expanded=True):
                metrics = results.get('metrics', {})
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown("**Returns**")
                    st.write(f"Total: {metrics.get('total_return', 0):.2%}")
                    st.write(f"Annual: {metrics.get('annual_return', 0):.2%}")
                    st.write(f"Monthly Avg: {metrics.get('monthly_return', 0):.2%}")
                
                with col2:
                    st.markdown("**Risk Metrics**")
                    st.write(f"Volatility: {metrics.get('volatility', 0):.2%}")
                    st.write(f"Sortino: {metrics.get('sortino_ratio', 0):.2f}")
                    st.write(f"Calmar: {metrics.get('calmar_ratio', 0):.2f}")
                
                with col3:
                    st.markdown("**Trading Stats**")
                    st.write(f"Total Trades: {metrics.get('total_trades', 0)}")
                    st.write(f"Avg Trade: {metrics.get('avg_trade_return', 0):.2%}")
                    st.write(f"Best Trade: {metrics.get('best_trade', 0):.2%}")
                
                with col4:
                    st.markdown("**Drawdown**")
                    st.write(f"Max DD: {metrics.get('max_drawdown', 0):.2%}")
                    st.write(f"Avg DD: {metrics.get('avg_drawdown', 0):.2%}")
                    st.write(f"Recovery: {metrics.get('recovery_time', 0)} days")
            
            # Trade analysis
            if 'trades' in results:
                with st.expander("📋 Trade Analysis", expanded=False):
                    trades_df = pd.DataFrame(results['trades'])
                    
                    # Summary by signal type
                    if 'signal_source' in trades_df.columns:
                        signal_summary = trades_df.groupby('signal_source').agg({
                            'return': ['count', 'mean', 'sum'],
                            'win': 'mean'
                        }).round(4)
                        
                        st.write("**Performance by Signal Source**")
                        st.dataframe(signal_summary)
                    
                    # Monthly breakdown
                    if 'timestamp' in trades_df.columns:
                        trades_df['month'] = pd.to_datetime(trades_df['timestamp']).dt.to_period('M')
                        monthly_summary = trades_df.groupby('month').agg({
                            'return': ['count', 'mean', 'sum']
                        }).round(4)
                        
                        st.write("**Monthly Performance**")
                        st.dataframe(monthly_summary)
    
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
                    
                    if result and result.get('status') == 'success':
                        st.success("✅ Analysis completed!")
                        st.session_state['feature_importance'] = result.get('features', {})
                    else:
                        # Fallback to basic feature importance
                        result = api_client.get("/ml/feature-importance")
                        if result:
                            st.session_state['feature_importance'] = result
                        else:
                            st.error("❌ Analysis failed")
        
        with col2:
            # Top features preview
            if 'feature_importance' in st.session_state:
                features = st.session_state['feature_importance']
                
                st.markdown("### 🏆 Top Features")
                
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                
                # Show top 5 features
                sorted_features = sorted(features.items(), 
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
                        category_importance[cat] = {
                            'total': sum(cat_features.values()),
                            'average': np.mean(list(cat_features.values())),
                            'count': len(cat_features),
                            'features': cat_features
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
                        
                        for feature, importance in sorted(cat_features.items(), 
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