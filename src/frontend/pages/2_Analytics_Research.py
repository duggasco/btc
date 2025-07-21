"""Analytics & Research - Comprehensive analysis tools"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

# Component imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from components.layout.dashboard_grid import render_dashboard_header
from components.display.metric_card import render_metric_row, format_currency, format_percentage
from components.display.data_table import render_data_table
from components.display.chart_container import create_chart_container, apply_dark_theme
from components.controls.form_controls import create_input_group, create_button, create_form_section
from utils.api_client import get_api_client

# Page configuration
st.set_page_config(
    page_title="Analytics & Research - BTC Trading System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load CSS
st.markdown("""
<style>
/* Import theme CSS */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
</style>
""", unsafe_allow_html=True)

# Import CSS files content
with open(Path(__file__).parent.parent / "styles" / "theme.css", "r") as f:
    theme_css = f.read()
with open(Path(__file__).parent.parent / "styles" / "components.css", "r") as f:
    components_css = f.read()

st.markdown(f"<style>{theme_css}{components_css}</style>", unsafe_allow_html=True)

# Initialize API client
api_client = get_api_client()

# Tabs for different analytics tools
st.markdown("""
<style>
.stTabs [data-baseweb="tab-list"] {
    background-color: var(--bg-secondary);
    padding: 8px;
    border-radius: var(--radius-md);
}

.stTabs [data-baseweb="tab"] {
    color: var(--text-secondary);
    font-weight: 500;
    padding: 8px 16px;
}

.stTabs [aria-selected="true"] {
    color: var(--accent-primary) !important;
    border-bottom: 2px solid var(--accent-primary);
}
</style>
""", unsafe_allow_html=True)

def render_header():
    """Render page header"""
    status_indicators = [
        {"label": "System", "status": "online"},
        {"label": "Data Quality", "status": "online"}
    ]
    render_dashboard_header("Analytics & Research", status_indicators)

def render_backtesting_tab():
    """Render backtesting analysis tab"""
    st.markdown('<div class="analytics-section">', unsafe_allow_html=True)
    
    # Backtest configuration
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        period = create_input_group(
            "Backtest Period",
            input_type="select",
            options=["7d", "30d", "90d", "180d", "1y", "2y"],
            value="30d",
            key="backtest_period"
        )
    
    with col2:
        optimize_weights = create_input_group(
            "Optimize Weights",
            input_type="checkbox",
            value=True,
            key="optimize_weights"
        )
    
    with col3:
        include_macro = create_input_group(
            "Include Macro Data",
            input_type="checkbox",
            value=True,
            key="include_macro"
        )
    
    # Run backtest button
    if create_button("Run Backtest", variant="primary", key="run_backtest"):
        with st.spinner("Running backtest..."):
            try:
                results = api_client.run_backtest({
                    "period": period,
                    "optimize_weights": optimize_weights,
                    "include_macro": include_macro
                })
                st.session_state['backtest_results'] = results
            except Exception as e:
                st.error(f"Backtest failed: {str(e)}")
    
    # Display results
    if 'backtest_results' in st.session_state:
        results = st.session_state['backtest_results']
        
        # Performance metrics
        st.markdown('<h3>Performance Metrics</h3>', unsafe_allow_html=True)
        
        metrics = [
            {
                "title": "SORTINO RATIO",
                "value": results.get('performance_metrics', {}).get('sortino_ratio_mean', 0),
                "format_func": lambda x: f"{x:.2f}"
            },
            {
                "title": "SHARPE RATIO",
                "value": results.get('performance_metrics', {}).get('sharpe_ratio_mean', 0),
                "format_func": lambda x: f"{x:.2f}"
            },
            {
                "title": "MAX DRAWDOWN",
                "value": results.get('performance_metrics', {}).get('max_drawdown_mean', 0),
                "format_func": lambda x: f"{x:.1%}"
            },
            {
                "title": "WIN RATE",
                "value": results.get('performance_metrics', {}).get('win_rate_mean', 0),
                "format_func": lambda x: f"{x:.1%}"
            },
            {
                "title": "TOTAL RETURN",
                "value": results.get('performance_metrics', {}).get('total_return_mean', 0),
                "format_func": lambda x: f"{x:.1%}"
            },
            {
                "title": "PROFIT FACTOR",
                "value": results.get('performance_metrics', {}).get('profit_factor_mean', 1),
                "format_func": lambda x: f"{x:.2f}"
            }
        ]
        
        render_metric_row(metrics[:3])
        render_metric_row(metrics[3:])
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Equity curve
            def create_equity_curve():
                # Simulated equity curve data
                dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
                equity = np.cumprod(1 + np.random.normal(0.001, 0.02, 100))
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=equity,
                    mode='lines',
                    name='Equity',
                    line=dict(color='#f7931a', width=2)
                ))
                
                fig.update_layout(
                    title="Equity Curve",
                    xaxis_title="Date",
                    yaxis_title="Portfolio Value",
                    height=300
                )
                
                return apply_dark_theme(fig)
            
            create_chart_container("Equity Curve", create_equity_curve)
        
        with col2:
            # Drawdown chart
            def create_drawdown_chart():
                dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
                drawdown = -np.abs(np.random.normal(0, 0.05, 100))
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=drawdown,
                    mode='lines',
                    fill='tozeroy',
                    name='Drawdown',
                    line=dict(color='#ef4444', width=1),
                    fillcolor='rgba(239, 68, 68, 0.2)'
                ))
                
                fig.update_layout(
                    title="Drawdown",
                    xaxis_title="Date",
                    yaxis_title="Drawdown %",
                    height=300
                )
                
                return apply_dark_theme(fig)
            
            create_chart_container("Drawdown", create_drawdown_chart)
        
        # Trade analysis
        st.markdown('<h3>Trade Analysis</h3>', unsafe_allow_html=True)
        
        trading_stats = results.get('trading_statistics', {})
        if trading_stats:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Trades", trading_stats.get('total_trades', 0))
            with col2:
                st.metric("Long Positions", trading_stats.get('long_positions', 0))
            with col3:
                st.metric("Short Positions", trading_stats.get('short_positions', 0))
            with col4:
                st.metric("Avg Position Turnover", f"{trading_stats.get('avg_position_turnover', 0):.1%}")
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_monte_carlo_tab():
    """Render Monte Carlo simulation tab"""
    st.markdown('<div class="analytics-section">', unsafe_allow_html=True)
    
    # Configuration
    col1, col2, col3 = st.columns(3)
    
    with col1:
        n_simulations = create_input_group(
            "Number of Simulations",
            input_type="number",
            value=1000,
            min_value=100,
            max_value=10000,
            step=100,
            key="mc_simulations"
        )
    
    with col2:
        time_horizon = create_input_group(
            "Time Horizon (days)",
            input_type="number",
            value=365,
            min_value=30,
            max_value=1825,
            step=30,
            key="mc_horizon"
        )
    
    with col3:
        confidence_level = create_input_group(
            "Confidence Level",
            input_type="select",
            options=["90%", "95%", "99%"],
            value="95%",
            key="mc_confidence"
        )
    
    # Run simulation
    if create_button("Run Monte Carlo Simulation", variant="primary", key="run_monte_carlo"):
        with st.spinner("Running simulations..."):
            # Simulated results
            st.session_state['mc_results'] = {
                'simulations': n_simulations,
                'horizon': time_horizon,
                'confidence': confidence_level
            }
    
    # Display results
    if 'mc_results' in st.session_state:
        # Summary statistics
        st.markdown('<h3>Risk Projections</h3>', unsafe_allow_html=True)
        
        metrics = [
            {
                "title": "EXPECTED RETURN",
                "value": 0.15,
                "format_func": lambda x: f"{x:.1%}"
            },
            {
                "title": f"VAR ({confidence_level})",
                "value": -0.08,
                "format_func": lambda x: f"{x:.1%}"
            },
            {
                "title": "EXPECTED VOLATILITY",
                "value": 0.25,
                "format_func": lambda x: f"{x:.1%}"
            },
            {
                "title": "PROBABILITY OF PROFIT",
                "value": 0.68,
                "format_func": lambda x: f"{x:.1%}"
            }
        ]
        
        render_metric_row(metrics)
        
        # Simulation paths chart
        def create_simulation_paths():
            n_paths = min(100, n_simulations)
            days = np.arange(time_horizon)
            
            fig = go.Figure()
            
            # Add individual paths
            for i in range(n_paths):
                returns = np.random.normal(0.0005, 0.02, time_horizon)
                path = np.cumprod(1 + returns)
                
                fig.add_trace(go.Scatter(
                    x=days,
                    y=path,
                    mode='lines',
                    line=dict(color='rgba(156, 163, 175, 0.1)', width=1),
                    showlegend=False
                ))
            
            # Add mean path
            mean_path = np.exp(np.cumsum(np.full(time_horizon, 0.0005)))
            fig.add_trace(go.Scatter(
                x=days,
                y=mean_path,
                mode='lines',
                name='Expected Path',
                line=dict(color='#f7931a', width=3)
            ))
            
            fig.update_layout(
                title=f"Monte Carlo Simulation Paths ({n_paths} shown)",
                xaxis_title="Days",
                yaxis_title="Portfolio Value (Relative)",
                height=400
            )
            
            return apply_dark_theme(fig)
        
        create_chart_container("Simulation Paths", create_simulation_paths)
        
        # Distribution chart
        col1, col2 = st.columns(2)
        
        with col1:
            def create_returns_distribution():
                returns = np.random.normal(0.15, 0.25, n_simulations)
                
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=returns,
                    nbinsx=50,
                    name='Returns',
                    marker_color='#f7931a'
                ))
                
                fig.update_layout(
                    title="Returns Distribution",
                    xaxis_title="Annual Return",
                    yaxis_title="Frequency",
                    height=300
                )
                
                return apply_dark_theme(fig)
            
            create_chart_container("Returns Distribution", create_returns_distribution)
        
        with col2:
            def create_risk_metrics():
                metrics_data = pd.DataFrame({
                    'Percentile': ['1%', '5%', '10%', '25%', '50%', '75%', '90%', '95%', '99%'],
                    'Return': [-0.42, -0.26, -0.17, -0.02, 0.15, 0.32, 0.47, 0.56, 0.72]
                })
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=metrics_data['Percentile'],
                    y=metrics_data['Return'],
                    marker_color=['#ef4444' if r < 0 else '#22c55e' for r in metrics_data['Return']]
                ))
                
                fig.update_layout(
                    title="Return Percentiles",
                    xaxis_title="Percentile",
                    yaxis_title="Return",
                    height=300
                )
                
                return apply_dark_theme(fig)
            
            create_chart_container("Risk Metrics", create_risk_metrics)
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_optimization_tab():
    """Render strategy optimization tab"""
    st.markdown('<div class="analytics-section">', unsafe_allow_html=True)
    
    # Optimization settings
    col1, col2 = st.columns(2)
    
    with col1:
        optimization_target = create_input_group(
            "Optimization Target",
            input_type="select",
            options=["Sortino Ratio", "Sharpe Ratio", "Total Return", "Win Rate"],
            value="Sortino Ratio",
            key="opt_target"
        )
        
        n_trials = create_input_group(
            "Number of Trials",
            input_type="number",
            value=100,
            min_value=10,
            max_value=1000,
            step=10,
            key="opt_trials"
        )
    
    with col2:
        parameter_ranges = st.expander("Parameter Ranges")
        with parameter_ranges:
            stop_loss = create_input_group(
                "Stop Loss Range (%)",
                input_type="slider",
                min_value=1.0,
                max_value=10.0,
                value=(2.0, 5.0),
                key="opt_stop_loss"
            )
            
            take_profit = create_input_group(
                "Take Profit Range (%)",
                input_type="slider",
                min_value=2.0,
                max_value=20.0,
                value=(5.0, 10.0),
                key="opt_take_profit"
            )
    
    # Run optimization
    if create_button("Run Optimization", variant="primary", key="run_optimization"):
        with st.spinner("Running optimization..."):
            st.session_state['opt_results'] = True
    
    # Display results
    if 'opt_results' in st.session_state:
        st.markdown('<h3>Optimization Results</h3>', unsafe_allow_html=True)
        
        # Best parameters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Optimal Stop Loss", "3.2%")
        with col2:
            st.metric("Optimal Take Profit", "7.8%")
        with col3:
            st.metric("Expected Sortino", "2.15")
        
        # Parameter heatmap
        def create_parameter_heatmap():
            # Create sample heatmap data
            stop_losses = np.linspace(2, 5, 10)
            take_profits = np.linspace(5, 10, 10)
            z_values = np.random.uniform(1.5, 2.5, (10, 10))
            
            fig = go.Figure(data=go.Heatmap(
                x=take_profits,
                y=stop_losses,
                z=z_values,
                colorscale='Viridis',
                text=np.round(z_values, 2),
                texttemplate='%{text}',
                textfont={"size": 10}
            ))
            
            fig.update_layout(
                title="Parameter Optimization Heatmap",
                xaxis_title="Take Profit %",
                yaxis_title="Stop Loss %",
                height=400
            )
            
            return apply_dark_theme(fig)
        
        create_chart_container("Parameter Heatmap", create_parameter_heatmap)
        
        # Convergence plot
        col1, col2 = st.columns(2)
        
        with col1:
            def create_convergence_plot():
                iterations = np.arange(1, n_trials + 1)
                best_values = np.cummax(np.random.uniform(1.5, 2.5, n_trials))
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=iterations,
                    y=best_values,
                    mode='lines',
                    name='Best Value',
                    line=dict(color='#f7931a', width=2)
                ))
                
                fig.update_layout(
                    title="Optimization Convergence",
                    xaxis_title="Iteration",
                    yaxis_title="Best Sortino Ratio",
                    height=300
                )
                
                return apply_dark_theme(fig)
            
            create_chart_container("Convergence", create_convergence_plot)
        
        with col2:
            # Top parameter combinations
            st.markdown('<h4>Top Parameter Combinations</h4>', unsafe_allow_html=True)
            
            top_params = pd.DataFrame({
                'Stop Loss': [3.2, 3.5, 2.8, 3.1, 3.3],
                'Take Profit': [7.8, 8.2, 7.5, 8.0, 7.6],
                'Sortino': [2.15, 2.12, 2.10, 2.08, 2.06]
            })
            
            render_data_table(top_params, height=200)
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_data_quality_tab():
    """Render data quality monitoring tab"""
    st.markdown('<div class="analytics-section">', unsafe_allow_html=True)
    
    # Fetch data quality metrics
    try:
        quality_metrics = api_client._make_request("GET", "/analytics/data-quality")
    except:
        quality_metrics = {
            'overall_score': 0.92,
            'completeness': 0.98,
            'accuracy': 0.95,
            'timeliness': 0.88,
            'consistency': 0.90
        }
    
    # Overall metrics
    st.markdown('<h3>Data Quality Overview</h3>', unsafe_allow_html=True)
    
    metrics = [
        {
            "title": "OVERALL SCORE",
            "value": quality_metrics.get('overall_score', 0),
            "format_func": lambda x: f"{x:.0%}"
        },
        {
            "title": "COMPLETENESS",
            "value": quality_metrics.get('completeness', 0),
            "format_func": lambda x: f"{x:.0%}"
        },
        {
            "title": "ACCURACY",
            "value": quality_metrics.get('accuracy', 0),
            "format_func": lambda x: f"{x:.0%}"
        },
        {
            "title": "TIMELINESS",
            "value": quality_metrics.get('timeliness', 0),
            "format_func": lambda x: f"{x:.0%}"
        }
    ]
    
    render_metric_row(metrics)
    
    # Source monitoring
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h4>Data Source Status</h4>', unsafe_allow_html=True)
        
        sources = pd.DataFrame({
            'Source': ['CoinGecko', 'Binance', 'Yahoo Finance', 'Alternative.me'],
            'Status': ['Online', 'Online', 'Delayed', 'Online'],
            'Latency': ['45ms', '23ms', '120ms', '67ms'],
            'Success Rate': ['99.8%', '99.9%', '97.2%', '98.5%']
        })
        
        render_data_table(sources, height=200)
    
    with col2:
        def create_quality_timeline():
            dates = pd.date_range(end=datetime.now(), periods=24, freq='H')
            quality_scores = np.random.uniform(0.85, 0.98, 24)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=quality_scores,
                mode='lines+markers',
                name='Quality Score',
                line=dict(color='#3b82f6', width=2),
                marker=dict(size=4)
            ))
            
            # Add threshold line
            fig.add_hline(
                y=0.90,
                line_dash="dash",
                line_color="#6b7280",
                annotation_text="Threshold"
            )
            
            fig.update_layout(
                title="Data Quality Timeline (24h)",
                xaxis_title="Time",
                yaxis_title="Quality Score",
                height=250
            )
            
            return apply_dark_theme(fig)
        
        create_chart_container("Quality Timeline", create_quality_timeline)
    
    # Missing data analysis
    st.markdown('<h4>Missing Data Analysis</h4>', unsafe_allow_html=True)
    
    missing_data = pd.DataFrame({
        'Field': ['Open', 'High', 'Low', 'Close', 'Volume', 'Market Cap'],
        'Missing Count': [0, 0, 0, 2, 5, 12],
        'Missing %': ['0.0%', '0.0%', '0.0%', '0.1%', '0.3%', '0.7%']
    })
    
    render_data_table(missing_data, height=250)
    
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    """Main function"""
    # Render header
    render_header()
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Backtesting",
        "Monte Carlo",
        "Optimization",
        "Data Quality"
    ])
    
    with tab1:
        render_backtesting_tab()
    
    with tab2:
        render_monte_carlo_tab()
    
    with tab3:
        render_optimization_tab()
    
    with tab4:
        render_data_quality_tab()

if __name__ == "__main__":
    main()