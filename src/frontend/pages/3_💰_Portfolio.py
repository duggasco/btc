import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime, timedelta
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components.api_client import APIClient
from components.charts import create_portfolio_chart, create_performance_chart
from components.metrics import display_portfolio_metrics, display_risk_metrics
from utils.helpers import format_currency, format_percentage, calculate_sharpe_ratio

st.set_page_config(page_title="Portfolio Management", page_icon="💰", layout="wide")

# Initialize API client
@st.cache_resource
def get_api_client():
    return APIClient(os.getenv("API_BASE_URL", "http://backend:8080"))

api_client = get_api_client()

st.title("💰 Portfolio Management")
st.markdown("Track positions, analyze performance, and manage risk")

# Fetch portfolio data
try:
    portfolio_metrics = api_client.get("/portfolio/metrics") or {}
    positions = api_client.get("/portfolio/positions") or []
    trades = api_client.get("/trades/all") or []
    performance = api_client.get("/analytics/performance") or {}
    btc_price_data = api_client.get("/btc/latest") or {}
    btc_price = btc_price_data.get("latest_price", 0)
except Exception as e:
    st.error(f"Error fetching portfolio data: {str(e)}")
    st.stop()

# Portfolio overview metrics
display_portfolio_metrics(portfolio_metrics)

# Additional performance metrics
if performance:
    display_risk_metrics(performance)

# Tabs for different views
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Current Positions",
    "📈 Performance Analysis",
    "🔄 Trade History",
    "⚠️ Risk Management",
    "📊 Analytics",
    "💸 P&L Analysis"
])

# Current Positions Tab
with tab1:
    st.markdown("### Active Positions")
    
    if positions:
        # Convert to DataFrame for easier manipulation
        positions_df = pd.DataFrame(positions)
        
        # Calculate additional metrics for each position
        positions_data = []
        for _, position in positions_df.iterrows():
            entry_price = position.get("entry_price", 0)
            size = position.get("size", 0)
            current_price = position.get("current_price", btc_price)
            
            value = size * current_price
            pnl = (current_price - entry_price) * size
            pnl_percent = ((current_price / entry_price) - 1) * 100 if entry_price > 0 else 0
            
            # Calculate position duration
            entry_time = pd.to_datetime(position.get("entry_time", datetime.now()))
            duration = (datetime.now() - entry_time).days
            
            positions_data.append({
                "Symbol": position.get("symbol", "BTC-USD"),
                "Side": position.get("side", "long").upper(),
                "Size": size,
                "Entry Price": entry_price,
                "Current Price": current_price,
                "Value": value,
                "P&L": pnl,
                "P&L %": pnl_percent,
                "Duration": f"{duration}d",
                "Status": "🟢" if pnl > 0 else "🔴"
            })
        
        positions_display = pd.DataFrame(positions_data)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            total_value = positions_display["Value"].sum()
            st.metric("Total Position Value", format_currency(total_value))
        with col2:
            total_pnl = positions_display["P&L"].sum()
            st.metric("Unrealized P&L", format_currency(total_pnl))
        with col3:
            avg_pnl_pct = positions_display["P&L %"].mean()
            st.metric("Average P&L %", format_percentage(avg_pnl_pct))
        with col4:
            winning_positions = len(positions_display[positions_display["P&L"] > 0])
            st.metric("Winning Positions", f"{winning_positions}/{len(positions_display)}")
        
        # Display positions table
        st.dataframe(positions_display, use_container_width=True, hide_index=True)
    else:
        st.info("No active positions")

# Performance Analysis Tab
with tab2:
    st.markdown("### Portfolio Performance Analysis")
    
    # Get historical performance data
    perf_history = api_client.get("/portfolio/performance/history") or {}
    
    if perf_history and "equity_curve" in perf_history:
        # Time period selector
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            period = st.selectbox(
                "Time Period",
                ["1D", "1W", "1M", "3M", "6M", "1Y", "All"],
                index=2
            )
        
        # Performance metrics
        st.markdown("#### Key Performance Indicators")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        metrics = perf_history.get("metrics", {})
        
        with col1:
            total_return = metrics.get("total_return", 0)
            st.metric(
                "Total Return",
                format_percentage(total_return),
                delta=f"{total_return:.1f}%"
            )
        
        with col2:
            sharpe_ratio = metrics.get("sharpe_ratio", 0)
            st.metric(
                "Sharpe Ratio",
                f"{sharpe_ratio:.2f}",
                help="Risk-adjusted return metric"
            )
        
        with col3:
            max_drawdown = metrics.get("max_drawdown", 0)
            st.metric(
                "Max Drawdown",
                format_percentage(max_drawdown),
                delta=f"{max_drawdown:.1f}%"
            )
        
        with col4:
            win_rate = metrics.get("win_rate", 0)
            st.metric(
                "Win Rate",
                format_percentage(win_rate),
                help="Percentage of profitable trades"
            )
        
        with col5:
            profit_factor = metrics.get("profit_factor", 0)
            st.metric(
                "Profit Factor",
                f"{profit_factor:.2f}",
                help="Gross profit / Gross loss"
            )
        
        # Equity curve chart
        st.markdown("#### Portfolio Equity Curve")
        
        equity_data = pd.DataFrame(perf_history["equity_curve"])
        equity_data['timestamp'] = pd.to_datetime(equity_data['timestamp'])
        
        fig = go.Figure()
        
        # Portfolio value line
        fig.add_trace(go.Scatter(
            x=equity_data['timestamp'],
            y=equity_data['value'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#667eea', width=3),
            fill='tonexty',
            fillcolor='rgba(102, 126, 234, 0.1)'
        ))
        
        # Add benchmark if available
        if 'benchmark' in equity_data.columns:
            fig.add_trace(go.Scatter(
                x=equity_data['timestamp'],
                y=equity_data['benchmark'],
                mode='lines',
                name='Buy & Hold',
                line=dict(color='gray', width=2, dash='dash')
            ))
        
        fig.update_layout(
            title="Portfolio Value Over Time",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Drawdown chart
        st.markdown("#### Drawdown Analysis")
        
        if 'drawdown' in equity_data.columns:
            fig_dd = go.Figure()
            
            fig_dd.add_trace(go.Scatter(
                x=equity_data['timestamp'],
                y=equity_data['drawdown'] * 100,
                mode='lines',
                name='Drawdown',
                line=dict(color='red', width=2),
                fill='tozeroy',
                fillcolor='rgba(255, 0, 0, 0.1)'
            ))
            
            fig_dd.update_layout(
                title="Portfolio Drawdown",
                xaxis_title="Date",
                yaxis_title="Drawdown (%)",
                height=300,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_dd, use_container_width=True)
        
        # Monthly returns heatmap
        st.markdown("#### Monthly Returns Heatmap")
        
        if 'monthly_returns' in perf_history:
            monthly_df = pd.DataFrame(perf_history['monthly_returns'])
            
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=monthly_df.values,
                x=monthly_df.columns,
                y=monthly_df.index,
                colorscale='RdBu',
                zmid=0,
                text=monthly_df.values,
                texttemplate='%{text:.1f}%',
                textfont={"size": 10}
            ))
            
            fig_heatmap.update_layout(
                title="Monthly Returns (%)",
                height=400
            )
            
            st.plotly_chart(fig_heatmap, use_container_width=True)
    else:
        st.info("No performance history available. Start trading to see performance analytics.")

# Trade History Tab
with tab3:
    st.markdown("### Trade History")
    
    if trades:
        # Convert to DataFrame
        trades_df = pd.DataFrame(trades)
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
        
        # Filters
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            date_filter = st.date_input(
                "Date Range",
                value=(datetime.now() - timedelta(days=30), datetime.now()),
                key="trade_date_range"
            )
        
        with col2:
            side_filter = st.selectbox(
                "Side",
                ["All", "Buy", "Sell"],
                key="trade_side"
            )
        
        with col3:
            status_filter = st.selectbox(
                "Status",
                ["All", "Closed", "Open"],
                key="trade_status"
            )
        
        with col4:
            search_term = st.text_input(
                "Search",
                placeholder="Search trades...",
                key="trade_search"
            )
        
        # Apply filters
        filtered_trades = trades_df.copy()
        
        if date_filter:
            start_date, end_date = date_filter
            filtered_trades = filtered_trades[
                (filtered_trades['timestamp'].dt.date >= start_date) &
                (filtered_trades['timestamp'].dt.date <= end_date)
            ]
        
        if side_filter != "All":
            filtered_trades = filtered_trades[filtered_trades['side'] == side_filter.lower()]
        
        if status_filter != "All":
            filtered_trades = filtered_trades[filtered_trades['status'] == status_filter.lower()]
        
        if search_term:
            # Search across multiple columns
            mask = filtered_trades.astype(str).apply(
                lambda x: x.str.contains(search_term, case=False)
            ).any(axis=1)
            filtered_trades = filtered_trades[mask]
        
        # Summary statistics
        st.markdown("#### Trade Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Trades", len(filtered_trades))
        
        with col2:
            closed_trades = filtered_trades[filtered_trades['status'] == 'closed']
            profitable_trades = closed_trades[closed_trades['pnl'] > 0] if 'pnl' in closed_trades else pd.DataFrame()
            win_rate = len(profitable_trades) / len(closed_trades) * 100 if len(closed_trades) > 0 else 0
            st.metric("Win Rate", f"{win_rate:.1f}%")
        
        with col3:
            total_pnl = closed_trades['pnl'].sum() if 'pnl' in closed_trades and len(closed_trades) > 0 else 0
            st.metric("Total P&L", format_currency(total_pnl))
        
        with col4:
            avg_trade = closed_trades['pnl'].mean() if 'pnl' in closed_trades and len(closed_trades) > 0 else 0
            st.metric("Avg Trade P&L", format_currency(avg_trade))
        
        # Trade table
        st.markdown("#### Trade Details")
        
        # Format for display
        display_columns = ['timestamp', 'side', 'size', 'entry_price', 'exit_price', 'pnl', 'pnl_percent', 'status', 'duration']
        available_columns = [col for col in display_columns if col in filtered_trades.columns]
        
        display_df = filtered_trades[available_columns].copy()
        
        # Format columns
        if 'timestamp' in display_df:
            display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
        if 'pnl' in display_df:
            display_df['pnl'] = display_df['pnl'].apply(lambda x: format_currency(x))
        if 'pnl_percent' in display_df:
            display_df['pnl_percent'] = display_df['pnl_percent'].apply(lambda x: f"{x:.2f}%")
        if 'entry_price' in display_df:
            display_df['entry_price'] = display_df['entry_price'].apply(lambda x: format_currency(x))
        if 'exit_price' in display_df:
            display_df['exit_price'] = display_df['exit_price'].apply(lambda x: format_currency(x) if pd.notna(x) else "-")
        
        # Style the dataframe
        def style_pnl(val):
            if isinstance(val, str) and '$' in val:
                value = float(val.replace('$', '').replace(',', ''))
                return 'color: green' if value > 0 else 'color: red' if value < 0 else ''
            return ''
        
        styled_df = display_df.style.map(style_pnl, subset=['pnl'] if 'pnl' in display_df else [])
        
        st.dataframe(styled_df, use_container_width=True, height=500)
        
        # Export button
        csv = filtered_trades.to_csv(index=False)
        st.download_button(
            label="📥 Export Trade History",
            data=csv,
            file_name=f"trade_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime='text/csv'
        )
    else:
        st.info("No trade history available. Start trading to see your transaction history.")

# Risk Management Tab
with tab4:
    st.markdown("### Risk Management")
    
    risk_metrics = api_client.get("/analytics/risk-metrics") or {}
    
    # Risk overview
    st.markdown("#### Current Risk Exposure")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        var_95 = risk_metrics.get("var_95", 0)
        st.metric(
            "VaR (95%)",
            format_currency(var_95),
            help="Value at Risk - Maximum expected loss in 95% of scenarios"
        )
    
    with col2:
        cvar_95 = risk_metrics.get("cvar_95", 0)
        st.metric(
            "CVaR (95%)",
            format_currency(cvar_95),
            help="Conditional VaR - Expected loss beyond VaR threshold"
        )
    
    with col3:
        exposure = risk_metrics.get("total_exposure", 0)
        max_exposure = risk_metrics.get("max_exposure", 100)
        exposure_pct = (exposure / max_exposure * 100) if max_exposure > 0 else 0
        st.metric(
            "Portfolio Exposure",
            f"{exposure_pct:.1f}%",
            help="Current exposure vs maximum allowed"
        )
    
    with col4:
        leverage = risk_metrics.get("leverage", 1.0)
        st.metric(
            "Leverage",
            f"{leverage:.2f}x",
            help="Total position value / Account equity"
        )
    
    # Risk limits
    st.markdown("#### Risk Limits & Compliance")
    
    limits = risk_metrics.get("limits", {})
    compliance = risk_metrics.get("compliance", {})
    
    limit_data = []
    for limit_type, limit_value in limits.items():
        current_value = compliance.get(limit_type, {}).get("current", 0)
        status = compliance.get(limit_type, {}).get("status", "OK")
        
        limit_data.append({
            "Risk Limit": limit_type.replace("_", " ").title(),
            "Current": current_value,
            "Limit": limit_value,
            "Usage": f"{(current_value / limit_value * 100):.1f}%" if limit_value > 0 else "0%",
            "Status": "✅" if status == "OK" else "⚠️"
        })
    
    if limit_data:
        limits_df = pd.DataFrame(limit_data)
        st.dataframe(limits_df, use_container_width=True, hide_index=True)
    
    # Position sizing calculator
    st.markdown("#### Position Sizing Calculator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        account_balance = st.number_input(
            "Account Balance ($)",
            min_value=100.0,
            value=portfolio_metrics.get("total_equity", 10000.0),
            step=100.0
        )
        
        risk_per_trade = st.slider(
            "Risk Per Trade (%)",
            min_value=0.5,
            max_value=5.0,
            value=2.0,
            step=0.5
        )
        
        entry_price = st.number_input(
            "Entry Price ($)",
            min_value=1.0,
            value=float(btc_price) if btc_price else 50000.0,
            step=100.0
        )
        
        stop_loss_price = st.number_input(
            "Stop Loss Price ($)",
            min_value=1.0,
            value=entry_price * 0.95,
            step=100.0
        )
    
    with col2:
        # Calculate position size
        risk_amount = account_balance * (risk_per_trade / 100)
        price_risk = abs(entry_price - stop_loss_price)
        position_size = risk_amount / price_risk if price_risk > 0 else 0
        position_value = position_size * entry_price
        
        st.markdown("##### Recommended Position")
        st.metric("Position Size", f"{position_size:.6f} BTC")
        st.metric("Position Value", format_currency(position_value))
        st.metric("Risk Amount", format_currency(risk_amount))
        st.metric("Risk/Reward", f"1:{((entry_price * 1.02 - entry_price) / price_risk):.1f}")
    
    # Risk heatmap
    st.markdown("#### Risk Correlation Matrix")
    
    if "correlation_matrix" in risk_metrics:
        corr_data = risk_metrics["correlation_matrix"]
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_data["values"],
            x=corr_data["labels"],
            y=corr_data["labels"],
            colorscale='RdBu',
            zmid=0,
            text=corr_data["values"],
            texttemplate='%{text:.2f}',
            textfont={"size": 12}
        ))
        
        fig.update_layout(
            title="Asset Correlation Heatmap",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Analytics Tab
with tab5:
    st.markdown("### Portfolio Analytics")
    
    # Performance attribution
    st.markdown("#### Performance Attribution")
    
    attribution = api_client.get("/analytics/attribution") or {}
    
    if attribution:
        # Factor contributions
        factors = attribution.get("factors", {})
        
        if factors:
            fig = go.Figure()
            
            factor_names = list(factors.keys())
            contributions = list(factors.values())
            colors = ['green' if c > 0 else 'red' for c in contributions]
            
            fig.add_trace(go.Bar(
                x=contributions,
                y=factor_names,
                orientation='h',
                marker_color=colors
            ))
            
            fig.update_layout(
                title="Performance Attribution by Factor",
                xaxis_title="Contribution (%)",
                yaxis_title="Factor",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Strategy performance comparison
    st.markdown("#### Strategy Performance Comparison")
    
    strategies = api_client.get("/analytics/strategies") or {}
    
    if strategies and "comparison" in strategies:
        comparison_df = pd.DataFrame(strategies["comparison"])
        
        # Create subplot for different metrics
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Returns", "Sharpe Ratio", "Win Rate", "Max Drawdown")
        )
        
        # Returns comparison
        fig.add_trace(go.Bar(
            x=comparison_df['strategy'],
            y=comparison_df['total_return'],
            name='Total Return',
            marker_color='lightblue'
        ), row=1, col=1)
        
        # Sharpe ratio
        fig.add_trace(go.Bar(
            x=comparison_df['strategy'],
            y=comparison_df['sharpe_ratio'],
            name='Sharpe Ratio',
            marker_color='lightgreen'
        ), row=1, col=2)
        
        # Win rate
        fig.add_trace(go.Bar(
            x=comparison_df['strategy'],
            y=comparison_df['win_rate'],
            name='Win Rate',
            marker_color='lightcoral'
        ), row=2, col=1)
        
        # Max drawdown
        fig.add_trace(go.Bar(
            x=comparison_df['strategy'],
            y=comparison_df['max_drawdown'],
            name='Max Drawdown',
            marker_color='lightyellow'
        ), row=2, col=2)
        
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Time-based analysis
    st.markdown("#### Time-Based Performance Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Performance by day of week
        dow_data = api_client.get("/analytics/performance-by-dow") or {}
        
        if dow_data and "weekly_performance" in dow_data:
            # Extract days and returns from the structured response
            performance_data = dow_data["weekly_performance"]
            days = [item["day"] for item in performance_data]
            returns = [item["avg_return"] * 100 for item in performance_data]  # Convert to percentage
            
            fig = go.Figure(data=go.Bar(
                x=days,
                y=returns,
                marker_color=['green' if r > 0 else 'red' for r in returns]
            ))
            
            fig.update_layout(
                title="Average Returns by Day of Week",
                xaxis_title="Day",
                yaxis_title="Average Return (%)",
                height=350
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Performance by hour
        hour_data = api_client.get("/analytics/performance-by-hour") or {}
        
        if hour_data and "hourly_performance" in hour_data:
            # Extract hours and returns from the structured response
            performance_data = hour_data["hourly_performance"]
            hours = [item["hour"] for item in performance_data]
            returns = [item["avg_return"] * 100 for item in performance_data]  # Convert to percentage
            
            fig = go.Figure(data=go.Scatter(
                x=hours,
                y=returns,
                mode='lines+markers',
                line=dict(color='purple', width=2)
            ))
            
            fig.update_layout(
                title="Average Returns by Hour (UTC)",
                xaxis_title="Hour",
                yaxis_title="Average Return (%)",
                height=350
            )
            
            st.plotly_chart(fig, use_container_width=True)

# P&L Analysis Tab
with tab6:
    st.markdown("### Profit & Loss Analysis")
    
    pnl_data = api_client.get("/analytics/pnl-analysis") or {}
    
    if pnl_data:
        # P&L summary
        st.markdown("#### P&L Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        summary = pnl_data.get("summary", {})
        
        with col1:
            gross_profit = summary.get("gross_profit", 0)
            st.metric("Gross Profit", format_currency(gross_profit))
        
        with col2:
            gross_loss = summary.get("gross_loss", 0)
            st.metric("Gross Loss", format_currency(gross_loss))
        
        with col3:
            net_pnl = summary.get("net_pnl", 0)
            st.metric("Net P&L", format_currency(net_pnl))
        
        with col4:
            profit_factor = gross_profit / abs(gross_loss) if gross_loss != 0 else 0
            st.metric("Profit Factor", f"{profit_factor:.2f}")
        
        # P&L distribution
        st.markdown("#### P&L Distribution")
        
        if "distribution" in pnl_data:
            pnl_values = pnl_data["distribution"]
            
            fig = go.Figure()
            
            fig.add_trace(go.Histogram(
                x=pnl_values,
                nbinsx=50,
                marker_color='lightblue',
                opacity=0.7,
                name='P&L Distribution'
            ))
            
            # Add average line
            avg_pnl = np.mean(pnl_values)
            fig.add_vline(x=avg_pnl, line_dash="dash", line_color="red",
                         annotation_text=f"Avg: ${avg_pnl:,.0f}")
            
            # Add zero line
            fig.add_vline(x=0, line_dash="solid", line_color="black", line_width=1)
            
            fig.update_layout(
                title="Trade P&L Distribution",
                xaxis_title="P&L ($)",
                yaxis_title="Frequency",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Cumulative P&L
        st.markdown("#### Cumulative P&L")
        
        if "cumulative_pnl" in pnl_data:
            cum_data = pd.DataFrame(pnl_data["cumulative_pnl"])
            cum_data['timestamp'] = pd.to_datetime(cum_data['timestamp'])
            
            fig = go.Figure()
            
            # Cumulative P&L line
            fig.add_trace(go.Scatter(
                x=cum_data['timestamp'],
                y=cum_data['cumulative_pnl'],
                mode='lines',
                name='Cumulative P&L',
                line=dict(color='blue', width=3),
                fill='tozeroy',
                fillcolor='rgba(0, 100, 200, 0.1)'
            ))
            
            # Add individual trades as markers
            if 'trade_pnl' in cum_data.columns:
                profits = cum_data[cum_data['trade_pnl'] > 0]
                losses = cum_data[cum_data['trade_pnl'] < 0]
                
                if not profits.empty:
                    fig.add_trace(go.Scatter(
                        x=profits['timestamp'],
                        y=profits['cumulative_pnl'],
                        mode='markers',
                        name='Profitable Trades',
                        marker=dict(color='green', size=8)
                    ))
                
                if not losses.empty:
                    fig.add_trace(go.Scatter(
                        x=losses['timestamp'],
                        y=losses['cumulative_pnl'],
                        mode='markers',
                        name='Losing Trades',
                        marker=dict(color='red', size=8)
                    ))
            
            fig.update_layout(
                title="Cumulative P&L Over Time",
                xaxis_title="Date",
                yaxis_title="Cumulative P&L ($)",
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Best and worst trades
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Best Trades")
            best_trades = pnl_data.get("best_trades", [])
            
            if best_trades:
                best_df = pd.DataFrame(best_trades)
                best_df['timestamp'] = pd.to_datetime(best_df['timestamp']).dt.strftime('%Y-%m-%d')
                best_df['pnl'] = best_df['pnl'].apply(lambda x: format_currency(x))
                best_df['pnl_percent'] = best_df['pnl_percent'].apply(lambda x: f"{x:.2f}%")
                
                st.dataframe(best_df[['timestamp', 'symbol', 'side', 'pnl', 'pnl_percent']], 
                           use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("#### Worst Trades")
            worst_trades = pnl_data.get("worst_trades", [])
            
            if worst_trades:
                worst_df = pd.DataFrame(worst_trades)
                worst_df['timestamp'] = pd.to_datetime(worst_df['timestamp']).dt.strftime('%Y-%m-%d')
                worst_df['pnl'] = worst_df['pnl'].apply(lambda x: format_currency(x))
                worst_df['pnl_percent'] = worst_df['pnl_percent'].apply(lambda x: f"{x:.2f}%")
                
                st.dataframe(worst_df[['timestamp', 'symbol', 'side', 'pnl', 'pnl_percent']], 
                           use_container_width=True, hide_index=True)
    else:
        st.info("No P&L data available. Complete some trades to see profit and loss analysis.")

# Add auto-refresh option
if st.sidebar.checkbox("Auto-refresh (30s)", value=False):
    import time
    time.sleep(30)
    st.rerun()

