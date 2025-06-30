import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import json
import os
from datetime import datetime, timedelta
import time
import numpy as np

st.set_page_config(
    page_title="BTC Trading System - Enhanced",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1e1e1e;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #333;
        margin: 10px 0;
    }
    .signal-buy { color: #00ff00; }
    .signal-sell { color: #ff0000; }
    .signal-hold { color: #ffa500; }
    .profit { color: #00ff00; }
    .loss { color: #ff0000; }
</style>
""", unsafe_allow_html=True)

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8080")

@st.cache_data(ttl=60)
def fetch_api_data(endpoint):
    try:
        response = requests.get(f"{API_BASE_URL}{endpoint}", timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot connect to API server at {API_BASE_URL}")
        return None
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

def post_api_data(endpoint, data):
    try:
        response = requests.post(f"{API_BASE_URL}{endpoint}", json=data, timeout=30)
        return response.json()
    except Exception as e:
        st.error(f"Error posting data: {str(e)}")
        return None

def format_currency(value):
    """Format value as currency"""
    if value >= 0:
        return f"${value:,.2f}"
    else:
        return f"-${abs(value):,.2f}"

def format_percentage(value):
    """Format value as percentage with color"""
    if value > 0:
        return f'<span class="profit">+{value:.2%}</span>'
    elif value < 0:
        return f'<span class="loss">{value:.2%}</span>'
    else:
        return f'{value:.2%}'

def create_enhanced_candlestick_chart(btc_data):
    """Create comprehensive candlestick chart with technical indicators"""
    if not btc_data or 'data' not in btc_data:
        return None
    
    df = pd.DataFrame(btc_data['data'])
    if df.empty:
        return None
        
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Create subplots
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('BTC Price with Indicators', 'Volume & OBV', 'RSI & Stochastic', 'MACD'),
        row_heights=[0.5, 0.2, 0.15, 0.15]
    )
    
    # Main price chart
    fig.add_trace(
        go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="BTC-USD"
        ),
        row=1, col=1
    )
    
    # Add moving averages
    if 'sma_20' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['sma_20'],
                name='SMA 20',
                line=dict(color='orange', width=1)
            ),
            row=1, col=1
        )
    
    if 'sma_50' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['sma_50'],
                name='SMA 50',
                line=dict(color='blue', width=1)
            ),
            row=1, col=1
        )
    
    # Add Bollinger Bands
    if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['bb_upper'],
                name='BB Upper',
                line=dict(color='gray', width=1, dash='dash')
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['bb_lower'],
                name='BB Lower',
                line=dict(color='gray', width=1, dash='dash'),
                fill='tonexty',
                fillcolor='rgba(128,128,128,0.1)'
            ),
            row=1, col=1
        )
    
    # Add VWAP
    if 'vwap' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['vwap'],
                name='VWAP',
                line=dict(color='purple', width=1, dash='dot')
            ),
            row=1, col=1
        )
    
    # Volume chart
    colors = ['red' if row['close'] < row['open'] else 'green' for _, row in df.iterrows()]
    fig.add_trace(
        go.Bar(
            x=df['timestamp'],
            y=df['volume'],
            name='Volume',
            marker_color=colors,
            opacity=0.5
        ),
        row=2, col=1
    )
    
    # OBV on secondary axis
    if 'obv' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['obv'],
                name='OBV',
                line=dict(color='cyan', width=1),
                yaxis='y5'
            ),
            row=2, col=1
        )
    
    # RSI
    if 'rsi' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['rsi'],
                name='RSI',
                line=dict(color='yellow', width=1)
            ),
            row=3, col=1
        )
        # Add RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    # Stochastic
    if 'stoch_k' in df.columns and 'stoch_d' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['stoch_k'],
                name='Stoch %K',
                line=dict(color='lime', width=1)
            ),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['stoch_d'],
                name='Stoch %D',
                line=dict(color='lightgreen', width=1, dash='dash')
            ),
            row=3, col=1
        )
    
    # MACD
    if 'macd' in df.columns and 'macd_signal' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['macd'],
                name='MACD',
                line=dict(color='blue', width=1)
            ),
            row=4, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['macd_signal'],
                name='Signal',
                line=dict(color='red', width=1)
            ),
            row=4, col=1
        )
        # MACD histogram
        macd_hist = df['macd'] - df['macd_signal']
        colors = ['green' if val > 0 else 'red' for val in macd_hist]
        fig.add_trace(
            go.Bar(
                x=df['timestamp'],
                y=macd_hist,
                name='MACD Hist',
                marker_color=colors,
                opacity=0.3
            ),
            row=4, col=1
        )
    
    # Update layout
    fig.update_layout(
        title="BTC Technical Analysis Dashboard",
        xaxis_rangeslider_visible=False,
        height=1000,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update y-axis labels
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="RSI/Stoch", row=3, col=1)
    fig.update_yaxes(title_text="MACD", row=4, col=1)
    
    return fig

def create_portfolio_metrics_cards(metrics):
    """Create metric cards for portfolio overview"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total P&L",
            format_currency(metrics.get('total_pnl', 0)),
            delta=f"{metrics.get('total_pnl', 0) / max(metrics.get('total_invested', 1), 1):.2%}"
        )
        st.metric(
            "Realized P&L",
            format_currency(metrics.get('realized_pnl', 0))
        )
    
    with col2:
        st.metric(
            "Unrealized P&L",
            format_currency(metrics.get('unrealized_pnl', 0)),
            delta=f"{metrics.get('unrealized_pnl', 0) / max(metrics.get('current_value', 1) - metrics.get('unrealized_pnl', 0), 1):.2%}" if metrics.get('current_value', 0) > 0 else None
        )
        st.metric(
            "Portfolio Value",
            format_currency(metrics.get('current_value', 0))
        )
    
    with col3:
        st.metric(
            "Win Rate",
            f"{metrics.get('win_rate', 0):.1%}",
            delta="Good" if metrics.get('win_rate', 0) > 0.5 else "Poor"
        )
        st.metric(
            "Sharpe Ratio",
            f"{metrics.get('sharpe_ratio', 0):.2f}",
            delta="Good" if metrics.get('sharpe_ratio', 0) > 1 else "Low"
        )
    
    with col4:
        st.metric(
            "Max Drawdown",
            f"{metrics.get('max_drawdown', 0):.1%}",
            delta="High Risk" if metrics.get('max_drawdown', 0) < -0.2 else "Controlled"
        )
        st.metric(
            "Total Trades",
            metrics.get('total_trades', 0)
        )

def main():
    st.title("₿ Enhanced BTC Trading System")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("Navigation")
        page = st.selectbox(
            "Select Page",
            ["Dashboard", "Trading", "Portfolio Analytics", "Technical Analysis", 
             "Signals & Factors", "Trade History", "Risk Management", "Configuration"]
        )
        
        st.markdown("---")
        
        # API Status
        with st.spinner("Checking API connection..."):
            api_status = fetch_api_data("/")
        
        if api_status:
            st.success("✅ API Connected")
            st.caption(f"Version: {api_status.get('version', 'Unknown')}")
        else:
            st.error("❌ API Disconnected")
        
        # Quick Stats
        st.markdown("---")
        st.subheader("Quick Stats")
        
        # Get current price from dedicated endpoint
        try:
            price_data = fetch_api_data("/price/current")
            if price_data and price_data.get('price'):
                current_price = price_data['price']
                st.metric("BTC Price", format_currency(current_price))
                st.caption(f"Source: {price_data.get('source', 'live')}")
            else:
                # Fallback to metrics endpoint
                metrics = fetch_api_data("/portfolio/metrics")
                if metrics and metrics.get('current_btc_price'):
                    st.metric("BTC Price", format_currency(metrics['current_btc_price']))
        except:
            st.metric("BTC Price", "Loading...")
        
        # Portfolio P&L
        metrics = fetch_api_data("/portfolio/metrics")
        if metrics:
            total_pnl = metrics.get('total_pnl', 0)
            color = "profit" if total_pnl >= 0 else "loss"
            st.markdown(f"Total P&L: <span class='{color}'>{format_currency(total_pnl)}</span>", 
                       unsafe_allow_html=True)
    
    # Main content based on selected page
    if page == "Dashboard":
        show_dashboard()
    elif page == "Trading":
        show_trading()
    elif page == "Portfolio Analytics":
        show_portfolio_analytics()
    elif page == "Technical Analysis":
        show_technical_analysis()
    elif page == "Signals & Factors":
        show_signals_factors()
    elif page == "Trade History":
        show_trade_history()
    elif page == "Risk Management":
        show_risk_management()
    elif page == "Configuration":
        show_configuration()

def show_dashboard():
    """Enhanced dashboard with comprehensive overview"""
    st.header("📊 Trading Dashboard")
    
    # Fetch all required data
    with st.spinner("Loading dashboard data..."):
        portfolio_metrics = fetch_api_data("/portfolio/metrics")
        latest_signal = fetch_api_data("/signals/latest")
        btc_data = fetch_api_data("/market/btc-data?period=7d&interval=1h")
        technical_data = fetch_api_data("/analytics/technical")
    
    # Portfolio Metrics Cards
    if portfolio_metrics:
        st.subheader("Portfolio Overview")
        create_portfolio_metrics_cards(portfolio_metrics)
    
    # Signal and Technical Overview
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Price Chart
        st.subheader("BTC Price & Technical Indicators")
        if btc_data:
            fig = create_enhanced_candlestick_chart(btc_data)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Latest Signal
        st.subheader("Latest Signal")
        if latest_signal:
            signal = latest_signal['signal']
            confidence = latest_signal['confidence']
            predicted_price = latest_signal['predicted_price']
            
            signal_color = {
                'buy': '#00FF00',
                'sell': '#FF0000', 
                'hold': '#FFA500'
            }.get(signal, '#808080')
            
            st.markdown(f"""
            <div style="padding: 20px; border-radius: 10px; background-color: {signal_color}20; 
                        border: 2px solid {signal_color}; text-align: center;">
                <h2 style="color: {signal_color}; margin: 0;">{signal.upper()}</h2>
                <p style="margin: 10px 0;">Confidence: {confidence:.1%}</p>
                <p style="margin: 10px 0;">Target: {format_currency(predicted_price)}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Signal Factors
            if 'factors' in latest_signal and latest_signal['factors']:
                st.markdown("### Key Factors")
                factors = latest_signal['factors']
                
                # Display top factors
                factor_items = [(k, v) for k, v in factors.items() if isinstance(v, (int, float))]
                factor_items.sort(key=lambda x: abs(x[1]), reverse=True)
                
                for factor, value in factor_items[:5]:
                    if isinstance(value, float):
                        color = "green" if value > 0 else "red" if value < 0 else "gray"
                        st.markdown(f"**{factor.replace('_', ' ').title()}**: "
                                  f"<span style='color: {color}'>{value:.3f}</span>", 
                                  unsafe_allow_html=True)
        
        # Technical Summary
        st.subheader("Technical Summary")
        if technical_data and 'current_indicators' in technical_data:
            indicators = technical_data['current_indicators']
            trend = technical_data['trend_analysis']['overall_trend']
            
            trend_color = {
                'bullish': 'green',
                'bearish': 'red',
                'neutral': 'gray'
            }.get(trend, 'gray')
            
            st.markdown(f"**Trend**: <span style='color: {trend_color}'>{trend.upper()}</span>", 
                       unsafe_allow_html=True)
            
            # Key indicators
            st.markdown(f"**RSI**: {indicators['rsi']:.1f} ({indicators['rsi_signal']})")
            st.markdown(f"**Fear & Greed**: {indicators['fear_greed']:.0f}")
            
            # Momentum
            momentum = technical_data['momentum_analysis']
            st.markdown(f"**Volume**: {momentum['volume_trend']}")

def show_trading():
    """Enhanced trading interface"""
    st.header("💰 Trading Interface")
    
    # Current positions
    positions = fetch_api_data("/positions/")
    if positions:
        st.subheader("Current Positions")
        positions_df = pd.DataFrame(positions)
        
        if not positions_df.empty:
            # Format columns
            for col in ['current_value', 'unrealized_pnl', 'avg_buy_price']:
                if col in positions_df.columns:
                    positions_df[col] = positions_df[col].apply(format_currency)
            
            if 'unrealized_pnl_pct' in positions_df.columns:
                positions_df['unrealized_pnl_pct'] = positions_df['unrealized_pnl_pct'].apply(
                    lambda x: format_percentage(x)
                )
            
            st.dataframe(
                positions_df[['lot_id', 'symbol', 'total_size', 'avg_buy_price', 
                            'current_value', 'unrealized_pnl', 'unrealized_pnl_pct']],
                use_container_width=True
            )
    
    # Trading Form
    st.subheader("Execute Trade")
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.form("trade_form"):
            st.subheader("New Trade")
            
            trade_type = st.selectbox("Trade Type", ["buy", "sell", "hold"])
            
            # Get current price
            tech_data = fetch_api_data("/analytics/technical")
            current_price = tech_data['current_indicators']['price'] if tech_data else 45000
            
            price = st.number_input("Price ($)", min_value=1.0, value=float(current_price), step=0.01)
            size = st.number_input("Size (BTC)", min_value=0.001, value=0.1, step=0.001, format="%.3f")
            
            # Lot selection for sells
            lot_id = None
            if trade_type == "sell" and positions:
                lot_ids = ["New Lot"] + [p['lot_id'] for p in positions]
                selected_lot = st.selectbox("Lot ID", lot_ids)
                lot_id = None if selected_lot == "New Lot" else selected_lot
            else:
                lot_id = st.text_input("Lot ID (optional)")
            
            # Trade value preview
            trade_value = price * size
            st.info(f"Trade Value: {format_currency(trade_value)}")
            
            submitted = st.form_submit_button("Execute Trade", type="primary")
            
            if submitted:
                trade_data = {
                    "symbol": "BTC-USD",
                    "trade_type": trade_type,
                    "price": price,
                    "size": size,
                    "lot_id": lot_id if lot_id else None
                }
                
                result = post_api_data("/trades/", trade_data)
                if result and result.get('status') == 'success':
                    st.success(f"✅ Trade executed successfully! Trade ID: {result['trade_id']}")
                    st.experimental_rerun()
                else:
                    st.error("❌ Trade execution failed")
    
    with col2:
        # Limit Orders
        st.subheader("Limit Orders")
        
        with st.form("limit_form"):
            limit_type = st.selectbox("Limit Type", ["stop_loss", "take_profit", "buy_limit", "sell_limit"])
            limit_price = st.number_input("Limit Price ($)", min_value=1.0, value=float(current_price * 0.95), step=0.01)
            limit_size = st.number_input("Size (BTC)", min_value=0.001, value=0.1, step=0.001, format="%.3f")
            
            submitted_limit = st.form_submit_button("Create Limit Order")
            
            if submitted_limit:
                limit_data = {
                    "symbol": "BTC-USD",
                    "limit_type": limit_type,
                    "price": limit_price,
                    "size": limit_size
                }
                
                result = post_api_data("/limits/", limit_data)
                if result and result.get('status') == 'success':
                    st.success(f"✅ Limit order created! ID: {result['limit_id']}")
                    st.experimental_rerun()
        
        # Active Limits
        limits = fetch_api_data("/limits/")
        if limits:
            st.subheader("Active Limit Orders")
            limits_df = pd.DataFrame(limits)
            if not limits_df.empty:
                st.dataframe(limits_df[['id', 'limit_type', 'price', 'size', 'created_at']], 
                           use_container_width=True)

def show_portfolio_analytics():
    """Comprehensive portfolio analytics page"""
    st.header("📈 Portfolio Analytics")
    
    # Fetch analytics data
    with st.spinner("Loading analytics..."):
        metrics = fetch_api_data("/portfolio/metrics")
        pnl_data = fetch_api_data("/analytics/pnl")
        performance_data = fetch_api_data("/analytics/performance")
    
    if not metrics:
        st.warning("No portfolio data available")
        return
    
    # Performance Overview
    st.subheader("Performance Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Return", 
                 f"{(metrics['total_pnl'] / max(metrics['total_invested'], 1)):.2%}",
                 delta=format_currency(metrics['total_pnl']))
        st.metric("Sharpe Ratio", f"{performance_data.get('sharpe_ratio', 0):.2f}")
    
    with col2:
        st.metric("Win Rate", f"{metrics['win_rate']:.1%}")
        st.metric("Profit Factor", f"{performance_data.get('profit_factor', 0):.2f}")
    
    with col3:
        st.metric("Max Drawdown", f"{performance_data.get('max_drawdown', 0):.1%}")
        st.metric("Sortino Ratio", f"{performance_data.get('sortino_ratio', 0):.2f}")
    
    # P&L Charts
    if pnl_data:
        col1, col2 = st.columns(2)
        
        with col1:
            # Daily P&L
            st.subheader("Daily P&L")
            daily_df = pd.DataFrame(pnl_data['daily_pnl'])
            if not daily_df.empty:
                fig = go.Figure()
                colors = ['green' if pnl > 0 else 'red' for pnl in daily_df['pnl']]
                fig.add_trace(go.Bar(
                    x=daily_df['date'],
                    y=daily_df['pnl'],
                    marker_color=colors,
                    name='Daily P&L'
                ))
                fig.update_layout(
                    title="Daily Profit/Loss",
                    xaxis_title="Date",
                    yaxis_title="P&L ($)",
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Cumulative P&L
            st.subheader("Cumulative P&L")
            cum_df = pd.DataFrame(pnl_data['cumulative_pnl'])
            if not cum_df.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=cum_df['date'],
                    y=cum_df['pnl'],
                    mode='lines',
                    fill='tozeroy',
                    line=dict(color='blue', width=2),
                    name='Cumulative P&L'
                ))
                fig.update_layout(
                    title="Cumulative Profit/Loss",
                    xaxis_title="Date",
                    yaxis_title="Cumulative P&L ($)",
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Returns Distribution
    if performance_data and 'returns' in performance_data:
        st.subheader("Returns Distribution")
        returns = performance_data['returns']
        
        if returns:
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=returns,
                nbinsx=30,
                name='Returns',
                marker_color='blue',
                opacity=0.7
            ))
            fig.add_vline(x=0, line_dash="dash", line_color="black")
            fig.add_vline(x=np.mean(returns), line_dash="dash", line_color="green", 
                         annotation_text="Mean")
            fig.update_layout(
                title="Daily Returns Distribution",
                xaxis_title="Return (%)",
                yaxis_title="Frequency",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # P&L by Lot
    if pnl_data and 'by_lot' in pnl_data:
        st.subheader("P&L by Trading Lot")
        lot_pnl = pnl_data['by_lot']
        
        if lot_pnl:
            lot_df = pd.DataFrame([
                {'Lot ID': k, 'P&L': v} for k, v in lot_pnl.items()
            ])
            lot_df = lot_df.sort_values('P&L', ascending=False)
            
            fig = go.Figure()
            colors = ['green' if pnl > 0 else 'red' for pnl in lot_df['P&L']]
            fig.add_trace(go.Bar(
                x=lot_df['Lot ID'],
                y=lot_df['P&L'],
                marker_color=colors,
                text=[format_currency(pnl) for pnl in lot_df['P&L']],
                textposition='auto'
            ))
            fig.update_layout(
                title="Profit/Loss by Trading Lot",
                xaxis_title="Lot ID",
                yaxis_title="P&L ($)",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

def show_technical_analysis():
    """Technical analysis page with all indicators"""
    st.header("📊 Technical Analysis")
    
    # Time period selector
    col1, col2 = st.columns([2, 1])
    with col1:
        period = st.selectbox("Time Period", ["1d", "7d", "1mo", "3mo", "6mo", "1y"], index=2)
    with col2:
        interval = st.selectbox("Interval", ["1h", "4h", "1d"], index=0)
    
    # Fetch data
    with st.spinner("Loading market data..."):
        btc_data = fetch_api_data(f"/market/btc-data?period={period}&interval={interval}")
        tech_analysis = fetch_api_data("/analytics/technical")
    
    if btc_data:
        # Main chart
        fig = create_enhanced_candlestick_chart(btc_data)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    # Current Technical Readings
    if tech_analysis:
        st.subheader("Current Technical Indicators")
        
        indicators = tech_analysis['current_indicators']
        trend = tech_analysis['trend_analysis']
        momentum = tech_analysis['momentum_analysis']
        
        # Create indicator cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("### Oscillators")
            st.metric("RSI", f"{indicators['rsi']:.1f}", delta=indicators['rsi_signal'])
            st.metric("Stochastic K", f"{indicators['stochastic_k']:.1f}", 
                     delta=indicators['stochastic_signal'])
            st.metric("MFI", f"{indicators['mfi']:.1f}")
        
        with col2:
            st.markdown("### Trend")
            st.metric("Overall Trend", trend['overall_trend'].upper())
            st.metric("Price vs SMA20", trend['price_vs_sma20'].upper())
            st.metric("Price vs SMA50", trend['price_vs_sma50'].upper())
        
        with col3:
            st.markdown("### Momentum")
            st.metric("MACD", f"{indicators['macd']:.2f}")
            st.metric("MACD Signal", f"{indicators['macd_signal_line']:.2f}")
            st.metric("ROC", f"{momentum['roc']:.2f}%")
        
        with col4:
            st.markdown("### Market Structure")
            st.metric("Fear & Greed", f"{indicators['fear_greed']:.0f}")
            st.metric("BB Position", f"{indicators['bollinger_position']:.2f}", 
                     delta=indicators['bollinger_signal'])
            st.metric("Volume Trend", momentum['volume_trend'].upper())
        
        # Detailed Analysis
        st.subheader("Signal Analysis")
        
        # Create a summary table
        signal_data = []
        
        # RSI Signal
        rsi_val = indicators['rsi']
        if rsi_val < 30:
            signal_data.append({"Indicator": "RSI", "Value": f"{rsi_val:.1f}", 
                              "Signal": "Oversold", "Action": "Consider Buy"})
        elif rsi_val > 70:
            signal_data.append({"Indicator": "RSI", "Value": f"{rsi_val:.1f}", 
                              "Signal": "Overbought", "Action": "Consider Sell"})
        else:
            signal_data.append({"Indicator": "RSI", "Value": f"{rsi_val:.1f}", 
                              "Signal": "Neutral", "Action": "Hold"})
        
        # Bollinger Bands
        bb_pos = indicators['bollinger_position']
        if bb_pos < 0.2:
            signal_data.append({"Indicator": "Bollinger Bands", "Value": f"{bb_pos:.2f}", 
                              "Signal": "Near Lower Band", "Action": "Potential Buy"})
        elif bb_pos > 0.8:
            signal_data.append({"Indicator": "Bollinger Bands", "Value": f"{bb_pos:.2f}", 
                              "Signal": "Near Upper Band", "Action": "Potential Sell"})
        else:
            signal_data.append({"Indicator": "Bollinger Bands", "Value": f"{bb_pos:.2f}", 
                              "Signal": "Middle Range", "Action": "Neutral"})
        
        # MACD
        macd_diff = indicators['macd'] - indicators['macd_signal_line']
        if macd_diff > 0:
            signal_data.append({"Indicator": "MACD", "Value": f"{macd_diff:.2f}", 
                              "Signal": "Bullish Cross", "Action": "Buy Signal"})
        else:
            signal_data.append({"Indicator": "MACD", "Value": f"{macd_diff:.2f}", 
                              "Signal": "Bearish Cross", "Action": "Sell Signal"})
        
        # Trend
        if trend['overall_trend'] == 'bullish':
            signal_data.append({"Indicator": "Trend Analysis", "Value": "Bullish", 
                              "Signal": "Uptrend", "Action": "Follow Trend"})
        elif trend['overall_trend'] == 'bearish':
            signal_data.append({"Indicator": "Trend Analysis", "Value": "Bearish", 
                              "Signal": "Downtrend", "Action": "Caution"})
        else:
            signal_data.append({"Indicator": "Trend Analysis", "Value": "Neutral", 
                              "Signal": "Sideways", "Action": "Wait"})
        
        signal_df = pd.DataFrame(signal_data)
        st.dataframe(signal_df, use_container_width=True)

def show_signals_factors():
    """Detailed signals and factors analysis"""
    st.header("🤖 Trading Signals & Factor Analysis")
    
    # Latest Signal
    latest_signal = fetch_api_data("/signals/latest")
    
    if latest_signal:
        # Signal Overview
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            signal = latest_signal['signal']
            confidence = latest_signal['confidence']
            predicted_price = latest_signal['predicted_price']
            
            signal_color = {
                'buy': '#00FF00',
                'sell': '#FF0000', 
                'hold': '#FFA500'
            }.get(signal, '#808080')
            
            st.markdown(f"""
            <div style="padding: 30px; border-radius: 15px; background-color: {signal_color}20; 
                        border: 3px solid {signal_color}; text-align: center; margin: 20px 0;">
                <h1 style="color: {signal_color}; margin: 0;">{signal.upper()}</h1>
                <h3 style="margin: 10px 0;">Confidence: {confidence:.1%}</h3>
                <h3 style="margin: 10px 0;">Predicted Price: {format_currency(predicted_price)}</h3>
                <p style="margin: 10px 0; color: gray;">
                    Generated: {latest_signal['timestamp'].replace('T', ' ').split('.')[0]}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Factor Analysis
        if 'factors' in latest_signal and latest_signal['factors']:
            st.subheader("Factor Breakdown")
            
            factors = latest_signal['factors']
            
            # Separate numeric and non-numeric factors
            numeric_factors = {k: v for k, v in factors.items() if isinstance(v, (int, float))}
            other_factors = {k: v for k, v in factors.items() if not isinstance(v, (int, float))}
            
            # Create factor visualization
            if numeric_factors:
                # Sort by absolute value
                sorted_factors = sorted(numeric_factors.items(), key=lambda x: abs(x[1]), reverse=True)
                
                # Create horizontal bar chart
                factor_names = [f[0].replace('_', ' ').title() for f in sorted_factors]
                factor_values = [f[1] for f in sorted_factors]
                colors = ['green' if v > 0 else 'red' for v in factor_values]
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    y=factor_names,
                    x=factor_values,
                    orientation='h',
                    marker_color=colors,
                    text=[f"{v:.3f}" for v in factor_values],
                    textposition='outside'
                ))
                
                fig.update_layout(
                    title="Signal Factor Contributions",
                    xaxis_title="Factor Strength",
                    yaxis_title="",
                    height=max(400, len(factor_names) * 30),
                    showlegend=False
                )
                
                # Add center line
                fig.add_vline(x=0, line_dash="dash", line_color="gray")
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Display other factors
            if other_factors:
                st.subheader("Additional Factors")
                for key, value in other_factors.items():
                    st.write(f"**{key.replace('_', ' ').title()}**: {value}")
    
    # Historical Signals
    st.subheader("Signal History")
    
    signals = fetch_api_data("/signals/history?limit=20")
    if signals:
        signals_df = pd.DataFrame(signals)
        
        # Format columns
        if 'price_prediction' in signals_df.columns:
            signals_df['price_prediction'] = signals_df['price_prediction'].apply(format_currency)
        
        if 'confidence' in signals_df.columns:
            signals_df['confidence'] = signals_df['confidence'].apply(lambda x: f"{x:.1%}")
        
        # Color code signals
        def style_signal(val):
            if val == 'buy':
                return 'color: green'
            elif val == 'sell':
                return 'color: red'
            else:
                return 'color: orange'
        
        styled_df = signals_df[['timestamp', 'signal', 'confidence', 'price_prediction']].style.applymap(
            style_signal, subset=['signal']
        )
        
        st.dataframe(styled_df, use_container_width=True)
        
        # Signal performance analysis
        if len(signals_df) > 5:
            st.subheader("Signal Performance")
            
            # Count signals
            signal_counts = signals_df['signal'].value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Signal distribution
                fig = go.Figure(data=[go.Pie(
                    labels=signal_counts.index,
                    values=signal_counts.values,
                    marker_colors=['green', 'red', 'orange']
                )])
                fig.update_layout(title="Signal Distribution")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Average confidence by signal
                # Fix: Ensure confidence is numeric before aggregation
                signals_df_copy = signals_df.copy()
                # Convert confidence to numeric, removing any % signs
                if signals_df_copy['confidence'].dtype == 'object':
                    signals_df_copy['confidence'] = signals_df_copy['confidence'].str.rstrip('%').astype(float) / 100
                
                avg_conf = signals_df_copy.groupby('signal')['confidence'].mean()
                
                fig = go.Figure(data=[go.Bar(
                    x=avg_conf.index,
                    y=avg_conf.values,
                    marker_color=['green' if x == 'buy' else 'red' if x == 'sell' else 'orange' 
                                 for x in avg_conf.index]
                )])
                fig.update_layout(
                    title="Average Confidence by Signal Type",
                    yaxis_title="Confidence",
                    xaxis_title="Signal"
                )
                st.plotly_chart(fig, use_container_width=True)

def show_trade_history():
    """Trade history and analysis"""
    st.header("📜 Trade History")
    
    # Fetch trades
    trades = fetch_api_data("/trades/?limit=200")
    
    if not trades:
        st.warning("No trades found")
        return
    
    trades_df = pd.DataFrame(trades)
    trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
    
    # Summary statistics
    st.subheader("Trading Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Trades", len(trades_df))
        buy_count = len(trades_df[trades_df['trade_type'] == 'buy'])
        st.metric("Buy Orders", buy_count)
    
    with col2:
        total_volume = trades_df['size'].sum()
        st.metric("Total Volume", f"{total_volume:.3f} BTC")
        sell_count = len(trades_df[trades_df['trade_type'] == 'sell'])
        st.metric("Sell Orders", sell_count)
    
    with col3:
        avg_trade_size = trades_df['size'].mean()
        st.metric("Avg Trade Size", f"{avg_trade_size:.3f} BTC")
        hold_count = len(trades_df[trades_df['trade_type'] == 'hold'])
        st.metric("Hold Orders", hold_count)
    
    with col4:
        days_trading = (trades_df['timestamp'].max() - trades_df['timestamp'].min()).days
        st.metric("Days Trading", days_trading)
        unique_lots = trades_df['lot_id'].nunique()
        st.metric("Trading Lots", unique_lots)
    
    # Trade distribution over time
    st.subheader("Trading Activity")
    
    # Group by date and trade type
    daily_trades = trades_df.groupby([trades_df['timestamp'].dt.date, 'trade_type']).size().unstack(fill_value=0)
    
    fig = go.Figure()
    for trade_type in ['buy', 'sell', 'hold']:
        if trade_type in daily_trades.columns:
            fig.add_trace(go.Scatter(
                x=daily_trades.index,
                y=daily_trades[trade_type],
                mode='lines',
                name=trade_type.capitalize(),
                stackgroup='one'
            ))
    
    fig.update_layout(
        title="Daily Trading Activity by Type",
        xaxis_title="Date",
        yaxis_title="Number of Trades",
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Recent trades table
    st.subheader("Recent Trades")
    
    # Format the dataframe
    display_df = trades_df.copy()
    display_df['price'] = display_df['price'].apply(format_currency)
    display_df['trade_value'] = (trades_df['price'] * trades_df['size']).apply(format_currency)
    display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Add trade type styling
    def style_trade_type(val):
        if val == 'buy':
            return 'background-color: rgba(0, 255, 0, 0.2)'
        elif val == 'sell':
            return 'background-color: rgba(255, 0, 0, 0.2)'
        else:
            return 'background-color: rgba(255, 165, 0, 0.2)'
    
    styled_df = display_df[['timestamp', 'trade_type', 'price', 'size', 'trade_value', 'lot_id']].head(50).style.applymap(
        style_trade_type, subset=['trade_type']
    )
    
    st.dataframe(styled_df, use_container_width=True)
    
    # Trade analysis by lot
    st.subheader("Performance by Trading Lot")
    
    lot_analysis = []
    for lot_id in trades_df['lot_id'].unique():
        lot_trades = trades_df[trades_df['lot_id'] == lot_id]
        
        buy_trades = lot_trades[lot_trades['trade_type'] == 'buy']
        sell_trades = lot_trades[lot_trades['trade_type'] == 'sell']
        
        if not buy_trades.empty:
            avg_buy = (buy_trades['price'] * buy_trades['size']).sum() / buy_trades['size'].sum()
            total_bought = buy_trades['size'].sum()
            
            if not sell_trades.empty:
                avg_sell = (sell_trades['price'] * sell_trades['size']).sum() / sell_trades['size'].sum()
                total_sold = sell_trades['size'].sum()
                
                realized_pnl = (avg_sell - avg_buy) * total_sold
                pnl_pct = (avg_sell - avg_buy) / avg_buy
                
                lot_analysis.append({
                    'Lot ID': lot_id,
                    'Avg Buy': avg_buy,
                    'Avg Sell': avg_sell,
                    'Size Traded': min(total_bought, total_sold),
                    'Realized P&L': realized_pnl,
                    'P&L %': pnl_pct,
                    'Status': 'Closed' if total_bought <= total_sold else 'Open'
                })
            else:
                lot_analysis.append({
                    'Lot ID': lot_id,
                    'Avg Buy': avg_buy,
                    'Avg Sell': None,
                    'Size Traded': 0,
                    'Realized P&L': 0,
                    'P&L %': 0,
                    'Status': 'Open'
                })
    
    if lot_analysis:
        lot_df = pd.DataFrame(lot_analysis)
        lot_df = lot_df.sort_values('Realized P&L', ascending=False)
        
        # Format columns
        for col in ['Avg Buy', 'Avg Sell', 'Realized P&L']:
            lot_df[col] = lot_df[col].apply(lambda x: format_currency(x) if pd.notna(x) else '-')
        
        lot_df['P&L %'] = lot_df['P&L %'].apply(lambda x: format_percentage(x))
        lot_df['Size Traded'] = lot_df['Size Traded'].apply(lambda x: f"{x:.3f}")
        
        st.dataframe(lot_df, use_container_width=True)

def show_risk_management():
    """Risk management dashboard"""
    st.header("⚠️ Risk Management")
    
    # Fetch data
    metrics = fetch_api_data("/portfolio/metrics")
    performance = fetch_api_data("/analytics/performance")
    positions = fetch_api_data("/positions/")
    tech_data = fetch_api_data("/analytics/technical")
    
    if not metrics:
        st.warning("No portfolio data available")
        return
    
    # Risk Overview
    st.subheader("Risk Metrics Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        max_dd = performance.get('max_drawdown', 0) if performance else 0
        dd_color = "red" if max_dd < -0.2 else "orange" if max_dd < -0.1 else "green"
        st.metric("Max Drawdown", f"{max_dd:.1%}")
        st.markdown(f"<p style='color: {dd_color}'>Risk Level: "
                   f"{'High' if max_dd < -0.2 else 'Medium' if max_dd < -0.1 else 'Low'}</p>", 
                   unsafe_allow_html=True)
    
    with col2:
        volatility = performance.get('volatility', 0) if performance else 0
        ann_vol = volatility * np.sqrt(252)
        vol_color = "red" if ann_vol > 1 else "orange" if ann_vol > 0.5 else "green"
        st.metric("Annual Volatility", f"{ann_vol:.1%}")
        st.markdown(f"<p style='color: {vol_color}'>Volatility: "
                   f"{'High' if ann_vol > 1 else 'Medium' if ann_vol > 0.5 else 'Low'}</p>", 
                   unsafe_allow_html=True)
    
    with col3:
        sharpe = performance.get('sharpe_ratio', 0) if performance else 0
        sharpe_color = "green" if sharpe > 1 else "orange" if sharpe > 0 else "red"
        st.metric("Sharpe Ratio", f"{sharpe:.2f}")
        st.markdown(f"<p style='color: {sharpe_color}'>Risk-Adjusted Return: "
                   f"{'Good' if sharpe > 1 else 'Fair' if sharpe > 0 else 'Poor'}</p>", 
                   unsafe_allow_html=True)
    
    with col4:
        sortino = performance.get('sortino_ratio', 0) if performance else 0
        sortino_color = "green" if sortino > 1.5 else "orange" if sortino > 0 else "red"
        st.metric("Sortino Ratio", f"{sortino:.2f}")
        st.markdown(f"<p style='color: {sortino_color}'>Downside Risk: "
                   f"{'Low' if sortino > 1.5 else 'Medium' if sortino > 0 else 'High'}</p>", 
                   unsafe_allow_html=True)
    
    # Position Risk Analysis
    if positions:
        st.subheader("Position Risk Analysis")
        
        positions_df = pd.DataFrame(positions)
        if not positions_df.empty and 'current_value' in positions_df.columns:
            total_value = positions_df['current_value'].sum()
            
            # Position concentration
            positions_df['concentration'] = positions_df['current_value'] / total_value
            
            # Create position risk table
            risk_df = positions_df[['lot_id', 'total_size', 'current_value', 'unrealized_pnl', 
                                   'unrealized_pnl_pct', 'concentration']].copy()
            
            # Add risk scores
            risk_df['risk_score'] = risk_df.apply(
                lambda row: 'High' if row['concentration'] > 0.3 or row['unrealized_pnl_pct'] < -0.2 
                else 'Medium' if row['concentration'] > 0.2 or row['unrealized_pnl_pct'] < -0.1 
                else 'Low', axis=1
            )
            
            # Format columns
            risk_df['current_value'] = risk_df['current_value'].apply(format_currency)
            risk_df['unrealized_pnl'] = risk_df['unrealized_pnl'].apply(format_currency)
            risk_df['unrealized_pnl_pct'] = risk_df['unrealized_pnl_pct'].apply(format_percentage)
            risk_df['concentration'] = risk_df['concentration'].apply(lambda x: f"{x:.1%}")
            
            st.dataframe(risk_df, use_container_width=True)
            
            # Position concentration pie chart
            fig = go.Figure(data=[go.Pie(
                labels=positions_df['lot_id'],
                values=positions_df['current_value'],
                title="Position Concentration"
            )])
            st.plotly_chart(fig, use_container_width=True)
    
    # Market Risk Indicators
    st.subheader("Market Risk Indicators")
    
    if tech_data and 'current_indicators' in tech_data:
        indicators = tech_data['current_indicators']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Volatility gauge
            atr = indicators.get('atr', 0)
            price = indicators.get('price', 1)
            atr_pct = (atr / price) * 100
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=atr_pct,
                title={'text': "Current Volatility (ATR %)"},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [None, 10]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 2], 'color': "lightgreen"},
                        {'range': [2, 5], 'color': "yellow"},
                        {'range': [5, 10], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 7
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Fear & Greed gauge
            fear_greed = indicators.get('fear_greed', 50)
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=fear_greed,
                title={'text': "Fear & Greed Index"},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 25], 'color': "red"},
                        {'range': [25, 45], 'color': "orange"},
                        {'range': [45, 55], 'color': "yellow"},
                        {'range': [55, 75], 'color': "lightgreen"},
                        {'range': [75, 100], 'color': "green"}
                    ]
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            # RSI gauge
            rsi = indicators.get('rsi', 50)
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=rsi,
                title={'text': "RSI"},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "green"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    # Risk Recommendations
    st.subheader("Risk Management Recommendations")
    
    recommendations = []
    
    # Check various risk factors
    if performance:
        if performance.get('max_drawdown', 0) < -0.3:
            recommendations.append({
                "Risk": "Excessive Drawdown",
                "Level": "High",
                "Action": "Consider reducing position sizes or implementing stop-losses"
            })
        
        if performance.get('volatility', 0) * np.sqrt(252) > 1.5:
            recommendations.append({
                "Risk": "High Volatility",
                "Level": "High",
                "Action": "Reduce leverage and consider hedging strategies"
            })
        
        if performance.get('sharpe_ratio', 0) < 0:
            recommendations.append({
                "Risk": "Negative Risk-Adjusted Returns",
                "Level": "Medium",
                "Action": "Review trading strategy and consider position adjustments"
            })
    
    if positions and len(positions) > 0:
        max_concentration = max([p.get('current_value', 0) / metrics.get('current_value', 1) 
                               for p in positions])
        if max_concentration > 0.5:
            recommendations.append({
                "Risk": "Position Concentration",
                "Level": "High",
                "Action": "Diversify by reducing largest positions"
            })
    
    if tech_data and 'current_indicators' in tech_data:
        indicators = tech_data['current_indicators']
        if indicators.get('fear_greed', 50) > 80:
            recommendations.append({
                "Risk": "Market Euphoria",
                "Level": "Medium",
                "Action": "Consider taking profits on winning positions"
            })
        elif indicators.get('fear_greed', 50) < 20:
            recommendations.append({
                "Risk": "Market Panic",
                "Level": "Low",
                "Action": "Potential buying opportunity for contrarian traders"
            })
    
    if recommendations:
        rec_df = pd.DataFrame(recommendations)
        
        # Style the dataframe
        def style_risk_level(val):
            if val == 'High':
                return 'background-color: rgba(255, 0, 0, 0.3)'
            elif val == 'Medium':
                return 'background-color: rgba(255, 165, 0, 0.3)'
            else:
                return 'background-color: rgba(0, 255, 0, 0.3)'
        
        styled_rec = rec_df.style.applymap(style_risk_level, subset=['Level'])
        st.dataframe(styled_rec, use_container_width=True)
    else:
        st.success("✅ No significant risk warnings at this time")

def show_configuration():
    """Configuration page for API keys and settings"""
    st.header("⚙️ System Configuration")
    
    # Initialize session state for API keys
    if 'api_keys' not in st.session_state:
        st.session_state.api_keys = {
            'admin_key': '',
            'fear_greed_api_key': '',
            'coingecko_api_key': '',
            'additional_keys': {}
        }
    
    # API Configuration Section
    st.subheader("API Keys Configuration")
    
    with st.form("api_config_form"):
        st.info("🔐 Configure API keys for enhanced data sources. Leave blank to use free/default sources.")
        
        # Admin API Key (for authentication)
        admin_key = st.text_input(
            "Admin API Key",
            type="password",
            value=st.session_state.api_keys.get('admin_key', ''),
            help="Required to save configuration (default: admin123)"
        )
        
        st.markdown("---")
        
        # Data Source API Keys
        col1, col2 = st.columns(2)
        
        with col1:
            fear_greed_key = st.text_input(
                "Fear & Greed API Key (Optional)",
                type="password",
                value=st.session_state.api_keys.get('fear_greed_api_key', ''),
                help="Enhanced Fear & Greed data access"
            )
            
            coingecko_key = st.text_input(
                "CoinGecko API Key (Optional)",
                type="password",
                value=st.session_state.api_keys.get('coingecko_api_key', ''),
                help="Higher rate limits for price data"
            )
        
        with col2:
            # Additional API keys
            st.markdown("### Additional API Keys")
            
            # Placeholder for future APIs
            binance_key = st.text_input(
                "Binance API Key (Optional)",
                type="password",
                help="For real-time order book data"
            )
            
            glassnode_key = st.text_input(
                "Glassnode API Key (Optional)",
                type="password",
                help="For on-chain metrics"
            )
        
        # Submit button
        submitted = st.form_submit_button("Save Configuration", type="primary")
        
        if submitted:
            # Update session state
            st.session_state.api_keys = {
                'admin_key': admin_key,
                'fear_greed_api_key': fear_greed_key,
                'coingecko_api_key': coingecko_key,
                'additional_keys': {
                    'binance': binance_key,
                    'glassnode': glassnode_key
                }
            }
            
            # Send to backend
            try:
                headers = {"api-key": admin_key or "admin123"}
                config_data = {
                    "fear_greed_api_key": fear_greed_key,
                    "coingecko_api_key": coingecko_key,
                    "additional_api_keys": {
                        k: v for k, v in {
                            'binance': binance_key,
                            'glassnode': glassnode_key
                        }.items() if v
                    }
                }
                
                response = requests.post(
                    f"{API_BASE_URL}/config/api-keys",
                    json=config_data,
                    headers=headers,
                    timeout=10
                )
                
                if response.status_code == 200:
                    st.success("✅ Configuration saved successfully!")
                else:
                    st.error(f"❌ Failed to save configuration: {response.text}")
                    
            except Exception as e:
                st.error(f"❌ Error saving configuration: {str(e)}")
    
    # Display current configuration status
    st.subheader("Current Configuration Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("API Connection", "Connected" if fetch_api_data("/") else "Disconnected")
    
    with col2:
        # Check current price source
        try:
            price_data = fetch_api_data("/price/current")
            if price_data:
                st.metric("Price Source", price_data.get('source', 'Unknown').capitalize())
        except:
            st.metric("Price Source", "Default")
    
    with col3:
        # Check if API keys are configured
        configured_keys = sum(1 for v in st.session_state.api_keys.values() 
                            if v and isinstance(v, str) and v.strip())
        st.metric("Configured APIs", configured_keys)
    
    # Data Sources Information
    st.subheader("📊 Available Data Sources")
    
    data_sources = [
        {
            "Source": "Yahoo Finance",
            "Type": "Price Data",
            "API Key": "Not Required",
            "Status": "✅ Active",
            "Description": "Historical and current BTC price data"
        },
        {
            "Source": "Alternative.me",
            "Type": "Fear & Greed Index",
            "API Key": "Optional",
            "Status": "✅ Active",
            "Description": "Crypto market sentiment indicator"
        },
        {
            "Source": "CoinGecko",
            "Type": "Real-time Price",
            "API Key": "Optional (Higher limits)",
            "Status": "✅ Active",
            "Description": "Current BTC price and market data"
        },
        {
            "Source": "Binance",
            "Type": "Price & Volume",
            "API Key": "Optional",
            "Status": "✅ Active",
            "Description": "Real-time market data and order book"
        },
        {
            "Source": "Glassnode",
            "Type": "On-chain Metrics",
            "API Key": "Required",
            "Status": "⏸️ Inactive",
            "Description": "Advanced on-chain analytics"
        }
    ]
    
    df_sources = pd.DataFrame(data_sources)
    st.dataframe(df_sources, use_container_width=True)
    
    # System Settings
    st.subheader("🔧 System Settings")
    
    with st.expander("Advanced Settings"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.number_input(
                "Signal Update Interval (minutes)",
                min_value=1,
                max_value=60,
                value=5,
                help="How often to generate new trading signals"
            )
            
            st.number_input(
                "Data Cache TTL (seconds)",
                min_value=10,
                max_value=300,
                value=60,
                help="How long to cache API responses"
            )
        
        with col2:
            st.selectbox(
                "Default Time Period",
                ["1d", "7d", "1mo", "3mo", "6mo", "1y"],
                index=2,
                help="Default period for price data"
            )
            
            st.selectbox(
                "Default Interval",
                ["1m", "5m", "15m", "1h", "4h", "1d"],
                index=3,
                help="Default candle interval"
            )
        
        st.warning("⚠️ Advanced settings are for display only in this version")
    
    # API Documentation
    st.subheader("📚 API Documentation")
    
    with st.expander("Backend API Endpoints"):
        st.markdown("""
        ### Available Endpoints
        
        - `GET /` - Health check
        - `GET /portfolio/metrics` - Portfolio performance metrics
        - `GET /signals/latest` - Latest trading signal with factors
        - `GET /market/btc-data` - BTC market data with indicators
        - `GET /analytics/technical` - Current technical indicators
        - `POST /trades/` - Execute a trade
        - `GET /positions/` - Current positions
        - `GET /price/current` - Real-time BTC price
        
        ### Authentication
        Some endpoints require the `api-key` header for authentication.
        
        ### Example Usage
        ```python
        import requests
        
        # Get current price
        response = requests.get(f"{API_BASE_URL}/price/current")
        price_data = response.json()
        
        # Update API keys (requires admin key)
        headers = {"api-key": "admin123"}
        config = {
            "fear_greed_api_key": "your-key",
            "coingecko_api_key": "your-key"
        }
        response = requests.post(
            f"{API_BASE_URL}/config/api-keys",
            json=config,
            headers=headers
        )
        ```
        """)

if __name__ == "__main__":
    main()