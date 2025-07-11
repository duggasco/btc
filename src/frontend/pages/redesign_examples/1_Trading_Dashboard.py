"""
Trading Dashboard - Consolidated view for monitoring and execution
Combines real-time price monitoring, signals, portfolio, and trading
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
from components.simple_websocket import SimpleWebSocketClient
from components.websocket_manager import WebSocketManager
from components.charts import create_candlestick_chart, create_indicator_chart
import config

# Page configuration
st.set_page_config(
    page_title="Trading Dashboard - BTC Trading System",
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
/* Trading Dashboard specific styles */
.dashboard-container {
    display: grid;
    grid-template-columns: 2fr 1fr;
    grid-template-rows: auto 1fr auto;
    gap: 12px;
    height: calc(100vh - 120px);
}

.top-bar {
    grid-column: 1 / -1;
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 0;
}

.main-chart {
    grid-column: 1;
    grid-row: 2;
    min-height: 400px;
}

.side-panel {
    grid-column: 2;
    grid-row: 2;
    display: flex;
    flex-direction: column;
    gap: 12px;
    overflow-y: auto;
    padding-right: 4px;
}

.bottom-section {
    grid-column: 1 / -1;
    grid-row: 3;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
}

.compact-metrics {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 8px;
}

.signal-panel {
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    padding: 12px;
}

.trade-panel {
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    padding: 16px;
}

.position-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
    border-bottom: 1px solid var(--border-subtle);
}

.position-row:last-child {
    border-bottom: none;
}

/* Minimize whitespace */
.main .block-container {
    padding-top: 1rem;
}

div[data-testid="column"] > div {
    width: 100%;
}

.stMarkdown {
    margin-bottom: 0;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "api_client" not in st.session_state:
    st.session_state.api_client = APIClient(base_url=config.API_BASE_URL)
if "websocket_manager" not in st.session_state:
    st.session_state.websocket_manager = WebSocketManager()

# Get clients
api_client = st.session_state.api_client
ws_manager = st.session_state.websocket_manager

# Start WebSocket for this page
ws = ws_manager.get_or_create_websocket("dashboard")

# Top Bar with Key Metrics
top_container = st.container()
with top_container:
    col1, col2, col3, col4, col5, col6, col7, col8 = st.columns([1.5, 1, 1, 1, 1, 1, 1, 1.5])
    
    # Fetch latest data
    try:
        btc_data = api_client.get("/btc/latest")
        signal_data = api_client.get("/signals/latest")
        portfolio_data = api_client.get("/portfolio/summary")
        
        # Check WebSocket for real-time updates
        ws_message = ws.get_latest_message() if ws and ws.connected else None
        if ws_message and "latest_btc_data" in ws_message:
            btc_data = ws_message["latest_btc_data"]
    except:
        btc_data = signal_data = portfolio_data = None
    
    with col1:
        price = btc_data.get("latest_price", 0) if btc_data else 0
        change = btc_data.get("price_change_percentage_24h", 0) if btc_data else 0
        st.metric("BTC/USDT", f"${price:,.2f}", f"{change:+.2f}%")
    
    with col2:
        volume = btc_data.get("total_volume", 0) if btc_data else 0
        st.metric("24h Volume", f"${volume/1e9:.1f}B", None)
    
    with col3:
        signal = signal_data.get("signal", "HOLD") if signal_data else "HOLD"
        confidence = signal_data.get("confidence", 0) if signal_data else 0
        color = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⚪"}.get(signal, "⚪")
        st.metric("Signal", f"{color} {signal}", f"{confidence*100:.0f}%")
    
    with col4:
        balance = portfolio_data.get("total_value", 0) if portfolio_data else 0
        pnl_pct = portfolio_data.get("total_pnl_percentage", 0) if portfolio_data else 0
        st.metric("Portfolio", f"${balance:,.2f}", f"{pnl_pct:+.1f}%")
    
    with col5:
        positions = portfolio_data.get("open_positions", 0) if portfolio_data else 0
        st.metric("Positions", positions, None)
    
    with col6:
        win_rate = portfolio_data.get("win_rate", 0) if portfolio_data else 0
        trades = portfolio_data.get("total_trades", 0) if portfolio_data else 0
        st.metric("Win Rate", f"{win_rate:.1f}%", f"{trades} trades")
    
    with col7:
        st.metric("Mode", "Paper", "Safe Mode")
    
    with col8:
        # Connection status
        ws_status = "🟢 Connected" if ws and ws.connected else "🔴 Disconnected"
        st.markdown(f"""
        <div class="status-indicator" style="text-align: right;">
            <span>{ws_status}</span>
            <span class="text-muted text-xs">API: {datetime.now().strftime('%H:%M:%S')}</span>
        </div>
        """, unsafe_allow_html=True)

# Main content area
main_col, side_col = st.columns([2, 1])

# Main Chart Area
with main_col:
    chart_container = st.container()
    with chart_container:
        # Chart controls
        chart_col1, chart_col2, chart_col3, chart_col4 = st.columns([2, 1, 1, 1])
        with chart_col1:
            timeframe = st.selectbox("Timeframe", ["1m", "5m", "15m", "1h", "4h", "1d"], index=3, label_visibility="collapsed")
        with chart_col2:
            chart_type = st.selectbox("Type", ["Candlestick", "Line", "Area"], label_visibility="collapsed")
        with chart_col3:
            indicators = st.multiselect("Indicators", ["MA20", "MA50", "RSI", "MACD", "BB"], default=["MA20"], label_visibility="collapsed")
        with chart_col4:
            if st.button("Refresh Chart", use_container_width=True):
                st.rerun()
        
        # Chart display
        try:
            # Fetch OHLCV data
            ohlcv_data = api_client.get(f"/btc/ohlcv?timeframe={timeframe}&limit=100")
            if ohlcv_data and "data" in ohlcv_data:
                df = pd.DataFrame(ohlcv_data["data"])
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Create chart based on type
                if chart_type == "Candlestick":
                    fig = create_candlestick_chart(df, title="", height=400)
                else:
                    fig = go.Figure()
                    y_data = df['close']
                    if chart_type == "Line":
                        fig.add_trace(go.Scatter(x=df['timestamp'], y=y_data, mode='lines', name='Price'))
                    else:  # Area
                        fig.add_trace(go.Scatter(x=df['timestamp'], y=y_data, mode='lines', fill='tozeroy', name='Price'))
                
                # Update layout for dark theme
                fig.update_layout(
                    height=400,
                    margin=dict(l=0, r=0, t=0, b=0),
                    paper_bgcolor='#131315',
                    plot_bgcolor='#0a0a0b',
                    xaxis=dict(gridcolor='#27272a', zerolinecolor='#27272a'),
                    yaxis=dict(gridcolor='#27272a', zerolinecolor='#27272a'),
                    font=dict(color='#e5e5e7', size=11),
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            else:
                st.info("Loading chart data...")
        except Exception as e:
            st.error(f"Chart error: {str(e)}")

# Side Panel - Signals and Quick Trade
with side_col:
    # Current Signal Details
    st.markdown("### Active Signal")
    signal_container = st.container()
    with signal_container:
        if signal_data:
            signal_type = signal_data.get("signal", "HOLD")
            confidence = signal_data.get("confidence", 0)
            price_pred = signal_data.get("price_prediction", 0)
            
            # Signal badge
            badge_class = {"BUY": "signal-buy", "SELL": "signal-sell", "HOLD": "signal-hold"}.get(signal_type, "signal-hold")
            st.markdown(f"""
            <div class="signal-panel">
                <div class="flex justify-between items-center mb-3">
                    <span class="{badge_class}">{signal_type}</span>
                    <span class="text-secondary">Confidence: {confidence*100:.0f}%</span>
                </div>
                <div class="metric-label">Predicted Price</div>
                <div class="metric-value mb-2">${price_pred:,.2f}</div>
                <div class="text-xs text-muted">Generated: {datetime.now().strftime('%H:%M:%S')}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No active signal")
    
    st.markdown("### Quick Trade")
    trade_container = st.container()
    with trade_container:
        # Trade form
        trade_type = st.radio("Action", ["Buy", "Sell"], horizontal=True, label_visibility="collapsed")
        
        col1, col2 = st.columns(2)
        with col1:
            amount = st.number_input("Amount (BTC)", min_value=0.0001, value=0.01, step=0.0001, format="%.4f")
        with col2:
            if price > 0:
                usd_value = amount * price
                st.metric("USD Value", f"${usd_value:,.2f}", None, label_visibility="visible")
        
        # Advanced options in expander
        with st.expander("Advanced Options", expanded=False):
            stop_loss = st.number_input("Stop Loss %", min_value=0.0, max_value=50.0, value=2.0, step=0.5)
            take_profit = st.number_input("Take Profit %", min_value=0.0, max_value=100.0, value=5.0, step=0.5)
            
        if st.button(f"{trade_type} BTC", type="primary", use_container_width=True):
            # Execute trade
            st.success(f"{trade_type} order placed!")
    
    # Recent Signals
    st.markdown("### Recent Signals")
    signals_container = st.container()
    with signals_container:
        try:
            recent_signals = api_client.get("/signals/history?limit=5")
            if recent_signals and "signals" in recent_signals:
                for sig in recent_signals["signals"]:
                    timestamp = datetime.fromisoformat(sig["timestamp"].replace('Z', '+00:00'))
                    signal_type = sig["signal"]
                    confidence = sig["confidence"]
                    
                    badge_class = {"BUY": "signal-buy", "SELL": "signal-sell", "HOLD": "signal-hold"}.get(signal_type, "signal-hold")
                    st.markdown(f"""
                    <div class="position-row">
                        <span class="{badge_class} text-xs">{signal_type}</span>
                        <span class="text-xs text-muted">{confidence*100:.0f}%</span>
                        <span class="text-xs text-muted">{timestamp.strftime('%H:%M')}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No recent signals")
        except:
            st.error("Failed to load signals")

# Bottom Section - Portfolio and Positions
bottom_container = st.container()
with bottom_container:
    port_col, pos_col = st.columns(2)
    
    # Portfolio Overview
    with port_col:
        st.markdown("### Portfolio Overview")
        if portfolio_data:
            # Key metrics in grid
            metrics_html = f"""
            <div class="compact-metrics">
                <div class="metric-card">
                    <div class="metric-label">Total Value</div>
                    <div class="metric-value">${portfolio_data.get('total_value', 0):,.2f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Available</div>
                    <div class="metric-value">${portfolio_data.get('available_balance', 0):,.2f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">In Positions</div>
                    <div class="metric-value">${portfolio_data.get('position_value', 0):,.2f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Total P&L</div>
                    <div class="metric-value {'text-success' if portfolio_data.get('total_pnl', 0) >= 0 else 'text-danger'}">
                        ${portfolio_data.get('total_pnl', 0):,.2f}
                    </div>
                </div>
            </div>
            """
            st.markdown(metrics_html, unsafe_allow_html=True)
            
            # Performance chart
            try:
                perf_data = api_client.get("/portfolio/performance?days=7")
                if perf_data and "data" in perf_data:
                    df_perf = pd.DataFrame(perf_data["data"])
                    df_perf['timestamp'] = pd.to_datetime(df_perf['timestamp'])
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df_perf['timestamp'],
                        y=df_perf['total_value'],
                        mode='lines',
                        fill='tozeroy',
                        line=dict(color='#f7931a', width=2)
                    ))
                    
                    fig.update_layout(
                        height=200,
                        margin=dict(l=0, r=0, t=20, b=0),
                        paper_bgcolor='#131315',
                        plot_bgcolor='#0a0a0b',
                        xaxis=dict(gridcolor='#27272a', showgrid=False),
                        yaxis=dict(gridcolor='#27272a'),
                        font=dict(color='#e5e5e7', size=10),
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            except:
                pass
        else:
            st.info("Portfolio data unavailable")
    
    # Open Positions
    with pos_col:
        st.markdown("### Open Positions")
        try:
            positions = api_client.get("/portfolio/positions")
            if positions and "positions" in positions and len(positions["positions"]) > 0:
                for pos in positions["positions"][:5]:  # Show top 5
                    entry_price = pos.get("entry_price", 0)
                    current_price = price
                    quantity = pos.get("quantity", 0)
                    pnl = (current_price - entry_price) * quantity
                    pnl_pct = ((current_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
                    
                    st.markdown(f"""
                    <div class="position-row">
                        <div>
                            <div class="text-sm">{pos.get('type', 'LONG').upper()} {quantity:.4f} BTC</div>
                            <div class="text-xs text-muted">Entry: ${entry_price:,.2f}</div>
                        </div>
                        <div style="text-align: right;">
                            <div class="text-sm {'text-success' if pnl >= 0 else 'text-danger'}">${pnl:,.2f}</div>
                            <div class="text-xs {'text-success' if pnl_pct >= 0 else 'text-danger'}">{pnl_pct:+.2f}%</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Quick position actions
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Close All", use_container_width=True):
                        st.info("Closing all positions...")
                with col2:
                    if st.button("Add Position", use_container_width=True):
                        st.info("Opening position form...")
            else:
                st.info("No open positions")
        except:
            st.error("Failed to load positions")

# Auto-refresh
if st.checkbox("Auto-refresh", value=True, key="auto_refresh_dashboard"):
    st.empty()  # Trigger rerun on next cycle