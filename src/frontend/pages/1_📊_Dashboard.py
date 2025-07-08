import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime, timedelta
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components.websocket_client import EnhancedWebSocketClient
from components.api_client import APIClient
from components.charts import create_candlestick_chart, create_portfolio_chart
from components.metrics import display_price_metrics, display_portfolio_metrics
from utils.constants import TIME_PERIODS, CHART_COLORS
from utils.helpers import format_currency, format_percentage

st.set_page_config(page_title="Real-Time Dashboard", page_icon="📊", layout="wide")

# Custom CSS for dashboard
st.markdown("""
<style>
.price-display {
    font-size: 3em;
    font-weight: bold;
    text-align: center;
    margin: 20px 0;
}
.price-positive { color: #00ff88; }
.price-negative { color: #ff3366; }
.signal-indicator {
    padding: 10px 20px;
    border-radius: 25px;
    text-align: center;
    font-weight: bold;
    margin: 10px 0;
}
.buy-signal { background: linear-gradient(135deg, #00ff88, #00cc66); color: white; }
.sell-signal { background: linear-gradient(135deg, #ff3366, #cc0033); color: white; }
.hold-signal { background: linear-gradient(135deg, #8b92a8, #5a6178); color: white; }
</style>
""", unsafe_allow_html=True)

# Initialize clients
@st.cache_resource
def get_websocket_client():
    client = EnhancedWebSocketClient(os.getenv("WS_URL", "ws://backend:8000/ws"))
    client.connect()
    client.subscribe("prices")
    client.subscribe("signals")
    return client

@st.cache_resource  
def get_api_client():
    return APIClient(os.getenv("API_BASE_URL", "http://backend:8080"))

ws_client = get_websocket_client()
api_client = get_api_client()

# Header with WebSocket status
col1, col2, col3 = st.columns([3, 1, 1])
with col1:
    st.title("📊 Real-Time BTC Dashboard")
with col2:
    ws_status = "🟢 Connected" if ws_client.is_connected() else "🔴 Disconnected"
    st.markdown(f'<div style="text-align: right; padding: 10px;">{ws_status}</div>', unsafe_allow_html=True)
with col3:
    if st.button("🔄 Refresh", use_container_width=True):
        st.rerun()

# Time period selector
time_period = st.selectbox(
    "Select Time Period",
    options=list(TIME_PERIODS.keys()),
    format_func=lambda x: TIME_PERIODS[x],
    index=2  # Default to 1 day
)

# Main layout
col1, col2 = st.columns([3, 1])

with col1:
    # Price chart container
    chart_container = st.container()
    
    # Technical indicators
    with st.expander("📈 Technical Indicators", expanded=True):
        indicator_cols = st.columns(6)
        selected_indicators = []
        
        with indicator_cols[0]:
            if st.checkbox("SMA 20", value=True):
                selected_indicators.append("sma_20")
        with indicator_cols[1]:
            if st.checkbox("SMA 50", value=True):
                selected_indicators.append("sma_50")
        with indicator_cols[2]:
            if st.checkbox("EMA 12"):
                selected_indicators.append("ema_12")
        with indicator_cols[3]:
            if st.checkbox("EMA 26"):
                selected_indicators.append("ema_26")
        with indicator_cols[4]:
            if st.checkbox("BB"):
                selected_indicators.extend(["bb_upper", "bb_middle", "bb_lower"])
        with indicator_cols[5]:
            if st.checkbox("VWAP"):
                selected_indicators.append("vwap")
    
    # Volume analysis
    volume_container = st.container()

with col2:
    # Current price display
    st.markdown("### 💰 Current Price")
    price_container = st.container()
    
    # Signal display
    st.markdown("### 🎯 AI Signal")
    signal_container = st.container()
    
    # Market stats
    st.markdown("### 📊 Market Stats")
    stats_container = st.container()
    
    # Quick metrics
    st.markdown("### 📈 Quick Metrics")
    metrics_container = st.container()

# Auto-refresh settings
refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 1, 60, 5)
auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)

# Sidebar additional controls
with st.sidebar:
    st.markdown("### 🎨 Chart Settings")
    chart_type = st.radio("Chart Type", ["Candlestick", "Line", "OHLC"])
    show_volume = st.checkbox("Show Volume", value=True)
    show_signals = st.checkbox("Show Buy/Sell Signals", value=True)
    
    st.markdown("### 🔔 Price Alerts")
    price_alert_enabled = st.checkbox("Enable Price Alerts")
    if price_alert_enabled:
        price_alert = st.number_input("Alert Price ($)", min_value=0.0, value=0.0, step=1000.0)
        alert_type = st.radio("Alert Type", ["Above", "Below"])

# Main update loop
placeholder = st.empty()

# Track previous values for alerts
if "prev_price" not in st.session_state:
    st.session_state.prev_price = 0
if "alert_triggered" not in st.session_state:
    st.session_state.alert_triggered = False

while True:
    with placeholder.container():
        try:
            # Get WebSocket messages
            messages = ws_client.get_messages()
            
            # Get latest data from API
            btc_data = api_client.get("/btc/latest") or {}
            latest_signal = api_client.get("/signals/enhanced/latest") or {}
            market_data = api_client.get(f"/market/btc-data?period={time_period}") or {}
            portfolio_metrics = api_client.get("/portfolio/metrics") or {}
            
            # Process WebSocket messages for real-time updates
            for msg in messages:
                if msg.get("type") == "price_update":
                    if btc_data:
                        btc_data["current_price"] = msg["data"]["price"]
                        btc_data["timestamp"] = msg["data"]["timestamp"]
                elif msg.get("type") == "signal_update":
                    latest_signal.update(msg["data"])
            
            # Update price display
            with price_container:
                if btc_data and "current_price" in btc_data:
                    current_price = btc_data["current_price"]
                    price_change = btc_data.get("price_change_percentage_24h", 0)
                    
                    price_class = "price-positive" if price_change >= 0 else "price-negative"
                    st.markdown(f"""
                    <div class="price-display {price_class}">
                        ${current_price:,.2f}
                    </div>
                    <div style="text-align: center; font-size: 1.2em;">
                        {format_percentage(price_change)}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Check price alert
                    if price_alert_enabled and price_alert > 0:
                        if alert_type == "Above" and current_price >= price_alert and st.session_state.prev_price < price_alert:
                            if not st.session_state.alert_triggered:
                                st.balloons()
                                st.warning(f"🔔 Price Alert! BTC is above ${price_alert:,.2f}")
                                st.session_state.alert_triggered = True
                        elif alert_type == "Below" and current_price <= price_alert and st.session_state.prev_price > price_alert:
                            if not st.session_state.alert_triggered:
                                st.balloons()
                                st.warning(f"🔔 Price Alert! BTC is below ${price_alert:,.2f}")
                                st.session_state.alert_triggered = True
                        elif (alert_type == "Above" and current_price < price_alert) or (alert_type == "Below" and current_price > price_alert):
                            st.session_state.alert_triggered = False
                    
                    st.session_state.prev_price = current_price
            
            # Update signal display
            with signal_container:
                if latest_signal:
                    signal = latest_signal.get("signal", "hold")
                    confidence = latest_signal.get("confidence", 0)
                    predicted_price = latest_signal.get("predicted_price", 0)
                    composite_confidence = latest_signal.get("composite_confidence", confidence)
                    
                    signal_class = {
                        "buy": "buy-signal",
                        "sell": "sell-signal",
                        "hold": "hold-signal"
                    }.get(signal, "hold-signal")
                    
                    st.markdown(f"""
                    <div class="signal-indicator {signal_class}">
                        {signal.upper()}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Confidence", f"{confidence:.1%}")
                    with col2:
                        st.metric("Target", f"${predicted_price:,.0f}")
                    
                    # Signal strength indicator
                    strength = min(100, int(composite_confidence * 100))
                    st.progress(strength / 100, text=f"Strength: {strength}%")
                    
                    # Key factors
                    if "key_factors" in latest_signal:
                        with st.expander("📊 Key Factors"):
                            for factor in latest_signal["key_factors"]:
                                st.write(f"• {factor}")
            
            # Update market stats
            with stats_container:
                if btc_data:
                    st.metric("24h High", f"${btc_data.get('high_24h', 0):,.2f}")
                    st.metric("24h Low", f"${btc_data.get('low_24h', 0):,.2f}")
                    st.metric("24h Volume", f"${btc_data.get('total_volume', 0)/1e6:.1f}M")
                    st.metric("Market Cap", f"${btc_data.get('market_cap', 0)/1e9:.1f}B")
            
            # Update quick metrics
            with metrics_container:
                if portfolio_metrics:
                    total_value = portfolio_metrics.get("total_value", 0)
                    total_pnl = portfolio_metrics.get("total_pnl", 0)
                    win_rate = portfolio_metrics.get("win_rate", 0)
                    sharpe = portfolio_metrics.get("sharpe_ratio", 0)
                    
                    st.metric("Portfolio", f"${total_value:,.2f}")
                    pnl_color = "🟢" if total_pnl >= 0 else "🔴"
                    st.metric("P&L", f"{pnl_color} ${abs(total_pnl):,.2f}")
                    st.metric("Win Rate", f"{win_rate:.1%}")
                    st.metric("Sharpe", f"{sharpe:.2f}")
            
            # Update main chart
            with chart_container:
                if market_data and "data" in market_data:
                    df = pd.DataFrame(market_data["data"])
                    if not df.empty:
                        df["timestamp"] = pd.to_datetime(df["timestamp"])
                        df.set_index("timestamp", inplace=True)
                        
                        # Calculate additional indicators if not present
                        if "sma_20" not in df.columns and len(df) >= 20:
                            df["sma_20"] = df["close"].rolling(window=20).mean()
                        if "sma_50" not in df.columns and len(df) >= 50:
                            df["sma_50"] = df["close"].rolling(window=50).mean()
                        if "ema_12" not in df.columns:
                            df["ema_12"] = df["close"].ewm(span=12).mean()
                        if "ema_26" not in df.columns:
                            df["ema_26"] = df["close"].ewm(span=26).mean()
                        
                        # Bollinger Bands
                        if "bb_middle" not in df.columns:
                            df["bb_middle"] = df["close"].rolling(window=20).mean()
                            bb_std = df["close"].rolling(window=20).std()
                            df["bb_upper"] = df["bb_middle"] + (bb_std * 2)
                            df["bb_lower"] = df["bb_middle"] - (bb_std * 2)
                        
                        # VWAP
                        if "vwap" not in df.columns and "volume" in df.columns:
                            df["vwap"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()
                        
                        # Prepare signals data if available
                        signals_df = None
                        if show_signals and "signals" in market_data:
                            signals_df = pd.DataFrame(market_data["signals"])
                            if not signals_df.empty:
                                signals_df["timestamp"] = pd.to_datetime(signals_df["timestamp"])
                                signals_df.set_index("timestamp", inplace=True)
                        
                        # Create the chart
                        fig = create_candlestick_chart(df, indicators=selected_indicators, signals=signals_df)
                        
                        # Customize based on chart type
                        if chart_type == "Line":
                            # Convert to line chart
                            fig.data[0].visible = False  # Hide candlestick
                            fig.add_trace(go.Scatter(
                                x=df.index,
                                y=df["close"],
                                mode="lines",
                                name="BTC Price",
                                line=dict(color=CHART_COLORS["primary"], width=2)
                            ))
                        elif chart_type == "OHLC":
                            # Convert to OHLC
                            fig.data[0].visible = False  # Hide candlestick
                            fig.add_trace(go.Ohlc(
                                x=df.index,
                                open=df["open"],
                                high=df["high"],
                                low=df["low"],
                                close=df["close"],
                                name="OHLC"
                            ))
                        
                        st.plotly_chart(fig, use_container_width=True)
            
            # Update volume chart
            with volume_container:
                if show_volume and market_data and "data" in market_data:
                    df = pd.DataFrame(market_data["data"])
                    if not df.empty and "volume" in df.columns:
                        df["timestamp"] = pd.to_datetime(df["timestamp"])
                        
                        # Volume analysis metrics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        avg_volume = df["volume"].mean()
                        current_volume = df["volume"].iloc[-1] if len(df) > 0 else 0
                        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
                        volume_trend = "📈" if volume_ratio > 1.2 else "📉" if volume_ratio < 0.8 else "➡️"
                        
                        with col1:
                            st.metric("Current Volume", f"{current_volume/1e6:.1f}M BTC")
                        with col2:
                            st.metric("Avg Volume", f"{avg_volume/1e6:.1f}M BTC")
                        with col3:
                            st.metric("Volume Ratio", f"{volume_ratio:.1f}x", delta=volume_trend)
                        with col4:
                            # Buy/Sell volume estimation
                            buy_volume = df[df["close"] > df["open"]]["volume"].sum()
                            total_volume = df["volume"].sum()
                            buy_ratio = buy_volume / total_volume if total_volume > 0 else 0.5
                            st.metric("Buy Volume", f"{buy_ratio:.1%}")
            
            # Add timestamp and data quality indicator
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f'<p style="text-align: center; color: #8b92a8; font-size: 0.9em;">Last updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>', unsafe_allow_html=True)
            with col2:
                data_quality = "🟢 Excellent" if ws_client.is_connected() else "🟡 Good"
                st.markdown(f'<p style="text-align: right; color: #8b92a8; font-size: 0.9em;">Data Quality: {data_quality}</p>', unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error updating dashboard: {str(e)}")
            st.info("Please check if the backend is running properly")
    
    # Break if auto-refresh is off
    if not auto_refresh:
        break
    
    # Wait before next update
    time.sleep(refresh_interval)
