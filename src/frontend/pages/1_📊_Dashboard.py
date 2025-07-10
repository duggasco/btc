import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime, timedelta
import time
import numpy as np
import atexit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components.simple_websocket import get_simple_websocket_client
from components.api_client import APIClient
from components.charts import create_candlestick_chart, create_portfolio_chart
from components.metrics import display_price_metrics, display_portfolio_metrics
from components.auto_refresh import AutoRefreshManager
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

# Initialize auto-refresh manager
refresh_manager = AutoRefreshManager('Dashboard')

# Initialize clients
@st.cache_resource  
def get_api_client():
    return APIClient(os.getenv("API_BASE_URL", "http://backend:8080"))

api_client = get_api_client()

# Get WebSocket client and subscribe to channels
ws_client = get_simple_websocket_client()
if ws_client:
    ws_client.subscribe("prices")
    ws_client.subscribe("signals")

# Header with WebSocket status
col1, col2, col3 = st.columns([3, 1, 1])
with col1:
    st.title("📊 Real-Time BTC Dashboard")
with col2:
    ws_status = "🟢 Connected" if ws_client and ws_client.is_connected() else "🔴 Disconnected"
    st.markdown(f'<div style="text-align: right; padding: 10px;">{ws_status}</div>', unsafe_allow_html=True)
with col3:
    if st.button("🔄 Refresh", use_container_width=True):
        st.rerun()

# Time period selector
time_period = st.selectbox(
    "Select Time Period",
    options=list(TIME_PERIODS.keys()),
    format_func=lambda x: TIME_PERIODS[x],
    index=2,  # Default to 1 day
    key="time_period_selector"
)

# Auto-refresh settings using the new manager
auto_refresh_enabled, refresh_interval = refresh_manager.render_controls(
    sidebar=True,
    default_interval=5,
    default_enabled=True
)

# Sidebar additional controls
with st.sidebar:
    st.markdown("### 🎨 Chart Settings")
    chart_type = st.radio("Chart Type", ["Candlestick", "Line", "OHLC"], key="chart_type_radio")
    show_volume = st.checkbox("Show Volume", value=True, key="show_volume_checkbox")
    show_signals = st.checkbox("Show Buy/Sell Signals", value=True, key="show_signals_checkbox")
    
    st.markdown("### 🔔 Price Alerts")
    price_alert_enabled = st.checkbox("Enable Price Alerts", key="price_alert_checkbox")
    if price_alert_enabled:
        price_alert = st.number_input("Alert Price ($)", min_value=0.0, value=0.0, step=1000.0, key="price_alert_input")
        alert_type = st.radio("Alert Type", ["Above", "Below"], key="alert_type_radio")

# Initialize session state for indicators
if "selected_indicators" not in st.session_state:
    st.session_state.selected_indicators = ["sma_20", "sma_50"]
if "prev_price" not in st.session_state:
    st.session_state.prev_price = 0
if "alert_triggered" not in st.session_state:
    st.session_state.alert_triggered = False

# Technical indicators selector (outside the loop)
with st.expander("📈 Technical Indicators", expanded=True):
    indicator_cols = st.columns(6)
    
    with indicator_cols[0]:
        if st.checkbox("SMA 20", value="sma_20" in st.session_state.selected_indicators, key="sma20_checkbox"):
            if "sma_20" not in st.session_state.selected_indicators:
                st.session_state.selected_indicators.append("sma_20")
        else:
            if "sma_20" in st.session_state.selected_indicators:
                st.session_state.selected_indicators.remove("sma_20")
                
    with indicator_cols[1]:
        if st.checkbox("SMA 50", value="sma_50" in st.session_state.selected_indicators, key="sma50_checkbox"):
            if "sma_50" not in st.session_state.selected_indicators:
                st.session_state.selected_indicators.append("sma_50")
        else:
            if "sma_50" in st.session_state.selected_indicators:
                st.session_state.selected_indicators.remove("sma_50")
                
    with indicator_cols[2]:
        if st.checkbox("EMA 12", value="ema_12" in st.session_state.selected_indicators, key="ema12_checkbox"):
            if "ema_12" not in st.session_state.selected_indicators:
                st.session_state.selected_indicators.append("ema_12")
        else:
            if "ema_12" in st.session_state.selected_indicators:
                st.session_state.selected_indicators.remove("ema_12")
                
    with indicator_cols[3]:
        if st.checkbox("EMA 26", value="ema_26" in st.session_state.selected_indicators, key="ema26_checkbox"):
            if "ema_26" not in st.session_state.selected_indicators:
                st.session_state.selected_indicators.append("ema_26")
        else:
            if "ema_26" in st.session_state.selected_indicators:
                st.session_state.selected_indicators.remove("ema_26")
                
    with indicator_cols[4]:
        if st.checkbox("BB", value="bb_upper" in st.session_state.selected_indicators, key="bb_checkbox"):
            if "bb_upper" not in st.session_state.selected_indicators:
                st.session_state.selected_indicators.extend(["bb_upper", "bb_middle", "bb_lower"])
        else:
            if "bb_upper" in st.session_state.selected_indicators:
                st.session_state.selected_indicators = [x for x in st.session_state.selected_indicators if x not in ["bb_upper", "bb_middle", "bb_lower"]]
                
    with indicator_cols[5]:
        if st.checkbox("VWAP", value="vwap" in st.session_state.selected_indicators, key="vwap_checkbox"):
            if "vwap" not in st.session_state.selected_indicators:
                st.session_state.selected_indicators.append("vwap")
        else:
            if "vwap" in st.session_state.selected_indicators:
                st.session_state.selected_indicators.remove("vwap")

# Main dashboard content
try:
    # Get WebSocket messages
    messages = []
    if ws_client:
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
                btc_data["latest_price"] = msg["data"]["price"]
                btc_data["timestamp"] = msg["data"]["timestamp"]
        elif msg.get("type") == "signal_update":
            latest_signal.update(msg["data"])
    
    # Main layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Price chart
        st.markdown("### 📊 BTC Price Chart")
        
        # Main chart
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
                fig = create_candlestick_chart(df, indicators=st.session_state.selected_indicators, signals=signals_df)
                
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
        
        # Volume analysis
        if show_volume and market_data and "data" in market_data:
            st.markdown("### 📊 Volume Analysis")
            df = pd.DataFrame(market_data["data"])
            if not df.empty and "volume" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                
                # Volume analysis metrics
                vol_col1, vol_col2, vol_col3, vol_col4 = st.columns(4)
                
                avg_volume = df["volume"].mean()
                current_volume = df["volume"].iloc[-1] if len(df) > 0 else 0
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
                volume_trend = "📈" if volume_ratio > 1.2 else "📉" if volume_ratio < 0.8 else "➡️"
                
                with vol_col1:
                    st.metric("Current Volume", f"{current_volume/1e6:.1f}M BTC")
                with vol_col2:
                    st.metric("Avg Volume", f"{avg_volume/1e6:.1f}M BTC")
                with vol_col3:
                    st.metric("Volume Ratio", f"{volume_ratio:.1f}x", delta=volume_trend)
                with vol_col4:
                    # Buy/Sell volume estimation
                    buy_volume = df[df["close"] > df["open"]]["volume"].sum()
                    total_volume = df["volume"].sum()
                    buy_ratio = buy_volume / total_volume if total_volume > 0 else 0.5
                    st.metric("Buy Volume", f"{buy_ratio:.1%}")
    
    with col2:
        # Current price display
        st.markdown("### 💰 Current Price")
        if btc_data and "latest_price" in btc_data:
            current_price = btc_data["latest_price"]
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
        
        # Signal display
        st.markdown("### 🎯 AI Signal")
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
            
        # Market stats
        st.markdown("### 📊 Market Stats")
        if btc_data:
            st.metric("24h High", f"${btc_data.get('high_24h', 0):,.2f}")
            st.metric("24h Low", f"${btc_data.get('low_24h', 0):,.2f}")
            st.metric("24h Volume", f"${btc_data.get('total_volume', 0)/1e6:.1f}M")
            st.metric("Market Cap", f"${btc_data.get('market_cap', 0)/1e9:.1f}B")
            
        # Quick metrics
        st.markdown("### 📈 Quick Metrics")
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
    
    # Add timestamp and data quality indicator
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f'<p style="text-align: center; color: #8b92a8; font-size: 0.9em;">Last updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>', unsafe_allow_html=True)
    with col2:
        data_quality = "🟢 Excellent" if ws_client and ws_client.is_connected() else "🟡 Good"
        st.markdown(f'<p style="text-align: right; color: #8b92a8; font-size: 0.9em;">Data Quality: {data_quality}</p>', unsafe_allow_html=True)
    
except Exception as e:
    st.error(f"Error updating dashboard: {str(e)}")
    st.info("Please check if the backend is running properly")

# No cleanup needed for simple WebSocket client as it doesn't use background threads

# Handle auto-refresh using the new manager
refresh_manager.handle_auto_refresh()
