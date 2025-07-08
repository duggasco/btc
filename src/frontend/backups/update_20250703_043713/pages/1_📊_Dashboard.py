import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime, timedelta
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components.websocket_client import EnhancedWebSocketClient
from components.api_client import APIClient
from components.charts import create_candlestick_chart, create_portfolio_chart
from components.metrics import display_price_metrics, display_portfolio_metrics

st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")

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

# Header
st.title("📊 Real-Time BTC Dashboard")

# WebSocket status indicator
ws_status = "🟢 Connected" if ws_client.is_connected() else "🔴 Disconnected"
st.markdown(f"<div class='ws-status {'ws-connected' if ws_client.is_connected() else 'ws-disconnected'}'>{ws_status}</div>", unsafe_allow_html=True)

# Real-time price display
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("## 💰 Bitcoin Price")
    
    # Price metrics container
    price_container = st.container()
    
    # Chart containers
    chart_container = st.container()

with col2:
    st.markdown("## 📈 Market Stats")
    stats_container = st.container()
    
    st.markdown("## 🎯 Current Signal")
    signal_container = st.container()

# Auto-refresh mechanism
refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 1, 60, 5)
auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)

# Create placeholder for live updates
if auto_refresh:
    placeholder = st.empty()
    
    # Update loop
    while True:
        with placeholder.container():
            # Get WebSocket messages
            messages = ws_client.get_messages()
            
            # Get latest data from API
            btc_data = api_client.get("/btc/latest")
            latest_signal = api_client.get("/signals/enhanced/latest")
            
            # Process WebSocket messages
            for msg in messages:
                if msg.get('type') == 'price_update':
                    if btc_data:
                        btc_data['current_price'] = msg['data']['price']
                        btc_data['timestamp'] = msg['data']['timestamp']
            
            # Update price metrics
            with price_container:
                if btc_data:
                    display_price_metrics(btc_data)
            
            # Update chart
            with chart_container:
                # Fetch historical data
                historical_data = api_client.get("/market/btc-data", params={"period": "24h"})
                
                if historical_data and 'data' in historical_data:
                    df = pd.DataFrame(historical_data['data'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df.set_index('timestamp', inplace=True)
                    
                    # Create candlestick chart
                    fig = create_candlestick_chart(df, indicators=['sma_20', 'sma_50'])
                    st.plotly_chart(fig, use_container_width=True)
            
            # Update market stats
            with stats_container:
                if btc_data:
                    st.metric("Market Cap", f"${btc_data.get('market_cap', 0)/1e9:.2f}B")
                    st.metric("Circulating Supply", f"{btc_data.get('circulating_supply', 0)/1e6:.2f}M")
                    st.metric("Market Cap Rank", f"#{btc_data.get('market_cap_rank', 1)}")
            
            # Update signal
            with signal_container:
                if latest_signal:
                    signal = latest_signal.get('signal', 'hold')
                    confidence = latest_signal.get('confidence', 0)
                    predicted_price = latest_signal.get('predicted_price', 0)
                    
                    # Signal indicator
                    if signal == 'buy':
                        st.success(f"🟢 **BUY**")
                        st.metric("Confidence", f"{confidence:.1%}")
                        st.metric("Target", f"${predicted_price:,.2f}")
                    elif signal == 'sell':
                        st.error(f"🔴 **SELL**")
                        st.metric("Confidence", f"{confidence:.1%}")
                        st.metric("Target", f"${predicted_price:,.2f}")
                    else:
                        st.info(f"⚪ **HOLD**")
                        st.metric("Confidence", f"{confidence:.1%}")
                        st.metric("Target", f"${predicted_price:,.2f}")
                    
                    # Signal details
                    if st.button("View Details"):
                        st.switch_page("pages/2_📈_Signals.py")
        
        # Wait before next update
        time.sleep(refresh_interval)

# Sidebar additional info
with st.sidebar:
    st.markdown("### 📊 Quick Stats")
    
    # Fetch portfolio metrics
    portfolio_metrics = api_client.get("/portfolio/metrics")
    
    if portfolio_metrics:
        st.metric("Total Value", f"${portfolio_metrics.get('total_value', 0):,.2f}")
        st.metric("Total P&L", f"${portfolio_metrics.get('total_pnl', 0):,.2f}")
        st.metric("Win Rate", f"{portfolio_metrics.get('win_rate', 0):.1%}")
