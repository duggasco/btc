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

st.set_page_config(
    page_title="BTC Trading System",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

def create_candlestick_chart(btc_data):
    if not btc_data or 'data' not in btc_data:
        return None
    
    df = pd.DataFrame(btc_data['data'])
    if df.empty:
        return None
        
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('BTC Price', 'Volume'),
        row_width=[0.7, 0.3]
    )
    
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
    
    if 'sma_20' in df.columns and df['sma_20'].notna().any():
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['sma_20'],
                name='SMA 20',
                line=dict(color='orange', width=1)
            ),
            row=1, col=1
        )
    
    fig.add_trace(
        go.Bar(
            x=df['timestamp'],
            y=df['volume'],
            name='Volume',
            marker_color='rgba(128,128,128,0.5)'
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        title="BTC Price Analysis",
        xaxis_rangeslider_visible=False,
        height=600
    )
    
    return fig

def main():
    st.title("₿ BTC Trading System")
    st.markdown("---")
    
    with st.sidebar:
        st.header("Navigation")
        page = st.selectbox(
            "Select Page",
            ["Dashboard", "Trading", "Portfolio", "Signals", "Limits", "Analytics"]
        )
        
        with st.spinner("Checking API connection..."):
            api_status = fetch_api_data("/")
        
        if api_status:
            st.success("✅ API Connected")
        else:
            st.error("❌ API Disconnected")
    
    if page == "Dashboard":
        show_dashboard()
    elif page == "Trading":
        show_trading()
    elif page == "Portfolio":
        show_portfolio()
    elif page == "Signals":
        show_signals()
    elif page == "Limits":
        show_limits()
    elif page == "Analytics":
        show_analytics()

def show_dashboard():
    st.header("📊 Trading Dashboard")
    
    with st.spinner("Loading dashboard data..."):
        portfolio_metrics = fetch_api_data("/portfolio/metrics")
        latest_signal = fetch_api_data("/signals/latest")
        btc_data = fetch_api_data("/market/btc-data?period=1mo")
    
    col1, col2, col3, col4 = st.columns(4)
    
    if portfolio_metrics:
        with col1:
            st.metric("Total Trades", portfolio_metrics.get('total_trades', 0))
        with col2:
            total_pnl = portfolio_metrics.get('total_pnl', 0)
            st.metric("Total P&L", f"${total_pnl:,.2f}")
        with col3:
            st.metric("Positions", portfolio_metrics.get('positions_count', 0))
        with col4:
            current_price = portfolio_metrics.get('current_btc_price', 0)
            st.metric("BTC Price", f"${current_price:,.2f}" if current_price else "N/A")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("BTC Price Chart")
        if btc_data:
            fig = create_candlestick_chart(btc_data)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Latest Signal")
        if latest_signal:
            signal_color = {
                'buy': '#00FF00',
                'sell': '#FF0000', 
                'hold': '#FFA500'
            }.get(latest_signal['signal'], '#808080')
            
            st.markdown(f"""
            <div style="padding: 20px; border-radius: 10px; background-color: {signal_color}20; border: 2px solid {signal_color}; color: white; text-align: center;">
                <h2 style="color: {signal_color};">{latest_signal['signal'].upper()}</h2>
                <p>Confidence: {latest_signal['confidence']:.1%}</p>
                <p>Predicted Price: ${latest_signal['predicted_price']:,.2f}</p>
            </div>
            """, unsafe_allow_html=True)

def show_trading():
    st.header("💰 Trading Interface")
    
    with st.form("trade_form"):
        st.subheader("Execute Trade")
        
        col1, col2 = st.columns(2)
        
        with col1:
            trade_type = st.selectbox("Trade Type", ["buy", "sell", "hold"])
            size = st.number_input("Size (BTC)", min_value=0.001, value=0.1, step=0.001, format="%.3f")
        
        with col2:
            price = st.number_input("Price ($)", min_value=1.0, value=45000.0, step=0.01)
            lot_id = st.text_input("Lot ID (optional)")
        
        submitted = st.form_submit_button("Execute Trade")
        
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
                st.success(f"Trade executed successfully! Trade ID: {result['trade_id']}")

def show_portfolio():
    st.header("📈 Portfolio Overview")
    st.info("Portfolio functionality implemented in full version")

def show_signals():
    st.header("🤖 AI Trading Signals")
    latest_signal = fetch_api_data("/signals/latest")
    if latest_signal:
        st.json(latest_signal)

def show_limits():
    st.header("⚠️ Trading Limits & Orders")
    st.info("Limits functionality implemented in full version")

def show_analytics():
    st.header("📊 Advanced Analytics")
    st.info("Analytics functionality implemented in full version")

if __name__ == "__main__":
    main()
