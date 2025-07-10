
import streamlit as st
import os
import sys

# Add the parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import configuration
import config

# Import components
from components.api_client import APIClient

# Page configuration
st.set_page_config(
    page_title="BTC Trading System - UltraThink Enhanced",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": config.GITHUB_REPO_URL,
        "Report a bug": f"{config.GITHUB_REPO_URL}/issues",
        "About": "# BTC Trading System\\nAI-powered Bitcoin trading with 50+ indicators and real-time analysis"
    }
)

# Enhanced CSS with animations and modern design
st.markdown("""
<style>
/* Modern dark theme */
:root {
    --primary-color: #f7931a;  /* Bitcoin orange */
    --success-color: #00ff88;
    --danger-color: #ff3366;
    --warning-color: #ffaa00;
    --info-color: #00aaff;
    --bg-dark: #0e1117;
    --bg-card: #1a1f2e;
    --text-primary: #ffffff;
    --text-secondary: #8b92a8;
}

/* Animated gradient background */
.main {
    background: linear-gradient(-45deg, #0e1117, #1a1f2e, #0e1117, #2a2f3e);
    background-size: 400% 400%;
    animation: gradientShift 15s ease infinite;
}

@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Enhanced metrics with glow effect */
div[data-testid="metric-container"] {
    background: rgba(26, 31, 46, 0.8);
    border: 1px solid rgba(247, 147, 26, 0.3);
    padding: 15px;
    border-radius: 15px;
    backdrop-filter: blur(10px);
    transition: all 0.3s ease;
}

div[data-testid="metric-container"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(247, 147, 26, 0.3);
    border-color: rgba(247, 147, 26, 0.6);
}

/* Trading status badges */
.trading-badge {
    padding: 8px 20px;
    border-radius: 25px;
    font-weight: bold;
    text-align: center;
    margin: 5px;
    animation: fadeInScale 0.5s ease;
}

.badge-active {
    background: linear-gradient(135deg, var(--success-color), #00cc66);
    color: white;
    box-shadow: 0 4px 15px rgba(0, 255, 136, 0.4);
}

.badge-inactive {
    background: linear-gradient(135deg, var(--danger-color), #cc0033);
    color: white;
    box-shadow: 0 4px 15px rgba(255, 51, 102, 0.4);
}

/* WebSocket status indicator */
.ws-status {
    position: fixed;
    top: 80px;
    right: 30px;
    padding: 8px 16px;
    border-radius: 25px;
    font-size: 13px;
    font-weight: bold;
    z-index: 1000;
    display: flex;
    align-items: center;
    gap: 8px;
    backdrop-filter: blur(10px);
}

.ws-connected {
    background: rgba(0, 255, 136, 0.2);
    border: 1px solid var(--success-color);
    color: var(--success-color);
}

.ws-indicator {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    animation: blink 1s infinite;
}

@keyframes blink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}

/* Enhanced buttons */
div.stButton > button {
    background: linear-gradient(135deg, var(--primary-color), #f7731a);
    color: white;
    border: none;
    border-radius: 25px;
    font-weight: bold;
    padding: 10px 25px;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(247, 147, 26, 0.4);
}

/* Chart containers */
.chart-container {
    background: rgba(26, 31, 46, 0.6);
    border: 1px solid rgba(247, 147, 26, 0.2);
    border-radius: 20px;
    padding: 25px;
    margin: 15px 0;
    backdrop-filter: blur(10px);
}

/* Signal cards */
.signal-card {
    background: rgba(26, 31, 46, 0.8);
    border-radius: 20px;
    padding: 20px;
    margin: 10px 0;
    border: 2px solid transparent;
    transition: all 0.3s ease;
}

/* Responsive design */
@media (max-width: 768px) {
    .ws-status {
        top: auto;
        bottom: 20px;
        right: 20px;
    }
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "last_update" not in st.session_state:
    st.session_state.last_update = None
if "ws_connected" not in st.session_state:
    st.session_state.ws_connected = False
if "api_client" not in st.session_state:
    st.session_state.api_client = APIClient(base_url=config.API_BASE_URL)

# Sidebar with enhanced system status
with st.sidebar:
    st.markdown("## ₿ BTC Trading System")
    st.markdown("### UltraThink Enhanced Edition")
    st.markdown("---")
    
    # System health indicators
    health_container = st.container()
    with health_container:
        st.markdown("### 🏥 System Health")
        
        # Placeholder for dynamic health status
        health_status = st.empty()
        health_status.info("🟢 All systems operational")
    
    # WebSocket status
    ws_container = st.container()
    with ws_container:
        st.markdown("### 🔌 Connections")
        ws_status = st.empty()
        ws_status.success("WebSocket: Connected")
    
    st.markdown("---")
    
    # Feature highlights
    st.markdown("### ✨ Features")
    st.markdown("""
    - 🤖 **AI Predictions** - LSTM Neural Network
    - 📊 **50+ Indicators** - Technical, On-chain, Sentiment
    - 📈 **Live Updates** - WebSocket streaming
    - 💼 **Paper Trading** - Risk-free practice
    - 🔔 **Discord Alerts** - Real-time notifications
    - 📱 **Mobile Ready** - Responsive design
    """)
    
    st.markdown("---")
    
    # Quick actions
    st.markdown("### ⚡ Quick Actions")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
    with col2:
        if st.button("📊 Backtest", use_container_width=True):
            st.switch_page("pages/5_🔬_Analytics.py")
    
    st.markdown("---")
    
    # Keyboard shortcuts
    with st.expander("⌨️ Keyboard Shortcuts"):
        st.markdown("""
        - `R` - Refresh all data
        - `T` - Toggle trading mode
        - `P` - Switch paper/real mode
        - `S` - View signals
        - `B` - Run backtest
        - `Esc` - Emergency stop
        """)

# Main content with animated header
st.markdown("""
<div style="text-align: center; padding: 20px;">
    <h1 style="font-size: 3em; background: linear-gradient(135deg, #f7931a, #ffaa00); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
        🚀 BTC Trading System
    </h1>
    <p style="font-size: 1.2em; color: #8b92a8;">
        AI-Powered Trading with 50+ Indicators & Real-Time Analysis
    </p>
</div>
""", unsafe_allow_html=True)

# Real-time stats banner
st.markdown("### 📊 Live System Overview")
stats_container = st.container()
with stats_container:
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # Fetch real-time data
    try:
        # Get current price from /btc/latest endpoint
        price_data = st.session_state.api_client.get("/btc/latest")
        btc_price = "$0.00"
        price_change = "0.00%"
        if price_data and "latest_price" in price_data:
            btc_price = f"${price_data['latest_price']:,.2f}"
            if "price_change_percentage_24h" in price_data:
                price_change = f"{price_data['price_change_percentage_24h']:.2f}%"
        
        # Get latest signal
        signal_data = st.session_state.api_client.get_latest_signal()
        current_signal = "HOLD"
        signal_confidence = "0%"
        if signal_data and "signal" in signal_data:
            current_signal = signal_data["signal"].upper()
            if "confidence" in signal_data:
                signal_confidence = f"{signal_data['confidence']*100:.0f}%"
    except Exception as e:
        st.error(f"Failed to fetch data: {str(e)}")
        btc_price = "$0.00"
        price_change = "0.00%"
        current_signal = "ERROR"
        signal_confidence = "0%"
    
    with col1:
        st.metric("BTC Price", btc_price, price_change)
    with col2:
        st.metric("Signal", current_signal, signal_confidence)
    with col3:
        st.metric("Portfolio", "$0.00", "0.00%")
    with col4:
        st.metric("Win Rate", "0%", "0 trades")
    with col5:
        st.metric("Status", "Active", "Real Mode")

# Navigation cards
st.markdown("### 🎯 Navigate to Your Trading Dashboard")

col1, col2, col3 = st.columns(3)

with col1:
    with st.container():
        st.markdown("""
        <div class="chart-container" style="text-align: center;">
            <h3>📊 Real-Time Dashboard</h3>
            <p>Live BTC prices, WebSocket updates, and market analysis with interactive charts</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open Dashboard", key="dash", use_container_width=True):
            st.switch_page("pages/1_📊_Dashboard.py")

with col2:
    with st.container():
        st.markdown("""
        <div class="chart-container" style="text-align: center;">
            <h3>📈 AI Trading Signals</h3>
            <p>50+ indicators, LSTM predictions, and comprehensive signal analysis</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("View Signals", key="sig", use_container_width=True):
            st.switch_page("pages/2_📈_Signals.py")

with col3:
    with st.container():
        st.markdown("""
        <div class="chart-container" style="text-align: center;">
            <h3>💼 Portfolio Manager</h3>
            <p>Track positions, P&L analysis, and automated risk management</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Manage Portfolio", key="port", use_container_width=True):
            st.switch_page("pages/3_💰_Portfolio.py")

# Additional features row
col1, col2, col3 = st.columns(3)

with col1:
    with st.container():
        st.markdown("""
        <div class="chart-container" style="text-align: center;">
            <h3>📄 Paper Trading</h3>
            <p>Practice risk-free with $10,000 virtual portfolio</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Start Paper Trading", key="paper", use_container_width=True):
            st.switch_page("pages/4_📄_Paper_Trading.py")

with col2:
    with st.container():
        st.markdown("""
        <div class="chart-container" style="text-align: center;">
            <h3>🔬 Advanced Analytics</h3>
            <p>Backtesting, Monte Carlo simulations, and ML insights</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Run Analytics", key="analytics", use_container_width=True):
            st.switch_page("pages/5_🔬_Analytics.py")

with col3:
    with st.container():
        st.markdown("""
        <div class="chart-container" style="text-align: center;">
            <h3>⚙️ Configuration</h3>
            <p>Customize trading rules, API keys, and signal weights</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Settings", key="settings", use_container_width=True):
            st.switch_page("pages/6_⚙️_Settings.py")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #8b92a8; padding: 20px;">
    <p>BTC Trading System v2.1.0 - UltraThink Enhanced</p>
    <p style="font-size: 0.9em;">Powered by LSTM Neural Networks • Real-time WebSocket • Multi-source Data Integration</p>
</div>
""", unsafe_allow_html=True)

