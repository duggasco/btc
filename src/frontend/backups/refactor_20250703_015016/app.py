import streamlit as st
import os
import sys

# Add the parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Page configuration
st.set_page_config(
    page_title="BTC Trading System - UltraThink Enhanced",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/btc-trading-system',
        'Report a bug': "https://github.com/yourusername/btc-trading-system/issues",
        'About': "# BTC Trading System\nAI-powered Bitcoin trading with 50+ indicators and real-time analysis"
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

/* Animated value changes */
.metric-delta-positive {
    color: var(--success-color) !important;
    font-weight: bold;
    animation: pulse 2s infinite;
}

.metric-delta-negative {
    color: var(--danger-color) !important;
    font-weight: bold;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.7; }
    100% { opacity: 1; }
}

/* Trading status badges with animations */
.trading-badge {
    padding: 8px 20px;
    border-radius: 25px;
    font-weight: bold;
    text-align: center;
    margin: 5px;
    animation: fadeInScale 0.5s ease;
    position: relative;
    overflow: hidden;
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

.badge-paper {
    background: linear-gradient(135deg, var(--warning-color), #ff8800);
    color: white;
    box-shadow: 0 4px 15px rgba(255, 170, 0, 0.4);
}

/* WebSocket status with live indicator */
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

.ws-disconnected {
    background: rgba(255, 51, 102, 0.2);
    border: 1px solid var(--danger-color);
    color: var(--danger-color);
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

/* Enhanced buttons with hover effects */
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

div.stButton > button:before {
    content: "";
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
    transition: left 0.6s;
}

div.stButton > button:hover:before {
    left: 100%;
}

/* Chart containers with glass effect */
.chart-container {
    background: rgba(26, 31, 46, 0.6);
    border: 1px solid rgba(247, 147, 26, 0.2);
    border-radius: 20px;
    padding: 25px;
    margin: 15px 0;
    backdrop-filter: blur(10px);
}

/* Signal cards with animation */
.signal-card {
    background: rgba(26, 31, 46, 0.8);
    border-radius: 20px;
    padding: 20px;
    margin: 10px 0;
    border: 2px solid transparent;
    transition: all 0.3s ease;
}

.signal-buy {
    border-color: var(--success-color);
    box-shadow: 0 0 20px rgba(0, 255, 136, 0.3);
}

.signal-sell {
    border-color: var(--danger-color);
    box-shadow: 0 0 20px rgba(255, 51, 102, 0.3);
}

.signal-hold {
    border-color: var(--text-secondary);
    box-shadow: 0 0 20px rgba(139, 146, 168, 0.2);
}

/* Loading animation */
.loading-spinner {
    border: 3px solid rgba(247, 147, 26, 0.3);
    border-top: 3px solid var(--primary-color);
    border-radius: 50%;
    width: 40px;
    height: 40px;
    animation: spin 1s linear infinite;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

/* Responsive improvements */
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
if 'last_update' not in st.session_state:
    st.session_state.last_update = None
if 'ws_connected' not in st.session_state:
    st.session_state.ws_connected = False

# Sidebar with enhanced system status
with st.sidebar:
    st.markdown("## ₿ BTC Trading System")
    st.markdown("### UltraThink Enhanced Edition")
    st.markdown("---")
    
    # System health indicators with live updates
    health_container = st.container()
    with health_container:
        st.markdown("### 🏥 System Health")
        health_placeholder = st.empty()
        
    # WebSocket status with animation
    ws_container = st.container()
    with ws_container:
        st.markdown("### 🔌 Connections")
        ws_status_placeholder = st.empty()
    
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
<div style='text-align: center; padding: 20px;'>
    <h1 style='font-size: 3em; background: linear-gradient(135deg, #f7931a, #ffaa00); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
        🚀 BTC Trading System
    </h1>
    <p style='font-size: 1.2em; color: #8b92a8;'>
        AI-Powered Trading with 50+ Indicators & Real-Time Analysis
    </p>
</div>
""", unsafe_allow_html=True)

# Real-time stats banner
stats_container = st.container()
with stats_container:
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown("""
        <div class='signal-card'>
            <h4>📊 BTC Price</h4>
            <h2 id='btc-price'>Loading...</h2>
            <p id='price-change'>...</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='signal-card'>
            <h4>🎯 Signal</h4>
            <h2 id='current-signal'>Loading...</h2>
            <p id='signal-confidence'>...</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='signal-card'>
            <h4>💰 Portfolio</h4>
            <h2 id='portfolio-value'>Loading...</h2>
            <p id='portfolio-pnl'>...</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class='signal-card'>
            <h4>📈 Win Rate</h4>
            <h2 id='win-rate'>Loading...</h2>
            <p id='total-trades'>...</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown("""
        <div class='signal-card'>
            <h4>🔥 Active</h4>
            <h2 id='active-positions'>Loading...</h2>
            <p id='trading-mode'>...</p>
        </div>
        """, unsafe_allow_html=True)

# Feature cards
st.markdown("### 🎯 Navigate to Your Trading Dashboard")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class='chart-container' style='text-align: center;'>
        <h3>📊 Real-Time Dashboard</h3>
        <p>Live BTC prices, WebSocket updates, and market analysis with interactive charts</p>
        <br>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open Dashboard", key="dash", use_container_width=True):
        st.switch_page("pages/1_📊_Dashboard.py")

with col2:
    st.markdown("""
    <div class='chart-container' style='text-align: center;'>
        <h3>📈 AI Trading Signals</h3>
        <p>50+ indicators, LSTM predictions, and comprehensive signal analysis</p>
        <br>
    </div>
    """, unsafe_allow_html=True)
    if st.button("View Signals", key="sig", use_container_width=True):
        st.switch_page("pages/2_📈_Signals.py")

with col3:
    st.markdown("""
    <div class='chart-container' style='text-align: center;'>
        <h3>💼 Portfolio Manager</h3>
        <p>Track positions, P&L analysis, and automated risk management</p>
        <br>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Manage Portfolio", key="port", use_container_width=True):
        st.switch_page("pages/3_💰_Portfolio.py")

# Additional features
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class='chart-container' style='text-align: center;'>
        <h3>📄 Paper Trading</h3>
        <p>Practice risk-free with $10,000 virtual portfolio</p>
        <br>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Start Paper Trading", key="paper", use_container_width=True):
        st.switch_page("pages/4_📄_Paper_Trading.py")

with col2:
    st.markdown("""
    <div class='chart-container' style='text-align: center;'>
        <h3>🔬 Advanced Analytics</h3>
        <p>Backtesting, Monte Carlo simulations, and ML insights</p>
        <br>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Run Analytics", key="analytics", use_container_width=True):
        st.switch_page("pages/5_🔬_Analytics.py")

with col3:
    st.markdown("""
    <div class='chart-container' style='text-align: center;'>
        <h3>⚙️ Configuration</h3>
        <p>Customize trading rules, API keys, and signal weights</p>
        <br>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Settings", key="settings", use_container_width=True):
        st.switch_page("pages/6_⚙️_Settings.py")

# Live updates notice
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px;'>
    <p style='color: #8b92a8;'>
        <span class='ws-indicator' style='background: #00ff88; display: inline-block;'></span>
        Real-time data streaming • AI predictions every 5 minutes • 50+ indicators analyzed
    </p>
</div>
""", unsafe_allow_html=True)

# Footer with version info
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #8b92a8; padding: 20px;'>
    <p>BTC Trading System v2.1.0 - UltraThink Enhanced</p>
    <p style='font-size: 0.9em;'>Powered by LSTM Neural Networks • Real-time WebSocket • Multi-source Data Integration</p>
</div>
""", unsafe_allow_html=True)

# JavaScript for live updates (placeholder)
st.markdown("""
<script>
// This would connect to WebSocket in a real implementation
// For now, showing the structure
console.log('BTC Trading System initialized');
</script>
""", unsafe_allow_html=True)
