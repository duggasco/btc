import streamlit as st
import os
import sys
from pathlib import Path

# Add the parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import configuration
import config

# Import components
from components.api_client import APIClient

# Page configuration
st.set_page_config(
    page_title="BTC Trading System",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": config.GITHUB_REPO_URL,
        "Report a bug": f"{config.GITHUB_REPO_URL}/issues",
        "About": "# BTC Trading System\nAI-powered Bitcoin trading with 50+ indicators and real-time analysis"
    }
)

def inject_custom_css():
    """Load and inject custom CSS files"""
    css_dir = Path(__file__).parent / "styles"
    css_files = ["professional_theme.css"]
    
    combined_css = ""
    for css_file in css_files:
        css_path = css_dir / css_file
        if css_path.exists():
            with open(css_path, 'r') as f:
                combined_css += f.read() + "\n"
    
    if combined_css:
        st.markdown(f"<style>{combined_css}</style>", unsafe_allow_html=True)

# Inject custom CSS
inject_custom_css()

# Initialize session state
if "api_client" not in st.session_state:
    st.session_state.api_client = APIClient(base_url=config.API_BASE_URL)

# Main content
st.markdown("""
<div class="global-header">
    <h1>BTC Trading System</h1>
    <p class="text-secondary">AI-Powered Trading with 50+ Indicators</p>
</div>
""", unsafe_allow_html=True)

# Real-time stats banner
st.markdown("### Live System Overview")
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

# Navigation cards for all pages
st.markdown("### Select Your Trading Interface")

# First row - 3 columns
col1, col2, col3 = st.columns(3)

with col1:
    with st.container():
        st.markdown("""
        <div class="card">
            <div class="card-header">
                <h3 class="card-title">Dashboard</h3>
            </div>
            <p class="text-secondary">Real-time monitoring with live price charts, AI signals, and market overview</p>
            <ul class="feature-list">
                <li>Live BTC price updates</li>
                <li>Real-time signal alerts</li>
                <li>Market trends & sentiment</li>
                <li>Quick trade execution</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open Dashboard", key="dashboard", use_container_width=True):
            st.switch_page("pages/1_🏠_Dashboard.py")

with col2:
    with st.container():
        st.markdown("""
        <div class="card">
            <div class="card-header">
                <h3 class="card-title">Analytics</h3>
            </div>
            <p class="text-secondary">Advanced backtesting, optimization, Monte Carlo analysis, and performance metrics</p>
            <ul class="feature-list">
                <li>Strategy backtesting</li>
                <li>Parameter optimization</li>
                <li>Risk analysis</li>
                <li>Performance reports</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open Analytics", key="analytics", use_container_width=True):
            st.switch_page("pages/4_🔬_Analytics.py")

with col3:
    with st.container():
        st.markdown("""
        <div class="card">
            <div class="card-header">
                <h3 class="card-title">Signals</h3>
            </div>
            <p class="text-secondary">Comprehensive signal analysis with 50+ technical indicators and AI predictions</p>
            <ul class="feature-list">
                <li>LSTM model predictions</li>
                <li>Technical indicators</li>
                <li>Signal history tracking</li>
                <li>Confidence analysis</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        if st.button("View Signals", key="signals", use_container_width=True):
            st.switch_page("pages/2_📈_Signals.py")

# Second row - 3 columns
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    with st.container():
        st.markdown("""
        <div class="card">
            <div class="card-header">
                <h3 class="card-title">Portfolio</h3>
            </div>
            <p class="text-secondary">Track your positions, P&L, and trading performance</p>
            <ul class="feature-list">
                <li>Position tracking</li>
                <li>P&L visualization</li>
                <li>Trade history</li>
                <li>Performance metrics</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        if st.button("View Portfolio", key="portfolio", use_container_width=True):
            st.switch_page("pages/3_💰_Portfolio.py")

with col2:
    with st.container():
        st.markdown("""
        <div class="card">
            <div class="card-header">
                <h3 class="card-title">Paper Trading</h3>
            </div>
            <p class="text-secondary">Practice trading without risking real money</p>
            <ul class="feature-list">
                <li>Risk-free practice</li>
                <li>Detailed analytics</li>
                <li>Trading journal</li>
                <li>Performance tracking</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open Paper Trading", key="paper_trading", use_container_width=True):
            st.switch_page("pages/5_📄_Paper_Trading.py")

with col3:
    with st.container():
        st.markdown("""
        <div class="card">
            <div class="card-header">
                <h3 class="card-title">Settings</h3>
            </div>
            <p class="text-secondary">Configure trading parameters and monitor system</p>
            <ul class="feature-list">
                <li>Trading rules</li>
                <li>API config</li>
                <li>Data quality</li>
                <li>Diagnostics</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Configure System", key="settings", use_container_width=True):
            st.switch_page("pages/6_⚙️_Settings.py")

# Footer
st.markdown("---")
st.markdown("""
<div class="footer">
    <p class="text-muted">BTC Trading System v2.1.0 - Professional Edition</p>
    <p class="text-xs text-secondary">Powered by LSTM Neural Networks | Real-time WebSocket | Multi-source Data Integration</p>
</div>
""", unsafe_allow_html=True)

# Add custom CSS for the new elements
st.markdown("""
<style>
/* Navigation and layout styles */
.global-header {
    text-align: center;
    padding: 2rem 0;
    margin-bottom: 2rem;
}

.global-header h1 {
    font-size: 2.5rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 0.5rem;
}

.feature-list {
    list-style: none;
    padding: 0;
    margin: 1rem 0;
}

.feature-list li {
    padding: 0.25rem 0;
    font-size: 0.875rem;
    color: var(--text-secondary);
}

.feature-list li:before {
    content: "→ ";
    color: var(--accent-primary);
    font-weight: bold;
}

.footer {
    text-align: center;
    padding: 2rem 0;
}
</style>
""", unsafe_allow_html=True)