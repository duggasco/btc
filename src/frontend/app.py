"""BTC Trading System - Professional UI"""
import streamlit as st
import os
import sys
from pathlib import Path

# Add the parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import configuration
import config

# Import components
from utils.api_client import get_api_client

# Page configuration
st.set_page_config(
    page_title="BTC Trading System",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        "Get Help": config.GITHUB_REPO_URL,
        "Report a bug": f"{config.GITHUB_REPO_URL}/issues",
        "About": "# BTC Trading System\nProfessional Bitcoin trading platform with AI-powered analysis"
    }
)

def inject_custom_css():
    """Load and inject custom CSS files"""
    css_dir = Path(__file__).parent / "styles"
    
    # Try to load theme CSS
    theme_css = ""
    theme_path = css_dir / "theme.css"
    if theme_path.exists():
        with open(theme_path, 'r') as f:
            theme_css = f.read()
    
    # Try to load components CSS
    components_css = ""
    components_path = css_dir / "components.css"
    if components_path.exists():
        with open(components_path, 'r') as f:
            components_css = f.read()
    
    # Combine and inject CSS
    combined_css = theme_css + "\n" + components_css
    if combined_css:
        st.markdown(f"<style>{combined_css}</style>", unsafe_allow_html=True)

# Inject custom CSS
inject_custom_css()

# Initialize session state
if "api_client" not in st.session_state:
    st.session_state.api_client = get_api_client()

# Professional header
st.markdown("""
<div class="main-header">
    <h1 class="main-title">BTC Trading System</h1>
    <p class="main-subtitle">Professional Trading Platform</p>
</div>
""", unsafe_allow_html=True)

# Real-time metrics bar
metrics_container = st.container()
with metrics_container:
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # Fetch real-time data
    try:
        # Get current price
        price_data = st.session_state.api_client.get_current_price()
        btc_price = price_data.get('price', 0)
        price_change = price_data.get('change_24h', 0)
        
        # Get latest signal
        signal_data = st.session_state.api_client.get_latest_signal()
        current_signal = signal_data.get('signal', 'hold').upper()
        signal_confidence = signal_data.get('confidence', 0) * 100
        
        # Get portfolio metrics
        portfolio = st.session_state.api_client.get_portfolio()
        portfolio_value = portfolio.get('total_value', 0)
        portfolio_change = portfolio.get('change_24h', 0)
        win_rate = portfolio.get('win_rate', 0)
        total_trades = portfolio.get('total_trades', 0)
        
        # Get system status
        system_status = st.session_state.api_client.get_system_status()
        system_health = "Online" if system_status.get('healthy', False) else "Offline"
        
    except Exception as e:
        # Fallback values
        btc_price = 0
        price_change = 0
        current_signal = "OFFLINE"
        signal_confidence = 0
        portfolio_value = 0
        portfolio_change = 0
        win_rate = 0
        total_trades = 0
        system_health = "Error"
    
    with col1:
        st.metric("BTC Price", f"${btc_price:,.2f}", f"{price_change:+.2f}%")
    with col2:
        st.metric("Signal", current_signal, f"{signal_confidence:.0f}% confidence")
    with col3:
        st.metric("Portfolio", f"${portfolio_value:,.2f}", f"{portfolio_change:+.2f}%")
    with col4:
        st.metric("Win Rate", f"{win_rate:.1f}%", f"{total_trades} trades")
    with col5:
        st.metric("System", system_health, "")

# Main navigation - 3 professional pages
st.markdown("## Trading Platform", unsafe_allow_html=True)

# Create three columns for the main pages
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="nav-card">
        <div class="nav-card-header">
            <h2 class="nav-card-title">Trading Dashboard</h2>
            <span class="nav-card-icon">📊</span>
        </div>
        <div class="nav-card-body">
            <p class="nav-card-description">
                Unified trading interface with real-time monitoring, 
                signal execution, and portfolio management
            </p>
            <ul class="nav-feature-list">
                <li>Live price charts & indicators</li>
                <li>AI-powered trading signals</li>
                <li>Position & P/L tracking</li>
                <li>One-click order execution</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Launch Trading Dashboard", key="trading_dashboard", use_container_width=True):
        st.switch_page("pages/1_Trading_Dashboard.py")

with col2:
    st.markdown("""
    <div class="nav-card">
        <div class="nav-card-header">
            <h2 class="nav-card-title">Analytics & Research</h2>
            <span class="nav-card-icon">📈</span>
        </div>
        <div class="nav-card-body">
            <p class="nav-card-description">
                Comprehensive analysis tools for strategy development, 
                backtesting, and risk assessment
            </p>
            <ul class="nav-feature-list">
                <li>Historical backtesting</li>
                <li>Monte Carlo simulations</li>
                <li>Strategy optimization</li>
                <li>Data quality monitoring</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Open Analytics Suite", key="analytics_research", use_container_width=True):
        st.switch_page("pages/2_Analytics_Research.py")

with col3:
    st.markdown("""
    <div class="nav-card">
        <div class="nav-card-header">
            <h2 class="nav-card-title">Settings & Configuration</h2>
            <span class="nav-card-icon">⚙️</span>
        </div>
        <div class="nav-card-body">
            <p class="nav-card-description">
                System configuration, API management, and maintenance 
                tools for optimal performance
            </p>
            <ul class="nav-feature-list">
                <li>Trading rules & limits</li>
                <li>API key management</li>
                <li>Alert notifications</li>
                <li>System maintenance</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("System Configuration", key="settings_config", use_container_width=True):
        st.switch_page("pages/3_Settings_Configuration.py")

# Footer
st.markdown("---")
st.markdown("""
<div class="main-footer">
    <p>BTC Trading System v3.0 - Professional Edition</p>
    <p class="footer-subtitle">AI-Powered Trading Platform with Institutional-Grade Analytics</p>
</div>
""", unsafe_allow_html=True)

# Additional styling for the main page
st.markdown("""
<style>
/* Main page specific styles */
.main-header {
    text-align: center;
    padding: 2rem 0;
    margin-bottom: 2rem;
    border-bottom: 1px solid var(--border-subtle);
}

.main-title {
    font-size: 2.5rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0;
    letter-spacing: -0.02em;
}

.main-subtitle {
    font-size: 1rem;
    color: var(--text-secondary);
    margin-top: 0.5rem;
}

/* Navigation cards */
.nav-card {
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: var(--space-4);
    height: 100%;
    transition: all var(--transition-normal);
}

.nav-card:hover {
    border-color: var(--border-focus);
    transform: translateY(-2px);
    box-shadow: var(--shadow-md);
}

.nav-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: var(--space-3);
}

.nav-card-title {
    font-size: 1.25rem;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0;
}

.nav-card-icon {
    font-size: 1.5rem;
    opacity: 0.8;
}

.nav-card-description {
    color: var(--text-secondary);
    font-size: 0.875rem;
    line-height: 1.5;
    margin-bottom: var(--space-3);
}

.nav-feature-list {
    list-style: none;
    padding: 0;
    margin: 0;
}

.nav-feature-list li {
    padding: var(--space-1) 0;
    font-size: 0.813rem;
    color: var(--text-secondary);
    position: relative;
    padding-left: var(--space-4);
}

.nav-feature-list li:before {
    content: "•";
    color: var(--accent-primary);
    font-weight: bold;
    position: absolute;
    left: 0;
}

/* Footer */
.main-footer {
    text-align: center;
    padding: var(--space-6) 0 var(--space-4);
    color: var(--text-muted);
}

.footer-subtitle {
    font-size: 0.75rem;
    margin-top: var(--space-1);
}

/* Button overrides for professional look */
.stButton > button {
    background: var(--accent-primary);
    color: white;
    border: none;
    padding: var(--space-3) var(--space-4);
    font-weight: 500;
    border-radius: var(--radius-sm);
    transition: all var(--transition-fast);
    font-size: 0.875rem;
}

.stButton > button:hover {
    background: var(--accent-primary);
    opacity: 0.9;
    transform: translateY(-1px);
}

/* Metric styling */
[data-testid="metric-container"] {
    background: var(--bg-secondary);
    padding: var(--space-3);
    border-radius: var(--radius-sm);
    border: 1px solid var(--border-subtle);
}

[data-testid="metric-container"] [data-testid="metric-label"] {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted);
}

[data-testid="metric-container"] [data-testid="metric-value"] {
    font-size: 1.25rem;
    font-weight: 600;
    color: var(--text-primary);
}

[data-testid="metric-container"] [data-testid="metric-delta"] {
    font-size: 0.75rem;
}
</style>
""", unsafe_allow_html=True)