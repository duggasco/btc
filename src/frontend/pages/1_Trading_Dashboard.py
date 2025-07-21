"""Trading Dashboard - Main consolidated view"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go

# Component imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from components.layout.dashboard_grid import render_dashboard_header, create_dashboard_grid
from components.display.metric_card import render_metric_row, format_currency, format_percentage, format_bitcoin
from components.display.signal_badge import render_signal_badge, render_signal_panel
from components.display.data_table import create_trade_table, create_position_table
from components.display.chart_container import create_chart_container, create_price_chart, create_portfolio_chart
from components.controls.form_controls import create_trading_form, create_button
from utils.api_client import get_api_client

# Page configuration
st.set_page_config(
    page_title="Trading Dashboard - BTC Trading System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load custom CSS
def inject_custom_css():
    """Load and inject custom CSS files"""
    css_dir = Path(__file__).parent.parent / "styles"
    
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

# Initialize API client
api_client = get_api_client()

# Initialize session state
if 'refresh_interval' not in st.session_state:
    st.session_state.refresh_interval = 30
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = True

def fetch_dashboard_data():
    """Fetch all data needed for the dashboard"""
    try:
        # Fetch current price and market data
        price_data = api_client.get_current_price()
        
        # Fetch latest signal
        signal_data = api_client.get_latest_signal()
        
        # Fetch portfolio data
        portfolio = api_client.get_portfolio()
        
        # Fetch recent trades
        trades = api_client.get_recent_trades(limit=20)
        
        # Fetch positions
        positions = api_client.get_positions()
        
        # Fetch historical data for chart
        historical_data = api_client.get_historical_data(period="24h")
        
        return {
            'price': price_data,
            'signal': signal_data,
            'portfolio': portfolio,
            'trades': trades,
            'positions': positions,
            'historical': historical_data
        }
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

def render_header_section(data):
    """Render the header with status indicators"""
    status_indicators = [
        {"label": "API", "status": "online" if data else "offline"},
        {"label": "Model", "status": "online" if data and data.get('signal') else "offline"},
        {"label": "Trading", "status": "online"}
    ]
    
    render_dashboard_header("Trading Dashboard", status_indicators)

def render_metrics_section(data):
    """Render key metrics"""
    if not data:
        return
    
    price_info = data.get('price', {})
    portfolio = data.get('portfolio', {})
    
    metrics = [
        {
            "title": "BTC PRICE",
            "value": price_info.get('price', 0),
            "change": price_info.get('change_24h', 0),
            "change_prefix": "+" if price_info.get('change_24h', 0) >= 0 else "",
            "format_func": format_currency
        },
        {
            "title": "PORTFOLIO VALUE",
            "value": portfolio.get('total_value', 0),
            "change": portfolio.get('change_24h', 0),
            "change_prefix": "+" if portfolio.get('change_24h', 0) >= 0 else "",
            "format_func": format_currency
        },
        {
            "title": "BTC BALANCE",
            "value": portfolio.get('btc_balance', 0),
            "format_func": format_bitcoin
        },
        {
            "title": "24H VOLUME",
            "value": price_info.get('volume_24h', 0),
            "format_func": format_currency
        }
    ]
    
    render_metric_row(metrics)

def render_main_chart(data):
    """Render the main price chart"""
    if not data or 'historical' not in data:
        st.info("No historical data available")
        return
    
    def create_chart():
        historical = data.get('historical', [])
        if not historical:
            return None
        df = pd.DataFrame(historical)
        if not df.empty and 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        return create_price_chart(df, "BTC/USD Price")
    
    # Chart controls
    controls = [
        {
            'type': 'select',
            'label': 'Timeframe',
            'options': ['1H', '4H', '1D', '1W'],
            'default': '1D',
            'callback': lambda x: st.rerun(),
            'key': 'chart_timeframe'
        }
    ]
    
    create_chart_container("Price Chart", create_chart, controls)

def render_signal_section(data):
    """Render signals panel"""
    if not data:
        return
    
    st.markdown('<div class="card signals-panel">', unsafe_allow_html=True)
    st.markdown('<h3>Trading Signals</h3>', unsafe_allow_html=True)
    
    # Current signal
    current_signal = data.get('signal', {})
    if current_signal:
        st.markdown('<div class="current-signal">', unsafe_allow_html=True)
        st.markdown('<h4>Current Signal</h4>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            render_signal_badge(
                current_signal.get('signal', 'hold'),
                current_signal.get('confidence', 50)
            )
        with col2:
            st.markdown(f"""
            <div class="signal-details">
                <div>Price: ${current_signal.get('price', 0):,.2f}</div>
                <div>Time: {current_signal.get('timestamp', 'N/A')}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Recent signals
    recent_signals = api_client.get_recent_signals(limit=10)
    if recent_signals:
        st.markdown('<h4>Recent Signals</h4>', unsafe_allow_html=True)
        for signal in recent_signals:
            render_signal_row(signal)
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_portfolio_section(data):
    """Render portfolio overview"""
    if not data:
        return
    
    portfolio = data.get('portfolio', {})
    positions = data.get('positions', [])
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3>Portfolio Overview</h3>', unsafe_allow_html=True)
    
    # Portfolio metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Total P&L",
            format_currency(portfolio.get('total_pnl', 0)),
            f"{portfolio.get('total_pnl_pct', 0):.2f}%"
        )
    
    with col2:
        st.metric(
            "Open Positions",
            len(positions),
            f"{sum(1 for p in positions if p.get('side') == 'long')} Long"
        )
    
    with col3:
        st.metric(
            "Win Rate",
            f"{portfolio.get('win_rate', 0):.1f}%",
            f"{portfolio.get('total_trades', 0)} trades"
        )
    
    # Positions table
    if positions:
        st.markdown('<h4>Open Positions</h4>', unsafe_allow_html=True)
        positions_df = pd.DataFrame(positions)
        create_position_table(positions_df)
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_order_execution_section():
    """Render order execution panel"""
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3>Order Execution</h3>', unsafe_allow_html=True)
    
    # Trading form
    order_data = create_trading_form()
    
    if order_data['submit']:
        # Execute order
        try:
            result = api_client.place_order(
                side=order_data['side'].lower(),
                amount=order_data['amount'],
                order_type=order_data['order_type'].lower(),
                price=order_data.get('price')
            )
            if result.get('success'):
                st.success(f"Order placed successfully! ID: {result.get('order_id')}")
            else:
                st.error(f"Order failed: {result.get('error')}")
        except Exception as e:
            st.error(f"Error placing order: {str(e)}")
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_recent_trades_section(data):
    """Render recent trades"""
    if not data or 'trades' not in data:
        return
    
    trades = data.get('trades', [])
    if not trades:
        st.info("No recent trades")
        return
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3>Recent Trades</h3>', unsafe_allow_html=True)
    
    trades_df = pd.DataFrame(trades)
    create_trade_table(trades_df)
    
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    """Main dashboard function"""
    # Fetch data
    data = fetch_dashboard_data()
    
    # Render header
    render_header_section(data)
    
    # Render metrics
    st.markdown('<div style="margin: 20px 0;">', unsafe_allow_html=True)
    render_metrics_section(data)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Create main grid layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Main chart
        render_main_chart(data)
        
        # Bottom panels
        col1_1, col1_2 = st.columns(2)
        
        with col1_1:
            render_portfolio_section(data)
        
        with col1_2:
            render_order_execution_section()
    
    with col2:
        # Signals panel
        render_signal_section(data)
        
        # Recent trades
        render_recent_trades_section(data)
    
    # Auto-refresh
    if st.session_state.auto_refresh:
        st.empty()
        import time
        time.sleep(st.session_state.refresh_interval)
        st.rerun()

if __name__ == "__main__":
    main()