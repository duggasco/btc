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

from components.simple_websocket import get_simple_websocket_client
from components.api_client import APIClient
from components.charts import create_candlestick_chart, create_portfolio_chart
from components.metrics import display_price_metrics, display_portfolio_metrics
from components.auto_refresh import AutoRefreshManager
from components.page_styling import setup_page
from utils.constants import TIME_PERIODS, CHART_COLORS
from utils.helpers import format_currency, format_percentage
from utils.timezone import format_datetime_est, format_time_est

# Setup page with consistent styling
api_client = setup_page(
    page_name="Dashboard",
    page_title="Trading Terminal",
    page_subtitle="Real-time market data, AI signals, and portfolio management"
)

# Initialize auto refresh manager
refresh_manager = AutoRefreshManager("dashboard")

# Additional page-specific CSS
st.markdown("""
<style>
/* Top metrics bar */
.metrics-bar {
    background: var(--bg-secondary);
    border-radius: 12px;
    padding: 15px 20px;
    margin-bottom: 20px;
    border: 1px solid var(--border-subtle);
}

.metric-item {
    display: inline-block;
    margin-right: 30px;
}

.metric-label {
    color: var(--text-muted);
    font-size: 0.85em;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.metric-value {
    color: #ffffff;
    font-size: 1.3em;
    font-weight: 600;
    margin-top: 4px;
}

.metric-positive { color: #00ff88; }
.metric-negative { color: #ff3366; }

/* Price display */
.price-display {
    font-size: 2.5em;
    font-weight: bold;
    margin: 10px 0;
}

/* Signal indicator */
.signal-badge {
    display: inline-block;
    padding: 8px 20px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.9em;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.signal-buy { 
    background: linear-gradient(135deg, #00ff88, #00cc66); 
    color: #0e1117;
}

.signal-sell { 
    background: linear-gradient(135deg, #ff3366, #cc0033); 
    color: white;
}

.signal-hold { 
    background: linear-gradient(135deg, #8b92a8, #5a6178); 
    color: white;
}

/* Tabs styling */
div[role="tablist"] {
    background: rgba(26, 31, 46, 0.8);
    border-radius: 8px;
    padding: 5px;
    margin-bottom: 20px;
}

div[role="tab"] {
    border-radius: 6px;
    margin: 0 2px;
}

div[role="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #667eea, #764ba2);
}

/* Cards */
.info-card {
    background: rgba(26, 31, 46, 0.8);
    border-radius: 12px;
    padding: 20px;
    border: 1px solid rgba(247, 147, 26, 0.2);
    margin-bottom: 15px;
}

.info-card h4 {
    color: #f7931a;
    margin-bottom: 15px;
}

/* Tables */
.dataframe {
    background: rgba(26, 31, 46, 0.6) !important;
    border-radius: 8px;
    overflow: hidden;
}

/* Trade form */
.trade-form {
    background: rgba(26, 31, 46, 0.9);
    border-radius: 12px;
    padding: 25px;
    border: 1px solid rgba(247, 147, 26, 0.3);
}

/* Indicator grid */
.indicator-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 15px;
    margin: 15px 0;
}

.indicator-item {
    background: rgba(26, 31, 46, 0.6);
    border-radius: 8px;
    padding: 15px;
    border-left: 3px solid #f7931a;
}

.indicator-name {
    color: #8b92a8;
    font-size: 0.85em;
    margin-bottom: 5px;
}

.indicator-value {
    color: #ffffff;
    font-size: 1.1em;
    font-weight: 600;
}

/* Position cards */
.position-card {
    background: rgba(26, 31, 46, 0.8);
    border-radius: 10px;
    padding: 15px;
    margin: 10px 0;
    border: 1px solid rgba(247, 147, 26, 0.2);
    transition: all 0.3s ease;
}

.position-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(247, 147, 26, 0.2);
}

/* Status indicators */
.status-connected {
    color: #00ff88;
    font-weight: 600;
}

.status-disconnected {
    color: #ff3366;
    font-weight: 600;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 8px 20px;
    font-weight: 600;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
}

/* Success/Error messages */
.success-message {
    background: rgba(0, 255, 136, 0.1);
    border: 1px solid #00ff88;
    border-radius: 8px;
    padding: 10px 15px;
    color: #00ff88;
}

.error-message {
    background: rgba(255, 51, 102, 0.1);
    border: 1px solid #ff3366;
    border-radius: 8px;
    padding: 10px 15px;
    color: #ff3366;
}
</style>
""", unsafe_allow_html=True)

# Initialize auto-refresh manager
refresh_manager = AutoRefreshManager('Trading')

# Initialize API client
@st.cache_resource
def get_api_client():
    return APIClient(os.getenv("API_BASE_URL", "http://backend:8080"))

api_client = get_api_client()

# Get WebSocket client
ws_client = get_simple_websocket_client()
if ws_client:
    ws_client.subscribe("prices")
    ws_client.subscribe("signals")

# Initialize session state
if "selected_indicators" not in st.session_state:
    st.session_state.selected_indicators = ["sma_20", "sma_50", "ema_12", "ema_26"]
if "trade_mode" not in st.session_state:
    st.session_state.trade_mode = "paper"
if "position_size" not in st.session_state:
    st.session_state.position_size = 0.1
if "price_alert" not in st.session_state:
    st.session_state.price_alert = 0
if "alert_triggered" not in st.session_state:
    st.session_state.alert_triggered = False
if "chart_type" not in st.session_state:
    st.session_state.chart_type = "Candlestick"
if "market_data" not in st.session_state:
    st.session_state.market_data = None

# Header with key metrics and WebSocket status
def render_header():
    """Render the top header with key metrics"""
    try:
        # Fetch latest data
        btc_data = api_client.get("/btc/latest") or {}
        latest_signal = api_client.get("/signals/enhanced/latest") or {}
        portfolio_metrics = api_client.get("/portfolio/metrics") or {}
        
        # Process WebSocket messages
        if ws_client:
            messages = ws_client.get_messages()
            for msg in messages:
                if msg.get("type") == "price_update":
                    btc_data["latest_price"] = msg["data"]["price"]
                elif msg.get("type") == "signal_update":
                    latest_signal.update(msg["data"])
        
        # Create metrics bar
        st.markdown('<div class="metrics-bar">', unsafe_allow_html=True)
        
        col1, col2, col3, col4, col5, col6, col7, col8 = st.columns([1.5, 1, 1, 1, 1, 1, 1, 0.8])
        
        with col1:
            current_price = btc_data.get("latest_price", 0)
            price_change = btc_data.get("price_change_percentage_24h", 0)
            price_class = "metric-positive" if price_change >= 0 else "metric-negative"
            
            st.markdown(f'''
            <div class="metric-item">
                <div class="metric-label">BTC/USD</div>
                <div class="metric-value price-display {price_class}">${current_price:,.2f}</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with col2:
            signal = latest_signal.get("signal", "hold")
            confidence = latest_signal.get("confidence", 0)
            
            st.markdown(f'''
            <div class="metric-item">
                <div class="metric-label">AI Signal</div>
                <div class="signal-badge signal-{signal}">{signal}</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with col3:
            st.markdown(f'''
            <div class="metric-item">
                <div class="metric-label">Confidence</div>
                <div class="metric-value">{confidence:.1%}</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with col4:
            total_value = portfolio_metrics.get("total_value", 0)
            st.markdown(f'''
            <div class="metric-item">
                <div class="metric-label">Portfolio</div>
                <div class="metric-value">${total_value:,.2f}</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with col5:
            total_pnl = portfolio_metrics.get("total_pnl", 0)
            pnl_class = "metric-positive" if total_pnl >= 0 else "metric-negative"
            st.markdown(f'''
            <div class="metric-item">
                <div class="metric-label">P&L</div>
                <div class="metric-value {pnl_class}">${abs(total_pnl):,.2f}</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with col6:
            win_rate = portfolio_metrics.get("win_rate", 0)
            st.markdown(f'''
            <div class="metric-item">
                <div class="metric-label">Win Rate</div>
                <div class="metric-value">{win_rate:.1%}</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with col7:
            volume_24h = btc_data.get("total_volume", 0)
            st.markdown(f'''
            <div class="metric-item">
                <div class="metric-label">24h Volume</div>
                <div class="metric-value">${volume_24h/1e6:.1f}M</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with col8:
            ws_status = "connected" if ws_client and ws_client.is_connected() else "disconnected"
            status_class = "status-connected" if ws_status == "connected" else "status-disconnected"
            st.markdown(f'''
            <div class="metric-item">
                <div class="metric-label">Status</div>
                <div class="metric-value {status_class}">{"●" if ws_status == "connected" else "○"}</div>
            </div>
            ''', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        return btc_data, latest_signal, portfolio_metrics
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return {}, {}, {}

# Main app
st.title("BTC Trading Terminal")

# Render header
btc_data, latest_signal, portfolio_metrics = render_header()

# Sidebar controls
with st.sidebar:
    st.markdown("### Trading Controls")
    
    # Time period selector
    time_period = st.selectbox(
        "Chart Period",
        options=list(TIME_PERIODS.keys()),
        format_func=lambda x: TIME_PERIODS[x],
        index=5,  # Default to 1 day
        key="time_period_selector"
    )
    
    # Chart type selector
    with st.expander("Chart Settings", expanded=True):
        chart_type = st.selectbox(
            "Chart Type",
            ["Candlestick", "Line", "OHLC"],
            index=0,
            key="chart_type_selector"
        )
        st.session_state.chart_type = chart_type
        
        # Technical Indicators
        st.markdown("#### Technical Indicators")
        indicators_col1, indicators_col2 = st.columns(2)
        
        with indicators_col1:
            sma_enabled = st.checkbox("SMA", value=True, key="sma_checkbox_sidebar")
            if sma_enabled:
                if "sma_20" not in st.session_state.selected_indicators:
                    st.session_state.selected_indicators.append("sma_20")
                if "sma_50" not in st.session_state.selected_indicators:
                    st.session_state.selected_indicators.append("sma_50")
            else:
                st.session_state.selected_indicators = [x for x in st.session_state.selected_indicators if not x.startswith("sma_")]
            
            ema_enabled = st.checkbox("EMA", value=True, key="ema_checkbox_sidebar")
            if ema_enabled:
                if "ema_12" not in st.session_state.selected_indicators:
                    st.session_state.selected_indicators.append("ema_12")
                if "ema_26" not in st.session_state.selected_indicators:
                    st.session_state.selected_indicators.append("ema_26")
            else:
                st.session_state.selected_indicators = [x for x in st.session_state.selected_indicators if not x.startswith("ema_")]
        
        with indicators_col2:
            bb_enabled = st.checkbox("BB", value=False, key="bb_checkbox_sidebar")
            if bb_enabled:
                if "bb_upper" not in st.session_state.selected_indicators:
                    st.session_state.selected_indicators.extend(["bb_upper", "bb_middle", "bb_lower"])
            else:
                st.session_state.selected_indicators = [x for x in st.session_state.selected_indicators if not x.startswith("bb_")]
            
            vwap_enabled = st.checkbox("VWAP", value=False, key="vwap_checkbox_sidebar")
            if vwap_enabled:
                if "vwap" not in st.session_state.selected_indicators:
                    st.session_state.selected_indicators.append("vwap")
            else:
                if "vwap" in st.session_state.selected_indicators:
                    st.session_state.selected_indicators.remove("vwap")
    
    # Volume Analysis
    with st.expander("Volume Analysis", expanded=False):
        st.markdown("#### Volume Metrics")
        
        # Get volume data from session state or fetch if needed
        if st.session_state.market_data and "data" in st.session_state.market_data:
            volume_data = pd.DataFrame(st.session_state.market_data["data"])
            if not volume_data.empty and "volume" in volume_data.columns:
                # Handle None values in volume data
                volume_data["volume"] = volume_data["volume"].fillna(0)
                current_volume = volume_data["volume"].iloc[-1] if len(volume_data) > 0 else 0
                avg_volume = volume_data["volume"].mean()
                volume_std = volume_data["volume"].std()
                
                # Volume metrics
                st.metric("Current Volume", f"{current_volume/1e6:.2f}M")
                st.metric("Avg Volume (24h)", f"{avg_volume/1e6:.2f}M")
                
                # Volume trend
                volume_change = ((current_volume - avg_volume) / avg_volume * 100) if avg_volume > 0 else 0
                st.metric("Volume Change", f"{volume_change:+.1f}%", 
                         delta="High" if volume_change > 20 else "Low" if volume_change < -20 else "Normal")
                
                # Volume anomaly detection
                if volume_std > 0:
                    z_score = (current_volume - avg_volume) / volume_std
                    if abs(z_score) > 2:
                        st.warning(f"⚠️ Unusual volume detected (Z-score: {z_score:.2f})")
        else:
            st.info("Volume data will be available after chart loads")
    
    # Trading mode
    st.markdown("### Trading Mode")
    trade_mode = st.radio(
        "Select Mode",
        ["Paper Trading", "Live Trading"],
        index=0,
        key="trading_mode_radio"
    )
    st.session_state.trade_mode = "paper" if trade_mode == "Paper Trading" else "live"
    
    if st.session_state.trade_mode == "live":
        st.warning("⚠️ Live trading with real funds")
    else:
        st.info("📝 Paper trading mode active")
    
    # Price Alerts
    with st.expander("Price Alerts", expanded=False):
        # Initialize alerts in session state
        if "price_alerts" not in st.session_state:
            st.session_state.price_alerts = []
        
        # Add new alert
        st.markdown("#### Add Alert")
        alert_price = st.number_input(
            "Alert Price ($)", 
            min_value=0.0, 
            value=float(btc_data.get("latest_price", 50000)), 
            step=1000.0,
            key="alert_price_input"
        )
        alert_type = st.selectbox("Alert Type", ["Above", "Below"], key="alert_type_select")
        
        if st.button("Add Alert", key="add_alert_button"):
            alert = {
                "price": alert_price,
                "type": alert_type,
                "triggered": False,
                "created_at": datetime.now()
            }
            st.session_state.price_alerts.append(alert)
            st.success(f"Alert added: BTC {alert_type.lower()} ${alert_price:,.2f}")
        
        # Display active alerts
        if st.session_state.price_alerts:
            st.markdown("#### Active Alerts")
            current_price = btc_data.get("latest_price", 0)
            
            for i, alert in enumerate(st.session_state.price_alerts):
                if not alert["triggered"]:
                    # Check if alert should trigger
                    if (alert["type"] == "Above" and current_price >= alert["price"]) or \
                       (alert["type"] == "Below" and current_price <= alert["price"]):
                        alert["triggered"] = True
                        st.warning(f"🔔 Alert triggered! BTC is {alert['type'].lower()} ${alert['price']:,.2f}")
                    else:
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            distance = abs(current_price - alert["price"])
                            distance_pct = (distance / alert["price"] * 100) if alert["price"] > 0 else 0
                            st.text(f"{alert['type']} ${alert['price']:,.0f} ({distance_pct:.1f}% away)")
                        with col2:
                            if st.button("×", key=f"remove_alert_{i}"):
                                st.session_state.price_alerts.pop(i)
                                st.rerun()
    
    # Auto-refresh settings
    auto_refresh_enabled, refresh_interval = refresh_manager.render_controls(
        sidebar=False,
        default_interval=5,
        default_enabled=True
    )

# Main content - Tabbed interface
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Signals", "💰 Portfolio", "💸 Trade"])

# Overview Tab
with tab1:
    # Main chart area
    chart_col, info_col = st.columns([3, 1])
    
    with chart_col:
        # Fetch market data
        market_data = api_client.get(f"/market/btc-data?period={time_period}") or {}
        st.session_state.market_data = market_data  # Store for sidebar access
        
        if market_data and "data" in market_data:
            df = pd.DataFrame(market_data["data"])
            if not df.empty:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df.set_index("timestamp", inplace=True)
                
                # Calculate indicators
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
                
                # Create the chart based on selected type
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                   vertical_spacing=0.03, row_heights=[0.7, 0.3])
                
                # Main price chart
                if st.session_state.chart_type == "Candlestick":
                    fig.add_trace(go.Candlestick(
                        x=df.index,
                        open=df["open"],
                        high=df["high"],
                        low=df["low"],
                        close=df["close"],
                        name="OHLC",
                        increasing_line_color=CHART_COLORS["bullish"],
                        decreasing_line_color=CHART_COLORS["bearish"]
                    ), row=1, col=1)
                elif st.session_state.chart_type == "Line":
                    fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df["close"],
                        mode='lines',
                        name='Price',
                        line=dict(color=CHART_COLORS["primary"], width=2)
                    ), row=1, col=1)
                elif st.session_state.chart_type == "OHLC":
                    fig.add_trace(go.Ohlc(
                        x=df.index,
                        open=df["open"],
                        high=df["high"],
                        low=df["low"],
                        close=df["close"],
                        name="OHLC",
                        increasing_line_color=CHART_COLORS["bullish"],
                        decreasing_line_color=CHART_COLORS["bearish"]
                    ), row=1, col=1)
                
                # Add selected indicators
                for indicator in st.session_state.selected_indicators:
                    if indicator in df.columns:
                        color = CHART_COLORS.get(indicator, '#8b92a8')
                        if indicator.startswith('bb_'):
                            # Bollinger Bands styling
                            if indicator == 'bb_upper':
                                fig.add_trace(go.Scatter(
                                    x=df.index, y=df[indicator],
                                    name='BB Upper', line=dict(color='rgba(139, 146, 168, 0.5)', dash='dash'),
                                    mode='lines'
                                ), row=1, col=1)
                            elif indicator == 'bb_middle':
                                fig.add_trace(go.Scatter(
                                    x=df.index, y=df[indicator],
                                    name='BB Middle', line=dict(color='rgba(139, 146, 168, 0.8)'),
                                    mode='lines'
                                ), row=1, col=1)
                            elif indicator == 'bb_lower':
                                fig.add_trace(go.Scatter(
                                    x=df.index, y=df[indicator],
                                    name='BB Lower', line=dict(color='rgba(139, 146, 168, 0.5)', dash='dash'),
                                    mode='lines', fill='tonexty', fillcolor='rgba(139, 146, 168, 0.1)'
                                ), row=1, col=1)
                        else:
                            fig.add_trace(go.Scatter(
                                x=df.index, y=df[indicator],
                                name=indicator.upper().replace('_', ' '),
                                line=dict(color=color, width=2),
                                mode='lines'
                            ), row=1, col=1)
                
                # Add volume subplot
                fig.add_trace(go.Bar(
                    x=df.index,
                    y=df["volume"],
                    name="Volume",
                    marker_color=df.apply(lambda x: CHART_COLORS["bullish"] if x["close"] >= x["open"] else CHART_COLORS["bearish"], axis=1),
                    opacity=0.5
                ), row=2, col=1)
                
                # Update layout
                fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
                fig.update_yaxes(title_text="Price ($)", row=1, col=1)
                fig.update_yaxes(title_text="Volume", row=2, col=1)
                
                fig.update_layout(
                    height=600,
                    title=f"BTC/USD {TIME_PERIODS[time_period]} Chart",
                    showlegend=True,
                    legend=dict(x=0, y=1, bgcolor='rgba(26, 31, 46, 0.8)'),
                    template="plotly_dark",
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Technical indicators selector
        with st.expander("Technical Indicators", expanded=False):
            cols = st.columns(6)
            indicators = [
                ("SMA 20", "sma_20"),
                ("SMA 50", "sma_50"),
                ("EMA 12", "ema_12"),
                ("EMA 26", "ema_26"),
                ("Bollinger Bands", "bb_upper"),
                ("VWAP", "vwap")
            ]
            
            for i, (name, key) in enumerate(indicators):
                with cols[i % 6]:
                    if st.checkbox(name, value=key in st.session_state.selected_indicators, key=f"{key}_checkbox"):
                        if key not in st.session_state.selected_indicators:
                            if key == "bb_upper":
                                st.session_state.selected_indicators.extend(["bb_upper", "bb_middle", "bb_lower"])
                            else:
                                st.session_state.selected_indicators.append(key)
                    else:
                        if key in st.session_state.selected_indicators:
                            if key == "bb_upper":
                                st.session_state.selected_indicators = [x for x in st.session_state.selected_indicators if x not in ["bb_upper", "bb_middle", "bb_lower"]]
                            else:
                                st.session_state.selected_indicators.remove(key)
    
    with info_col:
        # Current signal details
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown("#### AI Analysis")
        
        if latest_signal:
            predicted_price = latest_signal.get("predicted_price", 0)
            current_price = btc_data.get("latest_price", 0)
            price_diff = predicted_price - current_price if current_price > 0 else 0
            price_diff_pct = (price_diff / current_price * 100) if current_price > 0 else 0
            
            st.metric("Target Price", f"${predicted_price:,.0f}", delta=f"{price_diff_pct:+.1f}%")
            
            # Signal strength
            composite_confidence = latest_signal.get("composite_confidence", 0)
            strength = min(100, int(composite_confidence * 100))
            st.progress(strength / 100, text=f"Strength: {strength}%")
            
            # Key factors
            if "key_factors" in latest_signal:
                st.markdown("**Key Factors:**")
                for factor in latest_signal["key_factors"][:3]:
                    st.markdown(f"• {factor}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Market stats
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown("#### Market Stats")
        
        if btc_data:
            st.metric("24h High", f"${btc_data.get('high_24h', 0):,.2f}")
            st.metric("24h Low", f"${btc_data.get('low_24h', 0):,.2f}")
            
            # Volume analysis - handle both lowercase and uppercase keys
            data_points = market_data.get("data", [])
            if data_points:
                last_point = data_points[-1]
                # Try both lowercase and uppercase volume keys
                current_volume = last_point.get("volume") or last_point.get("Volume") or 0
                current_volume = current_volume if current_volume is not None else 0
            else:
                current_volume = 0
            
            # Calculate average volume, handling None values and both key cases
            volumes = []
            for d in data_points:
                vol = d.get("volume") or d.get("Volume") or 0
                volumes.append(vol if vol is not None else 0)
            
            avg_volume = sum(volumes) / len(volumes) if volumes else 0
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
            
            st.metric("Volume Ratio", f"{volume_ratio:.1f}x", delta="High" if volume_ratio > 1.2 else "Low" if volume_ratio < 0.8 else "Normal")
        
        st.markdown('</div>', unsafe_allow_html=True)

# Signals Tab
with tab2:
    # Fetch comprehensive signals
    comprehensive_signals = api_client.get("/signals/comprehensive") or {}
    feature_importance = api_client.get("/analytics/feature-importance") or {}
    
    # Signal overview
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown("### Technical Indicators")
        
        if comprehensive_signals:
            # Combine all technical indicators
            tech_indicators = {}
            for category in ['technical', 'momentum', 'volatility', 'volume', 'trend']:
                if category in comprehensive_signals:
                    tech_indicators.update(comprehensive_signals[category])
            
            if tech_indicators:
                # Create indicator grid
                st.markdown('<div class="indicator-grid">', unsafe_allow_html=True)
                
                # Display key indicators
                key_indicators = [
                    ("RSI", "rsi", tech_indicators.get("rsi", 0)),
                    ("MACD", "macd", tech_indicators.get("macd", 0)),
                    ("ATR", "atr", tech_indicators.get("atr", 0)),
                    ("OBV", "obv", tech_indicators.get("obv", 0) / 1e6),
                    ("ADX", "adx", tech_indicators.get("adx", 0)),
                    ("Stoch %K", "stoch_k", tech_indicators.get("stoch_k", 0))
                ]
                
                cols = st.columns(3)
                for i, (name, key, value) in enumerate(key_indicators):
                    with cols[i % 3]:
                        # Determine sentiment
                        sentiment_class = ""
                        if key == "rsi":
                            sentiment_class = "metric-positive" if value < 30 else "metric-negative" if value > 70 else ""
                        elif key == "macd":
                            sentiment_class = "metric-positive" if value > 0 else "metric-negative"
                        
                        st.markdown(f'''
                        <div class="indicator-item">
                            <div class="indicator-name">{name}</div>
                            <div class="indicator-value {sentiment_class}">{value:.2f}{"M" if key == "obv" else ""}</div>
                        </div>
                        ''', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown("### Signal Components")
        
        if comprehensive_signals:
            # Show signal breakdown
            components = {
                "Technical": comprehensive_signals.get("technical", {}).get("signal_strength", 0),
                "Momentum": comprehensive_signals.get("momentum", {}).get("signal_strength", 0),
                "Volume": comprehensive_signals.get("volume", {}).get("signal_strength", 0),
                "Sentiment": comprehensive_signals.get("sentiment", {}).get("overall_sentiment", 50) / 100,
                "On-Chain": comprehensive_signals.get("on_chain", {}).get("signal_strength", 0)
            }
            
            for name, strength in components.items():
                st.markdown(f"**{name}**")
                st.progress(strength, text=f"{strength:.0%}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Additional signal details
    tab2_1, tab2_2, tab2_3 = st.tabs(["On-Chain Metrics", "Sentiment Analysis", "Signal History"])
    
    with tab2_1:
        if comprehensive_signals and "on_chain" in comprehensive_signals:
            onchain_data = comprehensive_signals["on_chain"]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                nvt_ratio = onchain_data.get("nvt_ratio", 0)
                nvt_signal = "metric-positive" if nvt_ratio < 50 else "metric-negative" if nvt_ratio > 100 else ""
                st.markdown(f'''
                <div class="info-card">
                    <h4>NVT Ratio</h4>
                    <div class="metric-value {nvt_signal}">{nvt_ratio:.1f}</div>
                    <small>Network Value to Transactions</small>
                </div>
                ''', unsafe_allow_html=True)
            
            with col2:
                mvrv_ratio = onchain_data.get("mvrv_ratio", 1.0)
                mvrv_signal = "metric-positive" if mvrv_ratio < 1 else "metric-negative" if mvrv_ratio > 3 else ""
                st.markdown(f'''
                <div class="info-card">
                    <h4>MVRV Ratio</h4>
                    <div class="metric-value {mvrv_signal}">{mvrv_ratio:.2f}</div>
                    <small>Market/Realized Value</small>
                </div>
                ''', unsafe_allow_html=True)
            
            with col3:
                exchange_netflow = onchain_data.get("exchange_outflow", 0) - onchain_data.get("exchange_inflow", 0)
                flow_signal = "metric-positive" if exchange_netflow > 1000 else "metric-negative" if exchange_netflow < -1000 else ""
                st.markdown(f'''
                <div class="info-card">
                    <h4>Exchange Net Flow</h4>
                    <div class="metric-value {flow_signal}">{exchange_netflow:,.0f} BTC</div>
                    <small>{"Accumulation" if exchange_netflow > 0 else "Distribution"}</small>
                </div>
                ''', unsafe_allow_html=True)
            
            with col4:
                whale_transactions = onchain_data.get("whale_transactions", 0)
                st.markdown(f'''
                <div class="info-card">
                    <h4>Whale Activity</h4>
                    <div class="metric-value">{whale_transactions}</div>
                    <small>Large transactions (24h)</small>
                </div>
                ''', unsafe_allow_html=True)
    
    with tab2_2:
        if comprehensive_signals and "sentiment" in comprehensive_signals:
            sentiment_data = comprehensive_signals["sentiment"]
            
            # Overall sentiment gauge
            overall_sentiment = sentiment_data.get("overall_sentiment", 50)
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = overall_sentiment,
                title = {'text': "Overall Market Sentiment"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 25], 'color': "red"},
                        {'range': [25, 40], 'color': "orange"},
                        {'range': [40, 60], 'color': "yellow"},
                        {'range': [60, 75], 'color': "lightgreen"},
                        {'range': [75, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Sentiment breakdown
            col1, col2, col3 = st.columns(3)
            
            with col1:
                fear_greed = sentiment_data.get("fear_greed_index", 50)
                st.metric("Fear & Greed Index", fear_greed)
            
            with col2:
                news_sentiment = sentiment_data.get("news_sentiment", 0)
                st.metric("News Sentiment", f"{news_sentiment:.2f}")
            
            with col3:
                social_sentiment = (sentiment_data.get("reddit_sentiment", 0) + sentiment_data.get("twitter_sentiment", 0)) / 2
                st.metric("Social Sentiment", f"{social_sentiment:.0f}")
    
    with tab2_3:
        # Signal history
        signal_history = api_client.get("/signals/history?hours=24") or []
        
        if signal_history:
            history_df = pd.DataFrame(signal_history)
            history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
            
            # Create signal timeline chart
            fig = go.Figure()
            
            # Add buy signals
            buy_df = history_df[history_df['signal'] == 'buy']
            if not buy_df.empty:
                fig.add_trace(go.Scatter(
                    x=buy_df['timestamp'],
                    y=buy_df['price_prediction'],
                    mode='markers',
                    name='Buy Signals',
                    marker=dict(
                        color='green',
                        size=buy_df['confidence'] * 20,
                        symbol='triangle-up'
                    )
                ))
            
            # Add sell signals
            sell_df = history_df[history_df['signal'] == 'sell']
            if not sell_df.empty:
                fig.add_trace(go.Scatter(
                    x=sell_df['timestamp'],
                    y=sell_df['price_prediction'],
                    mode='markers',
                    name='Sell Signals',
                    marker=dict(
                        color='red',
                        size=sell_df['confidence'] * 20,
                        symbol='triangle-down'
                    )
                ))
            
            fig.update_layout(
                title="Signal History (24h)",
                xaxis_title="Time",
                yaxis_title="Price ($)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)

# Portfolio Tab
with tab3:
    # Fetch portfolio data
    positions = api_client.get("/portfolio/positions") or []
    trades = api_client.get("/trades/all") or []
    performance = api_client.get("/analytics/performance") or {}
    
    # Portfolio overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_value = portfolio_metrics.get("total_value", 0)
        st.metric("Total Value", f"${total_value:,.2f}")
    
    with col2:
        total_pnl = portfolio_metrics.get("total_pnl", 0)
        pnl_pct = portfolio_metrics.get("total_pnl_percent", 0)
        st.metric("Total P&L", f"${total_pnl:,.2f}", delta=f"{pnl_pct:.2f}%")
    
    with col3:
        sharpe_ratio = portfolio_metrics.get("sharpe_ratio", 0)
        st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
    
    with col4:
        max_drawdown = portfolio_metrics.get("max_drawdown", 0)
        st.metric("Max Drawdown", f"{max_drawdown:.1f}%")
    
    # Tabs for portfolio details
    tab3_1, tab3_2, tab3_3 = st.tabs(["Active Positions", "Performance", "Trade History"])
    
    with tab3_1:
        if positions:
            # Display positions
            positions_df = pd.DataFrame(positions)
            
            # Calculate metrics
            for idx, position in enumerate(positions):
                entry_price = position.get("entry_price", 0)
                size = position.get("size", 0)
                current_price = btc_data.get("latest_price", 0)
                
                value = size * current_price
                pnl = (current_price - entry_price) * size
                pnl_pct = ((current_price / entry_price) - 1) * 100 if entry_price > 0 else 0
                
                col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])
                
                with col1:
                    st.markdown(f'''
                    <div class="position-card">
                        <strong>{position.get("symbol", "BTC-USD")}</strong> - {position.get("side", "LONG").upper()}
                    </div>
                    ''', unsafe_allow_html=True)
                
                with col2:
                    st.metric("Size", f"{size:.6f} BTC")
                
                with col3:
                    st.metric("Entry", f"${entry_price:,.2f}")
                
                with col4:
                    st.metric("Value", f"${value:,.2f}")
                
                with col5:
                    st.metric("P&L", f"${pnl:,.2f}", delta=f"{pnl_pct:.2f}%")
        else:
            st.info("No active positions")
    
    with tab3_2:
        # Performance chart
        perf_history = api_client.get("/portfolio/performance/history") or {}
        
        if perf_history and "equity_curve" in perf_history:
            equity_data = pd.DataFrame(perf_history["equity_curve"])
            equity_data['timestamp'] = pd.to_datetime(equity_data['timestamp'])
            
            fig = go.Figure()
            
            # Portfolio value line
            fig.add_trace(go.Scatter(
                x=equity_data['timestamp'],
                y=equity_data['value'],
                mode='lines',
                name='Portfolio Value',
                line=dict(color=CHART_COLORS['primary'], width=3),
                fill='tonexty',
                fillcolor='rgba(102, 126, 234, 0.1)'
            ))
            
            # Add benchmark if available
            if 'benchmark' in equity_data.columns:
                fig.add_trace(go.Scatter(
                    x=equity_data['timestamp'],
                    y=equity_data['benchmark'],
                    mode='lines',
                    name='Buy & Hold',
                    line=dict(color='gray', width=2, dash='dash')
                ))
            
            fig.update_layout(
                title="Portfolio Equity Curve",
                xaxis_title="Date",
                yaxis_title="Portfolio Value ($)",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3_3:
        if trades:
            # Trade history table
            trades_df = pd.DataFrame(trades)
            trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
            
            # Format for display
            display_df = trades_df[['timestamp', 'side', 'size', 'entry_price', 'exit_price', 'pnl', 'status']].copy()
            display_df['timestamp'] = display_df['timestamp'].apply(format_datetime_est)
            
            st.dataframe(display_df, use_container_width=True, height=400)

# Trade Tab
with tab4:
    # Trading form
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown('<div class="trade-form">', unsafe_allow_html=True)
        st.markdown("### Place Order")
        
        # Order type
        order_type = st.radio("Order Type", ["Market", "Limit"], key="order_type_radio")
        
        # Side
        side = st.radio("Side", ["Buy", "Sell"], key="order_side_radio")
        
        # Size
        st.session_state.position_size = st.number_input(
            "Position Size (BTC)",
            min_value=0.001,
            max_value=10.0,
            value=st.session_state.position_size,
            step=0.01,
            key="position_size_input"
        )
        
        # Limit price (if limit order)
        if order_type == "Limit":
            limit_price = st.number_input(
                "Limit Price ($)",
                min_value=1.0,
                value=float(btc_data.get("latest_price", 50000)),
                step=100.0,
                key="limit_price_input"
            )
        
        # Risk management
        stop_loss = st.number_input(
            "Stop Loss (%)",
            min_value=0.0,
            max_value=50.0,
            value=2.0,
            step=0.5,
            key="stop_loss_input"
        )
        
        take_profit = st.number_input(
            "Take Profit (%)",
            min_value=0.0,
            max_value=100.0,
            value=5.0,
            step=0.5,
            key="take_profit_input"
        )
        
        # Calculate order value
        current_price = btc_data.get("latest_price", 0)
        order_value = st.session_state.position_size * (limit_price if order_type == "Limit" else current_price)
        
        st.markdown(f"**Order Value:** ${order_value:,.2f}")
        
        # Submit button
        if st.button("Submit Order", type="primary", use_container_width=True):
            # Prepare order data
            order_data = {
                "symbol": "BTC-USD",
                "side": side.lower(),
                "size": st.session_state.position_size,
                "order_type": order_type.lower(),
                "stop_loss_pct": stop_loss,
                "take_profit_pct": take_profit
            }
            
            if order_type == "Limit":
                order_data["limit_price"] = limit_price
            
            # Submit order based on mode
            if st.session_state.trade_mode == "paper":
                response = api_client.post("/paper-trading/trade", order_data)
            else:
                response = api_client.post("/trading/order", order_data)
            
            if response and response.get("status") == "success":
                st.success(f"✅ Order submitted successfully!")
                st.balloons()
            else:
                st.error(f"❌ Order failed: {response.get('message', 'Unknown error')}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Order book and recent trades
        st.markdown("### Market Depth")
        
        # Simulated order book
        order_book_data = {
            "bids": [
                {"price": current_price - 50, "size": 0.5},
                {"price": current_price - 100, "size": 1.2},
                {"price": current_price - 150, "size": 2.1},
                {"price": current_price - 200, "size": 3.5},
                {"price": current_price - 250, "size": 5.0}
            ],
            "asks": [
                {"price": current_price + 50, "size": 0.8},
                {"price": current_price + 100, "size": 1.5},
                {"price": current_price + 150, "size": 2.8},
                {"price": current_price + 200, "size": 4.2},
                {"price": current_price + 250, "size": 6.0}
            ]
        }
        
        # Create order book visualization
        fig = go.Figure()
        
        # Bids
        bids_df = pd.DataFrame(order_book_data["bids"])
        fig.add_trace(go.Bar(
            x=bids_df["size"],
            y=bids_df["price"],
            orientation='h',
            name='Bids',
            marker_color=CHART_COLORS["bullish"],
            opacity=0.8
        ))
        
        # Asks
        asks_df = pd.DataFrame(order_book_data["asks"])
        fig.add_trace(go.Bar(
            x=asks_df["size"],
            y=asks_df["price"],
            orientation='h',
            name='Asks',
            marker_color=CHART_COLORS["bearish"],
            opacity=0.8
        ))
        
        # Current price line
        fig.add_hline(y=current_price, line_dash="dash", line_color="white",
                      annotation_text=f"Current: ${current_price:,.2f}")
        
        fig.update_layout(
            title="Order Book",
            xaxis_title="Size (BTC)",
            yaxis_title="Price ($)",
            height=400,
            barmode='overlay'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Recent trades
        st.markdown("### Recent Trades")
        
        recent_trades = trades[-10:] if trades else []
        if recent_trades:
            recent_df = pd.DataFrame(recent_trades)
            recent_df = recent_df[['timestamp', 'side', 'size', 'entry_price']].copy()
            recent_df['timestamp'] = pd.to_datetime(recent_df['timestamp']).apply(format_time_est)
            recent_df.columns = ['Time', 'Side', 'Size', 'Price']
            
            st.dataframe(recent_df, use_container_width=True, height=200)

# Footer with last update time
st.markdown(f'''
<div style="text-align: center; color: #8b92a8; font-size: 0.9em; margin-top: 30px;">
    Last updated: {format_datetime_est(datetime.now(), '%Y-%m-%d %H:%M:%S')} | 
    Data Quality: {"Excellent" if ws_client and ws_client.is_connected() else "Good"}
</div>
''', unsafe_allow_html=True)

# Handle auto-refresh
refresh_manager.handle_auto_refresh()