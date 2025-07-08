import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime, timedelta
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components.api_client import APIClient
from components.charts import create_portfolio_chart, create_performance_chart
from components.metrics import display_portfolio_metrics, display_risk_metrics
from utils.helpers import format_currency, format_percentage, calculate_sharpe_ratio

st.set_page_config(page_title="Portfolio Management", page_icon="💰", layout="wide")

# Initialize API client
@st.cache_resource
def get_api_client():
    return APIClient(os.getenv("API_BASE_URL", "http://backend:8080"))

api_client = get_api_client()

st.title("💰 Portfolio Management")
st.markdown("Track positions, analyze performance, and manage risk")

# Fetch portfolio data
try:
    portfolio_metrics = api_client.get("/portfolio/metrics") or {}
    positions = api_client.get("/portfolio/positions") or []
    trades = api_client.get("/trades/all") or []
    performance = api_client.get("/analytics/performance") or {}
    btc_price = api_client.get("/btc/latest", {}).get("current_price", 0)
except Exception as e:
    st.error(f"Error fetching portfolio data: {str(e)}")
    st.stop()

# Portfolio overview metrics
display_portfolio_metrics(portfolio_metrics)

# Additional performance metrics
if performance:
    display_risk_metrics(performance)

# Tabs for different views
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Current Positions",
    "📈 Performance Analysis",
    "🔄 Trade History",
    "⚠️ Risk Management",
    "📊 Analytics",
    "💸 P&L Analysis"
])

# Current Positions Tab
with tab1:
    st.markdown("### Active Positions")
    
    if positions:
        # Convert to DataFrame for easier manipulation
        positions_df = pd.DataFrame(positions)
        
        # Calculate additional metrics for each position
        positions_data = []
        for _, position in positions_df.iterrows():
            entry_price = position.get("entry_price", 0)
            size = position.get("size", 0)
            current_price = position.get("current_price", btc_price)
            
            value = size * current_price
            pnl = (current_price - entry_price) * size
            pnl_percent = ((current_price / entry_price) - 1) * 100 if entry_price > 0 else 0
            
            # Calculate position duration
            entry_time = pd.to_datetime(position.get("entry_time", datetime.now()))
            duration = (datetime.now() - entry_time).days
            
            positions_data.append({
                "Symbol": position.get("symbol", "BTC-USD"),
                "Side": position.get("side", "long").upper(),
                "Size": size,
                "Entry Price": entry_price,
                "Current Price": current_price,
                "Value": value,
                "P&L": pnl,
                "P&L %": pnl_percent,
                "Duration": f"{duration}d",
                "Status": "🟢" if pnl > 0 else "🔴"
            })
        
        positions_display = pd.DataFrame(positions_data)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            total_value = positions_display["Value"].sum()
            st.metric("Total Position Value", format_currency(total_value))
        with col2:
            total_pnl = positions_display["P&L"].sum()
            st.metric("Unrealized P&L", format_currency(total_pnl))
        with col3:
            avg_pnl_pct = positions_display["P&L %"].mean()
            st.metric("Average P&L %", format_percentage(avg_pnl_pct))
        with col4:
            winning_positions = len(positions_display[positions_display["P&L"] > 0])
            st.metric("Winning Positions", f"{winning_positions}/{len(positions_display)}")
        
        # Display positions table
        st.dataframe(positions_display, use_container_width=True, hide_index=True)
    else:
        st.info("No active positions")

# Additional tabs would continue here...

