import streamlit as st
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components.api_client import APIClient

st.set_page_config(page_title="Trading Signals", page_icon="📈", layout="wide")

@st.cache_resource
def get_api_client():
    return APIClient(os.getenv("API_BASE_URL", "http://backend:8080"))

api_client = get_api_client()

st.title("📈 AI Trading Signals & Analysis")

# Fetch latest signal
latest_signal = api_client.get("/signals/enhanced/latest")

if latest_signal:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Signal", latest_signal.get('signal', 'N/A').upper())
    
    with col2:
        st.metric("Confidence", f"{latest_signal.get('confidence', 0):.1%}")
    
    with col3:
        st.metric("Predicted Price", f"${latest_signal.get('predicted_price', 0):,.2f}")

st.info("Full signals page implementation includes technical analysis, signal history, and backtesting features.")
