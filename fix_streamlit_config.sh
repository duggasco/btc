#!/bin/bash

# fix_streamlit_config.sh
# Fixes the st.set_page_config() order issue in streamlit_app.py

echo "Fixing Streamlit configuration order..."

# Create a backup of the original file
cp streamlit_app.py streamlit_app.py.backup
echo "✅ Created backup: streamlit_app.py.backup"

# Create a temporary file with the fixed content
cat > streamlit_app_temp.py << 'EOF'
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
import numpy as np
import logging
import websocket
import threading
import queue

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="BTC Trading System - UltraThink Enhanced",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# NOW add the custom CSS
st.markdown("""
<style>
/* Paper trading specific styles */
.paper-trading-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 20px;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin-bottom: 20px;
}

.paper-badge {
    background-color: #9C27B0;
    color: white;
    padding: 5px 10px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: bold;
}

.profit-positive {
    color: #4CAF50 !important;
    font-weight: bold;
}

.profit-negative {
    color: #f44336 !important;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

EOF

# Extract the rest of the file after line 56 (after the original st.set_page_config)
tail -n +57 streamlit_app.py >> streamlit_app_temp.py

# Replace the original file with the fixed version
mv streamlit_app_temp.py streamlit_app.py
echo "✅ Fixed st.set_page_config() order"

# Restart the frontend container
echo "Restarting frontend container..."
docker compose restart frontend

echo "✅ Fix complete! Frontend is restarting..."
echo ""
echo "To restore the backup if needed:"
echo "  mv streamlit_app.py.backup streamlit_app.py"
