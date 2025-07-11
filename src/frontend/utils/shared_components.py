"""
Shared component utilities for consistent UI across pages
"""
import streamlit as st
from typing import Dict, List, Optional, Tuple
import pandas as pd
from datetime import datetime

def create_metric_card(label: str, value: str, delta: Optional[str] = None, delta_color: str = "normal") -> None:
    """Create a styled metric card"""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {f'<div class="metric-change {delta_color}">{delta}</div>' if delta else ''}
    </div>
    """, unsafe_allow_html=True)

def create_status_indicator(label: str, status: str, connected: bool = True) -> None:
    """Create a status indicator with dot"""
    status_class = "connected" if connected else "disconnected"
    st.markdown(f"""
    <div class="status-indicator">
        <span class="status-dot {status_class}"></span>
        <span>{label}: {status}</span>
    </div>
    """, unsafe_allow_html=True)

def create_signal_badge(signal: str) -> str:
    """Create a signal badge HTML"""
    signal_lower = signal.lower()
    badge_class = f"signal-{signal_lower}"
    return f'<span class="signal-badge {badge_class}">{signal}</span>'

def format_currency(value: float, decimals: int = 2) -> str:
    """Format value as currency"""
    return f"${value:,.{decimals}f}"

def format_percentage(value: float, decimals: int = 2, include_sign: bool = True) -> str:
    """Format value as percentage"""
    sign = "+" if value > 0 and include_sign else ""
    return f"{sign}{value:.{decimals}f}%"

def create_section_header(title: str, subtitle: Optional[str] = None) -> None:
    """Create a consistent section header"""
    st.markdown(f"""
    <div class="section-header">
        <h2>{title}</h2>
        {f'<p class="text-secondary">{subtitle}</p>' if subtitle else ''}
    </div>
    """, unsafe_allow_html=True)

def create_info_box(title: str, content: str, box_type: str = "info") -> None:
    """Create an information box"""
    st.markdown(f"""
    <div class="info-box {box_type}">
        <div class="info-box-title">{title}</div>
        <div class="info-box-content">{content}</div>
    </div>
    """, unsafe_allow_html=True)

def create_tabs(tab_names: List[str], tab_key: str) -> int:
    """Create custom styled tabs and return selected index"""
    selected_tab = st.session_state.get(f"{tab_key}_selected", 0)
    
    cols = st.columns(len(tab_names))
    for idx, (col, name) in enumerate(zip(cols, tab_names)):
        with col:
            if st.button(name, key=f"{tab_key}_{idx}", use_container_width=True):
                st.session_state[f"{tab_key}_selected"] = idx
                st.rerun()
    
    return selected_tab

def create_data_table(df: pd.DataFrame, 
                     highlight_columns: Optional[List[str]] = None,
                     format_dict: Optional[Dict] = None) -> None:
    """Create a styled data table"""
    if df.empty:
        st.info("No data available")
        return
    
    # Apply formatting
    styled_df = df.style
    
    if highlight_columns:
        for col in highlight_columns:
            if col in df.columns:
                styled_df = styled_df.background_gradient(subset=[col], cmap='RdYlGn')
    
    if format_dict:
        styled_df = styled_df.format(format_dict)
    
    st.dataframe(styled_df, use_container_width=True)

def create_time_filter(key: str = "time_filter") -> Tuple[datetime, datetime]:
    """Create a time range filter and return selected range"""
    col1, col2 = st.columns(2)
    
    with col1:
        start_date = st.date_input("Start Date", key=f"{key}_start")
    with col2:
        end_date = st.date_input("End Date", key=f"{key}_end")
    
    return start_date, end_date

def create_refresh_button(label: str = "Refresh Data") -> bool:
    """Create a refresh button with consistent styling"""
    return st.button(label, key="refresh_btn", type="primary")

def display_connection_status(api_status: bool, ws_status: bool) -> None:
    """Display connection status indicators"""
    col1, col2 = st.columns(2)
    with col1:
        create_status_indicator("API", "Online" if api_status else "Offline", api_status)
    with col2:
        create_status_indicator("WebSocket", "Connected" if ws_status else "Disconnected", ws_status)

def create_empty_state(message: str, suggestion: Optional[str] = None) -> None:
    """Display an empty state message"""
    st.markdown(f"""
    <div class="empty-state">
        <p class="empty-state-message">{message}</p>
        {f'<p class="empty-state-suggestion">{suggestion}</p>' if suggestion else ''}
    </div>
    """, unsafe_allow_html=True)