"""
Card components for displaying metrics, info, alerts, and charts.
"""
import streamlit as st
from typing import Optional, Union, Dict, Any, List
import plotly.graph_objects as go


def render_metric_card(
    title: str,
    value: Union[str, float, int],
    delta: Optional[Union[str, float]] = None,
    delta_color: Optional[str] = None,
    subtitle: Optional[str] = None,
    icon: Optional[str] = None,
    key: Optional[str] = None
):
    """
    Render a compact metric card.
    
    Args:
        title: Card title
        value: Main metric value
        delta: Change value (optional)
        delta_color: Color for delta ("normal", "inverse", "off")
        subtitle: Additional info below value
        icon: Icon to display
        key: Unique key
    """
    with st.container():
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        
        # Header with icon
        if icon:
            st.markdown(f'<div class="metric-header">{icon} {title}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="metric-header">{title}</div>', unsafe_allow_html=True)
        
        # Use Streamlit's metric for consistent styling
        st.metric(
            label="",
            value=value,
            delta=delta,
            delta_color=delta_color,
            label_visibility="collapsed"
        )
        
        # Subtitle if provided
        if subtitle:
            st.markdown(f'<div class="metric-subtitle">{subtitle}</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)


def render_info_card(
    title: str,
    content: str,
    expandable: bool = True,
    expanded: bool = False,
    icon: Optional[str] = None,
    footer: Optional[str] = None,
    key: Optional[str] = None
):
    """
    Render an information card with optional expansion.
    
    Args:
        title: Card title
        content: Card content (supports markdown)
        expandable: Whether card can be expanded
        expanded: Initial expansion state
        icon: Icon to display
        footer: Footer text
        key: Unique key
    """
    if expandable:
        with st.expander(f"{icon} {title}" if icon else title, expanded=expanded):
            st.markdown(content)
            if footer:
                st.markdown(f'<div class="card-footer">{footer}</div>', unsafe_allow_html=True)
    else:
        with st.container():
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            st.markdown(f'<h4>{icon} {title}</h4>' if icon else f'<h4>{title}</h4>', unsafe_allow_html=True)
            st.markdown(content)
            if footer:
                st.markdown(f'<div class="card-footer">{footer}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)


def render_alert_card(
    message: str,
    status: str = "info",
    title: Optional[str] = None,
    dismissible: bool = True,
    key: Optional[str] = None
):
    """
    Render an alert card with status.
    
    Args:
        message: Alert message
        status: Alert type ("success", "info", "warning", "error")
        title: Alert title
        dismissible: Whether alert can be dismissed
        key: Unique key
    """
    # Map status to Streamlit alert types
    status_map = {
        "success": "success",
        "info": "info",
        "warning": "warning",
        "error": "error"
    }
    
    # Icons for each status
    icon_map = {
        "success": "✅",
        "info": "ℹ️",
        "warning": "⚠️",
        "error": "❌"
    }
    
    alert_type = status_map.get(status, "info")
    icon = icon_map.get(status, "ℹ️")
    
    # Handle dismissible alerts
    if dismissible and key:
        if f"{key}_dismissed" not in st.session_state:
            st.session_state[f"{key}_dismissed"] = False
        
        if not st.session_state[f"{key}_dismissed"]:
            col1, col2 = st.columns([11, 1])
            with col1:
                if title:
                    st.markdown(f"**{icon} {title}**")
                getattr(st, alert_type)(message)
            with col2:
                if st.button("✕", key=f"{key}_dismiss", help="Dismiss"):
                    st.session_state[f"{key}_dismissed"] = True
                    st.rerun()
    else:
        if title:
            st.markdown(f"**{icon} {title}**")
        getattr(st, alert_type)(message)


def render_chart_card(
    title: str,
    chart: Optional[go.Figure] = None,
    data: Optional[Dict[str, Any]] = None,
    chart_type: str = "line",
    height: int = 300,
    footer: Optional[str] = None,
    key: Optional[str] = None
):
    """
    Render a card with embedded chart.
    
    Args:
        title: Card title
        chart: Plotly figure (if provided, overrides data)
        data: Data for simple charts
        chart_type: Type of chart if using data
        height: Chart height in pixels
        footer: Footer text
        key: Unique key
    """
    with st.container():
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown(f'<h4>{title}</h4>', unsafe_allow_html=True)
        
        if chart:
            st.plotly_chart(chart, use_container_width=True, height=height)
        elif data:
            # Create simple chart from data
            fig = create_simple_chart(data, chart_type, height)
            st.plotly_chart(fig, use_container_width=True)
        
        if footer:
            st.markdown(f'<div class="card-footer">{footer}</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)


def create_simple_chart(data: Dict[str, Any], chart_type: str, height: int) -> go.Figure:
    """Create a simple chart from data."""
    fig = go.Figure()
    
    if chart_type == "line":
        fig.add_trace(go.Scatter(
            x=data.get("x", []),
            y=data.get("y", []),
            name=data.get("name", "Value"),
            mode="lines+markers"
        ))
    elif chart_type == "bar":
        fig.add_trace(go.Bar(
            x=data.get("x", []),
            y=data.get("y", []),
            name=data.get("name", "Value")
        ))
    elif chart_type == "area":
        fig.add_trace(go.Scatter(
            x=data.get("x", []),
            y=data.get("y", []),
            name=data.get("name", "Value"),
            fill="tozeroy"
        ))
    
    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False
    )
    
    return fig


def render_stat_cards(
    stats: List[Dict[str, Any]],
    columns: int = 4,
    key: Optional[str] = None
):
    """
    Render multiple metric cards in a grid.
    
    Args:
        stats: List of stat dictionaries with title, value, delta, etc.
        columns: Number of columns in grid
        key: Base key for components
    """
    rows = (len(stats) + columns - 1) // columns
    
    for row in range(rows):
        cols = st.columns(columns)
        for col_idx in range(columns):
            stat_idx = row * columns + col_idx
            if stat_idx < len(stats):
                stat = stats[stat_idx]
                with cols[col_idx]:
                    render_metric_card(
                        title=stat.get("title", ""),
                        value=stat.get("value", ""),
                        delta=stat.get("delta"),
                        delta_color=stat.get("delta_color"),
                        subtitle=stat.get("subtitle"),
                        icon=stat.get("icon"),
                        key=f"{key}_{stat_idx}" if key else None
                    )