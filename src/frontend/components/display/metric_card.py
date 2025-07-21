"""Metric Card Display Component"""
import streamlit as st

def render_metric_card(title, value, change=None, change_prefix="", format_func=None):
    """
    Render a compact metric card
    
    Args:
        title: Card title
        value: Main metric value
        change: Change value (optional)
        change_prefix: Prefix for change (e.g., '+', '-')
        format_func: Function to format the value
    """
    # Format value if function provided
    display_value = format_func(value) if format_func else str(value)
    
    # Determine change class
    change_class = ""
    if change is not None:
        if isinstance(change, (int, float)):
            change_class = "positive" if change >= 0 else "negative"
            change_display = f"{change_prefix}{abs(change):.2f}%"
        else:
            change_display = str(change)
    
    # Build HTML
    card_html = f"""
    <div class="metric-card">
        <div class="metric-card-header">
            <div class="metric-card-title">{title}</div>
        </div>
        <div class="metric-card-value">{display_value}</div>
    """
    
    if change is not None:
        card_html += f"""
        <div class="metric-card-change {change_class}">
            <span>{change_display}</span>
        </div>
        """
    
    card_html += "</div>"
    
    st.markdown(card_html, unsafe_allow_html=True)

def render_metric_row(metrics):
    """
    Render a row of metric cards
    
    Args:
        metrics: List of metric dictionaries with keys: title, value, change, format_func
    """
    cols = st.columns(len(metrics))
    
    for col, metric in zip(cols, metrics):
        with col:
            render_metric_card(
                title=metric.get("title"),
                value=metric.get("value"),
                change=metric.get("change"),
                change_prefix=metric.get("change_prefix", ""),
                format_func=metric.get("format_func")
            )

def format_currency(value):
    """Format value as currency"""
    return f"${value:,.2f}"

def format_bitcoin(value):
    """Format value as Bitcoin"""
    return f"₿{value:.8f}"

def format_percentage(value):
    """Format value as percentage"""
    return f"{value:.2f}%"

def format_number(value):
    """Format large numbers with commas"""
    return f"{value:,.0f}"