"""
Demo page showcasing all UI components.
Run with: streamlit run demo.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Import all UI components
from . import *

st.set_page_config(
    page_title="UI Components Demo",
    page_icon="🎨",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: var(--secondary-background-color);
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .metric-header {
        font-size: 0.875rem;
        color: var(--text-color);
        margin-bottom: 0.5rem;
    }
    .metric-subtitle {
        font-size: 0.75rem;
        color: var(--text-color);
        opacity: 0.7;
    }
    .info-card {
        background-color: var(--secondary-background-color);
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .chart-card {
        background-color: var(--secondary-background-color);
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .card-footer {
        margin-top: 1rem;
        padding-top: 1rem;
        border-top: 1px solid var(--text-color);
        opacity: 0.7;
        font-size: 0.875rem;
    }
</style>
""", unsafe_allow_html=True)

# Page header
create_page_header(
    title="UI Components Demo",
    subtitle="Showcase of all reusable UI components",
    icon="🎨",
    breadcrumbs=["Home", "Components", "Demo"],
    actions=[
        {"label": "Refresh", "key": "refresh_demo", "callback": lambda: st.rerun()},
        {"label": "Export", "key": "export_demo", "type": "primary"}
    ]
)

# Tab navigation
tab_idx = render_tabs(
    ["Cards", "Tables", "Forms", "Layout", "Tabs"],
    key="main_tabs"
)

if tab_idx == 0:  # Cards
    st.header("Card Components")
    
    # Metric Cards
    st.subheader("Metric Cards")
    metrics = [
        {"title": "Total Revenue", "value": "$125,432", "delta": "+12.5%", "icon": "💰", "subtitle": "Last 30 days"},
        {"title": "Active Users", "value": "1,234", "delta": "+5.2%", "icon": "👥", "subtitle": "Currently online"},
        {"title": "Conversion Rate", "value": "3.4%", "delta": "-0.5%", "delta_color": "inverse", "icon": "📈"},
        {"title": "Avg Response Time", "value": "245ms", "delta": "-12ms", "icon": "⚡", "subtitle": "99th percentile"}
    ]
    render_stat_cards(metrics, columns=4)
    
    add_vertical_space(2)
    
    # Info Cards
    st.subheader("Info Cards")
    col1, col2 = st.columns(2)
    
    with col1:
        render_info_card(
            title="About This System",
            content="""
            This is a comprehensive Bitcoin trading system with:
            - Real-time price monitoring
            - AI-powered predictions
            - Paper trading capabilities
            - Advanced analytics
            """,
            icon="ℹ️",
            footer="Last updated: 2 hours ago",
            expandable=True,
            expanded=False
        )
    
    with col2:
        render_info_card(
            title="Quick Stats",
            content="""
            **Performance Metrics:**
            - Win Rate: 65%
            - Sharpe Ratio: 1.85
            - Max Drawdown: -12%
            - Total Trades: 156
            """,
            icon="📊",
            expandable=False
        )
    
    # Alert Cards
    st.subheader("Alert Cards")
    render_alert_card(
        "System is operating normally. All services are online.",
        status="success",
        title="System Status",
        key="success_alert"
    )
    
    render_alert_card(
        "API rate limit approaching. Consider reducing request frequency.",
        status="warning",
        title="Rate Limit Warning",
        key="warning_alert"
    )
    
    # Chart Cards
    st.subheader("Chart Cards")
    
    # Generate sample data
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    prices = 40000 + np.cumsum(np.random.randn(30) * 500)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=prices,
        mode='lines',
        name='BTC Price',
        line=dict(color='#00D4FF', width=2)
    ))
    fig.update_layout(
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        height=300
    )
    
    render_chart_card(
        title="Bitcoin Price Trend",
        chart=fig,
        footer="Data updated every 5 minutes"
    )

elif tab_idx == 1:  # Tables
    st.header("Table Components")
    
    # Generate sample data
    df = pd.DataFrame({
        'Symbol': ['BTC', 'ETH', 'SOL', 'ADA', 'DOT'],
        'Price': [42150.23, 2234.56, 98.34, 0.45, 6.78],
        'Change 24h': [2.34, -1.23, 5.67, -0.89, 3.45],
        'Volume': [23456789012, 12345678901, 3456789012, 1234567890, 2345678901],
        'Market Cap': [823456789012, 268345678901, 43456789012, 16234567890, 8345678901]
    })
    
    # Condensed Table
    st.subheader("Condensed Table")
    render_condensed_table(
        df,
        title="Top Cryptocurrencies",
        max_rows=3,
        highlight_columns=['Symbol', 'Price'],
        format_dict={
            'Price': "${:.2f}",
            'Change 24h': "{:.2f}%",
            'Volume': "${:,.0f}",
            'Market Cap': "${:,.0f}"
        }
    )
    
    add_vertical_space(2)
    
    # Enhanced Data Table
    st.subheader("Enhanced Data Table with Features")
    table = DataTable(
        df,
        key="crypto_table",
        page_size=3,
        sortable=True,
        exportable=True,
        searchable=True
    )
    table.render()
    
    add_vertical_space(2)
    
    # Comparison Table
    st.subheader("Comparison Table")
    comparison_data = pd.DataFrame({
        'Strategy A': [15.2, 1.85, 0.65, -12.5],
        'Strategy B': [18.7, 2.10, 0.72, -15.3],
        'Strategy C': [12.4, 1.65, 0.58, -10.2]
    }, index=['Return %', 'Sharpe Ratio', 'Win Rate', 'Max Drawdown %'])
    
    render_comparison_table(
        comparison_data.T,
        baseline_column='Strategy A',
        comparison_columns=['Strategy B', 'Strategy C'],
        metrics=['Return %', 'Sharpe Ratio', 'Win Rate', 'Max Drawdown %'],
        title="Strategy Comparison"
    )

elif tab_idx == 2:  # Forms
    st.header("Form Components")
    
    # Input Groups
    st.subheader("Input Groups")
    
    col1, col2 = st.columns(2)
    
    with col1:
        email = render_input_group(
            label="Email Address",
            input_type="email",
            key="demo_email",
            placeholder="user@example.com",
            required=True,
            help_text="We'll never share your email"
        )
        
        amount = render_input_group(
            label="Investment Amount",
            input_type="number",
            key="demo_amount",
            default=10000.0,
            prefix="$",
            suffix="USD",
            min_value=100.0,
            max_value=1000000.0,
            step=100.0
        )
    
    with col2:
        strategy = render_input_group(
            label="Trading Strategy",
            input_type="select",
            key="demo_strategy",
            options=["Conservative", "Moderate", "Aggressive"],
            default="Moderate",
            help_text="Select your risk tolerance"
        )
        
        features = render_input_group(
            label="Features to Enable",
            input_type="multiselect",
            key="demo_features",
            options=["Auto Trading", "Stop Loss", "Take Profit", "Trailing Stop"],
            default=["Stop Loss", "Take Profit"]
        )
    
    add_vertical_space(2)
    
    # Preset Selector
    st.subheader("Preset Configurations")
    
    presets = {
        "Conservative": {
            "position_size": 0.1,
            "stop_loss": 0.02,
            "take_profit": 0.05,
            "max_trades": 5
        },
        "Moderate": {
            "position_size": 0.2,
            "stop_loss": 0.03,
            "take_profit": 0.08,
            "max_trades": 10
        },
        "Aggressive": {
            "position_size": 0.3,
            "stop_loss": 0.05,
            "take_profit": 0.15,
            "max_trades": 20
        }
    }
    
    selected_preset = render_preset_selector(
        label="Select Trading Preset",
        presets=presets,
        key="demo_preset"
    )
    
    # Dynamic Form
    st.subheader("Dynamic Form Example")
    
    form_sections = [
        {
            "name": "basic",
            "title": "Basic Information",
            "columns": 2,
            "fields": [
                {"name": "first_name", "label": "First Name", "type": "text", "required": True},
                {"name": "last_name", "label": "Last Name", "type": "text", "required": True},
                {"name": "phone", "label": "Phone", "type": "text"},
                {"name": "country", "label": "Country", "type": "select", "kwargs": {"options": ["USA", "UK", "Canada", "Other"]}}
            ]
        },
        {
            "name": "trading",
            "title": "Trading Preferences",
            "columns": 1,
            "fields": [
                {"name": "experience", "label": "Trading Experience", "type": "slider", "kwargs": {"min_value": 0, "max_value": 10}},
                {"name": "goals", "label": "Trading Goals", "type": "textarea", "placeholder": "Describe your trading goals..."}
            ]
        }
    ]
    
    form_data = render_dynamic_form(
        sections=form_sections,
        key="demo_dynamic_form",
        submit_label="Submit Application"
    )
    
    if form_data:
        st.success("Form submitted successfully!")
        st.json(form_data)

elif tab_idx == 3:  # Layout
    st.header("Layout Components")
    
    # Grid Layout
    st.subheader("Grid Layout")
    
    grid_items = [
        {"title": "Item 1", "value": 123},
        {"title": "Item 2", "value": 456},
        {"title": "Item 3", "value": 789},
        {"title": "Item 4", "value": 101},
        {"title": "Item 5", "value": 112},
        {"title": "Item 6", "value": 131}
    ]
    
    def render_grid_item(item):
        with st.container():
            st.metric(item["title"], item["value"])
    
    create_grid(grid_items, columns=3, gap="medium", render_func=render_grid_item)
    
    add_vertical_space(2)
    
    # Sections
    st.subheader("Section Containers")
    
    section1 = create_section(
        title="Expandable Section",
        subtitle="This section can be collapsed",
        icon="📁",
        expandable=True,
        expanded=True,
        help_text="This is a help message for the section"
    )
    
    with section1:
        st.write("This is the content inside the expandable section.")
        st.code("print('Hello, World!')")
    
    # Card Grid
    st.subheader("Card Grid Layout")
    
    cards = [
        {
            "title": "Performance",
            "metric": {"label": "Total Return", "value": "+24.5%", "delta": "+2.3%"},
            "content": "Last 30 days performance"
        },
        {
            "title": "Risk Metrics",
            "metric": {"label": "Sharpe Ratio", "value": "1.85"},
            "content": "Risk-adjusted returns"
        },
        {
            "title": "Activity",
            "metric": {"label": "Trades Today", "value": "12", "delta": "+3"},
            "content": "Trading activity"
        }
    ]
    
    create_card_grid(cards, columns=3)

elif tab_idx == 4:  # Tabs
    st.header("Tab Components")
    
    st.subheader("Basic Tabs")
    basic_tab = render_tabs(
        ["Overview", "Details", "Settings"],
        key="demo_basic_tabs"
    )
    
    if basic_tab == 0:
        st.write("This is the Overview tab content")
    elif basic_tab == 1:
        st.write("This is the Details tab content")
    else:
        st.write("This is the Settings tab content")
    
    add_vertical_space(2)
    
    st.subheader("Icon Tabs")
    icon_tabs = [
        {"name": "Dashboard", "icon": "📊"},
        {"name": "Analytics", "icon": "📈"},
        {"name": "Reports", "icon": "📄"},
        {"name": "Settings", "icon": "⚙️"}
    ]
    
    icon_tab = render_icon_tabs(icon_tabs, key="demo_icon_tabs")
    
    if icon_tab == 0:
        st.write("Dashboard content goes here")
    elif icon_tab == 1:
        st.write("Analytics content goes here")
    elif icon_tab == 2:
        st.write("Reports content goes here")
    else:
        st.write("Settings content goes here")

# Footer
st.markdown("---")
st.markdown("*UI Components Demo - Bitcoin Trading System*")