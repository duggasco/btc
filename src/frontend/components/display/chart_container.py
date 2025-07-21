"""Chart Container Display Component"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

def create_chart_container(title, chart_func, controls=None):
    """
    Create a chart container with header and controls
    
    Args:
        title: Chart title
        chart_func: Function that returns a plotly figure
        controls: List of control dictionaries
    """
    # Chart header with title
    st.markdown(f"<h3 class='chart-title'>{title}</h3>", unsafe_allow_html=True)
    
    # Render controls in a single row if provided
    if controls:
        # Create a container for controls to avoid deep nesting
        control_container = st.container()
        with control_container:
            # Use a horizontal layout for controls
            cols = st.columns([1] * len(controls) + [3])  # Last column takes remaining space
            for i, control in enumerate(controls):
                with cols[i]:
                    if control['type'] == 'button':
                        if st.button(control['label'], key=control.get('key')):
                            control['callback']()
                    elif control['type'] == 'select':
                        value = st.selectbox(
                            control.get('label', ''),
                            control['options'],
                            key=control.get('key'),
                            label_visibility="collapsed"
                        )
                        if value != control.get('default'):
                            control['callback'](value)
    
    # Chart container
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    
    # Render chart
    fig = chart_func()
    if fig:
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    st.markdown('</div>', unsafe_allow_html=True)

def apply_dark_theme(fig):
    """Apply dark theme to plotly figure"""
    fig.update_layout(
        paper_bgcolor='rgba(19, 19, 21, 0)',
        plot_bgcolor='rgba(19, 19, 21, 0)',
        font_color='#e5e5e7',
        font_family='Inter, sans-serif',
        font_size=12,
        margin=dict(l=0, r=0, t=30, b=0),
        showlegend=True,
        legend=dict(
            bgcolor='rgba(26, 26, 29, 0.8)',
            bordercolor='#27272a',
            borderwidth=1
        ),
        xaxis=dict(
            gridcolor='#27272a',
            zerolinecolor='#27272a',
            tickfont=dict(size=11)
        ),
        yaxis=dict(
            gridcolor='#27272a',
            zerolinecolor='#27272a',
            tickfont=dict(size=11)
        )
    )
    return fig

def create_price_chart(df, title="BTC Price"):
    """Create a candlestick price chart"""
    if df.empty:
        return None
    
    fig = go.Figure()
    
    # Add candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price',
        increasing_line_color='#22c55e',
        decreasing_line_color='#ef4444'
    ))
    
    # Add volume bars
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['Volume'],
        name='Volume',
        yaxis='y2',
        marker_color='rgba(156, 163, 175, 0.3)'
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        yaxis=dict(title='Price ($)', side='left'),
        yaxis2=dict(title='Volume', side='right', overlaying='y'),
        hovermode='x unified',
        height=400
    )
    
    return apply_dark_theme(fig)

def create_line_chart(df, y_column, title="", color=None):
    """Create a simple line chart"""
    if df.empty:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df[y_column],
        mode='lines',
        name=y_column,
        line=dict(color=color or '#f7931a', width=2)
    ))
    
    fig.update_layout(
        title=title,
        showlegend=False,
        height=300
    )
    
    return apply_dark_theme(fig)

def create_portfolio_chart(portfolio_history):
    """Create portfolio value chart"""
    if not portfolio_history or len(portfolio_history) == 0:
        return None
    
    fig = go.Figure()
    
    # Add portfolio value line
    fig.add_trace(go.Scatter(
        x=list(range(len(portfolio_history))),
        y=portfolio_history,
        mode='lines+markers',
        name='Portfolio Value',
        line=dict(color='#f7931a', width=2),
        marker=dict(size=4)
    ))
    
    # Add baseline
    fig.add_hline(
        y=portfolio_history[0] if portfolio_history else 0,
        line_dash="dash",
        line_color="#6b7280",
        annotation_text="Initial Value"
    )
    
    fig.update_layout(
        title="Portfolio Performance",
        xaxis_title="Time",
        yaxis_title="Value ($)",
        height=300
    )
    
    return apply_dark_theme(fig)