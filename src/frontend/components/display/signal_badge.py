"""Signal Badge Display Component"""
import streamlit as st

def render_signal_badge(signal, confidence=None):
    """
    Render a signal badge
    
    Args:
        signal: Signal type ('buy', 'sell', 'hold')
        confidence: Confidence percentage (optional)
    """
    signal_class = f"signal-{signal.lower()}"
    signal_text = signal.upper()
    
    badge_html = f"""
    <span class="signal-badge {signal_class}">
        {signal_text}
    """
    
    if confidence is not None:
        badge_html += f"""<span class="signal-confidence">{confidence:.0f}%</span>"""
    
    badge_html += "</span>"
    
    st.markdown(badge_html, unsafe_allow_html=True)

def render_signal_row(signal_data):
    """
    Render a row with signal information
    
    Args:
        signal_data: Dictionary with signal information
    """
    col1, col2, col3, col4 = st.columns([2, 1, 1, 2])
    
    with col1:
        st.markdown(f"""
        <div class="signal-timestamp">{signal_data.get('timestamp', '')}</div>
        """, unsafe_allow_html=True)
    
    with col2:
        render_signal_badge(
            signal_data.get('signal', 'hold'),
            signal_data.get('confidence')
        )
    
    with col3:
        price = signal_data.get('price', 0)
        st.markdown(f"""
        <div class="signal-price">${price:,.2f}</div>
        """, unsafe_allow_html=True)
    
    with col4:
        reason = signal_data.get('reason', '')
        st.markdown(f"""
        <div class="signal-reason">{reason}</div>
        """, unsafe_allow_html=True)

def render_signal_panel(signals, title="Recent Signals"):
    """
    Render a panel of signals
    
    Args:
        signals: List of signal dictionaries
        title: Panel title
    """
    st.markdown(f"""
    <div class="signals-panel">
        <h3>{title}</h3>
        <div class="signals-list">
    """, unsafe_allow_html=True)
    
    if not signals:
        st.info("No signals available")
    else:
        for signal in signals[:10]:  # Show last 10 signals
            render_signal_row(signal)
            st.markdown("<hr style='margin: 8px 0; opacity: 0.2;'>", unsafe_allow_html=True)
    
    st.markdown("</div></div>", unsafe_allow_html=True)