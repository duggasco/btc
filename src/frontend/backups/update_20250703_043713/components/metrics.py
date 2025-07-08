import streamlit as st
from typing import Dict, Any, Optional

def format_currency(value: float, symbol: str = "$", decimals: int = 2) -> str:
    """Format value as currency"""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return f"{symbol}0.00"
    return f"{symbol}{value:,.{decimals}f}"

def format_percentage(value: float, decimals: int = 2, with_sign: bool = True) -> str:
    """Format value as percentage"""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "0.00%"
    
    formatted = f"{value:.{decimals}f}%"
    if with_sign and value > 0:
        formatted = f"+{formatted}"
    return formatted

def display_price_metrics(data: Dict[str, Any], columns: Optional[list] = None):
    """Display price-related metrics in columns"""
    if not columns:
        columns = st.columns(4)
    
    with columns[0]:
        st.metric(
            "Current Price",
            format_currency(data.get('current_price', 0)),
            delta=format_percentage(data.get('price_change_percentage_24h', 0))
        )
    
    with columns[1]:
        st.metric(
            "24h High",
            format_currency(data.get('high_24h', 0))
        )
    
    with columns[2]:
        st.metric(
            "24h Low", 
            format_currency(data.get('low_24h', 0))
        )
    
    with columns[3]:
        st.metric(
            "24h Volume",
            format_currency(data.get('total_volume', 0), decimals=0)
        )

def display_portfolio_metrics(metrics: Dict[str, Any]):
    """Display portfolio metrics"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Value",
            format_currency(metrics.get('total_value', 0))
        )
    
    with col2:
        pnl = metrics.get('total_pnl', 0)
        pnl_pct = metrics.get('total_pnl_percent', 0)
        st.metric(
            "Total P&L",
            format_currency(pnl),
            delta=format_percentage(pnl_pct)
        )
    
    with col3:
        st.metric(
            "Win Rate",
            format_percentage(metrics.get('win_rate', 0) * 100, decimals=1)
        )
    
    with col4:
        st.metric(
            "Sharpe Ratio",
            f"{metrics.get('sharpe_ratio', 0):.2f}"
        )

def display_signal_metrics(signal_data: Dict[str, Any]):
    """Display signal metrics with visual indicators"""
    signal = signal_data.get('signal', 'hold')
    confidence = signal_data.get('confidence', 0)
    
    # Signal indicator
    if signal == 'buy':
        st.success(f"🟢 **BUY Signal** ({confidence:.1%} confidence)")
    elif signal == 'sell':
        st.error(f"🔴 **SELL Signal** ({confidence:.1%} confidence)")
    else:
        st.info(f"⚪ **HOLD Signal** ({confidence:.1%} confidence)")
    
    # Additional details
    if 'predicted_price' in signal_data:
        st.metric(
            "Predicted Price",
            format_currency(signal_data['predicted_price'])
        )

def display_risk_metrics(risk_data: Dict[str, Any]):
    """Display risk management metrics"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Value at Risk (95%)",
            format_currency(risk_data.get('var_95', 0))
        )
    
    with col2:
        st.metric(
            "Max Drawdown",
            format_percentage(risk_data.get('max_drawdown', 0) * 100)
        )
    
    with col3:
        st.metric(
            "Beta",
            f"{risk_data.get('beta', 0):.2f}"
        )
