import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional, List
from utils.helpers import format_currency, format_percentage

def display_price_metrics(data: Dict[str, Any], columns: Optional[List] = None):
    """Display enhanced price metrics"""
    if not columns:
        columns = st.columns(4)
    
    with columns[0]:
        current_price = data.get("current_price", 0)
        price_change = data.get("price_change_percentage_24h", 0)
        st.metric(
            "Current Price",
            format_currency(current_price),
            delta=format_percentage(price_change),
            delta_color="normal"
        )
    
    with columns[1]:
        high_24h = data.get("high_24h", 0)
        high_diff = ((current_price / high_24h) - 1) * 100 if high_24h > 0 else 0
        st.metric(
            "24h High",
            format_currency(high_24h),
            delta=f"{high_diff:.1f}% from high"
        )
    
    with columns[2]:
        low_24h = data.get("low_24h", 0)
        low_diff = ((current_price / low_24h) - 1) * 100 if low_24h > 0 else 0
        st.metric(
            "24h Low", 
            format_currency(low_24h),
            delta=f"{low_diff:.1f}% from low"
        )
    
    with columns[3]:
        volume = data.get("total_volume", 0)
        volume_change = data.get("volume_change_24h", 0)
        st.metric(
            "24h Volume",
            f"${volume/1e9:.2f}B" if volume > 1e9 else f"${volume/1e6:.1f}M",
            delta=format_percentage(volume_change) if volume_change else None
        )

def display_portfolio_metrics(metrics: Dict[str, Any]):
    """Display comprehensive portfolio metrics"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_value = metrics.get("total_value", 0)
        value_change = metrics.get("value_change_24h", 0)
        st.metric(
            "Total Value",
            format_currency(total_value),
            delta=format_currency(value_change) if value_change else None
        )
    
    with col2:
        pnl = metrics.get("total_pnl", 0)
        pnl_pct = metrics.get("total_pnl_percent", 0)
        st.metric(
            "Total P&L",
            format_currency(pnl),
            delta=format_percentage(pnl_pct),
            delta_color="normal"
        )
    
    with col3:
        win_rate = metrics.get("win_rate", 0)
        win_rate_change = metrics.get("win_rate_change", 0)
        st.metric(
            "Win Rate",
            format_percentage(win_rate * 100, decimals=1),
            delta=f"{win_rate_change:+.1f}pp" if win_rate_change else None
        )
    
    with col4:
        sharpe = metrics.get("sharpe_ratio", 0)
        sharpe_change = metrics.get("sharpe_change", 0)
        st.metric(
            "Sharpe Ratio",
            f"{sharpe:.2f}",
            delta=f"{sharpe_change:+.2f}" if sharpe_change else None
        )

def display_risk_metrics(risk_data: Dict[str, Any]):
    """Display comprehensive risk management metrics"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        var_95 = risk_data.get("var_95", 0)
        st.metric(
            "VaR (95%)",
            format_currency(var_95),
            help="Value at Risk - Maximum expected loss at 95% confidence"
        )
    
    with col2:
        max_dd = risk_data.get("max_drawdown", 0)
        current_dd = risk_data.get("current_drawdown", 0)
        st.metric(
            "Max Drawdown",
            format_percentage(max_dd * 100),
            delta=f"Current: {current_dd*100:.1f}%"
        )
    
    with col3:
        beta = risk_data.get("beta", 0)
        st.metric(
            "Beta",
            f"{beta:.2f}",
            help="Market correlation coefficient"
        )
    
    with col4:
        risk_score = risk_data.get("risk_score", 50)
        risk_level = "Low" if risk_score < 30 else "High" if risk_score > 70 else "Medium"
        risk_color = "🟢" if risk_score < 30 else "🔴" if risk_score > 70 else "🟡"
        st.metric(
            "Risk Score",
            f"{risk_color} {risk_score}/100",
            delta=risk_level
        )


def display_signal_metrics(signal_data: Dict[str, Any]):
    """Display AI signal metrics"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        signal = signal_data.get("signal", "HOLD")
        signal_color = "🟢" if signal == "BUY" else "🔴" if signal == "SELL" else "🟡"
        st.metric(
            "Signal",
            f"{signal_color} {signal}",
            help="Current AI trading signal"
        )
    
    with col2:
        confidence = signal_data.get("confidence", 0)
        st.metric(
            "Confidence",
            format_percentage(confidence * 100),
            delta=f"{(confidence - 0.5) * 100:+.1f}%" if confidence != 0.5 else None
        )
    
    with col3:
        predicted_price = signal_data.get("predicted_price", 0)
        current_price = signal_data.get("current_price", 0)
        price_change = ((predicted_price / current_price) - 1) * 100 if current_price > 0 else 0
        st.metric(
            "Predicted Price",
            format_currency(predicted_price),
            delta=f"{price_change:+.1f}%"
        )
    
    with col4:
        strength = signal_data.get("signal_strength", 0)
        strength_level = "Strong" if strength > 0.7 else "Weak" if strength < 0.3 else "Moderate"
        st.metric(
            "Signal Strength",
            f"{strength:.2f}",
            delta=strength_level
        )
