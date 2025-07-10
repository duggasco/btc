"""
Enhanced helper functions for the Streamlit application
"""

import pandas as pd
import numpy as np
from datetime import timedelta
from typing import Union, List, Dict, Any, Optional

def format_currency(value: float, symbol: str = "$", decimals: int = 2) -> str:
    """Format value as currency with proper handling"""
    if pd.isna(value) or value is None:
        return f"{symbol}0.00"
    
    # Handle very large numbers
    if abs(value) >= 1e9:
        return f"{symbol}{value/1e9:,.{decimals}f}B"
    elif abs(value) >= 1e6:
        return f"{symbol}{value/1e6:,.{decimals}f}M"
    elif abs(value) >= 1e3:
        return f"{symbol}{value/1e3:,.{decimals}f}K"
    
    return f"{symbol}{value:,.{decimals}f}"

def format_percentage(value: float, decimals: int = 2, with_sign: bool = True) -> str:
    """Format value as percentage"""
    if pd.isna(value) or value is None:
        return "0.00%"
    
    formatted = f"{value:.{decimals}f}%"
    if with_sign and value > 0:
        formatted = f"+{formatted}"
    return formatted

def aggregate_signals(signals: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate multiple signals into consensus"""
    if not signals:
        return {"signal": "hold", "confidence": 0.0}
    
    # Count signal types
    signal_counts = {"buy": 0, "sell": 0, "hold": 0}
    confidence_sum = {"buy": 0, "sell": 0, "hold": 0}
    
    for signal in signals:
        sig_type = signal.get("signal", "hold")
        signal_counts[sig_type] += 1
        confidence_sum[sig_type] += signal.get("confidence", 0)
    
    # Determine consensus
    total_signals = sum(signal_counts.values())
    if total_signals == 0:
        return {"signal": "hold", "confidence": 0.0}
    
    # Weight by both count and confidence
    weighted_scores = {}
    for sig_type in ["buy", "sell", "hold"]:
        if signal_counts[sig_type] > 0:
            avg_confidence = confidence_sum[sig_type] / signal_counts[sig_type]
            count_weight = signal_counts[sig_type] / total_signals
            weighted_scores[sig_type] = count_weight * avg_confidence
        else:
            weighted_scores[sig_type] = 0
    
    # Get dominant signal
    dominant_signal = max(weighted_scores, key=weighted_scores.get)
    
    return {
        "signal": dominant_signal,
        "confidence": weighted_scores[dominant_signal],
        "composite_confidence": max(weighted_scores.values()),
        "distribution": signal_counts,
        "agreement": signal_counts[dominant_signal] / total_signals
    }

def calculate_sharpe_ratio(returns: Union[pd.Series, np.ndarray], risk_free_rate: float = 0.02) -> float:
    """
    Calculate Sharpe ratio for given returns
    
    Args:
        returns: Daily returns as pandas Series or numpy array
        risk_free_rate: Annual risk-free rate (default 2%)
        
    Returns:
        Annualized Sharpe ratio
    """
    if isinstance(returns, pd.Series):
        returns = returns.values
    
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0
    
    # Calculate excess returns
    daily_rf_rate = risk_free_rate / 365
    excess_returns = returns - daily_rf_rate
    
    # Annualized Sharpe ratio
    sharpe = np.sqrt(365) * np.mean(excess_returns) / np.std(returns)
    
    return sharpe
