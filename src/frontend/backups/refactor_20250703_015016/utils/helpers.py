"""
Helper functions for the Streamlit application
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Union, List, Dict, Any

def format_currency(value: float, symbol: str = "$", decimals: int = 2) -> str:
    """Format value as currency"""
    if pd.isna(value):
        return f"{symbol}0.00"
    return f"{symbol}{value:,.{decimals}f}"

def format_percentage(value: float, decimals: int = 2, with_sign: bool = True) -> str:
    """Format value as percentage"""
    if pd.isna(value):
        return "0.00%"
    
    formatted = f"{value:.{decimals}f}%"
    if with_sign and value > 0:
        formatted = f"+{formatted}"
    return formatted

def calculate_returns(prices: pd.Series) -> pd.Series:
    """Calculate returns from price series"""
    return prices.pct_change().fillna(0)

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """Calculate Sharpe ratio"""
    excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
    if excess_returns.std() == 0:
        return 0
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

def calculate_max_drawdown(prices: pd.Series) -> float:
    """Calculate maximum drawdown"""
    cumulative = (1 + calculate_returns(prices)).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()

def get_time_period(period: str) -> timedelta:
    """Convert period string to timedelta"""
    period_map = {
        "1h": timedelta(hours=1),
        "4h": timedelta(hours=4),
        "1d": timedelta(days=1),
        "7d": timedelta(days=7),
        "30d": timedelta(days=30),
        "90d": timedelta(days=90),
        "180d": timedelta(days=180),
        "1y": timedelta(days=365),
        "2y": timedelta(days=730)
    }
    return period_map.get(period, timedelta(days=7))

def validate_signal(signal: Dict[str, Any]) -> bool:
    """Validate signal data structure"""
    required_fields = ['signal', 'confidence', 'timestamp']
    return all(field in signal for field in required_fields)

def aggregate_signals(signals: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate multiple signals into consensus"""
    if not signals:
        return {"signal": "hold", "confidence": 0.0}
    
    # Count signal types
    signal_counts = {"buy": 0, "sell": 0, "hold": 0}
    total_confidence = 0
    
    for signal in signals:
        if validate_signal(signal):
            signal_counts[signal['signal']] += 1
            total_confidence += signal['confidence']
    
    # Determine consensus
    total_signals = sum(signal_counts.values())
    if total_signals == 0:
        return {"signal": "hold", "confidence": 0.0}
    
    # Get dominant signal
    dominant_signal = max(signal_counts, key=signal_counts.get)
    consensus_confidence = signal_counts[dominant_signal] / total_signals
    avg_confidence = total_confidence / total_signals
    
    return {
        "signal": dominant_signal,
        "confidence": consensus_confidence * avg_confidence,
        "distribution": signal_counts
    }
