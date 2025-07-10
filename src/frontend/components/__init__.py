"""
Components package for reusable UI elements
"""

from .websocket_client import EnhancedWebSocketClient
from .websocket_manager import get_websocket_client, register_page, is_websocket_connected, cleanup_websocket
from .api_client import APIClient
from .auto_refresh import AutoRefreshManager
from .charts import (
    create_candlestick_chart,
    create_portfolio_chart,
    create_signal_chart,
    create_correlation_heatmap,
    create_performance_chart
)
from .metrics import (
    display_price_metrics,
    display_portfolio_metrics,
    display_signal_metrics,
    display_risk_metrics
)

__all__ = [
    'EnhancedWebSocketClient',
    'get_websocket_client',
    'register_page',
    'is_websocket_connected',
    'cleanup_websocket',
    'APIClient',
    'AutoRefreshManager',
    'create_candlestick_chart',
    'create_portfolio_chart', 
    'create_signal_chart',
    'create_correlation_heatmap',
    'create_performance_chart',
    'display_price_metrics',
    'display_portfolio_metrics',
    'display_signal_metrics',
    'display_risk_metrics'
]
