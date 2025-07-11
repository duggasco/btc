import requests
import json
import os
from datetime import datetime
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class DiscordNotifier:
    def __init__(self, webhook_url: Optional[str] = None):
        self.webhook_url = webhook_url or os.getenv('DISCORD_WEBHOOK_URL')
        self.enabled = bool(self.webhook_url)
        
        if not self.enabled:
            logger.warning("Discord webhook URL not provided. Notifications disabled.")
        
        # Track last signal to detect changes
        self.last_signal = None
        self.last_price = None
        self.daily_high = None
        self.daily_low = None
        self.daily_open = None
        
    def _send_webhook(self, embed: Dict[str, Any]) -> bool:
        """Send embed to Discord webhook"""
        if not self.enabled:
            return False
            
        try:
            response = requests.post(
                self.webhook_url,
                json={"embeds": [embed]},
                timeout=10
            )
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Failed to send Discord notification: {e}")
            return False
    
    def _format_price(self, price: float) -> str:
        """Format price with proper decimals and commas"""
        return f"${price:,.2f}"
    
    def _get_signal_color(self, signal: str) -> int:
        """Get color code for signal type"""
        colors = {
            'buy': 0x00FF00,  # Green
            'sell': 0xFF0000,  # Red
            'hold': 0xFFFF00,  # Yellow
            'bullish': 0x00FF00,
            'bearish': 0xFF0000,
            'neutral': 0xFFFF00
        }
        return colors.get(signal.lower(), 0x808080)
    
    def notify_signal_update(self, signal: str, confidence: float, predicted_price: float, current_price: float):
        """Notify when AI signal updates"""
        # Check if signal changed
        signal_changed = self.last_signal != signal
        self.last_signal = signal
        
        embed = {
            "title": f"AI Signal Update{' - CHANGED!' if signal_changed else ''}",
            "color": self._get_signal_color(signal),
            "fields": [
                {"name": "Signal", "value": f"**{signal.upper()}**", "inline": True},
                {"name": "Confidence", "value": f"{confidence:.1%}", "inline": True},
                {"name": "Current Price", "value": self._format_price(current_price), "inline": True},
                {"name": "Predicted Price", "value": self._format_price(predicted_price), "inline": True},
                {"name": "Expected Move", "value": f"{((predicted_price/current_price - 1) * 100):+.1f}%", "inline": True}
            ],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if signal_changed:
            embed["fields"].append({
                "name": "Signal Changed",
                "value": f"Previous signal was {self.last_signal}",
                "inline": False
            })
        
        self._send_webhook(embed)
    
    def notify_price_alert(self, current_price: float, price_change_pct: float, is_positive: bool):
        """Notify significant price movements (+/- 2.5%)"""
        self.last_price = current_price
        
        embed = {
            "title": f"Price Alert - {'+' if is_positive else ''}{price_change_pct:.1f}% Move",
            "color": 0x00FF00 if is_positive else 0xFF0000,
            "fields": [
                {"name": "Current Price", "value": self._format_price(current_price), "inline": True},
                {"name": "24h Change", "value": f"{price_change_pct:+.1f}%", "inline": True},
                {"name": "Daily Range", "value": f"{self._format_price(self.daily_low or current_price)} - {self._format_price(self.daily_high or current_price)}", "inline": True}
            ],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self._send_webhook(embed)
    
    def notify_trade_executed(self, trade_id: str, trade_type: str, price: float, size: float, 
                            lot_id: str, trade_value: float):
        """Notify when a trade is executed"""
        embed = {
            "title": f"Trade Executed - {trade_type.upper()}",
            "color": self._get_signal_color(trade_type),
            "fields": [
                {"name": "Trade ID", "value": f"`{trade_id[:8]}...`", "inline": True},
                {"name": "Type", "value": trade_type.upper(), "inline": True},
                {"name": "Price", "value": self._format_price(price), "inline": True},
                {"name": "Size", "value": f"{size:.6f} BTC", "inline": True},
                {"name": "Value", "value": self._format_price(trade_value), "inline": True},
                {"name": "Lot ID", "value": f"`{lot_id[:8]}...`" if lot_id else "Auto-generated", "inline": True},
                {"name": "Timestamp", "value": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "inline": False}
            ],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self._send_webhook(embed)
    
    def notify_limit_triggered(self, limit_type: str, trigger_price: float, current_price: float, 
                             size: Optional[float] = None):
        """Notify when a limit order is triggered"""
        embed = {
            "title": f"Limit Order Triggered - {limit_type.replace('_', ' ').title()}",
            "color": 0xFF9500,  # Orange
            "fields": [
                {"name": "Limit Type", "value": limit_type.replace('_', ' ').title(), "inline": True},
                {"name": "Trigger Price", "value": self._format_price(trigger_price), "inline": True},
                {"name": "Current Price", "value": self._format_price(current_price), "inline": True}
            ],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if size:
            embed["fields"].append({"name": "Size", "value": f"{size:.6f} BTC", "inline": True})
        
        self._send_webhook(embed)
    
    def notify_pnl_change(self, total_pnl: float, daily_pnl: float, pnl_change_pct: float):
        """Notify major P&L changes"""
        is_positive = pnl_change_pct > 0
        
        embed = {
            "title": f"Major P&L Change - {'+' if is_positive else ''}{pnl_change_pct:.1f}%",
            "color": 0x00FF00 if is_positive else 0xFF0000,
            "fields": [
                {"name": "Total P&L", "value": self._format_price(total_pnl), "inline": True},
                {"name": "Today's P&L", "value": self._format_price(daily_pnl), "inline": True},
                {"name": "Change %", "value": f"{pnl_change_pct:+.1f}%", "inline": True}
            ],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self._send_webhook(embed)
    
    def notify_system_status(self, status: str, message: str):
        """Notify system status changes"""
        colors = {
            'online': 0x00FF00,
            'offline': 0xFF0000,
            'warning': 0xFFFF00,
            'error': 0xFF0000
        }
        
        embed = {
            "title": f"System Status - {status.title()}",
            "color": colors.get(status.lower(), 0x808080),
            "description": message,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self._send_webhook(embed)
    
    def update_daily_stats(self, open_price: float, high: float, low: float):
        """Update daily price statistics for alerts"""
        self.daily_open = open_price
        self.daily_high = high
        self.daily_low = low
    
    def send_notification(self, message: str, notification_type: str = "info"):
        """Generic notification method for sending custom messages"""
        type_colors = {
            'info': 0x3498db,      # Blue
            'success': 0x2ecc71,   # Green
            'warning': 0xf39c12,   # Orange
            'error': 0xe74c3c,     # Red
            'signal': 0x9b59b6     # Purple
        }
        
        type_icons = {
            'info': '[INFO]',
            'success': '[SUCCESS]',
            'warning': '[WARNING]',
            'error': '[ERROR]',
            'signal': '[SIGNAL]'
        }
        
        embed = {
            "title": f"{type_icons.get(notification_type, '[NOTIFICATION]')} {notification_type.title()} Notification",
            "description": message,
            "color": type_colors.get(notification_type, 0x808080),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return self._send_webhook(embed)
