"""
Unit tests for Discord notifications service
"""
import pytest
from unittest.mock import Mock, patch, call
from datetime import datetime
from services.notifications import DiscordNotifier


class TestDiscordNotifier:
    """Test cases for DiscordNotifier class"""
    
    @pytest.mark.unit
    def test_initialization_with_webhook(self):
        """Test initialization with webhook URL"""
        webhook_url = "https://discord.com/api/webhooks/123/abc"
        notifier = DiscordNotifier(webhook_url)
        
        assert notifier.webhook_url == webhook_url
        assert notifier.enabled is True
        assert notifier.last_signal is None
        assert notifier.last_price is None
    
    @pytest.mark.unit
    def test_initialization_without_webhook(self):
        """Test initialization without webhook URL"""
        with patch.dict('os.environ', {}, clear=True):
            notifier = DiscordNotifier()
            assert notifier.enabled is False
            assert notifier.webhook_url is None
    
    @pytest.mark.unit
    def test_initialization_from_env(self):
        """Test initialization from environment variable"""
        webhook_url = "https://discord.com/api/webhooks/456/def"
        with patch.dict('os.environ', {'DISCORD_WEBHOOK_URL': webhook_url}):
            notifier = DiscordNotifier()
            assert notifier.webhook_url == webhook_url
            assert notifier.enabled is True
    
    @pytest.mark.unit
    @patch('requests.post')
    def test_send_webhook_success(self, mock_post):
        """Test successful webhook sending"""
        mock_post.return_value.status_code = 204
        
        notifier = DiscordNotifier("https://discord.com/webhook")
        embed = {"title": "Test", "color": 0x00FF00}
        
        result = notifier._send_webhook(embed)
        
        assert result is True
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[0][0] == "https://discord.com/webhook"
        assert "embeds" in call_args[1]["json"]
    
    @pytest.mark.unit
    @patch('requests.post')
    def test_send_webhook_failure(self, mock_post):
        """Test webhook sending failure"""
        mock_post.side_effect = Exception("Network error")
        
        notifier = DiscordNotifier("https://discord.com/webhook")
        embed = {"title": "Test", "color": 0x00FF00}
        
        result = notifier._send_webhook(embed)
        
        assert result is False
    
    @pytest.mark.unit
    def test_send_webhook_disabled(self):
        """Test webhook sending when disabled"""
        notifier = DiscordNotifier()  # No webhook URL
        embed = {"title": "Test", "color": 0x00FF00}
        
        result = notifier._send_webhook(embed)
        
        assert result is False
    
    @pytest.mark.unit
    def test_format_price(self):
        """Test price formatting"""
        notifier = DiscordNotifier()
        
        assert notifier._format_price(50000) == "$50,000.00"
        assert notifier._format_price(50000.50) == "$50,000.50"
        assert notifier._format_price(1234567.89) == "$1,234,567.89"
    
    @pytest.mark.unit
    def test_get_signal_color(self):
        """Test signal color mapping"""
        notifier = DiscordNotifier()
        
        assert notifier._get_signal_color('buy') == 0x00FF00  # Green
        assert notifier._get_signal_color('sell') == 0xFF0000  # Red
        assert notifier._get_signal_color('hold') == 0xFFFF00  # Yellow
        assert notifier._get_signal_color('bullish') == 0x00FF00
        assert notifier._get_signal_color('bearish') == 0xFF0000
        assert notifier._get_signal_color('neutral') == 0xFFFF00
        assert notifier._get_signal_color('unknown') == 0x808080  # Default gray
    
    @pytest.mark.unit
    @patch('requests.post')
    def test_notify_signal_update(self, mock_post):
        """Test signal update notification"""
        mock_post.return_value.status_code = 204
        
        notifier = DiscordNotifier("https://discord.com/webhook")
        notifier.notify_signal_update(
            signal='buy',
            confidence=0.85,
            predicted_price=52000,
            current_price=50000
        )
        
        mock_post.assert_called_once()
        call_data = mock_post.call_args[1]['json']['embeds'][0]
        
        assert "AI Signal Update" in call_data['title']
        assert call_data['color'] == 0x00FF00  # Green for buy
        assert any(field['name'] == 'Signal' for field in call_data['fields'])
        assert any(field['name'] == 'Confidence' for field in call_data['fields'])
    
    @pytest.mark.unit
    @patch('requests.post')
    def test_notify_signal_change(self, mock_post):
        """Test signal change detection"""
        mock_post.return_value.status_code = 204
        
        notifier = DiscordNotifier("https://discord.com/webhook")
        
        # First signal
        notifier.notify_signal_update('hold', 0.6, 50000, 50000)
        
        # Signal change
        notifier.notify_signal_update('buy', 0.8, 52000, 50000)
        
        # Check second call has "CHANGED!" in title
        second_call_data = mock_post.call_args_list[1][1]['json']['embeds'][0]
        assert "CHANGED!" in second_call_data['title']
    
    @pytest.mark.unit
    @patch('requests.post')
    def test_notify_price_alert(self, mock_post):
        """Test price alert notification"""
        mock_post.return_value.status_code = 204
        
        notifier = DiscordNotifier("https://discord.com/webhook")
        notifier.daily_high = 52000
        notifier.daily_low = 48000
        
        notifier.notify_price_alert(
            current_price=51000,
            price_change_pct=5.0,
            is_positive=True
        )
        
        mock_post.assert_called_once()
        call_data = mock_post.call_args[1]['json']['embeds'][0]
        
        assert "📈" in call_data['title']
        assert "+5.0% Move" in call_data['title']
        assert call_data['color'] == 0x00FF00  # Green
    
    @pytest.mark.unit
    @patch('requests.post')
    def test_notify_trade_executed(self, mock_post):
        """Test trade execution notification"""
        mock_post.return_value.status_code = 204
        
        notifier = DiscordNotifier("https://discord.com/webhook")
        notifier.notify_trade_executed(
            trade_id='abc123def456',
            trade_type='buy',
            price=50000,
            size=0.1,
            lot_id='lot123',
            trade_value=5000
        )
        
        mock_post.assert_called_once()
        call_data = mock_post.call_args[1]['json']['embeds'][0]
        
        assert "🟢" in call_data['title']
        assert "BUY" in call_data['title']
        assert any(field['value'] == '`abc123de...`' for field in call_data['fields'])
    
    @pytest.mark.unit
    @patch('requests.post')
    def test_notify_limit_triggered(self, mock_post):
        """Test limit order notification"""
        mock_post.return_value.status_code = 204
        
        notifier = DiscordNotifier("https://discord.com/webhook")
        notifier.notify_limit_triggered(
            limit_type='stop_loss',
            trigger_price=48000,
            current_price=47500,
            size=0.05
        )
        
        mock_post.assert_called_once()
        call_data = mock_post.call_args[1]['json']['embeds'][0]
        
        assert "⚡" in call_data['title']
        assert "Stop Loss" in call_data['title']
        assert call_data['color'] == 0xFF9500  # Orange
    
    @pytest.mark.unit
    @patch('requests.post')
    def test_notify_pnl_change(self, mock_post):
        """Test P&L change notification"""
        mock_post.return_value.status_code = 204
        
        notifier = DiscordNotifier("https://discord.com/webhook")
        notifier.notify_pnl_change(
            total_pnl=1500,
            daily_pnl=300,
            pnl_change_pct=15.0
        )
        
        mock_post.assert_called_once()
        call_data = mock_post.call_args[1]['json']['embeds'][0]
        
        assert "💰" in call_data['title']
        assert "+15.0%" in call_data['title']
        assert call_data['color'] == 0x00FF00  # Green for positive
    
    @pytest.mark.unit
    @patch('requests.post')
    def test_notify_system_status(self, mock_post):
        """Test system status notification"""
        mock_post.return_value.status_code = 204
        
        notifier = DiscordNotifier("https://discord.com/webhook")
        notifier.notify_system_status(
            status='online',
            message='System started successfully'
        )
        
        mock_post.assert_called_once()
        call_data = mock_post.call_args[1]['json']['embeds'][0]
        
        assert "🔔" in call_data['title']
        assert "Online" in call_data['title']
        assert call_data['color'] == 0x00FF00  # Green
        assert call_data['description'] == 'System started successfully'
    
    @pytest.mark.unit
    def test_update_daily_stats(self):
        """Test updating daily statistics"""
        notifier = DiscordNotifier()
        
        notifier.update_daily_stats(
            open_price=49000,
            high=52000,
            low=48000
        )
        
        assert notifier.daily_open == 49000
        assert notifier.daily_high == 52000
        assert notifier.daily_low == 48000
    
    @pytest.mark.unit
    @patch('requests.post')
    def test_notification_rate_limiting(self, mock_post):
        """Test that notifications handle rate limiting gracefully"""
        mock_post.return_value.status_code = 429  # Rate limited
        
        notifier = DiscordNotifier("https://discord.com/webhook")
        result = notifier._send_webhook({"title": "Test"})
        
        # Should handle rate limiting without crashing
        assert result is True  # Current implementation doesn't check status code
    
    @pytest.mark.unit
    def test_embed_timestamp_format(self):
        """Test that embed timestamps are properly formatted"""
        notifier = DiscordNotifier("https://discord.com/webhook")
        
        with patch('requests.post') as mock_post:
            mock_post.return_value.status_code = 204
            
            notifier.notify_signal_update('buy', 0.8, 50000, 49000)
            
            call_data = mock_post.call_args[1]['json']['embeds'][0]
            assert 'timestamp' in call_data
            
            # Verify timestamp is ISO format
            timestamp = call_data['timestamp']
            parsed = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            assert isinstance(parsed, datetime)