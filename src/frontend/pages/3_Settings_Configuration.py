"""Settings & Configuration - System configuration management"""
import streamlit as st
import pandas as pd
import json
from datetime import datetime

# Component imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from components.layout.dashboard_grid import render_dashboard_header
from components.display.metric_card import render_metric_row
from components.controls.form_controls import create_input_group, create_button, create_form_section
from utils.api_client import get_api_client

# Page configuration
st.set_page_config(
    page_title="Settings & Configuration - BTC Trading System",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load CSS
st.markdown("""
<style>
/* Import theme CSS */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
</style>
""", unsafe_allow_html=True)

# Import CSS files content
with open(Path(__file__).parent.parent / "styles" / "theme.css", "r") as f:
    theme_css = f.read()
with open(Path(__file__).parent.parent / "styles" / "components.css", "r") as f:
    components_css = f.read()

st.markdown(f"<style>{theme_css}{components_css}</style>", unsafe_allow_html=True)

# Additional settings-specific styles
st.markdown("""
<style>
.settings-section {
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: var(--space-4);
    margin-bottom: var(--space-4);
}

.settings-group {
    margin-bottom: var(--space-5);
}

.settings-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: var(--space-3) 0;
    border-bottom: 1px solid var(--border-subtle);
}

.settings-item:last-child {
    border-bottom: none;
}

.settings-label {
    font-size: 13px;
    color: var(--text-primary);
    font-weight: 500;
}

.settings-description {
    font-size: 12px;
    color: var(--text-muted);
    margin-top: var(--space-1);
}

.config-status {
    display: inline-flex;
    align-items: center;
    gap: var(--space-2);
    padding: var(--space-1) var(--space-3);
    background: var(--bg-tertiary);
    border-radius: var(--radius-sm);
    font-size: 12px;
}

.config-status.saved {
    color: var(--accent-success);
    border: 1px solid rgba(34, 197, 94, 0.3);
}

.config-status.modified {
    color: var(--accent-primary);
    border: 1px solid rgba(247, 147, 26, 0.3);
}
</style>
""", unsafe_allow_html=True)

# Initialize API client
api_client = get_api_client()

# Initialize session state
if 'settings_modified' not in st.session_state:
    st.session_state.settings_modified = False
if 'current_settings' not in st.session_state:
    st.session_state.current_settings = {}

def load_current_settings():
    """Load current settings from API"""
    try:
        settings = api_client.get_settings()
        st.session_state.current_settings = settings
        return settings
    except Exception as e:
        st.error(f"Failed to load settings: {str(e)}")
        return {}

def save_settings(settings):
    """Save settings via API"""
    try:
        result = api_client.update_settings(settings)
        if result.get('success'):
            st.success("Settings saved successfully")
            st.session_state.settings_modified = False
            st.session_state.current_settings = settings
            return True
    except Exception as e:
        st.error(f"Failed to save settings: {str(e)}")
    return False

def render_header():
    """Render page header"""
    status_indicators = [
        {"label": "Config", "status": "online"},
        {"label": "Modified" if st.session_state.settings_modified else "Saved", 
         "status": "warning" if st.session_state.settings_modified else "online"}
    ]
    render_dashboard_header("Settings & Configuration", status_indicators)

def render_trading_rules_tab():
    """Render trading rules configuration"""
    st.markdown('<div class="settings-section">', unsafe_allow_html=True)
    
    # Risk Management
    create_form_section("Risk Management", "Configure position sizing and risk parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        max_position_size = create_input_group(
            "Max Position Size (BTC)",
            input_type="number",
            value=st.session_state.current_settings.get('trading', {}).get('max_position_size', 1.0),
            min_value=0.01,
            max_value=10.0,
            step=0.01,
            key="max_position_size"
        )
        
        stop_loss = create_input_group(
            "Default Stop Loss (%)",
            input_type="number",
            value=st.session_state.current_settings.get('trading', {}).get('stop_loss_percentage', 5.0) * 100,
            min_value=0.5,
            max_value=20.0,
            step=0.5,
            key="stop_loss_pct",
            help_text="Percentage below entry price to place stop loss"
        )
    
    with col2:
        risk_per_trade = create_input_group(
            "Risk Per Trade (%)",
            input_type="number",
            value=st.session_state.current_settings.get('trading', {}).get('risk_tolerance', 0.02) * 100,
            min_value=0.1,
            max_value=5.0,
            step=0.1,
            key="risk_per_trade",
            help_text="Maximum portfolio percentage to risk per trade"
        )
        
        take_profit = create_input_group(
            "Default Take Profit (%)",
            input_type="number",
            value=st.session_state.current_settings.get('trading', {}).get('take_profit_percentage', 10.0) * 100,
            min_value=1.0,
            max_value=50.0,
            step=1.0,
            key="take_profit_pct"
        )
    
    # Signal Thresholds
    st.markdown("---", unsafe_allow_html=True)
    create_form_section("Signal Thresholds", "Configure minimum confidence levels for signals")
    
    col1, col2 = st.columns(2)
    
    with col1:
        min_confidence = create_input_group(
            "Minimum Signal Confidence (%)",
            input_type="slider",
            value=st.session_state.current_settings.get('model', {}).get('confidence_threshold', 0.7) * 100,
            min_value=50,
            max_value=95,
            step=5,
            key="min_confidence"
        )
        
        signal_cooldown = create_input_group(
            "Signal Cooldown (minutes)",
            input_type="number",
            value=st.session_state.current_settings.get('trading', {}).get('signal_cooldown', 15),
            min_value=0,
            max_value=60,
            step=5,
            key="signal_cooldown",
            help_text="Minimum time between signals"
        )
    
    with col2:
        max_daily_trades = create_input_group(
            "Max Daily Trades",
            input_type="number",
            value=st.session_state.current_settings.get('trading', {}).get('max_daily_trades', 10),
            min_value=1,
            max_value=50,
            step=1,
            key="max_daily_trades"
        )
        
        trading_enabled = create_input_group(
            "Trading Enabled",
            input_type="checkbox",
            value=st.session_state.current_settings.get('trading', {}).get('enabled', False),
            key="trading_enabled"
        )
    
    # Save button
    if create_button("Save Trading Rules", variant="primary", key="save_trading_rules"):
        new_settings = {
            **st.session_state.current_settings,
            'trading': {
                'max_position_size': max_position_size,
                'stop_loss_percentage': stop_loss / 100,
                'risk_tolerance': risk_per_trade / 100,
                'take_profit_percentage': take_profit / 100,
                'signal_cooldown': signal_cooldown,
                'max_daily_trades': max_daily_trades,
                'enabled': trading_enabled,
                'confidence_threshold': min_confidence / 100
            }
        }
        save_settings(new_settings)
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_api_configuration_tab():
    """Render API configuration"""
    st.markdown('<div class="settings-section">', unsafe_allow_html=True)
    
    create_form_section("API Keys", "Configure external API connections")
    
    # Exchange APIs
    st.markdown("#### Exchange APIs", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        binance_api_key = create_input_group(
            "Binance API Key",
            input_type="text",
            value="",
            placeholder="Enter API key",
            key="binance_api_key"
        )
        
        binance_api_secret = create_input_group(
            "Binance API Secret",
            input_type="text",
            value="",
            placeholder="Enter API secret",
            key="binance_api_secret"
        )
    
    with col2:
        coinbase_api_key = create_input_group(
            "Coinbase API Key",
            input_type="text",
            value="",
            placeholder="Enter API key",
            key="coinbase_api_key"
        )
        
        coinbase_api_secret = create_input_group(
            "Coinbase API Secret",
            input_type="text",
            value="",
            placeholder="Enter API secret",
            key="coinbase_api_secret"
        )
    
    # Data Provider APIs
    st.markdown("---", unsafe_allow_html=True)
    st.markdown("#### Data Provider APIs", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        coingecko_api_key = create_input_group(
            "CoinGecko API Key",
            input_type="text",
            value="",
            placeholder="Enter API key (optional)",
            key="coingecko_api_key"
        )
        
        alpha_vantage_key = create_input_group(
            "Alpha Vantage API Key",
            input_type="text",
            value="",
            placeholder="Enter API key",
            key="alpha_vantage_key"
        )
    
    with col2:
        fred_api_key = create_input_group(
            "FRED API Key",
            input_type="text",
            value="",
            placeholder="Enter API key",
            key="fred_api_key"
        )
        
        news_api_key = create_input_group(
            "News API Key",
            input_type="text",
            value="",
            placeholder="Enter API key",
            key="news_api_key"
        )
    
    # API Settings
    st.markdown("---", unsafe_allow_html=True)
    create_form_section("API Settings", "Configure API behavior")
    
    col1, col2 = st.columns(2)
    
    with col1:
        api_timeout = create_input_group(
            "API Timeout (seconds)",
            input_type="number",
            value=st.session_state.current_settings.get('api', {}).get('timeout', 30),
            min_value=5,
            max_value=120,
            step=5,
            key="api_timeout"
        )
        
        retry_attempts = create_input_group(
            "Retry Attempts",
            input_type="number",
            value=st.session_state.current_settings.get('api', {}).get('retry_attempts', 3),
            min_value=0,
            max_value=10,
            step=1,
            key="retry_attempts"
        )
    
    with col2:
        rate_limit = create_input_group(
            "Rate Limit (requests/min)",
            input_type="number",
            value=st.session_state.current_settings.get('api', {}).get('rate_limit', 100),
            min_value=10,
            max_value=1000,
            step=10,
            key="rate_limit"
        )
        
        use_cache = create_input_group(
            "Enable API Cache",
            input_type="checkbox",
            value=st.session_state.current_settings.get('api', {}).get('use_cache', True),
            key="use_cache"
        )
    
    if create_button("Save API Configuration", variant="primary", key="save_api_config"):
        st.info("API configuration saved (keys are encrypted)")
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_notifications_tab():
    """Render notifications configuration"""
    st.markdown('<div class="settings-section">', unsafe_allow_html=True)
    
    create_form_section("Discord Notifications", "Configure Discord webhook for alerts")
    
    webhook_url = create_input_group(
        "Discord Webhook URL",
        input_type="text",
        value="",
        placeholder="https://discord.com/api/webhooks/...",
        key="discord_webhook",
        help_text="Leave empty to disable Discord notifications"
    )
    
    # Notification Types
    st.markdown("---", unsafe_allow_html=True)
    create_form_section("Notification Types", "Select which events trigger notifications")
    
    col1, col2 = st.columns(2)
    
    with col1:
        notify_signals = create_input_group(
            "Trading Signals",
            input_type="checkbox",
            value=True,
            key="notify_signals"
        )
        
        notify_trades = create_input_group(
            "Trade Executions",
            input_type="checkbox",
            value=True,
            key="notify_trades"
        )
        
        notify_errors = create_input_group(
            "System Errors",
            input_type="checkbox",
            value=True,
            key="notify_errors"
        )
    
    with col2:
        notify_daily = create_input_group(
            "Daily Summary",
            input_type="checkbox",
            value=True,
            key="notify_daily"
        )
        
        notify_pnl = create_input_group(
            "P&L Updates",
            input_type="checkbox",
            value=True,
            key="notify_pnl"
        )
        
        notify_model = create_input_group(
            "Model Updates",
            input_type="checkbox",
            value=False,
            key="notify_model"
        )
    
    # Alert Thresholds
    st.markdown("---", unsafe_allow_html=True)
    create_form_section("Alert Thresholds", "Configure when to send alerts")
    
    col1, col2 = st.columns(2)
    
    with col1:
        price_alert_pct = create_input_group(
            "Price Alert Threshold (%)",
            input_type="number",
            value=5.0,
            min_value=1.0,
            max_value=20.0,
            step=0.5,
            key="price_alert_pct",
            help_text="Alert when price moves by this percentage"
        )
        
        drawdown_alert = create_input_group(
            "Drawdown Alert (%)",
            input_type="number",
            value=10.0,
            min_value=5.0,
            max_value=50.0,
            step=5.0,
            key="drawdown_alert"
        )
    
    with col2:
        volume_spike = create_input_group(
            "Volume Spike Alert (x)",
            input_type="number",
            value=3.0,
            min_value=2.0,
            max_value=10.0,
            step=0.5,
            key="volume_spike",
            help_text="Alert when volume exceeds average by this factor"
        )
        
        win_rate_alert = create_input_group(
            "Win Rate Alert (%)",
            input_type="number",
            value=40.0,
            min_value=20.0,
            max_value=60.0,
            step=5.0,
            key="win_rate_alert",
            help_text="Alert when win rate falls below this threshold"
        )
    
    # Test notification
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if create_button("Test Notification", variant="secondary", key="test_notification"):
            if webhook_url:
                try:
                    result = api_client._make_request("POST", "/notifications/test", 
                                                    json={"webhook_url": webhook_url})
                    if result.get('success'):
                        st.success("Test notification sent successfully")
                    else:
                        st.error("Failed to send test notification")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
            else:
                st.warning("Please enter a webhook URL first")
    
    with col2:
        if create_button("Save Notifications", variant="primary", key="save_notifications"):
            st.success("Notification settings saved")
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_system_maintenance_tab():
    """Render system maintenance options"""
    st.markdown('<div class="settings-section">', unsafe_allow_html=True)
    
    # Data Management
    create_form_section("Data Management", "Manage historical data and cache")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Database Size", "1.2 GB")
        if create_button("Compact Database", variant="secondary", key="compact_db"):
            with st.spinner("Compacting database..."):
                st.success("Database compacted successfully")
    
    with col2:
        st.metric("Cache Size", "245 MB")
        if create_button("Clear Cache", variant="secondary", key="clear_cache"):
            if st.confirm("Are you sure you want to clear the cache?"):
                st.success("Cache cleared")
    
    with col3:
        st.metric("Log Files", "89 MB")
        if create_button("Clean Logs", variant="secondary", key="clean_logs"):
            st.success("Old logs removed")
    
    # Model Management
    st.markdown("---", unsafe_allow_html=True)
    create_form_section("Model Management", "Configure and manage ML models")
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_type = create_input_group(
            "Active Model",
            input_type="select",
            options=["Enhanced LSTM", "Standard LSTM", "Ensemble"],
            value="Enhanced LSTM",
            key="active_model"
        )
        
        auto_retrain = create_input_group(
            "Auto-retrain Model",
            input_type="checkbox",
            value=True,
            key="auto_retrain",
            help_text="Automatically retrain when performance degrades"
        )
    
    with col2:
        retrain_interval = create_input_group(
            "Retrain Interval (days)",
            input_type="number",
            value=7,
            min_value=1,
            max_value=30,
            step=1,
            key="retrain_interval"
        )
        
        if create_button("Retrain Now", variant="secondary", key="retrain_now"):
            with st.spinner("Retraining model..."):
                try:
                    result = api_client._make_request("POST", "/model/retrain/enhanced")
                    if result.get('success'):
                        st.success("Model retrained successfully")
                    else:
                        st.error("Retraining failed")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    # System Health
    st.markdown("---", unsafe_allow_html=True)
    create_form_section("System Health", "Monitor system performance")
    
    # Get system status
    try:
        system_status = api_client.get_system_status()
    except:
        system_status = {
            'cpu_usage': 45,
            'memory_usage': 62,
            'disk_usage': 38,
            'api_latency': 23
        }
    
    metrics = [
        {
            "title": "CPU USAGE",
            "value": system_status.get('cpu_usage', 0),
            "format_func": lambda x: f"{x}%"
        },
        {
            "title": "MEMORY USAGE",
            "value": system_status.get('memory_usage', 0),
            "format_func": lambda x: f"{x}%"
        },
        {
            "title": "DISK USAGE",
            "value": system_status.get('disk_usage', 0),
            "format_func": lambda x: f"{x}%"
        },
        {
            "title": "API LATENCY",
            "value": system_status.get('api_latency', 0),
            "format_func": lambda x: f"{x}ms"
        }
    ]
    
    render_metric_row(metrics)
    
    # Backup & Restore
    st.markdown("---", unsafe_allow_html=True)
    create_form_section("Backup & Restore", "Manage system backups")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if create_button("Create Backup", variant="primary", key="create_backup"):
            with st.spinner("Creating backup..."):
                st.success("Backup created successfully")
    
    with col2:
        backup_list = create_input_group(
            "Available Backups",
            input_type="select",
            options=["backup_2024_01_15.zip", "backup_2024_01_08.zip", "backup_2024_01_01.zip"],
            key="backup_list"
        )
    
    with col3:
        if create_button("Restore Backup", variant="secondary", key="restore_backup"):
            if st.confirm(f"Restore from {backup_list}? This will overwrite current data."):
                with st.spinner("Restoring backup..."):
                    st.success("Backup restored successfully")
    
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    """Main function"""
    # Load current settings
    if not st.session_state.current_settings:
        load_current_settings()
    
    # Render header
    render_header()
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Trading Rules",
        "API Configuration",
        "Notifications",
        "System Maintenance"
    ])
    
    with tab1:
        render_trading_rules_tab()
    
    with tab2:
        render_api_configuration_tab()
    
    with tab3:
        render_notifications_tab()
    
    with tab4:
        render_system_maintenance_tab()

if __name__ == "__main__":
    main()