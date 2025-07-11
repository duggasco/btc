import streamlit as st
import pandas as pd
import json
import sys
import os
from datetime import datetime, timedelta
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components.api_client import APIClient
from components.page_styling import setup_page
from components.data_uploader import DataUploader
from utils.helpers import format_currency, format_percentage

# Setup page with consistent styling
api_client = setup_page(
    page_name="Settings",
    page_title="System Settings",
    page_subtitle="Configure trading parameters, API connections, and monitor system health"
)

# Additional page-specific CSS
st.markdown("""
<style>
/* Settings specific styling */
.settings-section {
    background: var(--bg-secondary);
    padding: 1.5rem;
    border-radius: 10px;
    border: 1px solid var(--border-subtle);
    margin-bottom: 1.5rem;
}

.settings-section h3 {
    color: var(--text-primary);
    margin-bottom: 1.5rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid var(--border-subtle);
}

.config-item {
    background: var(--bg-tertiary);
    padding: 1rem;
    border-radius: 8px;
    margin-bottom: 1rem;
    border: 1px solid var(--border-subtle);
}

.config-label {
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 0.5rem;
}

.config-value {
    color: var(--text-secondary);
    font-size: 0.95rem;
}

.save-button {
    background: var(--accent-primary);
    color: white;
    padding: 0.8rem 2rem;
    border-radius: 8px;
    border: none;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s ease;
}

.save-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(247, 147, 26, 0.4);
    background: #e6851a;
}

.status-indicator {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    margin-right: 0.5rem;
}

.status-active { background: var(--accent-success); }
.status-inactive { background: var(--accent-danger); }

.data-quality-metric {
    background: var(--bg-tertiary);
    padding: 1.2rem;
    border-radius: 8px;
    text-align: center;
    margin-bottom: 1rem;
    border: 1px solid var(--border-subtle);
}

.metric-value {
    font-size: 2rem;
    font-weight: 600;
    color: var(--text-primary);
}

.metric-label {
    color: var(--text-secondary);
    font-size: 0.9rem;
    margin-top: 0.5rem;
}

/* Status badges */
.status-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 4px;
    font-size: 0.875rem;
    font-weight: 500;
}

.status-online { 
    background: rgba(34, 197, 94, 0.15); 
    color: var(--accent-success);
    border: 1px solid rgba(34, 197, 94, 0.3);
}
.status-offline { 
    background: rgba(239, 68, 68, 0.15); 
    color: var(--accent-danger);
    border: 1px solid rgba(239, 68, 68, 0.3);
}
.status-warning { 
    background: rgba(245, 158, 11, 0.15); 
    color: var(--accent-warning);
    border: 1px solid rgba(245, 158, 11, 0.3);
}
</style>
""", unsafe_allow_html=True)

# The api_client is already initialized by setup_page

# Main tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🔧 Trading Config",
    "🔌 API Settings",
    "📊 Data Quality",
    "🔔 Notifications",
    "🛡️ System Health",
    "📤 Data Upload"
])

# Trading Configuration Tab
with tab1:
    st.subheader("Trading Configuration")
    
    # Load current configuration
    try:
        trading_config = api_client.get("/config/trading") or {}
        signal_weights = api_client.get("/config/signal-weights") or {}
        risk_config = api_client.get("/config/risk-management") or {}
    except:
        trading_config = {}
        signal_weights = {}
        risk_config = {}
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Trading Rules
        st.markdown('<div class="settings-section">', unsafe_allow_html=True)
        st.markdown("### Trading Rules")
        
        # Position sizing
        st.markdown("#### Position Sizing")
        default_position_size = st.slider(
            "Default Position Size (%)",
            min_value=1,
            max_value=100,
            value=trading_config.get("default_position_size", 10),
            help="Percentage of capital to allocate per trade"
        )
        
        max_positions = st.number_input(
            "Maximum Concurrent Positions",
            min_value=1,
            max_value=20,
            value=trading_config.get("max_positions", 5),
            help="Maximum number of open positions allowed"
        )
        
        # Risk management
        st.markdown("#### Risk Management")
        col_r1, col_r2 = st.columns(2)
        
        with col_r1:
            stop_loss = st.number_input(
                "Default Stop Loss (%)",
                min_value=0.0,
                max_value=50.0,
                value=risk_config.get("default_stop_loss", 5.0),
                step=0.5,
                help="Default stop loss percentage"
            )
            
            max_drawdown = st.number_input(
                "Max Drawdown Limit (%)",
                min_value=5.0,
                max_value=50.0,
                value=risk_config.get("max_drawdown_limit", 20.0),
                step=1.0,
                help="Stop trading if drawdown exceeds this limit"
            )
        
        with col_r2:
            take_profit = st.number_input(
                "Default Take Profit (%)",
                min_value=0.0,
                max_value=100.0,
                value=risk_config.get("default_take_profit", 10.0),
                step=0.5,
                help="Default take profit percentage"
            )
            
            risk_per_trade = st.number_input(
                "Max Risk Per Trade (%)",
                min_value=0.1,
                max_value=10.0,
                value=risk_config.get("max_risk_per_trade", 2.0),
                step=0.1,
                help="Maximum capital at risk per trade"
            )
        
        # Signal thresholds
        st.markdown("#### Signal Thresholds")
        col_s1, col_s2 = st.columns(2)
        
        with col_s1:
            min_confidence = st.slider(
                "Minimum Signal Confidence",
                min_value=0.0,
                max_value=1.0,
                value=trading_config.get("min_confidence", 0.6),
                step=0.05,
                help="Minimum confidence required to execute trades"
            )
        
        with col_s2:
            signal_cooldown = st.number_input(
                "Signal Cooldown (minutes)",
                min_value=0,
                max_value=60,
                value=trading_config.get("signal_cooldown", 5),
                help="Wait time between signals"
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Signal Weights
        st.markdown('<div class="settings-section">', unsafe_allow_html=True)
        st.markdown("### Signal Weights")
        
        st.info("Adjust the importance of different signal types (must sum to 100%)")
        
        col_w1, col_w2 = st.columns(2)
        
        with col_w1:
            technical_weight = st.slider(
                "Technical Analysis",
                min_value=0,
                max_value=100,
                value=int(signal_weights.get("technical", 40) * 100),
                help="Weight for technical indicators"
            )
            
            sentiment_weight = st.slider(
                "Sentiment Analysis",
                min_value=0,
                max_value=100,
                value=int(signal_weights.get("sentiment", 20) * 100),
                help="Weight for market sentiment"
            )
        
        with col_w2:
            onchain_weight = st.slider(
                "On-Chain Metrics",
                min_value=0,
                max_value=100,
                value=int(signal_weights.get("onchain", 30) * 100),
                help="Weight for blockchain data"
            )
            
            macro_weight = st.slider(
                "Macro Indicators",
                min_value=0,
                max_value=100,
                value=int(signal_weights.get("macro", 10) * 100),
                help="Weight for macroeconomic data"
            )
        
        # Weight validation
        total_weight = technical_weight + sentiment_weight + onchain_weight + macro_weight
        if total_weight != 100:
            st.error(f"Weights must sum to 100%. Current total: {total_weight}%")
        else:
            st.success("✅ Weight configuration is valid")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Trading modes
        st.markdown('<div class="settings-section">', unsafe_allow_html=True)
        st.markdown("### Trading Modes")
        
        paper_trading = st.checkbox(
            "Paper Trading Mode",
            value=trading_config.get("paper_trading_enabled", True),
            help="Practice with virtual funds"
        )
        
        auto_trading = st.checkbox(
            "Auto Trading",
            value=trading_config.get("auto_trading_enabled", False),
            help="Execute trades automatically based on signals",
            disabled=paper_trading
        )
        
        if auto_trading and not paper_trading:
            st.warning("⚠️ Auto trading with real funds enabled!")
        
        st.markdown("#### Order Types")
        
        use_market_orders = st.checkbox(
            "Use Market Orders",
            value=trading_config.get("use_market_orders", True),
            help="Execute at current market price"
        )
        
        use_limit_orders = st.checkbox(
            "Use Limit Orders",
            value=trading_config.get("use_limit_orders", False),
            help="Set specific entry prices"
        )
        
        if use_limit_orders:
            limit_offset = st.number_input(
                "Limit Order Offset (%)",
                min_value=0.0,
                max_value=5.0,
                value=trading_config.get("limit_offset", 0.1),
                step=0.1,
                help="Price offset for limit orders"
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Save button
        if st.button("💾 Save Trading Configuration", use_container_width=True):
            # Prepare configuration data
            config_data = {
                "trading": {
                    "default_position_size": default_position_size,
                    "max_positions": max_positions,
                    "min_confidence": min_confidence,
                    "signal_cooldown": signal_cooldown,
                    "paper_trading_enabled": paper_trading,
                    "auto_trading_enabled": auto_trading,
                    "use_market_orders": use_market_orders,
                    "use_limit_orders": use_limit_orders,
                    "limit_offset": limit_offset if use_limit_orders else 0
                },
                "risk_management": {
                    "default_stop_loss": stop_loss,
                    "default_take_profit": take_profit,
                    "max_drawdown_limit": max_drawdown,
                    "max_risk_per_trade": risk_per_trade
                },
                "signal_weights": {
                    "technical": technical_weight / 100,
                    "sentiment": sentiment_weight / 100,
                    "onchain": onchain_weight / 100,
                    "macro": macro_weight / 100
                }
            }
            
            result = api_client.post("/config/update", config_data)
            if result and result.get("status") == "success":
                st.success("✅ Configuration saved successfully!")
            else:
                st.error("❌ Failed to save configuration")

# API Settings Tab
with tab2:
    st.subheader("API Configuration")
    
    # Load API configuration
    try:
        api_config = api_client.get("/config/api") or {}
    except:
        api_config = {}
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="settings-section">', unsafe_allow_html=True)
        st.markdown("### Exchange APIs")
        
        # Exchange selection
        exchange = st.selectbox(
            "Primary Exchange",
            ["Binance", "Coinbase", "Kraken", "Bybit"],
            index=["Binance", "Coinbase", "Kraken", "Bybit"].index(api_config.get("primary_exchange", "Binance"))
        )
        
        # API credentials (masked)
        api_key = st.text_input(
            "API Key",
            value="*" * 20 if api_config.get("has_api_key") else "",
            type="password",
            help="Your exchange API key"
        )
        
        api_secret = st.text_input(
            "API Secret",
            value="*" * 20 if api_config.get("has_api_secret") else "",
            type="password",
            help="Your exchange API secret"
        )
        
        # Test connection
        if st.button("🔗 Test Connection", use_container_width=True):
            result = api_client.post("/config/test-exchange", {
                "exchange": exchange,
                "api_key": api_key if api_key != "*" * 20 else None,
                "api_secret": api_secret if api_secret != "*" * 20 else None
            })
            
            if result and result.get("status") == "connected":
                st.success("✅ Connection successful!")
                st.metric("Account Balance", f"${result.get('balance', 0):,.2f}")
            else:
                st.error("❌ Connection failed. Please check your credentials.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="settings-section">', unsafe_allow_html=True)
        st.markdown("### Data Providers")
        
        # Data source configuration
        st.markdown("#### Price Data")
        price_source = st.selectbox(
            "Primary Price Source",
            ["CoinGecko", "CryptoCompare", "CoinMarketCap"],
            index=["CoinGecko", "CryptoCompare", "CoinMarketCap"].index(api_config.get("price_source", "CoinGecko"))
        )
        
        st.markdown("#### On-Chain Data")
        onchain_source = st.selectbox(
            "On-Chain Data Provider",
            ["Glassnode", "IntoTheBlock", "CryptoQuant"],
            index=["Glassnode", "IntoTheBlock", "CryptoQuant"].index(api_config.get("onchain_source", "Glassnode"))
        )
        
        onchain_api_key = st.text_input(
            "On-Chain API Key",
            value="*" * 20 if api_config.get("has_onchain_key") else "",
            type="password",
            help="API key for on-chain data provider"
        )
        
        st.markdown("#### News & Sentiment")
        news_api_key = st.text_input(
            "News API Key",
            value="*" * 20 if api_config.get("has_news_key") else "",
            type="password",
            help="API key for news data"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Rate limits
    st.markdown('<div class="settings-section">', unsafe_allow_html=True)
    st.markdown("### Rate Limits & Caching")
    
    col_r1, col_r2, col_r3 = st.columns(3)
    
    with col_r1:
        rate_limit = st.number_input(
            "API Rate Limit (req/min)",
            min_value=1,
            max_value=1000,
            value=api_config.get("rate_limit", 60),
            help="Maximum API requests per minute"
        )
    
    with col_r2:
        cache_ttl = st.number_input(
            "Cache TTL (seconds)",
            min_value=0,
            max_value=3600,
            value=api_config.get("cache_ttl", 300),
            help="How long to cache API responses"
        )
    
    with col_r3:
        retry_attempts = st.number_input(
            "Retry Attempts",
            min_value=0,
            max_value=10,
            value=api_config.get("retry_attempts", 3),
            help="Number of retry attempts on failure"
        )
    
    st.markdown('</div>', unsafe_allow_html=True)

# Data Quality Tab
with tab3:
    st.subheader("Data Quality Monitoring")
    
    # Fetch data quality metrics
    try:
        data_quality = api_client.get("/analytics/data-quality") or {}
    except:
        data_quality = {}
    
    if data_quality:
        # Summary metrics
        summary = data_quality.get("summary", {})
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="data-quality-metric">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{summary.get("total_datapoints", 0):,}</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Total Datapoints</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            completeness = summary.get("overall_completeness", 0)
            color = "#0ecb81" if completeness > 90 else "#ffd700" if completeness > 70 else "#f6465d"
            st.markdown('<div class="data-quality-metric">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value" style="color: {color}">{completeness:.1f}%</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Data Completeness</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="data-quality-metric">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{summary.get("total_missing_dates", 0)}</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Missing Dates</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            sources = len(data_quality.get("by_source", {}))
            st.markdown('<div class="data-quality-metric">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{sources}</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Active Sources</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Data by type
        st.markdown("### Data Coverage by Type")
        
        by_type = data_quality.get("by_type", {})
        if by_type:
            type_df = pd.DataFrame([
                {
                    "Type": data_type.title(),
                    "Datapoints": info.get("total_datapoints", 0),
                    "Completeness": f"{info.get('completeness', 0):.1f}%",
                    "Sources": len(info.get("sources", {}))
                }
                for data_type, info in by_type.items()
            ])
            
            st.dataframe(type_df, use_container_width=True, hide_index=True)
        
        # Coverage heatmap
        coverage = data_quality.get("coverage", {})
        if coverage:
            st.markdown("### Data Coverage by Time Period")
            
            # Create heatmap data
            periods = ["last_24h", "last_7d", "last_30d", "last_90d"]
            data_types = ["price", "volume", "onchain", "sentiment"]
            
            heatmap_data = []
            for period in periods:
                row = []
                for dtype in data_types:
                    value = coverage.get(period, {}).get(dtype, 0)
                    row.append(value)
                heatmap_data.append(row)
            
            fig = go.Figure(data=go.Heatmap(
                z=heatmap_data,
                x=[t.title() for t in data_types],
                y=["24 Hours", "7 Days", "30 Days", "90 Days"],
                colorscale="RdYlGn",
                zmid=50,
                text=[[f"{val:.0f}%" for val in row] for row in heatmap_data],
                texttemplate="%{text}",
                textfont={"size": 12}
            ))
            
            fig.update_layout(
                title="Data Coverage Heatmap (%)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Gap analysis
        gaps = data_quality.get("gaps", [])
        if gaps:
            st.markdown("### Data Gaps")
            st.warning(f"Found {len(gaps)} data gaps in the historical record")
            
            with st.expander("View Gap Details"):
                for gap in gaps[:10]:  # Show first 10 gaps
                    st.write(f"• {gap['start']} to {gap['end']} ({gap['days']} days)")
        
        # Actions
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Refresh Data Quality", use_container_width=True):
                st.rerun()
        
        with col2:
            if st.button("🔧 Fill Data Gaps", use_container_width=True):
                result = api_client.post("/data/fill-gaps", {})
                if result and result.get("status") == "success":
                    st.success(f"✅ Filled {result.get('gaps_filled', 0)} gaps")
                else:
                    st.error("❌ Failed to fill gaps")
        
        with col3:
            if st.button("🗑️ Clear Cache", use_container_width=True):
                result = api_client.post("/cache/clear", {})
                if result and result.get("status") == "success":
                    st.success("✅ Cache cleared")
                else:
                    st.error("❌ Failed to clear cache")
    else:
        st.info("No data quality metrics available. The system may still be initializing.")

# Notifications Tab
with tab4:
    st.subheader("Notification Settings")
    
    # Load notification config
    try:
        notif_config = api_client.get("/config/notifications") or {}
    except:
        notif_config = {}
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="settings-section">', unsafe_allow_html=True)
        st.markdown("### Notification Channels")
        
        # Email notifications
        email_enabled = st.checkbox(
            "Email Notifications",
            value=notif_config.get("email_enabled", False)
        )
        
        if email_enabled:
            email_address = st.text_input(
                "Email Address",
                value=notif_config.get("email_address", ""),
                help="Your email for notifications"
            )
        
        # Discord notifications
        discord_enabled = st.checkbox(
            "Discord Notifications",
            value=notif_config.get("discord_enabled", False)
        )
        
        if discord_enabled:
            discord_webhook = st.text_input(
                "Discord Webhook URL",
                value="*" * 50 if notif_config.get("has_discord_webhook") else "",
                type="password",
                help="Your Discord webhook URL"
            )
        
        # Telegram notifications
        telegram_enabled = st.checkbox(
            "Telegram Notifications",
            value=notif_config.get("telegram_enabled", False)
        )
        
        if telegram_enabled:
            telegram_token = st.text_input(
                "Telegram Bot Token",
                value="*" * 30 if notif_config.get("has_telegram_token") else "",
                type="password",
                help="Your Telegram bot token"
            )
            
            telegram_chat_id = st.text_input(
                "Telegram Chat ID",
                value=notif_config.get("telegram_chat_id", ""),
                help="Your Telegram chat ID"
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="settings-section">', unsafe_allow_html=True)
        st.markdown("### Notification Events")
        
        # Signal notifications
        st.markdown("#### Trading Signals")
        notify_buy_signals = st.checkbox(
            "Buy Signals",
            value=notif_config.get("notify_buy_signals", True)
        )
        
        notify_sell_signals = st.checkbox(
            "Sell Signals",
            value=notif_config.get("notify_sell_signals", True)
        )
        
        signal_min_confidence = st.slider(
            "Minimum Confidence for Alerts",
            min_value=0.0,
            max_value=1.0,
            value=notif_config.get("signal_min_confidence", 0.7),
            step=0.05
        )
        
        # Trade notifications
        st.markdown("#### Trade Execution")
        notify_trade_executed = st.checkbox(
            "Trade Executed",
            value=notif_config.get("notify_trade_executed", True)
        )
        
        notify_stop_loss = st.checkbox(
            "Stop Loss Hit",
            value=notif_config.get("notify_stop_loss", True)
        )
        
        notify_take_profit = st.checkbox(
            "Take Profit Hit",
            value=notif_config.get("notify_take_profit", True)
        )
        
        # System notifications
        st.markdown("#### System Events")
        notify_errors = st.checkbox(
            "System Errors",
            value=notif_config.get("notify_errors", True)
        )
        
        notify_data_issues = st.checkbox(
            "Data Quality Issues",
            value=notif_config.get("notify_data_issues", True)
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Test notification
    if st.button("📧 Send Test Notification", use_container_width=True):
        result = api_client.post("/notifications/test", {
            "channels": {
                "email": email_enabled,
                "discord": discord_enabled,
                "telegram": telegram_enabled
            }
        })
        
        if result and result.get("status") == "success":
            st.success("✅ Test notification sent!")
        else:
            st.error("❌ Failed to send test notification")

# System Health Tab
with tab5:
    st.subheader("System Health & Monitoring")
    
    # Fetch system health
    try:
        health = api_client.get("/health/detailed") or {}
    except:
        health = {}
    
    if health:
        # Overall status
        overall_status = health.get("status", "unknown")
        status_color = "#0ecb81" if overall_status == "healthy" else "#ffd700" if overall_status == "degraded" else "#f6465d"
        
        st.markdown(f"""
        <div style="text-align: center; padding: 2rem; background: {status_color}20; border-radius: 10px; margin-bottom: 2rem;">
            <h2 style="color: {status_color}; margin: 0;">System Status: {overall_status.upper()}</h2>
            <p style="margin: 0.5rem 0 0 0;">Last checked: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Component status
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="settings-section">', unsafe_allow_html=True)
            st.markdown("### Component Status")
            
            components = health.get("components", {})
            for component, status in components.items():
                status_class = "status-active" if status.get("healthy") else "status-inactive"
                st.markdown(f"""
                <div class="config-item">
                    <span class="status-indicator {status_class}"></span>
                    <strong>{component.replace('_', ' ').title()}</strong>
                    <div style="margin-left: 1.5rem; color: #6b7280; font-size: 0.9rem;">
                        {status.get('message', 'No details available')}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="settings-section">', unsafe_allow_html=True)
            st.markdown("### System Metrics")
            
            metrics = health.get("metrics", {})
            
            # CPU usage
            cpu_usage = metrics.get("cpu_usage", 0)
            st.metric("CPU Usage", f"{cpu_usage:.1f}%")
            st.progress(cpu_usage / 100)
            
            # Memory usage
            memory_usage = metrics.get("memory_usage", 0)
            st.metric("Memory Usage", f"{memory_usage:.1f}%")
            st.progress(memory_usage / 100)
            
            # Disk usage
            disk_usage = metrics.get("disk_usage", 0)
            st.metric("Disk Usage", f"{disk_usage:.1f}%")
            st.progress(disk_usage / 100)
            
            # Uptime
            uptime_seconds = metrics.get("uptime", 0)
            uptime_days = uptime_seconds // 86400
            uptime_hours = (uptime_seconds % 86400) // 3600
            st.metric("Uptime", f"{uptime_days}d {uptime_hours}h")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Recent errors
        errors = health.get("recent_errors", [])
        if errors:
            st.markdown('<div class="settings-section">', unsafe_allow_html=True)
            st.markdown("### Recent Errors")
            
            for error in errors[:5]:  # Show last 5 errors
                error_time = datetime.fromisoformat(error['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
                st.error(f"**{error_time}** - {error['component']}: {error['message']}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # System actions
        st.markdown('<div class="settings-section">', unsafe_allow_html=True)
        st.markdown("### System Actions")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🔄 Restart Services", use_container_width=True):
                if st.session_state.get('confirm_restart'):
                    result = api_client.post("/system/restart", {})
                    if result and result.get("status") == "success":
                        st.success("✅ Services restarting...")
                    else:
                        st.error("❌ Failed to restart services")
                    del st.session_state['confirm_restart']
                else:
                    st.session_state['confirm_restart'] = True
                    st.warning("Click again to confirm restart")
        
        with col2:
            if st.button("🗑️ Clear Logs", use_container_width=True):
                result = api_client.post("/system/clear-logs", {})
                if result and result.get("status") == "success":
                    st.success("✅ Logs cleared")
                else:
                    st.error("❌ Failed to clear logs")
        
        with col3:
            if st.button("💾 Backup Data", use_container_width=True):
                result = api_client.post("/system/backup", {})
                if result and result.get("status") == "success":
                    st.success(f"✅ Backup created: {result.get('filename')}")
                else:
                    st.error("❌ Failed to create backup")
        
        with col4:
            if st.button("📥 Export Logs", use_container_width=True):
                result = api_client.get("/system/logs/export")
                if result and result.get("logs"):
                    st.download_button(
                        label="Download Logs",
                        data=result["logs"],
                        file_name=f"system_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.error("Unable to fetch system health data. The backend may be offline.")

# Auto-refresh for system health
if tab5 and st.checkbox("Auto-refresh health (10s)", value=False):
    time.sleep(10)
    st.rerun()

# Data Upload Tab
with tab6:
    st.subheader("Data Upload")
    
    st.markdown('<div class="settings-section">', unsafe_allow_html=True)
    st.markdown("### Upload Historical Data")
    
    st.info("""
    Upload historical price data to enhance the system's prediction capabilities.
    Supported formats: CSV, Excel (.xlsx, .xls)
    """)
    
    # Initialize the data uploader component
    data_uploader = DataUploader(api_client)
    
    # Render the upload interface
    data_uploader.render()
    
    st.markdown('</div>', unsafe_allow_html=True)