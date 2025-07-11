import streamlit as st
import pandas as pd
import json
import sys
import os
from datetime import datetime, timedelta
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components.api_client import APIClient
from utils.helpers import format_currency, format_percentage

st.set_page_config(page_title="Settings", page_icon="Settings", layout="wide")

# Custom CSS for settings
st.markdown("""
<style>
.settings-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 20px;
    border-radius: 10px;
    color: white;
    margin-bottom: 20px;
}
.config-section {
    background: #f0f2f6;
    padding: 20px;
    border-radius: 10px;
    margin: 20px 0;
}
.config-card {
    background: #ffffff;
    padding: 15px;
    border-radius: 8px;
    margin: 10px 0;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}
.param-group {
    background: #f8f9fa;
    padding: 15px;
    border-radius: 8px;
    margin: 10px 0;
    border-left: 4px solid #667eea;
}
.warning-box {
    background: #fff3cd;
    border-left: 4px solid #ffc107;
    padding: 15px;
    border-radius: 5px;
    margin: 10px 0;
}
.success-box {
    background: #d4edda;
    border-left: 4px solid #28a745;
    padding: 15px;
    border-radius: 5px;
    margin: 10px 0;
}
.model-status {
    padding: 10px 20px;
    border-radius: 20px;
    text-align: center;
    font-weight: bold;
    margin: 10px 0;
}
.status-trained { background: #d4edda; color: #155724; }
.status-not-trained { background: #f8d7da; color: #721c24; }
.status-training { background: #fff3cd; color: #856404; }
</style>
""", unsafe_allow_html=True)

# Initialize API client
@st.cache_resource
def get_api_client():
    return APIClient(os.getenv("API_BASE_URL", "http://backend:8000"))

api_client = get_api_client()

def validate_weights(weights: dict) -> bool:
    """Validate that weights sum to approximately 1.0"""
    total = sum(weights.values())
    return 0.99 <= total <= 1.01

def show_settings():
    """Main settings interface"""
    
    st.title("System Settings")
    
    # Header
    st.markdown("""
    <div class="settings-header">
        <h2>Configuration Management</h2>
        <p>Customize trading parameters, model settings, and system preferences</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get current configuration
    try:
        current_config = api_client.get("/config/current") or {}
        model_status = api_client.get("/ml/status") or {}
        enhanced_status = api_client.get("/enhanced-lstm/status") or {}
    except Exception as e:
        st.error(f"Error loading configuration: {str(e)}")
        current_config = {}
        model_status = {}
        enhanced_status = {}
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "Trading Configuration",
        "Signal Weights",
        "Notifications",
        "Model Settings",
        "🔧 System Preferences",
        "💾 Backup & Restore",
        "Data Quality",
        "📤 Data Upload"
    ])
    
    with tab1:
        st.subheader("Trading Configuration")
        
        # Trading rules
        st.markdown('<div class="config-section">', unsafe_allow_html=True)
        st.markdown("### Trading Rules")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="param-group">', unsafe_allow_html=True)
            st.markdown("**Position Management**")
            
            min_trade_size = st.number_input(
                "Minimum Trade Size (BTC)",
                min_value=0.0001,
                max_value=1.0,
                value=current_config.get('trading_rules', {}).get('min_trade_size', 0.001),
                format="%.4f",
                help="Smallest allowed trade size"
            )
            
            max_position_size = st.slider(
                "Max Position Size (%)",
                min_value=1,
                max_value=50,
                value=int(current_config.get('trading_rules', {}).get('max_position_size', 10) * 100),
                help="Maximum percentage of portfolio in a single position"
            )
            
            max_open_positions = st.number_input(
                "Max Open Positions",
                min_value=1,
                max_value=20,
                value=current_config.get('trading_rules', {}).get('max_open_positions', 5),
                help="Maximum number of concurrent positions"
            )
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="param-group">', unsafe_allow_html=True)
            st.markdown("**Risk Management**")
            
            stop_loss_pct = st.slider(
                "Stop Loss (%)",
                min_value=1,
                max_value=20,
                value=int(current_config.get('trading_rules', {}).get('stop_loss_pct', 5)),
                help="Automatic stop loss percentage"
            )
            
            take_profit_pct = st.slider(
                "Take Profit (%)",
                min_value=5,
                max_value=50,
                value=int(current_config.get('trading_rules', {}).get('take_profit_pct', 10)),
                help="Automatic take profit percentage"
            )
            
            trailing_stop = st.checkbox(
                "Enable Trailing Stop",
                value=current_config.get('trading_rules', {}).get('trailing_stop', False),
                help="Adjust stop loss as price moves favorably"
            )
            
            if trailing_stop:
                trailing_stop_pct = st.slider(
                    "Trailing Stop Distance (%)",
                    min_value=1,
                    max_value=10,
                    value=int(current_config.get('trading_rules', {}).get('trailing_stop_pct', 3))
                )
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Signal thresholds
        st.markdown("### Signal Thresholds")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            buy_threshold = st.slider(
                "Buy Signal Threshold",
                min_value=0.3,
                max_value=0.9,
                value=current_config.get('trading_rules', {}).get('buy_threshold', 0.6),
                step=0.05,
                help="Minimum confidence for buy signals"
            )
        
        with col2:
            sell_threshold = st.slider(
                "Sell Signal Threshold",
                min_value=0.3,
                max_value=0.9,
                value=current_config.get('trading_rules', {}).get('sell_threshold', 0.6),
                step=0.05,
                help="Minimum confidence for sell signals"
            )
        
        with col3:
            signal_timeout = st.number_input(
                "Signal Timeout (minutes)",
                min_value=1,
                max_value=60,
                value=current_config.get('trading_rules', {}).get('signal_timeout', 5),
                help="How long signals remain valid"
            )
        
        # Trading schedule
        st.markdown("### Trading Schedule")
        
        col1, col2 = st.columns(2)
        
        with col1:
            enable_schedule = st.checkbox(
                "Enable Trading Schedule",
                value=current_config.get('trading_rules', {}).get('enable_schedule', False),
                help="Only trade during specified hours"
            )
            
            if enable_schedule:
                start_hour = st.time_input(
                    "Trading Start Time",
                    value=datetime.strptime(
                        current_config.get('trading_rules', {}).get('start_time', '09:00'),
                        '%H:%M'
                    ).time()
                )
                
                end_hour = st.time_input(
                    "Trading End Time",
                    value=datetime.strptime(
                        current_config.get('trading_rules', {}).get('end_time', '17:00'),
                        '%H:%M'
                    ).time()
                )
        
        with col2:
            weekend_trading = st.checkbox(
                "Enable Weekend Trading",
                value=current_config.get('trading_rules', {}).get('weekend_trading', True),
                help="Allow trading on weekends"
            )
            
            exclude_holidays = st.checkbox(
                "Exclude Market Holidays",
                value=current_config.get('trading_rules', {}).get('exclude_holidays', False),
                help="Skip trading on major market holidays"
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Save trading configuration
        if st.button("💾 Save Trading Configuration", type="primary", use_container_width=True):
            trading_config = {
                "min_trade_size": min_trade_size,
                "max_position_size": max_position_size / 100,
                "max_open_positions": max_open_positions,
                "stop_loss_pct": stop_loss_pct,
                "take_profit_pct": take_profit_pct,
                "buy_threshold": buy_threshold,
                "sell_threshold": sell_threshold,
                "signal_timeout": signal_timeout,
                "weekend_trading": weekend_trading,
                "exclude_holidays": exclude_holidays
            }
            
            if trailing_stop:
                trading_config["trailing_stop"] = True
                trading_config["trailing_stop_pct"] = trailing_stop_pct
            
            if enable_schedule:
                trading_config["enable_schedule"] = True
                trading_config["start_time"] = start_hour.strftime('%H:%M')
                trading_config["end_time"] = end_hour.strftime('%H:%M')
            
            result = api_client.post("/config/update", {"trading_rules": trading_config})
            
            if result and result.get('status') == 'success':
                st.success("Trading configuration saved successfully!")
            else:
                st.error("Failed to save configuration")
    
    with tab2:
        st.subheader("Signal Weight Configuration")
        
        st.info("Adjust the importance of different signal categories. Weights must sum to 100%.")
        
        # Current weights
        current_weights = current_config.get('signal_weights', {
            'technical': 0.40,
            'onchain': 0.35,
            'sentiment': 0.15,
            'macro': 0.10
        })
        
        # Weight sliders
        st.markdown('<div class="config-section">', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            technical_weight = st.slider(
                "Technical Indicators Weight",
                min_value=0,
                max_value=100,
                value=int(current_weights.get('technical', 0.40) * 100),
                help="Weight for technical analysis signals (RSI, MACD, etc.)"
            )
            
            onchain_weight = st.slider(
                "On-chain Metrics Weight",
                min_value=0,
                max_value=100,
                value=int(current_weights.get('onchain', 0.35) * 100),
                help="Weight for blockchain metrics (active addresses, etc.)"
            )
        
        with col2:
            sentiment_weight = st.slider(
                "Sentiment Analysis Weight",
                min_value=0,
                max_value=100,
                value=int(current_weights.get('sentiment', 0.15) * 100),
                help="Weight for social sentiment and Fear & Greed"
            )
            
            macro_weight = st.slider(
                "Macro Indicators Weight",
                min_value=0,
                max_value=100,
                value=int(current_weights.get('macro', 0.10) * 100),
                help="Weight for macro economic indicators"
            )
        
        # Total weight validation
        total_weight = technical_weight + onchain_weight + sentiment_weight + macro_weight
        
        if total_weight != 100:
            st.warning(f"Total weight: {total_weight}%. Must equal 100%.")
            
            # Auto-adjust button
            if st.button("🔧 Auto-adjust to 100%"):
                # Proportionally adjust weights
                if total_weight > 0:
                    technical_weight = int((technical_weight / total_weight) * 100)
                    onchain_weight = int((onchain_weight / total_weight) * 100)
                    sentiment_weight = int((sentiment_weight / total_weight) * 100)
                    macro_weight = 100 - technical_weight - onchain_weight - sentiment_weight
                    st.rerun()
        else:
            st.success("Weights are properly balanced!")
        
        # Visual weight distribution
        st.markdown("### Weight Distribution")
        
        import plotly.graph_objects as go
        
        fig = go.Figure(data=[go.Pie(
            labels=['Technical', 'On-chain', 'Sentiment', 'Macro'],
            values=[technical_weight, onchain_weight, sentiment_weight, macro_weight],
            hole=.3,
            marker_colors=['#667eea', '#764ba2', '#f093fb', '#4facfe']
        )])
        
        fig.update_layout(
            height=300,
            showlegend=True,
            margin=dict(l=0, r=0, t=0, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Individual indicator weights
        with st.expander("Advanced: Individual Indicator Weights", expanded=False):
            st.info("Fine-tune weights for specific indicators within each category")
            
            # Technical indicators
            st.markdown("**Technical Indicators**")
            col1, col2, col3 = st.columns(3)
            
            tech_indicators = {
                'rsi': 'RSI',
                'macd': 'MACD',
                'bb': 'Bollinger Bands',
                'sma': 'SMA Cross',
                'ema': 'EMA Cross',
                'stoch': 'Stochastic'
            }
            
            tech_weights = {}
            for i, (key, name) in enumerate(tech_indicators.items()):
                col = [col1, col2, col3][i % 3]
                with col:
                    tech_weights[key] = st.slider(
                        name,
                        min_value=0.0,
                        max_value=1.0,
                        value=current_config.get('indicator_weights', {}).get(key, 0.5),
                        step=0.1
                    )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Save weights
        if st.button("💾 Save Signal Weights", type="primary", use_container_width=True):
            if total_weight == 100:
                weights_config = {
                    "signal_weights": {
                        "technical": technical_weight / 100,
                        "onchain": onchain_weight / 100,
                        "sentiment": sentiment_weight / 100,
                        "macro": macro_weight / 100
                    }
                }
                
                if 'tech_weights' in locals():
                    weights_config["indicator_weights"] = tech_weights
                
                result = api_client.post("/config/update", weights_config)
                
                if result and result.get('status') == 'success':
                    st.success("Signal weights saved successfully!")
                else:
                    st.error("Failed to save weights")
            else:
                st.error("Please ensure weights sum to 100% before saving")
    
    with tab3:
        st.subheader("Notification Settings")
        
        # Discord webhook
        st.markdown('<div class="config-section">', unsafe_allow_html=True)
        st.markdown("### Discord Notifications")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            discord_webhook = st.text_input(
                "Discord Webhook URL",
                value=current_config.get('notifications', {}).get('discord_webhook', ''),
                type="password",
                help="Your Discord webhook URL for notifications"
            )
        
        with col2:
            if st.button("🧪 Test Webhook"):
                if discord_webhook:
                    result = api_client.post("/notifications/test", {"webhook_url": discord_webhook})
                    if result and result.get('status') == 'success':
                        st.success("Test message sent!")
                    else:
                        st.error("Failed to send test message")
                else:
                    st.warning("Please enter a webhook URL first")
        
        # Notification types
        st.markdown("### Notification Types")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="param-group">', unsafe_allow_html=True)
            st.markdown("**Trading Notifications**")
            
            notify_trades = st.checkbox(
                "Trade Executions",
                value=current_config.get('notifications', {}).get('notify_trades', True),
                help="Notify when trades are executed"
            )
            
            notify_signals = st.checkbox(
                "Strong Signals",
                value=current_config.get('notifications', {}).get('notify_signals', True),
                help="Notify for high-confidence signals"
            )
            
            signal_confidence_threshold = st.slider(
                "Signal Confidence Threshold",
                min_value=0.6,
                max_value=0.9,
                value=current_config.get('notifications', {}).get('signal_threshold', 0.7),
                step=0.05,
                disabled=not notify_signals
            )
            
            notify_pnl = st.checkbox(
                "Daily P&L Summary",
                value=current_config.get('notifications', {}).get('notify_pnl', True),
                help="Daily profit/loss summary"
            )
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="param-group">', unsafe_allow_html=True)
            st.markdown("**System Notifications**")
            
            notify_errors = st.checkbox(
                "System Errors",
                value=current_config.get('notifications', {}).get('notify_errors', True),
                help="Notify on system errors"
            )
            
            notify_model = st.checkbox(
                "Model Updates",
                value=current_config.get('notifications', {}).get('notify_model', True),
                help="Notify when models are retrained"
            )
            
            notify_market = st.checkbox(
                "Market Alerts",
                value=current_config.get('notifications', {}).get('notify_market', True),
                help="Significant price movements"
            )
            
            market_move_threshold = st.slider(
                "Market Move Threshold (%)",
                min_value=1,
                max_value=10,
                value=current_config.get('notifications', {}).get('market_threshold', 5),
                disabled=not notify_market
            )
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Notification schedule
        st.markdown("### Notification Schedule")
        
        col1, col2 = st.columns(2)
        
        with col1:
            quiet_hours = st.checkbox(
                "Enable Quiet Hours",
                value=current_config.get('notifications', {}).get('quiet_hours', False),
                help="Disable non-critical notifications during certain hours"
            )
            
            if quiet_hours:
                quiet_start = st.time_input(
                    "Quiet Hours Start",
                    value=datetime.strptime("22:00", '%H:%M').time()
                )
                
                quiet_end = st.time_input(
                    "Quiet Hours End",
                    value=datetime.strptime("08:00", '%H:%M').time()
                )
        
        with col2:
            batch_notifications = st.checkbox(
                "Batch Notifications",
                value=current_config.get('notifications', {}).get('batch_mode', False),
                help="Group multiple notifications together"
            )
            
            if batch_notifications:
                batch_interval = st.selectbox(
                    "Batch Interval",
                    ["5 minutes", "15 minutes", "30 minutes", "1 hour"],
                    index=1
                )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Save notification settings
        if st.button("💾 Save Notification Settings", type="primary", use_container_width=True):
            notifications_config = {
                "notifications": {
                    "discord_webhook": discord_webhook,
                    "notify_trades": notify_trades,
                    "notify_signals": notify_signals,
                    "signal_threshold": signal_confidence_threshold if notify_signals else 0.7,
                    "notify_pnl": notify_pnl,
                    "notify_errors": notify_errors,
                    "notify_model": notify_model,
                    "notify_market": notify_market,
                    "market_threshold": market_move_threshold if notify_market else 5,
                    "quiet_hours": quiet_hours,
                    "batch_mode": batch_notifications
                }
            }
            
            if quiet_hours:
                notifications_config["notifications"]["quiet_start"] = quiet_start.strftime('%H:%M')
                notifications_config["notifications"]["quiet_end"] = quiet_end.strftime('%H:%M')
            
            if batch_notifications:
                notifications_config["notifications"]["batch_interval"] = batch_interval
            
            result = api_client.post("/config/update", notifications_config)
            
            if result and result.get('status') == 'success':
                st.success("Notification settings saved successfully!")
            else:
                st.error("Failed to save settings")
    
    with tab4:
        st.subheader("Model Settings")
        
        # Model status
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Original LSTM Model")
            # Access nested lstm status from /ml/status endpoint
            lstm_status = model_status.get('lstm', {})
            if lstm_status.get('trained'):
                st.markdown('<div class="model-status status-trained">Model Trained</div>', 
                           unsafe_allow_html=True)
                st.write(f"Last trained: {lstm_status.get('last_update', 'Unknown')}")
                st.write(f"Model accuracy: {lstm_status.get('accuracy', 'Unknown')}")
                st.write(f"Model version: {lstm_status.get('version', 'Unknown')}")
            else:
                st.markdown('<div class="model-status status-not-trained">Model Not Trained</div>', 
                           unsafe_allow_html=True)
        
        with col2:
            st.markdown("### Enhanced LSTM Model")
            # Use model_trained field from /enhanced-lstm/status endpoint
            if enhanced_status.get('model_trained'):
                st.markdown('<div class="model-status status-trained">Enhanced Model Trained</div>', 
                           unsafe_allow_html=True)
                st.write(f"Last trained: {enhanced_status.get('last_training_date', 'Unknown')}")
                st.write(f"Features used: {enhanced_status.get('selected_features', 'Unknown')}")
                # Display training metrics if available
                metrics = enhanced_status.get('training_metrics', {})
                if metrics:
                    st.write(f"Test RMSE: {metrics.get('avg_rmse', 'Unknown'):.4f}" if isinstance(metrics.get('avg_rmse'), (int, float)) else f"Test RMSE: {metrics.get('avg_rmse', 'Unknown')}")
            else:
                st.markdown('<div class="model-status status-not-trained">Enhanced Model Not Trained</div>', 
                           unsafe_allow_html=True)
        
        # Model configuration
        st.markdown('<div class="config-section">', unsafe_allow_html=True)
        st.markdown("### Model Configuration")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Architecture**")
            hidden_size = st.number_input(
                "Hidden Layer Size",
                min_value=32,
                max_value=256,
                value=current_config.get('model', {}).get('hidden_size', 100),
                step=32,
                help="Number of units in LSTM hidden layers"
            )
            
            num_layers = st.number_input(
                "Number of Layers",
                min_value=1,
                max_value=5,
                value=current_config.get('model', {}).get('num_layers', 2),
                help="Number of LSTM layers"
            )
            
            dropout = st.slider(
                "Dropout Rate",
                min_value=0.0,
                max_value=0.5,
                value=current_config.get('model', {}).get('dropout', 0.2),
                step=0.05,
                help="Dropout rate for regularization"
            )
        
        with col2:
            st.markdown("**Training**")
            sequence_length = st.number_input(
                "Sequence Length",
                min_value=30,
                max_value=120,
                value=current_config.get('model', {}).get('sequence_length', 60),
                step=10,
                help="Number of time steps to look back"
            )
            
            batch_size = st.selectbox(
                "Batch Size",
                [16, 32, 64, 128, 256],
                index=[16, 32, 64, 128, 256].index(
                    current_config.get('model', {}).get('batch_size', 32)
                ),
                help="Training batch size"
            )
            
            learning_rate = st.select_slider(
                "Learning Rate",
                options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
                value=current_config.get('model', {}).get('learning_rate', 0.001),
                format_func=lambda x: f"{x:.4f}"
            )
        
        with col3:
            st.markdown("**Features**")
            use_attention = st.checkbox(
                "Use Attention Mechanism",
                value=current_config.get('model', {}).get('use_attention', True),
                help="Enable attention for better temporal focus"
            )
            
            use_batch_norm = st.checkbox(
                "Use Batch Normalization",
                value=current_config.get('model', {}).get('use_batch_norm', True),
                help="Normalize inputs for stable training"
            )
            
            ensemble_size = st.number_input(
                "Ensemble Size",
                min_value=1,
                max_value=10,
                value=current_config.get('model', {}).get('ensemble_size', 3),
                help="Number of models in ensemble"
            )
        
        # Training schedule
        st.markdown("### Training Schedule")
        
        col1, col2 = st.columns(2)
        
        # Mapping between interval labels and seconds
        interval_mapping = {
            "Daily": 86400,      # 24 hours
            "Weekly": 604800,    # 7 days
            "Bi-weekly": 1209600,  # 14 days
            "Monthly": 2592000   # 30 days
        }
        
        with col1:
            auto_retrain = st.checkbox(
                "Enable Auto-Retraining",
                value=current_config.get('model', {}).get('auto_retrain', True),
                help="Automatically retrain models periodically"
            )
            
            if auto_retrain:
                # Reverse mapping for display
                reverse_mapping = {v: k for k, v in interval_mapping.items()}
                
                # Get current interval value from config (in seconds)
                current_interval_seconds = current_config.get('model', {}).get('retrain_interval', 604800)
                current_interval_label = reverse_mapping.get(current_interval_seconds, 'Weekly')
                
                retrain_interval = st.selectbox(
                    "Retraining Interval",
                    ["Daily", "Weekly", "Bi-weekly", "Monthly"],
                    index=["Daily", "Weekly", "Bi-weekly", "Monthly"].index(current_interval_label)
                )
                
                retrain_time = st.time_input(
                    "Retraining Time",
                    value=datetime.strptime(
                        current_config.get('model', {}).get('retrain_time', '03:00'),
                        '%H:%M'
                    ).time(),
                    help="Time to start retraining (UTC)"
                )
        
        with col2:
            early_stopping = st.checkbox(
                "Enable Early Stopping",
                value=current_config.get('model', {}).get('early_stopping', True),
                help="Stop training when validation loss stops improving"
            )
            
            if early_stopping:
                patience = st.number_input(
                    "Early Stopping Patience",
                    min_value=5,
                    max_value=50,
                    value=current_config.get('model', {}).get('patience', 10),
                    help="Epochs to wait before stopping"
                )
            
            max_epochs = st.number_input(
                "Maximum Epochs",
                min_value=50,
                max_value=500,
                value=current_config.get('model', {}).get('max_epochs', 100),
                help="Maximum training epochs"
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Training actions
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Train Original LSTM", type="primary", use_container_width=True):
                with st.spinner("Training model... This may take several minutes"):
                    result = api_client.post("/ml/train", {})
                    if result and result.get('status') == 'training_started':
                        st.success("Model training started successfully!")
                        st.info(f"Estimated time: {result.get('estimated_time', 'Unknown')}")
                        st.info("Check back in a few minutes to see the trained model status.")
                    elif result and result.get('status') == 'unsupported':
                        st.error(f"{result.get('message', 'Model type not supported')}")
                    else:
                        st.error("Training failed to start")
        
        with col2:
            if st.button("Train Enhanced LSTM", type="primary", use_container_width=True):
                with st.spinner("Training enhanced model... This may take 5-10 minutes"):
                    result = api_client.post("/enhanced-lstm/train", {})
                    if result:
                        status = result.get('status')
                        if status == 'success':
                            st.success("Enhanced model trained successfully!")
                            metrics = result.get('training_metrics', {})
                            if metrics:
                                st.write(f"Test RMSE: {metrics.get('avg_rmse', 'N/A'):.4f}" if isinstance(metrics.get('avg_rmse'), (int, float)) else f"Test RMSE: {metrics.get('avg_rmse', 'N/A')}")
                            st.rerun()
                        elif status == 'already_trained':
                            st.info("Model was already trained recently")
                            st.write(f"Last trained: {result.get('last_training_date', 'Unknown')}")
                        elif status == 'error':
                            st.error(f"{result.get('message', 'Training failed')}")
                            if result.get('suggestion'):
                                st.warning(f"{result.get('suggestion')}")
                        else:
                            st.error("Training failed with unknown status")
                    else:
                        st.error("No response from training endpoint")
        
        with col3:
            if st.button("💾 Save Model Settings", use_container_width=True):
                model_config = {
                    "model": {
                        "hidden_size": hidden_size,
                        "num_layers": num_layers,
                        "dropout": dropout,
                        "sequence_length": sequence_length,
                        "batch_size": batch_size,
                        "learning_rate": learning_rate,
                        "use_attention": use_attention,
                        "use_batch_norm": use_batch_norm,
                        "ensemble_size": ensemble_size,
                        "auto_retrain": auto_retrain,
                        "early_stopping": early_stopping,
                        "max_epochs": max_epochs
                    }
                }
                
                if auto_retrain:
                    # Convert interval label back to seconds for storage
                    model_config["model"]["retrain_interval"] = interval_mapping.get(retrain_interval, 604800)
                    model_config["model"]["retrain_time"] = retrain_time.strftime('%H:%M')
                
                if early_stopping:
                    model_config["model"]["patience"] = patience
                
                result = api_client.post("/config/update", model_config)
                
                if result and result.get('status') == 'success':
                    st.success("Model settings saved successfully!")
                else:
                    st.error("Failed to save settings")
    
    with tab5:
        st.subheader("System Preferences")
        
        # API configuration
        st.markdown('<div class="config-section">', unsafe_allow_html=True)
        st.markdown("### API Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="param-group">', unsafe_allow_html=True)
            st.markdown("**Rate Limiting**")
            
            rate_limit_enabled = st.checkbox(
                "Enable Rate Limiting",
                value=current_config.get('system', {}).get('rate_limit_enabled', True),
                help="Prevent API abuse"
            )
            
            if rate_limit_enabled:
                rate_limit_requests = st.number_input(
                    "Requests per Minute",
                    min_value=10,
                    max_value=1000,
                    value=current_config.get('system', {}).get('rate_limit_requests', 100)
                )
                
                rate_limit_burst = st.number_input(
                    "Burst Limit",
                    min_value=5,
                    max_value=50,
                    value=current_config.get('system', {}).get('rate_limit_burst', 20),
                    help="Maximum burst requests"
                )
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="param-group">', unsafe_allow_html=True)
            st.markdown("**Caching**")
            
            enable_caching = st.checkbox(
                "Enable Response Caching",
                value=current_config.get('system', {}).get('enable_caching', True),
                help="Cache API responses for performance"
            )
            
            if enable_caching:
                cache_ttl = st.number_input(
                    "Cache TTL (seconds)",
                    min_value=10,
                    max_value=3600,
                    value=current_config.get('system', {}).get('cache_ttl', 60)
                )
                
                cache_size = st.selectbox(
                    "Maximum Cache Size",
                    ["100 MB", "500 MB", "1 GB", "5 GB"],
                    index=["100 MB", "500 MB", "1 GB", "5 GB"].index(
                        current_config.get('system', {}).get('cache_size', '500 MB')
                    )
                )
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Data sources
        st.markdown("### Data Sources")
        
        st.info("Configure external data source preferences and API keys")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="param-group">', unsafe_allow_html=True)
            st.markdown("**Primary Sources**")
            
            primary_price_source = st.selectbox(
                "Primary Price Source",
                ["Binance", "CoinGecko", "Yahoo Finance", "Kraken"],
                index=["Binance", "CoinGecko", "Yahoo Finance", "Kraken"].index(
                    current_config.get('data_sources', {}).get('primary_price', 'Binance')
                )
            )
            
            primary_onchain_source = st.selectbox(
                "Primary On-chain Source",
                ["Blockchain.info", "Blockchair", "Glassnode"],
                index=["Blockchain.info", "Blockchair", "Glassnode"].index(
                    current_config.get('data_sources', {}).get('primary_onchain', 'Blockchain.info')
                )
            )
            
            enable_fallbacks = st.checkbox(
                "Enable Fallback Sources",
                value=current_config.get('data_sources', {}).get('enable_fallbacks', True),
                help="Use alternative sources if primary fails"
            )
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="param-group">', unsafe_allow_html=True)
            st.markdown("**API Keys**")
            
            st.warning("API keys are optional but provide better data access")
            
            news_api_key = st.text_input(
                "News API Key",
                value=current_config.get('api_keys', {}).get('news_api', ''),
                type="password",
                help="For news sentiment analysis"
            )
            
            fred_api_key = st.text_input(
                "FRED API Key",
                value=current_config.get('api_keys', {}).get('fred', ''),
                type="password",
                help="For macro economic data"
            )
            
            glassnode_api_key = st.text_input(
                "Glassnode API Key",
                value=current_config.get('api_keys', {}).get('glassnode', ''),
                type="password",
                help="For advanced on-chain metrics"
            )
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # System behavior
        st.markdown("### System Behavior")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Logging**")
            log_level = st.selectbox(
                "Log Level",
                ["DEBUG", "INFO", "WARNING", "ERROR"],
                index=["DEBUG", "INFO", "WARNING", "ERROR"].index(
                    current_config.get('system', {}).get('log_level', 'INFO')
                )
            )
            
            log_to_file = st.checkbox(
                "Log to File",
                value=current_config.get('system', {}).get('log_to_file', True)
            )
        
        with col2:
            st.markdown("**Performance**")
            parallel_requests = st.checkbox(
                "Enable Parallel Requests",
                value=current_config.get('system', {}).get('parallel_requests', True),
                help="Fetch data from multiple sources in parallel"
            )
            
            request_timeout = st.number_input(
                "Request Timeout (seconds)",
                min_value=5,
                max_value=60,
                value=current_config.get('system', {}).get('request_timeout', 30)
            )
        
        with col3:
            st.markdown("**Maintenance**")
            auto_cleanup = st.checkbox(
                "Auto Cleanup Old Data",
                value=current_config.get('system', {}).get('auto_cleanup', True),
                help="Automatically remove old data"
            )
            
            if auto_cleanup:
                retention_days = st.number_input(
                    "Data Retention (days)",
                    min_value=30,
                    max_value=365,
                    value=current_config.get('system', {}).get('retention_days', 90)
                )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Save system preferences
        if st.button("💾 Save System Preferences", type="primary", use_container_width=True):
            system_config = {
                "system": {
                    "rate_limit_enabled": rate_limit_enabled,
                    "enable_caching": enable_caching,
                    "log_level": log_level,
                    "log_to_file": log_to_file,
                    "parallel_requests": parallel_requests,
                    "request_timeout": request_timeout,
                    "auto_cleanup": auto_cleanup
                },
                "data_sources": {
                    "primary_price": primary_price_source,
                    "primary_onchain": primary_onchain_source,
                    "enable_fallbacks": enable_fallbacks
                }
            }
            
            if rate_limit_enabled:
                system_config["system"]["rate_limit_requests"] = rate_limit_requests
                system_config["system"]["rate_limit_burst"] = rate_limit_burst
            
            if enable_caching:
                system_config["system"]["cache_ttl"] = cache_ttl
                system_config["system"]["cache_size"] = cache_size
            
            if auto_cleanup:
                system_config["system"]["retention_days"] = retention_days
            
            # Only save API keys if they're not empty
            api_keys = {}
            if news_api_key:
                api_keys["news_api"] = news_api_key
            if fred_api_key:
                api_keys["fred"] = fred_api_key
            if glassnode_api_key:
                api_keys["glassnode"] = glassnode_api_key
            
            if api_keys:
                system_config["api_keys"] = api_keys
            
            result = api_client.post("/config/update", system_config)
            
            if result and result.get('status') == 'success':
                st.success("System preferences saved successfully!")
            else:
                st.error("Failed to save preferences")
    
    with tab6:
        st.subheader("Backup & Restore")
        
        # Backup section
        st.markdown('<div class="config-section">', unsafe_allow_html=True)
        st.markdown("### Create Backup")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("Create a backup of your current configuration and data")
            
            backup_items = st.multiselect(
                "Items to Backup",
                ["Configuration", "Trading History", "Model Weights", "Paper Trading Data"],
                default=["Configuration", "Trading History"]
            )
            
            backup_name = st.text_input(
                "Backup Name",
                value=f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                help="Name for the backup file"
            )
        
        with col2:
            st.markdown('<div class="warning-box">', unsafe_allow_html=True)
            st.markdown("**Important:**")
            st.write("• Backups include sensitive data")
            st.write("• Store backups securely")
            st.write("• API keys are encrypted")
            st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("📦 Create Backup", type="primary", use_container_width=True):
            with st.spinner("Creating backup..."):
                backup_data = {
                    "name": backup_name,
                    "items": backup_items,
                    "timestamp": datetime.now().isoformat()
                }
                
                result = api_client.post("/backup/create", backup_data)
                
                if result and result.get('status') == 'success':
                    st.success("Backup created successfully!")
                    
                    # Download backup
                    backup_content = result.get('backup_data', {})
                    backup_json = json.dumps(backup_content, indent=2)
                    
                    st.download_button(
                        label="📥 Download Backup",
                        data=backup_json,
                        file_name=f"{backup_name}.json",
                        mime="application/json"
                    )
                else:
                    st.error("Failed to create backup")
        
        # Restore section
        st.markdown("### Restore from Backup")
        
        uploaded_file = st.file_uploader(
            "Choose a backup file",
            type=['json'],
            help="Select a previously created backup file"
        )
        
        if uploaded_file is not None:
            try:
                backup_data = json.loads(uploaded_file.read())
                
                st.write(f"**Backup Information:**")
                st.write(f"• Name: {backup_data.get('name', 'Unknown')}")
                st.write(f"• Created: {backup_data.get('timestamp', 'Unknown')}")
                st.write(f"• Items: {', '.join(backup_data.get('items', []))}")
                
                restore_items = st.multiselect(
                    "Items to Restore",
                    backup_data.get('items', []),
                    default=backup_data.get('items', [])
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("Restore Backup", type="primary", use_container_width=True):
                        with st.spinner("Restoring backup..."):
                            restore_data = {
                                "backup_data": backup_data,
                                "items": restore_items
                            }
                            
                            result = api_client.post("/backup/restore", restore_data)
                            
                            if result and result.get('status') == 'success':
                                st.success("Backup restored successfully!")
                                st.info("Please restart the application for all changes to take effect")
                            else:
                                st.error("Failed to restore backup")
                
                with col2:
                    st.warning("Restoring will overwrite current settings!")
                    
            except json.JSONDecodeError:
                st.error("Invalid backup file format")
            except Exception as e:
                st.error(f"Error reading backup file: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Export/Import configuration
        st.markdown("### Configuration Export/Import")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Export Current Configuration"):
                config_export = json.dumps(current_config, indent=2)
                st.download_button(
                    label="📥 Download Configuration",
                    data=config_export,
                    file_name=f"config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("🔧 Reset to Defaults"):
                if st.checkbox("I understand this will reset all settings"):
                    result = api_client.post("/config/reset", {})
                    if result and result.get('status') == 'success':
                        st.success("Configuration reset to defaults!")
                        st.rerun()
                    else:
                        st.error("Failed to reset configuration")
    
    with tab7:
        st.subheader("Data Quality Metrics")
        
        # Fetch data quality metrics
        with st.spinner("Loading data quality metrics..."):
            data_quality = api_client.get("/analytics/data-quality")
        
        if data_quality and not data_quality.get('summary', {}).get('error'):
            # Summary section
            st.markdown('<div class="config-section">', unsafe_allow_html=True)
            st.markdown("### Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Datapoints",
                    f"{data_quality['summary']['total_datapoints']:,}",
                    help="Total number of data points across all sources"
                )
            
            with col2:
                st.metric(
                    "Missing Dates",
                    f"{data_quality['summary']['total_missing_dates']:,}",
                    help="Total number of missing dates in the data"
                )
            
            with col3:
                st.metric(
                    "Overall Completeness",
                    f"{data_quality['summary']['overall_completeness']:.1f}%",
                    help="Percentage of expected data that is available"
                )
            
            with col4:
                st.metric(
                    "Last Updated",
                    pd.to_datetime(data_quality['summary']['last_updated']).strftime('%Y-%m-%d %H:%M'),
                    help="When these metrics were calculated"
                )
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Data by Type
            st.markdown("### Data Completeness by Type")
            
            # Create a dataframe for visualization
            type_data = []
            for data_type, metrics in data_quality.get('by_type', {}).items():
                type_data.append({
                    'Type': data_type.capitalize(),
                    'Datapoints': metrics['total_datapoints'],
                    'Missing': metrics['missing_dates'],
                    'Completeness %': metrics['completeness'],
                    'Start Date': pd.to_datetime(metrics['date_range']['start']).strftime('%Y-%m-%d') if metrics['date_range']['start'] else 'N/A',
                    'End Date': pd.to_datetime(metrics['date_range']['end']).strftime('%Y-%m-%d') if metrics['date_range']['end'] else 'N/A'
                })
            
            if type_data:
                type_df = pd.DataFrame(type_data)
                
                # Style the dataframe
                styled_df = type_df.style.background_gradient(
                    subset=['Completeness %'],
                    cmap='RdYlGn',
                    vmin=0,
                    vmax=100
                ).format({
                    'Datapoints': '{:,}',
                    'Missing': '{:,}',
                    'Completeness %': '{:.1f}%'
                })
                
                st.dataframe(styled_df, use_container_width=True)
            
            # Data by Source
            st.markdown("### Data Availability by Source")
            
            # Create expandable sections for each data type
            for data_type, type_metrics in data_quality.get('by_type', {}).items():
                with st.expander(f"{data_type.capitalize()} Data Sources", expanded=False):
                    source_data = []
                    for source, metrics in type_metrics.get('sources', {}).items():
                        source_data.append({
                            'Source': source.capitalize(),
                            'Datapoints': metrics['datapoints'],
                            'Missing': metrics['missing_dates'],
                            'Completeness %': metrics['completeness'],
                            'Last Update': pd.to_datetime(metrics['last_update']).strftime('%Y-%m-%d %H:%M') if metrics['last_update'] else 'N/A'
                        })
                    
                    if source_data:
                        source_df = pd.DataFrame(source_data)
                        st.dataframe(
                            source_df.style.background_gradient(
                                subset=['Completeness %'],
                                cmap='RdYlGn',
                                vmin=0,
                                vmax=100
                            ).format({
                                'Datapoints': '{:,}',
                                'Missing': '{:,}',
                                'Completeness %': '{:.1f}%'
                            }),
                            use_container_width=True
                        )
            
            # Coverage by Time Period
            st.markdown("### Data Coverage by Time Period")
            
            coverage_data = data_quality.get('coverage', {})
            if coverage_data:
                # Create heatmap data
                periods = list(coverage_data.keys())
                data_types = ['price', 'volume', 'onchain', 'sentiment', 'macro']
                
                heatmap_data = []
                for period in periods:
                    row = []
                    for dtype in data_types:
                        row.append(coverage_data[period].get(dtype, 0))
                    heatmap_data.append(row)
                
                # Create heatmap using plotly
                import plotly.graph_objects as go
                
                fig = go.Figure(data=go.Heatmap(
                    z=heatmap_data,
                    x=[t.capitalize() for t in data_types],
                    y=[p.replace('_', ' ').title() for p in periods],
                    colorscale='RdYlGn',
                    text=[[f'{val:.0f}%' for val in row] for row in heatmap_data],
                    texttemplate='%{text}',
                    textfont={"size": 12},
                    colorbar=dict(title="Coverage %")
                ))
                
                fig.update_layout(
                    title="Data Coverage Heatmap",
                    xaxis_title="Data Type",
                    yaxis_title="Time Period",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Data Gaps
            st.markdown("### Data Gaps")
            
            gaps = data_quality.get('gaps', [])
            if gaps:
                st.warning(f"Found {len(gaps)} gaps in the data")
                
                gap_data = []
                for gap in gaps:
                    gap_data.append({
                        'Start': pd.to_datetime(gap['start']).strftime('%Y-%m-%d'),
                        'End': pd.to_datetime(gap['end']).strftime('%Y-%m-%d'),
                        'Days Missing': gap['days'],
                        'Symbol': gap['symbol'],
                        'Granularity': gap['granularity']
                    })
                
                gap_df = pd.DataFrame(gap_data)
                st.dataframe(gap_df, use_container_width=True)
            else:
                st.success("No significant data gaps found!")
            
            # Cache Metrics
            cache_metrics = data_quality.get('cache_metrics', {})
            if cache_metrics:
                st.markdown("### Cache Performance")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Total Cache Entries",
                        f"{cache_metrics.get('total_entries', 0):,}",
                        help="Total number of cached API responses"
                    )
                
                with col2:
                    st.metric(
                        "Active Cache Entries",
                        f"{cache_metrics.get('active_entries', 0):,}",
                        help="Number of non-expired cache entries"
                    )
                
                with col3:
                    cache_efficiency = (cache_metrics.get('active_entries', 0) / cache_metrics.get('total_entries', 1) * 100) if cache_metrics.get('total_entries', 0) > 0 else 0
                    st.metric(
                        "Cache Efficiency",
                        f"{cache_efficiency:.1f}%",
                        help="Percentage of cache entries that are still valid"
                    )
            
            # Actions
            st.markdown("### Actions")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("Refresh Metrics", use_container_width=True):
                    st.rerun()
            
            with col2:
                if st.button("📥 Fill Data Gaps", use_container_width=True):
                    with st.spinner("Attempting to fill data gaps..."):
                        # This would trigger a background job to fetch missing data
                        st.info("Data gap filling initiated. This may take several minutes.")
            
            with col3:
                if st.button("🧹 Clear Cache", use_container_width=True):
                    if st.checkbox("I understand this will clear all cached data"):
                        # This would clear the cache
                        st.warning("Cache clearing functionality not yet implemented")
            
            # File Upload Section
            st.markdown("### Data Upload")
            st.markdown('<div class="config-section">', unsafe_allow_html=True)
            
            st.info("Upload historical data from CSV or Excel files to fill gaps or add new data sources")
            
            # Show example data formats
            with st.expander("📋 Example Data Formats", expanded=False):
                st.markdown("""
                **Price Data Example (CSV):**
                ```
                date,open,high,low,close,volume
                2024-01-01,42150.00,42500.00,41900.00,42300.00,15234.56
                2024-01-02,42300.00,42800.00,42100.00,42650.00,18923.12
                ```
                
                **Volume Data Example (CSV):**
                ```
                timestamp,volume,type
                2024-01-01 00:00:00,12345.67,trading
                2024-01-01 01:00:00,13456.78,trading
                ```
                
                **On-chain Data Example (CSV):**
                ```
                date,active_addresses,transaction_count,hash_rate
                2024-01-01,850000,320000,450.5
                2024-01-02,870000,335000,455.2
                ```
                
                **Sentiment Data Example (CSV):**
                ```
                date,fear_greed_index,social_volume,sentiment_score
                2024-01-01,65,12500,0.72
                2024-01-02,68,13200,0.75
                ```
                """)
            
            # File uploader
            uploaded_file = st.file_uploader(
                "Choose a data file",
                type=['csv', 'xlsx', 'xls'],
                help="Upload CSV or Excel file containing historical data. File should have columns for date/timestamp and relevant data values. Maximum file size: 100MB"
            )
            
            if uploaded_file is not None:
                # Check file size (100MB limit)
                file_size_mb = uploaded_file.size / (1024 * 1024)
                if file_size_mb > 100:
                    st.error(f"File too large ({file_size_mb:.1f}MB). Maximum allowed size is 100MB.")
                    st.info("Consider splitting your data into smaller files or compressing it.")
                else:
                    st.info(f"📁 File: {uploaded_file.name} ({file_size_mb:.2f}MB)")
                
                    # Configuration options
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Data type selection
                        data_type = st.selectbox(
                            "Data Type",
                            ["price", "volume", "onchain", "sentiment", "macro"],
                            help="Select the type of data being uploaded"
                        )
                        
                        # Source input with suggestions
                        source_suggestions = {
                            "price": ["binance", "coingecko", "kraken", "coinbase", "custom"],
                            "volume": ["binance", "coingecko", "custom"],
                            "onchain": ["blockchain.info", "glassnode", "custom"],
                            "sentiment": ["newsapi", "reddit", "twitter", "custom"],
                            "macro": ["fred", "yahoo", "custom"]
                        }
                        
                        source = st.text_input(
                            "Data Source",
                            help=f"Suggested sources: {', '.join(source_suggestions.get(data_type, ['custom']))}"
                        )
                    
                    with col2:
                        # Conflict resolution
                        conflict_strategy = st.radio(
                            "Conflict Resolution",
                            ["skip", "overwrite", "average"],
                            help="How to handle data that already exists:\n• Skip: Keep existing data\n• Overwrite: Replace with new data\n• Average: Average the values"
                        )
                        
                        # Date format hint
                        date_format = st.text_input(
                            "Date Format (optional)",
                            placeholder="YYYY-MM-DD",
                            help="Specify date format if auto-detection fails. Examples: YYYY-MM-DD, DD/MM/YYYY, MM-DD-YYYY"
                        )
                
                # Preview data
                try:
                    # Reset file pointer to beginning
                    uploaded_file.seek(0)
                    
                    # Read file based on type
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)
                    
                    # Show preview
                    st.markdown("#### Data Preview (first 10 rows)")
                    st.dataframe(df.head(10), use_container_width=True)
                    
                    # Column mapping
                    st.markdown("#### Column Mapping")
                    st.info("Map your file columns to the expected data fields")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        date_column = st.selectbox(
                            "Date/Time Column",
                            df.columns.tolist(),
                            help="Select the column containing dates or timestamps"
                        )
                    
                    with col2:
                        value_column = st.selectbox(
                            "Value Column",
                            df.columns.tolist(),
                            help="Select the column containing the main data values"
                        )
                    
                    with col3:
                        # Additional columns based on data type
                        additional_columns = {}
                        if data_type == "price":
                            additional_columns["high"] = st.selectbox(
                                "High Price Column (optional)",
                                ["None"] + df.columns.tolist()
                            )
                            additional_columns["low"] = st.selectbox(
                                "Low Price Column (optional)",
                                ["None"] + df.columns.tolist()
                            )
                            additional_columns["close"] = st.selectbox(
                                "Close Price Column (optional)",
                                ["None"] + df.columns.tolist()
                            )
                        elif data_type == "volume":
                            additional_columns["volume_type"] = st.selectbox(
                                "Volume Type",
                                ["trading", "onchain", "both"]
                            )
                    
                    # Data validation
                    st.markdown("#### Data Validation")
                    
                    # Check date parsing
                    try:
                        if date_format:
                            sample_dates = pd.to_datetime(df[date_column], format=date_format)
                        else:
                            sample_dates = pd.to_datetime(df[date_column])
                        st.success(f"Date column parsed successfully. Date range: {sample_dates.min()} to {sample_dates.max()}")
                    except Exception as e:
                        st.error(f"Error parsing dates: {str(e)}")
                        st.info("Try specifying a date format above")
                    
                    # Check numeric values
                    try:
                        numeric_values = pd.to_numeric(df[value_column], errors='coerce')
                        valid_count = numeric_values.notna().sum()
                        total_count = len(df)
                        
                        if valid_count == total_count:
                            st.success(f"All {total_count} values are numeric")
                        else:
                            st.warning(f"{total_count - valid_count} non-numeric values found (will be skipped)")
                    except Exception as e:
                        st.error(f"Error checking values: {str(e)}")
                    
                    # Upload button
                    if st.button("📤 Upload Data", type="primary", use_container_width=True):
                        if not source:
                            st.error("Please specify a data source")
                        else:
                            with st.spinner("Uploading data..."):
                                # Prepare upload data
                                upload_params = {
                                    "data_type": data_type,
                                    "source": source.lower(),
                                    "conflict_strategy": conflict_strategy,
                                    "date_column": date_column,
                                    "value_column": value_column,
                                    "date_format": date_format if date_format else None,
                                    "additional_columns": {k: v for k, v in additional_columns.items() if v != "None"}
                                }
                                
                                # Reset file pointer for re-reading
                                uploaded_file.seek(0)
                                
                                # Re-read the dataframe to ensure we have fresh data
                                if uploaded_file.name.endswith('.csv'):
                                    df_upload = pd.read_csv(uploaded_file)
                                else:
                                    df_upload = pd.read_excel(uploaded_file)
                                
                                # Convert dataframe to CSV for upload
                                csv_data = df_upload.to_csv(index=False)
                                
                                # Use the upload method
                                result = api_client.upload_data(uploaded_file.name, csv_data, upload_params)
                                
                                if result and result.get('status') == 'success':
                                    st.success(f"Successfully uploaded {result.get('records_imported', 0)} records!")
                                    
                                    # Show import summary
                                    if result.get('summary'):
                                        col1, col2, col3, col4 = st.columns(4)
                                        with col1:
                                            st.metric("Records Imported", result['summary'].get('imported', 0))
                                        with col2:
                                            st.metric("Records Skipped", result['summary'].get('skipped', 0))
                                        with col3:
                                            st.metric("Errors", result['summary'].get('errors', 0))
                                        with col4:
                                            st.metric("Time Taken", f"{result['summary'].get('duration', 0):.1f}s")
                                    
                                    # Refresh metrics
                                    st.info("Refreshing data quality metrics...")
                                    time.sleep(2)
                                    st.rerun()
                                else:
                                    error_msg = result.get('message', 'Unknown error') if result else 'No response from server'
                                    st.error(f"Upload failed: {error_msg}")
                                    
                                    # Show detailed errors if available
                                    if result and result.get('errors'):
                                        with st.expander("Error Details"):
                                            for error in result['errors'][:10]:  # Show max 10 errors
                                                st.write(f"• {error}")
                
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
                    st.info("Please ensure your file is a valid CSV or Excel file")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
        else:
            st.error("Failed to load data quality metrics")
            if data_quality and data_quality.get('summary', {}).get('error'):
                st.error(f"Error: {data_quality['summary']['error']}")
    
    with tab8:
        st.subheader("Data Upload")
        
        # Import the data uploader component
        from components.data_uploader import DataUploader
        
        # Create and render the uploader
        uploader = DataUploader(api_client)
        uploader.render()

# Auto-refresh option
if st.sidebar.checkbox("Auto-refresh (60s)", value=False):
    time.sleep(60)
    st.rerun()

# Show the page
show_settings()