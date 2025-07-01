import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import json
import os
from datetime import datetime, timedelta
import time
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="BTC Trading System - UltraThink Enhanced",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Use backend service name when running in Docker, fallback to localhost
# Fixed port to 8080 to match backend_api.py
API_BASE_URL = os.getenv("API_BASE_URL", "http://backend:8080")

# Enhanced cache function with better error handling
@st.cache_data(ttl=60)
def fetch_api_data(endpoint):
    """Fetch data from API with enhanced error handling"""
    try:
        response = requests.get(f"{API_BASE_URL}{endpoint}", timeout=30)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            # Log missing endpoint but don't show error to user
            logger.warning(f"Endpoint not found: {endpoint}")
            return None
        else:
            logger.error(f"API Error {response.status_code} for {endpoint}")
            return None
    except requests.exceptions.ConnectionError:
        # Try localhost if backend name fails (for local development)
        try:
            alt_url = f"http://localhost:8080{endpoint}"
            response = requests.get(alt_url, timeout=30)
            if response.status_code == 200:
                return response.json()
        except:
            # Silent fail for better UX
            logger.error(f"Cannot connect to API server for {endpoint}")
            return None
    except Exception as e:
        logger.error(f"Error fetching data from {endpoint}: {str(e)}")
        return None

def fetch_or_calculate_analytics(endpoint: str, fallback_data: dict = None):
    """Fetch analytics data with fallback calculation"""
    data = fetch_api_data(endpoint)
    if not data and fallback_data:
        return fallback_data
    return data

def post_api_data(endpoint, data):
    """Post data to API with error handling"""
    try:
        response = requests.post(f"{API_BASE_URL}{endpoint}", json=data, timeout=30)
        if response.status_code in [200, 201]:
            return response.json()
        else:
            logger.error(f"API Error {response.status_code} for POST {endpoint}")
            return None
    except requests.exceptions.ConnectionError:
        # Try localhost if backend name fails
        try:
            alt_url = f"http://localhost:8080{endpoint}"
            response = requests.post(alt_url, json=data, timeout=30)
            if response.status_code in [200, 201]:
                return response.json()
        except:
            st.error(f"Cannot connect to API server")
            return None
    except Exception as e:
        st.error(f"Error posting data: {str(e)}")
        return None

def delete_api_data(endpoint):
    """Delete API data"""
    try:
        response = requests.delete(f"{API_BASE_URL}{endpoint}", timeout=30)
        return response.json() if response.status_code == 200 else None
    except:
        return None

def create_candlestick_chart(btc_data, include_enhanced_indicators=False):
    """Create interactive candlestick chart with indicators"""
    if not btc_data or 'data' not in btc_data:
        return None
    
    df = pd.DataFrame(btc_data['data'])
    if df.empty:
        return None
    
    # Convert timestamps
    df['Date'] = pd.to_datetime(df['Date'])
    
    fig = make_subplots(
        rows=3 if include_enhanced_indicators else 2, 
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2] if include_enhanced_indicators else [0.7, 0.3],
        subplot_titles=("Price & Indicators", "Volume", "Enhanced Signals") if include_enhanced_indicators else ("Price", "Volume")
    )
    
    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df['Date'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='OHLC'
        ),
        row=1, col=1
    )
    
    # Add enhanced indicators if available
    if include_enhanced_indicators:
        # Multiple EMAs
        for period in [9, 21, 50, 200]:
            if f'ema_{period}' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['Date'],
                        y=df[f'ema_{period}'],
                        mode='lines',
                        name=f'EMA {period}',
                        line=dict(width=1)
                    ),
                    row=1, col=1
                )
        
        # Bollinger Bands if available
        if 'bb_upper' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['Date'],
                    y=df['bb_upper'],
                    mode='lines',
                    name='BB Upper',
                    line=dict(color='rgba(250,128,114,0.5)', width=1)
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=df['Date'],
                    y=df['bb_lower'],
                    mode='lines',
                    name='BB Lower',
                    line=dict(color='rgba(250,128,114,0.5)', width=1)
                ),
                row=1, col=1
            )
    else:
        # Basic indicators only
        if 'sma_50' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['Date'],
                    y=df['sma_50'],
                    mode='lines',
                    name='SMA 50',
                    line=dict(color='orange', width=2)
                ),
                row=1, col=1
            )
        
        if 'sma_200' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['Date'],
                    y=df['sma_200'],
                    mode='lines',
                    name='SMA 200',
                    line=dict(color='red', width=2)
                ),
                row=1, col=1
            )
    
    # Volume
    colors = ['red' if row['Close'] < row['Open'] else 'green' for _, row in df.iterrows()]
    fig.add_trace(
        go.Bar(
            x=df['Date'],
            y=df['Volume'],
            name='Volume',
            marker_color=colors
        ),
        row=2, col=1
    )
    
    # Add enhanced signals subplot if requested
    if include_enhanced_indicators and 'rsi' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['Date'],
                y=df['rsi'],
                mode='lines',
                name='RSI',
                line=dict(color='purple', width=1)
            ),
            row=3, col=1
        )
        # Add RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    # Update layout
    fig.update_layout(
        title='BTC/USD Price Chart',
        yaxis_title='Price (USD)',
        xaxis_rangeslider_visible=False,
        template='plotly_dark',
        height=800 if include_enhanced_indicators else 600
    )
    
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    if include_enhanced_indicators:
        fig.update_yaxes(title_text="RSI", row=3, col=1)
    
    return fig

def show_dashboard():
    """Display main dashboard with key metrics"""
    st.header("📊 Trading Dashboard - UltraThink Enhanced")
    
    # Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with st.spinner("Loading dashboard data..."):
        portfolio_metrics = fetch_api_data("/portfolio/metrics")
        # Try enhanced signal first, fallback to regular
        latest_signal = fetch_api_data("/signals/enhanced/latest")
        if not latest_signal:
            latest_signal = fetch_api_data("/signals/latest")
        btc_data = fetch_api_data("/btc/latest")
        backtest_results = fetch_api_data("/backtest/results/latest")
        system_status = fetch_api_data("/system/status")
    
    # Display metrics
    with col1:
        current_price = btc_data['latest_price'] if btc_data else 0
        st.metric("BTC Price", f"${current_price:,.2f}")
    
    with col2:
        signal_value = latest_signal.get('signal', 'N/A') if latest_signal else 'N/A'
        signal_confidence = latest_signal.get('confidence', 0) if latest_signal else 0
        delta_text = f"{signal_confidence:.1%} conf" if signal_confidence > 0 else None
        st.metric("Signal", signal_value.upper(), delta=delta_text)
    
    with col3:
        if portfolio_metrics:
            total_pnl = portfolio_metrics.get('total_pnl', 0)
            pnl_pct = (total_pnl / portfolio_metrics.get('total_invested', 1)) * 100 if portfolio_metrics.get('total_invested', 0) > 0 else 0
            st.metric("Total P&L", f"${total_pnl:,.2f}", f"{pnl_pct:+.2f}%")
        else:
            st.metric("Total P&L", "$0.00", "0.00%")
    
    with col4:
        if backtest_results and 'performance_metrics' in backtest_results:
            sortino = backtest_results['performance_metrics'].get('sortino_ratio', 0)
            st.metric("Sortino Ratio", f"{sortino:.3f}")
        else:
            st.metric("Sortino Ratio", "N/A")
    
    with col5:
        if system_status and 'enhanced_features' in system_status:
            active_features = sum(1 for v in system_status['enhanced_features'].values() if v)
            total_features = len(system_status['enhanced_features'])
            st.metric("Enhanced Features", f"{active_features}/{total_features}")
        else:
            st.metric("Enhanced Features", "Loading...")
    
    # Main chart
    if btc_data:
        include_enhanced = system_status and system_status.get('enhanced_features', {}).get('50_plus_signals', False)
        chart = create_candlestick_chart(btc_data, include_enhanced_indicators=include_enhanced)
        if chart:
            st.plotly_chart(chart, use_container_width=True)
    
    # Enhanced signal analysis
    if latest_signal and 'analysis' in latest_signal:
        with st.expander("📈 Enhanced Signal Analysis", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Signal Components")
                analysis = latest_signal.get('analysis', {})
                
                # Consensus ratio gauge
                consensus = analysis.get('consensus_ratio', 0)
                fig_consensus = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = consensus * 100,
                    title = {'text': "Signal Consensus %"},
                    gauge = {'axis': {'range': [None, 100]},
                            'bar': {'color': "darkgreen" if consensus > 0.7 else "orange" if consensus > 0.5 else "red"},
                            'steps': [
                                {'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 70], 'color': "gray"},
                                {'range': [70, 100], 'color': "lightgreen"}
                            ]}
                ))
                fig_consensus.update_layout(height=200)
                st.plotly_chart(fig_consensus, use_container_width=True)
                
                # Signal distribution
                if 'signal_distribution' in analysis:
                    st.markdown("**Signal Distribution:**")
                    for signal, count in analysis['signal_distribution'].items():
                        st.write(f"- {signal.upper()}: {count}")
            
            with col2:
                st.markdown("### Price Prediction")
                if 'price_confidence_interval' in analysis:
                    ci_low, ci_high = analysis['price_confidence_interval']
                    predicted = latest_signal.get('predicted_price', 0)
                    
                    # Create confidence interval chart
                    fig_ci = go.Figure()
                    fig_ci.add_trace(go.Scatter(
                        x=['Lower', 'Predicted', 'Upper'],
                        y=[ci_low, predicted, ci_high],
                        mode='lines+markers',
                        name='Price Range',
                        line=dict(color='cyan', width=3),
                        marker=dict(size=10)
                    ))
                    fig_ci.update_layout(
                        title="95% Confidence Interval",
                        yaxis_title="Price ($)",
                        height=200,
                        showlegend=False
                    )
                    st.plotly_chart(fig_ci, use_container_width=True)
                
                # Feature importance
                if 'feature_importance' in analysis and analysis['feature_importance']:
                    st.markdown("**Top Feature Importance:**")
                    for feature, importance in list(analysis['feature_importance'].items())[:5]:
                        st.write(f"- {feature}: {importance:.3f}")

def show_trading():
    """Trading interface with manual and automated controls"""
    st.header("💹 Trading Interface")
    
    # Trading status
    trading_status = fetch_api_data("/trading/status")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if trading_status:
            status_color = "🟢" if trading_status.get('is_active') else "🔴"
            st.markdown(f"### {status_color} Trading Status")
            st.write(f"**Active:** {trading_status.get('is_active', False)}")
            st.write(f"**Mode:** {trading_status.get('mode', 'N/A')}")
        else:
            st.markdown("### 🔴 Trading Status")
            st.write("Unable to fetch trading status")
    
    with col2:
        st.markdown("### 📊 Current Signal")
        latest_signal = fetch_api_data("/signals/enhanced/latest")
        if not latest_signal:
            latest_signal = fetch_api_data("/signals/latest")
        
        if latest_signal:
            signal = latest_signal.get('signal', 'hold')
            confidence = latest_signal.get('confidence', 0)
            
            # Signal indicator with confidence
            if signal == 'buy':
                st.success(f"**BUY** (Confidence: {confidence:.1%})")
            elif signal == 'sell':
                st.error(f"**SELL** (Confidence: {confidence:.1%})")
            else:
                st.info(f"**HOLD** (Confidence: {confidence:.1%})")
            
            if 'predicted_price' in latest_signal:
                st.write(f"Predicted Price: ${latest_signal['predicted_price']:,.2f}")
    
    with col3:
        st.markdown("### 🎯 Actions")
        
        # Toggle trading
        if st.button("Toggle Trading", type="primary", use_container_width=True):
            if trading_status and trading_status.get('is_active'):
                result = post_api_data("/trading/stop", {})
                if result:
                    st.success("Trading stopped")
            else:
                result = post_api_data("/trading/start", {})
                if result:
                    st.success("Trading started")
            st.rerun()
        
        # Manual trade buttons
        col_buy, col_sell = st.columns(2)
        with col_buy:
            if st.button("Manual BUY", type="secondary", use_container_width=True):
                result = post_api_data("/trades/execute", {
                    "signal": "buy",
                    "size": 0.001,
                    "reason": "manual"
                })
                if result:
                    st.success("Buy order executed")
                    st.rerun()
        
        with col_sell:
            if st.button("Manual SELL", type="secondary", use_container_width=True):
                result = post_api_data("/trades/execute", {
                    "signal": "sell",
                    "size": 0.001,
                    "reason": "manual"
                })
                if result:
                    st.success("Sell order executed")
                    st.rerun()
    
    # Recent trades
    st.markdown("### 📋 Recent Trades")
    trades = fetch_api_data("/trades/recent?limit=10")
    
    if trades:
        trades_df = pd.DataFrame(trades)
        if not trades_df.empty:
            trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
            trades_df['value'] = trades_df['price'] * trades_df['size']
            
            # Style the dataframe
            def style_trade_type(val):
                if val == 'buy':
                    return 'background-color: #28a745; color: white'
                elif val == 'sell':
                    return 'background-color: #dc3545; color: white'
                return ''
            
            styled_df = trades_df[['timestamp', 'trade_type', 'price', 'size', 'value', 'reason']].style.applymap(
                style_trade_type, subset=['trade_type']
            )
            
            st.dataframe(styled_df, hide_index=True, use_container_width=True)
    else:
        st.info("No recent trades")

def show_portfolio():
    """Portfolio management and performance tracking"""
    st.header("💼 Portfolio Management")
    
    portfolio_metrics = fetch_api_data("/portfolio/metrics")
    positions = fetch_api_data("/portfolio/positions")
    trades = fetch_api_data("/trades/all")
    
    if not portfolio_metrics:
        st.warning("No portfolio data available. Start trading to see your portfolio!")
        return
    
    # Portfolio Summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Invested", f"${portfolio_metrics.get('total_invested', 0):,.2f}")
    
    with col2:
        current_value = portfolio_metrics.get('total_invested', 0) + portfolio_metrics.get('total_pnl', 0)
        st.metric("Current Value", f"${current_value:,.2f}")
    
    with col3:
        total_pnl = portfolio_metrics.get('total_pnl', 0)
        pnl_pct = (total_pnl / portfolio_metrics.get('total_invested', 1)) * 100 if portfolio_metrics.get('total_invested', 0) > 0 else 0
        st.metric("Total P&L", f"${total_pnl:,.2f}", f"{pnl_pct:+.2f}%")
    
    with col4:
        st.metric("Total BTC", f"{portfolio_metrics.get('total_volume', 0):.6f}")
    
    # Performance Chart
    if trades:
        trades_df = pd.DataFrame(trades)
        if not trades_df.empty:
            trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
            trades_df = trades_df.sort_values('timestamp')
            
            # Calculate cumulative metrics
            trades_df['trade_value'] = trades_df['price'] * trades_df['size']
            trades_df['signed_value'] = trades_df.apply(
                lambda x: -x['trade_value'] if x['trade_type'] == 'buy' else x['trade_value'], 
                axis=1
            )
            trades_df['cumulative_pnl'] = trades_df['signed_value'].cumsum()
            trades_df['cumulative_invested'] = trades_df.apply(
                lambda x: x['trade_value'] if x['trade_type'] == 'buy' else 0, 
                axis=1
            ).cumsum()
            
            # Create performance chart
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=("Cumulative P&L", "Portfolio Value"),
                row_heights=[0.5, 0.5]
            )
            
            # P&L Chart
            fig.add_trace(
                go.Scatter(
                    x=trades_df['timestamp'],
                    y=trades_df['cumulative_pnl'],
                    mode='lines',
                    name='Cumulative P&L',
                    line=dict(color='green' if trades_df['cumulative_pnl'].iloc[-1] > 0 else 'red', width=2)
                ),
                row=1, col=1
            )
            
            # Portfolio Value Chart
            trades_df['portfolio_value'] = trades_df['cumulative_invested'] + trades_df['cumulative_pnl']
            fig.add_trace(
                go.Scatter(
                    x=trades_df['timestamp'],
                    y=trades_df['portfolio_value'],
                    mode='lines',
                    name='Portfolio Value',
                    line=dict(color='blue', width=2)
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=trades_df['timestamp'],
                    y=trades_df['cumulative_invested'],
                    mode='lines',
                    name='Total Invested',
                    line=dict(color='orange', width=1, dash='dash')
                ),
                row=2, col=1
            )
            
            fig.update_layout(
                height=600,
                showlegend=True,
                template='plotly_dark'
            )
            
            fig.update_yaxes(title_text="P&L ($)", row=1, col=1)
            fig.update_yaxes(title_text="Value ($)", row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Positions
    st.subheader("Current Positions")
    if positions:
        positions_df = pd.DataFrame(positions)
        if not positions_df.empty:
            # Add current metrics
            current_price = portfolio_metrics.get('current_btc_price', 0)
            positions_df['current_value'] = positions_df['total_size'] * current_price
            positions_df['purchase_value'] = positions_df['total_size'] * positions_df['avg_buy_price']
            positions_df['unrealized_pnl'] = positions_df['current_value'] - positions_df['purchase_value']
            positions_df['pnl_percent'] = (positions_df['unrealized_pnl'] / positions_df['purchase_value']) * 100
            
            # Display positions
            for _, position in positions_df.iterrows():
                with st.expander(f"Lot {position['lot_id'][:8]}... - {position['total_size']:.6f} BTC"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write(f"**Size:** {position['total_size']:.6f} BTC")
                        st.write(f"**Avg Price:** ${position['avg_buy_price']:,.2f}")
                    with col2:
                        st.write(f"**Current Value:** ${position['current_value']:,.2f}")
                        st.write(f"**Purchase Value:** ${position['purchase_value']:,.2f}")
                    with col3:
                        pnl_color = "green" if position['unrealized_pnl'] > 0 else "red"
                        st.markdown(f"**P&L:** <span style='color:{pnl_color}'>${position['unrealized_pnl']:,.2f} ({position['pnl_percent']:+.2f}%)</span>", unsafe_allow_html=True)
                        if st.button(f"Sell", key=f"sell_{position['lot_id']}"):
                            result = post_api_data("/trades/execute", {
                                "signal": "sell",
                                "size": position['total_size'],
                                "lot_id": position['lot_id'],
                                "reason": "manual_position_close"
                            })
                            if result:
                                st.success(f"Sold {position['total_size']:.6f} BTC")
                                st.rerun()
    else:
        st.info("No open positions")

def show_signals():
    """Display basic trading signals"""
    st.header("📡 Trading Signals")
    
    # Fetch all signals
    btc_data = fetch_api_data("/btc/latest")
    
    if btc_data and 'data' in btc_data:
        df = pd.DataFrame(btc_data['data'])
        
        # Calculate comprehensive signals
        signals_data = []
        
        # Technical Indicators
        if 'rsi' in df.columns:
            latest_rsi = df['rsi'].iloc[-1]
            signals_data.append({
                "Indicator": "RSI",
                "Value": f"{latest_rsi:.2f}",
                "Signal": "Oversold" if latest_rsi < 30 else "Overbought" if latest_rsi > 70 else "Neutral",
                "Interpretation": "Bullish" if latest_rsi < 30 else "Bearish" if latest_rsi > 70 else "Neutral"
            })
        
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            macd_diff = df['macd'].iloc[-1] - df['macd_signal'].iloc[-1]
            signals_data.append({
                "Indicator": "MACD",
                "Value": f"{macd_diff:.2f}",
                "Signal": "Bullish Cross" if macd_diff > 0 else "Bearish Cross",
                "Interpretation": "Bullish" if macd_diff > 0 else "Bearish"
            })
        
        # Moving Average Signals
        if 'sma_50' in df.columns and 'sma_200' in df.columns:
            golden_cross = df['sma_50'].iloc[-1] > df['sma_200'].iloc[-1]
            signals_data.append({
                "Indicator": "Golden/Death Cross",
                "Value": "50 > 200" if golden_cross else "50 < 200",
                "Signal": "Golden Cross" if golden_cross else "Death Cross",
                "Interpretation": "Bullish" if golden_cross else "Bearish"
            })
        
        # Bollinger Bands
        if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
            latest_close = df['Close'].iloc[-1]
            bb_position = "Above Upper" if latest_close > df['bb_upper'].iloc[-1] else "Below Lower" if latest_close < df['bb_lower'].iloc[-1] else "Within Bands"
            signals_data.append({
                "Indicator": "Bollinger Bands",
                "Value": bb_position,
                "Signal": "Overbought" if bb_position == "Above Upper" else "Oversold" if bb_position == "Below Lower" else "Normal",
                "Interpretation": "Bearish" if bb_position == "Above Upper" else "Bullish" if bb_position == "Below Lower" else "Neutral"
            })
        
        # Volume Analysis
        if 'obv' in df.columns:
            obv_trend = "Increasing" if df['obv'].iloc[-1] > df['obv'].iloc[-5] else "Decreasing"
            signals_data.append({
                "Indicator": "On-Balance Volume",
                "Value": obv_trend,
                "Signal": "Accumulation" if obv_trend == "Increasing" else "Distribution",
                "Interpretation": "Bullish" if obv_trend == "Increasing" else "Bearish"
            })
        
        # Display signals table
        if signals_data:
            signals_df = pd.DataFrame(signals_data)
            
            # Style the dataframe
            def style_interpretation(val):
                if val == 'Bullish':
                    return 'background-color: #28a745; color: white'
                elif val == 'Bearish':
                    return 'background-color: #dc3545; color: white'
                return 'background-color: #6c757d; color: white'
            
            styled_df = signals_df.style.applymap(style_interpretation, subset=['Interpretation'])
            st.dataframe(styled_df, hide_index=True, use_container_width=True)
            
            # Summary
            bullish_count = sum(1 for s in signals_data if s['Interpretation'] == 'Bullish')
            bearish_count = sum(1 for s in signals_data if s['Interpretation'] == 'Bearish')
            neutral_count = sum(1 for s in signals_data if s['Interpretation'] == 'Neutral')
            
            st.write("**Signal Summary:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Bullish Signals", bullish_count, f"{(bullish_count/len(signals_data)*100):.0f}%")
            with col2:
                st.metric("Bearish Signals", bearish_count, f"{(bearish_count/len(signals_data)*100):.0f}%")
            with col3:
                st.metric("Neutral Signals", neutral_count, f"{(neutral_count/len(signals_data)*100):.0f}%")
            
            # Aggregate Signal
            if bullish_count > bearish_count + 2:
                st.success("**Overall Market Sentiment: BULLISH** - Multiple indicators suggest upward momentum")
            elif bearish_count > bullish_count + 2:
                st.error("**Overall Market Sentiment: BEARISH** - Multiple indicators suggest downward pressure")
            else:
                st.info("**Overall Market Sentiment: NEUTRAL** - Mixed signals, exercise caution")
    else:
        st.warning("Unable to load BTC data for signal calculations")

def show_advanced_signals():
    """Display advanced AI trading signals with UltraThink features"""
    st.header("🧠 Advanced AI Trading Signals - UltraThink")
    
    # Tabs for different signal views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Enhanced Signal", 
        "Comprehensive Signals", 
        "Signal History", 
        "Feature Importance",
        "50+ Indicators"
    ])
    
    with tab1:
        st.subheader("Enhanced AI Signal with Confidence Analysis")
        
        enhanced_signal = fetch_api_data("/signals/enhanced/latest")
        if enhanced_signal:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Enhanced signal display
                signal = enhanced_signal.get('signal', 'hold')
                confidence = enhanced_signal.get('confidence', 0.5)
                predicted_price = enhanced_signal.get('predicted_price', 0)
                
                # Create a gauge chart for confidence
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = confidence * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Signal Confidence %"},
                    delta = {'reference': 70, 'relative': True},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 80], 'color': "gray"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                fig_gauge.update_layout(height=300)
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                # Signal recommendation
                if signal == 'buy':
                    st.success(f"🔵 **BUY Signal** - Predicted: ${predicted_price:,.2f}")
                elif signal == 'sell':
                    st.error(f"🔴 **SELL Signal** - Predicted: ${predicted_price:,.2f}")
                else:
                    st.info(f"⚪ **HOLD Signal** - Predicted: ${predicted_price:,.2f}")
            
            with col2:
                st.markdown("### Analysis Details")
                
                if 'analysis' in enhanced_signal:
                    analysis = enhanced_signal['analysis']
                    
                    # Consensus details
                    if 'signal_distribution' in analysis:
                        st.write("**Signal Distribution:**")
                        dist = analysis['signal_distribution']
                        total = sum(dist.values())
                        for sig, count in dist.items():
                            pct = (count/total)*100 if total > 0 else 0
                            st.progress(pct/100, text=f"{sig.upper()}: {count} ({pct:.0f}%)")
                    
                    # Price confidence interval
                    if 'price_confidence_interval' in analysis:
                        ci = analysis['price_confidence_interval']
                        st.info(f"**95% Price Confidence Interval:**\n${ci[0]:,.2f} - ${ci[1]:,.2f}")
                    
                    # Top features
                    if 'feature_importance' in analysis and analysis['feature_importance']:
                        st.write("**Top Contributing Features:**")
                        for feat, imp in list(analysis['feature_importance'].items())[:5]:
                            st.write(f"• {feat}: {imp:.3f}")
        else:
            st.warning("Enhanced signals not available. Please check if the system is running.")
    
    with tab2:
        st.subheader("Comprehensive Signal Analysis")
        
        comprehensive = fetch_api_data("/signals/comprehensive")
        if comprehensive and comprehensive.get('status') != 'no_data':
            
            # Signal categories breakdown
            categories = ['technical', 'on_chain', 'sentiment', 'macro']
            category_scores = {}
            
            for category in categories:
                if f'{category}_signals' in comprehensive:
                    signals = comprehensive[f'{category}_signals']
                    bullish = sum(1 for s in signals.values() if isinstance(s, dict) and s.get('signal') == 'bullish')
                    bearish = sum(1 for s in signals.values() if isinstance(s, dict) and s.get('signal') == 'bearish')
                    total = len(signals)
                    score = (bullish - bearish) / total if total > 0 else 0
                    category_scores[category] = {
                        'score': score,
                        'bullish': bullish,
                        'bearish': bearish,
                        'total': total
                    }
            
            # Display category scores
            cols = st.columns(len(categories))
            for i, (cat, data) in enumerate(category_scores.items()):
                with cols[i]:
                    color = "green" if data['score'] > 0.2 else "red" if data['score'] < -0.2 else "gray"
                    st.metric(
                        cat.replace('_', ' ').title(),
                        f"{data['score']:.2f}",
                        f"↑{data['bullish']} ↓{data['bearish']}"
                    )
            
            # Composite score
            if 'composite_signal' in comprehensive:
                comp = comprehensive['composite_signal']
                st.markdown("---")
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.metric("Composite Signal", comp['signal'].upper(), f"Score: {comp['score']:.3f}")
                with col2:
                    st.metric("Confidence", f"{comp['confidence']:.1%}")
                with col3:
                    st.metric("Active Signals", comp.get('active_signals', 0))
        else:
            st.info("Comprehensive signals are being calculated. Please wait...")
    
    with tab3:
        st.subheader("Signal History (Last 24 Hours)")
        
        # Fetch signal history
        signal_history = fetch_api_data("/signals/history?hours=24")
        
        if signal_history:
            history_df = pd.DataFrame(signal_history)
            history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
            
            # Create signal timeline
            fig = go.Figure()
            
            # Add signal changes
            for signal_type in ['buy', 'sell', 'hold']:
                mask = history_df['signal'] == signal_type
                if mask.any():
                    fig.add_trace(go.Scatter(
                        x=history_df[mask]['timestamp'],
                        y=history_df[mask]['confidence'],
                        mode='markers',
                        name=signal_type.upper(),
                        marker=dict(
                            size=10,
                            color='green' if signal_type == 'buy' else 'red' if signal_type == 'sell' else 'gray'
                        )
                    ))
            
            fig.update_layout(
                title="Signal Changes Over Time",
                xaxis_title="Time",
                yaxis_title="Confidence",
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Signal statistics
            st.markdown("### Signal Statistics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                buy_count = len(history_df[history_df['signal'] == 'buy'])
                st.metric("Buy Signals", buy_count)
            
            with col2:
                sell_count = len(history_df[history_df['signal'] == 'sell'])
                st.metric("Sell Signals", sell_count)
            
            with col3:
                avg_conf = history_df['confidence'].mean()
                st.metric("Avg Confidence", f"{avg_conf:.1%}")
        else:
            st.info("Signal history data is not available yet. This feature requires backend updates.")
    
    with tab4:
        st.subheader("Feature Importance Analysis")
        
        # Get feature importance - using correct endpoint
        feature_importance = fetch_api_data("/analytics/feature-importance")
        
        if feature_importance and 'feature_importance' in feature_importance:
            # Create feature importance chart
            features = list(feature_importance['feature_importance'].keys())[:20]  # Top 20
            importances = [feature_importance['feature_importance'][f] for f in features]
            
            fig = go.Figure(go.Bar(
                x=importances,
                y=features,
                orientation='h',
                marker_color='lightblue'
            ))
            
            fig.update_layout(
                title="Top 20 Most Important Features",
                xaxis_title="Importance Score",
                yaxis_title="Feature",
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature categories breakdown
            st.markdown("### Feature Category Importance")
            
            category_importance = {
                'Technical': 0,
                'On-chain': 0,
                'Sentiment': 0,
                'Macro': 0,
                'Other': 0
            }
            
            for feature, importance in feature_importance['feature_importance'].items():
                if any(ind in feature.lower() for ind in ['rsi', 'macd', 'sma', 'ema', 'bb', 'volume']):
                    category_importance['Technical'] += importance
                elif any(ind in feature.lower() for ind in ['chain', 'network', 'hash', 'difficulty']):
                    category_importance['On-chain'] += importance
                elif any(ind in feature.lower() for ind in ['sentiment', 'fear', 'greed', 'social']):
                    category_importance['Sentiment'] += importance
                elif any(ind in feature.lower() for ind in ['macro', 'dxy', 'gold', 'sp500']):
                    category_importance['Macro'] += importance
                else:
                    category_importance['Other'] += importance
            
            # Normalize
            total = sum(category_importance.values())
            if total > 0:
                category_importance = {k: v/total for k, v in category_importance.items()}
            
            # Display as pie chart
            fig_pie = go.Figure(data=[go.Pie(
                labels=list(category_importance.keys()),
                values=list(category_importance.values()),
                hole=.3
            )])
            
            fig_pie.update_layout(
                title="Feature Importance by Category",
                height=400
            )
            
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("Feature importance analysis not available. Run a backtest to generate this data.")
    
    with tab5:
        st.subheader("All 50+ Technical Indicators")
        
        # Fetch comprehensive indicator data
        indicators = fetch_api_data("/indicators/all")
        
        if indicators and not indicators.get('status') == 'waiting':
            # Group indicators by category
            indicator_groups = {
                'Momentum': ['rsi', 'stochastic', 'williams_r', 'roc', 'tsi', 'uo'],
                'Trend': ['adx', 'aroon', 'cci', 'dpo', 'ichimoku', 'parabolic_sar'],
                'Volatility': ['atr', 'bollinger_bands', 'keltner_channel', 'donchian_channel'],
                'Volume': ['obv', 'cmf', 'mfi', 'vwap', 'volume_profile'],
                'Market Structure': ['pivot_points', 'fibonacci_retracements', 'support_resistance']
            }
            
            for group_name, indicator_list in indicator_groups.items():
                with st.expander(f"{group_name} Indicators", expanded=True):
                    group_data = {ind: val for ind, val in indicators.items() 
                                if any(key in ind.lower() for key in indicator_list)}
                    
                    if group_data:
                        # Create a formatted display
                        cols = st.columns(3)
                        for i, (ind, val) in enumerate(group_data.items()):
                            col_idx = i % 3
                            with cols[col_idx]:
                                if isinstance(val, dict):
                                    st.write(f"**{ind}:**")
                                    for k, v in val.items():
                                        st.write(f"  • {k}: {v}")
                                else:
                                    st.write(f"**{ind}:** {val}")
                    else:
                        st.write("No indicators in this category")
        else:
            st.info("Comprehensive indicator data is being calculated. This feature requires the backend to expose individual indicator values.")

def show_limits():
    """Trading limits and order management"""
    st.header("🎯 Trading Limits & Orders")
    
    # Create new limit order
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Create Limit Order")
        
        with st.form("limit_order_form"):
            limit_type = st.selectbox(
                "Limit Type",
                ["stop_loss", "take_profit", "buy_limit", "sell_limit"],
                help="Type of limit order"
            )
            
            price = st.number_input(
                "Price ($)",
                min_value=0.01,
                value=50000.00,
                step=100.0,
                help="Price at which to trigger the order"
            )
            
            size = st.number_input(
                "Size (BTC)",
                min_value=0.0001,
                value=0.001,
                step=0.0001,
                format="%.4f",
                help="Amount of BTC to trade"
            )
            
            expiry_days = st.number_input(
                "Expiry (days)",
                min_value=1,
                value=7,
                help="Days until order expires"
            )
            
            submitted = st.form_submit_button("Create Order", type="primary")
            
            if submitted:
                order_data = {
                    "limit_type": limit_type,
                    "price": price,
                    "size": size,
                    "expiry_days": expiry_days
                }
                
                result = post_api_data("/limits", order_data)
                if result:
                    st.success(f"✅ {limit_type} order created at ${price:,.2f}")
                    st.rerun()
    
    with col2:
        st.subheader("Order Guidelines")
        st.info("""
        **Stop Loss**: Sell when price drops below threshold
        
        **Take Profit**: Sell when price rises above threshold
        
        **Buy Limit**: Buy when price drops to target
        
        **Sell Limit**: Sell when price rises to target
        
        All orders expire after the specified days if not triggered.
        """)
    
    # Active limits
    st.subheader("Active Limit Orders")
    limits = fetch_api_data("/limits")
    
    if limits:
        limits_df = pd.DataFrame(limits)
        if not limits_df.empty:
            limits_df['created_at'] = pd.to_datetime(limits_df['created_at']).dt.strftime('%Y-%m-%d %H:%M')
            limits_df['expires_at'] = pd.to_datetime(limits_df['expires_at']).dt.strftime('%Y-%m-%d %H:%M')
            
            # Add current price reference
            btc_data = fetch_api_data("/btc/latest")
            current_price = btc_data['latest_price'] if btc_data else 0
            
            limits_df['distance'] = ((limits_df['price'] - current_price) / current_price * 100)
            limits_df['distance_str'] = limits_df['distance'].apply(lambda x: f"{x:+.2f}%")
            
            # Display each limit order
            for _, limit in limits_df.iterrows():
                col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                
                with col1:
                    icon = "🔴" if "loss" in limit['limit_type'] else "🟢" if "profit" in limit['limit_type'] else "🔵"
                    st.write(f"{icon} **{limit['limit_type'].replace('_', ' ').title()}**")
                    st.write(f"Price: ${limit['price']:,.2f} ({limit['distance_str']} from current)")
                
                with col2:
                    st.write(f"**Size:** {limit['size']:.4f} BTC")
                    st.write(f"**Status:** {limit['status']}")
                
                with col3:
                    st.write(f"**Created:** {limit['created_at']}")
                    st.write(f"**Expires:** {limit['expires_at']}")
                
                with col4:
                    if st.button("Cancel", key=f"cancel_{limit['id']}"):
                        result = delete_api_data(f"/limits/{limit['id']}")
                        if result:
                            st.success("Order cancelled")
                            st.rerun()
                
                st.markdown("---")
    else:
        st.info("No active limit orders")

def show_analytics():
    """Advanced analytics with fallback for missing endpoints"""
    st.header("📈 Advanced Analytics")
    
    # Tabs for different analytics views
    tab1, tab2, tab3, tab4 = st.tabs([
        "Performance Metrics", 
        "Risk Analysis", 
        "Market Correlations",
        "Strategy Optimization"
    ])
    
    with tab1:
        st.subheader("Performance Metrics")
        
        # Get comprehensive metrics with fallback
        metrics = fetch_or_calculate_analytics("/analytics/performance", {
            'total_return': 0,
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'max_drawdown': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'calmar_ratio': 0,
            'omega_ratio': 1
        })
        
        backtest_results = fetch_api_data("/backtest/results/latest")
        
        if metrics or backtest_results:
            col1, col2, col3, col4 = st.columns(4)
            
            # Display key metrics
            perf = backtest_results.get('performance_metrics', {}) if backtest_results else metrics
            
            with col1:
                sharpe = perf.get('sharpe_ratio', 0)
                st.metric("Sharpe Ratio", f"{sharpe:.3f}")
                
                sortino = perf.get('sortino_ratio', 0)
                st.metric("Sortino Ratio", f"{sortino:.3f}")
            
            with col2:
                max_dd = perf.get('max_drawdown', 0)
                st.metric("Max Drawdown", f"{max_dd:.2%}")
                
                calmar = perf.get('calmar_ratio', 0)
                st.metric("Calmar Ratio", f"{calmar:.3f}")
            
            with col3:
                win_rate = perf.get('win_rate', 0)
                st.metric("Win Rate", f"{win_rate:.1%}")
                
                profit_factor = perf.get('profit_factor', 0)
                st.metric("Profit Factor", f"{profit_factor:.2f}")
            
            with col4:
                total_return = perf.get('total_return', 0)
                st.metric("Total Return", f"{total_return:.2%}")
                
                # Enhanced metrics
                omega = perf.get('omega_ratio', 0)
                if omega and omega != float('inf'):
                    st.metric("Omega Ratio", f"{omega:.3f}")
            
            # Detailed performance chart
            if 'equity_curve' in perf:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=perf['equity_curve'],
                    mode='lines',
                    name='Equity Curve',
                    line=dict(color='blue', width=2)
                ))
                
                fig.update_layout(
                    title="Strategy Equity Curve",
                    xaxis_title="Time",
                    yaxis_title="Portfolio Value",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Performance metrics require running a backtest or having trade history.")
    
    with tab2:
        st.subheader("Risk Analysis")
        
        risk_metrics = fetch_api_data("/analytics/risk")
        
        if risk_metrics:
            # VaR and CVaR
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Value at Risk (VaR)")
                var_95 = risk_metrics.get('var_95', 0)
                var_99 = risk_metrics.get('var_99', 0)
                
                st.metric("95% VaR", f"{var_95:.2%}")
                st.metric("99% VaR", f"{var_99:.2%}")
            
            with col2:
                st.markdown("### Conditional VaR (CVaR)")
                cvar_95 = risk_metrics.get('cvar_95', 0)
                cvar_99 = risk_metrics.get('cvar_99', 0)
                
                st.metric("95% CVaR", f"{cvar_95:.2%}")
                st.metric("99% CVaR", f"{cvar_99:.2%}")
            
            # Risk distribution
            if 'returns_distribution' in risk_metrics and risk_metrics['returns_distribution']:
                returns = risk_metrics['returns_distribution']
                
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=returns,
                    nbinsx=50,
                    name='Returns Distribution'
                ))
                
                # Add VaR lines
                fig.add_vline(x=var_95, line_dash="dash", line_color="orange", 
                            annotation_text="95% VaR")
                fig.add_vline(x=var_99, line_dash="dash", line_color="red", 
                            annotation_text="99% VaR")
                
                fig.update_layout(
                    title="Returns Distribution with Risk Metrics",
                    xaxis_title="Return",
                    yaxis_title="Frequency",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Risk analysis requires trade history. Start trading to see risk metrics.")
    
    with tab3:
        st.subheader("Market Correlations")
        
        correlations = fetch_api_data("/analytics/correlations")
        
        if correlations and correlations.get('correlation_matrix'):
            # Correlation heatmap
            corr_matrix = correlations.get('correlation_matrix', {})
            
            if corr_matrix:
                # Convert to DataFrame
                corr_df = pd.DataFrame(corr_matrix)
                
                # Create heatmap
                fig = go.Figure(data=go.Heatmap(
                    z=corr_df.values,
                    x=corr_df.columns,
                    y=corr_df.index,
                    colorscale='RdBu',
                    zmid=0,
                    text=corr_df.values,
                    texttemplate='%{text:.2f}',
                    textfont={"size": 10}
                ))
                
                fig.update_layout(
                    title="Feature Correlation Matrix",
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Key correlations
            if 'key_correlations' in correlations:
                st.markdown("### Key Correlations with BTC Price")
                
                key_corr = correlations['key_correlations']
                sorted_corr = sorted(key_corr.items(), key=lambda x: abs(x[1]), reverse=True)
                
                # Display top correlations
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Positive Correlations:**")
                    for feature, corr in sorted_corr:
                        if corr > 0.3:
                            st.write(f"• {feature}: {corr:.3f}")
                
                with col2:
                    st.write("**Negative Correlations:**")
                    for feature, corr in sorted_corr:
                        if corr < -0.3:
                            st.write(f"• {feature}: {corr:.3f}")
        else:
            st.info("Correlation analysis requires enhanced features to be calculated. Run the system for a while to collect data.")
    
    with tab4:
        st.subheader("Strategy Optimization")
        
        optimization = fetch_api_data("/analytics/optimization")
        
        if optimization and 'optimal_weights' in optimization:
            weights = optimization['optimal_weights']
            
            st.markdown("### Optimal Signal Weights")
            
            # Main category weights
            main_weights = {
                'Technical': weights.get('technical_weight', 0),
                'On-chain': weights.get('onchain_weight', 0),
                'Sentiment': weights.get('sentiment_weight', 0),
                'Macro': weights.get('macro_weight', 0)
            }
            
            fig = go.Figure(data=[go.Bar(
                x=list(main_weights.keys()),
                y=list(main_weights.values()),
                marker_color=['blue', 'green', 'orange', 'red']
            )])
            
            fig.update_layout(
                title="Optimal Category Weights",
                yaxis_title="Weight",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Sub-weights
            with st.expander("Detailed Sub-weights"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Technical Sub-weights:**")
                    st.write(f"• Momentum: {weights.get('momentum_weight', 0):.3f}")
                    st.write(f"• Trend: {weights.get('trend_weight', 0):.3f}")
                    st.write(f"• Volatility: {weights.get('volatility_weight', 0):.3f}")
                    st.write(f"• Volume: {weights.get('volume_weight', 0):.3f}")
                
                with col2:
                    st.markdown("**On-chain Sub-weights:**")
                    st.write(f"• Flow: {weights.get('flow_weight', 0):.3f}")
                    st.write(f"• Network: {weights.get('network_weight', 0):.3f}")
                    st.write(f"• Holder: {weights.get('holder_weight', 0):.3f}")
                
                with col3:
                    st.markdown("**Sentiment Sub-weights:**")
                    st.write(f"• Social: {weights.get('social_weight', 0):.3f}")
                    st.write(f"• Derivatives: {weights.get('derivatives_weight', 0):.3f}")
                    st.write(f"• Fear/Greed: {weights.get('fear_greed_weight', 0):.3f}")
            
            # Optimization history
            if 'optimization_history' in optimization and optimization['optimization_history']:
                history = optimization['optimization_history']
                
                st.markdown("### Optimization Progress")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=history,
                    mode='lines',
                    name='Objective Value',
                    line=dict(color='green', width=2)
                ))
                
                fig.update_layout(
                    title="Bayesian Optimization Progress",
                    xaxis_title="Iteration",
                    yaxis_title="Objective Value (Sortino Ratio)",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run optimization through backtesting to see optimal weights and progress.")

def calculate_portfolio_metrics(trades_df):
    """Calculate portfolio performance metrics from trades"""
    if trades_df.empty:
        return {
            'total_return': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'total_trades': 0
        }
    
    # Calculate returns
    trades_df = trades_df.sort_values('timestamp')
    trades_df['trade_value'] = trades_df['price'] * trades_df['size']
    trades_df['signed_value'] = trades_df['trade_value'] * trades_df['trade_type'].map({
        'buy': -1, 
        'sell': 1, 
        'hold': 0
    })
    
    # Cumulative P&L
    trades_df['cumulative_pnl'] = trades_df['signed_value'].cumsum()
    
    # Total return
    total_invested = trades_df[trades_df['trade_type'] == 'buy']['trade_value'].sum()
    total_return = trades_df['cumulative_pnl'].iloc[-1] / total_invested if total_invested > 0 else 0
    
    # Max drawdown
    cummax = trades_df['cumulative_pnl'].cummax()
    drawdown = (trades_df['cumulative_pnl'] - cummax) / cummax.replace(0, 1)
    max_drawdown = drawdown.min()
    
    # Win rate (for completed round trips)
    sells = trades_df[trades_df['trade_type'] == 'sell']
    if len(sells) > 0:
        profitable_trades = sells[sells['signed_value'] > 0]
        win_rate = len(profitable_trades) / len(sells)
    else:
        win_rate = 0
    
    # Simple Sharpe (annualized)
    if len(trades_df) > 1:
        daily_returns = trades_df['cumulative_pnl'].pct_change().dropna()
        if daily_returns.std() > 0:
            sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(365)
        else:
            sharpe_ratio = 0
    else:
        sharpe_ratio = 0
    
    # Profit factor
    wins = trades_df[trades_df['signed_value'] > 0]['signed_value'].sum()
    losses = abs(trades_df[trades_df['signed_value'] < 0]['signed_value'].sum())
    profit_factor = wins / losses if losses > 0 else float('inf')
    
    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'total_trades': len(trades_df)
    }

def show_backtesting():
    """Enhanced backtesting system interface"""
    st.header("🔬 Enhanced Backtesting System")
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Backtest Results", 
        "Run Backtest", 
        "Walk-Forward Analysis",
        "Optimization Results"
    ])
    
    with tab1:
        show_backtest_results()
    
    with tab2:
        show_run_backtest()
    
    with tab3:
        show_walk_forward_analysis()
    
    with tab4:
        show_optimization_results()

def show_backtest_results():
    """Display latest backtest results"""
    st.subheader("📊 Latest Backtest Results")
    
    results = fetch_api_data("/backtest/results/latest")
    
    if not results:
        st.info("No backtest results available. Run a backtest to see results.")
        return
    
    # Summary metrics
    if 'performance_metrics' in results:
        metrics = results['performance_metrics']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Return", f"{metrics.get('total_return', 0):.2%}")
            st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.3f}")
        
        with col2:
            st.metric("Sortino Ratio", f"{metrics.get('sortino_ratio', 0):.3f}")
            st.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0):.2%}")
        
        with col3:
            st.metric("Win Rate", f"{metrics.get('win_rate', 0):.1%}")
            st.metric("Profit Factor", f"{metrics.get('profit_factor', 0):.2f}")
        
        with col4:
            st.metric("Total Trades", metrics.get('num_trades', 0))
            if 'omega_ratio' in metrics and metrics['omega_ratio'] != float('inf'):
                st.metric("Omega Ratio", f"{metrics['omega_ratio']:.3f}")
    
    # Composite score
    if 'composite_score' in results:
        st.markdown("---")
        score = results['composite_score']
        
        # Create gauge for composite score
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Strategy Composite Score"},
            gauge = {
                'axis': {'range': [None, 1]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 0.3], 'color': "lightgray"},
                    {'range': [0.3, 0.7], 'color': "gray"},
                    {'range': [0.7, 1], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.5
                }
            }
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # Recommendations
    if 'recommendations' in results:
        st.markdown("### 💡 Recommendations")
        for rec in results['recommendations']:
            if rec.startswith("✅"):
                st.success(rec)
            elif rec.startswith("⚠️"):
                st.warning(rec)
            else:
                st.info(rec)
    
    # Detailed results
    with st.expander("Detailed Backtest Results"):
        st.json(results)
    
    # Backtest history
    st.markdown("### 📜 Backtest History")
    history = fetch_api_data("/backtest/results/history")
    
    if history:
        history_df = pd.DataFrame(history)
        history_df['timestamp'] = pd.to_datetime(history_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
        
        # Format numeric columns
        for col in ['composite_score', 'sortino_ratio', 'max_drawdown']:
            if col in history_df.columns:
                if col == 'max_drawdown':
                    history_df[col] = history_df[col].apply(lambda x: f"{x:.2%}")
                else:
                    history_df[col] = history_df[col].apply(lambda x: f"{x:.3f}")
        
        st.dataframe(
            history_df[['timestamp', 'composite_score', 'sortino_ratio', 'max_drawdown']], 
            hide_index=True
        )

def show_run_backtest():
    """Interface to run new backtests"""
    st.subheader("🚀 Run Enhanced Backtest")
    
    # Check if backtest is in progress
    status = fetch_api_data("/backtest/status")
    
    if status and status.get('in_progress'):
        st.warning("⏳ Backtest is currently in progress. Please wait...")
        
        # Add a progress bar
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.1)
            progress_bar.progress(i + 1)
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Backtest Parameters")
        
        period = st.selectbox(
            "Data Period",
            ["1mo", "3mo", "6mo", "1y", "2y", "3y"],
            index=3,
            help="Historical data period for backtesting"
        )
        
        use_enhanced = st.checkbox(
            "Use Enhanced Features",
            value=True,
            help="Include 50+ indicators, macro data, and sentiment analysis"
        )
        
        optimize_weights = st.checkbox(
            "Optimize Signal Weights",
            value=True,
            help="Use Bayesian optimization to find optimal feature weights"
        )
        
        include_macro = st.checkbox(
            "Include Macro Indicators",
            value=True,
            help="Include DXY, Gold, S&P500 correlations"
        )
        
        # Advanced settings in expander
        with st.expander("Advanced Settings"):
            settings = fetch_api_data("/config/backtest-settings") or {}
            
            training_days = st.number_input(
                "Training Window (days)",
                min_value=100,
                max_value=2000,
                value=settings.get('training_window_days', 1008),
                help="Number of days for training data"
            )
            
            test_days = st.number_input(
                "Test Window (days)",
                min_value=10,
                max_value=180,
                value=settings.get('test_window_days', 90),
                help="Number of days for test data"
            )
            
            n_trials = st.number_input(
                "Optimization Trials",
                min_value=10,
                max_value=100,
                value=50,
                help="Number of Bayesian optimization trials"
            )
            
            transaction_cost = st.number_input(
                "Transaction Cost (%)",
                min_value=0.0,
                max_value=1.0,
                value=settings.get('transaction_cost', 0.0025) * 100,
                step=0.01,
                help="Trading fees as percentage"
            ) / 100
    
    with col2:
        st.markdown("### Expected Outcomes")
        
        if use_enhanced:
            st.info("""
            **Enhanced Backtest Features:**
            
            ✨ **50+ Technical Indicators**
            - Momentum, trend, volatility, volume
            - Market structure analysis
            
            🌍 **Macro Integration**
            - DXY, Gold, S&P500 correlations
            - Cross-market analysis
            
            🧠 **AI-Powered Optimization**
            - Bayesian optimization
            - Feature importance analysis
            - Confidence intervals
            
            📊 **Advanced Metrics**
            - Omega ratio, CVaR
            - Walk-forward validation
            - Monte Carlo simulation
            
            ⏱️ **Estimated time**: 10-20 minutes
            """)
        else:
            st.info("""
            **Standard Backtest Features:**
            
            📈 **Basic Indicators**
            - Core technical analysis
            - Price action signals
            
            🔄 **Walk-Forward Analysis**
            - Train on historical windows
            - Test on future data
            
            📊 **Standard Metrics**
            - Sharpe, Sortino ratios
            - Maximum drawdown
            - Win rate analysis
            
            ⏱️ **Estimated time**: 5-10 minutes
            """)
    
    # Run backtest button
    if st.button("🎯 Run Enhanced Backtest", type="primary", use_container_width=True):
        with st.spinner("Running enhanced backtest... This may take several minutes."):
            # Prepare request - using correct endpoint
            request_data = {
                "period": period,
                "optimize_weights": optimize_weights,
                "force": True,
                "use_enhanced_weights": use_enhanced,
                "include_macro": include_macro,
                "n_optimization_trials": n_trials if optimize_weights else 0,
                "settings": {
                    "training_window_days": training_days,
                    "test_window_days": test_days,
                    "n_trials": n_trials,
                    "transaction_cost": transaction_cost
                }
            }
            
            # Use correct enhanced endpoint
            result = post_api_data("/backtest/enhanced/run", request_data)
            
            if result:
                if 'task_id' in result or 'status' in result:
                    st.success(f"✅ Backtest started successfully!")
                    st.info("The backtest is running in the background. Check back in a few minutes.")
                elif 'summary' in result:
                    # Immediate results
                    st.success("✅ Backtest completed!")
                    st.metric("Composite Score", f"{result['summary'].get('composite_score', 0):.3f}")
                else:
                    st.error("Unexpected response format")
            else:
                st.error("Failed to start backtest. Please check if the backend service is running.")

def show_walk_forward_analysis():
    """Display walk-forward analysis results"""
    st.subheader("🔄 Walk-Forward Analysis")
    
    wf_results = fetch_api_data("/backtest/walk-forward/results")
    
    if not wf_results or not wf_results.get('window_results'):
        st.info("Walk-forward analysis results not available. This feature requires backend implementation.")
        
        # Show placeholder explanation
        st.markdown("""
        ### What is Walk-Forward Analysis?
        
        Walk-forward analysis tests strategy robustness by:
        - Training on historical data windows
        - Testing on subsequent out-of-sample periods
        - Rolling forward through time
        - Measuring consistency across different market conditions
        
        This helps identify if the strategy is overfitted or genuinely predictive.
        """)
        return
    
    # Overview
    st.markdown("### Analysis Overview")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Windows", wf_results.get('total_windows', 0))
    
    with col2:
        st.metric("Avg Window Return", f"{wf_results.get('avg_return', 0):.2%}")
    
    with col3:
        consistency = wf_results.get('consistency_score', 0)
        st.metric("Consistency Score", f"{consistency:.2%}")
    
    # Window performance chart
    if 'window_results' in wf_results:
        windows_df = pd.DataFrame(wf_results['window_results'])
        
        fig = go.Figure()
        
        # Add returns for each window
        fig.add_trace(go.Bar(
            x=windows_df.index,
            y=windows_df['return'],
            name='Window Return',
            marker_color=['green' if r > 0 else 'red' for r in windows_df['return']]
        ))
        
        # Add average line
        avg_return = windows_df['return'].mean()
        fig.add_hline(y=avg_return, line_dash="dash", line_color="blue", 
                     annotation_text=f"Avg: {avg_return:.2%}")
        
        fig.update_layout(
            title="Walk-Forward Window Performance",
            xaxis_title="Window",
            yaxis_title="Return",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Stability analysis
    with st.expander("Stability Analysis"):
        if 'stability_metrics' in wf_results:
            metrics = wf_results['stability_metrics']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Performance Stability:**")
                st.write(f"• Return Std Dev: {metrics.get('return_std', 0):.3f}")
                st.write(f"• Sharpe Std Dev: {metrics.get('sharpe_std', 0):.3f}")
                st.write(f"• Max Window DD: {metrics.get('max_window_dd', 0):.2%}")
            
            with col2:
                st.markdown("**Signal Stability:**")
                st.write(f"• Signal Consistency: {metrics.get('signal_consistency', 0):.2%}")
                st.write(f"• Weight Stability: {metrics.get('weight_stability', 0):.2%}")
                st.write(f"• Feature Importance Var: {metrics.get('feature_var', 0):.3f}")

def show_optimization_results():
    """Display optimization results"""
    st.subheader("🎯 Optimization Results")
    
    opt_results = fetch_api_data("/backtest/optimization/results")
    
    if not opt_results or not opt_results.get('best_params'):
        st.info("Optimization results not available. Run a backtest with optimization enabled.")
        
        # Show placeholder
        st.markdown("""
        ### What is Bayesian Optimization?
        
        The system uses advanced Bayesian optimization to:
        - Find optimal signal weights
        - Balance risk vs return
        - Minimize overfitting
        - Maximize out-of-sample performance
        
        Run a backtest with "Optimize Signal Weights" enabled to see results.
        """)
        return
    
    # Best parameters
    if 'best_params' in opt_results:
        st.markdown("### Optimal Parameters Found")
        
        params = opt_results['best_params']
        
        # Create radar chart for weights
        categories = ['Technical', 'On-chain', 'Sentiment', 'Macro']
        values = [
            params.get('technical_weight', 0),
            params.get('onchain_weight', 0),
            params.get('sentiment_weight', 0),
            params.get('macro_weight', 0)
        ]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Optimal Weights'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 0.5]
                )),
            showlegend=False,
            title="Optimal Category Weights",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Optimization history
    if 'optimization_history' in opt_results and opt_results['optimization_history']:
        st.markdown("### Optimization Progress")
        
        history = opt_results['optimization_history']
        trials_df = pd.DataFrame(history)
        
        fig = go.Figure()
        
        # Add all trials
        fig.add_trace(go.Scatter(
            x=trials_df.index,
            y=trials_df['value'],
            mode='markers',
            name='Trial Value',
            marker=dict(
                size=8,
                color=trials_df['value'],
                colorscale='Viridis',
                showscale=True
            )
        ))
        
        # Add best value line
        trials_df['best'] = trials_df['value'].cummax()
        fig.add_trace(go.Scatter(
            x=trials_df.index,
            y=trials_df['best'],
            mode='lines',
            name='Best Value',
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title="Bayesian Optimization Progress",
            xaxis_title="Trial",
            yaxis_title="Objective Value",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Parameter importance
    if 'param_importance' in opt_results and opt_results['param_importance']:
        st.markdown("### Parameter Importance")
        
        importance = opt_results['param_importance']
        
        fig = go.Figure(go.Bar(
            x=list(importance.values()),
            y=list(importance.keys()),
            orientation='h',
            marker_color='lightblue'
        ))
        
        fig.update_layout(
            title="Parameter Importance Analysis",
            xaxis_title="Importance",
            yaxis_title="Parameter",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

def show_configuration():
    """System configuration interface"""
    st.header("⚙️ System Configuration")
    
    # Tabs for different settings
    tab1, tab2, tab3, tab4 = st.tabs([
        "Model Settings",
        "Trading Rules", 
        "Signal Weights",
        "System Info"
    ])
    
    with tab1:
        show_model_configuration()
    
    with tab2:
        show_trading_rules()
    
    with tab3:
        show_signal_weights()
    
    with tab4:
        show_system_info()

def show_model_configuration():
    """Model configuration interface"""
    st.subheader("🧠 LSTM Model Configuration")
    
    current_config = fetch_api_data("/config/model") or {
        'input_size': 16,
        'hidden_size': 50,
        'num_layers': 2,
        'dropout': 0.2,
        'sequence_length': 60,
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 50,
        'use_attention': False,
        'ensemble_size': 5
    }
    
    with st.form("model_config_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Model Architecture")
            
            input_size = st.number_input(
                "Input Features",
                min_value=10,
                max_value=100,
                value=current_config.get('input_size', 16),
                help="Number of input features for LSTM"
            )
            
            hidden_size = st.number_input(
                "Hidden Size",
                min_value=32,
                max_value=256,
                value=current_config.get('hidden_size', 50),
                help="LSTM hidden layer size"
            )
            
            num_layers = st.number_input(
                "Number of Layers",
                min_value=1,
                max_value=5,
                value=current_config.get('num_layers', 2),
                help="Number of LSTM layers"
            )
            
            dropout = st.slider(
                "Dropout Rate",
                min_value=0.0,
                max_value=0.5,
                value=current_config.get('dropout', 0.2),
                step=0.05,
                help="Dropout rate for regularization"
            )
        
        with col2:
            st.markdown("### Training Parameters")
            
            sequence_length = st.number_input(
                "Sequence Length",
                min_value=30,
                max_value=120,
                value=current_config.get('sequence_length', 60),
                help="Number of time steps to look back"
            )
            
            learning_rate = st.number_input(
                "Learning Rate",
                min_value=0.0001,
                max_value=0.01,
                value=current_config.get('learning_rate', 0.001),
                format="%.4f",
                help="Model learning rate"
            )
            
            batch_size = st.selectbox(
                "Batch Size",
                [16, 32, 64, 128],
                index=[16, 32, 64, 128].index(current_config.get('batch_size', 32)),
                help="Training batch size"
            )
            
            epochs = st.number_input(
                "Training Epochs",
                min_value=10,
                max_value=200,
                value=current_config.get('epochs', 50),
                help="Number of training epochs"
            )
        
        st.markdown("### Enhanced Features")
        
        use_attention = st.checkbox(
            "Use Attention Mechanism",
            value=current_config.get('use_attention', False),
            help="Enable multi-head attention for better feature focus"
        )
        
        ensemble_size = st.number_input(
            "Ensemble Size",
            min_value=1,
            max_value=10,
            value=current_config.get('ensemble_size', 5),
            help="Number of models for ensemble predictions"
        )
        
        submitted = st.form_submit_button("Update Configuration", type="primary")
        
        if submitted:
            config_data = {
                "input_size": input_size,
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "dropout": dropout,
                "sequence_length": sequence_length,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "epochs": epochs,
                "use_attention": use_attention,
                "ensemble_size": ensemble_size
            }
            
            result = post_api_data("/config/model", config_data)
            if result:
                st.success("✅ Model configuration updated successfully")
                st.info("Note: Changes will take effect after model retraining")
            else:
                st.info("Model configuration endpoint not yet implemented in backend.")

def show_trading_rules():
    """Trading rules configuration"""
    st.subheader("📋 Trading Rules Configuration")
    
    current_rules = fetch_api_data("/config/trading-rules") or {
        'min_trade_size': 0.001,
        'max_position_size': 0.1,
        'position_scaling': 'confidence_based',
        'stop_loss_pct': 5.0,
        'take_profit_pct': 10.0,
        'max_daily_trades': 10,
        'buy_threshold': 0.6,
        'strong_buy_threshold': 0.8,
        'sell_threshold': 0.6,
        'strong_sell_threshold': 0.8
    }
    
    with st.form("trading_rules_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Position Sizing")
            
            min_trade_size = st.number_input(
                "Minimum Trade Size (BTC)",
                min_value=0.0001,
                max_value=0.01,
                value=current_rules.get('min_trade_size', 0.001),
                format="%.4f",
                step=0.0001
            )
            
            max_position_size = st.number_input(
                "Maximum Position Size (BTC)",
                min_value=0.01,
                max_value=1.0,
                value=current_rules.get('max_position_size', 0.1),
                format="%.2f",
                step=0.01
            )
            
            position_scaling = st.selectbox(
                "Position Scaling Method",
                ["fixed", "confidence_based", "volatility_adjusted", "kelly_criterion"],
                index=["fixed", "confidence_based", "volatility_adjusted", "kelly_criterion"].index(
                    current_rules.get('position_scaling', 'confidence_based')
                )
            )
        
        with col2:
            st.markdown("### Risk Management")
            
            stop_loss_pct = st.number_input(
                "Stop Loss (%)",
                min_value=0.0,
                max_value=20.0,
                value=current_rules.get('stop_loss_pct', 5.0),
                step=0.5,
                help="Automatic stop loss percentage"
            )
            
            take_profit_pct = st.number_input(
                "Take Profit (%)",
                min_value=0.0,
                max_value=50.0,
                value=current_rules.get('take_profit_pct', 10.0),
                step=0.5,
                help="Automatic take profit percentage"
            )
            
            max_daily_trades = st.number_input(
                "Max Daily Trades",
                min_value=1,
                max_value=50,
                value=current_rules.get('max_daily_trades', 10),
                help="Maximum trades per day"
            )
        
        st.markdown("### Signal Thresholds")
        
        col3, col4 = st.columns(2)
        
        with col3:
            buy_threshold = st.slider(
                "Buy Signal Threshold",
                min_value=0.5,
                max_value=0.9,
                value=current_rules.get('buy_threshold', 0.6),
                step=0.05,
                help="Minimum confidence for buy signals"
            )
            
            strong_buy_threshold = st.slider(
                "Strong Buy Threshold",
                min_value=0.7,
                max_value=0.95,
                value=current_rules.get('strong_buy_threshold', 0.8),
                step=0.05,
                help="Threshold for increased position size"
            )
        
        with col4:
            sell_threshold = st.slider(
                "Sell Signal Threshold",
                min_value=0.5,
                max_value=0.9,
                value=current_rules.get('sell_threshold', 0.6),
                step=0.05,
                help="Minimum confidence for sell signals"
            )
            
            strong_sell_threshold = st.slider(
                "Strong Sell Threshold",
                min_value=0.7,
                max_value=0.95,
                value=current_rules.get('strong_sell_threshold', 0.8),
                step=0.05,
                help="Threshold for full position exit"
            )
        
        submitted = st.form_submit_button("Update Trading Rules", type="primary")
        
        if submitted:
            rules_data = {
                "min_trade_size": min_trade_size,
                "max_position_size": max_position_size,
                "position_scaling": position_scaling,
                "stop_loss_pct": stop_loss_pct,
                "take_profit_pct": take_profit_pct,
                "max_daily_trades": max_daily_trades,
                "buy_threshold": buy_threshold,
                "strong_buy_threshold": strong_buy_threshold,
                "sell_threshold": sell_threshold,
                "strong_sell_threshold": strong_sell_threshold
            }
            
            result = post_api_data("/config/trading-rules", rules_data)
            if result:
                st.success("✅ Trading rules updated successfully")
            else:
                st.info("Trading rules endpoint not yet implemented in backend.")

def show_signal_weights():
    """Signal weights configuration"""
    st.subheader("⚖️ Signal Weights Configuration")
    
    # Get current weights - note the different endpoint structure
    current_weights_response = fetch_api_data("/config/signal-weights")
    
    # Parse the response format
    if current_weights_response and 'main_categories' in current_weights_response:
        current_weights = {
            'technical_weight': current_weights_response['main_categories'].get('technical', 0.4),
            'onchain_weight': current_weights_response['main_categories'].get('onchain', 0.35),
            'sentiment_weight': current_weights_response['main_categories'].get('sentiment', 0.15),
            'macro_weight': current_weights_response['main_categories'].get('macro', 0.1),
            **current_weights_response.get('technical_sub', {}),
            **current_weights_response.get('onchain_sub', {}),
            **current_weights_response.get('sentiment_sub', {})
        }
    else:
        # Default weights
        current_weights = {
            'technical_weight': 0.4,
            'onchain_weight': 0.35,
            'sentiment_weight': 0.15,
            'macro_weight': 0.1,
            'momentum_weight': 0.3,
            'trend_weight': 0.4,
            'volatility_weight': 0.15,
            'volume_weight': 0.15,
            'flow_weight': 0.4,
            'network_weight': 0.3,
            'holder_weight': 0.3,
            'social_weight': 0.5,
            'derivatives_weight': 0.3,
            'fear_greed_weight': 0.2
        }
    
    # Get optimal weights from latest backtest
    backtest_results = fetch_api_data("/backtest/results/latest")
    optimal_weights = None
    if backtest_results and 'optimal_weights' in backtest_results:
        optimal_weights = backtest_results['optimal_weights']
    
    with st.form("signal_weights_form"):
        st.markdown("### Main Category Weights")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            technical_weight = st.slider(
                "Technical Analysis",
                min_value=0.0,
                max_value=1.0,
                value=current_weights.get('technical_weight', 0.4),
                step=0.05
            )
            
            onchain_weight = st.slider(
                "On-chain Analysis",
                min_value=0.0,
                max_value=1.0,
                value=current_weights.get('onchain_weight', 0.35),
                step=0.05
            )
            
            sentiment_weight = st.slider(
                "Sentiment Analysis",
                min_value=0.0,
                max_value=1.0,
                value=current_weights.get('sentiment_weight', 0.15),
                step=0.05
            )
            
            macro_weight = st.slider(
                "Macro Indicators",
                min_value=0.0,
                max_value=1.0,
                value=current_weights.get('macro_weight', 0.1),
                step=0.05
            )
        
        with col2:
            if optimal_weights:
                st.markdown("**Optimal Values:**")
                st.write(f"Tech: {optimal_weights.get('technical_weight', 0):.2f}")
                st.write(f"Chain: {optimal_weights.get('onchain_weight', 0):.2f}")
                st.write(f"Sent: {optimal_weights.get('sentiment_weight', 0):.2f}")
                st.write(f"Macro: {optimal_weights.get('macro_weight', 0):.2f}")
                
                if st.button("Apply Optimal"):
                    technical_weight = optimal_weights.get('technical_weight', 0.4)
                    onchain_weight = optimal_weights.get('onchain_weight', 0.35)
                    sentiment_weight = optimal_weights.get('sentiment_weight', 0.15)
                    macro_weight = optimal_weights.get('macro_weight', 0.1)
        
        # Normalize weights
        total_weight = technical_weight + onchain_weight + sentiment_weight + macro_weight
        if total_weight > 0:
            st.info(f"Weights will be normalized. Current sum: {total_weight:.2f}")
        
        # Sub-weights in expander
        with st.expander("Advanced Sub-weights"):
            st.markdown("### Technical Sub-weights")
            col1, col2 = st.columns(2)
            
            with col1:
                momentum_weight = st.slider("Momentum", 0.0, 1.0, 
                    current_weights.get('momentum_weight', 0.25), 0.05)
                trend_weight = st.slider("Trend", 0.0, 1.0, 
                    current_weights.get('trend_weight', 0.35), 0.05)
            
            with col2:
                volatility_weight = st.slider("Volatility", 0.0, 1.0, 
                    current_weights.get('volatility_weight', 0.2), 0.05)
                volume_weight = st.slider("Volume", 0.0, 1.0, 
                    current_weights.get('volume_weight', 0.2), 0.05)
            
            st.markdown("### On-chain Sub-weights")
            col3, col4 = st.columns(2)
            
            with col3:
                flow_weight = st.slider("Flow Metrics", 0.0, 1.0, 
                    current_weights.get('flow_weight', 0.4), 0.05)
                network_weight = st.slider("Network Activity", 0.0, 1.0, 
                    current_weights.get('network_weight', 0.3), 0.05)
            
            with col4:
                holder_weight = st.slider("Holder Behavior", 0.0, 1.0, 
                    current_weights.get('holder_weight', 0.3), 0.05)
        
        submitted = st.form_submit_button("Update Weights", type="primary")
        
        if submitted:
            # Send in flat format expected by endpoint
            weights_data = {
                "technical_weight": technical_weight,
                "onchain_weight": onchain_weight,
                "sentiment_weight": sentiment_weight,
                "macro_weight": macro_weight,
                "momentum_weight": momentum_weight,
                "trend_weight": trend_weight,
                "volatility_weight": volatility_weight,
                "volume_weight": volume_weight,
                "flow_weight": flow_weight,
                "network_weight": network_weight,
                "holder_weight": holder_weight,
                "social_weight": current_weights.get('social_weight', 0.5),
                "derivatives_weight": current_weights.get('derivatives_weight', 0.3),
                "fear_greed_weight": current_weights.get('fear_greed_weight', 0.2)
            }
            
            result = post_api_data("/config/signal-weights", weights_data)
            if result:
                st.success("✅ Signal weights updated successfully")

def show_system_info():
    """Display system information"""
    st.subheader("ℹ️ System Information")
    
    # Get system status
    status = fetch_api_data("/system/status")
    
    if status:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### System Components")
            
            components = status.get('components', {})
            for component, state in components.items():
                if state == "healthy" or state == "active":
                    st.success(f"✅ {component}: {state}")
                else:
                    st.error(f"❌ {component}: {state}")
        
        with col2:
            st.markdown("### Enhanced Features")
            
            features = status.get('enhanced_features', {})
            for feature, enabled in features.items():
                feature_name = feature.replace('_', ' ').title()
                if enabled:
                    st.success(f"✅ {feature_name}")
                else:
                    st.warning(f"⭕ {feature_name}")
    
    # Model info
    model_info = fetch_api_data("/model/info")
    
    if model_info:
        st.markdown("### Model Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Model Version", model_info.get('version', 'N/A'))
            st.metric("Last Trained", model_info.get('last_trained', 'N/A'))
        
        with col2:
            st.metric("Training Samples", f"{model_info.get('training_samples', 0):,}")
            st.metric("Features", model_info.get('n_features', 0))
        
        with col3:
            st.metric("Accuracy", f"{model_info.get('accuracy', 0):.2%}")
            st.metric("Validation Loss", f"{model_info.get('val_loss', 0):.4f}")
    else:
        st.info("Model information endpoint not yet implemented.")
    
    # Database stats
    db_stats = fetch_api_data("/database/stats")
    
    if db_stats:
        st.markdown("### Database Statistics")
        
        stats_df = pd.DataFrame([
            {"Table": "Trades", "Count": db_stats.get('trades_count', 0)},
            {"Table": "Signals", "Count": db_stats.get('signals_count', 0)},
            {"Table": "Backtest Results", "Count": db_stats.get('backtest_count', 0)},
            {"Table": "Limit Orders", "Count": db_stats.get('limits_count', 0)}
        ])
        
        st.dataframe(stats_df, hide_index=True)
    else:
        st.info("Database statistics endpoint not yet implemented.")
    
    # System actions
    st.markdown("### System Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Retrain Model", type="secondary"):
            result = post_api_data("/model/retrain", {})
            if result:
                st.success("Model retraining started")
            else:
                st.info("Model retrain endpoint not yet implemented.")
    
    with col2:
        if st.button("🧹 Clear Cache", type="secondary"):
            st.cache_data.clear()
            st.success("Cache cleared")
    
    with col3:
        if st.button("📥 Export Data", type="secondary"):
            result = fetch_api_data("/database/export")
            if result:
                st.download_button(
                    label="Download Export",
                    data=json.dumps(result, indent=2),
                    file_name=f"btc_trading_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            else:
                st.info("Database export endpoint not yet implemented.")

# Main app
def main():
    st.title("🚀 BTC Trading System - UltraThink Enhanced")
    st.markdown("---")
    
    with st.sidebar:
        st.header("Navigation")
        page = st.selectbox(
            "Select Page",
            ["Dashboard", "Trading", "Portfolio", "Signals", "Advanced Signals", 
             "Limits", "Analytics", "Backtesting", "Configuration"]
        )
        
        st.markdown("---")
        
        # API Status
        with st.spinner("Checking API connection..."):
            api_status = fetch_api_data("/health")
        
        if api_status:
            st.success("✅ API Connected")
            if 'components' in api_status:
                with st.expander("System Status"):
                    for component, status in api_status['components'].items():
                        if component != 'enhanced_features':
                            st.write(f"• {component}: {status}")
                    
                    # Show enhanced features status
                    if 'enhanced_features' in api_status:
                        st.write("\n**Enhanced Features:**")
                        for feature, enabled in api_status['enhanced_features'].items():
                            status_icon = "✅" if enabled else "❌"
                            st.write(f"{status_icon} {feature.replace('_', ' ').title()}")
        else:
            st.error("❌ API Disconnected")
            st.info("Please ensure the backend service is running on port 8080")
        
        # Auto-refresh toggle
        auto_refresh = st.checkbox("Auto-refresh (60s)", value=False)
        if auto_refresh:
            st.empty()  # Placeholder for refresh
        
        # Version info
        st.markdown("---")
        st.caption("UltraThink Enhanced v2.0")
        st.caption("50+ Indicators | AI Optimization")
    
    # Page routing
    if page == "Dashboard":
        show_dashboard()
    elif page == "Trading":
        show_trading()
    elif page == "Portfolio":
        show_portfolio()
    elif page == "Signals":
        show_signals()
    elif page == "Advanced Signals":
        show_advanced_signals()
    elif page == "Limits":
        show_limits()
    elif page == "Analytics":
        show_analytics()
    elif page == "Backtesting":
        show_backtesting()
    elif page == "Configuration":
        show_configuration()
    
    # Auto-refresh logic
    if auto_refresh:
        time.sleep(60)
        st.rerun()

if __name__ == "__main__":
    main()