import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime, timedelta
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components.api_client import APIClient
from components.charts import create_signal_chart, create_correlation_heatmap
from utils.helpers import format_currency, format_percentage, aggregate_signals
from utils.constants import CHART_COLORS

st.set_page_config(page_title="Trading Signals", page_icon="Signals", layout="wide")

# Initialize API client
@st.cache_resource
def get_api_client():
    return APIClient(os.getenv("API_BASE_URL", "http://backend:8080"))

api_client = get_api_client()

# Custom CSS for signals page
st.markdown("""
<style>
.signal-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 20px;
    margin: 20px 0;
}
.indicator-card {
    background: rgba(26, 31, 46, 0.8);
    border-radius: 15px;
    padding: 20px;
    border: 1px solid rgba(247, 147, 26, 0.3);
    transition: all 0.3s ease;
}
.indicator-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(247, 147, 26, 0.3);
}
.indicator-value {
    font-size: 2em;
    font-weight: bold;
    margin: 10px 0;
}
.bullish { color: #00ff88; }
.bearish { color: #ff3366; }
.neutral { color: #8b92a8; }
</style>
""", unsafe_allow_html=True)

st.title("AI Trading Signals & Analysis")
st.markdown("Comprehensive analysis with 50+ indicators and LSTM predictions")

# Fetch signal data
try:
    latest_signal = api_client.get("/signals/enhanced/latest") or {}
    comprehensive_signals = api_client.get("/signals/comprehensive") or {}
    signal_history = api_client.get("/signals/history?hours=24") or []
    feature_importance = api_client.get("/analytics/feature-importance") or {}
except Exception as e:
    st.error(f"Error fetching signal data: {str(e)}")
    st.stop()

# Main signal display
if latest_signal:
    col1, col2, col3, col4 = st.columns(4)
    
    signal = latest_signal.get("signal", "hold")
    confidence = latest_signal.get("confidence", 0)
    predicted_price = latest_signal.get("predicted_price", 0)
    composite_confidence = latest_signal.get("composite_confidence", confidence)
    
    with col1:
        signal_color = {
            "buy": "background: linear-gradient(135deg, #00ff88, #00cc66);",
            "sell": "background: linear-gradient(135deg, #ff3366, #cc0033);",
            "hold": "background: linear-gradient(135deg, #8b92a8, #5a6178);"
        }.get(signal, "")
        
        st.markdown(f"""
        <div style="{signal_color} padding: 20px; border-radius: 15px; text-align: center;">
            <h2 style="color: white; margin: 0;">{signal.upper()}</h2>
            <p style="color: white; margin: 5px 0;">AI Signal</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.metric("Composite Confidence", f"{composite_confidence:.1%}", 
                 delta=f"{(composite_confidence - 0.5) * 100:.1f}pp")
    
    with col3:
        current_price = latest_signal.get("current_price", 0)
        price_diff = predicted_price - current_price if current_price > 0 else 0
        price_diff_pct = (price_diff / current_price * 100) if current_price > 0 else 0
        st.metric("Predicted Price", f"${predicted_price:,.2f}", 
                 delta=f"{price_diff_pct:+.1f}%")
    
    with col4:
        signal_strength = latest_signal.get("signal_strength", "Medium")
        strength_color = {
            "Strong": "[Strong]",
            "Medium": "[Medium]",
            "Weak": "[Weak]"
        }.get(signal_strength, "[Unknown]")
        st.metric("Signal Strength", f"{strength_color} {signal_strength}")

# Tabs for different signal views
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Technical Indicators",
    "On-Chain Metrics",
    "Sentiment Analysis",
    "Macro Indicators",
    "Signal History"
])

# Technical Indicators Tab
with tab1:
    st.markdown("### Technical Analysis Indicators")
    
    if comprehensive_signals:
        # Combine all technical-related indicators from categorized response
        tech_indicators = {}
        for category in ['technical', 'momentum', 'volatility', 'volume', 'trend']:
            if category in comprehensive_signals:
                tech_indicators.update(comprehensive_signals[category])
        
        # Only proceed if we have indicators
        if tech_indicators:
            # Categorize indicators
            momentum_indicators = {}
            trend_indicators = {}
            volatility_indicators = {}
            volume_indicators = {}
            
            for indicator, value in tech_indicators.items():
                if any(x in indicator.lower() for x in ["rsi", "macd", "stoch", "momentum", "roc"]):
                    momentum_indicators[indicator] = value
                elif any(x in indicator.lower() for x in ["sma", "ema", "wma", "trend", "adx"]):
                    trend_indicators[indicator] = value
                elif any(x in indicator.lower() for x in ["bb", "atr", "volatility", "std"]):
                    volatility_indicators[indicator] = value
                elif any(x in indicator.lower() for x in ["volume", "obv", "vwap", "mfi"]):
                    volume_indicators[indicator] = value
                else:
                    trend_indicators[indicator] = value  # Default category
            
            # Display indicators by category
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Momentum Indicators")
                for indicator, value in momentum_indicators.items():
                    # Determine sentiment
                    sentiment = "neutral"
                    if "rsi" in indicator.lower():
                        if value < 30:
                            sentiment = "bullish"
                        elif value > 70:
                            sentiment = "bearish"
                    elif "macd" in indicator.lower() and "signal" in indicator.lower():
                        if value > 0:
                            sentiment = "bullish"
                        else:
                            sentiment = "bearish"
                    
                    col_ind1, col_ind2, col_ind3 = st.columns([2, 1, 1])
                    with col_ind1:
                        st.markdown(f"**{indicator.replace('_', ' ').title()}**")
                    with col_ind2:
                        st.markdown(f'<span class="{sentiment}">{value:.2f}</span>', unsafe_allow_html=True)
                    with col_ind3:
                        if sentiment == "bullish":
                            st.markdown("[Bullish]")
                        elif sentiment == "bearish":
                            st.markdown("[Bearish]")
                        else:
                            st.markdown("[Neutral]")
                
                st.markdown("#### Trend Indicators")
                for indicator, value in trend_indicators.items():
                    col_ind1, col_ind2 = st.columns([2, 1])
                    with col_ind1:
                        st.markdown(f"**{indicator.replace('_', ' ').title()}**")
                    with col_ind2:
                        st.markdown(f"{value:.2f}")
            
            with col2:
                st.markdown("#### Volatility Indicators")
                for indicator, value in volatility_indicators.items():
                    col_ind1, col_ind2 = st.columns([2, 1])
                    with col_ind1:
                        st.markdown(f"**{indicator.replace('_', ' ').title()}**")
                    with col_ind2:
                        st.markdown(f"{value:.2f}")
                
                st.markdown("#### Volume Indicators")
                for indicator, value in volume_indicators.items():
                    col_ind1, col_ind2 = st.columns([2, 1])
                    with col_ind1:
                        st.markdown(f"**{indicator.replace('_', ' ').title()}**")
                    with col_ind2:
                        if "volume" in indicator.lower():
                            st.markdown(f"{value/1e6:.1f}M")
                        else:
                            st.markdown(f"{value:.2f}")
    else:
        st.info("Technical indicators are not available. Waiting for comprehensive signals to be calculated.")

# On-Chain Metrics Tab
with tab2:
    st.markdown("### On-Chain Metrics Analysis")
    
    if comprehensive_signals and "on_chain" in comprehensive_signals:
        onchain_data = comprehensive_signals["on_chain"]
        
        # Network Activity
        st.markdown("#### Network Activity")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            active_addresses = onchain_data.get("active_addresses", 0)
            st.metric("Active Addresses", f"{active_addresses/1000:.1f}K", 
                     help="Number of unique addresses active in transactions")
        
        with col2:
            transaction_count = onchain_data.get("transaction_count", 0)
            st.metric("Daily Transactions", f"{transaction_count/1000:.1f}K",
                     help="Total number of transactions")
        
        with col3:
            hash_rate = onchain_data.get("hash_rate", 0)
            st.metric("Hash Rate", f"{hash_rate/1e18:.1f} EH/s",
                     help="Network security metric")
        
        with col4:
            difficulty = onchain_data.get("difficulty", 0)
            st.metric("Mining Difficulty", f"{difficulty/1e12:.1f}T",
                     help="Mining difficulty level")
        
        # Value Metrics
        st.markdown("#### Value Metrics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            nvt_ratio = onchain_data.get("nvt_ratio", 0)
            nvt_signal = "bullish" if nvt_ratio < 50 else "bearish" if nvt_ratio > 100 else "neutral"
            st.markdown(f"""
            <div class="indicator-card">
                <h4>NVT Ratio</h4>
                <div class="indicator-value {nvt_signal}">{nvt_ratio:.1f}</div>
                <p>Network Value to Transactions</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            mvrv_ratio = onchain_data.get("mvrv_ratio", 1.0)
            mvrv_signal = "bullish" if mvrv_ratio < 1 else "bearish" if mvrv_ratio > 3 else "neutral"
            st.markdown(f"""
            <div class="indicator-card">
                <h4>MVRV Ratio</h4>
                <div class="indicator-value {mvrv_signal}">{mvrv_ratio:.2f}</div>
                <p>Market Value to Realized Value</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            realized_cap = onchain_data.get("realized_cap", 0)
            st.markdown(f"""
            <div class="indicator-card">
                <h4>Realized Cap</h4>
                <div class="indicator-value">${realized_cap/1e9:.1f}B</div>
                <p>Aggregate cost basis</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Exchange Flows
        st.markdown("#### 🏦 Exchange Flows")
        col1, col2 = st.columns(2)
        
        with col1:
            exchange_inflow = onchain_data.get("exchange_inflow", 0)
            exchange_outflow = onchain_data.get("exchange_outflow", 0)
            net_flow = exchange_outflow - exchange_inflow
            
            flow_fig = go.Figure(data=[
                go.Bar(name='Inflow', x=['Exchanges'], y=[exchange_inflow], marker_color='red'),
                go.Bar(name='Outflow', x=['Exchanges'], y=[exchange_outflow], marker_color='green')
            ])
            flow_fig.update_layout(
                title="Exchange BTC Flow (24h)",
                barmode='group',
                height=300,
                showlegend=True
            )
            st.plotly_chart(flow_fig, use_container_width=True)
        
        with col2:
            # Flow interpretation
            flow_signal = "bullish" if net_flow > 1000 else "bearish" if net_flow < -1000 else "neutral"
            st.markdown(f"""
            <div class="indicator-card">
                <h4>Net Exchange Flow</h4>
                <div class="indicator-value {flow_signal}">{net_flow:,.0f} BTC</div>
                <p>{"Accumulation" if net_flow > 0 else "Distribution"}</p>
                <small>Positive = More leaving exchanges (bullish)</small>
            </div>
            """, unsafe_allow_html=True)
        
        # Whale Activity
        st.markdown("#### 🐋 Whale Activity")
        whale_transactions = onchain_data.get("whale_transactions", 0)
        large_holder_pct = onchain_data.get("large_holder_percentage", 0)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Whale Transactions (>100 BTC)", whale_transactions,
                     help="Number of large transactions in last 24h")
        with col2:
            st.metric("Large Holder %", f"{large_holder_pct:.1f}%",
                     help="Percentage held by addresses with >1000 BTC")
    else:
        st.info("On-chain metrics data is not available. Please ensure the backend is fetching blockchain data.")

# Sentiment Analysis Tab
with tab3:
    st.markdown("### Market Sentiment Analysis")
    
    if comprehensive_signals and "sentiment" in comprehensive_signals:
        sentiment_data = comprehensive_signals["sentiment"]
        
        # Overall Sentiment Score
        overall_sentiment = sentiment_data.get("overall_sentiment", 50)
        sentiment_class = "bullish" if overall_sentiment > 60 else "bearish" if overall_sentiment < 40 else "neutral"
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            # Create gauge chart for overall sentiment
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = overall_sentiment,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Overall Market Sentiment"},
                delta = {'reference': 50},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 25], 'color': "red"},
                        {'range': [25, 40], 'color': "orange"},
                        {'range': [40, 60], 'color': "yellow"},
                        {'range': [60, 75], 'color': "lightgreen"},
                        {'range': [75, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Individual Sentiment Metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 😱 Fear & Greed Index")
            fear_greed = sentiment_data.get("fear_greed_index", 50)
            fear_greed_text = (
                "Extreme Fear" if fear_greed < 20 else
                "Fear" if fear_greed < 40 else
                "Neutral" if fear_greed < 60 else
                "Greed" if fear_greed < 80 else
                "Extreme Greed"
            )
            
            st.markdown(f"""
            <div class="indicator-card">
                <h4>{fear_greed_text}</h4>
                <div class="indicator-value {sentiment_class}">{fear_greed}</div>
                <p>Market emotion indicator</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Social sentiment breakdown
            st.markdown("#### Social Media Sentiment")
            reddit_sentiment = sentiment_data.get("reddit_sentiment", 0)
            twitter_sentiment = sentiment_data.get("twitter_sentiment", 0)
            
            social_fig = go.Figure(data=[
                go.Bar(
                    x=['Reddit', 'Twitter'],
                    y=[reddit_sentiment, twitter_sentiment],
                    marker_color=['orange', 'lightblue']
                )
            ])
            social_fig.update_layout(
                title="Social Platform Sentiment",
                yaxis_title="Sentiment Score",
                height=300
            )
            st.plotly_chart(social_fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 📰 News Sentiment")
            news_sentiment = sentiment_data.get("news_sentiment", 0)
            news_volume = sentiment_data.get("news_volume", 0)
            
            news_signal = "bullish" if news_sentiment > 0.6 else "bearish" if news_sentiment < 0.4 else "neutral"
            st.markdown(f"""
            <div class="indicator-card">
                <h4>News Sentiment Score</h4>
                <div class="indicator-value {news_signal}">{news_sentiment:.2f}</div>
                <p>{news_volume} articles analyzed (24h)</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Funding rates
            st.markdown("#### 💸 Market Positioning")
            funding_rate = sentiment_data.get("funding_rate", 0)
            long_short_ratio = sentiment_data.get("long_short_ratio", 1.0)
            
            position_fig = go.Figure()
            position_fig.add_trace(go.Indicator(
                mode = "number+delta",
                value = long_short_ratio,
                title = {"text": "Long/Short Ratio"},
                delta = {'reference': 1.0, 'relative': True},
                domain = {'x': [0, 0.5], 'y': [0, 1]}
            ))
            position_fig.add_trace(go.Indicator(
                mode = "number+delta",
                value = funding_rate * 100,
                title = {"text": "Funding Rate %"},
                delta = {'reference': 0},
                domain = {'x': [0.5, 1], 'y': [0, 1]}
            ))
            position_fig.update_layout(height=200)
            st.plotly_chart(position_fig, use_container_width=True)
        
        # Sentiment Trends
        st.markdown("#### Sentiment Trends (7 Days)")
        if "sentiment_history" in sentiment_data:
            history_df = pd.DataFrame(sentiment_data["sentiment_history"])
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=history_df['timestamp'],
                y=history_df['overall'],
                mode='lines',
                name='Overall Sentiment',
                line=dict(width=3, color='blue')
            ))
            fig.add_trace(go.Scatter(
                x=history_df['timestamp'],
                y=history_df['fear_greed'],
                mode='lines',
                name='Fear & Greed',
                line=dict(width=2, color='orange', dash='dot')
            ))
            fig.add_hline(y=50, line_dash="dash", line_color="gray",
                         annotation_text="Neutral")
            
            fig.update_layout(
                title="Sentiment Evolution",
                xaxis_title="Date",
                yaxis_title="Sentiment Score",
                height=400,
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Sentiment data is not available. Please check if sentiment APIs are configured.")

# Macro Indicators Tab
with tab4:
    st.markdown("### Macro Economic Indicators")
    
    if comprehensive_signals and "macro_indicators" in comprehensive_signals:
        macro_data = comprehensive_signals["macro_indicators"]
        
        # Correlation Overview
        st.markdown("#### BTC Correlations")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            sp500_corr = macro_data.get("sp500_correlation", 0)
            corr_strength = "Strong" if abs(sp500_corr) > 0.7 else "Moderate" if abs(sp500_corr) > 0.4 else "Weak"
            st.markdown(f"""
            <div class="indicator-card">
                <h4>S&P 500</h4>
                <div class="indicator-value">{sp500_corr:+.2f}</div>
                <p>{corr_strength} correlation</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            gold_corr = macro_data.get("gold_correlation", 0)
            corr_strength = "Strong" if abs(gold_corr) > 0.7 else "Moderate" if abs(gold_corr) > 0.4 else "Weak"
            st.markdown(f"""
            <div class="indicator-card">
                <h4>Gold</h4>
                <div class="indicator-value">{gold_corr:+.2f}</div>
                <p>{corr_strength} correlation</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            dxy_corr = macro_data.get("dxy_correlation", 0)
            corr_strength = "Strong" if abs(dxy_corr) > 0.7 else "Moderate" if abs(dxy_corr) > 0.4 else "Weak"
            st.markdown(f"""
            <div class="indicator-card">
                <h4>Dollar Index</h4>
                <div class="indicator-value">{dxy_corr:+.2f}</div>
                <p>{corr_strength} correlation</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            vix_value = macro_data.get("vix", 0)
            vix_signal = "high" if vix_value > 30 else "moderate" if vix_value > 20 else "low"
            st.markdown(f"""
            <div class="indicator-card">
                <h4>VIX</h4>
                <div class="indicator-value">{vix_value:.1f}</div>
                <p>{vix_signal.title()} volatility</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Economic Indicators
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Monetary Indicators")
            
            # Interest rates
            fed_rate = macro_data.get("fed_funds_rate", 0)
            real_yield = macro_data.get("real_yield_10y", 0)
            
            rates_df = pd.DataFrame({
                'Indicator': ['Fed Funds Rate', '10Y Real Yield', '2Y Treasury', '10Y Treasury'],
                'Value': [
                    fed_rate,
                    real_yield,
                    macro_data.get("treasury_2y", 0),
                    macro_data.get("treasury_10y", 0)
                ]
            })
            
            fig = go.Figure(data=[
                go.Bar(
                    x=rates_df['Indicator'],
                    y=rates_df['Value'],
                    marker_color=['red', 'orange', 'yellow', 'green']
                )
            ])
            fig.update_layout(
                title="Interest Rate Environment",
                yaxis_title="Rate (%)",
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Global Markets")
            
            # Create correlation heatmap
            correlations = {
                'BTC': [1.0, sp500_corr, gold_corr, dxy_corr],
                'S&P 500': [sp500_corr, 1.0, 0.3, -0.5],
                'Gold': [gold_corr, 0.3, 1.0, -0.7],
                'DXY': [dxy_corr, -0.5, -0.7, 1.0]
            }
            
            corr_df = pd.DataFrame(correlations, 
                                 index=['BTC', 'S&P 500', 'Gold', 'DXY'])
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_df.values,
                x=corr_df.columns,
                y=corr_df.index,
                colorscale='RdBu',
                zmid=0,
                text=corr_df.values,
                texttemplate='%{text:.2f}',
                textfont={"size": 12}
            ))
            fig.update_layout(
                title="Asset Correlation Matrix",
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Macro Risk Assessment
        st.markdown("#### Macro Risk Assessment")
        
        inflation_rate = macro_data.get("inflation_rate", 0)
        gdp_growth = macro_data.get("gdp_growth", 0)
        unemployment = macro_data.get("unemployment_rate", 0)
        
        risk_score = 50  # Base score
        if inflation_rate > 4:
            risk_score += 20
        if gdp_growth < 0:
            risk_score += 15
        if unemployment > 5:
            risk_score += 10
        if vix_value > 25:
            risk_score += 15
        
        risk_level = "High" if risk_score > 70 else "Medium" if risk_score > 50 else "Low"
        risk_color = "bearish" if risk_score > 70 else "neutral" if risk_score > 50 else "bullish"
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f"""
            <div class="indicator-card" style="text-align: center;">
                <h3>Macro Risk Level</h3>
                <div class="indicator-value {risk_color}" style="font-size: 3em;">{risk_level}</div>
                <p>Score: {risk_score}/100</p>
                <small>Based on inflation, GDP, unemployment, and market volatility</small>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Macro indicator data is not available. Configure FRED API key for enhanced data.")

# Signal History Tab
with tab5:
    st.markdown("### Signal History & Performance")
    
    if signal_history:
        # Convert to DataFrame
        history_df = pd.DataFrame(signal_history)
        history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
        
        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)
        
        # Calculate accuracy if actual outcomes are available
        if 'actual_movement' in history_df.columns:
            correct_signals = history_df[
                ((history_df['signal'] == 'buy') & (history_df['actual_movement'] > 0)) |
                ((history_df['signal'] == 'sell') & (history_df['actual_movement'] < 0))
            ]
            accuracy = len(correct_signals) / len(history_df) * 100 if len(history_df) > 0 else 0
        else:
            accuracy = 0
        
        with col1:
            st.metric("Total Signals (24h)", len(history_df))
        
        with col2:
            buy_signals = len(history_df[history_df['signal'] == 'buy'])
            sell_signals = len(history_df[history_df['signal'] == 'sell'])
            st.metric("Buy/Sell Ratio", f"{buy_signals}/{sell_signals}")
        
        with col3:
            avg_confidence = history_df['confidence'].mean() if 'confidence' in history_df else 0
            st.metric("Avg Confidence", f"{avg_confidence:.1%}")
        
        with col4:
            st.metric("Accuracy", f"{accuracy:.1f}%", 
                     help="Based on actual price movements after signals")
        
        # Signal timeline
        st.markdown("#### Signal Timeline")
        
        fig = go.Figure()
        
        # Add buy signals
        buy_df = history_df[history_df['signal'] == 'buy']
        if not buy_df.empty:
            fig.add_trace(go.Scatter(
                x=buy_df['timestamp'],
                y=buy_df['price_prediction'],
                mode='markers',
                name='Buy Signals',
                marker=dict(
                    color='green',
                    size=buy_df['confidence'] * 20,
                    symbol='triangle-up'
                ),
                text=buy_df['confidence'].apply(lambda x: f"Confidence: {x:.1%}"),
                hovertemplate='%{x}<br>Price: $%{y:,.0f}<br>%{text}<extra></extra>'
            ))
        
        # Add sell signals
        sell_df = history_df[history_df['signal'] == 'sell']
        if not sell_df.empty:
            fig.add_trace(go.Scatter(
                x=sell_df['timestamp'],
                y=sell_df['price_prediction'],
                mode='markers',
                name='Sell Signals',
                marker=dict(
                    color='red',
                    size=sell_df['confidence'] * 20,
                    symbol='triangle-down'
                ),
                text=sell_df['confidence'].apply(lambda x: f"Confidence: {x:.1%}"),
                hovertemplate='%{x}<br>Price: $%{y:,.0f}<br>%{text}<extra></extra>'
            ))
        
        # Add price line if available
        if 'price' in history_df.columns:
            fig.add_trace(go.Scatter(
                x=history_df['timestamp'],
                y=history_df['price_prediction'],
                mode='lines',
                name='BTC Price',
                line=dict(color='gray', width=1, dash='dot')
            ))
        
        fig.update_layout(
            title="Trading Signals Over Time",
            xaxis_title="Time",
            yaxis_title="Price ($)",
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Signal distribution
        col1, col2 = st.columns(2)
        
        with col1:
            # Signal type distribution
            signal_counts = history_df['signal'].value_counts()
            
            fig = go.Figure(data=[go.Pie(
                labels=signal_counts.index,
                values=signal_counts.values,
                hole=.3,
                marker_colors=['green', 'red', 'gray']
            )])
            fig.update_layout(
                title="Signal Distribution",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Confidence distribution
            fig = go.Figure(data=[
                go.Histogram(
                    x=history_df['confidence'],
                    nbinsx=20,
                    marker_color='blue',
                    opacity=0.7
                )
            ])
            fig.update_layout(
                title="Confidence Distribution",
                xaxis_title="Confidence",
                yaxis_title="Count",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Recent signals table
        st.markdown("#### Recent Signals")
        
        recent_signals = history_df.sort_values('timestamp', ascending=False).head(10)
        display_df = recent_signals[['timestamp', 'signal', 'confidence', 'price_prediction']].copy()
        display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
        display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x:.1%}")
        display_df['price_prediction'] = display_df['price_prediction'].apply(lambda x: f"${x:,.0f}")
        # Rename columns for better display
        display_df = display_df.rename(columns={'price_prediction': 'Price', 'timestamp': 'Time', 'signal': 'Signal', 'confidence': 'Confidence'})
        
        st.dataframe(
            display_df.style.map(
                lambda x: 'color: green' if x == 'buy' else 'color: red' if x == 'sell' else '',
                subset=['Signal']
            ),
            use_container_width=True,
            height=400
        )
    else:
        st.info("No signal history available. Signals will appear here as they are generated.")

# Add auto-refresh option
if st.sidebar.checkbox("Auto-refresh (30s)", value=False):
    import time
    time.sleep(30)
    st.rerun()
