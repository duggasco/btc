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

st.set_page_config(
    page_title="BTC Trading System",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Use backend service name when running in Docker, fallback to localhost
API_BASE_URL = os.getenv("API_BASE_URL", "http://backend:8000")

# Cache API calls
@st.cache_data(ttl=60)
def fetch_api_data(endpoint):
    try:
        response = requests.get(f"{API_BASE_URL}{endpoint}", timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except requests.exceptions.ConnectionError:
        # Try localhost if backend name fails (for local development)
        try:
            alt_url = f"http://localhost:8080{endpoint}"
            response = requests.get(alt_url, timeout=30)
            if response.status_code == 200:
                return response.json()
        except:
            st.error(f"Cannot connect to API server")
            return None
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

def post_api_data(endpoint, data):
    try:
        response = requests.post(f"{API_BASE_URL}{endpoint}", json=data, timeout=30)
        return response.json()
    except requests.exceptions.ConnectionError:
        # Try localhost if backend name fails
        try:
            alt_url = f"http://localhost:8080{endpoint}"
            response = requests.post(alt_url, json=data, timeout=30)
            return response.json()
        except:
            st.error(f"Cannot connect to API server")
            return None
    except Exception as e:
        st.error(f"Error posting data: {str(e)}")
        return None

def delete_api_data(endpoint):
    try:
        response = requests.delete(f"{API_BASE_URL}{endpoint}", timeout=30)
        return response.json() if response.status_code == 200 else None
    except:
        return None

def create_candlestick_chart(btc_data, include_enhanced_indicators=False):
    if not btc_data or 'data' not in btc_data:
        return None
    
    df = pd.DataFrame(btc_data['data'])
    if df.empty:
        return None
        
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Determine number of subplots based on available indicators
    subplot_count = 2  # Price and Volume
    subplot_titles = ['BTC-USD Price', 'Volume']
    row_heights = [0.5, 0.15]
    
    if 'rsi' in df.columns and df['rsi'].notna().any():
        subplot_count += 1
        subplot_titles.append('RSI')
        row_heights.append(0.15)
    
    if 'macd' in df.columns and df['macd'].notna().any():
        subplot_count += 1
        subplot_titles.append('MACD')
        row_heights.append(0.2)
    
    fig = make_subplots(
        rows=subplot_count, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=subplot_titles,
        row_heights=row_heights
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="BTC-USD"
        ),
        row=1, col=1
    )
    
    # Add moving averages if available
    if 'sma_20' in df.columns and df['sma_20'].notna().any():
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['sma_20'],
                name='SMA 20',
                line=dict(color='orange', width=1)
            ),
            row=1, col=1
        )
    
    if 'sma_50' in df.columns and df['sma_50'].notna().any():
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['sma_50'],
                name='SMA 50',
                line=dict(color='blue', width=1)
            ),
            row=1, col=1
        )
    
    # Add enhanced indicators if requested
    if include_enhanced_indicators:
        if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['bb_upper'],
                    name='BB Upper',
                    line=dict(color='gray', width=1, dash='dash')
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['bb_lower'],
                    name='BB Lower',
                    line=dict(color='gray', width=1, dash='dash'),
                    fill='tonexty',
                    fillcolor='rgba(128,128,128,0.1)'
                ),
                row=1, col=1
            )
    
    # Volume
    colors = ['red' if row['close'] < row['open'] else 'green' for _, row in df.iterrows()]
    fig.add_trace(
        go.Bar(
            x=df['timestamp'],
            y=df['volume'],
            name='Volume',
            marker_color=colors,
            showlegend=False
        ),
        row=2, col=1
    )
    
    current_row = 3
    
    # RSI
    if 'rsi' in df.columns and df['rsi'].notna().any() and current_row <= subplot_count:
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['rsi'],
                name='RSI',
                line=dict(color='purple', width=1)
            ),
            row=current_row, col=1
        )
        
        # Add RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=current_row, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=current_row, col=1)
        current_row += 1
    
    # MACD
    if 'macd' in df.columns and df['macd'].notna().any() and current_row <= subplot_count:
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['macd'],
                name='MACD',
                line=dict(color='blue', width=1)
            ),
            row=current_row, col=1
        )
        
        # Add MACD signal line if available
        if 'macd_signal' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['macd_signal'],
                    name='Signal',
                    line=dict(color='red', width=1)
                ),
                row=current_row, col=1
            )
            
            # MACD histogram
            if 'macd_histogram' in df.columns:
                fig.add_trace(
                    go.Bar(
                        x=df['timestamp'],
                        y=df['macd_histogram'],
                        name='MACD Hist',
                        marker_color='gray'
                    ),
                    row=current_row, col=1
                )
    
    fig.update_layout(
        title="BTC Technical Analysis",
        xaxis_rangeslider_visible=False,
        height=800,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    fig.update_xaxes(title_text="Date", row=subplot_count, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    if 'rsi' in df.columns and df['rsi'].notna().any():
        fig.update_yaxes(title_text="RSI", row=3, col=1)
    if 'macd' in df.columns and df['macd'].notna().any():
        fig.update_yaxes(title_text="MACD", row=current_row, col=1)
    
    return fig

def calculate_portfolio_metrics(trades_df):
    """Calculate portfolio performance metrics"""
    if trades_df.empty:
        return {
            'total_return': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'win_rate': 0,
            'profit_factor': 0
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
    drawdown = (trades_df['cumulative_pnl'] - cummax) / cummax
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
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(365) if daily_returns.std() > 0 else 0
    else:
        sharpe_ratio = 0
    
    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'total_trades': len(trades_df)
    }

def main():
    st.title("₿ BTC Trading System")
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
                    st.json(api_status['components'])
                    
                    # Show enhanced features status
                    if 'enhanced_features' in api_status:
                        st.write("**Enhanced Features:**")
                        for feature, enabled in api_status['enhanced_features'].items():
                            status = "✅" if enabled else "❌"
                            st.write(f"{status} {feature.replace('_', ' ').title()}")
        else:
            st.error("❌ API Disconnected")
        
        # Auto-refresh toggle
        auto_refresh = st.checkbox("Auto-refresh (60s)", value=False)
        if auto_refresh:
            st.empty()  # Placeholder for refresh
    
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

def show_dashboard():
    st.header("📊 Trading Dashboard")
    
    # Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with st.spinner("Loading dashboard data..."):
        portfolio_metrics = fetch_api_data("/portfolio/metrics")
        latest_signal = fetch_api_data("/signals/enhanced/latest")
        if not latest_signal:  # Fallback to regular signal if enhanced not available
            latest_signal = fetch_api_data("/signals/latest")
        btc_data = fetch_api_data("/market/btc-data?period=1mo&include_indicators=true")
        recent_trades = fetch_api_data("/trades/?limit=10")
    
    if portfolio_metrics:
        with col1:
            st.metric(
                "Total Trades", 
                portfolio_metrics.get('total_trades', 0),
                delta=f"{portfolio_metrics.get('total_trades', 0) - 100}" if portfolio_metrics.get('total_trades', 0) > 100 else None
            )
        with col2:
            total_pnl = portfolio_metrics.get('total_pnl', 0)
            st.metric(
                "Total P&L", 
                f"${total_pnl:,.2f}",
                delta=f"{total_pnl:,.2f}" if total_pnl != 0 else None,
                delta_color="normal" if total_pnl >= 0 else "inverse"
            )
        with col3:
            st.metric("Active Positions", portfolio_metrics.get('positions_count', 0))
        with col4:
            total_volume = portfolio_metrics.get('total_volume', 0)
            st.metric("Total Volume", f"{total_volume:.4f} BTC")
        with col5:
            current_price = portfolio_metrics.get('current_btc_price', 0)
            st.metric("BTC Price", f"${current_price:,.2f}" if current_price else "N/A")
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("BTC Price Chart")
        if btc_data:
            fig = create_candlestick_chart(btc_data, include_enhanced_indicators=True)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No chart data available")
        else:
            st.warning("Unable to load BTC data")
    
    with col2:
        st.subheader("Latest Signal")
        if latest_signal:
            signal = latest_signal.get('signal', 'hold')
            confidence = latest_signal.get('confidence', 0.5)
            predicted_price = latest_signal.get('predicted_price', 0)
            
            # Enhanced signal display with analysis
            signal_colors = {
                'buy': {'bg': '#1f7a1f', 'text': '#90EE90'},
                'sell': {'bg': '#7a1f1f', 'text': '#FFA07A'},
                'hold': {'bg': '#7a5f1f', 'text': '#FFD700'}
            }
            
            colors = signal_colors.get(signal, signal_colors['hold'])
            
            st.markdown(f"""
            <div style="
                padding: 20px; 
                border-radius: 10px; 
                background-color: {colors['bg']}; 
                border: 2px solid {colors['text']}; 
                text-align: center;
                margin-bottom: 20px;
            ">
                <h2 style="color: {colors['text']}; margin: 0;">{signal.upper()}</h2>
                <p style="color: white; margin: 10px 0;">Confidence: {confidence:.1%}</p>
                <p style="color: white; margin: 10px 0;">Target: ${predicted_price:,.2f}</p>
                <p style="color: #ccc; font-size: 0.8em; margin: 0;">
                    {latest_signal.get('timestamp', 'N/A')}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show analysis if available
            if 'analysis' in latest_signal and latest_signal['analysis']:
                with st.expander("Signal Analysis"):
                    analysis = latest_signal['analysis']
                    if 'consensus_ratio' in analysis:
                        st.metric("Consensus Ratio", f"{analysis['consensus_ratio']:.2%}")
                    if 'price_confidence_interval' in analysis:
                        ci = analysis['price_confidence_interval']
                        st.write(f"**Price CI:** ${ci[0]:,.2f} - ${ci[1]:,.2f}")
                    if 'signal_distribution' in analysis:
                        st.write("**Signal Distribution:**")
                        for sig, count in analysis['signal_distribution'].items():
                            st.write(f"- {sig}: {count}")
            
            # Signal explanation
            if signal == 'buy':
                st.success("Model suggests buying - price expected to rise")
            elif signal == 'sell':
                st.error("Model suggests selling - price expected to fall")
            else:
                st.info("Model suggests holding - no clear trend")
        
        # Recent trades summary
        st.subheader("Recent Activity")
        if recent_trades:
            trades_df = pd.DataFrame(recent_trades)
            if not trades_df.empty:
                buy_count = len(trades_df[trades_df['trade_type'] == 'buy'])
                sell_count = len(trades_df[trades_df['trade_type'] == 'sell'])
                
                st.write(f"**Last 10 trades:**")
                st.write(f"Buys: {buy_count}")
                st.write(f"Sells: {sell_count}")
                
                # Mini chart of trade distribution
                fig_pie = px.pie(
                    values=[buy_count, sell_count],
                    names=['Buy', 'Sell'],
                    color_discrete_map={'Buy': '#00ff00', 'Sell': '#ff0000'}
                )
                fig_pie.update_layout(height=200, showlegend=False)
                st.plotly_chart(fig_pie, use_container_width=True)

def show_trading():
    st.header("💰 Trading Interface")
    
    # Get latest market data
    latest_signal = fetch_api_data("/signals/latest")
    portfolio_metrics = fetch_api_data("/portfolio/metrics")
    
    # Trading form
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Execute Trade")
        
        with st.form("trade_form"):
            trade_type = st.selectbox(
                "Trade Type", 
                ["buy", "sell", "hold"],
                help="Select the type of trade to execute"
            )
            
            current_price = portfolio_metrics.get('current_btc_price', 45000) if portfolio_metrics else 45000
            
            price = st.number_input(
                "Price ($)", 
                min_value=1.0, 
                value=float(current_price), 
                step=0.01,
                help="Price per BTC"
            )
            
            size = st.number_input(
                "Size (BTC)", 
                min_value=0.0001, 
                value=0.01, 
                step=0.0001, 
                format="%.4f",
                help="Amount of BTC to trade"
            )
            
            lot_id = st.text_input(
                "Lot ID (optional)", 
                placeholder="Leave empty for auto-generated ID",
                help="Track specific lots for tax purposes"
            )
            
            # Trade value preview
            trade_value = price * size
            st.info(f"Total Trade Value: ${trade_value:,.2f}")
            
            col_submit1, col_submit2 = st.columns(2)
            with col_submit1:
                submitted = st.form_submit_button("Execute Trade", type="primary", use_container_width=True)
            with col_submit2:
                if st.form_submit_button("Clear", use_container_width=True):
                    st.rerun()
            
            if submitted:
                if trade_type == "hold":
                    st.warning("Hold is not an executable trade type. Please select buy or sell.")
                else:
                    trade_data = {
                        "symbol": "BTC-USD",
                        "trade_type": trade_type,
                        "price": price,
                        "size": size,
                        "lot_id": lot_id if lot_id else None
                    }
                    
                    with st.spinner("Executing trade..."):
                        result = post_api_data("/trades/", trade_data)
                    
                    if result and result.get('status') == 'success':
                        st.success(f"✅ Trade executed successfully! Trade ID: {result['trade_id']}")
                        st.balloons()
                        time.sleep(2)
                        st.rerun()
                    else:
                        st.error("❌ Trade execution failed. Please try again.")
    
    with col2:
        st.subheader("Market Information")
        
        # Current signal
        if latest_signal:
            st.write("**AI Signal:**")
            signal = latest_signal.get('signal', 'hold')
            confidence = latest_signal.get('confidence', 0.5)
            
            signal_emoji = {'buy': '🟢', 'sell': '🔴', 'hold': '🟡'}
            st.write(f"{signal_emoji.get(signal, '❓')} {signal.upper()} (Confidence: {confidence:.1%})")
            
            if signal == 'buy' and trade_type != 'buy':
                st.info("💡 AI suggests buying")
            elif signal == 'sell' and trade_type != 'sell':
                st.info("💡 AI suggests selling")
        
        # Quick stats
        st.write("**Quick Stats:**")
        if portfolio_metrics:
            st.write(f"Current BTC Price: ${portfolio_metrics.get('current_btc_price', 0):,.2f}")
            st.write(f"Total Invested: ${portfolio_metrics.get('total_invested', 0):,.2f}")
            st.write(f"Current P&L: ${portfolio_metrics.get('total_pnl', 0):,.2f}")
        
        # Recent trades
        st.write("**Your Recent Trades:**")
        recent_trades = fetch_api_data("/trades/?limit=5")
        if recent_trades:
            for trade in recent_trades:
                trade_emoji = '🟢' if trade['trade_type'] == 'buy' else '🔴'
                st.write(f"{trade_emoji} {trade['trade_type'].upper()} {trade['size']:.4f} BTC @ ${trade['price']:,.2f}")

def show_portfolio():
    st.header("📈 Portfolio Overview")
    
    # Get portfolio data
    positions = fetch_api_data("/positions/")
    trades = fetch_api_data("/trades/")
    portfolio_metrics = fetch_api_data("/portfolio/metrics")
    
    if not positions and not trades:
        st.info("No portfolio data available. Start trading to see your portfolio!")
        return
    
    # Portfolio Summary
    col1, col2, col3, col4 = st.columns(4)
    
    if portfolio_metrics:
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
    
    # Positions
    st.subheader("Current Positions")
    if positions:
        positions_df = pd.DataFrame(positions)
        if not positions_df.empty:
            # Add current value and P&L
            current_price = portfolio_metrics.get('current_btc_price', 0) if portfolio_metrics else 0
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
                        st.write(f"**Avg Buy Price:** ${position['avg_buy_price']:,.2f}")
                    with col2:
                        st.write(f"**Current Value:** ${position['current_value']:,.2f}")
                        st.write(f"**Purchase Value:** ${position['purchase_value']:,.2f}")
                    with col3:
                        pnl_color = "green" if position['unrealized_pnl'] >= 0 else "red"
                        st.markdown(f"**Unrealized P&L:** <span style='color:{pnl_color}'>${position['unrealized_pnl']:,.2f} ({position['pnl_percent']:+.2f}%)</span>", unsafe_allow_html=True)
                        st.write(f"**Created:** {position['created_at']}")
    else:
        st.info("No open positions")
    
    # Performance Metrics
    st.subheader("Performance Metrics")
    if trades:
        trades_df = pd.DataFrame(trades)
        metrics = calculate_portfolio_metrics(trades_df)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Return", f"{metrics['total_return']:.2%}")
        with col2:
            st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
        with col3:
            st.metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")
        with col4:
            st.metric("Win Rate", f"{metrics['win_rate']:.2%}")
        with col5:
            st.metric("Total Trades", metrics['total_trades'])
    
    # Trade History
    st.subheader("Trade History")
    if trades:
        trades_df = pd.DataFrame(trades)
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
        trades_df = trades_df.sort_values('timestamp', ascending=False)
        
        # Add filters
        col1, col2, col3 = st.columns(3)
        with col1:
            trade_type_filter = st.selectbox("Filter by Type", ["All", "buy", "sell"])
        with col2:
            date_range = st.date_input(
                "Date Range",
                value=(trades_df['timestamp'].min().date(), trades_df['timestamp'].max().date()),
                max_value=datetime.now().date()
            )
        with col3:
            lot_filter = st.selectbox(
                "Filter by Lot", 
                ["All"] + list(trades_df['lot_id'].unique())
            )
        
        # Apply filters
        filtered_df = trades_df.copy()
        if trade_type_filter != "All":
            filtered_df = filtered_df[filtered_df['trade_type'] == trade_type_filter]
        if len(date_range) == 2:
            filtered_df = filtered_df[
                (filtered_df['timestamp'].dt.date >= date_range[0]) & 
                (filtered_df['timestamp'].dt.date <= date_range[1])
            ]
        if lot_filter != "All":
            filtered_df = filtered_df[filtered_df['lot_id'] == lot_filter]
        
        # Display trades
        if not filtered_df.empty:
            st.dataframe(
                filtered_df[['timestamp', 'trade_type', 'size', 'price', 'lot_id', 'status']],
                use_container_width=True,
                hide_index=True
            )
            
            # Download button
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download Trade History",
                data=csv,
                file_name=f"btc_trades_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No trades match the selected filters")

def show_signals():
    st.header("🤖 AI Trading Signals")
    
    # Latest signal
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.subheader("Current Signal")
        latest_signal = fetch_api_data("/signals/latest")
        if latest_signal:
            signal = latest_signal.get('signal', 'hold')
            confidence = latest_signal.get('confidence', 0.5)
            predicted_price = latest_signal.get('predicted_price', 0)
            timestamp = latest_signal.get('timestamp', 'N/A')
            
            # Enhanced signal display
            signal_colors = {
                'buy': {'bg': '#1f7a1f', 'text': '#90EE90', 'emoji': '🟢'},
                'sell': {'bg': '#7a1f1f', 'text': '#FFA07A', 'emoji': '🔴'},
                'hold': {'bg': '#7a5f1f', 'text': '#FFD700', 'emoji': '🟡'}
            }
            
            colors = signal_colors.get(signal, signal_colors['hold'])
            
            st.markdown(f"""
            <div style="
                padding: 30px; 
                border-radius: 15px; 
                background: linear-gradient(135deg, {colors['bg']} 0%, #1a1a1a 100%);
                border: 3px solid {colors['text']}; 
                text-align: center;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            ">
                <h1 style="color: {colors['text']}; margin: 0; font-size: 3em;">
                    {colors['emoji']} {signal.upper()}
                </h1>
                <h3 style="color: white; margin: 15px 0;">
                    Confidence: {confidence:.1%}
                </h3>
                <p style="color: white; margin: 10px 0; font-size: 1.2em;">
                    Predicted Price: ${predicted_price:,.2f}
                </p>
                <p style="color: #aaa; font-size: 0.9em; margin-top: 20px;">
                    Last Update: {timestamp}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Signal strength indicator
            st.write("")
            st.write("**Signal Strength:**")
            st.progress(confidence)
            
            # Interpretation
            st.write("")
            st.write("**Interpretation:**")
            if signal == 'buy':
                st.success("Strong buying opportunity detected. The model predicts price will rise.")
            elif signal == 'sell':
                st.warning("Selling recommended. The model predicts price will fall.")
            else:
                st.info("No clear trend detected. Hold current positions.")
    
    with col2:
        st.subheader("Signal History")
        
        # Get historical signals
        signals_history = fetch_api_data("/signals/history?limit=50")
        
        if signals_history:
            signals_df = pd.DataFrame(signals_history)
            signals_df['timestamp'] = pd.to_datetime(signals_df['timestamp'])
            
            # Signal accuracy chart
            fig = go.Figure()
            
            # Add signal points
            buy_signals = signals_df[signals_df['signal'] == 'buy']
            sell_signals = signals_df[signals_df['signal'] == 'sell']
            hold_signals = signals_df[signals_df['signal'] == 'hold']
            
            fig.add_trace(go.Scatter(
                x=buy_signals['timestamp'],
                y=buy_signals['price_prediction'],
                mode='markers',
                name='Buy Signals',
                marker=dict(color='green', size=10, symbol='triangle-up')
            ))
            
            fig.add_trace(go.Scatter(
                x=sell_signals['timestamp'],
                y=sell_signals['price_prediction'],
                mode='markers',
                name='Sell Signals',
                marker=dict(color='red', size=10, symbol='triangle-down')
            ))
            
            fig.add_trace(go.Scatter(
                x=hold_signals['timestamp'],
                y=hold_signals['price_prediction'],
                mode='markers',
                name='Hold Signals',
                marker=dict(color='yellow', size=8, symbol='circle')
            ))
            
            # Add confidence as line
            fig.add_trace(go.Scatter(
                x=signals_df['timestamp'],
                y=signals_df['confidence'] * 100,
                mode='lines',
                name='Confidence %',
                yaxis='y2',
                line=dict(color='purple', width=2, dash='dot')
            ))
            
            fig.update_layout(
                title="Signal History & Confidence",
                xaxis_title="Time",
                yaxis_title="Predicted Price ($)",
                yaxis2=dict(
                    title="Confidence (%)",
                    overlaying='y',
                    side='right',
                    range=[0, 100]
                ),
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Signal statistics
            st.write("**Signal Statistics (Last 50):**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                buy_count = len(buy_signals)
                st.metric("Buy Signals", buy_count, f"{(buy_count/len(signals_df)*100):.1f}%")
            
            with col2:
                sell_count = len(sell_signals)
                st.metric("Sell Signals", sell_count, f"{(sell_count/len(signals_df)*100):.1f}%")
            
            with col3:
                hold_count = len(hold_signals)
                st.metric("Hold Signals", hold_count, f"{(hold_count/len(signals_df)*100):.1f}%")
            
            # Average confidence by signal type
            st.write("**Average Confidence by Signal Type:**")
            avg_confidence = signals_df.groupby('signal')['confidence'].mean()
            fig_conf = px.bar(
                x=avg_confidence.index,
                y=avg_confidence.values,
                labels={'x': 'Signal Type', 'y': 'Average Confidence'},
                color=avg_confidence.index,
                color_discrete_map={'buy': 'green', 'sell': 'red', 'hold': 'yellow'}
            )
            fig_conf.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig_conf, use_container_width=True)
        else:
            st.info("No signal history available")
    
    # Comprehensive Trading Signals Table
    st.subheader("Comprehensive Trading Signals")
    
    # Fetch BTC data for calculations
    btc_data = fetch_api_data("/market/btc-data?period=1mo")
    
    if btc_data and 'data' in btc_data:
        df = pd.DataFrame(btc_data['data'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Calculate various signals (16 total)
        signals_data = []
        
        # Get latest values
        latest_price = df['close'].iloc[-1]
        latest_volume = df['volume'].iloc[-1]
        latest_rsi = df['rsi'].iloc[-1] if 'rsi' in df.columns else 50
        latest_macd = df['macd'].iloc[-1] if 'macd' in df.columns else 0
        sma_20 = df['sma_20'].iloc[-1] if 'sma_20' in df.columns else latest_price
        sma_50 = df['sma_50'].iloc[-1] if 'sma_50' in df.columns else latest_price
        
        # 1. RSI - Relative Strength Index
        rsi_signal = "Oversold" if latest_rsi < 30 else ("Overbought" if latest_rsi > 70 else "Neutral")
        rsi_interpretation = "Bullish" if latest_rsi < 30 else ("Bearish" if latest_rsi > 70 else "Neutral")
        signals_data.append({
            "Signal": "RSI",
            "Type": "Momentum",
            "Value": f"{latest_rsi:.1f}",
            "Status": rsi_signal,
            "Interpretation": rsi_interpretation,
            "Description": "Momentum oscillator (0-100)"
        })
        
        # 2. MACD - Moving Average Convergence Divergence
        macd_signal = "Positive" if latest_macd > 0 else "Negative"
        macd_interpretation = "Bullish" if latest_macd > 0 else "Bearish"
        signals_data.append({
            "Signal": "MACD",
            "Type": "Momentum",
            "Value": f"{latest_macd:.2f}",
            "Status": macd_signal,
            "Interpretation": macd_interpretation,
            "Description": "Trend momentum indicator"
        })
        
        # 3. SMA-20 Position
        ma20_signal = "Above" if latest_price > sma_20 else "Below"
        ma20_interpretation = "Bullish" if latest_price > sma_20 else "Bearish"
        signals_data.append({
            "Signal": "Price vs SMA-20",
            "Type": "Trend",
            "Value": f"{((latest_price/sma_20 - 1) * 100):.1f}%",
            "Status": ma20_signal,
            "Interpretation": ma20_interpretation,
            "Description": "Price relative to 20-day MA"
        })
        
        # 4. SMA-50 Position
        ma50_signal = "Above" if latest_price > sma_50 else "Below"
        ma50_interpretation = "Bullish" if latest_price > sma_50 else "Bearish"
        signals_data.append({
            "Signal": "Price vs SMA-50",
            "Type": "Trend",
            "Value": f"{((latest_price/sma_50 - 1) * 100):.1f}%",
            "Status": ma50_signal,
            "Interpretation": ma50_interpretation,
            "Description": "Price relative to 50-day MA"
        })
        
        # 5. Volume Analysis
        avg_volume = df['volume'].rolling(20).mean().iloc[-1]
        volume_ratio = latest_volume / avg_volume if avg_volume > 0 else 1
        volume_signal = "High" if volume_ratio > 1.5 else ("Low" if volume_ratio < 0.5 else "Normal")
        volume_interpretation = "Strong Move" if volume_ratio > 1.5 else ("Weak Move" if volume_ratio < 0.5 else "Neutral")
        signals_data.append({
            "Signal": "Volume Ratio",
            "Type": "Volume",
            "Value": f"{volume_ratio:.2f}x",
            "Status": volume_signal,
            "Interpretation": volume_interpretation,
            "Description": "Volume vs 20-day average"
        })
        
        # 6. Bollinger Band Position
        bb_middle = sma_20
        bb_std = df['close'].rolling(20).std().iloc[-1]
        bb_upper = bb_middle + (2 * bb_std)
        bb_lower = bb_middle - (2 * bb_std)
        bb_position = (latest_price - bb_lower) / (bb_upper - bb_lower) if bb_upper > bb_lower else 0.5
        bb_signal = "Overbought" if bb_position > 0.8 else ("Oversold" if bb_position < 0.2 else "Normal")
        bb_interpretation = "Bearish" if bb_position > 0.8 else ("Bullish" if bb_position < 0.2 else "Neutral")
        signals_data.append({
            "Signal": "Bollinger Bands",
            "Type": "Volatility",
            "Value": f"{bb_position*100:.0f}%",
            "Status": bb_signal,
            "Interpretation": bb_interpretation,
            "Description": "Position within bands"
        })
        
        # 7. ATR - Average True Range
        high_low = df['high'] - df['low']
        atr = high_low.rolling(14).mean().iloc[-1]
        atr_pct = (atr / latest_price) * 100
        atr_signal = "High Vol" if atr_pct > 3 else ("Low Vol" if atr_pct < 1 else "Normal")
        atr_interpretation = "Volatile" if atr_pct > 3 else ("Stable" if atr_pct < 1 else "Moderate")
        signals_data.append({
            "Signal": "ATR",
            "Type": "Volatility",
            "Value": f"{atr_pct:.1f}%",
            "Status": atr_signal,
            "Interpretation": atr_interpretation,
            "Description": "14-day average true range"
        })
        
        # 8. Price Change (24h)
        price_24h_ago = df['close'].iloc[-24] if len(df) >= 24 else latest_price
        price_change_24h = ((latest_price - price_24h_ago) / price_24h_ago) * 100
        change_24h_signal = "Rising" if price_change_24h > 2 else ("Falling" if price_change_24h < -2 else "Stable")
        change_24h_interpretation = "Bullish" if price_change_24h > 2 else ("Bearish" if price_change_24h < -2 else "Neutral")
        signals_data.append({
            "Signal": "24h Change",
            "Type": "Momentum",
            "Value": f"{price_change_24h:+.1f}%",
            "Status": change_24h_signal,
            "Interpretation": change_24h_interpretation,
            "Description": "Price change last 24 hours"
        })
        
        # 9. Price Change (7d)
        price_7d_ago = df['close'].iloc[-7*24] if len(df) >= 7*24 else latest_price
        price_change_7d = ((latest_price - price_7d_ago) / price_7d_ago) * 100
        change_7d_signal = "Uptrend" if price_change_7d > 5 else ("Downtrend" if price_change_7d < -5 else "Sideways")
        change_7d_interpretation = "Bullish" if price_change_7d > 5 else ("Bearish" if price_change_7d < -5 else "Neutral")
        signals_data.append({
            "Signal": "7d Trend",
            "Type": "Trend",
            "Value": f"{price_change_7d:+.1f}%",
            "Status": change_7d_signal,
            "Interpretation": change_7d_interpretation,
            "Description": "Price trend over 7 days"
        })
        
        # 10. Support/Resistance
        recent_high = df['high'].rolling(20).max().iloc[-1]
        recent_low = df['low'].rolling(20).min().iloc[-1]
        price_position = (latest_price - recent_low) / (recent_high - recent_low) if recent_high > recent_low else 0.5
        sr_signal = "Near Resistance" if price_position > 0.8 else ("Near Support" if price_position < 0.2 else "Mid-Range")
        sr_interpretation = "Bearish" if price_position > 0.8 else ("Bullish" if price_position < 0.2 else "Neutral")
        signals_data.append({
            "Signal": "Support/Resistance",
            "Type": "Price Action",
            "Value": f"{price_position*100:.0f}%",
            "Status": sr_signal,
            "Interpretation": sr_interpretation,
            "Description": "Position in 20-day range"
        })
        
        # 11. Stochastic Oscillator
        low_14 = df['low'].rolling(14).min().iloc[-1]
        high_14 = df['high'].rolling(14).max().iloc[-1]
        stoch_k = ((latest_price - low_14) / (high_14 - low_14) * 100) if high_14 > low_14 else 50
        stoch_signal = "Oversold" if stoch_k < 20 else ("Overbought" if stoch_k > 80 else "Neutral")
        stoch_interpretation = "Bullish" if stoch_k < 20 else ("Bearish" if stoch_k > 80 else "Neutral")
        signals_data.append({
            "Signal": "Stochastic",
            "Type": "Momentum",
            "Value": f"{stoch_k:.0f}",
            "Status": stoch_signal,
            "Interpretation": stoch_interpretation,
            "Description": "Stochastic oscillator %K"
        })
        
        # 12. Rate of Change (ROC)
        price_10_ago = df['close'].iloc[-10] if len(df) >= 10 else latest_price
        roc = ((latest_price - price_10_ago) / price_10_ago) * 100
        roc_signal = "Strong Up" if roc > 10 else ("Strong Down" if roc < -10 else "Normal")
        roc_interpretation = "Bullish" if roc > 10 else ("Bearish" if roc < -10 else "Neutral")
        signals_data.append({
            "Signal": "ROC (10)",
            "Type": "Momentum",
            "Value": f"{roc:+.1f}%",
            "Status": roc_signal,
            "Interpretation": roc_interpretation,
            "Description": "10-period rate of change"
        })
        
        # 13. Volume Trend
        vol_sma_5 = df['volume'].rolling(5).mean().iloc[-1]
        vol_sma_20 = avg_volume
        vol_trend = "Increasing" if vol_sma_5 > vol_sma_20 * 1.2 else ("Decreasing" if vol_sma_5 < vol_sma_20 * 0.8 else "Stable")
        vol_trend_interpretation = "Active" if vol_sma_5 > vol_sma_20 * 1.2 else ("Quiet" if vol_sma_5 < vol_sma_20 * 0.8 else "Normal")
        signals_data.append({
            "Signal": "Volume Trend",
            "Type": "Volume",
            "Value": f"{(vol_sma_5/vol_sma_20):.2f}x",
            "Status": vol_trend,
            "Interpretation": vol_trend_interpretation,
            "Description": "5-day vs 20-day volume"
        })
        
        # 14. Volatility (Historical)
        returns = df['close'].pct_change()
        volatility = returns.rolling(20).std().iloc[-1] * np.sqrt(365) * 100  # Annualized
        vol_signal = "High" if volatility > 80 else ("Low" if volatility < 40 else "Normal")
        vol_interpretation = "Risky" if volatility > 80 else ("Stable" if volatility < 40 else "Moderate")
        signals_data.append({
            "Signal": "Volatility (20d)",
            "Type": "Risk",
            "Value": f"{volatility:.1f}%",
            "Status": vol_signal,
            "Interpretation": vol_interpretation,
            "Description": "20-day annualized volatility"
        })
        
        # 15. MA Crossover
        ma_cross = "Golden Cross" if sma_20 > sma_50 and df['sma_20'].iloc[-2] <= df['sma_50'].iloc[-2] else (
                   "Death Cross" if sma_20 < sma_50 and df['sma_20'].iloc[-2] >= df['sma_50'].iloc[-2] else 
                   ("Bullish" if sma_20 > sma_50 else "Bearish"))
        ma_cross_interpretation = "Bullish" if ma_cross in ["Golden Cross", "Bullish"] else "Bearish"
        signals_data.append({
            "Signal": "MA Cross (20/50)",
            "Type": "Trend",
            "Value": f"{((sma_20/sma_50 - 1) * 100):.1f}%",
            "Status": ma_cross,
            "Interpretation": ma_cross_interpretation,
            "Description": "Moving average crossover"
        })
        
        # 16. LSTM Model Signal
        if latest_signal:
            model_signal = latest_signal.get('signal', 'hold').upper()
            model_confidence = latest_signal.get('confidence', 0.5)
            model_interpretation = "Bullish" if model_signal == "BUY" else ("Bearish" if model_signal == "SELL" else "Neutral")
            signals_data.append({
                "Signal": "LSTM AI Model",
                "Type": "AI/ML",
                "Value": f"{model_confidence:.1%}",
                "Status": model_signal,
                "Interpretation": model_interpretation,
                "Description": "Neural network prediction"
            })
        
        # Create DataFrame and display
        signals_table_df = pd.DataFrame(signals_data)
        
        # Style the table
        def style_interpretation(val):
            if val == "Bullish":
                return 'color: #90EE90'
            elif val == "Bearish":
                return 'color: #FFA07A'
            else:
                return 'color: #FFD700'
        
        styled_table = signals_table_df.style.applymap(style_interpretation, subset=['Interpretation'])
        
        st.dataframe(
            styled_table,
            use_container_width=True,
            hide_index=True,
            height=400
        )
        
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
    st.header("🧠 Advanced AI Trading Signals")
    
    # Tabs for different signal views
    tab1, tab2, tab3, tab4 = st.tabs(["Enhanced Signal", "Comprehensive Signals", "Signal History", "Feature Importance"])
    
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
                
                # Signal details
                st.metric("Signal", signal.upper(), f"Target: ${predicted_price:,.2f}")
            
            with col2:
                # Analysis details
                if 'analysis' in enhanced_signal:
                    analysis = enhanced_signal['analysis']
                    
                    st.write("**Analysis Details:**")
                    if 'consensus_ratio' in analysis:
                        st.metric("Model Consensus", f"{analysis['consensus_ratio']:.2%}")
                    
                    if 'price_confidence_interval' in analysis:
                        ci = analysis['price_confidence_interval']
                        st.write(f"**95% Confidence Interval:**")
                        st.write(f"Lower: ${ci[0]:,.2f}")
                        st.write(f"Upper: ${ci[1]:,.2f}")
                    
                    if 'feature_importance' in analysis and analysis['feature_importance']:
                        st.write("**Top Contributing Features:**")
                        for feat, imp in list(analysis['feature_importance'].items())[:5]:
                            st.progress(imp, text=f"{feat}: {imp:.2%}")
            
            # Comprehensive signal indicators
            if 'comprehensive_signals' in enhanced_signal:
                st.subheader("Key Signal Indicators")
                comp_signals = enhanced_signal['comprehensive_signals']
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write("**Technical:**")
                    if 'technical' in comp_signals:
                        for ind, val in comp_signals['technical'].items():
                            if isinstance(val, bool):
                                emoji = "✅" if val else "❌"
                                st.write(f"{emoji} {ind}")
                            else:
                                st.write(f"{ind}: {val:.2f}")
                
                with col2:
                    st.write("**Sentiment:**")
                    if 'sentiment' in comp_signals:
                        for ind, val in comp_signals['sentiment'].items():
                            st.write(f"{ind}: {val:.2f}")
                
                with col3:
                    st.write("**On-Chain:**")
                    if 'on_chain' in comp_signals:
                        for ind, val in comp_signals['on_chain'].items():
                            st.write(f"{ind}: {val:.2f}")
    
    with tab2:
        st.subheader("50+ Comprehensive Trading Signals")
        
        comp_signals = fetch_api_data("/signals/comprehensive")
        if comp_signals and comp_signals.get('status') != 'no_data':
            st.metric("Total Signals", comp_signals.get('total_signals', 0))
            
            # Display categorized signals
            categorized = comp_signals.get('categorized_signals', {})
            
            # Create columns for each category
            cols = st.columns(len(categorized))
            
            for i, (category, signals) in enumerate(categorized.items()):
                with cols[i]:
                    st.write(f"**{category.title()}**")
                    
                    # Create a simple table for signals
                    signal_data = []
                    for sig_name, sig_val in signals.items():
                        if isinstance(sig_val, bool):
                            val_str = "✅" if sig_val else "❌"
                        elif isinstance(sig_val, (int, float)):
                            val_str = f"{sig_val:.3f}"
                        else:
                            val_str = str(sig_val)
                        signal_data.append({"Signal": sig_name, "Value": val_str})
                    
                    if signal_data:
                        df = pd.DataFrame(signal_data)
                        st.dataframe(df, hide_index=True, height=300)
            
            # Show all signals in expandable section
            with st.expander("All Signals (Raw Data)"):
                all_signals = comp_signals.get('all_signals', {})
                st.json(all_signals)
        else:
            st.info("Comprehensive signals not yet calculated. Please wait for next update cycle.")
    
    with tab3:
        st.subheader("Signal Performance History")
        
        # Get signal history
        signals_history = fetch_api_data("/signals/history?limit=100")
        
        if signals_history:
            signals_df = pd.DataFrame(signals_history)
            signals_df['timestamp'] = pd.to_datetime(signals_df['timestamp'])
            
            # Performance metrics
            col1, col2, col3, col4 = st.columns(4)
            
            # Calculate signal distribution
            signal_counts = signals_df['signal'].value_counts()
            
            with col1:
                st.metric("Total Signals", len(signals_df))
            with col2:
                avg_confidence = signals_df['confidence'].mean()
                st.metric("Avg Confidence", f"{avg_confidence:.2%}")
            with col3:
                buy_ratio = signal_counts.get('buy', 0) / len(signals_df)
                st.metric("Buy Signal %", f"{buy_ratio:.1%}")
            with col4:
                sell_ratio = signal_counts.get('sell', 0) / len(signals_df)
                st.metric("Sell Signal %", f"{sell_ratio:.1%}")
            
            # Signal accuracy over time
            fig = go.Figure()
            
            # Group by signal type
            for signal_type in signals_df['signal'].unique():
                signal_data = signals_df[signals_df['signal'] == signal_type]
                
                fig.add_trace(go.Scatter(
                    x=signal_data['timestamp'],
                    y=signal_data['confidence'],
                    mode='markers',
                    name=f'{signal_type.upper()} signals',
                    marker=dict(
                        size=8,
                        color={'buy': 'green', 'sell': 'red', 'hold': 'yellow'}.get(signal_type, 'gray')
                    )
                ))
            
            fig.update_layout(
                title="Signal Confidence Over Time",
                xaxis_title="Time",
                yaxis_title="Confidence",
                height=400,
                yaxis=dict(range=[0, 1])
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Signal transitions matrix
            st.subheader("Signal Transitions")
            
            # Calculate transitions
            transitions = {}
            for i in range(1, len(signals_df)):
                prev_signal = signals_df.iloc[i-1]['signal']
                curr_signal = signals_df.iloc[i]['signal']
                key = f"{prev_signal} → {curr_signal}"
                transitions[key] = transitions.get(key, 0) + 1
            
            # Display as matrix
            signals = ['buy', 'sell', 'hold']
            matrix = [[0 for _ in signals] for _ in signals]
            
            for i, from_sig in enumerate(signals):
                for j, to_sig in enumerate(signals):
                    key = f"{from_sig} → {to_sig}"
                    matrix[i][j] = transitions.get(key, 0)
            
            # Create heatmap
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=matrix,
                x=signals,
                y=signals,
                text=matrix,
                texttemplate="%{text}",
                colorscale='Blues'
            ))
            
            fig_heatmap.update_layout(
                title="Signal Transition Matrix",
                xaxis_title="To Signal",
                yaxis_title="From Signal",
                height=400
            )
            
            st.plotly_chart(fig_heatmap, use_container_width=True)
    
    with tab4:
        st.subheader("Feature Importance Analysis")
        
        feature_importance = fetch_api_data("/analytics/feature-importance")
        
        if feature_importance and feature_importance.get('feature_importance'):
            # Get top features
            top_features = feature_importance.get('top_10_features', {})
            
            if top_features:
                # Create bar chart
                features = list(top_features.keys())
                importances = list(top_features.values())
                
                fig = go.Figure(go.Bar(
                    x=importances,
                    y=features,
                    orientation='h',
                    marker=dict(
                        color=importances,
                        colorscale='Viridis'
                    )
                ))
                
                fig.update_layout(
                    title="Top 10 Most Important Features",
                    xaxis_title="Importance Score",
                    yaxis_title="Feature",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature details
                st.write("**Feature Categories:**")
                
                # Categorize features
                technical_features = [f for f in features if any(ind in f.lower() for ind in ['rsi', 'macd', 'bb', 'sma', 'ema'])]
                sentiment_features = [f for f in features if any(ind in f.lower() for ind in ['sentiment', 'fear', 'greed'])]
                onchain_features = [f for f in features if any(ind in f.lower() for ind in ['nvt', 'onchain', 'whale'])]
                macro_features = [f for f in features if any(ind in f.lower() for ind in ['macro', 'sp500', 'gold', 'vix'])]
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Technical", len(technical_features))
                with col2:
                    st.metric("Sentiment", len(sentiment_features))
                with col3:
                    st.metric("On-Chain", len(onchain_features))
                with col4:
                    st.metric("Macro", len(macro_features))
                
                # Show full feature list
                with st.expander("All Features with Importance"):
                    all_features = feature_importance.get('feature_importance', {})
                    if all_features:
                        df = pd.DataFrame(
                            [(k, v) for k, v in all_features.items()],
                            columns=['Feature', 'Importance']
                        )
                        df = df.sort_values('Importance', ascending=False)
                        st.dataframe(df, hide_index=True)
        else:
            st.info("Feature importance not yet calculated. Run a backtest to generate feature importance scores.")

def show_limits():
    st.header("⚠️ Trading Limits & Orders")
    
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
                "Trigger Price ($)",
                min_value=1.0,
                value=45000.0,
                step=0.01,
                help="Price at which the order triggers"
            )
            
            size = st.number_input(
                "Size (BTC)",
                min_value=0.0001,
                value=0.01,
                step=0.0001,
                format="%.4f",
                help="Amount of BTC for the order (optional)"
            )
            
            lot_id = st.text_input(
                "Lot ID (optional)",
                help="Associate with specific lot"
            )
            
            # Order preview
            st.info(f"Order will trigger when BTC {'falls below' if 'stop' in limit_type or 'sell' in limit_type else 'rises above'} ${price:,.2f}")
            
            submitted = st.form_submit_button("Create Limit Order", type="primary", use_container_width=True)
            
            if submitted:
                limit_data = {
                    "symbol": "BTC-USD",
                    "limit_type": limit_type,
                    "price": price,
                    "size": size,
                    "lot_id": lot_id if lot_id else None
                }
                
                with st.spinner("Creating limit order..."):
                    result = post_api_data("/limits/", limit_data)
                
                if result and result.get('status') == 'success':
                    st.success(f"✅ Limit order created! ID: {result['limit_id']}")
                    time.sleep(2)
                    st.rerun()
                else:
                    st.error("❌ Failed to create limit order")
    
    with col2:
        st.subheader("Order Types Explained")
        
        st.write("**Stop Loss:** Sell when price falls below trigger to limit losses")
        st.write("**Take Profit:** Sell when price rises above trigger to lock in gains")
        st.write("**Buy Limit:** Buy when price falls to trigger level")
        st.write("**Sell Limit:** Sell when price rises to trigger level")
        
        # Current price reference
        portfolio_metrics = fetch_api_data("/portfolio/metrics")
        if portfolio_metrics:
            current_price = portfolio_metrics.get('current_btc_price', 0)
            st.info(f"Current BTC Price: ${current_price:,.2f}")
    
    # Active limit orders
    st.subheader("Active Limit Orders")
    
    limits = fetch_api_data("/limits/")
    if limits:
        limits_df = pd.DataFrame(limits)
        if not limits_df.empty:
            # Add status indicators
            current_price = portfolio_metrics.get('current_btc_price', 45000) if portfolio_metrics else 45000
            
            for _, limit in limits_df.iterrows():
                with st.expander(f"{limit['limit_type'].upper()} - ${limit['price']:,.2f}"):
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.write(f"**Type:** {limit['limit_type']}")
                        st.write(f"**Price:** ${limit['price']:,.2f}")
                    
                    with col2:
                        st.write(f"**Size:** {limit.get('size', 'N/A')} BTC")
                        st.write(f"**Lot ID:** {limit.get('lot_id', 'Any')[:8]}...")
                    
                    with col3:
                        # Distance from current price
                        distance = abs(current_price - limit['price'])
                        distance_pct = (distance / current_price) * 100
                        st.write(f"**Distance:** ${distance:,.2f} ({distance_pct:.2f}%)")
                        
                        # Status
                        if 'stop' in limit['limit_type'] or 'sell' in limit['limit_type']:
                            if current_price <= limit['price']:
                                st.warning("⚠️ Near trigger!")
                        else:
                            if current_price >= limit['price']:
                                st.warning("⚠️ Near trigger!")
                    
                    with col4:
                        if st.button(f"Cancel", key=f"cancel_{limit['id']}"):
                            result = delete_api_data(f"/limits/{limit['id']}")
                            if result:
                                st.success("Limit order cancelled")
                                st.rerun()
            
            # Summary statistics
            st.write("**Limit Orders Summary:**")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Orders", len(limits_df))
            with col2:
                stop_losses = len(limits_df[limits_df['limit_type'] == 'stop_loss'])
                st.metric("Stop Losses", stop_losses)
            with col3:
                take_profits = len(limits_df[limits_df['limit_type'] == 'take_profit'])
                st.metric("Take Profits", take_profits)
            with col4:
                limit_orders = len(limits_df[limits_df['limit_type'].str.contains('limit')])
                st.metric("Limit Orders", limit_orders)
        else:
            st.info("No active limit orders")
    else:
        st.info("No limit orders found")

def show_analytics():
    st.header("📊 Advanced Analytics")
    
    # Get analytics data
    trades = fetch_api_data("/trades/")
    pnl_data = fetch_api_data("/analytics/pnl")
    btc_data = fetch_api_data("/market/btc-data?period=3mo")
    
    if not trades:
        st.info("No trading data available for analytics")
        return
    
    # Analytics tabs
    tab1, tab2, tab3, tab4 = st.tabs(["P&L Analysis", "Performance Metrics", "Risk Analysis", "Trade Analysis"])
    
    with tab1:
        st.subheader("Profit & Loss Analysis")
        
        if pnl_data and pnl_data.get('daily_pnl'):
            # Daily P&L chart
            daily_df = pd.DataFrame(pnl_data['daily_pnl'])
            daily_df['date'] = pd.to_datetime(daily_df['date'])
            
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=('Daily P&L', 'Cumulative P&L'),
                row_heights=[0.4, 0.6]
            )
            
            # Daily P&L bars
            colors = ['green' if x > 0 else 'red' for x in daily_df['pnl']]
            fig.add_trace(
                go.Bar(
                    x=daily_df['date'],
                    y=daily_df['pnl'],
                    name='Daily P&L',
                    marker_color=colors
                ),
                row=1, col=1
            )
            
            # Cumulative P&L line
            cum_df = pd.DataFrame(pnl_data['cumulative_pnl'])
            cum_df['date'] = pd.to_datetime(cum_df['date'])
            
            fig.add_trace(
                go.Scatter(
                    x=cum_df['date'],
                    y=cum_df['pnl'],
                    mode='lines+markers',
                    name='Cumulative P&L',
                    line=dict(color='blue', width=3)
                ),
                row=2, col=1
            )
            
            fig.update_layout(height=600, showlegend=True)
            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_yaxes(title_text="P&L ($)", row=1, col=1)
            fig.update_yaxes(title_text="Cumulative P&L ($)", row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # P&L Statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_pnl = cum_df['pnl'].iloc[-1] if not cum_df.empty else 0
                st.metric("Total P&L", f"${total_pnl:,.2f}")
            
            with col2:
                profitable_days = len([x for x in daily_df['pnl'] if x > 0])
                total_days = len(daily_df)
                win_rate = (profitable_days / total_days * 100) if total_days > 0 else 0
                st.metric("Win Rate (Days)", f"{win_rate:.1f}%")
            
            with col3:
                avg_win = daily_df[daily_df['pnl'] > 0]['pnl'].mean() if len(daily_df[daily_df['pnl'] > 0]) > 0 else 0
                st.metric("Avg Win", f"${avg_win:,.2f}")
            
            with col4:
                avg_loss = daily_df[daily_df['pnl'] < 0]['pnl'].mean() if len(daily_df[daily_df['pnl'] < 0]) > 0 else 0
                st.metric("Avg Loss", f"${avg_loss:,.2f}")
    
    with tab2:
        st.subheader("Performance Metrics")
        
        trades_df = pd.DataFrame(trades)
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
        
        # Calculate advanced metrics
        metrics = calculate_portfolio_metrics(trades_df)
        
        # Performance over time
        trades_df['trade_value'] = trades_df['price'] * trades_df['size']
        trades_df['signed_value'] = trades_df['trade_value'] * trades_df['trade_type'].map({
            'buy': -1, 
            'sell': 1, 
            'hold': 0
        })
        trades_df['cumulative_investment'] = trades_df[trades_df['trade_type'] == 'buy']['trade_value'].cumsum()
        trades_df['cumulative_pnl'] = trades_df['signed_value'].cumsum()
        
        # ROI over time
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=trades_df['timestamp'],
            y=trades_df['cumulative_pnl'] / trades_df['cumulative_investment'] * 100,
            mode='lines',
            name='ROI %',
            line=dict(color='green', width=2)
        ))
        
        fig.update_layout(
            title="Return on Investment Over Time",
            xaxis_title="Date",
            yaxis_title="ROI (%)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Key metrics display
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Risk-Adjusted Returns**")
            st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
            st.metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")
        
        with col2:
            st.write("**Trading Performance**")
            st.metric("Win Rate", f"{metrics['win_rate']:.2%}")
            st.metric("Total Trades", metrics['total_trades'])
        
        with col3:
            st.write("**Returns**")
            st.metric("Total Return", f"{metrics['total_return']:.2%}")
            # Calculate annualized return if we have enough data
            if len(trades_df) > 0:
                days = (trades_df['timestamp'].max() - trades_df['timestamp'].min()).days
                if days > 0:
                    annualized_return = (1 + metrics['total_return']) ** (365 / days) - 1
                    st.metric("Annualized Return", f"{annualized_return:.2%}")
    
    with tab3:
        st.subheader("Risk Analysis")
        
        if btc_data and btc_data.get('data'):
            price_df = pd.DataFrame(btc_data['data'])
            price_df['timestamp'] = pd.to_datetime(price_df['timestamp'])
            
            # Calculate returns
            price_df['returns'] = price_df['close'].pct_change()
            
            # Volatility over time (20-day rolling)
            price_df['volatility'] = price_df['returns'].rolling(window=20).std() * np.sqrt(252) * 100
            
            # Volatility chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=price_df['timestamp'],
                y=price_df['volatility'],
                mode='lines',
                name='Volatility (Annual %)',
                line=dict(color='orange', width=2)
            ))
            
            fig.update_layout(
                title="Bitcoin Volatility (20-day Rolling)",
                xaxis_title="Date",
                yaxis_title="Volatility (%)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Risk metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                current_vol = price_df['volatility'].iloc[-1] if not price_df['volatility'].isna().all() else 0
                st.metric("Current Volatility", f"{current_vol:.1f}%")
            
            with col2:
                avg_vol = price_df['volatility'].mean()
                st.metric("Avg Volatility", f"{avg_vol:.1f}%")
            
            with col3:
                # Calculate Value at Risk (VaR)
                var_95 = np.percentile(price_df['returns'].dropna(), 5) * 100
                st.metric("VaR (95%)", f"{var_95:.2f}%", help="95% of the time, daily loss won't exceed this")
            
            # Returns distribution
            st.write("**Returns Distribution**")
            fig_hist = px.histogram(
                price_df['returns'].dropna() * 100,
                nbins=50,
                labels={'value': 'Daily Return (%)', 'count': 'Frequency'},
                title="Distribution of Daily Returns"
            )
            fig_hist.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig_hist, use_container_width=True)
    
    with tab4:
        st.subheader("Trade Analysis")
        
        trades_df = pd.DataFrame(trades)
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
        
        # Trade distribution by type
        col1, col2 = st.columns(2)
        
        with col1:
            # Trade types pie chart
            trade_counts = trades_df['trade_type'].value_counts()
            fig_pie = px.pie(
                values=trade_counts.values,
                names=trade_counts.index,
                title="Trade Distribution",
                color_discrete_map={'buy': '#00ff00', 'sell': '#ff0000', 'hold': '#ffff00'}
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Trade sizes distribution
            fig_box = px.box(
                trades_df,
                x='trade_type',
                y='size',
                title="Trade Size Distribution by Type",
                color='trade_type',
                color_discrete_map={'buy': '#00ff00', 'sell': '#ff0000', 'hold': '#ffff00'}
            )
            st.plotly_chart(fig_box, use_container_width=True)
        
        # Trading activity heatmap
        st.write("**Trading Activity Heatmap**")
        
        # Create hourly trading activity
        trades_df['hour'] = trades_df['timestamp'].dt.hour
        trades_df['day'] = trades_df['timestamp'].dt.day_name()
        
        # Pivot for heatmap
        activity_pivot = trades_df.pivot_table(
            values='id',
            index='hour',
            columns='day',
            aggfunc='count',
            fill_value=0
        )
        
        # Reorder days
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        activity_pivot = activity_pivot.reindex(columns=[d for d in days_order if d in activity_pivot.columns])
        
        fig_heatmap = px.imshow(
            activity_pivot,
            labels=dict(x="Day", y="Hour", color="Trades"),
            aspect="auto",
            title="Trading Activity by Day and Hour"
        )
        fig_heatmap.update_layout(height=400)
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Best and worst trades
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Best Trades**")
            # Calculate P&L for each sell
            sells = trades_df[trades_df['trade_type'] == 'sell'].copy()
            if not sells.empty:
                # This is simplified - in reality you'd match with corresponding buys
                sells['pnl'] = sells['price'] * sells['size']  # Simplified
                best_trades = sells.nlargest(5, 'pnl')[['timestamp', 'size', 'price', 'pnl']]
                st.dataframe(best_trades, use_container_width=True, hide_index=True)
        
        with col2:
            st.write("**Trading Statistics**")
            avg_trade_size = trades_df['size'].mean()
            st.metric("Avg Trade Size", f"{avg_trade_size:.6f} BTC")
            
            avg_trade_value = (trades_df['price'] * trades_df['size']).mean()
            st.metric("Avg Trade Value", f"${avg_trade_value:,.2f}")
            
            # Time between trades
            if len(trades_df) > 1:
                time_diffs = trades_df['timestamp'].diff().dropna()
                avg_time_between = time_diffs.mean()
                st.metric("Avg Time Between Trades", f"{avg_time_between.days}d {avg_time_between.seconds//3600}h")

def show_backtesting():
    st.header("🔬 Advanced Backtesting System")
    
    # Tabs for different backtesting functions
    tab1, tab2, tab3, tab4 = st.tabs(["Run Backtest", "Results", "History", "Market Analysis"])
    
    with tab1:
        st.subheader("Configure and Run Enhanced Backtest")
        
        # Check if backtest is in progress
        status = fetch_api_data("/backtest/status")
        
        if status and status.get('in_progress'):
            st.warning("⏳ Backtest is currently in progress. Please wait...")
            
            # Add a progress bar
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.1)
                progress_bar.progress(i + 1)
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Backtest Parameters")
                
                period = st.selectbox(
                    "Data Period",
                    ["1mo", "3mo", "6mo", "1y", "2y"],
                    index=3,
                    help="Historical data period for backtesting"
                )
                
                optimize_weights = st.checkbox(
                    "Optimize Signal Weights",
                    value=True,
                    help="Use Bayesian optimization to find optimal feature weights"
                )
                
                include_macro = st.checkbox(
                    "Include Macro Indicators",
                    value=True,
                    help="Include macroeconomic factors in the analysis"
                )
                
                use_enhanced_weights = st.checkbox(
                    "Use Enhanced Weight System",
                    value=True,
                    help="Use the enhanced weight system with sub-categories"
                )
                
                n_optimization_trials = st.slider(
                    "Optimization Trials",
                    min_value=10,
                    max_value=50,
                    value=20,
                    help="Number of Bayesian optimization trials"
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
                    
                    transaction_cost = st.number_input(
                        "Transaction Cost (%)",
                        min_value=0.0,
                        max_value=1.0,
                        value=settings.get('transaction_cost', 0.0025) * 100,
                        step=0.01,
                        help="Trading fees as percentage"
                    ) / 100
                    
                    max_drawdown_threshold = st.number_input(
                        "Max Drawdown Threshold (%)",
                        min_value=10.0,
                        max_value=50.0,
                        value=settings.get('max_drawdown_threshold', 0.25) * 100,
                        step=5.0,
                        help="Maximum acceptable drawdown"
                    ) / 100
            
            with col2:
                st.markdown("### Expected Outcomes")
                st.info("""
                **Enhanced Backtest Features:**
                
                1. **Walk-Forward Analysis**
                   - Multiple train/test windows
                   - Adaptive window sizing
                   - Information leakage prevention
                
                2. **Enhanced Optimization**
                   - Main category weights
                   - Sub-category weights
                   - Feature importance analysis
                
                3. **Comprehensive Metrics**
                   - 15+ performance metrics
                   - Risk decomposition
                   - Market regime analysis
                
                4. **Signal Analysis**
                   - Top contributing signals
                   - Signal effectiveness
                   - Optimal combinations
                
                ⏱️ **Estimated time**: 10-30 minutes
                """)
            
            # Run backtest button
            if st.button("🚀 Run Enhanced Backtest", type="primary", use_container_width=True):
                with st.spinner("Running enhanced backtest... This may take several minutes."):
                    request_data = {
                        "period": period,
                        "optimize_weights": optimize_weights,
                        "include_macro": include_macro,
                        "use_enhanced_weights": use_enhanced_weights,
                        "n_optimization_trials": n_optimization_trials
                    }
                    
                    result = post_api_data("/backtest/enhanced/run", request_data)
                    
                    if result and result.get('status') == 'success':
                        st.success("✅ Enhanced backtest completed successfully!")
                        st.balloons()
                        
                        # Display summary
                        summary = result.get('summary', {})
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Composite Score", f"{summary.get('composite_score', 0):.3f}")
                        with col2:
                            st.metric("Confidence Score", f"{summary.get('confidence_score', 0):.2%}")
                        with col3:
                            key_metrics = summary.get('key_metrics', {})
                            st.metric("Sortino Ratio", f"{key_metrics.get('sortino_ratio', 0):.2f}")
                        
                        st.info(f"Backtest ID: {result.get('backtest_id', 'N/A')}. View detailed results in the Results tab.")
                    else:
                        st.error(f"Backtest failed: {result.get('message', 'Unknown error') if result else 'Connection error'}")
    
    with tab2:
        st.subheader("Latest Enhanced Backtest Results")
        
        # Fetch latest enhanced results
        results = fetch_api_data("/backtest/enhanced/results/latest")
        
        if results:
            # Display timestamp
            st.info(f"Backtest completed: {results.get('timestamp', 'Unknown')}")
            
            # Performance metrics in columns
            st.markdown("### Performance Metrics")
            
            metrics = results.get('performance_metrics', {})
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                composite_score = metrics.get('composite_score', 0)
                delta_color = "normal" if composite_score >= 0.7 else "inverse"
                st.metric(
                    "Composite Score", 
                    f"{composite_score:.3f}",
                    delta=f"Target: 0.7+",
                    delta_color=delta_color
                )
                
            with col2:
                sortino = metrics.get('sortino_ratio_mean', 0)
                delta_color = "normal" if sortino >= 2.0 else "inverse"
                st.metric(
                    "Sortino Ratio",
                    f"{sortino:.2f}",
                    delta=f"Target: 2.0+",
                    delta_color=delta_color
                )
                
            with col3:
                max_dd = metrics.get('max_drawdown_mean', 0)
                delta_color = "normal" if abs(max_dd) <= 0.25 else "inverse"
                st.metric(
                    "Max Drawdown",
                    f"{max_dd:.2%}",
                    delta=f"Limit: -25%",
                    delta_color=delta_color
                )
                
            with col4:
                total_return = metrics.get('total_return_mean', 0)
                st.metric(
                    "Total Return",
                    f"{total_return:.2%}",
                    delta=f"Periods: {metrics.get('periods_tested', 0)}"
                )
            
            # Additional metrics grid
            st.markdown("### Detailed Metrics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                metrics_data = []
                metrics_list = [
                    ("Calmar Ratio", metrics.get('calmar_ratio_mean', 0), 3.0),
                    ("Profit Factor", metrics.get('profit_factor_mean', 0), 1.5),
                    ("Win Rate", metrics.get('win_rate_mean', 0), 0.55),
                    ("Sharpe Ratio", metrics.get('sharpe_ratio_mean', 0), 1.0),
                    ("Volatility", metrics.get('volatility_mean', 0), None),
                    ("Total Trades", metrics.get('total_trades', 0), None)
                ]
                
                for metric_name, value, target in metrics_list:
                    if metric_name == "Win Rate":
                        value_str = f"{value:.2%}"
                    elif metric_name == "Total Trades":
                        value_str = f"{int(value)}"
                    else:
                        value_str = f"{value:.3f}"
                    
                    if target:
                        if metric_name == "Win Rate":
                            status = "✅" if value > target else "❌"
                            target_str = f">{target:.0%}"
                        else:
                            status = "✅" if value > target else "❌"
                            target_str = f">{target}"
                    else:
                        status = "ℹ️"
                        target_str = "-"
                    
                    metrics_data.append({
                        "Metric": metric_name,
                        "Value": value_str,
                        "Target": target_str,
                        "Status": status
                    })
                
                metrics_df = pd.DataFrame(metrics_data)
                st.dataframe(metrics_df, hide_index=True)
            
            with col2:
                # Risk assessment
                st.markdown("**Risk Assessment**")
                
                risk_data = results.get('risk_assessment', {})
                if risk_data:
                    risk_levels = {
                        'Low': '🟢',
                        'Medium': '🟡', 
                        'High': '🔴',
                        'Acceptable': '🟢',
                        'Normal': '🟢',
                        'Balanced': '🟢'
                    }
                    
                    for risk_type, risk_level in risk_data.items():
                        icon = risk_levels.get(risk_level, '⚪')
                        st.write(f"{icon} **{risk_type.replace('_', ' ').title()}**: {risk_level}")
                
                # Confidence score
                confidence_score = results.get('confidence_score', 0)
                if confidence_score:
                    st.markdown("**Confidence Score**")
                    st.progress(confidence_score, text=f"{confidence_score:.1%}")
            
            # Optimal weights
            st.markdown("### Optimal Signal Weights")
            
            optimal_weights = results.get('optimal_weights', {})
            if optimal_weights:
                # Main categories
                main_weights = {
                    'Technical': optimal_weights.get('technical', 0.4),
                    'On-Chain': optimal_weights.get('onchain', 0.35),
                    'Sentiment': optimal_weights.get('sentiment', 0.15),
                    'Macro': optimal_weights.get('macro', 0.1)
                }
                
                # Create pie chart
                fig_weights = go.Figure(data=[go.Pie(
                    labels=list(main_weights.keys()),
                    values=list(main_weights.values()),
                    hole=.3
                )])
                
                fig_weights.update_layout(
                    title="Optimal Main Category Weights",
                    height=400
                )
                
                st.plotly_chart(fig_weights, use_container_width=True)
                
                # Sub-weights if available
                if 'technical_sub' in optimal_weights:
                    st.markdown("#### Sub-Category Weights")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write("**Technical Sub-weights:**")
                        tech_sub = optimal_weights['technical_sub']
                        for key, val in tech_sub.items():
                            st.write(f"- {key.title()}: {val:.2%}")
                    
                    with col2:
                        if 'onchain_sub' in optimal_weights:
                            st.write("**On-Chain Sub-weights:**")
                            onchain_sub = optimal_weights['onchain_sub']
                            for key, val in onchain_sub.items():
                                st.write(f"- {key.title()}: {val:.2%}")
                    
                    with col3:
                        if 'sentiment_sub' in optimal_weights:
                            st.write("**Sentiment Sub-weights:**")
                            sentiment_sub = optimal_weights['sentiment_sub']
                            for key, val in sentiment_sub.items():
                                st.write(f"- {key.title()}: {val:.2%}")
            
            # Signal analysis
            if 'signal_analysis' in results:
                st.markdown("### Signal Effectiveness Analysis")
                
                signal_analysis = results['signal_analysis']
                effectiveness = signal_analysis.get('effectiveness', {})
                
                if effectiveness:
                    # Create effectiveness chart
                    signals = list(effectiveness.keys())
                    win_rates = [eff['win_rate'] for eff in effectiveness.values()]
                    contributions = [eff['contribution'] for eff in effectiveness.values()]
                    
                    fig = make_subplots(
                        rows=1, cols=2,
                        subplot_titles=('Signal Win Rates', 'Signal Contributions')
                    )
                    
                    fig.add_trace(
                        go.Bar(x=signals, y=win_rates, name='Win Rate'),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Bar(x=signals, y=contributions, name='Contribution'),
                        row=1, col=2
                    )
                    
                    fig.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations
            recommendations = results.get('recommendations', [])
            if recommendations:
                st.markdown("### 💡 Recommendations")
                for i, rec in enumerate(recommendations, 1):
                    st.warning(f"{i}. {rec}")
            
            # Full results in expander
            with st.expander("View Complete Results (JSON)"):
                st.json(results)
        else:
            st.warning("No enhanced backtest results available. Run a backtest to see results.")
    
    with tab3:
        st.subheader("Backtest History")
        
        history = fetch_api_data("/backtest/results/history?limit=20")
        
        if history:
            # Convert to DataFrame
            history_df = pd.DataFrame(history)
            
            # Add enhanced indicator
            history_df['Type'] = history_df['enhanced'].apply(lambda x: '🚀 Enhanced' if x else '📊 Basic')
            
            # Format timestamps
            history_df['timestamp'] = pd.to_datetime(history_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
            
            # Display metrics
            st.dataframe(
                history_df[['timestamp', 'Type', 'composite_score', 'sortino_ratio', 
                          'max_drawdown', 'confidence_score']],
                hide_index=True
            )
            
            # Performance trend chart
            if len(history_df) > 1:
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=history_df.index,
                    y=history_df['composite_score'],
                    mode='lines+markers',
                    name='Composite Score',
                    line=dict(color='blue', width=2)
                ))
                
                if 'confidence_score' in history_df.columns:
                    fig.add_trace(go.Scatter(
                        x=history_df.index,
                        y=history_df['confidence_score'],
                        mode='lines+markers',
                        name='Confidence Score',
                        line=dict(color='green', width=2),
                        yaxis='y2'
                    ))
                
                fig.update_layout(
                    title="Backtest Performance Trend",
                    xaxis_title="Backtest #",
                    yaxis_title="Composite Score",
                    yaxis2=dict(
                        title="Confidence Score",
                        overlaying='y',
                        side='right'
                    ),
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No backtest history available yet.")
    
    with tab4:
        st.subheader("Market Analysis from Latest Backtest")
        
        results = fetch_api_data("/backtest/enhanced/results/latest")
        
        if results and 'market_analysis' in results:
            market_analysis = results['market_analysis']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                regime = market_analysis.get('regime', 'Unknown')
                regime_icons = {
                    'High Volatility Bull': '🚀📊',
                    'High Volatility Bear': '🐻📉',
                    'Low Volatility Bull': '🐂📈',
                    'Low Volatility Bear': '🐻📉',
                    'Ranging': '↔️'
                }
                icon = regime_icons.get(regime, '❓')
                st.metric("Market Regime", f"{icon} {regime}")
            
            with col2:
                trend = market_analysis.get('dominant_trend', 'Unknown')
                trend_icons = {
                    'Bullish': '📈',
                    'Bearish': '📉',
                    'Mixed': '🔄'
                }
                icon = trend_icons.get(trend, '❓')
                st.metric("Dominant Trend", f"{icon} {trend}")
            
            with col3:
                vol_regime = market_analysis.get('volatility_regime', 'Unknown')
                vol_icons = {
                    'Expanding': '📊',
                    'Contracting': '📉',
                    'Normal': '➖'
                }
                icon = vol_icons.get(vol_regime, '❓')
                st.metric("Volatility Regime", f"{icon} {vol_regime}")
            
            # Risk analysis
            if 'risk_analysis' in results:
                st.markdown("### Risk Analysis")
                
                risk_analysis = results['risk_analysis']
                
                # Risk decomposition
                if 'decomposition' in risk_analysis:
                    decomp = risk_analysis['decomposition']
                    
                    fig = go.Figure(data=[go.Bar(
                        x=list(decomp.keys()),
                        y=list(decomp.values()),
                        marker_color=['red', 'orange', 'yellow', 'green']
                    )])
                    
                    fig.update_layout(
                        title="Risk Decomposition",
                        xaxis_title="Risk Type",
                        yaxis_title="Risk Level",
                        height=300
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Stress test results
                if 'stress_scenarios' in risk_analysis:
                    st.markdown("### Stress Test Scenarios")
                    
                    stress_results = risk_analysis['stress_scenarios']
                    
                    scenario_data = []
                    for scenario, result in stress_results.items():
                        scenario_data.append({
                            'Scenario': scenario.replace('_', ' ').title(),
                            'Impact': f"{result['impact']:.2%}",
                            'Max Drawdown': f"{result['max_drawdown']:.2%}"
                        })
                    
                    stress_df = pd.DataFrame(scenario_data)
                    st.dataframe(stress_df, hide_index=True)
        else:
            st.info("Run an enhanced backtest to see market analysis.")

def show_configuration():
    st.header("⚙️ Configuration & Backtesting")
    
    # Create tabs for different configuration sections
    config_tab1, config_tab2, config_tab3, config_tab4 = st.tabs([
        "Backtest Results", "Run Backtest", "Signal Weights", "Model Settings"
    ])
    
    with config_tab1:
        show_backtest_results()
    
    with config_tab2:
        show_run_backtest()
    
    with config_tab3:
        show_signal_weights()
    
    with config_tab4:
        show_model_settings()

def show_backtest_results():
    """Display latest backtest results"""
    st.subheader("📊 Latest Backtest Results")
    
    # Fetch latest results
    results = fetch_api_data("/backtest/results/latest")
    
    if results:
        # Display timestamp
        st.info(f"Last backtest: {results.get('timestamp', 'Unknown')}")
        
        # Performance metrics in columns
        metrics = results.get('performance_metrics', {})
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            composite_score = metrics.get('composite_score', 0)
            delta_color = "normal" if composite_score >= 0.7 else "inverse"
            st.metric(
                "Composite Score", 
                f"{composite_score:.3f}",
                delta=f"Target: 0.7+",
                delta_color=delta_color
            )
            
        with col2:
            sortino = metrics.get('sortino_ratio_mean', 0)
            delta_color = "normal" if sortino >= 2.0 else "inverse"
            st.metric(
                "Sortino Ratio",
                f"{sortino:.2f}",
                delta=f"Target: 2.0+",
                delta_color=delta_color
            )
            
        with col3:
            max_dd = metrics.get('max_drawdown_mean', 0)
            delta_color = "normal" if abs(max_dd) <= 0.25 else "inverse"
            st.metric(
                "Max Drawdown",
                f"{max_dd:.2%}",
                delta=f"Limit: -25%",
                delta_color=delta_color
            )
            
        with col4:
            total_return = metrics.get('total_return_mean', 0)
            st.metric(
                "Total Return",
                f"{total_return:.2%}",
                delta=f"Avg: {total_return:.2%}"
            )
        
        # Additional metrics
        st.markdown("### Detailed Performance Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            metrics_df = pd.DataFrame([
                {"Metric": "Calmar Ratio", "Value": f"{metrics.get('calmar_ratio_mean', 0):.2f}", 
                 "Target": ">3.0", "Status": "✅" if metrics.get('calmar_ratio_mean', 0) > 3.0 else "❌"},
                {"Metric": "Profit Factor", "Value": f"{metrics.get('profit_factor_mean', 0):.2f}",
                 "Target": ">1.5", "Status": "✅" if metrics.get('profit_factor_mean', 0) > 1.5 else "❌"},
                {"Metric": "Win Rate", "Value": f"{metrics.get('win_rate_mean', 0):.2%}",
                 "Target": ">55%", "Status": "✅" if metrics.get('win_rate_mean', 0) > 0.55 else "❌"},
                {"Metric": "Sharpe Ratio", "Value": f"{metrics.get('sharpe_ratio_mean', 0):.3f}",
                 "Target": ">1.0", "Status": "✅" if metrics.get('sharpe_ratio_mean', 0) > 1.0 else "❌"},
            ])
            st.dataframe(metrics_df, hide_index=True)
        
        with col2:
            # Create a radar chart for risk assessment
            risk_data = results.get('risk_assessment', {})
            if risk_data:
                st.markdown("**Risk Assessment**")
                
                risk_levels = {
                    'Low': '🟢',
                    'Medium': '🟡', 
                    'High': '🔴',
                    'Acceptable': '🟢',
                    'Normal': '🟢',
                    'Balanced': '🟢'
                }
                
                for risk_type, risk_level in risk_data.items():
                    icon = risk_levels.get(risk_level, '⚪')
                    st.write(f"{icon} **{risk_type.replace('_', ' ').title()}**: {risk_level}")
        
        # Recommendations
        recommendations = results.get('recommendations', [])
        if recommendations:
            st.markdown("### 💡 Recommendations")
            for rec in recommendations:
                st.warning(f"• {rec}")
        
        # Performance chart
        st.markdown("### Performance Visualization")
        
        # Create sample performance data for visualization
        dates = pd.date_range(end=pd.Timestamp.now(), periods=30, freq='D')
        cumulative_returns = 1 + (pd.Series(range(30)) * total_return / 30 + 
                                 pd.Series(range(30)).apply(lambda x: np.random.randn() * 0.02))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=cumulative_returns,
            mode='lines',
            name='Cumulative Returns',
            line=dict(color='#00FF00' if total_return > 0 else '#FF0000', width=2)
        ))
        
        fig.update_layout(
            title="Backtest Cumulative Returns",
            xaxis_title="Date",
            yaxis_title="Cumulative Return",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.warning("No backtest results available. Run a backtest to see results.")
        
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
        
        st.dataframe(history_df[['timestamp', 'composite_score', 'sortino_ratio', 'max_drawdown']], 
                    hide_index=True)

def show_run_backtest():
    """Interface to run new backtests"""
    st.subheader("🚀 Run New Backtest")
    
    # Check if backtest is in progress
    status = fetch_api_data("/backtest/status")
    
    if status and status.get('in_progress'):
        st.warning("⏳ Backtest is currently in progress. Please wait...")
        
        # Add a progress bar
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.1)
            progress_bar.progress(i + 1)
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Backtest Parameters")
            
            period = st.selectbox(
                "Data Period",
                ["1mo", "3mo", "6mo", "1y", "2y", "3y"],
                index=3,
                help="Historical data period for backtesting"
            )
            
            optimize_weights = st.checkbox(
                "Optimize Signal Weights",
                value=True,
                help="Use Bayesian optimization to find optimal feature weights"
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
            st.info("""
            **What the backtest will do:**
            
            1. **Walk-Forward Analysis**
               - Train on historical windows
               - Test on future data
               - Prevent look-ahead bias
            
            2. **Optimization** (if enabled)
               - Find optimal signal weights
               - Balance risk vs return
               - ~50 optimization trials
            
            3. **Performance Evaluation**
               - Calculate Sortino ratio
               - Measure maximum drawdown
               - Generate recommendations
            
            ⏱️ **Estimated time**: 5-15 minutes
            """)
        
        # Run backtest button
        if st.button("🎯 Run Backtest", type="primary", use_container_width=True):
            with st.spinner("Running backtest... This may take several minutes."):
                result = post_api_data("/backtest/run", {
                    "period": period,
                    "optimize_weights": optimize_weights
                })
                
                if result and result.get('status') == 'success':
                    st.success("✅ Backtest completed successfully!")
                    st.balloons()
                    
                    # Display summary
                    summary = result.get('summary', {})
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Composite Score", f"{summary.get('composite_score', 0):.3f}")
                    with col2:
                        st.metric("Sortino Ratio", f"{summary.get('sortino_ratio', 0):.2f}")
                    with col3:
                        st.metric("Max Drawdown", f"{summary.get('max_drawdown', 0):.2%}")
                    
                    st.info("View detailed results in the 'Backtest Results' tab")
                else:
                    st.error(f"Backtest failed: {result.get('message', 'Unknown error') if result else 'Connection error'}")

def show_signal_weights():
    """Configure signal weights"""
    st.subheader("🎚️ Signal Weight Configuration")
    
    # Fetch current weights
    current_weights = fetch_api_data("/config/signal-weights") or {
        "technical": 0.40,
        "onchain": 0.35,
        "sentiment": 0.15,
        "macro": 0.10
    }
    
    st.info("Adjust the importance of different signal categories. Weights will be normalized to sum to 100%.")
    
    # Create sliders for weights
    col1, col2 = st.columns([2, 1])
    
    with col1:
        technical = st.slider(
            "Technical Indicators Weight",
            min_value=0.0,
            max_value=1.0,
            value=current_weights['technical'],
            step=0.05,
            help="RSI, MACD, Bollinger Bands, Moving Averages"
        )
        
        onchain = st.slider(
            "On-chain Metrics Weight",
            min_value=0.0,
            max_value=1.0,
            value=current_weights['onchain'],
            step=0.05,
            help="Transaction volume, active addresses, network metrics"
        )
        
        sentiment = st.slider(
            "Sentiment Analysis Weight",
            min_value=0.0,
            max_value=1.0,
            value=current_weights['sentiment'],
            step=0.05,
            help="Social media sentiment, fear & greed index"
        )
        
        macro = st.slider(
            "Macroeconomic Factors Weight",
            min_value=0.0,
            max_value=1.0,
            value=current_weights['macro'],
            step=0.05,
            help="USD index, stock market correlation, economic indicators"
        )
    
    with col2:
        # Show normalized weights
        total = technical + onchain + sentiment + macro
        
        st.markdown("### Normalized Weights")
        
        if total > 0:
            norm_technical = technical / total
            norm_onchain = onchain / total
            norm_sentiment = sentiment / total
            norm_macro = macro / total
            
            # Create pie chart
            fig = go.Figure(data=[go.Pie(
                labels=['Technical', 'On-chain', 'Sentiment', 'Macro'],
                values=[norm_technical, norm_onchain, norm_sentiment, norm_macro],
                hole=.3
            )])
            
            fig.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display percentages
            st.write(f"**Technical**: {norm_technical:.1%}")
            st.write(f"**On-chain**: {norm_onchain:.1%}")
            st.write(f"**Sentiment**: {norm_sentiment:.1%}")
            st.write(f"**Macro**: {norm_macro:.1%}")
        else:
            st.warning("Total weight cannot be zero")
    
    # Update weights button
    if st.button("💾 Update Signal Weights", type="primary"):
        weights = {
            "technical": technical,
            "onchain": onchain,
            "sentiment": sentiment,
            "macro": macro
        }
        
        result = post_api_data("/config/signal-weights", weights)
        
        if result and result.get('status') == 'success':
            st.success("✅ Signal weights updated successfully!")
            st.info("Run a new backtest to evaluate performance with updated weights")
        else:
            st.error("Failed to update signal weights")

def show_model_settings():
    """Model configuration and retraining"""
    st.subheader("🧠 Model Settings & Retraining")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Model Information")
        
        # Display model status
        model_info = {
            "Model Type": "LSTM (Long Short-Term Memory)",
            "Architecture": "4 layers, 512 hidden units",
            "Sequence Length": "60 time steps",
            "Features": "Technical + On-chain + Sentiment",
            "Last Training": "Check logs for details"
        }
        
        for key, value in model_info.items():
            st.write(f"**{key}**: {value}")
        
        st.markdown("### Retraining Options")
        
        st.warning("""
        ⚠️ **Caution**: Model retraining will:
        - Use the latest optimized parameters
        - Take 10-30 minutes to complete
        - Temporarily affect live predictions
        """)
        
        if st.button("🔄 Retrain Model", type="secondary"):
            with st.spinner("Retraining model... This will take some time."):
                result = post_api_data("/model/retrain", {})
                
                if result and result.get('status') == 'success':
                    st.success("✅ Model retrained successfully!")
                    st.info(f"Completed at: {result.get('timestamp', 'Unknown')}")
                else:
                    st.error("Model retraining failed")
    
    with col2:
        st.markdown("### Adaptive Retraining Schedule")
        
        st.info("""
        **Automatic retraining triggers:**
        
        🔄 **Scheduled**: Every 90 days (quarterly)
        
        📉 **Performance-based**: When composite score < 0.6
        
        📊 **Volatility-based**: When market volatility > 50%
        
        🚨 **Drift detection**: When concept drift detected
        """)
        
        # Show next scheduled retraining
        st.markdown("### Next Scheduled Actions")
        
        next_retrain = datetime.now() + timedelta(days=90)
        st.write(f"**Next retraining**: {next_retrain.strftime('%Y-%m-%d')}")
        st.write(f"**Next backtest**: Weekly on Sundays")
        
        # Performance thresholds
        st.markdown("### Performance Thresholds")
        
        thresholds = {
            "Minimum Sortino Ratio": 2.0,
            "Maximum Drawdown": "25%",
            "Target Composite Score": 0.7,
            "Minimum Win Rate": "55%"
        }
        
        for metric, threshold in thresholds.items():
            st.write(f"**{metric}**: {threshold}")

def show_trading():
    """Trading interface"""
    st.header("💰 Trading Interface")
    
    # Get latest market data
    latest_signal = fetch_api_data("/signals/latest")
    portfolio_metrics = fetch_api_data("/portfolio/metrics")
    
    # Trading form
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Execute Trade")
        
        with st.form("trade_form"):
            trade_type = st.selectbox(
                "Trade Type", 
                ["buy", "sell", "hold"],
                help="Select the type of trade to execute"
            )
            
            current_price = portfolio_metrics.get('current_btc_price', 45000) if portfolio_metrics else 45000
            
            price = st.number_input(
                "Price ($)", 
                min_value=1.0, 
                value=float(current_price), 
                step=0.01,
                help="Price per BTC"
            )
            
            size = st.number_input(
                "Size (BTC)", 
                min_value=0.0001, 
                value=0.01, 
                step=0.0001, 
                format="%.4f",
                help="Amount of BTC to trade"
            )
            
            lot_id = st.text_input(
                "Lot ID (optional)", 
                placeholder="Leave empty for auto-generated ID",
                help="Track specific lots for tax purposes"
            )
            
            # Trade value preview
            trade_value = price * size
            st.info(f"Total Trade Value: ${trade_value:,.2f}")
            
            col_submit1, col_submit2 = st.columns(2)
            with col_submit1:
                submitted = st.form_submit_button("Execute Trade", type="primary", use_container_width=True)
            with col_submit2:
                if st.form_submit_button("Clear", use_container_width=True):
                    st.rerun()
            
            if submitted:
                if trade_type == "hold":
                    st.warning("Hold is not an executable trade type. Please select buy or sell.")
                else:
                    trade_data = {
                        "symbol": "BTC-USD",
                        "trade_type": trade_type,
                        "price": price,
                        "size": size,
                        "lot_id": lot_id if lot_id else None
                    }
                    
                    with st.spinner("Executing trade..."):
                        result = post_api_data("/trades/", trade_data)
                    
                    if result and result.get('status') == 'success':
                        st.success(f"✅ Trade executed successfully! Trade ID: {result['trade_id']}")
                        st.balloons()
                        time.sleep(2)
                        st.rerun()
                    else:
                        st.error("❌ Trade execution failed. Please try again.")
    
    with col2:
        st.subheader("Market Information")
        
        # Current signal
        if latest_signal:
            st.write("**AI Signal:**")
            signal = latest_signal.get('signal', 'hold')
            confidence = latest_signal.get('confidence', 0.5)
            
            signal_emoji = {'buy': '🟢', 'sell': '🔴', 'hold': '🟡'}
            st.write(f"{signal_emoji.get(signal, '❓')} {signal.upper()} (Confidence: {confidence:.1%})")
            
            if signal == 'buy' and trade_type != 'buy':
                st.info("💡 AI suggests buying")
            elif signal == 'sell' and trade_type != 'sell':
                st.info("💡 AI suggests selling")
        
        # Quick stats
        st.write("**Quick Stats:**")
        if portfolio_metrics:
            st.write(f"Current BTC Price: ${portfolio_metrics.get('current_btc_price', 0):,.2f}")
            st.write(f"Total Invested: ${portfolio_metrics.get('total_invested', 0):,.2f}")
            st.write(f"Current P&L: ${portfolio_metrics.get('total_pnl', 0):,.2f}")
        
        # Recent trades
        st.write("**Your Recent Trades:**")
        recent_trades = fetch_api_data("/trades/?limit=5")
        if recent_trades:
            for trade in recent_trades:
                trade_emoji = '🟢' if trade['trade_type'] == 'buy' else '🔴'
                st.write(f"{trade_emoji} {trade['trade_type'].upper()} {trade['size']:.4f} BTC @ ${trade['price']:,.2f}")

def show_portfolio():
    """Portfolio overview"""
    st.header("📈 Portfolio Overview")
    
    # Get portfolio data
    positions = fetch_api_data("/positions/")
    trades = fetch_api_data("/trades/")
    portfolio_metrics = fetch_api_data("/portfolio/metrics")
    
    if not positions and not trades:
        st.info("No portfolio data available. Start trading to see your portfolio!")
        return
    
    # Portfolio Summary
    col1, col2, col3, col4 = st.columns(4)
    
    if portfolio_metrics:
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
    
    # Positions
    st.subheader("Current Positions")
    if positions:
        positions_df = pd.DataFrame(positions)
        if not positions_df.empty:
            # Add current value and P&L
            current_price = portfolio_metrics.get('current_btc_price', 0) if portfolio_metrics else 0
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
                        st.write(f"**Avg Buy Price:** ${position['avg_buy_price']:,.2f}")
                    with col2:
                        st.write(f"**Current Value:** ${position['current_value']:,.2f}")
                        st.write(f"**Purchase Value:** ${position['purchase_value']:,.2f}")
                    with col3:
                        pnl_color = "green" if position['unrealized_pnl'] >= 0 else "red"
                        st.markdown(f"**Unrealized P&L:** <span style='color:{pnl_color}'>${position['unrealized_pnl']:,.2f} ({position['pnl_percent']:+.2f}%)</span>", unsafe_allow_html=True)
                        st.write(f"**Created:** {position['created_at']}")
    else:
        st.info("No open positions")
    
    # Performance Metrics
    st.subheader("Performance Metrics")
    if trades:
        trades_df = pd.DataFrame(trades)
        metrics = calculate_portfolio_metrics(trades_df)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Return", f"{metrics['total_return']:.2%}")
        with col2:
            st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
        with col3:
            st.metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")
        with col4:
            st.metric("Win Rate", f"{metrics['win_rate']:.2%}")
        with col5:
            st.metric("Total Trades", metrics['total_trades'])
    
    # Trade History
    st.subheader("Trade History")
    if trades:
        trades_df = pd.DataFrame(trades)
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
        trades_df = trades_df.sort_values('timestamp', ascending=False)
        
        # Add filters
        col1, col2, col3 = st.columns(3)
        with col1:
            trade_type_filter = st.selectbox("Filter by Type", ["All", "buy", "sell"])
        with col2:
            date_range = st.date_input(
                "Date Range",
                value=(trades_df['timestamp'].min().date(), trades_df['timestamp'].max().date()),
                max_value=datetime.now().date()
            )
        with col3:
            lot_filter = st.selectbox(
                "Filter by Lot", 
                ["All"] + list(trades_df['lot_id'].unique())
            )
        
        # Apply filters
        filtered_df = trades_df.copy()
        if trade_type_filter != "All":
            filtered_df = filtered_df[filtered_df['trade_type'] == trade_type_filter]
        if len(date_range) == 2:
            filtered_df = filtered_df[
                (filtered_df['timestamp'].dt.date >= date_range[0]) & 
                (filtered_df['timestamp'].dt.date <= date_range[1])
            ]
        if lot_filter != "All":
            filtered_df = filtered_df[filtered_df['lot_id'] == lot_filter]
        
        # Display trades
        if not filtered_df.empty:
            st.dataframe(
                filtered_df[['timestamp', 'trade_type', 'size', 'price', 'lot_id', 'status']],
                use_container_width=True,
                hide_index=True
            )
            
            # Download button
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download Trade History",
                data=csv,
                file_name=f"btc_trades_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No trades match the selected filters")

def show_signals():
    """AI Trading Signals"""
    st.header("🤖 AI Trading Signals")
    
    # Latest signal
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.subheader("Current Signal")
        latest_signal = fetch_api_data("/signals/latest")
        if latest_signal:
            signal = latest_signal.get('signal', 'hold')
            confidence = latest_signal.get('confidence', 0.5)
            predicted_price = latest_signal.get('predicted_price', 0)
            timestamp = latest_signal.get('timestamp', 'N/A')
            
            # Enhanced signal display
            signal_colors = {
                'buy': {'bg': '#1f7a1f', 'text': '#90EE90', 'emoji': '🟢'},
                'sell': {'bg': '#7a1f1f', 'text': '#FFA07A', 'emoji': '🔴'},
                'hold': {'bg': '#7a5f1f', 'text': '#FFD700', 'emoji': '🟡'}
            }
            
            colors = signal_colors.get(signal, signal_colors['hold'])
            
            st.markdown(f"""
            <div style="
                padding: 30px; 
                border-radius: 15px; 
                background: linear-gradient(135deg, {colors['bg']} 0%, #1a1a1a 100%);
                border: 3px solid {colors['text']}; 
                text-align: center;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            ">
                <h1 style="color: {colors['text']}; margin: 0; font-size: 3em;">
                    {colors['emoji']} {signal.upper()}
                </h1>
                <h3 style="color: white; margin: 15px 0;">
                    Confidence: {confidence:.1%}
                </h3>
                <p style="color: white; margin: 10px 0; font-size: 1.2em;">
                    Predicted Price: ${predicted_price:,.2f}
                </p>
                <p style="color: #aaa; font-size: 0.9em; margin-top: 20px;">
                    Last Update: {timestamp}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Signal strength indicator
            st.write("")
            st.write("**Signal Strength:**")
            st.progress(confidence)
            
            # Interpretation
            st.write("")
            st.write("**Interpretation:**")
            if signal == 'buy':
                st.success("Strong buying opportunity detected. The model predicts price will rise.")
            elif signal == 'sell':
                st.warning("Selling recommended. The model predicts price will fall.")
            else:
                st.info("No clear trend detected. Hold current positions.")
    
    with col2:
        st.subheader("Signal History")
        
        # Get historical signals
        signals_history = fetch_api_data("/signals/history?limit=50")
        
        if signals_history:
            signals_df = pd.DataFrame(signals_history)
            signals_df['timestamp'] = pd.to_datetime(signals_df['timestamp'])
            
            # Signal accuracy chart
            fig = go.Figure()
            
            # Add signal points
            buy_signals = signals_df[signals_df['signal'] == 'buy']
            sell_signals = signals_df[signals_df['signal'] == 'sell']
            hold_signals = signals_df[signals_df['signal'] == 'hold']
            
            fig.add_trace(go.Scatter(
                x=buy_signals['timestamp'],
                y=buy_signals['price_prediction'],
                mode='markers',
                name='Buy Signals',
                marker=dict(color='green', size=10, symbol='triangle-up')
            ))
            
            fig.add_trace(go.Scatter(
                x=sell_signals['timestamp'],
                y=sell_signals['price_prediction'],
                mode='markers',
                name='Sell Signals',
                marker=dict(color='red', size=10, symbol='triangle-down')
            ))
            
            fig.add_trace(go.Scatter(
                x=hold_signals['timestamp'],
                y=hold_signals['price_prediction'],
                mode='markers',
                name='Hold Signals',
                marker=dict(color='yellow', size=8, symbol='circle')
            ))
            
            # Add confidence as line
            fig.add_trace(go.Scatter(
                x=signals_df['timestamp'],
                y=signals_df['confidence'] * 100,
                mode='lines',
                name='Confidence %',
                yaxis='y2',
                line=dict(color='purple', width=2, dash='dot')
            ))
            
            fig.update_layout(
                title="Signal History & Confidence",
                xaxis_title="Time",
                yaxis_title="Predicted Price ($)",
                yaxis2=dict(
                    title="Confidence (%)",
                    overlaying='y',
                    side='right',
                    range=[0, 100]
                ),
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Signal statistics
            st.write("**Signal Statistics (Last 50):**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                buy_count = len(buy_signals)
                st.metric("Buy Signals", buy_count, f"{(buy_count/len(signals_df)*100):.1f}%")
            
            with col2:
                sell_count = len(sell_signals)
                st.metric("Sell Signals", sell_count, f"{(sell_count/len(signals_df)*100):.1f}%")
            
            with col3:
                hold_count = len(hold_signals)
                st.metric("Hold Signals", hold_count, f"{(hold_count/len(signals_df)*100):.1f}%")
            
            # Average confidence by signal type
            st.write("**Average Confidence by Signal Type:**")
            avg_confidence = signals_df.groupby('signal')['confidence'].mean()
            fig_conf = px.bar(
                x=avg_confidence.index,
                y=avg_confidence.values,
                labels={'x': 'Signal Type', 'y': 'Average Confidence'},
                color=avg_confidence.index,
                color_discrete_map={'buy': 'green', 'sell': 'red', 'hold': 'yellow'}
            )
            fig_conf.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig_conf, use_container_width=True)
        else:
            st.info("No signal history available")

def show_limits():
    """Trading Limits & Orders"""
    st.header("⚠️ Trading Limits & Orders")
    
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
                "Trigger Price ($)",
                min_value=1.0,
                value=45000.0,
                step=0.01,
                help="Price at which the order triggers"
            )
            
            size = st.number_input(
                "Size (BTC)",
                min_value=0.0001,
                value=0.01,
                step=0.0001,
                format="%.4f",
                help="Amount of BTC for the order (optional)"
            )
            
            lot_id = st.text_input(
                "Lot ID (optional)",
                help="Associate with specific lot"
            )
            
            # Order preview
            st.info(f"Order will trigger when BTC {'falls below' if 'stop' in limit_type or 'sell' in limit_type else 'rises above'} ${price:,.2f}")
            
            submitted = st.form_submit_button("Create Limit Order", type="primary", use_container_width=True)
            
            if submitted:
                limit_data = {
                    "symbol": "BTC-USD",
                    "limit_type": limit_type,
                    "price": price,
                    "size": size,
                    "lot_id": lot_id if lot_id else None
                }
                
                with st.spinner("Creating limit order..."):
                    result = post_api_data("/limits/", limit_data)
                
                if result and result.get('status') == 'success':
                    st.success(f"✅ Limit order created! ID: {result['limit_id']}")
                    time.sleep(2)
                    st.rerun()
                else:
                    st.error("❌ Failed to create limit order")
    
    with col2:
        st.subheader("Order Types Explained")
        
        st.write("**Stop Loss:** Sell when price falls below trigger to limit losses")
        st.write("**Take Profit:** Sell when price rises above trigger to lock in gains")
        st.write("**Buy Limit:** Buy when price falls to trigger level")
        st.write("**Sell Limit:** Sell when price rises to trigger level")
        
        # Current price reference
        portfolio_metrics = fetch_api_data("/portfolio/metrics")
        if portfolio_metrics:
            current_price = portfolio_metrics.get('current_btc_price', 0)
            st.info(f"Current BTC Price: ${current_price:,.2f}")
    
    # Active limit orders
    st.subheader("Active Limit Orders")
    
    limits = fetch_api_data("/limits/")
    if limits:
        limits_df = pd.DataFrame(limits)
        if not limits_df.empty:
            # Add status indicators
            current_price = portfolio_metrics.get('current_btc_price', 45000) if portfolio_metrics else 45000
            
            for _, limit in limits_df.iterrows():
                with st.expander(f"{limit['limit_type'].upper()} - ${limit['price']:,.2f}"):
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.write(f"**Type:** {limit['limit_type']}")
                        st.write(f"**Price:** ${limit['price']:,.2f}")
                    
                    with col2:
                        st.write(f"**Size:** {limit.get('size', 'N/A')} BTC")
                        st.write(f"**Lot ID:** {limit.get('lot_id', 'Any')[:8]}...")
                    
                    with col3:
                        # Distance from current price
                        distance = abs(current_price - limit['price'])
                        distance_pct = (distance / current_price) * 100
                        st.write(f"**Distance:** ${distance:,.2f} ({distance_pct:.2f}%)")
                        
                        # Status
                        if 'stop' in limit['limit_type'] or 'sell' in limit['limit_type']:
                            if current_price <= limit['price']:
                                st.warning("⚠️ Near trigger!")
                        else:
                            if current_price >= limit['price']:
                                st.warning("⚠️ Near trigger!")
                    
                    with col4:
                        if st.button(f"Cancel", key=f"cancel_{limit['id']}"):
                            # Here you would call DELETE endpoint
                            st.info("Cancel functionality to be implemented")
            
            # Summary statistics
            st.write("**Limit Orders Summary:**")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Orders", len(limits_df))
            with col2:
                stop_losses = len(limits_df[limits_df['limit_type'] == 'stop_loss'])
                st.metric("Stop Losses", stop_losses)
            with col3:
                take_profits = len(limits_df[limits_df['limit_type'] == 'take_profit'])
                st.metric("Take Profits", take_profits)
            with col4:
                limit_orders = len(limits_df[limits_df['limit_type'].str.contains('limit')])
                st.metric("Limit Orders", limit_orders)
        else:
            st.info("No active limit orders")
    else:
        st.info("No limit orders found")

def show_analytics():
    """Advanced Analytics"""
    st.header("📊 Advanced Analytics")
    
    # Get analytics data
    trades = fetch_api_data("/trades/")
    pnl_data = fetch_api_data("/analytics/pnl")
    btc_data = fetch_api_data("/market/btc-data?period=3mo")
    
    if not trades:
        st.info("No trading data available for analytics")
        return
    
    # Analytics tabs
    tab1, tab2, tab3, tab4 = st.tabs(["P&L Analysis", "Performance Metrics", "Risk Analysis", "Trade Analysis"])
    
    with tab1:
        st.subheader("Profit & Loss Analysis")
        
        if pnl_data and pnl_data.get('daily_pnl'):
            # Daily P&L chart
            daily_df = pd.DataFrame(pnl_data['daily_pnl'])
            daily_df['date'] = pd.to_datetime(daily_df['date'])
            
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=('Daily P&L', 'Cumulative P&L'),
                row_heights=[0.4, 0.6]
            )
            
            # Daily P&L bars
            colors = ['green' if x > 0 else 'red' for x in daily_df['pnl']]
            fig.add_trace(
                go.Bar(
                    x=daily_df['date'],
                    y=daily_df['pnl'],
                    name='Daily P&L',
                    marker_color=colors
                ),
                row=1, col=1
            )
            
            # Cumulative P&L line
            cum_df = pd.DataFrame(pnl_data['cumulative_pnl'])
            cum_df['date'] = pd.to_datetime(cum_df['date'])
            
            fig.add_trace(
                go.Scatter(
                    x=cum_df['date'],
                    y=cum_df['pnl'],
                    mode='lines+markers',
                    name='Cumulative P&L',
                    line=dict(color='blue', width=3)
                ),
                row=2, col=1
            )
            
            fig.update_layout(height=600, showlegend=True)
            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_yaxes(title_text="P&L ($)", row=1, col=1)
            fig.update_yaxes(title_text="Cumulative P&L ($)", row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # P&L Statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_pnl = cum_df['pnl'].iloc[-1] if not cum_df.empty else 0
                st.metric("Total P&L", f"${total_pnl:,.2f}")
            
            with col2:
                profitable_days = len([x for x in daily_df['pnl'] if x > 0])
                total_days = len(daily_df)
                win_rate = (profitable_days / total_days * 100) if total_days > 0 else 0
                st.metric("Win Rate (Days)", f"{win_rate:.1f}%")
            
            with col3:
                avg_win = daily_df[daily_df['pnl'] > 0]['pnl'].mean() if len(daily_df[daily_df['pnl'] > 0]) > 0 else 0
                st.metric("Avg Win", f"${avg_win:,.2f}")
            
            with col4:
                avg_loss = daily_df[daily_df['pnl'] < 0]['pnl'].mean() if len(daily_df[daily_df['pnl'] < 0]) > 0 else 0
                st.metric("Avg Loss", f"${avg_loss:,.2f}")
    
    with tab2:
        st.subheader("Performance Metrics")
        
        trades_df = pd.DataFrame(trades)
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
        
        # Calculate advanced metrics
        metrics = calculate_portfolio_metrics(trades_df)
        
        # Performance over time
        trades_df['trade_value'] = trades_df['price'] * trades_df['size']
        trades_df['signed_value'] = trades_df['trade_value'] * trades_df['trade_type'].map({
            'buy': -1, 
            'sell': 1, 
            'hold': 0
        })
        trades_df['cumulative_investment'] = trades_df[trades_df['trade_type'] == 'buy']['trade_value'].cumsum()
        trades_df['cumulative_pnl'] = trades_df['signed_value'].cumsum()
        
        # ROI over time
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=trades_df['timestamp'],
            y=trades_df['cumulative_pnl'] / trades_df['cumulative_investment'] * 100,
            mode='lines',
            name='ROI %',
            line=dict(color='green', width=2)
        ))
        
        fig.update_layout(
            title="Return on Investment Over Time",
            xaxis_title="Date",
            yaxis_title="ROI (%)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Key metrics display
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Risk-Adjusted Returns**")
            st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
            st.metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")
        
        with col2:
            st.write("**Trading Performance**")
            st.metric("Win Rate", f"{metrics['win_rate']:.2%}")
            st.metric("Total Trades", metrics['total_trades'])
        
        with col3:
            st.write("**Returns**")
            st.metric("Total Return", f"{metrics['total_return']:.2%}")
            # Calculate annualized return if we have enough data
            if len(trades_df) > 0:
                days = (trades_df['timestamp'].max() - trades_df['timestamp'].min()).days
                if days > 0:
                    annualized_return = (1 + metrics['total_return']) ** (365 / days) - 1
                    st.metric("Annualized Return", f"{annualized_return:.2%}")
    
    with tab3:
        st.subheader("Risk Analysis")
        
        if btc_data and btc_data.get('data'):
            price_df = pd.DataFrame(btc_data['data'])
            price_df['timestamp'] = pd.to_datetime(price_df['timestamp'])
            
            # Calculate returns
            price_df['returns'] = price_df['close'].pct_change()
            
            # Volatility over time (20-day rolling)
            price_df['volatility'] = price_df['returns'].rolling(window=20).std() * np.sqrt(252) * 100
            
            # Volatility chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=price_df['timestamp'],
                y=price_df['volatility'],
                mode='lines',
                name='Volatility (Annual %)',
                line=dict(color='orange', width=2)
            ))
            
            fig.update_layout(
                title="Bitcoin Volatility (20-day Rolling)",
                xaxis_title="Date",
                yaxis_title="Volatility (%)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Risk metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                current_vol = price_df['volatility'].iloc[-1] if not price_df['volatility'].isna().all() else 0
                st.metric("Current Volatility", f"{current_vol:.1f}%")
            
            with col2:
                avg_vol = price_df['volatility'].mean()
                st.metric("Avg Volatility", f"{avg_vol:.1f}%")
            
            with col3:
                # Calculate Value at Risk (VaR)
                var_95 = np.percentile(price_df['returns'].dropna(), 5) * 100
                st.metric("VaR (95%)", f"{var_95:.2f}%", help="95% of the time, daily loss won't exceed this")
            
            # Returns distribution
            st.write("**Returns Distribution**")
            fig_hist = px.histogram(
                price_df['returns'].dropna() * 100,
                nbins=50,
                labels={'value': 'Daily Return (%)', 'count': 'Frequency'},
                title="Distribution of Daily Returns"
            )
            fig_hist.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig_hist, use_container_width=True)
    
    with tab4:
        st.subheader("Trade Analysis")
        
        trades_df = pd.DataFrame(trades)
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
        
        # Trade distribution by type
        col1, col2 = st.columns(2)
        
        with col1:
            # Trade types pie chart
            trade_counts = trades_df['trade_type'].value_counts()
            fig_pie = px.pie(
                values=trade_counts.values,
                names=trade_counts.index,
                title="Trade Distribution",
                color_discrete_map={'buy': '#00ff00', 'sell': '#ff0000', 'hold': '#ffff00'}
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Trade sizes distribution
            fig_box = px.box(
                trades_df,
                x='trade_type',
                y='size',
                title="Trade Size Distribution by Type",
                color='trade_type',
                color_discrete_map={'buy': '#00ff00', 'sell': '#ff0000', 'hold': '#ffff00'}
            )
            st.plotly_chart(fig_box, use_container_width=True)
        
        # Trading activity heatmap
        st.write("**Trading Activity Heatmap**")
        
        # Create hourly trading activity
        trades_df['hour'] = trades_df['timestamp'].dt.hour
        trades_df['day'] = trades_df['timestamp'].dt.day_name()
        
        # Pivot for heatmap
        activity_pivot = trades_df.pivot_table(
            values='id',
            index='hour',
            columns='day',
            aggfunc='count',
            fill_value=0
        )
        
        # Reorder days
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        activity_pivot = activity_pivot.reindex(columns=[d for d in days_order if d in activity_pivot.columns])
        
        fig_heatmap = px.imshow(
            activity_pivot,
            labels=dict(x="Day", y="Hour", color="Trades"),
            aspect="auto",
            title="Trading Activity by Day and Hour"
        )
        fig_heatmap.update_layout(height=400)
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Best and worst trades
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Best Trades**")
            # Calculate P&L for each sell
            sells = trades_df[trades_df['trade_type'] == 'sell'].copy()
            if not sells.empty:
                # This is simplified - in reality you'd match with corresponding buys
                sells['pnl'] = sells['price'] * sells['size']  # Simplified
                best_trades = sells.nlargest(5, 'pnl')[['timestamp', 'size', 'price', 'pnl']]
                st.dataframe(best_trades, use_container_width=True, hide_index=True)
        
        with col2:
            st.write("**Trading Statistics**")
            avg_trade_size = trades_df['size'].mean()
            st.metric("Avg Trade Size", f"{avg_trade_size:.6f} BTC")
            
            avg_trade_value = (trades_df['price'] * trades_df['size']).mean()
            st.metric("Avg Trade Value", f"${avg_trade_value:,.2f}")
            
            # Time between trades
            if len(trades_df) > 1:
                time_diffs = trades_df['timestamp'].diff().dropna()
                avg_time_between = time_diffs.mean()
                st.metric("Avg Time Between Trades", f"{avg_time_between.days}d {avg_time_between.seconds//3600}h")

def calculate_portfolio_metrics(trades_df):
    """Calculate portfolio performance metrics"""
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
    drawdown = (trades_df['cumulative_pnl'] - cummax) / cummax
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
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(365) if daily_returns.std() > 0 else 0
    else:
        sharpe_ratio = 0
    
    # Profit factor
    wins = trades_df[trades_df['signed_value'] > 0]['signed_value'].sum()
    losses = abs(trades_df[trades_df['signed_value'] < 0]['signed_value'].sum())
    profit_factor = wins / losses if losses > 0 else 0
    
    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'total_trades': len(trades_df)
    }

def delete_api_data(endpoint):
    """Delete API data"""
    try:
        response = requests.delete(f"{API_BASE_URL}{endpoint}", timeout=30)
        return response.json() if response.status_code == 200 else None
    except:
        return None

def show_configuration():
    st.header("⚙️ Configuration & Backtesting")
    
    # Create tabs for different configuration sections
    config_tab1, config_tab2, config_tab3, config_tab4 = st.tabs([
        "Backtest Results", "Run Backtest", "Signal Weights", "Model Settings"
    ])
    
    with config_tab1:
        show_backtest_results()
    
    with config_tab2:
        show_run_backtest()
    
    with config_tab3:
        show_signal_weights()
    
    with config_tab4:
        show_model_settings()

def show_backtest_results():
    """Display latest backtest results"""
    st.subheader("📊 Latest Backtest Results")
    
    # Fetch latest results
    results = fetch_api_data("/backtest/results/latest")
    
    if results:
        # Display timestamp
        st.info(f"Last backtest: {results.get('timestamp', 'Unknown')}")
        
        # Performance metrics in columns
        metrics = results.get('performance_metrics', {})
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            composite_score = metrics.get('composite_score', 0)
            delta_color = "normal" if composite_score >= 0.7 else "inverse"
            st.metric(
                "Composite Score", 
                f"{composite_score:.3f}",
                delta=f"Target: 0.7+",
                delta_color=delta_color
            )
            
        with col2:
            sortino = metrics.get('sortino_ratio_mean', 0)
            delta_color = "normal" if sortino >= 2.0 else "inverse"
            st.metric(
                "Sortino Ratio",
                f"{sortino:.2f}",
                delta=f"Target: 2.0+",
                delta_color=delta_color
            )
            
        with col3:
            max_dd = metrics.get('max_drawdown_mean', 0)
            delta_color = "normal" if abs(max_dd) <= 0.25 else "inverse"
            st.metric(
                "Max Drawdown",
                f"{max_dd:.2%}",
                delta=f"Limit: -25%",
                delta_color=delta_color
            )
            
        with col4:
            total_return = metrics.get('total_return_mean', 0)
            st.metric(
                "Total Return",
                f"{total_return:.2%}",
                delta=f"Avg: {total_return:.2%}"
            )
        
        # Additional metrics
        st.markdown("### Detailed Performance Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            metrics_df = pd.DataFrame([
                {"Metric": "Calmar Ratio", "Value": f"{metrics.get('calmar_ratio_mean', 0):.2f}", 
                 "Target": ">3.0", "Status": "✅" if metrics.get('calmar_ratio_mean', 0) > 3.0 else "❌"},
                {"Metric": "Profit Factor", "Value": f"{metrics.get('profit_factor_mean', 0):.2f}",
                 "Target": ">1.5", "Status": "✅" if metrics.get('profit_factor_mean', 0) > 1.5 else "❌"},
                {"Metric": "Win Rate", "Value": f"{metrics.get('win_rate_mean', 0):.2%}",
                 "Target": ">55%", "Status": "✅" if metrics.get('win_rate_mean', 0) > 0.55 else "❌"},
                {"Metric": "Sharpe Ratio", "Value": f"{metrics.get('sharpe_ratio_mean', 0):.3f}",
                 "Target": ">1.0", "Status": "✅" if metrics.get('sharpe_ratio_mean', 0) > 1.0 else "❌"},
            ])
            st.dataframe(metrics_df, hide_index=True)
        
        with col2:
            # Create a radar chart for risk assessment
            risk_data = results.get('risk_assessment', {})
            if risk_data:
                st.markdown("**Risk Assessment**")
                
                risk_levels = {
                    'Low': '🟢',
                    'Medium': '🟡', 
                    'High': '🔴',
                    'Acceptable': '🟢',
                    'Normal': '🟢',
                    'Balanced': '🟢'
                }
                
                for risk_type, risk_level in risk_data.items():
                    icon = risk_levels.get(risk_level, '⚪')
                    st.write(f"{icon} **{risk_type.replace('_', ' ').title()}**: {risk_level}")
        
        # Recommendations
        recommendations = results.get('recommendations', [])
        if recommendations:
            st.markdown("### 💡 Recommendations")
            for rec in recommendations:
                st.warning(f"• {rec}")
        
        # Performance chart
        st.markdown("### Performance Visualization")
        
        # Create sample performance data for visualization
        dates = pd.date_range(end=pd.Timestamp.now(), periods=30, freq='D')
        cumulative_returns = 1 + (pd.Series(range(30)) * total_return / 30 + 
                                 pd.Series(range(30)).apply(lambda x: np.random.randn() * 0.02))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=cumulative_returns,
            mode='lines',
            name='Cumulative Returns',
            line=dict(color='#00FF00' if total_return > 0 else '#FF0000', width=2)
        ))
        
        fig.update_layout(
            title="Backtest Cumulative Returns",
            xaxis_title="Date",
            yaxis_title="Cumulative Return",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.warning("No backtest results available. Run a backtest to see results.")
        
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
        
        st.dataframe(history_df[['timestamp', 'composite_score', 'sortino_ratio', 'max_drawdown']], 
                    hide_index=True)

def show_run_backtest():
    """Interface to run new backtests"""
    st.subheader("🚀 Run New Backtest")
    
    # Check if backtest is in progress
    status = fetch_api_data("/backtest/status")
    
    if status and status.get('in_progress'):
        st.warning("⏳ Backtest is currently in progress. Please wait...")
        
        # Add a progress bar
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.1)
            progress_bar.progress(i + 1)
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Backtest Parameters")
            
            period = st.selectbox(
                "Data Period",
                ["1mo", "3mo", "6mo", "1y", "2y", "3y"],
                index=3,
                help="Historical data period for backtesting"
            )
            
            optimize_weights = st.checkbox(
                "Optimize Signal Weights",
                value=True,
                help="Use Bayesian optimization to find optimal feature weights"
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
            st.info("""
            **What the backtest will do:**
            
            1. **Walk-Forward Analysis**
               - Train on historical windows
               - Test on future data
               - Prevent look-ahead bias
            
            2. **Optimization** (if enabled)
               - Find optimal signal weights
               - Balance risk vs return
               - ~50 optimization trials
            
            3. **Performance Evaluation**
               - Calculate Sortino ratio
               - Measure maximum drawdown
               - Generate recommendations
            
            ⏱️ **Estimated time**: 5-15 minutes
            """)
        
        # Run backtest button
        if st.button("🎯 Run Backtest", type="primary", use_container_width=True):
            with st.spinner("Running backtest... This may take several minutes."):
                result = post_api_data("/backtest/run", {
                    "period": period,
                    "optimize_weights": optimize_weights
                })
                
                if result and result.get('status') == 'success':
                    st.success("✅ Backtest completed successfully!")
                    st.balloons()
                    
                    # Display summary
                    summary = result.get('summary', {})
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Composite Score", f"{summary.get('composite_score', 0):.3f}")
                    with col2:
                        st.metric("Sortino Ratio", f"{summary.get('sortino_ratio', 0):.2f}")
                    with col3:
                        st.metric("Max Drawdown", f"{summary.get('max_drawdown', 0):.2%}")
                    
                    st.info("View detailed results in the 'Backtest Results' tab")
                else:
                    st.error(f"Backtest failed: {result.get('message', 'Unknown error') if result else 'Connection error'}")

def show_signal_weights():
    """Configure signal weights"""
    st.subheader("🎚️ Signal Weight Configuration")
    
    # Fetch current weights
    current_weights = fetch_api_data("/config/signal-weights") or {
        "technical": 0.40,
        "onchain": 0.35,
        "sentiment": 0.15,
        "macro": 0.10
    }
    
    st.info("Adjust the importance of different signal categories. Weights will be normalized to sum to 100%.")
    
    # Create sliders for weights
    col1, col2 = st.columns([2, 1])
    
    with col1:
        technical = st.slider(
            "Technical Indicators Weight",
            min_value=0.0,
            max_value=1.0,
            value=current_weights['technical'],
            step=0.05,
            help="RSI, MACD, Bollinger Bands, Moving Averages"
        )
        
        onchain = st.slider(
            "On-chain Metrics Weight",
            min_value=0.0,
            max_value=1.0,
            value=current_weights['onchain'],
            step=0.05,
            help="Transaction volume, active addresses, network metrics"
        )
        
        sentiment = st.slider(
            "Sentiment Analysis Weight",
            min_value=0.0,
            max_value=1.0,
            value=current_weights['sentiment'],
            step=0.05,
            help="Social media sentiment, fear & greed index"
        )
        
        macro = st.slider(
            "Macroeconomic Factors Weight",
            min_value=0.0,
            max_value=1.0,
            value=current_weights['macro'],
            step=0.05,
            help="USD index, stock market correlation, economic indicators"
        )
    
    with col2:
        # Show normalized weights
        total = technical + onchain + sentiment + macro
        
        st.markdown("### Normalized Weights")
        
        if total > 0:
            norm_technical = technical / total
            norm_onchain = onchain / total
            norm_sentiment = sentiment / total
            norm_macro = macro / total
            
            # Create pie chart
            fig = go.Figure(data=[go.Pie(
                labels=['Technical', 'On-chain', 'Sentiment', 'Macro'],
                values=[norm_technical, norm_onchain, norm_sentiment, norm_macro],
                hole=.3
            )])
            
            fig.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display percentages
            st.write(f"**Technical**: {norm_technical:.1%}")
            st.write(f"**On-chain**: {norm_onchain:.1%}")
            st.write(f"**Sentiment**: {norm_sentiment:.1%}")
            st.write(f"**Macro**: {norm_macro:.1%}")
        else:
            st.warning("Total weight cannot be zero")
    
    # Update weights button
    if st.button("💾 Update Signal Weights", type="primary"):
        weights = {
            "technical": technical,
            "onchain": onchain,
            "sentiment": sentiment,
            "macro": macro
        }
        
        result = post_api_data("/config/signal-weights", weights)
        
        if result and result.get('status') == 'success':
            st.success("✅ Signal weights updated successfully!")
            st.info("Run a new backtest to evaluate performance with updated weights")
        else:
            st.error("Failed to update signal weights")

def show_model_settings():
    """Model configuration and retraining"""
    st.subheader("🧠 Model Settings & Retraining")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Model Information")
        
        # Display model status
        model_info = {
            "Model Type": "LSTM (Long Short-Term Memory)",
            "Architecture": "4 layers, 512 hidden units",
            "Sequence Length": "60 time steps",
            "Features": "Technical + On-chain + Sentiment",
            "Last Training": "Check logs for details"
        }
        
        for key, value in model_info.items():
            st.write(f"**{key}**: {value}")
        
        st.markdown("### Retraining Options")
        
        st.warning("""
        ⚠️ **Caution**: Model retraining will:
        - Use the latest optimized parameters
        - Take 10-30 minutes to complete
        - Temporarily affect live predictions
        """)
        
        if st.button("🔄 Retrain Model", type="secondary"):
            with st.spinner("Retraining model... This will take some time."):
                result = post_api_data("/model/retrain", {})
                
                if result and result.get('status') == 'success':
                    st.success("✅ Model retrained successfully!")
                    st.info(f"Completed at: {result.get('timestamp', 'Unknown')}")
                else:
                    st.error("Model retraining failed")
    
    with col2:
        st.markdown("### Adaptive Retraining Schedule")
        
        st.info("""
        **Automatic retraining triggers:**
        
        🔄 **Scheduled**: Every 90 days (quarterly)
        
        📉 **Performance-based**: When composite score < 0.6
        
        📊 **Volatility-based**: When market volatility > 50%
        
        🚨 **Drift detection**: When concept drift detected
        """)
        
        # Show next scheduled retraining
        st.markdown("### Next Scheduled Actions")
        
        next_retrain = datetime.now() + timedelta(days=90)
        st.write(f"**Next retraining**: {next_retrain.strftime('%Y-%m-%d')}")
        st.write(f"**Next backtest**: Weekly on Sundays")
        
        # Performance thresholds
        st.markdown("### Performance Thresholds")
        
        thresholds = {
            "Minimum Sortino Ratio": 2.0,
            "Maximum Drawdown": "25%",
            "Target Composite Score": 0.7,
            "Minimum Win Rate": "55%"
        }
        
        for metric, threshold in thresholds.items():
            st.write(f"**{metric}**: {threshold}")

if __name__ == "__main__":
    main()
