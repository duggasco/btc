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
    page_icon="?",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Use backend service name when running in Docker, fallback to localhost
API_BASE_URL = os.getenv("API_BASE_URL", "http://backend:8080")

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

def create_candlestick_chart(btc_data):
    if not btc_data or 'data' not in btc_data:
        return None
    
    df = pd.DataFrame(btc_data['data'])
    if df.empty:
        return None
        
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
    
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=('BTC-USD Price', 'Volume', 'RSI', 'MACD'),
        row_heights=[0.5, 0.15, 0.15, 0.2]
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
    
    # RSI
    if 'rsi' in df.columns and df['rsi'].notna().any():
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['rsi'],
                name='RSI',
                line=dict(color='purple', width=1)
            ),
            row=3, col=1
        )
        
        # Add RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    # MACD
    if 'macd' in df.columns and df['macd'].notna().any():
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['macd'],
                name='MACD',
                line=dict(color='blue', width=1)
            ),
            row=4, col=1
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
                row=4, col=1
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
    
    fig.update_xaxes(title_text="Date", row=4, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1)
    fig.update_yaxes(title_text="MACD", row=4, col=1)
    
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
    st.title("? BTC Trading System")
    st.markdown("---")
    
    with st.sidebar:
        st.header("Navigation")
        page = st.selectbox(
            "Select Page",
            ["Dashboard", "Trading", "Portfolio", "Signals", "Limits", "Analytics"]
        )
        
        st.markdown("---")
        
        # API Status
        with st.spinner("Checking API connection..."):
            api_status = fetch_api_data("/health")
        
        if api_status:
            st.success("? API Connected")
            if 'components' in api_status:
                with st.expander("System Status"):
                    st.json(api_status['components'])
        else:
            st.error("? API Disconnected")
        
        # Auto-refresh toggle
        auto_refresh = st.checkbox("Auto-refresh (60s)", value=True)
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
    elif page == "Limits":
        show_limits()
    elif page == "Analytics":
        show_analytics()
    
    # Auto-refresh logic
    if auto_refresh:
        time.sleep(60)
        st.rerun()

def show_dashboard():
    st.header("?? Trading Dashboard")
    
    # Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with st.spinner("Loading dashboard data..."):
        portfolio_metrics = fetch_api_data("/portfolio/metrics")
        latest_signal = fetch_api_data("/signals/latest")
        btc_data = fetch_api_data("/market/btc-data?period=1mo")
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
            fig = create_candlestick_chart(btc_data)
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
            
            # Signal card with color coding
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
            
            # Signal explanation
            if signal == 'buy':
                st.success("?? Model suggests buying - price expected to rise")
            elif signal == 'sell':
                st.error("?? Model suggests selling - price expected to fall")
            else:
                st.info("?? Model suggests holding - no clear trend")
        
        # Recent trades summary
        st.subheader("Recent Activity")
        if recent_trades:
            trades_df = pd.DataFrame(recent_trades)
            if not trades_df.empty:
                buy_count = len(trades_df[trades_df['trade_type'] == 'buy'])
                sell_count = len(trades_df[trades_df['trade_type'] == 'sell'])
                
                st.write(f"**Last 10 trades:**")
                st.write(f"?? Buys: {buy_count}")
                st.write(f"?? Sells: {sell_count}")
                
                # Mini chart of trade distribution
                fig_pie = px.pie(
                    values=[buy_count, sell_count],
                    names=['Buy', 'Sell'],
                    color_discrete_map={'Buy': '#00ff00', 'Sell': '#ff0000'}
                )
                fig_pie.update_layout(height=200, showlegend=False)
                st.plotly_chart(fig_pie, use_container_width=True)

def show_trading():
    st.header("?? Trading Interface")
    
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
                        st.success(f"? Trade executed successfully! Trade ID: {result['trade_id']}")
                        st.balloons()
                        time.sleep(2)
                        st.rerun()
                    else:
                        st.error("? Trade execution failed. Please try again.")
    
    with col2:
        st.subheader("Market Information")
        
        # Current signal
        if latest_signal:
            st.write("**AI Signal:**")
            signal = latest_signal.get('signal', 'hold')
            confidence = latest_signal.get('confidence', 0.5)
            
            signal_emoji = {'buy': '??', 'sell': '??', 'hold': '??'}
            st.write(f"{signal_emoji.get(signal, '??')} {signal.upper()} (Confidence: {confidence:.1%})")
            
            if signal == 'buy' and trade_type != 'buy':
                st.info("?? AI suggests buying")
            elif signal == 'sell' and trade_type != 'sell':
                st.info("?? AI suggests selling")
        
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
                trade_emoji = '??' if trade['trade_type'] == 'buy' else '??'
                st.write(f"{trade_emoji} {trade['trade_type'].upper()} {trade['size']:.4f} BTC @ ${trade['price']:,.2f}")

def show_portfolio():
    st.header("?? Portfolio Overview")
    
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
    st.header("?? AI Trading Signals")
    
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
                'buy': {'bg': '#1f7a1f', 'text': '#90EE90', 'emoji': '??'},
                'sell': {'bg': '#7a1f1f', 'text': '#FFA07A', 'emoji': '??'},
                'hold': {'bg': '#7a5f1f', 'text': '#FFD700', 'emoji': '??'}
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
    st.subheader("?? Comprehensive Trading Signals")
    
    # Fetch BTC data for calculations
    btc_data = fetch_api_data("/market/btc-data?period=1mo")
    
    if btc_data and 'data' in btc_data:
        df = pd.DataFrame(btc_data['data'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
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
            st.success("?? **Overall Market Sentiment: BULLISH** - Multiple indicators suggest upward momentum")
        elif bearish_count > bullish_count + 2:
            st.error("?? **Overall Market Sentiment: BEARISH** - Multiple indicators suggest downward pressure")
        else:
            st.info("?? **Overall Market Sentiment: NEUTRAL** - Mixed signals, exercise caution")
    else:
        st.warning("Unable to load BTC data for signal calculations")

def show_limits():
    st.header("? Trading Limits & Orders")
    
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
                    st.success(f"? Limit order created! ID: {result['limit_id']}")
                    time.sleep(2)
                    st.rerun()
                else:
                    st.error("? Failed to create limit order")
    
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
                                st.warning("?? Near trigger!")
                        else:
                            if current_price >= limit['price']:
                                st.warning("?? Near trigger!")
                    
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
    st.header("?? Advanced Analytics")
    
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

if __name__ == "__main__":
    main()