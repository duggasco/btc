import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime, timedelta
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components.api_client import APIClient
from components.charts import create_portfolio_chart
from utils.helpers import format_currency, format_percentage
from utils.constants import CHART_COLORS

st.set_page_config(page_title="Paper Trading", page_icon="Paper", layout="wide")

# Custom CSS for paper trading
st.markdown("""
<style>
.paper-trading-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 20px;
    border-radius: 10px;
    color: white;
    margin-bottom: 20px;
}
.position-card {
    background: #f8f9fa;
    padding: 15px;
    border-radius: 8px;
    margin: 10px 0;
    border-left: 4px solid #667eea;
}
.profit-positive { color: #00ff88; font-weight: bold; }
.profit-negative { color: #ff3366; font-weight: bold; }
.order-form {
    background: #ffffff;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}
.metric-card {
    background: #f0f2f6;
    padding: 15px;
    border-radius: 8px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# Initialize API client
@st.cache_resource
def get_api_client():
    return APIClient(os.getenv("API_BASE_URL", "http://backend:8000"))

api_client = get_api_client()

def format_pnl(value: float, percentage: float = None) -> str:
    """Format P&L with color coding"""
    color_class = "profit-positive" if value >= 0 else "profit-negative"
    pnl_str = f"${value:,.2f}"
    if percentage is not None:
        pnl_str += f" ({percentage:+.2f}%)"
    return f'<span class="{color_class}">{pnl_str}</span>'

def create_equity_curve(history: list) -> go.Figure:
    """Create portfolio equity curve chart"""
    if not history:
        return go.Figure().add_annotation(text="No trading history yet", 
                                        xref="paper", yref="paper", 
                                        x=0.5, y=0.5, showarrow=False)
    
    df = pd.DataFrame(history)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    fig = go.Figure()
    
    # Portfolio value line
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['portfolio_value'],
        mode='lines',
        name='Portfolio Value',
        line=dict(color=CHART_COLORS['primary'], width=3),
        fill='tonexty',
        fillcolor='rgba(102, 126, 234, 0.1)'
    ))
    
    # Starting balance reference line
    fig.add_hline(y=10000, line_dash="dash", line_color="gray",
                  annotation_text="Starting Balance: $10,000")
    
    # Add markers for trades
    trades_df = df[df['action'].isin(['buy', 'sell'])]
    for _, trade in trades_df.iterrows():
        color = 'green' if trade['action'] == 'buy' else 'red'
        fig.add_vline(x=trade['timestamp'], line_dash="dot", 
                      line_color=color, opacity=0.5)
    
    fig.update_layout(
        title="Portfolio Equity Curve",
        xaxis_title="Time",
        yaxis_title="Portfolio Value ($)",
        template="plotly_white",
        height=400,
        hovermode='x unified'
    )
    
    return fig

def create_trade_distribution(trades: list) -> go.Figure:
    """Create trade P&L distribution chart"""
    if not trades:
        return go.Figure().add_annotation(text="No completed trades yet", 
                                        xref="paper", yref="paper", 
                                        x=0.5, y=0.5, showarrow=False)
    
    df = pd.DataFrame(trades)
    df['pnl'] = df['exit_price'] - df['entry_price']
    df['pnl_pct'] = (df['pnl'] / df['entry_price']) * 100
    
    # Create histogram
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=df['pnl_pct'],
        nbinsx=20,
        name='Trade P&L Distribution',
        marker_color=CHART_COLORS['primary'],
        opacity=0.8
    ))
    
    # Add average line
    avg_pnl = df['pnl_pct'].mean()
    fig.add_vline(x=avg_pnl, line_dash="dash", line_color="red",
                  annotation_text=f"Avg: {avg_pnl:.2f}%")
    
    fig.update_layout(
        title="Trade P&L Distribution",
        xaxis_title="P&L %",
        yaxis_title="Number of Trades",
        template="plotly_white",
        height=400
    )
    
    return fig

def show_paper_trading():
    """Main paper trading interface"""
    
    st.title("Paper Trading Simulator")
    
    # Fetch current status and data
    try:
        pt_status = api_client.get_paper_trading_status() or {}
        current_price_data = api_client.get_current_price() or {}
        latest_signal = api_client.get_latest_signal() or {}
        
        # Try enhanced LSTM first, fallback to regular
        if latest_signal.get('source') != 'enhanced_lstm':
            enhanced_signal = api_client.get("/signals/enhanced/latest")
            if enhanced_signal and enhanced_signal.get('source') == 'enhanced_lstm':
                latest_signal = enhanced_signal
                
    except Exception as e:
        st.error(f"Error connecting to backend: {str(e)}")
        st.info("Please ensure the backend is running")
        return
    
    # Extract data
    is_enabled = pt_status.get('enabled', False)
    portfolio = pt_status.get('portfolio', {})
    positions = pt_status.get('positions', [])
    performance = pt_status.get('performance', {})
    trade_history = pt_status.get('trades', [])
    current_price = current_price_data.get('price', 0)
    
    # Paper Trading Header
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        status_text = "Active" if is_enabled else "Inactive"
        st.markdown(f"""
        <div class="paper-trading-header">
            <h2>Paper Trading Status: {status_text}</h2>
            <p>Practice trading with virtual funds - No real money at risk</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if st.button("Toggle Paper Trading", use_container_width=True,
                     disabled=st.session_state.get('confirming_toggle', False)):
            st.session_state.confirming_toggle = True
            
    with col3:
        if st.button("Reset Portfolio", use_container_width=True,
                     disabled=st.session_state.get('confirming_reset', False)):
            st.session_state.confirming_reset = True
    
    # Confirmation dialogs
    if st.session_state.get('confirming_toggle', False):
        with st.container():
            st.warning(f"{'Disable' if is_enabled else 'Enable'} paper trading?")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Confirm", key="confirm_toggle"):
                    result = api_client.toggle_paper_trading()
                    if result:
                        st.success(f"Paper trading {'disabled' if is_enabled else 'enabled'}")
                        st.session_state.confirming_toggle = False
                        st.rerun()
            with col2:
                if st.button("Cancel", key="cancel_toggle"):
                    st.session_state.confirming_toggle = False
                    st.rerun()
    
    if st.session_state.get('confirming_reset', False):
        with st.container():
            st.warning("Reset portfolio to $10,000? All positions and history will be cleared!")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Confirm Reset", key="confirm_reset"):
                    result = api_client.post("/paper-trading/reset", {})
                    if result:
                        st.success("Portfolio reset to $10,000")
                        st.session_state.confirming_reset = False
                        st.rerun()
            with col2:
                if st.button("Cancel", key="cancel_reset"):
                    st.session_state.confirming_reset = False
                    st.rerun()
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview",
        "Trading",
        "Positions",
        "History",
        "Analytics"
    ])
    
    with tab1:
        # Portfolio Overview
        st.subheader("Portfolio Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            balance = portfolio.get('balance', 10000)
            st.metric("Cash Balance", f"${balance:,.2f}")
            
        with col2:
            btc_holdings = portfolio.get('btc_holdings', 0)
            btc_value = btc_holdings * current_price if current_price else 0
            st.metric("BTC Holdings", f"{btc_holdings:.6f}", f"${btc_value:,.2f}")
            
        with col3:
            total_value = balance + btc_value
            initial_balance = 10000
            total_pnl = total_value - initial_balance
            pnl_pct = (total_pnl / initial_balance) * 100
            st.metric("Total Portfolio", f"${total_value:,.2f}", 
                     f"{total_pnl:+,.2f} ({pnl_pct:+.2f}%)")
            
        with col4:
            win_rate = performance.get('win_rate', 0)
            color = "green" if win_rate >= 50 else "red"
            st.metric("Win Rate", f"{win_rate:.1f}%")
        
        # Performance metrics in expandable section
        with st.expander("Detailed Performance Metrics", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Total Trades", performance.get('total_trades', 0))
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                sharpe = performance.get('sharpe_ratio', 0)
                st.metric("Sharpe Ratio", f"{sharpe:.2f}")
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                max_dd = performance.get('max_drawdown', 0)
                st.metric("Max Drawdown", f"{max_dd:.2f}%")
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col4:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                profit_factor = performance.get('profit_factor', 0)
                st.metric("Profit Factor", f"{profit_factor:.2f}")
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Equity curve
        if 'history' in pt_status:
            st.plotly_chart(create_equity_curve(pt_status['history']), 
                           use_container_width=True)
    
    with tab2:
        # Trading Interface
        st.subheader("Execute Trades")
        
        if not is_enabled:
            st.warning("Paper trading is disabled. Enable it to place trades.")
        else:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Order form
                st.markdown('<div class="order-form">', unsafe_allow_html=True)
                
                # Current market info
                col_price, col_signal = st.columns(2)
                with col_price:
                    st.metric("Current BTC Price", f"${current_price:,.2f}")
                with col_signal:
                    if latest_signal:
                        signal = latest_signal.get('signal', 'hold')
                        confidence = latest_signal.get('confidence', 0)
                        signal_emoji = "[BUY]" if signal == 'buy' else "[SELL]" if signal == 'sell' else "[HOLD]"
                        source = "Enhanced LSTM" if latest_signal.get('source') == 'enhanced_lstm' else "LSTM"
                        st.metric(f"{source} Signal", 
                                 f"{signal_emoji} {signal.upper()}", 
                                 f"Confidence: {confidence:.1%}")
                
                st.markdown("---")
                
                # Order type selection
                order_type = st.radio("Order Type", ["Market Order", "Limit Order"], 
                                     horizontal=True)
                
                # Trade direction
                trade_direction = st.radio("Direction", ["Buy", "Sell"], horizontal=True)
                
                # Amount input
                if trade_direction == "Buy":
                    max_amount = balance / current_price if current_price > 0 else 0
                    amount = st.number_input("BTC Amount", 
                                           min_value=0.0001, 
                                           max_value=max_amount,
                                           value=min(0.001, max_amount),
                                           format="%.6f")
                    cost = amount * current_price
                    st.info(f"Cost: ${cost:,.2f}")
                else:
                    max_amount = btc_holdings
                    amount = st.number_input("BTC Amount", 
                                           min_value=0.0001, 
                                           max_value=max_amount,
                                           value=min(0.001, max_amount),
                                           format="%.6f")
                    proceeds = amount * current_price
                    st.info(f"Proceeds: ${proceeds:,.2f}")
                
                # Limit price for limit orders
                limit_price = None
                if order_type == "Limit Order":
                    limit_price = st.number_input("Limit Price", 
                                                 min_value=1.0,
                                                 value=float(current_price),
                                                 format="%.2f")
                
                # Execute button
                if st.button(f"Place {trade_direction} Order", 
                            use_container_width=True, type="primary"):
                    # Prepare order data
                    order_data = {
                        "type": trade_direction.lower(),
                        "amount": amount,
                        "order_type": "limit" if order_type == "Limit Order" else "market"
                    }
                    
                    if limit_price:
                        order_data["limit_price"] = limit_price
                    
                    # Execute trade
                    result = api_client.post("/paper-trading/trade", order_data)
                    
                    if result and result.get('success'):
                        st.success(f"{trade_direction} order executed successfully!")
                        st.balloons()
                        time.sleep(1)
                        st.rerun()
                    else:
                        error_msg = result.get('error', 'Unknown error') if result else 'Connection error'
                        st.error(f"Order failed: {error_msg}")
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                # Quick trade suggestions based on signals
                st.subheader("Signal-Based Suggestions")
                
                if latest_signal:
                    signal = latest_signal.get('signal', 'hold')
                    confidence = latest_signal.get('confidence', 0)
                    
                    if signal == 'buy' and confidence > 0.6:
                        st.success("Strong Buy Signal")
                        suggested_amount = min(0.01, (balance * 0.1) / current_price)
                        st.write(f"Suggested amount: {suggested_amount:.6f} BTC")
                        st.write(f"Risk: ${suggested_amount * current_price:,.2f}")
                    elif signal == 'sell' and confidence > 0.6:
                        st.error("Strong Sell Signal")
                        suggested_amount = min(btc_holdings * 0.5, btc_holdings)
                        st.write(f"Suggested amount: {suggested_amount:.6f} BTC")
                    else:
                        st.info("Hold - No strong signal")
                        st.write("Wait for stronger signals")
                
                # Risk warnings
                st.markdown("---")
                st.subheader("Risk Management")
                st.write("• Never risk more than 2% per trade")
                st.write("• Use stop losses to limit downside")
                st.write("• Diversify your positions")
                st.write("• This is practice - learn from mistakes!")
    
    with tab3:
        # Current Positions
        st.subheader("Current Positions")
        
        if positions:
            for pos in positions:
                entry_price = pos.get('entry_price', 0)
                size = pos.get('size', 0)
                entry_value = entry_price * size
                current_value = current_price * size
                pnl = current_value - entry_value
                pnl_pct = (pnl / entry_value) * 100 if entry_value > 0 else 0
                
                with st.container():
                    st.markdown('<div class="position-card">', unsafe_allow_html=True)
                    
                    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                    
                    with col1:
                        position_type = pos.get('type', 'long').upper()
                        st.write(f"**{position_type} Position**")
                        st.write(f"Entry: ${entry_price:,.2f}")
                        st.write(f"Size: {size:.6f} BTC")
                        
                    with col2:
                        st.write("**Current Value**")
                        st.write(f"${current_value:,.2f}")
                        
                    with col3:
                        st.write("**P&L**")
                        st.markdown(format_pnl(pnl, pnl_pct), unsafe_allow_html=True)
                        
                    with col4:
                        if st.button("Close Position", key=f"close_{pos.get('id', '')}"):
                            result = api_client.post("/paper-trading/close-position", 
                                                   {"position_id": pos.get('id')})
                            if result:
                                st.success("Position closed!")
                                st.rerun()
                    
                    st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("No open positions. Start trading to build your portfolio!")
    
    with tab4:
        # Trade History
        st.subheader("Trade History")
        
        if trade_history:
            # Convert to DataFrame for easier manipulation
            df = pd.DataFrame(trade_history)
            
            # Add calculated columns
            if 'exit_price' in df.columns:
                df['pnl'] = (df['exit_price'] - df['entry_price']) * df['size']
                df['pnl_pct'] = (df['pnl'] / (df['entry_price'] * df['size'])) * 100
            
            # Format timestamp
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
            
            # Display options
            col1, col2 = st.columns([3, 1])
            with col1:
                search = st.text_input("Search trades", placeholder="Search by type, date...")
            with col2:
                sort_by = st.selectbox("Sort by", ["timestamp", "pnl", "size"])
            
            # Filter and sort
            if search:
                mask = df.astype(str).apply(lambda x: x.str.contains(search, case=False)).any(axis=1)
                df = df[mask]
            
            df = df.sort_values(sort_by, ascending=False)
            
            # Display styled DataFrame
            st.dataframe(
                df[['timestamp', 'type', 'entry_price', 'exit_price', 'size', 'pnl', 'pnl_pct']].style.format({
                    'entry_price': '${:,.2f}',
                    'exit_price': '${:,.2f}',
                    'size': '{:.6f}',
                    'pnl': '${:,.2f}',
                    'pnl_pct': '{:+.2f}%'
                }).map(
                    lambda x: 'color: green' if x > 0 else 'color: red' if x < 0 else '',
                    subset=['pnl', 'pnl_pct']
                ),
                use_container_width=True,
                height=400
            )
            
            # Summary statistics
            st.markdown("---")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_pnl = df['pnl'].sum() if 'pnl' in df.columns else 0
                st.metric("Total P&L", f"${total_pnl:,.2f}")
                
            with col2:
                avg_win = df[df['pnl'] > 0]['pnl'].mean() if 'pnl' in df.columns else 0
                st.metric("Average Win", f"${avg_win:,.2f}")
                
            with col3:
                avg_loss = abs(df[df['pnl'] < 0]['pnl'].mean()) if 'pnl' in df.columns else 0
                st.metric("Average Loss", f"${avg_loss:,.2f}")
                
            with col4:
                best_trade = df['pnl'].max() if 'pnl' in df.columns else 0
                st.metric("Best Trade", f"${best_trade:,.2f}")
        else:
            st.info("No trade history yet. Start trading to see your performance!")
    
    with tab5:
        # Analytics
        st.subheader("Trading Analytics")
        
        if trade_history and len(trade_history) > 0:
            # P&L Distribution
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(create_trade_distribution(trade_history), 
                               use_container_width=True)
            
            with col2:
                # Win/Loss pie chart
                df = pd.DataFrame(trade_history)
                if 'exit_price' in df.columns:
                    df['pnl'] = (df['exit_price'] - df['entry_price']) * df['size']
                    wins = len(df[df['pnl'] > 0])
                    losses = len(df[df['pnl'] < 0])
                    
                    fig = go.Figure(data=[go.Pie(
                        labels=['Wins', 'Losses'],
                        values=[wins, losses],
                        marker_colors=['#00ff88', '#ff3366']
                    )])
                    
                    fig.update_layout(
                        title="Win/Loss Ratio",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            # Monthly performance
            if len(trade_history) > 30:  # Only show if enough history
                st.markdown("---")
                st.subheader("Monthly Performance")
                
                df['month'] = pd.to_datetime(df['timestamp']).dt.to_period('M')
                monthly_pnl = df.groupby('month')['pnl'].sum()
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=monthly_pnl.index.astype(str),
                    y=monthly_pnl.values,
                    marker_color=['green' if x > 0 else 'red' for x in monthly_pnl.values]
                ))
                
                fig.update_layout(
                    title="Monthly P&L",
                    xaxis_title="Month",
                    yaxis_title="P&L ($)",
                    template="plotly_white",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Complete some trades to see analytics!")
            
            # Show tips for beginners
            st.markdown("---")
            st.subheader("📚 Paper Trading Tips")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **Getting Started:**
                - Start with small positions (0.001 BTC)
                - Follow the AI signals when confidence > 60%
                - Use limit orders to control entry price
                - Keep a trading journal
                """)
                
            with col2:
                st.markdown("""
                **Risk Management:**
                - Never risk more than 2% per trade
                - Set mental stop losses
                - Take profits gradually
                - Learn from losses
                """)

# Auto-refresh option
if st.sidebar.checkbox("Auto-refresh (5s)", value=False):
    time.sleep(5)
    st.rerun()

# Show the page
show_paper_trading()