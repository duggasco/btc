import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
from typing import Dict, List, Optional, Tuple
import logging

from components.page_styling import setup_page

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config and styling
api_client = setup_page(
    page_name="Paper Trading",
    page_title="Paper Trading Simulator",
    page_subtitle="Practice trading without risking real money"
)

# Helper functions
def format_pnl(value: float, percentage: float = None) -> str:
    """Format P&L with color coding"""
    color_class = "profit-positive" if value >= 0 else "profit-negative"
    pnl_str = f"${value:,.2f}"
    if percentage is not None:
        pnl_str += f" ({percentage:+.2f}%)"
    return f'<span class="{color_class}">{pnl_str}</span>'

def create_equity_curve(history: list) -> go.Figure:
    """Create portfolio equity curve with professional styling"""
    if not history:
        return go.Figure().add_annotation(
            text="No trading history yet", 
            xref="paper", yref="paper", 
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color='#6b7280')
        )
    
    df = pd.DataFrame(history)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Calculate cumulative values
    initial_value = 10000
    df['portfolio_value'] = initial_value  # This would come from the history data
    
    fig = go.Figure()
    
    # Portfolio value line
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['portfolio_value'],
        mode='lines',
        name='Portfolio Value',
        line=dict(color='#3b82f6', width=3),
        fill='tonexty',
        fillcolor='rgba(59, 130, 246, 0.1)'
    ))
    
    # Add initial value reference line
    fig.add_hline(
        y=initial_value, 
        line_dash="dash", 
        line_color="#6b7280",
        annotation_text="Initial Balance"
    )
    
    # Update layout
    fig.update_layout(
        title="Portfolio Performance",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        hovermode='x unified',
        template='plotly_dark',
        height=400,
        showlegend=False,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    # Format y-axis
    fig.update_yaxis(tickformat='$,.0f')
    
    return fig

def create_pnl_distribution(trades: List[Dict]) -> go.Figure:
    """Create P&L distribution chart"""
    if not trades:
        return go.Figure().add_annotation(
            text="No completed trades yet", 
            xref="paper", yref="paper", 
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color='#6b7280')
        )
    
    df = pd.DataFrame(trades)
    if 'pnl' not in df.columns:
        return go.Figure().add_annotation(
            text="No P&L data available", 
            xref="paper", yref="paper", 
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color='#6b7280')
        )
    
    # Create histogram
    fig = go.Figure()
    
    positive_pnl = df[df['pnl'] > 0]['pnl']
    negative_pnl = df[df['pnl'] < 0]['pnl']
    
    if len(positive_pnl) > 0:
        fig.add_trace(go.Histogram(
            x=positive_pnl,
            name='Profits',
            marker_color='#0ecb81',
            opacity=0.7,
            nbinsx=20
        ))
    
    if len(negative_pnl) > 0:
        fig.add_trace(go.Histogram(
            x=negative_pnl,
            name='Losses',
            marker_color='#f6465d',
            opacity=0.7,
            nbinsx=20
        ))
    
    fig.update_layout(
        title="P&L Distribution",
        xaxis_title="P&L ($)",
        yaxis_title="Frequency",
        template='plotly_dark',
        height=400,
        barmode='overlay',
        showlegend=True,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    fig.update_xaxis(tickformat='$,.0f')
    
    return fig

def create_win_loss_pie(trades: List[Dict]) -> go.Figure:
    """Create win/loss pie chart"""
    if not trades:
        return go.Figure().add_annotation(
            text="No trades to analyze", 
            xref="paper", yref="paper", 
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color='#6b7280')
        )
    
    df = pd.DataFrame(trades)
    if 'pnl' not in df.columns:
        return go.Figure().add_annotation(
            text="No P&L data available", 
            xref="paper", yref="paper", 
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color='#6b7280')
        )
    
    wins = len(df[df['pnl'] > 0])
    losses = len(df[df['pnl'] < 0])
    breakeven = len(df[df['pnl'] == 0])
    
    fig = go.Figure(data=[go.Pie(
        labels=['Wins', 'Losses', 'Breakeven'],
        values=[wins, losses, breakeven],
        hole=.4,
        marker_colors=['#0ecb81', '#f6465d', '#6b7280']
    )])
    
    total_trades = wins + losses + breakeven
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    
    fig.add_annotation(
        text=f"{win_rate:.1f}%<br>Win Rate",
        x=0.5, y=0.5,
        font=dict(size=20, color='white'),
        showarrow=False
    )
    
    fig.update_layout(
        title="Trade Outcomes",
        template='plotly_dark',
        height=400,
        showlegend=True,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig

def create_monthly_performance(trades: List[Dict]) -> go.Figure:
    """Create monthly performance chart"""
    if not trades:
        return go.Figure().add_annotation(
            text="No trade history available", 
            xref="paper", yref="paper", 
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color='#6b7280')
        )
    
    df = pd.DataFrame(trades)
    if 'timestamp' not in df.columns or 'pnl' not in df.columns:
        return go.Figure().add_annotation(
            text="Insufficient data for monthly analysis", 
            xref="paper", yref="paper", 
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color='#6b7280')
        )
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['month'] = df['timestamp'].dt.to_period('M')
    
    monthly_pnl = df.groupby('month')['pnl'].sum().reset_index()
    monthly_pnl['month_str'] = monthly_pnl['month'].astype(str)
    monthly_pnl['color'] = monthly_pnl['pnl'].apply(lambda x: '#0ecb81' if x > 0 else '#f6465d')
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=monthly_pnl['month_str'],
        y=monthly_pnl['pnl'],
        marker_color=monthly_pnl['color'],
        text=monthly_pnl['pnl'].apply(lambda x: f"${x:,.0f}"),
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Monthly P&L Performance",
        xaxis_title="Month",
        yaxis_title="P&L ($)",
        template='plotly_dark',
        height=400,
        showlegend=False,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    fig.update_yaxis(tickformat='$,.0f')
    
    return fig

def get_trading_tips(signal: str, confidence: float, portfolio: Dict) -> List[Dict]:
    """Generate trading tips based on current market conditions"""
    tips = []
    
    # General risk management tips
    tips.append({
        "icon": "⚠️",
        "title": "Risk Management",
        "content": "Never risk more than 2% of your portfolio on a single trade"
    })
    
    # Signal-specific tips
    if signal == 'buy' and confidence > 0.7:
        tips.append({
            "icon": "🟢",
            "title": "Strong Buy Signal",
            "content": "Consider entering a position, but use a stop loss at -3%"
        })
    elif signal == 'sell' and confidence > 0.7:
        tips.append({
            "icon": "🔴",
            "title": "Strong Sell Signal",
            "content": "Consider reducing exposure or closing long positions"
        })
    else:
        tips.append({
            "icon": "⏸️",
            "title": "Neutral Market",
            "content": "Wait for stronger signals before entering new positions"
        })
    
    # Portfolio-specific tips
    btc_holdings = portfolio.get('btc_holdings', 0)
    balance = portfolio.get('balance', 10000)
    
    if btc_holdings == 0:
        tips.append({
            "icon": "💡",
            "title": "No Positions",
            "content": "Start with small positions (0.1-1% of portfolio) to learn"
        })
    elif btc_holdings * 50000 > balance * 0.5:  # Assuming BTC ~$50k
        tips.append({
            "icon": "⚖️",
            "title": "Rebalance Alert",
            "content": "BTC position is >50% of portfolio. Consider rebalancing"
        })
    
    # Market condition tips
    tips.append({
        "icon": "📊",
        "title": "Market Analysis",
        "content": "Check multiple timeframes before making trading decisions"
    })
    
    return tips

def get_quick_trade_suggestions(price: float, signal: str, confidence: float, 
                              balance: float, btc_holdings: float) -> List[Dict]:
    """Generate quick trade suggestions"""
    suggestions = []
    
    # Calculate position sizes
    small_position = min(0.001, (balance * 0.01) / price)
    medium_position = min(0.01, (balance * 0.05) / price)
    large_position = min(0.1, (balance * 0.10) / price)
    
    if signal == 'buy' and confidence > 0.6:
        suggestions.append({
            "action": "buy",
            "size": small_position,
            "label": f"Small Buy ({small_position:.6f} BTC)",
            "risk": small_position * price,
            "confidence": "Low Risk"
        })
        suggestions.append({
            "action": "buy",
            "size": medium_position,
            "label": f"Medium Buy ({medium_position:.6f} BTC)",
            "risk": medium_position * price,
            "confidence": "Medium Risk"
        })
        if confidence > 0.8:
            suggestions.append({
                "action": "buy",
                "size": large_position,
                "label": f"Large Buy ({large_position:.6f} BTC)",
                "risk": large_position * price,
                "confidence": "High Risk"
            })
    
    elif signal == 'sell' and confidence > 0.6 and btc_holdings > 0:
        suggestions.append({
            "action": "sell",
            "size": btc_holdings * 0.25,
            "label": "Sell 25%",
            "risk": 0,
            "confidence": "Take Profits"
        })
        suggestions.append({
            "action": "sell",
            "size": btc_holdings * 0.5,
            "label": "Sell 50%",
            "risk": 0,
            "confidence": "Risk Reduction"
        })
        if confidence > 0.8:
            suggestions.append({
                "action": "sell",
                "size": btc_holdings,
                "label": "Sell All",
                "risk": 0,
                "confidence": "Exit Position"
            })
    
    else:
        suggestions.append({
            "action": "hold",
            "size": 0,
            "label": "Hold Position",
            "risk": 0,
            "confidence": "Wait for Signal"
        })
    
    return suggestions

# Main app
def main():
    # Fetch current data
    try:
        pt_status = api_client.get_paper_trading_status() or {}
        current_price_data = api_client.get_current_price() or {}
        latest_signal = api_client.get_latest_signal() or {}
        
        # Try enhanced LSTM
        if latest_signal.get('source') != 'enhanced_lstm':
            enhanced_signal = api_client.get("/signals/enhanced/latest")
            if enhanced_signal and enhanced_signal.get('source') == 'enhanced_lstm':
                latest_signal = enhanced_signal
                
    except Exception as e:
        st.error(f"Error connecting to backend: {str(e)}")
        st.info("Please ensure the backend server is running")
        return
    
    # Extract data
    is_enabled = pt_status.get('enabled', False)
    portfolio = pt_status.get('portfolio', {})
    positions = pt_status.get('positions', [])
    performance = pt_status.get('performance', {})
    trade_history = pt_status.get('trades', [])
    current_price = current_price_data.get('price', 0)
    
    # Header with status and controls
    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
    
    with col1:
        status_html = f"""
        <div style='display: flex; align-items: center; gap: 15px;'>
            <span style='font-size: 24px; font-weight: bold;'>Status:</span>
            <span style='font-size: 20px; color: {"#0ecb81" if is_enabled else "#f6465d"};'>
                {"🟢 Active" if is_enabled else "🔴 Inactive"}
            </span>
            {f"<span style='color: #6b7280;'>|</span><span style='color: #3b82f6;'>BTC: ${current_price:,.2f}</span>" if current_price else ""}
        </div>
        """
        st.markdown(status_html, unsafe_allow_html=True)
    
    with col2:
        if st.button("⚡ Toggle Trading", use_container_width=True, type="secondary"):
            result = api_client.toggle_paper_trading()
            if result:
                st.success(f"Paper trading {'disabled' if is_enabled else 'enabled'}")
                st.rerun()
    
    with col3:
        if st.button("🔄 Reset Portfolio", use_container_width=True):
            if st.session_state.get('confirm_reset'):
                result = api_client.post("/paper-trading/reset", {})
                if result:
                    st.success("Portfolio reset successfully!")
                    del st.session_state['confirm_reset']
                    st.rerun()
            else:
                st.session_state['confirm_reset'] = True
                st.warning("Click again to confirm reset")
    
    with col4:
        if st.button("📥 Export History", use_container_width=True):
            if trade_history:
                df = pd.DataFrame(trade_history)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"paper_trading_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", 
        "💱 Execute Trades", 
        "📋 Positions", 
        "📈 Analytics", 
        "📜 History"
    ])
    
    # Tab 1: Overview
    with tab1:
        # Portfolio metrics
        st.markdown("### 💼 Portfolio Overview")
        
        balance = portfolio.get('balance', 10000)
        btc_holdings = portfolio.get('btc_holdings', 0)
        btc_value = btc_holdings * current_price if current_price else 0
        total_value = balance + btc_value
        initial_balance = 10000
        total_pnl = total_value - initial_balance
        pnl_pct = (total_pnl / initial_balance) * 100
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "💵 Cash Balance", 
                f"${balance:,.2f}",
                f"{(balance/initial_balance - 1)*100:+.1f}%" if balance != initial_balance else None
            )
        
        with col2:
            st.metric(
                "₿ BTC Holdings", 
                f"{btc_holdings:.6f}",
                f"${btc_value:,.2f}"
            )
        
        with col3:
            st.metric(
                "💼 Total Portfolio", 
                f"${total_value:,.2f}",
                f"{total_pnl:+,.2f} ({pnl_pct:+.2f}%)"
            )
        
        with col4:
            win_rate = performance.get('win_rate', 0)
            total_trades = performance.get('total_trades', 0)
            st.metric(
                "📈 Win Rate", 
                f"{win_rate:.1f}%",
                f"{total_trades} trades"
            )
        
        # Quick stats
        st.markdown("### 📊 Performance Metrics")
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("**Sharpe Ratio**")
            sharpe = performance.get('sharpe_ratio', 0)
            color = "profit-positive" if sharpe > 1 else "profit-negative" if sharpe < 0 else ""
            st.markdown(f"<h3 class='{color}'>{sharpe:.2f}</h3>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("**Profit Factor**")
            pf = performance.get('profit_factor', 0)
            color = "profit-positive" if pf > 1.5 else "profit-negative" if pf < 1 else ""
            st.markdown(f"<h3 class='{color}'>{pf:.2f}</h3>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("**Max Drawdown**")
            dd = performance.get('max_drawdown', 0)
            color = "profit-negative" if dd < -10 else ""
            st.markdown(f"<h3 class='{color}'>{dd:.2f}%</h3>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("**Avg Trade**")
            avg_trade = performance.get('avg_trade_pnl', 0)
            color = "profit-positive" if avg_trade > 0 else "profit-negative"
            st.markdown(f"<h3 class='{color}'>${avg_trade:.2f}</h3>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col5:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("**Best Trade**")
            best = performance.get('best_trade', 0)
            st.markdown(f"<h3 class='profit-positive'>${best:.2f}</h3>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col6:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("**Worst Trade**")
            worst = performance.get('worst_trade', 0)
            st.markdown(f"<h3 class='profit-negative'>${worst:.2f}</h3>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Trading tips
        st.markdown("### 💡 Trading Tips for Beginners")
        
        tips = get_trading_tips(
            latest_signal.get('signal', 'hold'),
            latest_signal.get('confidence', 0),
            portfolio
        )
        
        tip_cols = st.columns(len(tips))
        for i, tip in enumerate(tips):
            with tip_cols[i]:
                st.markdown(f"""
                <div class="tip-card">
                    <div style="font-size: 32px; margin-bottom: 10px;">{tip['icon']}</div>
                    <h4>{tip['title']}</h4>
                    <p style="color: #9ca3af; font-size: 14px;">{tip['content']}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Tab 2: Execute Trades
    with tab2:
        if not is_enabled:
            st.warning("⚠️ Enable paper trading to place orders")
            st.info("Click the 'Toggle Trading' button in the header to start")
        else:
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.markdown("### 📝 Place Order")
                
                # Market info
                st.markdown('<div class="order-form">', unsafe_allow_html=True)
                
                col_price, col_signal, col_model = st.columns(3)
                
                with col_price:
                    st.metric("Current Price", f"${current_price:,.2f}")
                
                with col_signal:
                    if latest_signal:
                        signal = latest_signal.get('signal', 'hold')
                        confidence = latest_signal.get('confidence', 0)
                        signal_color = "#0ecb81" if signal == 'buy' else "#f6465d" if signal == 'sell' else "#6b7280"
                        st.markdown(f"""
                        <div style='text-align: center;'>
                            <p style='margin: 0; color: #9ca3af; font-size: 14px;'>AI Signal</p>
                            <h3 style='margin: 0; color: {signal_color};'>{signal.upper()}</h3>
                            <p style='margin: 0; color: #6b7280; font-size: 12px;'>{confidence:.1%} conf.</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col_model:
                    source = "Enhanced LSTM" if latest_signal.get('source') == 'enhanced_lstm' else "LSTM"
                    st.markdown(f"""
                    <div style='text-align: center;'>
                        <p style='margin: 0; color: #9ca3af; font-size: 14px;'>Model</p>
                        <h3 style='margin: 0; color: #3b82f6; font-size: 16px;'>{source}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Order type selection
                order_type = st.radio(
                    "Order Type",
                    ["Market Order", "Limit Order", "Stop Loss", "Take Profit"],
                    horizontal=True,
                    help="Market orders execute immediately, limit orders wait for target price"
                )
                
                # Trade direction
                trade_direction = st.radio(
                    "Direction",
                    ["Buy", "Sell"],
                    horizontal=True
                )
                
                # Amount input
                col_amount, col_value = st.columns(2)
                
                with col_amount:
                    if trade_direction == "Buy":
                        max_amount = balance / current_price if current_price > 0 else 0
                        amount = st.number_input(
                            "BTC Amount",
                            min_value=0.0001,
                            max_value=max_amount,
                            value=min(0.001, max_amount),
                            format="%.6f",
                            help="Amount of BTC to buy"
                        )
                    else:
                        amount = st.number_input(
                            "BTC Amount",
                            min_value=0.0001,
                            max_value=max(0.0001, btc_holdings),
                            value=min(0.001, btc_holdings) if btc_holdings > 0 else 0.0001,
                            format="%.6f",
                            help="Amount of BTC to sell"
                        )
                
                with col_value:
                    value = amount * current_price
                    st.metric(
                        "Order Value",
                        f"${value:,.2f}",
                        f"{(value/balance)*100:.1f}% of cash" if trade_direction == "Buy" else f"{(amount/btc_holdings)*100:.1f}% of BTC"
                    )
                
                # Price inputs for limit/stop orders
                limit_price = None
                stop_price = None
                
                if order_type in ["Limit Order", "Stop Loss", "Take Profit"]:
                    col_price1, col_price2 = st.columns(2)
                    
                    with col_price1:
                        if order_type == "Limit Order":
                            limit_price = st.number_input(
                                "Limit Price",
                                min_value=1.0,
                                value=float(current_price * 0.99 if trade_direction == "Buy" else current_price * 1.01),
                                format="%.2f",
                                help="Price at which to execute the order"
                            )
                        elif order_type == "Stop Loss":
                            stop_price = st.number_input(
                                "Stop Price",
                                min_value=1.0,
                                value=float(current_price * 0.97),
                                format="%.2f",
                                help="Price at which to trigger stop loss"
                            )
                        else:  # Take Profit
                            stop_price = st.number_input(
                                "Target Price",
                                min_value=1.0,
                                value=float(current_price * 1.03),
                                format="%.2f",
                                help="Price at which to take profits"
                            )
                    
                    with col_price2:
                        if limit_price:
                            diff_pct = ((limit_price - current_price) / current_price) * 100
                            st.metric("Price Difference", f"{diff_pct:+.2f}%")
                        elif stop_price:
                            diff_pct = ((stop_price - current_price) / current_price) * 100
                            st.metric("Price Difference", f"{diff_pct:+.2f}%")
                
                # Place order button
                col_button, col_space = st.columns([2, 3])
                
                with col_button:
                    button_type = "primary" if trade_direction == "Buy" else "secondary"
                    if st.button(
                        f"Place {trade_direction} Order",
                        use_container_width=True,
                        type=button_type,
                        disabled=(amount <= 0)
                    ):
                        order_data = {
                            "type": trade_direction.lower(),
                            "amount": amount,
                            "order_type": order_type.lower().replace(" ", "_")
                        }
                        
                        if limit_price:
                            order_data["limit_price"] = limit_price
                        if stop_price:
                            order_data["stop_price"] = stop_price
                        
                        with st.spinner("Executing order..."):
                            result = api_client.post("/paper-trading/trade", order_data)
                        
                        if result and result.get('success'):
                            st.success(f"✅ {trade_direction} order executed successfully!")
                            st.balloons()
                            time.sleep(1)
                            st.rerun()
                        else:
                            error_msg = result.get('error', 'Unknown error') if result else 'Connection error'
                            st.error(f"❌ Order failed: {error_msg}")
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown("### ⚡ Quick Trade Suggestions")
                
                suggestions = get_quick_trade_suggestions(
                    current_price,
                    latest_signal.get('signal', 'hold'),
                    latest_signal.get('confidence', 0),
                    balance,
                    btc_holdings
                )
                
                for i, suggestion in enumerate(suggestions):
                    st.markdown('<div class="suggestion-card">', unsafe_allow_html=True)
                    
                    col_info, col_btn = st.columns([3, 1])
                    
                    with col_info:
                        action_color = "#0ecb81" if suggestion['action'] == 'buy' else "#f6465d" if suggestion['action'] == 'sell' else "#6b7280"
                        st.markdown(f"""
                        <h4 style='margin: 0; color: {action_color};'>{suggestion['label']}</h4>
                        <p style='margin: 5px 0; color: #9ca3af; font-size: 14px;'>
                            Risk: ${suggestion['risk']:,.2f} | {suggestion['confidence']}
                        </p>
                        """, unsafe_allow_html=True)
                    
                    with col_btn:
                        if suggestion['action'] != 'hold':
                            if st.button("Execute", key=f"quick_{i}", use_container_width=True):
                                order_data = {
                                    "type": suggestion['action'],
                                    "amount": suggestion['size'],
                                    "order_type": "market"
                                }
                                result = api_client.post("/paper-trading/trade", order_data)
                                if result and result.get('success'):
                                    st.success("Order executed!")
                                    st.rerun()
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Advanced order types
                with st.expander("🔧 Advanced Order Types"):
                    st.markdown("""
                    **OCO (One-Cancels-Other)**
                    Place both stop loss and take profit orders simultaneously
                    """)
                    
                    if btc_holdings > 0:
                        col1, col2 = st.columns(2)
                        with col1:
                            oco_stop = st.number_input(
                                "Stop Loss Price",
                                value=float(current_price * 0.95),
                                format="%.2f"
                            )
                        with col2:
                            oco_target = st.number_input(
                                "Take Profit Price",
                                value=float(current_price * 1.05),
                                format="%.2f"
                            )
                        
                        if st.button("Place OCO Order", use_container_width=True):
                            st.info("OCO orders coming soon!")
                    else:
                        st.info("You need BTC holdings to place OCO orders")
    
    # Tab 3: Positions
    with tab3:
        st.markdown("### 📋 Current Positions")
        
        if positions:
            # Position summary
            total_positions = len(positions)
            total_value = sum(pos.get('size', 0) * current_price for pos in positions)
            total_pnl = sum((current_price - pos.get('entry_price', 0)) * pos.get('size', 0) for pos in positions)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Open Positions", total_positions)
            with col2:
                st.metric("Total Value", f"${total_value:,.2f}")
            with col3:
                pnl_color = "profit-positive" if total_pnl >= 0 else "profit-negative"
                st.markdown(f"<p style='margin: 0; color: #9ca3af; font-size: 14px;'>Total P&L</p>", unsafe_allow_html=True)
                st.markdown(f"<h3 style='margin: 0;' class='{pnl_color}'>${total_pnl:,.2f}</h3>", unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Individual positions
            for i, pos in enumerate(positions):
                entry_price = pos.get('entry_price', 0)
                size = pos.get('size', 0)
                entry_value = entry_price * size
                current_value = current_price * size
                pnl = current_value - entry_value
                pnl_pct = (pnl / entry_value) * 100 if entry_value > 0 else 0
                
                st.markdown('<div class="position-card">', unsafe_allow_html=True)
                
                col1, col2, col3, col4, col5 = st.columns([2, 1.5, 1.5, 1.5, 1])
                
                with col1:
                    position_type = pos.get('type', 'long').upper()
                    entry_time = pd.to_datetime(pos.get('timestamp', '')).strftime('%Y-%m-%d %H:%M') if pos.get('timestamp') else 'Unknown'
                    st.markdown(f"""
                    <h4 style='margin: 0; color: #3b82f6;'>{position_type} Position #{i+1}</h4>
                    <p style='margin: 5px 0; color: #6b7280; font-size: 12px;'>{entry_time}</p>
                    <p style='margin: 0;'>Entry: ${entry_price:,.2f} | Size: {size:.6f} BTC</p>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("**Current Value**")
                    st.write(f"${current_value:,.2f}")
                    st.caption(f"Price: ${current_price:,.2f}")
                
                with col3:
                    st.markdown("**P&L**")
                    st.markdown(format_pnl(pnl, pnl_pct), unsafe_allow_html=True)
                
                with col4:
                    # Position management
                    st.markdown("**Actions**")
                    action = st.selectbox(
                        "Action",
                        ["Hold", "Close 25%", "Close 50%", "Close All"],
                        key=f"action_{i}",
                        label_visibility="collapsed"
                    )
                
                with col5:
                    if action != "Hold":
                        if st.button("Execute", key=f"close_{i}", use_container_width=True):
                            close_amount = size
                            if action == "Close 25%":
                                close_amount = size * 0.25
                            elif action == "Close 50%":
                                close_amount = size * 0.5
                            
                            result = api_client.post("/paper-trading/trade", {
                                "type": "sell",
                                "amount": close_amount,
                                "order_type": "market"
                            })
                            
                            if result and result.get('success'):
                                st.success(f"Closed {close_amount:.6f} BTC")
                                st.rerun()
                
                # Risk management for position
                with st.expander(f"Risk Management for Position #{i+1}"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        suggested_stop = entry_price * 0.97
                        st.markdown(f"""
                        **Suggested Stop Loss**
                        <p style='color: #f6465d; font-size: 18px;'>${suggested_stop:,.2f}</p>
                        <p style='color: #6b7280; font-size: 12px;'>-3% from entry</p>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        suggested_target = entry_price * 1.05
                        st.markdown(f"""
                        **Suggested Take Profit**
                        <p style='color: #0ecb81; font-size: 18px;'>${suggested_target:,.2f}</p>
                        <p style='color: #6b7280; font-size: 12px;'>+5% from entry</p>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        risk_reward = 5 / 3  # 5% profit vs 3% loss
                        st.markdown(f"""
                        **Risk/Reward Ratio**
                        <p style='color: #3b82f6; font-size: 18px;'>{risk_reward:.2f}:1</p>
                        <p style='color: #6b7280; font-size: 12px;'>Favorable ratio</p>
                        """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("📭 No open positions")
            st.markdown("""
            <div class='tip-card' style='margin-top: 20px;'>
                <h4>Getting Started</h4>
                <ol style='text-align: left; margin: 10px 0;'>
                    <li>Go to the Execute Trades tab</li>
                    <li>Start with a small position (0.001 BTC)</li>
                    <li>Use market orders for immediate execution</li>
                    <li>Set stop losses to manage risk</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)
    
    # Tab 4: Analytics
    with tab4:
        st.markdown("### 📊 Trading Analytics")
        
        if not trade_history:
            st.info("📈 Start trading to see analytics")
            st.write("Analytics will appear here once you've made some trades")
        else:
            # Performance charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Equity curve
                if 'history' in pt_status:
                    st.plotly_chart(
                        create_equity_curve(pt_status['history']),
                        use_container_width=True
                    )
                else:
                    st.info("Equity curve will appear after more trades")
            
            with col2:
                # P&L distribution
                st.plotly_chart(
                    create_pnl_distribution(trade_history),
                    use_container_width=True
                )
            
            # Win/Loss analysis
            col1, col2 = st.columns(2)
            
            with col1:
                # Win/Loss pie chart
                st.plotly_chart(
                    create_win_loss_pie(trade_history),
                    use_container_width=True
                )
            
            with col2:
                # Monthly performance
                st.plotly_chart(
                    create_monthly_performance(trade_history),
                    use_container_width=True
                )
            
            # Detailed statistics
            with st.expander("📊 Detailed Statistics", expanded=True):
                df = pd.DataFrame(trade_history)
                
                if 'pnl' in df.columns:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.markdown("**Returns Analysis**")
                        total_return = (total_value / initial_balance - 1) * 100
                        avg_return = df['pnl'].mean() / initial_balance * 100
                        st.metric("Total Return", f"{total_return:.2f}%")
                        st.metric("Avg Return/Trade", f"{avg_return:.2f}%")
                        st.metric("Best Day", f"{df.groupby(df['timestamp'].dt.date)['pnl'].sum().max():.2f}")
                    
                    with col2:
                        st.markdown("**Risk Metrics**")
                        if len(df) > 1:
                            returns = df['pnl'] / initial_balance
                            volatility = returns.std() * np.sqrt(252) * 100  # Annualized
                            st.metric("Volatility", f"{volatility:.2f}%")
                        st.metric("Max Loss", f"${abs(df['pnl'].min()):.2f}")
                        st.metric("Avg Loss", f"${abs(df[df['pnl'] < 0]['pnl'].mean()):.2f}")
                    
                    with col3:
                        st.markdown("**Trade Statistics**")
                        avg_duration = "N/A"  # Would need entry/exit times
                        st.metric("Total Trades", len(df))
                        st.metric("Winning Trades", len(df[df['pnl'] > 0]))
                        st.metric("Losing Trades", len(df[df['pnl'] < 0]))
                    
                    with col4:
                        st.markdown("**Efficiency**")
                        profit_trades = df[df['pnl'] > 0]['pnl'].sum()
                        loss_trades = abs(df[df['pnl'] < 0]['pnl'].sum())
                        expectancy = df['pnl'].mean()
                        
                        st.metric("Total Profits", f"${profit_trades:.2f}")
                        st.metric("Total Losses", f"${loss_trades:.2f}")
                        st.metric("Expectancy", f"${expectancy:.2f}")
            
            # Trading journal insights
            with st.expander("📝 Trading Journal Insights"):
                # Best and worst trades
                best_trades = df.nlargest(3, 'pnl')[['timestamp', 'type', 'entry_price', 'exit_price', 'pnl']]
                worst_trades = df.nsmallest(3, 'pnl')[['timestamp', 'type', 'entry_price', 'exit_price', 'pnl']]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**🏆 Best Trades**")
                    st.dataframe(
                        best_trades.style.format({
                            'entry_price': '${:,.2f}',
                            'exit_price': '${:,.2f}',
                            'pnl': '${:,.2f}'
                        }),
                        use_container_width=True
                    )
                
                with col2:
                    st.markdown("**📉 Worst Trades**")
                    st.dataframe(
                        worst_trades.style.format({
                            'entry_price': '${:,.2f}',
                            'exit_price': '${:,.2f}',
                            'pnl': '${:,.2f}'
                        }),
                        use_container_width=True
                    )
                
                # Trading patterns
                st.markdown("**📊 Trading Patterns**")
                
                # Time of day analysis
                if 'timestamp' in df.columns:
                    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
                    hourly_pnl = df.groupby('hour')['pnl'].mean()
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=hourly_pnl.index,
                        y=hourly_pnl.values,
                        marker_color=['#0ecb81' if x > 0 else '#f6465d' for x in hourly_pnl.values]
                    ))
                    
                    fig.update_layout(
                        title="Average P&L by Hour of Day",
                        xaxis_title="Hour",
                        yaxis_title="Average P&L ($)",
                        template='plotly_dark',
                        height=300
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    # Tab 5: History
    with tab5:
        st.markdown("### 📜 Trade History")
        
        if trade_history:
            # Filters
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
            
            with col1:
                search = st.text_input(
                    "🔍 Search",
                    placeholder="Search by type, date, price...",
                    help="Search across all fields"
                )
            
            with col2:
                filter_type = st.selectbox(
                    "Type",
                    ["All", "Buy", "Sell"],
                    help="Filter by trade type"
                )
            
            with col3:
                sort_by = st.selectbox(
                    "Sort by",
                    ["timestamp", "pnl", "size", "entry_price"],
                    help="Sort trades by field"
                )
            
            with col4:
                sort_order = st.radio(
                    "Order",
                    ["Desc", "Asc"],
                    horizontal=True
                )
            
            # Apply filters
            df = pd.DataFrame(trade_history)
            
            # Search filter
            if search:
                mask = df.astype(str).apply(lambda x: x.str.contains(search, case=False)).any(axis=1)
                df = df[mask]
            
            # Type filter
            if filter_type != "All":
                df = df[df['type'] == filter_type.lower()]
            
            # Sort
            df = df.sort_values(sort_by, ascending=(sort_order == "Asc"))
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Trades", len(df))
            
            with col2:
                if 'pnl' in df.columns:
                    total_pnl = df['pnl'].sum()
                    color = "profit-positive" if total_pnl > 0 else "profit-negative"
                    st.markdown(f"<p style='margin: 0; color: #9ca3af; font-size: 14px;'>Total P&L</p>", unsafe_allow_html=True)
                    st.markdown(f"<h3 style='margin: 0;' class='{color}'>${total_pnl:,.2f}</h3>", unsafe_allow_html=True)
            
            with col3:
                if 'pnl' in df.columns:
                    wins = len(df[df['pnl'] > 0])
                    win_rate = (wins / len(df) * 100) if len(df) > 0 else 0
                    st.metric("Win Rate", f"{win_rate:.1f}%")
            
            with col4:
                if 'size' in df.columns:
                    total_volume = df['size'].sum()
                    st.metric("Total Volume", f"{total_volume:.4f} BTC")
            
            # Display table
            display_columns = ['timestamp', 'type', 'entry_price', 'size']
            if 'exit_price' in df.columns:
                display_columns.append('exit_price')
            if 'pnl' in df.columns:
                display_columns.extend(['pnl', 'pnl_pct'])
            
            # Format display
            format_dict = {
                'entry_price': '${:,.2f}',
                'exit_price': '${:,.2f}',
                'size': '{:.6f}',
                'pnl': '${:,.2f}',
                'pnl_pct': '{:+.2f}%'
            }
            
            # Apply conditional formatting
            styled_df = df[display_columns].style.format(format_dict)
            
            if 'pnl' in df.columns:
                styled_df = styled_df.map(
                    lambda x: 'color: #0ecb81' if x > 0 else 'color: #f6465d' if x < 0 else '',
                    subset=['pnl', 'pnl_pct']
                )
            
            st.dataframe(
                styled_df,
                use_container_width=True,
                height=600
            )
            
            # Export options
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 1, 3])
            
            with col1:
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Export CSV",
                    data=csv,
                    file_name=f"trade_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                # Create summary report
                if 'pnl' in df.columns:
                    summary = {
                        "Report Date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        "Total Trades": len(df),
                        "Total P&L": f"${df['pnl'].sum():,.2f}",
                        "Win Rate": f"{(len(df[df['pnl'] > 0]) / len(df) * 100):.1f}%",
                        "Best Trade": f"${df['pnl'].max():,.2f}",
                        "Worst Trade": f"${df['pnl'].min():,.2f}",
                        "Average Trade": f"${df['pnl'].mean():,.2f}"
                    }
                    
                    summary_text = "\n".join([f"{k}: {v}" for k, v in summary.items()])
                    
                    st.download_button(
                        label="📊 Export Report",
                        data=summary_text,
                        file_name=f"trading_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
        else:
            st.info("📭 No trade history yet")
            st.markdown("""
            <div class='tip-card' style='margin-top: 20px;'>
                <h4>Start Paper Trading</h4>
                <p>Your trade history will appear here once you start trading.</p>
                <p>Paper trading lets you practice without risking real money!</p>
            </div>
            """, unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()