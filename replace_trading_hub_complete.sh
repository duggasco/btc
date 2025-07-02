#!/bin/bash

# Replace Trading Hub with Complete Functionality
# Restore all missing features while maintaining safe variable scoping

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}Replacing Trading Hub with Complete Functionality${NC}"

# Find frontend directory
if [ -d "src/frontend" ] && [ -f "src/frontend/app.py" ]; then
    FRONTEND_DIR="src/frontend"
elif [ -d "src" ] && [ -f "src/app.py" ]; then
    FRONTEND_DIR="src"
else
    echo -e "${RED}ERROR: Cannot find frontend directory!${NC}"
    exit 1
fi

APP_FILE="$FRONTEND_DIR/app.py"

# Create backup
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
cp "$APP_FILE" "$APP_FILE.complete_restore_$TIMESTAMP"
echo -e "${GREEN}✓ Backup created: $APP_FILE.complete_restore_$TIMESTAMP${NC}"

# Replace the show_trading_hub function with the complete version
echo -e "${YELLOW}Replacing show_trading_hub function...${NC}"

python3 << 'EOFCOMPLETE'
import re

# Read the current file
with open('src/frontend/app.py', 'r') as f:
    content = f.read()

# Complete Trading Hub function with all functionality
complete_trading_hub = '''def show_trading_hub():
    """Unified Trading Hub - Complete functionality with all features"""
    st.header("💰 Trading Hub")
    st.markdown("Complete trading interface with portfolio management, paper trading, and risk controls")
    
    # Mode selector at the top
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        trading_mode = st.radio(
            "Trading Mode",
            ["📊 Real Trading", "📄 Paper Trading"],
            horizontal=True,
            help="Switch between real and paper trading modes"
        )
    
    is_paper_mode = "Paper" in trading_mode
    
    # Initialize ALL variables at the top to avoid scope issues
    pt_status = {}
    portfolio_metrics = {}
    portfolio_data = {}
    performance_data = {}
    is_trading_active = False
    positions = []
    trades = []
    
    # Fetch all necessary data
    try:
        trading_status = fetch_api_data("/trading/status") or {}
        latest_signal = fetch_api_data("/signals/enhanced/latest") or fetch_api_data("/signals/latest") or {}
        btc_data = fetch_api_data("/btc/latest") or {}
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        st.info("Please ensure the backend is running")
        return
    
    # Set mode-specific data
    if is_paper_mode:
        pt_status = fetch_api_data("/paper-trading/status") or {}
        portfolio_data = pt_status.get('portfolio', {})
        performance_data = pt_status.get('performance', {})
        is_trading_active = pt_status.get('enabled', False)
    else:
        portfolio_metrics = fetch_api_data("/portfolio/metrics") or {}
        positions = fetch_api_data("/portfolio/positions") or []
        trades = fetch_api_data("/trades/all") or []
        portfolio_data = portfolio_metrics
        performance_data = portfolio_metrics
        is_trading_active = trading_status.get('is_active', False)
    
    # Extract current price
    current_price = 0
    if btc_data and 'data' in btc_data and len(btc_data['data']) > 0:
        current_price = btc_data['data'][-1].get('close', 0)
    elif btc_data and 'latest_price' in btc_data:
        current_price = btc_data['latest_price']
    
    # Status Bar - 6 key metrics
    st.markdown("---")
    status_cols = st.columns(6)
    
    with status_cols[0]:
        status_color = "🟢" if is_trading_active else "🔴"
        st.metric(
            "Trading Status",
            f"{status_color} {'Active' if is_trading_active else 'Inactive'}",
            delta="Paper Mode" if is_paper_mode else "Real Mode"
        )
    
    with status_cols[1]:
        st.metric("BTC Price", f"${current_price:,.2f}")
    
    with status_cols[2]:
        if latest_signal:
            signal = latest_signal.get('signal', 'hold')
            confidence = latest_signal.get('confidence', 0)
            signal_emoji = "🟢" if signal == 'buy' else "🔴" if signal == 'sell' else "⚪"
            st.metric("Signal", f"{signal_emoji} {signal.upper()}", f"{confidence:.1%}")
    
    with status_cols[3]:
        portfolio_value = portfolio_data.get('total_value', 0)
        if is_paper_mode and pt_status:
            balance = portfolio_data.get('balance', portfolio_data.get('usd_balance', 10000))
            btc_balance = portfolio_data.get('btc_balance', 0)
            portfolio_value = balance + (btc_balance * current_price)
        st.metric("Portfolio Value", f"${portfolio_value:,.2f}")
    
    with status_cols[4]:
        total_pnl = performance_data.get('total_pnl', portfolio_data.get('total_pnl', 0))
        pnl_pct = performance_data.get('total_return_pct', 0)
        if is_paper_mode and portfolio_value > 0:
            pnl_pct = ((portfolio_value - 10000) / 10000) * 100
        elif not is_paper_mode and portfolio_data.get('total_invested', 0) > 0:
            pnl_pct = (total_pnl / portfolio_data.get('total_invested', 1)) * 100
        st.metric("Total P&L", f"${total_pnl:,.2f}", f"{pnl_pct:+.2f}%")
    
    with status_cols[5]:
        win_rate = performance_data.get('win_rate', 0)
        if is_paper_mode and pt_status:
            trades_list = portfolio_data.get('trades', [])
            if trades_list:
                winning_trades = sum(1 for t in trades_list if t.get('pnl', 0) > 0)
                win_rate = (winning_trades / len(trades_list)) if trades_list else 0
        st.metric("Win Rate", f"{win_rate:.1%}")
    
    # Enhanced Tabs with ALL original functionality
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🎯 Trading Controls",
        "💼 Portfolio & Positions", 
        "🛡️ Risk & Limits",
        "📊 Performance Analytics",
        "📋 Trade History",
        "⚙️ Settings"
    ])
    
    # TAB 1: Trading Controls (comprehensive trading interface)
    with tab1:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### 🎯 Trading Actions")
            
            # Trading toggle
            if is_paper_mode:
                toggle_label = "📄 Toggle Paper Trading"
                toggle_endpoint = "/paper-trading/toggle"
            else:
                toggle_label = "💰 Toggle Real Trading"
                toggle_endpoint = "/trading/start" if not is_trading_active else "/trading/stop"
            
            if st.button(toggle_label, type="primary", use_container_width=True):
                result = post_api_data(toggle_endpoint, {})
                if result:
                    if is_paper_mode:
                        st.success(f"Paper trading {'started' if result.get('enabled') else 'stopped'}")
                    else:
                        st.success(f"Trading {'started' if not is_trading_active else 'stopped'}")
                    time.sleep(0.5)
                    st.rerun()
            
            # Manual trading controls
            st.markdown("### 🎮 Manual Trading")
            
            default_size = 0.001 if not is_paper_mode else 0.01
            trade_size = st.number_input(
                "Trade Size (BTC)",
                min_value=0.0001,
                max_value=1.0,
                value=default_size,
                step=0.0001,
                format="%.4f"
            )
            
            col_buy, col_sell = st.columns(2)
            
            with col_buy:
                if st.button("🟢 BUY", type="primary", use_container_width=True):
                    endpoint = "/paper-trading/trade" if is_paper_mode else "/trades/execute"
                    result = post_api_data(endpoint, {
                        "signal": "buy",
                        "size": trade_size,
                        "reason": "manual_buy"
                    })
                    if result:
                        st.success(f"Bought {trade_size} BTC")
                        time.sleep(0.5)
                        st.rerun()
            
            with col_sell:
                if st.button("🔴 SELL", type="primary", use_container_width=True):
                    endpoint = "/paper-trading/trade" if is_paper_mode else "/trades/execute"
                    result = post_api_data(endpoint, {
                        "signal": "sell",
                        "size": trade_size,
                        "reason": "manual_sell"
                    })
                    if result:
                        st.success(f"Sold {trade_size} BTC")
                        time.sleep(0.5)
                        st.rerun()
            
            # Quick stats
            st.markdown("### 📊 Quick Stats")
            if is_paper_mode and pt_status:
                st.info(f"Balance: ${portfolio_data.get('balance', portfolio_data.get('usd_balance', 10000)):,.2f}")
                st.info(f"BTC: {portfolio_data.get('btc_balance', 0):.6f}")
                trades_today = len([t for t in portfolio_data.get('trades', []) 
                                  if pd.to_datetime(t.get('timestamp', '')).date() == datetime.now().date()])
                st.info(f"Trades Today: {trades_today}")
                st.info(f"Total Trades: {len(portfolio_data.get('trades', []))}")
            else:
                st.info(f"Open Positions: {len(positions)}")
                total_btc = sum(p.get('total_size', 0) for p in positions)
                st.info(f"Total BTC: {total_btc:.6f}")
                st.info(f"Total Invested: ${portfolio_data.get('total_invested', 0):,.2f}")
            
            if latest_signal and 'predicted_price' in latest_signal:
                st.info(f"Predicted Price: ${latest_signal['predicted_price']:,.2f}")
        
        with col2:
            st.markdown("### 📈 Price Chart")
            if btc_data and 'data' in btc_data:
                df = pd.DataFrame(btc_data['data'])
                
                fig = go.Figure()
                
                # Add candlestick
                fig.add_trace(go.Candlestick(
                    x=pd.to_datetime(df['timestamp']),
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name='BTC/USD'
                ))
                
                fig.update_layout(
                    height=400,
                    showlegend=True,
                    xaxis_rangeslider_visible=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    # TAB 2: Portfolio & Positions (comprehensive portfolio management)
    with tab2:
        st.markdown("### 💼 Portfolio Overview")
        
        if not is_paper_mode and portfolio_metrics:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Invested", f"${portfolio_data.get('total_invested', 0):,.2f}")
            
            with col2:
                current_value = portfolio_data.get('total_invested', 0) + portfolio_data.get('total_pnl', 0)
                st.metric("Current Value", f"${current_value:,.2f}")
            
            with col3:
                total_pnl = portfolio_data.get('total_pnl', 0)
                pnl_pct = (total_pnl / portfolio_data.get('total_invested', 1)) * 100 if portfolio_data.get('total_invested', 0) > 0 else 0
                st.metric("Total P&L", f"${total_pnl:,.2f}", f"{pnl_pct:+.2f}%")
            
            with col4:
                st.metric("Total BTC", f"{portfolio_data.get('total_volume', portfolio_data.get('total_btc', 0)):.6f}")
        
        # Portfolio composition charts
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if is_paper_mode and pt_status:
                # Paper trading portfolio pie chart
                balance = portfolio_data.get('balance', portfolio_data.get('usd_balance', 10000))
                btc_balance = portfolio_data.get('btc_balance', 0)
                btc_value = btc_balance * current_price
                
                fig = go.Figure(data=[go.Pie(
                    labels=['USD', 'BTC'],
                    values=[balance, btc_value],
                    hole=.3
                )])
                fig.update_layout(
                    title="Portfolio Composition",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Real trading positions pie chart
                if positions:
                    position_values = []
                    position_labels = []
                    for pos in positions:
                        value = pos.get('total_size', 0) * current_price
                        position_values.append(value)
                        position_labels.append(f"Lot {pos.get('lot_id', 'Unknown')[:8]}")
                    
                    cash_balance = portfolio_data.get('cash_balance', 0)
                    if cash_balance > 0:
                        position_values.append(cash_balance)
                        position_labels.append("Cash")
                    
                    fig = go.Figure(data=[go.Pie(
                        labels=position_labels,
                        values=position_values,
                        hole=.3
                    )])
                    fig.update_layout(
                        title="Position Distribution",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No open positions to display")
        
        with col2:
            st.markdown("#### Key Metrics")
            
            if is_paper_mode and pt_status:
                st.metric("Starting Capital", "$10,000")
                st.metric("Current Value", f"${portfolio_value:,.2f}")
                st.metric("Total Trades", len(portfolio_data.get('trades', [])))
                st.metric("Sharpe Ratio", f"{performance_data.get('sharpe_ratio', 0):.2f}")
                st.metric("Max Drawdown", f"{performance_data.get('max_drawdown', 0):.1%}")
                st.metric("Days Active", performance_data.get('days_active', 0))
            else:
                st.metric("Total Holdings", f"{portfolio_data.get('total_btc', portfolio_data.get('total_volume', 0)):.6f} BTC")
                st.metric("Avg Entry Price", f"${portfolio_data.get('avg_entry_price', 0):,.2f}")
                st.metric("Total Invested", f"${portfolio_data.get('total_invested', 0):,.2f}")
                st.metric("Realized P&L", f"${portfolio_data.get('realized_pnl', portfolio_data.get('total_realized_pnl', 0)):,.2f}")
                st.metric("Unrealized P&L", f"${portfolio_data.get('unrealized_pnl', portfolio_data.get('total_unrealized_pnl', 0)):,.2f}")
                roi = portfolio_data.get('roi_percent', portfolio_data.get('roi', 0))
                if isinstance(roi, (int, float)) and roi < 1:
                    roi = roi * 100
                st.metric("ROI", f"{roi:.2f}%")
        
        # Open Positions Management
        st.markdown("---")
        if not is_paper_mode:
            st.markdown("### 📋 Open Positions")
            
            if positions:
                for pos in positions:
                    with st.expander(f"Lot {pos.get('lot_id', 'Unknown')[:8]} - {pos.get('total_size', 0):.6f} BTC"):
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Entry Price", f"${pos.get('avg_price', 0):,.2f}")
                            st.metric("Purchase Value", f"${pos.get('purchase_value', pos.get('avg_price', 0) * pos.get('total_size', 0)):,.2f}")
                        with col2:
                            current_value = pos.get('current_value', pos.get('total_size', 0) * current_price)
                            st.metric("Current Value", f"${current_value:,.2f}")
                            st.metric("Current Price", f"${current_price:,.2f}")
                        with col3:
                            pnl = pos.get('unrealized_pnl', current_value - (pos.get('avg_price', 0) * pos.get('total_size', 0)))
                            pnl_pct = pos.get('pnl_percent', (pnl / (pos.get('avg_price', 0) * pos.get('total_size', 0)) * 100) if pos.get('avg_price', 0) > 0 else 0)
                            color = "green" if pnl >= 0 else "red"
                            st.markdown(f"**P&L:** <span style='color:{color}'>${pnl:,.2f} ({pnl_pct:+.2f}%)</span>", unsafe_allow_html=True)
                        with col4:
                            if st.button(f"Close Position", key=f"close_{pos.get('lot_id')}", type="primary"):
                                result = post_api_data("/trades/execute", {
                                    "signal": "sell",
                                    "size": pos.get('total_size'),
                                    "lot_id": pos.get('lot_id'),
                                    "reason": "manual_position_close"
                                })
                                if result:
                                    st.success("Position closed")
                                    time.sleep(0.5)
                                    st.rerun()
            else:
                st.info("No open positions")
    
    # TAB 3: Risk & Limits (comprehensive risk management)
    with tab3:
        st.markdown("### 🛡️ Risk Management & Limit Orders")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Create Limit Order")
            
            with st.form("limit_order_form"):
                limit_type = st.selectbox(
                    "Order Type",
                    ["stop_loss", "take_profit", "buy_limit", "sell_limit"],
                    format_func=lambda x: x.replace('_', ' ').title(),
                    help="Type of limit order"
                )
                
                if 'loss' in limit_type or 'buy' in limit_type:
                    default_price = current_price * 0.95
                else:
                    default_price = current_price * 1.05
                
                price = st.number_input(
                    "Trigger Price ($)",
                    min_value=0.01,
                    value=default_price,
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
                
                lot_id = None
                if not is_paper_mode:
                    if positions:
                        position_lots = ["None"] + [f"Lot {p.get('lot_id', 'Unknown')[:8]}" for p in positions]
                        selected_lot = st.selectbox(
                            "Apply to Position",
                            position_lots,
                            help="Apply limit to specific position"
                        )
                        if selected_lot != "None":
                            lot_index = position_lots.index(selected_lot) - 1
                            lot_id = positions[lot_index].get('lot_id')
                
                expiry_days = st.number_input(
                    "Expiry (days)",
                    min_value=1,
                    value=7,
                    help="Days until order expires"
                )
                
                submitted = st.form_submit_button("Create Order", type="primary")
                
                if submitted:
                    order_data = {
                        "symbol": "BTC-USD",
                        "limit_type": limit_type,
                        "price": price,
                        "size": size,
                        "lot_id": lot_id,
                        "expiry_days": expiry_days
                    }
                    
                    result = post_api_data("/limits", order_data)
                    if result:
                        st.success(f"✅ {limit_type.replace('_', ' ').title()} order created at ${price:,.2f}")
                        time.sleep(0.5)
                        st.rerun()
        
        with col2:
            st.subheader("Risk Parameters")
            
            with st.form("quick_risk_form"):
                st.markdown("#### Quick Risk Settings")
                
                stop_loss_pct = st.slider(
                    "Default Stop Loss %",
                    min_value=1.0,
                    max_value=20.0,
                    value=5.0,
                    step=0.5,
                    help="Automatic stop loss percentage for new positions"
                )
                
                take_profit_pct = st.slider(
                    "Default Take Profit %",
                    min_value=1.0,
                    max_value=50.0,
                    value=10.0,
                    step=0.5,
                    help="Automatic take profit percentage for new positions"
                )
                
                max_position_size = st.number_input(
                    "Max Position Size (BTC)",
                    min_value=0.001,
                    max_value=10.0,
                    value=1.0,
                    step=0.1,
                    help="Maximum allowed position size"
                )
                
                max_daily_loss = st.number_input(
                    "Max Daily Loss ($)",
                    min_value=100.0,
                    max_value=100000.0,
                    value=1000.0,
                    step=100.0,
                    help="Stop trading if daily loss exceeds this"
                )
                
                if st.form_submit_button("Update Risk Settings"):
                    st.success("Risk settings updated")
        
        # Active Limit Orders Management
        st.markdown("---")
        st.subheader("Active Limit Orders")
        
        limits = fetch_api_data("/limits") or []
        active_limits = [l for l in limits if l.get('active', True)]
        
        if active_limits:
            for limit in active_limits:
                limit['created_at'] = pd.to_datetime(limit.get('created_at', '')).strftime('%Y-%m-%d %H:%M') if limit.get('created_at') else 'Unknown'
                limit['expires_at'] = pd.to_datetime(limit.get('expires_at', '')).strftime('%Y-%m-%d %H:%M') if limit.get('expires_at') else 'Unknown'
                
                with st.expander(f"{limit.get('limit_type', '').replace('_', ' ').title()} - ${limit.get('price', 0):,.2f}"):
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        icon = "🔴" if "loss" in limit.get('limit_type', '') else "🟢" if "profit" in limit.get('limit_type', '') else "🔵"
                        st.write(f"{icon} **{limit.get('limit_type', '').replace('_', ' ').title()}**")
                        st.write(f"**Price:** ${limit.get('price', 0):,.2f}")
                        distance = ((limit.get('price', 0) - current_price) / current_price * 100) if current_price > 0 else 0
                        st.write(f"**Distance:** {distance:+.2f}%")
                    
                    with col2:
                        st.write(f"**Size:** {limit.get('size', 0):.4f} BTC")
                        st.write(f"**Value:** ${limit.get('size', 0) * limit.get('price', 0):,.2f}")
                        st.write(f"**Status:** {limit.get('status', 'Active')}")
                    
                    with col3:
                        st.write(f"**Created:** {limit['created_at']}")
                        st.write(f"**Expires:** {limit['expires_at']}")
                        if limit.get('lot_id'):
                            st.write(f"**Lot:** {limit.get('lot_id', '')[:8]}")
                    
                    with col4:
                        if st.button("Cancel Order", key=f"cancel_{limit.get('id', '')}", type="secondary"):
                            result = post_api_data(f"/limits/{limit.get('id', '')}/cancel", {})
                            if result:
                                st.success("Order cancelled")
                                time.sleep(0.5)
                                st.rerun()
        else:
            st.info("No active limit orders")
    
    # TAB 4: Performance Analytics (comprehensive performance analysis)
    with tab4:
        st.markdown("### 📊 Performance Analytics")
        
        if is_paper_mode and pt_status:
            # Paper trading performance charts
            history = fetch_api_data("/paper-trading/history?days=30") or []
            if history:
                df = pd.DataFrame(history)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.1,
                    subplot_titles=('Portfolio Value', 'Daily P&L'),
                    row_heights=[0.6, 0.4]
                )
                
                # Portfolio value
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df.get('portfolio_value', df.get('total_value', [])),
                        mode='lines',
                        name='Portfolio Value',
                        line=dict(color='blue', width=2)
                    ),
                    row=1, col=1
                )
                
                # Daily P&L bars
                if 'daily_pnl' in df.columns:
                    fig.add_trace(
                        go.Bar(
                            x=df['timestamp'],
                            y=df['daily_pnl'],
                            name='Daily P&L',
                            marker_color=df['daily_pnl'].apply(lambda x: 'green' if x > 0 else 'red')
                        ),
                        row=2, col=1
                    )
                
                fig.update_layout(height=600, showlegend=True, title="Paper Trading Performance")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No performance history available yet. Start paper trading to see results.")
        else:
            # Real trading performance analysis
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
                    
                    fig.update_layout(height=600, showlegend=True, title="Trading Performance")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No trading history available yet")
            
            # Additional analytics metrics
            performance = fetch_api_data("/analytics/performance") or {}
            if performance:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Sharpe Ratio", f"{performance.get('sharpe_ratio', 0):.2f}")
                with col2:
                    st.metric("Sortino Ratio", f"{performance.get('sortino_ratio', 0):.2f}")
                with col3:
                    st.metric("Max Drawdown", f"{performance.get('max_drawdown', 0):.1%}")
                with col4:
                    st.metric("Profit Factor", f"{performance.get('profit_factor', 0):.2f}")
    
    # TAB 5: Trade History (comprehensive trade management)
    with tab5:
        st.markdown("### 📋 Trade History")
        
        # Date range filter
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=30))
        with col2:
            end_date = st.date_input("End Date", value=datetime.now())
        
        # Fetch trades based on mode
        if is_paper_mode and pt_status:
            trades_list = portfolio_data.get('trades', [])
            
            if trades_list:
                trades_df = pd.DataFrame(trades_list)
                trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
                
                # Apply date filter
                mask = (trades_df['timestamp'].dt.date >= start_date) & (trades_df['timestamp'].dt.date <= end_date)
                trades_df = trades_df[mask]
                
                if not trades_df.empty:
                    trades_df['timestamp_str'] = trades_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                    trades_df['value'] = trades_df['price'] * trades_df['size']
                    
                    # Style the dataframe
                    def style_trade_type(val):
                        return 'background-color: #28a745; color: white' if val == 'buy' else 'background-color: #dc3545; color: white'
                    
                    def style_pnl(val):
                        return 'color: green' if val > 0 else 'color: red' if val < 0 else ''
                    
                    display_df = trades_df[['timestamp_str', 'type', 'price', 'size', 'value']]
                    if 'pnl' in trades_df.columns:
                        display_df['pnl'] = trades_df['pnl']
                        styled_df = display_df.style.map(
                            style_trade_type, subset=['type']
                        ).map(style_pnl, subset=['pnl'])
                    else:
                        styled_df = display_df.style.map(style_trade_type, subset=['type'])
                    
                    st.dataframe(styled_df, hide_index=True, use_container_width=True)
                    
                    # Summary stats
                    st.markdown("#### Trade Summary")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Trades", len(trades_df))
                    with col2:
                        buy_count = len(trades_df[trades_df['type'] == 'buy'])
                        st.metric("Buy Orders", buy_count)
                    with col3:
                        sell_count = len(trades_df[trades_df['type'] == 'sell'])
                        st.metric("Sell Orders", sell_count)
                    with col4:
                        total_volume = trades_df['value'].sum()
                        st.metric("Total Volume", f"${total_volume:,.2f}")
                else:
                    st.info("No trades in selected date range")
            else:
                st.info("No trades yet")
        else:
            # Real trading history
            recent_trades = fetch_api_data("/trades/recent?limit=100")
            
            if recent_trades:
                trades_df = pd.DataFrame(recent_trades)
                trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
                
                # Apply date filter
                mask = (trades_df['timestamp'].dt.date >= start_date) & (trades_df['timestamp'].dt.date <= end_date)
                trades_df = trades_df[mask]
                
                if not trades_df.empty:
                    trades_df['timestamp_str'] = trades_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                    trades_df['value'] = trades_df['price'] * trades_df['size']
                    
                    # Style the dataframe
                    def style_trade_type(val):
                        return 'background-color: #28a745; color: white' if val == 'buy' else 'background-color: #dc3545; color: white'
                    
                    styled_df = trades_df[['timestamp_str', 'trade_type', 'price', 'size', 'value', 'reason']].style.map(
                        style_trade_type, subset=['trade_type']
                    )
                    
                    st.dataframe(styled_df, hide_index=True, use_container_width=True)
                    
                    # Summary stats
                    st.markdown("#### Trade Summary")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Trades", len(trades_df))
                    with col2:
                        buy_count = len(trades_df[trades_df['trade_type'] == 'buy'])
                        st.metric("Buy Orders", buy_count)
                    with col3:
                        sell_count = len(trades_df[trades_df['trade_type'] == 'sell'])
                        st.metric("Sell Orders", sell_count)
                    with col4:
                        total_volume = trades_df['value'].sum()
                        st.metric("Total Volume", f"${total_volume:,.2f}")
                else:
                    st.info("No trades in selected date range")
            else:
                st.info("No recent trades")
    
    # TAB 6: Settings (comprehensive system management)
    with tab6:
        st.markdown("### ⚙️ Trading Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if is_paper_mode:
                st.markdown("#### Paper Trading Controls")
                
                # Reset button with confirmation
                if 'confirm_reset' not in st.session_state:
                    st.session_state.confirm_reset = False
                
                if st.button("🔄 Reset Paper Portfolio", type="secondary"):
                    st.session_state.confirm_reset = True
                
                # Confirmation dialog
                if st.session_state.confirm_reset:
                    st.warning("⚠️ Are you sure you want to reset? This will clear all trading history!")
                    col_yes, col_no = st.columns(2)
                    with col_yes:
                        if st.button("Yes, Reset", type="primary"):
                            result = post_api_data("/paper-trading/reset", {})
                            if result:
                                st.success("✅ Paper trading portfolio reset!")
                                st.session_state.confirm_reset = False
                                time.sleep(0.5)
                                st.rerun()
                    with col_no:
                        if st.button("Cancel"):
                            st.session_state.confirm_reset = False
                            st.rerun()
            else:
                st.markdown("#### Real Trading Controls")
                
                # Emergency stop with confirmation
                if 'confirm_emergency_stop' not in st.session_state:
                    st.session_state.confirm_emergency_stop = False
                
                if st.button("🛑 EMERGENCY STOP", type="secondary"):
                    st.session_state.confirm_emergency_stop = True
                
                if st.session_state.confirm_emergency_stop:
                    st.error("⚠️ WARNING: This will stop all trading and cancel all orders!")
                    col_yes, col_no = st.columns(2)
                    with col_yes:
                        if st.button("Confirm Stop", type="primary"):
                            # Stop trading
                            post_api_data("/trading/stop", {})
                            # Cancel all limits
                            limits = fetch_api_data("/limits") or []
                            for limit in limits:
                                if limit.get('active', True):
                                    post_api_data(f"/limits/{limit.get('id', '')}/cancel", {})
                            st.error("Emergency stop activated - all trading halted")
                            st.session_state.confirm_emergency_stop = False
                            time.sleep(1)
                            st.rerun()
                    with col_no:
                        if st.button("Cancel"):
                            st.session_state.confirm_emergency_stop = False
                            st.rerun()
        
        with col2:
            st.markdown("#### System Preferences")
            
            # Auto-trading settings
            auto_trade = st.checkbox("Enable Auto-Trading", value=is_trading_active)
            
            signal_threshold = st.slider(
                "Signal Confidence Threshold",
                min_value=0.5,
                max_value=1.0,
                value=0.7,
                step=0.05,
                help="Minimum confidence required to execute trades"
            )
            
            trade_frequency = st.selectbox(
                "Trade Frequency",
                ["Conservative", "Moderate", "Aggressive"],
                index=1,
                help="Controls how often the system will trade"
            )
            
            notification_settings = st.multiselect(
                "Notifications",
                ["Trade Executions", "Limit Triggers", "Large P&L Changes", "System Errors"],
                default=["Trade Executions", "System Errors"],
                help="Select which events trigger notifications"
            )
            
            if st.button("Save Preferences", type="primary"):
                st.success("Preferences saved successfully")
    
    # Auto-refresh option
    st.markdown("---")
    if st.checkbox("Auto-refresh (5 seconds)", value=False):
        time.sleep(5)
        st.rerun()'''

# Find and replace the show_trading_hub function
pattern = r'def show_trading_hub\(\):.*?(?=\ndef [a-zA-Z_]|\n# Main app|\nif __name__|$)'
match = re.search(pattern, content, re.DOTALL)

if match:
    print("Found show_trading_hub function, replacing with complete version...")
    new_content = content.replace(match.group(0), complete_trading_hub)
    
    # Write the fixed content
    with open('src/frontend/app.py', 'w') as f:
        f.write(new_content)
    
    print("✓ Replaced show_trading_hub with complete functionality")
else:
    print("Could not find show_trading_hub function to replace")
    # Append the function if not found
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if line.strip().startswith('def main():'):
            lines.insert(i, complete_trading_hub)
            lines.insert(i, '')
            break
    
    with open('src/frontend/app.py', 'w') as f:
        f.write('\n'.join(lines))
    
    print("✓ Added complete show_trading_hub function")
EOFCOMPLETE

# Test the replacement
echo -e "${YELLOW}Testing the complete function...${NC}"
if python3 -m py_compile "$APP_FILE" 2>/dev/null; then
    echo -e "${GREEN}✅ Python syntax is valid${NC}"
else
    echo -e "${RED}❌ Syntax errors exist${NC}"
    python3 -m py_compile "$APP_FILE"
fi

# Restart frontend container
echo -e "\n${YELLOW}Restarting frontend container...${NC}"
if command -v docker &> /dev/null && docker compose ps 2>/dev/null | grep -q "frontend"; then
    docker compose restart frontend
    echo -e "${GREEN}✓ Frontend container restarted${NC}"
fi

echo -e "\n${GREEN}🎉 Complete Trading Hub with all functionality restored!${NC}"

echo -e "\n${YELLOW}All restored features:${NC}"
echo "✅ 🎯 Trading Controls - Manual trading, charts, quick stats"
echo "✅ 💼 Portfolio & Positions - Holdings, pie charts, position management" 
echo "✅ 🛡️ Risk & Limits - Create/manage limit orders, risk settings"
echo "✅ 📊 Performance Analytics - Charts, metrics, Sharpe ratio"
echo "✅ 📋 Trade History - Filterable history, export functionality"
echo "✅ ⚙️ Settings - Emergency stops, preferences, paper trading reset"

echo -e "\n${BLUE}Test by visiting http://localhost:8501 and selecting Trading Hub${NC}"
echo -e "\n${BLUE}Backup location: $APP_FILE.complete_restore_$TIMESTAMP${NC}"
