#!/bin/bash

# Trading Hub Integration Fix - Proper Implementation
# This script fixes the Trading Hub to work with the existing selectbox navigation

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'
BOLD='\033[1m'

echo -e "${PURPLE}${BOLD}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║              Trading Hub Integration Fix                     ║"
echo "║         Properly integrate with selectbox navigation         ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Verify we're in project root
if [ ! -f "docker-compose.yml" ] && [ ! -f "docker-compose.yaml" ]; then
    echo -e "${RED}ERROR: Not in project root!${NC}"
    echo "Run this from your project root (where docker-compose.yml is)"
    exit 1
fi

PROJECT_ROOT=$(pwd)
echo -e "${GREEN}✓ Project root: $PROJECT_ROOT${NC}"

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
PAGES_DIR="$FRONTEND_DIR/pages"
echo -e "${GREEN}✓ Frontend directory: $FRONTEND_DIR${NC}"

# Create backup
echo -e "\n${YELLOW}Creating backup...${NC}"
BACKUP_DIR="backups/trading_hub_fix_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

cp "$APP_FILE" "$BACKUP_DIR/app.py.backup"
if [ -d "$PAGES_DIR" ]; then
    cp -r "$PAGES_DIR" "$BACKUP_DIR/"
fi
echo -e "${GREEN}✓ Backup created: $BACKUP_DIR${NC}"

# Remove the incorrectly created Trading Hub page file
echo -e "\n${YELLOW}Cleaning up incorrect page file...${NC}"
if [ -f "$PAGES_DIR/2_💰_Trading_Hub.py" ]; then
    rm -f "$PAGES_DIR/2_💰_Trading_Hub.py"
    echo -e "${GREEN}✓ Removed incorrect Trading Hub page file${NC}"
fi

# Check if Trading Hub already exists in selectbox
echo -e "\n${YELLOW}Checking current app.py structure...${NC}"
if grep -q "Trading Hub" "$APP_FILE"; then
    echo -e "${YELLOW}⚠ Trading Hub already exists in selectbox. Updating function...${NC}"
    TRADING_HUB_EXISTS=true
else
    echo -e "${BLUE}ℹ Adding Trading Hub to selectbox navigation...${NC}"
    TRADING_HUB_EXISTS=false
fi

# Create the Trading Hub function to add to app.py
echo -e "\n${YELLOW}Creating Trading Hub function...${NC}"

cat > /tmp/trading_hub_function.py << 'EOFHUBFUNC'

def show_trading_hub():
    """Unified Trading Hub - All trading functionality in one place"""
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
    
    # Fetch all necessary data
    try:
        trading_status = fetch_api_data("/trading/status") or {}
        latest_signal = fetch_api_data("/signals/enhanced/latest") or fetch_api_data("/signals/latest") or {}
        btc_data = fetch_api_data("/btc/latest") or {}
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        st.info("Please ensure the backend is running")
        return
    
    if is_paper_mode:
        pt_status = fetch_api_data("/paper-trading/status") or {}
        portfolio_data = pt_status.get('portfolio', {})
        performance_data = pt_status.get('performance', {})
        is_trading_active = pt_status.get('enabled', False)
    else:
        portfolio_metrics = fetch_api_data("/portfolio/metrics") or {}
        positions = fetch_api_data("/portfolio/positions") or []
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
            trades = portfolio_data.get('trades', [])
            if trades:
                winning_trades = sum(1 for t in trades if t.get('pnl', 0) > 0)
                win_rate = (winning_trades / len(trades)) if trades else 0
        st.metric("Win Rate", f"{win_rate:.1%}")
    
    # Main Content Area with Enhanced Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Trading Controls",
        "💼 Portfolio & Positions", 
        "🛡️ Risk Management",
        "📊 Performance"
    ])
    
    with tab1:
        show_trading_hub_controls(is_paper_mode, is_trading_active, current_price, 
                                  latest_signal, btc_data, portfolio_data, pt_status)
    
    with tab2:
        show_trading_hub_portfolio(is_paper_mode, portfolio_data, performance_data, 
                                   current_price, pt_status, portfolio_metrics)
    
    with tab3:
        show_trading_hub_risk(is_paper_mode, current_price)
    
    with tab4:
        show_trading_hub_performance(is_paper_mode, portfolio_data, pt_status)
    
    # Auto-refresh option
    st.markdown("---")
    if st.checkbox("Auto-refresh (5 seconds)", value=False):
        time.sleep(5)
        st.rerun()

def show_trading_hub_controls(is_paper_mode, is_trading_active, current_price, 
                              latest_signal, btc_data, portfolio_data, pt_status):
    """Trading controls tab"""
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
        else:
            positions = fetch_api_data("/portfolio/positions") or []
            st.info(f"Open Positions: {len(positions)}")
            total_btc = sum(p.get('total_size', 0) for p in positions)
            st.info(f"Total BTC: {total_btc:.6f}")
    
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

def show_trading_hub_portfolio(is_paper_mode, portfolio_data, performance_data, 
                               current_price, pt_status, portfolio_metrics):
    """Portfolio tab"""
    st.markdown("### 💼 Portfolio Overview")
    
    if is_paper_mode and pt_status:
        # Paper trading portfolio
        balance = portfolio_data.get('balance', portfolio_data.get('usd_balance', 10000))
        btc_balance = portfolio_data.get('btc_balance', 0)
        btc_value = btc_balance * current_price
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("USD Balance", f"${balance:,.2f}")
        with col2:
            st.metric("BTC Balance", f"{btc_balance:.6f}")
        with col3:
            st.metric("BTC Value", f"${btc_value:,.2f}")
        
        # Portfolio composition pie chart
        if balance > 0 or btc_value > 0:
            fig = go.Figure(data=[go.Pie(
                labels=['USD', 'BTC'],
                values=[balance, btc_value],
                hole=.3
            )])
            fig.update_layout(title="Portfolio Composition", height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Recent trades
        trades = portfolio_data.get('trades', [])
        if trades:
            st.markdown("### 📋 Recent Trades")
            recent_trades = trades[-10:][::-1]  # Last 10, most recent first
            
            for trade in recent_trades:
                with st.expander(f"{trade.get('type', 'Unknown').upper()} - {trade.get('timestamp', '')}"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write(f"**Price:** ${trade.get('price', 0):,.2f}")
                    with col2:
                        st.write(f"**Amount:** {trade.get('amount', 0):.6f} BTC")
                    with col3:
                        st.write(f"**Value:** ${trade.get('value', 0):,.2f}")
    else:
        # Real trading portfolio
        if portfolio_metrics:
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
            
            # Open positions
            positions = fetch_api_data("/portfolio/positions") or []
            if positions:
                st.markdown("### 📋 Open Positions")
                for pos in positions:
                    with st.expander(f"Position {pos.get('lot_id', 'Unknown')[:8]}"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.write(f"**Size:** {pos.get('total_size', 0):.6f} BTC")
                        with col2:
                            st.write(f"**Avg Price:** ${pos.get('avg_price', 0):,.2f}")
                        with col3:
                            pnl = pos.get('unrealized_pnl', 0)
                            color = "green" if pnl >= 0 else "red"
                            st.markdown(f"**P&L:** <span style='color:{color}'>${pnl:,.2f}</span>", unsafe_allow_html=True)

def show_trading_hub_risk(is_paper_mode, current_price):
    """Risk management tab"""
    st.markdown("### 🛡️ Risk Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Create Limit Order")
        
        with st.form("quick_limit_form"):
            limit_type = st.selectbox(
                "Order Type",
                ["stop_loss", "take_profit", "buy_limit", "sell_limit"],
                format_func=lambda x: x.replace('_', ' ').title()
            )
            
            if 'loss' in limit_type or 'buy' in limit_type:
                default_price = current_price * 0.95
            else:
                default_price = current_price * 1.05
            
            price = st.number_input(
                "Trigger Price ($)",
                min_value=0.01,
                value=default_price,
                step=100.0
            )
            
            size = st.number_input(
                "Size (BTC)",
                min_value=0.0001,
                value=0.001,
                step=0.0001,
                format="%.4f"
            )
            
            if st.form_submit_button("Create Order", type="primary"):
                order_data = {
                    "symbol": "BTC-USD",
                    "limit_type": limit_type,
                    "price": price,
                    "size": size
                }
                
                result = post_api_data("/limits", order_data)
                if result:
                    st.success(f"✅ {limit_type.replace('_', ' ').title()} order created")
                    time.sleep(0.5)
                    st.rerun()
    
    with col2:
        st.subheader("Active Limits")
        limits = fetch_api_data("/limits") or []
        active_limits = [l for l in limits if l.get('active', True)]
        
        if active_limits:
            for limit in active_limits:
                icon = "🔴" if "loss" in limit.get('limit_type', '') else "🟢"
                distance = ((limit.get('price', 0) - current_price) / current_price * 100) if current_price > 0 else 0
                st.write(f"{icon} {limit.get('limit_type', '').replace('_', ' ').title()}: ${limit.get('price', 0):,.2f} ({distance:+.2f}%)")
        else:
            st.info("No active limit orders")

def show_trading_hub_performance(is_paper_mode, portfolio_data, pt_status):
    """Performance analytics tab"""
    st.markdown("### 📊 Performance Analytics")
    
    if is_paper_mode and pt_status:
        performance = pt_status.get('performance', {})
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Return", f"{performance.get('total_return', 0):.1%}")
        with col2:
            st.metric("Win Rate", f"{performance.get('win_rate', 0):.1%}")
        with col3:
            st.metric("Sharpe Ratio", f"{performance.get('sharpe_ratio', 0):.2f}")
        with col4:
            st.metric("Max Drawdown", f"{performance.get('max_drawdown', 0):.1%}")
        
        # Performance chart
        history = fetch_api_data("/paper-trading/history?days=30") or []
        if history:
            df = pd.DataFrame(history)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df.get('total_value', df.get('portfolio_value', [])),
                mode='lines',
                name='Portfolio Value',
                line=dict(color='blue', width=2)
            ))
            
            fig.update_layout(
                title="Portfolio Performance",
                height=400,
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Performance analytics available for paper trading and with trade history")

EOFHUBFUNC

echo -e "${GREEN}✓ Trading Hub function created${NC}"

# Now update app.py to include Trading Hub
echo -e "\n${YELLOW}Updating app.py...${NC}"

# Backup current app.py again
cp "$APP_FILE" "$APP_FILE.pre_trading_hub"

# Add Trading Hub to selectbox if it doesn't exist
if [ "$TRADING_HUB_EXISTS" = false ]; then
    # Add Trading Hub to the selectbox options
    sed -i.bak 's/\["Dashboard", "Trading", "Portfolio/["Dashboard", "Trading Hub", "Trading", "Portfolio/' "$APP_FILE"
    echo -e "${GREEN}✓ Added Trading Hub to selectbox navigation${NC}"
fi

# Add the Trading Hub function to app.py (append before main function)
# Find the line with 'def main():' and insert the trading hub functions before it
awk '
/^def main\(\):/ {
    while ((getline line < "/tmp/trading_hub_function.py") > 0) {
        print line
    }
    close("/tmp/trading_hub_function.py")
    print ""
}
{print}
' "$APP_FILE" > "$APP_FILE.tmp" && mv "$APP_FILE.tmp" "$APP_FILE"

# Add Trading Hub routing to the page routing section
if ! grep -q "elif page == \"Trading Hub\":" "$APP_FILE"; then
    # Find the line with 'elif page == "Trading":' and add Trading Hub before it
    sed -i.bak '/elif page == "Trading":/i\
    elif page == "Trading Hub":\
        show_trading_hub()' "$APP_FILE"
    echo -e "${GREEN}✓ Added Trading Hub routing${NC}"
fi

# Clean up temporary files
rm -f /tmp/trading_hub_function.py
rm -f "$APP_FILE.bak" 2>/dev/null || true

# Verify the integration
echo -e "\n${YELLOW}Verifying integration...${NC}"

if grep -q "Trading Hub" "$APP_FILE" && grep -q "show_trading_hub" "$APP_FILE"; then
    echo -e "${GREEN}✓ Trading Hub successfully integrated into app.py${NC}"
else
    echo -e "${RED}✗ Integration verification failed${NC}"
    echo -e "${YELLOW}Restoring from backup...${NC}"
    cp "$APP_FILE.pre_trading_hub" "$APP_FILE"
    exit 1
fi

# Restart containers
echo -e "\n${YELLOW}Restarting containers...${NC}"
cd "$PROJECT_ROOT"

if docker compose ps | grep -q "frontend"; then
    docker compose restart frontend
    echo -e "${GREEN}✓ Frontend container restarted${NC}"
else
    echo -e "${YELLOW}⚠ Frontend not running. Start with: docker compose up -d${NC}"
fi

# Success message
echo -e "\n${GREEN}${BOLD}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                ✅ TRADING HUB FIX SUCCESSFUL!                ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo -e "${BLUE}What was fixed:${NC}"
echo "✅ Removed incorrect pages/2_💰_Trading_Hub.py file"
echo "✅ Added 'Trading Hub' to selectbox navigation in app.py"
echo "✅ Created show_trading_hub() function with 4 comprehensive tabs"
echo "✅ Integrated with existing app.py navigation system"
echo "✅ Added proper routing for Trading Hub page"

echo -e "\n${BLUE}Trading Hub Features:${NC}"
echo "🎯 Trading Controls - Manual trading, auto-trading toggle"
echo "💼 Portfolio & Positions - View holdings and performance"
echo "🛡️ Risk Management - Create and manage limit orders"
echo "📊 Performance - Analytics and charts"

echo -e "\n${BLUE}Next Steps:${NC}"
echo "1. Wait 10-15 seconds for container restart"
echo "2. Open http://localhost:8501"
echo "3. In the sidebar, select 'Trading Hub' from dropdown"
echo "4. Test all functionality in the 4 tabs"

echo -e "\n${BLUE}Backup location:${NC} $BACKUP_DIR"
echo -e "\n${GREEN}🎉 Trading Hub is now properly integrated!${NC}"
