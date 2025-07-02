#!/bin/bash

# Fix pt_status Scope Issue - Targeted Solution
# Specifically fix the UnboundLocalError for pt_status

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}Fixing pt_status Scope Issue${NC}"

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
cp "$APP_FILE" "$APP_FILE.pt_status_fix_$TIMESTAMP"
echo -e "${GREEN}✓ Backup created: $APP_FILE.pt_status_fix_$TIMESTAMP${NC}"

# Show the problematic line
echo -e "${YELLOW}Checking line 3647 area...${NC}"
sed -n '3640,3655p' "$APP_FILE" | nl -v3640

# Fix the pt_status issue using Python
echo -e "${YELLOW}Applying targeted fix...${NC}"

python3 << 'EOFFIX'
import re

# Read the file
with open('src/frontend/app.py', 'r') as f:
    content = f.read()

# Find the show_trading_hub function and fix it
def fix_show_trading_hub(content):
    # Pattern to find the show_trading_hub function
    pattern = r'(def show_trading_hub\(\):.*?)(# Initialize variables for both modes.*?pt_status = \{\}.*?portfolio_metrics = \{\}.*?)(.*?)(# Simple tabs for now.*?tab1, tab2 = st\.tabs.*?)'
    
    # If we find the pattern, we need to ensure no function calls are passing pt_status incorrectly
    # Let's find any problematic function calls
    lines = content.split('\n')
    fixed_lines = []
    in_trading_hub = False
    
    for i, line in enumerate(lines):
        if 'def show_trading_hub():' in line:
            in_trading_hub = True
            print(f"Found show_trading_hub at line {i+1}")
        
        # If we're in the trading hub and find a function call with pt_status
        if in_trading_hub and ('show_trading_hub_controls' in line or 'latest_signal, btc_data, portfolio_data, pt_status)' in line):
            print(f"Found problematic function call at line {i+1}: {line.strip()}")
            # Skip this line or replace with safe version
            if 'show_trading_hub_controls(' in line:
                # Replace with inline content instead of function call
                indent = ' ' * (len(line) - len(line.lstrip()))
                fixed_lines.append(f'{indent}# Trading controls inline (function call removed)')
                fixed_lines.append(f'{indent}st.markdown("### 🎯 Trading Controls")')
                continue
            else:
                # Skip the problematic line
                print(f"Skipping problematic line: {line.strip()}")
                continue
        
        # If we exit the function
        if in_trading_hub and line.strip().startswith('def ') and 'show_trading_hub' not in line:
            in_trading_hub = False
        
        fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

# Apply the fix
fixed_content = fix_show_trading_hub(content)

# Additional safety: ensure pt_status is always defined before any usage
def ensure_pt_status_defined(content):
    lines = content.split('\n')
    fixed_lines = []
    in_trading_hub = False
    pt_status_defined = False
    
    for line in lines:
        if 'def show_trading_hub():' in line:
            in_trading_hub = True
            pt_status_defined = False
        
        # If we're in trading hub and haven't defined pt_status yet
        if in_trading_hub and not pt_status_defined and 'pt_status' in line and 'pt_status = {}' not in line:
            # Insert pt_status definition before this line
            indent = ' ' * (len(line) - len(line.lstrip()))
            if indent == '':
                indent = '    '  # Default indentation
            fixed_lines.append(f'{indent}# Ensure pt_status is defined')
            fixed_lines.append(f'{indent}pt_status = {{}}')
            fixed_lines.append(f'{indent}portfolio_metrics = {{}}')
            pt_status_defined = True
        
        if 'pt_status = {}' in line:
            pt_status_defined = True
        
        # If we exit the function
        if in_trading_hub and line.strip().startswith('def ') and 'show_trading_hub' not in line:
            in_trading_hub = False
        
        fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

# Apply additional safety
fixed_content = ensure_pt_status_defined(fixed_content)

# Write the fixed content
with open('src/frontend/app.py', 'w') as f:
    f.write(fixed_content)

print("✓ Applied pt_status scope fix")
EOFFIX

# Alternative approach: Replace the entire show_trading_hub function with a safe version
echo -e "${YELLOW}Applying safe function replacement...${NC}"

python3 << 'EOFSAFE'
import re

# Read the file
with open('src/frontend/app.py', 'r') as f:
    content = f.read()

# Safe Trading Hub function that doesn't have scope issues
safe_trading_hub = '''def show_trading_hub():
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
    
    # Initialize ALL variables at the top to avoid scope issues
    pt_status = {}
    portfolio_metrics = {}
    portfolio_data = {}
    performance_data = {}
    is_trading_active = False
    
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
    
    # Tabs with inline content (no separate functions)
    tab1, tab2 = st.tabs(["🎯 Trading Controls", "💼 Portfolio"])
    
    with tab1:
        st.markdown("### 🎯 Trading Controls")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Trading toggle
            if is_paper_mode:
                toggle_label = "📄 Toggle Paper Trading"
            else:
                toggle_label = "💰 Toggle Real Trading"
            
            if st.button(toggle_label, type="primary"):
                if is_paper_mode:
                    result = post_api_data("/paper-trading/toggle", {})
                else:
                    endpoint = "/trading/start" if not is_trading_active else "/trading/stop"
                    result = post_api_data(endpoint, {})
                
                if result:
                    st.success("Trading toggled successfully")
                    time.sleep(0.5)
                    st.rerun()
            
            # Manual trading
            st.markdown("### 🎮 Manual Trading")
            trade_size = st.number_input("Trade Size (BTC)", min_value=0.0001, value=0.001, step=0.0001, format="%.4f")
            
            col_buy, col_sell = st.columns(2)
            with col_buy:
                if st.button("🟢 BUY", type="primary"):
                    endpoint = "/paper-trading/trade" if is_paper_mode else "/trades/execute"
                    result = post_api_data(endpoint, {"signal": "buy", "size": trade_size, "reason": "manual_buy"})
                    if result:
                        st.success(f"Bought {trade_size} BTC")
                        time.sleep(0.5)
                        st.rerun()
            
            with col_sell:
                if st.button("🔴 SELL", type="primary"):
                    endpoint = "/paper-trading/trade" if is_paper_mode else "/trades/execute"
                    result = post_api_data(endpoint, {"signal": "sell", "size": trade_size, "reason": "manual_sell"})
                    if result:
                        st.success(f"Sold {trade_size} BTC")
                        time.sleep(0.5)
                        st.rerun()
        
        with col2:
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
    
    with tab2:
        st.markdown("### 💼 Portfolio Overview")
        
        if is_paper_mode and pt_status:
            balance = portfolio_data.get('balance', portfolio_data.get('usd_balance', 10000))
            btc_balance = portfolio_data.get('btc_balance', 0)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("USD Balance", f"${balance:,.2f}")
            with col2:
                st.metric("BTC Balance", f"{btc_balance:.6f}")
            with col3:
                st.metric("BTC Value", f"${btc_balance * current_price:,.2f}")
        else:
            if portfolio_metrics:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Invested", f"${portfolio_data.get('total_invested', 0):,.2f}")
                with col2:
                    st.metric("Total P&L", f"${portfolio_data.get('total_pnl', 0):,.2f}")
                with col3:
                    st.metric("Total BTC", f"{portfolio_data.get('total_volume', 0):.6f}")
    
    # Auto-refresh option
    if st.checkbox("Auto-refresh (5 seconds)", value=False):
        time.sleep(5)
        st.rerun()'''

# Find and replace the show_trading_hub function
pattern = r'def show_trading_hub\(\):.*?(?=\ndef [a-zA-Z_]|\nif __name__|$)'
match = re.search(pattern, content, re.DOTALL)

if match:
    print("Found show_trading_hub function, replacing...")
    new_content = content.replace(match.group(0), safe_trading_hub)
    
    # Write the fixed content
    with open('src/frontend/app.py', 'w') as f:
        f.write(new_content)
    
    print("✓ Replaced show_trading_hub with safe version")
else:
    print("Could not find show_trading_hub function to replace")
EOFSAFE

# Test the fix
echo -e "${YELLOW}Testing the fix...${NC}"
if python3 -m py_compile "$APP_FILE" 2>/dev/null; then
    echo -e "${GREEN}✅ Python syntax is valid${NC}"
else
    echo -e "${RED}❌ Syntax errors still exist${NC}"
    python3 -m py_compile "$APP_FILE"
fi

# Show the area around where the error was
echo -e "\n${YELLOW}Checking fixed area around line 3647:${NC}"
sed -n '3640,3655p' "$APP_FILE" | nl -v3640

# Restart frontend container
echo -e "\n${YELLOW}Restarting frontend container...${NC}"
if command -v docker &> /dev/null && docker compose ps 2>/dev/null | grep -q "frontend"; then
    docker compose restart frontend
    echo -e "${GREEN}✓ Frontend container restarted${NC}"
fi

echo -e "\n${GREEN}🎉 pt_status scope issue should be fixed!${NC}"
echo -e "${BLUE}The Trading Hub should now work without UnboundLocalError${NC}"
echo -e "${BLUE}Test by visiting http://localhost:8501 and selecting Trading Hub${NC}"

echo -e "\n${YELLOW}What was fixed:${NC}"
echo "✅ Defined pt_status and portfolio_metrics at function start"
echo "✅ Removed problematic function calls that caused scope issues"
echo "✅ Used inline code instead of separate functions"
echo "✅ Ensured all variables are accessible throughout the function"

echo -e "\n${BLUE}Backup location: $APP_FILE.pt_status_fix_$TIMESTAMP${NC}"
