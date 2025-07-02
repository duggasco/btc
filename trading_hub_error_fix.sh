#!/bin/bash

# Trading Hub UnboundLocalError Fix
# Fixes the pt_status variable scope issue

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}Fixing Trading Hub UnboundLocalError...${NC}"

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
echo -e "${YELLOW}Creating backup...${NC}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
cp "$APP_FILE" "$APP_FILE.backup_$TIMESTAMP"
echo -e "${GREEN}✓ Backup created: $APP_FILE.backup_$TIMESTAMP${NC}"

# Fix the pt_status variable scope issue
echo -e "${YELLOW}Applying fix...${NC}"

# Use Python to fix the variable scope issue more precisely
python3 << 'EOF'
import re

# Read the current app.py file
with open('src/frontend/app.py', 'r') as f:
    content = f.read()

# Find the show_trading_hub function and fix the variable scope
# Pattern to match the problematic section
pattern = r'(def show_trading_hub\(\):.*?# Fetch all necessary data.*?try:.*?except Exception as e:.*?return\s+)(if is_paper_mode:.*?pt_status = fetch_api_data\("/paper-trading/status"\) or \{\}.*?is_trading_active = trading_status\.get\(\'is_active\', False\))'

def fix_function(match):
    function_start = match.group(1)
    conditional_block = match.group(2)
    
    # Create the fixed version with pt_status initialized
    fixed_block = '''# Initialize pt_status for both modes
    pt_status = {}
    
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
        is_trading_active = trading_status.get('is_active', False)'''
    
    return function_start + fixed_block

# Apply the fix
fixed_content = re.sub(pattern, fix_function, content, flags=re.DOTALL)

# If the regex didn't match, try a simpler approach
if fixed_content == content:
    print("Regex approach didn't work, trying line-by-line fix...")
    
    lines = content.split('\n')
    fixed_lines = []
    in_trading_hub = False
    found_if_paper_mode = False
    
    for i, line in enumerate(lines):
        if 'def show_trading_hub():' in line:
            in_trading_hub = True
        
        if in_trading_hub and 'if is_paper_mode:' in line and not found_if_paper_mode:
            # Insert pt_status initialization before the if statement
            indent = len(line) - len(line.lstrip())
            spaces = ' ' * indent
            fixed_lines.append(f'{spaces}# Initialize pt_status for both modes')
            fixed_lines.append(f'{spaces}pt_status = {{}}')
            fixed_lines.append('')
            found_if_paper_mode = True
        
        fixed_lines.append(line)
        
        # Reset when we exit the function
        if in_trading_hub and line.strip().startswith('def ') and 'show_trading_hub' not in line:
            in_trading_hub = False
            found_if_paper_mode = False
    
    fixed_content = '\n'.join(fixed_lines)

# Write the fixed content back to the file
with open('src/frontend/app.py', 'w') as f:
    f.write(fixed_content)

print("✓ Applied pt_status variable scope fix")
EOF

echo -e "${GREEN}✓ Fix applied successfully${NC}"

# Verify the fix
echo -e "${YELLOW}Verifying fix...${NC}"
if grep -A 10 "def show_trading_hub():" "$APP_FILE" | grep -q "pt_status = {}"; then
    echo -e "${GREEN}✓ pt_status initialization found${NC}"
else
    echo -e "${YELLOW}⚠ Automatic fix may not have worked, applying manual fix...${NC}"
    
    # Manual fix as fallback
    python3 << 'EOF'
# Read the file
with open('src/frontend/app.py', 'r') as f:
    lines = f.readlines()

# Find the show_trading_hub function and add pt_status initialization
fixed_lines = []
for i, line in enumerate(lines):
    fixed_lines.append(line)
    
    # Look for the line with "try:" inside show_trading_hub function
    if 'try:' in line and i > 0:
        # Check if we're in the show_trading_hub function by looking back
        context = ''.join(lines[max(0, i-20):i])
        if 'def show_trading_hub():' in context and 'pt_status = {}' not in context:
            # Add pt_status initialization after the try block and before is_paper_mode check
            for j in range(i+1, min(len(lines), i+10)):
                if 'if is_paper_mode:' in lines[j]:
                    # Insert before this line
                    indent = len(lines[j]) - len(lines[j].lstrip())
                    spaces = ' ' * indent
                    fixed_lines.insert(-1 + (j-i), f'{spaces}# Initialize pt_status for both modes\n')
                    fixed_lines.insert(-1 + (j-i), f'{spaces}pt_status = {{}}\n')
                    fixed_lines.insert(-1 + (j-i), f'{spaces}\n')
                    break
            break

# Write back
with open('src/frontend/app.py', 'w') as f:
    f.writelines(fixed_lines)

print("✓ Manual fix applied")
EOF
fi

# Additional fix: Ensure portfolio_metrics is defined in both branches
echo -e "${YELLOW}Ensuring portfolio_metrics is defined...${NC}"
python3 << 'EOF'
with open('src/frontend/app.py', 'r') as f:
    content = f.read()

# Make sure portfolio_metrics is initialized
if 'portfolio_metrics = {}' not in content:
    # Add portfolio_metrics initialization
    content = content.replace(
        'pt_status = {}',
        'pt_status = {}\n    portfolio_metrics = {}'
    )

# Write back
with open('src/frontend/app.py', 'w') as f:
    f.write(content)

print("✓ Ensured portfolio_metrics is defined")
EOF

# Restart frontend container
echo -e "${YELLOW}Restarting frontend container...${NC}"
if command -v docker &> /dev/null && docker compose ps 2>/dev/null | grep -q "frontend"; then
    docker compose restart frontend
    echo -e "${GREEN}✓ Frontend container restarted${NC}"
else
    echo -e "${YELLOW}⚠ Docker not available or frontend not running${NC}"
fi

echo -e "\n${GREEN}🎉 Fix completed!${NC}"
echo -e "${BLUE}The UnboundLocalError should now be resolved.${NC}"
echo -e "${BLUE}Test by visiting http://localhost:8501 and selecting Trading Hub${NC}"

echo -e "\n${YELLOW}What was fixed:${NC}"
echo "✅ Added pt_status = {} initialization before conditional blocks"
echo "✅ Ensured portfolio_metrics is properly defined"
echo "✅ Fixed variable scope issue in show_trading_hub function"

echo -e "\n${BLUE}Backup location: $APP_FILE.backup_$TIMESTAMP${NC}"
