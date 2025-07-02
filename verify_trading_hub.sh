#!/bin/bash

# Trading Hub Verification Script
# Check the current state and verify the fix

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}Trading Hub Verification Report${NC}"
echo "========================================"

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

echo -e "\n${YELLOW}1. Checking file structure...${NC}"
echo "   App file: $APP_FILE"
echo "   Pages dir: $PAGES_DIR"

if [ -f "$APP_FILE" ]; then
    echo -e "   ${GREEN}✓${NC} app.py exists"
else
    echo -e "   ${RED}✗${NC} app.py missing"
    exit 1
fi

echo -e "\n${YELLOW}2. Checking for incorrect Trading Hub page file...${NC}"
if [ -f "$PAGES_DIR/2_💰_Trading_Hub.py" ]; then
    echo -e "   ${RED}✗${NC} Incorrect Trading Hub page file still exists: $PAGES_DIR/2_💰_Trading_Hub.py"
    echo -e "   ${YELLOW}   This file should be removed as it conflicts with the selectbox navigation${NC}"
else
    echo -e "   ${GREEN}✓${NC} No conflicting Trading Hub page file"
fi

echo -e "\n${YELLOW}3. Checking selectbox navigation...${NC}"
if grep -q '"Trading Hub"' "$APP_FILE"; then
    echo -e "   ${GREEN}✓${NC} Trading Hub found in selectbox options"
else
    echo -e "   ${RED}✗${NC} Trading Hub NOT found in selectbox options"
    echo -e "   ${YELLOW}   Expected to find 'Trading Hub' in the selectbox list${NC}"
fi

echo -e "\n${YELLOW}4. Checking Trading Hub function...${NC}"
if grep -q "def show_trading_hub" "$APP_FILE"; then
    echo -e "   ${GREEN}✓${NC} show_trading_hub() function found"
else
    echo -e "   ${RED}✗${NC} show_trading_hub() function NOT found"
fi

echo -e "\n${YELLOW}5. Checking routing...${NC}"
if grep -q 'elif page == "Trading Hub":' "$APP_FILE"; then
    echo -e "   ${GREEN}✓${NC} Trading Hub routing found"
else
    echo -e "   ${RED}✗${NC} Trading Hub routing NOT found"
fi

echo -e "\n${YELLOW}6. Checking current selectbox options...${NC}"
selectbox_line=$(grep -n "Select Page" -A 10 "$APP_FILE" | grep -E '\[.*\]' | head -1)
if [ -n "$selectbox_line" ]; then
    echo "   Current options: $selectbox_line"
else
    echo -e "   ${YELLOW}⚠${NC} Could not extract selectbox options"
fi

echo -e "\n${YELLOW}7. Checking navigation structure...${NC}"
if grep -q "page = st.selectbox" "$APP_FILE"; then
    echo -e "   ${GREEN}✓${NC} Uses selectbox navigation (correct)"
else
    echo -e "   ${YELLOW}⚠${NC} Selectbox navigation not found"
fi

# Check if pages directory has any conflicting files
echo -e "\n${YELLOW}8. Checking pages directory...${NC}"
if [ -d "$PAGES_DIR" ]; then
    echo "   Pages directory contents:"
    ls -la "$PAGES_DIR"/*.py 2>/dev/null | awk '{print "     " $9}' || echo -e "     ${BLUE}(no .py files)${NC}"
else
    echo -e "   ${BLUE}ℹ${NC} No pages directory (this is fine)"
fi

echo -e "\n${YELLOW}9. Docker status...${NC}"
if command -v docker &> /dev/null && docker compose ps 2>/dev/null | grep -q "frontend"; then
    frontend_status=$(docker compose ps | grep frontend | awk '{print $4}' | head -1)
    echo -e "   ${GREEN}✓${NC} Frontend container status: $frontend_status"
    
    if [ "$frontend_status" = "running" ]; then
        echo -e "   ${GREEN}✓${NC} Frontend is running on http://localhost:8501"
    else
        echo -e "   ${YELLOW}⚠${NC} Frontend container not running"
    fi
else
    echo -e "   ${YELLOW}⚠${NC} Docker not available or frontend container not found"
fi

echo -e "\n${YELLOW}Summary:${NC}"
echo "========"

# Determine overall status
issues=0

if [ -f "$PAGES_DIR/2_💰_Trading_Hub.py" ]; then
    echo -e "${RED}❌ Issue: Conflicting Trading Hub page file exists${NC}"
    issues=$((issues + 1))
fi

if ! grep -q '"Trading Hub"' "$APP_FILE"; then
    echo -e "${RED}❌ Issue: Trading Hub not in selectbox${NC}"
    issues=$((issues + 1))
fi

if ! grep -q "def show_trading_hub" "$APP_FILE"; then
    echo -e "${RED}❌ Issue: show_trading_hub() function missing${NC}"
    issues=$((issues + 1))
fi

if ! grep -q 'elif page == "Trading Hub":' "$APP_FILE"; then
    echo -e "${RED}❌ Issue: Trading Hub routing missing${NC}"
    issues=$((issues + 1))
fi

if [ $issues -eq 0 ]; then
    echo -e "${GREEN}✅ All checks passed! Trading Hub should be working correctly.${NC}"
    echo -e "${BLUE}To test: Go to http://localhost:8501 and select 'Trading Hub' from the sidebar dropdown${NC}"
else
    echo -e "${RED}❌ Found $issues issues that need to be fixed${NC}"
    echo -e "${BLUE}Run the Trading Hub fix script to resolve these issues${NC}"
fi
