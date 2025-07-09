#!/bin/bash

# Complete test runner for BTC Trading System

echo "=============================================="
echo "BTC Trading System - Complete Test Suite"
echo "=============================================="

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Clean up any existing test containers
echo -e "${YELLOW}Cleaning up existing test containers...${NC}"
docker compose -f docker-compose.test-complete.yml down -v 2>/dev/null || true

# Create test reports directory
mkdir -p test_reports

# Build fresh images
echo -e "${YELLOW}Building backend image with latest changes...${NC}"
docker compose -f docker-compose.test-complete.yml build backend

echo -e "${YELLOW}Building test runner image...${NC}"
docker compose -f docker-compose.test-complete.yml build test-runner

# Start backend
echo -e "${YELLOW}Starting backend service...${NC}"
docker compose -f docker-compose.test-complete.yml up -d backend

# Wait for backend to be healthy
echo -e "${YELLOW}Waiting for backend to be healthy...${NC}"
RETRIES=30
RETRY_COUNT=0
while [ $RETRY_COUNT -lt $RETRIES ]; do
    HEALTH=$(docker inspect btc-backend-test --format='{{.State.Health.Status}}' 2>/dev/null)
    if [ "$HEALTH" == "healthy" ]; then
        echo -e "${GREEN}Backend is healthy!${NC}"
        break
    fi
    echo -n "."
    sleep 2
    RETRY_COUNT=$((RETRY_COUNT + 1))
done

if [ $RETRY_COUNT -eq $RETRIES ]; then
    echo -e "${RED}Backend failed to become healthy!${NC}"
    echo "Backend logs:"
    docker logs btc-backend-test --tail 50
    exit 1
fi

# Run tests
echo -e "${YELLOW}Running API tests...${NC}"
docker compose -f docker-compose.test-complete.yml run --rm test-runner

# Check if test reports were generated
REPORT_COUNT=$(ls -1 test_reports/api_test_report_*.json 2>/dev/null | wc -l)
if [ $REPORT_COUNT -gt 0 ]; then
    echo -e "${GREEN}Test reports generated successfully!${NC}"
    
    # Display summary from latest report
    LATEST_REPORT=$(ls -t test_reports/api_test_report_*.json | head -1)
    echo -e "\n${BLUE}Test Summary:${NC}"
    python3 -c "
import json
import sys

try:
    with open('$LATEST_REPORT', 'r') as f:
        data = json.load(f)
        summary = data['summary']
        
        # Color codes
        GREEN = '\033[0;32m'
        RED = '\033[0;31m'
        YELLOW = '\033[1;33m'
        NC = '\033[0m'
        
        # Calculate color based on pass rate
        pass_rate = summary['pass_rate']
        if pass_rate >= 80:
            color = GREEN
        elif pass_rate >= 50:
            color = YELLOW
        else:
            color = RED
        
        print(f'Total Tests: {summary[\"total_tests\"]}')
        print(f'{GREEN}Passed: {summary[\"passed\"]}{NC}')
        print(f'{RED}Failed: {summary[\"failed\"]}{NC}')
        print(f'{color}Pass Rate: {pass_rate:.1f}%{NC}')
        
        # Show failed categories if any
        if summary['failed'] > 0:
            print(f'\n{RED}Failed Categories:{NC}')
            categories = data.get('categories', {})
            for cat, info in categories.items():
                if info['failed'] > 0:
                    print(f'  - {cat}: {info[\"failed\"]} failures')
                    
except Exception as e:
    print(f'Error reading report: {e}', file=sys.stderr)
"
else
    echo -e "${RED}No test reports generated!${NC}"
fi

# Optionally show logs
echo -e "\n${YELLOW}Show container logs? (y/n)${NC}"
read -n 1 -r SHOW_LOGS
echo
if [[ $SHOW_LOGS =~ ^[Yy]$ ]]; then
    echo -e "\n${BLUE}Backend logs:${NC}"
    docker logs btc-backend-test --tail 30
fi

# Cleanup
echo -e "\n${YELLOW}Cleaning up test environment...${NC}"
docker compose -f docker-compose.test-complete.yml down

echo -e "\n${GREEN}Test suite completed!${NC}"

# Exit with appropriate code based on test results
if [ $REPORT_COUNT -gt 0 ]; then
    # Check pass rate from latest report
    PASS_RATE=$(python3 -c "
import json
try:
    with open('$LATEST_REPORT', 'r') as f:
        data = json.load(f)
        print(int(data['summary']['pass_rate']))
except:
    print(0)
")
    if [ $PASS_RATE -ge 70 ]; then
        exit 0
    else
        exit 1
    fi
else
    exit 1
fi