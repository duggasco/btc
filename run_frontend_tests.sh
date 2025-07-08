#!/bin/bash

# Run frontend functionality tests in Docker

echo "======================================"
echo "Running Frontend Functionality Tests"
echo "======================================"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Create test reports directory
mkdir -p test_reports

echo -e "${YELLOW}Building test containers...${NC}"
docker compose -f docker-compose.test-frontend.yml build

echo -e "${YELLOW}Starting test environment...${NC}"
docker compose -f docker-compose.test-frontend.yml up -d backend frontend

# Wait for services to be healthy
echo -e "${YELLOW}Waiting for services to be ready...${NC}"
sleep 10

# Check if services are healthy
BACKEND_HEALTH=$(docker inspect btc-trading-backend-test --format='{{.State.Health.Status}}' 2>/dev/null)
FRONTEND_HEALTH=$(docker inspect btc-trading-frontend-test --format='{{.State.Health.Status}}' 2>/dev/null)

if [ "$BACKEND_HEALTH" != "healthy" ]; then
    echo -e "${RED}Backend is not healthy! Status: $BACKEND_HEALTH${NC}"
    echo "Checking backend logs:"
    docker logs btc-trading-backend-test --tail 20
    exit 1
fi

if [ "$FRONTEND_HEALTH" != "healthy" ]; then
    echo -e "${RED}Frontend is not healthy! Status: $FRONTEND_HEALTH${NC}"
    echo "Checking frontend logs:"
    docker logs btc-trading-frontend-test --tail 20
    exit 1
fi

echo -e "${GREEN}Services are healthy!${NC}"

# Run the tests
echo -e "${YELLOW}Running frontend tests...${NC}"
docker compose -f docker-compose.test-frontend.yml run --rm test-frontend

# Copy test reports
echo -e "${YELLOW}Copying test reports...${NC}"
docker cp btc-test-frontend:/app/test_report_*.json ./test_reports/ 2>/dev/null || true

# Show test summary
if [ -f test_reports/test_report_*.json ]; then
    echo -e "${GREEN}Test reports saved in ./test_reports/${NC}"
    # Parse and display summary from latest report
    LATEST_REPORT=$(ls -t test_reports/test_report_*.json | head -1)
    if [ -f "$LATEST_REPORT" ]; then
        echo -e "${YELLOW}Test Summary:${NC}"
        python3 -c "
import json
with open('$LATEST_REPORT', 'r') as f:
    data = json.load(f)
    print(f\"Total Tests: {data['total_tests']}\")
    print(f\"Passed: {data['passed']}\")
    print(f\"Failed: {data['failed']}\")
    print(f\"Pass Rate: {data['pass_rate']:.1f}%\")
"
    fi
fi

# Optionally view logs
echo -e "${YELLOW}Do you want to view the logs? (y/n)${NC}"
read -r VIEW_LOGS

if [ "$VIEW_LOGS" = "y" ]; then
    echo -e "${YELLOW}Backend logs:${NC}"
    docker logs btc-trading-backend-test --tail 50
    echo -e "${YELLOW}Frontend logs:${NC}"
    docker logs btc-trading-frontend-test --tail 50
fi

# Cleanup
echo -e "${YELLOW}Cleaning up test environment...${NC}"
docker compose -f docker-compose.test-frontend.yml down

echo -e "${GREEN}Testing complete!${NC}"