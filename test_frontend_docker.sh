#!/bin/bash

# Run frontend tests using Docker

echo "========================================"
echo "Running Frontend Functionality Tests"
echo "========================================"

# Ensure containers are running
echo "Checking if containers are running..."
if ! docker ps | grep -q btc-trading-backend; then
    echo "Starting backend container..."
    docker compose up -d backend
    sleep 10
fi

if ! docker ps | grep -q btc-trading-frontend; then
    echo "Starting frontend container..."
    docker compose up -d frontend
    sleep 10
fi

# Create test reports directory
mkdir -p test_reports

# Run tests in a temporary container
echo "Running comprehensive API tests..."
docker run --rm \
    --network btc_trading-network \
    -e API_BASE_URL=http://btc-trading-backend:8000 \
    -v $(pwd)/test_reports:/app/reports \
    btc-test-frontend \
    python /app/test_all_endpoints.py

echo "Tests completed!"
echo "Check test reports in ./test_reports/"

# Show summary if report exists
LATEST_REPORT=$(ls -t test_reports/api_test_report_*.json 2>/dev/null | head -1)
if [ -f "$LATEST_REPORT" ]; then
    echo ""
    echo "Test Summary:"
    python3 -c "
import json
with open('$LATEST_REPORT', 'r') as f:
    data = json.load(f)
    summary = data['summary']
    print(f\"Total Tests: {summary['total_tests']}\")
    print(f\"Passed: {summary['passed']}\")
    print(f\"Failed: {summary['failed']}\")
    print(f\"Pass Rate: {summary['pass_rate']:.1f}%\")
"
fi