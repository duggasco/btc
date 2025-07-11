#!/bin/bash

# Script to run Data Quality tab tests in Docker environment

echo "=== Data Quality Tab Test Runner ==="
echo "Running automated tests for the Data Quality tab in Settings page"
echo ""

# Check if Docker containers are running
if ! docker compose -f docker/docker-compose.yml ps | grep -q "Up"; then
    echo "Error: Docker containers are not running!"
    echo "Please start them with: docker compose -f docker/docker-compose.yml up -d"
    exit 1
fi

# Wait for services to be ready
echo "Waiting for services to be ready..."
sleep 5

# Install test dependencies if needed
echo "Installing test dependencies..."
docker exec btc-trading-backend pip install selenium pytest pytest-timeout

# Run the automated tests
echo ""
echo "Running automated Selenium tests..."
docker exec btc-trading-backend python -m pytest /app/tests/e2e/test_data_quality_tab.py -v --tb=short

# Check test results
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ All automated tests passed!"
else
    echo ""
    echo "❌ Some tests failed. Please check the output above."
fi

# Run a quick visual test by opening the page
echo ""
echo "=== Visual Test Instructions ==="
echo "1. Open your browser to: http://localhost:8501"
echo "2. Navigate to Settings page (⚙️ Settings)"
echo "3. Click on the 'Data Quality' tab (last tab)"
echo "4. Use the manual checklist at: tests/manual/data_quality_tab_checklist.md"
echo ""
echo "For comprehensive testing, please follow the manual test checklist."

# Optional: Generate test report
echo ""
echo "Generating test report..."
REPORT_FILE="tests/reports/data_quality_tab_test_report_$(date +%Y%m%d_%H%M%S).txt"
mkdir -p tests/reports

cat > "$REPORT_FILE" << EOF
Data Quality Tab Test Report
Generated: $(date)

Automated Test Results:
- Selenium E2E tests: $([ $? -eq 0 ] && echo "PASSED" || echo "FAILED")

Manual Testing Required:
- Visual appearance verification
- Responsive design testing
- Cross-browser compatibility
- Performance testing
- Accessibility testing

Please complete manual testing using: tests/manual/data_quality_tab_checklist.md
EOF

echo "Test report saved to: $REPORT_FILE"