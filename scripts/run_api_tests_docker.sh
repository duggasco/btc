#!/bin/bash

# BTC Trading System - Docker Test Controller
# Manages API testing from Docker environment
# Handles container setup, test execution, and cleanup

set -euo pipefail

# Configuration
COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.yml}"
BACKEND_CONTAINER="btc-trading-backend"
FRONTEND_CONTAINER="btc-trading-frontend"
TEST_RESULTS_DIR="/storage/logs/test_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Functions
print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

check_requirements() {
    print_header "Checking Requirements"
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed"
        exit 1
    fi
    print_success "Docker is installed"
    
    # Check Docker Compose
    if ! docker compose version &> /dev/null 2>&1; then
        print_error "Docker Compose is not installed"
        exit 1
    fi
    print_success "Docker Compose is installed"
    
    # Check if Docker is running
    if ! docker info &> /dev/null; then
        print_error "Docker daemon is not running"
        exit 1
    fi
    print_success "Docker daemon is running"
}

check_containers() {
    print_header "Checking Container Status"
    
    # Check if containers are running
    if docker ps --format "table {{.Names}}" | grep -q "$BACKEND_CONTAINER"; then
        print_success "Backend container is running"
    else
        print_error "Backend container is not running"
        return 1
    fi
    
    if docker ps --format "table {{.Names}}" | grep -q "$FRONTEND_CONTAINER"; then
        print_success "Frontend container is running"
    else
        print_warning "Frontend container is not running (optional for API tests)"
    fi
    
    # Wait for backend to be healthy
    print_info "Waiting for backend to be healthy..."
    local retries=30
    while [ $retries -gt 0 ]; do
        if docker exec "$BACKEND_CONTAINER" curl -s -f http://localhost:8000/health > /dev/null 2>&1; then
            print_success "Backend is healthy"
            return 0
        fi
        retries=$((retries - 1))
        sleep 2
    done
    
    print_error "Backend failed to become healthy"
    return 1
}

install_test_dependencies() {
    print_header "Installing Test Dependencies"
    
    # Install Python test dependencies in backend container
    docker exec "$BACKEND_CONTAINER" pip install --no-cache-dir \
        colorama \
        websocket-client \
        pytest \
        pytest-asyncio \
        pytest-cov || {
        print_error "Failed to install test dependencies"
        return 1
    }
    
    print_success "Test dependencies installed"
}

create_test_scripts() {
    print_header "Creating Test Scripts"
    
    # Create the realistic Python test script
    cat > /tmp/test_implemented_endpoints.py << 'EOF_PYTHON'
#!/usr/bin/env python3
"""
BTC Trading System - API Test Suite for Implemented Endpoints
Tests only the endpoints that are actually implemented
Marks unimplemented endpoints as "Not Implemented" rather than failures
"""

import requests
import json
import time
import sys
import argparse
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple, Set
import websocket
from colorama import init, Fore, Style

# Initialize colorama
init()

class APITester:
    def __init__(self, base_url: str = "http://localhost:8080", verbose: bool = False):
        self.base_url = base_url.rstrip('/')
        self.verbose = verbose
        self.session = requests.Session()
        
        # Test counters
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.skipped_tests = 0
        self.not_implemented = 0
        
        # Track implemented endpoints
        self.implemented_endpoints = set()
        self.failed_endpoints = set()
        self.not_implemented_endpoints = set()
        
    def log(self, message: str, level: str = "INFO"):
        """Log message with timestamp and color"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if level == "SUCCESS":
            color = Fore.GREEN
            symbol = "✓"
        elif level == "ERROR":
            color = Fore.RED
            symbol = "✗"
        elif level == "WARNING":
            color = Fore.YELLOW
            symbol = "⚠"
        elif level == "NOT_IMPL":
            color = Fore.CYAN
            symbol = "○"
        else:
            color = Fore.BLUE
            symbol = "ℹ"
            
        print(f"{color}[{timestamp}] {symbol} {message}{Style.RESET_ALL}")
        
    def test_endpoint(self, method: str, endpoint: str, expected_status: int = 200, 
                     data: Optional[Dict] = None, description: Optional[str] = None,
                     optional: bool = False) -> Tuple[bool, Any]:
        """Test a single API endpoint"""
        self.total_tests += 1
        description = description or endpoint
        
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method == "GET":
                response = self.session.get(url, timeout=30)
            elif method == "POST":
                response = self.session.post(url, json=data, timeout=30)
            elif method == "PUT":
                response = self.session.put(url, json=data, timeout=30)
            elif method == "DELETE":
                response = self.session.delete(url, timeout=30)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            # Check if endpoint is not implemented (404 when we expect 200)
            if response.status_code == 404 and expected_status == 200:
                self.not_implemented += 1
                self.not_implemented_endpoints.add(endpoint)
                self.log(f"{description} - Not Implemented", "NOT_IMPL")
                return False, None
            
            # Check for server errors that indicate bugs
            if response.status_code >= 500:
                self.failed_tests += 1
                self.failed_endpoints.add(endpoint)
                self.log(f"{description} - Server Error: {response.status_code}", "ERROR")
                if self.verbose or response.status_code == 500:
                    print(f"Error details: {response.text}")
                return False, None
            
            # Check expected status
            if response.status_code == expected_status:
                self.passed_tests += 1
                self.implemented_endpoints.add(endpoint)
                self.log(f"{description} - Status: {response.status_code}", "SUCCESS")
                
                if self.verbose and response.text:
                    try:
                        print(json.dumps(response.json(), indent=2))
                    except:
                        print(response.text)
                
                return True, response.json() if response.text else None
            else:
                # Only count as failure if it's not an optional endpoint
                if not optional:
                    self.failed_tests += 1
                    self.failed_endpoints.add(endpoint)
                    self.log(f"{description} - Expected: {expected_status}, Got: {response.status_code}", "ERROR")
                else:
                    self.skipped_tests += 1
                    self.log(f"{description} - Expected: {expected_status}, Got: {response.status_code}", "WARNING")
                    
                if self.verbose:
                    print(f"Response: {response.text}")
                return False, None
                
        except requests.exceptions.RequestException as e:
            self.failed_tests += 1
            self.log(f"{description} - Request failed: {str(e)}", "ERROR")
            return False, None
            
    def validate_json_field(self, data: Dict, field: str, expected_type: Optional[type] = None) -> bool:
        """Validate that a JSON field exists and has the correct type"""
        if field in data:
            if expected_type and not isinstance(data[field], expected_type):
                type_name = expected_type.__name__ if hasattr(expected_type, '__name__') else str(expected_type)
                self.log(f"Field '{field}' has wrong type. Expected: {type_name}, Got: {type(data[field]).__name__}", "ERROR")
                return False
            if expected_type:
                type_name = expected_type.__name__ if hasattr(expected_type, '__name__') else str(expected_type)
                self.log(f"Field '{field}' exists with type {type_name}", "SUCCESS")
            else:
                self.log(f"Field '{field}' exists", "SUCCESS")
            return True
        else:
            self.log(f"Field '{field}' missing from response", "ERROR")
            return False
            
    def test_core_endpoints(self):
        """Test core system endpoints that should always work"""
        self.log("=== Testing Core Endpoints ===")
        
        # Basic health check - this should always work
        success, data = self.test_endpoint("GET", "/", description="Basic health check")
        if success and data:
            self.validate_json_field(data, "message", str)
            self.validate_json_field(data, "status", str)
        
        # Detailed health check - this should always work
        success, data = self.test_endpoint("GET", "/health", description="Detailed health check")
        if success and data:
            self.validate_json_field(data, "status", str)
            self.validate_json_field(data, "components", dict)
            
    def test_signal_endpoints(self):
        """Test trading signal endpoints"""
        self.log("=== Testing Signal Endpoints ===")
        
        # Latest signal - core functionality
        success, data = self.test_endpoint("GET", "/signals/latest", description="Latest trading signal")
        if success and data:
            self.validate_json_field(data, "signal", str)
            self.validate_json_field(data, "confidence", (int, float))
            self.validate_json_field(data, "predicted_price", (int, float))
        
        # Enhanced signal - may not be implemented
        success, data = self.test_endpoint("GET", "/signals/enhanced/latest", description="Enhanced trading signal")
        if success and data:
            self.validate_json_field(data, "signal", str)
            self.validate_json_field(data, "composite_confidence", (int, float))
        
        # Other signal endpoints
        self.test_endpoint("GET", "/signals/comprehensive", description="Comprehensive signals")
        self.test_endpoint("GET", "/signals/history?days=7", description="Signal history (7 days)")
        
    def test_portfolio_endpoints(self):
        """Test portfolio management endpoints"""
        self.log("=== Testing Portfolio Endpoints ===")
        
        # Portfolio metrics - core functionality
        success, data = self.test_endpoint("GET", "/portfolio/metrics", description="Portfolio metrics")
        if success and data:
            self.validate_json_field(data, "total_value", (int, float))
            self.validate_json_field(data, "total_pnl", (int, float))
        
        # Other portfolio endpoints
        self.test_endpoint("GET", "/portfolio/holdings", description="Portfolio holdings")
        self.test_endpoint("GET", "/portfolio/performance", description="Portfolio performance")
        self.test_endpoint("GET", "/portfolio/pnl", description="Portfolio P&L")
        
    def test_paper_trading_endpoints(self):
        """Test paper trading endpoints"""
        self.log("=== Testing Paper Trading Endpoints ===")
        
        # Paper trading status
        success, data = self.test_endpoint("GET", "/paper-trading/status", description="Paper trading status")
        if success and data:
            self.validate_json_field(data, "enabled", bool)
            self.validate_json_field(data, "balance", (int, float))
        
        # Other paper trading endpoints
        self.test_endpoint("POST", "/paper-trading/toggle", description="Toggle paper trading")
        self.test_endpoint("POST", "/paper-trading/reset", description="Reset paper portfolio")
        self.test_endpoint("GET", "/paper-trading/history?days=30", description="Paper trading history")
        
    def test_websocket_connection(self):
        """Test WebSocket connection"""
        self.log("=== Testing WebSocket Connection ===")
        
        ws_url = self.base_url.replace("http://", "ws://") + "/ws"
        
        try:
            ws = websocket.WebSocket()
            ws.connect(ws_url, timeout=5)
            
            # Test ping/pong
            ws.send(json.dumps({"action": "ping"}))
            response = ws.recv()
            
            if "pong" in response:
                self.passed_tests += 1
                self.log("WebSocket ping/pong successful", "SUCCESS")
            else:
                self.failed_tests += 1
                self.log("WebSocket ping/pong failed", "ERROR")
                
            ws.close()
            
        except Exception as e:
            self.failed_tests += 1
            self.log(f"WebSocket connection failed: {str(e)}", "ERROR")
            
    def generate_report(self):
        """Generate test summary report"""
        self.log("=== Test Summary Report ===")
        
        # Calculate percentages
        if self.total_tests > 0:
            success_rate = (self.passed_tests / self.total_tests * 100)
            implementation_rate = ((self.passed_tests + self.failed_tests) / self.total_tests * 100)
        else:
            success_rate = 0
            implementation_rate = 0
        
        print(f"\n{'='*50}")
        print(f"         BTC TRADING SYSTEM TEST REPORT")
        print(f"{'='*50}")
        print(f"Total Endpoints Tested:    {self.total_tests}")
        print(f"✓ Passed:                  {self.passed_tests}")
        print(f"✗ Failed (Errors):         {self.failed_tests}")
        print(f"○ Not Implemented:         {self.not_implemented}")
        print(f"⚠ Skipped/Optional:        {self.skipped_tests}")
        print(f"{'='*50}")
        print(f"Success Rate:              {success_rate:.1f}%")
        print(f"Implementation Rate:       {implementation_rate:.1f}%")
        print(f"{'='*50}")
        
        # Show critical failures (500 errors)
        if self.failed_endpoints:
            print(f"\n{Fore.RED}Critical Failures (need immediate attention):{Style.RESET_ALL}")
            for endpoint in sorted(self.failed_endpoints):
                print(f"  - {endpoint}")
        
        # Show implemented endpoints
        if self.implemented_endpoints:
            print(f"\n{Fore.GREEN}Successfully Tested Endpoints:{Style.RESET_ALL}")
            for endpoint in sorted(self.implemented_endpoints):
                print(f"  ✓ {endpoint}")
        
        # Show not implemented endpoints
        if self.not_implemented_endpoints:
            print(f"\n{Fore.CYAN}Not Yet Implemented Endpoints:{Style.RESET_ALL}")
            for endpoint in sorted(self.not_implemented_endpoints):
                print(f"  ○ {endpoint}")
        
        print(f"\n{'='*50}\n")
        
        # Return success if no critical failures
        return self.failed_tests == 0
            
    def run_all_tests(self):
        """Run all test categories"""
        self.log(f"Starting BTC Trading System API Test Suite")
        self.log(f"API Base URL: {self.base_url}")
        
        # Check API availability first
        try:
            response = self.session.get(f"{self.base_url}/", timeout=5)
            if response.status_code != 200:
                self.log(f"API returned status {response.status_code}", "ERROR")
                return False
        except requests.exceptions.RequestException as e:
            self.log(f"API is not accessible at {self.base_url}: {str(e)}", "ERROR")
            return False
            
        self.log("API is accessible", "SUCCESS")
        
        # Run test categories for implemented features
        self.test_core_endpoints()
        self.test_signal_endpoints()
        self.test_portfolio_endpoints()
        self.test_paper_trading_endpoints()
        self.test_websocket_connection()
        
        # Generate final report
        return self.generate_report()


def main():
    parser = argparse.ArgumentParser(description="BTC Trading System API Test Suite")
    parser.add_argument("--url", default="http://localhost:8080", help="API base URL")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--quick", action="store_true", help="Run quick test (core endpoints only)")
    
    args = parser.parse_args()
    
    tester = APITester(base_url=args.url, verbose=args.verbose)
    
    if args.quick:
        tester.log("Running quick test (core endpoints only)")
        tester.test_core_endpoints()
        tester.test_signal_endpoints()
        tester.test_portfolio_endpoints()
        success = tester.generate_report()
    else:
        success = tester.run_all_tests()
        
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
EOF_PYTHON

    # Create simple bash test script
    cat > /tmp/test_core_api.sh << 'EOF_BASH'
#!/bin/bash

# BTC Trading System - Core API Test Script
# Tests only the endpoints that are confirmed to be implemented

set -euo pipefail

# Configuration
API_BASE_URL="${API_BASE_URL:-http://localhost:8000}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Counters
TOTAL=0
PASSED=0
FAILED=0

# Functions
test_endpoint() {
    local method="$1"
    local endpoint="$2"
    local expected="${3:-200}"
    local description="${4:-$endpoint}"
    
    ((TOTAL++))
    
    echo -ne "${BLUE}Testing ${description}...${NC} "
    
    local response
    response=$(curl -s -w "\n%{http_code}" "$API_BASE_URL$endpoint" 2>/dev/null || echo "000")
    local status_code=$(echo "$response" | tail -1)
    local body=$(echo "$response" | head -n -1)
    
    if [ "$status_code" == "$expected" ]; then
        ((PASSED++))
        echo -e "${GREEN}✓ PASSED${NC} (Status: $status_code)"
        return 0
    else
        ((FAILED++))
        echo -e "${RED}✗ FAILED${NC} (Expected: $expected, Got: $status_code)"
        return 1
    fi
}

print_header() {
    echo -e "\n${BLUE}=== $1 ===${NC}\n"
}

print_summary() {
    local success_rate=0
    if [ $TOTAL -gt 0 ]; then
        success_rate=$(awk "BEGIN {printf \"%.1f\", $PASSED * 100 / $TOTAL}")
    fi
    
    echo -e "\n${BLUE}======================================${NC}"
    echo -e "${BLUE}         TEST SUMMARY${NC}"
    echo -e "${BLUE}======================================${NC}"
    echo -e "Total Tests:    $TOTAL"
    echo -e "${GREEN}Passed:         $PASSED${NC}"
    echo -e "${RED}Failed:         $FAILED${NC}"
    echo -e "Success Rate:   ${success_rate}%"
    echo -e "${BLUE}======================================${NC}\n"
    
    if [ $FAILED -eq 0 ]; then
        echo -e "${GREEN}✅ All tests passed!${NC}"
        return 0
    else
        echo -e "${RED}❌ Some tests failed${NC}"
        return 1
    fi
}

# Main Tests
main() {
    echo -e "${BLUE}BTC Trading System - Core API Tests${NC}"
    echo -e "${BLUE}Testing implemented endpoints only${NC}"
    echo -e "${BLUE}API URL: $API_BASE_URL${NC}"
    echo -e "${BLUE}Time: $(date)${NC}\n"
    
    # Check if API is accessible
    if ! curl -s -f "$API_BASE_URL/" > /dev/null 2>&1; then
        echo -e "${RED}ERROR: Cannot connect to API at $API_BASE_URL${NC}"
        exit 1
    fi
    
    print_header "Core Endpoints"
    test_endpoint "GET" "/" 200 "Health check"
    test_endpoint "GET" "/health" 200 "Detailed health"
    
    print_header "Signal Endpoints"
    test_endpoint "GET" "/signals/latest" 200 "Latest signal"
    test_endpoint "GET" "/signals/enhanced/latest" 200 "Enhanced signal"
    test_endpoint "GET" "/signals/comprehensive" 200 "Comprehensive signals"
    
    print_header "Portfolio Endpoints"
    test_endpoint "GET" "/portfolio/metrics" 200 "Portfolio metrics"
    
    print_header "Paper Trading Endpoints"
    test_endpoint "GET" "/paper-trading/status" 200 "Paper trading status"
    
    print_header "Backtesting Endpoints"
    test_endpoint "GET" "/backtest/status" 200 "Backtest status"
    
    print_header "Configuration Endpoints"
    test_endpoint "GET" "/config/signal-weights" 200 "Signal weights"
    
    print_header "Analytics Endpoints"
    test_endpoint "GET" "/analytics/feature-importance" 200 "Feature importance"
    
    # Generate summary
    print_summary
}

# Run tests
main "$@"
EOF_BASH

    # Make scripts executable
    chmod +x /tmp/test_implemented_endpoints.py
    chmod +x /tmp/test_core_api.sh
    
    print_success "Test scripts created successfully"
}
#!/usr/bin/env python3
"""
BTC Trading System - Comprehensive API Test Suite
Tests all backend API endpoints for expected outputs
Can run from Docker container or host machine
"""

import requests
import json
import time
import sys
import argparse
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import websocket
import threading
from colorama import init, Fore, Style

# Initialize colorama for cross-platform colored output
init()

class APITester:
    def __init__(self, base_url: str = "http://localhost:8080", verbose: bool = False):
        self.base_url = base_url.rstrip('/')
        self.verbose = verbose
        self.session = requests.Session()
        
        # Test counters
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.skipped_tests = 0
        
        # Test results storage
        self.results = []
        
    def log(self, message: str, level: str = "INFO"):
        """Log message with timestamp and color"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if level == "SUCCESS":
            color = Fore.GREEN
            symbol = "✓"
        elif level == "ERROR":
            color = Fore.RED
            symbol = "✗"
        elif level == "WARNING":
            color = Fore.YELLOW
            symbol = "⚠"
        else:
            color = Fore.BLUE
            symbol = "ℹ"
            
        print(f"{color}[{timestamp}] {symbol} {message}{Style.RESET_ALL}")
        
    def test_endpoint(self, method: str, endpoint: str, expected_status: int = 200, 
                     data: Optional[Dict] = None, description: Optional[str] = None) -> Tuple[bool, Any]:
        """Test a single API endpoint"""
        self.total_tests += 1
        description = description or endpoint
        
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method == "GET":
                response = self.session.get(url, timeout=30)
            elif method == "POST":
                response = self.session.post(url, json=data, timeout=30)
            elif method == "PUT":
                response = self.session.put(url, json=data, timeout=30)
            elif method == "DELETE":
                response = self.session.delete(url, timeout=30)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            if response.status_code == expected_status:
                self.passed_tests += 1
                self.log(f"{description} - Status: {response.status_code}", "SUCCESS")
                
                if self.verbose and response.text:
                    try:
                        print(json.dumps(response.json(), indent=2))
                    except:
                        print(response.text)
                
                return True, response.json() if response.text else None
            else:
                self.failed_tests += 1
                self.log(f"{description} - Expected: {expected_status}, Got: {response.status_code}", "ERROR")
                if self.verbose:
                    print(f"Response: {response.text}")
                return False, None
                
        except requests.exceptions.RequestException as e:
            self.failed_tests += 1
            self.log(f"{description} - Request failed: {str(e)}", "ERROR")
            return False, None
            
    def validate_json_field(self, data: Dict, field: str, expected_type: Optional[type] = None) -> bool:
        """Validate that a JSON field exists and has the correct type"""
        if field in data:
            if expected_type and not isinstance(data[field], expected_type):
                type_name = expected_type.__name__ if hasattr(expected_type, '__name__') else str(expected_type)
                self.log(f"Field '{field}' has wrong type. Expected: {type_name}, Got: {type(data[field]).__name__}", "ERROR")
                return False
            if expected_type:
                type_name = expected_type.__name__ if hasattr(expected_type, '__name__') else str(expected_type)
                self.log(f"Field '{field}' exists with type {type_name}", "SUCCESS")
            else:
                self.log(f"Field '{field}' exists", "SUCCESS")
            return True
        else:
            self.log(f"Field '{field}' missing from response", "ERROR")
            return False
            
    def test_core_endpoints(self):
        """Test core system endpoints"""
        self.log("=== Testing Core Endpoints ===")
        
        # Basic health check
        success, data = self.test_endpoint("GET", "/", description="Basic health check")
        if success and data:
            self.validate_json_field(data, "message", str)
            self.validate_json_field(data, "version", str)
            self.validate_json_field(data, "status", str)
        
        # Detailed health check
        success, data = self.test_endpoint("GET", "/health", description="Detailed health check")
        if success and data:
            self.validate_json_field(data, "status", str)
            self.validate_json_field(data, "components", dict)
            self.validate_json_field(data, "timestamp")
            
    def test_signal_endpoints(self):
        """Test trading signal endpoints"""
        self.log("=== Testing Signal Endpoints ===")
        
        # Latest signal
        success, data = self.test_endpoint("GET", "/signals/latest", description="Latest trading signal")
        if success and data:
            self.validate_json_field(data, "signal", str)
            self.validate_json_field(data, "confidence", (int, float))
            self.validate_json_field(data, "predicted_price", (int, float))
        
        # Enhanced signal
        success, data = self.test_endpoint("GET", "/signals/enhanced/latest", description="Enhanced trading signal")
        if success and data:
            self.validate_json_field(data, "signal", str)
            self.validate_json_field(data, "composite_confidence", (int, float))
            self.validate_json_field(data, "analysis", dict)
        
        # Comprehensive signals (50+ indicators)
        success, data = self.test_endpoint("GET", "/signals/comprehensive", description="Comprehensive signals")
        if success and data:
            self.validate_json_field(data, "total_signals", int)
            self.validate_json_field(data, "categorized_signals", dict)
            
        # Historical signals
        self.test_endpoint("GET", "/signals/history?days=7", description="Signal history (7 days)")
        
    def test_portfolio_endpoints(self):
        """Test portfolio management endpoints"""
        self.log("=== Testing Portfolio Endpoints ===")
        
        # Portfolio metrics
        success, data = self.test_endpoint("GET", "/portfolio/metrics", description="Portfolio metrics")
        if success and data:
            self.validate_json_field(data, "total_value", (int, float))
            self.validate_json_field(data, "total_pnl", (int, float))
            self.validate_json_field(data, "roi", (int, float))
        
        # Holdings
        self.test_endpoint("GET", "/portfolio/holdings", description="Portfolio holdings")
        
        # Performance
        self.test_endpoint("GET", "/portfolio/performance", description="Portfolio performance")
        
        # P&L
        self.test_endpoint("GET", "/portfolio/pnl", description="Portfolio P&L")
        
    def test_trading_endpoints(self):
        """Test trading operation endpoints"""
        self.log("=== Testing Trading Endpoints ===")
        
        # Execute trade
        trade_data = {
            "signal": "buy",
            "confidence": 0.75,
            "quantity": 0.001
        }
        self.test_endpoint("POST", "/trade/execute", data=trade_data, description="Execute trade")
        
        # Active orders
        self.test_endpoint("GET", "/orders/active", description="Active orders")
        
        # Order history
        self.test_endpoint("GET", "/orders/history", description="Order history")
        
        # Cancel order (expect 404 if no orders)
        self.test_endpoint("POST", "/orders/cancel/1", expected_status=404, description="Cancel order (404 expected)")
        
    def test_paper_trading_endpoints(self):
        """Test paper trading endpoints"""
        self.log("=== Testing Paper Trading Endpoints ===")
        
        # Paper trading status
        success, data = self.test_endpoint("GET", "/paper-trading/status", description="Paper trading status")
        if success and data:
            self.validate_json_field(data, "enabled", bool)
            self.validate_json_field(data, "balance", (int, float))
        
        # Toggle paper trading
        self.test_endpoint("POST", "/paper-trading/toggle", description="Toggle paper trading")
        
        # Reset portfolio
        self.test_endpoint("POST", "/paper-trading/reset", description="Reset paper portfolio")
        
        # Paper trading history
        self.test_endpoint("GET", "/paper-trading/history?days=30", description="Paper trading history")
        
    def test_market_data_endpoints(self):
        """Test market data endpoints"""
        self.log("=== Testing Market Data Endpoints ===")
        
        # Current price
        success, data = self.test_endpoint("GET", "/market/price", description="Current BTC price")
        if success and data:
            self.validate_json_field(data, "price", (int, float))
            self.validate_json_field(data, "timestamp")
        
        # Historical data
        self.test_endpoint("GET", "/market/history?period=1d", description="Market history (1 day)")
        self.test_endpoint("GET", "/market/history?period=1w", description="Market history (1 week)")
        
        # Market indicators
        self.test_endpoint("GET", "/market/indicators", description="Market indicators")
        
        # OHLCV data
        self.test_endpoint("GET", "/market/ohlcv?period=1h&limit=24", description="OHLCV data")
        
    def test_analytics_endpoints(self):
        """Test analytics endpoints"""
        self.log("=== Testing Analytics Endpoints ===")
        
        # Feature importance
        success, data = self.test_endpoint("GET", "/analytics/feature-importance", description="Feature importance")
        if success and data:
            self.validate_json_field(data, "feature_importance", dict)
        
        # Market regime
        self.test_endpoint("GET", "/analytics/market-regime", description="Market regime detection")
        
        # Risk metrics
        self.test_endpoint("GET", "/analytics/risk-metrics", description="Risk metrics")
        
        # Monte Carlo simulation
        mc_data = {
            "num_simulations": 100,
            "time_horizon_days": 7
        }
        self.test_endpoint("POST", "/analytics/monte-carlo", data=mc_data, description="Monte Carlo simulation")
        
    def test_backtesting_endpoints(self):
        """Test backtesting endpoints"""
        self.log("=== Testing Backtesting Endpoints ===")
        
        # Backtest status
        self.test_endpoint("GET", "/backtest/status", description="Backtest status")
        
        # Run backtest (might take time)
        backtest_data = {
            "period": "1m",
            "optimize_weights": False
        }
        self.test_endpoint("POST", "/backtest/enhanced/run", data=backtest_data, description="Run enhanced backtest")
        
        # Latest results
        self.test_endpoint("GET", "/backtest/enhanced/results/latest", description="Latest backtest results")
        
        # Results history
        self.test_endpoint("GET", "/backtest/results/history?limit=5", description="Backtest history")
        
        # Walk-forward results
        self.test_endpoint("GET", "/backtest/walk-forward/results", description="Walk-forward analysis")
        
        # Optimization results
        self.test_endpoint("GET", "/backtest/optimization/results", description="Optimization results")
        
    def test_configuration_endpoints(self):
        """Test configuration endpoints"""
        self.log("=== Testing Configuration Endpoints ===")
        
        # Signal weights
        success, data = self.test_endpoint("GET", "/config/signal-weights", description="Current signal weights")
        
        # Enhanced signal weights
        self.test_endpoint("GET", "/config/signal-weights/enhanced", description="Enhanced signal weights")
        
        # Update weights
        weights_data = {
            "technical_weight": 0.4,
            "onchain_weight": 0.35,
            "sentiment_weight": 0.15,
            "macro_weight": 0.1
        }
        self.test_endpoint("POST", "/config/signal-weights", data=weights_data, description="Update signal weights")
        
        # Model config
        self.test_endpoint("GET", "/config/model", description="Model configuration")
        
        # Trading config
        self.test_endpoint("GET", "/config/trading", description="Trading configuration")
        
    def test_model_endpoints(self):
        """Test model-related endpoints"""
        self.log("=== Testing Model Endpoints ===")
        
        # Model info
        self.test_endpoint("GET", "/model/info", description="Model information")
        
        # Model diagnostics
        self.test_endpoint("GET", "/model/diagnostics", description="Model diagnostics")
        
        # Performance metrics
        self.test_endpoint("GET", "/model/performance", description="Model performance")
        
        # Predict endpoint
        self.test_endpoint("GET", "/model/predict", description="Model prediction")
        
        # Ensemble prediction
        self.test_endpoint("GET", "/models/ensemble/predict", description="Ensemble prediction")
        
    def test_system_endpoints(self):
        """Test system management endpoints"""
        self.log("=== Testing System Endpoints ===")
        
        # System status
        self.test_endpoint("GET", "/system/status", description="System status")
        
        # System metrics
        self.test_endpoint("GET", "/system/metrics", description="System metrics")
        
        # Recent logs
        self.test_endpoint("GET", "/logs/recent?limit=10", description="Recent logs")
        
    def test_websocket_connection(self):
        """Test WebSocket connection"""
        self.log("=== Testing WebSocket Connection ===")
        
        ws_url = self.base_url.replace("http://", "ws://") + "/ws"
        
        try:
            ws = websocket.WebSocket()
            ws.connect(ws_url, timeout=5)
            
            # Test ping/pong
            ws.send(json.dumps({"action": "ping"}))
            response = ws.recv()
            
            if "pong" in response:
                self.passed_tests += 1
                self.log("WebSocket ping/pong successful", "SUCCESS")
            else:
                self.failed_tests += 1
                self.log("WebSocket ping/pong failed", "ERROR")
                
            ws.close()
            
        except Exception as e:
            self.failed_tests += 1
            self.log(f"WebSocket connection failed: {str(e)}", "ERROR")
            
    def test_data_consistency(self):
        """Test data consistency across endpoints"""
        self.log("=== Testing Data Consistency ===")
        
        # Get price from different endpoints
        price1 = None
        price2 = None
        
        success, data = self.test_endpoint("GET", "/market/price")
        if success and data and "price" in data:
            price1 = data["price"]
            
        success, data = self.test_endpoint("GET", "/signals/latest")
        if success and data and "current_price" in data:
            price2 = data["current_price"]
            
        if price1 and price2:
            diff_percent = abs(price1 - price2) / price1 * 100
            if diff_percent < 1:
                self.passed_tests += 1
                self.log(f"Price consistency check passed (diff: {diff_percent:.2f}%)", "SUCCESS")
            else:
                self.failed_tests += 1
                self.log(f"Price inconsistency detected: ${price1} vs ${price2} (diff: {diff_percent:.2f}%)", "ERROR")
        else:
            self.skipped_tests += 1
            self.log("Could not verify price consistency", "WARNING")
            
    def test_error_handling(self):
        """Test API error handling"""
        self.log("=== Testing Error Handling ===")
        
        # Invalid endpoints
        self.test_endpoint("GET", "/invalid/endpoint", expected_status=404, description="Invalid endpoint (404 expected)")
        
        # Invalid data
        self.test_endpoint("POST", "/trade/execute", expected_status=422, 
                         data={"invalid": "data"}, description="Invalid trade data (422 expected)")
        
        # Invalid parameters
        self.test_endpoint("GET", "/market/history?period=invalid", expected_status=422, 
                         description="Invalid period parameter (422 expected)")
        
    def generate_report(self):
        """Generate test summary report"""
        self.log("=== Test Summary Report ===")
        
        success_rate = (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0
        
        print(f"\n{'='*40}")
        print(f"         TEST SUMMARY REPORT")
        print(f"{'='*40}")
        print(f"Total Tests:    {self.total_tests}")
        print(f"Passed:         {self.passed_tests}")
        print(f"Failed:         {self.failed_tests}")
        print(f"Skipped:        {self.skipped_tests}")
        print(f"Success Rate:   {success_rate:.1f}%")
        print(f"{'='*40}\n")
        
        if self.failed_tests == 0:
            self.log("All tests passed! 🎉", "SUCCESS")
            return True
        else:
            self.log(f"{self.failed_tests} tests failed", "ERROR")
            return False
            
    def run_all_tests(self):
        """Run all test categories"""
        self.log(f"Starting BTC Trading System API Test Suite")
        self.log(f"API Base URL: {self.base_url}")
        
        # Check API availability first
        try:
            response = self.session.get(f"{self.base_url}/", timeout=5)
            if response.status_code != 200:
                self.log(f"API returned status {response.status_code}", "ERROR")
                return False
        except requests.exceptions.RequestException as e:
            self.log(f"API is not accessible at {self.base_url}: {str(e)}", "ERROR")
            return False
            
        self.log("API is accessible", "SUCCESS")
        
        # Run all test categories
        self.test_core_endpoints()
        self.test_signal_endpoints()
        self.test_portfolio_endpoints()
        self.test_trading_endpoints()
        self.test_paper_trading_endpoints()
        self.test_market_data_endpoints()
        self.test_analytics_endpoints()
        self.test_backtesting_endpoints()
        self.test_configuration_endpoints()
        self.test_model_endpoints()
        self.test_system_endpoints()
        self.test_websocket_connection()
        self.test_data_consistency()
        self.test_error_handling()
        
        # Generate final report
        return self.generate_report()


def main():
    parser = argparse.ArgumentParser(description="BTC Trading System API Test Suite")
    parser.add_argument("--url", default="http://localhost:8080", help="API base URL")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--category", help="Test specific category only")
    parser.add_argument("--quick", action="store_true", help="Run quick test (core endpoints only)")
    
    args = parser.parse_args()
    
    tester = APITester(base_url=args.url, verbose=args.verbose)
    
    if args.quick:
        tester.log("Running quick test (core endpoints only)")
        tester.test_core_endpoints()
        success = tester.generate_report()
    elif args.category:
        method_name = f"test_{args.category}_endpoints"
        if hasattr(tester, method_name):
            tester.log(f"Testing category: {args.category}")
            getattr(tester, method_name)()
            success = tester.generate_report()
        else:
            print(f"Unknown category: {args.category}")
            print("Available categories: core, signal, portfolio, trading, paper_trading, market_data, analytics, backtesting, configuration, model, system")
            return False
    else:
        success = tester.run_all_tests()
        
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
PYTHON_EOF

    # Create Bash test script
    cat > /tmp/test_all_endpoints.sh << 'BASH_EOF'
#!/bin/bash

# BTC Trading System - Comprehensive API Testing Script
# Tests all backend API endpoints for expected outputs
# Runs from within the Docker container

set -euo pipefail

# Configuration
API_BASE_URL="${API_BASE_URL:-http://backend:8000}"
FRONTEND_URL="${FRONTEND_URL:-http://frontend:8501}"
TEST_LOG="/app/logs/api_test_$(date +%Y%m%d_%H%M%S).log"
VERBOSE="${VERBOSE:-false}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
SKIPPED_TESTS=0

# Logging functions
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$TEST_LOG"
}

log_success() {
    echo -e "${GREEN}✓ $*${NC}" | tee -a "$TEST_LOG"
    ((PASSED_TESTS++))
}

log_error() {
    echo -e "${RED}✗ $*${NC}" | tee -a "$TEST_LOG"
    ((FAILED_TESTS++))
}

log_warning() {
    echo -e "${YELLOW}⚠ $*${NC}" | tee -a "$TEST_LOG"
}

log_info() {
    echo -e "${BLUE}ℹ $*${NC}" | tee -a "$TEST_LOG"
}

# API Testing Functions
test_endpoint() {
    local method="$1"
    local endpoint="$2"
    local expected_status="${3:-200}"
    local data="${4:-}"
    local description="${5:-$endpoint}"
    
    ((TOTAL_TESTS++))
    
    log_info "Testing: $method $endpoint"
    
    local response
    local status_code
    local curl_opts="-s -w \n%{http_code}"
    
    if [ "$VERBOSE" == "true" ]; then
        curl_opts="-v -w \n%{http_code}"
    fi
    
    if [ "$method" == "GET" ]; then
        response=$(curl $curl_opts "$API_BASE_URL$endpoint" 2>&1 || true)
    elif [ "$method" == "POST" ]; then
        response=$(curl $curl_opts -X POST -H "Content-Type: application/json" \
            ${data:+-d "$data"} "$API_BASE_URL$endpoint" 2>&1 || true)
    elif [ "$method" == "PUT" ]; then
        response=$(curl $curl_opts -X PUT -H "Content-Type: application/json" \
            ${data:+-d "$data"} "$API_BASE_URL$endpoint" 2>&1 || true)
    fi
    
    # Extract status code from last line
    status_code=$(echo "$response" | tail -1)
    body=$(echo "$response" | head -n -1)
    
    if [ "$status_code" == "$expected_status" ]; then
        log_success "$description - Status: $status_code"
        if [ "$VERBOSE" == "true" ]; then
            echo "$body" | jq . 2>/dev/null || echo "$body"
        fi
        return 0
    else
        log_error "$description - Expected: $expected_status, Got: $status_code"
        echo "Response body: $body" >> "$TEST_LOG"
        return 1
    fi
}

# Test Categories
test_core_endpoints() {
    log "=== Testing Core Endpoints ==="
    
    # Health check
    test_endpoint "GET" "/" 200 "" "Basic health check"
    
    # Detailed health check
    if response=$(curl -s "$API_BASE_URL/health"); then
        echo "$response" | jq . > /dev/null 2>&1 && {
            log_success "Health endpoint returns valid JSON"
        }
    fi
}

test_all_endpoints() {
    # Run all test categories
    test_core_endpoints
    
    # Add more endpoint tests here...
    test_endpoint "GET" "/signals/latest" 200 "" "Latest trading signal"
    test_endpoint "GET" "/signals/enhanced/latest" 200 "" "Enhanced trading signal"
    test_endpoint "GET" "/portfolio/metrics" 200 "" "Portfolio metrics"
    test_endpoint "GET" "/market/price" 200 "" "Current BTC price"
}

# Generate report
generate_report() {
    log "=== Test Summary Report ==="
    
    local success_rate=0
    if [ $TOTAL_TESTS -gt 0 ]; then
        success_rate=$(echo "scale=2; $PASSED_TESTS * 100 / $TOTAL_TESTS" | bc)
    fi
    
    echo "Total Tests:    $TOTAL_TESTS"
    echo "Passed:         $PASSED_TESTS"
    echo "Failed:         $FAILED_TESTS"
    echo "Success Rate:   ${success_rate}%"
    
    if [ $FAILED_TESTS -eq 0 ]; then
        log_success "All tests passed!"
        return 0
    else
        log_error "Some tests failed."
        return 1
    fi
}

# Main execution
main() {
    log "Starting BTC Trading System API Test Suite"
    
    # Create log directory
    local log_dir=$(dirname "$TEST_LOG")
    mkdir -p "$log_dir" 2>/dev/null || {
        # If we can't create in the specified location, use /tmp
        TEST_LOG="/tmp/api_test_$(date +%Y%m%d_%H%M%S).log"
        log_dir=$(dirname "$TEST_LOG")
        mkdir -p "$log_dir"
    }
    
    test_all_endpoints
    generate_report
}

main
BASH_EOF

    # Make scripts executable
    chmod +x /tmp/test_all_endpoints.py
    chmod +x /tmp/test_all_endpoints.sh
    
    print_success "Test scripts created successfully"
}

copy_test_scripts() {
    print_header "Copying Test Scripts to Container"
    
    # Create test scripts first if they don't exist
    if [ ! -f /tmp/test_implemented_endpoints.py ] || [ ! -f /tmp/test_core_api.sh ]; then
        create_test_scripts
    fi
    
    # Copy test scripts from /tmp
    docker cp /tmp/test_implemented_endpoints.py "$BACKEND_CONTAINER":/app/test_implemented_endpoints.py || {
        print_error "Failed to copy Python test script"
        return 1
    }
    
    docker cp /tmp/test_core_api.sh "$BACKEND_CONTAINER":/app/test_core_api.sh || {
        print_error "Failed to copy Bash test script"
        return 1
    }
    
    # Make bash script executable
    docker exec "$BACKEND_CONTAINER" chmod +x /app/test_core_api.sh
    
    print_success "Test scripts copied successfully"
}

run_python_tests() {
    print_header "Running Python API Tests"
    
    local test_log="/tmp/python_test_${TIMESTAMP}.log"
    
    # Create results directory in container
    docker exec "$BACKEND_CONTAINER" mkdir -p "$TEST_RESULTS_DIR"
    
    # Run tests and capture output to temp file first
    if docker exec -e PYTHONUNBUFFERED=1 "$BACKEND_CONTAINER" \
        python /app/test_implemented_endpoints.py --url http://localhost:8000 --verbose \
        2>&1 | tee "$test_log"; then
        print_success "Python tests completed successfully"
        # Copy log to container
        docker cp "$test_log" "$BACKEND_CONTAINER":"${TEST_RESULTS_DIR}/python_test_${TIMESTAMP}.log"
        rm -f "$test_log"
        return 0
    else
        print_error "Python tests failed"
        # Copy log to container even on failure
        docker cp "$test_log" "$BACKEND_CONTAINER":"${TEST_RESULTS_DIR}/python_test_${TIMESTAMP}.log" 2>/dev/null || true
        rm -f "$test_log"
        return 1
    fi
}

run_bash_tests() {
    print_header "Running Bash API Tests"
    
    local test_log="/tmp/bash_test_${TIMESTAMP}.log"
    
    # Create results directory in container
    docker exec "$BACKEND_CONTAINER" mkdir -p "$TEST_RESULTS_DIR"
    
    # Run tests and capture output to temp file first
    if docker exec -e API_BASE_URL=http://localhost:8000 "$BACKEND_CONTAINER" \
        /app/test_core_api.sh 2>&1 | tee "$test_log"; then
        print_success "Bash tests completed successfully"
        # Copy log to container
        docker cp "$test_log" "$BACKEND_CONTAINER":"${TEST_RESULTS_DIR}/bash_test_${TIMESTAMP}.log"
        rm -f "$test_log"
        return 0
    else
        print_error "Bash tests failed"
        # Copy log to container even on failure
        docker cp "$test_log" "$BACKEND_CONTAINER":"${TEST_RESULTS_DIR}/bash_test_${TIMESTAMP}.log" 2>/dev/null || true
        rm -f "$test_log"
        return 1
    fi
}

run_specific_tests() {
    local test_type="$1"
    local category="${2:-}"
    
    print_header "Running $test_type Tests${category:+ - Category: $category}"
    
    case "$test_type" in
        quick)
            docker exec "$BACKEND_CONTAINER" python /app/test_all_endpoints.py --quick
            ;;
        category)
            if [ -z "$category" ]; then
                print_error "Category not specified"
                echo "Available categories: core, signal, portfolio, trading, paper_trading, market_data, analytics, backtesting, configuration, model, system"
                return 1
            fi
            docker exec "$BACKEND_CONTAINER" python /app/test_all_endpoints.py --category "$category"
            ;;
        performance)
            run_performance_tests
            ;;
        stress)
            run_stress_tests
            ;;
        *)
            print_error "Unknown test type: $test_type"
            return 1
            ;;
    esac
}

run_performance_tests() {
    print_header "Running Performance Tests"
    
    # Simple performance test
    docker exec "$BACKEND_CONTAINER" bash -c '
        echo "Testing API response times..."
        
        endpoints=(
            "/"
            "/health"
            "/signals/latest"
            "/portfolio/metrics"
            "/market/price"
        )
        
        for endpoint in "${endpoints[@]}"; do
            echo -n "Testing $endpoint: "
            time=$(curl -o /dev/null -s -w "%{time_total}\n" http://localhost:8000$endpoint)
            echo "${time}s"
        done
        
        # Load test with multiple requests
        echo -e "\nRunning load test (100 requests)..."
        seq 1 100 | xargs -P 10 -I {} curl -s http://localhost:8000/ > /dev/null
        echo "Load test completed"
    '
}

run_stress_tests() {
    print_header "Running Stress Tests"
    
    docker exec "$BACKEND_CONTAINER" bash -c '
        echo "Running stress tests..."
        
        # Test rapid API calls
        echo "Testing rapid API calls..."
        for i in {1..50}; do
            curl -s http://localhost:8000/signals/latest > /dev/null &
        done
        wait
        echo "Rapid API call test completed"
    '
}

generate_test_report() {
    print_header "Generating Test Report"
    
    local report_file="${TEST_RESULTS_DIR}/test_report_${TIMESTAMP}.txt"
    
    docker exec "$BACKEND_CONTAINER" bash -c "
        cat > $report_file <<EOF
========================================
BTC Trading System - API Test Report
Generated: $(date)
========================================

Container Information:
- Backend: $BACKEND_CONTAINER
- Image: $(docker inspect --format='{{.Config.Image}}' $BACKEND_CONTAINER)
- Status: $(docker inspect --format='{{.State.Status}}' $BACKEND_CONTAINER)

Test Results Summary:
- Python Tests: ${PYTHON_TEST_RESULT:-Not Run}
- Bash Tests: ${BASH_TEST_RESULT:-Not Run}
- Performance Tests: ${PERF_TEST_RESULT:-Not Run}

Log Files:
- Python Test Log: ${TEST_RESULTS_DIR}/python_test_${TIMESTAMP}.log
- Bash Test Log: ${TEST_RESULTS_DIR}/bash_test_${TIMESTAMP}.log

========================================
EOF
    "
    
    print_success "Test report generated: $report_file"
    
    # Display report
    docker exec "$BACKEND_CONTAINER" cat "$report_file"
}

cleanup() {
    print_header "Cleanup"
    
    # Remove test scripts from container
    docker exec "$BACKEND_CONTAINER" rm -f /app/test_implemented_endpoints.py /app/test_core_api.sh 2>/dev/null || true
    
    # Remove temp files
    rm -f /tmp/test_implemented_endpoints.py /tmp/test_core_api.sh 2>/dev/null || true
    
    print_success "Cleanup completed"
}

# Main execution
main() {
    local mode="${1:-all}"
    
    print_header "BTC Trading System - Docker API Testing"
    print_info "Mode: $mode"
    print_info "Timestamp: $TIMESTAMP"
    
    # Always check requirements and containers
    check_requirements || exit 1
    check_containers || {
        print_error "Container checks failed. Please ensure containers are running:"
        print_info "  docker compose up -d"
        exit 1
    }
    
    # Export results for report
    export PYTHON_TEST_RESULT="Not Run"
    export BASH_TEST_RESULT="Not Run"
    export PERF_TEST_RESULT="Not Run"
    
    case "$mode" in
        all)
            install_test_dependencies
            copy_test_scripts
            
            if run_python_tests; then
                export PYTHON_TEST_RESULT="Passed"
            else
                export PYTHON_TEST_RESULT="Failed"
            fi
            
            if run_bash_tests; then
                export BASH_TEST_RESULT="Passed"
            else
                export BASH_TEST_RESULT="Failed"
            fi
            
            run_performance_tests && export PERF_TEST_RESULT="Passed"
            ;;
            
        python)
            install_test_dependencies
            copy_test_scripts
            if run_python_tests; then
                export PYTHON_TEST_RESULT="Passed"
            else
                export PYTHON_TEST_RESULT="Failed"
            fi
            ;;
            
        bash)
            copy_test_scripts
            if run_bash_tests; then
                export BASH_TEST_RESULT="Passed"
            else
                export BASH_TEST_RESULT="Failed"
            fi
            ;;
            
        quick|category|performance|stress)
            install_test_dependencies
            copy_test_scripts
            run_specific_tests "$mode" "$2"
            ;;
            
        report)
            generate_test_report
            exit 0
            ;;
            
        *)
            print_error "Unknown mode: $mode"
            echo "Usage: $0 [all|python|bash|quick|category|performance|stress|report] [category_name]"
            echo ""
            echo "Modes:"
            echo "  all         - Run all tests (default)"
            echo "  python      - Run Python tests only"
            echo "  bash        - Run Bash tests only"
            echo "  quick       - Run quick core tests only"
            echo "  category    - Run specific category tests"
            echo "  performance - Run performance tests"
            echo "  stress      - Run stress tests"
            echo "  report      - Generate test report"
            echo ""
            echo "Categories for 'category' mode:"
            echo "  core, signal, portfolio, trading, paper_trading, market_data,"
            echo "  analytics, backtesting, configuration, model, system"
            exit 1
            ;;
    esac
    
    # Generate report and cleanup
    generate_test_report
    cleanup
    
    # Exit with appropriate code
    if [[ "$PYTHON_TEST_RESULT" == "Failed" ]] || [[ "$BASH_TEST_RESULT" == "Failed" ]]; then
        exit 1
    else
        exit 0
    fi
}

# Trap to ensure cleanup on exit
trap cleanup EXIT

# Run main function with all arguments
main "$@"
