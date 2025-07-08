#!/usr/bin/env python3
"""
Simplified test script for frontend functionality (no WebSocket dependency)
Tests all REST API endpoints for implemented features
"""

import json
import time
from datetime import datetime, timedelta
import urllib.request
import urllib.error
from typing import Dict, List, Tuple, Optional

# Configuration
import os
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8090")
TEST_RESULTS = []

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_header(text: str):
    """Print a formatted header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text:^60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}\n")

def print_test(name: str, result: bool, details: str = ""):
    """Print test result"""
    status = f"{Colors.GREEN}✓ PASS{Colors.END}" if result else f"{Colors.RED}✗ FAIL{Colors.END}"
    print(f"{status} {name}")
    if details:
        print(f"    {Colors.YELLOW}{details}{Colors.END}")
    TEST_RESULTS.append((name, result, details))

def test_api_endpoint(endpoint: str, method: str = "GET", data: dict = None) -> Tuple[bool, str]:
    """Test a single API endpoint"""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        
        if method == "GET":
            req = urllib.request.Request(url)
        elif method == "POST":
            json_data = json.dumps(data or {}).encode('utf-8')
            req = urllib.request.Request(url, data=json_data, method='POST')
            req.add_header('Content-Type', 'application/json')
        else:
            return False, f"Unsupported method: {method}"
        
        with urllib.request.urlopen(req) as response:
            status_code = response.getcode()
            if status_code == 200:
                return True, f"Status: {status_code}"
            else:
                return False, f"Status: {status_code}"
    except urllib.error.HTTPError as e:
        return False, f"HTTP Error {e.code}: {e.reason}"
    except urllib.error.URLError as e:
        return False, f"URL Error: {str(e.reason)}"
    except Exception as e:
        return False, f"Error: {str(e)}"

def test_dashboard_functionality():
    """Test Dashboard page functionality"""
    print_header("Testing Dashboard Functionality")
    
    # Test current price endpoint
    success, details = test_api_endpoint("/btc/latest")
    print_test("Current BTC price", success, details)
    
    # Test latest signal
    success, details = test_api_endpoint("/signals/latest")
    print_test("Latest trading signal", success, details)
    
    # Test enhanced LSTM signal
    success, details = test_api_endpoint("/signals/enhanced/latest")
    print_test("Enhanced LSTM signal", success, details)
    
    # Test price history
    success, details = test_api_endpoint("/btc/history/1d")
    print_test("Price history (1 day)", success, details)
    
    # Test market metrics
    success, details = test_api_endpoint("/btc/metrics")
    print_test("Market metrics", success, details)

def test_signals_functionality():
    """Test Signals page functionality"""
    print_header("Testing Signals Page Functionality")
    
    # Test comprehensive signals
    success, details = test_api_endpoint("/signals/comprehensive")
    print_test("Comprehensive signals data", success, details)
    
    # Test signal history
    success, details = test_api_endpoint("/signals/history?hours=24")
    print_test("Signal history (24h)", success, details)
    
    # Test feature importance
    success, details = test_api_endpoint("/analytics/feature-importance")
    print_test("Feature importance analysis", success, details)

def test_portfolio_functionality():
    """Test Portfolio page functionality"""
    print_header("Testing Portfolio Page Functionality")
    
    # Test portfolio metrics
    success, details = test_api_endpoint("/portfolio/metrics")
    print_test("Portfolio metrics", success, details)
    
    # Test positions
    success, details = test_api_endpoint("/portfolio/positions")
    print_test("Current positions", success, details)
    
    # Test trades
    success, details = test_api_endpoint("/trades/all")
    print_test("Trade history", success, details)
    
    # Test performance analytics
    success, details = test_api_endpoint("/analytics/performance")
    print_test("Performance analytics", success, details)
    
    # Test performance history
    success, details = test_api_endpoint("/portfolio/performance/history")
    print_test("Performance history", success, details)
    
    # Test risk metrics
    success, details = test_api_endpoint("/analytics/risk-metrics")
    print_test("Risk metrics", success, details)
    
    # Test attribution analysis
    success, details = test_api_endpoint("/analytics/attribution")
    print_test("Performance attribution", success, details)
    
    # Test P&L analysis
    success, details = test_api_endpoint("/analytics/pnl-analysis")
    print_test("P&L analysis", success, details)

def test_paper_trading_functionality():
    """Test Paper Trading functionality"""
    print_header("Testing Paper Trading Functionality")
    
    # Test paper trading status
    success, details = test_api_endpoint("/paper-trading/status")
    print_test("Paper trading status", success, details)
    
    # Test toggle paper trading
    success, details = test_api_endpoint("/paper-trading/toggle", "POST")
    print_test("Toggle paper trading", success, details)
    
    # Test portfolio reset
    success, details = test_api_endpoint("/paper-trading/reset", "POST")
    print_test("Reset paper trading portfolio", success, details)

def test_analytics_functionality():
    """Test Analytics page functionality"""
    print_header("Testing Analytics Page Functionality")
    
    # Test backtesting
    backtest_params = {
        "start_date": (datetime.now() - timedelta(days=30)).isoformat(),
        "end_date": datetime.now().isoformat(),
        "initial_capital": 10000
    }
    success, details = test_api_endpoint("/analytics/backtest", "POST", backtest_params)
    print_test("Backtesting analysis", success, details)
    
    # Test Monte Carlo simulation
    monte_carlo_params = {
        "num_simulations": 100,
        "time_horizon": 30
    }
    success, details = test_api_endpoint("/analytics/monte-carlo", "POST", monte_carlo_params)
    print_test("Monte Carlo simulation", success, details)
    
    # Test feature importance
    success, details = test_api_endpoint("/analytics/feature-importance")
    print_test("Feature importance (Analytics)", success, details)
    
    # Test market regime
    success, details = test_api_endpoint("/analytics/market-regime")
    print_test("Market regime detection", success, details)

def test_settings_functionality():
    """Test Settings page functionality"""
    print_header("Testing Settings Page Functionality")
    
    # Test get current config
    success, details = test_api_endpoint("/config/current")
    print_test("Get current configuration", success, details)
    
    # Test update config
    config_update = {
        "trading_rules": {
            "max_position_size": 0.5,
            "stop_loss_percentage": 5.0
        }
    }
    success, details = test_api_endpoint("/config/update", "POST", config_update)
    print_test("Update configuration", success, details)
    
    # Test reset to defaults
    success, details = test_api_endpoint("/config/reset", "POST")
    print_test("Reset configuration to defaults", success, details)
    
    # Test enhanced LSTM status
    success, details = test_api_endpoint("/enhanced-lstm/status")
    print_test("Enhanced LSTM status", success, details)
    
    # Test data availability
    success, details = test_api_endpoint("/enhanced-lstm/data-status")
    print_test("Enhanced LSTM data status", success, details)

def test_enhanced_lstm_integration():
    """Test Enhanced LSTM integration"""
    print_header("Testing Enhanced LSTM Integration")
    
    # Test enhanced LSTM status
    success, details = test_api_endpoint("/enhanced-lstm/status")
    print_test("Enhanced LSTM model status", success, details)
    
    # Test enhanced prediction
    success, details = test_api_endpoint("/enhanced-lstm/predict")
    print_test("Enhanced LSTM prediction", success, details)
    
    # Test data availability for 50+ signals
    success, details = test_api_endpoint("/enhanced-lstm/data-status")
    print_test("Data availability for 50+ signals", success, details)
    
    # Test fallback to original LSTM
    success, details = test_api_endpoint("/signals/latest")
    print_test("Fallback to original LSTM", success, details)

def generate_test_report():
    """Generate final test report"""
    print_header("Test Summary Report")
    
    total_tests = len(TEST_RESULTS)
    passed_tests = sum(1 for _, result, _ in TEST_RESULTS if result)
    failed_tests = total_tests - passed_tests
    pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    print(f"{Colors.BOLD}Total Tests:{Colors.END} {total_tests}")
    print(f"{Colors.GREEN}Passed:{Colors.END} {passed_tests}")
    print(f"{Colors.RED}Failed:{Colors.END} {failed_tests}")
    print(f"{Colors.BOLD}Pass Rate:{Colors.END} {pass_rate:.1f}%")
    
    if failed_tests > 0:
        print(f"\n{Colors.RED}{Colors.BOLD}Failed Tests:{Colors.END}")
        for name, result, details in TEST_RESULTS:
            if not result:
                print(f"  - {name}: {details}")
    
    # Save report to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"test_report_{timestamp}.json"
    with open(report_file, 'w') as f:
        json.dump({
            "timestamp": timestamp,
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "pass_rate": pass_rate,
            "results": [(name, result, details) for name, result, details in TEST_RESULTS]
        }, f, indent=2)
    
    print(f"\n{Colors.BLUE}Report saved to: {report_file}{Colors.END}")

def main():
    """Main test execution"""
    print(f"{Colors.BOLD}{Colors.BLUE}")
    print("=" * 60)
    print("BTC Trading System - Frontend Functionality Test Suite")
    print("=" * 60)
    print(f"{Colors.END}")
    
    print(f"\nTesting against: {API_BASE_URL}")
    print("\nPress Ctrl+C to stop at any time\n")
    
    try:
        # Test connectivity first
        print("Testing backend connectivity...")
        success, details = test_api_endpoint("/health")
        if not success:
            # Try alternate health endpoint
            success, details = test_api_endpoint("/")
            if not success:
                print(f"{Colors.RED}ERROR: Cannot connect to backend at {API_BASE_URL}{Colors.END}")
                print(f"Details: {details}")
                print("\nPlease ensure the backend is running:")
                print("  docker compose up -d")
                return
        
        print(f"{Colors.GREEN}✓ Backend is accessible{Colors.END}\n")
        
        # Run all test suites
        test_dashboard_functionality()
        test_signals_functionality()
        test_portfolio_functionality()
        test_paper_trading_functionality()
        test_analytics_functionality()
        test_settings_functionality()
        test_enhanced_lstm_integration()
        
        # Generate report
        generate_test_report()
        
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Testing interrupted by user{Colors.END}")
        generate_test_report()
    except Exception as e:
        print(f"\n{Colors.RED}Unexpected error: {str(e)}{Colors.END}")
        generate_test_report()

if __name__ == "__main__":
    main()