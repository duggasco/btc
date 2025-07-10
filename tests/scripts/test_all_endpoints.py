#!/usr/bin/env python3
"""
Comprehensive API endpoint test for the BTC Trading System
Tests all endpoints needed by the frontend implementation
"""

import json
import os
import time
from datetime import datetime, timedelta
import urllib.request
import urllib.error
from typing import Dict, List, Tuple, Optional

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
TEST_RESULTS = []

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_header(text: str):
    """Print a formatted header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text:^70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}\n")

def print_subheader(text: str):
    """Print a formatted subheader"""
    print(f"\n{Colors.BOLD}{Colors.PURPLE}{text}{Colors.END}")
    print(f"{Colors.PURPLE}{'-'*len(text)}{Colors.END}")

def print_test(name: str, endpoint: str, result: bool, details: str = ""):
    """Print test result"""
    status = f"{Colors.GREEN}✓{Colors.END}" if result else f"{Colors.RED}✗{Colors.END}"
    print(f"{status} {name:<40} {Colors.YELLOW}{endpoint}{Colors.END}")
    if details and not result:
        print(f"  {Colors.RED}→ {details}{Colors.END}")
    TEST_RESULTS.append((name, endpoint, result, details))

def test_api_endpoint(endpoint: str, method: str = "GET", data: dict = None, headers: dict = None) -> Tuple[bool, str, any]:
    """Test a single API endpoint and return response data"""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        
        if method == "GET":
            req = urllib.request.Request(url)
        elif method in ["POST", "PUT", "DELETE"]:
            json_data = json.dumps(data or {}).encode('utf-8')
            req = urllib.request.Request(url, data=json_data, method=method)
            req.add_header('Content-Type', 'application/json')
        else:
            return False, f"Unsupported method: {method}", None
        
        # Add custom headers
        if headers:
            for key, value in headers.items():
                req.add_header(key, value)
        
        with urllib.request.urlopen(req) as response:
            status_code = response.getcode()
            response_data = json.loads(response.read().decode('utf-8'))
            if status_code == 200:
                return True, f"Status: {status_code}", response_data
            else:
                return False, f"Status: {status_code}", response_data
    except urllib.error.HTTPError as e:
        try:
            error_data = json.loads(e.read().decode('utf-8'))
            return False, f"HTTP {e.code}: {error_data.get('detail', e.reason)}", None
        except:
            return False, f"HTTP {e.code}: {e.reason}", None
    except urllib.error.URLError as e:
        return False, f"Connection Error: {str(e.reason)}", None
    except Exception as e:
        return False, f"Error: {str(e)}", None

def test_core_endpoints():
    """Test core system endpoints"""
    print_subheader("Core System Endpoints")
    
    # Health check
    success, details, _ = test_api_endpoint("/health")
    print_test("Health Check", "/health", success, details)
    
    # System info
    success, details, _ = test_api_endpoint("/")
    print_test("System Info", "/", success, details)

def test_btc_endpoints():
    """Test BTC price and data endpoints"""
    print_subheader("BTC Price & Data Endpoints")
    
    # Current price
    success, details, _ = test_api_endpoint("/btc/latest")
    print_test("Current BTC Price", "/btc/latest", success, details)
    
    # Price history endpoints
    timeframes = ["1h", "4h", "1d", "1w", "1m"]
    for tf in timeframes:
        success, details, _ = test_api_endpoint(f"/btc/history/{tf}")
        print_test(f"Price History ({tf})", f"/btc/history/{tf}", success, details)
    
    # Market metrics
    success, details, _ = test_api_endpoint("/btc/metrics")
    print_test("Market Metrics", "/btc/metrics", success, details)

def test_signal_endpoints():
    """Test trading signal endpoints"""
    print_subheader("Trading Signal Endpoints")
    
    # Latest signals
    success, details, _ = test_api_endpoint("/signals/latest")
    print_test("Latest Signal", "/signals/latest", success, details)
    
    success, details, _ = test_api_endpoint("/signals/enhanced/latest")
    print_test("Enhanced LSTM Signal", "/signals/enhanced/latest", success, details)
    
    # Comprehensive signals
    success, details, _ = test_api_endpoint("/signals/comprehensive")
    print_test("Comprehensive Signals", "/signals/comprehensive", success, details)
    
    # Signal history
    success, details, _ = test_api_endpoint("/signals/history?hours=24")
    print_test("Signal History (24h)", "/signals/history?hours=24", success, details)
    
    # Indicator endpoints
    success, details, _ = test_api_endpoint("/indicators/technical")
    print_test("Technical Indicators", "/indicators/technical", success, details)
    
    success, details, _ = test_api_endpoint("/indicators/onchain")
    print_test("On-chain Indicators", "/indicators/onchain", success, details)
    
    success, details, _ = test_api_endpoint("/indicators/sentiment")
    print_test("Sentiment Indicators", "/indicators/sentiment", success, details)
    
    success, details, _ = test_api_endpoint("/indicators/macro")
    print_test("Macro Indicators", "/indicators/macro", success, details)

def test_portfolio_endpoints():
    """Test portfolio management endpoints"""
    print_subheader("Portfolio Management Endpoints")
    
    # Portfolio metrics
    success, details, _ = test_api_endpoint("/portfolio/metrics")
    print_test("Portfolio Metrics", "/portfolio/metrics", success, details)
    
    # Positions
    success, details, _ = test_api_endpoint("/portfolio/positions")
    print_test("Current Positions", "/portfolio/positions", success, details)
    
    # Performance
    success, details, _ = test_api_endpoint("/portfolio/performance/history")
    print_test("Performance History", "/portfolio/performance/history", success, details)
    
    # Trades
    success, details, _ = test_api_endpoint("/trades/all")
    print_test("All Trades", "/trades/all", success, details)
    
    success, details, _ = test_api_endpoint("/trades/recent?limit=10")
    print_test("Recent Trades", "/trades/recent?limit=10", success, details)

def test_paper_trading_endpoints():
    """Test paper trading endpoints"""
    print_subheader("Paper Trading Endpoints")
    
    # Status
    success, details, _ = test_api_endpoint("/paper-trading/status")
    print_test("Paper Trading Status", "/paper-trading/status", success, details)
    
    # Toggle
    success, details, _ = test_api_endpoint("/paper-trading/toggle", "POST")
    print_test("Toggle Paper Trading", "/paper-trading/toggle", success, details)
    
    # Reset
    success, details, _ = test_api_endpoint("/paper-trading/reset", "POST")
    print_test("Reset Portfolio", "/paper-trading/reset", success, details)
    
    # Execute trade
    trade_data = {
        "type": "buy",
        "amount": 0.001,
        "order_type": "market"
    }
    success, details, _ = test_api_endpoint("/paper-trading/trade", "POST", trade_data)
    print_test("Execute Trade", "/paper-trading/trade", success, details)
    
    # History
    success, details, _ = test_api_endpoint("/paper-trading/history")
    print_test("Trade History", "/paper-trading/history", success, details)

def test_analytics_endpoints():
    """Test analytics endpoints"""
    print_subheader("Analytics Endpoints")
    
    # Performance analytics
    success, details, _ = test_api_endpoint("/analytics/performance")
    print_test("Performance Analytics", "/analytics/performance", success, details)
    
    # Risk metrics
    success, details, _ = test_api_endpoint("/analytics/risk-metrics")
    print_test("Risk Metrics", "/analytics/risk-metrics", success, details)
    
    # Attribution
    success, details, _ = test_api_endpoint("/analytics/attribution")
    print_test("Performance Attribution", "/analytics/attribution", success, details)
    
    # P&L Analysis
    success, details, _ = test_api_endpoint("/analytics/pnl-analysis")
    print_test("P&L Analysis", "/analytics/pnl-analysis", success, details)
    
    # Feature importance
    success, details, _ = test_api_endpoint("/analytics/feature-importance")
    print_test("Feature Importance", "/analytics/feature-importance", success, details)
    
    # Market regime
    success, details, _ = test_api_endpoint("/analytics/market-regime")
    print_test("Market Regime", "/analytics/market-regime", success, details)
    
    # Backtesting
    backtest_params = {
        "start_date": (datetime.now() - timedelta(days=30)).isoformat(),
        "end_date": datetime.now().isoformat(),
        "initial_capital": 10000
    }
    success, details, _ = test_api_endpoint("/analytics/backtest", "POST", backtest_params)
    print_test("Backtesting", "/analytics/backtest", success, details)
    
    # Monte Carlo
    monte_carlo_params = {
        "num_simulations": 100,
        "time_horizon": 30
    }
    success, details, _ = test_api_endpoint("/analytics/monte-carlo", "POST", monte_carlo_params)
    print_test("Monte Carlo Simulation", "/analytics/monte-carlo", success, details)
    
    # Strategy optimization
    opt_params = {
        "strategy": "momentum",
        "optimize_for": "sharpe_ratio"
    }
    success, details, _ = test_api_endpoint("/analytics/optimize-strategy", "POST", opt_params)
    print_test("Strategy Optimization", "/analytics/optimize-strategy", success, details)
    
    # Time-based analytics
    success, details, _ = test_api_endpoint("/analytics/performance-by-dow")
    print_test("Performance by Day of Week", "/analytics/performance-by-dow", success, details)
    
    success, details, _ = test_api_endpoint("/analytics/performance-by-hour")
    print_test("Performance by Hour", "/analytics/performance-by-hour", success, details)
    
    # Strategy comparison
    success, details, _ = test_api_endpoint("/analytics/strategies")
    print_test("Strategy Comparison", "/analytics/strategies", success, details)

def test_config_endpoints():
    """Test configuration endpoints"""
    print_subheader("Configuration Endpoints")
    
    # Get current config
    success, details, _ = test_api_endpoint("/config/current")
    print_test("Get Current Config", "/config/current", success, details)
    
    # Update config
    config_update = {
        "trading_rules": {
            "max_position_size": 0.5,
            "stop_loss_percentage": 5.0
        }
    }
    success, details, _ = test_api_endpoint("/config/update", "POST", config_update)
    print_test("Update Config", "/config/update", success, details)
    
    # Reset config
    success, details, _ = test_api_endpoint("/config/reset", "POST")
    print_test("Reset Config", "/config/reset", success, details)
    
    # Export config
    success, details, _ = test_api_endpoint("/config/export")
    print_test("Export Config", "/config/export", success, details)
    
    # Import config
    import_data = {
        "config": {
            "trading": {"max_position_size": 0.5},
            "signals": {"rsi_weight": 0.2},
            "risk": {"stop_loss": 0.05},
            "data": {"lookback_days": 30}
        }
    }
    success, details, _ = test_api_endpoint("/config/import", "POST", import_data)
    print_test("Import Config", "/config/import", success, details)

def test_enhanced_lstm_endpoints():
    """Test Enhanced LSTM endpoints"""
    print_subheader("Enhanced LSTM Endpoints")
    
    # Status
    success, details, _ = test_api_endpoint("/enhanced-lstm/status")
    print_test("Enhanced LSTM Status", "/enhanced-lstm/status", success, details)
    
    # Data status
    success, details, _ = test_api_endpoint("/enhanced-lstm/data-status")
    print_test("Data Status", "/enhanced-lstm/data-status", success, details)
    
    # Prediction
    success, details, _ = test_api_endpoint("/enhanced-lstm/predict")
    print_test("Enhanced Prediction", "/enhanced-lstm/predict", success, details)
    
    # Train (skip if model exists)
    success, details, data = test_api_endpoint("/enhanced-lstm/status")
    if success and data and not data.get('is_trained', False):
        success, details, _ = test_api_endpoint("/enhanced-lstm/train", "POST")
        print_test("Train Enhanced Model", "/enhanced-lstm/train", success, details)
    else:
        print_test("Train Enhanced Model", "/enhanced-lstm/train", True, "Model already trained")

def test_notification_endpoints():
    """Test notification endpoints"""
    print_subheader("Notification Endpoints")
    
    # Test notification
    test_data = {"message": "Test notification from API testing"}
    success, details, _ = test_api_endpoint("/notifications/test", "POST", test_data)
    print_test("Test Notification", "/notifications/test", success, details)
    
    # Send notification
    notif_data = {
        "type": "signal",
        "title": "Test Signal",
        "message": "Test signal notification"
    }
    success, details, _ = test_api_endpoint("/notifications/send", "POST", notif_data)
    print_test("Send Notification", "/notifications/send", success, details)

def generate_summary_report():
    """Generate and display test summary"""
    print_header("Test Summary Report")
    
    total_tests = len(TEST_RESULTS)
    passed_tests = sum(1 for _, _, result, _ in TEST_RESULTS if result)
    failed_tests = total_tests - passed_tests
    pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    # Group results by category
    categories = {}
    for name, endpoint, result, details in TEST_RESULTS:
        category = endpoint.split('/')[1] if '/' in endpoint else 'core'
        if category not in categories:
            categories[category] = {'passed': 0, 'failed': 0, 'endpoints': []}
        
        if result:
            categories[category]['passed'] += 1
        else:
            categories[category]['failed'] += 1
            categories[category]['endpoints'].append((endpoint, details))
    
    # Display summary
    print(f"{Colors.BOLD}Overall Results:{Colors.END}")
    print(f"  Total Tests: {total_tests}")
    print(f"  {Colors.GREEN}Passed: {passed_tests}{Colors.END}")
    print(f"  {Colors.RED}Failed: {failed_tests}{Colors.END}")
    print(f"  {Colors.BOLD}Pass Rate: {pass_rate:.1f}%{Colors.END}")
    
    # Display by category
    print(f"\n{Colors.BOLD}Results by Category:{Colors.END}")
    for category, stats in sorted(categories.items()):
        total = stats['passed'] + stats['failed']
        cat_rate = (stats['passed'] / total * 100) if total > 0 else 0
        
        color = Colors.GREEN if cat_rate >= 80 else Colors.YELLOW if cat_rate >= 50 else Colors.RED
        print(f"\n  {Colors.BOLD}{category.upper()}{Colors.END}: {color}{cat_rate:.0f}%{Colors.END} ({stats['passed']}/{total})")
        
        if stats['failed'] > 0:
            print(f"    {Colors.RED}Failed endpoints:{Colors.END}")
            for endpoint, details in stats['endpoints']:
                print(f"      • {endpoint}")
                print(f"        {Colors.YELLOW}{details}{Colors.END}")
    
    # Save detailed report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"/app/reports/api_test_report_{timestamp}.json"
    
    report_data = {
        "timestamp": timestamp,
        "api_base_url": API_BASE_URL,
        "summary": {
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "pass_rate": pass_rate
        },
        "categories": categories,
        "results": [
            {
                "name": name,
                "endpoint": endpoint,
                "success": result,
                "details": details
            }
            for name, endpoint, result, details in TEST_RESULTS
        ]
    }
    
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    with open(report_file, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print(f"\n{Colors.BLUE}Detailed report saved to: {report_file}{Colors.END}")
    
    # Return exit code based on pass rate
    return 0 if pass_rate >= 70 else 1

def main():
    """Main test execution"""
    print(f"{Colors.BOLD}{Colors.BLUE}")
    print("="*70)
    print("BTC Trading System - Comprehensive API Test Suite".center(70))
    print("="*70)
    print(f"{Colors.END}")
    
    print(f"\nTesting API at: {Colors.YELLOW}{API_BASE_URL}{Colors.END}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    try:
        # Test connectivity
        print("Testing backend connectivity...")
        success, details, _ = test_api_endpoint("/health")
        if not success:
            # Try alternate endpoint
            success, details, _ = test_api_endpoint("/")
        
        if not success:
            print(f"{Colors.RED}ERROR: Cannot connect to backend at {API_BASE_URL}{Colors.END}")
            print(f"Details: {details}")
            return 1
        
        print(f"{Colors.GREEN}✓ Backend is accessible{Colors.END}\n")
        
        # Run all test suites
        test_core_endpoints()
        test_btc_endpoints()
        test_signal_endpoints()
        test_portfolio_endpoints()
        test_paper_trading_endpoints()
        test_analytics_endpoints()
        test_config_endpoints()
        test_enhanced_lstm_endpoints()
        test_notification_endpoints()
        
        # Generate summary
        return generate_summary_report()
        
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Testing interrupted by user{Colors.END}")
        return 1
    except Exception as e:
        print(f"\n{Colors.RED}Unexpected error: {str(e)}{Colors.END}")
        return 1

if __name__ == "__main__":
    exit(main())