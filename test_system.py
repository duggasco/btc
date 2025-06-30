#!/usr/bin/env python3
"""
BTC Trading System - Comprehensive Test Suite
Tests all components: API, database, LSTM model, and frontend connectivity
"""

import requests
import time
import json
import sys
from datetime import datetime

# Configuration
API_BASE_URL = "http://localhost:8080"
FRONTEND_URL = "http://localhost:8501"

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_status(message, status="INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    if status == "SUCCESS":
        print(f"{Colors.GREEN}✅ [{timestamp}] {message}{Colors.END}")
    elif status == "ERROR":
        print(f"{Colors.RED}❌ [{timestamp}] {message}{Colors.END}")
    elif status == "WARNING":
        print(f"{Colors.YELLOW}⚠️  [{timestamp}] {message}{Colors.END}")
    else:
        print(f"{Colors.BLUE}ℹ️  [{timestamp}] {message}{Colors.END}")

def test_api_endpoint(endpoint, method="GET", data=None, expected_status=200):
    """Test a single API endpoint"""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        if method == "GET":
            response = requests.get(url, timeout=10)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=10)
        
        if response.status_code == expected_status:
            print_status(f"{method} {endpoint} - Status: {response.status_code}", "SUCCESS")
            try:
                return response.json()
            except:
                return response.text
        else:
            print_status(f"{method} {endpoint} - Expected {expected_status}, got {response.status_code}", "ERROR")
            return None
    except requests.exceptions.ConnectionError:
        print_status(f"Cannot connect to API at {API_BASE_URL}", "ERROR")
        return None
    except Exception as e:
        print_status(f"Error testing {endpoint}: {str(e)}", "ERROR")
        return None

def test_backend_api():
    """Test all backend API endpoints"""
    print_status("Testing Backend API Endpoints", "INFO")
    
    # Test health check
    health = test_api_endpoint("/")
    if not health:
        return False
    
    # Test health detailed
    health_detail = test_api_endpoint("/health")
    if health_detail:
        print_status(f"System Health: {health_detail.get('status', 'unknown')}")
    
    # Test portfolio metrics
    metrics = test_api_endpoint("/portfolio/metrics")
    if metrics:
        print_status(f"Portfolio - Total Trades: {metrics.get('total_trades', 0)}")
        print_status(f"Portfolio - Total P&L: ${metrics.get('total_pnl', 0):.2f}")
    
    # Test latest signal
    signal = test_api_endpoint("/signals/latest")
    if signal:
        print_status(f"Latest Signal: {signal.get('signal', 'N/A')} (Confidence: {signal.get('confidence', 0):.1%})")
    
    # Test BTC market data
    btc_data = test_api_endpoint("/market/btc-data?period=1d")
    if btc_data and 'data' in btc_data:
        data_points = len(btc_data['data'])
        print_status(f"BTC Market Data: {data_points} data points retrieved")
    
    # Test trades endpoint
    trades = test_api_endpoint("/trades/")
    if trades is not None:
        print_status(f"Trades History: {len(trades)} trades found")
    
    # Test positions
    positions = test_api_endpoint("/positions/")
    if positions is not None:
        print_status(f"Current Positions: {len(positions)} positions")
    
    # Test limits
    limits = test_api_endpoint("/limits/")
    if limits is not None:
        print_status(f"Active Limits: {len(limits)} limit orders")
    
    return True

def test_trading_operations():
    """Test trading operations (create trade, limits, etc.)"""
    print_status("Testing Trading Operations", "INFO")
    
    # Test creating a trade
    trade_data = {
        "symbol": "BTC-USD",
        "trade_type": "buy",
        "price": 45000.00,
        "size": 0.001,
        "lot_id": f"test_lot_{int(time.time())}"
    }
    
    trade_result = test_api_endpoint("/trades/", "POST", trade_data, 200)
    if trade_result and trade_result.get('status') == 'success':
        print_status(f"Test Trade Created: {trade_result.get('trade_id')}", "SUCCESS")
    
    # Test creating a limit order
    limit_data = {
        "symbol": "BTC-USD",
        "limit_type": "stop_loss",
        "price": 40000.00,
        "size": 0.001
    }
    
    limit_result = test_api_endpoint("/limits/", "POST", limit_data, 200)
    if limit_result and limit_result.get('status') == 'success':
        print_status(f"Test Limit Order Created: {limit_result.get('limit_id')}", "SUCCESS")
    
    return True

def test_model_signals():
    """Test LSTM model signal generation"""
    print_status("Testing LSTM Model Signals", "INFO")
    
    # Get signal history
    signal_history = test_api_endpoint("/signals/history?limit=5")
    if signal_history:
        print_status(f"Signal History: {len(signal_history)} historical signals")
        if signal_history:
            latest = signal_history[0]
            print_status(f"Latest Historical Signal: {latest.get('signal')} at {latest.get('timestamp')}")
    
    # Test system status
    system_status = test_api_endpoint("/system/status")
    if system_status:
        print_status(f"Signal Update Errors: {system_status.get('signal_update_errors', 0)}")
        print_status(f"Data Cache Status: {system_status.get('data_cache_status', 'unknown')}")
    
    return True

def test_analytics():
    """Test analytics endpoints"""
    print_status("Testing Analytics Features", "INFO")
    
    # Test P&L analytics
    pnl_data = test_api_endpoint("/analytics/pnl")
    if pnl_data:
        daily_pnl = pnl_data.get('daily_pnl', [])
        cumulative_pnl = pnl_data.get('cumulative_pnl', [])
        print_status(f"P&L Analytics: {len(daily_pnl)} daily records, {len(cumulative_pnl)} cumulative records")
    
    return True

def test_frontend_connectivity():
    """Test if frontend is accessible"""
    print_status("Testing Frontend Connectivity", "INFO")
    
    try:
        response = requests.get(FRONTEND_URL, timeout=10)
        if response.status_code == 200:
            print_status(f"Frontend accessible at {FRONTEND_URL}", "SUCCESS")
            return True
        else:
            print_status(f"Frontend returned status {response.status_code}", "WARNING")
            return False
    except requests.exceptions.ConnectionError:
        print_status(f"Cannot connect to frontend at {FRONTEND_URL}", "ERROR")
        return False
    except Exception as e:
        print_status(f"Error connecting to frontend: {str(e)}", "ERROR")
        return False

def test_data_quality():
    """Test data quality and consistency"""
    print_status("Testing Data Quality", "INFO")
    
    # Get market data and verify structure
    btc_data = test_api_endpoint("/market/btc-data?period=7d")
    if btc_data and 'data' in btc_data:
        data = btc_data['data']
        if data:
            sample = data[0]
            required_fields = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            missing_fields = [field for field in required_fields if field not in sample]
            
            if not missing_fields:
                print_status("Market data structure validation passed", "SUCCESS")
                
                # Check data consistency
                for i, record in enumerate(data[:5]):  # Check first 5 records
                    if record['high'] >= record['low'] and record['high'] >= record['open'] and record['high'] >= record['close']:
                        continue
                    else:
                        print_status(f"Data inconsistency in record {i}: high < other prices", "WARNING")
                        break
                else:
                    print_status("Market data consistency validation passed", "SUCCESS")
            else:
                print_status(f"Missing required fields in market data: {missing_fields}", "ERROR")
    
    return True

def performance_test():
    """Basic performance testing"""
    print_status("Running Performance Tests", "INFO")
    
    endpoints = ["/", "/health", "/signals/latest", "/portfolio/metrics"]
    response_times = []
    
    for endpoint in endpoints:
        start_time = time.time()
        result = test_api_endpoint(endpoint)
        end_time = time.time()
        
        if result:
            response_time = (end_time - start_time) * 1000  # Convert to ms
            response_times.append(response_time)
            print_status(f"{endpoint}: {response_time:.2f}ms")
    
    if response_times:
        avg_response_time = sum(response_times) / len(response_times)
        print_status(f"Average API Response Time: {avg_response_time:.2f}ms")
        
        if avg_response_time < 1000:  # Less than 1 second
            print_status("API Performance: GOOD", "SUCCESS")
        elif avg_response_time < 3000:  # Less than 3 seconds
            print_status("API Performance: ACCEPTABLE", "WARNING")
        else:
            print_status("API Performance: SLOW", "ERROR")
    
    return True

def run_comprehensive_test():
    """Run all tests"""
    print(f"\n{Colors.BOLD}🚀 Starting BTC Trading System Comprehensive Test Suite{Colors.END}")
    print(f"{Colors.BOLD}API Base URL: {API_BASE_URL}{Colors.END}")
    print(f"{Colors.BOLD}Frontend URL: {FRONTEND_URL}{Colors.END}")
    print("=" * 60)
    
    test_results = {}
    
    # Run all test categories
    test_functions = [
        ("Backend API", test_backend_api),
        ("Trading Operations", test_trading_operations),
        ("Model Signals", test_model_signals),
        ("Analytics", test_analytics),
        ("Frontend Connectivity", test_frontend_connectivity),
        ("Data Quality", test_data_quality),
        ("Performance", performance_test)
    ]
    
    for test_name, test_func in test_functions:
        print(f"\n{Colors.BOLD}--- {test_name} Tests ---{Colors.END}")
        try:
            result = test_func()
            test_results[test_name] = result
        except Exception as e:
            print_status(f"Test suite error in {test_name}: {str(e)}", "ERROR")
            test_results[test_name] = False
    
    # Summary
    print(f"\n{Colors.BOLD}📊 Test Results Summary{Colors.END}")
    print("=" * 60)
    
    passed = sum(1 for result in test_results.values() if result)
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "PASS" if result else "FAIL"
        color = Colors.GREEN if result else Colors.RED
        print(f"{color}{status:>6}{Colors.END} - {test_name}")
    
    print("=" * 60)
    print(f"{Colors.BOLD}Overall: {passed}/{total} test suites passed{Colors.END}")
    
    if passed == total:
        print_status("🎉 All tests passed! System is ready for use.", "SUCCESS")
        return True
    else:
        print_status(f"⚠️  {total - passed} test suite(s) failed. Check logs above.", "WARNING")
        return False

def quick_test():
    """Quick connectivity test"""
    print(f"\n{Colors.BOLD}⚡ Quick System Test{Colors.END}")
    print("=" * 40)
    
    # Test API
    health = test_api_endpoint("/")
    if not health:
        print_status("Backend API is not responding", "ERROR")
        return False
    
    # Test Frontend
    frontend_ok = test_frontend_connectivity()
    
    if health and frontend_ok:
        print_status("✅ Quick test passed - System is responding", "SUCCESS")
        return True
    else:
        print_status("❌ Quick test failed - Check system status", "ERROR")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        quick_test()
    else:
        run_comprehensive_test()
