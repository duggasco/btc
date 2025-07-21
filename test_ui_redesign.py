#!/usr/bin/env python3
"""Test script for the new 3-page UI redesign"""
import os
import sys
import subprocess
import time
import requests
import json

def print_status(message, status="INFO"):
    """Print colored status messages"""
    colors = {
        "INFO": "\033[94m",
        "SUCCESS": "\033[92m",
        "WARNING": "\033[93m",
        "ERROR": "\033[91m"
    }
    reset = "\033[0m"
    print(f"{colors.get(status, '')}[{status}] {message}{reset}")

def test_backend_api():
    """Test backend API endpoints"""
    print_status("Testing backend API endpoints...")
    
    base_url = "http://localhost:8090"
    endpoints = [
        ("/health", "GET"),
        ("/system/status", "GET"),
        ("/price/current", "GET"),
        ("/signals/enhanced/latest", "GET"),
        ("/portfolio/metrics", "GET"),
        ("/config/current", "GET"),
    ]
    
    results = []
    for endpoint, method in endpoints:
        try:
            url = f"{base_url}{endpoint}"
            response = requests.request(method, url, timeout=5)
            status = "SUCCESS" if response.status_code == 200 else "ERROR"
            results.append((endpoint, response.status_code, status))
            print_status(f"{endpoint}: {response.status_code}", status)
        except Exception as e:
            results.append((endpoint, "ERROR", str(e)))
            print_status(f"{endpoint}: {str(e)}", "ERROR")
    
    return results

def test_frontend_pages():
    """Test frontend page loading"""
    print_status("Testing frontend pages...")
    
    pages = [
        "app.py",
        "pages/1_Trading_Dashboard.py",
        "pages/2_Analytics_Research.py",
        "pages/3_Settings_Configuration.py"
    ]
    
    results = []
    for page in pages:
        page_path = f"/root/btc/src/frontend/{page}"
        if os.path.exists(page_path):
            print_status(f"{page}: Found", "SUCCESS")
            results.append((page, "EXISTS"))
        else:
            print_status(f"{page}: Not found", "ERROR")
            results.append((page, "MISSING"))
    
    return results

def test_css_files():
    """Test CSS files exist"""
    print_status("Testing CSS files...")
    
    css_files = [
        "src/frontend/styles/theme.css",
        "src/frontend/styles/components.css"
    ]
    
    results = []
    for css_file in css_files:
        css_path = f"/root/btc/{css_file}"
        if os.path.exists(css_path):
            with open(css_path, 'r') as f:
                content = f.read()
                if '--bg-primary' in content and '--accent-primary' in content:
                    print_status(f"{css_file}: Valid", "SUCCESS")
                    results.append((css_file, "VALID"))
                else:
                    print_status(f"{css_file}: Missing theme variables", "WARNING")
                    results.append((css_file, "INVALID"))
        else:
            print_status(f"{css_file}: Not found", "ERROR")
            results.append((css_file, "MISSING"))
    
    return results

def test_components():
    """Test component files exist"""
    print_status("Testing component files...")
    
    components = [
        "components/layout/dashboard_grid.py",
        "components/display/metric_card.py",
        "components/display/signal_badge.py",
        "components/display/data_table.py",
        "components/display/chart_container.py",
        "components/controls/form_controls.py"
    ]
    
    results = []
    for component in components:
        component_path = f"/root/btc/src/frontend/{component}"
        if os.path.exists(component_path):
            print_status(f"{component}: Found", "SUCCESS")
            results.append((component, "EXISTS"))
        else:
            print_status(f"{component}: Not found", "ERROR")
            results.append((component, "MISSING"))
    
    return results

def test_ui_integration():
    """Test UI integration points"""
    print_status("Testing UI integration...")
    
    # Check if old pages are removed
    old_pages = [
        "1_🏠_Dashboard.py",
        "2_📈_Signals.py",
        "3_💰_Portfolio.py",
        "4_🔬_Analytics.py",
        "5_📄_Paper_Trading.py",
        "6_⚙️_Settings.py"
    ]
    
    results = []
    for old_page in old_pages:
        page_path = f"/root/btc/src/frontend/pages/{old_page}"
        if not os.path.exists(page_path):
            print_status(f"Old page {old_page}: Removed", "SUCCESS")
            results.append((old_page, "REMOVED"))
        else:
            print_status(f"Old page {old_page}: Still exists", "ERROR")
            results.append((old_page, "EXISTS"))
    
    return results

def generate_report(all_results):
    """Generate test report"""
    print("\n" + "="*60)
    print("UI REDESIGN TEST REPORT")
    print("="*60)
    
    total_tests = sum(len(results) for results in all_results.values())
    passed_tests = 0
    for category, results in all_results.items():
        if category == "api":
            # API results have 3 values (endpoint, status_code, status)
            passed_tests += sum(1 for _, status_code, status in results if status == "SUCCESS")
        else:
            # Other results have 2 values (name, status)
            passed_tests += sum(1 for _, status in results if status in ["EXISTS", "VALID", "REMOVED"])
    
    print(f"\nTotal Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    print("\n" + "-"*60)
    print("SUMMARY:")
    print("-"*60)
    
    if passed_tests == total_tests:
        print_status("All tests passed! UI redesign is complete.", "SUCCESS")
    else:
        print_status(f"{total_tests - passed_tests} tests failed. Review the issues above.", "WARNING")
    
    # Save report
    report_path = "/root/btc/ui_redesign_test_report.json"
    with open(report_path, 'w') as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": total_tests - passed_tests,
            "results": all_results
        }, f, indent=2)
    
    print(f"\nDetailed report saved to: {report_path}")

def main():
    """Main test function"""
    print("="*60)
    print("BTC TRADING SYSTEM - UI REDESIGN TEST")
    print("="*60)
    print()
    
    all_results = {}
    
    # Test CSS files
    all_results["css"] = test_css_files()
    print()
    
    # Test components
    all_results["components"] = test_components()
    print()
    
    # Test frontend pages
    all_results["pages"] = test_frontend_pages()
    print()
    
    # Test UI integration
    all_results["integration"] = test_ui_integration()
    print()
    
    # Test backend API
    try:
        api_results = test_backend_api()
        all_results["api"] = api_results
    except Exception as e:
        print_status(f"Backend API test skipped: {str(e)}", "WARNING")
        all_results["api"] = [("skipped", "Backend not running")]
    
    # Generate report
    generate_report(all_results)

if __name__ == "__main__":
    main()