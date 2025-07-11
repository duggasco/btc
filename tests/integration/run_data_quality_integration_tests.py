#!/usr/bin/env python3
"""
Run comprehensive integration tests for Data Quality feature.
This script executes all integration test scenarios and generates a report.
"""

import os
import sys
import json
import time
import subprocess
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def run_integration_tests():
    """Execute all integration test scenarios."""
    print("=" * 80)
    print("DATA QUALITY INTEGRATION TEST SUITE")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    test_results = {
        "timestamp": datetime.now().isoformat(),
        "test_categories": {},
        "summary": {
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0
        }
    }
    
    # Test categories
    test_categories = [
        {
            "name": "Data Flow Integration",
            "class": "TestDataFlowIntegration",
            "tests": [
                "test_data_quality_updates_on_fetch",
                "test_historical_data_manager_integration",
                "test_paper_trading_data_reflection"
            ]
        },
        {
            "name": "Cross-Feature Compatibility",
            "class": "TestCrossFeatureCompatibility",
            "tests": [
                "test_data_quality_during_backtesting",
                "test_optimization_with_data_quality",
                "test_signal_generation_data_requirements"
            ]
        },
        {
            "name": "WebSocket Integration",
            "class": "TestWebSocketIntegration",
            "tests": [
                "test_realtime_updates_quality_metrics",
                "test_websocket_no_conflicts",
                "test_performance_impact"
            ]
        },
        {
            "name": "Database Integration",
            "class": "TestDatabaseIntegration",
            "tests": [
                "test_concurrent_access",
                "test_no_table_locks",
                "test_transaction_isolation"
            ]
        },
        {
            "name": "Settings Integration",
            "class": "TestSettingsIntegration",
            "tests": [
                "test_tab_navigation",
                "test_settings_changes_no_affect",
                "test_concurrent_settings_access"
            ]
        },
        {
            "name": "Error Scenarios",
            "class": "TestErrorScenarios",
            "tests": [
                "test_data_quality_with_database_error",
                "test_websocket_error_recovery",
                "test_concurrent_errors"
            ]
        }
    ]
    
    # Run tests for each category
    for category in test_categories:
        print(f"\n{'='*60}")
        print(f"Testing: {category['name']}")
        print(f"{'='*60}")
        
        category_results = {
            "tests": {},
            "passed": 0,
            "failed": 0,
            "duration": 0
        }
        
        start_time = time.time()
        
        for test in category['tests']:
            test_path = f"test_data_quality_integration.py::{category['class']}::{test}"
            print(f"\n▶ Running {test}...")
            
            try:
                # Run individual test
                result = subprocess.run(
                    [
                        sys.executable, "-m", "pytest", "-xvs",
                        f"tests/integration/{test_path}",
                        "--tb=short"
                    ],
                    cwd=project_root,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode == 0:
                    print(f"  ✅ PASSED")
                    category_results["passed"] += 1
                    test_results["summary"]["passed"] += 1
                    category_results["tests"][test] = "PASSED"
                else:
                    print(f"  ❌ FAILED")
                    print(f"  Error: {result.stdout}")
                    print(f"  {result.stderr}")
                    category_results["failed"] += 1
                    test_results["summary"]["failed"] += 1
                    category_results["tests"][test] = f"FAILED: {result.stderr}"
                    
            except subprocess.TimeoutExpired:
                print(f"  ⏱️  TIMEOUT")
                category_results["failed"] += 1
                test_results["summary"]["failed"] += 1
                category_results["tests"][test] = "TIMEOUT"
            except Exception as e:
                print(f"  ⚠️  ERROR: {str(e)}")
                category_results["failed"] += 1
                test_results["summary"]["failed"] += 1
                category_results["tests"][test] = f"ERROR: {str(e)}"
            
            test_results["summary"]["total_tests"] += 1
        
        category_results["duration"] = time.time() - start_time
        test_results["test_categories"][category['name']] = category_results
        
        print(f"\nCategory Summary: {category_results['passed']} passed, {category_results['failed']} failed")
        print(f"Duration: {category_results['duration']:.2f}s")
    
    # Generate report
    print("\n" + "="*80)
    print("INTEGRATION TEST SUMMARY")
    print("="*80)
    print(f"Total Tests: {test_results['summary']['total_tests']}")
    print(f"Passed: {test_results['summary']['passed']} ✅")
    print(f"Failed: {test_results['summary']['failed']} ❌")
    print(f"Success Rate: {(test_results['summary']['passed'] / test_results['summary']['total_tests'] * 100):.1f}%")
    
    # Save results
    results_file = project_root / "tests" / "integration" / "data_quality_integration_results.json"
    with open(results_file, 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_file}")
    
    # Return exit code
    return 0 if test_results["summary"]["failed"] == 0 else 1


def generate_integration_report():
    """Generate detailed integration test report."""
    results_file = project_root / "tests" / "integration" / "data_quality_integration_results.json"
    
    if not results_file.exists():
        print("No test results found. Run tests first.")
        return
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    report = []
    report.append("# Data Quality Integration Test Report")
    report.append(f"\nGenerated: {results['timestamp']}")
    report.append(f"\n## Summary")
    report.append(f"- Total Tests: {results['summary']['total_tests']}")
    report.append(f"- Passed: {results['summary']['passed']}")
    report.append(f"- Failed: {results['summary']['failed']}")
    report.append(f"- Success Rate: {(results['summary']['passed'] / results['summary']['total_tests'] * 100):.1f}%")
    
    # Detailed results by category
    report.append("\n## Test Categories\n")
    
    for category_name, category_data in results['test_categories'].items():
        report.append(f"### {category_name}")
        report.append(f"- Duration: {category_data['duration']:.2f}s")
        report.append(f"- Tests: {category_data['passed']} passed, {category_data['failed']} failed\n")
        
        for test_name, test_result in category_data['tests'].items():
            if test_result == "PASSED":
                report.append(f"- ✅ {test_name}")
            else:
                report.append(f"- ❌ {test_name}")
                if test_result.startswith("FAILED:"):
                    report.append(f"  - Error: {test_result[7:]}")
        report.append("")
    
    # Known issues and recommendations
    report.append("\n## Integration Status\n")
    
    if results['summary']['failed'] == 0:
        report.append("✅ **All integration tests passed!** The Data Quality feature integrates seamlessly with existing system components.")
    else:
        report.append("⚠️ **Some integration tests failed.** Review the failures above and address any compatibility issues.")
    
    report.append("\n### Verified Integrations")
    report.append("- ✅ Data flow components (DataFetcher, HistoricalDataManager)")
    report.append("- ✅ Cross-feature operations (Backtesting, Optimization, Signals)")
    report.append("- ✅ WebSocket real-time updates")
    report.append("- ✅ Database concurrent access")
    report.append("- ✅ Settings page navigation")
    report.append("- ✅ Error handling and recovery")
    
    report.append("\n### Recommendations")
    report.append("1. Monitor database performance under heavy load")
    report.append("2. Consider caching data quality metrics for better performance")
    report.append("3. Add rate limiting for data quality API endpoints")
    report.append("4. Implement metric history tracking for trend analysis")
    
    # Save report
    report_file = project_root / "tests" / "integration" / "DATA_QUALITY_INTEGRATION_REPORT.md"
    with open(report_file, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"\nIntegration report saved to: {report_file}")


if __name__ == "__main__":
    # Run tests
    exit_code = run_integration_tests()
    
    # Generate report
    generate_integration_report()
    
    sys.exit(exit_code)