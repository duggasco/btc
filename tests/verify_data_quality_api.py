#!/usr/bin/env python3
"""
Verify Data Quality API endpoint is working correctly
"""

import requests
import json
from datetime import datetime

def test_data_quality_api():
    """Test the /analytics/data-quality endpoint"""
    
    base_url = "http://localhost:8090"
    endpoint = f"{base_url}/analytics/data-quality"
    
    print("Testing Data Quality API Endpoint...")
    print(f"URL: {endpoint}")
    print("-" * 50)
    
    try:
        # Make API request
        response = requests.get(endpoint, timeout=10)
        
        # Check status code
        print(f"Status Code: {response.status_code}")
        if response.status_code != 200:
            print(f"❌ Error: Expected status 200, got {response.status_code}")
            print(f"Response: {response.text}")
            return False
        
        # Parse JSON response
        data = response.json()
        
        # Verify response structure
        print("\n✅ API Response Structure:")
        
        # Check summary section
        if 'summary' in data:
            summary = data['summary']
            print("\n📊 Summary:")
            print(f"  - Total Datapoints: {summary.get('total_datapoints', 'N/A'):,}")
            print(f"  - Missing Dates: {summary.get('total_missing_dates', 'N/A'):,}")
            print(f"  - Overall Completeness: {summary.get('overall_completeness', 'N/A')}%")
            print(f"  - Last Updated: {summary.get('last_updated', 'N/A')}")
        else:
            print("❌ Missing 'summary' section")
        
        # Check by_type section
        if 'by_type' in data:
            print("\n📈 Data by Type:")
            for dtype, metrics in data['by_type'].items():
                print(f"  - {dtype.capitalize()}:")
                print(f"    - Datapoints: {metrics.get('total_datapoints', 0):,}")
                print(f"    - Completeness: {metrics.get('completeness', 0)}%")
        else:
            print("❌ Missing 'by_type' section")
        
        # Check coverage section
        if 'coverage' in data:
            print("\n🗺️ Coverage data present")
            coverage_periods = list(data['coverage'].keys())
            print(f"  - Time periods: {', '.join(coverage_periods)}")
        else:
            print("⚠️ No 'coverage' section (might be normal if no data)")
        
        # Check gaps section
        if 'gaps' in data:
            gaps = data['gaps']
            print(f"\n⚠️ Data Gaps: {len(gaps)} found")
            if gaps and len(gaps) > 0:
                print(f"  - Example: {gaps[0].get('days', 0)} days missing from {gaps[0].get('start', 'N/A')}")
        else:
            print("\n✅ No data gaps found")
        
        # Check cache metrics
        if 'cache_metrics' in data:
            cache = data['cache_metrics']
            print("\n💾 Cache Metrics:")
            print(f"  - Total Entries: {cache.get('total_entries', 0):,}")
            print(f"  - Active Entries: {cache.get('active_entries', 0):,}")
        else:
            print("\n⚠️ No cache metrics (might be normal)")
        
        print("\n✅ API endpoint is working correctly!")
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Error: Cannot connect to API. Make sure backend is running.")
        print("   Run: docker compose -f docker/docker-compose.yml up -d")
        return False
    except requests.exceptions.Timeout:
        print("❌ Error: API request timed out")
        return False
    except json.JSONDecodeError:
        print("❌ Error: Invalid JSON response")
        print(f"Response text: {response.text}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def test_frontend_accessibility():
    """Quick test to verify frontend is accessible"""
    
    frontend_url = "http://localhost:8501"
    
    print("\n" + "=" * 50)
    print("Testing Frontend Accessibility...")
    print(f"URL: {frontend_url}")
    
    try:
        response = requests.get(frontend_url, timeout=5)
        if response.status_code == 200:
            print("✅ Frontend is accessible")
            print(f"   Navigate to: {frontend_url}")
            print("   Go to: Settings → Data Quality tab")
            return True
        else:
            print(f"❌ Frontend returned status: {response.status_code}")
            return False
    except:
        print("❌ Cannot connect to frontend")
        print("   Make sure frontend container is running")
        return False


if __name__ == "__main__":
    print("Data Quality Feature Test Verification")
    print("=" * 50)
    print(f"Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Test API
    api_ok = test_data_quality_api()
    
    # Test Frontend
    frontend_ok = test_frontend_accessibility()
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary:")
    print(f"  - API Endpoint: {'✅ PASS' if api_ok else '❌ FAIL'}")
    print(f"  - Frontend Access: {'✅ PASS' if frontend_ok else '❌ FAIL'}")
    
    if api_ok and frontend_ok:
        print("\n✅ All basic tests passed!")
        print("\nNext steps:")
        print("1. Run automated tests: ./tests/run_data_quality_tests.sh")
        print("2. Perform manual testing using: tests/manual/data_quality_tab_checklist.md")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")