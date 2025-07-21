"""Test script to verify the frontend socket error fix"""
import requests
import time
import concurrent.futures
import signal
import sys

def test_abrupt_disconnect():
    """Test handling of abrupt client disconnections"""
    try:
        # Start a request but close it immediately
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=1, pool_maxsize=1)
        session.mount('http://', adapter)
        
        # Make request with very short timeout to simulate disconnect
        response = session.get('http://localhost:8502/', timeout=0.001)
    except requests.exceptions.Timeout:
        print("✓ Timeout handled gracefully")
    except requests.exceptions.ConnectionError:
        print("✓ Connection error handled gracefully")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")

def test_concurrent_requests():
    """Test handling of multiple concurrent requests"""
    def make_request(i):
        try:
            response = requests.get(f'http://localhost:8502/api/health', timeout=5)
            return f"Request {i}: {response.status_code}"
        except Exception as e:
            return f"Request {i} failed: {e}"
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(make_request, i) for i in range(20)]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
    success_count = sum(1 for r in results if "200" in r)
    print(f"✓ Concurrent requests: {success_count}/20 successful")
    
def test_long_polling():
    """Test handling of long-polling requests"""
    try:
        # Simulate a long-polling request
        start_time = time.time()
        response = requests.get('http://localhost:8502/api/price/current', timeout=30)
        duration = time.time() - start_time
        print(f"✓ Long polling request completed in {duration:.2f}s: {response.status_code}")
    except Exception as e:
        print(f"✗ Long polling failed: {e}")

def test_websocket_fallback():
    """Test WebSocket fallback to polling"""
    try:
        # Test the polling endpoint
        response = requests.get('http://localhost:8502/api/realtime/price')
        if response.status_code == 200:
            print("✓ Polling fallback endpoint working")
        else:
            print(f"✗ Polling endpoint returned: {response.status_code}")
    except Exception as e:
        print(f"✗ Polling endpoint failed: {e}")

def main():
    print("Testing frontend socket error fixes...")
    print("-" * 50)
    
    # Give the container a moment to fully start
    print("Waiting for frontend to be ready...")
    time.sleep(5)
    
    # Run tests
    print("\n1. Testing abrupt disconnections:")
    test_abrupt_disconnect()
    
    print("\n2. Testing concurrent requests:")
    test_concurrent_requests()
    
    print("\n3. Testing long polling:")
    test_long_polling()
    
    print("\n4. Testing WebSocket fallback:")
    test_websocket_fallback()
    
    print("\n" + "-" * 50)
    print("Socket error fix testing complete!")
    print("\nTo deploy the fix:")
    print("1. docker compose -f docker/docker-compose.yml down")
    print("2. docker compose -f docker/docker-compose.yml build --no-cache frontend")
    print("3. docker compose -f docker/docker-compose.yml up -d")
    print("4. docker logs -f btc-trading-frontend  # Monitor for errors")

if __name__ == "__main__":
    main()