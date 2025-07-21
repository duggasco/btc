#!/usr/bin/env python3
import requests
import sys

# Test direct backend access
try:
    print("Testing backend API directly...")
    response = requests.get("http://localhost:8090/health", timeout=5)
    print(f"Backend health check: {response.status_code}")
    print(f"Response: {response.json()}")
    
    response = requests.get("http://localhost:8090/price/current", timeout=5)
    print(f"\nBackend current price: {response.json()}")
except Exception as e:
    print(f"Error accessing backend: {e}")

# Test frontend dashboard API
try:
    print("\n\nTesting frontend dashboard API...")
    response = requests.get("http://localhost:8502/api/dashboard-data", timeout=10)
    print(f"Frontend dashboard API: {response.status_code}")
    data = response.json()
    print(f"Price: ${data['price']['current']}")
    print(f"Signal: {data['signal']['current']} (confidence: {data['signal']['confidence']}%)")
    print(f"Paper trading enabled: {data['paper_trading_enabled']}")
    print(f"System status: {data['system_status']}")
except Exception as e:
    print(f"Error accessing frontend: {e}")