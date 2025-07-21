#!/usr/bin/env python3
"""Test price display in the frontend"""
import requests
from bs4 import BeautifulSoup
import time

# Wait for frontend to fully load
print("Testing price display...")
time.sleep(2)

# Get the main page HTML
response = requests.get("http://localhost:8502/")
soup = BeautifulSoup(response.text, 'html.parser')

# Find all elements that might contain price
price_elements = []

# Check metrics bar
btc_price = soup.find(id='btc-price')
if btc_price:
    price_elements.append(('Metrics Bar BTC Price', btc_price.text))

# Check main price display
current_price = soup.find(id='current-price-value')
if current_price:
    price_elements.append(('Main Price Display', current_price.text))

# Check signal price
signal_price = soup.find(id='signal-price-value')
if signal_price:
    price_elements.append(('Signal Price', signal_price.text))

# Get dashboard data
api_response = requests.get("http://localhost:8502/api/dashboard-data")
api_data = api_response.json()

print("\n=== Price Display Test Results ===")
print("\nHTML Elements:")
for name, value in price_elements:
    print(f"  {name}: {value}")

print(f"\nAPI Response:")
print(f"  Current Price: ${api_data['price']['current']:,.2f}")
print(f"  24h Change: {api_data['price']['change_24h']:.2f}%")
print(f"  Signal Price: ${api_data['signal']['price']:,.2f}")

# Check for any "95000" in the HTML
if "95000" in response.text or "95,000" in response.text:
    print("\nWARNING: Found '95000' or '95,000' in the HTML!")
    # Find where it appears
    lines = response.text.split('\n')
    for i, line in enumerate(lines):
        if "95000" in line or "95,000" in line:
            print(f"  Line {i+1}: {line.strip()[:100]}...")
else:
    print("\nNo '95000' found in HTML - prices should be displaying correctly")

print("\n=== End Test ===")