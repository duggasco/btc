#!/usr/bin/env python3
"""Test paper trading functionality in the Flask frontend"""
import requests
import json
import time

BASE_URL = "http://localhost:8502"

def test_paper_trading():
    print("Testing Paper Trading Functionality")
    print("="*50)
    
    # 1. Check initial status
    print("\n1. Checking initial paper trading status...")
    response = requests.get(f"{BASE_URL}/api/paper-trading/status")
    status = response.json()
    print(f"Paper trading enabled: {bool(status['enabled'])}")
    print(f"Initial balance: ${status['portfolio']['usd_balance']}")
    
    # 2. Enable paper trading if not already enabled
    if not status['enabled']:
        print("\n2. Enabling paper trading...")
        response = requests.post(f"{BASE_URL}/api/paper-trading/toggle", 
                               json={"action": "enable"})
        print(f"Response: {response.json()['message']}")
    
    # 3. Get dashboard data
    print("\n3. Getting dashboard data...")
    response = requests.get(f"{BASE_URL}/api/dashboard-data")
    dashboard = response.json()
    print(f"Current BTC price: ${dashboard['price']['current']}")
    print(f"Current signal: {dashboard['signal']['current']} (confidence: {dashboard['signal']['confidence']}%)")
    print(f"Portfolio value: ${dashboard['portfolio']['total_value']}")
    print(f"Paper trading enabled in dashboard: {bool(dashboard['paper_trading_enabled'])}")
    
    # 4. Execute a paper trade
    print("\n4. Executing paper buy trade...")
    trade_data = {
        "trade_type": "buy",
        "quantity": 0.05,
        "price": None  # Market order
    }
    response = requests.post(f"{BASE_URL}/api/paper-trading/trade", json=trade_data)
    if response.status_code == 200:
        result = response.json()
        print(f"Trade executed: {result['message']}")
        if 'details' in result and 'trade' in result['details']:
            trade = result['details']['trade']
            print(f"  - Price: ${trade['price']}")
            print(f"  - Size: {trade['size']} BTC")
            print(f"  - Value: ${trade['value']}")
    else:
        print(f"Trade failed: {response.text}")
    
    # 5. Check updated portfolio
    time.sleep(1)  # Wait for update
    print("\n5. Checking updated portfolio...")
    response = requests.get(f"{BASE_URL}/api/dashboard-data")
    dashboard = response.json()
    portfolio = dashboard['portfolio']
    print(f"BTC balance: {portfolio['btc_balance']} BTC")
    print(f"USD balance: ${portfolio['usd_balance']}")
    print(f"Total value: ${portfolio['total_value']}")
    print(f"P&L: ${portfolio['pnl']} ({portfolio['pnl_percentage']}%)")
    
    # 6. Test chart data endpoint
    print("\n6. Testing chart data endpoint...")
    response = requests.get(f"{BASE_URL}/api/chart-data?timeframe=1H")
    if response.status_code == 200:
        chart_data = response.json()
        print(f"Chart data points: {len(chart_data)}")
        if chart_data:
            latest = chart_data[-1]
            print(f"Latest candle: ${latest['close']} at {latest['timestamp']}")
    
    print("\n" + "="*50)
    print("Paper Trading Test Complete!")
    print("\nNOTE: The UI will show:")
    print("- Trading mode toggle (Live/Paper) in the sidebar")
    print("- Paper badge on portfolio values when in paper mode")
    print("- 'Execute Paper Order' button text when in paper mode")
    print("- All trades executed are simulated with no real funds")

if __name__ == "__main__":
    test_paper_trading()