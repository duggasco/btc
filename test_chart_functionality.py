#!/usr/bin/env python3
"""Test chart functionality after fixes"""
import requests
import json

def test_chart_endpoints():
    """Test all chart timeframes"""
    base_url = "http://localhost:8502/api/chart-data"
    timeframes = ['1H', '4H', '1D', '1W', '1M']
    
    print("Testing Chart Data Endpoints")
    print("=" * 50)
    
    for tf in timeframes:
        try:
            response = requests.get(f"{base_url}?timeframe={tf}")
            data = response.json()
            
            if isinstance(data, list) and len(data) > 0:
                first = data[0]
                last = data[-1]
                
                print(f"\nTimeframe: {tf}")
                print(f"  Data points: {len(data)}")
                print(f"  First date: {first.get('timestamp', 'N/A')}")
                print(f"  Last date: {last.get('timestamp', 'N/A')}")
                print(f"  Last close: ${last.get('close', 0):,.2f}")
                
                # Check data integrity
                has_ohlcv = all(
                    all(k in item for k in ['open', 'high', 'low', 'close', 'volume'])
                    for item in data[:5]  # Check first 5 items
                )
                print(f"  Has OHLCV data: {'✓' if has_ohlcv else '✗'}")
                
                # Check if close prices are non-zero
                non_zero_closes = sum(1 for item in data if item.get('close', 0) > 0)
                print(f"  Non-zero closes: {non_zero_closes}/{len(data)}")
                
            else:
                print(f"\nTimeframe: {tf} - No data returned")
                
        except Exception as e:
            print(f"\nTimeframe: {tf} - Error: {e}")
    
    # Test backend price history directly
    print("\n" + "=" * 50)
    print("Testing Backend Price History")
    print("=" * 50)
    
    backend_url = "http://localhost:8090/price/history"
    test_days = [2, 7, 30, 180]
    
    for days in test_days:
        try:
            response = requests.get(f"{backend_url}?days={days}")
            data = response.json()
            
            if isinstance(data, list) and len(data) > 0:
                print(f"\nDays: {days}")
                print(f"  Data points: {len(data)}")
                print(f"  Last price: ${data[-1].get('price', data[-1].get('close', 0)):,.2f}")
            else:
                print(f"\nDays: {days} - No data")
                
        except Exception as e:
            print(f"\nDays: {days} - Error: {e}")

if __name__ == "__main__":
    test_chart_endpoints()