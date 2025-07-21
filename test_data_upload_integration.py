#!/usr/bin/env python3
"""Test data upload functionality end-to-end"""
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def create_sample_data_files():
    """Create sample data files for testing"""
    
    # Create test directory
    os.makedirs('test_data', exist_ok=True)
    
    # 1. Create OHLCV data for BTC
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    btc_ohlcv = pd.DataFrame({
        'date': dates,
        'open': np.random.uniform(110000, 120000, 30),
        'high': np.random.uniform(115000, 125000, 30),
        'low': np.random.uniform(105000, 115000, 30),
        'close': np.random.uniform(110000, 120000, 30),
        'volume': np.random.uniform(10000, 50000, 30)
    })
    btc_ohlcv.to_csv('test_data/btc_ohlcv.csv', index=False)
    print("Created: test_data/btc_ohlcv.csv")
    
    # 2. Create OHLCV data for GLD (Gold ETF)
    gld_ohlcv = pd.DataFrame({
        'date': dates,
        'open': np.random.uniform(180, 190, 30),
        'high': np.random.uniform(185, 195, 30),
        'low': np.random.uniform(175, 185, 30),
        'close': np.random.uniform(180, 190, 30),
        'volume': np.random.uniform(1000000, 5000000, 30)
    })
    gld_ohlcv.to_csv('test_data/gld_ohlcv.csv', index=False)
    print("Created: test_data/gld_ohlcv.csv")
    
    # 3. Create VIX data
    vix_data = pd.DataFrame({
        'date': dates,
        'open': np.random.uniform(12, 25, 30),
        'high': np.random.uniform(15, 30, 30),
        'low': np.random.uniform(10, 20, 30),
        'close': np.random.uniform(12, 25, 30),
        'volume': 0  # VIX doesn't have volume
    })
    vix_data.to_csv('test_data/vix_ohlcv.csv', index=False)
    print("Created: test_data/vix_ohlcv.csv")
    
    # 4. Create On-chain data
    onchain_data = pd.DataFrame({
        'date': dates,
        'active_addresses': np.random.randint(800000, 1200000, 30),
        'transaction_count': np.random.randint(250000, 400000, 30),
        'hash_rate': np.random.uniform(400e18, 600e18, 30),
        'difficulty': np.random.uniform(60e12, 80e12, 30),
        'block_size': np.random.uniform(1.2, 1.8, 30),
        'fees_total': np.random.uniform(20, 40, 30),
        'supply': 19700000 + np.arange(30) * 900  # Simulating supply growth
    })
    onchain_data.to_csv('test_data/btc_onchain.csv', index=False)
    print("Created: test_data/btc_onchain.csv")
    
    # 5. Create Sentiment data
    sentiment_data = pd.DataFrame({
        'timestamp': pd.date_range(end=datetime.now(), periods=100, freq='H'),
        'fear_greed_index': np.random.randint(20, 80, 100),
        'reddit_sentiment': np.random.uniform(0.3, 0.8, 100),
        'twitter_sentiment': np.random.uniform(0.2, 0.9, 100),
        'news_sentiment': np.random.uniform(0.4, 0.7, 100),
        'google_trends': np.random.randint(40, 100, 100)
    })
    sentiment_data.to_csv('test_data/btc_sentiment.csv', index=False)
    print("Created: test_data/btc_sentiment.csv")
    
    # 6. Create Macro data (mixed indicators)
    macro_data = pd.DataFrame({
        'date': dates.repeat(4),  # 4 indicators per day
        'symbol': ['DXY', 'TNX', 'GLD', 'SPY'] * 30,
        'value': np.random.uniform(50, 150, 120),
        'indicator_type': ['currency_index', 'bond_yield', 'commodity', 'equity_index'] * 30
    })
    macro_data.to_csv('test_data/macro_indicators.csv', index=False)
    print("Created: test_data/macro_indicators.csv")
    
    # 7. Create Excel file with multiple sheets
    with pd.ExcelWriter('test_data/combined_data.xlsx') as writer:
        btc_ohlcv.to_excel(writer, sheet_name='BTC_OHLCV', index=False)
        onchain_data.to_excel(writer, sheet_name='OnChain', index=False)
        sentiment_data.head(30).to_excel(writer, sheet_name='Sentiment', index=False)
    print("Created: test_data/combined_data.xlsx")
    
    print("\nAll test files created successfully!")
    return True

def test_upload_endpoints():
    """Test the upload endpoints"""
    import requests
    
    base_url = "http://localhost:8502"
    
    # Test 1: Check if data upload page loads
    print("\n1. Testing data upload page...")
    response = requests.get(f"{base_url}/data/upload")
    print(f"   Status: {response.status_code}")
    print(f"   Page loads: {'Yes' if response.status_code == 200 else 'No'}")
    
    # Test 2: Check if quality page loads
    print("\n2. Testing data quality page...")
    response = requests.get(f"{base_url}/data/quality")
    print(f"   Status: {response.status_code}")
    print(f"   Page loads: {'Yes' if response.status_code == 200 else 'No'}")
    
    # Test 3: Test file upload preview
    print("\n3. Testing file upload preview...")
    with open('test_data/btc_ohlcv.csv', 'rb') as f:
        files = {'file': ('btc_ohlcv.csv', f, 'text/csv')}
        response = requests.post(f"{base_url}/data/api/preview", files=files)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Preview successful: {data.get('success', False)}")
            if 'preview' in data:
                print(f"   Columns: {data['preview'].get('columns', [])}")
                print(f"   Rows: {data['preview'].get('total_rows', 0)}")
    
    # Test 4: Test actual file upload
    print("\n4. Testing actual file upload...")
    with open('test_data/btc_ohlcv.csv', 'rb') as f:
        files = {'file': ('btc_ohlcv.csv', f, 'text/csv')}
        data = {
            'data_type': 'ohlcv',
            'symbol': 'BTC',
            'source': 'test_upload'
        }
        response = requests.post(f"{base_url}/data/api/upload", files=files, data=data)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"   Upload successful: {result.get('success', False)}")
            print(f"   Message: {result.get('message', 'No message')}")
    
    # Test 5: Check data quality metrics
    print("\n5. Testing data quality metrics...")
    response = requests.get(f"{base_url}/data/api/quality-metrics")
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        if 'summary' in data:
            print(f"   Total records: {data['summary'].get('total_records', 0)}")
            print(f"   Overall coverage: {data['summary'].get('overall_coverage', 0)}%")

def main():
    print("BTC Trading System - Data Upload Integration Test")
    print("=" * 50)
    
    # Create sample data files
    print("\nCreating sample data files...")
    create_sample_data_files()
    
    # Test upload functionality
    print("\nTesting upload endpoints...")
    try:
        test_upload_endpoints()
    except requests.exceptions.ConnectionError:
        print("\nERROR: Could not connect to Flask frontend.")
        print("Make sure the Flask frontend is running on port 8502")
    except Exception as e:
        print(f"\nERROR: {e}")
    
    print("\n" + "=" * 50)
    print("Test complete!")
    print("\nYou can now:")
    print("1. Navigate to http://localhost:8502/data/upload to test the UI")
    print("2. Upload the test files from the 'test_data' directory")
    print("3. Check data quality at http://localhost:8502/data/quality")
    print("4. View upload history at http://localhost:8502/data/history")

if __name__ == "__main__":
    main()