#!/bin/bash
# Test data upload functionality inside Docker container

echo "Testing Data Upload Functionality in Docker"
echo "=========================================="

# Create test Python script
cat > /tmp/test_data_upload.py << 'EOF'
#!/usr/bin/env python3
"""Test data upload functionality end-to-end"""
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import requests
import json

def create_sample_data_files():
    """Create sample data files for testing"""
    
    # Create test directory
    os.makedirs('/tmp/test_data', exist_ok=True)
    
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
    btc_ohlcv.to_csv('/tmp/test_data/btc_ohlcv.csv', index=False)
    print("Created: /tmp/test_data/btc_ohlcv.csv")
    
    # 2. Create OHLCV data for GLD (Gold ETF)
    gld_ohlcv = pd.DataFrame({
        'date': dates,
        'open': np.random.uniform(180, 190, 30),
        'high': np.random.uniform(185, 195, 30),
        'low': np.random.uniform(175, 185, 30),
        'close': np.random.uniform(180, 190, 30),
        'volume': np.random.uniform(1000000, 5000000, 30)
    })
    gld_ohlcv.to_csv('/tmp/test_data/gld_ohlcv.csv', index=False)
    print("Created: /tmp/test_data/gld_ohlcv.csv")
    
    # 3. Create VIX data
    vix_data = pd.DataFrame({
        'date': dates,
        'open': np.random.uniform(12, 25, 30),
        'high': np.random.uniform(15, 30, 30),
        'low': np.random.uniform(10, 20, 30),
        'close': np.random.uniform(12, 25, 30),
        'volume': 0  # VIX doesn't have volume
    })
    vix_data.to_csv('/tmp/test_data/vix_ohlcv.csv', index=False)
    print("Created: /tmp/test_data/vix_ohlcv.csv")
    
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
    onchain_data.to_csv('/tmp/test_data/btc_onchain.csv', index=False)
    print("Created: /tmp/test_data/btc_onchain.csv")
    
    # 5. Create Sentiment data
    sentiment_data = pd.DataFrame({
        'timestamp': pd.date_range(end=datetime.now(), periods=100, freq='H'),
        'fear_greed_index': np.random.randint(20, 80, 100),
        'reddit_sentiment': np.random.uniform(0.3, 0.8, 100),
        'twitter_sentiment': np.random.uniform(0.2, 0.9, 100),
        'news_sentiment': np.random.uniform(0.4, 0.7, 100),
        'google_trends': np.random.randint(40, 100, 100)
    })
    sentiment_data.to_csv('/tmp/test_data/btc_sentiment.csv', index=False)
    print("Created: /tmp/test_data/btc_sentiment.csv")
    
    # 6. Create Macro data (mixed indicators)
    macro_data = pd.DataFrame({
        'date': dates.repeat(4),  # 4 indicators per day
        'symbol': ['DXY', 'TNX', 'GLD', 'SPY'] * 30,
        'value': np.random.uniform(50, 150, 120),
        'indicator_type': ['currency_index', 'bond_yield', 'commodity', 'equity_index'] * 30
    })
    macro_data.to_csv('/tmp/test_data/macro_indicators.csv', index=False)
    print("Created: /tmp/test_data/macro_indicators.csv")
    
    print("\nAll test files created successfully!")
    return True

def test_backend_upload():
    """Test backend upload endpoints directly"""
    print("\n\nTesting Backend Upload Endpoints")
    print("================================")
    
    base_url = "http://backend:8000"
    
    # Test 1: Upload OHLCV data
    print("\n1. Testing OHLCV data upload to backend...")
    with open('/tmp/test_data/btc_ohlcv.csv', 'rb') as f:
        files = {'file': ('btc_ohlcv.csv', f, 'text/csv')}
        data = {
            'data_type': 'price',
            'symbol': 'BTC',
            'source': 'test_upload'
        }
        try:
            response = requests.post(f"{base_url}/data/upload", files=files, data=data)
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"   Success: {result.get('status') == 'success'}")
                print(f"   Rows processed: {result.get('rows_processed', 0)}")
            else:
                print(f"   Error: {response.text}")
        except Exception as e:
            print(f"   Error: {e}")
    
    # Test 2: Get data quality metrics
    print("\n2. Testing data quality endpoint...")
    try:
        response = requests.get(f"{base_url}/analytics/data-quality")
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Got quality metrics: Yes")
            if 'summary' in data:
                print(f"   Total records: {data['summary'].get('total_records', 0)}")
    except Exception as e:
        print(f"   Error: {e}")

def test_frontend_pages():
    """Test frontend pages and API endpoints"""
    print("\n\nTesting Frontend Pages and APIs")
    print("===============================")
    
    base_url = "http://frontend:8502"
    
    # Test 1: Check if data upload page loads
    print("\n1. Testing data upload page...")
    try:
        response = requests.get(f"{base_url}/data/upload", timeout=5)
        print(f"   Status: {response.status_code}")
        print(f"   Page loads: {'Yes' if response.status_code == 200 else 'No'}")
        if response.status_code == 200:
            print(f"   Contains 'Data Upload': {'Data Upload' in response.text}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 2: Check if quality page loads
    print("\n2. Testing data quality page...")
    try:
        response = requests.get(f"{base_url}/data/quality", timeout=5)
        print(f"   Status: {response.status_code}")
        print(f"   Page loads: {'Yes' if response.status_code == 200 else 'No'}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 3: Test file upload preview
    print("\n3. Testing file upload preview...")
    with open('/tmp/test_data/btc_ohlcv.csv', 'rb') as f:
        files = {'file': ('btc_ohlcv.csv', f, 'text/csv')}
        try:
            response = requests.post(f"{base_url}/data/api/preview", files=files, timeout=10)
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"   Preview successful: {data.get('success', False)}")
                if 'preview' in data:
                    print(f"   Columns: {data['preview'].get('columns', [])}")
                    print(f"   Rows: {data['preview'].get('total_rows', 0)}")
        except Exception as e:
            print(f"   Error: {e}")
    
    # Test 4: Test quality metrics API
    print("\n4. Testing quality metrics API...")
    try:
        response = requests.get(f"{base_url}/data/api/quality-metrics", timeout=5)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            if 'summary' in data:
                print(f"   API works: Yes")
                print(f"   Has data: {data['summary'].get('total_records', 0) > 0}")
    except Exception as e:
        print(f"   Error: {e}")

def main():
    print("BTC Trading System - Data Upload Integration Test")
    print("=" * 50)
    
    # Create sample data files
    print("\nCreating sample data files...")
    create_sample_data_files()
    
    # Test backend
    test_backend_upload()
    
    # Test frontend
    test_frontend_pages()
    
    print("\n" + "=" * 50)
    print("Test Summary:")
    print("- Sample data files created in /tmp/test_data/")
    print("- Backend upload endpoints tested")
    print("- Frontend pages and APIs tested")
    print("\nFor manual testing, visit:")
    print("- http://localhost:8502/data/upload")
    print("- http://localhost:8502/data/quality")
    print("- http://localhost:8502/data/history")

if __name__ == "__main__":
    main()
EOF

# Run test in backend container (has all dependencies)
echo "Running test script in backend container..."
docker exec btc-trading-backend python3 /tmp/test_data_upload.py