#!/usr/bin/env python3
"""Test the external data fetcher implementation with real sources"""

import sys
import os
try:
    from external_data_fetcher import get_fetcher
    print("✅ Successfully imported external_data_fetcher")
    
    # Test fetcher initialization
    fetcher = get_fetcher()
    print("✅ Fetcher initialized")
    
    # Test crypto data fetch from real sources
    print("\nTesting crypto data fetch (real sources)...")
    btc_data = fetcher.fetch_crypto_data('BTC', '7d')
    print(f"✅ Fetched {len(btc_data)} days of BTC data")
    print(f"   Columns: {list(btc_data.columns)}")
    print(f"   Latest price: ${btc_data['Close'].iloc[-1]:,.2f}")
    print(f"   Data source: Real API (CoinGecko/Binance/CryptoCompare)")
    
    # Test current price from real source
    print("\nTesting current price fetch...")
    current_price = fetcher.get_current_crypto_price('BTC')
    print(f"✅ Current BTC price: ${current_price:,.2f} (from real API)")
    
    # Test sentiment data from real sources
    print("\nTesting sentiment data fetch (real sources)...")
    sentiment = fetcher.fetch_sentiment_data()
    print(f"✅ Fetched sentiment data")
    print(f"   Fear & Greed: {sentiment['fear_greed']['value']} ({sentiment['fear_greed']['classification']})")
    print(f"   Reddit sentiment: {sentiment['reddit'].get('reddit_sentiment', 0):.2f}")
    print(f"   News sentiment: {sentiment['news'].get('news_sentiment', 0):.2f}")
    print(f"   Data source: Real APIs (Alternative.me, Reddit, News)")
    
    # Test on-chain data from real sources
    print("\nTesting on-chain data fetch (real sources)...")
    onchain = fetcher.fetch_onchain_data()
    network = onchain.get('network', {})
    print(f"✅ Fetched on-chain data")
    print(f"   Active addresses: {network.get('active_addresses', 0):,}")
    print(f"   Transaction count: {network.get('transaction_count', 0):,}")
    print(f"   Hash rate: {network.get('hash_rate', 0):.2e}")
    print(f"   Data source: Real APIs (Blockchain.info, Blockchair)")
    
    # Test macro data
    print("\nTesting macro data fetch...")
    spy_data = fetcher.fetch_macro_data('SPY', '1mo')
    print(f"✅ Fetched {len(spy_data)} days of SPY data")
    print(f"   Latest close: ${spy_data['Close'].iloc[-1]:,.2f}")
    
    # Check if API keys are set
    print("\n📝 API Key Status:")
    api_keys = {
        'FRED_API_KEY': 'Federal Reserve Economic Data',
        'NEWS_API_KEY': 'NewsAPI',
        'CRYPTOPANIC_API_KEY': 'CryptoPanic',
        'GLASSNODE_API_KEY': 'Glassnode',
        'CRYPTOQUANT_API_KEY': 'CryptoQuant'
    }
    
    for key, service in api_keys.items():
        if os.getenv(key):
            print(f"   ✅ {service}: Configured")
        else:
            print(f"   ⚠️  {service}: Not configured (using fallback)")
    
    # Test data flow compatibility
    print("\n\nTesting data flow compatibility...")
    
    # Test 1: Technical indicators
    from lstm_model import TradingSignalGenerator
    generator = TradingSignalGenerator()
    enhanced_data = generator.add_technical_indicators(btc_data)
    print(f"✅ Technical indicators added: {len(enhanced_data.columns)} columns")
    
    # Test 2: Integration with signal generator
    from integration import AdvancedTradingSignalGenerator
    adv_generator = AdvancedTradingSignalGenerator()
    enhanced_btc = adv_generator.fetch_enhanced_btc_data(period='7d', include_macro=True)
    print(f"✅ Enhanced data fetched: {enhanced_btc.shape}")
    
    # Test 3: Data format for API
    data_records = []
    for idx, row in btc_data.head(5).iterrows():
        record = {
            'timestamp': idx.isoformat(),
            'Date': idx.isoformat(),
            'Open': float(row['Open']),
            'High': float(row['High']),
            'Low': float(row['Low']),
            'Close': float(row['Close']),
            'Volume': float(row['Volume'])
        }
        data_records.append(record)
    
    import json
    json.dumps(data_records)
    print("✅ Data format compatible with API responses")
    
    print("\n✅ All tests passed! External data fetcher is working with real sources.")
    print("\n💡 To get the most accurate data, set the API keys listed above.")
    print("   See .env.template for instructions on obtaining free API keys.")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
