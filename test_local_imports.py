#!/usr/bin/env python3
"""Test imports for local development"""

import sys
import os

# Add src paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src/backend'))

print("Testing local imports...")
print("=" * 50)

success = True

# Test each module
tests = [
    ("models.database", "DatabaseManager"),
    ("models.lstm", "TradingSignalGenerator"),
    ("models.lstm", "LSTMTradingModel"),
    ("models.paper_trading", "PersistentPaperTrading"),
    ("services.data_fetcher", "get_fetcher"),
    ("services.backtesting", "BacktestConfig"),
    ("services.integration", "AdvancedTradingSignalGenerator"),
    ("services.notifications", "DiscordNotifier"),
]

for module, item in tests:
    try:
        exec(f"from {module} import {item}")
        print(f"✅ {module}.{item}")
    except ImportError as e:
        print(f"❌ {module}.{item}: {e}")
        success = False

print("=" * 50)

if success:
    print("✅ All local imports working!")
    
    # Now test if main.py would work with proper imports
    print("\nTesting main.py imports (with compatibility)...")
    try:
        # Set up the module aliases that main.py expects
        import models.paper_trading
        import models.database
        import models.lstm
        import services.data_fetcher
        import services.integration
        import services.backtesting
        import services.notifications
        
        sys.modules['paper_trading_persistence'] = models.paper_trading
        sys.modules['database_models'] = models.database
        sys.modules['lstm_model'] = models.lstm
        sys.modules['external_data_fetcher'] = services.data_fetcher
        sys.modules['integration'] = services.integration
        sys.modules['backtesting_system'] = services.backtesting
        sys.modules['discord_notifications'] = services.notifications
        
        print("✅ Module aliases set up successfully")
        print("\nYou can now import main.py or run it directly!")
        
    except Exception as e:
        print(f"⚠️  Could not set up module aliases: {e}")
else:
    print("❌ Some imports failed. Please check the errors above.")
    exit(1)
