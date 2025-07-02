#!/bin/bash

# Simple script to fix imports for local development
set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() { echo -e "${GREEN}✅ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_error() { echo -e "${RED}❌ $1${NC}"; }
print_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }

echo "🔧 Fixing imports for local development"
echo "======================================"

# Check if we're in the right directory
if [ ! -f "src/backend/api/main.py" ]; then
    print_error "Please run this script from the project root directory"
    print_info "Current directory: $(pwd)"
    exit 1
fi

# Step 1: Create all __init__.py files
print_info "Creating __init__.py files..."
directories=(
    "src"
    "src/backend" 
    "src/backend/models"
    "src/backend/services"
    "src/backend/api"
    "src/backend/api/routes"
    "src/frontend"
    "src/frontend/components"
    "src/frontend/pages"
)

for dir in "${directories[@]}"; do
    mkdir -p "$dir"
    touch "$dir/__init__.py"
done
print_status "All __init__.py files created"

# Step 2: Create a simple test to verify local imports work
print_info "Creating local import test..."

cat > test_local_imports.py << 'EOF'
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
EOF

chmod +x test_local_imports.py

# Step 3: Create a run script for local development
print_info "Creating local run script..."

cat > run_local.py << 'EOF'
#!/usr/bin/env python3
"""Run the API locally with proper imports"""

import sys
import os

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src/backend'))

# Set up module aliases for main.py
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

# Now import and run the main app
from api.main import app
import uvicorn

if __name__ == "__main__":
    print("Starting BTC Trading System API (Local Development)...")
    print("API will be available at: http://localhost:8000")
    print("API docs: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
EOF

chmod +x run_local.py

# Step 4: Run the test
print_info "Running import test..."
echo
python3 test_local_imports.py

echo
print_status "Setup complete!"
echo
print_info "To run the API locally, use:"
echo "  python3 run_local.py"
echo
print_info "To run with Docker, use:"
echo "  ./init_deploy.sh"
echo
print_warning "Note: The imports in main.py are designed for Docker deployment."
print_warning "The run_local.py script handles the import differences automatically."
