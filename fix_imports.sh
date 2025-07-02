#!/bin/bash

# Fix Final Import Issues
# Corrects the backtesting import and any other remaining issues

set -e

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() { echo -e "${GREEN}✅ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_error() { echo -e "${RED}❌ $1${NC}"; }
print_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }

# Fix the backtesting import issue
fix_backtesting_import() {
    print_info "Fixing backtesting import..."
    
    # The file is backtesting.py, not backtesting_system.py
    # So imports should be from services.backtesting
    
    # Fix in all Python files
    find src/backend -name "*.py" -type f | grep -v __pycache__ | while read -r file; do
        # Fix various forms of backtesting imports
        sed -i 's/from backtesting_system import/from services.backtesting import/g' "$file"
        sed -i 's/import backtesting_system/import services.backtesting/g' "$file"
        sed -i 's/from services.backtesting_system import/from services.backtesting import/g' "$file"
        sed -i 's/import services.backtesting_system/import services.backtesting/g' "$file"
    done
    
    print_status "Backtesting imports fixed"
}

# Show current file structure to verify
show_structure() {
    print_info "Current backend structure:"
    echo ""
    echo "src/backend/"
    echo "├── api/"
    find src/backend/api -name "*.py" -type f | grep -v __pycache__ | sort | while read -r file; do
        echo "│   └── $(basename "$file")"
    done
    echo "├── models/"
    find src/backend/models -name "*.py" -type f | grep -v __pycache__ | sort | while read -r file; do
        echo "│   └── $(basename "$file")"
    done
    echo "└── services/"
    find src/backend/services -name "*.py" -type f | grep -v __pycache__ | sort | while read -r file; do
        echo "    └── $(basename "$file")"
    done
    echo ""
}

# Create a comprehensive import map
create_import_map() {
    print_info "Creating import map..."
    
    cat > /tmp/import_map.txt << 'EOF'
# Import Mapping for BTC Trading System

## Models (src/backend/models/)
- paper_trading.py → from models.paper_trading import PersistentPaperTrading
- database.py → from models.database import DatabaseManager, TradingModel, etc.
- lstm.py → from models.lstm import LSTMModel, create_lstm_model

## Services (src/backend/services/)
- data_fetcher.py → from services.data_fetcher import get_fetcher, ExternalDataFetcher
- trading_system.py → from services.trading_system import TradingSystem
- backtesting.py → from services.backtesting import BacktestConfig, WalkForwardBacktester, etc.

## API (src/backend/api/)
- main.py → from api.main import app

## Common Import Mistakes to Fix:
- external_data_fetcher → services.data_fetcher
- backtesting_system → services.backtesting
- paper_trading_persistence → models.paper_trading
- database_models → models.database
- lstm_model → models.lstm
EOF

    cat /tmp/import_map.txt
    rm -f /tmp/import_map.txt
}

# Apply all fixes
apply_all_fixes() {
    print_info "Applying comprehensive import fixes..."
    
    # Fix all known import issues
    find src/backend -name "*.py" -type f | grep -v __pycache__ | while read -r file; do
        # Create backup
        cp "$file" "$file.bak" 2>/dev/null || true
        
        # Models imports
        sed -i 's/from paper_trading_persistence import/from models.paper_trading import/g' "$file"
        sed -i 's/import paper_trading_persistence/import models.paper_trading/g' "$file"
        sed -i 's/from database_models import/from models.database import/g' "$file"
        sed -i 's/import database_models/import models.database/g' "$file"
        sed -i 's/from lstm_model import/from models.lstm import/g' "$file"
        sed -i 's/import lstm_model/import models.lstm/g' "$file"
        
        # Services imports
        sed -i 's/from external_data_fetcher import/from services.data_fetcher import/g' "$file"
        sed -i 's/import external_data_fetcher/import services.data_fetcher/g' "$file"
        sed -i 's/from data_fetcher import/from services.data_fetcher import/g' "$file"
        sed -i 's/from backtesting_system import/from services.backtesting import/g' "$file"
        sed -i 's/import backtesting_system/import services.backtesting/g' "$file"
        sed -i 's/from trading_system import/from services.trading_system import/g' "$file"
        sed -i 's/import trading_system/import services.trading_system/g' "$file"
        
        # Fix any lingering issues with services.backtesting_system
        sed -i 's/from services.backtesting_system import/from services.backtesting import/g' "$file"
        sed -i 's/import services.backtesting_system/import services.backtesting/g' "$file"
    done
    
    print_status "All import fixes applied"
}

# Test imports in container
test_imports() {
    print_info "Testing imports..."
    
    cat > /tmp/test_all_imports.py << 'EOF'
#!/usr/bin/env python3
import sys
sys.path.insert(0, '/app/src')

print("Testing all imports...")
print("=" * 50)

# Test model imports
try:
    from models.paper_trading import PersistentPaperTrading
    print("✓ models.paper_trading")
except ImportError as e:
    print(f"✗ models.paper_trading: {e}")

try:
    from models.database import DatabaseManager
    print("✓ models.database")
except ImportError as e:
    print(f"✗ models.database: {e}")

try:
    from models.lstm import LSTMModel
    print("✓ models.lstm")
except ImportError as e:
    print(f"✗ models.lstm: {e}")

# Test service imports
try:
    from services.data_fetcher import get_fetcher
    print("✓ services.data_fetcher")
except ImportError as e:
    print(f"✗ services.data_fetcher: {e}")

try:
    from services.backtesting import BacktestConfig, WalkForwardBacktester
    print("✓ services.backtesting")
except ImportError as e:
    print(f"✗ services.backtesting: {e}")

try:
    from services.trading_system import TradingSystem
    print("✓ services.trading_system")
except ImportError as e:
    print(f"✗ services.trading_system: {e}")

# Test API import
try:
    from api.main import app
    print("✓ api.main")
except ImportError as e:
    print(f"✗ api.main: {e}")

print("=" * 50)
print("Import test complete")
EOF

    # Try to run in container
    if docker ps | grep -q btc-trading-backend; then
        docker cp /tmp/test_all_imports.py btc-trading-backend:/tmp/
        docker exec btc-trading-backend python /tmp/test_all_imports.py || print_warning "Some imports may still have issues"
    else
        print_warning "Backend container not running, skipping import test"
    fi
    
    rm -f /tmp/test_all_imports.py
}

# Main execution
main() {
    echo -e "${BLUE}===============================================${NC}"
    echo -e "${BLUE}   Final Import Fix Script${NC}"
    echo -e "${BLUE}===============================================${NC}"
    echo ""
    
    # Show current structure
    show_structure
    
    # Create import map for reference
    create_import_map
    
    # Apply fixes
    fix_backtesting_import
    apply_all_fixes
    
    # Test imports
    test_imports
    
    echo ""
    echo -e "${GREEN}✅ Import fixes complete!${NC}"
    echo ""
    echo -e "${BLUE}Next steps:${NC}"
    echo "1. Restart the backend: docker compose restart backend"
    echo "2. Check logs: docker compose logs -f backend"
    echo ""
    echo -e "${YELLOW}If you still see import errors:${NC}"
    echo "- Check the exact filename in src/backend/services/"
    echo "- Ensure all __init__.py files exist"
    echo "- Verify the module name matches the import statement"
    echo ""
}

# Run main
main
