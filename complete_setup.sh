#!/bin/bash

# Complete setup script for BTC Trading System
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

echo "🚀 BTC Trading System - Complete Setup"
echo "======================================"

# Check if we're in the right directory
if [ ! -f "src/backend/api/main.py" ]; then
    print_error "Please run this script from the project root directory"
    print_info "Current directory: $(pwd)"
    exit 1
fi

# Step 1: Check Python version
print_info "Checking Python version..."
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
print_status "Python version: $PYTHON_VERSION"

# Step 2: Create virtual environment (recommended)
print_info "Setting up Python environment..."

if [ ! -d "venv" ]; then
    print_info "Creating virtual environment..."
    python3 -m venv venv
    print_status "Virtual environment created"
else
    print_info "Virtual environment already exists"
fi

# Activate virtual environment
print_info "Activating virtual environment..."
source venv/bin/activate || {
    print_warning "Could not activate virtual environment automatically"
    print_info "Please run: source venv/bin/activate"
}

# Step 3: Upgrade pip
print_info "Upgrading pip..."
pip install --upgrade pip

# Step 4: Install backend dependencies
print_info "Installing backend dependencies..."
if [ -f "src/backend/requirements.txt" ]; then
    pip install -r src/backend/requirements.txt
    print_status "Backend dependencies installed"
else
    print_error "Backend requirements.txt not found!"
    print_info "Installing common dependencies manually..."
    
    # Install essential packages
    pip install \
        fastapi==0.104.1 \
        uvicorn[standard]==0.24.0 \
        pandas==2.1.3 \
        numpy==1.24.3 \
        torch==2.1.1 \
        scikit-learn==1.3.2 \
        requests==2.31.0 \
        yfinance==0.2.33 \
        optuna==3.5.0 \
        ta==0.10.2 \
        scipy==1.11.4 \
        websocket-client==1.6.4
fi

# Step 5: Install frontend dependencies (optional)
if [ -f "src/frontend/requirements.txt" ]; then
    print_info "Installing frontend dependencies..."
    pip install -r src/frontend/requirements.txt
    print_status "Frontend dependencies installed"
fi

# Step 6: Create all __init__.py files
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

# Step 7: Create directories for data storage
print_info "Creating data directories..."
mkdir -p data models logs config
print_status "Data directories created"

# Step 8: Create test script
print_info "Creating test script..."

cat > test_system.py << 'EOF'
#!/usr/bin/env python3
"""Complete system test for BTC Trading System"""

import sys
import os

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src/backend'))

def test_dependencies():
    """Test that all required packages are installed"""
    print("\n1. Testing Python dependencies...")
    print("=" * 50)
    
    dependencies = [
        ("pandas", "2.1.3"),
        ("numpy", "1.24.3"),
        ("torch", "2.1.1"),
        ("sklearn", "scikit-learn"),
        ("fastapi", None),
        ("uvicorn", None),
        ("requests", None),
        ("yfinance", None),
        ("optuna", None),
        ("ta", None),
        ("scipy", None)
    ]
    
    all_installed = True
    for package, display_name in dependencies:
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {display_name or package} ({version})")
        except ImportError:
            print(f"❌ {display_name or package} - NOT INSTALLED")
            all_installed = False
    
    return all_installed

def test_local_imports():
    """Test that all local modules can be imported"""
    print("\n2. Testing local imports...")
    print("=" * 50)
    
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
    
    all_passed = True
    for module, item in tests:
        try:
            exec(f"from {module} import {item}")
            print(f"✅ {module}.{item}")
        except ImportError as e:
            print(f"❌ {module}.{item}: {e}")
            all_passed = False
    
    return all_passed

def test_api_import():
    """Test that the main API can be imported"""
    print("\n3. Testing API import...")
    print("=" * 50)
    
    try:
        # Set up module aliases
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
        
        # Now try to import the API
        from api.main import app
        print("✅ API imported successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to import API: {e}")
        return False

def main():
    print("🔍 Running complete system test...")
    
    # Test dependencies
    deps_ok = test_dependencies()
    
    # Test local imports
    imports_ok = test_local_imports()
    
    # Test API
    api_ok = test_api_import()
    
    print("\n" + "=" * 50)
    print("Test Summary:")
    print(f"  Dependencies: {'✅ PASS' if deps_ok else '❌ FAIL'}")
    print(f"  Local imports: {'✅ PASS' if imports_ok else '❌ FAIL'}")
    print(f"  API import: {'✅ PASS' if api_ok else '❌ FAIL'}")
    
    if deps_ok and imports_ok and api_ok:
        print("\n✅ All tests passed! System is ready to run.")
        print("\nTo run the API locally:")
        print("  python3 run_local.py")
        print("\nTo run with Docker:")
        print("  ./init_deploy.sh")
        return 0
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        if not deps_ok:
            print("\n💡 To fix dependency issues:")
            print("  1. Make sure you're in the virtual environment: source venv/bin/activate")
            print("  2. Install dependencies: pip install -r src/backend/requirements.txt")
        return 1

if __name__ == "__main__":
    sys.exit(main())
EOF

chmod +x test_system.py

# Step 9: Create run_local.py
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

# Set environment variables
os.environ['DATABASE_PATH'] = os.path.join(os.path.dirname(__file__), 'data', 'trading_system.db')
os.environ['MODEL_PATH'] = os.path.join(os.path.dirname(__file__), 'models')
os.environ['LOG_PATH'] = os.path.join(os.path.dirname(__file__), 'logs')
os.environ['CONFIG_PATH'] = os.path.join(os.path.dirname(__file__), 'config')

# Now import and run the main app
from api.main import app
import uvicorn

if __name__ == "__main__":
    print("Starting BTC Trading System API (Local Development)...")
    print("API will be available at: http://localhost:8000")
    print("API docs: http://localhost:8000/docs")
    print("\nPress CTRL+C to stop")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
EOF

chmod +x run_local.py

# Step 10: Run the test
print_info "Running system test..."
python3 test_system.py

# Final instructions
echo
print_info "Setup complete!"
echo
if [ -d "venv" ]; then
    print_warning "Remember to activate the virtual environment:"
    echo "  source venv/bin/activate"
    echo
fi
print_info "To run the API locally:"
echo "  python3 run_local.py"
echo
print_info "To run the frontend:"
echo "  streamlit run src/frontend/app.py"
echo
print_info "To run with Docker:"
echo "  ./init_deploy.sh"
