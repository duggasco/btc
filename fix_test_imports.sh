#!/bin/bash

# Quick setup script for import compatibility
set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
print_success() { echo -e "${GREEN}✅ $1${NC}"; }

# Create a compatibility layer for imports
create_import_compatibility() {
    print_info "Creating import compatibility layer..."
    
    # Create src/backend/import_compat.py
    cat > src/backend/import_compat.py << 'EOF'
"""
Import compatibility layer for local vs Docker environments
"""
import os
import sys

# Detect if we're in Docker (files are copied with different names)
IS_DOCKER = os.path.exists('backend_api.py') and not os.path.exists('src/backend/api/main.py')

if IS_DOCKER:
    # Docker environment - files are in the same directory with different names
    # No changes needed, imports work as-is
    pass
else:
    # Local environment - add src paths
    import os
    import sys
    
    # Get the project root (parent of src)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    
    # Add paths
    sys.path.insert(0, os.path.join(project_root, 'src'))
    sys.path.insert(0, os.path.join(project_root, 'src', 'backend'))
    
    # Create import aliases for compatibility
    try:
        import models.paper_trading as paper_trading_persistence
        import services.data_fetcher as external_data_fetcher
        import models.database as database_models
        import models.lstm as lstm_model
        import services.integration as integration
        import services.backtesting as backtesting_system
        import services.notifications as discord_notifications
        
        # Add to sys.modules for import compatibility
        sys.modules['paper_trading_persistence'] = paper_trading_persistence
        sys.modules['external_data_fetcher'] = external_data_fetcher
        sys.modules['database_models'] = database_models
        sys.modules['lstm_model'] = lstm_model
        sys.modules['integration'] = integration
        sys.modules['backtesting_system'] = backtesting_system
        sys.modules['discord_notifications'] = discord_notifications
        
    except ImportError as e:
        print(f"Warning: Could not set up import compatibility: {e}")
EOF
    
    # Update main.py to use the compatibility layer
    if [ -f "src/backend/api/main.py" ]; then
        # Check if import_compat is already imported
        if ! grep -q "import import_compat" src/backend/api/main.py; then
            # Add import at the beginning of the file (after the initial imports)
            python3 - << 'EOF'
import re

with open('src/backend/api/main.py', 'r') as f:
    content = f.read()

# Add import_compat after the first import block
lines = content.split('\n')
import_added = False
new_lines = []

for i, line in enumerate(lines):
    new_lines.append(line)
    # Add after the first group of imports
    if not import_added and line.strip() and not line.startswith('import') and not line.startswith('from') and i > 10:
        new_lines.insert(-1, '\n# Import compatibility layer')
        new_lines.insert(-1, 'try:')
        new_lines.insert(-1, '    import import_compat')
        new_lines.insert(-1, 'except ImportError:')
        new_lines.insert(-1, '    pass  # Not needed in Docker environment')
        new_lines.insert(-1, '')
        import_added = True

# Write back
with open('src/backend/api/main.py', 'w') as f:
    f.write('\n'.join(new_lines))

print("Added import compatibility layer to main.py")
EOF
        fi
    fi
    
    print_success "Import compatibility layer created"
}

# Quick test function
quick_test() {
    print_info "Running quick import test..."
    
    python3 - << 'EOF'
import sys
import os

# Try local environment first
try:
    sys.path.insert(0, 'src')
    sys.path.insert(0, 'src/backend')
    
    # Test a few key imports
    from models.database import DatabaseManager
    from models.lstm import TradingSignalGenerator
    from services.data_fetcher import get_fetcher
    
    print("✅ Local environment imports working!")
except ImportError:
    # Try Docker environment
    try:
        from database_models import DatabaseManager
        from lstm_model import TradingSignalGenerator
        from external_data_fetcher import get_fetcher
        
        print("✅ Docker environment imports working!")
    except ImportError as e:
        print(f"❌ Import test failed: {e}")
        sys.exit(1)
EOF
}

# Main execution
echo "🔧 Setting up import compatibility"
echo "=================================="

# Create compatibility layer
create_import_compatibility

# Run quick test
quick_test

echo
print_success "Setup complete!"
echo
print_info "You can now run the system in either environment:"
echo "  - Local: python src/backend/api/main.py"
echo "  - Docker: ./init_deploy.sh"
echo
print_info "The import compatibility layer will handle the differences automatically."
