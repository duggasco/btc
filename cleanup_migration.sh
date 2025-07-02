#!/bin/bash

# BTC Trading System - Post-Migration Cleanup Script
# This script removes old files after successful migration while preserving backups

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

print_status() { echo -e "${GREEN}✅ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_error() { echo -e "${RED}❌ $1${NC}"; }
print_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }

# List of old files to be removed (from root directory)
OLD_PYTHON_FILES=(
    "backend_api.py"
    "database_models.py"
    "lstm_model.py"
    "paper_trading_persistence.py"
    "external_data_fetcher.py"
    "backtesting_system.py"
    "integration.py"
    "discord_notifications.py"
    "streamlit_app.py"
    "test_external_fetcher.py"
    "test_system.py"
)

OLD_DOCKER_FILES=(
    "Dockerfile.backend"
    "Dockerfile.frontend"
    "docker-compose.yml"
)

OLD_SCRIPT_FILES=(
    "init_and_deploy.sh"
    "create_gitkeeps.sh"
)

OLD_DOC_FILES=(
    "trading_signals_api.md"
)

OLD_REQUIREMENT_FILES=(
    "requirements-backend.txt"
    "requirements-frontend.txt"
)

# Check if we're in the right directory
if [ ! -d "src" ] || [ ! -d "docker" ] || [ ! -d "scripts" ]; then
    print_error "This script should be run AFTER the migration script"
    print_error "Required directories (src/, docker/, scripts/) not found"
    exit 1
fi

# Check if backup exists
BACKUP_COUNT=$(find . -maxdepth 1 -type d -name "backup_*" | wc -l)
if [ "$BACKUP_COUNT" -eq 0 ]; then
    print_error "No backup directory found. This is unsafe!"
    print_error "Please ensure you have a backup before cleaning up"
    exit 1
fi

echo -e "${BOLD}BTC Trading System - Post-Migration Cleanup${NC}"
echo "=================================================="
echo ""

# Show what will be removed
print_info "This script will remove the following old files:"
echo ""

echo "Python Files:"
for file in "${OLD_PYTHON_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  - $file"
    fi
done

echo ""
echo "Docker Files:"
for file in "${OLD_DOCKER_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  - $file"
    fi
done

echo ""
echo "Script Files:"
for file in "${OLD_SCRIPT_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  - $file"
    fi
done

echo ""
echo "Documentation Files:"
for file in "${OLD_DOC_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  - $file"
    fi
done

echo ""
echo "Requirements Files:"
for file in "${OLD_REQUIREMENT_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  - $file"
    fi
done

echo ""
print_warning "The following will be PRESERVED:"
echo "  ✓ All backup directories (backup_*)"
echo "  ✓ storage/ directory and all its contents"
echo "  ✓ .env file (both in root and config/)"
echo "  ✓ .gitignore"
echo "  ✓ README.md"
echo "  ✓ The new src/ structure"
echo "  ✓ The new docker/ directory"
echo "  ✓ The new scripts/ directory"
echo "  ✓ The new docs/ directory"
echo "  ✓ The new config/ directory"
echo "  ✓ run.sh helper script"
echo "  ✓ MIGRATION_INFO.md"

# Safety check - verify new structure exists
echo ""
print_info "Performing safety checks..."

SAFETY_CHECKS=(
    "src/backend/api/main.py"
    "src/backend/models/database.py"
    "src/backend/models/lstm.py"
    "src/backend/services/data_fetcher.py"
    "src/frontend/app.py"
    "docker/docker-compose.yml"
    "docker/backend.Dockerfile"
    "scripts/init_deploy.sh"
)

ALL_SAFE=true
for check in "${SAFETY_CHECKS[@]}"; do
    if [ ! -f "$check" ]; then
        print_error "Missing migrated file: $check"
        ALL_SAFE=false
    fi
done

if [ "$ALL_SAFE" = false ]; then
    print_error "Safety check failed! Some migrated files are missing."
    print_error "Please run the migration script again or check for errors."
    exit 1
fi

print_status "Safety checks passed - all migrated files exist"

# Confirmation prompt
echo ""
echo -e "${BOLD}${YELLOW}⚠️  WARNING: This action cannot be undone!${NC}"
echo -e "${YELLOW}(Your backup will be preserved in case you need to restore)${NC}"
echo ""
read -p "Are you sure you want to remove the old files? (yes/NO): " confirm

if [ "$confirm" != "yes" ]; then
    print_info "Cleanup cancelled. No files were removed."
    exit 0
fi

# Second confirmation for safety
echo ""
read -p "Type 'CLEANUP' to confirm removal of old files: " second_confirm

if [ "$second_confirm" != "CLEANUP" ]; then
    print_info "Cleanup cancelled. No files were removed."
    exit 0
fi

# Perform cleanup
echo ""
print_info "Starting cleanup..."

# Function to safely remove a file
safe_remove() {
    local file=$1
    if [ -f "$file" ]; then
        rm -f "$file"
        echo "  Removed: $file"
    fi
}

# Remove Python files
print_info "Removing old Python files..."
for file in "${OLD_PYTHON_FILES[@]}"; do
    safe_remove "$file"
done

# Remove Docker files
print_info "Removing old Docker files..."
for file in "${OLD_DOCKER_FILES[@]}"; do
    safe_remove "$file"
done

# Remove script files
print_info "Removing old script files..."
for file in "${OLD_SCRIPT_FILES[@]}"; do
    safe_remove "$file"
done

# Remove documentation files
print_info "Removing old documentation files..."
for file in "${OLD_DOC_FILES[@]}"; do
    safe_remove "$file"
done

# Remove requirements files
print_info "Removing old requirements files..."
for file in "${OLD_REQUIREMENT_FILES[@]}"; do
    safe_remove "$file"
done

# Clean up any .pyc files and __pycache__ in root
print_info "Cleaning Python cache files from root..."
find . -maxdepth 1 -name "*.pyc" -delete 2>/dev/null || true
find . -maxdepth 1 -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Remove any .bak files from sed operations
print_info "Cleaning backup files from sed operations..."
find . -name "*.bak" -delete 2>/dev/null || true

print_status "Cleanup completed successfully!"

# Show final directory structure
echo ""
print_info "Current directory structure:"
echo ""
echo "."
echo "├── src/"
echo "│   ├── backend/"
echo "│   │   ├── api/"
echo "│   │   ├── models/"
echo "│   │   └── services/"
echo "│   └── frontend/"
echo "├── docker/"
echo "├── scripts/"
echo "├── tests/"
echo "├── docs/"
echo "├── config/"
echo "├── storage/ (unchanged)"
echo "├── backup_*/ (preserved)"
echo "├── .env"
echo "├── .gitignore"
echo "├── README.md"
echo "├── run.sh"
echo "└── MIGRATION_INFO.md"

# Count removed files
REMOVED_COUNT=0
for file in "${OLD_PYTHON_FILES[@]}" "${OLD_DOCKER_FILES[@]}" "${OLD_SCRIPT_FILES[@]}" "${OLD_DOC_FILES[@]}" "${OLD_REQUIREMENT_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        ((REMOVED_COUNT++))
    fi
done

echo ""
print_status "Summary:"
echo "  - Removed $REMOVED_COUNT old files"
echo "  - Preserved all backups"
echo "  - Kept new structure intact"

# Show backup information
echo ""
print_info "Backup Information:"
for backup in backup_*/; do
    if [ -d "$backup" ]; then
        echo "  - $backup ($(du -sh "$backup" | cut -f1))"
    fi
done

echo ""
print_info "Quick commands:"
echo "  Start system:  cd docker && docker-compose up -d"
echo "  Run tests:     ./run.sh test"
echo "  View backup:   ls -la backup_*/"
echo ""

# Create a cleanup log
LOG_FILE="cleanup_$(date +%Y%m%d_%H%M%S).log"
cat > "$LOG_FILE" << EOF
BTC Trading System - Cleanup Log
Date: $(date)

Files Removed:
$(for file in "${OLD_PYTHON_FILES[@]}" "${OLD_DOCKER_FILES[@]}" "${OLD_SCRIPT_FILES[@]}" "${OLD_DOC_FILES[@]}" "${OLD_REQUIREMENT_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "  - $file"
    fi
done)

Total files removed: $REMOVED_COUNT

Backup directories preserved:
$(ls -d backup_*/ 2>/dev/null)

Cleanup completed successfully.
EOF

print_info "Cleanup log saved to: $LOG_FILE"

# Final safety reminder
echo ""
print_warning "Remember: Your old files are safely stored in the backup directory"
print_info "If you need to restore, simply copy files from backup_*/ to their original locations"
