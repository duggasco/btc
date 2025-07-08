#!/bin/bash

# Script to commit and push navigation fix changes to git
# Handles staging, committing with detailed message, and pushing

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# Function to print colored output
print_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
print_success() { echo -e "${GREEN}✅ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_error() { echo -e "${RED}❌ $1${NC}"; }

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    print_error "Not in a git repository!"
    exit 1
fi

# Check for uncommitted changes
print_info "Checking git status..."
git status --porcelain

# Check if there are changes to commit
if [ -z "$(git status --porcelain)" ]; then
    print_warning "No changes to commit"
    exit 0
fi

# Show current branch
CURRENT_BRANCH=$(git branch --show-current)
print_info "Current branch: $CURRENT_BRANCH"

# Ask for confirmation on branch
read -p "Continue committing to branch '$CURRENT_BRANCH'? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_info "Aborted by user"
    exit 0
fi

# Check if app.py has changes
if git status --porcelain | grep -q "src/frontend/app.py"; then
    print_success "Found changes in app.py"
else
    print_warning "No changes found in app.py"
    print_info "Looking for other navigation-related changes..."
fi

# Stage the navigation fix changes
print_info "Staging navigation fix files..."

# Stage the main app file if modified
if [ -f "src/frontend/app.py" ] && git status --porcelain | grep -q "src/frontend/app.py"; then
    git add src/frontend/app.py
    print_success "Staged: src/frontend/app.py"
fi

# Stage the fix scripts if they exist
if [ -f "fix_navigation.sh" ]; then
    read -p "Add fix_navigation.sh to repository? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git add fix_navigation.sh
        print_success "Staged: fix_navigation.sh"
    fi
fi

# Stage the Discord fix if it exists and was modified
if [ -f "init_deploy.sh" ] && git status --porcelain | grep -q "init_deploy.sh"; then
    git add init_deploy.sh
    print_success "Staged: init_deploy.sh (Discord warning fix)"
fi

# Show what will be committed
print_info "Files to be committed:"
git diff --cached --name-status

# Ask for confirmation
read -p "Proceed with commit? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    git reset
    print_info "Changes unstaged. Commit aborted."
    exit 0
fi

# Create detailed commit message
COMMIT_MSG="fix(frontend): Remove redundant navigation pages

- Consolidated Trading, Portfolio, Paper Trading, and Limits into Trading Hub
- Merged Signals into Advanced Signals page
- Simplified navigation from 11 to 6 main pages
- Improved user experience with clearer navigation structure

Consolidated pages:
- Trading → Trading Hub
- Portfolio → Trading Hub
- Paper Trading → Trading Hub
- Limits → Trading Hub
- Signals → Advanced Signals

Final navigation structure:
- Dashboard
- Trading Hub (all trading functionality)
- Advanced Signals (all signal analysis)
- Analytics
- Backtesting
- Configuration"

# Add Discord fix note if init_deploy.sh was staged
if git diff --cached --name-only | grep -q "init_deploy.sh"; then
    COMMIT_MSG="$COMMIT_MSG

Also fixed:
- Added missing discord_warning() function in init_deploy.sh"
fi

# Commit the changes
print_info "Committing changes..."
git commit -m "$COMMIT_MSG"
print_success "Changes committed successfully!"

# Show the commit
print_info "Commit details:"
git log --oneline -1

# Ask about pushing
print_warning "Ready to push to remote repository"
read -p "Push to origin/$CURRENT_BRANCH? (y/n): " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_info "Pushing to origin/$CURRENT_BRANCH..."
    
    # Check if upstream is set
    if ! git config "branch.$CURRENT_BRANCH.remote" > /dev/null 2>&1; then
        print_warning "No upstream branch set"
        read -p "Set upstream to origin/$CURRENT_BRANCH? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            git push --set-upstream origin "$CURRENT_BRANCH"
        else
            print_info "Aborted push"
            exit 0
        fi
    else
        git push
    fi
    
    print_success "Changes pushed successfully!"
    
    # Show remote status
    print_info "Remote status:"
    git log --oneline origin/"$CURRENT_BRANCH"..HEAD
    
    if [ -z "$(git log --oneline origin/"$CURRENT_BRANCH"..HEAD)" ]; then
        print_success "Local and remote are in sync"
    fi
else
    print_info "Push cancelled. Changes are committed locally."
    print_info "To push later, run: git push"
fi

# Final summary
echo
print_success "Navigation fix has been committed!"
print_info "Summary:"
echo "  - Removed redundant pages from navigation"
echo "  - Consolidated functionality into Trading Hub and Advanced Signals"
echo "  - Improved user experience with cleaner interface"

# Reminder about restarting services
echo
print_warning "Remember to restart your services to see the changes:"
print_info "  docker-compose restart frontend"
print_info "  OR"
print_info "  ./init_deploy.sh restart"

