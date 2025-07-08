#!/bin/bash

# BTC Trading System - Initialization and Deployment Script
set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# Discord webhook configuration
DISCORD_WEBHOOK_URL="${DISCORD_WEBHOOK_URL:-}"

print_status() { echo -e "${GREEN}✅ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_error() { echo -e "${RED}❌ $1${NC}"; exit 1; }
print_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }

# Discord notification functions
send_discord_notification() {
    local message="$1"
    local color="${2:-3447003}"  # Default blue color
    
    if [ -n "$DISCORD_WEBHOOK_URL" ]; then
        curl -s -H "Content-Type: application/json" -X POST \
            -d "{\"embeds\": [{\"description\": \"$message\", \"color\": $color}]}" \
            "$DISCORD_WEBHOOK_URL" || true
    fi
}

discord_info() { send_discord_notification "ℹ️ $1" 3447003; }      # Blue
discord_success() { send_discord_notification "✅ $1" 3066993; }   # Green
discord_error() { send_discord_notification "❌ $1" 15158332; }    # Red
discord_warning() { send_discord_notification "⚠️ $1" 16776960; }  # Yellow - THIS WAS MISSING!

check_dependencies() {
    print_info "Checking dependencies..."
    discord_info "Checking system dependencies..."
    
    if ! command -v docker &> /dev/null; then
        discord_error "Docker is not installed"
        print_error "Docker is not installed"
    fi
    
    if ! command -v docker compose &> /dev/null && ! docker compose version &> /dev/null; then
        discord_error "Docker Compose is not installed"
        print_error "Docker Compose is not installed"
    fi
    
    if ! docker info &> /dev/null; then
        discord_error "Docker is not running"
        print_error "Docker is not running"
    fi
    
    print_status "Dependencies check passed"
    discord_success "All dependencies verified"
}

create_storage() {
    print_info "Creating storage directories..."
    discord_info "Creating storage directories..."
    
    # Create all necessary directories at /storage
    sudo mkdir -p /storage/{data,models,logs/{backend,frontend,system},config,backups,uploads,exports}
    
    # Set proper permissions (assuming current user should have access)
    sudo chown -R $(whoami):$(whoami) /storage
    chmod -R 755 /storage
    
    # Create .gitkeep files to preserve directory structure
    find /storage -type d -exec touch {}/.gitkeep \;
    
    print_status "Storage directories created at /storage"
    discord_success "Storage directories created successfully at /storage"
}

create_config() {
    print_info "Creating configuration files..."
    discord_info "Creating configuration files..."
    
    # Check if .env exists and has Discord webhook, preserve it
    local existing_webhook=""
    if [ -f .env ]; then
        existing_webhook=$(grep "^DISCORD_WEBHOOK_URL=" .env | cut -d'=' -f2- || true)
    fi
    
    # Use existing webhook or environment variable
    local webhook_to_use="${DISCORD_WEBHOOK_URL:-$existing_webhook}"
    
    # Create root .env file
    cat > .env << EOL
DATABASE_PATH=/app/data/trading_system.db
MODEL_PATH=/app/models
LOG_PATH=/app/logs
CONFIG_PATH=/app/config
API_HOST=0.0.0.0
API_PORT=8000
STREAMLIT_HOST=0.0.0.0
STREAMLIT_PORT=8501
DISCORD_WEBHOOK_URL=${webhook_to_use}
EOL

    if [ -n "$webhook_to_use" ]; then
        print_status "Discord webhook URL configured"
        discord_info "Discord notifications enabled"
    else
        print_warning "Discord webhook URL not set - notifications disabled"
        print_info "To enable Discord notifications, set DISCORD_WEBHOOK_URL in .env file"
    fi

    # Create trading config if it doesn't exist
    if [ ! -f /storage/config/trading_rules.json ]; then
        cat > /storage/config/trading_rules.json << EOL
{
    "min_trade_size": 0.001,
    "max_position_size": 0.1,
    "stop_loss_pct": 5.0,
    "take_profit_pct": 10.0,
    "buy_threshold": 0.6,
    "sell_threshold": 0.6
}
EOL
    fi

    print_status "Configuration files created"
    discord_success "Configuration completed"
}

build_and_start() {
    print_info "Building and starting services..."
    discord_info "Building Docker images and starting services..."
    
    docker compose build
    docker compose up -d
    
    print_info "Waiting for services to start..."
    sleep 10
    
    print_status "Services started"
    discord_success "All services are up and running"
}

run_tests() {
    print_info "Running system tests..."
    discord_info "Running comprehensive system tests..."
    
    # Wait a bit more for services to be fully ready
    sleep 5
    
    if command -v python3 &> /dev/null; then
        if [ -f test_system.py ]; then
            python3 test_system.py quick || {
                discord_error "System tests failed"
                print_warning "Some tests failed - check logs for details"
            }
        else
            print_warning "test_system.py not found - skipping tests"
        fi
    else
        print_warning "Python3 not found - skipping tests"
    fi
}

send_system_status() {
    local backend_status="❌ Not running"
    local frontend_status="❌ Not running"
    
    if curl -f http://localhost:8080/health >/dev/null 2>&1; then
        backend_status="✅ Running"
    fi
    
    if curl -f http://localhost:8501/_stcore/health >/dev/null 2>&1; then
        frontend_status="✅ Running"
    fi
    
    local status_message="**BTC Trading System Status**\n\n"
    status_message+="Backend API: $backend_status\n"
    status_message+="Frontend UI: $frontend_status\n"
    status_message+="\n**Access URLs:**\n"
    status_message+="• Backend: http://localhost:8080\n"
    status_message+="• Frontend: http://localhost:8501\n"
    status_message+="• API Docs: http://localhost:8080/docs"
    
    send_discord_notification "$status_message" 3447003
}

show_status() {
    echo ""
    echo -e "${GREEN}🎉 BTC Trading System is running!${NC}"
    echo ""
    echo -e "${BLUE}📊 Services:${NC}"
    echo "  Backend API:     http://localhost:8080"
    echo "  Frontend UI:     http://localhost:8501"
    echo "  API Docs:        http://localhost:8080/docs"
    echo ""
    echo -e "${BLUE}📝 Management Commands:${NC}"
    echo "  View logs:       docker compose logs -f"
    echo "  Stop services:   docker compose down"
    echo "  Restart:         docker compose restart"
    echo ""
    echo -e "${BLUE}📁 Storage:${NC}"
    echo "  Data:            /storage/data/"
    echo "  Models:          /storage/models/"
    echo "  Logs:            /storage/logs/"
    echo "  Config:          /storage/config/"
    echo ""
    
    # Check Discord status
    if [ -f .env ] && grep -q "DISCORD_WEBHOOK_URL=." .env; then
        echo -e "${GREEN}🔔 Discord notifications: ENABLED${NC}"
    else
        echo -e "${YELLOW}🔕 Discord notifications: DISABLED${NC}"
        echo "  To enable: Add DISCORD_WEBHOOK_URL to .env file"
    fi
    echo ""
    
    if [ -n "$DISCORD_WEBHOOK_URL" ]; then
        send_system_status
    fi
}

# Error handler
handle_error() {
    local error_msg="Deployment failed at line $1"
    print_error "$error_msg"
    discord_error "$error_msg"
    exit 1
}

trap 'handle_error $LINENO' ERR

# Main execution
case "${1:-deploy}" in
    "deploy"|"init")
        discord_info "🚀 Starting BTC Trading System deployment..."
        check_dependencies
        create_storage
        create_config
        build_and_start
        run_tests
        show_status
        discord_success "🎉 BTC Trading System deployed successfully!"
        ;;
    "start")
        discord_info "Starting BTC Trading System..."
        docker compose up -d
        sleep 5
        run_tests
        show_status
        discord_success "System started successfully"
        ;;
    "stop")
        discord_info "Stopping BTC Trading System..."
        docker compose down
        print_status "Services stopped"
        discord_success "System stopped successfully"
        ;;
    "restart")
        discord_info "Restarting BTC Trading System..."
        docker compose restart
        sleep 5
        run_tests
        show_status
        discord_success "System restarted successfully"
        ;;
    "logs")
        docker compose logs -f
        ;;
    "build")
        discord_info "Building Docker images..."
        docker compose build --no-cache
        print_status "Build complete"
        discord_success "Docker images built successfully"
        ;;
    "test")
        run_tests
        ;;
    "clean")
        discord_warning "Cleaning up Docker resources..."
        docker compose down --volumes --remove-orphans
        docker system prune -f
        print_status "Cleanup complete"
        discord_success "Cleanup completed successfully"
        ;;
    "status")
        docker compose ps
        echo ""
        show_status
        ;;
    "test-enhanced")
        discord_info "Running enhanced LSTM tests..."
        print_info "Testing enhanced LSTM modules..."
        print_info "Note: This may take 5-10 minutes due to TA-Lib compilation..."
        
        # Build and run enhanced tests with extended timeout
        export DOCKER_BUILDKIT_TIMEOUT=600
        export COMPOSE_HTTP_TIMEOUT=600
        
        docker compose -f docker-compose.test-enhanced.yml build --no-cache
        docker compose -f docker-compose.test-enhanced.yml up --abort-on-container-exit --timeout 600
        docker compose -f docker-compose.test-enhanced.yml down
        
        # If system is running, test the endpoints
        if docker compose ps | grep -q "Up"; then
            print_info "Testing enhanced LSTM endpoints..."
            docker run --rm --network=btc_default -v $(pwd)/scripts:/scripts python:3.11-slim \
                bash -c "pip install requests && python /scripts/test_enhanced_system.py"
        fi
        
        discord_success "Enhanced LSTM tests completed"
        ;;
    *)
        echo "Usage: $0 {deploy|start|stop|restart|logs|build|test|test-enhanced|clean|status}"
        echo ""
        echo "Commands:"
        echo "  deploy        - Full deployment (default)"
        echo "  test-enhanced - Test enhanced LSTM modules"
        echo "  start     - Start services"
        echo "  stop      - Stop services"
        echo "  restart   - Restart services"
        echo "  logs      - View logs"
        echo "  build     - Rebuild containers"
        echo "  test      - Run tests"
        echo "  clean     - Clean up containers and volumes"
        echo "  status    - Show service status"
        echo ""
        echo "Discord notifications: Set DISCORD_WEBHOOK_URL environment variable"
        echo ""
        echo "Note: All persistent data is stored in /storage"
        ;;
esac
