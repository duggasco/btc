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

print_status() { echo -e "${GREEN}? $1${NC}"; }
print_warning() { echo -e "${YELLOW}??  $1${NC}"; }
print_error() { echo -e "${RED}? $1${NC}"; exit 1; }
print_info() { echo -e "${BLUE}??  $1${NC}"; }

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

discord_info() { send_discord_notification "?? $1" 3447003; }      # Blue
discord_success() { send_discord_notification "? $1" 3066993; }   # Green
discord_error() { send_discord_notification "? $1" 15158332; }    # Red
discord_warning() { send_discord_notification "?? $1" 16776960; }  # Yellow

check_dependencies() {
    print_info "Checking dependencies..."
    discord_info "Checking system dependencies..."
    
    if ! command -v docker &> /dev/null; then
        discord_error "Docker is not installed"
        print_error "Docker is not installed"
    fi
    
    if ! command -v docker compose &> /dev/null; then
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
    mkdir -p storage/{data,models,logs/{backend,frontend,system},config,backups,uploads,exports}
    chmod -R 755 storage
    print_status "Storage directories created"
    discord_success "Storage directories created successfully"
}

create_config() {
    print_info "Creating configuration files..."
    discord_info "Creating configuration files..."
    
    cat > .env << 'EOL'
DATABASE_PATH=/app/data/trading_system.db
MODEL_PATH=/app/models
LOG_PATH=/app/logs
CONFIG_PATH=/app/config
API_HOST=0.0.0.0
API_PORT=8000
STREAMLIT_HOST=0.0.0.0
STREAMLIT_PORT=8501
EOL

    if [ ! -f storage/config/trading_config.json ]; then
        cat > storage/config/trading_config.json << 'EOL'
{
    "trading": {
        "default_symbol": "BTC-USD",
        "risk_tolerance": 0.02,
        "max_position_size": 1.0,
        "stop_loss_percentage": 0.05,
        "take_profit_percentage": 0.10
    },
    "model": {
        "sequence_length": 60,
        "update_frequency": 300,
        "confidence_threshold": 0.7
    },
    "api": {
        "timeout": 30,
        "retry_attempts": 3,
        "rate_limit": 100
    }
}
EOL
    fi
    
    print_status "Configuration files created"
    discord_success "Configuration files created"
}

build_and_start() {
    print_info "Building and starting services..."
    discord_info "Starting Docker build process..."
    
    docker compose down --remove-orphans 2>/dev/null || true
    
    if docker compose build --no-cache; then
        discord_success "Docker images built successfully"
    else
        discord_error "Docker build failed"
        print_error "Docker build failed"
    fi
    
    if docker compose up -d; then
        discord_success "Docker containers started"
    else
        discord_error "Failed to start Docker containers"
        print_error "Failed to start containers"
    fi
    
    print_info "Waiting for services to be ready..."
    discord_info "Waiting for services to initialize..."
    sleep 15
    
    # Check backend health
    for i in {1..30}; do
        if curl -f http://localhost:8080/health >/dev/null 2>&1; then
            print_status "Backend is healthy"
            discord_success "Backend API is healthy and responding"
            break
        fi
        if [ $i -eq 30 ]; then
            discord_error "Backend failed to start after 30 attempts"
            print_error "Backend failed to start"
        fi
        sleep 2
    done
    
    # Check frontend health
    for i in {1..30}; do
        if curl -f http://localhost:8501/_stcore/health >/dev/null 2>&1; then
            print_status "Frontend is healthy"
            discord_success "Frontend UI is healthy and responding"
            break
        fi
        if [ $i -eq 30 ]; then
            discord_warning "Frontend may still be starting"
            print_warning "Frontend may still be starting"
        fi
        sleep 2
    done
    
    print_status "Services started successfully"
}

run_tests() {
    print_info "Running system tests..."
    discord_info "Running system tests..."
    if [ -f test_system.py ]; then
        if python3 test_system.py quick; then
            discord_success "System tests passed"
        else
            discord_warning "Some tests failed"
        fi
    else
        print_warning "test_system.py not found, skipping tests"
        discord_warning "Test file not found, skipping tests"
    fi
}

send_system_status() {
    local backend_status="? Down"
    local frontend_status="? Down"
    
    if curl -f http://localhost:8080/health >/dev/null 2>&1; then
        backend_status="? Running"
    fi
    
    if curl -f http://localhost:8501/_stcore/health >/dev/null 2>&1; then
        frontend_status="? Running"
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
    echo -e "${GREEN}?? BTC Trading System is running!${NC}"
    echo ""
    echo -e "${BLUE}?? Services:${NC}"
    echo "  Backend API:     http://localhost:8080"
    echo "  Frontend UI:     http://localhost:8501"
    echo "  API Docs:        http://localhost:8080/docs"
    echo ""
    echo -e "${BLUE}?? Management Commands:${NC}"
    echo "  View logs:       docker compose logs -f"
    echo "  Stop services:   docker compose down"
    echo "  Restart:         docker compose restart"
    echo "  Run tests:       python3 test_system.py"
    echo ""
    echo -e "${BLUE}?? Storage:${NC}"
    echo "  Data:            ./storage/data/"
    echo "  Models:          ./storage/models/"
    echo "  Logs:            ./storage/logs/"
    echo "  Config:          ./storage/config/"
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

case "${1:-deploy}" in
    "deploy"|"init")
        discord_info "?? Starting BTC Trading System deployment..."
        check_dependencies
        create_storage
        create_config
        build_and_start
        run_tests
        show_status
        discord_success "?? BTC Trading System deployed successfully!"
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
    *)
        echo "Usage: $0 {deploy|start|stop|restart|logs|build|test|clean|status}"
        echo ""
        echo "Commands:"
        echo "  deploy    - Full deployment (default)"
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
        ;;
esac