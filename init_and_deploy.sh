#!/bin/bash

# BTC Trading System - Initialization and Deployment Script
set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() { echo -e "${GREEN}✅ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_error() { echo -e "${RED}❌ $1${NC}"; exit 1; }
print_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }

check_dependencies() {
    print_info "Checking dependencies..."
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed"
    fi
    
    if ! command -v docker compose &> /dev/null; then
        print_error "Docker Compose is not installed"
    fi
    
    if ! docker info &> /dev/null; then
        print_error "Docker is not running"
    fi
    
    print_status "Dependencies check passed"
}

create_storage() {
    print_info "Creating storage directories..."
    mkdir -p storage/{data,models,logs/{backend,frontend,system},config,backups,uploads,exports}
    chmod -R 755 storage
    print_status "Storage directories created"
}

create_config() {
    print_info "Creating configuration files..."
    
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
}

build_and_start() {
    print_info "Building and starting services..."
    
    docker compose down --remove-orphans 2>/dev/null || true
    docker compose build --no-cache
    docker compose up -d
    
    print_info "Waiting for services to be ready..."
    sleep 15
    
    # Check backend health
    for i in {1..30}; do
        if curl -f http://localhost:8080/health >/dev/null 2>&1; then
            print_status "Backend is healthy"
            break
        fi
        if [ $i -eq 30 ]; then
            print_error "Backend failed to start"
        fi
        sleep 2
    done
    
    # Check frontend health
    for i in {1..30}; do
        if curl -f http://localhost:8501/_stcore/health >/dev/null 2>&1; then
            print_status "Frontend is healthy"
            break
        fi
        if [ $i -eq 30 ]; then
            print_warning "Frontend may still be starting"
        fi
        sleep 2
    done
    
    print_status "Services started successfully"
}

run_tests() {
    print_info "Running system tests..."
    if [ -f test_system.py ]; then
        python3 test_system.py quick
    else
        print_warning "test_system.py not found, skipping tests"
    fi
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
    echo "  Run tests:       python3 test_system.py"
    echo ""
    echo -e "${BLUE}📂 Storage:${NC}"
    echo "  Data:            ./storage/data/"
    echo "  Models:          ./storage/models/"
    echo "  Logs:            ./storage/logs/"
    echo "  Config:          ./storage/config/"
    echo ""
}

case "${1:-deploy}" in
    "deploy"|"init")
        check_dependencies
        create_storage
        create_config
        build_and_start
        run_tests
        show_status
        ;;
    "start")
        docker compose up -d
        sleep 5
        run_tests
        show_status
        ;;
    "stop")
        docker compose down
        print_status "Services stopped"
        ;;
    "restart")
        docker compose restart
        sleep 5
        run_tests
        show_status
        ;;
    "logs")
        docker compose logs -f
        ;;
    "build")
        docker compose build --no-cache
        print_status "Build complete"
        ;;
    "test")
        run_tests
        ;;
    "clean")
        docker compose down --volumes --remove-orphans
        docker system prune -f
        print_status "Cleanup complete"
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
        ;;
esac
