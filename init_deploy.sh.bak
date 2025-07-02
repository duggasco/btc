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
discord_warning() { send_discord_notification "⚠️ $1" 16776960; }  # Yellow

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

    # Copy .env to docker directory for docker-compose
    cp .env docker/.env

    if [ -n "$webhook_to_use" ]; then
        print_status "Discord webhook URL configured"
        discord_info "Discord notifications enabled"
    else
        print_warning "Discord webhook URL not set - notifications disabled"
        print_info "To enable Discord notifications, set DISCORD_WEBHOOK_URL in .env file"
    fi

    # Create trading config if it doesn't exist
    if [ ! -f /storage/config/trading_config.json ]; then
        cat > /storage/config/trading_config.json << 'EOL'
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

prepare_source_files() {
    print_info "Preparing source files for Docker context..."
    
    # Create docker/src directory structure if it doesn't exist
    mkdir -p docker/src/{backend/{models,services,api},frontend}
    
    # Copy backend Python files to docker context
    print_info "Copying backend source files..."
    
    # Backend API
    cp -f src/backend/api/main.py docker/backend_api.py 2>/dev/null || print_warning "main.py not found"
    
    # Backend models
    cp -f src/backend/models/database.py docker/database_models.py 2>/dev/null || print_warning "database.py not found"
    cp -f src/backend/models/lstm.py docker/lstm_model.py 2>/dev/null || print_warning "lstm.py not found"
    cp -f src/backend/models/paper_trading.py docker/paper_trading_persistence.py 2>/dev/null || print_warning "paper_trading.py not found"
    
    # Backend services
    cp -f src/backend/services/data_fetcher.py docker/external_data_fetcher.py 2>/dev/null || print_warning "data_fetcher.py not found"
    cp -f src/backend/services/integration.py docker/integration.py 2>/dev/null || print_warning "integration.py not found"
    cp -f src/backend/services/backtesting.py docker/backtesting_system.py 2>/dev/null || print_warning "backtesting.py not found"
    cp -f src/backend/services/notifications.py docker/discord_notifications.py 2>/dev/null || print_warning "notifications.py not found"
    
    # Frontend
    cp -f src/frontend/app.py docker/streamlit_app.py 2>/dev/null || print_warning "app.py not found"
    
    # Copy requirements files
    cp -f src/backend/requirements.txt docker/requirements-backend.txt 2>/dev/null || print_warning "backend requirements.txt not found"
    cp -f src/frontend/requirements.txt docker/requirements-frontend.txt 2>/dev/null || print_warning "frontend requirements.txt not found"
    
    # Update Dockerfiles with correct naming
    update_dockerfiles
    
    # Update docker-compose.yml to use /storage
    update_docker_compose
    
    print_status "Source files prepared for Docker build"
}

update_dockerfiles() {
    print_info "Updating Dockerfiles..."
    
    # Create backend Dockerfile with correct file references
    cat > docker/Dockerfile.backend << 'EOF'
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including network tools
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    locales \
    wget \
    dnsutils \
    iputils-ping \
    net-tools \
    ca-certificates \
    && sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen \
    && locale-gen \
    && rm -rf /var/lib/apt/lists/*

# Set locale environment variables
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8
ENV PYTHONIOENCODING=utf-8

# Update CA certificates
RUN update-ca-certificates

# Copy requirements and install Python dependencies
COPY requirements-backend.txt .
RUN pip install --no-cache-dir --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org -r requirements-backend.txt

# Copy application files
COPY database_models.py .
COPY lstm_model.py .
COPY external_data_fetcher.py .
COPY integration.py .
COPY backtesting_system.py .
COPY backend_api.py .
COPY discord_notifications.py .
COPY paper_trading_persistence.py .

# Create necessary directories
RUN mkdir -p /app/data /app/models /app/logs /app/config

# Set environment variables
ENV PYTHONPATH=/app
ENV DATABASE_PATH=/app/data/trading_system.db
ENV MODEL_PATH=/app/models
ENV LOG_PATH=/app/logs
ENV CONFIG_PATH=/app/config

# Network environment variables
ENV PYTHONHTTPSVERIFY=0
ENV REQUESTS_CA_BUNDLE=""
ENV CURL_CA_BUNDLE=""

# Expose port
EXPOSE 8000

# Health check with network test
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "backend_api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
EOF

    # Create frontend Dockerfile
    cat > docker/Dockerfile.frontend << 'EOF'
FROM python:3.11-slim

WORKDIR /app

# Install dependencies and configure locale for UTF-8 support
RUN apt-get update && apt-get install -y \
    curl \
    locales \
    && rm -rf /var/lib/apt/lists/* \
    && sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen \
    && locale-gen

# Set locale environment variables
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8
ENV PYTHONIOENCODING=utf-8

COPY requirements-frontend.txt .
RUN pip install --no-cache-dir -r requirements-frontend.txt

COPY streamlit_app.py .

RUN mkdir -p ~/.streamlit /app/logs /app/config

RUN echo "\
[server]\n\
headless = true\n\
enableCORS = false\n\
port = 8501\n\
[theme]\n\
base = \"dark\"\n\
" > ~/.streamlit/config.toml

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "streamlit_app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
EOF

    print_status "Dockerfiles updated"
}

update_docker_compose() {
    print_info "Updating docker-compose.yml to use /storage..."
    
    cat > docker/docker-compose.yml << 'EOF'
services:
  backend:
    build:
      context: .
      dockerfile: Dockerfile.backend
    container_name: btc-trading-backend
    ports:
      - "8080:8000"
    volumes:
      # Map to /storage instead of ./storage
      - /storage/data:/app/data
      - /storage/models:/app/models
      - /storage/logs/backend:/app/logs
      - /storage/config:/app/config
      # Mount the Python files for hot reloading
      - ./backend_api.py:/app/backend_api.py:ro
      - ./database_models.py:/app/database_models.py:ro
      - ./lstm_model.py:/app/lstm_model.py:ro
      - ./external_data_fetcher.py:/app/external_data_fetcher.py:ro
      - ./integration.py:/app/integration.py:ro
      - ./backtesting_system.py:/app/backtesting_system.py:ro
      - ./discord_notifications.py:/app/discord_notifications.py:ro
      - ./paper_trading_persistence.py:/app/paper_trading_persistence.py:ro
    environment:
      - DATABASE_PATH=/app/data/trading_system.db
      - MODEL_PATH=/app/models
      - LOG_PATH=/app/logs
      - CONFIG_PATH=/app/config
      - PYTHONUNBUFFERED=1
      - DISCORD_WEBHOOK_URL=${DISCORD_WEBHOOK_URL}
      # Add proxy settings for yfinance
      - HTTP_PROXY=
      - HTTPS_PROXY=
      - NO_PROXY=localhost,127.0.0.1
    networks:
      - trading-network
    restart: unless-stopped
    dns:
      - 8.8.8.8
      - 8.8.4.4
      - 1.1.1.1
    extra_hosts:
      - "host.docker.internal:host-gateway"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

  frontend:
    build:
      context: .
      dockerfile: Dockerfile.frontend
    container_name: btc-trading-frontend
    ports:
      - "8501:8501"
    volumes:
      # Map to /storage instead of ./storage
      - /storage/logs/frontend:/app/logs
      - /storage/config:/app/config
      # Mount streamlit app for hot reloading
      - ./streamlit_app.py:/app/streamlit_app.py:ro
    environment:
      - API_BASE_URL=http://backend:8000
      - LOG_PATH=/app/logs
      - CONFIG_PATH=/app/config
      - PYTHONUNBUFFERED=1
      - STREAMLIT_SERVER_ENABLE_CORS=false
      - STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false
    depends_on:
      backend:
        condition: service_healthy
    networks:
      - trading-network
    restart: unless-stopped
    dns:
      - 8.8.8.8
      - 8.8.4.4
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

  # Optional Redis cache for performance
  redis:
    image: redis:7-alpine
    container_name: btc-trading-redis
    ports:
      - "6379:6379"
    networks:
      - trading-network
    restart: unless-stopped
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 3s
      retries: 3
    profiles:
      - cache

networks:
  trading-network:
    driver: bridge
    ipam:
      driver: default
      config:
        - subnet: 172.20.0.0/16

volumes:
  btc_data:
    driver: local
  btc_models:
    driver: local
  btc_logs:
    driver: local
  redis_data:
    driver: local
EOF

    print_status "docker-compose.yml updated to use /storage"
}

build_and_start() {
    print_info "Building and starting services..."
    discord_info "Starting Docker build process..."
    
    cd docker
    
    # Stop existing containers
    docker compose down --remove-orphans 2>/dev/null || true
    
    # Build images
    if docker compose build --no-cache; then
        discord_success "Docker images built successfully"
    else
        discord_error "Docker build failed"
        print_error "Docker build failed"
    fi
    
    # Start services
    if docker compose up -d; then
        discord_success "Docker containers started"
    else
        discord_error "Failed to start Docker containers"
        print_error "Failed to start containers"
    fi
    
    cd ..
    
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
    
    # Copy test script to docker directory and update API URL
    if [ -f scripts/test_system.py ]; then
        cp scripts/test_system.py docker/test_system.py
        cd docker
        
        # Update API URL in test script
        sed -i 's|API_BASE_URL = "http://localhost:8080"|API_BASE_URL = "http://localhost:8080"|g' test_system.py 2>/dev/null || \
        sed -i '' 's|API_BASE_URL = "http://localhost:8080"|API_BASE_URL = "http://localhost:8080"|g' test_system.py 2>/dev/null || true
        
        if python3 test_system.py quick; then
            discord_success "System tests passed"
        else
            discord_warning "Some tests failed"
        fi
        
        cd ..
    else
        print_warning "test_system.py not found, skipping tests"
        discord_warning "Test file not found, skipping tests"
    fi
}

send_system_status() {
    local backend_status="❌ Down"
    local frontend_status="❌ Down"
    
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
    echo "  View logs:       cd docker && docker compose logs -f"
    echo "  Stop services:   cd docker && docker compose down"
    echo "  Restart:         cd docker && docker compose restart"
    echo "  Run tests:       cd docker && python3 test_system.py"
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
        prepare_source_files
        build_and_start
        run_tests
        show_status
        discord_success "🎉 BTC Trading System deployed successfully!"
        ;;
    "start")
        discord_info "Starting BTC Trading System..."
        cd docker && docker compose up -d && cd ..
        sleep 5
        run_tests
        show_status
        discord_success "System started successfully"
        ;;
    "stop")
        discord_info "Stopping BTC Trading System..."
        cd docker && docker compose down && cd ..
        print_status "Services stopped"
        discord_success "System stopped successfully"
        ;;
    "restart")
        discord_info "Restarting BTC Trading System..."
        cd docker && docker compose restart && cd ..
        sleep 5
        run_tests
        show_status
        discord_success "System restarted successfully"
        ;;
    "logs")
        cd docker && docker compose logs -f
        ;;
    "build")
        discord_info "Building Docker images..."
        prepare_source_files
        cd docker && docker compose build --no-cache && cd ..
        print_status "Build complete"
        discord_success "Docker images built successfully"
        ;;
    "test")
        run_tests
        ;;
    "clean")
        discord_warning "Cleaning up Docker resources..."
        cd docker && docker compose down --volumes --remove-orphans && cd ..
        docker system prune -f
        print_status "Cleanup complete"
        discord_success "Cleanup completed successfully"
        ;;
    "status")
        cd docker && docker compose ps && cd ..
        echo ""
        show_status
        ;;
    "prepare")
        prepare_source_files
        print_status "Source files prepared"
        ;;
    *)
        echo "Usage: $0 {deploy|start|stop|restart|logs|build|test|clean|status|prepare}"
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
        echo "  prepare   - Prepare source files for Docker"
        echo ""
        echo "Discord notifications: Set DISCORD_WEBHOOK_URL environment variable"
        echo ""
        echo "Note: All persistent data is stored in /storage"
        ;;
esac
