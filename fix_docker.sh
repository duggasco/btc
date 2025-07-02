#!/bin/bash

# Docker Migration Fix Script
# Fixes the read-only filesystem issue

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

# Fix 1: Update Dockerfile.backend to not create directories that will be volume mounts
fix_backend_dockerfile() {
    print_info "Fixing Dockerfile.backend..."
    
    cat > Dockerfile.backend << 'EOF'
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

# Copy requirements first for better caching
COPY src/backend/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org -r /tmp/requirements.txt

# Don't create directories that will be volume mounts
# Docker will create them automatically when mounting

# Set environment variables
ENV PYTHONPATH=/app/src:/app
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

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Change working directory to where the source will be mounted
WORKDIR /app/src

# Run the application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
EOF
    
    print_status "Dockerfile.backend fixed"
}

# Fix 2: Update docker-compose.yml to mount source to /app/src instead of /app
fix_docker_compose() {
    print_info "Fixing docker-compose.yml..."
    
    cat > docker-compose.yml << 'EOF'
services:
  backend:
    build:
      context: .
      dockerfile: Dockerfile.backend
    container_name: btc-trading-backend
    ports:
      - "8080:8000"
    volumes:
      # Map persistent storage - Docker will create these directories
      - /storage/data:/app/data
      - /storage/models:/app/models
      - /storage/logs/backend:/app/logs
      - /storage/config:/app/config
      # Mount source code to /app/src (not /app) to avoid conflicts
      - ./src/backend:/app/src:ro
    environment:
      - DATABASE_PATH=/app/data/trading_system.db
      - MODEL_PATH=/app/models
      - LOG_PATH=/app/logs
      - CONFIG_PATH=/app/config
      - PYTHONPATH=/app/src:/app
      - PYTHONUNBUFFERED=1
      - DISCORD_WEBHOOK_URL=${DISCORD_WEBHOOK_URL}
      # Network settings
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
      # Map persistent storage
      - /storage/logs/frontend:/app/logs
      - /storage/config:/app/config
      # Mount source code to /app/src
      - ./src/frontend:/app/src:ro
    environment:
      - API_BASE_URL=http://backend:8000
      - LOG_PATH=/app/logs
      - CONFIG_PATH=/app/config
      - PYTHONPATH=/app/src:/app
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
  redis_data:
    driver: local
EOF
    
    print_status "docker-compose.yml fixed"
}

# Fix 3: Update Dockerfile.frontend similarly
fix_frontend_dockerfile() {
    print_info "Fixing Dockerfile.frontend..."
    
    cat > Dockerfile.frontend << 'EOF'
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

# Copy requirements for better caching
COPY src/frontend/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Configure Streamlit
RUN mkdir -p ~/.streamlit && \
    echo "\
[server]\n\
headless = true\n\
enableCORS = false\n\
port = 8501\n\
[theme]\n\
base = \"dark\"\n\
" > ~/.streamlit/config.toml

# Set Python path
ENV PYTHONPATH=/app/src:/app

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Change working directory to where source will be mounted
WORKDIR /app/src

# Run the application
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
EOF
    
    print_status "Dockerfile.frontend fixed"
}

# Main function
main() {
    echo -e "${BLUE}===============================================${NC}"
    echo -e "${BLUE}   Docker Read-Only Filesystem Fix${NC}"
    echo -e "${BLUE}===============================================${NC}"
    echo ""
    
    print_info "This script fixes the read-only filesystem issue"
    echo ""
    
    # Stop running containers first
    print_info "Stopping any running containers..."
    docker compose down 2>/dev/null || true
    
    # Apply fixes
    fix_backend_dockerfile
    fix_docker_compose
    fix_frontend_dockerfile
    
    echo ""
    print_status "All fixes applied!"
    echo ""
    echo -e "${BLUE}The key changes:${NC}"
    echo "1. Dockerfiles no longer create directories that will be volume mounts"
    echo "2. Source code is now mounted to /app/src instead of /app"
    echo "3. PYTHONPATH updated to include both /app/src and /app"
    echo "4. Working directory changed to /app/src for both services"
    echo ""
    echo -e "${GREEN}Next steps:${NC}"
    echo "1. Rebuild the containers: docker compose build --no-cache"
    echo "2. Start the system: ./init_deploy.sh deploy"
    echo ""
    
    read -p "Would you like to rebuild and start now? (y/N) " -n 1 -r
    echo ""
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Rebuilding containers..."
        if docker compose build --no-cache; then
            print_status "Build successful"
            echo ""
            print_info "Starting services..."
            ./init_deploy.sh deploy
        else
            print_error "Build failed"
        fi
    fi
}

# Run main
main
