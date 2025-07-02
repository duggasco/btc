#!/bin/bash

# BTC Trading System - Directory Restructuring Script
# This script safely reorganizes files without breaking functionality

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() { echo -e "${GREEN}✅ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_error() { echo -e "${RED}❌ $1${NC}"; }
print_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }

# Check if we're in the right directory
if [ ! -f "backend_api.py" ] || [ ! -f "streamlit_app.py" ]; then
    print_error "Please run this script from the root directory of the BTC Trading System"
    exit 1
fi

print_info "Starting BTC Trading System directory restructuring..."
print_warning "This script will create a backup before making changes"

# Create backup
BACKUP_DIR="backup_$(date +%Y%m%d_%H%M%S)"
print_info "Creating backup in $BACKUP_DIR..."
mkdir -p "$BACKUP_DIR"
cp -r . "$BACKUP_DIR/" 2>/dev/null || true
print_status "Backup created"

# Create new directory structure
print_info "Creating new directory structure..."

mkdir -p src/backend/{api/routes,models,services}
mkdir -p src/frontend/{pages,components}
mkdir -p docker
mkdir -p scripts
mkdir -p tests
mkdir -p docs
mkdir -p config

# Create __init__.py files
touch src/__init__.py
touch src/backend/__init__.py
touch src/backend/api/__init__.py
touch src/backend/api/routes/__init__.py
touch src/backend/models/__init__.py
touch src/backend/services/__init__.py
touch src/frontend/__init__.py
touch src/frontend/pages/__init__.py
touch src/frontend/components/__init__.py
touch tests/__init__.py

print_status "Directory structure created"

# Move backend files
print_info "Moving backend files..."

# Move main API file
cp backend_api.py src/backend/api/main.py

# Move models
cp database_models.py src/backend/models/database.py
cp lstm_model.py src/backend/models/lstm.py
cp paper_trading_persistence.py src/backend/models/paper_trading.py

# Move services
cp external_data_fetcher.py src/backend/services/data_fetcher.py
cp backtesting_system.py src/backend/services/backtesting.py
cp integration.py src/backend/services/integration.py
cp discord_notifications.py src/backend/services/notifications.py

# Move requirements
cp requirements-backend.txt src/backend/requirements.txt

print_status "Backend files moved"

# Move frontend files
print_info "Moving frontend files..."

cp streamlit_app.py src/frontend/app.py
cp requirements-frontend.txt src/frontend/requirements.txt

print_status "Frontend files moved"

# Move Docker files
print_info "Moving Docker files..."

cp Dockerfile.backend docker/backend.Dockerfile
cp Dockerfile.frontend docker/frontend.Dockerfile
cp docker-compose.yml docker/docker-compose.yml

print_status "Docker files moved"

# Move scripts
print_info "Moving scripts..."

cp init_and_deploy.sh scripts/init_deploy.sh
cp create_gitkeeps.sh scripts/create_gitkeeps.sh
cp test_system.py scripts/test_system.py

# Make scripts executable
chmod +x scripts/*.sh

print_status "Scripts moved"

# Move tests
print_info "Moving test files..."

cp test_external_fetcher.py tests/test_data_fetcher.py

print_status "Test files moved"

# Move documentation
print_info "Moving documentation..."

cp trading_signals_api.md docs/API.md
cp README.md README.md  # Keep in root
cp .gitignore .gitignore  # Keep in root

print_status "Documentation moved"

# Move config
print_info "Moving configuration..."

if [ -f ".env" ]; then
    cp .env config/.env
    cp .env .env  # Keep a copy in root for Docker
fi

print_status "Configuration moved"

# Update imports in Python files
print_info "Updating Python imports..."

# Update backend imports
# Update main.py (backend_api.py)
sed -i.bak 's/from paper_trading_persistence import/from ..models.paper_trading import/g' src/backend/api/main.py
sed -i.bak 's/from external_data_fetcher import/from ..services.data_fetcher import/g' src/backend/api/main.py
sed -i.bak 's/from database_models import/from ..models.database import/g' src/backend/api/main.py
sed -i.bak 's/from lstm_model import/from ..models.lstm import/g' src/backend/api/main.py
sed -i.bak 's/from integration import/from ..services.integration import/g' src/backend/api/main.py
sed -i.bak 's/from discord_notifications import/from ..services.notifications import/g' src/backend/api/main.py
sed -i.bak 's/from backtesting_system import/from ..services.backtesting import/g' src/backend/api/main.py

# Update integration.py
sed -i.bak 's/from lstm_model import/from ..models.lstm import/g' src/backend/services/integration.py
sed -i.bak 's/from database_models import/from ..models.database import/g' src/backend/services/integration.py
sed -i.bak 's/from backtesting_system import/from .backtesting import/g' src/backend/services/integration.py
sed -i.bak 's/from external_data_fetcher import/from .data_fetcher import/g' src/backend/services/integration.py

# Update lstm_model.py
sed -i.bak 's/from external_data_fetcher import/from ..services.data_fetcher import/g' src/backend/models/lstm.py
sed -i.bak 's/from backtesting_system import/from ..services.backtesting import/g' src/backend/models/lstm.py

# Remove backup files created by sed
find src -name "*.bak" -delete

print_status "Python imports updated"

# Update Docker files
print_info "Updating Docker configurations..."

# Update docker-compose.yml paths
cat > docker/docker-compose.yml << 'EOF'
services:
  backend:
    build:
      context: ..
      dockerfile: docker/backend.Dockerfile
    container_name: btc-trading-backend
    ports:
      - "8080:8000"
    volumes:
      - ../storage/data:/app/data
      - ../storage/models:/app/models
      - ../storage/logs/backend:/app/logs
      - ../storage/config:/app/config
      # Mount the Python files for hot reloading
      - ../src/backend:/app/src/backend:ro
    environment:
      - DATABASE_PATH=/app/data/trading_system.db
      - MODEL_PATH=/app/models
      - LOG_PATH=/app/logs
      - CONFIG_PATH=/app/config
      - PYTHONUNBUFFERED=1
      - PYTHONPATH=/app
      - DISCORD_WEBHOOK_URL=${DISCORD_WEBHOOK_URL}
    networks:
      - trading-network
    restart: unless-stopped
    dns:
      - 8.8.8.8
      - 8.8.4.4
      - 1.1.1.1
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  frontend:
    build:
      context: ..
      dockerfile: docker/frontend.Dockerfile
    container_name: btc-trading-frontend
    ports:
      - "8501:8501"
    volumes:
      - ../storage/logs/frontend:/app/logs
      - ../storage/config:/app/config
      - ../src/frontend:/app/src/frontend:ro
    environment:
      - API_BASE_URL=http://backend:8000
      - LOG_PATH=/app/logs
      - CONFIG_PATH=/app/config
      - PYTHONUNBUFFERED=1
      - PYTHONPATH=/app
    depends_on:
      backend:
        condition: service_healthy
    networks:
      - trading-network
    restart: unless-stopped

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
    profiles:
      - cache

networks:
  trading-network:
    driver: bridge

volumes:
  redis_data:
    driver: local
EOF

# Update backend Dockerfile
cat > docker/backend.Dockerfile << 'EOF'
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc g++ curl locales wget dnsutils iputils-ping net-tools ca-certificates \
    && sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen \
    && locale-gen \
    && rm -rf /var/lib/apt/lists/*

# Set locale
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8
ENV PYTHONIOENCODING=utf-8

# Copy requirements
COPY src/backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY src/backend /app/src/backend

# Create directories
RUN mkdir -p /app/data /app/models /app/logs /app/config

# Set Python path
ENV PYTHONPATH=/app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.backend.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
EOF

# Update frontend Dockerfile
cat > docker/frontend.Dockerfile << 'EOF'
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y curl locales \
    && rm -rf /var/lib/apt/lists/* \
    && sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen \
    && locale-gen

# Set locale
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8
ENV PYTHONIOENCODING=utf-8

# Copy requirements
COPY src/frontend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY src/frontend /app/src/frontend

# Create directories
RUN mkdir -p ~/.streamlit /app/logs /app/config

# Configure Streamlit
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

CMD ["streamlit", "run", "src/frontend/app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
EOF

print_status "Docker configurations updated"

# Update deployment script
print_info "Updating deployment script..."

sed -i.bak 's|docker compose|cd docker && docker compose|g' scripts/init_deploy.sh
sed -i.bak 's|docker-compose|cd docker && docker-compose|g' scripts/init_deploy.sh

# Remove backup
rm -f scripts/init_deploy.sh.bak

print_status "Deployment script updated"

# Create new .gitignore in src directories
echo "__pycache__/" > src/.gitignore
echo "*.pyc" >> src/.gitignore
echo ".pytest_cache/" >> src/.gitignore

# Create run script for easy execution
cat > run.sh << 'EOF'
#!/bin/bash
# Quick run script for the reorganized structure

case "$1" in
    "backend")
        cd src/backend && uvicorn api.main:app --reload --port 8000
        ;;
    "frontend")
        cd src/frontend && streamlit run app.py
        ;;
    "docker")
        cd docker && docker-compose up -d
        ;;
    "test")
        python scripts/test_system.py
        ;;
    *)
        echo "Usage: ./run.sh {backend|frontend|docker|test}"
        ;;
esac
EOF

chmod +x run.sh

# Create migration info
cat > MIGRATION_INFO.md << 'EOF'
# Directory Structure Migration - Completed

The BTC Trading System has been reorganized into a cleaner structure.

## New Structure:
- `src/backend/` - All backend Python code
- `src/frontend/` - Streamlit frontend
- `docker/` - Docker configurations
- `scripts/` - Utility scripts
- `tests/` - Test files
- `docs/` - Documentation
- `config/` - Configuration files
- `storage/` - Data storage (unchanged)

## Important Notes:
1. A backup was created in `backup_[timestamp]/`
2. Original files are preserved
3. Docker commands now run from the `docker/` directory
4. Use `./run.sh` for quick commands

## Quick Commands:
- Start with Docker: `cd docker && docker-compose up -d`
- Run backend: `./run.sh backend`
- Run frontend: `./run.sh frontend`
- Run tests: `./run.sh test`

## Rollback:
If needed, restore from the backup directory.
EOF

print_status "Migration completed successfully!"
print_info "A backup was created in: $BACKUP_DIR"
print_info "Original files have been preserved"
print_warning "Please review MIGRATION_INFO.md for important information"
print_info "To start the system: cd docker && docker-compose up -d"

# Summary
echo ""
echo "Summary of changes:"
echo "1. ✅ Created new directory structure"
echo "2. ✅ Moved all files to appropriate locations"
echo "3. ✅ Updated Python imports"
echo "4. ✅ Updated Docker configurations"
echo "5. ✅ Created backup in $BACKUP_DIR"
echo "6. ✅ Added run.sh helper script"
echo ""
echo "Next steps:"
echo "1. Review the new structure"
echo "2. Test the system: ./run.sh test"
echo "3. Start with Docker: cd docker && docker-compose up -d"
