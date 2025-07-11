#!/bin/bash

echo "Testing new Trading page..."
echo "=========================="

# Check if Docker Compose is running
echo "1. Checking Docker services..."
docker compose -f docker/docker-compose.yml ps

# Test if the page loads in the frontend container
echo -e "\n2. Testing page import in frontend container..."
docker compose -f docker/docker-compose.yml exec -T frontend python3 -c "
import sys
sys.path.insert(0, '/app/src/frontend')
try:
    # Test imports
    from pages.1_Trading import *
    print('✓ Trading page imports successfully')
except Exception as e:
    print(f'✗ Error importing Trading page: {e}')
"

# Check if WebSocket and API endpoints are accessible
echo -e "\n3. Testing API connectivity..."
curl -s http://localhost:8090/btc/latest | head -n 1

echo -e "\n\nTest complete. If all checks passed, the Trading page should work correctly."