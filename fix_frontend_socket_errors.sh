#!/bin/bash
# Fix frontend socket errors deployment script

echo "=== Fixing Frontend Socket Errors ==="
echo ""
echo "Changes made:"
echo "1. Changed SocketIO from 'eventlet' to 'threading' mode to avoid DNS issues"
echo "2. Updated Gunicorn to use gthread worker class with proper configuration"
echo "3. Added gunicorn.conf.py with robust error handling"
echo "4. Added Flask error handlers for broken pipe/connection reset errors"
echo "5. Added keep-alive headers and connection management"
echo ""
echo "Deploying fixes..."
echo ""

# Navigate to docker directory
cd /root/btc/docker || exit 1

# Stop the frontend container
echo "→ Stopping frontend container..."
docker compose down frontend

# Rebuild the frontend image with no cache
echo "→ Rebuilding frontend image..."
docker compose build --no-cache frontend

# Start the services
echo "→ Starting services..."
docker compose up -d

# Wait for services to be ready
echo "→ Waiting for services to start..."
sleep 10

# Check service health
echo "→ Checking service health..."
docker ps | grep btc-trading

# Show recent logs
echo ""
echo "→ Recent frontend logs:"
docker logs --tail 20 btc-trading-frontend

echo ""
echo "=== Deployment Complete ==="
echo ""
echo "Monitor for socket errors with:"
echo "  docker logs -f btc-trading-frontend | grep -i 'error\\|errno'"
echo ""
echo "Test the fix with:"
echo "  python test_frontend_socket_fix.py"