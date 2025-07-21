#!/bin/bash
# Simple script to rebuild containers with openpyxl fix

echo "Stopping containers..."
docker compose -f docker/docker-compose.yml down

echo "Rebuilding backend container..."
docker compose -f docker/docker-compose.yml build --no-cache backend

echo "Rebuilding frontend container..."
docker compose -f docker/docker-compose.yml build --no-cache frontend

echo "Starting containers..."
docker compose -f docker/docker-compose.yml up -d

echo "Done! Containers rebuilt with openpyxl dependency."