#!/bin/bash

# Create .gitkeep files for empty directories
echo "Creating .gitkeep files for storage directories..."

directories=(
    "storage/data"
    "storage/models"
    "storage/logs/backend"
    "storage/logs/frontend"
    "storage/logs/system"
    "storage/backups"
    "storage/uploads" 
    "storage/exports"
)

for dir in "${directories[@]}"; do
    mkdir -p "$dir"
    touch "$dir/.gitkeep"
    echo "Created: $dir/.gitkeep"
done

echo "✅ All .gitkeep files created successfully"
