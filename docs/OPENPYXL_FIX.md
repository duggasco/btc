# OpenPyXL Dependency Fix

## Issue
The data upload tool was showing error: "Missing optional dependency 'openpyxl'. Use pip or conda to install openpyxl."

## Root Cause
The `openpyxl` library was missing from the backend's requirements.txt file, even though it was present in the frontend's requirements.txt.

## Fix Applied
1. Added `openpyxl==3.1.2` to `/src/backend/requirements.txt`
2. The dependency was already present in `/src/frontend_flask/requirements.txt`

## Deployment Steps
To apply this fix, rebuild the Docker containers:

```bash
# Stop existing containers
docker compose -f docker/docker-compose.yml down

# Rebuild backend container with new dependency
docker compose -f docker/docker-compose.yml build --no-cache backend

# Start all containers
docker compose -f docker/docker-compose.yml up -d
```

Alternatively, run the deployment script:
```bash
./init_deploy.sh
```

## Verification
After rebuilding, the data upload tool should be able to handle both CSV and XLSX files without the openpyxl error.