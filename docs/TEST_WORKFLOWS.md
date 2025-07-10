# Docker Test Workflows

This document describes the available Docker-based test workflows for the BTC Trading System.

All Docker configuration files are now located in the `./docker` directory.

## Test Workflows

### 1. Basic Test Suite (`docker/docker/docker-compose.test.yml`)
- **Dockerfile**: `docker/docker/Dockerfile.test`
- **Purpose**: Runs the complete test suite using tests/run_tests.py
- **Usage**: 
  ```bash
  docker compose -f docker/docker-compose.test.yml up
  ```
- **Features**:
  - Runs all unit and integration tests
  - Generates coverage reports in `./htmlcov`
  - Outputs test results to `./test-results`

### 2. Frontend API Testing (`docker/docker-compose.test-frontend.yml`)
- **Dockerfile**: `docker/docker/Dockerfile.test-frontend`
- **Purpose**: Tests API endpoints with a running backend
- **Usage**:
  ```bash
  docker compose -f docker/docker-compose.test-frontend.yml up
  # or
  ./tests/scripts/run_complete_tests.sh
  ```
- **Features**:
  - Starts backend service
  - Runs comprehensive API endpoint tests
  - Generates test reports in `./test_reports`
  - Includes health checks

### 3. Enhanced LSTM Testing (`docker/docker-compose.test-enhanced.yml`)
- **Dockerfile**: `docker/docker/Dockerfile.test-enhanced`
- **Purpose**: Tests enhanced LSTM model functionality
- **Usage**:
  ```bash
  docker compose -f docker/docker-compose.test-enhanced.yml up
  # or
  ./init_deploy.sh test-enhanced
  ```
- **Features**:
  - Focuses on enhanced LSTM integration tests
  - Lightweight testing for ML components

## Test Dockerfiles

### Main Test Images

1. **`docker/docker/Dockerfile.test`**
   - Full test environment with all dependencies
   - Includes backend, frontend, and test requirements
   - Default: runs all tests via tests/run_tests.py

2. **`docker/docker/Dockerfile.test-simple`**
   - Minimal test environment
   - Only essential dependencies
   - Default: runs unit tests only (faster)
   - Usage: For quick unit test runs

3. **`docker/docker/Dockerfile.test-frontend`**
   - API testing environment
   - Includes test scripts for endpoint testing
   - Used by frontend test workflow

4. **`docker/docker/Dockerfile.test-enhanced`**
   - Enhanced LSTM testing environment
   - Focused on ML model testing
   - Minimal dependencies for faster builds

## Removed/Deprecated Files

The following files were removed during consolidation:
- `docker/docker/Dockerfile.test-e2e` - Unreferenced, functionality covered by other tests
- `docker/docker/Dockerfile.test-full` - Redundant with `docker/docker/Dockerfile.test`
- `docker-compose.test-complete.yml` - Redundant with `test-frontend.yml`

## Test Commands

### Run all tests
```bash
docker compose -f docker/docker-compose.test.yml up
```

### Run unit tests only (fast)
```bash
docker build -f docker/Dockerfile.test-simple -t btc-test-simple .
docker run --rm btc-test-simple
```

### Run API tests with backend
```bash
./tests/scripts/run_complete_tests.sh
```

### Run enhanced LSTM tests
```bash
./init_deploy.sh test-enhanced
```

### Run specific test category
```bash
docker compose -f docker/docker-compose.test.yml run test-runner python /app/tests/run_tests.py unit
```

## Test Reports

- **Coverage Reports**: `./htmlcov/index.html`
- **Test Results**: `./test-results/`
- **API Test Reports**: `./test_reports/api_test_report_*.json`

## Best Practices

1. Use `docker/docker-compose.test.yml` for comprehensive testing during development
2. Use `docker/docker/Dockerfile.test-simple` for quick unit test runs in CI
3. Use `docker/docker-compose.test-frontend.yml` for API integration testing
4. Always clean up test containers: `docker compose -f <compose-file> down -v`