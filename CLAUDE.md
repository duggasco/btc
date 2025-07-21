# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a comprehensive Bitcoin trading system with AI-powered signals using LSTM neural networks. The system features real-time price updates via WebSocket, paper trading capabilities, and a modern web interface built with FastAPI (backend) and Streamlit (frontend).

## Development Guidelines

- Do not use emojis in our developed products
- Always be succinct and precise in our documentation
- Always store documentation in ./docs folder

## UI Architecture (v4.0 - Flask Edition)

The system now uses Flask for the frontend, replacing Streamlit to resolve rendering issues and provide better production capabilities.

### Frontend Structure
- **Framework**: Flask with Jinja2 templates
- **Location**: `src/frontend_flask/`
- **Port**: 8502 (changed from 8501)

### Pages
1. **Trading Dashboard** (`/`)
   - Real-time price monitoring with candlestick charts
   - Portfolio tracking and P&L display
   - Order execution interface with paper trading toggle
   - Trading signals panel
   - Integrated paper trading mode with visual indicators
   
2. **Analytics & Research** (`/analytics/`)
   - Advanced backtesting with walk-forward analysis
   - Monte Carlo simulations for risk assessment
   - Strategy optimization using real backend endpoints
   - Feature importance and performance metrics
   - Data quality monitoring
   
3. **Settings & Configuration** (`/settings/`)
   - Trading rules configuration (position sizing, stop loss, take profit)
   - Signal weight adjustment with visual distribution
   - Model configuration and retraining interface
   - System health monitoring
   - Database maintenance operations

4. **Data Management** (`/data/`) - Added 2025-01-11
   - **Data Upload** (`/data/upload`): Bulk CSV/XLSX file upload with drag-and-drop
   - **Data Quality** (`/data/quality`): Comprehensive quality metrics and gap analysis
   - **Upload History** (`/data/history`): Track uploads with rollback functionality

### Architecture Components
- **Blueprints**: Modular route handlers in `blueprints/`
  - `dashboard.py` - Trading dashboard routes
  - `analytics.py` - Analytics and research routes
  - `settings.py` - Settings and configuration routes
  - `data.py` - Data management routes (upload, quality, history)
  - `api.py` - API proxy routes
  - `websocket.py` - WebSocket handlers
- **Templates**: Jinja2 HTML templates in `templates/`
- **Static Assets**: CSS and JavaScript in `static/`
- **JavaScript Modules**: 
  - `api-client.js` - Backend API communication
  - `charts.js` - Plotly.js chart management
  - `realtime-updates.js` - Polling-based updates
  - `dashboard.js` - Dashboard-specific logic
  - `data-upload.js` - File upload handling with drag-and-drop
  - `data-quality.js` - Quality metrics visualization
  - `data-history.js` - Upload history management

### Styling
- **Theme**: Dark professional theme with CSS variables
- **Colors**: Near-black backgrounds (#0a0a0b), Bitcoin orange accent (#f7931a)
- **Typography**: Inter font family with consistent sizing
- **Files**: `static/css/theme.css`, `static/css/components.css`, `static/css/flask-overrides.css`

### Migration Notes
- Migrated from Streamlit to Flask to resolve HTML rendering issues
- No more column nesting limitations
- Full control over HTML/CSS/JS
- Better suited for production deployment
- Port configuration: Flask runs on 8502 (similar to Streamlit's 8501)

### Flask Frontend Key Features
- **WebSocket Support**: Real-time updates via Flask-SocketIO
- **Modular Architecture**: Blueprints for dashboard, analytics, settings, API, and websocket
- **Production Ready**: Gunicorn with sync workers (eventlet caused DNS issues)
- **Responsive Design**: Mobile-friendly with CSS Grid and Flexbox
- **Dark Theme**: Consistent with trading system design (Bitcoin orange #f7931a accent)
- **JavaScript Modules**:
  - `api-client.js`: Centralized API communication
  - `websocket-client.js`: Real-time WebSocket connection management
  - `charts.js`: Plotly.js chart management with dark theme
  - `realtime-updates.js`: Fallback polling when WebSocket unavailable
- **Security**: CORS enabled, session management, CSRF protection ready

### Paper Trading Integration (Added 2025-01-11)
- **Dashboard Integration**: Radio toggle in sidebar to switch between Live/Paper trading modes
- **Visual Indicators**: 
  - Paper badge on portfolio values when in paper mode
  - Order button text changes to "Execute Paper Order"
  - Trading mode info shows initial capital and status
- **API Endpoints**:
  - `/api/paper-trading/status` - Get current status
  - `/api/paper-trading/toggle` - Enable/disable paper trading
  - `/api/paper-trading/trade` - Execute paper trades
- **Seamless Experience**: Same charts, inputs, and workflows for both modes
- **Performance Tracking**: Win rate, Sharpe ratio, P&L tracked separately for paper trades

### Important Flask Implementation Details
- Always use port 8502 for consistency across development and production
- WebSocket automatically falls back to polling if connection fails
- All API calls go through the centralized API client for consistency
- CSS files are properly linked (not using `<link>` tags like in Streamlit)
- No column nesting limitations - use CSS Grid/Flexbox freely
- Form submissions use proper POST requests with JSON payloads
- Real-time updates subscribe to specific event types for efficiency

## Database Architecture

The system uses three SQLite databases:

1. **api_cache.db** - Caches API responses to reduce external calls
   - Tables: api_cache, cache_invalidation_log, cache_stats
   
2. **trading_system.db** - Core trading data and analytics
   - Tables: trades, positions, trading_limits, model_signals, backtest_results, signal_performance, feature_importance, market_regime, paper_portfolio, paper_trades, paper_performance
   - Key relationships: trades/positions via lot_id, paper_trades/paper_portfolio via portfolio_id
   
3. **historical_data.db** - Historical market data for analysis
   - Tables: ohlcv_data, onchain_data, sentiment_data

For detailed database schema and ERD, see: ./docs/database-erd.md

## Docker Deployment

### Docker Configuration
- **Backend Port**: 8090 (maps internal 8000)
- **Frontend Port**: 8502
- **Compose File**: `docker/docker-compose.yml`
- **Deployment Script**: `init_deploy.sh`
- **Network**: Both containers use `trading-network` for internal communication
- **API Communication**: Frontend calls backend via `http://backend:8000` internally

### Known Issues and Fixes

#### 1. Frontend Container Volume Mount Error
**Issue**: Frontend container fails with "read-only file system" error when mounting logs directory.

**Root Cause**: The Dockerfile switches to non-root user before Docker can create mount points for volumes.

**Fix Applied**:
1. In `src/frontend_flask/Dockerfile`, create directories before switching user:
   ```dockerfile
   # Create directories for volumes before switching to non-root user
   RUN mkdir -p /app/logs /app/config
   ```

2. In `docker/docker-compose.yml`, remove source code mount that conflicts:
   ```yaml
   volumes:
     - ../storage/logs/frontend:/app/logs
     - ../storage/config:/app/config
     # Removed: - ../src/frontend_flask:/app:ro
   ```

3. Add `curl` to Dockerfile for health checks:
   ```dockerfile
   RUN apt-get update && apt-get install -y \
       gcc \
       g++ \
       curl \
       && rm -rf /var/lib/apt/lists/*
   ```

4. Update `requirements.txt` with missing dependencies:
   ```
   numpy==1.24.3
   pandas==2.0.3
   ```

#### 2. Frontend DNS Resolution Issue
**Issue**: Frontend returns dummy data, eventlet worker causes DNS resolution timeout for Docker service names.

**Root Cause**: Eventlet's monkey patching interferes with DNS resolution in Docker networks.

**Fix Applied**:
1. Changed from eventlet to sync workers in `src/frontend_flask/Dockerfile`:
   ```dockerfile
   CMD ["gunicorn", "--bind", "0.0.0.0:8502", "--workers", "4", "--timeout", "120", "app:create_app()"]
   ```

2. Updated `docker/docker-compose.yml` to use correct backend URL:
   ```yaml
   environment:
     - API_BASE_URL=http://backend:8000
   ```

#### 3. Pricing Data Display Issue (Fixed 2025-01-11)
**Issue**: Frontend displays incorrect price ($95,000) instead of real-time price (~$117,500).

**Root Cause**: Backend `/price/current` endpoint returned hardcoded fallback value from `data_fetcher.py` instead of real-time price from enhanced data fetcher.

**Fix Applied**:
1. Updated `/price/current` endpoint in `src/backend/api/main.py` to prioritize real-time price:
   ```python
   # First try to get real-time price from enhanced data fetcher
   current_price = get_current_btc_price()
   ```

2. Only use fallback price when real-time price is unavailable and sanity check > $100,000.

#### 4. Chart Data Not Updating (Fixed 2025-01-11)
**Issue**: Pricing chart doesn't reflect actual data or update when changing time periods.

**Root Cause**: Multiple issues:
- Backend expects `days` parameter but frontend sends `timeframe`
- Backend returns `price` field but frontend expects `close` field
- Missing timeframe to days conversion

**Fix Applied**:
1. Added timeframe conversion in `src/frontend_flask/blueprints/dashboard.py`:
   ```python
   timeframe_to_days = {
       '1H': 2,    # 2 days for hourly data
       '4H': 7,    # 7 days for 4-hour data
       '1D': 30,   # 30 days for daily data
       '1W': 180,  # 180 days for weekly data
       '1M': 365   # 365 days for monthly data
   }
   ```

2. Fixed field mapping to check both `close` and `price` fields:
   ```python
   'close': float(item.get('close', item.get('price', 0)))
   ```

3. Removed broken fallback to `/btc/latest` endpoint.

## Data Management System (Added 2025-01-11)

### Overview
Comprehensive data quality and management tools for uploading, monitoring, and managing historical data that feeds into LSTM models for signal generation.

### Data Upload Features
- **File Support**: CSV and XLSX bulk upload with drag-and-drop
- **Data Types Supported**:
  - **OHLCV Data**: Full candlestick data requiring all fields: timestamp, open, high, low, close, volume (Fixed 2025-01-12)
  - **On-Chain Data**: Blockchain metrics (active addresses, transaction count, hash rate, etc.)
  - **Sentiment Data**: Fear/greed index, Reddit/Twitter sentiment, news sentiment
  - **Macro Data**: Economic indicators (DXY, bond yields, commodities)
- **Preview**: View data before upload with column detection
- **Validation**: Automatic type checking and data validation
- **Sample Templates**: Download CSV templates for each data type
- **Custom Sources**: Accept any custom source name for data uploads (Fixed 2025-01-12)

### Data Quality Dashboard
- **Completeness Metrics**: Track data coverage by type and symbol
- **Gap Detection**: Identify missing time periods in data
- **Coverage Heatmap**: Visual representation of data availability
- **Source Quality**: Rate data sources by reliability and completeness
- **Cache Performance**: Monitor API cache hit rates and efficiency

### Upload History Management
- **Upload Tracking**: Complete history of all data uploads
- **Filtering**: Search by data type, symbol, date range
- **Rollback**: Ability to undo uploads if needed
- **Upload Details**: View metadata and statistics for each upload

### Technical Implementation
- **Backend Integration**: Uses existing `DataUploadService` and `HistoricalDataManager`
- **Database Storage**: Data stored in `historical_data.db` with tables for OHLCV, on-chain, and sentiment data
- **Conflict Resolution**: Multiple strategies (REPLACE, MERGE, FILL_GAPS, AVERAGE)
- **API Endpoints**:
  - `POST /data/api/upload` - Upload data files
  - `POST /data/api/preview` - Preview file before upload
  - `GET /data/api/quality-metrics` - Get quality metrics
  - `GET /data/api/upload-history` - Get upload history
  - `GET /data/api/sample-formats` - Get sample data formats
  - `GET /data/api/download-template/<type>` - Download CSV templates

### Data Upload Field Mapping
The frontend automatically maps common field names to expected backend fields:
- Frontend `data_type` values: `ohlcv`, `onchain`, `sentiment`, `macro`
- Backend `data_type` values: `price`, `onchain`, `sentiment`, `macro`
- File type detection from extension (csv/xlsx)

### Navigation
Added dropdown menu in main navigation:
```
Data Management ▼
├── Data Upload
├── Data Quality
└── Upload History
```

### Deployment Commands
```bash
# Stop containers
docker compose -f docker/docker-compose.yml down

# Rebuild with no cache
docker compose -f docker/docker-compose.yml build --no-cache frontend

# Start services
docker compose -f docker/docker-compose.yml up -d

# Check health status
docker ps | grep btc-trading
```

#### 5. Data Upload Source Restriction (Fixed 2025-01-12)
**Issue**: Data upload was restricted to hardcoded list of sources, preventing custom data source names.

**Root Cause**: Backend validated against a fixed `VALID_SOURCES` list and frontend defaulted to "manual_upload" which wasn't in the valid list.

**Fix Applied**:
1. Updated backend `DataUploadService` to accept any source name:
   - Renamed `VALID_SOURCES` to `SUGGESTED_SOURCES`
   - Removed source validation check
   - API now returns `custom_allowed: true` flag

2. Updated frontend to guide users about custom sources:
   - Added helpful text explaining custom sources are allowed
   - Updated placeholder to show custom source examples
   - Listed suggested sources for reference

3. Updated API documentation to reflect new behavior

#### 6. OHLCV Data Upload Requirements (Fixed 2025-01-12)
**Issue**: Data upload only expected price and volume fields instead of full OHLCV dataset.

**Root Cause**: DataUploadService treated OHLCV as simple "price" data with optional OHLCV fields, storing in generic JSON table instead of proper ohlcv_data table.

**Fix Applied**:
1. Updated DataUploadService validation to require all OHLCV fields:
   - Required fields: `timestamp`, `open`, `high`, `low`, `close`, `volume`
   - Changed from storing in generic historical_data table to proper ohlcv_data table
   
2. Modified storage method to use dedicated OHLCV table:
   - Creates proper table schema with individual columns
   - Ensures data integrity and efficient querying
   
3. Enhanced statistics generation for OHLCV-specific metrics

#### 7. Mock BTC Price Data Issue (Fixed 2025-07-16)
**Issue**: Frontend displays incorrect historical BTC price ($31,961) instead of current market price (~$119,000).

**Root Cause**: Multiple cascading failures:
- All external APIs (Binance, CoinGecko, CryptoCompare) failing due to rate limits or geographical restrictions
- System falling back to old historical data from database
- Historical data contained outdated price of $31,961

**Fix Applied**:
1. Enhanced `get_current_btc_price()` fallback logic in `src/backend/api/main.py`:
   - Only use historical data if less than 24 hours old AND price > $50,000
   - Added direct CoinGecko API call as additional fallback
   - Updated last resort fallback to realistic market price ($119,000)

2. Improved `enhanced_data_fetcher.py` error handling:
   - Better logging for API failures
   - Added direct requests fallback for CoinGecko
   - Increased timeout from 5s to 10s for better reliability

3. Better error visibility:
   - Added detailed logging to track which price source is being used
   - Log warnings when falling back to historical or default prices

**Result**: Frontend now displays accurate BTC price even when primary APIs fail

#### 8. Data Upload Zero Results Display Issue (Fixed 2025-07-16)
**Issue**: Frontend data upload tool shows "zero results successfully loaded" even when data is uploaded successfully.

**Root Cause**: Flask blueprint wasn't properly checking if backend returned success=true before displaying success message. It only checked HTTP status code (200).

**Fix Applied**:
1. Updated Flask blueprint `/data/api/upload` handler in `src/frontend_flask/blueprints/data.py`:
   - Added proper validation to check backend `success` field
   - Added handling for both boolean and integer success values (true, 1, "1")
   - Returns error response if backend returns 200 but success=false

2. Enhanced logging:
   ```python
   backend_success = result.get('success')
   if backend_success is True or backend_success == 1 or backend_success == "1":
       rows_inserted = result.get('rows_inserted', 0)
       logger.info(f"Backend response: success={backend_success} (type: {type(backend_success).__name__}), rows_inserted={rows_inserted}")
   ```

3. Proper error handling when backend reports failure despite 200 status

**Result**: Frontend now correctly displays "Successfully uploaded X records" matching actual upload results

## Data Deletion Feature (Added 2025-07-17)

### Overview
Added comprehensive data deletion capabilities to the Data Management system, allowing users to safely remove historical data from the databases with multiple safeguards.

### Backend API Endpoints
- **GET /data/available** - Lists all available data grouped by type, source, and date range
  - Returns data from ohlcv_data, onchain_data, and sentiment_data tables
  - Groups by symbol and source with record counts
  
- **DELETE /data/delete** - Deletes data with specified criteria
  - Parameters: data_type, source, symbol, start_date, end_date, confirm
  - Requires confirm=true to prevent accidental deletion
  - Returns deleted record count and details
  
- **GET /data/stats/{data_type}** - Shows statistics for specific data before deletion
  - Returns record count, date range, and type-specific stats (e.g., price stats for OHLCV)

### Frontend Implementation
- **New Route**: `/data/manage` - Data Management page
- **New Template**: `templates/data/manage.html`
- **New JavaScript**: `static/js/data-manage.js`
- **Flask Blueprint Updates**: Added proxy endpoints in `blueprints/data.py`

### UI Features
1. **Available Data Table**
   - Shows all data groups with type, source, symbol, date range, and record count
   - Stats button to view detailed statistics
   - Delete button for each data group

2. **Statistics Modal**
   - Displays detailed information before deletion
   - OHLCV data shows average/min/max prices and total volume
   - General stats show record count and date range

3. **Delete Confirmation Modal**
   - Red warning header and alert
   - Shows deletion details in table format
   - Requires typing "DELETE" to enable confirmation button
   - Safety measure prevents accidental deletion

4. **Advanced Deletion Options**
   - Filter by data type, source, symbol
   - Optional date range filtering
   - Preview deletion before confirming

### Safety Features
- Multiple confirmation steps required
- Backend requires explicit confirm=true parameter
- Type "DELETE" confirmation in UI
- Clear warnings about permanent deletion
- All deletions logged in backend for audit trail
- No bulk delete all - must specify criteria

### Navigation
Added to Data Management dropdown menu:
```
Data Management ▼
├── Data Upload
├── Data Quality
├── Upload History
├── ─────────────── (divider)
└── Manage Data
```

### Technical Details
- Uses parameterized SQL queries to prevent injection
- Transactions ensure data integrity
- Returns verification of deleted record count
- Frontend uses Bootstrap modals for confirmations
- Real-time data refresh after deletion