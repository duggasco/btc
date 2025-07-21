# Flask Migration Guide

## Overview

This guide documents the migration from Streamlit to Flask for the BTC Trading System frontend.

## Why Migrate to Flask?

### Issues with Streamlit
1. **HTML Rendering Problems**: Raw HTML/CSS was being displayed instead of rendered
2. **Column Nesting Limitations**: Maximum 2-level nesting caused layout constraints
3. **Limited Control**: Difficult to implement custom JavaScript functionality
4. **Performance**: Not optimized for production trading applications

### Benefits of Flask
1. **Full Control**: Complete control over HTML, CSS, and JavaScript
2. **Production Ready**: Better suited for production deployment
3. **Scalability**: Can handle more concurrent users
4. **Flexibility**: Easy integration with WebSockets and real-time features
5. **Performance**: Faster response times and lower resource usage

## Migration Steps

### Phase 1: Setup (Completed)
- Created Flask application structure
- Set up blueprints for modular routing
- Configured Docker deployment
- Migrated CSS assets

### Phase 2: Core Functionality (Completed)
- Implemented base HTML template with navigation
- Created Trading Dashboard with all features
- Built JavaScript modules for API communication
- Set up real-time updates via polling

### Phase 3: Remaining Pages (Pending)
- Analytics & Research page full implementation
- Settings & Configuration page full implementation
- WebSocket integration for real-time updates

### Phase 4: Cleanup (Pending)
- Remove old Streamlit code
- Update all documentation
- Remove Streamlit dependencies from requirements

## Architecture Changes

### Old Structure (Streamlit)
```
src/frontend/
├── app.py              # Main Streamlit app
├── pages/              # Streamlit pages
├── components/         # UI components
├── styles/            # CSS files
└── utils/             # Utilities
```

### New Structure (Flask)
```
src/frontend_flask/
├── app.py             # Flask application
├── blueprints/        # Route handlers
├── templates/         # HTML templates
├── static/           # CSS, JS, images
└── utils/            # Utilities
```

## Key Implementation Details

### 1. HTML Templates
- Using Jinja2 templating engine
- Base template for consistent layout
- Modular component structure

### 2. JavaScript Architecture
- `api-client.js`: Centralized API communication
- `charts.js`: Plotly.js chart management
- `realtime-updates.js`: Polling-based updates
- `dashboard.js`: Page-specific functionality

### 3. CSS Organization
- Maintained existing dark theme
- Added Flask-specific overrides
- Responsive grid layout

### 4. API Integration
- RESTful endpoints for data fetching
- Mock data for development
- Ready for WebSocket upgrade

## Deployment Changes

### Docker Configuration
```yaml
# Old (Streamlit)
ports:
  - "8501:8501"
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]

# New (Flask)
ports:
  - "8502:8502"
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8502/api/health"]
```

### Environment Variables
- `FLASK_ENV`: Set to 'production' for deployment
- `SECRET_KEY`: Required for session management
- `API_BASE_URL`: Backend API endpoint

## Development Workflow

1. **Local Development**:
   ```bash
   cd src/frontend_flask
   pip install -r requirements.txt
   python app.py
   ```

2. **Docker Development**:
   ```bash
   cd docker
   docker-compose up frontend
   ```

3. **Production Deployment**:
   - Uses Gunicorn with eventlet workers
   - Configured for high concurrency
   - Health checks enabled

## Testing

Run the test script to verify the migration:
```bash
python test_flask_frontend.py
```

## Future Enhancements

1. **WebSocket Integration**
   - Real-time price updates
   - Live trading signals
   - Portfolio updates

2. **Advanced Features**
   - Multi-chart layouts
   - Advanced order types
   - Custom indicators

3. **Performance Optimizations**
   - CDN for static assets
   - Response caching
   - Database connection pooling

## Troubleshooting

### Common Issues

1. **CSS Not Loading**
   - Ensure static files are properly mapped in Docker
   - Check Flask static file configuration

2. **API Connection Errors**
   - Verify API_BASE_URL environment variable
   - Check network connectivity between containers

3. **Chart Display Issues**
   - Ensure Plotly.js is loaded
   - Check browser console for JavaScript errors

## Rollback Plan

If issues arise, the old Streamlit frontend is preserved in `src/frontend/`. To rollback:
1. Update docker-compose.yml to use old frontend configuration
2. Rebuild and restart containers

## Conclusion

The Flask migration provides a more robust, scalable, and maintainable frontend for the BTC Trading System. With full control over the UI and better production capabilities, the system is now ready for serious trading operations.