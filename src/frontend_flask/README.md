# BTC Trading System - Flask Frontend

This is the Flask-based web frontend for the BTC Trading System, replacing the previous Streamlit implementation.

## Features

- **Trading Dashboard**: Real-time price monitoring, portfolio tracking, and order execution
- **Analytics & Research**: Backtesting, optimization, and performance analysis
- **Settings & Configuration**: System settings and API configuration
- **Real-time Updates**: Polling-based updates (WebSocket support planned)
- **Dark Professional Theme**: Consistent with the trading system design

## Technology Stack

- Flask 2.3.3
- Plotly.js for interactive charts
- Vanilla JavaScript for frontend logic
- Gunicorn with eventlet for production deployment

## Project Structure

```
src/frontend_flask/
├── app.py                 # Main Flask application
├── config.py             # Configuration settings
├── requirements.txt      # Python dependencies
├── Dockerfile           # Docker configuration
├── blueprints/          # Flask blueprints
│   ├── dashboard.py     # Trading dashboard routes
│   ├── analytics.py     # Analytics routes
│   ├── settings.py      # Settings routes
│   └── api.py          # Internal API routes
├── templates/           # HTML templates
│   ├── base.html       # Base template
│   └── dashboard/      # Dashboard templates
├── static/             # Static assets
│   ├── css/           # Stylesheets
│   └── js/            # JavaScript files
└── utils/             # Utility modules
```

## Development Setup

1. Install dependencies:
```bash
cd src/frontend_flask
pip install -r requirements.txt
```

2. Set environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. Run development server:
```bash
python app.py
```

## Docker Deployment

The frontend is configured to run in Docker as part of the main system:

```bash
cd docker
docker-compose up frontend
```

## API Integration

The frontend communicates with the FastAPI backend through:
- REST API endpoints for data fetching
- Planned WebSocket support for real-time updates

## Key Differences from Streamlit

1. **Full HTML/CSS/JS Control**: No more raw HTML rendering issues
2. **Better Performance**: Direct control over rendering and updates
3. **Scalability**: Can handle more concurrent users
4. **Customization**: Complete control over UI components
5. **Production Ready**: Better suited for production deployment

## Environment Variables

- `SECRET_KEY`: Flask secret key for sessions
- `API_BASE_URL`: Backend API URL
- `FLASK_ENV`: Environment (development/production)
- `ENABLE_WEBSOCKET`: Enable WebSocket support (future)
- `ENABLE_PAPER_TRADING`: Enable paper trading features
- `ENABLE_REAL_TRADING`: Enable real trading features

## Future Enhancements

- WebSocket integration for real-time updates
- Advanced charting features
- Mobile-responsive design improvements
- Additional analytics visualizations