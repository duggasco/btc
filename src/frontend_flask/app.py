"""BTC Trading System - Flask Frontend Application"""
import os
import sys
from flask import Flask, render_template
from flask_cors import CORS
from flask_socketio import SocketIO
from datetime import datetime
import logging

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import blueprints
from blueprints.dashboard import dashboard_bp
from blueprints.analytics import analytics_bp
from blueprints.settings import settings_bp
from blueprints.api import api_bp
from blueprints.data import data_bp
from blueprints.websocket import init_socketio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global SocketIO instance
socketio = SocketIO(cors_allowed_origins="*", async_mode='threading')

def create_app():
    """Create and configure the Flask application"""
    app = Flask(__name__)
    
    # Configuration
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    app.config['API_BASE_URL'] = os.environ.get('API_BASE_URL', 'http://localhost:8000')
    
    # Enable CORS
    CORS(app)
    
    # Initialize SocketIO
    socketio.init_app(app)
    
    # Initialize WebSocket handlers and get emit functions
    websocket_emitters = init_socketio(socketio)
    
    # Store emitters in app config for use in other blueprints
    app.config['WEBSOCKET_EMITTERS'] = websocket_emitters
    
    # Register blueprints
    app.register_blueprint(dashboard_bp, url_prefix='/')
    app.register_blueprint(analytics_bp, url_prefix='/analytics')
    app.register_blueprint(settings_bp, url_prefix='/settings')
    app.register_blueprint(data_bp, url_prefix='/data')
    app.register_blueprint(api_bp, url_prefix='/api')
    
    # Error handlers
    @app.errorhandler(404)
    def not_found_error(error):
        return render_template('404.html'), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return render_template('500.html'), 500
    
    @app.errorhandler(OSError)
    def handle_oserror(error):
        """Handle OS errors like broken pipes gracefully"""
        import errno
        if error.errno in (errno.EPIPE, errno.ECONNRESET):
            # Client disconnected, log and return empty response
            logger.debug("Client disconnected: %s", error)
            return '', 200
        # Re-raise other OS errors
        raise error
    
    @app.before_request
    def before_request():
        """Setup request context"""
        from flask import g
        g.start_time = datetime.now()
    
    @app.after_request
    def after_request(response):
        """Add headers to prevent connection issues"""
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Connection'] = 'keep-alive'
        return response
    
    # Context processors
    @app.context_processor
    def inject_globals():
        return {
            'now': datetime.now(),
            'app_name': 'BTC Trading System',
            'version': '4.0',
            'websocket_enabled': True
        }
    
    return app

if __name__ == '__main__':
    app = create_app()
    # Use SocketIO run method instead of Flask's run
    socketio.run(app, host='0.0.0.0', port=8502, debug=True)