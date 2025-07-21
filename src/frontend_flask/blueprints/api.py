"""API Blueprint for internal Flask API endpoints"""
from flask import Blueprint, jsonify
import logging

logger = logging.getLogger(__name__)

api_bp = Blueprint('api', __name__)

@api_bp.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'flask-frontend'
    })

@api_bp.route('/config')
def get_config():
    """Get frontend configuration"""
    return jsonify({
        'api_base_url': '/api',
        'websocket_url': '/ws',
        'auto_refresh_interval': 5000,
        'features': {
            'websocket': True,
            'paper_trading': True,
            'real_trading': False
        }
    })