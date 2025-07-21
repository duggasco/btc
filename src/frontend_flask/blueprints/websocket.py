"""WebSocket Blueprint for real-time updates"""
from flask import Blueprint, request
from flask_socketio import emit, join_room, leave_room
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)

# Create blueprint (note: WebSocket routes are handled differently)
websocket_bp = Blueprint('websocket', __name__)

def init_socketio(socketio):
    """Initialize SocketIO event handlers"""
    
    @socketio.on('connect')
    def handle_connect():
        """Handle client connection"""
        logger.info(f"Client connected: {request.sid}")
        join_room('updates')
        emit('connected', {'status': 'Connected to real-time updates'})
    
    @socketio.on('disconnect')
    def handle_disconnect():
        """Handle client disconnection"""
        logger.info(f"Client disconnected: {request.sid}")
        leave_room('updates')
    
    @socketio.on('subscribe')
    def handle_subscribe(data):
        """Handle subscription to specific update types"""
        update_types = data.get('types', [])
        for update_type in update_types:
            join_room(f"updates:{update_type}")
        emit('subscribed', {'types': update_types})
    
    @socketio.on('unsubscribe')
    def handle_unsubscribe(data):
        """Handle unsubscription from update types"""
        update_types = data.get('types', [])
        for update_type in update_types:
            leave_room(f"updates:{update_type}")
        emit('unsubscribed', {'types': update_types})
    
    # Server-side emit functions to be called from other parts of the app
    def emit_price_update(price_data):
        """Emit price update to all connected clients"""
        socketio.emit('price_update', {
            'timestamp': datetime.now().isoformat(),
            'data': price_data
        }, room='updates:price')
    
    def emit_signal_update(signal_data):
        """Emit trading signal update"""
        socketio.emit('signal_update', {
            'timestamp': datetime.now().isoformat(),
            'data': signal_data
        }, room='updates:signals')
    
    def emit_portfolio_update(portfolio_data):
        """Emit portfolio update"""
        socketio.emit('portfolio_update', {
            'timestamp': datetime.now().isoformat(),
            'data': portfolio_data
        }, room='updates:portfolio')
    
    def emit_trade_update(trade_data):
        """Emit trade execution update"""
        socketio.emit('trade_update', {
            'timestamp': datetime.now().isoformat(),
            'data': trade_data
        }, room='updates:trades')
    
    def emit_system_update(system_data):
        """Emit system status update"""
        socketio.emit('system_update', {
            'timestamp': datetime.now().isoformat(),
            'data': system_data
        }, room='updates:system')
    
    # Return emit functions for use in other modules
    return {
        'emit_price_update': emit_price_update,
        'emit_signal_update': emit_signal_update,
        'emit_portfolio_update': emit_portfolio_update,
        'emit_trade_update': emit_trade_update,
        'emit_system_update': emit_system_update
    }