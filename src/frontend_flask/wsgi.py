"""WSGI entry point for production deployment"""
import eventlet
# Monkey patch early, before any imports
eventlet.monkey_patch()

from app import create_app, socketio

app = create_app()

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=8502)