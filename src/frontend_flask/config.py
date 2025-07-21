"""Flask Frontend Configuration"""
import os
from datetime import timedelta

class Config:
    """Base configuration"""
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    API_BASE_URL = os.environ.get('API_BASE_URL', 'http://localhost:8090')
    
    # Session configuration
    SESSION_COOKIE_NAME = 'btc_trading_session'
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    PERMANENT_SESSION_LIFETIME = timedelta(hours=24)
    
    # WebSocket configuration
    WEBSOCKET_URL = os.environ.get('WEBSOCKET_URL', 'ws://localhost:8090/ws')
    
    # UI Configuration
    ITEMS_PER_PAGE = 20
    AUTO_REFRESH_INTERVAL = 5000  # milliseconds
    
    # Chart configuration
    CHART_THEME = 'dark'
    DEFAULT_TIMEFRAME = '1D'
    
    # Features
    ENABLE_WEBSOCKET = True
    ENABLE_PAPER_TRADING = True
    ENABLE_REAL_TRADING = False

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False
    SESSION_COOKIE_SECURE = True

class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = True
    TESTING = True
    WTF_CSRF_ENABLED = False

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}