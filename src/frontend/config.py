"""
Frontend configuration settings
"""
import os

# API Settings
API_BASE_URL = os.getenv("API_BASE_URL", "http://backend:8000")
API_TIMEOUT = int(os.getenv("API_TIMEOUT", "30"))
API_RETRY_ATTEMPTS = int(os.getenv("API_RETRY_ATTEMPTS", "3"))

# WebSocket Settings
WS_BASE_URL = os.getenv("WS_BASE_URL", "ws://backend:8000/ws")
WS_RECONNECT_INTERVAL = int(os.getenv("WS_RECONNECT_INTERVAL", "5"))

# Cache Settings
CACHE_TTL = int(os.getenv("CACHE_TTL", "60"))  # seconds
CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", "1000"))

# Rate Limiting
RATE_LIMIT_MAX = int(os.getenv("RATE_LIMIT_MAX", "100"))
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))  # seconds

# UI Settings
DEFAULT_REFRESH_INTERVAL = int(os.getenv("DEFAULT_REFRESH_INTERVAL", "5"))  # seconds
PAPER_TRADING_INITIAL_BALANCE = float(os.getenv("PAPER_TRADING_INITIAL_BALANCE", "10000"))

# External Links
GITHUB_REPO_URL = os.getenv("GITHUB_REPO_URL", "https://github.com/duggasco/btc")
DOCUMENTATION_URL = os.getenv("DOCUMENTATION_URL", "https://github.com/duggasco/btc/wiki")

# Development Settings
DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() == "true"