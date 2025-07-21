"""Gunicorn configuration file for BTC Trading Flask Frontend"""
import multiprocessing
import os

# Server socket
bind = "0.0.0.0:8502"
backlog = 2048

# Worker processes
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "gthread"
threads = 2
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50

# Timeout settings
timeout = 120
graceful_timeout = 30
keepalive = 5

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = "btc-trading-frontend"

# Server mechanics
daemon = False
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None

# SSL/TLS
keyfile = None
certfile = None

# Error handling - ignore client disconnections
def worker_int(worker):
    """Handle worker interrupt gracefully"""
    worker.log.info("Worker received INT or QUIT signal")

def when_ready(server):
    """Server is ready. Notify systemd if applicable."""
    server.log.info("Server is ready. Spawning workers")

def worker_abort(worker):
    """Handle worker abort - log but don't crash"""
    worker.log.info("Worker received SIGABRT, restarting...")

def pre_request(worker, req):
    """Log request start"""
    worker.log.debug("%s %s", req.method, req.path)

def post_request(worker, req, environ, resp):
    """Log request completion"""
    worker.log.debug("%s %s %s", req.method, req.path, resp.status)

# Handle broken pipe errors gracefully
def handle_error(request, client_address):
    """Handle client errors gracefully"""
    import errno
    import sys
    exc_info = sys.exc_info()
    if exc_info[0] is OSError and exc_info[1].errno in (errno.EPIPE, errno.ECONNRESET):
        # Client disconnected, ignore
        return
    # Re-raise other errors
    raise exc_info[1].with_traceback(exc_info[2])