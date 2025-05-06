import os

# Timeout configuration
timeout = 120  # Increased for .keras model
graceful_timeout = 120
keepalive = 5

# Worker configuration
workers = 1  # Single worker for 512 MB RAM
worker_class = 'sync'
worker_connections = 1000

# Logging
accesslog = '-'
errorlog = '-'
loglevel = 'info'

# Prevent memory leaks
max_requests = 500  # Frequent restarts for .keras model
max_requests_jitter = 50

# Preload app to share model memory
preload_app = True

# Limits
limit_request_line = 4096
limit_request_fields = 100
limit_request_field_size = 8190