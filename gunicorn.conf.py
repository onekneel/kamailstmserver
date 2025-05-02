import os

# Timeout configuration
timeout = 60  # Reduced to 60 seconds; sufficient for frame processing
graceful_timeout = 60
keepalive = 5

# Worker configuration
workers = 1  # Use 1 worker to stay within 512 MB RAM
worker_class = 'sync'  # Keep sync for CPU-bound tasks
worker_connections = 1000  # Ignored for sync workers, but harmless

# Logging
accesslog = '-'
errorlog = '-'
loglevel = 'info'  # Switch to info to reduce logging overhead

# Prevent memory leaks
max_requests = 1000
max_requests_jitter = 50

# Limits
limit_request_line = 4096
limit_request_fields = 100
limit_request_field_size = 8190