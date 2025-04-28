import os

# Timeout configuration
timeout = int(os.environ.get('WORKER_TIMEOUT', 120))
graceful_timeout = timeout
keepalive = 5

# Worker configuration
workers = 2
worker_class = 'sync'
worker_connections = 1000

# Logging
accesslog = '-'
errorlog = '-'
loglevel = 'info'

# Prevent memory leaks
max_requests = 1000
max_requests_jitter = 50

# Limits
limit_request_line = 4096
limit_request_fields = 100
limit_request_field_size = 8190
