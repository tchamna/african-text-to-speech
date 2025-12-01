# gunicorn.conf.py
# Conservative Gunicorn config for running the app in constrained hosts (e.g., Azure App Service)
# Adjust values according to your plan. Keep workers small to avoid multiple large model copies.

workers = 1
worker_class = 'sync'
threads = 2
timeout = 120
keepalive = 5
preload_app = False

# Optional: tune request/headers sizes to suit your environment
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190

# Logging (use Gunicorn defaults unless you override via env)
accesslog = '-'  # stdout
errorlog = '-'   # stderr
loglevel = 'info'
