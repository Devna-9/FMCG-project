from app import app

# Dedicated WSGI callable for Gunicorn/Render.
application = app.server
