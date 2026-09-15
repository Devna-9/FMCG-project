# FMCG Dashboard - Render Deployment

Files in this folder are intended to be uploaded to the root of a GitHub repository.

## Render
- Runtime: Python
- Python version: 3.11.10
- Build: `pip install --upgrade pip && pip install -r requirements.txt`
- Start: `gunicorn app:server --workers 1 --threads 4 --timeout 120`

The app builds its master dataset from the supplied CSV files at startup.
The CSV loader validates headers and skips malformed rows with a warning instead of
crashing the entire deployment.

Important: replace the old repository files with this package and trigger a **Clear
build cache & deploy** in Render. If the log still shows Python 3.14, the service is
not using the new repository configuration; set `PYTHON_VERSION=3.11.10` in Render's
Environment settings and redeploy.
