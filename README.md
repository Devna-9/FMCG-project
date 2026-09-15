# FMCG Dashboard

A Render-ready Dash dashboard built from the supplied FMCG campaign, transaction, lookup and feedback files.

## Files required in the GitHub repository

- `app.py`
- `requirements.txt`
- `render.yaml`
- `runtime.txt`
- `Campaign Response  Data.csv`
- `Campaign Details.csv`
- `MasterLookUp.csv`
- `Transaction data.csv`
- `74responses.txt`

The application builds the partner-level master table at startup, so a separate `Final_Master_Data.csv` is not required.

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

Then open `http://127.0.0.1:8050`.

## Render

Create a new Web Service from the GitHub repository. Render can use the included `render.yaml`, or use:

Build Command:
`pip install -r requirements.txt`

Start Command:
`gunicorn app:server`

The data files must be committed to the repository because the app reads them at startup.
