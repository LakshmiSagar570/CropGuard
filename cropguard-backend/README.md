# CropGuard Backend

## Run locally

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:GEMINI_API_KEY="YOUR_KEY"
uvicorn app:app --reload
```

## Endpoints
- `GET /health`
- `POST /diagnose` (multipart form with `file`, `crop`, `district`, `season`, `language`)