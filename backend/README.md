# Kerb Backend

FastAPI service that serves predictions from the real-data-trained
`house_price.pkl`. See the root `README.md` for full project docs,
architecture, and API reference.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8000
```

## Tests

```bash
pytest tests/ -v
```

## Structure

```
app/
├── main.py                    # FastAPI app, CORS, lifespan-based model loading
├── api/routes/prediction.py   # GET /health, POST /predict
├── core/config.py             # pydantic-settings configuration
├── schemas/prediction.py      # request/response models
├── services/
│   ├── preprocessing.py       # builds the exact feature row the model expects
│   └── inference.py           # loads + wraps the trained pipeline
└── utils/logging_config.py
```
