FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY models ./models

ENV MODEL_PATH=models/house_price.pkl
ENV LOCATIONS_PATH=models/locations.json
ENV CORS_ORIGINS=http://localhost:5173

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
