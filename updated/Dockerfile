# ─── Stage: runtime ───────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Системні залежності (потрібні для torch/transformers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Встановлення Python-залежностей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копіюємо код API
COPY api.py .

# Копіюємо збережені моделі
COPY saved_models_4_3_bert/ saved_models_4_3_bert/

EXPOSE 8000

ENV MODEL_PATH=saved_models_4_3_bert/bert_absa

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
