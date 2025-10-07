FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# Copiamos primero el pyproject para cachear deps
COPY pyproject.toml /app/
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -e .

# Ahora el código
COPY src /app/src

# Entrena un mini-modelo al construir
RUN python -m gameradar.train

ENV ARTIFACTS_DIR=/app/artifacts
EXPOSE 8000

CMD ["uvicorn", "gameradar.api:app", "--host", "0.0.0.0", "--port", "8000"]
