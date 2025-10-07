FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# Copia metadatos + README (evita avisos)
COPY pyproject.toml README.md /app/

# Copia el código ANTES del editable install
COPY src /app/src

# Instala deps y el paquete en editable
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

# Entrena el modelo durante la build
RUN python -m gameradar.train

ENV ARTIFACTS_DIR=/app/artifacts
EXPOSE 8000

CMD ["uvicorn", "gameradar.api:app", "--host", "0.0.0.0", "--port", "8000"]
