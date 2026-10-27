# ABA Graph Studio: Vite frontend built once, served by the FastAPI app that also answers /api.
#
# The mined data is NOT copied in. Most of ABA_mining is not in git (outputs/task3_llm_pairs
# alone is several GB), so an image carrying it could not be rebuilt from a clone. It is mounted
# at /data instead and found through ABA_MINING_ROOT -- see docker-compose.yml.

# --- stage 1: build the frontend -------------------------------------------------------------
FROM node:22-alpine AS frontend

WORKDIR /build
# package files first: this layer is cached until the dependencies themselves change.
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci --no-audit --no-fund || npm install --no-audit --no-fund

COPY frontend/ ./
RUN npm run build


# --- stage 2: the server ----------------------------------------------------------------------
FROM python:3.13-slim AS runtime

# PYTHONDONTWRITEBYTECODE: the source is read-only in the image, .pyc files would only add size.
# PYTHONUNBUFFERED: uvicorn logs appear in `docker logs` immediately rather than on flush.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ABA_MINING_ROOT=/data

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY --from=frontend /build/dist ./frontend/dist

# .cache must exist AND be owned before the named volume is attached: Docker seeds a fresh
# volume from the image path, so a directory created later at run time would belong to root and
# the unprivileged user could not write its derived files.
RUN mkdir -p /app/.cache \
    && useradd --create-home --uid 10001 aba \
    && chown -R aba /app
USER aba

EXPOSE 8000

# The app serves the SPA from frontend/dist and the API under /api, so one port is enough.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/api/hotels').read()"

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
