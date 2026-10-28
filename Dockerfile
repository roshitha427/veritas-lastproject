FROM node:22-bookworm-slim AS frontend
WORKDIR /web
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim-bookworm AS runtime
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HOST=0.0.0.0 \
    PORT=5000 \
    DATA_DIR=/var/data \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2
WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app \
    && mkdir -p /var/data \
    && chown app:app /var/data
COPY requirements*.txt ./
ARG INSTALL_INFERENCE=true
RUN if [ "$INSTALL_INFERENCE" = "true" ]; then \
      pip install -r requirements-inference.txt; \
    elif [ "$INSTALL_INFERENCE" = "false" ]; then \
      pip install -r requirements.txt; \
    else echo 'INSTALL_INFERENCE must be true or false' >&2; exit 1; fi
COPY --chown=app:app backend/ backend/
COPY --chown=app:app ml/ ml/
COPY --chown=app:app scripts/ scripts/
COPY --chown=app:app models/ models/
COPY --from=frontend --chown=app:app /web/dist/ frontend/dist/
# Use a public, non-secret artifact URL; never place signed URLs or keys in ARGs.
ARG MODEL_BUNDLE_URL
ARG MODEL_BUNDLE_SHA256
RUN python -m scripts.install_models --if-configured --destination /app/models \
    && chown -R app:app /app/models
EXPOSE 5000
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
  CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.getenv('PORT','5000')+'/api/health',timeout=8)"
# The entrypoint fixes a fresh disk's ownership before dropping to the app user.
ENTRYPOINT ["python", "-m", "scripts.startup"]
CMD ["python", "-m", "scripts.serve"]
