# Multi-platform Docker image - serves the dashboard and the API in one container.
#
# Supported deployment targets:
#   Render.com  : PORT is set automatically by Render (defaults to 8000 here)
#   Hugging Face: set PORT=7860 in Space env vars; app_port: 7860 in README front matter
#   Local dev   : docker run --rm -p 8000:8000 prognostix-ml
#                 open http://localhost:8000
#
# For the local development stack (separate API + nginx containers, port 8000/8080)
# use docker/docker-compose.yml instead.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libgomp1 is required by scikit-learn and torch; curl backs the health check.
RUN apt-get update \
 && apt-get install --no-install-recommends -y libgomp1 curl \
 && rm -rf /var/lib/apt/lists/*

# Spaces runs the container as UID 1000; own every path we write to.
RUN useradd --create-home --uid 1000 user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    MPLCONFIGDIR=/home/user/.cache/matplotlib

WORKDIR /home/user/app

# Dependencies first, so code changes do not invalidate this layer.
# The default torch wheels carry ~2 GB of CUDA libraries that a CPU Space cannot use.
COPY --chown=user requirements.txt ./
RUN pip install --user --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

COPY --chown=user configs/ ./configs/
COPY --chown=user src/ ./src/
COPY --chown=user api/ ./api/
COPY --chown=user monitoring/ ./monitoring/
COPY --chown=user scripts/ ./scripts/
COPY --chown=user frontend/ ./frontend/
COPY --chown=user data/ ./data/
COPY --chown=user artifacts/ ./artifacts/

ENV PYTHONPATH=/home/user/app \
    PROGNOSTIX_CONFIG=/home/user/app/configs/config.yaml \
    # Default port - Render.com overrides this automatically via its own PORT env var;
    # for Hugging Face Spaces set PORT=7860 in the Space environment variables.
    PORT=8000

# Trained models are not tracked in git (artifacts/models/ is ignored), so the
# image builds its own from the committed raw CMAPSS data. prepare_data.py also
# writes artifacts/scaler.joblib, which the API needs to standardise incoming
# sensor readings - without it every prediction would return 503.
RUN python scripts/prepare_data.py

# Skipped when a checkpoint was copied in from the build context, so local
# rebuilds stay fast while a fresh clone (the Spaces case) trains one.
RUN if ls artifacts/models/*.pt >/dev/null 2>&1; then \
        echo "Using the checkpoint already present in artifacts/models/"; \
    else \
        python scripts/train.py --model gru --no-plots; \
    fi

EXPOSE 8000

# /health answers 200 even with no model loaded (reporting "degraded"), which is
# the right liveness signal: the process is up and can describe its own problem.
# Shell form used so $PORT is expanded at container start time.
HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
    CMD curl -fsS http://localhost:${PORT:-8000}/health || exit 1

# Shell form (not exec form) so the $PORT variable is substituted at runtime.
CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}
