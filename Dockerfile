# Hugging Face Spaces image - one container serving the dashboard and the API.
#
# Spaces requirements this satisfies: a root-level Dockerfile, a non-root user
# with UID 1000, and the app listening on port 7860 (declared as `app_port` in
# the README front matter).
#
# Build and run locally exactly as the Space does:
#   docker build -t prognostix-space .
#   docker run --rm -p 7860:7860 prognostix-space
#   open http://localhost:7860
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
    PROGNOSTIX_CONFIG=/home/user/app/configs/config.yaml

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

EXPOSE 7860

# /health answers 200 even with no model loaded (reporting "degraded"), which is
# the right liveness signal: the process is up and can describe its own problem.
HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
    CMD curl -fsS http://localhost:7860/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
