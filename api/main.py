"""FastAPI application factory and ASGI entry point.

Serve with::

    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

The operations dashboard is served at ``/``, the API under ``/api/v1``, service
metadata at ``/api/info`` and interactive documentation at ``/docs``. The default
model is loaded during startup so the first request does not pay for it; anything
else loads lazily.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from api.routes import router
from api.schemas import ErrorResponse
from src import __version__
from src.config import PROJECT_ROOT, get_config, setup_logging
from src.inference.predictor import ModelRegistry

logger = logging.getLogger(__name__)

__all__ = ["app", "create_app"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, release them on shutdown.

    A missing checkpoint is logged, not fatal: the process still starts, ``/health``
    reports ``degraded`` and ``/models`` shows what is actually available. That is
    the difference between a container that reports its problem and one that
    crash-loops.
    """
    config = get_config()
    app.state.started_at = time.time()
    app.state.registry = ModelRegistry(config)

    loaded = app.state.registry.preload()
    available = app.state.registry.available()
    if loaded:
        logger.info("Preloaded model(s): %s", ", ".join(loaded))
    elif available:
        logger.warning(
            "No model preloaded; %s available for lazy loading", ", ".join(available)
        )
    else:
        logger.error(
            "No trained model found in %s - /predict will return 503 until one is "
            "trained (python scripts/train.py --model gru)",
            config.path("models"),
        )

    yield

    app.state.registry = None
    logger.info("Shutdown complete")


def create_app() -> FastAPI:
    """Build the FastAPI application."""
    setup_logging()
    config = get_config()
    settings = config.api

    app = FastAPI(
        title=str(settings.title),
        description=str(settings.description),
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.get("cors_origins", ["*"])),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    prefix = str(settings.get("prefix", "")).rstrip("/")
    app.include_router(router, prefix=prefix)
    if prefix:
        # Unprefixed aliases keep container health checks and scrapers simple.
        app.include_router(router, include_in_schema=False)

    _register_error_handlers(app)

    @app.get("/api/info", tags=["service"])
    def service_info() -> dict[str, object]:
        """Service metadata and the paths worth knowing."""
        return {
            "service": str(settings.title),
            "version": __version__,
            "docs": "/docs",
            "dashboard": "/",
            "endpoints": {
                "info": "/api/info",
                "health": f"{prefix}/health",
                "models": f"{prefix}/models",
                "predict": f"{prefix}/predict",
                "predict_batch": f"{prefix}/predict/batch",
                "fleet": f"{prefix}/fleet",
                "leaderboard": f"{prefix}/leaderboard",
                "drift": f"{prefix}/monitoring/drift",
                "metrics": f"{prefix}/metrics",
            },
        }

    # The dashboard is served from the application root, so a single container
    # answers both the UI and the API and the browser needs no CORS exception.
    # Mounted last: routes registered above (/api/v1/*, /docs, /health, ...) are
    # matched first, and the mount only catches what is left.
    #
    # Resolved against the project root rather than the process working
    # directory - StaticFiles raises at startup if the directory is missing, and
    # a relative path would make that depend on where uvicorn was launched.
    dashboard = PROJECT_ROOT / "frontend"
    if dashboard.is_dir():
        app.mount("/", StaticFiles(directory=dashboard, html=True), name="frontend")
    else:
        logger.warning("No frontend directory at %s; serving the API only", dashboard)

    return app


def _register_error_handlers(app: FastAPI) -> None:
    """Return a consistent :class:`ErrorResponse` body for every failure mode."""

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        detail = "; ".join(
            f"{'.'.join(str(part) for part in error['loc'][1:])}: {error['msg']}"
            for error in exc.errors()
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ErrorResponse(
                detail=detail or "Request validation failed",
                error_type="validation_error",
            ).model_dump(mode="json"),
        )

    @app.exception_handler(FileNotFoundError)
    async def missing_artifact(request: Request, exc: FileNotFoundError):
        logger.warning("Missing artifact: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=ErrorResponse(
                detail=str(exc), error_type="artifact_missing"
            ).model_dump(mode="json"),
        )

    @app.exception_handler(ValueError)
    async def invalid_value(request: Request, exc: ValueError):
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ErrorResponse(detail=str(exc), error_type="value_error").model_dump(
                mode="json"
            ),
        )

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                detail="Internal server error", error_type=type(exc).__name__
            ).model_dump(mode="json"),
        )


app = create_app()


if __name__ == "__main__":  # pragma: no cover - manual entry point
    import uvicorn

    settings = get_config().api
    uvicorn.run(
        "api.main:app",
        host=str(settings.get("host", "0.0.0.0")),
        port=int(settings.get("port", 8000)),
        reload=True,
    )
