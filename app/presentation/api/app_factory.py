from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from app.presentation.api.exception_handlers import register_exception_handlers
from app.presentation.api.routers import health, audio
from app.infrastructure.database import init_db
from app.presentation.api.dependencies import get_emotion_service


def create_app() -> FastAPI:
    app = FastAPI(
        title="AI2Emotion API",
        version="1.0.0",
        description="Backend API for audio emotion analysis.",
    )

    app.include_router(health.router)
    app.include_router(audio.router)

    # Mount built Vite frontend if available, otherwise fall back to legacy static UI.
    project_root = Path(__file__).parent.parent.parent.parent
    frontend_dist = project_root / "frontend" / "dist"
    legacy_static_dir = project_root / "static"

    if frontend_dist.exists():
        app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")
    elif legacy_static_dir.exists():
        app.mount("/", StaticFiles(directory=str(legacy_static_dir), html=True), name="static")

    register_exception_handlers(app)

    @app.on_event("startup")
    def warmup_models() -> None:
        print("[Startup] Initializing database...", flush=True)
        init_db()
        print("[Startup] Warming up emotion model...", flush=True)
        get_emotion_service().model_provider.get_resources()
        print("[Startup] API ready (direct inference mode)", flush=True)

    return app


app = create_app()