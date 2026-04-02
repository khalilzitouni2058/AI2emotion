from fastapi import FastAPI

from app.presentation.api.exception_handlers import register_exception_handlers
from app.presentation.api.routers import health, text,audio


def create_app() -> FastAPI:
    app = FastAPI(
        title="AI2Emotion API",
        version="1.0.0",
        description="Backend API for audio and text emotion analysis.",
    )

    app.include_router(health.router)
    app.include_router(text.router)
    app.include_router(audio.router)

    register_exception_handlers(app)

    return app


app = create_app()