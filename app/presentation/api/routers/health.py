from fastapi import APIRouter, Depends

from app.presentation.api.dependencies import get_emotion_service
from app.services.emotion_service import EmotionService

router = APIRouter(prefix="/api/v1/health", tags=["health"])


@router.get("")
def health() -> dict:
    """
    Basic liveness check.
    """
    return {
        "status": "ok",
        "message": "API is running",
    }


@router.get("/ready")
def ready(
    service: EmotionService = Depends(get_emotion_service),
) -> dict:
    """
    Readiness check. Forces model resources to be available.
    """
    service.model_provider.get_resources()
    service.text_service.text_model_provider.get_resources()

    return {
        "status": "ready",
        "message": "API is ready to serve requests",
    }