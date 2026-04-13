from functools import lru_cache

from app.services.emotion_service import EmotionService
from app.infrastructure.database import get_db as _get_db


@lru_cache
def get_emotion_service() -> EmotionService:
    """
    Return a shared EmotionService instance for API requests.
    """
    return EmotionService()


def get_db():
    """
    Database session dependency for FastAPI routes.
    """
    return _get_db()