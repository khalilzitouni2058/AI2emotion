from functools import lru_cache

from app.services.emotion_service import EmotionService


@lru_cache
def get_emotion_service() -> EmotionService:
    """
    Return a shared EmotionService instance for API requests.
    """
    return EmotionService()