from uuid import uuid4

from fastapi import APIRouter, Depends

from app.presentation.api.dependencies import get_emotion_service
from app.presentation.api.schemas.requests import TextAnalysisRequest
from app.presentation.api.schemas.responses import ApiResponse
from app.presentation.api.utils.serializers import to_serializable
from app.services.emotion_service import EmotionService

router = APIRouter(prefix="/api/v1/analyze", tags=["analysis"])


@router.post("/text", response_model=ApiResponse)
def analyze_text(
    payload: TextAnalysisRequest,
    service: EmotionService = Depends(get_emotion_service),
) -> ApiResponse:
    """
    Analyze emotion from text input.
    """
    request_id = str(uuid4())

    result = service.analyze(text=payload.text)
    result_dict = to_serializable(result)

    if payload.source_name:
        result_dict["metadata"]["source_name"] = payload.source_name

    return ApiResponse(
        success=True,
        request_id=request_id,
        data=result_dict,
        error=None,
    )