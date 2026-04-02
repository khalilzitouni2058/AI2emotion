from uuid import uuid4

from fastapi import APIRouter, UploadFile, File, Form, Depends

from app.presentation.api.dependencies import get_emotion_service
from app.presentation.api.schemas.responses import ApiResponse
from app.presentation.api.utils.serializers import to_serializable
from app.presentation.api.utils.upload_manager import (
    save_upload,
    validate_audio_file,
    cleanup_file,
)
from app.services.emotion_service import EmotionService

router = APIRouter(prefix="/api/v1/analyze", tags=["analysis"])


@router.post("/audio", response_model=ApiResponse)
async def analyze_audio(
    file: UploadFile = File(...),
    source_name: str | None = Form(default=None),
    service: EmotionService = Depends(get_emotion_service),
) -> ApiResponse:
    """
    Analyze emotion from uploaded audio file.
    """
    request_id = str(uuid4())

    validate_audio_file(file)

    file_path = save_upload(file)

    try:
        result = service.analyze(audio_path=str(file_path))
        result_dict = to_serializable(result)

        result_dict["metadata"]["source_name"] = source_name or file.filename

        return ApiResponse(
            success=True,
            request_id=request_id,
            data=result_dict,
            error=None,
        )

    finally:
        cleanup_file(file_path)