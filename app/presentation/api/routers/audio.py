from uuid import uuid4
from pathlib import Path
import time
from typing import Any

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from app.presentation.api.dependencies import get_emotion_service
from app.presentation.api.utils.serializers import to_serializable
from app.presentation.api.utils.upload_manager import (
    save_upload,
    validate_audio_file,
    cleanup_file,
)
from app.infrastructure.models import EmotionAnalysis
from app.services.metahuman_animator import MetahumanAnimator

router = APIRouter(prefix="/api/v1/analyze", tags=["analysis"])


def _log_background(message: str) -> None:
    print(f"[AudioBackground] {message}", flush=True)


def run_audio_analysis_sync(
    request_id: str,
    file_path: str,
    source_name: str,
    profile: str,
) -> dict[str, Any]:
    """Run analysis synchronously and return final response payload."""
    _log_background(f"request_id={request_id} sync mode started | file={Path(file_path).name}")

    try:
        service = get_emotion_service()
        stage_state = {"current_stage": None, "stage_start": None}
        analysis_started = time.perf_counter()
        stage_history: list[dict[str, Any]] = []
        chunk_history: list[dict[str, Any]] = []

        def on_progress(
            chunk_index: int,
            total_chunks: int,
            start_time: float,
            end_time: float,
            emotion: str,
            confidence: float,
        ) -> None:
            if chunk_index <= 0 or total_chunks <= 0:
                return

            progress = min(100, int((chunk_index / total_chunks) * 100))
            if len(chunk_history) < 200:
                chunk_history.append(
                    {
                        "chunk_index": int(chunk_index),
                        "total_chunks": int(total_chunks),
                        "start_time": round(float(start_time), 3),
                        "end_time": round(float(end_time), 3),
                        "emotion": str(emotion),
                        "confidence": round(float(confidence), 3),
                        "progress": progress,
                    }
                )

            _log_background(
                f"request_id={request_id} chunk={chunk_index}/{total_chunks} "
                f"range={start_time:.2f}-{end_time:.2f}s emotion={emotion} "
                f"confidence={confidence:.3f} progress={progress}%"
            )

        def on_stage_update(message: str) -> None:
            now = time.perf_counter()
            current_stage = stage_state["current_stage"]
            stage_start = stage_state["stage_start"]

            if current_stage is not None and stage_start is not None:
                elapsed = now - stage_start
                stage_history.append(
                    {
                        "name": current_stage,
                        "elapsed_seconds": round(elapsed, 3),
                    }
                )
                _log_background(
                    f"request_id={request_id} stage_complete={current_stage} elapsed={elapsed:.2f}s"
                )

            stage_state["current_stage"] = message
            stage_state["stage_start"] = now
            _log_background(f"request_id={request_id} stage={message}")

        if profile == "fast":
            result = service.analyze_audio_fast(
                audio_path=file_path,
                progress_callback=on_progress,
                stage_callback=on_stage_update,
            )
        else:
            result = service.analyze(
                audio_path=file_path,
                progress_callback=on_progress,
                stage_callback=on_stage_update,
            )

        if stage_state["current_stage"] is not None and stage_state["stage_start"] is not None:
            final_elapsed = time.perf_counter() - stage_state["stage_start"]
            stage_history.append(
                {
                    "name": str(stage_state["current_stage"]),
                    "elapsed_seconds": round(final_elapsed, 3),
                }
            )
            _log_background(
                f"request_id={request_id} stage_complete={stage_state['current_stage']} elapsed={final_elapsed:.2f}s"
            )

        result_dict = to_serializable(result)

        if "metadata" not in result_dict or result_dict["metadata"] is None:
            result_dict["metadata"] = {}

        result_dict["metadata"]["source_name"] = source_name
        result_dict["workflow"] = {
            "profile": profile,
            "total_elapsed_seconds": round(time.perf_counter() - analysis_started, 3),
            "stages": stage_history,
            "chunks_sample": chunk_history,
        }

        # Include UE5-ready animation payload built from segments and transitions.
        result_dict["metahuman_animation"] = MetahumanAnimator.to_ue5_metahuman_format(result)

        _log_background(f"request_id={request_id} sync mode completed")
        return {
            "status": "done",
            "success": True,
            "request_id": request_id,
            "data": result_dict,
            "error": None,
            "progress": 100,
            "stage": "completed",
        }
    except Exception as exc:
        _log_background(f"request_id={request_id} sync mode failed | error={exc}")
        return {
            "status": "failed",
            "success": False,
            "request_id": request_id,
            "data": None,
            "error": str(exc),
            "progress": 0,
            "stage": "failed",
        }
    finally:
        cleanup_file(Path(file_path))
        _log_background(f"request_id={request_id} sync mode upload cleaned")


def _save_analysis_to_db(
    filename: str,
    request_id: str,
    result_payload: dict[str, Any],
    profile: str,
) -> None:
    """Save the exact API response payload to database."""
    try:
        # Import here to avoid circular imports
        from app.infrastructure.database import SessionLocal

        db = SessionLocal()
        try:
            if result_payload.get("success") and result_payload.get("data"):
                processing_time_ms = (
                    result_payload.get("data", {})
                    .get("workflow", {})
                    .get("total_elapsed_seconds", 0)
                    * 1000
                )

                # Persist each successful analysis as its own row.
                unique_filename = f"{filename}__{request_id}"
                analysis = EmotionAnalysis(
                    audio_filename=unique_filename,
                    analysis_json=result_payload,
                    status=str(result_payload.get("status", "completed")),
                    error_message=result_payload.get("error"),
                    processing_time_ms=processing_time_ms,
                )
                db.add(analysis)

                db.commit()
                _log_background(f"Saved analysis for {filename} to database")
        finally:
            db.close()
    except Exception as e:
        _log_background(f"Failed to save analysis to database: {e}")


@router.get("/result/{request_id}")
async def get_analysis_result(request_id: str):
    """Backward-compatible endpoint for clients that poll analysis by request_id."""
    from app.infrastructure.database import SessionLocal

    db = SessionLocal()
    try:
        rows = (
            db.query(EmotionAnalysis)
            .order_by(EmotionAnalysis.created_at.desc())
            .limit(200)
            .all()
        )

        for row in rows:
            payload = row.analysis_json if isinstance(row.analysis_json, dict) else None
            if payload and payload.get("request_id") == request_id:
                return JSONResponse(content=payload)

        raise HTTPException(status_code=404, detail="Result not found")
    finally:
        db.close()


@router.get("/stored/by-source")
async def get_saved_result_by_source(source_name: str):
    """Return the latest saved payload for a given metadata.source_name."""
    from app.infrastructure.database import SessionLocal

    target = source_name.strip()
    if not target:
        raise HTTPException(status_code=400, detail="source_name is required")

    db = SessionLocal()
    try:
        rows = (
            db.query(EmotionAnalysis)
            .order_by(EmotionAnalysis.created_at.desc())
            .limit(1000)
            .all()
        )

        for row in rows:
            payload = row.analysis_json if isinstance(row.analysis_json, dict) else None
            if not payload:
                continue

            payload_source_name = (
                payload.get("data", {})
                .get("metadata", {})
                .get("source_name")
            )

            if isinstance(payload_source_name, str) and payload_source_name.strip() == target:
                return JSONResponse(content=payload)

        raise HTTPException(status_code=404, detail=f"No saved result found for source_name '{target}'")
    finally:
        db.close()


@router.post("/audio")
async def analyze_audio(
    file: UploadFile = File(...),
    source_name: str | None = Form(default=None),
    profile: str = Form(default="fast"),
):
    request_id = str(uuid4())

    validate_audio_file(file)
    file_path = save_upload(file)

    clean_source_name = source_name.strip() if source_name else file.filename
    selected_profile = profile.strip().lower() if profile else "fast"
    if selected_profile not in {"fast", "balanced", "main"}:
        selected_profile = "fast"

    # main profile aliases to the same full-path analyzer used by balanced.
    if selected_profile == "main":
        selected_profile = "balanced"

    result_payload = run_audio_analysis_sync(
        request_id=request_id,
        file_path=str(file_path),
        source_name=clean_source_name,
        profile=selected_profile,
    )

    # Save to database automatically after analysis completes
    if result_payload.get("success"):
        _save_analysis_to_db(
            filename=clean_source_name,
            request_id=request_id,
            result_payload=result_payload,
            profile=selected_profile,
        )

    return JSONResponse(
        content=result_payload,
        headers={
            "Connection": "close"
        }
    )
