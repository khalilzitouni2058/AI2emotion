"""Web platform routes for emotion analysis database."""

from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.infrastructure.database import SessionLocal, get_db
from app.infrastructure.file_storage import FileStorage
from app.infrastructure.models import AnalysisJob, EmotionAnalysis

router = APIRouter(prefix="/api/v1/platform", tags=["emotion-platform"])

file_storage = FileStorage()
progress_store: dict[str, dict] = {}


def _log(message: str) -> None:
    print(f"[Platform] {message}", flush=True)


def _stage_progress(message: str) -> int | None:
    text = message.lower()
    if "preparing audio" in text:
        return 5
    if "audio loaded" in text:
        return 15
    if "building chunk timeline" in text:
        return 20
    if "chunk windows ready" in text:
        return 20
    if "using batched chunk inference" in text:
        return 25
    if "running chunk batch" in text:
        return 30
    if "smoothing" in text:
        return 90
    if "summary" in text:
        return 95
    return None


def _set_progress(request_id: str, **updates) -> None:
    current = progress_store.get(request_id, {})
    current.update(updates)
    progress_store[request_id] = current


def _job_to_progress(job: AnalysisJob) -> dict:
    return {
        "request_id": job.request_id,
        "filename": job.audio_filename,
        "status": job.status,
        "stage": job.stage,
        "progress": job.progress,
        "total_chunks": job.total_chunks,
        "current_chunk": job.current_chunk,
        "error": job.error_message,
        "completed": job.status in {"completed", "failed"},
        "processing_time_ms": job.processing_time_ms,
        "created_at": job.created_at.isoformat(),
    }


def _job_to_progress_update(job: AnalysisJob) -> dict:
    payload = _job_to_progress(job)
    payload.pop("request_id", None)
    return payload


@router.post("/upload-and-analyze")
async def upload_and_analyze(
    file: UploadFile = File(...),
    profile: str = "fast",
    request_id: str | None = Form(default=None),
    db: Session = Depends(get_db),
):
    """Save the upload, enqueue a job, and return immediately."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    request_id = request_id or str(uuid4())
    audio_filename = file.filename
    _log(f"request_id={request_id} received file={audio_filename} mode=async")

    _set_progress(
        request_id,
        filename=audio_filename,
        status="processing",
        stage="starting",
        progress=0,
        total_chunks=0,
        current_chunk=0,
        error=None,
        completed=False,
    )

    existing = db.query(EmotionAnalysis).filter(
        EmotionAnalysis.audio_filename == audio_filename,
    ).first()

    if existing:
        _log(f"request_id={request_id} file already analyzed, returning cached result")
        _set_progress(
            request_id,
            status="completed",
            stage="already_exists",
            progress=100,
            completed=True,
        )
        return {
            "request_id": request_id,
            "status": "already_exists",
            "filename": audio_filename,
            "analysis": existing.analysis_json,
            "created_at": existing.created_at.isoformat(),
        }

    queued_job = db.query(AnalysisJob).filter(
        AnalysisJob.audio_filename == audio_filename,
        AnalysisJob.status.in_(["queued", "processing"]),
    ).order_by(AnalysisJob.created_at.desc()).first()

    if queued_job:
        _log(f"request_id={request_id} file already queued, returning existing job")
        _set_progress(queued_job.request_id, **_job_to_progress_update(queued_job))
        return {
            "request_id": queued_job.request_id,
            "status": queued_job.status,
            "filename": queued_job.audio_filename,
            "message": "Analysis already queued",
        }

    try:
        file_path = file_storage.save_uploaded_file(
            file,
            saved_name=f"{request_id}_{audio_filename}",
        )
        _log(f"request_id={request_id} file saved at {file_path}")

        job = AnalysisJob(
            request_id=request_id,
            audio_filename=audio_filename,
            file_path=file_path,
            profile=profile,
            status="queued",
            stage="queued for worker",
            progress=1,
            total_chunks=0,
            current_chunk=0,
        )
        db.add(job)
        db.commit()
        db.refresh(job)

        _set_progress(request_id, **_job_to_progress_update(job))

        return {
            "request_id": request_id,
            "status": "queued",
            "filename": audio_filename,
            "message": "Analysis started in background",
        }

    except Exception as exc:
        _log(f"request_id={request_id} failed to enqueue: {str(exc)}")
        _set_progress(
            request_id,
            status="failed",
            stage="failed to enqueue",
            error=str(exc),
            completed=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to enqueue analysis: {str(exc)}",
        )


@router.get("/progress/{request_id}")
async def get_progress(request_id: str):
    db = SessionLocal()
    try:
        job = db.query(AnalysisJob).filter(AnalysisJob.request_id == request_id).first()
        if job:
            return _job_to_progress(job)

        progress = progress_store.get(request_id)
        if not progress:
            raise HTTPException(status_code=404, detail=f"No progress found for '{request_id}'")
        return progress
    finally:
        db.close()


@router.get("/result/{filename}")
async def get_result(filename: str, db: Session = Depends(get_db)):
    """Retrieve analysis results for an audio file by filename."""
    analysis = db.query(EmotionAnalysis).filter(
        EmotionAnalysis.audio_filename == filename,
    ).first()

    if not analysis:
        raise HTTPException(status_code=404, detail=f"No analysis found for '{filename}'")

    return {
        "filename": filename,
        "status": analysis.status,
        "analysis": analysis.analysis_json,
        "processing_time_ms": analysis.processing_time_ms,
        "created_at": analysis.created_at.isoformat(),
        "error": analysis.error_message if analysis.status == "failed" else None,
    }


@router.get("/result/{filename}/json")
async def get_result_json(filename: str, db: Session = Depends(get_db)):
    """Retrieve analysis results as raw JSON (for plugin consumption)."""
    analysis = db.query(EmotionAnalysis).filter(
        EmotionAnalysis.audio_filename == filename,
    ).first()

    if not analysis:
        raise HTTPException(status_code=404, detail=f"No analysis found for '{filename}'")

    if analysis.status != "completed":
        raise HTTPException(status_code=202, detail=f"Analysis status: {analysis.status}")

    return analysis.analysis_json


@router.get("/list")
async def list_analyses(
    limit: int = 50,
    offset: int = 0,
    status_filter: str = None,
    db: Session = Depends(get_db),
):
    query = db.query(EmotionAnalysis)

    if status_filter:
        query = query.filter(EmotionAnalysis.status == status_filter)

    total = query.count()
    analyses = query.order_by(EmotionAnalysis.created_at.desc()).offset(offset).limit(limit).all()

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "items": [
            {
                "filename": a.audio_filename,
                "status": a.status,
                "processing_time_ms": a.processing_time_ms,
                "created_at": a.created_at.isoformat(),
            }
            for a in analyses
        ],
    }


@router.delete("/result/{filename}")
async def delete_result(filename: str, db: Session = Depends(get_db)):
    analysis = db.query(EmotionAnalysis).filter(
        EmotionAnalysis.audio_filename == filename,
    ).first()

    if not analysis:
        raise HTTPException(status_code=404, detail=f"No analysis found for '{filename}'")

    db.delete(analysis)
    db.commit()

    return {
        "status": "deleted",
        "filename": filename,
    }
