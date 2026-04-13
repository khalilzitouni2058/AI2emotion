from pathlib import Path
from uuid import uuid4
import shutil

from fastapi import UploadFile


UPLOAD_DIR = Path("data/uploads")
UPLOAD_COPY_BUFFER_BYTES = 1024 * 1024


def save_upload(file: UploadFile) -> Path:
    """
    Save uploaded file to data storage.
    """
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    filename = f"{uuid4()}_{file.filename}"
    file_path = UPLOAD_DIR / filename

    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer, length=UPLOAD_COPY_BUFFER_BYTES)

    return file_path


def validate_audio_file(file: UploadFile) -> None:
    """
    Validate uploaded audio file.
    """
    if not file.filename:
        raise ValueError("No file provided.")

    if not file.filename.lower().endswith(".wav"):
        raise ValueError("Only WAV files are supported.")


def cleanup_file(file_path: Path) -> None:
    """
    Delete temporary file.
    """
    try:
        if file_path.exists():
            file_path.unlink()
    except Exception:
        pass