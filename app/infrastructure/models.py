"""Database models for emotion analysis results."""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, Float
from sqlalchemy.dialects.sqlite import JSON
from app.infrastructure.database import Base


class EmotionAnalysis(Base):
    """Model for storing emotion analysis results."""
    
    __tablename__ = "emotion_analyses"
    
    id = Column(Integer, primary_key=True, index=True)
    audio_filename = Column(String(255), unique=True, index=True, nullable=False)
    analysis_json = Column(JSON, nullable=False)  # Full analysis results
    status = Column(String(50), default="completed", index=True)  # "processing", "completed", "failed"
    error_message = Column(Text, nullable=True)  # Error details if failed
    processing_time_ms = Column(Float, nullable=True)  # How long analysis took
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<EmotionAnalysis(filename={self.audio_filename}, status={self.status})>"


class AnalysisJob(Base):
    """Model for queued background analysis jobs and live progress tracking."""

    __tablename__ = "analysis_jobs"

    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(String(64), unique=True, index=True, nullable=False)
    audio_filename = Column(String(255), index=True, nullable=False)
    file_path = Column(String(512), nullable=False)
    profile = Column(String(50), default="fast", index=True)
    status = Column(String(50), default="queued", index=True)
    stage = Column(String(255), default="queued", nullable=False)
    progress = Column(Float, default=0.0)
    total_chunks = Column(Integer, nullable=True)
    current_chunk = Column(Integer, nullable=True)
    error_message = Column(Text, nullable=True)
    processing_time_ms = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<AnalysisJob(request_id={self.request_id}, status={self.status})>"
