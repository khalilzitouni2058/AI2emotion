"""Domain schemas representing core data structures."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class EmotionSegment:
    """Represents a single analyzed segment of audio."""

    chunk_id: int
    start_time: float
    end_time: float
    timestamp: float
    duration: float
    emotion: str
    confidence: float
    probabilities: Dict[str, float]

    # Optional raw values (for smoothing comparison)
    emotion_raw: Optional[str] = None
    confidence_raw: Optional[float] = None
    probabilities_raw: Optional[Dict[str, float]] = None


@dataclass
class EmotionTransition:
    """Represents a transition between two emotions."""

    transition_id: int
    from_emotion: str
    to_emotion: str
    transition_time: float
    previous_segment: int
    current_segment: int


@dataclass
class AnalysisMetadata:
    """Metadata about the analysis process."""

    total_duration: float
    total_chunks: int
    window_size: float
    hop_size: float
    sampling_rate: int
    smoothing_method: Optional[str]
    analysis_timestamp: str

    # Optional fields
    audio_file: Optional[str] = None
    processing_mode: Optional[str] = None
    sub_window_size: Optional[float] = None
    sub_hop_size: Optional[float] = None


@dataclass
class AnalysisSummary:
    """Summary statistics of the analysis."""

    total_transitions: int
    emotion_distribution: Dict[str, int]


@dataclass
class EmotionAnalysisResult:
    """Top-level result object."""

    metadata: AnalysisMetadata
    emotion_segments: List[EmotionSegment] = field(default_factory=list)
    emotion_transitions: List[EmotionTransition] = field(default_factory=list)
    summary: AnalysisSummary = None