"""Domain schemas representing core data structures."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class EmotionSegment:
    """Represents a single analyzed unit of input."""

    chunk_id: int
    start_time: float
    end_time: float
    timestamp: float
    duration: float
    emotion: str
    confidence: float
    probabilities: Dict[str, float]

    # Optional raw values (used when smoothing/comparing predictions)
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

    input_type: str  # "audio" or "text"
    analysis_timestamp: str

    # Generic optional metadata
    source_name: Optional[str] = None
    processing_mode: Optional[str] = None
    smoothing_method: Optional[str] = None

    # Audio-specific optional metadata
    total_duration: Optional[float] = None
    total_chunks: Optional[int] = None
    window_size: Optional[float] = None
    hop_size: Optional[float] = None
    sampling_rate: Optional[int] = None
    sub_window_size: Optional[float] = None
    sub_hop_size: Optional[float] = None

    # Text-specific optional metadata
    text_length: Optional[int] = None


@dataclass
class AnalysisSummary:
    """Summary statistics of the analysis."""

    total_transitions: int
    emotion_distribution: Dict[str, int]


@dataclass
class EmotionAnalysisResult:
    """Top-level result object returned by the application."""

    metadata: AnalysisMetadata
    emotion_segments: List[EmotionSegment] = field(default_factory=list)
    emotion_transitions: List[EmotionTransition] = field(default_factory=list)
    summary: Optional[AnalysisSummary] = None