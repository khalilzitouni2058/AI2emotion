"""Application configuration for the emotion analysis system."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Central application settings."""

    # Model settings
    model_id: str = "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3"

    # Audio processing settings
    audio_duration_threshold: float = 15.0
    audio_sampling_rate: int = 16000
    chunk_window_size: float = 1.9
    chunk_hop_size: float = 0.5

    # Inner sliding-window settings
    sub_window_size: float = 0.625
    sub_hop_size: float = 0.31

    # Full-audio preprocessing
    max_audio_duration: float = 30.0

    # Output / display
    print_width: int = 60

    # Smoothing mode selection
    smoothing_mode: str = "offline"  # options: offline, streaming
    offline_smoothing_method: str = "interpolate"  # options: interpolate, median, none
    streaming_smoothing_method: str = "ema"  # options: ema, none
    streaming_use_hysteresis: bool = False

    # Legacy / fallback smoothing parameters
    smoothing_method: str = "combined"
    smoothing_window_size: int = 3
    smoothing_ema_alpha: float = 0.3
    smoothing_confidence_threshold: float = 0.7
    smoothing_min_frames: int = 2


settings = Settings()