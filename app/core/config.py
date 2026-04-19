"""Application configuration for the emotion analysis system."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Central application settings."""

    # Audio model settings
    audio_model_id: str = "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3"

    # Text model settings (multilingual, supports Arabic)
    text_model_id: str = "AnasAlokla/multilingual_go_emotions_V1.2"
    max_text_length: int = 512

    # Audio processing settings
    audio_duration_threshold: float = 30.0  # Use chunked path for files > 30s
    audio_sampling_rate: int = 16000
    chunk_window_size: float = 2.5
    chunk_hop_size: float = 1.5
    chunk_inference_batch_size: int = 4

    # Inner sliding-window settings
    sub_window_size: float = 2.5
    sub_hop_size: float = 2.5

    # Full-audio preprocessing
    max_audio_duration: float = 30.0  # Whisper's hard limit per inference

    # Output / display
    print_width: int = 60

    # Smoothing mode selection
    smoothing_mode: str = "none"  # Disable smoothing for fast analysis
    offline_smoothing_method: str = "none"  # options: interpolate, median, none
    streaming_smoothing_method: str = "ema"  # options: ema, none
    streaming_use_hysteresis: bool = False

    # Legacy / fallback smoothing parameters
    smoothing_method: str = "none"  # Disabled for fast path
    smoothing_window_size: int = 3
    smoothing_ema_alpha: float = 0.3
    smoothing_confidence_threshold: float = 0.7
    smoothing_min_frames: int = 2


settings = Settings()