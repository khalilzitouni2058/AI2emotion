"""Configuration constants for emotion analysis."""

# Model settings
MODEL_ID = "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3"

# Audio processing settings
AUDIO_DURATION_THRESHOLD = 15.0  # seconds - threshold between single prediction and chunked analysis
AUDIO_SAMPLING_RATE = 16000  # Hz
CHUNK_WINDOW_SIZE = 1.9  # seconds
CHUNK_HOP_SIZE = 0.5  # seconds
# NVIDIA-like inner sweep (double sliding window)
SUB_WINDOW_SIZE = 0.625  # seconds
SUB_HOP_SIZE = 0.31  # seconds
MAX_AUDIO_DURATION = 30.0  # seconds - max duration for preprocessing

# Output settings
PRINT_WIDTH = 60

# Smoothing/Denoising settings
SMOOTHING_MODE = 'offline'  # Options: 'offline', 'streaming'
OFFLINE_SMOOTHING_METHOD = 'interpolate'  # Options: 'interpolate', 'median', None
STREAMING_SMOOTHING_METHOD = 'ema'  # Options: 'ema', None
STREAMING_USE_HYSTERESIS = False  # Optional tiny hysteresis for streaming

SMOOTHING_METHOD = 'combined'  # Legacy fallback when SMOOTHING_MODE is None
SMOOTHING_WINDOW_SIZE = 3  # For median filter (must be odd)
SMOOTHING_EMA_ALPHA = 0.3  # For exponential moving average (0.0-1.0, higher = more weight on recent)
SMOOTHING_CONFIDENCE_THRESHOLD = 0.7  # For hysteresis filter (0.0-1.0)
SMOOTHING_MIN_FRAMES = 2  # For hysteresis filter - min consecutive frames to switch emotion
