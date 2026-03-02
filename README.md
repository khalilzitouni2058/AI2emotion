# Emotion Analysis Workflow

A modular emotion recognition system that analyzes audio files and tracks emotional changes over time with noise reduction.

## Table of Contents

- [Quick Start](#quick-start)
- [Workflow Overview](#workflow-overview)
- [Detailed Process](#detailed-process)
- [File Structure](#file-structure)
- [Configuration](#configuration)
- [Output Format](#output-format)
- [Examples](#examples)

---

## Quick Start

```bash
python main.py
```

Analyzes the audio file specified in `main.py` and outputs:

- JSON file with emotion predictions and statistics
- Console output with summary and raw vs smoothed comparison

---

## Workflow Overview

```
┌─────────────────┐
│  Audio Input    │
│   (*.wav)       │
└────────┬────────┘
         │
         ▼
┌──────────────────────────┐
│  Load Audio & Get Info   │
│  ├─ Sampling: 16 kHz     │
│  └─ Get duration         │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────────┐
│  Sliding Window Chunking     │
│  ├─ Window: 1.9s            │
│  ├─ Hop: 0.5s (overlapping) │
│  └─ Generate chunk segments  │
└────────┬─────────────────────┘
         │
         ▼
    ┌────────────────────────────────┐
    │ For Each Chunk:                │
    │ ├─ Sub-window averaging        │
    │ │  ├─ Window: 0.625s          │
    │ │  ├─ Hop: 0.31s              │
    │ │  └─ Average probabilities    │
    │ ├─ Get emotion from avg probs  │
    │ └─ Extract confidence score    │
    └────────┬───────────────────────┘
             │
             ▼
    ┌──────────────────────────────┐
    │  Apply Smoothing (Mode)      │
    │                              │
    │  OFFLINE Mode:               │
    │  ├─ Interpolation            │
    │  └─ Median filter            │
    │                              │
    │  STREAMING Mode:             │
    │  ├─ EMA smoothing            │
    │  └─ Optional: Hysteresis     │
    └────────┬─────────────────────┘
             │
             ▼
    ┌──────────────────────────────┐
    │ Track Emotion Transitions    │
    │ & Calculate Statistics       │
    └────────┬─────────────────────┘
             │
             ▼
    ┌──────────────────────────────┐
    │ Output JSON + Print Results  │
    │ (Raw vs Smoothed Analysis)   │
    └──────────────────────────────┘
```

---

## Detailed Process

### 1. Input: Load Audio File

**What happens:**

- Audio file is loaded using librosa at 16 kHz sampling rate
- Duration is calculated in seconds
- Audio is stored as numpy array for processing

**Example:**

```
File: w.wav
Duration: 13.38 seconds
Sampling Rate: 16000 Hz
```

---

### 2. Sliding Window Chunking

**Process:**

```
sliding_window_segmentation()
    ↓
Generate overlapping chunks with window_size and hop_size

For 13.38 second audio with window_size=1.9s, hop_size=0.5s:

[0.00 - 1.90] ← Chunk 1
[0.50 - 2.40] ← Chunk 2
[1.00 - 2.90] ← Chunk 3
[1.50 - 3.40] ← Chunk 4
...
[12.08 - 13.38] ← Final chunk

Note: Overlapping windows provide continuous coverage
```

**Configuration:**

- `CHUNK_WINDOW_SIZE = 1.9` seconds (chunk duration)
- `CHUNK_HOP_SIZE = 0.5` seconds (overlap between chunks)
- Overlapping provides temporal consistency

---

### 3. Sub-Window Averaging (NEW)

**Process:**

```
For each chunk, apply inner sliding window for better probability estimation:

_average_subwindow_probabilities()
    ↓
For a 1.9s chunk with SUB_WINDOW_SIZE=0.625s, SUB_HOP_SIZE=0.31s:

[0.00 - 0.625] ← Sub-window 1
[0.31 - 0.935] ← Sub-window 2
[0.62 - 1.245] ← Sub-window 3
...

For each sub-window:
    ├─ Get emotion probabilities from model
    └─ Collect into list

Average all sub-window probabilities
    ↓
Use averaged probabilities as chunk prediction
```

**Why sub-windows?**

- NVIDIA-inspired double sliding window technique
- Smooths predictions at the chunk level
- Reduces jitter from single predictions
- More stable emotion detection per chunk

**Example:**

```
Chunk predictions before sub-window averaging:
  Sub-window 1: [0.1, 0.8, 0.1] (happy)
  Sub-window 2: [0.15, 0.75, 0.1] (happy)
  Sub-window 3: [0.2, 0.7, 0.1] (happy)

Averaged: [0.15, 0.75, 0.1] → happy (more stable)
Confidence: 0.75
```

---

### 4. Raw Predictions

**Process:**

```
For each chunk:
    extract_segment()
        ↓
    apply_subwindow_averaging()
        ↓
    get_emotion_from_avg_probabilities()
        ↓
    extract_confidence_score
        ↓
    Store prediction with metadata
```

**Example Output - RAW:**

```
Chunk 1 (0-1.9s):     happy (0.92)
Chunk 2 (0.5-2.4s):   happy (0.89)
Chunk 3 (1.0-2.9s):   happy (0.88)
Chunk 4 (1.5-3.4s):   fearful (0.85) ← Slight change
...
```

---

### 5. Smoothing/Denoising

**Process:**

```
apply_smoothing()
    ↓
SMOOTHING_MODE selection:

┌─── OFFLINE MODE ──┐
│                   │
├─ Interpolation    │ Smooth probability curves with interpolation
│                   │
└─ Median Filter    │ Vote-based emotion smoothing
│
└─── STREAMING MODE ─┐
    │
    ├─ EMA            │ Exponential Moving Average (real-time friendly)
    └─ Hysteresis     │ Optional: Prevent rapid emotion switching
```

**Offline Interpolation (Default):**

```
Input: Raw probability distributions for each chunk

Steps:
1. Stack all probability vectors
2. Interpolate between adjacent probabilities
3. Re-extract emotions from smoothed probabilities

Result: Smooth probability curves over time
```

**Offline Median Filter:**

```
Input: Sequence of raw emotions

Steps:
1. Sliding window of size 3 (or configured)
2. Count emotions in window
3. Vote for most common emotion
4. Replace with majority vote

Result: Emotion sequence with outliers removed
```

**Streaming EMA (Exponential Moving Average):**

```
Input: Sequence of raw emotions, confidences

Process per frame:
1. Look at recent predictions (last 5)
2. Weight by exponential decay (recent = higher weight)
3. Pick emotion with highest weighted count

Result: Smooth, responsive emotion stream
```

**Example: Before & After**

```
BEFORE (RAW):       happy → happy → fearful → fearful → happy
                    (2 transitions, some noise)

AFTER (OFFLINE):    happy → happy → happy → happy → happy
                    (0 transitions, smooth)

AFTER (STREAMING):  happy → happy → happy → fearful → fearful
                    (1 smooth transition)
```

---

### 6. Track Transitions

**Process:**

```
Compare consecutive smoothed predictions:

happy → happy: NO transition
happy → happy: NO transition
happy → fearful: TRANSITION (record it)
fearful → fearful: NO transition

Total Transitions: 1
```

Transitions include:

- From emotion → To emotion
- Transition time
- Previous and current segment IDs

---

### 7. Calculate Statistics

**Process:**

```
emotion_distribution = Count emotions in smoothed predictions

Emotion      Count
happy:       15
fearful:     3
surprised:   2
```

---

## File Structure

```
project/
├── main.py                    # Entry point (minimal, just imports)
├── processor.py               # Main orchestrator
│   ├── process_short_audio()
│   ├── process_long_audio()
│   └── run_analysis()
├── emotion_model.py           # Model predictions
│   ├── predict_emotion_full()
│   ├── predict_emotion_segment()
│   └── analyze_emotion_over_time()
├── audio_utils.py             # Audio processing
│   ├── get_audio_duration()
│   ├── preprocess_audio()
│   ├── sliding_window_segmentation()
│   └── group_emotion_ranges()
├── smoothing.py               # Denoising filters
│   ├── median_filter()
│   ├── exponential_moving_average_filter()
│   ├── hysteresis_filter()
│   └── apply_smoothing()
├── file_handler.py            # JSON I/O
│   ├── save_results_to_json()
│   └── create_short_audio_result()
├── output_formatter.py        # Display functions
│   ├── print_header()
│   ├── print_audio_info()
│   └── print_emotion_ranges()
├── report_generator.py        # Statistics & comparison
│   ├── print_raw_vs_smoothed_comparison()
│   └── print_smoothing_stats()
├── config.py                  # All configuration constants
├── SMOOTHING_GUIDE.md         # Tuning parameters guide
└── README.md                  # This file
```

---

## Configuration

Edit `config.py` to adjust behavior:

### Audio Processing

```python
AUDIO_SAMPLING_RATE = 16000        # Sample rate (Hz)
CHUNK_WINDOW_SIZE = 1.9            # Window size for chunking (seconds)
CHUNK_HOP_SIZE = 0.5               # Step size between chunks (seconds, overlapping)
SUB_WINDOW_SIZE = 0.625            # Sub-window size for averaging (seconds)
SUB_HOP_SIZE = 0.31                # Sub-window hop size (seconds)
MAX_AUDIO_DURATION = 30.0          # Max duration for preprocessing (seconds)
```

### Smoothing Mode Selection

```python
SMOOTHING_MODE = 'offline'                    # Options: 'offline', 'streaming'

# Offline smoothing (full audio analysis post-processing)
OFFLINE_SMOOTHING_METHOD = 'interpolate'      # Options: 'interpolate', 'median', None

# Streaming smoothing (real-time processing)
STREAMING_SMOOTHING_METHOD = 'ema'            # Options: 'ema', None
STREAMING_USE_HYSTERESIS = False              # Optional: Hysteresis filter for stability
```

### Smoothing Parameters

```python
SMOOTHING_WINDOW_SIZE = 3                     # For median filter (recommend: odd number)
SMOOTHING_EMA_ALPHA = 0.3                     # For EMA (0.0-1.0, higher = more weight on recent)
SMOOTHING_CONFIDENCE_THRESHOLD = 0.7          # For hysteresis (0.0-1.0)
SMOOTHING_MIN_FRAMES = 2                      # Consecutive frames needed to switch emotion
```

**See `SMOOTHING_GUIDE.md` for tuning recommendations.**

---

## Output Format

### Console Output

```
============================================================
Audio Duration: 13.38 seconds
============================================================
EMOTION ANALYSIS OVER TIME (SLIDING WINDOW)
============================================================
Audio Duration: 13.38 seconds
Sampling Rate: 16000 Hz
Computing emotion predictions...
Chunk 1/22: 0.00s - 1.90s (t=0.95s) | happy (0.92)
Chunk 2/22: 0.50s - 2.40s (t=1.45s) | happy (0.91)
Chunk 3/22: 1.00s - 2.90s (t=1.95s) | happy (0.89)
...
Chunk 21/22: 11.58s - 13.48s (t=12.53s) | fearful (0.85)
Chunk 22/22: 12.08s - 13.38s (t=12.73s) | fearful (0.84)

Applying offline interpolation smoothing...

Results saved to: test_emotion_analysis.json
============================================================
ANALYSIS SUMMARY
============================================================
Total Duration: 13.38 seconds
Total Chunks: 22
Emotion Transitions: 1
Emotion Distribution: {'happy': 15, 'fearful': 7}

EMOTION TIME RANGES:
1. From 0.00 to 9.50 → happy
2. From 10.00 to 13.38 → fearful

Emotion transitions detected: 1
```

### JSON Output

```json
{
  "metadata": {
    "audio_file": "C:\\Users\\kzito\\OneDrive\\Documents\\ISS\\test.wav",
    "total_duration": 13.38,
    "total_chunks": 22,
    "window_size": 1.9,
    "hop_size": 0.5,
    "sampling_rate": 16000,
    "smoothing_method": "interpolate",
    "analysis_timestamp": "2026-02-17T14:30:22.185934",
    "sub_window_size": 0.625,
    "sub_hop_size": 0.31
  },
  "emotion_segments": [
    {
      "chunk_id": 1,
      "start_time": 0.0,
      "end_time": 1.9,
      "timestamp": 0.95,
      "duration": 1.9,
      "emotion_raw": "happy",
      "emotion": "happy",
      "confidence": 0.924,
      "probabilities": {
        "angry": 0.001,
        "calm": 0.01,
        "fearful": 0.02,
        "happy": 0.924,
        "sad": 0.035,
        "surprised": 0.01
      }
    },
    {
      "chunk_id": 2,
      "start_time": 0.5,
      "end_time": 2.4,
      "timestamp": 1.45,
      "duration": 1.9,
      "emotion_raw": "happy",
      "emotion": "happy",
      "confidence": 0.912,
      "probabilities": {
        "angry": 0.002,
        "calm": 0.015,
        "fearful": 0.025,
        "happy": 0.912,
        "sad": 0.03,
        "surprised": 0.016
      }
    }
  ],
  "emotion_transitions": [
    {
      "transition_id": 1,
      "from_emotion": "happy",
      "to_emotion": "fearful",
      "transition_time": 10.0,
      "previous_segment": 15,
      "current_segment": 16
    }
  ],
  "summary": {
    "total_transitions": 1,
    "emotion_distribution": {
      "happy": 15,
      "fearful": 7
    }
  }
}
```

### Key Fields Explained

| Field                  | Meaning                                 |
| ---------------------- | --------------------------------------- |
| `emotion_raw`          | Direct model output from sub-window avg |
| `emotion`              | After smoothing (offline or streaming)  |
| `confidence`           | Highest probability in emotion class    |
| `probabilities`        | Full probability distribution           |
| `timestamp`            | Center time of chunk                    |
| `emotion_transitions`  | When smoothed emotion actually changes  |
| `emotion_distribution` | Count of each smoothed emotion          |
| `smoothing_method`     | 'interpolate', 'median', 'ema', etc.    |
| `sub_window_size`      | Inner sliding window size               |
| `sub_hop_size`         | Inner sliding window hop size           |

---

## Examples

### Example 1: Short Speech (5 seconds)

```
Input: greeting.wav (5 seconds)
↓
Load audio at 16 kHz
↓
Sliding window chunking with 1.9s window, 0.5s hop:
  Chunk 1: [0.00 - 1.90]
  Chunk 2: [0.50 - 2.40]
  Chunk 3: [1.00 - 2.90]
  Chunk 4: [1.50 - 3.40]
  Chunk 5: [2.00 - 3.90]
  Chunk 6: [2.50 - 4.40]
  Chunk 7: [3.00 - 4.90]
↓
Sub-window averaging for each chunk (0.625s window, 0.31s hop)
↓
Raw predictions:
  Chunk 1: happy (0.92)
  Chunk 2: happy (0.91)
  ...
↓
Apply offline interpolation smoothing
↓
Output: {
  "emotion_segments": [...],
  "emotion_transitions": [],
  "summary": {"emotion_distribution": {"happy": 7}}
}
```

### Example 2: Interview (30 minutes)

```
Input: interview.wav (30 minutes = 1800 seconds)
↓
Load audio at 16 kHz
↓
Sliding window chunking with 1.9s window, 0.5s hop:
  Total chunks ≈ (1800 - 1.9) / 0.5 ≈ 3596 chunks
↓
For each chunk, apply sub-window averaging (0.625s window, 0.31s hop)
↓
Raw predictions (overlapping chunks):
  Chunk 1: happy (0.92)
  Chunk 2: happy (0.91)
  Chunk 3: happy (0.89)
  ...
  Chunk 500: angry (0.88)
  Chunk 501: angry (0.92)
  ...
↓
Apply smoothing (e.g., offline interpolation):
  Smooths probability curves over time
  Reduces jitter between chunks
↓
Track transitions:
  happy → angry at ~256s
  angry → calm at ~512s
  calm → happy at ~768s
↓
Output: JSON with detailed emotion timeline and transitions
```

### Example 3: Streaming vs Offline Mode

**OFFLINE MODE:**

```
After collecting all chunks:
  1. Get all probability distributions
  2. Interpolate smooth curves through probability space
  3. Re-extract emotions from smoothed probabilities

Advantage: Best global smoothness, requires full audio
Disadvantage: Requires batch processing
```

**STREAMING MODE:**

```
Process chunks as they arrive:
  1. For each chunk: Apply EMA filter to recent emotions
  2. Optional: Apply hysteresis to prevent rapid switching
  3. Output emotion immediately (no waiting for future data)

Advantage: Real-time capable, responsive
Disadvantage: May see some transitions that offline would smooth
```

---

## Troubleshooting

### Emotions Change Too Frequently (Noisy Output)

**Symptom:** Many transitions between emotions

**Fix - Use Offline Interpolation:**

```python
SMOOTHING_MODE = 'offline'
OFFLINE_SMOOTHING_METHOD = 'interpolate'  # Smoothest option
```

Or:

```python
SMOOTHING_MODE = 'offline'
OFFLINE_SMOOTHING_METHOD = 'median'       # Alternative
SMOOTHING_WINDOW_SIZE = 5                 # Larger window for more smoothing
```

### Smooth But Missing Real Changes

**Symptom:** All emotions merge into one, real transitions disappear

**Fix - Use Streaming Mode:**

```python
SMOOTHING_MODE = 'streaming'
STREAMING_SMOOTHING_METHOD = 'ema'
SMOOTHING_EMA_ALPHA = 0.5                 # Higher = more responsive
```

Or with hysteresis:

```python
STREAMING_USE_HYSTERESIS = True
SMOOTHING_CONFIDENCE_THRESHOLD = 0.5      # Lower threshold (easier to switch)
SMOOTHING_MIN_FRAMES = 1                  # Switch faster
```

### Need Finer Time Resolution

**Symptom:** Want more granular emotion changes

**Adjust Window/Hop Sizes:**

```python
CHUNK_WINDOW_SIZE = 1.0                   # Smaller window
CHUNK_HOP_SIZE = 0.25                     # Smaller hop (more chunks)
SUB_WINDOW_SIZE = 0.3                     # Smaller sub-window
SUB_HOP_SIZE = 0.15                       # Smaller sub-hop
```

### Need Coarser Time Resolution

**Symptom:** Too many chunks, want aggregated view

**Adjust Window/Hop Sizes:**

```python
CHUNK_WINDOW_SIZE = 3.0                   # Larger window
CHUNK_HOP_SIZE = 1.0                      # Larger hop, less overlap
SUB_WINDOW_SIZE = 1.0                     # Larger sub-window
SUB_HOP_SIZE = 0.5                        # Larger sub-hop
```

### Low Confidence Scores

**Symptom:** All confidences < 0.5

**Check:**

- Is audio quality poor (low volume, background noise)?
- Try different smoothing mode or check raw predictions
- Display raw probabilities for all emotion classes in JSON output

---

## Key Takeaways

1. **Always Chunked** → All audio uses overlapping sliding window analysis
2. **Overlapping Windows** → 1.9s window, 0.5s hop ensures temporal continuity
3. **Sub-window Averaging** → Double sliding window for stable predictions at chunk level
4. **Flexible Smoothing** → Choose between offline (interpolation/median) or streaming (EMA) modes
5. **Two Output Fields** → `emotion_raw` (from sub-window averaging) and `emotion` (smoothed)
6. **Smooth Tracking** → Overlapping chunks + sub-window averaging = smooth emotion transitions
7. **Confidence Tracking** → Each prediction includes confidence scores from model
8. **Extensive Metadata** → JSON stores raw probabilities, timestamps, and transition info

---

## Next Steps

- Edit `audio_file` in `main.py` to analyze different audio files
- Choose smoothing mode in `config.py`: offline (batch) or streaming (real-time)
- Adjust window/hop sizes for desired time resolution
- Check raw probabilities in JSON to debug low confidence scores
- Use JSON output for downstream analysis (dashboards, reports, etc.)
