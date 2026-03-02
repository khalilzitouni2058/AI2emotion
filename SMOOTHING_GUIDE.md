"""
SMOOTHING CONFIGURATION GUIDE

Adjust smoothing in config.py to control how noisy predictions are filtered.

=============================================================================
SMOOTHING METHODS
=============================================================================

1. 'median' - Takes most common emotion in sliding window
   ├─ PROS: Simple, stable
   └─ CONS: Delayed response to real changes

2. 'ema' - Exponential moving average weighted towards recent predictions
   ├─ PROS: Responsive to changes
   └─ CONS: Can still follow noisy predictions

3. 'hysteresis' - Only switches emotions when high confidence is sustained
   ├─ PROS: Prevents accidental switches
   └─ CONS: Slow to respond, high confidence threshold may be limiting

4. 'combined' (DEFAULT) - Median filter first, then hysteresis
   ├─ PROS: Best for reducing noise while staying responsive
   └─ CONS: Most conservative

5. None - No smoothing
   ├─ PROS: Raw model output
   └─ CONS: Very noisy with many false transitions

=============================================================================
TUNING PARAMETERS
=============================================================================

1. SMOOTHING_WINDOW_SIZE (for median filter)
   Current: 3
   Try: 3 (conservative), 5 (more aggressive), 7 (very aggressive)
   Impact: Larger = more smoothing but slower to detect changes

2. SMOOTHING_EMA_ALPHA (for exponential moving average)
   Current: 0.3
   Try: 0.1-0.2 (more memory), 0.3-0.5 (balanced), 0.7+ (recent-focused)
   Impact: Higher = more responsive to recent changes

3. SMOOTHING_CONFIDENCE_THRESHOLD (for hysteresis)
   Current: 0.7
   Try: 0.5 (permissive), 0.7 (balanced), 0.9 (strict)
   Impact: Higher = requires more confidence to switch emotions

4. SMOOTHING_MIN_FRAMES (for hysteresis)
   Current: 2
   Try: 1 (responsive), 2 (balanced), 3+ (stable)
   Impact: Higher = more frames needed to confirm emotion switch

=============================================================================
RECOMMENDED PRESETS
=============================================================================

VERY NOISY DATA (lots of false transitions in raw):
SMOOTHING_METHOD = 'combined'
SMOOTHING_WINDOW_SIZE = 5
SMOOTHING_CONFIDENCE_THRESHOLD = 0.8
SMOOTHING_MIN_FRAMES = 3

BALANCED (default):
SMOOTHING_METHOD = 'combined'
SMOOTHING_WINDOW_SIZE = 3
SMOOTHING_CONFIDENCE_THRESHOLD = 0.7
SMOOTHING_MIN_FRAMES = 2

RESPONSIVE (trust raw predictions):
SMOOTHING_METHOD = 'ema'
SMOOTHING_EMA_ALPHA = 0.4
(Other parameters ignored)

MINIMAL SMOOTHING (light denoising):
SMOOTHING_METHOD = 'median'
SMOOTHING_WINDOW_SIZE = 3

RAW OUTPUT (no smoothing):
SMOOTHING_METHOD = None

=============================================================================
HOW TO DEBUG
=============================================================================

The JSON output includes:

- "emotion_raw": Raw model prediction
- "emotion": Final smoothed prediction
- "confidence": Model confidence score

Use these to see if smoothing is helping or hurting your specific audio.

Example JSON segment:
{
"chunk_id": 1,
"emotion_raw": "happy",
"emotion": "happy",
"confidence": 0.95
}
"""
