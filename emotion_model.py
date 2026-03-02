"""Emotion model prediction functions."""

import torch
import torch.nn.functional as F
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
from config import (
    MODEL_ID,
    SMOOTHING_MODE,
    OFFLINE_SMOOTHING_METHOD,
    STREAMING_SMOOTHING_METHOD,
    STREAMING_USE_HYSTERESIS,
    SMOOTHING_METHOD,
    SMOOTHING_WINDOW_SIZE,
    SMOOTHING_EMA_ALPHA,
    SMOOTHING_CONFIDENCE_THRESHOLD,
    SMOOTHING_MIN_FRAMES,
    SUB_WINDOW_SIZE,
    SUB_HOP_SIZE,
)
from audio_utils import preprocess_audio, sliding_window_segmentation
from smoothing import apply_smoothing, hysteresis_filter, interpolate_probabilities
from datetime import datetime


# Initialize model and feature extractor
model = AutoModelForAudioClassification.from_pretrained(MODEL_ID)
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID, do_normalize=True)
id2label = model.config.id2label


def _get_device():
    """Get the appropriate device (GPU or CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _probs_to_dict(probabilities):
    return {id2label[i]: round(float(probabilities[i].item()), 3) for i in range(len(probabilities))}


def predict_emotion_full(audio_path, max_duration=30.0, audio_duration=None):
    """
    Predict emotion for full audio file without chunking.
    
    Args:
        audio_path: Path to audio file
        max_duration: Maximum duration in seconds
        audio_duration: Optional audio duration for result metadata
        
    Returns:
        Dictionary with emotion analysis result (compatible with analyze_emotion_over_time)
    """
    inputs = preprocess_audio(audio_path, feature_extractor, max_duration)
    
    device = _get_device()
    model.to(device)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probabilities = F.softmax(logits, dim=-1)[0].detach().cpu()
    predicted_id = torch.argmax(logits, dim=-1).item()
    emotion = id2label[predicted_id]
    confidence = float(probabilities[predicted_id].item())
    
    # Return result in same format as analyze_emotion_over_time for consistency
    if audio_duration is None:
        audio_duration = max_duration
    
    prediction = {
        "chunk_id": 1,
        "start_time": 0.0,
        "end_time": round(audio_duration, 3),
        "timestamp": round(audio_duration / 2.0, 3),
        "duration": round(audio_duration, 3),
        "emotion": emotion,
        "confidence": round(confidence, 3),
        "probabilities": _probs_to_dict(probabilities),
    }
    
    result = {
        "metadata": {
            "total_duration": round(audio_duration, 3),
            "total_chunks": 1,
            "window_size": audio_duration,
            "hop_size": audio_duration,
            "sampling_rate": 16000,
            "smoothing_method": "none",
            "analysis_timestamp": datetime.now().isoformat(),
            "processing_mode": "full_audio",
            "sub_window_size": SUB_WINDOW_SIZE,
            "sub_hop_size": SUB_HOP_SIZE,
        },
        "emotion_segments": [prediction],
        "emotion_transitions": [],
        "summary": {
            "total_transitions": 0,
            "emotion_distribution": {emotion: 1}
        }
    }
    
    return result


def predict_emotion_segment(audio_array, sampling_rate, return_confidence=False):
    """
    Predict emotion for a specific audio segment.
    
    Args:
        audio_array: Audio data array
        sampling_rate: Sample rate in Hz
        return_confidence: If True, return (emotion, confidence) tuple
        
    Returns:
        Emotion label, or (emotion, confidence) tuple if return_confidence=True
    """
    inputs = feature_extractor(
        audio_array,
        sampling_rate=sampling_rate,
        return_tensors="pt",
    )
    
    device = _get_device()
    model.to(device)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    predicted_id = torch.argmax(logits, dim=-1).item()
    emotion = id2label[predicted_id]
    
    if return_confidence:
        # Get confidence using softmax
        probabilities = F.softmax(logits, dim=-1)
        confidence = probabilities[0, predicted_id].item()
        return emotion, confidence
    
    return emotion


def _predict_probabilities(audio_array, sampling_rate):
    inputs = feature_extractor(
        audio_array,
        sampling_rate=sampling_rate,
        return_tensors="pt",
    )

    device = _get_device()
    model.to(device)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probabilities = F.softmax(logits, dim=-1)[0].detach().cpu()
    return probabilities


def _average_subwindow_probabilities(segment_audio, sampling_rate):
    segment_duration = len(segment_audio) / sampling_rate

    if segment_duration <= 0:
        return None

    if segment_duration <= SUB_WINDOW_SIZE:
        return _predict_probabilities(segment_audio, sampling_rate)

    subwindows = sliding_window_segmentation(segment_duration, SUB_WINDOW_SIZE, SUB_HOP_SIZE)
    if not subwindows:
        return _predict_probabilities(segment_audio, sampling_rate)

    probs_list = []
    for start, end in subwindows:
        start_sample = int(start * sampling_rate)
        end_sample = int(end * sampling_rate)
        sub_audio = segment_audio[start_sample:end_sample]
        if len(sub_audio) > 0:
            probs_list.append(_predict_probabilities(sub_audio, sampling_rate))

    if not probs_list:
        return _predict_probabilities(segment_audio, sampling_rate)

    stacked = torch.stack(probs_list, dim=0)
    return torch.mean(stacked, dim=0)


def analyze_emotion_over_time(audio_array, sampling_rate, audio_duration, window_size, hop_size):
    """
    Analyze emotion changes over time using sliding window.
    
    Args:
        audio_array: Audio data array
        sampling_rate: Sample rate in Hz
        audio_duration: Total duration in seconds
        window_size: Window size in seconds
        hop_size: Hop size in seconds
        
    Returns:
        Dictionary with emotion segments and transitions
    """
    print(f"Audio Duration: {audio_duration:.2f} seconds")
    print(f"Sampling Rate: {sampling_rate} Hz")
    print(f"Computing emotion predictions...")
    
    # Generate chunks
    chunks = sliding_window_segmentation(audio_duration, window_size, hop_size)
    
    # Predict emotion for each chunk (with confidence scores)
    predictions = []
    raw_emotions = []
    confidences = []
    avg_probs_list = []
    
    for i, (start, end) in enumerate(chunks):
        start_sample = int(start * sampling_rate)
        end_sample = int(end * sampling_rate)
        segment_audio = audio_array[start_sample:end_sample]
        center_time = start + ((end - start) / 2.0)
        
        if len(segment_audio) > 0:
            avg_probs = _average_subwindow_probabilities(segment_audio, sampling_rate)
            if avg_probs is None:
                continue

            avg_probs_list.append(avg_probs)
            predicted_id = int(torch.argmax(avg_probs).item())
            emotion = id2label[predicted_id]
            confidence = float(avg_probs[predicted_id].item())

            raw_emotions.append(emotion)
            confidences.append(confidence)
            predictions.append({
                "chunk_id": i + 1,
                "start_time": round(start, 3),
                "end_time": round(end, 3),
                "timestamp": round(center_time, 3),
                "duration": round(end - start, 3),
                "emotion": emotion,
                "confidence": round(confidence, 3),
                "probabilities": _probs_to_dict(avg_probs),
            })
            print(
                f"Chunk {i+1}/{len(chunks)}: {start:.2f}s - {end:.2f}s "
                f"(t={center_time:.2f}s) | {emotion} ({confidence:.2f})"
            )
    
    # Apply smoothing to predictions
    smoothing_method_used = None
    if len(raw_emotions) > 1:
        if SMOOTHING_MODE == 'offline':
            if OFFLINE_SMOOTHING_METHOD == 'interpolate':
                print("\nApplying offline interpolation smoothing...")
                smoothed_probs = interpolate_probabilities(avg_probs_list)
                for i, pred in enumerate(predictions):
                    pred["emotion_raw"] = pred["emotion"]
                    pred["confidence_raw"] = pred["confidence"]
                    pred["probabilities_raw"] = pred["probabilities"]
                    predicted_id = int(torch.argmax(smoothed_probs[i]).item())
                    pred["emotion"] = id2label[predicted_id]
                    pred["confidence"] = round(float(smoothed_probs[i][predicted_id].item()), 3)
                    pred["probabilities"] = _probs_to_dict(smoothed_probs[i])
                smoothing_method_used = 'interpolate'
            elif OFFLINE_SMOOTHING_METHOD == 'median':
                print("\nApplying offline median smoothing...")
                smoothed_emotions = apply_smoothing(
                    raw_emotions,
                    method='median',
                    window_size=SMOOTHING_WINDOW_SIZE,
                )
                for i, pred in enumerate(predictions):
                    pred["emotion_raw"] = pred["emotion"]
                    pred["emotion"] = smoothed_emotions[i]
                smoothing_method_used = 'median'
        elif SMOOTHING_MODE == 'streaming':
            if STREAMING_SMOOTHING_METHOD == 'ema':
                print("\nApplying streaming EMA smoothing...")
                smoothed_emotions = apply_smoothing(
                    raw_emotions,
                    method='ema',
                    alpha=SMOOTHING_EMA_ALPHA,
                )
                if STREAMING_USE_HYSTERESIS:
                    smoothed_emotions = hysteresis_filter(
                        smoothed_emotions,
                        confidences,
                        confidence_threshold=SMOOTHING_CONFIDENCE_THRESHOLD,
                        min_consecutive_frames=SMOOTHING_MIN_FRAMES,
                    )
                    smoothing_method_used = 'ema+hysteresis'
                else:
                    smoothing_method_used = 'ema'

                for i, pred in enumerate(predictions):
                    pred["emotion_raw"] = pred["emotion"]
                    pred["emotion"] = smoothed_emotions[i]
        else:
            if SMOOTHING_METHOD:
                print(f"\nApplying {SMOOTHING_METHOD} smoothing...")
                smoothed_emotions = apply_smoothing(
                    raw_emotions,
                    confidences=confidences,
                    method=SMOOTHING_METHOD,
                    window_size=SMOOTHING_WINDOW_SIZE,
                    alpha=SMOOTHING_EMA_ALPHA,
                    confidence_threshold=SMOOTHING_CONFIDENCE_THRESHOLD,
                    min_consecutive_frames=SMOOTHING_MIN_FRAMES
                )

                # Update predictions with smoothed emotions
                for i, pred in enumerate(predictions):
                    pred["emotion_raw"] = pred["emotion"]
                    pred["emotion"] = smoothed_emotions[i]
                smoothing_method_used = SMOOTHING_METHOD
    
    # Track emotion transitions (based on smoothed predictions)
    transitions = []
    for i in range(1, len(predictions)):
        if predictions[i]["emotion"] != predictions[i-1]["emotion"]:
            transitions.append({
                "transition_id": len(transitions) + 1,
                "from_emotion": predictions[i-1]["emotion"],
                "to_emotion": predictions[i]["emotion"],
                "transition_time": predictions[i].get("timestamp", predictions[i]["start_time"]),
                "previous_segment": predictions[i-1]["chunk_id"],
                "current_segment": predictions[i]["chunk_id"]
            })
    
    # Create comprehensive output
    result = {
        "metadata": {
            "total_duration": round(audio_duration, 3),
            "total_chunks": len(predictions),
            "window_size": window_size,
            "hop_size": hop_size,
            "sampling_rate": sampling_rate,
            "smoothing_method": smoothing_method_used or SMOOTHING_METHOD,
            "analysis_timestamp": datetime.now().isoformat(),
            "sub_window_size": SUB_WINDOW_SIZE,
            "sub_hop_size": SUB_HOP_SIZE,
        },
        "emotion_segments": predictions,
        "emotion_transitions": transitions,
        "summary": {
            "total_transitions": len(transitions),
            "emotion_distribution": {}
        }
    }
    
    # Calculate emotion distribution (based on smoothed predictions)
    for pred in predictions:
        emotion = pred["emotion"]
        if emotion not in result["summary"]["emotion_distribution"]:
            result["summary"]["emotion_distribution"][emotion] = 0
        result["summary"]["emotion_distribution"][emotion] += 1
    
    return result
