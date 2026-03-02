"""Audio processing utilities."""

import librosa
import numpy as np
from config import AUDIO_SAMPLING_RATE


def get_audio_duration(audio_path):
    """
    Get duration of audio file in seconds.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Tuple of (duration in seconds, audio_array, sampling_rate)
    """
    audio_array, sampling_rate = librosa.load(audio_path, sr=AUDIO_SAMPLING_RATE)
    duration = len(audio_array) / sampling_rate
    return duration, audio_array, sampling_rate


def preprocess_audio(audio_path, feature_extractor, max_duration=30.0):
    """
    Preprocess audio file for model input.
    
    Args:
        audio_path: Path to audio file
        feature_extractor: Feature extractor from model
        max_duration: Maximum duration in seconds
        
    Returns:
        Input tensors for model
    """
    audio_array, sampling_rate = librosa.load(audio_path, sr=AUDIO_SAMPLING_RATE)
    
    max_length = int(feature_extractor.sampling_rate * max_duration)
    if len(audio_array) > max_length:
        audio_array = audio_array[:max_length]
    else:
        audio_array = np.pad(audio_array, (0, max_length - len(audio_array)))

    inputs = feature_extractor(
        audio_array,
        sampling_rate=feature_extractor.sampling_rate,
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    return inputs


def sliding_window_segmentation(audio_duration, window_size, hop_size):
    """
    Generate overlapping chunks using sliding window segmentation.
    
    Args:
        audio_duration: Total duration of audio in seconds
        window_size: Window size in seconds
        hop_size: Hop/stride size in seconds
    
    Returns:
        List of tuples containing (start_time, end_time) for each chunk
    """
    chunks = []
    start_time = 0.0
    
    while start_time + window_size <= audio_duration:
        end_time = start_time + window_size
        chunks.append((start_time, end_time))
        start_time += hop_size
    
    # Handle last chunk if there's remaining audio
    if start_time < audio_duration:
        end_time = min(start_time + window_size, audio_duration)
        chunks.append((start_time, end_time))
    
    return chunks


def group_emotion_ranges(emotion_segments):
    """
    Group consecutive segments with same emotion into ranges.
    
    Args:
        emotion_segments: List of emotion segment dictionaries
        
    Returns:
        List of emotion ranges with start, end, and emotion
    """
    if not emotion_segments:
        return []
    
    emotion_ranges = []
    current_emotion = emotion_segments[0]['emotion']
    start_time = emotion_segments[0]['start_time']
    
    for i, segment in enumerate(emotion_segments[1:], 1):
        if segment['emotion'] != current_emotion:
            end_time = emotion_segments[i - 1]['end_time']
            emotion_ranges.append({
                'start': start_time,
                'end': end_time,
                'emotion': current_emotion
            })
            current_emotion = segment['emotion']
            start_time = segment['start_time']
    
    # Add the last emotion range
    end_time = emotion_segments[-1]['end_time']
    emotion_ranges.append({
        'start': start_time,
        'end': end_time,
        'emotion': current_emotion
    })
    
    return emotion_ranges
