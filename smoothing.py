"""Smoothing and denoising filters for emotion predictions."""

from collections import Counter
import torch


def exponential_moving_average_filter(predictions, alpha=0.3):
    """
    Apply exponential moving average smoothing to emotion predictions.
    Creates a weighted average that favors recent predictions.
    
    Args:
        predictions: List of emotion prediction strings
        alpha: Smoothing factor (0.0-1.0). Higher = more weight on recent values
        
    Returns:
        List of smoothed predictions
    """
    if not predictions:
        return predictions
    
    if len(predictions) == 1:
        return predictions
    
    # For string predictions, use voting with decay weights
    smoothed = [predictions[0]]
    
    for i in range(1, len(predictions)):
        # Weight recent predictions higher
        window_size = min(5, i + 1)  # Look at last 5 predictions
        window = predictions[max(0, i - window_size + 1):i + 1]
        
        # Count emotions with decay (recent ones weighted more)
        weighted_counts = {}
        for j, emotion in enumerate(window):
            weight = alpha ** (len(window) - 1 - j)  # Exponential decay
            weighted_counts[emotion] = weighted_counts.get(emotion, 0) + weight
        
        # Pick emotion with highest weighted count
        smoothed_emotion = max(weighted_counts, key=weighted_counts.get)
        smoothed.append(smoothed_emotion)
    
    return smoothed


def median_filter(predictions, window_size=3):
    """
    Apply median filtering to emotion predictions.
    Takes the most common emotion in a sliding window.
    
    Args:
        predictions: List of emotion prediction strings
        window_size: Size of median filter window (must be odd)
        
    Returns:
        List of filtered predictions
    """
    if not predictions:
        return predictions
    
    if window_size < 1:
        return predictions
    
    if window_size % 2 == 0:
        window_size += 1  # Make window size odd
    
    filtered = []
    half_window = window_size // 2
    
    for i in range(len(predictions)):
        start = max(0, i - half_window)
        end = min(len(predictions), i + half_window + 1)
        
        # Get most common emotion in window
        window_emotions = predictions[start:end]
        most_common = Counter(window_emotions).most_common(1)[0][0]
        filtered.append(most_common)
    
    return filtered


def hysteresis_filter(predictions, confidences, confidence_threshold=0.7, min_consecutive_frames=2):
    """
    Apply hysteresis filtering: only switch emotions if new emotion
    has high confidence for several consecutive frames.
    
    Args:
        predictions: List of emotion prediction strings
        confidences: List of confidence scores (0.0-1.0) for each prediction
        confidence_threshold: Minimum confidence to allow emotion switch
        min_consecutive_frames: Number of consecutive high-confidence frames needed to switch
        
    Returns:
        List of hysteresis-filtered predictions
    """
    if not predictions or not confidences:
        return predictions
    
    if len(predictions) != len(confidences):
        return predictions
    
    filtered = [predictions[0]]
    current_emotion = predictions[0]
    switch_candidate = None
    switch_count = 0
    
    for i in range(1, len(predictions)):
        emotion = predictions[i]
        confidence = confidences[i]
        
        if emotion == current_emotion:
            # Same emotion - continue
            filtered.append(emotion)
            switch_candidate = None
            switch_count = 0
        elif confidence >= confidence_threshold:
            # Different emotion with high confidence
            if emotion == switch_candidate:
                switch_count += 1
                if switch_count >= min_consecutive_frames - 1:
                    # Switch emotions
                    current_emotion = emotion
                    switch_candidate = None
                    switch_count = 0
            else:
                # New candidate
                switch_candidate = emotion
                switch_count = 1
            
            filtered.append(current_emotion)  # Keep current until fully switched
        else:
            # Different emotion but low confidence - ignore
            filtered.append(current_emotion)
            switch_candidate = None
            switch_count = 0
    
    return filtered


def interpolate_probabilities(probabilities_list):
    """
    Linearly interpolate probabilities between keyframes (argmax changes).

    Args:
        probabilities_list: List of probability tensors (1D)

    Returns:
        List of probability tensors with interpolation applied
    """
    if not probabilities_list:
        return probabilities_list

    if len(probabilities_list) == 1:
        return probabilities_list

    argmaxes = [int(torch.argmax(p).item()) for p in probabilities_list]
    keyframes = [0]
    for i in range(1, len(argmaxes)):
        if argmaxes[i] != argmaxes[i - 1]:
            keyframes.append(i)
    if keyframes[-1] != len(probabilities_list) - 1:
        keyframes.append(len(probabilities_list) - 1)

    smoothed = [p.clone() for p in probabilities_list]
    for k in range(len(keyframes) - 1):
        start_idx = keyframes[k]
        end_idx = keyframes[k + 1]
        if end_idx == start_idx:
            continue
        start_prob = probabilities_list[start_idx]
        end_prob = probabilities_list[end_idx]
        span = end_idx - start_idx
        for i in range(start_idx + 1, end_idx):
            t = (i - start_idx) / span
            smoothed[i] = (1.0 - t) * start_prob + t * end_prob

    return smoothed


def apply_smoothing(predictions, confidences=None, method='median', **kwargs):
    """
    Apply smoothing to predictions.
    
    Args:
        predictions: List of emotion predictions
        confidences: List of confidence scores (required for 'hysteresis')
        method: 'ema', 'median', 'hysteresis', or 'combined'
        **kwargs: Additional parameters for specific methods
        
    Returns:
        Smoothed predictions
    """
    if method == 'ema':
        alpha = kwargs.get('alpha', 0.3)
        return exponential_moving_average_filter(predictions, alpha=alpha)
    
    elif method == 'median':
        window_size = kwargs.get('window_size', 3)
        return median_filter(predictions, window_size=window_size)
    
    elif method == 'hysteresis':
        if confidences is None:
            raise ValueError("Confidences required for hysteresis filter")
        confidence_threshold = kwargs.get('confidence_threshold', 0.7)
        min_frames = kwargs.get('min_consecutive_frames', 2)
        return hysteresis_filter(predictions, confidences, 
                                confidence_threshold=confidence_threshold,
                                min_consecutive_frames=min_frames)
    
    elif method == 'combined':
        # Apply median first, then hysteresis if confidences available
        smoothed = median_filter(predictions, window_size=kwargs.get('window_size', 3))
        if confidences is not None:
            smoothed = hysteresis_filter(smoothed, confidences,
                                        confidence_threshold=kwargs.get('confidence_threshold', 0.7),
                                        min_consecutive_frames=kwargs.get('min_consecutive_frames', 2))
        return smoothed
    
    else:
        return predictions
