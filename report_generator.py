"""Summary and statistics reporting for emotion analysis."""

def print_raw_vs_smoothed_comparison(analysis_result):
    """Print comparison between raw and smoothed predictions."""
    segments = analysis_result.get('emotion_segments', [])
    
    if not segments:
        return
    
    # Check if smoothing was applied
    has_raw = 'emotion_raw' in segments[0]
    if not has_raw:
        return
    
    print("\n" + "-" * 60)
    print("RAW vs SMOOTHED COMPARISON:")
    print("-" * 60)
    print(f"{'Chunk':<8} {'Raw Emotion':<15} {'Confidence':<12} {'Smoothed':<15}")
    print("-" * 60)
    
    for seg in segments:
        chunk_id = seg.get('chunk_id', '?')
        raw = seg.get('emotion_raw', '?')
        smoothed = seg.get('emotion', '?')
        confidence = seg.get('confidence', 0)
        
        marker = " ✓" if raw == smoothed else " →"
        print(f"{chunk_id:<8} {raw:<15} {confidence:<12.3f} {smoothed:<15}{marker}")


def print_smoothing_stats(analysis_result):
    """Print statistics about smoothing effectiveness."""
    segments = analysis_result.get('emotion_segments', [])
    
    if not segments:
        return
    
    # Check if smoothing was applied
    has_raw = 'emotion_raw' in segments[0]
    if not has_raw:
        return
    
    raw_emotions = [seg.get('emotion_raw') for seg in segments]
    smoothed_emotions = [seg.get('emotion') for seg in segments]
    changes = sum(1 for r, s in zip(raw_emotions, smoothed_emotions) if r != s)
    
    print("\n" + "-" * 60)
    print("SMOOTHING STATISTICS:")
    print("-" * 60)
    print(f"Total chunks analyzed: {len(segments)}")
    print(f"Predictions changed by smoothing: {changes}/{len(segments)} ({100*changes//len(segments)}%)")
    
    # Count raw transitions
    raw_transitions = sum(1 for i in range(1, len(raw_emotions)) if raw_emotions[i] != raw_emotions[i-1])
    smoothed_transitions = analysis_result.get('summary', {}).get('total_transitions', 0)
    print(f"Raw transitions: {raw_transitions}")
    print(f"Smoothed transitions: {smoothed_transitions}")
    
    smoothing_method = analysis_result.get('metadata', {}).get('smoothing_method')
    if smoothing_method:
        print(f"Smoothing method: {smoothing_method}")


def print_detailed_emotion_analysis(analysis_result):
    """Print detailed analysis with raw and smoothed predictions."""
    print_raw_vs_smoothed_comparison(analysis_result)
    print_smoothing_stats(analysis_result)
