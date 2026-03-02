"""Main processing logic for emotion analysis."""

import traceback
from audio_utils import get_audio_duration, group_emotion_ranges
from emotion_model import analyze_emotion_over_time, predict_emotion_full
from file_handler import save_results_to_json, add_audio_file_to_metadata
from output_formatter import (
    print_audio_info,
    print_long_audio_start,
    print_analysis_summary,
    print_emotion_ranges,
    print_emotion_transitions,
)
from report_generator import print_detailed_emotion_analysis
from config import CHUNK_WINDOW_SIZE, CHUNK_HOP_SIZE, AUDIO_DURATION_THRESHOLD


def process_short_audio(audio_path, audio_duration):
    """
    Process audio file shorter than threshold (without chunking).
    
    Args:
        audio_path: Path to audio file
        audio_duration: Duration in seconds
        
    Returns:
        Result dictionary or None if error
    """
    print(f"Audio is {audio_duration:.2f}s (under {AUDIO_DURATION_THRESHOLD}s threshold).")
    print("Using full audio prediction (no chunking)...\n")
    
    try:
        analysis_result = predict_emotion_full(
            audio_path=audio_path,
            audio_duration=audio_duration
        )
        
        add_audio_file_to_metadata(analysis_result, audio_path)
        
        output_path = audio_path.replace('.wav', '_emotion_analysis.json')
        save_results_to_json(analysis_result, output_path)
        
        # Print results
        print_analysis_summary(analysis_result)
        
        emotion_ranges = group_emotion_ranges(analysis_result['emotion_segments'])
        print_emotion_ranges(emotion_ranges)
        
        if analysis_result['emotion_transitions']:
            print_emotion_transitions(analysis_result['emotion_transitions'])
        
        # Print detailed smoothing analysis
        print_detailed_emotion_analysis(analysis_result)
        
        return analysis_result
        
    except Exception as e:
        print(f"Error in emotion analysis: {e}")
        traceback.print_exc()
        return None


def process_long_audio(audio_path, audio_array, sampling_rate, audio_duration):
    """
    Process audio file longer than threshold (with chunking).
    
    Args:
        audio_path: Path to audio file
        audio_array: Audio data array
        sampling_rate: Sample rate in Hz
        audio_duration: Duration in seconds
        
    Returns:
        Result dictionary or None if error
    """
    print_long_audio_start()
    
    try:
        analysis_result = analyze_emotion_over_time(
            audio_array=audio_array,
            sampling_rate=sampling_rate,
            audio_duration=audio_duration,
            window_size=CHUNK_WINDOW_SIZE,
            hop_size=CHUNK_HOP_SIZE
        )
        
        add_audio_file_to_metadata(analysis_result, audio_path)
        
        output_path = audio_path.replace('.wav', '_emotion_analysis.json')
        save_results_to_json(analysis_result, output_path)
        
        # Print results
        print_analysis_summary(analysis_result)
        
        emotion_ranges = group_emotion_ranges(analysis_result['emotion_segments'])
        print_emotion_ranges(emotion_ranges)
        
        if analysis_result['emotion_transitions']:
            print_emotion_transitions(analysis_result['emotion_transitions'])
        
        # Print detailed smoothing analysis
        print_detailed_emotion_analysis(analysis_result)
        
        return analysis_result
        
    except Exception as e:
        print(f"Error in emotion analysis: {e}")
        traceback.print_exc()
        return None


def run_analysis(audio_path):
    """
    Main entry point for emotion analysis.
    Automatically selects between short audio (full) or long audio (chunked) based on duration.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Result dictionary or None if error
    """
    try:
        audio_duration, audio_array, sampling_rate = get_audio_duration(audio_path)
        print_audio_info(audio_duration)
        
        # Choose processing method based on audio duration
        if audio_duration < AUDIO_DURATION_THRESHOLD:
            return process_short_audio(audio_path, audio_duration)
        else:
            return process_long_audio(audio_path, audio_array, sampling_rate, audio_duration)
            
    except Exception as e:
        print(f"Error loading audio file: {e}")
        traceback.print_exc()
        return None
