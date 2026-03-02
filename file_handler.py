"""File I/O operations."""

import json
from datetime import datetime


def save_results_to_json(data, output_path):
    """
    Save analysis results to JSON file.
    
    Args:
        data: Dictionary with results to save
        output_path: Output file path
    """
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to: {output_path}")


def create_short_audio_result(audio_path, audio_duration, emotion):
    """
    Create result dictionary for short audio (single prediction).
    
    Args:
        audio_path: Path to audio file
        audio_duration: Duration in seconds
        emotion: Predicted emotion
        
    Returns:
        Result dictionary
    """
    return {
        "metadata": {
            "audio_file": audio_path,
            "total_duration": round(audio_duration, 3),
            "method": "single_prediction",
            "analysis_timestamp": datetime.now().isoformat()
        },
        "predicted_emotion": emotion
    }


def add_audio_file_to_metadata(analysis_result, audio_path):
    """
    Add audio file path to analysis result metadata.
    
    Args:
        analysis_result: Result dictionary
        audio_path: Path to audio file
    """
    analysis_result['metadata']['audio_file'] = audio_path
