"""Output formatting and printing functions."""

from config import PRINT_WIDTH


def print_header(title):
    """Print a formatted header."""
    print("=" * PRINT_WIDTH)
    print(title)
    print("=" * PRINT_WIDTH)


def print_audio_info(duration):
    """Print audio duration information."""
    print_header(f"Audio Duration: {duration:.2f} seconds")


def print_short_audio_start():
    """Print start message for short audio processing."""
    print("Audio is less than 11 seconds - predicting without chunks")
    print_header("FULL AUDIO EMOTION PREDICTION")


def print_emotion_result(emotion):
    """Print the predicted emotion."""
    print(f"Predicted Emotion: {emotion}\n")


def print_long_audio_start():
    """Print start message for long audio processing."""
    print("Audio is 11 seconds or longer - using chunked analysis")
    print_header("EMOTION ANALYSIS OVER TIME (SLIDING WINDOW)")


def print_analysis_summary(analysis_result):
    """Print summary of chunked analysis."""
    print_header("ANALYSIS SUMMARY")
    print(f"Total Duration: {analysis_result['metadata']['total_duration']:.2f} seconds")
    print(f"Total Chunks: {analysis_result['metadata']['total_chunks']}")
    print(f"Emotion Transitions: {analysis_result['summary']['total_transitions']}")
    print(f"Emotion Distribution: {analysis_result['summary']['emotion_distribution']}")


def print_emotion_ranges(emotion_ranges):
    """Print emotion time ranges."""
    print("\n" + "-" * PRINT_WIDTH)
    print("EMOTION TIME RANGES:")
    print("-" * PRINT_WIDTH)
    for i, emotion_range in enumerate(emotion_ranges, 1):
        print(f"{i}. From {emotion_range['start']:.2f} to {emotion_range['end']:.2f} → {emotion_range['emotion']}")


def print_emotion_transitions(transitions, max_show=10):
    """Print emotion transitions."""
    print("\n" + "-" * PRINT_WIDTH)
    print("EMOTION TRANSITIONS:")
    print("-" * PRINT_WIDTH)
    for trans in transitions[:max_show]:
        print(f"  {trans['transition_time']:.2f}s: {trans['from_emotion']} → {trans['to_emotion']}")
    if len(transitions) > max_show:
        print(f"  ... and {len(transitions) - max_show} more transitions")
