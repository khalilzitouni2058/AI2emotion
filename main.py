"""
Main entry point for emotion analysis application.
"""

from processor import run_analysis


if __name__ == "__main__":
    audio_file = "C:\\Users\\kzito\\OneDrive\\Documents\\ISS\\angry.wav"
    run_analysis(audio_file)