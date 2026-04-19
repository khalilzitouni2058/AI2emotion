"""Service layer for audio loading, preprocessing, and segmentation."""

from typing import Any, Dict, List, Tuple

import librosa
import numpy as np

from app.core.config import settings


class AudioService:
    """Provides reusable audio operations for the application."""

    def get_audio_duration(self, audio_path: str) -> Tuple[float, np.ndarray, int]:
        """
        Load audio and return its duration, waveform, and sampling rate.

        Args:
            audio_path: Path to the audio file

        Returns:
            Tuple of (duration_seconds, audio_array, sampling_rate)
        """
        audio_array, sampling_rate = librosa.load(
            audio_path,
            sr=settings.audio_sampling_rate,
        )
        duration = len(audio_array) / sampling_rate
        return duration, audio_array, sampling_rate

    def preprocess_audio(
        self,
        audio_path: str,
        feature_extractor: Any,
        max_duration: float | None = None,
        audio_array: np.ndarray | None = None,
        sampling_rate: int | None = None,
    ) -> Dict[str, Any]:
        """
        Preprocess audio file for model inference.

        Args:
            audio_path: Path to the audio file
            feature_extractor: Hugging Face feature extractor
            max_duration: Maximum duration in seconds

        Returns:
            Model input tensors
        """
        if max_duration is None:
            max_duration = settings.max_audio_duration

        if audio_array is None:
            audio_array, sampling_rate = librosa.load(
                audio_path,
                sr=settings.audio_sampling_rate,
            )

        if sampling_rate is None:
            sampling_rate = settings.audio_sampling_rate

        max_length = int(feature_extractor.sampling_rate * max_duration)
        if len(audio_array) > max_length:
            audio_array = audio_array[:max_length]
        else:
            audio_array = np.pad(audio_array, (0, max_length - len(audio_array)))

        inputs = feature_extractor(
            audio_array,
            sampling_rate=sampling_rate,
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        return inputs

    def sliding_window_segmentation(
        self,
        audio_duration: float,
        window_size: float,
        hop_size: float,
    ) -> List[Tuple[float, float]]:
        """
        Generate overlapping chunks using sliding window segmentation.

        Args:
            audio_duration: Total duration in seconds
            window_size: Window size in seconds
            hop_size: Hop size in seconds

        Returns:
            List of (start_time, end_time) tuples
        """
        chunks: List[Tuple[float, float]] = []
        start_time = 0.0

        while start_time + window_size <= audio_duration:
            end_time = start_time + window_size
            chunks.append((start_time, end_time))
            start_time += hop_size

        if start_time < audio_duration:
            end_time = min(start_time + window_size, audio_duration)
            chunks.append((start_time, end_time))

        return chunks

    def group_emotion_ranges(self, emotion_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Group consecutive segments with the same emotion into time ranges.

        Args:
            emotion_segments: List of segment dictionaries

        Returns:
            List of grouped emotion ranges
        """
        if not emotion_segments:
            return []

        emotion_ranges = []
        current_emotion = emotion_segments[0]["emotion"]
        start_time = emotion_segments[0]["start_time"]

        for index, segment in enumerate(emotion_segments[1:], 1):
            if segment["emotion"] != current_emotion:
                end_time = emotion_segments[index - 1]["end_time"]
                emotion_ranges.append(
                    {
                        "start": start_time,
                        "end": end_time,
                        "emotion": current_emotion,
                    }
                )
                current_emotion = segment["emotion"]
                start_time = segment["start_time"]

        emotion_ranges.append(
            {
                "start": start_time,
                "end": emotion_segments[-1]["end_time"],
                "emotion": current_emotion,
            }
        )

        return emotion_ranges