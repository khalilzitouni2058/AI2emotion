"""Service layer for smoothing and denoising emotion predictions."""

from collections import Counter
from typing import List, Optional

import torch


class SmoothingService:
    """Provides smoothing methods for emotion predictions."""

    def exponential_moving_average_filter(
        self,
        predictions: List[str],
        alpha: float = 0.3,
    ) -> List[str]:
        """
        Apply exponential moving average smoothing to emotion predictions.
        """
        if not predictions:
            return predictions

        if len(predictions) == 1:
            return predictions

        smoothed = [predictions[0]]

        for index in range(1, len(predictions)):
            window_size = min(5, index + 1)
            window = predictions[max(0, index - window_size + 1): index + 1]

            weighted_counts = {}
            for offset, emotion in enumerate(window):
                weight = alpha ** (len(window) - 1 - offset)
                weighted_counts[emotion] = weighted_counts.get(emotion, 0) + weight

            smoothed_emotion = max(weighted_counts, key=weighted_counts.get)
            smoothed.append(smoothed_emotion)

        return smoothed

    def median_filter(
        self,
        predictions: List[str],
        window_size: int = 3,
    ) -> List[str]:
        """
        Apply median-style majority-vote filtering.
        """
        if not predictions:
            return predictions

        if window_size < 1:
            return predictions

        if window_size % 2 == 0:
            window_size += 1

        filtered = []
        half_window = window_size // 2

        for index in range(len(predictions)):
            start = max(0, index - half_window)
            end = min(len(predictions), index + half_window + 1)
            window_emotions = predictions[start:end]
            most_common = Counter(window_emotions).most_common(1)[0][0]
            filtered.append(most_common)

        return filtered

    def hysteresis_filter(
        self,
        predictions: List[str],
        confidences: List[float],
        confidence_threshold: float = 0.7,
        min_consecutive_frames: int = 2,
    ) -> List[str]:
        """
        Only switch emotion if the new emotion is confident and sustained.
        """
        if not predictions or not confidences:
            return predictions

        if len(predictions) != len(confidences):
            return predictions

        filtered = [predictions[0]]
        current_emotion = predictions[0]
        switch_candidate: Optional[str] = None
        switch_count = 0

        for index in range(1, len(predictions)):
            emotion = predictions[index]
            confidence = confidences[index]

            if emotion == current_emotion:
                filtered.append(emotion)
                switch_candidate = None
                switch_count = 0

            elif confidence >= confidence_threshold:
                if emotion == switch_candidate:
                    switch_count += 1
                    if switch_count >= min_consecutive_frames - 1:
                        current_emotion = emotion
                        switch_candidate = None
                        switch_count = 0
                else:
                    switch_candidate = emotion
                    switch_count = 1

                filtered.append(current_emotion)

            else:
                filtered.append(current_emotion)
                switch_candidate = None
                switch_count = 0

        return filtered

    def interpolate_probabilities(
        self,
        probabilities_list: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        Linearly interpolate probabilities between keyframes.
        """
        if not probabilities_list:
            return probabilities_list

        if len(probabilities_list) == 1:
            return probabilities_list

        argmaxes = [int(torch.argmax(probabilities).item()) for probabilities in probabilities_list]
        keyframes = [0]

        for index in range(1, len(argmaxes)):
            if argmaxes[index] != argmaxes[index - 1]:
                keyframes.append(index)

        if keyframes[-1] != len(probabilities_list) - 1:
            keyframes.append(len(probabilities_list) - 1)

        smoothed = [probabilities.clone() for probabilities in probabilities_list]

        for keyframe_index in range(len(keyframes) - 1):
            start_idx = keyframes[keyframe_index]
            end_idx = keyframes[keyframe_index + 1]

            if end_idx == start_idx:
                continue

            start_prob = probabilities_list[start_idx]
            end_prob = probabilities_list[end_idx]
            span = end_idx - start_idx

            for index in range(start_idx + 1, end_idx):
                interpolation_ratio = (index - start_idx) / span
                smoothed[index] = (
                    (1.0 - interpolation_ratio) * start_prob
                    + interpolation_ratio * end_prob
                )

        return smoothed

    def apply_smoothing(
        self,
        predictions: List[str],
        confidences: Optional[List[float]] = None,
        method: str = "median",
        **kwargs,
    ) -> List[str]:
        """
        Apply the selected smoothing method.
        """
        if method == "ema":
            alpha = kwargs.get("alpha", 0.3)
            return self.exponential_moving_average_filter(predictions, alpha=alpha)

        if method == "median":
            window_size = kwargs.get("window_size", 3)
            return self.median_filter(predictions, window_size=window_size)

        if method == "hysteresis":
            if confidences is None:
                raise ValueError("Confidences are required for hysteresis smoothing.")

            confidence_threshold = kwargs.get("confidence_threshold", 0.7)
            min_frames = kwargs.get("min_consecutive_frames", 2)
            return self.hysteresis_filter(
                predictions,
                confidences,
                confidence_threshold=confidence_threshold,
                min_consecutive_frames=min_frames,
            )

        if method == "combined":
            smoothed = self.median_filter(
                predictions,
                window_size=kwargs.get("window_size", 3),
            )

            if confidences is not None:
                smoothed = self.hysteresis_filter(
                    smoothed,
                    confidences,
                    confidence_threshold=kwargs.get("confidence_threshold", 0.7),
                    min_consecutive_frames=kwargs.get("min_consecutive_frames", 2),
                )

            return smoothed

        return predictions