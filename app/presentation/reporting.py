"""Detailed reporting utilities for emotion analysis."""

from app.domain.schemas import EmotionAnalysisResult
from app.core.config import settings


class Reporting:
    """Provides detailed CLI reporting for emotion analysis."""

    def print_raw_vs_smoothed(self, result: EmotionAnalysisResult) -> None:
        """
        Print comparison between raw and smoothed predictions.
        """
        segments = result.emotion_segments

        if not segments:
            return

        # Check if smoothing exists
        if segments[0].emotion_raw is None:
            return

        print("\n" + "-" * settings.print_width)
        print("RAW vs SMOOTHED COMPARISON:")
        print("-" * settings.print_width)

        print(f"{'Chunk':<8} {'Raw Emotion':<15} {'Confidence':<12} {'Smoothed':<15}")
        print("-" * settings.print_width)

        for segment in segments:
            raw = segment.emotion_raw or "N/A"
            smoothed = segment.emotion
            confidence = segment.confidence

            marker = " ✓" if raw == smoothed else " →"

            print(
                f"{segment.chunk_id:<8} "
                f"{raw:<15} "
                f"{confidence:<12.3f} "
                f"{smoothed:<15}{marker}"
            )

    def print_smoothing_stats(self, result: EmotionAnalysisResult) -> None:
        """
        Print statistics about smoothing effectiveness.
        """
        segments = result.emotion_segments

        if not segments:
            return

        if segments[0].emotion_raw is None:
            return

        raw_emotions = [seg.emotion_raw for seg in segments]
        smoothed_emotions = [seg.emotion for seg in segments]

        changes = sum(
            1 for raw, smooth in zip(raw_emotions, smoothed_emotions)
            if raw != smooth
        )

        print("\n" + "-" * settings.print_width)
        print("SMOOTHING STATISTICS:")
        print("-" * settings.print_width)

        print(f"Total chunks analyzed: {len(segments)}")
        print(
            f"Predictions changed by smoothing: "
            f"{changes}/{len(segments)} ({100 * changes // len(segments)}%)"
        )

        # Count raw transitions
        raw_transitions = sum(
            1 for i in range(1, len(raw_emotions))
            if raw_emotions[i] != raw_emotions[i - 1]
        )

        smoothed_transitions = result.summary.total_transitions

        print(f"Raw transitions: {raw_transitions}")
        print(f"Smoothed transitions: {smoothed_transitions}")

        if result.metadata.smoothing_method:
            print(f"Smoothing method: {result.metadata.smoothing_method}")

    def print_detailed_analysis(self, result: EmotionAnalysisResult) -> None:
        """
        Print full detailed analysis.
        """
        self.print_raw_vs_smoothed(result)
        self.print_smoothing_stats(result)