"""CLI formatting utilities for displaying emotion analysis results."""

from app.core.config import settings
from app.domain.schemas import EmotionAnalysisResult, EmotionSegment


class CLIFormatter:
    """Handles console output formatting."""

    def print_header(self, title: str) -> None:
        print("=" * settings.print_width)
        print(title)
        print("=" * settings.print_width)

    def print_audio_info(self, duration: float) -> None:
        self.print_header(f"Audio Duration: {duration:.2f} seconds")

    def print_analysis_summary(self, result: EmotionAnalysisResult) -> None:
        self.print_header("ANALYSIS SUMMARY")

        metadata = result.metadata
        summary = result.summary

        print(f"Total Duration: {metadata.total_duration:.2f} seconds")
        print(f"Total Chunks: {metadata.total_chunks}")
        print(f"Smoothing Method: {metadata.smoothing_method}")
        print(f"Emotion Transitions: {summary.total_transitions}")
        print(f"Emotion Distribution: {summary.emotion_distribution}")

    def print_emotion_ranges(self, segments: list[EmotionSegment]) -> None:
        """
        Print grouped emotion ranges.
        """
        if not segments:
            return

        print("\n" + "-" * settings.print_width)
        print("EMOTION TIME RANGES:")
        print("-" * settings.print_width)

        ranges = self._group_emotion_ranges(segments)

        for index, r in enumerate(ranges, 1):
            print(
                f"{index}. From {r['start']:.2f} to {r['end']:.2f} → {r['emotion']}"
            )

    def print_emotion_transitions(self, result: EmotionAnalysisResult, max_show: int = 10) -> None:
        transitions = result.emotion_transitions

        if not transitions:
            return

        print("\n" + "-" * settings.print_width)
        print("EMOTION TRANSITIONS:")
        print("-" * settings.print_width)

        for transition in transitions[:max_show]:
            print(
                f"{transition.transition_time:.2f}s: "
                f"{transition.from_emotion} → {transition.to_emotion}"
            )

        if len(transitions) > max_show:
            print(f"... and {len(transitions) - max_show} more transitions")

    def _group_emotion_ranges(self, segments: list[EmotionSegment]):
        """
        Internal helper to group consecutive segments with same emotion.
        """
        ranges = []

        current_emotion = segments[0].emotion
        start_time = segments[0].start_time

        for index, segment in enumerate(segments[1:], 1):
            if segment.emotion != current_emotion:
                ranges.append(
                    {
                        "start": start_time,
                        "end": segments[index - 1].end_time,
                        "emotion": current_emotion,
                    }
                )
                current_emotion = segment.emotion
                start_time = segment.start_time

        ranges.append(
            {
                "start": start_time,
                "end": segments[-1].end_time,
                "emotion": current_emotion,
            }
        )

        return ranges