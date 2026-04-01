"""CLI entrypoint for emotion analysis application."""

from app.services.emotion_service import EmotionService
from app.presentation.cli_formatter import CLIFormatter
from app.presentation.reporting import Reporting
from app.infrastructure.file_storage import FileStorage


def main():
    # === CONFIGURE INPUT ===
    audio_path = "example.wav"  # 🔁 Change this to your test file

    # === INITIALIZE COMPONENTS ===
    emotion_service = EmotionService()
    formatter = CLIFormatter()
    reporter = Reporting()
    storage = FileStorage()

    # === RUN ANALYSIS ===
    result = emotion_service.analyze(audio_path)

    if result is None:
        print("❌ Analysis failed.")
        return

    # === DISPLAY RESULTS ===
    formatter.print_audio_info(result.metadata.total_duration)
    formatter.print_analysis_summary(result)
    formatter.print_emotion_ranges(result.emotion_segments)
    formatter.print_emotion_transitions(result)

    # Detailed reporting (optional but useful)
    reporter.print_detailed_analysis(result)

    # === SAVE OUTPUT (optional) ===
    output_path = audio_path.replace(".wav", "_analysis.json")
    storage.save_json(result, output_path)

    print(f"\n✅ Results saved to: {output_path}")


if __name__ == "__main__":
    main()