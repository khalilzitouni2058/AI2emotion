"""CLI entrypoint for emotion analysis application."""

import time

from app.services.emotion_service import EmotionService
from app.presentation.cli_formatter import CLIFormatter
from app.presentation.reporting import Reporting
from app.infrastructure.file_storage import FileStorage


def main():
    # === CONFIGURE INPUT ===
    audio_path = "c:\\Users\\kzito\\Downloads\\mike_tyson.wav"  # 🔁 Change this to your test file

    # === INITIALIZE COMPONENTS ===
    emotion_service = EmotionService()
    formatter = CLIFormatter()
    reporter = Reporting()
    storage = FileStorage()

    print("[Init] Warming up audio model...")
    emotion_service.model_provider.get_resources()
    print("[Init] Model warmup complete")

    stage_state = {"current_stage": None, "stage_start": None}

    def on_chunk_processed(
        chunk_index: int,
        total_chunks: int,
        start_time: float,
        end_time: float,
        emotion: str,
        confidence: float,
    ) -> None:
        if chunk_index == 0:
            print(f"[Chunk 0/{total_chunks}] initializing inference...")
            return

        print(
            f"[Chunk {chunk_index}/{total_chunks}] "
            f"{start_time:.2f}s-{end_time:.2f}s -> {emotion} ({confidence:.3f})"
        )

    def on_stage_update(message: str) -> None:
        now = time.perf_counter()
        current_stage = stage_state["current_stage"]
        stage_start = stage_state["stage_start"]

        if current_stage is not None and stage_start is not None:
            elapsed = now - stage_start
            print(f"[Stage Complete] {current_stage} ({elapsed:.2f}s)")

        stage_state["current_stage"] = message
        stage_state["stage_start"] = now
        print(f"[Stage] {message}")

    # === RUN ANALYSIS ===
    result = emotion_service.analyze(
        audio_path=audio_path,
        progress_callback=on_chunk_processed,
        stage_callback=on_stage_update,
    )

    # Print duration for the final stage, which has no subsequent stage transition.
    if stage_state["current_stage"] is not None and stage_state["stage_start"] is not None:
        final_elapsed = time.perf_counter() - stage_state["stage_start"]
        print(f"[Stage Complete] {stage_state['current_stage']} ({final_elapsed:.2f}s)")

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