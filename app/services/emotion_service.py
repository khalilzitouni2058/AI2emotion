"""Core application service for emotion analysis."""

from datetime import datetime
import logging
from time import perf_counter
from typing import Callable, Dict, List, Optional

import torch
import torch.nn.functional as F

from app.core.config import settings
from app.domain.schemas import (
    AnalysisMetadata,
    AnalysisSummary,
    EmotionAnalysisResult,
    EmotionSegment,
    EmotionTransition,
)
from app.infrastructure.model_provider import ModelProvider
from app.services.audio_service import AudioService
from app.services.smoothing_service import SmoothingService
from app.services.text_service import TextService


logger = logging.getLogger(__name__)

ChunkProgressCallback = Callable[[int, int, float, float, str, float], None]
StageCallback = Callable[[str], None]


class EmotionService:
    """Main application service responsible for end-to-end emotion analysis."""

    def __init__(
        self,
        model_provider: Optional[ModelProvider] = None,
        audio_service: Optional[AudioService] = None,
        smoothing_service: Optional[SmoothingService] = None,
        text_service: Optional[TextService] = None,
    ) -> None:
        self.model_provider = model_provider or ModelProvider()
        self.audio_service = audio_service or AudioService()
        self.smoothing_service = smoothing_service or SmoothingService()
        self.text_service = text_service or TextService()

    @staticmethod
    def _log_timing(stage: str, elapsed_seconds: float, **details) -> None:
        extra = ""
        if details:
            extra = " | " + ", ".join(f"{key}={value}" for key, value in details.items())
        logger.info("[EmotionService] %s took %.3fs%s", stage, elapsed_seconds, extra)

    def _resolve_effective_smoothing_method(
        self,
        applied_method: Optional[str],
        prediction_count: int,
    ) -> str:
        """Resolve the smoothing method actually used for this analysis run."""
        if prediction_count <= 1:
            return "none"

        if applied_method:
            return applied_method

        if settings.smoothing_mode == "offline":
            return settings.offline_smoothing_method

        if settings.smoothing_mode == "streaming":
            if (
                settings.streaming_smoothing_method == "ema"
                and settings.streaming_use_hysteresis
            ):
                return "ema+hysteresis"
            return settings.streaming_smoothing_method

        return settings.smoothing_method or "none"

    def _probs_to_dict(self, probabilities: torch.Tensor) -> Dict[str, float]:
        """Convert probability tensor to label-probability mapping."""
        id2label = self.model_provider.get_id2label()
        return {
            id2label[index]: round(float(probabilities[index].item()), 3)
            for index in range(len(probabilities))
        }

    def _predict_probabilities(
        self,
        audio_array,
        sampling_rate: int,
    ) -> torch.Tensor:
        """Run inference and return probabilities for a single audio array."""
        resources = self.model_provider.get_resources()
        max_audio_samples = getattr(
            resources.feature_extractor,
            "n_samples",
            int(resources.feature_extractor.sampling_rate * settings.max_audio_duration),
        )

        feature_start = perf_counter()
        inputs = resources.feature_extractor(
            audio_array,
            sampling_rate=sampling_rate,
            padding="max_length",
            truncation=True,
            max_length=max_audio_samples,
            return_tensors="pt",
        )
        self._log_timing(
            "feature_extraction",
            perf_counter() - feature_start,
            samples=len(audio_array),
        )

        inputs = {key: value.to(resources.device) for key, value in inputs.items()}

        inference_start = perf_counter()
        if resources.device.type == "cuda":
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = resources.model(**inputs)
            # Synchronize GPU operations before CPU transfer to avoid blocking
            torch.cuda.synchronize()
        else:
            with torch.inference_mode():
                outputs = resources.model(**inputs)
        self._log_timing("model_inference", perf_counter() - inference_start)

        probabilities = F.softmax(outputs.logits.float(), dim=-1)[0].detach().cpu()
        return probabilities

    def _predict_probabilities_batch(
        self,
        audio_arrays,
        sampling_rate: int,
    ) -> List[torch.Tensor]:
        """Run inference for a batch of audio arrays and return probabilities per item."""
        resources = self.model_provider.get_resources()
        max_audio_samples = getattr(
            resources.feature_extractor,
            "n_samples",
            int(resources.feature_extractor.sampling_rate * settings.max_audio_duration),
        )

        feature_start = perf_counter()
        inputs = resources.feature_extractor(
            audio_arrays,
            sampling_rate=sampling_rate,
            padding="max_length",
            truncation=True,
            max_length=max_audio_samples,
            return_tensors="pt",
        )
        self._log_timing(
            "feature_extraction_batch",
            perf_counter() - feature_start,
            segments=len(audio_arrays),
        )

        inputs = {key: value.to(resources.device) for key, value in inputs.items()}

        inference_start = perf_counter()
        if resources.device.type == "cuda":
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = resources.model(**inputs)
            # Synchronize GPU operations before CPU transfer to avoid blocking
            torch.cuda.synchronize()
        else:
            with torch.inference_mode():
                outputs = resources.model(**inputs)
        self._log_timing(
            "model_inference_batch",
            perf_counter() - inference_start,
            segments=len(audio_arrays),
        )

        probabilities = F.softmax(outputs.logits.float(), dim=-1).detach().cpu()
        return [probabilities[index] for index in range(probabilities.shape[0])]

    def _average_subwindow_probabilities(
        self,
        segment_audio,
        sampling_rate: int,
    ) -> Optional[torch.Tensor]:
        """Average probabilities across sub-windows inside a chunk."""
        segment_duration = len(segment_audio) / sampling_rate

        if segment_duration <= 0:
            return None

        if segment_duration <= settings.sub_window_size:
            return self._predict_probabilities(segment_audio, sampling_rate)

        subwindows = self.audio_service.sliding_window_segmentation(
            audio_duration=segment_duration,
            window_size=settings.sub_window_size,
            hop_size=settings.sub_hop_size,
        )

        if not subwindows:
            return self._predict_probabilities(segment_audio, sampling_rate)

        sub_audio_segments = []

        for start_time, end_time in subwindows:
            start_sample = int(start_time * sampling_rate)
            end_sample = int(end_time * sampling_rate)
            sub_audio = segment_audio[start_sample:end_sample]

            if len(sub_audio) > 0:
                sub_audio_segments.append(sub_audio)

        if not sub_audio_segments:
            return self._predict_probabilities(segment_audio, sampling_rate)

        probabilities_list = self._predict_probabilities_batch(
            sub_audio_segments,
            sampling_rate,
        )

        stacked = torch.stack(probabilities_list, dim=0)
        return torch.mean(stacked, dim=0)

    def predict_emotion_full(
        self,
        audio_path: str,
        audio_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        audio_array=None,
        sampling_rate: Optional[int] = None,
    ) -> EmotionAnalysisResult:
        """
        Predict emotion for a short audio file using full-audio inference.
        """
        resources = self.model_provider.get_resources()

        if max_duration is None:
            max_duration = settings.max_audio_duration

        preprocess_start = perf_counter()
        inputs = self.audio_service.preprocess_audio(
            audio_path=audio_path,
            feature_extractor=resources.feature_extractor,
            max_duration=max_duration,
            audio_array=audio_array,
            sampling_rate=sampling_rate,
        )
        self._log_timing("audio_preprocessing", perf_counter() - preprocess_start)

        inputs = {key: value.to(resources.device) for key, value in inputs.items()}

        inference_start = perf_counter()
        if resources.device.type == "cuda":
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = resources.model(**inputs)
        else:
            with torch.inference_mode():
                outputs = resources.model(**inputs)
        self._log_timing("model_inference", perf_counter() - inference_start)

        logits = outputs.logits
        probabilities = F.softmax(logits.float(), dim=-1)[0].detach().cpu()
        predicted_id = torch.argmax(logits, dim=-1).item()

        emotion = resources.id2label[predicted_id]
        confidence = float(probabilities[predicted_id].item())

        if audio_duration is None:
            audio_duration, _, _ = self.audio_service.get_audio_duration(audio_path)

        segment = EmotionSegment(
            chunk_id=1,
            start_time=0.0,
            end_time=round(audio_duration, 3),
            timestamp=round(audio_duration / 2.0, 3),
            duration=round(audio_duration, 3),
            emotion=emotion,
            confidence=round(confidence, 3),
            probabilities=self._probs_to_dict(probabilities),
        )

        metadata = AnalysisMetadata(
            input_type="audio",
            analysis_timestamp=datetime.now().isoformat(),
            source_name=audio_path,
            processing_mode="full_audio",
            smoothing_method="none",
            total_duration=round(audio_duration, 3),
            total_chunks=1,
            window_size=round(audio_duration, 3),
            hop_size=round(audio_duration, 3),
            sampling_rate=settings.audio_sampling_rate,
            sub_window_size=settings.sub_window_size,
            sub_hop_size=settings.sub_hop_size,
        )

        summary = AnalysisSummary(
            total_transitions=0,
            emotion_distribution={emotion: 1},
        )

        return EmotionAnalysisResult(
            metadata=metadata,
            emotion_segments=[segment],
            emotion_transitions=[],
            summary=summary,
        )

    def analyze_emotion_over_time(
        self,
        audio_array,
        sampling_rate: int,
        audio_duration: float,
        audio_path: Optional[str] = None,
        window_size: Optional[float] = None,
        hop_size: Optional[float] = None,
        progress_callback: Optional[ChunkProgressCallback] = None,
        stage_callback: Optional[StageCallback] = None,
        force_direct_chunk_batch: bool = False,
        skip_smoothing: bool = False,
    ) -> EmotionAnalysisResult:
        """
        Analyze emotion changes over time using sliding-window segmentation.
        """
        if window_size is None:
            window_size = settings.chunk_window_size

        if hop_size is None:
            hop_size = settings.chunk_hop_size

        chunks = self.audio_service.sliding_window_segmentation(
            audio_duration=audio_duration,
            window_size=window_size,
            hop_size=hop_size,
        )

        if stage_callback is not None:
            stage_callback(f"Chunk windows ready: {len(chunks)}")

        predictions: List[EmotionSegment] = []
        raw_emotions: List[str] = []
        confidences: List[float] = []
        avg_probs_list: List[torch.Tensor] = []

        id2label = self.model_provider.get_id2label()
        total_chunks = len(chunks)

        if progress_callback is not None and total_chunks > 0:
            # Emit an immediate progress event so CLI shows activity before first inference.
            progress_callback(0, total_chunks, 0.0, 0.0, "initializing", 0.0)

        def _append_prediction(
            chunk_index: int,
            start_time: float,
            end_time: float,
            center_time: float,
            avg_probs: torch.Tensor,
        ) -> None:
            predicted_id = int(torch.argmax(avg_probs).item())
            emotion = id2label[predicted_id]
            confidence = float(avg_probs[predicted_id].item())

            raw_emotions.append(emotion)
            confidences.append(confidence)
            avg_probs_list.append(avg_probs)

            predictions.append(
                EmotionSegment(
                    chunk_id=chunk_index + 1,
                    start_time=round(start_time, 3),
                    end_time=round(end_time, 3),
                    timestamp=round(center_time, 3),
                    duration=round(end_time - start_time, 3),
                    emotion=emotion,
                    confidence=round(confidence, 3),
                    probabilities=self._probs_to_dict(avg_probs),
                )
            )

            if progress_callback is not None:
                progress_callback(
                    chunk_index + 1,
                    total_chunks,
                    round(start_time, 3),
                    round(end_time, 3),
                    emotion,
                    round(confidence, 3),
                )

        use_direct_chunk_batch = force_direct_chunk_batch or (
            settings.sub_window_size >= window_size
            and settings.sub_hop_size >= settings.sub_window_size
        )

        if use_direct_chunk_batch:
            if stage_callback is not None:
                stage_callback("Using batched chunk inference")

            chunk_payloads = []

            for index, (start_time, end_time) in enumerate(chunks):
                start_sample = int(start_time * sampling_rate)
                end_sample = int(end_time * sampling_rate)
                segment_audio = audio_array[start_sample:end_sample]
                center_time = start_time + ((end_time - start_time) / 2.0)

                if len(segment_audio) == 0:
                    continue

                chunk_payloads.append(
                    (index, start_time, end_time, center_time, segment_audio)
                )

            batch_size = max(1, settings.chunk_inference_batch_size)
            total_batches = (len(chunk_payloads) + batch_size - 1) // batch_size
            for batch_index, batch_start in enumerate(
                range(0, len(chunk_payloads), batch_size),
                start=1,
            ):
                batch = chunk_payloads[batch_start: batch_start + batch_size]
                if stage_callback is not None:
                    stage_callback(
                        f"Running chunk batch {batch_index}/{total_batches} "
                        f"(size={len(batch)})"
                    )

                batch_audio_arrays = [payload[4] for payload in batch]

                try:
                    batch_probs = self._predict_probabilities_batch(
                        batch_audio_arrays,
                        sampling_rate,
                    )
                except RuntimeError as exc:
                    logger.warning(
                        "Batch inference failed, falling back to per-chunk inference: %s",
                        exc,
                    )
                    if stage_callback is not None:
                        stage_callback(
                            "Batch inference fallback: switching to per-chunk inference"
                        )

                    for payload in batch:
                        avg_probs = self._predict_probabilities(
                            payload[4],
                            sampling_rate,
                        )
                        _append_prediction(
                            chunk_index=payload[0],
                            start_time=payload[1],
                            end_time=payload[2],
                            center_time=payload[3],
                            avg_probs=avg_probs,
                        )
                    continue

                for payload, avg_probs in zip(batch, batch_probs):
                    _append_prediction(
                        chunk_index=payload[0],
                        start_time=payload[1],
                        end_time=payload[2],
                        center_time=payload[3],
                        avg_probs=avg_probs,
                    )
        else:
            if stage_callback is not None:
                stage_callback("Using subwindow chunk inference")

            for index, (start_time, end_time) in enumerate(chunks):
                start_sample = int(start_time * sampling_rate)
                end_sample = int(end_time * sampling_rate)
                segment_audio = audio_array[start_sample:end_sample]
                center_time = start_time + ((end_time - start_time) / 2.0)

                if len(segment_audio) == 0:
                    continue

                avg_probs = self._average_subwindow_probabilities(
                    segment_audio=segment_audio,
                    sampling_rate=sampling_rate,
                )

                if avg_probs is None:
                    continue

                _append_prediction(
                    chunk_index=index,
                    start_time=start_time,
                    end_time=end_time,
                    center_time=center_time,
                    avg_probs=avg_probs,
                )

        smoothing_method_used = None

        if skip_smoothing:
            smoothing_method_used = "none"

        if not skip_smoothing and len(raw_emotions) > 1:
            if settings.smoothing_mode == "offline":
                if settings.offline_smoothing_method == "interpolate":
                    smoothed_probs = self.smoothing_service.interpolate_probabilities(
                        avg_probs_list
                    )

                    for index, segment in enumerate(predictions):
                        segment.emotion_raw = segment.emotion
                        segment.confidence_raw = segment.confidence
                        segment.probabilities_raw = dict(segment.probabilities)

                        predicted_id = int(torch.argmax(smoothed_probs[index]).item())
                        segment.emotion = id2label[predicted_id]
                        segment.confidence = round(
                            float(smoothed_probs[index][predicted_id].item()),
                            3,
                        )
                        segment.probabilities = self._probs_to_dict(
                            smoothed_probs[index]
                        )

                    smoothing_method_used = "interpolate"

                elif settings.offline_smoothing_method == "median":
                    smoothed_emotions = self.smoothing_service.apply_smoothing(
                        raw_emotions,
                        method="median",
                        window_size=settings.smoothing_window_size,
                    )

                    for index, segment in enumerate(predictions):
                        segment.emotion_raw = segment.emotion
                        segment.emotion = smoothed_emotions[index]

                    smoothing_method_used = "median"

            elif settings.smoothing_mode == "streaming":
                if settings.streaming_smoothing_method == "ema":
                    smoothed_emotions = self.smoothing_service.apply_smoothing(
                        raw_emotions,
                        method="ema",
                        alpha=settings.smoothing_ema_alpha,
                    )

                    if settings.streaming_use_hysteresis:
                        smoothed_emotions = self.smoothing_service.hysteresis_filter(
                            smoothed_emotions,
                            confidences,
                            confidence_threshold=settings.smoothing_confidence_threshold,
                            min_consecutive_frames=settings.smoothing_min_frames,
                        )
                        smoothing_method_used = "ema+hysteresis"
                    else:
                        smoothing_method_used = "ema"

                    for index, segment in enumerate(predictions):
                        segment.emotion_raw = segment.emotion
                        segment.emotion = smoothed_emotions[index]

            else:
                if settings.smoothing_method:
                    smoothed_emotions = self.smoothing_service.apply_smoothing(
                        raw_emotions,
                        confidences=confidences,
                        method=settings.smoothing_method,
                        window_size=settings.smoothing_window_size,
                        alpha=settings.smoothing_ema_alpha,
                        confidence_threshold=settings.smoothing_confidence_threshold,
                        min_consecutive_frames=settings.smoothing_min_frames,
                    )

                    for index, segment in enumerate(predictions):
                        segment.emotion_raw = segment.emotion
                        segment.emotion = smoothed_emotions[index]

                    smoothing_method_used = settings.smoothing_method

        transitions: List[EmotionTransition] = []

        for index in range(1, len(predictions)):
            previous_segment = predictions[index - 1]
            current_segment = predictions[index]

            if current_segment.emotion != previous_segment.emotion:
                transitions.append(
                    EmotionTransition(
                        transition_id=len(transitions) + 1,
                        from_emotion=previous_segment.emotion,
                        to_emotion=current_segment.emotion,
                        transition_time=current_segment.timestamp,
                        previous_segment=previous_segment.chunk_id,
                        current_segment=current_segment.chunk_id,
                    )
                )

        emotion_distribution: Dict[str, int] = {}
        for segment in predictions:
            emotion_distribution[segment.emotion] = (
                emotion_distribution.get(segment.emotion, 0) + 1
            )

        metadata = AnalysisMetadata(
            input_type="audio",
            analysis_timestamp=datetime.now().isoformat(),
            source_name=audio_path,
            processing_mode="chunked",
            smoothing_method=self._resolve_effective_smoothing_method(
                smoothing_method_used,
                len(predictions),
            ),
            total_duration=round(audio_duration, 3),
            total_chunks=len(predictions),
            window_size=window_size,
            hop_size=hop_size,
            sampling_rate=sampling_rate,
            sub_window_size=settings.sub_window_size,
            sub_hop_size=settings.sub_hop_size,
        )

        summary = AnalysisSummary(
            total_transitions=len(transitions),
            emotion_distribution=emotion_distribution,
        )

        return EmotionAnalysisResult(
            metadata=metadata,
            emotion_segments=predictions,
            emotion_transitions=transitions,
            summary=summary,
        )

    def analyze_audio_fast(
        self,
        audio_path: str,
        progress_callback: Optional[ChunkProgressCallback] = None,
        stage_callback: Optional[StageCallback] = None,
    ) -> EmotionAnalysisResult:
        """Low-latency audio analysis profile for API clients."""
        if stage_callback is not None:
            stage_callback("Preparing audio")

        load_start = perf_counter()
        audio_duration, audio_array, sampling_rate = self.audio_service.get_audio_duration(
            audio_path
        )
        self._log_timing(
            "audio_loading",
            perf_counter() - load_start,
            duration=round(audio_duration, 3),
            profile="fast",
        )

        if stage_callback is not None:
            stage_callback(f"Audio loaded ({audio_duration:.2f}s)")

        if audio_duration < settings.audio_duration_threshold:
            if stage_callback is not None:
                stage_callback("Running full-audio inference")
            return self.predict_emotion_full(
                audio_path=audio_path,
                audio_duration=audio_duration,
                audio_array=audio_array,
                sampling_rate=sampling_rate,
            )

        fast_window = max(settings.chunk_window_size, 4.0)
        fast_hop = max(settings.chunk_hop_size, 3.0)

        if stage_callback is not None:
            stage_callback(
                f"Running fast chunk profile (window={fast_window:.1f}, hop={fast_hop:.1f})"
            )

        chunk_start = perf_counter()
        result = self.analyze_emotion_over_time(
            audio_array=audio_array,
            sampling_rate=sampling_rate,
            audio_duration=audio_duration,
            audio_path=audio_path,
            window_size=fast_window,
            hop_size=fast_hop,
            progress_callback=progress_callback,
            stage_callback=stage_callback,
            force_direct_chunk_batch=True,
            skip_smoothing=True,
        )
        self._log_timing(
            "chunked_analysis",
            perf_counter() - chunk_start,
            chunks=len(result.emotion_segments),
            profile="fast",
        )
        return result

    def analyze_text(self, text: str) -> EmotionAnalysisResult:
        """
        Analyze emotion from text input.
        """
        return self.text_service.analyze_text(text)

    def analyze_audio(
        self,
        audio_path: str,
        progress_callback: Optional[ChunkProgressCallback] = None,
        stage_callback: Optional[StageCallback] = None,
    ) -> EmotionAnalysisResult:
        """
        Analyze emotion from audio input.
        """
        if stage_callback is not None:
            stage_callback("Preparing audio")

        load_start = perf_counter()
        audio_duration, audio_array, sampling_rate = self.audio_service.get_audio_duration(
            audio_path
        )
        self._log_timing("audio_loading", perf_counter() - load_start, duration=round(audio_duration, 3))

        if stage_callback is not None:
            stage_callback(f"Audio loaded ({audio_duration:.2f}s)")

        if audio_duration < settings.audio_duration_threshold:
            if stage_callback is not None:
                stage_callback("Running full-audio inference")
            return self.predict_emotion_full(
                audio_path=audio_path,
                audio_duration=audio_duration,
                audio_array=audio_array,
                sampling_rate=sampling_rate,
            )

        chunk_start = perf_counter()
        if stage_callback is not None:
            stage_callback("Building chunk timeline")

        result = self.analyze_emotion_over_time(
            audio_array=audio_array,
            sampling_rate=sampling_rate,
            audio_duration=audio_duration,
            audio_path=audio_path,
            window_size=settings.chunk_window_size,
            hop_size=settings.chunk_hop_size,
            progress_callback=progress_callback,
            stage_callback=stage_callback,
        )
        self._log_timing("chunked_analysis", perf_counter() - chunk_start, chunks=len(result.emotion_segments))
        return result

    def analyze(
        self,
        audio_path: Optional[str] = None,
        text: Optional[str] = None,
        progress_callback: Optional[ChunkProgressCallback] = None,
        stage_callback: Optional[StageCallback] = None,
    ) -> EmotionAnalysisResult:
        """
        Main unified entrypoint for emotion analysis.

        Supports:
        - audio input
        - text input

        Args:
            audio_path: Path to audio file
            text: Input text

        Returns:
            EmotionAnalysisResult
        """
        if audio_path and text:
            raise ValueError("Provide either audio_path or text, not both.")

        if not audio_path and not text:
            raise ValueError("You must provide either audio_path or text.")

        if text is not None:
            return self.analyze_text(text)

        return self.analyze_audio(
            audio_path,
            progress_callback=progress_callback,
            stage_callback=stage_callback,
        )