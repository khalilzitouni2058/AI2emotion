"""Core application service for emotion analysis."""

from datetime import datetime
from typing import Dict, List, Optional

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


class EmotionService:
    """Main application service responsible for end-to-end emotion analysis."""

    def __init__(
        self,
        model_provider: Optional[ModelProvider] = None,
        audio_service: Optional[AudioService] = None,
        smoothing_service: Optional[SmoothingService] = None,
    ) -> None:
        self.model_provider = model_provider or ModelProvider()
        self.audio_service = audio_service or AudioService()
        self.smoothing_service = smoothing_service or SmoothingService()

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

        inputs = resources.feature_extractor(
            audio_array,
            sampling_rate=sampling_rate,
            return_tensors="pt",
        )
        inputs = {key: value.to(resources.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = resources.model(**inputs)

        probabilities = F.softmax(outputs.logits, dim=-1)[0].detach().cpu()
        return probabilities

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

        probabilities_list = []

        for start_time, end_time in subwindows:
            start_sample = int(start_time * sampling_rate)
            end_sample = int(end_time * sampling_rate)
            sub_audio = segment_audio[start_sample:end_sample]

            if len(sub_audio) > 0:
                probabilities_list.append(
                    self._predict_probabilities(sub_audio, sampling_rate)
                )

        if not probabilities_list:
            return self._predict_probabilities(segment_audio, sampling_rate)

        stacked = torch.stack(probabilities_list, dim=0)
        return torch.mean(stacked, dim=0)

    def predict_emotion_full(
        self,
        audio_path: str,
        audio_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
    ) -> EmotionAnalysisResult:
        """
        Predict emotion for a short audio file using full-audio inference.
        """
        resources = self.model_provider.get_resources()

        if max_duration is None:
            max_duration = settings.max_audio_duration

        inputs = self.audio_service.preprocess_audio(
            audio_path=audio_path,
            feature_extractor=resources.feature_extractor,
            max_duration=max_duration,
        )
        inputs = {key: value.to(resources.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = resources.model(**inputs)

        logits = outputs.logits
        probabilities = F.softmax(logits, dim=-1)[0].detach().cpu()
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
            audio_file=audio_path,
            total_duration=round(audio_duration, 3),
            total_chunks=1,
            window_size=round(audio_duration, 3),
            hop_size=round(audio_duration, 3),
            sampling_rate=settings.audio_sampling_rate,
            smoothing_method="none",
            analysis_timestamp=datetime.now().isoformat(),
            processing_mode="full_audio",
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

        predictions: List[EmotionSegment] = []
        raw_emotions: List[str] = []
        confidences: List[float] = []
        avg_probs_list: List[torch.Tensor] = []

        id2label = self.model_provider.get_id2label()

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

            avg_probs_list.append(avg_probs)

            predicted_id = int(torch.argmax(avg_probs).item())
            emotion = id2label[predicted_id]
            confidence = float(avg_probs[predicted_id].item())

            raw_emotions.append(emotion)
            confidences.append(confidence)

            predictions.append(
                EmotionSegment(
                    chunk_id=index + 1,
                    start_time=round(start_time, 3),
                    end_time=round(end_time, 3),
                    timestamp=round(center_time, 3),
                    duration=round(end_time - start_time, 3),
                    emotion=emotion,
                    confidence=round(confidence, 3),
                    probabilities=self._probs_to_dict(avg_probs),
                )
            )

        smoothing_method_used = None

        if len(raw_emotions) > 1:
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
            audio_file=audio_path,
            total_duration=round(audio_duration, 3),
            total_chunks=len(predictions),
            window_size=window_size,
            hop_size=hop_size,
            sampling_rate=sampling_rate,
            smoothing_method=smoothing_method_used or settings.smoothing_method,
            analysis_timestamp=datetime.now().isoformat(),
            processing_mode="chunked",
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

    def analyze(self, audio_path: str) -> EmotionAnalysisResult:
        """
        Main entrypoint for emotion analysis.

        Chooses between full-audio prediction and chunked analysis
        based on configured audio duration threshold.
        """
        audio_duration, audio_array, sampling_rate = self.audio_service.get_audio_duration(
            audio_path
        )

        if audio_duration < settings.audio_duration_threshold:
            return self.predict_emotion_full(
                audio_path=audio_path,
                audio_duration=audio_duration,
            )

        return self.analyze_emotion_over_time(
            audio_array=audio_array,
            sampling_rate=sampling_rate,
            audio_duration=audio_duration,
            audio_path=audio_path,
            window_size=settings.chunk_window_size,
            hop_size=settings.chunk_hop_size,
        )