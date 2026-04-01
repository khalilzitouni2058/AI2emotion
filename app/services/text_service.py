"""Service layer for text emotion analysis."""

from datetime import datetime
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from app.core.config import settings
from app.domain.schemas import (
    AnalysisMetadata,
    AnalysisSummary,
    EmotionAnalysisResult,
    EmotionSegment,
)
from app.infrastructure.text_model_provider import TextModelProvider
from app.services.emotion_mapper import EmotionMapper


class TextService:
    """Provides text emotion analysis using a multilingual text model."""

    def __init__(
        self,
        text_model_provider: Optional[TextModelProvider] = None,
        emotion_mapper: Optional[EmotionMapper] = None,
    ) -> None:
        self.text_model_provider = text_model_provider or TextModelProvider()
        self.emotion_mapper = emotion_mapper or EmotionMapper()

    def _validate_text(self, text: str) -> str:
        """
        Validate and normalize input text.

        Args:
            text: Raw input text

        Returns:
            Cleaned text

        Raises:
            ValueError: If text is empty after normalization
        """
        if text is None:
            raise ValueError("Text input cannot be None.")

        cleaned_text = text.strip()
        if not cleaned_text:
            raise ValueError("Text input cannot be empty.")

        return cleaned_text

    def _predict_raw_probabilities(self, text: str) -> Dict[str, float]:
        """
        Run text model inference and return raw label probabilities.

        Args:
            text: Input text

        Returns:
            Mapping of raw model labels to probabilities
        """
        resources = self.text_model_provider.get_resources()

        encoded_inputs = resources.tokenizer(
            text,
            truncation=True,
            max_length=settings.max_text_length,
            return_tensors="pt",
        )
        encoded_inputs = {
            key: value.to(resources.device)
            for key, value in encoded_inputs.items()
        }

        with torch.no_grad():
            outputs = resources.model(**encoded_inputs)

        probabilities = F.softmax(outputs.logits, dim=-1)[0].detach().cpu()

        raw_probabilities = {
            resources.id2label[index]: float(probabilities[index].item())
            for index in range(len(probabilities))
        }

        return raw_probabilities

    def analyze_text(self, text: str, source_name: Optional[str] = None) -> EmotionAnalysisResult:
        """
        Analyze emotion from text input.

        Args:
            text: Input text
            source_name: Optional display/source name

        Returns:
            Structured emotion analysis result
        """
        cleaned_text = self._validate_text(text)
        raw_probabilities = self._predict_raw_probabilities(cleaned_text)

        aggregated_probabilities = self.emotion_mapper.aggregate_probabilities(
            raw_probabilities
        )
        top_emotion = self.emotion_mapper.get_top_emotion(raw_probabilities)
        top_confidence = float(aggregated_probabilities[top_emotion])

        segment = EmotionSegment(
            chunk_id=1,
            start_time=0.0,
            end_time=0.0,
            timestamp=0.0,
            duration=0.0,
            emotion=top_emotion,
            confidence=round(top_confidence, 3),
            probabilities={
                emotion: round(probability, 3)
                for emotion, probability in aggregated_probabilities.items()
            },
        )

        metadata = AnalysisMetadata(
            input_type="text",
            analysis_timestamp=datetime.now().isoformat(),
            source_name=source_name,
            processing_mode="text",
            smoothing_method=None,
            text_length=len(cleaned_text),
        )

        summary = AnalysisSummary(
            total_transitions=0,
            emotion_distribution={top_emotion: 1},
        )

        return EmotionAnalysisResult(
            metadata=metadata,
            emotion_segments=[segment],
            emotion_transitions=[],
            summary=summary,
        )