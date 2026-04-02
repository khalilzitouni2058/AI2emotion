"""Infrastructure component responsible for loading and providing the audio model."""

from dataclasses import dataclass
from typing import Optional, Dict

import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

from app.core.config import settings


@dataclass
class LoadedModelResources:
    """Container for loaded audio model resources."""

    model: AutoModelForAudioClassification
    feature_extractor: AutoFeatureExtractor
    id2label: Dict[int, str]
    device: torch.device


class ModelProvider:
    """Loads and exposes the audio emotion recognition model and related resources."""

    def __init__(self) -> None:
        self._resources: Optional[LoadedModelResources] = None

    def get_device(self) -> torch.device:
        """Return the best available device."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load(self) -> LoadedModelResources:
        """
        Load audio model resources lazily and cache them.

        Returns:
            LoadedModelResources containing model, feature extractor, labels, and device.
        """
        if self._resources is None:
            device = self.get_device()
            print(f"[Audio ModelProvider] Using device: {device}")

            model = AutoModelForAudioClassification.from_pretrained(
                settings.audio_model_id
            )
            feature_extractor = AutoFeatureExtractor.from_pretrained(
                settings.audio_model_id,
                do_normalize=True,
            )

            model.to(device)
            model.eval()

            self._resources = LoadedModelResources(
                model=model,
                feature_extractor=feature_extractor,
                id2label=model.config.id2label,
                device=device,
            )

        return self._resources

    def get_model(self) -> AutoModelForAudioClassification:
        """Return the loaded audio model."""
        return self.load().model

    def get_feature_extractor(self) -> AutoFeatureExtractor:
        """Return the loaded feature extractor."""
        return self.load().feature_extractor

    def get_id2label(self) -> Dict[int, str]:
        """Return the ID-to-label mapping."""
        return self.load().id2label

    def get_resources(self) -> LoadedModelResources:
        """Return all loaded resources."""
        return self.load()