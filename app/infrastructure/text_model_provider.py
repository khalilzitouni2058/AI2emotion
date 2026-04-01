"""Infrastructure component responsible for loading and providing the text model."""

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.core.config import settings


@dataclass
class LoadedTextModelResources:
    """Container for loaded text model resources."""

    model: AutoModelForSequenceClassification
    tokenizer: AutoTokenizer
    id2label: Dict[int, str]
    device: torch.device


class TextModelProvider:
    """Loads and exposes the text emotion recognition model and related resources."""

    def __init__(self) -> None:
        self._resources: Optional[LoadedTextModelResources] = None

    def get_device(self) -> torch.device:
        """Return the best available device."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load(self) -> LoadedTextModelResources:
        """
        Load text model resources lazily and cache them.

        Returns:
            LoadedTextModelResources containing model, tokenizer, labels, and device.
        """
        if self._resources is None:
            device = self.get_device()

            tokenizer = AutoTokenizer.from_pretrained(settings.text_model_id)
            model = AutoModelForSequenceClassification.from_pretrained(
                settings.text_model_id
            )

            model.to(device)
            model.eval()

            self._resources = LoadedTextModelResources(
                model=model,
                tokenizer=tokenizer,
                id2label=model.config.id2label,
                device=device,
            )

        return self._resources

    def get_model(self) -> AutoModelForSequenceClassification:
        """Return the loaded text model."""
        return self.load().model

    def get_tokenizer(self) -> AutoTokenizer:
        """Return the loaded tokenizer."""
        return self.load().tokenizer

    def get_id2label(self) -> Dict[int, str]:
        """Return the ID-to-label mapping."""
        return self.load().id2label

    def get_resources(self) -> LoadedTextModelResources:
        """Return all loaded resources."""
        return self.load()