"""Service for normalizing raw model emotion labels into canonical project labels."""

from typing import Dict


class EmotionMapper:
    """Maps raw emotion labels from different models into canonical labels."""

    CANONICAL_EMOTIONS = (
        "Angry",
        "Fear",
        "Happy",
        "Neutral",
        "Sad",
        "Surprise",
    )

    EMOTION_MAP = {
        "angry": "Angry",
        "anger": "Angry",
        "annoyance": "Angry",
        "disapproval": "Angry",

        "fear": "Fear",
        "fearful": "Fear",
        "nervousness": "Fear",

        "happy": "Happy",
        "joy": "Happy",
        "amusement": "Happy",
        "excitement": "Happy",
        "love": "Happy",
        "approval": "Happy",
        "gratitude": "Happy",
        "optimism": "Happy",
        "pride": "Happy",
        "relief": "Happy",
        "caring": "Happy",
        "admiration": "Happy",
        "desire": "Happy",

        "neutral": "Neutral",
        "curiosity": "Neutral",
        "remorse": "Neutral",
        "embarrassment": "Neutral",
        "realization": "Neutral",

        "sad": "Sad",
        "sadness": "Sad",
        "grief": "Sad",
        "disappointment": "Sad",

        "surprised": "Surprise",
        "surprise": "Surprise",
        "confusion": "Surprise",

        "disgust": "Angry",
    }

    DEFAULT_EMOTION = "Neutral"

    def normalize_label(self, raw_label: str) -> str:
        """
        Map a raw label to a canonical emotion label.
        """
        if not raw_label:
            return self.DEFAULT_EMOTION

        normalized_key = raw_label.strip().lower()
        return self.EMOTION_MAP.get(normalized_key, self.DEFAULT_EMOTION)

    def aggregate_probabilities(self, raw_probabilities: Dict[str, float]) -> Dict[str, float]:
        """
        Aggregate raw probabilities into canonical emotion probabilities.

        Args:
            raw_probabilities: Mapping of raw model labels to probabilities

        Returns:
            Mapping of canonical labels to aggregated probabilities
        """
        aggregated = {emotion: 0.0 for emotion in self.CANONICAL_EMOTIONS}

        for raw_label, probability in raw_probabilities.items():
            canonical_label = self.normalize_label(raw_label)
            aggregated[canonical_label] += float(probability)

        return {
            emotion: round(probability, 3)
            for emotion, probability in aggregated.items()
        }

    def get_top_emotion(self, raw_probabilities: Dict[str, float]) -> str:
        """
        Determine the top canonical emotion after aggregation.
        """
        aggregated = self.aggregate_probabilities(raw_probabilities)
        return max(aggregated, key=aggregated.get)