"""Service for transforming emotion analysis results into UE5 Metahuman animation format."""

from typing import Dict, List, Any, Optional
from datetime import datetime
from app.domain.schemas import EmotionAnalysisResult
from app.services.emotion_mapper import EmotionMapper


class MetahumanAnimator:
    """Converts emotion analysis results to UE5 Metahuman blend shape and animation format."""

    # Mapping from canonical emotions to Metahuman facial expressions
    EMOTION_TO_BLEND_SHAPES = {
        "Angry": {
            "BrowDownLeft": 1.0,
            "BrowDownRight": 1.0,
            "EyeSquintLeft": 0.7,
            "EyeSquintRight": 0.7,
            "MouthPressLeft": 0.7,
            "MouthPressRight": 0.7,
            "MouthFrownLeft": 0.55,
            "MouthFrownRight": 0.55,
            "JawOpen": 0.15,
        },
        "Fear": {
            "BrowUpLeft": 1.0,
            "BrowUpRight": 1.0,
            "EyeWideLeft": 1.0,
            "EyeWideRight": 1.0,
            "JawOpen": 0.6,
            "MouthOpen": 0.55,
            "MouthStretchLeft": 0.3,
            "MouthStretchRight": 0.3,
        },
        "Happy": {
            "BrowUpLeft": 0.5,
            "BrowUpRight": 0.5,
            "EyeSquintLeft": 1.0,
            "EyeSquintRight": 1.0,
            "EyeCheekRaiseLeft": 0.65,
            "EyeCheekRaiseRight": 0.65,
            "MouthSmileLeft": 1.0,
            "MouthSmileRight": 1.0,
            "MouthDimpleLeft": 0.45,
            "MouthDimpleRight": 0.45,
            "JawOpen": 0.2,
        },
        "Sad": {
            "BrowDownLeft": 0.7,
            "BrowDownRight": 0.7,
            "BrowInnerUp": 1.0,
            "EyeWideLeft": 0.5,
            "EyeWideRight": 0.5,
            "MouthFrownLeft": 0.95,
            "MouthFrownRight": 0.95,
            "MouthPressLeft": 0.35,
            "MouthPressRight": 0.35,
            "JawOpen": 0.1,
        },
        "Surprise": {
            "BrowUpLeft": 1.0,
            "BrowUpRight": 1.0,
            "EyeWideLeft": 1.0,
            "EyeWideRight": 1.0,
            "JawOpen": 0.8,
            "MouthOpen": 0.8,
            "MouthStretchLeft": 0.35,
            "MouthStretchRight": 0.35,
        },
        "Neutral": {
            "BrowDownLeft": 0.0,
            "BrowDownRight": 0.0,
            "EyeSquintLeft": 0.0,
            "EyeSquintRight": 0.0,
            "JawOpen": 0.05,
            "MouthOpen": 0.05,
            "MouthSmileLeft": 0.0,
            "MouthSmileRight": 0.0,
            "MouthFrownLeft": 0.0,
            "MouthFrownRight": 0.0,
        },
    }

    _mapper = EmotionMapper()

    @staticmethod
    def _canonical_emotion(emotion: str) -> str:
        """Map raw labels (e.g. fearful/surprised) to canonical labels used by blend-shape presets."""
        return MetahumanAnimator._mapper.normalize_label(emotion)

    @staticmethod
    def _get_dominant_emotion_curve(analysis: EmotionAnalysisResult) -> List[Dict[str, Any]]:
        """Generate time-series curve of dominant emotion with intensity."""
        curve = []
        for segment in analysis.emotion_segments:
            canonical_emotion = MetahumanAnimator._canonical_emotion(segment.emotion)
            curve.append({
                "time": segment.start_time,
                "emotion": canonical_emotion,
                "intensity": segment.confidence,
                "blend_shapes": MetahumanAnimator._scale_blend_shapes(
                    canonical_emotion,
                    segment.confidence
                ),
            })
        return curve

    @staticmethod
    def _scale_blend_shapes(emotion: str, intensity: float) -> Dict[str, float]:
        """Scale blend shape weights by emotion intensity."""
        canonical = MetahumanAnimator._canonical_emotion(emotion)
        base_shapes = MetahumanAnimator.EMOTION_TO_BLEND_SHAPES.get(canonical, {})
        return {shape: weight * intensity for shape, weight in base_shapes.items()}

    @staticmethod
    def _get_emotion_distribution_weights(analysis: EmotionAnalysisResult) -> Dict[str, float]:
        """Generate normalized blend weights from overall emotion distribution."""
        if not analysis.summary or not analysis.summary.emotion_distribution:
            return {}

        total = sum(analysis.summary.emotion_distribution.values())
        if total == 0:
            return {}

        weights = {}
        for emotion, count in analysis.summary.emotion_distribution.items():
            weights[emotion] = count / total

        return weights

    @staticmethod
    def to_ue5_metahuman_format(analysis: EmotionAnalysisResult) -> Dict[str, Any]:
        """
        Transform EmotionAnalysisResult into UE5 Metahuman animation format.

        Returns a dictionary suitable for Unreal Engine 5 skeletal mesh animation,
        blend shape interpolation, and facial motion capture playback.
        """
        emotion_curve = MetahumanAnimator._get_dominant_emotion_curve(analysis)
        dominant_emotion = emotion_curve[0]["emotion"] if emotion_curve else "Neutral"
        dominant_intensity = emotion_curve[0]["intensity"] if emotion_curve else 0.0

        return {
            "version": "1.0",
            "format": "ue5_metahuman",
            "generated_at": datetime.utcnow().isoformat(),
            "metadata": {
                "input_type": analysis.metadata.input_type,
                "total_duration": analysis.metadata.total_duration,
                "sampling_rate": analysis.metadata.sampling_rate,
                "source_name": analysis.metadata.source_name,
            },
            "dominant_emotion": {
                "emotion": dominant_emotion,
                "intensity": round(dominant_intensity, 3),
                "blend_shapes": MetahumanAnimator._scale_blend_shapes(
                    dominant_emotion,
                    dominant_intensity
                ),
            },
            "emotion_distribution": {
                "proportions": MetahumanAnimator._get_emotion_distribution_weights(analysis),
                "total_segments": len(analysis.emotion_segments),
            },
            # Time-series animation curve for facial motion
            "animation_curve": {
                "duration": analysis.metadata.total_duration or 0.0,
                "keyframes": emotion_curve,
                "transitions": [
                    {
                        "from_emotion": t.from_emotion,
                        "to_emotion": t.to_emotion,
                        "time": t.transition_time,
                        "segment_index": t.current_segment,
                    }
                    for t in analysis.emotion_transitions
                ],
            },
            # Blend shape ranges for Metahuman skeletal setup
            "blend_shape_ranges": {
                "min_intensity": 0.0,
                "max_intensity": 1.0,
                "default_curve_type": "ease_in_out",
            },
            # Raw segment data for frame-by-frame animation
            "segments": [
                {
                    "segment_id": seg.chunk_id,
                    "time_range": {
                        "start": round(seg.start_time, 3),
                        "end": round(seg.end_time, 3),
                        "duration": round(seg.duration, 3),
                    },
                    "emotion": seg.emotion,
                    "confidence": round(seg.confidence, 3),
                    "blend_shapes": MetahumanAnimator._scale_blend_shapes(
                        seg.emotion,
                        seg.confidence
                    ),
                    "probabilities": {
                        emotion: round(prob, 3)
                        for emotion, prob in seg.probabilities.items()
                    },
                }
                for seg in analysis.emotion_segments
            ],
        }
