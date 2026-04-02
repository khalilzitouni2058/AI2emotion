from pydantic import BaseModel, Field


class TextAnalysisRequest(BaseModel):
    """Request body for text emotion analysis."""

    text: str = Field(..., min_length=1, description="Text to analyze")
    source_name: str | None = Field(
        default=None,
        max_length=255,
        description="Optional client-provided source name",
    )